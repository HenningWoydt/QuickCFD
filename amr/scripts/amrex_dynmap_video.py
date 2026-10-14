#!/usr/bin/env python3
"""
2×2 comparison video:
  top-left    phi field
  top-right   refine-fast     (Dyn-HeiProMap, autorefine 101 every frame)
  bottom-left always repartition (Dyn-HeiProMap, partition fast every frame)
  bottom-right autorefine 30   (Dyn-HeiProMap, autorefine 30 every frame)

Usage:
  python3 amrex_dynmap_video.py \\
      --plt-dir amr/plotfiles/advection_periodic \\
      --hierarchy 4:2 --distance 1:10 \\
      --save amr/results/amrex_compare.mp4 --fps 30
"""

import argparse
import glob
import hashlib
import json
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
import multiprocessing as mp

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, BoundaryNorm


# ── partition cache ───────────────────────────────────────────────────────────

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def cache_dir_for(plt_dir):
    """amr/plotfiles/foo  →  amr/partitions/foo"""
    plt_dir = os.path.normpath(plt_dir)
    name    = os.path.basename(plt_dir)
    parent  = os.path.dirname(plt_dir)          # …/plotfiles
    amr_dir = os.path.dirname(parent)           # …/amr
    return os.path.join(amr_dir, 'partitions', name)


def cache_meta(binary, hierarchy, distance, fps, selected_indices, methods):
    return {
        'binary_hash':       file_sha256(binary),
        'hierarchy':         hierarchy,
        'distance':          distance,
        'fps':               fps,
        'selected_indices':  selected_indices,
        'methods':           methods,
    }


def save_partition_cache(cache_dir, binary, hierarchy, distance, fps,
                         selected_indices, methods, part_frames):
    """Save per-frame partition lists, timing, and metadata.

    part_frames: list of (part_refine, part_always, part_auto, ms_refine, ms_always, ms_auto)
    """
    os.makedirs(cache_dir, exist_ok=True)
    meta = cache_meta(binary, hierarchy, distance, fps, selected_indices, methods)
    with open(os.path.join(cache_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    for i, (pr, pa, pu, tr, ta, tu) in enumerate(part_frames):
        np.savez_compressed(
            os.path.join(cache_dir, f'frame_{i:06d}.npz'),
            refine=np.array(pr, dtype=np.int32),
            always=np.array(pa, dtype=np.int32),
            auto=np.array(pu,   dtype=np.int32),
            ms_refine=np.float32(tr),
            ms_always=np.float32(ta),
            ms_auto=np.float32(tu),
        )


def load_partition_cache(cache_dir, binary, hierarchy, distance, fps,
                         selected_indices, methods):
    """Return list of (part_refine, part_always, part_auto, ms_r, ms_a, ms_u)
    or None on mismatch."""
    meta_path = os.path.join(cache_dir, 'meta.json')
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        stored = json.load(f)
    expected = cache_meta(binary, hierarchy, distance, fps, selected_indices, methods)
    if stored != expected:
        return None
    n = len(glob.glob(os.path.join(cache_dir, 'frame_*.npz')))
    if n != len(selected_indices):
        return None
    result = []
    for i in range(n):
        d = np.load(os.path.join(cache_dir, f'frame_{i:06d}.npz'))
        result.append((
            d['refine'].tolist(), d['always'].tolist(), d['auto'].tolist(),
            float(d['ms_refine']), float(d['ms_always']), float(d['ms_auto']),
        ))
    return result


# ── AMReX readers ─────────────────────────────────────────────────────────────

def parse_cell_h(cell_h_path):
    with open(cell_h_path) as f:
        lines = f.readlines()
    boxes, fab_entries = [], []
    i = 0
    while i < len(lines):
        m = re.match(r'\((\d+)', lines[i].strip())
        if m:
            n = int(m.group(1))
            for j in range(i + 1, i + 1 + n):
                bm = re.match(r'\(\((\d+),(\d+)\)\s*\((\d+),(\d+)\)', lines[j].strip())
                if bm:
                    boxes.append((int(bm.group(1)), int(bm.group(2)),
                                  int(bm.group(3)), int(bm.group(4))))
            i += n + 2
            break
        i += 1
    for line in lines:
        m = re.match(r'FabOnDisk:\s+(\S+)\s+(\d+)', line.strip())
        if m:
            fab_entries.append((m.group(1), int(m.group(2))))
    return boxes, fab_entries


def read_fab(filepath, offset, nx, ny):
    with open(filepath, 'rb') as f:
        f.seek(offset)
        f.readline()
        data = np.frombuffer(f.read(nx * ny * 8), dtype='<f8')
    return data.reshape((ny, nx))


def read_level_raw(plotdir, level):
    """Return (grid, boxes, x0, y0, dx, dy) or None."""
    cell_h = os.path.join(plotdir, f'Level_{level}', 'Cell_H')
    level_dir = os.path.join(plotdir, f'Level_{level}')
    header_path = os.path.join(plotdir, 'Header')
    if not os.path.exists(cell_h):
        return None
    with open(header_path) as f:
        hlines = f.readlines()
    nvar = int(hlines[1].strip())
    # [0]=version [1]=nvar [2..nvar+1]=varnames [nvar+2]=spacedim [nvar+3]=time
    # [nvar+4]=max_level [nvar+5]=prob_lo [nvar+6]=prob_hi [nvar+7]=ref_ratios
    # [nvar+8]=domains [nvar+9]=stepping  [nvar+10+level]=dx
    base = 2 + nvar
    prob_lo = list(map(float, hlines[base + 3].split()))
    dx_line = hlines[base + 8 + level].split()
    dx, dy = float(dx_line[0]), float(dx_line[1])
    boxes, fab_entries = parse_cell_h(cell_h)
    all_x1 = max(b[2] for b in boxes) + 1
    all_y1 = max(b[3] for b in boxes) + 1
    grid = np.zeros((all_y1, all_x1))
    for (lo_x, lo_y, hi_x, hi_y), (fname, offset) in zip(boxes, fab_entries):
        nx_ = hi_x - lo_x + 1
        ny_ = hi_y - lo_y + 1
        grid[lo_y:hi_y + 1, lo_x:hi_x + 1] = read_fab(
            os.path.join(level_dir, fname), offset, nx_, ny_)
    return grid, boxes, prob_lo[0], prob_lo[1], dx, dy


# ── composite graph ────────────────────────────────────────────────────────────

def build_composite_graph(nx0, ny0, l1_mask, periodic=True):
    """Return (metis_lines, nodes, edges, l0_node, l1_node)."""
    nx1, ny1 = nx0 * 2, ny0 * 2
    l0_node = {}
    nodes = []
    for r in range(ny0):
        for c in range(nx0):
            if not l1_mask[r, c]:
                l0_node[(r, c)] = len(nodes)
                nodes.append((0, r, c))
    l1_node = {}
    for r0 in range(ny0):
        for c0 in range(nx0):
            if l1_mask[r0, c0]:
                for dr in range(2):
                    for dc in range(2):
                        r1, c1 = r0 * 2 + dr, c0 * 2 + dc
                        l1_node[(r1, c1)] = len(nodes)
                        nodes.append((1, r1, c1))

    def l0_nb(r, c, dr, dc):
        rn = (r + dr) % ny0 if periodic else r + dr
        cn = (c + dc) % nx0 if periodic else c + dc
        if not periodic and (rn < 0 or rn >= ny0 or cn < 0 or cn >= nx0):
            return -1
        if l1_mask[rn, cn]:
            return -1
        return l0_node.get((rn, cn), -1)

    def l0_l1_boundary(r0, c0, dr, dc):
        rn = (r0 + dr) % ny0 if periodic else r0 + dr
        cn = (c0 + dc) % nx0 if periodic else c0 + dc
        if not periodic and (rn < 0 or rn >= ny0 or cn < 0 or cn >= nx0):
            return []
        if not l1_mask[rn, cn]:
            return []
        r1n, c1n = rn * 2, cn * 2
        if dr != 0:
            return [l1_node.get((r1n + (0 if dr == 1 else 1), c1n + d), -1) for d in range(2)]
        else:
            return [l1_node.get((r1n + d, c1n + (0 if dc == 1 else 1)), -1) for d in range(2)]

    def l1_nb(r1, c1, dr, dc):
        rn, cn = r1 + dr, c1 + dc
        if periodic:
            rn = rn % ny1; cn = cn % nx1
        else:
            if rn < 0 or rn >= ny1 or cn < 0 or cn >= nx1:
                return -1
        nb = l1_node.get((rn, cn), -1)
        if nb >= 0:
            return nb
        r0n, c0n = rn // 2, cn // 2
        if periodic:
            r0n = r0n % ny0; c0n = c0n % nx0
        return l0_node.get((r0n, c0n), -1)

    edges = set()
    adj = [[] for _ in range(len(nodes))]

    def add_edge(u, v):
        if u < 0 or v < 0 or u == v:
            return
        key = (min(u, v), max(u, v))
        if key not in edges:
            edges.add(key)
            adj[u].append(v)
            adj[v].append(u)

    for nid, (level, r, c) in enumerate(nodes):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if level == 0:
                add_edge(nid, l0_nb(r, c, dr, dc))
                for nb2 in l0_l1_boundary(r, c, dr, dc):
                    add_edge(nid, nb2)
            else:
                add_edge(nid, l1_nb(r, c, dr, dc))

    N = len(nodes)
    M = len(edges)
    lines = [f"{N} {M} 011\n"]
    for nbs in adj:
        lines.append("1 " + " ".join(f"{nb + 1} 1" for nb in nbs) + "\n")

    return lines, nodes, edges, l0_node, l1_node


def cell_key(level, r, c):
    """Hashable cell identity, used as dict key."""
    return (level, r, c)


def partition_image(nodes, partition, ny0, nx0):
    """Build (ny0*2, nx0*2) partition image from node list + partition list."""
    img = np.full((ny0 * 2, nx0 * 2), -1, dtype=np.int32)
    for nid, (level, r, c) in enumerate(nodes):
        p = partition[nid]
        if level == 0:
            img[r * 2:(r + 1) * 2, c * 2:(c + 1) * 2] = p
        else:
            img[r, c] = p
    return img


def partition_stats(nodes, edges, partition, k, prev_part_map):
    """Return (max_balance, comm_cost, mig_cost) and update prev_part_map.

    max_balance : max(block_size) / avg_block_size  (1.0 = perfect balance)
    comm_cost   : number of cut edges
    mig_cost    : vertices present in both frames whose partition changed
    """
    n = len(partition)
    counts = [0] * k
    for p in partition:
        counts[p] += 1
    avg = n / k if k else 1
    max_balance = max(counts) / avg if avg > 0 else 1.0

    comm_cost = sum(1 for u, v in edges if partition[u] != partition[v])

    mig_cost = 0
    for nid, cell in enumerate(nodes):
        ck = cell_key(*cell)
        if ck in prev_part_map and prev_part_map[ck] != partition[nid]:
            mig_cost += 1
        prev_part_map[ck] = partition[nid]

    return max_balance, comm_cost, mig_cost


# ── Dyn-HeiProMap persistent process ─────────────────────────────────────────

class DynHeiProMap:
    """Persistent Dyn-HeiProMap subprocess.

    Uses dense sequential vertex IDs (0, 1, 2, ...) and maintains a
    cell→id registry.  Synchronisation uses stdout: the binary prints
    "Partition saved to <path>" after the file is fully closed, so we
    wait for that line rather than polling for file existence (which
    races against a partial write).
    """

    def __init__(self, binary, hierarchy, distance, threads, imbalance=0.03):
        self._tmpdir = tempfile.mkdtemp(prefix='dyn_hpm_')
        cmd = [binary,
               '--hierarchy', hierarchy,
               '--distance',  distance,
               '--threads',   str(threads),
               '--imbalance', str(imbalance)]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._seq        = 0            # partition file counter
        self._next_id    = 0            # next free vertex id (never reused)
        self._cell_to_id = {}           # cell_key → dyn vertex id
        self._saved_q    = queue.Queue()  # paths confirmed written by binary
        self._reader     = threading.Thread(target=self._drain_stdout, daemon=True)
        self._reader.start()

    def _drain_stdout(self):
        """Background thread: read binary stdout and enqueue confirmed save paths."""
        for line in self._proc.stdout:
            if line.startswith('Partition saved to '):
                self._saved_q.put(line.strip()[len('Partition saved to '):])

    def _send(self, line):
        self._proc.stdin.write(line + '\n')
        self._proc.stdin.flush()

    def load_graph(self, nodes, edges):
        """Load initial graph via exec file. Assigns IDs 0..N-1 in node order."""
        seq_path = os.path.join(self._tmpdir, 'init.my_seq')
        with open(seq_path, 'w') as f:
            for i, (lv, r, c) in enumerate(nodes):
                self._cell_to_id[cell_key(lv, r, c)] = i
                self._next_id = i + 1
                f.write(f'+v {i} 1\n')
            for u, v in edges:
                f.write(f'+e {u} {v} 1\n')
        self._send(f'exec {seq_path}')

    def apply_diff(self, prev_nodes, prev_edges, curr_nodes, curr_edges):
        """Compute and send the graph diff between two consecutive frames."""
        prev_set = {cell_key(*cell) for cell in prev_nodes}
        curr_set = {cell_key(*cell) for cell in curr_nodes}

        prev_local_to_id = {i: self._cell_to_id[cell_key(*cell)]
                            for i, cell in enumerate(prev_nodes)}
        prev_edge_ids = set()
        for u, v in prev_edges:
            eu, ev = prev_local_to_id[u], prev_local_to_id[v]
            prev_edge_ids.add((min(eu, ev), max(eu, ev)))

        # assign IDs to new cells
        for cell in curr_nodes:
            k = cell_key(*cell)
            if k not in self._cell_to_id:
                self._cell_to_id[k] = self._next_id
                self._next_id += 1

        curr_local_to_id = {i: self._cell_to_id[cell_key(*cell)]
                            for i, cell in enumerate(curr_nodes)}
        curr_edge_ids = set()
        for u, v in curr_edges:
            eu, ev = curr_local_to_id[u], curr_local_to_id[v]
            curr_edge_ids.add((min(eu, ev), max(eu, ev)))

        cmds = []
        for u, v in (prev_edge_ids - curr_edge_ids):
            cmds.append(f'-e {u} {v}')
        for cell in prev_set - curr_set:
            vid = self._cell_to_id.pop(cell_key(*cell))
            cmds.append(f'-v {vid}')
        for cell in curr_set - prev_set:
            vid = self._cell_to_id[cell_key(*cell)]
            cmds.append(f'+v {vid} 1')
        for u, v in (curr_edge_ids - prev_edge_ids):
            cmds.append(f'+e {u} {v} 1')

        for c in cmds:
            self._send(c)

    def partition(self, config='fast'):
        self._send(f'partition {config}')

    def autorefine(self, threshold=30, config='fast'):
        self._send(f'autorefine {threshold} {config}')

    def get_partition(self, curr_nodes):
        """Save partition and return list aligned to curr_nodes order.

        Waits for the binary to print "Partition saved to <path>" on stdout,
        which only happens after the file is fully closed — no partial-read race.
        """
        self._seq += 1
        path = os.path.join(self._tmpdir, f'part_{self._seq}.partition')
        self._send(f'save-partition {path}')
        deadline = time.time() + 60.0
        while True:
            try:
                saved = self._saved_q.get(timeout=1.0)
                if saved == path:
                    break
            except queue.Empty:
                if time.time() > deadline:
                    raise TimeoutError('Dyn-HeiProMap save-partition timed out')
        with open(path) as f:
            raw = [int(l.strip()) for l in f if l.strip()]
        os.remove(path)
        # raw[i] = partition block for vertex with id i
        return [raw[self._cell_to_id[cell_key(*cell)]] for cell in curr_nodes]

    def close(self):
        try:
            self._proc.stdin.close()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()




# ── render worker ──────────────────────────────────────────────────────────────

def _render_worker(args_tuple):
    indices, frames, cfg = args_tuple
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, BoundaryNorm
    import numpy as np

    k = cfg['k']
    norm_phi  = Normalize(vmin=cfg['phi_min'], vmax=cfg['phi_max'])
    xmin, xmax, ymin, ymax = cfg['domain']

    cmap_part = matplotlib.colormaps.get_cmap('tab10').resampled(k)
    bounds    = np.arange(-0.5, k + 0.5, 1)
    norm_part = BoundaryNorm(bounds, cmap_part.N)

    fig, axes = plt.subplots(2, 2, figsize=cfg['figsize'])
    fig.patch.set_facecolor('#1a1a2e')

    titles = [
        'phi field',
        f'refine-fast (Dyn-HeiProMap, k={k})',
        f'always repartition (Dyn-HeiProMap, k={k})',
        f'autorefine 30 (Dyn-HeiProMap, k={k})',
    ]
    for ax, title in zip(axes.flat, titles):
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        ax.set_facecolor('#111111')
        ax.tick_params(colors='white', labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor('#444444')
        ax.set_title(title, color='white', fontsize=8, pad=4)

    ax_phi, ax_static, ax_always, ax_auto = axes.flat

    im_phi = ax_phi.imshow(np.zeros((2, 2)), origin='lower', cmap='RdYlBu_r',
                           norm=norm_phi, extent=[xmin, xmax, ymin, ymax],
                           interpolation='nearest')
    im_static = ax_static.imshow(np.zeros((2, 2)), origin='lower', cmap=cmap_part,
                                 norm=norm_part, extent=[xmin, xmax, ymin, ymax],
                                 interpolation='nearest')
    im_always = ax_always.imshow(np.zeros((2, 2)), origin='lower', cmap=cmap_part,
                                 norm=norm_part, extent=[xmin, xmax, ymin, ymax],
                                 interpolation='nearest')
    im_auto   = ax_auto.imshow(np.zeros((2, 2)), origin='lower', cmap=cmap_part,
                               norm=norm_part, extent=[xmin, xmax, ymin, ymax],
                               interpolation='nearest')

    fig.colorbar(im_phi,    ax=ax_phi,    fraction=0.046, pad=0.04).set_label('phi',  color='white', fontsize=7)
    fig.colorbar(im_static, ax=ax_static, fraction=0.046, pad=0.04).set_label('rank', color='white', fontsize=7)
    fig.colorbar(im_always, ax=ax_always, fraction=0.046, pad=0.04).set_label('rank', color='white', fontsize=7)
    fig.colorbar(im_auto,   ax=ax_auto,   fraction=0.046, pad=0.04).set_label('rank', color='white', fontsize=7)
    for cb_ax in [ax_phi, ax_static, ax_always, ax_auto]:
        for cbar in fig.axes:
            if cbar is not cb_ax:
                cbar.yaxis.set_tick_params(color='white')
    # simpler: just set tick colors on all axes
    for ax in fig.axes:
        ax.tick_params(colors='white')
        ax.yaxis.label.set_color('white')

    suptitle = fig.suptitle('', color='white', fontsize=9, y=1.005)

    for ax in [ax_static, ax_always, ax_auto]:
        ax.set_xlabel('', color='#cccccc', fontsize=6.5, fontfamily='monospace')

    fig.tight_layout()

    out = []
    for idx in indices:
        fd = frames[idx]
        # phi
        src = fd['fine'] if fd['fine'] is not None else fd['coarse']
        g, x0, y0, dx, dy = src
        ny_g, nx_g = g.shape
        im_phi.set_data(g)
        im_phi.set_extent([x0, x0 + nx_g * dx, y0, y0 + ny_g * dy])
        # partitions
        im_static.set_data(fd['part_refine'])
        im_always.set_data(fd['part_always'])
        im_auto.set_data(fd['part_auto'])
        for im in [im_static, im_always, im_auto]:
            im.set_extent([xmin, xmax, ymin, ymax])

        def _fmt(bal, comm, mig):
            return f'bal {bal:.2f}  Σcomm {comm}  Σmig {mig}'

        ax_static.set_xlabel(_fmt(*fd['stats_refine']))
        ax_always.set_xlabel(_fmt(*fd['stats_always']))
        ax_auto.set_xlabel(_fmt(*fd['stats_auto']))

        suptitle.set_text(
            f"t = {fd['time']:.4f} s  |  "
            f"nodes: {fd['n_nodes']}  (L0: {fd['n_l0']}  L1: {fd['n_l1']})"
        )
        png = cfg['tmpdir'] + f'/f{idx:06d}.png'
        fig.savefig(png, dpi=cfg['dpi'], facecolor=fig.get_facecolor(),
                    bbox_inches='tight')
        out.append((idx, png))

    plt.close(fig)
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    _default_binary = os.path.join(
        os.path.dirname(__file__), '../../extern/amrex/build/Tests/Amr/'
        'Advection_AmrCore/2d/Test_Amr_Advection_AmrCore_2d')
    parser.add_argument('--plt-dir',    default='.')
    parser.add_argument('--dynheipromap',
                        default='/home/henning/Projects/HeiProMap/build/Dyn-HeiProMap')
    parser.add_argument('--hierarchy',  default='4:2')
    parser.add_argument('--distance',   default='1:10')
    parser.add_argument('--save',       default='amrex_compare.mp4')
    parser.add_argument('--fps',        type=int, default=30)
    parser.add_argument('--amrex-binary', default=_default_binary,
                        help='AMReX advection binary (for auto-run)')
    parser.add_argument('--inputs', default=None,
                        help='AMReX inputs file (auto-detected from --plt-dir name if omitted)')
    parser.add_argument('--mpirun', default='/usr/bin/mpirun.openmpi')
    parser.add_argument('--mpi-np', type=int, default=2)
    args = parser.parse_args()

    k = 1
    for p in args.hierarchy.split(':'): k *= int(p)

    plotdirs = sorted(p for p in glob.glob(os.path.join(args.plt_dir, 'plt*'))
                      if re.match(r'plt\d+$', os.path.basename(p)))
    if not plotdirs:
        # auto-detect inputs file from plt-dir name
        inputs = args.inputs
        if inputs is None:
            name = os.path.basename(os.path.normpath(args.plt_dir))
            inputs = os.path.join(os.path.dirname(__file__), '../../amr/inputs', name)
            inputs = os.path.normpath(inputs)
        if not os.path.exists(inputs):
            print(f'No plotfiles in {args.plt_dir} and no inputs file at {inputs}')
            print('Pass --inputs <file> to specify one.')
            return
        binary = os.path.normpath(args.amrex_binary)
        if not os.path.exists(binary):
            print(f'AMReX binary not found: {binary}'); return
        os.makedirs(args.plt_dir, exist_ok=True)
        print(f'No plotfiles found — running simulation: {os.path.basename(inputs)}')
        cmd = [args.mpirun, '-n', str(args.mpi_np), binary, inputs]
        result = subprocess.run(cmd, cwd=os.path.normpath(
            os.path.join(os.path.dirname(__file__), '../..')))
        if result.returncode != 0:
            print('Simulation failed.'); return
        plotdirs = sorted(p for p in glob.glob(os.path.join(args.plt_dir, 'plt*'))
                          if re.match(r'plt\d+$', os.path.basename(p)))
        if not plotdirs:
            print(f'Simulation ran but no plotfiles found in {args.plt_dir}'); return

    # read simulation times
    times = []
    for pd in plotdirs:
        try:
            with open(os.path.join(pd, 'Header')) as f:
                hlines = f.readlines()
            nvar = int(hlines[1].strip())
            times.append(float(hlines[2 + nvar + 1].strip()))
        except Exception:
            times.append(0.0)

    # subsample to target fps
    t_max, t_min = max(times), min(times)
    n_target = int((t_max - t_min) * args.fps) + 1
    if len(plotdirs) > n_target:
        ta = np.array(times)
        targets = np.linspace(t_min, t_max, n_target)
        selected_indices = sorted(set(int(np.argmin(np.abs(ta - t))) for t in targets))
        plotdirs = [plotdirs[i] for i in selected_indices]
        times    = [times[i]    for i in selected_indices]
    else:
        selected_indices = list(range(len(plotdirs)))

    print(f'{len(plotdirs)} frames to process.')

    # domain from first frame
    r0_first = read_level_raw(plotdirs[0], 0)
    _, _, x0_d, y0_d, dx0, dy0 = r0_first
    ny0, nx0 = r0_first[0].shape
    domain = (x0_d, x0_d + nx0 * dx0, y0_d, y0_d + ny0 * dy0)
    threads = min(4, mp.cpu_count())

    # ── phase 1: read all frames + build graphs ────────────────────────────────

    print('Phase 1: reading frames...')

    all_phi    = []
    frame_data = []

    for i, (pd, t) in enumerate(zip(plotdirs, times)):
        print(f'\r  {i + 1}/{len(plotdirs)}', end='', flush=True)

        r0 = read_level_raw(pd, 0)
        r1 = read_level_raw(pd, 1)
        if r0 is None:
            continue

        grid0, boxes0, *_ = r0
        l1_mask = np.zeros((ny0, nx0), dtype=bool)
        grid1 = None

        if r1 is not None:
            grid1, boxes1, *_ = r1
            for (lo_x, lo_y, hi_x, hi_y) in boxes1:
                l1_mask[lo_y // 2:hi_y // 2 + 1, lo_x // 2:hi_x // 2 + 1] = True

        _, nodes, edges, l0_node, l1_node = build_composite_graph(
            nx0, ny0, l1_mask)

        # phi source
        if grid1 is not None:
            phi_src = grid1
        else:
            phi_src = np.repeat(np.repeat(grid0, 2, axis=0), 2, axis=1)
        all_phi.append(phi_src.ravel())

        n_l0 = sum(1 for lv, _, _ in nodes if lv == 0)
        n_l1 = sum(1 for lv, _, _ in nodes if lv == 1)

        frame_data.append({
            'coarse':  (grid0,) + tuple(r0[2:]),
            'fine':    (grid1,) + tuple(r1[2:]) if r1 else None,
            'nodes':   nodes,
            'edges':   edges,
            'n_nodes': len(nodes),
            'n_l0':    n_l0,
            'n_l1':    n_l1,
            'time':    t,
        })

    print(f'\n  {len(frame_data)} frames read.')

    if not frame_data:
        print('No valid frames.'); return

    flat    = np.concatenate(all_phi)
    phi_min = float(np.percentile(flat, 2))
    phi_max = float(np.percentile(flat, 98))
    phi_max = max(phi_max, phi_min + 1e-6)

    # ── phase 2: three Dyn-HeiProMap processes (with partition cache) ─────────

    methods = {
        'refine': {'label': 'refine-fast',        'cmd': 'autorefine 101 fast'},
        'always': {'label': 'always-repartition', 'cmd': 'partition fast'},
        'auto':   {'label': 'autorefine-30',      'cmd': 'autorefine 30 fast'},
    }

    cdir = cache_dir_for(args.plt_dir)
    cached = load_partition_cache(cdir, args.dynheipromap,
                                  args.hierarchy, args.distance,
                                  args.fps, selected_indices, methods)

    if cached is not None:
        print(f'Phase 2: loading cached partitions from {cdir}')
        raw_parts = cached
    else:
        print('Phase 2: Dyn-HeiProMap (refine-fast + always repartition + autorefine 30)...')

        dyn_refine = DynHeiProMap(args.dynheipromap,
                                  args.hierarchy, args.distance, threads)
        dyn_always = DynHeiProMap(args.dynheipromap,
                                  args.hierarchy, args.distance, threads)
        dyn_auto   = DynHeiProMap(args.dynheipromap,
                                  args.hierarchy, args.distance, threads)

        fd0 = frame_data[0]
        dyn_refine.load_graph(fd0['nodes'], fd0['edges'])
        dyn_always.load_graph(fd0['nodes'], fd0['edges'])
        dyn_auto.load_graph(fd0['nodes'], fd0['edges'])

        raw_parts = []
        try:
            for i, fd in enumerate(frame_data):
                print(f'\r  {i + 1}/{len(frame_data)}', end='', flush=True)
                nodes = fd['nodes']
                edges = fd['edges']

                if i == 0:
                    t0 = time.perf_counter()
                    dyn_refine.partition('fast')
                    pr = dyn_refine.get_partition(nodes)
                    ms_r = (time.perf_counter() - t0) * 1e3

                    t0 = time.perf_counter()
                    dyn_always.partition('fast')
                    pa = dyn_always.get_partition(nodes)
                    ms_a = (time.perf_counter() - t0) * 1e3

                    t0 = time.perf_counter()
                    dyn_auto.partition('fast')
                    pu = dyn_auto.get_partition(nodes)
                    ms_u = (time.perf_counter() - t0) * 1e3
                else:
                    prev = frame_data[i - 1]
                    dyn_refine.apply_diff(prev['nodes'], prev['edges'], nodes, edges)
                    dyn_always.apply_diff(prev['nodes'], prev['edges'], nodes, edges)
                    dyn_auto.apply_diff(prev['nodes'], prev['edges'], nodes, edges)

                    t0 = time.perf_counter()
                    dyn_refine.autorefine(101, 'fast')
                    pr = dyn_refine.get_partition(nodes)
                    ms_r = (time.perf_counter() - t0) * 1e3

                    t0 = time.perf_counter()
                    dyn_always.partition('fast')
                    pa = dyn_always.get_partition(nodes)
                    ms_a = (time.perf_counter() - t0) * 1e3

                    t0 = time.perf_counter()
                    dyn_auto.autorefine(30, 'fast')
                    pu = dyn_auto.get_partition(nodes)
                    ms_u = (time.perf_counter() - t0) * 1e3

                raw_parts.append((pr, pa, pu, ms_r, ms_a, ms_u))
        finally:
            dyn_refine.close()
            dyn_always.close()
            dyn_auto.close()

        print(f'\n  saving partitions to {cdir}')
        save_partition_cache(cdir, args.dynheipromap,
                             args.hierarchy, args.distance,
                             args.fps, selected_indices, methods, raw_parts)

    # compute stats + build frame list
    prev_map_refine, prev_map_always, prev_map_auto = {}, {}, {}
    acc_comm_r = acc_mig_r = 0
    acc_comm_a = acc_mig_a = 0
    acc_comm_u = acc_mig_u = 0

    frames = []
    table_rows = []
    for i, (fd, (part_refine, part_always, part_auto, ms_r, ms_a, ms_u)) in enumerate(
            zip(frame_data, raw_parts)):
        nodes = fd['nodes']
        edges = fd['edges']

        bal_r, comm_r, mig_r = partition_stats(nodes, edges, part_refine, k, prev_map_refine)
        bal_a, comm_a, mig_a = partition_stats(nodes, edges, part_always, k, prev_map_always)
        bal_u, comm_u, mig_u = partition_stats(nodes, edges, part_auto,   k, prev_map_auto)

        acc_comm_r += comm_r; acc_mig_r += mig_r
        acc_comm_a += comm_a; acc_mig_a += mig_a
        acc_comm_u += comm_u; acc_mig_u += mig_u

        frames.append({
            'coarse':       fd['coarse'],
            'fine':         fd['fine'],
            'part_refine':  partition_image(nodes, part_refine, ny0, nx0),
            'part_always':  partition_image(nodes, part_always, ny0, nx0),
            'part_auto':    partition_image(nodes, part_auto,   ny0, nx0),
            'stats_refine': (bal_r, acc_comm_r, acc_mig_r),
            'stats_always': (bal_a, acc_comm_a, acc_mig_a),
            'stats_auto':   (bal_u, acc_comm_u, acc_mig_u),
            'n_nodes':      fd['n_nodes'],
            'n_l0':         fd['n_l0'],
            'n_l1':         fd['n_l1'],
            'time':         fd['time'],
        })
        table_rows.append((
            i, fd['time'], fd['n_nodes'],
            bal_r, acc_comm_r, acc_mig_r, ms_r,
            bal_a, acc_comm_a, acc_mig_a, ms_a,
            bal_u, acc_comm_u, acc_mig_u, ms_u,
        ))

    print(f'\n  {len(frames)} frames with all partitions.')

    # ── phase 3: render ────────────────────────────────────────────────────────

    render_cfg = {
        'phi_min': phi_min, 'phi_max': phi_max,
        'domain':  domain,
        'figsize': (14, 12),
        'dpi':     100,
        'k':       k,
    }

    import imageio, imageio.v2 as iio2
    ext  = os.path.splitext(args.save)[1].lower()
    extra = {'loop': 0} if ext == '.gif' else {
        'codec': 'libx264', 'macro_block_size': 1,
        'output_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p']}

    n        = len(frames)
    n_workers = max(1, mp.cpu_count() - 1)
    chunk    = max(1, n // n_workers)
    batches  = [list(range(j, min(j + chunk, n))) for j in range(0, n, chunk)]

    print(f'Rendering {n} frames on {n_workers} workers → {args.save}')
    with tempfile.TemporaryDirectory() as tmpdir:
        render_cfg = {**render_cfg, 'tmpdir': tmpdir}
        work = [(b, frames, render_cfg) for b in batches]
        png_map = {}
        done = 0
        with mp.Pool(n_workers) as pool:
            for result in pool.imap_unordered(_render_worker, work):
                for idx, png in result:
                    png_map[idx] = png
                done += len(result)
                pct = done * 100 // n
                bar = '#' * (pct // 5) + '.' * (20 - pct // 5)
                print(f'\r  [{bar}] {pct:3d}%  {done}/{n}', end='', flush=True)
        print()
        print('Assembling...')
        with imageio.get_writer(args.save, fps=args.fps, **extra) as writer:
            for j in range(n):
                writer.append_data(iio2.imread(png_map[j]))
    print(f'Saved → {args.save}')

    # ── per-frame table ─────────────────────────────────────────────────────────

    C = {
        'frm':   4,  't':    7,  'nd':  5,
        'bal':   5,  'Sco': 8,  'Smi': 7,  'ms':  7,
    }
    meth = [methods['refine']['label'], methods['always']['label'], methods['auto']['label']]
    hdr1 = (f"{'frm':>{C['frm']}}  {'t(s)':>{C['t']}}  {'nodes':>{C['nd']}}  "
            + '  '.join(
                f"{'── ' + m + ' ──':^{C['bal']+C['Sco']+C['Smi']+C['ms']+9}}"
                for m in meth))
    hdr2 = (f"{'':>{C['frm']}}  {'':>{C['t']}}  {'':>{C['nd']}}  "
            + '  '.join(
                f"{'bal':>{C['bal']}}  {'Σcomm':>{C['Sco']}}  {'Σmig':>{C['Smi']}}  {'ms':>{C['ms']}}"
                for _ in meth))
    sep  = '-' * len(hdr2)

    print()
    print(hdr1)
    print(hdr2)
    print(sep)
    for (frm, t, nd,
         br, cr, mr, tr,
         ba, ca, ma, ta,
         bu, cu, mu, tu) in table_rows:
        print(
            f"{frm:>{C['frm']}}  {t:>{C['t']}.4f}  {nd:>{C['nd']}}  "
            f"{br:>{C['bal']}.2f}  {cr:>{C['Sco']}}  {mr:>{C['Smi']}}  {tr:>{C['ms']}.1f}  "
            f"{ba:>{C['bal']}.2f}  {ca:>{C['Sco']}}  {ma:>{C['Smi']}}  {ta:>{C['ms']}.1f}  "
            f"{bu:>{C['bal']}.2f}  {cu:>{C['Sco']}}  {mu:>{C['Smi']}}  {tu:>{C['ms']}.1f}"
        )
    print(sep)
    cr_f, mr_f = table_rows[-1][4], table_rows[-1][5]
    ca_f, ma_f = table_rows[-1][8], table_rows[-1][9]
    cu_f, mu_f = table_rows[-1][12], table_rows[-1][13]
    ms_r_tot = sum(r[6]  for r in table_rows)
    ms_a_tot = sum(r[10] for r in table_rows)
    ms_u_tot = sum(r[14] for r in table_rows)
    print(
        f"{'TOT':>{C['frm']}}  {'':>{C['t']}}  {'':>{C['nd']}}  "
        f"{'':>{C['bal']}}  {cr_f:>{C['Sco']}}  {mr_f:>{C['Smi']}}  {ms_r_tot:>{C['ms']}.1f}  "
        f"{'':>{C['bal']}}  {ca_f:>{C['Sco']}}  {ma_f:>{C['Smi']}}  {ms_a_tot:>{C['ms']}.1f}  "
        f"{'':>{C['bal']}}  {cu_f:>{C['Sco']}}  {mu_f:>{C['Smi']}}  {ms_u_tot:>{C['ms']}.1f}"
    )

    # ── compact summary ────────────────────────────────────────────────────────

    sw, nw, cw, mw, tw = 22, 8, 10, 10, 10
    s_hdr = f"  {'method':<{sw}}  {'nodes':>{nw}}  {'Σcomm':>{cw}}  {'Σmig':>{mw}}  {'ms(tot)':>{tw}}"
    s_sep = '  ' + '-' * (sw + nw + cw + mw + tw + 8)
    print()
    print(s_hdr)
    print(s_sep)
    for label, comm, mig, ms in [
        (methods['refine']['label'], cr_f, mr_f, ms_r_tot),
        (methods['always']['label'], ca_f, ma_f, ms_a_tot),
        (methods['auto']['label'],   cu_f, mu_f, ms_u_tot),
    ]:
        print(f"  {label:<{sw}}  {table_rows[-1][2]:>{nw}}  {comm:>{cw}}  {mig:>{mw}}  {ms:>{tw}.1f}")
    print(s_sep)


if __name__ == '__main__':
    main()
