#!/usr/bin/env python3
"""
2×2 comparison video:
  top-left    phi field
  top-right   refine-fast     (Dyn-HeiProMap, autorefine 101 every frame)
  bottom-left always repartition (Dyn-HeiProMap, partition fast every frame)
  bottom-right autorefine 30   (Dyn-HeiProMap, autorefine 30 every frame)

With --render3d (3D plotfiles) each panel is a 3D view: density isosurfaces, coloured
by value in the field panel and by rank in the three partition panels.

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
import sys

import numpy as np
# offscreen 3D rendering (--render3d) goes through EGL; harmless otherwise
os.environ.setdefault('VTK_DEFAULT_OPENGL_WINDOW', 'vtkEGLRenderWindow')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, BoundaryNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amrex_common import (GRAPH_VERSION, read_plotfile, read_time, finest_shape,
                          finest_domain, build_composite_graph, partition_image, slice_2d,
                          select_frames, playback_fps, assemble_video, var_index,
                          IdRegistry)


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


def cache_meta(binary, hierarchy, distance, methods, n_plotfiles):
    return {
        'binary_hash':   file_sha256(binary),
        'hierarchy':     hierarchy,
        'distance':      distance,
        'methods':       methods,
        'n_plotfiles':   n_plotfiles,
        'graph_version': GRAPH_VERSION,
        'skip_unchanged': True,   # plotfiles whose graph did not change are not partitioned
    }


# One file per plotfile whose graph changed (plotfile 0 and every plotfile whose graph
# differs from the previous one; the others keep the previous partition and are not stored):
#   refine / always / auto  rank of every node for the three strategies
#   nodes                   (level, *index) of every node, in the same order
#   time                    simulation time
#   ms_refine / ms_always / ms_auto   partitioning time of each strategy
def frame_path(cache_dir, i):
    return os.path.join(cache_dir, f'frame_{i:06d}.npz')


def cache_is_valid(cache_dir, meta):
    meta_path = os.path.join(cache_dir, 'meta.json')
    if not os.path.exists(meta_path):
        return False
    with open(meta_path) as f:
        stored = json.load(f)
    n_stored = stored.pop('n_stored', None)
    if n_stored is None or stored != meta:
        return False
    return len(glob.glob(os.path.join(cache_dir, 'frame_*.npz'))) == n_stored


def reset_cache(cache_dir):
    """Remove a stale or partial cache before recomputing it."""
    os.makedirs(cache_dir, exist_ok=True)
    for p in glob.glob(os.path.join(cache_dir, 'frame_*.npz')) + \
             [os.path.join(cache_dir, 'meta.json')]:
        if os.path.exists(p):
            os.remove(p)


def save_frame(cache_dir, i, parts, ms, nodes, t):
    np.savez_compressed(
        frame_path(cache_dir, i),
        refine=np.array(parts[0], dtype=np.int32),
        always=np.array(parts[1], dtype=np.int32),
        auto=np.array(parts[2], dtype=np.int32),
        ms_refine=np.float32(ms[0]),
        ms_always=np.float32(ms[1]),
        ms_auto=np.float32(ms[2]),
        nodes=np.array(nodes, dtype=np.int32),
        time=np.float64(t),
    )


def load_frame(cache_dir, i):
    """Partitions and timings of plotfile i, or None if its graph did not change
    (such plotfiles are not stored; they keep the previous partition)."""
    if not os.path.exists(frame_path(cache_dir, i)):
        return None
    d = np.load(frame_path(cache_dir, i))
    parts = (d['refine'].tolist(), d['always'].tolist(), d['auto'].tolist())
    ms = (float(d['ms_refine']), float(d['ms_always']), float(d['ms_auto']))
    return parts, ms


def finish_cache(cache_dir, meta, n_stored):
    """Mark the cache complete (written last, so a partial cache is never valid)."""
    with open(os.path.join(cache_dir, 'meta.json'), 'w') as f:
        json.dump({**meta, 'n_stored': n_stored}, f, indent=2)


def cell_key(*cell):
    """Hashable cell identity (level, *index), used as dict key."""
    return tuple(cell)


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
        self._ids        = IdRegistry() # cell → vertex id (ids are never reused)
        self._cell_to_id = self._ids.cell_to_id
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
            f.write('\n'.join(self._ids.initial(nodes, edges)) + '\n')
        self._send(f'exec {seq_path}')

    def apply_diff(self, prev_nodes, prev_edges, curr_nodes, curr_edges):
        """Compute and send the graph diff between two consecutive frames."""
        cmds = self._ids.diff(prev_nodes, prev_edges, curr_nodes, curr_edges)
        for c in cmds:
            self._send(c)
        return len(cmds)

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
        f"{cfg['var']} field",
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

    fig.colorbar(im_phi,    ax=ax_phi,    fraction=0.046, pad=0.04).set_label(cfg['var'], color='white', fontsize=7)
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
        im_phi.set_data(fd['phi'])
        im_phi.set_extent([xmin, xmax, ymin, ymax])
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
            f"nodes: {fd['n_nodes']}  ("
            + "  ".join(f"L{lv}: {n}" for lv, n in enumerate(fd['n_per_level'])) + ")"
        )
        png = cfg['tmpdir'] + f'/f{idx:06d}.png'
        fig.savefig(png, dpi=cfg['dpi'], facecolor=fig.get_facecolor(),
                    bbox_inches='tight')
        out.append((idx, png))

    plt.close(fig)
    return out


def _render_worker_3d(args_tuple):
    """3D version of the render worker: isosurfaces of the field, coloured by
    value (top-left) or by the rank of the leaf cell each surface point is in."""
    indices, frames, cfg = args_tuple
    import numpy as np
    import pyvista as pv
    pv.OFF_SCREEN = True

    k = cfg['k']
    lo, hi = cfg['domain3d']
    shape_xyz = cfg['shape3d'][::-1]
    spacing = tuple((h - l) / n for l, h, n in zip(lo, hi, shape_xyz))
    levels = np.linspace(cfg['phi_min'], cfg['phi_max'], 5)[1:-1]
    cmap_part = matplotlib.colormaps.get_cmap('tab10').resampled(k)

    titles = [f"{cfg['var']} isosurfaces",
              f'refine-fast (Dyn-HeiProMap, k={k})',
              f'always repartition (Dyn-HeiProMap, k={k})',
              f'autorefine 30 (Dyn-HeiProMap, k={k})']
    pl = pv.Plotter(shape=(2, 2), off_screen=True, window_size=cfg['win3d'])
    pl.set_background('#1a1a2e')
    pl.link_views()

    out = []
    first = True
    for idx in indices:
        fd = frames[idx]
        grid = pv.ImageData(dimensions=tuple(n + 1 for n in shape_xyz),
                            origin=lo, spacing=spacing)
        grid.cell_data['v'] = fd['vol'].ravel()
        iso = grid.cell_data_to_point_data().contour(levels, scalars='v')

        # voxel under each surface point -> rank of the leaf cell there
        if iso.n_points:
            ijk = np.floor((iso.points - np.array(lo)) / np.array(spacing)).astype(int)
            ijk = np.clip(ijk, 0, np.array(shape_xyz) - 1)

        pl.clear_actors()  # clears every sub-plot
        for p, (r, c) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            pl.subplot(r, c)
            pl.add_mesh(grid.outline(), color='#888888')
            if iso.n_points:
                if p == 0:
                    pl.add_mesh(iso, scalars='v', cmap='RdYlBu_r', opacity=0.75,
                                clim=(cfg['phi_min'], cfg['phi_max']),
                                scalar_bar_args={'title': cfg['var'], 'color': 'white',
                                                 'n_labels': 4, 'height': 0.3,
                                                 'position_x': 0.88, 'position_y': 0.3,
                                                 'vertical': True})
                else:
                    key = ('refine', 'always', 'auto')[p - 1]
                    iso.point_data['rank'] = fd[f'part3d_{key}'][ijk[:, 2], ijk[:, 1], ijk[:, 0]]
                    pl.add_mesh(iso, scalars='rank', cmap=cmap_part, clim=(-0.5, k - 0.5),
                                show_scalar_bar=False, opacity=0.9)
            pl.add_text(titles[p], position='upper_edge', font_size=8, color='white')
            if p == 0:
                pl.add_text(f"t = {fd['time']:.4f} s  |  nodes: {fd['n_nodes']}  ("
                            + "  ".join(f"L{lv}: {n}" for lv, n in enumerate(fd['n_per_level']))
                            + ")", position='lower_left', font_size=7, color='white')
            else:
                bal, comm, mig = fd[('stats_refine', 'stats_always', 'stats_auto')[p - 1]]
                pl.add_text(f'bal {bal:.2f}  Σcomm {comm}  Σmig {mig}',
                            position='lower_left', font_size=7, color='white')
            if first:
                pl.view_isometric()
        first = False
        png = cfg['tmpdir'] + f'/f{idx:06d}.png'
        pl.screenshot(png)
        out.append((idx, png))

    pl.close()
    return out


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
    parser.add_argument('--play-fps', type=float, default=None,
                        help='video frame rate (default: play in simulation time)')
    parser.add_argument('--render3d', action='store_true',
                        help='render 3D isosurface views instead of 2D mid-plane slices')
    parser.add_argument('--var', default=None,
                        help='plotfile variable to show (default: first component)')
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
        if args.mpi_np <= 0:  # serial (non-MPI) build
            cmd = [binary, inputs]
        result = subprocess.run(cmd, cwd=os.path.normpath(
            os.path.join(os.path.dirname(__file__), '../..')))
        if result.returncode != 0:
            print('Simulation failed.'); return
        plotdirs = sorted(p for p in glob.glob(os.path.join(args.plt_dir, 'plt*'))
                          if re.match(r'plt\d+$', os.path.basename(p)))
        if not plotdirs:
            print(f'Simulation ran but no plotfiles found in {args.plt_dir}'); return

    # every plotfile gets a partition; only a sample of them is rendered
    all_plotdirs = plotdirs
    all_times = [read_time(pd) for pd in all_plotdirs]
    shown = select_frames(all_times, args.fps)
    shown_set = set(shown)
    times = [all_times[i] for i in shown]

    print(f'{len(all_plotdirs)} plotfiles, {len(shown)} of them rendered.')

    # common image resolution (finest level over the whole run) and domain
    shape  = finest_shape(all_plotdirs)
    hdr0   = read_plotfile(all_plotdirs[0], None, args.var)['hdr']
    if args.render3d and hdr0['ndim'] != 3:
        print('--render3d needs 3D plotfiles'); return
    domain = finest_domain(hdr0)
    var    = hdr0['varnames'][var_index(hdr0, args.var)]
    threads = min(4, mp.cpu_count())

    # ── phase 1+2: read every plotfile, partition it, accumulate statistics ───

    methods = {
        'refine': {'label': 'refine-fast',        'cmd': 'autorefine 101 fast'},
        'always': {'label': 'always-repartition', 'cmd': 'partition fast'},
        'auto':   {'label': 'autorefine-30',      'cmd': 'autorefine 30 fast'},
    }

    cdir = cache_dir_for(args.plt_dir)
    meta = cache_meta(args.dynheipromap, args.hierarchy, args.distance,
                      methods, len(all_plotdirs))
    cached = cache_is_valid(cdir, meta)

    dyns = None
    if cached:
        print(f'Partitions: loading cached partitions from {cdir}')
    else:
        print('Partitions: Dyn-HeiProMap (refine-fast + always repartition + autorefine 30)...')
        reset_cache(cdir)
        dyns = [DynHeiProMap(args.dynheipromap, args.hierarchy, args.distance, threads)
                for _ in range(3)]

    prev_maps = [{}, {}, {}]          # per method: previous partition of each cell
    acc_comm = [0, 0, 0]              # cumulative over ALL plotfiles
    acc_mig  = [0, 0, 0]
    all_phi    = []
    frames     = []
    table_rows = []
    prev_graph = None
    prev_parts = None
    n_stored   = 0          # plotfiles whose graph changed (the only ones partitioned)

    try:
        for i, pd in enumerate(all_plotdirs):
            print(f'\r  {i + 1}/{len(all_plotdirs)}', end='', flush=True)
            render = i in shown_set

            pf = read_plotfile(pd, shape, args.var, volume=args.render3d and render)
            nodes, edges, leaf = build_composite_graph(
                pf['grids'], pf['hdr']['ratios'], shape=shape)
            t = all_times[i]

            if cached:
                got = load_frame(cdir, i)
                changed = got is not None
                parts, ms = got if changed else (prev_parts, (0.0, 0.0, 0.0))
            else:
                if i == 0:
                    for d in dyns:
                        d.load_graph(nodes, edges)
                    changed = True
                else:
                    n_cmds = [d.apply_diff(prev_graph[0], prev_graph[1], nodes, edges)
                              for d in dyns]
                    changed = n_cmds[0] > 0
                if changed:
                    parts, ms = [], []
                    for d, cmd in zip(dyns, (('autorefine', 101), ('partition', None),
                                             ('autorefine', 30))):
                        t0 = time.perf_counter()
                        if i == 0 or cmd[0] == 'partition':
                            d.partition('fast')
                        else:
                            d.autorefine(cmd[1], 'fast')
                        parts.append(d.get_partition(nodes))
                        ms.append((time.perf_counter() - t0) * 1e3)
                    save_frame(cdir, i, parts, ms, nodes, t)
                    n_stored += 1
                else:
                    # graph unchanged: nothing to repartition, keep the last partition
                    parts, ms = prev_parts, (0.0, 0.0, 0.0)
            prev_parts = parts

            stats = [partition_stats(nodes, edges, parts[m], k, prev_maps[m])
                     for m in range(3)]
            for m in range(3):
                acc_comm[m] += stats[m][1]
                acc_mig[m]  += stats[m][2]

            n_per_level = [0] * pf['hdr']['nlev']
            for cell in nodes:
                n_per_level[cell[0]] += 1

            table_rows.append((
                i, t, len(nodes),
                stats[0][0], acc_comm[0], acc_mig[0], ms[0],
                stats[1][0], acc_comm[1], acc_mig[1], ms[1],
                stats[2][0], acc_comm[2], acc_mig[2], ms[2],
            ))

            if render:
                all_phi.append((pf['vol'] if args.render3d else pf['phi']).ravel())
                leaf2d = slice_2d(leaf)
                frames.append({
                    'phi':          pf['phi'],
                    'part_refine':  partition_image(leaf2d, parts[0]),
                    'part_always':  partition_image(leaf2d, parts[1]),
                    'part_auto':    partition_image(leaf2d, parts[2]),
                    'stats_refine': (stats[0][0], acc_comm[0], acc_mig[0]),
                    'stats_always': (stats[1][0], acc_comm[1], acc_mig[1]),
                    'stats_auto':   (stats[2][0], acc_comm[2], acc_mig[2]),
                    'n_nodes':      len(nodes),
                    'n_per_level':  n_per_level,
                    'time':         t,
                })
                if args.render3d:
                    frames[-1]['vol'] = pf['vol']
                    for key, part in zip(('refine', 'always', 'auto'), parts):
                        frames[-1][f'part3d_{key}'] = np.asarray(part, dtype=np.int8)[leaf]
            prev_graph = (nodes, edges)
    finally:
        if dyns:
            for d in dyns:
                d.close()

    if not cached:
        finish_cache(cdir, meta, n_stored)
        print(f'\n  partitions of the {n_stored} plotfiles whose graph changed '
              f'(of {len(all_plotdirs)}) saved to {cdir}')
    print(f'\n  {len(table_rows)} plotfiles, {len(frames)} frames to render.')

    flat    = np.concatenate(all_phi)
    phi_min = float(np.percentile(flat, 2))
    phi_max = float(np.percentile(flat, 98))
    phi_max = max(phi_max, phi_min + 1e-6)

    # ── phase 3: render ────────────────────────────────────────────────────────

    render_cfg = {
        'phi_min': phi_min, 'phi_max': phi_max,
        'domain':  domain,
        'figsize': (14, 12),
        'dpi':     100,
        'k':       k,
        'var':     var,
        'domain3d': (hdr0['prob_lo'], hdr0['prob_hi']),
        'shape3d':  shape,
        'win3d':    (1400, 1200),
    }

    n        = len(frames)
    n_workers = max(1, mp.cpu_count() - 1)
    if args.render3d:
        n_workers = min(n_workers, 8)  # one EGL context per process
    chunk    = max(1, n // n_workers)
    batches  = [list(range(j, min(j + chunk, n))) for j in range(0, n, chunk)]

    print(f'Rendering {n} frames on {n_workers} workers → {args.save}')
    with tempfile.TemporaryDirectory() as tmpdir:
        render_cfg = {**render_cfg, 'tmpdir': tmpdir}
        work = [(b, frames, render_cfg) for b in batches]
        png_map = {}
        done = 0
        with mp.Pool(n_workers) as pool:
            for result in pool.imap_unordered(
                    _render_worker_3d if args.render3d else _render_worker, work):
                for idx, png in result:
                    png_map[idx] = png
                done += len(result)
                pct = done * 100 // n
                bar = '#' * (pct // 5) + '.' * (20 - pct // 5)
                print(f'\r  [{bar}] {pct:3d}%  {done}/{n}', end='', flush=True)
        print()
        print('Assembling...')
        assemble_video(tmpdir, args.save,
                       args.play_fps or playback_fps(times, args.fps))
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

    lines = [hdr1, hdr2, sep]
    for (frm, t, nd,
         br, cr, mr, tr,
         ba, ca, ma, ta,
         bu, cu, mu, tu) in table_rows:
        lines.append(
            f"{frm:>{C['frm']}}  {t:>{C['t']}.4f}  {nd:>{C['nd']}}  "
            f"{br:>{C['bal']}.2f}  {cr:>{C['Sco']}}  {mr:>{C['Smi']}}  {tr:>{C['ms']}.1f}  "
            f"{ba:>{C['bal']}.2f}  {ca:>{C['Sco']}}  {ma:>{C['Smi']}}  {ta:>{C['ms']}.1f}  "
            f"{bu:>{C['bal']}.2f}  {cu:>{C['Sco']}}  {mu:>{C['Smi']}}  {tu:>{C['ms']}.1f}"
        )
    lines.append(sep)
    cr_f, mr_f = table_rows[-1][4], table_rows[-1][5]
    ca_f, ma_f = table_rows[-1][8], table_rows[-1][9]
    cu_f, mu_f = table_rows[-1][12], table_rows[-1][13]
    ms_r_tot = sum(r[6]  for r in table_rows)
    ms_a_tot = sum(r[10] for r in table_rows)
    ms_u_tot = sum(r[14] for r in table_rows)
    lines.append(
        f"{'TOT':>{C['frm']}}  {'':>{C['t']}}  {'':>{C['nd']}}  "
        f"{'':>{C['bal']}}  {cr_f:>{C['Sco']}}  {mr_f:>{C['Smi']}}  {ms_r_tot:>{C['ms']}.1f}  "
        f"{'':>{C['bal']}}  {ca_f:>{C['Sco']}}  {ma_f:>{C['Smi']}}  {ms_a_tot:>{C['ms']}.1f}  "
        f"{'':>{C['bal']}}  {cu_f:>{C['Sco']}}  {mu_f:>{C['Smi']}}  {ms_u_tot:>{C['ms']}.1f}"
    )
    stats_path = os.path.splitext(args.save)[0] + '_stats.txt'
    with open(stats_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nPer-plotfile statistics ({len(table_rows)} plotfiles) → {stats_path}')
    print(lines[-1])

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
