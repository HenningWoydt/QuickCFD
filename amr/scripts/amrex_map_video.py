#!/usr/bin/env python3
"""
Build a communication graph from AMReX plt00000, run HeiProMap, then render
a side-by-side video: left = phi field, right = partition coloring.
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import tempfile
import multiprocessing as mp

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, BoundaryNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amrex_common import (read_plotfile, read_time, finest_shape, finest_domain,
                          select_frames, playback_fps, assemble_video, upsample_to)

# ── build METIS graph from Level 0 grid ──────────────────────────────────────

def build_metis_graph(nx, ny, periodic):
    """
    Build a 4-connected grid graph. Returns lines of METIS file.
    Nodes are 1-indexed. All node weights = 1, all edge weights = 1.
    fmt = 011 means: vertex weights present, edge weights present.
    """
    N = nx * ny
    # Each cell has 4 neighbors; edges counted once → 2*nx*ny edges if periodic
    # With periodic BCs all cells have exactly 4 neighbors
    lines = []
    adj = []  # list of neighbor lists

    def idx(r, c):
        if periodic:
            return (r % ny) * nx + (c % nx)
        else:
            if r < 0 or r >= ny or c < 0 or c >= nx:
                return -1
            return r * nx + c

    for r in range(ny):
        for c in range(nx):
            neighbors = []
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nb = idx(r+dr, c+dc)
                if nb >= 0:
                    neighbors.append(nb + 1)  # 1-indexed
            adj.append(neighbors)

    # count edges (each counted once)
    n_edges = sum(len(a) for a in adj) // 2

    # fmt=011: has vertex weights and edge weights
    lines.append(f"{N} {n_edges} 011\n")
    for neighbors in adj:
        # vertex weight = 1, then neighbor edgeweight pairs
        parts = ["1"]
        for nb in neighbors:
            parts.append(f"{nb} 1")
        lines.append(" ".join(parts) + "\n")

    return lines


# ── render worker ─────────────────────────────────────────────────────────────

def _render_worker(args_tuple):
    indices, frames, render_cfg = args_tuple
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, BoundaryNorm
    import numpy as np

    cfg = render_cfg
    norm_phi = Normalize(vmin=cfg['phi_min'], vmax=cfg['phi_max'])
    xmin, xmax, ymin, ymax = cfg['domain']
    k = cfg['k']  # number of partitions

    cmap_part = matplotlib.colormaps['tab10'].resampled(k)
    bounds = np.arange(-0.5, k + 0.5, 1)
    norm_part = BoundaryNorm(bounds, cmap_part.N)

    fig, (ax_phi, ax_part) = plt.subplots(1, 2, figsize=cfg['figsize'])
    fig.patch.set_facecolor('#1a1a2e')

    for ax, title_str in [(ax_phi, 'phi field'), (ax_part, f'partition (k={k})')]:
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        ax.set_facecolor('#111111')
        ax.set_xlabel('x (m)', color='white')
        ax.set_ylabel('y (m)', color='white')
        ax.tick_params(colors='white')
        for sp in ax.spines.values(): sp.set_edgecolor('#444444')
        ax.set_title(title_str, color='white', pad=6)

    im_phi  = ax_phi.imshow(np.zeros((2,2)), origin='lower', cmap='RdYlBu_r',
                             norm=norm_phi, extent=[xmin,xmax,ymin,ymax],
                             interpolation='nearest')
    im_part = ax_part.imshow(np.zeros((2,2)), origin='lower', cmap=cmap_part,
                              norm=norm_part, extent=[xmin,xmax,ymin,ymax],
                              interpolation='nearest')

    cbar_phi  = fig.colorbar(im_phi,  ax=ax_phi,  fraction=0.046, pad=0.04)
    cbar_part = fig.colorbar(im_part, ax=ax_part, fraction=0.046, pad=0.04,
                              ticks=np.arange(k))
    cbar_phi.set_label('phi', color='white')
    cbar_part.set_label('rank', color='white')
    for cbar in [cbar_phi, cbar_part]:
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    suptitle = fig.suptitle('', color='white', y=1.01, fontsize=10)

    out = []
    for idx in indices:
        fd = frames[idx]
        im_phi.set_data(fd['phi'])
        im_phi.set_extent([xmin, xmax, ymin, ymax])

        part_grid = fd['partition_grid']
        im_part.set_data(part_grid)
        im_part.set_extent([xmin, xmax, ymin, ymax])

        suptitle.set_text(
            f"t = {fd['time']:.4f} s  |  "
            + "  ".join(f"L{lv}: {n}" for lv, n in enumerate(fd['cells']))
            + f"  total: {sum(fd['cells'])}"
        )

        png = cfg['tmpdir'] + f'/f{idx:06d}.png'
        fig.savefig(png, dpi=cfg['dpi'], facecolor=fig.get_facecolor(),
                    bbox_inches='tight')
        out.append((idx, png))

    plt.close(fig)
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plt-dir',    default='.',
                        help='Directory containing plt##### folders')
    parser.add_argument('--heipromap',  default='/home/henning/Projects/HeiProMap/build/HeiProMap')
    parser.add_argument('--hierarchy',  default='4:2')
    parser.add_argument('--distance',   default='1:10')
    parser.add_argument('--config',     default='eco')
    parser.add_argument('--save',       default='amrex_mapped.mp4')
    parser.add_argument('--fps',        type=int, default=30)
    args = parser.parse_args()

    plotdirs = sorted(glob.glob(os.path.join(args.plt_dir, 'plt?????')))
    if not plotdirs:
        print(f'No plt##### directories found in {args.plt_dir}'); return

    # ── step 1: read frame 0 to get grid dimensions ──────────────────────────
    print('Reading plt00000 for graph construction...')
    pf0 = read_plotfile(plotdirs[0])
    if pf0 is None:
        print('Could not read Level_0'); return
    if pf0['hdr']['ndim'] != 2:
        print('amrex_map_video.py only supports 2D plotfiles'); return
    ny0, nx0 = pf0['grids'][0].shape
    print(f'  Level 0 grid: {nx0}×{ny0} = {nx0*ny0} cells')

    # ── step 2: build METIS graph ─────────────────────────────────────────────
    print('Building communication graph...')
    metis_lines = build_metis_graph(nx0, ny0, periodic=True)

    scratchdir = '/tmp/amrex_heipromap'
    os.makedirs(scratchdir, exist_ok=True)
    graph_path     = os.path.join(scratchdir, 'grid.graph')
    partition_path = os.path.join(scratchdir, 'grid.partition')

    with open(graph_path, 'w') as f:
        f.writelines(metis_lines)
    print(f'  Graph written: {graph_path}  ({nx0*ny0} nodes, {metis_lines[0].split()[1]} edges)')

    # ── step 3: run HeiProMap ─────────────────────────────────────────────────
    k = 1
    for part in args.hierarchy.split(':'):
        k *= int(part)
    print(f'Running HeiProMap (k={k}, hierarchy={args.hierarchy})...')

    cmd = [
        args.heipromap,
        '--graph',     graph_path,
        '--mapping',   partition_path,
        '--hierarchy', args.hierarchy,
        '--distance',  args.distance,
        '--config',    args.config,
        '--threads',   str(min(8, mp.cpu_count())),
        '--imbalance', '0.03',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print('HeiProMap failed:')
        print(result.stderr)
        return
    print(result.stdout.strip().split('\n')[-1])  # print total time line

    # ── step 4: read partition ────────────────────────────────────────────────
    with open(partition_path) as f:
        partition = [int(l.strip()) for l in f if l.strip()]
    partition = np.array(partition, dtype=np.int32).reshape((ny0, nx0))
    print(f'  Partition read: {len(np.unique(partition))} unique ranks')

    # ── step 5: load all frames ───────────────────────────────────────────────
    print(f'Found {len(plotdirs)} plotfiles total.')

    times = [read_time(pd) for pd in plotdirs]
    selected = select_frames(times, args.fps)
    if len(selected) < len(plotdirs):
        plotdirs = [plotdirs[i] for i in selected]
        times    = [times[i]    for i in selected]
        print(f'Subsampled to {len(plotdirs)} frames at {args.fps} fps.')

    from concurrent.futures import ThreadPoolExecutor

    shape = finest_shape(plotdirs)
    part_fine = upsample_to(partition, shape)

    def _load(args_):
        pd, t, i = args_
        pf = read_plotfile(pd, shape)
        if pf is None:
            return None
        return {
            'phi': pf['phi'],
            'partition_grid': part_fine,
            'cells': [int((~np.isnan(g)).sum()) for g in pf['grids']],
            'frame': i, 'time': t,
        }

    print('Loading frames...')
    frames = []
    all_phi = []
    with ThreadPoolExecutor() as pool:
        load_args = [(pd, t, i) for i, (pd, t) in enumerate(zip(plotdirs, times))]
        for i, fd in enumerate(pool.map(_load, load_args)):
            if fd is not None:
                all_phi.append(fd['phi'].ravel())
                frames.append(fd)
            print(f'\r  {i+1}/{len(plotdirs)}', end='', flush=True)
    print()

    flat = np.concatenate(all_phi)
    phi_min = float(np.percentile(flat, 2))
    phi_max = float(np.percentile(flat, 98))
    phi_max = max(phi_max, phi_min + 1e-6)

    domain = finest_domain(pf0["hdr"])
    xmin, xmax, ymin, ymax = domain

    render_cfg = {
        'phi_min': phi_min, 'phi_max': phi_max,
        'domain':  domain,
        'figsize': (14, 6),
        'dpi':     100,
        'k':       k,
    }

    # ── step 6: render ────────────────────────────────────────────────────────
    n = len(frames)
    n_workers = max(1, mp.cpu_count() - 1)
    chunk = max(1, n // n_workers)
    batches = [list(range(i, min(i+chunk, n))) for i in range(0, n, chunk)]

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
        assemble_video(tmpdir, args.save, playback_fps(times, args.fps))
    print(f'Saved → {args.save}')


if __name__ == '__main__':
    main()
