#!/usr/bin/env python3
"""Visualize AMReX 2D plotfiles as animation or saved video."""

import argparse
import glob
import os
import re
import sys
import tempfile
import multiprocessing as mp

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize, LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amrex_common import (read_plotfile, read_time, finest_shape, finest_domain,
                          select_frames, playback_fps, assemble_video)


def frame_title(fd):
    levels = "  |  ".join(f"L{lv}: {n} cells" for lv, n in enumerate(fd['cells']))
    return f"t = {fd['time']:.4f} s  |  {levels}  |  total: {sum(fd['cells'])}"


def _render_worker(args_tuple):
    indices, frames, render_cfg = args_tuple
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    cfg = render_cfg
    norm = Normalize(vmin=cfg['vmin'], vmax=cfg['vmax'])
    xmin, xmax, ymin, ymax = cfg['domain']

    fig, ax = plt.subplots(figsize=cfg['figsize'])
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_facecolor('#111111')
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_xlabel('x (m)', color='white')
    ax.set_ylabel('y (m)', color='white')
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_edgecolor('#444444')
    title = ax.set_title('', color='white', pad=10)

    im = ax.imshow(np.zeros((2, 2)), origin='lower', cmap='RdYlBu_r',
                   norm=norm, extent=[xmin, xmax, ymin, ymax], interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('phi', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    out = []
    for idx in indices:
        fd = frames[idx]
        im.set_data(fd['phi'])
        im.set_extent([xmin, xmax, ymin, ymax])
        title.set_text(frame_title(fd))
        png = cfg['tmpdir'] + f'/f{idx:06d}.png'
        fig.savefig(png, dpi=cfg['dpi'], facecolor=fig.get_facecolor())
        out.append((idx, png))

    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plt-dir', default='.',
                        help='Directory containing plt##### folders')
    parser.add_argument('--save', default='', help='Output .mp4 or .gif')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--var', default=None,
                        help='plotfile variable to show (default: first component)')
    args = parser.parse_args()

    plotdirs = sorted(glob.glob(os.path.join(args.plt_dir, 'plt?????')))
    if not plotdirs:
        print(f'No plt##### directories found in {args.plt_dir}'); return

    print(f'Found {len(plotdirs)} plotfiles total.')

    # read simulation times from headers, subsample to the target fps
    times = [read_time(pd) for pd in plotdirs]
    selected = select_frames(times, args.fps)
    if len(selected) < len(plotdirs):
        plotdirs = [plotdirs[i] for i in selected]
        times    = [times[i]    for i in selected]
        print(f'Subsampled to {len(plotdirs)} frames at {args.fps} fps.')

    shape = finest_shape(plotdirs)
    frames = []
    all_vals = []
    for i, (pd, t) in enumerate(zip(plotdirs, times)):
        print(f'\r  reading {i+1}/{len(plotdirs)}', end='', flush=True)
        data = read_plotfile(pd, shape, args.var)
        if data is None:
            continue
        all_vals.append(data['phi'].ravel())
        frames.append({'phi': data['phi'], 'frame': i, 'time': t,
                       'cells': [int((~np.isnan(g)).sum()) for g in data['grids']]})
        domain = finest_domain(data['hdr'])
    print()

    if not frames:
        print('No valid frames.'); return

    flat = np.concatenate(all_vals)
    vmin = float(np.percentile(flat, 2))
    vmax = float(np.percentile(flat, 98))
    vmax = max(vmax, vmin + 1e-6)

    xmin, xmax, ymin, ymax = domain
    aspect = (xmax - xmin) / max(ymax - ymin, 1e-6)
    figsize = (9, 9/aspect) if aspect >= 1 else (9*aspect, 9)

    render_cfg = {
        'vmin': vmin, 'vmax': vmax,
        'domain': domain,
        'figsize': figsize,
        'dpi': 100,
    }

    if args.save:

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
        return

    # interactive
    matplotlib.use('TkAgg')
    import matplotlib.animation as animation

    norm = Normalize(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*domain[:2]); ax.set_ylim(*domain[2:])
    ax.set_aspect('equal')
    ax.set_facecolor('#111111')
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_xlabel('x (m)', color='white')
    ax.set_ylabel('y (m)', color='white')
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_edgecolor('#444444')

    im = ax.imshow(frames[0]['phi'], origin='lower', cmap='RdYlBu_r', norm=norm,
                   extent=[xmin, xmax, ymin, ymax], interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('phi', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    title = ax.set_title('', color='white', pad=10)

    def update(i):
        fd = frames[i]
        im.set_data(fd['phi'])
        title.set_text(frame_title(fd))
        return im, title

    ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                  interval=1000 // args.fps, blit=True)
    plt.show()


if __name__ == '__main__':
    main()
