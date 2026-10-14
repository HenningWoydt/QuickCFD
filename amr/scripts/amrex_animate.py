#!/usr/bin/env python3
"""Visualize AMReX 2D plotfiles as animation or saved video."""

import argparse
import glob
import os
import re
import struct
import tempfile
import multiprocessing as mp

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize, LinearSegmentedColormap


def read_fab(filepath, offset, nx, ny):
    """Read one FAB (float64 native-endian) from a binary data file."""
    n = nx * ny
    with open(filepath, 'rb') as f:
        # skip the ascii header line at start of each fab block
        # AMReX FABs start with a small ascii header then raw doubles
        f.seek(offset)
        # read header line to find where data starts
        line = f.readline()  # e.g. "FAB ((8, (64 11 52 0 1 12 0 1023)),(8, (1)))\n"
        data = np.frombuffer(f.read(n * 8), dtype='<f8')
    return data.reshape((ny, nx))


def parse_cell_h(cell_h_path):
    """Parse Cell_H to get fab locations: list of (file, offset, lo, hi)."""
    with open(cell_h_path) as f:
        lines = f.readlines()

    # find box list: line with "(N" starts it
    fabs = []
    boxes = []
    i = 0
    while i < len(lines):
        m = re.match(r'\((\d+)', lines[i].strip())
        if m:
            n = int(m.group(1))
            for j in range(i+1, i+1+n):
                bm = re.match(r'\(\((\d+),(\d+)\)\s*\((\d+),(\d+)\)', lines[j].strip())
                if bm:
                    boxes.append((int(bm.group(1)), int(bm.group(2)),
                                  int(bm.group(3)), int(bm.group(4))))
            i += n + 2
            break
        i += 1

    # find FabOnDisk lines
    fab_entries = []
    for line in lines:
        m = re.match(r'FabOnDisk:\s+(\S+)\s+(\d+)', line.strip())
        if m:
            fab_entries.append((m.group(1), int(m.group(2))))

    return boxes, fab_entries


def read_level(plotdir, level):
    """Read all FABs for a level, return (data_2d, x0, y0, dx, dy)."""
    header_path = os.path.join(plotdir, 'Header')
    cell_h = os.path.join(plotdir, f'Level_{level}', 'Cell_H')
    level_dir = os.path.join(plotdir, f'Level_{level}')

    if not os.path.exists(cell_h):
        return None

    # parse global header for dx at this level
    # Format: version, nvar, varnames..., spacedim, time, max_level,
    #         prob_lo, prob_hi, ref_ratios(1 line), domains(1 line), stepping(1 line), dx per level
    with open(header_path) as f:
        hlines = f.readlines()

    nvar = int(hlines[1].strip())
    idx = 2 + nvar          # after varnames
    # spacedim
    idx += 1                # skip spacedim
    # time
    idx += 1                # skip time
    # max_level
    idx += 1                # skip max_level
    prob_lo = list(map(float, hlines[idx].split())); idx += 1
    prob_hi = list(map(float, hlines[idx].split())); idx += 1
    idx += 1                # ref_ratios (one line)
    idx += 1                # domains (one line)
    idx += 1                # stepping (one line)
    # now dx lines, one per level
    dx_line = hlines[idx + level].split()
    dx, dy = float(dx_line[0]), float(dx_line[1])

    boxes, fab_entries = parse_cell_h(cell_h)

    # figure out full level domain size
    all_x1 = max(b[2] for b in boxes) + 1
    all_y1 = max(b[3] for b in boxes) + 1
    grid = np.zeros((all_y1, all_x1))

    for (lo_x, lo_y, hi_x, hi_y), (fname, offset) in zip(boxes, fab_entries):
        nx = hi_x - lo_x + 1
        ny = hi_y - lo_y + 1
        fpath = os.path.join(level_dir, fname)
        fab = read_fab(fpath, offset, nx, ny)
        grid[lo_y:hi_y+1, lo_x:hi_x+1] = fab

    x0, y0 = prob_lo[0], prob_lo[1]
    return grid, x0, y0, dx, dy


def read_plotfile(plotdir):
    """Read coarsest + finest level, return composited grid."""
    try:
        result0 = read_level(plotdir, 0)
        result1 = read_level(plotdir, 1)
    except Exception as e:
        print(f"  Warning reading {plotdir}: {e}")
        return None

    if result0 is None:
        return None

    grid, x0, y0, dx, dy = result0

    # overlay level 1 if available (higher resolution)
    if result1 is not None:
        g1, x0_1, y0_1, dx1, dy1 = result1
        # scale factor
        scale = round(dx / dx1)
        ny0, nx0 = grid.shape
        ny1, nx1 = g1.shape
        # coarsen level1 back to level0 resolution for composite display
        # just use level1 data as-is for the fine composite image
        return {'coarse': (grid, x0, y0, dx, dy),
                'fine':   (g1,  x0_1, y0_1, dx1, dy1)}

    return {'coarse': (grid, x0, y0, dx, dy), 'fine': None}


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
        # use fine grid if available, else coarse
        src = fd['fine'] if fd['fine'] is not None else fd['coarse']
        grid, x0, y0, dx, dy = src
        ny, nx = grid.shape
        extent = [x0, x0 + nx*dx, y0, y0 + ny*dy]
        im.set_data(grid)
        im.set_extent(extent)
        title.set_text(
            f"t = {fd['time']:.4f} s  |  "
            f"L0: {fd['cells_l0']} cells  |  "
            f"L1: {fd['cells_l1']} cells  |  "
            f"total: {fd['cells_l0'] + fd['cells_l1']}"
        )
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
    args = parser.parse_args()

    plotdirs = sorted(glob.glob(os.path.join(args.plt_dir, 'plt?????')))
    if not plotdirs:
        print(f'No plt##### directories found in {args.plt_dir}'); return

    print(f'Found {len(plotdirs)} plotfiles total.')

    # read simulation times from headers
    times = []
    for pd in plotdirs:
        try:
            with open(os.path.join(pd, 'Header')) as f:
                hlines = f.readlines()
            nvar = int(hlines[1].strip())
            t = float(hlines[2 + nvar + 1].strip())  # spacedim then time
            times.append(t)
        except Exception:
            times.append(0.0)

    # subsample to target fps: pick frames closest to evenly-spaced time points
    t_max = max(times) if times else 0.0
    t_min = min(times) if times else 0.0
    n_target = int((t_max - t_min) * args.fps) + 1
    if len(plotdirs) > n_target:
        targets = np.linspace(t_min, t_max, n_target)
        times_arr = np.array(times)
        selected = sorted(set(int(np.argmin(np.abs(times_arr - t))) for t in targets))
        plotdirs = [plotdirs[i] for i in selected]
        times    = [times[i]    for i in selected]
        print(f'Subsampled to {len(plotdirs)} frames at {args.fps} fps.')

    frames = []
    all_vals = []
    for i, (pd, t) in enumerate(zip(plotdirs, times)):
        print(f'\r  reading {i+1}/{len(plotdirs)}', end='', flush=True)
        data = read_plotfile(pd)
        if data is None:
            continue
        src = data['fine'] if data['fine'] is not None else data['coarse']
        all_vals.append(src[0].ravel())
        g0 = data['coarse'][0]
        g1 = data['fine'][0] if data['fine'] is not None else np.zeros((0,0))
        frames.append({**data, 'frame': i, 'time': t,
                        'cells_l0': g0.size, 'cells_l1': g1.size})
    print()

    if not frames:
        print('No valid frames.'); return

    flat = np.concatenate(all_vals)
    vmin = float(np.percentile(flat, 2))
    vmax = float(np.percentile(flat, 98))
    vmax = max(vmax, vmin + 1e-6)

    # domain from coarsest level of first frame
    g, x0, y0, dx, dy = frames[0]['coarse']
    ny, nx = g.shape
    domain = (x0, x0 + nx*dx, y0, y0 + ny*dy)
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
        import imageio, imageio.v2 as iio2
        ext = os.path.splitext(args.save)[1].lower()
        extra = {'loop': 0} if ext == '.gif' else {
            'codec': 'libx264', 'macro_block_size': 1,
            'output_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p']}

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
            with imageio.get_writer(args.save, fps=args.fps, **extra) as writer:
                for i in range(n):
                    writer.append_data(iio2.imread(png_map[i]))
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

    src0 = frames[0]['fine'] if frames[0]['fine'] is not None else frames[0]['coarse']
    g0, x0, y0, dx, dy = src0
    ny0, nx0 = g0.shape
    im = ax.imshow(g0, origin='lower', cmap='RdYlBu_r', norm=norm,
                   extent=[x0, x0+nx0*dx, y0, y0+ny0*dy], interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('phi', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    title = ax.set_title('', color='white', pad=10)

    def update(i):
        fd = frames[i]
        src = fd['fine'] if fd['fine'] is not None else fd['coarse']
        g, x0, y0, dx, dy = src
        ny, nx = g.shape
        im.set_data(g)
        im.set_extent([x0, x0+nx*dx, y0, y0+ny*dy])
        title.set_text(
            f"t = {fd['time']:.4f} s  |  "
            f"L0: {fd['cells_l0']} cells  |  "
            f"L1: {fd['cells_l1']} cells  |  "
            f"total: {fd['cells_l0'] + fd['cells_l1']}"
        )
        return im, title

    ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                  interval=1000 // args.fps, blit=True)
    plt.show()


if __name__ == '__main__':
    main()
