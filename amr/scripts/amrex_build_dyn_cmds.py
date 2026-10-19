#!/usr/bin/env python3
"""
Write the Dyn-HeiProMap command files that recreate how the graph of an AMReX run
changes over time: one file per plotfile (every plotfile, not a sample).

For each run, into amr/dyn_heipromap_cmds/<run>/ (gitignored, regenerate with this script):
  frame_000000.my_seq   builds the graph of the first plotfile (+v / +e)
  frame_000001.my_seq   changes that turn the graph of plotfile 0 into plotfile 1
  ...                   (-e, -v, +v, +e), one file per following plotfile
  frames.csv            frame, plotfile, time, vertices, edges (after the frame)
  vertices.csv          vertex id -> cell (level, index), first and last frame it exists

Executing frame_000000.my_seq, frame_000001.my_seq, ... in order in Dyn-HeiProMap
(`exec <file>`) reproduces exactly the graph sequence amrex_dynmap_video.py partitions:
same vertex ids, same command order. Vertex ids are handed out in order of first
appearance and never reused. Vertices and edges all have weight 1.

Usage:
  python3 amr/scripts/amrex_build_dyn_cmds.py                   # every run in amr/plotfiles
  python3 amr/scripts/amrex_build_dyn_cmds.py --run cns_sod advection_vortex
  python3 amr/scripts/amrex_build_dyn_cmds.py --plt-dir amr/plotfiles/advection_vortex \\
      --out-dir some/where
"""

import argparse
import glob
import multiprocessing as mp
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amrex_common import (read_plotfile, read_time, finest_shape,
                          build_composite_graph, IdRegistry)

AMR_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def plotfile_dirs(plt_dir):
    return sorted(p for p in glob.glob(os.path.join(plt_dir, 'plt*'))
                  if re.match(r'plt\d+$', os.path.basename(p)))


def build_run(job):
    plt_dir, out_dir = job
    name = os.path.basename(os.path.normpath(plt_dir))
    plotdirs = plotfile_dirs(plt_dir)
    if not plotdirs:
        return f'{name}: no plotfiles in {plt_dir}'
    os.makedirs(out_dir, exist_ok=True)
    shape = finest_shape(plotdirs)

    reg = IdRegistry()
    first, last, cell_of = {}, {}, {}     # per vertex id
    prev = None
    frames_csv = ['frame,plotfile,time,vertices,edges']

    for i, pd in enumerate(plotdirs):
        pf = read_plotfile(pd, shape)
        nodes, edges, _ = build_composite_graph(pf['grids'], pf['hdr']['ratios'], shape=shape)
        t = read_time(pd)

        before = reg.next_id
        cmds = reg.initial(nodes, edges) if prev is None else reg.diff(*prev, nodes, edges)
        for cell, vid in reg.cell_to_id.items():
            if vid >= before:
                first[vid] = i
                cell_of[vid] = cell
        for c in cmds:
            if c.startswith('-v '):
                last[int(c.split()[1])] = i - 1       # gone from this frame on
        n_add_v = sum(c.startswith('+v ') for c in cmds)
        n_del_v = sum(c.startswith('-v ') for c in cmds)
        n_add_e = sum(c.startswith('+e ') for c in cmds)
        n_del_e = sum(c.startswith('-e ') for c in cmds)

        with open(os.path.join(out_dir, f'frame_{i:06d}.my_seq'), 'w') as f:
            f.write(f'# {name}  frame {i}  {os.path.basename(pd)}  t={t:.10g}\n')
            f.write(f'# +{n_add_v} -{n_del_v} vertices, +{n_add_e} -{n_del_e} edges'
                    f'  ->  {len(nodes)} vertices, {len(edges)} edges\n')
            f.write('\n'.join(cmds) + '\n')
        frames_csv.append(f'{i},{os.path.basename(pd)},{t:.10g},{len(nodes)},{len(edges)}')
        prev = (nodes, edges)

    with open(os.path.join(out_dir, 'frames.csv'), 'w') as f:
        f.write('\n'.join(frames_csv) + '\n')
    ndim = len(next(iter(cell_of.values()))) - 1
    axes = ['x', 'y', 'z'][:ndim][::-1]            # index order is (z, y, x) / (y, x)
    with open(os.path.join(out_dir, 'vertices.csv'), 'w') as f:
        f.write('id,level,' + ','.join(f'i_{a}' for a in axes) + ',first_frame,last_frame\n')
        for vid in sorted(cell_of):
            cell = cell_of[vid]
            f.write(f'{vid},{cell[0]},' + ','.join(str(v) for v in cell[1:])
                    + f',{first[vid]},{last.get(vid, len(plotdirs) - 1)}\n')
    return (f'{name}: {len(plotdirs)} frames, {reg.next_id} vertex ids '
            f'→ {os.path.relpath(out_dir, os.path.dirname(AMR_DIR))}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', nargs='*', default=None,
                        help='run names under amr/plotfiles (default: all)')
    parser.add_argument('--plt-dir', default=None,
                        help='one plotfile directory instead of --run')
    parser.add_argument('--out-dir', default=None,
                        help='output directory (only with --plt-dir; default '
                             'amr/dyn_heipromap_cmds/<run>)')
    args = parser.parse_args()

    if args.plt_dir:
        name = os.path.basename(os.path.normpath(args.plt_dir))
        jobs = [(args.plt_dir, args.out_dir or os.path.join(AMR_DIR, 'dyn_heipromap_cmds', name))]
    else:
        base = os.path.join(AMR_DIR, 'plotfiles')
        names = args.run or sorted(os.path.basename(d)
                                   for d in glob.glob(os.path.join(base, '*')) if os.path.isdir(d))
        jobs = [(os.path.join(base, n), os.path.join(AMR_DIR, 'dyn_heipromap_cmds', n))
                for n in names]

    with mp.Pool(min(len(jobs), 9)) as pool:
        for line in pool.imap_unordered(build_run, jobs):
            print(line, flush=True)


if __name__ == '__main__':
    main()
