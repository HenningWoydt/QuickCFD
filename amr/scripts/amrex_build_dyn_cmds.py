#!/usr/bin/env python3
"""
Build a Dyn-HeiProMap input sequence from an AMReX plotfiles folder.

Outputs (into --out-dir):
  initial.graph          METIS graph for frame 0 (for --graph arg to Dyn-HeiProMap)
  frame_000000.my_seq    commands for frame 0: set-hierarchy, partition, save-partition
  frame_000001.my_seq    diff from frame 0→1: +v/-v/+e/-e, autorefine, save-partition
  ...
  frame_NNNNNN.my_seq    diff from N-1→N
  partitions/            directory where save-partition commands write to

Usage:
  python3 amrex_build_dyn_cmds.py \\
      --plt-dir amr/plotfiles/advection_periodic \\
      --out-dir amr/dyn_cmds/advection_periodic \\
      --hierarchy 4:2 --distance 1:10 \\
      --autorefine 20
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amrex_common import (read_plotfile, finest_shape, build_composite_graph,
                          nodes_edges_to_metis)


# ── stable ids ───────────────────────────────────────────────────────────────

def build_cell_id_map(nodes):
    """Return dict (level, r, c) -> stable integer id, independent of which
    cells are active in a given frame (stride 10000 per row, 10**7 per level)."""
    return {(lv, r, c): lv * 10_000_000 + r * 10_000 + c for (lv, r, c) in nodes}


# ── diff between two frames ──────────────────────────────────────────────────

def compute_diff(prev_nodes, prev_edges, curr_nodes, curr_edges, id_map_prev, id_map_curr):
    """
    Returns lists of Dyn-HeiProMap commands for the transition prev→curr.
    Uses stable cell ids.
    """
    prev_set = set((lv, r, c) for lv, r, c in prev_nodes)
    curr_set = set((lv, r, c) for lv, r, c in curr_nodes)

    added_cells   = curr_set - prev_set
    removed_cells = prev_set - curr_set

    # edge sets keyed by stable (vid_u, vid_v) pairs
    def edge_vid_set(nodes, edges, id_map):
        result = set()
        node_list = list(nodes)  # nodes is already ordered
        # build local_id -> vid
        local_to_vid = {}
        for local_id, cell in enumerate(node_list):
            local_to_vid[local_id] = id_map[cell]
        for e in edges:
            u, v = tuple(e)
            eu, ev = local_to_vid[u], local_to_vid[v]
            result.add((min(eu, ev), max(eu, ev)))
        return result

    prev_edge_vids = edge_vid_set(prev_nodes, prev_edges, id_map_prev)
    curr_edge_vids = edge_vid_set(curr_nodes, curr_edges, id_map_curr)

    added_edges   = curr_edge_vids - prev_edge_vids
    removed_edges = prev_edge_vids - curr_edge_vids

    cmds = []
    for cell in removed_cells:
        cmds.append(f"-v {id_map_prev[cell]}")
    for cell in added_cells:
        cmds.append(f"+v {id_map_curr[cell]} 1")
    for (u, v) in removed_edges:
        # only emit if both endpoints still exist in current graph
        cmds.append(f"-e {u} {v}")
    for (u, v) in added_edges:
        cmds.append(f"+e {u} {v} 1")

    return cmds


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--plt-dir',    required=True,
                        help='Path to experiment plotfiles folder (contains plt#####)')
    parser.add_argument('--out-dir',    required=True,
                        help='Output directory for graph + .my_seq files')
    parser.add_argument('--hierarchy',  default='4:2')
    parser.add_argument('--distance',   default='1:10')
    parser.add_argument('--config',     default='fast',
                        choices=['fast', 'eco', 'strong', 'super-strong'],
                        help='Config for initial partition (default: fast)')
    args = parser.parse_args()

    plotdirs = sorted(glob.glob(os.path.join(args.plt_dir, 'plt?????')))
    if not plotdirs:
        print(f'No plt##### found in {args.plt_dir}')
        return

    print(f'Found {len(plotdirs)} plotfiles in {args.plt_dir}')

    os.makedirs(args.out_dir, exist_ok=True)

    # determine grid dimensions from first frame
    shape = finest_shape(plotdirs)
    print(f'finest-level image: {shape[1]} x {shape[0]}')

    prev_nodes = None
    prev_edges = None
    prev_id_map = None

    for i, pd in enumerate(plotdirs):
        print(f'\r  frame {i+1}/{len(plotdirs)}  ({os.path.basename(pd)})', end='', flush=True)

        pf = read_plotfile(pd, shape)
        nodes, edges, _ = build_composite_graph(pf['grids'], pf['hdr']['ratios'], shape=shape)
        id_map = build_cell_id_map(nodes)

        seq_path = os.path.join(args.out_dir, f'frame_{i:06d}.my_seq')

        with open(seq_path, 'w') as f:
            if i == 0:
                # frame 0: full initial partition
                f.write(f"# frame 0 – initial partition\n")
                f.write(f"set-hierarchy {args.hierarchy} {args.distance}\n")
                f.write(f"partition {args.config}\n")
            else:
                # frame i: incremental graph updates only
                diff_cmds = compute_diff(
                    prev_nodes, prev_edges,
                    nodes,       edges,
                    prev_id_map, id_map,
                )
                n_added   = sum(1 for c in diff_cmds if c.startswith('+'))
                n_removed = sum(1 for c in diff_cmds if c.startswith('-'))
                f.write(f"# frame {i}  (+{n_added} / -{n_removed} ops)\n")
                for cmd in diff_cmds:
                    f.write(cmd + "\n")

        if i == 0:
            graph_path = os.path.join(args.out_dir, 'initial.graph')
            metis_lines = nodes_edges_to_metis(nodes, edges)
            with open(graph_path, 'w') as f:
                f.writelines(metis_lines)
            print(f'\n  initial graph: {len(nodes)} nodes, {len(edges)} edges → {graph_path}')

        prev_nodes  = nodes
        prev_edges  = edges
        prev_id_map = id_map

    print(f'\nDone. {len(plotdirs)} .my_seq files written to {args.out_dir}')
    print(f'\nTo run with Dyn-HeiProMap:')
    print(f'  Dyn-HeiProMap --graph {os.path.join(args.out_dir, "initial.graph")} \\')
    print(f'    --hierarchy {args.hierarchy} --distance {args.distance} \\')
    print(f'    --threads 4 --imbalance 0.03')
    print(f'  exec frame_000000.my_seq  # initial partition')
    print(f'  exec frame_000001.my_seq  # graph updates only, wire in your own logic')


if __name__ == '__main__':
    main()
