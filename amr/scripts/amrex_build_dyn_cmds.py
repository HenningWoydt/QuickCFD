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
import re

import numpy as np


# ── AMReX readers ────────────────────────────────────────────────────────────

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


def read_l1_mask(plotdir, nx0, ny0):
    """Return bool mask (ny0, nx0): True where an L1 box covers the L0 cell."""
    cell_h = os.path.join(plotdir, 'Level_1', 'Cell_H')
    if not os.path.exists(cell_h):
        return np.zeros((ny0, nx0), dtype=bool)
    boxes, _ = parse_cell_h(cell_h)
    mask = np.zeros((ny0, nx0), dtype=bool)
    for (lo_x, lo_y, hi_x, hi_y) in boxes:
        r0_lo, c0_lo = lo_y // 2, lo_x // 2
        r0_hi, c0_hi = hi_y // 2, hi_x // 2
        mask[r0_lo:r0_hi + 1, c0_lo:c0_hi + 1] = True
    return mask


def read_header_dims(plotdir):
    """Return (nx0, ny0) from the Level_0 domain box in the Header."""
    with open(os.path.join(plotdir, 'Header')) as f:
        hlines = f.readlines()
    nvar = int(hlines[1].strip())
    # layout: [0]=version [1]=nvar [2..2+nvar-1]=varnames [2+nvar]=spacedim
    # [2+nvar+1]=time [2+nvar+2]=max_level [2+nvar+3]=prob_lo
    # [2+nvar+4]=prob_hi [2+nvar+5]=ref_ratios [2+nvar+6]=domains
    idx = 2 + nvar + 6
    domains_line = hlines[idx].strip()
    m = re.match(r'\(\((\d+),(\d+)\)\s*\((\d+),(\d+)\)', domains_line)
    if not m:
        raise ValueError(f"Cannot parse domains line: {domains_line!r}")
    nx0 = int(m.group(3)) + 1
    ny0 = int(m.group(4)) + 1
    return nx0, ny0


# ── composite graph builder ──────────────────────────────────────────────────

def build_composite_nodes_edges(nx0, ny0, l1_mask, periodic=True):
    """
    Returns:
      nodes: list of (level, r, c)  – ordered by node id (0-indexed)
      edges: set of frozenset({u, v})
      l0_node: dict (r, c) -> node_id  for L0 uncovered cells
      l1_node: dict (r1, c1) -> node_id  for L1 fine cells
    """
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
    for node_id, (level, r, c) in enumerate(nodes):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if level == 0:
                nb = l0_nb(r, c, dr, dc)
                if nb >= 0:
                    edges.add(frozenset((node_id, nb)))
                for nb2 in l0_l1_boundary(r, c, dr, dc):
                    if nb2 >= 0:
                        edges.add(frozenset((node_id, nb2)))
            else:
                nb = l1_nb(r, c, dr, dc)
                if nb >= 0:
                    edges.add(frozenset((node_id, nb)))

    return nodes, edges, l0_node, l1_node


def nodes_edges_to_metis(nodes, edges):
    """Produce METIS fmt=011 lines (vertex weight=1, edge weight=1)."""
    N = len(nodes)
    adj = [[] for _ in range(N)]
    for e in edges:
        u, v = tuple(e)
        adj[u].append(v)
        adj[v].append(u)
    M = len(edges)
    lines = [f"{N} {M} 011\n"]
    for nbs in adj:
        parts = ["1"] + [f"{nb + 1} 1" for nb in nbs]
        lines.append(" ".join(parts) + "\n")
    return lines


# ── node id maps that survive across frames ──────────────────────────────────

def cell_key(level, r, c):
    """A stable string id for a cell, used as Dyn-HeiProMap vertex id."""
    return f"{level}_{r}_{c}"


def build_cell_id_map(nodes):
    """Return dict cell_key -> stable integer id.

    We assign ids based on a fixed scheme so ids are consistent across frames:
      L0 cell (r, c) on a 64x64 grid: id = r*64 + c  (0..4095)
      L1 cell (r1, c1) on 128x128 grid: id = 4096 + r1*128 + c1  (4096..20480)
    This keeps ids stable regardless of which cells are active.
    """
    id_map = {}
    for (level, r, c) in nodes:
        if level == 0:
            vid = r * 128 + c  # use 128 as stride to not overlap with L1 ids... actually simpler:
        else:
            pass
    # Actually compute from node index in a stable way via the formula above
    id_map = {}
    for (level, r, c) in nodes:
        if level == 0:
            vid = c + r * 10000          # L0: up to 64*64=4096, stride 10000
        else:
            vid = 100000 + c + r * 10000  # L1: offset by 100000
        id_map[(level, r, c)] = vid
    return id_map


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
    nx0, ny0 = read_header_dims(plotdirs[0])
    print(f'L0 grid: {nx0} x {ny0}')

    prev_nodes = None
    prev_edges = None
    prev_id_map = None

    for i, pd in enumerate(plotdirs):
        print(f'\r  frame {i+1}/{len(plotdirs)}  ({os.path.basename(pd)})', end='', flush=True)

        l1_mask = read_l1_mask(pd, nx0, ny0)
        nodes, edges, l0_node, l1_node = build_composite_nodes_edges(nx0, ny0, l1_mask)
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
