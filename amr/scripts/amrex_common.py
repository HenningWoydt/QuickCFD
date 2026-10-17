"""Shared AMReX plotfile readers, composite-mesh graph builder and video helpers.

Works for any number of AMR levels (max_level), any per-level refinement ratio,
2D and 3D plotfiles and any number of components. Arrays are indexed in numpy
order, i.e. (z, y, x) in 3D and (y, x) in 2D. For 3D runs the images used for
rendering are the mid-plane slice of the domain; the graph is the full 3D mesh.
"""

import os
import re
import subprocess

import numpy as np

# Bump when the node ordering or edge set produced by build_composite_graph
# changes, so cached partitions are not reused against a different graph.
GRAPH_VERSION = 4


# ── plotfile readers ─────────────────────────────────────────────────────────

BOX_RE = re.compile(r'\(\(([\d,]+)\)\s*\(([\d,]+)\)')


def parse_cell_h(cell_h_path):
    """Return (boxes, fab_entries): boxes as (lo, hi) tuples in AMReX (x, y, z)
    order, fab_entries as (data file, byte offset)."""
    with open(cell_h_path) as f:
        lines = f.readlines()
    boxes, fab_entries = [], []
    i = 0
    while i < len(lines):
        m = re.match(r'\((\d+)', lines[i].strip())
        if m:
            n = int(m.group(1))
            for j in range(i + 1, i + 1 + n):
                bm = BOX_RE.match(lines[j].strip())
                if bm:
                    boxes.append((tuple(int(v) for v in bm.group(1).split(',')),
                                  tuple(int(v) for v in bm.group(2).split(','))))
            i += n + 2
            break
        i += 1
    for line in lines:
        m = re.match(r'FabOnDisk:\s+(\S+)\s+(\d+)', line.strip())
        if m:
            fab_entries.append((m.group(1), int(m.group(2))))
    return boxes, fab_entries


def read_fab(filepath, offset, shape, ncomp, comp):
    """Read one component of a FAB (float64) as an array of the given numpy
    shape; components are stored one after the other."""
    n = int(np.prod(shape))
    with open(filepath, 'rb') as f:
        f.seek(offset)
        f.readline()  # ascii fab header line
        f.seek(comp * n * 8, os.SEEK_CUR)
        data = np.frombuffer(f.read(n * 8), dtype='<f8')
    return data.reshape(shape)


def read_header(plotdir):
    """Parse the plotfile Header.

    Layout: [0]=version [1]=nvar [2..nvar+1]=varnames then spacedim, time,
    finest_level, prob_lo, prob_hi, ref_ratios, domains, steps, dx per level.
    """
    with open(os.path.join(plotdir, 'Header')) as f:
        h = f.readlines()
    nvar = int(h[1].strip())
    base = 2 + nvar
    ndim = int(h[base])
    finest = int(h[base + 2])
    nlev = finest + 1
    domains = []
    for m in BOX_RE.finditer(h[base + 6]):
        hi = [int(v) for v in m.group(2).split(',')]
        domains.append(tuple(v + 1 for v in reversed(hi)))  # numpy order
    ratios = [int(x) for x in h[base + 5].split()][:finest]
    dx = [tuple(float(v) for v in h[base + 8 + lev].split()[:ndim]) for lev in range(nlev)]
    return {
        'time':     float(h[base + 1]),
        'ndim':     ndim,
        'varnames': [h[2 + i].strip() for i in range(nvar)],
        'nlev':     nlev,
        'prob_lo':  tuple(float(v) for v in h[base + 3].split()[:ndim]),
        'prob_hi':  tuple(float(v) for v in h[base + 4].split()[:ndim]),
        'ratios':   ratios,
        'domains':  domains,
        'dx':       dx,
    }


def var_index(hdr, var):
    """Component index for a variable name (or number); default is the first."""
    if var is None:
        return 0
    if var in hdr['varnames']:
        return hdr['varnames'].index(var)
    try:
        return int(var)
    except ValueError:
        raise SystemExit(f"unknown variable {var!r}; plotfile has {hdr['varnames']}")


def read_time(plotdir):
    try:
        return read_header(plotdir)['time']
    except Exception:
        return 0.0


def read_level(plotdir, level, hdr, comp=0):
    """Return the level's grid on its full domain (NaN where no box exists)."""
    cell_h = os.path.join(plotdir, f'Level_{level}', 'Cell_H')
    if not os.path.exists(cell_h):
        return None
    level_dir = os.path.join(plotdir, f'Level_{level}')
    boxes, fab_entries = parse_cell_h(cell_h)
    grid = np.full(hdr['domains'][level], np.nan)
    ncomp = len(hdr['varnames'])
    for (lo, hi), (fname, offset) in zip(boxes, fab_entries):
        sel = tuple(slice(l, h + 1) for l, h in zip(reversed(lo), reversed(hi)))
        shape = tuple(h - l + 1 for l, h in zip(reversed(lo), reversed(hi)))
        grid[sel] = read_fab(os.path.join(level_dir, fname), offset, shape, ncomp, comp)
    return grid


def finest_shape(plotdirs):
    """Largest finest-level shape over all plotfiles, i.e. the common image
    resolution when the number of levels changes during the run."""
    shape = None
    for pd in plotdirs:
        hdr = read_header(pd)
        # the last level that actually has data
        lev = hdr['nlev'] - 1
        while lev > 0 and not os.path.exists(os.path.join(pd, f'Level_{lev}', 'Cell_H')):
            lev -= 1
        cur = hdr['domains'][lev]
        shape = cur if shape is None else tuple(max(a, b) for a, b in zip(shape, cur))
    return shape


def upsample_to(img, shape):
    """Repeat pixels along every axis so that img has the given shape."""
    f = shape[0] // img.shape[0]
    if f == 1:
        return img
    for ax in range(img.ndim):
        img = np.repeat(img, f, axis=ax)
    return img


def slice_2d(img):
    """Mid-plane 2D slice of an image (unchanged for 2D)."""
    while img.ndim > 2:
        img = img[img.shape[0] // 2]
    return img


def read_plotfile(plotdir, shape=None, var=None, volume=False):
    """Return {'hdr', 'grids', 'phi'}: per-level grids (NaN = uncovered) and the
    composite 2D image of the chosen variable (mid-plane slice in 3D) at the
    finest resolution (or at `shape`). With volume=True the full N-d composite
    is also returned as 'vol' (float32)."""
    hdr = read_header(plotdir)
    comp = var_index(hdr, var)
    grids = []
    for lev in range(hdr['nlev']):
        g = read_level(plotdir, lev, hdr, comp)
        if g is None:
            break
        grids.append(g)
    if not grids:
        return None
    hdr['nlev'] = len(grids)
    phi = composite_phi(grids, hdr['ratios'])
    if shape is not None:
        phi = upsample_to(phi, shape)
    out = {'hdr': hdr, 'grids': grids, 'phi': slice_2d(phi)}
    if volume:
        out['vol'] = phi.astype(np.float32)
    return out


def composite_phi(grids, ratios):
    """Composite image at the finest resolution: each finer level overwrites the
    (upsampled) coarser ones wherever it has data."""
    img = grids[0]
    for lev in range(1, len(grids)):
        r = ratios[lev - 1]
        for ax in range(img.ndim):
            img = np.repeat(img, r, axis=ax)
        img = np.where(~np.isnan(grids[lev]), grids[lev], img)
    return img


def finest_domain(hdr):
    """(xmin, xmax, ymin, ymax) of the (x, y) plane of the domain."""
    return (hdr['prob_lo'][0], hdr['prob_hi'][0], hdr['prob_lo'][1], hdr['prob_hi'][1])


# ── composite mesh graph ─────────────────────────────────────────────────────

def build_composite_graph(grids, ratios, periodic=True, shape=None):
    """Graph of the composite mesh: one node per leaf cell (a cell with data
    that is not covered by a finer level), edges between cells that share a face.

    Returns (nodes, edges, leaf):
      nodes: list of (level, *index) tuples in numpy index order, by node id
      edges: list of (u, v) with u < v
      leaf:  int32 image at the finest resolution (or `shape`) holding the node
             id of the leaf cell covering each pixel
    """
    nlev = len(grids)
    ndim = grids[0].ndim
    masks = [~np.isnan(g) for g in grids]

    nodes = []
    leaf = np.full(grids[-1].shape, -1, dtype=np.int32)
    for lev in range(nlev):
        shp = grids[lev].shape
        is_leaf = masks[lev].copy()
        if lev + 1 < nlev:
            r = ratios[lev]
            split = tuple(x for n in shp for x in (n, r))
            fine = masks[lev + 1].reshape(split)
            is_leaf &= ~fine.any(axis=tuple(range(1, 2 * ndim, 2)))
        idx = np.nonzero(is_leaf)
        count = len(idx[0])
        ids = np.full(shp, -1, dtype=np.int32)
        ids[idx] = np.arange(len(nodes), len(nodes) + count, dtype=np.int32)
        nodes.extend((lev,) + tuple(int(c) for c in cell) for cell in zip(*idx))
        up = upsample_to(ids, leaf.shape)
        leaf = np.where(up >= 0, up, leaf)

    if shape is not None:
        leaf = upsample_to(leaf, shape)

    pairs = []
    for axis in range(ndim):
        a = leaf
        b = np.roll(leaf, -1, axis=axis)
        if not periodic:
            cut = [slice(None)] * ndim
            cut[axis] = slice(None, -1)
            a, b = a[tuple(cut)], b[tuple(cut)]
        keep = (a != b) & (a >= 0) & (b >= 0)
        u, v = a[keep], b[keep]
        pairs.append(np.stack([np.minimum(u, v), np.maximum(u, v)], axis=1))
    uv = np.unique(np.concatenate(pairs), axis=0)
    edges = [(int(u), int(v)) for u, v in uv]
    return nodes, edges, leaf


def partition_image(leaf, partition):
    """2D image of the partition at the finest resolution (mid-plane in 3D)."""
    return np.asarray(partition, dtype=np.int32)[slice_2d(leaf)]


def nodes_edges_to_metis(nodes, edges):
    """METIS fmt=011 lines (vertex weight 1, edge weight 1)."""
    adj = [[] for _ in range(len(nodes))]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    lines = [f"{len(nodes)} {len(edges)} 011\n"]
    for nbs in adj:
        lines.append(" ".join(["1"] + [f"{nb + 1} 1" for nb in nbs]) + "\n")
    return lines


# ── video helpers ────────────────────────────────────────────────────────────

def playback_fps(times, fps):
    """Frame rate that plays the video in simulation time: at most fps, lower
    when the plotfiles are sparser than fps frames per simulated second."""
    span = max(times) - min(times)
    if span <= 0 or len(times) < 2:
        return fps
    return min(fps, (len(times) - 1) / span)


def assemble_video(tmpdir, out_path, fps):
    """Encode tmpdir/f%06d.png into out_path (.mp4 or .gif) with system ffmpeg."""
    ext = os.path.splitext(out_path)[1].lower()
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-framerate', f'{fps:.6g}',
           '-i', os.path.join(tmpdir, 'f%06d.png')]
    if ext == '.gif':
        cmd += ['-loop', '0']
    else:
        cmd += ['-vf', 'pad=ceil(iw/16)*16:ceil(ih/16)*16:0:0:color=0x1a1a2e,'
                       'scale=out_color_matrix=bt709:out_range=tv,format=yuv420p',
                '-c:v', 'libx264', '-profile:v', 'main', '-bf', '0',
                '-colorspace', 'bt709', '-color_primaries', 'bt709',
                '-color_trc', 'bt709', '-color_range', 'tv',
                '-movflags', '+faststart']
    subprocess.run(cmd + [out_path], check=True)


def select_frames(times, fps):
    """Indices of the frames closest to evenly spaced times at `fps` frames per
    simulated second (all frames if there are no more than that)."""
    t_max, t_min = max(times), min(times)
    n_target = int((t_max - t_min) * fps) + 1
    if len(times) <= n_target:
        return list(range(len(times)))
    ta = np.array(times)
    targets = np.linspace(t_min, t_max, n_target)
    return sorted(set(int(np.argmin(np.abs(ta - t))) for t in targets))
