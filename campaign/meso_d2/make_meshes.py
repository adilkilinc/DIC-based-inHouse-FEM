# -*- coding: utf-8 -*-
"""Conforming triangle meshes of the Dataset 2 partitions (run locally; needs `triangle`).

Input   Case_studies/CS_2_d2_partition/outputs/partition_d2_optical.npz, label maps
        on the 1880 x 1705 DIC grid. The analysis grid of the campaign is rows
        0..1879 and columns 2..1701 (field_geometry crop_offset [0, 2]).
Units   pixel (i, j) of the mesh grid occupies [i, i+1] x [j, j+1] raw units
        (grid_origin = analysis-grid index of mesh pixel (0, 0); 0, 0 except for
        the --crop smoke-test meshes);
        1 raw unit = 3.45 um. Node coordinates in mm are raw * 0.00345, with
        coords[:, 0] along the image rows and coords[:, 1] along the columns
        (tensile axis), the convention of the pixel-scale and Dataset 1 meshes.

Steps   1. unindexed pixels of the twin-resolved map (label 0) take the nearest
           twin-resolved label of the same twin-merged grain;
        2. region boundaries are traced on the pixel cracks, split at junctions
           and simplified (Douglas-Peucker, vertices kept on the pixel corners);
           chains whose simplified segments would cross or overlap another
           chain are kept unsimplified;
        3. constrained quality triangulation (Triangle, min angle q, max area a);
        4. each face enclosed by boundary segments takes the majority label of
           its triangle centroids; the mapping purity of every region is the
           share of its pixels whose centre lies in a triangle of that region.

    python make_meshes.py [--eps 0.9] [--q 28] [--moderate 12] [--fine 4]
        -> 0_inputs/meshes/{moderate,fine}_{no_twins,twins}.npz,
           0_inputs/partitions_d2.npz, 0_inputs/meshes/mesh_report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import triangle
from numba import njit
from scipy import ndimage

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "Case_studies" / "CS_2_d2_partition" / "outputs" / "partition_d2_optical.npz"
PX_MM = 0.00345
ROWS, COL0, NCOL = 1880, 2, 1700


def sha256(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ partitions
def partitions(src):
    z = np.load(src)
    res = z["twin_resolved"][:, COL0:COL0 + NCOL].astype(np.int64)
    mer = z["twin_merged"][:, COL0:COL0 + NCOL].astype(np.int64)
    zero = res == 0
    n_zero = int(zero.sum())
    # fill label 0 inside the same merged grain, by 4-neighbour propagation
    it = 0
    while (res == 0).any() and it < 500:
        it += 1
        for sh in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = np.roll(res, sh, axis=(0, 1))
            nm = np.roll(mer, sh, axis=(0, 1))
            ok = (res == 0) & (nb > 0) & (nm == mer)
            if sh == (1, 0):
                ok[0, :] = False
            elif sh == (-1, 0):
                ok[-1, :] = False
            elif sh == (0, 1):
                ok[:, 0] = False
            else:
                ok[:, -1] = False
            res[ok] = nb[ok]
    left = res == 0
    new = 0
    if left.any():                      # merged grain without indexed point: one region per grain
        base = res.max()
        for m in np.unique(mer[left]):
            new += 1
            res[left & (mer == m)] = base + new
    # strict refinement check
    key = np.unique(res * 100000 + mer)
    if np.unique(key // 100000).size != key.size:
        raise SystemExit("twin-resolved region spans two merged grains")
    info = dict(unindexed_pixels_filled=n_zero, propagation_iterations=it, regions_created=new,
                regions_merged=int(np.unique(mer).size), regions_resolved=int(np.unique(res).size))
    return mer.astype(np.int32), res.astype(np.int32), info


# ------------------------------------------------------------------ boundary graph
def crack_edges(lab):
    """Corner-to-corner edges of the region boundaries and of the domain perimeter.
    Corner (i, j) has id i * (nc + 1) + j, i in 0..nr, j in 0..nc."""
    nr, nc = lab.shape
    W = nc + 1
    e = []
    ii, jj = np.nonzero(lab[1:, :] != lab[:-1, :])        # between rows i and i+1 -> corners (i+1, j)-(i+1, j+1)
    e.append(np.stack([(ii + 1) * W + jj, (ii + 1) * W + jj + 1], 1))
    ii, jj = np.nonzero(lab[:, 1:] != lab[:, :-1])        # between cols j and j+1 -> corners (i, j+1)-(i+1, j+1)
    e.append(np.stack([ii * W + jj + 1, (ii + 1) * W + jj + 1], 1))
    j = np.arange(nc)
    e.append(np.stack([j, j + 1], 1))                       # row 0
    e.append(np.stack([nr * W + j, nr * W + j + 1], 1))     # row nr
    i = np.arange(nr)
    e.append(np.stack([i * W, (i + 1) * W], 1))             # col 0
    e.append(np.stack([i * W + nc, (i + 1) * W + nc], 1))   # col nc
    return np.concatenate(e).astype(np.int64)


def chains(edges, n_corner, fixed):
    """Split the boundary graph into chains between junctions (degree != 2 or fixed)."""
    deg = np.bincount(edges.ravel(), minlength=n_corner)
    junction = (deg != 2) & (deg > 0)
    junction[fixed] = True
    order = np.argsort(np.concatenate([edges[:, 0], edges[:, 1]]), kind="stable")
    ends = np.concatenate([edges[:, 1], edges[:, 0]])[order]
    eid = np.concatenate([np.arange(len(edges)), np.arange(len(edges))])[order]
    start = np.zeros(n_corner + 1, np.int64)
    np.cumsum(deg, out=start[1:])
    used = np.zeros(len(edges), bool)
    out = []

    def walk(v, e):
        path = [v]
        while True:
            used[e] = True
            a, b = edges[e]
            w = b if a == v else a
            path.append(w)
            if junction[w] or w == path[0]:
                return path
            s = start[w]
            nxt = [eid[k] for k in range(s, s + deg[w]) if not used[eid[k]]]
            if not nxt:
                return path
            v, e = w, nxt[0]

    for v in np.nonzero(junction)[0]:
        for k in range(start[v], start[v] + deg[v]):
            if not used[eid[k]]:
                out.append(walk(v, eid[k]))
    for e in np.nonzero(~used)[0]:                          # loops without junction
        if not used[e]:
            out.append(walk(edges[e, 0], e))
    return out


def dp(pts, eps):
    """Douglas-Peucker on an open polyline; returns kept indices."""
    n = len(pts)
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        p, q = pts[a], pts[b]
        d = q - p
        L = np.hypot(*d)
        seg = pts[a + 1:b] - p
        dist = np.abs(seg[:, 0] * d[1] - seg[:, 1] * d[0]) / L if L > 0 else np.hypot(seg[:, 0], seg[:, 1])
        k = int(np.argmax(dist))
        if dist[k] > eps:
            m = a + 1 + k
            keep[m] = True
            stack += [(a, m), (m, b)]
    return np.nonzero(keep)[0]


def simplify(chain, coords, eps, force_interior):
    pts = coords[chain]
    if chain[0] == chain[-1]:                               # closed loop: split at the farthest vertex
        d = np.hypot(*(pts - pts[0]).T)
        m = int(np.argmax(d))
        k1 = dp(pts[:m + 1], eps)
        k2 = dp(pts[m:], eps) + m
        k1 = _force(pts[:m + 1], k1)
        k2 = _force(pts[m:], k2 - m) + m
        idx = np.unique(np.concatenate([k1, k2]))
        return [chain[i] for i in idx] + [chain[0]]
    k = dp(pts, eps)
    if force_interior:
        k = _force(pts, k)
    return [chain[i] for i in k]


def _force(pts, k):
    """Keep at least one interior vertex (the farthest from the chord)."""
    if len(k) > 2 or len(pts) < 3:
        return k
    p, q = pts[0], pts[-1]
    d = q - p
    L = np.hypot(*d)
    seg = pts[1:-1] - p
    dist = np.abs(seg[:, 0] * d[1] - seg[:, 1] * d[0]) / L if L > 0 else np.hypot(seg[:, 0], seg[:, 1])
    return np.array(sorted({0, len(pts) - 1, 1 + int(np.argmax(dist))}))


@njit(cache=True)
def _conflicts(P, S, owner, pairs, out):
    """Flag segment pairs of different chains that cross, touch or overlap."""
    n = 0
    for t in range(pairs.shape[0]):
        a, b = pairs[t, 0], pairs[t, 1]
        p1, p2 = S[a, 0], S[a, 1]
        q1, q2 = S[b, 0], S[b, 1]
        shared = int(p1 == q1) + int(p1 == q2) + int(p2 == q1) + int(p2 == q2)
        if owner[a] == owner[b] and shared >= 1:
            continue
        if shared >= 2:
            out[n, 0] = a; out[n, 1] = b; n += 1
            continue
        ax, ay = P[p1, 0], P[p1, 1]; bx, by = P[p2, 0], P[p2, 1]
        cx, cy = P[q1, 0], P[q1, 1]; dx, dy = P[q2, 0], P[q2, 1]
        d1 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        d2 = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax)
        d3 = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx)
        d4 = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx)
        hit = False
        if shared == 1:
            # only a collinear overlap matters when an endpoint is shared
            if d1 == 0.0 and d2 == 0.0:
                lo1 = min(ax, bx); hi1 = max(ax, bx); lo2 = min(cx, dx); hi2 = max(cx, dx)
                ly1 = min(ay, by); hy1 = max(ay, by); ly2 = min(cy, dy); hy2 = max(cy, dy)
                ox = min(hi1, hi2) - max(lo1, lo2); oy = min(hy1, hy2) - max(ly1, ly2)
                if ox > 0.0 or oy > 0.0:
                    hit = True
        else:
            if ((d1 > 0) != (d2 > 0) or d1 == 0.0 or d2 == 0.0) and ((d3 > 0) != (d4 > 0) or d3 == 0.0 or d4 == 0.0):
                # bounding boxes must overlap for the touching cases
                if (min(ax, bx) <= max(cx, dx) and min(cx, dx) <= max(ax, bx)
                        and min(ay, by) <= max(cy, dy) and min(cy, dy) <= max(ay, by)):
                    hit = True
        if hit:
            out[n, 0] = a; out[n, 1] = b; n += 1
    return n


def candidate_pairs(P, S, cell=8.0):
    lo = np.minimum(P[S[:, 0]], P[S[:, 1]]) // cell
    hi = np.maximum(P[S[:, 0]], P[S[:, 1]]) // cell
    span = (hi - lo + 1).astype(np.int64)
    reps = span[:, 0] * span[:, 1]
    sid = np.repeat(np.arange(len(S)), reps)
    k = np.arange(reps.sum()) - np.repeat(np.cumsum(reps) - reps, reps)
    cx = lo[sid, 0].astype(np.int64) + k // span[sid, 1]
    cy = lo[sid, 1].astype(np.int64) + k % span[sid, 1]
    key = cx * 100000 + cy
    o = np.argsort(key, kind="stable")
    key, sid = key[o], sid[o]
    b = np.flatnonzero(np.r_[True, key[1:] != key[:-1], True])
    pa, pb = [], []
    for s, e in zip(b[:-1], b[1:]):
        if e - s < 2:
            continue
        g = sid[s:e]
        i, j = np.triu_indices(len(g), 1)
        pa.append(g[i]); pb.append(g[j])
    if not pa:
        return np.zeros((0, 2), np.int64)
    p = np.stack([np.concatenate(pa), np.concatenate(pb)], 1)
    p.sort(axis=1)
    return np.unique(p, axis=0)


def pslg(lab, eps):
    nr, nc = lab.shape
    W = nc + 1
    n_corner = (nr + 1) * W
    corners = np.stack(np.divmod(np.arange(n_corner), W), 1).astype(np.float64)
    edges = crack_edges(lab)
    fixed = np.array([0, nc, nr * W, nr * W + nc])
    raw = chains(edges, n_corner, fixed)
    pair_count = {}
    for c in raw:
        k = (min(c[0], c[-1]), max(c[0], c[-1]))
        pair_count[k] = pair_count.get(k, 0) + 1
    plain = np.zeros(len(raw), bool)
    for rnd in range(20):
        simp = [c if plain[i] else simplify(c, corners, eps, pair_count[(min(c[0], c[-1]), max(c[0], c[-1]))] > 1)
                for i, c in enumerate(raw)]
        seg, own = [], []
        for i, c in enumerate(simp):
            c = np.asarray(c, np.int64)
            seg.append(np.stack([c[:-1], c[1:]], 1)); own.append(np.full(len(c) - 1, i))
        S = np.concatenate(seg); owner = np.concatenate(own)
        pairs = candidate_pairs(corners, S)
        out = np.zeros((max(len(pairs), 1), 2), np.int64)
        n = _conflicts(corners, S, owner, pairs, out)
        bad = np.unique(owner[out[:n].ravel()]) if n else np.zeros(0, np.int64)
        bad = bad[~plain[bad]]
        if bad.size == 0:
            break
        plain[bad] = True
    used = np.unique(S)
    remap = np.full(n_corner, -1, np.int64)
    remap[used] = np.arange(used.size)
    info = dict(chains=len(raw), chains_unsimplified=int(plain.sum()), conflict_rounds=rnd + 1,
                crack_edges=int(len(edges)), vertices=int(used.size), segments=int(len(S)),
                remaining_conflicts=int(n))
    return corners[used], remap[S], info


# ------------------------------------------------------------------ labelling
@njit(cache=True)
def _find(p, i):
    while p[i] != i:
        p[i] = p[p[i]]
        i = p[i]
    return i


@njit(cache=True)
def _faces(tri, nbr, seg_flag):
    n = tri.shape[0]
    p = np.arange(n)
    for t in range(n):
        for k in range(3):
            u = nbr[t, k]
            if u < 0 or seg_flag[t, k]:
                continue
            a, b = _find(p, t), _find(p, u)
            if a != b:
                p[max(a, b)] = min(a, b)
    for t in range(n):
        p[t] = _find(p, t)
    return p


@njit(cache=True)
def _raster(xy, el, nr, nc, out):
    """out[i, j] = triangle containing the pixel centre (i + 0.5, j + 0.5); -1 outside."""
    for e in range(el.shape[0]):
        x0, y0 = xy[el[e, 0], 0], xy[el[e, 0], 1]
        x1, y1 = xy[el[e, 1], 0], xy[el[e, 1], 1]
        x2, y2 = xy[el[e, 2], 0], xy[el[e, 2], 1]
        d = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if d == 0.0:
            continue
        i0 = max(int(np.floor(min(x0, x1, x2) - 0.5)), 0); i1 = min(int(np.ceil(max(x0, x1, x2))), nr - 1)
        j0 = max(int(np.floor(min(y0, y1, y2) - 0.5)), 0); j1 = min(int(np.ceil(max(y0, y1, y2))), nc - 1)
        for i in range(i0, i1 + 1):
            px = i + 0.5
            for j in range(j0, j1 + 1):
                py = j + 0.5
                a = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / d
                b = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / d
                c = 1.0 - a - b
                if a >= -1e-12 and b >= -1e-12 and c >= -1e-12:
                    out[i, j] = e


def mesh(lab, eps, q, amax):
    t0 = time.time()
    V, S, pinfo = pslg(lab, eps)
    t1 = time.time()
    T = triangle.triangulate(dict(vertices=V, segments=S), f"pq{q}a{amax}nQ")
    t2 = time.time()
    xy, el, nbr = T["vertices"], T["triangles"].astype(np.int64), T["neighbors"].astype(np.int64)
    nv = np.int64(len(xy))
    ts = np.sort(T["segments"].astype(np.int64), axis=1)
    skey = np.unique(ts[:, 0] * nv + ts[:, 1])
    seg_flag = np.zeros(el.shape, bool)
    for k in range(3):                                       # edge k is opposite vertex k
        a, b = el[:, (k + 1) % 3], el[:, (k + 2) % 3]
        seg_flag[:, k] = np.isin(np.minimum(a, b) * nv + np.maximum(a, b), skey)
    face = _faces(el, nbr, seg_flag)
    cen = xy[el].mean(axis=1)
    ci = np.clip(np.floor(cen).astype(np.int64), 0, [lab.shape[0] - 1, lab.shape[1] - 1])
    clab = lab[ci[:, 0], ci[:, 1]]
    p3 = xy[el]
    area = 0.5 * np.abs((p3[:, 1, 0] - p3[:, 0, 0]) * (p3[:, 2, 1] - p3[:, 0, 1])
                        - (p3[:, 2, 0] - p3[:, 0, 0]) * (p3[:, 1, 1] - p3[:, 0, 1]))
    # area-weighted majority label per face
    fu, finv = np.unique(face, return_inverse=True)
    key = finv.astype(np.int64) * (int(lab.max()) + 1) + clab
    ku, kinv = np.unique(key, return_inverse=True)
    w = np.bincount(kinv, weights=area)
    kf, kl = ku // (int(lab.max()) + 1), ku % (int(lab.max()) + 1)
    o = np.lexsort((-w, kf))
    first = np.r_[True, kf[o][1:] != kf[o][:-1]]
    face_label = np.empty(fu.size, np.int64)
    face_label[kf[o][first]] = kl[o][first]
    elab = face_label[finv]
    # purity on the pixel grid
    idx = np.full(lab.shape, -1, np.int64)
    _raster(np.ascontiguousarray(xy), np.ascontiguousarray(el), lab.shape[0], lab.shape[1], idx)
    if (idx < 0).any():
        raise SystemExit(f"{int((idx < 0).sum())} pixel centres outside the mesh")
    mlab = elab[idx]
    regions = np.unique(lab)
    n = int(lab.max()) + 1
    tot = np.bincount(lab.ravel(), minlength=n)
    hit = np.bincount(lab.ravel(), weights=(mlab == lab).ravel().astype(float), minlength=n)
    purity = hit[regions] / tot[regions]
    present = np.unique(elab)
    missing = np.setdiff1d(regions, present)
    # quality
    L = np.stack([np.hypot(*(p3[:, (k + 1) % 3] - p3[:, k]).T) for k in range(3)], 1)
    cosang = np.stack([(L[:, (k + 1) % 3] ** 2 + L[:, (k + 2) % 3] ** 2 - L[:, k] ** 2)
                       / (2 * L[:, (k + 1) % 3] * L[:, (k + 2) % 3]) for k in range(3)], 1)
    minang = np.degrees(np.arccos(np.clip(cosang, -1, 1))).min(axis=1)
    rr = 4 * np.sqrt(3) * area / np.sum(L ** 2, axis=1)
    per_region = np.bincount(np.searchsorted(present, elab), minlength=present.size)
    small = tot[regions] < 50
    info = dict(pslg=pinfo, eps_px=eps, q_deg=q, max_area_px2=amax, nodes=int(len(xy)), elements=int(len(el)),
                faces=int(fu.size), regions=int(regions.size), regions_without_element=missing.tolist(),
                pixel_agreement=float((mlab == lab).mean()), purity_min=float(purity.min()),
                purity_p1=float(np.percentile(purity, 1)), purity_median=float(np.median(purity)),
                min_angle_deg=float(minang.min()), min_angle_p1=float(np.percentile(minang, 1)),
                radius_ratio_min=float(rr.min()), radius_ratio_median=float(np.median(rr)),
                elements_per_region_min=int(per_region.min()), elements_per_region_median=float(np.median(per_region)),
                elements_per_small_region_min=int(per_region[np.isin(present, regions[small])].min()) if small.any() else None,
                area_px2_median=float(np.median(area)), seconds=dict(pslg=t1 - t0, triangle=t2 - t1, total=time.time() - t0))
    return xy, el.astype(np.int32), elab.astype(np.int32), regions, purity, info


def boundary_nodes(xy, nr, nc):
    on = (np.abs(xy[:, 0]) < 1e-9) | (np.abs(xy[:, 0] - nr) < 1e-9) | (np.abs(xy[:, 1]) < 1e-9) | (np.abs(xy[:, 1] - nc) < 1e-9)
    return np.nonzero(on)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--eps", type=float, default=0.9)
    ap.add_argument("--q", type=float, default=28.0)
    ap.add_argument("--moderate", type=float, default=12.0, help="max element area, px^2")
    ap.add_argument("--fine", type=float, default=4.0)
    ap.add_argument("--only", default="", help="comma list of meshes, for tests")
    ap.add_argument("--crop", type=int, default=0, help="test on a centre crop of this size")
    a = ap.parse_args()
    src = Path(a.src)
    mer, res, pinfo = partitions(src)
    origin = (0, 0)
    if a.crop:                                               # smoke-test meshes on a centre window
        origin = ((ROWS - a.crop) // 2, (NCOL - a.crop) // 2)
        c0, c1 = origin
        mer, res = mer[c0:c0 + a.crop, c1:c1 + a.crop], res[c0:c0 + a.crop, c1:c1 + a.crop]
    out = HERE / "0_inputs" / "meshes"
    out.mkdir(parents=True, exist_ok=True)
    if not a.crop:
        pf = HERE / "0_inputs" / "partitions_d2.npz"
        np.savez_compressed(pf, twin_merged=mer, twin_resolved=res, pixel_mm=PX_MM, rows=ROWS, col_offset=COL0)
        pinfo["file_sha256"] = sha256(pf)
    report = dict(source=str(src.name), source_sha256=sha256(src), partition=pinfo, meshes={})
    nr, nc = mer.shape
    for dens, amax in (("moderate", a.moderate), ("fine", a.fine)):
        for twins, lab in ((False, mer), (True, res)):
            name = f"{dens}_{'twins' if twins else 'no_twins'}"
            if a.only and name not in a.only.split(","):
                continue
            xy, el, elab, regions, purity, info = mesh(lab, a.eps, a.q, amax)
            # region index per element, ordered as `segmentation_label`
            mat_id = np.searchsorted(regions, elab).astype(np.int32)
            merged_of = None
            if twins:                                        # merged grain of every twin-resolved region
                m = np.zeros(int(res.max()) + 1, np.int64)
                m[res.ravel()] = mer.ravel()
                merged_of = m[regions].astype(np.int32)
            bc = boundary_nodes(xy, nr, nc)
            d = dict(schema=1, mesh=name, coords=xy * PX_MM, elements=el, mat_id=mat_id,
                     segmentation_label=regions.astype(np.int32), mapping_purity=purity, bc_nodes=bc.astype(np.int64),
                     partition="twin_resolved" if twins else "twin_merged", pixel_mm=PX_MM, grid_shape=np.array([nr, nc]),
                     grid_origin=np.array(origin))
            if merged_of is not None:
                d["merged_label_of_region"] = merged_of
            f = out / f"{name}{'_crop' if a.crop else ''}.npz"
            np.savez_compressed(f, **d)
            info.update(bc_nodes=int(bc.size), file=f.name, file_sha256=sha256(f))
            report["meshes"][name] = info
            print(name, json.dumps({k: v for k, v in info.items() if k not in ("pslg",)}, default=float), flush=True)
            print("   pslg", info["pslg"], flush=True)
    (out / ("mesh_report_crop.json" if a.crop else "mesh_report.json")).write_text(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
