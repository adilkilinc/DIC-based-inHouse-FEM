# -*- coding: utf-8 -*-
"""
fem_to_grid.py - rasterise unstructured CPS3 fields onto the DIC pixel grid.

Python-3 re-implementation of the barycentric rasteriser in
2_codes/odb_grid_4.py, with two changes that matter for this test:

  * the raster (element index + barycentric weights per pixel) is built ONCE
    per mesh and cached, so mapping another field or another frame is a single
    fancy-index;
  * the SAME routine is applied to the Abaqus element export and to the
    in-house FEM element fields, so the pixel maps differ only where the
    solvers differ - no rasterisation asymmetry in the comparison.

Element fields (S, E, PE, PEEQ) are constant inside a CPS3, so every pixel of a
triangle gets the element value.  Nodal fields (U, RF) are interpolated with the
triangle's shape functions, which is exact for a linear triangle.

    python fem_to_grid.py            # last frame, FEM + Abaqus
    MESO_GRID_ALL_FRAMES=1 python fem_to_grid.py

Outputs land in results/<mesh>/compare as grid_<src>_<field>.npy.
"""
import os
import sys
import time
import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
# Repository layout: the solver lives in ../solver, the runners and the
# configuration here. (Before the micro/meso split these sat together under
# fem/1_codes, so ROOT was two levels up - it is kept only for sys.path.)
ROOT = os.path.dirname(CODE_DIR)
SOLVER_DIR = os.path.join(ROOT, "solver")
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, SOLVER_DIR)
sys.path.insert(0, ROOT)

import meso_config as cfg
import inp_io

ALL_FRAMES = os.environ.get("MESO_GRID_ALL_FRAMES", "0") == "1"


# ── raster construction ─────────────────────────────────────────────────────
def build_raster(coords, elements, m_per_pixel=None, verbose=True):
    """Return dict(pixel_map, a, b, elements, shape).

    pixel_map (nx, ny) int32 : element index at each pixel, -1 outside the mesh
    a, b      (nx, ny) float32 : barycentric weights of vertices 0 and 1
                                 (the third is 1 - a - b)
    Pixels covered by several triangles keep the LAST one written, exactly as
    odb_grid_4.py does.
    """
    mpp = cfg.PX_TO_MM if m_per_pixel is None else m_per_pixel
    px = np.rint(coords[:, 0]/mpp).astype(np.int32)
    py = np.rint(coords[:, 1]/mpp).astype(np.int32)
    nx = int(px.max()) + 1
    ny = int(py.max()) + 1
    if verbose:
        print(f"[grid] pixel grid {nx} x {ny}  (m_per_pixel = {mpp})")

    vx = px[elements]          # (n_e, 3)
    vy = py[elements]
    n_e = len(elements)

    pixel_map = np.full((nx, ny), -1, dtype=np.int32)
    A = np.zeros((nx, ny), dtype=np.float32)
    Bw = np.zeros((nx, ny), dtype=np.float32)

    x1, x2, x3 = vx[:, 0].astype(np.int64), vx[:, 1].astype(np.int64), vx[:, 2].astype(np.int64)
    y1, y2, y3 = vy[:, 0].astype(np.int64), vy[:, 1].astype(np.int64), vy[:, 2].astype(np.int64)
    denom_all = ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)).astype(np.float64)
    xmin_all = np.clip(np.minimum(np.minimum(x1, x2), x3), 0, nx - 1)
    xmax_all = np.clip(np.maximum(np.maximum(x1, x2), x3), 0, nx - 1)
    ymin_all = np.clip(np.minimum(np.minimum(y1, y2), y3), 0, ny - 1)
    ymax_all = np.clip(np.maximum(np.maximum(y1, y2), y3), 0, ny - 1)

    t0 = time.time()
    report = max(1, n_e // 20)
    degenerate = 0
    for e in range(n_e):
        d = denom_all[e]
        if abs(d) < 1e-12:
            degenerate += 1
            continue
        xa, xb = xmin_all[e], xmax_all[e]
        ya, yb = ymin_all[e], ymax_all[e]
        if xa > xb or ya > yb:
            continue
        GX = np.arange(xa, xb + 1, dtype=np.int64)
        GY = np.arange(ya, yb + 1, dtype=np.int64)
        XX = GX[:, None].astype(np.float64)
        YY = GY[None, :].astype(np.float64)
        a = ((y2[e] - y3[e])*(XX - x3[e]) + (x3[e] - x2[e])*(YY - y3[e]))/d
        b = ((y3[e] - y1[e])*(XX - x3[e]) + (x1[e] - x3[e])*(YY - y3[e]))/d
        c = 1.0 - a - b
        inside = (a >= -1e-9) & (b >= -1e-9) & (c >= -1e-9)
        if not inside.any():
            continue
        ii, jj = np.nonzero(inside)
        gx = GX[ii]
        gy = GY[jj]
        pixel_map[gx, gy] = e
        A[gx, gy] = a[ii, jj]
        Bw[gx, gy] = b[ii, jj]
        if verbose and e % report == 0:
            print(f"  rasterising {e}/{n_e}", end="\r", flush=True)

    cov = 100.0*np.count_nonzero(pixel_map >= 0)/float(nx*ny)
    if verbose:
        print(f"[grid] rasterised {n_e} triangles in {time.time()-t0:.1f}s  "
              f"coverage {cov:.1f}%"
              + (f"  ({degenerate} degenerate skipped)" if degenerate else ""))
    return dict(pixel_map=pixel_map, a=A, b=Bw, elements=np.asarray(elements),
                shape=(nx, ny), coverage=cov)


def get_raster(prob=None, verbose=True):
    """Cached raster for the configured mesh."""
    cpath = os.path.join(cfg.CACHE, f"raster_{cfg.MESH_KEY}.npz")
    if os.path.isfile(cpath):
        d = np.load(cpath)
        if verbose:
            print(f"[grid] raster cache {cpath}")
        return dict(pixel_map=d["pixel_map"], a=d["a"], b=d["b"],
                    elements=d["elements"], shape=tuple(d["shape"]),
                    coverage=float(d["coverage"]))
    if prob is None:
        prob = inp_io.build_problem(verbose=verbose)
    r = build_raster(prob["coords"], prob["elements"], verbose=verbose)
    np.savez_compressed(cpath, pixel_map=r["pixel_map"], a=r["a"], b=r["b"],
                        elements=r["elements"], shape=np.array(r["shape"]),
                        coverage=r["coverage"])
    if verbose:
        print(f"[grid] raster cached -> {cpath}")
    return r


# ── field mapping ───────────────────────────────────────────────────────────
def elem_to_grid(vals, raster):
    """(n_elem,) element values -> (nx, ny) pixel map, NaN outside the mesh."""
    pm = raster["pixel_map"]
    out = np.full(pm.shape, np.nan, dtype=np.float64)
    m = pm >= 0
    out[m] = np.asarray(vals, dtype=np.float64)[pm[m]]
    return out


def node_to_grid(vals, raster):
    """(n_node,) nodal values -> (nx, ny) pixel map by linear interpolation."""
    pm = raster["pixel_map"]
    conn = raster["elements"]
    out = np.full(pm.shape, np.nan, dtype=np.float64)
    m = pm >= 0
    e = pm[m]
    a = raster["a"][m].astype(np.float64)
    b = raster["b"][m].astype(np.float64)
    c = 1.0 - a - b
    v = np.asarray(vals, dtype=np.float64)
    out[m] = a*v[conn[e, 0]] + b*v[conn[e, 1]] + c*v[conn[e, 2]]
    return out


# ── CLI: rasterise both solvers ─────────────────────────────────────────────
_ELEM_FIELDS = [("s11", "S", 0), ("s22", "S", 1), ("s12", "S", 2),
                ("e11", "E", 0), ("e22", "E", 1), ("e12", "E", 2),
                ("peeq", "PEEQ", None)]
_NODE_FIELDS = [("u1", "U", 0), ("u2", "U", 1)]


def _write(src, name, arr):
    p = os.path.join(cfg.CMP_OUT, f"grid_{src}_{name}.npy")
    np.save(p, arr)
    print(f"[grid] {os.path.basename(p)}  {arr.shape}  "
          f"{os.path.getsize(p)/1e6:.0f} MB")


def _rasterise(src, get_elem, get_node, raster, n_fr):
    frames = range(n_fr) if ALL_FRAMES else [n_fr - 1]
    for name, key, comp in _ELEM_FIELDS:
        stack = [elem_to_grid(get_elem(key, comp, k), raster) for k in frames]
        _write(src, name, np.array(stack) if ALL_FRAMES else stack[0])
    for name, key, comp in _NODE_FIELDS:
        stack = [node_to_grid(get_node(key, comp, k), raster) for k in frames]
        _write(src, name, np.array(stack) if ALL_FRAMES else stack[0])


def main():
    prob = inp_io.build_problem(verbose=True)
    raster = get_raster(prob)

    # ── in-house FEM ────────────────────────────────────────────────────────
    fp = os.path.join(cfg.FEM_OUT, f"{cfg.FEM_TAG}_frames.npz")
    if os.path.isfile(fp):
        d = np.load(fp)
        n_fr = len(d["time"])
        print(f"\n[grid] FEM: {n_fr} frames from {os.path.basename(fp)}")
        _rasterise("fem",
                   lambda k, c, i: d[k][i] if c is None else d[k][i][:, c],
                   lambda k, c, i: d[k][i][:, c],
                   raster, n_fr)
    else:
        S = os.path.join(cfg.FEM_OUT, f"{cfg.FEM_TAG}_S.npy")
        if os.path.isfile(S):
            fin = {k: np.load(os.path.join(cfg.FEM_OUT, f"{cfg.FEM_TAG}_{k}.npy"))
                   for k in ("S", "E", "PEEQ", "U")}
            print("\n[grid] FEM: final state only (no frames npz)")
            _rasterise("fem",
                       lambda k, c, i: fin[k] if c is None else fin[k][:, c],
                       lambda k, c, i: fin[k][:, c],
                       raster, 1)
        else:
            print("\n[grid] FEM fields not found - run fem_meso_driver.py first")

    # ── Abaqus element export ───────────────────────────────────────────────
    ap = os.path.join(cfg.AB_OUT, f"{cfg.JOB_NAME}_ab_S.npy")
    if os.path.isfile(ap):
        lab_e = np.load(os.path.join(cfg.AB_OUT, f"{cfg.JOB_NAME}_elem_labels.npy"))
        lab_n = np.load(os.path.join(cfg.AB_OUT, f"{cfg.JOB_NAME}_node_labels.npy"))
        pe = _perm(prob["elem_ids"], lab_e, "element")
        pn = _perm(prob["node_ids"], lab_n, "node")
        ab = {k: np.load(os.path.join(cfg.AB_OUT, f"{cfg.JOB_NAME}_ab_{k}.npy"),
                         mmap_mode="r")
              for k in ("S", "E", "PEEQ", "U")}
        n_fr = ab["S"].shape[0]
        print(f"\n[grid] Abaqus: {n_fr} frames from {cfg.AB_OUT}")
        _rasterise("abaqus",
                   lambda k, c, i: (np.asarray(ab[k][i])[pe] if c is None
                                    else np.asarray(ab[k][i])[pe, c]),
                   lambda k, c, i: np.asarray(ab[k][i])[pn, c],
                   raster, n_fr)
    else:
        print(f"\n[grid] Abaqus element export not found in {cfg.AB_OUT} - "
              f"run ../abaqus/odb_process_meso.py first")


def _perm(mesh_labels, odb_labels, what):
    """Index array reordering ODB-order rows into mesh order."""
    pos = {int(L): i for i, L in enumerate(odb_labels)}
    try:
        return np.array([pos[int(L)] for L in mesh_labels], dtype=np.int64)
    except KeyError as ex:
        raise RuntimeError(f"{what} label {ex} is in the mesh but not in the "
                           f"ODB export - the .inp and the .odb do not match. "
                           f"Re-run the Abaqus job.")


if __name__ == "__main__":
    main()
