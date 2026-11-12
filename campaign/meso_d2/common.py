# -*- coding: utf-8 -*-
"""Shared paths, definitions and helpers of the Dataset 2 mesoscale package.

The package sits inside the campaign folder and reads it without writing to
it: campaign.json, the production identification maps of Dataset 2, the
mesoscale solver (mesoscale/solver/fem_tri.py) and, for the
pixel-scale reference, 2_data/dataset2_*_f01_crop.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CAMPAIGN = Path(os.environ.get("D2MESO_CAMPAIGN", HERE.parent)).resolve()
INPUTS = HERE / "0_inputs"
PROBLEMS = HERE / "1_problems"
RESULTS = HERE / "2_results"
ANALYSIS = HERE / "3_analysis"
SOLVER = HERE.parents[1] / "mesoscale" / "solver"
PX_MM = 0.00345
MESHES = ("moderate_no_twins", "fine_no_twins", "moderate_twins", "fine_twins")
SHORT = {"moderate_no_twins": "mnt", "fine_no_twins": "fnt", "moderate_twins": "mt", "fine_twins": "ft"}


def sha256(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def campaign():
    return json.loads((CAMPAIGN / "campaign.json").read_text(encoding="utf-8"))


def package():
    return json.loads((HERE / "cases.json").read_text(encoding="utf-8"))


def settings():
    """Dataset 2 constants from campaign.json; solver settings of the locked mesoscale runs."""
    cfg = campaign()
    d2, idc, mc = cfg["datasets"]["dataset2"], cfg["identification"], cfg["meso"]
    hom = d2["materials"]["homogeneous"]
    root = os.environ.get(cfg["roots"]["dataset2"]["env"]) or cfg["roots"]["dataset2"]["hpc" if os.name != "nt" else "local"]
    key = idc["dataset2"]["production_key"]
    n = int(idc["dataset2"]["final_frame"])
    root = Path(root)
    geo = cfg["field_geometry"]["dataset2"]
    return dict(E=float(d2["E_mpa"]), nu=float(d2["nu"]), n_exp=float(d2["n_exp"]), k_scale=float(d2["K_scale_mpa"]),
                homogeneous=(float(hom["yield_constant_mpa"]), float(hom["K_constant_mpa"])),
                source_shape=tuple(d2["source_shape"]), field_shape=tuple(geo["field_shape"]),
                crop_offset=tuple(geo["crop_offset"]), key=key, final_frame=n,
                maps=CAMPAIGN / idc["out_dir"] / "dataset2" / key / idc["variant_dir"],
                U=root / f"disflow/{key}/U/U_{n}.npy", V=root / f"disflow/{key}/V/V_{n}.npy",
                tables=cfg["hardening_tables"], table=mc["table"], n_inc=int(mc["n_inc"]),
                max_nr=int(mc["max_nr"]), nr_tol=float(mc["nr_tol"]),
                snapshot_fractions=[float(x) for x in mc["snapshot_fractions"]])


# ---------------------------------------------------------------- equivalent strain
def evm_printed(exx, eyy, exy, nu):
    """Thesis Eq. (evmps); exy is the tensorial shear strain."""
    s = exx + eyy
    q = (1.0 - nu) ** 2 * (exx * exx + eyy * eyy - exx * eyy + 3.0 * exy * exy) + nu * s * s
    return 2.0 / (3.0 * (1.0 - nu)) * np.sqrt(np.maximum(q, 0.0))


def evm_voigt(E, nu):
    """Solver output E[:, 0:3] = e11, e22, engineering shear."""
    return evm_printed(E[..., 0], E[..., 1], 0.5 * E[..., 2], nu)


def vm_stress(S):
    return np.sqrt(np.maximum(S[..., 0] ** 2 - S[..., 0] * S[..., 1] + S[..., 1] ** 2 + 3 * S[..., 2] ** 2, 0.0))


def dic_evm(U_path, V_path, nu, shape, offset):
    """Measured equivalent strain on the analysis grid: the gradients of
    analyse_results.dic_evm_from (full field, then cropped), printed definition."""
    U = np.asarray(np.load(U_path, mmap_mode="r"), float)
    V = np.asarray(np.load(V_path, mmap_mode="r"), float)
    e11 = np.gradient(V, axis=0)
    e22 = np.gradient(U, axis=1)
    e12 = np.gradient(V, axis=1) + np.gradient(U, axis=0)
    ox, oy = offset
    return evm_printed(e11, e22, 0.5 * e12, nu)[ox:ox + shape[0], oy:oy + shape[1]]


# ---------------------------------------------------------------- maps and boundary data
def fill_nan_iter_4nbr(a, max_iter=64):
    """meso/build_materials.py::fill_nan_iter_4nbr, verbatim."""
    a = a.astype(float, copy=True)
    for _ in range(max_iter):
        nan = np.isnan(a)
        if not nan.any():
            break
        up = np.pad(a, ((1, 0), (0, 0)), constant_values=np.nan)[:-1, :]
        down = np.pad(a, ((0, 1), (0, 0)), constant_values=np.nan)[1:, :]
        left = np.pad(a, ((0, 0), (1, 0)), constant_values=np.nan)[:, :-1]
        right = np.pad(a, ((0, 0), (0, 1)), constant_values=np.nan)[:, 1:]
        stack = np.stack([up, down, left, right], axis=0)
        sums = np.nansum(stack, axis=0)
        counts = np.sum(np.isfinite(stack), axis=0)
        means = sums / np.where(counts == 0, 1, counts)
        a[nan & (counts > 0)] = means[nan & (counts > 0)]
    return a


def bilinear(F, r, c):
    """F sampled at fractional (row, col), indices clamped to the array: the
    value of pixel (i, j) sits at (i, j), the node convention of the pixel model."""
    r = np.clip(r, 0.0, F.shape[0] - 1.0)
    c = np.clip(c, 0.0, F.shape[1] - 1.0)
    r0 = np.minimum(np.floor(r).astype(np.int64), F.shape[0] - 2)
    c0 = np.minimum(np.floor(c).astype(np.int64), F.shape[1] - 2)
    fr, fc = r - r0, c - c0
    return ((1 - fr) * (1 - fc) * F[r0, c0] + fr * (1 - fc) * F[r0 + 1, c0]
            + (1 - fr) * fc * F[r0, c0 + 1] + fr * fc * F[r0 + 1, c0 + 1])


# ---------------------------------------------------------------- rasterisation
def raster_index(coords_mm, elements, shape, origin=(0, 0)):
    """Triangle containing the centre of every pixel of the analysis sub-grid
    starting at `origin`: pixel (i, j) centre at ((i + 0.5), (j + 0.5)) raw units."""
    from numba import njit

    @njit(cache=False)
    def _raster(xy, el, nr, nc, out):
        for e in range(el.shape[0]):
            x0, y0 = xy[el[e, 0], 0], xy[el[e, 0], 1]
            x1, y1 = xy[el[e, 1], 0], xy[el[e, 1], 1]
            x2, y2 = xy[el[e, 2], 0], xy[el[e, 2], 1]
            d = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if d == 0.0:
                continue
            i0 = max(int(np.floor(min(x0, x1, x2) - 0.5)), 0)
            i1 = min(int(np.ceil(max(x0, x1, x2))), nr - 1)
            j0 = max(int(np.floor(min(y0, y1, y2) - 0.5)), 0)
            j1 = min(int(np.ceil(max(y0, y1, y2))), nc - 1)
            for i in range(i0, i1 + 1):
                px = i + 0.5
                for j in range(j0, j1 + 1):
                    py = j + 0.5
                    a = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / d
                    b = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / d
                    c = 1.0 - a - b
                    if a >= -1e-12 and b >= -1e-12 and c >= -1e-12:
                        out[i, j] = e

    out = np.full(tuple(shape), -1, np.int64)
    _raster(np.ascontiguousarray(np.asarray(coords_mm, np.float64) / PX_MM),
            np.ascontiguousarray(elements, np.int64), int(shape[0]), int(shape[1]), out)
    return out


def corr(a, b):
    return float(np.corrcoef(np.ravel(a), np.ravel(b))[0, 1])


def dice10(a, b):
    aa = a >= np.percentile(a, 90)
    bb = b >= np.percentile(b, 90)
    return float(2 * np.count_nonzero(aa & bb) / max(np.count_nonzero(aa) + np.count_nonzero(bb), 1))


def metrics(a, b, mask):
    a, b = a[mask], b[mask]
    d = a - b
    return dict(r=corr(a, b), rel_rms_pct=float(100 * np.sqrt(np.mean(d * d)) / np.sqrt(np.mean(b * b))),
                dice_top10=dice10(a, b), contrast=float(a.std() / b.std()), mean_ratio=float(a.mean() / b.mean()))


def write_csv(path, rows):
    import csv
    if not rows:
        return
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
