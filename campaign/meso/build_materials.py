"""Regional materials of the Dataset 1 mesoscale meshes from method-B maps.

Geometry, face-to-region mapping and boundary data come from the frozen
problems (0_inputs/problems/<mesh>.npz, built 2026-09-22 with the corrected
majority mapping).  Only the material arrays are rebuilt here, with the
rules of the corrected closure chain (meso_config / inp_io):

  identified   region value = mean over the region's segmentation pixels of
               the 4-neighbour-filled yield map and of the filled K map in MPa
  hp_macro     Hall-Petch yield sigma0 + k d_eq^-1/2 (d_eq from the pixel
               area of the region; Hu (2025): twins 86 + 538, no twins
               74 + 522, d in um) with the uniform macroscopic K
  hp_hybrid    Hall-Petch yield with the identified regional K
  homogeneous  the macroscopic law of the campaign
  pixel        (fine_twins) raw maps sampled at every triangle centroid

  python meso/build_materials.py --maps-dir <dir with yield_macro_filled.npy
        and K_coef_filled.npy> --k-scale 461.8 --out meso/1_materials
  python meso/build_materials.py --validate-t075 <T075 dir> --k-scale 396
        (rebuilds the frozen T075 arrays of the 2026-09-22 problems)
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PX_TO_MM = 0.00184
SCALE_FACTOR = 1.84
MESHES = ("moderate_no_twins", "fine_no_twins", "moderate_twins", "fine_twins")
HP_FIT = {True: (86.0, 538.0), False: (74.0, 522.0)}        # sigma0 MPa, k MPa um^0.5
SEG = {True: "segmented_image.npy", False: "GB_without_twins_4.npy"}


def sha256(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fill_nan_iter_4nbr(a, max_iter=64):
    """meso_config.fill_nan_iter_4nbr, verbatim."""
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


def background_label(seg, n_region):
    """meso_config.seg_background_label without the environment override."""
    if not (seg == 0).any():
        return None
    n_nonzero = int(np.unique(seg[seg != 0]).size)
    if n_nonzero == n_region - 1:
        return None
    return 0


def label_means(seg_i, field):
    n = int(seg_i.max()) + 1
    cnt = np.bincount(seg_i.ravel(), minlength=n)
    tot = np.bincount(seg_i.ravel(), weights=np.asarray(field, np.float64).ravel(), minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return tot / cnt, cnt


def build(problem, seg, Y, Kmpa, twins, K_macro, homogeneous, float32_sy0=False):
    seg_i = seg.astype(np.int64)
    if not np.array_equal(seg, seg_i):
        raise ValueError("segmentation holds non-integer labels")
    lab = np.asarray(problem["segmentation_label"], np.int64)
    ym, cnt = label_means(seg_i, Y)
    km, _ = label_means(seg_i, Kmpa)
    if np.any(cnt[lab] == 0):
        raise ValueError("a mapped region label has no pixel")
    bg = background_label(seg_i, lab.size)
    s0, k_um = HP_FIT[twins]
    d_eq = 2.0 * np.sqrt(cnt * PX_TO_MM**2 / np.pi)
    with np.errstate(divide="ignore"):
        hp = s0 + (k_um / np.sqrt(1000.0)) / np.sqrt(d_eq)
    if bg is not None:
        hp[bg] = s0
    cast = (lambda a: a.astype(np.float32).astype(np.float64)) if float32_sy0 else (lambda a: a)
    out = dict(sy0_dic=cast(ym[lab]), K_dic=km[lab], sy0_hp=cast(hp[lab]),
               K_hp_macro=np.full(lab.size, float(K_macro)), K_hp_hybrid=km[lab])
    if homogeneous is None:                      # 2026-09-22 rule: global mean without background
        keep = seg_i != bg if bg is not None else np.ones(seg_i.shape, bool)
        hy, hk = float(np.mean(Y[keep])), float(np.mean(Kmpa[keep]))
    else:
        hy, hk = map(float, homogeneous)
    out["sy0_homogeneous"] = cast(np.full(lab.size, hy))
    out["K_homogeneous"] = np.full(lab.size, hk)
    info = dict(regions=int(lab.size), background_label=bg,
                pixels_per_region_median=float(np.median(cnt[lab])),
                yield_region_mean=float(ym[lab].mean()), yield_region_sd=float(ym[lab].std()),
                K_region_mean=float(km[lab].mean()), K_region_sd=float(km[lab].std()),
                yield_area_weighted=float(np.sum(ym[lab] * cnt[lab]) / cnt[lab].sum()),
                K_area_weighted=float(np.sum(km[lab] * cnt[lab]) / cnt[lab].sum()),
                hp_yield_mean=float(hp[lab].mean()), homogeneous=[hy, hk])
    return out, info


def pixel_sample(problem, Y, Kmpa, float32_sy0=False):
    nodes_raw = np.asarray(problem["coords"], np.float64) / SCALE_FACTOR
    el = np.asarray(problem["elements"], np.int64)
    c = np.rint(nodes_raw[el].mean(axis=1) * 1000.0).astype(np.int64) - 1
    c[:, 0] = np.clip(c[:, 0], 0, Y.shape[0] - 1)
    c[:, 1] = np.clip(c[:, 1], 0, Y.shape[1] - 1)
    sy = np.asarray(Y[c[:, 0], c[:, 1]], np.float64)
    if float32_sy0:
        sy = sy.astype(np.float32).astype(np.float64)
    return sy, np.asarray(Kmpa[c[:, 0], c[:, 1]], np.float64)


def bc_check(problem, U_path, V_path):
    """Largest difference between the frozen DIC boundary data and the same
    nodes sampled from the campaign's final DIC frame (inp_io convention)."""
    U = np.load(U_path, mmap_mode="r"); V = np.load(V_path, mmap_mode="r")
    c = np.asarray(problem["coords"], np.float64); b = np.asarray(problem["bc_nodes"], np.int64)
    px = np.clip(np.round(c[b, 0] * 1000.0 / SCALE_FACTOR).astype(np.int64), 0, U.shape[0] - 1)
    py = np.clip(np.round(c[b, 1] * 1000.0 / SCALE_FACTOR).astype(np.int64), 0, U.shape[1] - 1)
    u = np.asarray(problem["bc_u_dic"], np.float64)
    return float(max(np.max(np.abs(np.asarray(V[px, py], np.float64) * PX_TO_MM - u[:, 0])),
                     np.max(np.abs(np.asarray(U[px, py], np.float64) * PX_TO_MM - u[:, 1]))))


def load_maps(yield_path, K_path, k_scale):
    Y = fill_nan_iter_4nbr(np.load(yield_path).astype(np.float64))
    K = fill_nan_iter_4nbr(np.load(K_path).astype(np.float64) * float(k_scale))
    return Y, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps-dir")
    ap.add_argument("--validate-t075")
    ap.add_argument("--k-scale", type=float, required=True)
    ap.add_argument("--k-macro", type=float)
    ap.add_argument("--homogeneous", default="", help="sigma0:K of the macroscopic law")
    ap.add_argument("--inputs", default=str(HERE / "0_inputs"))
    ap.add_argument("--out", default=str(HERE / "1_materials"))
    ap.add_argument("--dic-U", help="final DIC frame used by the pixel-scale campaign")
    ap.add_argument("--dic-V")
    a = ap.parse_args()
    inp = Path(a.inputs)
    if a.validate_t075:
        d = Path(a.validate_t075)
        Y, K = load_maps(d / "yield_vm_thresh.npy", d / "K_coef_filled.npy", a.k_scale)
        rep = {}
        for mesh in MESHES:
            with np.load(inp / "problems" / f"{mesh}.npz") as z:
                pb = {k: z[k] for k in z.files}
            twins = "no_twins" not in mesh
            seg = np.load(inp / "segmentation" / SEG[twins])
            out, info = build(pb, seg, Y, K, twins, 380.0, None, float32_sy0=True)
            r = {}
            for k, v in out.items():
                ref = np.asarray(pb[k], np.float64)
                r[k] = dict(max_abs=float(np.max(np.abs(v - ref))), max_rel=float(np.max(np.abs(v - ref) / np.abs(ref))))
            if "sy0_pixel" in pb:
                sy, kp = pixel_sample(pb, Y, K, float32_sy0=True)
                r["sy0_pixel"] = dict(max_abs=float(np.max(np.abs(sy - pb["sy0_pixel"]))))
                r["K_pixel"] = dict(max_abs=float(np.max(np.abs(kp - pb["K_pixel"]))))
            rep[mesh] = dict(info=info, comparison=r)
            print(mesh, json.dumps(r))
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "validation_t075.json").write_text(json.dumps(rep, indent=2) + "\n")
        return
    md = Path(a.maps_dir)
    yp, kp = md / "yield_macro_filled.npy", md / "K_coef_filled.npy"
    Y, K = load_maps(yp, kp, a.k_scale)
    hom = tuple(map(float, a.homogeneous.split(":")))
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(maps={str(yp): sha256(yp), str(kp): sha256(kp)}, k_scale_mpa=a.k_scale,
                   k_macro_mpa=a.k_macro, homogeneous=list(hom), meshes={})
    for mesh in MESHES:
        pf = inp / "problems" / f"{mesh}.npz"
        with np.load(pf) as z:
            pb = {k: z[k] for k in z.files}
        twins = "no_twins" not in mesh
        sf = inp / "segmentation" / SEG[twins]
        out, info = build(pb, np.load(sf), Y, K, twins, a.k_macro, hom)
        if mesh == "fine_twins":
            out["sy0_pixel"], out["K_pixel"] = pixel_sample(pb, Y, K)
            info["pixel_yield_mean"] = float(out["sy0_pixel"].mean())
            info["pixel_K_mean"] = float(out["K_pixel"].mean())
        if a.dic_U:
            info["frozen_bc_vs_campaign_dic_max_mm"] = bc_check(pb, a.dic_U, a.dic_V)
        dst = out_dir / f"{mesh}.npz"
        np.savez(dst, **out)
        info.update(problem_sha256=sha256(pf), segmentation_sha256=sha256(sf), materials_sha256=sha256(dst))
        summary["meshes"][mesh] = info
        print(f"[materials] {mesh}: {info['regions']} regions, yield {info['yield_region_mean']:.1f} "
              f"+/- {info['yield_region_sd']:.1f} MPa, K {info['K_region_mean']:.1f} +/- {info['K_region_sd']:.1f} MPa, "
              f"HP yield {info['hp_yield_mean']:.1f} MPa; frozen BC vs campaign DIC "
              f"{info.get('frozen_bc_vs_campaign_dic_max_mm', float('nan')):.2e} mm", flush=True)
    (out_dir / "materials.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
