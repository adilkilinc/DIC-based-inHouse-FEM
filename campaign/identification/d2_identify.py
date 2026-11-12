"""Dataset 2 method-B identification for one DIC key (full field, 1537
loading frames of rb_sp2).

  python identification/d2_identify.py a80_it240 --reference write ...   (production key)
  python identification/d2_identify.py a50_it240 --reference read  ...

Strain history: the cached Evm_history.npy of the production identification
(out/yield_hardening/<key>/) when its metadata matches the loading window,
otherwise recomputed exactly as 02_yield_hardening_generic.py (not cached).
Machine stress per frame: 02's macro_stress_for_dic_files (data_labjack.csv,
frame times from the image names).  SLOP_REF is the plastic slope of the
field-mean history of the production key by the pixel rule, written to
<out-dir>/reference.json and read by the other keys.  The macroscopic law of
Dataset 2 stays the extensometer law of new_experiment_config.py; the
field-mean law is written for comparison only.

Environment: NEWEXP_CODE_DIR (new_experiment_config.py,
02_yield_hardening_generic.py), NEWEXP_DATA_DIR, NEWEXP_SRC_ROOT,
NEWEXP_MANIFEST, NEWEXP_DIS_ITERATIONS=240.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
import numpy as np
from scipy.ndimage import gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.environ.get("NEWEXP_CODE_DIR", os.path.join(HERE, "dataset2"))
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, HERE)
import new_experiment_config as cfg  # noqa: E402
import macro_reference as mr  # noqa: E402
import identify_common as ic  # noqa: E402

_spec = importlib.util.spec_from_file_location("m02", os.path.join(CODE_DIR, "02_yield_hardening_generic.py"))
m02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m02)


def evm_history(key, files_U, files_V, u_dir, v_dir):
    d = os.path.join(cfg.YH_DIR, key)
    p, pj = os.path.join(d, "Evm_history.npy"), os.path.join(d, "Evm_history.json")
    N = len(files_U)
    if os.path.isfile(p) and os.path.isfile(pj):
        meta = json.load(open(pj))
        if (meta.get("n_frames") == N and meta.get("key") == key and meta.get("frame_start") == cfg.FRAME_START
                and meta.get("frame_end") == cfg.FRAME_END and meta.get("bad_frames") == list(cfg.BAD_FRAMES.get(key, []))):
            E = np.load(p)
            return E, dict(source="cache", path=p, meta=meta)
        print(f"[d2] cache metadata does not match; recomputing ({meta})", flush=True)
    E = None
    for m, (a, b) in enumerate(zip(files_U, files_V)):
        if m % 200 == 0:
            print(f"  frame {m}/{N}", flush=True)
        Exy, Exx = np.gradient(np.load(os.path.join(u_dir, a)))
        Eyy, Eyx = np.gradient(np.load(os.path.join(v_dir, b)))
        dd = cfg.deviatoric_strain_plane_stress(Exx, Eyy, 0.5 * (Exy + Eyx), cfg.NU)
        e = cfg.evm_from_edev(*dd)
        if E is None:
            E = np.empty((N,) + e.shape, np.float32)
        E[m] = e
    E = m02.repair_bad_frames(E, cfg.BAD_FRAMES.get(key, []))
    # temporal Gaussian (sigma 1, as 02), column blocks in float64 to bound memory
    for j0 in range(0, E.shape[2], 128):
        E[:, :, j0:j0 + 128] = gaussian_filter1d(E[:, :, j0:j0 + 128].astype(np.float64), 1.0, axis=0)
    return E, dict(source="recomputed")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key")
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sigma0", type=float, required=True)
    ap.add_argument("--win-fracs", default="0.11:0.80")
    ap.add_argument("--reference", choices=("write", "read"), required=True)
    ap.add_argument("--k-scale", type=float, required=True)
    ap.add_argument("--n-exp", type=float, required=True)
    ap.add_argument("--E", type=float, required=True)
    a = ap.parse_args()
    t0 = time.time()
    variants = [ic.parse_variant(s) for s in a.variants]
    if any(v["window"] != "load" for v in variants):
        raise SystemExit("Dataset 2 uses the load window only")
    out_root = os.path.join(a.out_dir, a.key)
    os.makedirs(out_root, exist_ok=True)
    fr = tuple(map(float, a.win_fracs.split(":")))

    u_dir, v_dir = cfg.uv_dirs(a.key)
    fu = cfg.window_files(cfg.get_sorted_npy_files(u_dir, "U"))
    fv = cfg.window_files(cfg.get_sorted_npy_files(v_dir, "V"))
    if not fu or len(fu) != len(fv):
        raise SystemExit(f"U/V frames missing or mismatched in {u_dir}")
    stress, times = m02.macro_stress_for_dic_files(fu)
    Evm, src = evm_history(a.key, fu, fv, u_dir, v_dir)
    N = Evm.shape[0]
    if N != len(stress):
        raise SystemExit(f"{N} strain frames but {len(stress)} stress values")
    emean = Evm.mean(axis=(1, 2), dtype=np.float64)
    np.save(os.path.join(out_root, "frame_stress.npy"), stress)
    np.save(os.path.join(out_root, "field_mean_strain.npy"), emean)
    lo, hi = mr.load_window(stress, a.sigma0, *fr)
    windows = {"load": (lo, hi)}
    print(f"[d2] {a.key}: Evm {Evm.shape} ({src['source']}, {time.time()-t0:.0f} s); load window {lo}-{hi} "
          f"({stress[lo-1]:.1f}-{stress[hi-1]:.1f} MPa); config window "
          f"{cfg.ELASTIC_FIT_FIRST}-{cfg.ELASTIC_FIT_LAST}", flush=True)
    val = dict(key=a.key, evm_source={k: v for k, v in src.items() if k != "meta"},
               windows={"load": [lo, hi], "config": [cfg.ELASTIC_FIT_FIRST, cfg.ELASTIC_FIT_LAST]},
               frame_times=[float(times[0]), float(times[-1])],
               kernel_identity=ic.kernel_identity(Evm, lo, hi, 2e-4))
    print(f"[d2] stress kernel with s = frame number vs frame kernel: {val['kernel_identity']}", flush=True)
    prod = variants[0]
    own = mr.reference(emean, stress, lo, hi, prod["offset"], prod["abscissa"])
    val["field_mean_reference_own"] = own
    ref_path = os.path.join(a.out_dir, "reference.json")
    if a.reference == "write":
        law_dic = mr.macro_law(emean, stress, lo, hi, a.E, cfg.NU, a.n_exp)
        ref = dict(dataset="dataset2", production_key=a.key, variant=prod, fit_lo=lo, fit_hi=hi,
                   win_fracs=list(fr), sigma0_window_mpa=a.sigma0, slop_ref=own["slop_ref"], reference=own,
                   slop_ref_extensometer=cfg.SLOP_REF,
                   law_campaign=dict(E=a.E, nu=cfg.NU, n=a.n_exp, sigma0=a.sigma0, K=a.k_scale),
                   law_field_mean_for_comparison=law_dic,
                   frame_reference=mr.reference(emean, stress, lo, hi, prod["offset"], "frame"),
                   field_mean_strain=emean.tolist(), frame_stress=stress.tolist())
        with open(ref_path, "w") as fh:
            json.dump(ref, fh, indent=2)
        print(f"[d2] SLOP_REF (field mean, {prod['abscissa']} onset at frame {own['onset_count']}, "
              f"{own['yield_stress_mpa']:.1f} MPa) = {own['slop_ref']:.4e} (extensometer rule "
              f"{cfg.SLOP_REF:.4e}); field-mean law sigma0 {law_dic['sigma0_mpa']:.1f} MPa, "
              f"K {law_dic['K_mpa']:.1f} MPa (extensometer {a.sigma0} / {a.k_scale})", flush=True)
    else:
        ref = json.load(open(ref_path))
    slop_ref = float(ref["slop_ref"])
    val["slop_ref_used"] = slop_ref
    rows = ic.run_variants(Evm, variants, stress, windows, slop_ref, a.k_scale, out_root, m02,
                           cfg.replace_nan_with_neighbors, cfg.YIELD_SAFETY_FLOOR_MPA,
                           extra=dict(key=a.key, dataset="dataset2", E_mpa=a.E, nu=cfg.NU))
    # production-rule reproduction: frame abscissa, config window, extensometer
    # SLOP_REF must give the existing yield_hardening/<key> onset map
    prodS = os.path.join(cfg.YH_DIR, a.key, "S_p_yield_onset_frame.npy")
    if os.path.isfile(prodS):
        v = dict(name="repro", offset=cfg.ONSET_OFFSET, hi_shift=0, window="load", abscissa="frame")
        S_rep, _ = ic.onset_map(Evm, v, stress, cfg.ELASTIC_FIT_FIRST, cfg.ELASTIC_FIT_LAST)
        P = np.load(prodS)
        both = np.isfinite(S_rep) & np.isfinite(P)
        val["production_onset_reproduced"] = dict(identical=float(np.mean(S_rep[both] == P[both])),
                                                  nan_mismatch=int(np.sum(np.isfinite(S_rep) != np.isfinite(P))))
        print(f"[d2] production onset map reproduced: {val['production_onset_reproduced']}", flush=True)
    val["wall_s"] = round(time.time() - t0, 1)
    with open(os.path.join(out_root, "validation.json"), "w") as fh:
        json.dump(val, fh, indent=2)
    with open(os.path.join(out_root, "summary.json"), "w") as fh:
        json.dump(dict(key=a.key, variants=rows, validation=val), fh, indent=2)
    print(f"[d2] done in {time.time()-t0:.0f} s -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
