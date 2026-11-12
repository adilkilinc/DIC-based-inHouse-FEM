"""Dataset 1 method-B identification for one DIC key (full field).

  python identification/d1_identify.py a100_it240 --reference write ...   (production key)
  python identification/d1_identify.py a50_it240  --reference read  ...   (other keys)

The production key also fixes the field-mean reference: the macroscopic law
is recomputed from the field-mean history against the synchronised machine
stress and must agree with the law in campaign.json (--law, --law-tol), and
SLOP_REF is the plastic slope of the field-mean history by the pixel rule.
Both are written to <out-dir>/reference.json, which every other key reads,
so all keys share one SLOP_REF.

Environment: OLDEXP_CODE_DIR (alpha_config.py, 02_identify.py), OLDEXP_ROOT.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
import numpy as np
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.environ.get("OLDEXP_CODE_DIR", os.path.join(HERE, "dataset1"))
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, HERE)
import alpha_config as cfg  # noqa: E402
import macro_reference as mr  # noqa: E402
import identify_common as ic  # noqa: E402

_spec = importlib.util.spec_from_file_location("m1", os.path.join(CODE_DIR, "02_identify.py"))
m1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m1)
AREA_MM2 = 7.00 * 1.71        # Dataset 1 gauge section


def d1_frame_stress(path, n_frames, area=AREA_MM2, smooth=5):
    """Machine stress at the DIC frames: the DIC series is the first loading
    ramp of Traction_1.csv (frame 1 at 1 % of the rise, last frame at the
    first force peak, constant frame rate; 41 intervals over the ramp give
    30.6 Hz against the documented 30 Hz)."""
    raw = np.genfromtxt(path, delimiter=",", skip_header=1)
    t, F = raw[:, 0], raw[:, 1]
    ok = np.isfinite(t) & np.isfinite(F)
    t, F = t[ok], F[ok]
    Fs = gaussian_filter(F, smooth)
    i_cross = int(np.argmax(Fs > 1000.0))
    tc = t[i_cross]
    F_hold = float(np.median(Fs[(t > tc - 2.0) & (t < tc - 0.5)]))
    win = (t >= tc) & (t <= tc + 1.0)
    i_peak = int(np.nonzero(win)[0][np.argmax(Fs[win])])
    thr = F_hold + 0.01 * (Fs[i_peak] - F_hold)
    i_start = int(np.nonzero((Fs <= thr) & (np.arange(t.size) < i_cross))[0][-1])
    tk = np.linspace(t[i_start], t[i_peak], n_frames)
    stress = np.interp(tk, t, Fs) / area
    return stress, dict(t_start=float(t[i_start]), t_peak=float(t[i_peak]),
                        frame_rate_hz=float((n_frames - 1) / (t[i_peak] - t[i_start])),
                        F_hold_N=F_hold, F_peak_N=float(Fs[i_peak]), area_mm2=area)


def evm_history(key):
    """Equivalent-strain history exactly as 02_identify (full frame)."""
    u_dir, v_dir = cfg.uv_dirs(key)
    fu = cfg.get_sorted_npy_files(u_dir, "U")
    fv = cfg.get_sorted_npy_files(v_dir, "V")
    if not fu or len(fu) != len(fv):
        raise SystemExit(f"U/V frames missing or mismatched in {u_dir}")
    out = []
    for a, b in zip(fu, fv):
        U = np.load(os.path.join(u_dir, a)); V = np.load(os.path.join(v_dir, b))
        Exy, Exx = np.gradient(U)
        Eyy, Eyx = np.gradient(V)
        d = m1.deviatoric_strain_plane_stress(Exx, Eyy, 0.5 * (Exy + Eyx), cfg.NU)
        out.append(m1.evm_from_edev(*d))
    E = m1.repair_bad_frames(np.array(out), cfg.BAD_FRAMES.get(key, []))
    return gaussian_filter(E, sigma=[1, 0, 0]).astype(np.float32), u_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key")
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--traction", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sigma0", type=float, required=True)
    ap.add_argument("--win-fracs", default="0.11:0.80")
    ap.add_argument("--reference", choices=("write", "read"), required=True)
    ap.add_argument("--law", default="", help="E:nu:n:sigma0:K of campaign.json (write mode)")
    ap.add_argument("--law-tol", default="0.5:1.0", help="sigma0 MPa : K percent")
    ap.add_argument("--k-scale", type=float, required=True)
    a = ap.parse_args()
    t0 = time.time()
    variants = [ic.parse_variant(s) for s in a.variants]
    out_root = os.path.join(a.out_dir, a.key)
    os.makedirs(out_root, exist_ok=True)
    fr = tuple(map(float, a.win_fracs.split(":")))

    Evm, u_dir = evm_history(a.key)
    N = Evm.shape[0]
    emean = Evm.mean(axis=(1, 2)).astype(np.float64)
    stress, sinfo = d1_frame_stress(a.traction, N)
    np.save(os.path.join(out_root, "frame_stress.npy"), stress)
    np.save(os.path.join(out_root, "field_mean_strain.npy"), emean)
    lo, hi = mr.load_window(stress, a.sigma0, *fr)
    windows = {"load": (lo, hi), "auto": ic.auto_window(emean, 2e-4)}
    print(f"[d1] {a.key}: Evm {Evm.shape} from {u_dir} ({time.time()-t0:.0f} s); load window {lo}-{hi} "
          f"({stress[lo-1]:.1f}-{stress[hi-1]:.1f} MPa); auto window {windows['auto']}", flush=True)
    val = dict(key=a.key, frame_stress=sinfo, windows={k: list(v) for k, v in windows.items()},
               kernel_identity=ic.kernel_identity(Evm, lo, hi, 2e-4))
    print(f"[d1] stress kernel with s = frame number vs frame kernel: {val['kernel_identity']}", flush=True)
    prod = variants[0]
    own = mr.reference(emean, stress, lo, hi, prod["offset"], prod["abscissa"])
    val["field_mean_reference_own"] = own
    ref_path = os.path.join(a.out_dir, "reference.json")
    if a.reference == "write":
        E, nu, n, s0, K = map(float, a.law.split(":"))
        tol_s, tol_K = map(float, a.law_tol.split(":"))
        # offset from the record's own fitted elastic line (E=None), as for the
        # pixels and for the Dataset 2 extensometer law; E is the FEM modulus
        law = mr.macro_law(emean, stress, lo, hi, None, nu, n)
        law_sc = mr.self_consistent_law(emean, stress, s0, fr, None, nu, n)
        law_old_window = mr.macro_law(emean, stress, 4, 12, None, nu, n)
        law_line_of_E = mr.macro_law(emean, stress, lo, hi, E, nu, n)
        ok = abs(law["sigma0_mpa"] - s0) <= tol_s and abs(law["K_mpa"] / K - 1) * 100 <= tol_K
        ref = dict(dataset="dataset1", production_key=a.key, variant=prod, fit_lo=lo, fit_hi=hi,
                   win_fracs=list(fr), sigma0_window_mpa=a.sigma0, slop_ref=own["slop_ref"],
                   reference=own, law_campaign=dict(E=E, nu=nu, n=n, sigma0=s0, K=K),
                   law_recomputed=law, law_self_consistent=law_sc, law_window_4_12=law_old_window,
                   law_offset_from_line_of_FEM_E=law_line_of_E,
                   law_check="PASS" if ok else "FAIL", tolerance=dict(sigma0_mpa=tol_s, K_percent=tol_K),
                   field_mean_strain=emean.tolist(), frame_stress=stress.tolist(),
                   frame_reference=mr.reference(emean, stress, lo, hi, prod["offset"], "frame"))
        with open(ref_path, "w") as fh:
            json.dump(ref, fh, indent=2)
        print(f"[d1] macro law recomputed: sigma0 {law['sigma0_mpa']:.2f} MPa, K {law['K_mpa']:.1f} MPa, "
              f"E_app {law['E_app_mpa']/1e3:.1f} GPa (campaign {s0} / {K}) -> {ref['law_check']}", flush=True)
        print(f"[d1] SLOP_REF (field mean, {prod['abscissa']} onset at frame {own['onset_count']}, "
              f"{own['yield_stress_mpa']:.1f} MPa) = {own['slop_ref']:.4e}", flush=True)
        if not ok:
            raise SystemExit("macro law does not reproduce campaign.json; stopping before identification")
    else:
        ref = json.load(open(ref_path))
        if ref["fit_lo"] != lo or ref["fit_hi"] != hi:
            print(f"[d1] note: load window {lo}-{hi} differs from the reference key's "
                  f"{ref['fit_lo']}-{ref['fit_hi']}", flush=True)
    slop_ref = float(ref["slop_ref"])
    val["slop_ref_used"] = slop_ref
    rows = ic.run_variants(Evm, variants, stress, windows, slop_ref, a.k_scale, out_root, m1,
                           m1.replace_nan_with_neighbors, cfg.YIELD_SAFETY_FLOOR_MPA,
                           extra=dict(key=a.key, dataset="dataset1", nu=cfg.NU))
    val["wall_s"] = round(time.time() - t0, 1)
    with open(os.path.join(out_root, "validation.json"), "w") as fh:
        json.dump(val, fh, indent=2)
    with open(os.path.join(out_root, "summary.json"), "w") as fh:
        json.dump(dict(key=a.key, variants=rows, validation=val), fh, indent=2)
    print(f"[d1] done in {time.time()-t0:.0f} s -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
