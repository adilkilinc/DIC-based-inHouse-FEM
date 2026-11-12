"""Hardening maps with the power-law slope-ratio rule (Chapter 4, Eq. ludw2 and the text after it).

The locked rule assigns K_j = K s_ref / s_j.  For two points with a common stress
history, yield stress and exponent n, the Ludwik law gives instead
K_j = K (s_ref / s_j)^n.  This script builds the second map from the stored
first one, for Dataset 1 (a100_it240) and Dataset 2 (a80_it240):

  ratio    K_coef_raw.npy of the locked identification (s_ref / s_j)
  kept     the pixels kept by the locked rule (K_flag = 0, ratio in [0, 3]);
           the power rule is applied to them
  filled   the pixels flagged by the locked rule are filled with the same
           neighbour fill as the identification (02_identify / new_experiment_config)
  yield    unchanged (copied)

Gate: the same fill applied to the kept linear ratios must reproduce the locked
K_coef_filled.npy, which shows that only the exponent differs between the two maps.
When the locked ratio and flag maps are no longer on disk (Dataset 1 on Zeus, 14 Oct),
they are taken from this package's identification rerun (d_identify240.sbatch), which
must first reproduce the locked filled K and yield maps.

Writes 0_identification/<dataset>/<key>/kpow/{K_coef_raw, K_coef_filled, K_flag,
yield_macro_filled}.npy and diagnostics.json.

    python d_kpow_maps.py [--datasets dataset1 dataset2]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import common as K

TOL = 1e-5


def fill_function(dataset: str):
    """The neighbour fill used by the identification of that dataset."""
    if dataset == "dataset1":
        code = Path(os.environ.get("OLDEXP_CODE_DIR", K.FROZEN / "identification" / "dataset1"))
        sys.path.insert(0, str(code))
        spec = importlib.util.spec_from_file_location("m1_02_identify", code / "02_identify.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m.replace_nan_with_neighbors, str(code / "02_identify.py")
    code = Path(os.environ.get("NEWEXP_CODE_DIR", K.FROZEN / "identification" / "dataset2"))
    sys.path.insert(0, str(code))
    import new_experiment_config as cfg  # noqa: E402
    return cfg.replace_nan_with_neighbors, str(code / "new_experiment_config.py")


def save32(path: Path, arr):
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, np.asarray(arr, dtype=np.float32))
    os.replace(tmp, path)


def stats(a):
    f = np.asarray(a, np.float64)
    f = f[np.isfinite(f)]
    return dict(mean=float(f.mean()), std=float(f.std()), p5=float(np.percentile(f, 5)),
                p50=float(np.percentile(f, 50)), p95=float(np.percentile(f, 95)))


def build(dataset: str, key: str, n: float, variant: str) -> dict:
    t0 = time.time()
    src = K.FROZEN / "0_identification" / dataset / key / "off_stress"
    dst = K.PKG / "0_identification" / dataset / key / variant
    dst.mkdir(parents=True, exist_ok=True)
    rsrc, rerun = src, None
    if not K.locked_raw_available(dataset, key):
        # the locked ratio and flag maps are no longer on disk: use the identification rerun of this package
        # (d_identify240.sbatch: frozen code, same DIC, locked reference.json), which must reproduce the locked
        # filled K and yield maps exactly
        rsrc = K.PKG / "0_identification" / dataset / key / "off_stress"
        if not K.rerun_raw_available(dataset, key):
            raise SystemExit(f"[{dataset}] locked K_coef_raw/K_flag missing and no rerun in {rsrc}")
        rerun = {}
        for f in ("K_coef_filled.npy", "yield_macro_filled.npy"):
            x = np.load(rsrc / f).astype(np.float64); y = np.load(src / f).astype(np.float64)
            both = np.isfinite(x) & np.isfinite(y)
            rerun[f] = dict(max_abs_diff=float(np.max(np.abs(x[both] - y[both]))),
                            nan_mismatch=int(np.sum(np.isfinite(x) != np.isfinite(y))))
        print(f"[{dataset}] rerun vs locked maps: {rerun}", flush=True)
        if any(v["max_abs_diff"] > 1e-6 or v["nan_mismatch"] for v in rerun.values()):
            raise SystemExit(f"[{dataset}] the identification rerun does not reproduce the locked maps; nothing written")
    raw = np.load(rsrc / "K_coef_raw.npy").astype(np.float64)
    flag = np.load(rsrc / "K_flag.npy")
    locked = np.load(src / "K_coef_filled.npy").astype(np.float64)
    fill, fill_src = fill_function(dataset)
    kept = flag == 0
    if np.any(raw[kept] < 0) or np.any(raw[kept] > 3) or not np.all(np.isfinite(raw[kept])):
        raise SystemExit(f"{dataset}: kept ratios outside [0, 3]")
    lin = np.where(kept, raw, np.nan)
    lin_f = fill(lin)
    both = np.isfinite(lin_f) & np.isfinite(locked)
    dev = float(np.max(np.abs(lin_f[both] - locked[both])))
    nan_mismatch = int(np.sum(np.isfinite(lin_f) != np.isfinite(locked)))
    print(f"[{dataset}] gate: fill of the kept linear ratios vs locked K_coef_filled: max |diff| {dev:.2e}, "
          f"NaN mismatch {nan_mismatch}", flush=True)
    if dev > TOL or nan_mismatch:
        raise SystemExit(f"[{dataset}] the locked fill is not reproduced; nothing written")
    pw = np.where(kept, np.power(np.where(kept, raw, 0.0), n), np.nan)
    pw_f = fill(pw)
    raw_out = np.where(kept, pw, raw)
    save32(dst / "K_coef_raw.npy", raw_out)
    save32(dst / "K_coef_filled.npy", pw_f)
    np.save(dst / "K_flag.npy", flag)
    shutil.copy2(src / "yield_macro_filled.npy", dst / "yield_macro_filled.npy")
    diag = dict(dataset=dataset, key=key, variant=variant, exponent_n=n,
                rule="K_j/K = ratio^n on the pixels kept by the linear rule; flagged pixels filled",
                source_dir=str(src), fill_function=fill_src, gate_max_abs_diff=dev, gate_nan_mismatch=nan_mismatch,
                kept_fraction=float(kept.mean()),
                flag_fractions={str(i): float(np.mean(flag == i)) for i in range(4)},
                linear_filled=stats(locked), power_filled=stats(pw_f),
                r_power_vs_linear_filled=K.pearson(pw_f, locked),
                r_log_power_vs_log_linear_kept=K.pearson(np.log(np.maximum(pw[kept], 1e-12)),
                                                         np.log(np.maximum(raw[kept], 1e-12))),
                raw_flag_source=str(rsrc), rerun_vs_locked=rerun,
                source_sha256={**{p: K.sha256(rsrc / p) for p in ("K_coef_raw.npy", "K_flag.npy")},
                               **{p: K.sha256(src / p) for p in ("K_coef_filled.npy", "yield_macro_filled.npy")}},
                wall_s=round(time.time() - t0, 1))
    K.write_json(dst / "diagnostics.json", diag)
    print(f"[{dataset}] K/K_macro mean: linear {diag['linear_filled']['mean']:.3f} "
          f"(std {diag['linear_filled']['std']:.3f}), power {diag['power_filled']['mean']:.3f} "
          f"(std {diag['power_filled']['std']:.3f}); r(power, linear) {diag['r_power_vs_linear_filled']:.4f}",
          flush=True)
    return diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["dataset1", "dataset2"])
    a = ap.parse_args()
    cfg = K.campaign_config()
    fu = K.followup()["power_rule"]
    for ds in a.datasets:
        n = float(cfg["datasets"][ds]["n_exp"])
        build(ds, fu["sources"][ds], n, fu["variant_dir"])


if __name__ == "__main__":
    main()
