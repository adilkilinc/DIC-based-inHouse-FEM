"""Method-B identification steps shared by d1_identify.py and d2_identify.py.

Variant spec:  name:offset:<offset>:<hi_shift>:<window>:<abscissa>
  offset    equivalent plastic strain offset (2e-4 = 0.02 %)
  hi_shift  frames added to the end of the elastic window
  window    load  = frames carrying 0.11-0.80 sigma0 of machine stress
            auto  = largest window from frame 4 over which the field-mean
                    history stays within 0.1 x offset of its own line
                    (Dataset 1 only, sensitivity)
  abscissa  stress = elastic line fitted against the machine stress
            frame  = elastic line fitted against the frame number
Yield = machine stress at frame index S_p - 1.  K = SLOP_REF / plastic slope
(02_identify rule, frame abscissa), bounds [0, 3], 8-neighbour fill.
"""
import json
import os
import time
import numpy as np
from onset_kernels import offset_onset_block, offset_onset_stress_block

# noise guard: the elastic line must rise by more than the PRODUCTION offset
# across its window, whatever offset a sensitivity variant tests
RISE_MIN = 2e-4


def parse_variant(spec):
    f = spec.split(":")
    if len(f) != 6 or f[1] != "offset" or f[4] not in ("load", "auto") or f[5] not in ("stress", "frame"):
        raise SystemExit(f"bad variant spec {spec!r}; expected name:offset:<off>:<shift>:load|auto:stress|frame")
    return dict(name=f[0], offset=float(f[2]), hi_shift=int(f[3]), window=f[4], abscissa=f[5])


def auto_window(emean, offset, lo=4, tol_frac=0.1):
    x = np.arange(1, emean.size + 1, dtype=float)
    best = None
    for hi in range(lo + 4, emean.size):
        c = np.polyfit(x[lo - 1:hi], emean[lo - 1:hi], 1)
        if np.max(np.abs(emean[lo - 1:hi] - np.polyval(c, x[lo - 1:hi]))) <= tol_frac * offset:
            best = hi
        else:
            break
    if best is None:
        raise SystemExit("no linear initial window in the mean history")
    return lo, best


def onset_map(Evm, v, stress, lo, hi, block_rows=100):
    """S_p (full-series count, NaN where the elastic line does not rise) and
    the elastic-line strain at the descriptor frame."""
    N, nr, nc = Evm.shape
    sel = np.arange(N, dtype=np.int64)
    x = np.linspace(1, N, N)
    S_p = np.full((nr, nc), np.nan)
    e_el = np.full((nr, nc), np.nan)
    min_k = hi + 1
    for r0 in range(0, nr, block_rows):
        r1 = min(r0 + block_rows, nr)
        blk = np.ascontiguousarray(Evm[:, r0:r1, :])
        if v["abscissa"] == "stress":
            offset_onset_stress_block(blk, np.asarray(stress, np.float64), sel, int(lo), int(hi),
                                      int(min_k), float(v["offset"]), S_p[r0:r1], e_el[r0:r1], RISE_MIN)
        else:
            offset_onset_block(blk, x, sel, float(lo), float(hi), int(min_k),
                               float(v["offset"]), S_p[r0:r1], e_el[r0:r1], RISE_MIN)
    return S_p, e_el


def kernel_identity(Evm, lo, hi, offset, rows=64):
    """The stress kernel fed with s = frame number must reproduce the frame
    kernel (the two guards differ by one frame of rise, so pixels at the
    guard may switch between NaN and a value; those are counted apart)."""
    N = Evm.shape[0]
    r1 = min(rows, Evm.shape[1])
    blk = np.ascontiguousarray(Evm[:, :r1, :])
    sel = np.arange(N, dtype=np.int64)
    x = np.linspace(1, N, N)
    a = np.full(blk.shape[1:], np.nan); ea = np.full_like(a, np.nan)
    b = np.full_like(a, np.nan); eb = np.full_like(a, np.nan)
    offset_onset_block(blk, x, sel, float(lo), float(hi), hi + 1, offset, a, ea, RISE_MIN)
    offset_onset_stress_block(blk, x, sel, int(lo), int(hi), hi + 1, offset, b, eb, RISE_MIN)
    both = np.isfinite(a) & np.isfinite(b)
    return dict(pixels=int(a.size), both_finite=int(both.sum()),
                identical_onset=float(np.mean(a[both] == b[both])) if both.any() else float("nan"),
                max_e_el_diff=float(np.max(np.abs(ea[both] - eb[both]))) if both.any() else float("nan"),
                nan_mismatch=int(np.sum(np.isfinite(a) != np.isfinite(b))))


def yield_from_onset(S_p, stress):
    N = len(stress)
    y = np.full(S_p.shape, np.nan)
    ok = np.isfinite(S_p) & (S_p >= 1) & (S_p <= N)
    y[ok] = np.asarray(stress)[np.clip(S_p[ok].astype(int) - 1, 0, N - 1)]
    return y


def fill_yield(y, fill, floor):
    yf = fill(y.copy())
    if floor > 0:
        yf[yf < floor] = floor
    if np.isnan(yf).any():
        yf[np.isnan(yf)] = np.nanmedian(yf)
    return yf


def hardening(Evm, S_p, slop_ref, m, fill):
    """02 rule: plastic slope against the frame number, K = SLOP_REF / slope,
    bounds [0, 3] replaced by the neighbour fill.  Flag: 0 kept, 1 above 3,
    2 below 0, 3 undefined."""
    N = Evm.shape[0]
    x = np.linspace(1, N, N)
    S_k = np.where(np.isfinite(S_p), S_p, 0.0)
    slope = m.compute_slope_plastic(Evm, S_k, x)
    slope = m.update_slope_plastic(Evm, slope, S_k, x)
    with np.errstate(divide="ignore", invalid="ignore"):
        K = slop_ref / slope
    flag = np.zeros(K.shape, dtype=np.uint8)
    flag[~np.isfinite(K)] = 3
    flag[np.isfinite(K) & (K > 3)] = 1
    flag[np.isfinite(K) & (K < 0)] = 2
    raw = K.copy()
    K[(K > 3) | (K < 0)] = np.nan
    return fill(K), raw, flag


def stats(a):
    a = np.asarray(a, float)
    f = a[np.isfinite(a)]
    if f.size == 0:
        return {}
    q = np.percentile(f, [1, 5, 25, 50, 75, 95, 99])
    return dict(mean=float(f.mean()), std=float(f.std()), p1=float(q[0]), p5=float(q[1]),
                p25=float(q[2]), p50=float(q[3]), p75=float(q[4]), p95=float(q[5]),
                p99=float(q[6]), n=int(f.size))


def save32(path, arr):
    tmp = path + ".tmp.npy"
    np.save(tmp, np.asarray(arr, dtype=np.float32))
    os.replace(tmp, path)


def run_variants(Evm, variants, stress, windows, slop_ref, k_scale, out_root, m, fill, floor, extra=None):
    """Identify every variant; returns the per-variant diagnostics."""
    N = Evm.shape[0]
    rows = []
    for v in variants:
        t1 = time.time()
        lo, hi = windows[v["window"]]
        hi = hi + v["hi_shift"]
        vdir = os.path.join(out_root, v["name"])
        os.makedirs(vdir, exist_ok=True)
        S_p, e_el = onset_map(Evm, v, stress, lo, hi)
        y = yield_from_onset(S_p, stress)
        yf = fill_yield(y, fill, floor)
        Kf, Kraw, flag = hardening(Evm, S_p, slop_ref, m, fill)
        save32(os.path.join(vdir, "S_p.npy"), S_p)
        save32(os.path.join(vdir, "e_el.npy"), e_el)
        save32(os.path.join(vdir, "yield_macro.npy"), y)
        save32(os.path.join(vdir, "yield_macro_filled.npy"), yf)
        save32(os.path.join(vdir, "K_coef_raw.npy"), Kraw)
        save32(os.path.join(vdir, "K_coef_filled.npy"), Kf)
        np.save(os.path.join(vdir, "K_flag.npy"), flag)
        diag = dict(variant=v, fit_lo=int(lo), fit_hi=int(hi), min_k=int(hi + 1),
                    stress_window_mpa=[float(stress[lo - 1]), float(stress[hi - 1])],
                    n_frames=int(N), shape=list(S_p.shape),
                    onset=stats(S_p), onset_nan_frac=float(np.isnan(S_p).mean()),
                    onset_last_frac=float(np.nanmean(S_p >= N - 1)),
                    n_distinct_onsets=int(np.unique(S_p[np.isfinite(S_p)]).size),
                    yield_raw=stats(y), yield_filled=stats(yf),
                    K_coef=stats(Kf), K_mpa_mean=float(np.nanmean(Kf) * k_scale),
                    K_flag_fractions={str(i): float(np.mean(flag == i)) for i in range(4)},
                    slop_ref=float(slop_ref), k_scale_mpa=float(k_scale),
                    rise_min=RISE_MIN, wall_s=round(time.time() - t1, 1))
        if diag["onset_nan_frac"] > 0.2:
            print(f"[id] WARNING {v['name']}: {100*diag['onset_nan_frac']:.1f} % of pixels fail the "
                  f"elastic-rise guard", flush=True)
        if extra:
            diag.update(extra)
        with open(os.path.join(vdir, "diagnostics.json"), "w") as fh:
            json.dump(diag, fh, indent=2)
        print(f"[id] {v['name']}: window {lo}-{hi} ({v['abscissa']}), onset p5/50/95 "
              f"{np.nanpercentile(S_p, [5, 50, 95])}, NaN {100*diag['onset_nan_frac']:.2f} %, "
              f"censored {100*diag['onset_last_frac']:.2f} %, yield p50 {np.median(yf):.1f} MPa, "
              f"K mean {np.nanmean(Kf):.3f} ({diag['K_mpa_mean']:.1f} MPa) [{diag['wall_s']} s]", flush=True)
        rows.append(diag)
    return rows
