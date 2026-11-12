# -*- coding: utf-8 -*-
"""
02_yield_hardening_generic.py - pixel-wise yield stress + hardening
coefficient identification, generic and keyed over every alpha in
new_experiment_config.DATASET_KEYS, for specimen rb_sp2.

This is the generic/keyed generalisation of 2_elastoplasticity_corrected.py
(which hardcodes a single dataset, the production alpha=100 traction/ data),
rebuilt in the engineering style of the old experiment's
03_yield_hardening_alpha.py (per-key checkpointed onset detection, Evm
history caching, diagnostics.json, resumability) - but keeping
2_elastoplasticity_corrected.py's CORRECTED onset-detection kernel, because
that correction (and its MIN_ELASTIC_FRAMES parameter) was derived
specifically for this specimen's own Evm history and is NOT present in
03_yield_hardening_alpha.py as it stands on D: (that script's
dectect_maximum_r2 has neither the min_k parameter nor the index-clamp
fix - it was written for the old experiment's dataset and left as-is here
only for methodology reference, not copied verbatim).

CORRECTION MADE HERE, 2026-09-16: the elastic-fit window.
------------------------------------------------------------------
The offset detector fits each pixel's elastic line, then declares onset at
the first departure from that line. Until this date the window it fitted
over was [3, MIN_ELASTIC_FRAMES), i.e. window frames 4-30, on the reasoning
that this specimen's elastic branch lasts about 40 frames. The load record
says it does not: frames 1-15 are the load train taking up its slack, with
the cell reading -72 N and the extensometer motionless, and the elastic
branch runs from frame 15 to frame 163, where the stress reaches the
172.3 MPa proportional limit.

Fitting over frames 4-30 therefore fitted dead time. On a100_it240 the
median fitted slope came out negative, -6.6e-07 strain/frame; 71.7 % of
pixels failed the slope>0 guard and were pinned to the floor; their yield
stress was then read at frame 30, where the specimen carries 23 MPa; and
the identified yield median was 17.3 MPa against a macroscopic 172.3.

The window is now its own parameter, cfg.ELASTIC_FIT_FIRST/LAST = 27/122
(19-138 MPa, wholly inside the elastic branch), separate from the earliest
admissible onset frame cfg.MIN_ONSET_FRAME = 123. Pixels whose elastic fit
still does not rise are returned as NaN and neighbour-filled, rather than
pinned to the floor and entered as a real yield stress. The derivation,
with the frame-to-load table, is in new_experiment_config.py.

Corrections carried over from 2_elastoplasticity_corrected.py (both
already verified against synthetic regressions there; see its own
docstring for the full bug description):

  1. dectect_maximum_r2's trailing extension loop can push the onset index
     to N (one past the last valid 0-based frame). Clipped to
     [min_k, N-1].
  2. compute_stress_yield_map's onset-to-array-index mapping: S_p is a
     1-based frame COUNT, not a 0-based index. Mapped via
     clip(S_p - 1, 0, n_frames - 1), never S_p directly.

Also carried over from 2_elastoplasticity_corrected.py, specific to this
specimen:
  - MIN_ELASTIC_FRAMES (new_experiment_config.MIN_ELASTIC_FRAMES = 30):
    without it, the R^2-based detector finds a spurious peak at k~8 and
    reports physically-impossible ~47 MPa yield stresses. It remains the
    R^2 detector's floor; the offset detector no longer uses it, see below.
  - The temporal window new_experiment_config.FRAME_START:FRAME_END
    (loading phase only - excludes the pre-load plateau and the unloading
    tail), applied to every key's sorted U/V file list before anything
    else, exactly as 2_elastoplasticity_corrected.py applies it to the
    production traction/ data.
  - Temporal Gaussian smoothing (sigma=1, time axis only) in place of
    bad-frame interpolation - oscillations in V for this specimen are
    measurement noise, not acquisition dropouts. The BAD_FRAMES dict is
    still wired through (mirrors 03_yield_hardening_alpha.py) for any
    future dataset/key that DOES need it; empty by default.

Carried over from 03_yield_hardening_alpha.py (engineering, not physics):
  - Per-key checkpointed, numba-parallel, row-blocked onset computation
    (resumable at ~ONSET_CHECKPOINT_ROWS granularity instead of an
    all-or-nothing pass).
  - Evm history cached to disk (Evm_history.npy/.json) so an interrupted
    run never recomputes the expensive gradient+deviatoric+smoothing pass.
  - Yield stress is the smoothed nominal machine stress at each pixel's
    DIC-detected onset frame.  An older branch evaluated 3*G*eps_vM_total at
    the 0.02%-offset crossing; that necessarily counted the imposed 2e-4
    plastic offset as elastic strain and biased every yield value upward.
  - diagnostics.json per key, atomic saves, _outputs_complete() skip-if-done.
  - NO thresholding anywhere (deferred to a later stage, mirroring both
    03_yield_hardening_alpha.py and thresh_config.py's separation of
    concerns). The only clamps are the [0,3] hardening-ratio sanity bound
    and the numerical yield-stress safety floor - both pre-existing
    conventions, not the deferred physical threshold.

Writes, per key, to new_experiment_config.YH_DIR/<key>/:
    Evm_history.npy / .json         cached, fully-processed strain history
    S_p_yield_onset_frame.npy       (n1,n2) yield onset frame index (1-based)
    yield_stress_vm.npy             (n1,n2) raw, MPa, NaN where undetermined
    yield_stress_vm_filled.npy      (n1,n2) NaN-filled + safety-floor clamped
    K_hardening_coef_filled.npy     (n1,n2) dimensionless ratio, NaN-filled,
                                     clip [0,3] (multiply by cfg.K_SCALE for
                                     the FEM's per-pixel K in MPa)
    diagnostics.json                counts + timing

Run:  python 02_yield_hardening_generic.py            (every key in cfg.DATASET_KEYS)
      python 02_yield_hardening_generic.py a100 a500   (subset)

Sanity check once a100 is available: running this script for key "a100"
over new_experiment_config.FRAME_START:FRAME_END should reproduce (up to
floating-point / smoothing-window numerics) the ALREADY-EXISTING
results/yield_stress_vm.npy, results/K_hardening_coef_filled.npy etc. in
rb_sp2/results/, since it is the same source displacement data run through
the same (corrected) algorithm - a good first thing to diff once this
finishes for that key.
"""
import os
import sys
import json
import re
import time
import numpy as np
from numba import njit, prange
from scipy.ndimage import gaussian_filter

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
import new_experiment_config as cfg

# NEWEXP_FORCE=1 recomputes keys that already have complete outputs, and
# ignores any existing onset checkpoint / cached Evm history.
FORCE = os.environ.get("NEWEXP_FORCE", "0") == "1"


# ============================================================
# 1. regression / onset-detection primitives
#    (2_elastoplasticity_corrected.py's CORRECTED kernel - see module
#    docstring for the two fixes vs. the uncorrected original)
# ============================================================
@njit
def estimate_coef(x, y):
    n = x.shape[0]
    m_x = np.mean(x)
    m_y = np.mean(y)
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x
    slope = SS_xy / SS_xx
    intercept = m_y - slope * m_x
    return intercept, slope


@njit
def fct_slope(u, v, Evm, S_p, len_x):
    if 0 < S_p[u, v] < Evm.shape[0]:
        y_true = Evm[:, u, v]
        k = int(S_p[u, v])
        intercept, slope = estimate_coef(len_x[:k], y_true[:k])
        y_mean_pred = np.linspace(1, Evm.shape[0], Evm.shape[0])
        y_mean_pred[:] = intercept + slope * len_x[:Evm.shape[0]]
        y_mean_true_plastic = y_true - y_mean_pred
        try:
            intercept_p, slope_p = estimate_coef(
                len_x[k:Evm.shape[0]], y_mean_true_plastic[k:Evm.shape[0]])
        except Exception:
            slope_p = np.nan
    else:
        slope_p = np.nan
    return slope_p


@njit
def fct_slope_negative(u, v, Evm, S_p, len_x):
    slope = np.nan
    if 0 < S_p[u, v] < Evm.shape[0]:
        y_true = Evm[:, u, v]
        k = int(S_p[u, v])
        intercept, slope = estimate_coef(len_x[k:], y_true[k:])
    return slope


@njit
def find_local_maxima(arr):
    local_maxima = []
    for i in range(2, len(arr) - 2):
        if (arr[i] > arr[i - 1] and arr[i - 1] > arr[i - 2] and
                arr[i] > arr[i + 1] and arr[i + 1] > arr[i + 2]):
            local_maxima.append(i)
    return local_maxima


@njit
def dectect_maximum_r2(u, v, Evm, len_x, min_k):
    """R^2-based yield-onset detector.

    min_k : minimum frame index at which yield onset can be declared -
            without it, a line through 2-3 early points always gives
            R^2~1 and gets mis-identified as the elastic->plastic
            transition. See new_experiment_config.MIN_ELASTIC_FRAMES for
            this specimen's physical justification.

    CORRECTION vs. the uncorrected original: the trailing extension loop
    (`for w in range(l, N): ... if slope<=0: l+=1`) can push l to N (one
    past the last valid 0-based frame index) if the plastic-slope check
    keeps failing all the way to the end of the series. Clipped below to
    [min_k, N-1] so callers never receive an out-of-range onset frame.
    """
    Evm_uv = Evm[:, u, v]
    b = []
    N = Evm.shape[0]
    for k in range(min_k, N):
        intercept, slope = estimate_coef(len_x[3:k], Evm_uv[3:k])
        if slope > -100:
            y_mean_pred = np.linspace(1, N, N)
            y_mean_pred[:] = intercept + slope * len_x[:N]
            try:
                corr_matrix = np.corrcoef(Evm_uv[:k], y_mean_pred[:k])
                corr = corr_matrix[0, 1]
                b.append(corr ** 2)
            except Exception:
                b.append(-10000)
        else:
            b.append(-10000)

    # b[0] -> k=min_k,  b[i] -> k=min_k+i
    # b[1:][j] -> k=min_k+j+1  ->  l = j + (min_k+1)
    try:
        l = find_local_maxima(np.array(b[1:]))[0] + min_k + 1
    except Exception:
        max_value = max(b[1:])
        l = b.index(max_value) + min_k

    for w in range(l, N):
        _, slope = estimate_coef(len_x[:w], Evm_uv[:w])
        if slope <= 0:
            l += 1
    if l > N - 1:      # CORRECTION #1
        l = N - 1
    if l < min_k:
        l = min_k
    return l


@njit
def detect_offset_onset(u, v, Evm, len_x, fit_lo, fit_hi, min_k, offset):
    """Plastic-strain offset yield-onset detector.

    The direct per-pixel analogue of the macroscopic 0.02 % offset: fit this
    pixel's elastic line over a window that is known to be elastic, then
    declare onset at the first frame whose strain departs from that line by
    more than `offset`.

    The elastic window is FIXED at [fit_lo, fit_hi] and deliberately not
    iterated. An earlier version refitted the line over [3, onset) and then
    re-detected; that diverges. Each refit swallows part of the knee, which
    inflates the fitted slope, which pushes the onset later, which swallows
    more knee. On synthetic histories it ran away to the last frame every
    time.

    `fit_lo` and `fit_hi` are 1-based window frame numbers, both inclusive,
    and come from cfg.ELASTIC_FIT_FIRST / cfg.ELASTIC_FIT_LAST, where the
    load record that fixes them is written out. They are separate from
    `min_k` because they answer a different question: `fit_lo`/`fit_hi` say
    where the specimen is known to be elastic, `min_k` says how early an
    onset may be believed. Conflating the two is what put the elastic fit in
    the load train's slack take-up and produced a 17.3 MPa yield median.

    A pixel whose elastic fit does not rise by more than `offset` across its
    own fit window is returned as NaN rather than at the floor. Pinning it to
    `min_k` would silently enter a yield stress read at an arbitrary frame;
    NaN lets the 8-neighbour fill in the caller replace it with its
    surroundings, which is what such a pixel is worth.

    `offset` is an EQUIVALENT PLASTIC strain, so 2.0e-4 means 0.02 %, matching
    cfg.SIGMA0_MACRO_MPA on the macro curve.

    Returns a 1-based frame COUNT, the same convention dectect_maximum_r2 and
    fct_slope use, clipped to [min_k, N-1], or NaN.
    """
    Evm_uv = Evm[:, u, v]
    N = Evm.shape[0]

    a = fit_lo - 1              # 1-based inclusive -> 0-based inclusive
    b = fit_hi                  # 1-based inclusive -> 0-based exclusive
    if a < 0:
        a = 0
    if b > N:
        b = N
    if b - a < 6:
        return np.nan

    intercept, slope = estimate_coef(len_x[a:b], Evm_uv[a:b])
    if not (slope * (b - a) > offset):
        # The elastic line must rise by more than the offset across the very
        # window it was fitted through. Below that there is nothing for a
        # departure to be measured against: the fit is noise, and whichever
        # sign the noise happens to take would otherwise send the pixel to
        # the floor or to the last frame, both of them entered downstream as
        # a real yield stress. Returning NaN hands it to the 8-neighbour
        # fill instead. On this specimen the bulk clears the test by a factor
        # of three (6.2e-06 x 95 = 5.9e-04 against a 2.0e-04 offset), while a
        # pixel carrying only noise fails it by two orders of magnitude.
        return np.nan

    l = N - 1
    for k in range(min_k, N):
        if Evm_uv[k] - (intercept + slope * len_x[k]) > offset:
            l = k
            break

    if l > N - 1:
        l = N - 1
    if l < min_k:
        l = min_k
    return float(l)


@njit
def _onset_one(u, v, Evm, len_x, fit_lo, fit_hi, min_k, method_code, offset):
    """Dispatch between the two detectors inside the compiled kernel."""
    if method_code == 1:
        return detect_offset_onset(u, v, Evm, len_x, fit_lo, fit_hi,
                                   min_k, offset)
    return float(dectect_maximum_r2(u, v, Evm, len_x, min_k))


@njit(parallel=True)
def _onset_rows(Evm, len_x, fit_lo, fit_hi, min_k, r0, r1, out,
                method_code, offset):
    """Yield-onset detection for rows [r0, r1), written into `out`."""
    n2 = Evm.shape[2]
    for i in prange(r0, r1):
        for j in range(n2):
            out[i, j] = _onset_one(i, j, Evm, len_x, fit_lo, fit_hi,
                                   min_k, method_code, offset)


def _onset_rows_python(Evm, len_x, fit_lo, fit_hi, min_k, r0, r1, out,
                       method_code, offset):
    """Fallback with identical semantics, used only if the compiled kernel
    fails to compile on this numba version. Slow but correct, still
    checkpointed."""
    n2 = Evm.shape[2]
    for i in range(r0, r1):
        for j in range(n2):
            out[i, j] = _onset_one(i, j, Evm, len_x, fit_lo, fit_hi,
                                   min_k, method_code, offset)


def _save_atomic(path, arr):
    """Write to a temp file then replace, so a kill mid-write never leaves a
    half-written .npy that looks valid."""
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def compute_yield_onset_map(key, Evm, len_x, fit_lo, fit_hi, min_k,
                            block_rows=None):
    """S_p for every pixel, checkpointed every `block_rows` rows. On
    restart, an existing checkpoint is loaded and computation resumes from
    the first unfinished row."""
    if block_rows is None:
        block_rows = cfg.ONSET_CHECKPOINT_ROWS
    n1, n2 = Evm.shape[1], Evm.shape[2]
    d = cfg.yh_dir(key)
    ck_path = os.path.join(d, "_checkpoint_S_p.npy")
    ck_meta = os.path.join(d, "_checkpoint_S_p.json")

    S_p = np.full((n1, n2), np.nan)
    row0 = 0
    if FORCE and os.path.isfile(ck_path):
        print(f"[02] {key}: NEWEXP_FORCE=1 - ignoring existing onset checkpoint.")
    elif os.path.isfile(ck_path) and os.path.isfile(ck_meta):
        try:
            with open(ck_meta) as f:
                meta = json.load(f)
            if (meta.get("shape") == [n1, n2] and meta.get("n_frames") == Evm.shape[0]
                    and meta.get("key") == key and meta.get("min_k") == min_k
                    and meta.get("elastic_fit_first") == fit_lo
                    and meta.get("elastic_fit_last") == fit_hi
                    and meta.get("onset_method") == cfg.ONSET_METHOD
                    and meta.get("onset_offset") == cfg.ONSET_OFFSET):
                S_p = np.load(ck_path)
                row0 = int(meta.get("next_row", 0))
                print(f"[02] {key}: RESUMING yield onset from row {row0}/{n1} "
                      f"({100*row0/n1:.1f}% already done, checkpoint found)")
            else:
                print(f"[02] {key}: checkpoint found but shape/frames/key/min_k/"
                      f"elastic-fit-window differ from the current run - "
                      f"ignoring it and starting over.")
        except Exception as e:
            print(f"[02] {key}: unreadable checkpoint ({e}) - starting over.")

    if row0 >= n1:
        print(f"[02] {key}: yield onset already complete in checkpoint.")
        return S_p

    mcode = cfg.ONSET_METHOD_CODE
    offset = cfg.ONSET_OFFSET
    print(f"[02] {key}: onset detector = {cfg.ONSET_METHOD!r}"
          + (f", offset = {offset:.3e} equivalent plastic strain, "
             f"elastic fit over window frames {fit_lo}-{fit_hi}, "
             f"onset no earlier than frame {min_k}"
             if mcode == 1 else f", onset no earlier than frame {min_k}"))

    kernel = _onset_rows
    if row0 == 0:
        try:
            _onset_rows(Evm, len_x, fit_lo, fit_hi, min_k, 0, 1,
                        np.full((n1, n2), np.nan), mcode, offset)
        except Exception as e:
            print(f"[02] {key}: numba-parallel onset kernel unavailable ({e}); "
                  f"falling back to the slow Python loop (still checkpointed).")
            kernel = _onset_rows_python

    t_blk = time.time()
    for r0 in range(row0, n1, block_rows):
        r1 = min(r0 + block_rows, n1)
        kernel(Evm, len_x, fit_lo, fit_hi, min_k, r0, r1, S_p, mcode, offset)
        _save_atomic(ck_path, S_p)
        with open(ck_meta, "w") as f:
            json.dump(dict(key=key, next_row=int(r1), shape=[n1, n2],
                           n_frames=int(Evm.shape[0]), min_k=min_k,
                           elastic_fit_first=fit_lo, elastic_fit_last=fit_hi,
                           onset_method=cfg.ONSET_METHOD,
                           onset_offset=cfg.ONSET_OFFSET), f)
        done = r1 - row0
        rate = done / max(time.time() - t_blk, 1e-9)
        eta = (n1 - r1) / max(rate, 1e-9)
        print(f"[02] {key}: yield onset rows {r1}/{n1} "
              f"({100*r1/n1:.1f}%)  {rate:.1f} rows/s  ETA {eta/60:.1f} min",
              flush=True)
    return S_p


@njit(parallel=True)
def compute_slope_plastic(Evm, S_p, len_x):
    n1, n2 = Evm.shape[1], Evm.shape[2]
    slop_plastic = np.full((n1, n2), np.nan)
    for i in prange(n1):
        for j in range(n2):
            slop_plastic[i, j] = fct_slope(i, j, Evm, S_p, len_x)
    return slop_plastic


@njit(parallel=True)
def update_slope_plastic(Evm, slope_plastic, S_p, len_x):
    n1, n2 = Evm.shape[1], Evm.shape[2]
    for i in prange(n1):
        for j in range(n2):
            if slope_plastic[i, j] < 0:
                slope_plastic[i, j] = fct_slope_negative(i, j, Evm, S_p, len_x)
    return slope_plastic


# ============================================================
# 2. bad-frame repair (infrastructure only - empty by default for this
#    specimen; temporal Gaussian smoothing is what actually handles noise)
# ============================================================
def repair_bad_frames(Evm, bad_frames_1based):
    """Linearly interpolate the given 1-based frame indices from the nearest
    GOOD neighbours (skipping over other bad frames)."""
    if not bad_frames_1based:
        return Evm
    Evm = Evm.copy()
    N = Evm.shape[0]
    bad0 = sorted(set(i - 1 for i in bad_frames_1based))
    for i in bad0:
        lo = i - 1
        while lo in bad0 and lo >= 0:
            lo -= 1
        hi = i + 1
        while hi in bad0 and hi < N:
            hi += 1
        if lo < 0 and hi >= N:
            continue
        if lo < 0:
            Evm[i] = Evm[hi]
        elif hi >= N:
            Evm[i] = Evm[lo]
        else:
            w = (i - lo) / (hi - lo)
            Evm[i] = (1 - w) * Evm[lo] + w * Evm[hi]
    return Evm


# ============================================================
# 3. yield stress map from the synchronized machine record
# ============================================================
def _time_from_image_name(name):
    """Acquisition time encoded by ``<image>_<time>.tiff``."""
    m = re.match(r"^\d+_([0-9]+(?:\.[0-9]+)?)\.tiff?$",
                 os.path.basename(name), flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"cannot read acquisition time from image name {name!r}")
    return float(m.group(1))


def frame_times_for_dic_files(files_U):
    """Return the acquisition time for each windowed U file, in its order.

    On the HPC the authoritative manifest maps full-acquisition frame numbers
    to the subset of uploaded images.  Locally, where all 2551 images exist,
    the same mapping is recovered from their numeric filename ordering.
    """
    frame_numbers = cfg.frame_numbers_of(files_U)
    manifest = os.environ.get(
        "NEWEXP_MANIFEST", os.path.join(CODE_DIR, "frames_manifest.txt"))
    names_by_frame = {}
    if os.path.isfile(manifest):
        with open(manifest, encoding="utf-8") as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                k, name = line.rstrip("\n").split("\t", 1)
                names_by_frame[int(k)] = name

    if all(k in names_by_frame for k in frame_numbers):
        names = [names_by_frame[k] for k in frame_numbers]
    else:
        image_names = [n for n in os.listdir(cfg.IMG_DIR)
                       if re.match(r"^\d+_[0-9.]+\.tiff?$", n,
                                   flags=re.IGNORECASE)]
        image_names.sort(key=lambda n: int(n.split("_", 1)[0]))
        if image_names and max(frame_numbers) <= len(image_names):
            names = [image_names[k - 1] for k in frame_numbers]
        elif len(image_names) == len(frame_numbers):
            # Window-only upload without a manifest: safe only when both
            # ordered lists have the same length.
            names = image_names
        else:
            missing = [k for k in frame_numbers if k not in names_by_frame]
            raise RuntimeError(
                f"cannot map DIC frames to image times: manifest {manifest!r} "
                f"misses {missing[:10]} and {cfg.IMG_DIR!r} contains "
                f"{len(image_names)} usable images")

    times = np.asarray([_time_from_image_name(n) for n in names], dtype=float)
    if len(times) != len(files_U) or np.any(np.diff(times) <= 0):
        raise RuntimeError("DIC frame times are not a unique increasing sequence")
    return times


def macro_stress_for_dic_files(files_U):
    """Smoothed nominal machine stress interpolated onto every DIC frame."""
    if not os.path.isfile(cfg.MACRO_RECORD):
        raise FileNotFoundError(
            f"machine record not found: {cfg.MACRO_RECORD}\n"
            "Upload data_labjack.csv before running identification.")
    raw = np.genfromtxt(cfg.MACRO_RECORD, delimiter=";", skip_header=1)
    if raw.ndim != 2 or raw.shape[1] < 3:
        raise ValueError(f"unexpected machine-record shape {raw.shape}")
    t, force = raw[:, 0], raw[:, 2]
    valid = np.isfinite(t) & np.isfinite(force)
    t, force = t[valid], force[valid]
    order = np.argsort(t)
    t, force = t[order], force[order]
    force = gaussian_filter(force, cfg.MACRO_SMOOTH_SIGMA)
    frame_times = frame_times_for_dic_files(files_U)
    if frame_times[0] < t[0] or frame_times[-1] > t[-1]:
        raise RuntimeError(
            f"DIC times [{frame_times[0]:.3f}, {frame_times[-1]:.3f}] lie "
            f"outside machine times [{t[0]:.3f}, {t[-1]:.3f}]")
    stress = np.interp(frame_times, t, force) / cfg.MACRO_AREA_MM2
    return stress, frame_times


def compute_stress_yield_map(S_p, macro_stress):
    """Assign nominal stress at each pixel's DIC-detected onset frame.

    The result is an effective local-onset descriptor, not a directly
    measured local Cauchy stress.  S_p is a 1-based frame count, hence `-1`.
    """
    stress_yield = np.full(S_p.shape, np.nan, dtype=float)
    n_frames = len(macro_stress)
    mask = (S_p >= 1) & (S_p <= n_frames) & np.isfinite(S_p)
    indices = np.clip(S_p[mask].astype(int) - 1, 0, n_frames - 1)
    stress_yield[mask] = macro_stress[indices]
    return stress_yield


# ============================================================
# 4. per-dataset pipeline
# ============================================================
# The identification settings that, if changed, make an existing result stale.
# diagnostics.json records all of them, so a finished key can be checked
# against the config it was produced under instead of being trusted on the
# strength of its files existing. Without this, correcting the elastic-fit
# window and re-running would have silently reported the old answer, which is
# how a wrong number survives a fix.
_IDENT_SETTINGS = ("onset_method", "onset_offset", "onset_min_frame",
                   "elastic_fit_first", "elastic_fit_last",
                   "frame_start", "frame_end", "slop_ref",
                   "yield_stress_method", "macro_area_mm2",
                   "macro_smooth_sigma", "fem_E_mod_mpa")


def _current_ident_settings():
    return dict(onset_method=cfg.ONSET_METHOD, onset_offset=cfg.ONSET_OFFSET,
                onset_min_frame=int(cfg.ONSET_MIN_FRAME),
                elastic_fit_first=int(cfg.ELASTIC_FIT_FIRST),
                elastic_fit_last=int(cfg.ELASTIC_FIT_LAST),
                frame_start=cfg.FRAME_START, frame_end=cfg.FRAME_END,
                slop_ref=cfg.SLOP_REF,
                yield_stress_method=cfg.YIELD_STRESS_METHOD,
                macro_area_mm2=cfg.MACRO_AREA_MM2,
                macro_smooth_sigma=cfg.MACRO_SMOOTH_SIGMA,
                fem_E_mod_mpa=cfg.E_FEM)


def _outputs_complete(key):
    d = cfg.yh_dir(key)
    if not all(os.path.isfile(os.path.join(d, f)) for f in (
            "S_p_yield_onset_frame.npy", "yield_stress_vm.npy",
            "yield_stress_vm_filled.npy", "K_hardening_coef_filled.npy",
            "diagnostics.json")):
        return False
    try:
        with open(os.path.join(d, "diagnostics.json")) as f:
            old = json.load(f)
    except Exception:
        return False
    now = _current_ident_settings()
    stale = [k for k in _IDENT_SETTINGS if old.get(k) != now[k]]
    if stale:
        print(f"[02] {key}: existing result was produced under different "
              f"identification settings - recomputing.")
        for k in stale:
            print(f"        {k}: {old.get(k)!r} -> {now[k]!r}")
        return False
    return True


def process_dataset(key):
    t0 = time.time()
    print("=" * 78)
    print(f"[02] {key}")

    if _outputs_complete(key) and not FORCE:
        d = cfg.yh_dir(key)
        with open(os.path.join(d, "diagnostics.json")) as f:
            diag = json.load(f)
        print(f"[02] {key}: already complete -> {d}  (set NEWEXP_FORCE=1 to redo)")
        return diag

    u_dir, v_dir = cfg.uv_dirs(key)
    files_U = cfg.get_sorted_npy_files(u_dir, "U")
    files_V = cfg.get_sorted_npy_files(v_dir, "V")
    if len(files_U) != len(files_V):
        raise RuntimeError(f"[{key}] {len(files_U)} U frames vs "
                           f"{len(files_V)} V frames - mismatch")
    if not files_U:
        raise FileNotFoundError(
            f"[{key}] no U_*.npy files in {u_dir} - run "
            f"01_run_dic_generic.py for this key first.")

    # Loading-phase window only (excludes pre-load plateau and unloading
    # tail) - same convention 2_elastoplasticity_corrected.py uses on the
    # production data, applied here to every key's own frame list.
    # by frame NUMBER, not position, so a DIC run restricted to the loading
    # window is read correctly; identical to the old positional slice when
    # every frame of the acquisition is present. See new_experiment_config.
    files_U = cfg.window_files(files_U)
    files_V = cfg.window_files(files_V)
    N = len(files_U)
    print(f"[02] {key}: {N} frames (window [{cfg.FRAME_START}:{cfg.FRAME_END}]) "
          f"from {u_dir}")

    U_all = [np.load(os.path.join(u_dir, f), mmap_mode="r") for f in files_U]
    V_all = [np.load(os.path.join(v_dir, f), mmap_mode="r") for f in files_V]

    d = cfg.yh_dir(key)
    bad = cfg.BAD_FRAMES.get(key, [])
    len_x = np.linspace(1, N, N)
    min_k = cfg.ONSET_MIN_FRAME
    fit_lo, fit_hi = cfg.ELASTIC_FIT_FIRST, cfg.ELASTIC_FIT_LAST

    # ---- Evm(frame, i, j) - CACHED to disk ----
    evm_path = os.path.join(d, "Evm_history.npy")
    evm_meta = os.path.join(d, "Evm_history.json")
    Evm = None
    if os.path.isfile(evm_path) and os.path.isfile(evm_meta) and not FORCE:
        try:
            with open(evm_meta) as f:
                meta = json.load(f)
            if (meta.get("n_frames") == N and meta.get("bad_frames") == list(bad)
                    and meta.get("key") == key
                    and meta.get("frame_start") == cfg.FRAME_START
                    and meta.get("frame_end") == cfg.FRAME_END):
                Evm = np.load(evm_path)
                print(f"[02] {key}: loaded cached Evm history {Evm.shape} "
                      f"({Evm.nbytes/1e9:.2f} GB) - skipping recomputation")
            else:
                print(f"[02] {key}: cached Evm does not match current settings "
                      f"(frames/window/bad_frames/key) - recomputing.")
        except Exception as e:
            print(f"[02] {key}: unreadable Evm cache ({e}) - recomputing.")

    if Evm is None:
        print(f"[02] {key}: computing Evm history ({N} frames)...")
        Evm_all_dev = []
        for m in range(N):
            if m % 200 == 0:
                print(f"  frame {m}/{N}")
            Exy, Exx = np.gradient(U_all[m])
            Eyy, Eyx = np.gradient(V_all[m])
            Eshear = 0.5 * (Exy + Eyx)
            Exx_dev, Eyy_dev, Ezz_dev, Exy_dev = cfg.deviatoric_strain_plane_stress(
                Exx, Eyy, Eshear, cfg.NU)
            Evm_all_dev.append(cfg.evm_from_edev(Exx_dev, Eyy_dev, Ezz_dev, Exy_dev))
        Evm = np.array(Evm_all_dev)

        if bad:
            print(f"[02] {key}: repairing bad frames {bad} "
                  f"(linear interp from nearest good neighbours)")
        Evm = repair_bad_frames(Evm, bad)
        # Temporal Gaussian smoothing only (sigma=1, time axis) - this
        # specimen's V oscillations are measurement noise, not acquisition
        # dropouts (2_elastoplasticity_corrected.py's finding).
        Evm = gaussian_filter(Evm, sigma=[1, 0, 0])
        Evm = Evm.astype(np.float32)

        if cfg.SAVE_EVM_CACHE:
            _save_atomic(evm_path, Evm)
            with open(evm_meta, "w") as f:
                json.dump(dict(key=key, n_frames=int(N), bad_frames=list(bad),
                               shape=[int(s) for s in Evm.shape],
                               dtype="float32", gaussian_sigma=[1, 0, 0],
                               frame_start=cfg.FRAME_START,
                               frame_end=cfg.FRAME_END), f)
            print(f"[02] {key}: Evm history saved {Evm.shape} "
                  f"({Evm.nbytes/1e9:.2f} GB) -> {evm_path}")
        else:
            print(f"[02] {key}: NEWEXP_SAVE_EVM=0 - Evm not cached "
                  f"(would have been {Evm.nbytes/1e9:.2f} GB)")

    # ---- yield onset frame S_p (checkpointed - the expensive step) ----
    S_p = compute_yield_onset_map(key, Evm, len_x, fit_lo, fit_hi, min_k)
    _save_atomic(os.path.join(d, "S_p_yield_onset_frame.npy"), S_p)
    n_Sp_nan = int(np.isnan(S_p).sum())
    n_Sp_valid = int(((S_p >= 2) & (S_p <= N) & ~np.isnan(S_p)).sum())
    print(f"[02] {key}: S_p valid onset in {n_Sp_valid}/{S_p.size} px "
          f"({100*n_Sp_valid/S_p.size:.2f}%), NaN in {n_Sp_nan} px")
    # Onset distribution. A large share sitting exactly on the floor means the
    # elastic fit is failing, not that the material yields at the floor; that
    # is the signature the 2026-09-16 elastic-window correction was made for.
    _q = np.nanpercentile(S_p, [1, 5, 25, 50, 75, 95, 99])
    n_Sp_floor = int(np.nansum(S_p <= min_k))
    n_Sp_last = int(np.nansum(S_p >= N - 1))
    print(f"[02] {key}: onset frame percentiles "
          f"(1/5/25/50/75/95/99) = " + " ".join(f"{v:.0f}" for v in _q))
    print(f"[02] {key}: onset pinned at the floor ({min_k}) in "
          f"{100*n_Sp_floor/S_p.size:.1f}% of px, never detected (last frame) "
          f"in {100*n_Sp_last/S_p.size:.1f}%")

    # ---- yield stress: synchronized load-cell stress at local onset ----
    macro_stress, frame_times = macro_stress_for_dic_files(files_U)
    print(f"[02] {key}: yield stress method = {cfg.YIELD_STRESS_METHOD}; "
          f"machine stress spans {macro_stress[0]:.1f} to "
          f"{macro_stress[-1]:.1f} MPa over {frame_times[0]:.3f} to "
          f"{frame_times[-1]:.3f} s")
    stress_yield = compute_stress_yield_map(S_p, macro_stress)
    n_yield_nan = int(np.isnan(stress_yield).sum())
    _save_atomic(os.path.join(d, "yield_stress_vm.npy"), stress_yield)

    # ---- hardening slope / coefficient (no thresholding) ----
    slop_plastic = compute_slope_plastic(Evm, S_p, len_x)
    slop_plastic = update_slope_plastic(Evm, slop_plastic, S_p, len_x)
    hardening_coef = cfg.SLOP_REF / slop_plastic

    hardening_bounded = hardening_coef.copy()
    n_K_gt3 = int(np.sum(hardening_bounded > 3))
    n_K_lt0 = int(np.sum(hardening_bounded < 0))
    hardening_bounded[hardening_bounded > 3] = np.nan
    hardening_bounded[hardening_bounded < 0] = np.nan
    n_K_nan_before_fill = int(np.isnan(hardening_bounded).sum())
    hardening_filled = cfg.replace_nan_with_neighbors(hardening_bounded)

    # ---- yield: NaN-fill + numerical safety floor (NOT the deferred thresh) ----
    yield_filled = cfg.replace_nan_with_neighbors(stress_yield.copy())
    floor = cfg.YIELD_SAFETY_FLOOR_MPA
    n_floor_clamped = 0
    if floor > 0:
        below = yield_filled < floor
        n_floor_clamped = int(below.sum())
        yield_filled[below] = floor
    remaining_nan = int(np.isnan(yield_filled).sum())
    if remaining_nan:
        print(f"[02] {key}: WARNING {remaining_nan} px still NaN after "
              f"8-neighbour fill - filling with the dataset median as a "
              f"last resort.")
        yield_filled[np.isnan(yield_filled)] = np.nanmedian(yield_filled)

    wall = time.time() - t0
    _save_atomic(os.path.join(d, "yield_stress_vm_filled.npy"), yield_filled)
    _save_atomic(os.path.join(d, "K_hardening_coef_filled.npy"), hardening_filled)

    # _current_ident_settings() supplies frame_start/frame_end, the elastic-fit
    # window, the onset floor, the detector and slop_ref - the settings
    # _outputs_complete() compares a finished key against. Nothing below may
    # repeat one of those keys.
    diag = dict(
        key=key, n_frames=N,
        min_elastic_frames=min_k,
        wall_s=wall,
        n_pixels=int(S_p.size),
        n_Sp_nan=n_Sp_nan, n_Sp_valid=n_Sp_valid,
        n_Sp_at_floor=n_Sp_floor, n_Sp_at_last_frame=n_Sp_last,
        onset_frame_percentiles=dict(zip(
            ["p1", "p5", "p25", "p50", "p75", "p95", "p99"],
            [float(v) for v in _q])),
        median_yield_vm_mpa=float(np.nanmedian(yield_filled)),
        n_yield_nan_before_fill=n_yield_nan,
        n_yield_floor_clamped=n_floor_clamped,
        yield_safety_floor_mpa=floor,
        n_K_gt3_clipped=n_K_gt3, n_K_lt0_clipped=n_K_lt0,
        n_K_nan_before_fill=n_K_nan_before_fill,
        bad_frames_repaired=bad,
        sigma0_macro_mpa=getattr(cfg, "SIGMA0_MACRO_MPA", None),
        K_scale_mpa=cfg.K_SCALE,
        n_exp=cfg.N_EXP,
        mean_yield_vm_mpa=float(np.nanmean(yield_filled)),
        mean_K_coef=float(np.nanmean(hardening_filled)),
        mean_K_mpa=float(np.nanmean(hardening_filled) * cfg.K_SCALE),
        macro_record=os.path.basename(cfg.MACRO_RECORD),
        macro_stress_first_mpa=float(macro_stress[0]),
        macro_stress_last_mpa=float(macro_stress[-1]),
        **_current_ident_settings(),
    )
    with open(os.path.join(d, "diagnostics.json"), "w") as f:
        json.dump(diag, f, indent=2)

    for _f in ("_checkpoint_S_p.npy", "_checkpoint_S_p.json"):
        _p = os.path.join(d, _f)
        if os.path.isfile(_p):
            os.remove(_p)

    print(f"[02] {key}: yield NaN {n_yield_nan}/{S_p.size} before fill, "
          f"{n_floor_clamped} floor-clamped (<{floor} MPa) | "
          f"K: {n_K_gt3} clipped >3, {n_K_lt0} clipped <0")
    print(f"[02] {key}: mean yield {diag['mean_yield_vm_mpa']:.2f} MPa, "
          f"median {diag['median_yield_vm_mpa']:.2f} MPa "
          f"(macroscopic proportional limit {cfg.SIGMA0_MACRO_MPA:.1f} MPa) | "
          f"mean K {diag['mean_K_coef']:.4f} ({diag['mean_K_mpa']:.1f} MPa "
          f"after x{cfg.K_SCALE}) | {wall:.1f}s -> {d}")
    return diag


if __name__ == "__main__":
    requested = sys.argv[1:] or cfg.DATASET_KEYS
    bad_keys = [k for k in requested if k not in cfg.DATASET_KEYS]
    if bad_keys:
        raise SystemExit(f"unknown dataset key(s) {bad_keys}; "
                         f"available: {cfg.DATASET_KEYS}")
    t_all = time.time()
    diags = [process_dataset(k) for k in requested]
    print("=" * 78)
    for d in diags:
        print(f"  {d['key']:>10s}: mean yield {d['mean_yield_vm_mpa']:7.2f} MPa  "
              f"mean K {d['mean_K_mpa']:7.1f} MPa  {d['wall_s']:6.1f}s")
    print(f"TOTAL {time.time()-t_all:.1f}s")
    print("Next: compare diagnostics and maps across the 240-iteration alpha "
          "sweep, then run FEM only after the identification trends pass.")
