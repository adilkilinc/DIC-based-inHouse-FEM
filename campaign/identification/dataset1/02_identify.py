# -*- coding: utf-8 -*-
"""
03_yield_hardening_alpha.py - pixel-wise yield stress + hardening coefficient
identification, generalized from el_K_recalculate_to_Adil_modified_2.py
(the NON-threshold script) to run over every dataset key in
alpha_config.DATASET_KEYS (REF + every alpha), with NO thresholding anywhere.

What changed vs the original el_K script, and why:

  - Paths, frame count and the S_p validity bound (was hardcoded 42, tied to
    the a100/a1k datasets' 42 frames) are now derived per-dataset, so this
    runs correctly on REF (40 frames) and any alpha (its own frame count).

  - Bad-frame repair is no longer the hardcoded `Evm[31] = ...` lines (tied to
    one specific acquisition). It now reads alpha_config.BAD_FRAMES[key] - a
    list you fill in AFTER looking at 02's mean-U/V plots - and linearly
    interpolates across those frames from the nearest good neighbours. Empty
    by default; does nothing until you've inspected the plots.

  - vm_Jeff no longer computes both a Hooke-strain and a deviatoric-strain von
    Mises stress. They are algebraically identical under the plane-stress
    assumption both branches share (verified numerically, max relative
    difference ~1e-15) - the Hooke branch was a redundant duplicate. Kept:
    sigma_vm = 3*G*eps_vM, the direct elastic conjugate of the eps_vM already
    used for onset detection.

  - NO thresholding block (that is deferred to a later study - see chat
    notes). The only clamps applied are:
      (a) the hardening ratio sanity bound [0,3] - same as the original
          script, always applied, not part of the deferred thresholding.
      (b) a numerical safety floor on yield stress
          (alpha_config.YIELD_SAFETY_FLOOR_MPA) - guards the FEM's
          return-mapping from dividing by a non-positive/near-zero sy0.
          This is NOT the thresh=50 clamp from the deferred study: it only
          touches values below the floor (a few pathological pixels, if any),
          is loud about how many, and can be disabled (set to 0) to see the
          raw pathological-pixel count instead.
  - Both yield and K are NaN-filled (8-neighbour iterative, same as the
    original script's replace_nan_with_neighbors) so every map is directly
    usable by the FEM. Raw (with NaN, pre-floor) and filled versions are both
    saved for inspection.

Writes, per key, to 2_data/yield_hardening/<key>/:
    S_p_yield_onset_frame.npy      (n1,n2) fractional onset frame index
    yield_stress_vm.npy            (n1,n2) raw, MPa, NaN where undetermined
    yield_stress_vm_filled.npy     (n1,n2) NaN-filled + safety-floor clamped
    K_hardening_coef_filled.npy    (n1,n2) dimensionless ratio, NaN-filled,
                                    clip [0,3] (multiply by K_SCALE for the
                                    FEM's K in MPa - done in 04, not here)
    diagnostics.json               counts + timing, see below

Run:  python 03_yield_hardening_alpha.py            (REF + every alpha)
      python 03_yield_hardening_alpha.py REF a100    (subset)
"""
import os
import sys
import re
import json
import time
import numpy as np
from numba import njit, prange
from scipy.ndimage import gaussian_filter
from joblib import Parallel, delayed

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
import alpha_config as cfg

# ALPHA_FORCE=1 recomputes keys that already have complete outputs, and
# ignores any existing onset checkpoint.
FORCE = os.environ.get("ALPHA_FORCE", "0") == "1"


# ============================================================
# 1. regression / onset-detection primitives
#    (byte-identical logic to el_K_recalculate_to_Adil_modified_2.py)
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
def dectect_maximum_r2(u, v, Evm, len_x):
    """R^2-based yield-onset detector: finds the fit window [0:k] whose
    linear (elastic) fit best explains the data (max R^2), the first such
    local maximum after index 6, or the global max if none found; then
    extends forward while the residual plastic slope is non-positive."""
    Evm_uv = Evm[:, u, v]
    b = []
    N = Evm.shape[0]

    for k in range(5, N):
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

    try:
        l = find_local_maxima(np.array(b[1:]))[0] + 6
    except Exception:
        max_value = max(b[1:])
        l = b.index(max_value) + 6

    for w in range(l, N):
        _, slope = estimate_coef(len_x[:w], Evm_uv[:w])
        if slope <= 0:
            l += 1
    return l


@njit(parallel=True)
def _onset_rows(Evm, len_x, r0, r1, out):
    """Yield-onset detection for rows [r0, r1), written into `out`.

    The original loop was `for i in prange(...)` inside a PLAIN Python
    function - outside @njit, prange is just range, so it ran as a pure
    Python double loop over ~11.2M pixels calling an njit function once per
    pixel. That is the multi-hour step. Compiling the outer loop (same
    pattern already used by compute_slope_plastic) makes it genuinely
    parallel and orders of magnitude faster; the per-pixel result is
    identical since dectect_maximum_r2 is unchanged.

    Row-blocked so the caller can checkpoint between blocks.
    """
    n2 = Evm.shape[2]
    for i in prange(r0, r1):
        for j in range(n2):
            out[i, j] = dectect_maximum_r2(i, j, Evm, len_x)


def _onset_rows_python(Evm, len_x, r0, r1, out):
    """Fallback with identical semantics, used only if _onset_rows fails to
    compile (dectect_maximum_r2 builds a reflected list, which numba's
    parallel backend can be fussy about on some versions). Slow - the
    original behaviour - but correct, and still checkpointed."""
    n2 = Evm.shape[2]
    for i in range(r0, r1):
        for j in range(n2):
            out[i, j] = dectect_maximum_r2(i, j, Evm, len_x)


def _save_atomic(path, arr):
    """Write to a temp file then replace, so a kill mid-write can never
    leave a half-written .npy that looks valid."""
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def compute_yield_onset_map(key, Evm, len_x, block_rows=None):
    """S_p for every pixel, checkpointed every `block_rows` rows.

    On restart, an existing checkpoint is loaded and computation resumes
    from the first unfinished row, so an interrupted run costs at most one
    block instead of the whole dataset.
    """
    if block_rows is None:
        block_rows = cfg.ONSET_CHECKPOINT_ROWS
    n1, n2 = Evm.shape[1], Evm.shape[2]
    d = cfg.yh_dir(key)
    ck_path = os.path.join(d, "_checkpoint_S_p.npy")
    ck_meta = os.path.join(d, "_checkpoint_S_p.json")

    S_p = np.full((n1, n2), np.nan)
    row0 = 0
    if FORCE and os.path.isfile(ck_path):
        print(f"[03] {key}: ALPHA_FORCE=1 - ignoring existing onset checkpoint.")
    elif os.path.isfile(ck_path) and os.path.isfile(ck_meta):
        try:
            with open(ck_meta) as f:
                meta = json.load(f)
            if (meta.get("shape") == [n1, n2] and meta.get("n_frames") == Evm.shape[0]
                    and meta.get("key") == key):
                S_p = np.load(ck_path)
                row0 = int(meta.get("next_row", 0))
                print(f"[03] {key}: RESUMING yield onset from row {row0}/{n1} "
                      f"({100*row0/n1:.1f}% already done, checkpoint found)")
            else:
                print(f"[03] {key}: checkpoint found but shape/frames/key differ "
                      f"from the current run - ignoring it and starting over.")
        except Exception as e:
            print(f"[03] {key}: unreadable checkpoint ({e}) - starting over.")

    if row0 >= n1:
        print(f"[03] {key}: yield onset already complete in checkpoint.")
        return S_p

    kernel = _onset_rows
    if row0 == 0:
        # warm up / verify the compiled kernel on a single row before
        # committing to it; fall back to the slow path if it won't compile
        try:
            _probe = np.full((1, n2), np.nan)
            _onset_rows(Evm, len_x, 0, 1, np.full((n1, n2), np.nan))
        except Exception as e:
            print(f"[03] {key}: numba-parallel onset kernel unavailable ({e}); "
                  f"falling back to the slow Python loop (still checkpointed).")
            kernel = _onset_rows_python

    t_blk = time.time()
    for r0 in range(row0, n1, block_rows):
        r1 = min(r0 + block_rows, n1)
        kernel(Evm, len_x, r0, r1, S_p)
        _save_atomic(ck_path, S_p)
        with open(ck_meta, "w") as f:
            json.dump(dict(key=key, next_row=int(r1), shape=[n1, n2],
                           n_frames=int(Evm.shape[0])), f)
        done = r1 - row0
        rate = done / max(time.time() - t_blk, 1e-9)
        eta = (n1 - r1) / max(rate, 1e-9)
        print(f"[03] {key}: yield onset rows {r1}/{n1} "
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
# 2. strain / stress helpers
# ============================================================
def deviatoric_strain_plane_stress(exx, eyy, exy, nu):
    ezz = -nu / (1.0 - nu) * (exx + eyy)
    trace = exx + eyy + ezz
    exx_dev = exx - trace / 3.0
    eyy_dev = eyy - trace / 3.0
    ezz_dev = ezz - trace / 3.0
    exy_dev = exy
    return exx_dev, eyy_dev, ezz_dev, exy_dev


def evm_from_edev(exx_dev, eyy_dev, ezz_dev, exy_dev):
    j2 = exx_dev ** 2 + eyy_dev ** 2 + ezz_dev ** 2 + 2.0 * exy_dev ** 2
    return np.sqrt((2.0 / 3.0) * j2)


def sigma_vm_from_evm(evm, E, nu):
    """Elastic conjugate of the deviatoric eps_vM used everywhere in this
    script: sigma_vm = 3*G*eps_vM. (The plane-stress Hooke-strain formula
    the original script also computed gives the identical number - verified
    numerically to ~1e-15 relative - so only this one is kept.)"""
    g = E / (2.0 * (1.0 + nu))
    return 3.0 * g * evm


def replace_nan_with_neighbors(a, max_iter=64):
    """Iterative 8-neighbour NaN fill (same as the original el_K script)."""
    a = a.astype(float, copy=True)
    for _ in range(max_iter):
        nan = np.isnan(a)
        if not nan.any():
            break
        neighbors = [
            np.pad(a, ((1, 0), (1, 0)), constant_values=np.nan)[:-1, :-1],
            np.pad(a, ((1, 0), (0, 0)), constant_values=np.nan)[:-1, :],
            np.pad(a, ((1, 0), (0, 1)), constant_values=np.nan)[:-1, 1:],
            np.pad(a, ((0, 0), (1, 0)), constant_values=np.nan)[:, :-1],
            np.pad(a, ((0, 0), (0, 1)), constant_values=np.nan)[:, 1:],
            np.pad(a, ((0, 1), (1, 0)), constant_values=np.nan)[1:, :-1],
            np.pad(a, ((0, 1), (0, 0)), constant_values=np.nan)[1:, :],
            np.pad(a, ((0, 1), (0, 1)), constant_values=np.nan)[1:, 1:],
        ]
        stack = np.stack(neighbors, axis=0)
        sums = np.nansum(stack, axis=0)
        counts = np.sum(np.isfinite(stack), axis=0)
        means = sums / np.where(counts == 0, 1, counts)
        a[nan & (counts > 0)] = means[nan & (counts > 0)]
    return a


def repair_bad_frames(Evm, bad_frames_1based):
    """Linearly interpolate the given 1-based frame indices from the nearest
    GOOD neighbours (skipping over other bad frames), applied uniformly
    across every pixel - generalizes the original script's hardcoded
    Evm[31] = (Evm[29]+Evm[33])/2 pair-averaging to any list/run length."""
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
            continue   # every frame is "bad" - nothing sane to interpolate from
        if lo < 0:
            Evm[i] = Evm[hi]
        elif hi >= N:
            Evm[i] = Evm[lo]
        else:
            w = (i - lo) / (hi - lo)
            Evm[i] = (1 - w) * Evm[lo] + w * Evm[hi]
    return Evm


def compute_stress_yield_map(S_p, U_all, V_all, E, nu, n_jobs=-1):
    """Per-pixel yield stress: for each pixel, take the strain field at ITS
    OWN onset frame (nearest integer of S_p) and evaluate sigma_vm there.
    Vectorised over the small set of DISTINCT onset frames actually used
    (not one gradient computation per pixel)."""
    stress_yield = np.full(S_p.shape, np.nan, dtype=float)
    n_frames = len(U_all)

    mask = (S_p >= 2) & (S_p <= n_frames) & ~np.isnan(S_p)
    m_map = np.full(S_p.shape, -1, dtype=int)
    # S_p is a 1-based frame count (len_x starts at 1); U_all/V_all are 0-based
    # lists of length n_frames, so S_p==n_frames must map to index n_frames-1,
    # not n_frames (which is out of range - this crashed on any pixel whose
    # onset estimate landed exactly on the last frame, e.g. a never-yielding
    # pixel with S_p pinned at N).
    m_map[mask] = np.clip(S_p[mask].astype(int) - 1, 0, n_frames - 1)

    def process_m(m):
        U_i = np.asarray(U_all[m])
        V_i = np.asarray(V_all[m])
        Exy, Exx = np.gradient(U_i)
        Eyy, Eyx = np.gradient(V_i)
        Eshear = 0.5 * (Exy + Eyx)
        Exx_dev, Eyy_dev, Ezz_dev, Exy_dev = deviatoric_strain_plane_stress(
            Exx, Eyy, Eshear, nu)
        evm_dev = evm_from_edev(Exx_dev, Eyy_dev, Ezz_dev, Exy_dev)
        return m, sigma_vm_from_evm(evm_dev, E, nu)

    used_frames = sorted(set(m_map[mask].tolist()))
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(process_m)(m) for m in used_frames)

    for m, sigma_ij in results:
        idx = np.argwhere(m_map == m)
        if len(idx) == 0:
            continue
        i_coords, j_coords = idx[:, 0], idx[:, 1]
        stress_yield[i_coords, j_coords] = sigma_ij[i_coords, j_coords]
    return stress_yield


# ============================================================
# 3. per-dataset pipeline
# ============================================================
def _outputs_complete(key):
    """True if this key's final products are all already on disk."""
    d = cfg.yh_dir(key)
    return all(os.path.isfile(os.path.join(d, f)) for f in (
        "S_p_yield_onset_frame.npy", "yield_stress_vm.npy",
        "yield_stress_vm_filled.npy", "K_hardening_coef_filled.npy",
        "diagnostics.json"))


def process_dataset(key):
    t0 = time.time()
    print("=" * 78)
    print(f"[03] {key}")

    # skip keys that already finished, so re-running the script after an
    # interruption resumes the sweep instead of redoing completed datasets
    if _outputs_complete(key) and not FORCE:
        d = cfg.yh_dir(key)
        with open(os.path.join(d, "diagnostics.json")) as f:
            diag = json.load(f)
        print(f"[03] {key}: already complete -> {d}  (set ALPHA_FORCE=1 to redo)")
        return diag

    u_dir, v_dir = cfg.uv_dirs(key)
    files_U = cfg.get_sorted_npy_files(u_dir, "U")
    files_V = cfg.get_sorted_npy_files(v_dir, "V")
    if len(files_U) != len(files_V):
        raise RuntimeError(f"[{key}] {len(files_U)} U frames vs "
                           f"{len(files_V)} V frames - mismatch")
    N = len(files_U)
    print(f"[03] {key}: {N} frames from {u_dir}")

    U_all = [np.load(os.path.join(u_dir, f), mmap_mode="r") for f in files_U]
    V_all = [np.load(os.path.join(v_dir, f), mmap_mode="r") for f in files_V]

    d = cfg.yh_dir(key)
    bad = cfg.BAD_FRAMES.get(key, [])
    len_x = np.linspace(1, N, N)

    # ---- Evm(frame, i, j) - CACHED to disk ----
    # This is the fully-processed strain history (bad frames repaired +
    # temporal Gaussian). Saved so an interrupted run never recomputes it,
    # and so it can be inspected/reused independently of this script.
    # Stored as float32: it is a (N, 3600, 3100) array, which is ~3.7 GB in
    # float64 but ~1.9 GB in float32, and float32 has far more precision
    # than the DIC displacements it derives from.
    evm_path = os.path.join(d, "Evm_history.npy")
    evm_meta = os.path.join(d, "Evm_history.json")
    Evm = None
    if os.path.isfile(evm_path) and os.path.isfile(evm_meta) and not FORCE:
        try:
            with open(evm_meta) as f:
                meta = json.load(f)
            if (meta.get("n_frames") == N and meta.get("bad_frames") == list(bad)
                    and meta.get("key") == key):
                Evm = np.load(evm_path)
                print(f"[03] {key}: loaded cached Evm history {Evm.shape} "
                      f"({Evm.nbytes/1e9:.2f} GB) - skipping recomputation")
            else:
                print(f"[03] {key}: cached Evm does not match current settings "
                      f"(frames/bad_frames/key) - recomputing.")
        except Exception as e:
            print(f"[03] {key}: unreadable Evm cache ({e}) - recomputing.")

    if Evm is None:
        Evm_all_dev = []
        for m in range(N):
            Exy, Exx = np.gradient(U_all[m])
            Eyy, Eyx = np.gradient(V_all[m])
            Eshear = 0.5 * (Exy + Eyx)
            Exx_dev, Eyy_dev, Ezz_dev, Exy_dev = deviatoric_strain_plane_stress(
                Exx, Eyy, Eshear, cfg.NU)
            Evm_all_dev.append(evm_from_edev(Exx_dev, Eyy_dev, Ezz_dev, Exy_dev))
        Evm = np.array(Evm_all_dev)

        if bad:
            print(f"[03] {key}: repairing bad frames {bad} "
                  f"(linear interp from nearest good neighbours)")
        Evm = repair_bad_frames(Evm, bad)
        Evm = gaussian_filter(Evm, sigma=[1, 0, 0])   # temporal smoothing only
        Evm = Evm.astype(np.float32)

        if cfg.SAVE_EVM_CACHE:
            _save_atomic(evm_path, Evm)
            with open(evm_meta, "w") as f:
                json.dump(dict(key=key, n_frames=int(N), bad_frames=list(bad),
                               shape=[int(s) for s in Evm.shape],
                               dtype="float32", gaussian_sigma=[1, 0, 0]), f)
            print(f"[03] {key}: Evm history saved {Evm.shape} "
                  f"({Evm.nbytes/1e9:.2f} GB) -> {evm_path}")
        else:
            print(f"[03] {key}: ALPHA_SAVE_EVM=0 - Evm not cached "
                  f"(would have been {Evm.nbytes/1e9:.2f} GB)")

    # ---- yield onset frame S_p (checkpointed - the expensive step) ----
    S_p = compute_yield_onset_map(key, Evm, len_x)
    # save S_p as a real product immediately: everything below is comparatively
    # cheap, so a later failure should never cost the onset computation again
    _save_atomic(os.path.join(d, "S_p_yield_onset_frame.npy"), S_p)
    n_Sp_nan = int(np.isnan(S_p).sum())
    n_Sp_valid = int(((S_p >= 2) & (S_p <= N) & ~np.isnan(S_p)).sum())
    print(f"[03] {key}: S_p valid onset in {n_Sp_valid}/{S_p.size} px "
          f"({100*n_Sp_valid/S_p.size:.2f}%), NaN in {n_Sp_nan} px")

    # ---- yield stress ----
    stress_yield = compute_stress_yield_map(S_p, U_all, V_all, cfg.E_MOD, cfg.NU)
    n_yield_nan = int(np.isnan(stress_yield).sum())
    _save_atomic(os.path.join(d, "yield_stress_vm.npy"), stress_yield)

    # ---- hardening slope / coefficient (no thresholding) ----
    slop_plastic = compute_slope_plastic(Evm, S_p, len_x)
    slope_plastic = update_slope_plastic(Evm, slop_plastic, S_p, len_x)
    hardening_coef = cfg.SLOP_REF / slope_plastic

    hardening_bounded = hardening_coef.copy()
    n_K_gt3 = int(np.sum(hardening_bounded > 3))
    n_K_lt0 = int(np.sum(hardening_bounded < 0))
    hardening_bounded[hardening_bounded > 3] = np.nan
    hardening_bounded[hardening_bounded < 0] = np.nan
    n_K_nan_before_fill = int(np.isnan(hardening_bounded).sum())
    hardening_filled = replace_nan_with_neighbors(hardening_bounded)

    # ---- yield: NaN-fill + numerical safety floor (NOT the deferred thresh) ----
    yield_filled = replace_nan_with_neighbors(stress_yield.copy())
    floor = cfg.YIELD_SAFETY_FLOOR_MPA
    n_floor_clamped = 0
    if floor > 0:
        below = yield_filled < floor
        n_floor_clamped = int(below.sum())
        yield_filled[below] = floor
    remaining_nan = int(np.isnan(yield_filled).sum())
    if remaining_nan:
        print(f"[03] {key}: WARNING {remaining_nan} px still NaN after "
              f"8-neighbour fill (isolated/large NaN clusters) - filling "
              f"with the dataset median as a last resort.")
        yield_filled[np.isnan(yield_filled)] = np.nanmedian(yield_filled)

    wall = time.time() - t0
    _save_atomic(os.path.join(d, "yield_stress_vm_filled.npy"), yield_filled)
    _save_atomic(os.path.join(d, "K_hardening_coef_filled.npy"), hardening_filled)

    diag = dict(
        key=key, n_frames=N, wall_s=wall,
        n_pixels=int(S_p.size),
        n_Sp_nan=n_Sp_nan, n_Sp_valid=n_Sp_valid,
        n_yield_nan_before_fill=n_yield_nan,
        n_yield_floor_clamped=n_floor_clamped,
        yield_safety_floor_mpa=floor,
        n_K_gt3_clipped=n_K_gt3, n_K_lt0_clipped=n_K_lt0,
        n_K_nan_before_fill=n_K_nan_before_fill,
        bad_frames_repaired=bad,
        mean_yield_vm_mpa=float(np.nanmean(yield_filled)),
        mean_K_coef=float(np.nanmean(hardening_filled)),
        mean_K_mpa=float(np.nanmean(hardening_filled) * cfg.K_SCALE),
    )
    with open(os.path.join(d, "diagnostics.json"), "w") as f:
        json.dump(diag, f, indent=2)

    # key is complete - drop the onset checkpoint so a later re-run doesn't
    # resurrect a stale partial map (the final S_p .npy is the product now)
    for _f in ("_checkpoint_S_p.npy", "_checkpoint_S_p.json"):
        _p = os.path.join(d, _f)
        if os.path.isfile(_p):
            os.remove(_p)

    print(f"[03] {key}: yield NaN {n_yield_nan}/{S_p.size} before fill, "
          f"{n_floor_clamped} floor-clamped (<{floor} MPa) | "
          f"K: {n_K_gt3} clipped >3, {n_K_lt0} clipped <0")
    print(f"[03] {key}: mean yield {diag['mean_yield_vm_mpa']:.2f} MPa | "
          f"mean K {diag['mean_K_coef']:.4f} ({diag['mean_K_mpa']:.1f} MPa "
          f"after x{cfg.K_SCALE}) | {wall:.1f}s -> {d}")
    return diag


if __name__ == "__main__":
    keys = sys.argv[1:] or cfg.DATASET_KEYS
    bad_keys = [k for k in keys if k not in cfg.DATASET_KEYS]
    if bad_keys:
        raise SystemExit(f"unknown dataset key(s) {bad_keys}; "
                         f"available: {cfg.DATASET_KEYS}")
    t_all = time.time()
    diags = [process_dataset(k) for k in keys]
    print("=" * 78)
    for d in diags:
        print(f"  {d['key']:>6s}: mean yield {d['mean_yield_vm_mpa']:7.2f} MPa  "
              f"mean K {d['mean_K_mpa']:7.1f} MPa  {d['wall_s']:6.1f}s")
    print(f"TOTAL {time.time()-t_all:.1f}s")
    print("Now run:  python 04_run_fem_alpha.py")
