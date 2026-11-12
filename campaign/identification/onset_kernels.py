# -*- coding: utf-8 -*-
"""
onset_kernels.py - shared onset-detection kernels for the yield-definition
study (Dataset 1: 05_d1_onset_variants.py, Dataset 2: 04_onset_variants.py).

Both kernels work on a SELECTED subset of the frames (`sel`, 0-based frame
indices into the full series, x = 1-based frame numbers), so the same code
tests temporal-sampling sensitivity.  Results are always returned in the
full-series 1-based count convention of the identification scripts
(yield read at frame index S_p - 1).

r2_onset_fast
    O(N) re-implementation of dectect_maximum_r2 (Dataset 1 02_identify.py,
    Dataset 2 02_yield_hardening_generic.py).  R^2(k) = corr(y[:k], x[:k])^2
    for any non-zero slope, so all sums are prefix sums.  Branch structure is
    kept: slope guard, first strict local maximum of b[1:], fallback to the
    global maximum, trailing slope<=0 extension.  Dataset 2 kernel:
    clamp_hi=True (onset clamped to [min_k, N-1]), fb_add=min_k.  Dataset 1
    kernel (02_identify, min_k fixed at 5): clamp_hi=False (can return N),
    fb_add=6.  Checked against the original
    kernels on random pixel samples by the calling scripts.

offset_onset_sel
    Strain-offset rule of Dataset 2 (detect_offset_onset) generalised to a
    frame subset.  Elastic line fitted through the selected frames whose
    frame number lies in [fit_lo, fit_hi]; onset at the first selected frame
    with 0-based index >= min_k whose strain exceeds the line by `offset`.
    NaN when the line does not rise by more than `offset` across the window.
    At stride 1 it is identical to detect_offset_onset (checked by the
    caller).  Also returns the elastic-line strain at the frame where the
    yield descriptor is read, for the force-free Hooke descriptor.

offset_onset_stress_block
    The same strain-offset rule with the elastic line fitted against the
    machine stress of each frame instead of the frame number: e = a + b*s over
    the frames of the elastic window, onset at the first frame (0-based index
    >= min_k) whose strain exceeds a + b*s by `offset`.  The window, the
    earliest admissible onset and the returned count follow
    offset_onset_block exactly, so passing s = frame number reproduces it
    (checked by the callers).  The line must rise by more than `offset`
    across the stress range of the window.

rise_min (both offset kernels): the elastic line must rise by more than this
across its window, otherwise the pixel is NaN (noise guard of 02's
detect_offset_onset, where it equals the offset).  Default: the offset.
The identification scripts fix it at the production offset 2e-4 so the
offset-sensitivity variants change only the offset, not the guard.
"""
import numpy as np
from numba import njit, prange


@njit(error_model="numpy")
def _first_local_max(b, start, n_used):
    n = n_used - start
    for i in range(2, n - 2):
        a = b[start + i]
        if (a > b[start + i - 1] and b[start + i - 1] > b[start + i - 2] and
                a > b[start + i + 1] and b[start + i + 1] > b[start + i + 2]):
            return i
    return -1


@njit(error_model="numpy")
def r2_onset_fast(y, x, min_k, fit_from, clamp_hi, fb_add, work):
    """1-based onset count into the series (x, y); see module docstring.
    fb_add: constant added to the index of the global maximum in the
    no-local-maximum fallback (Dataset 2 kernel: min_k; Dataset 1: 6)."""
    N = y.shape[0]
    Sx = work[0:N + 1]
    Sy = work[N + 1:2 * N + 2]
    Sxx = work[2 * N + 2:3 * N + 3]
    Syy = work[3 * N + 3:4 * N + 4]
    Sxy = work[4 * N + 4:5 * N + 5]
    b = work[5 * N + 5:6 * N + 6]
    x0 = x[0]
    y0 = np.float64(y[0])
    Sx[0] = 0.0
    Sy[0] = 0.0
    Sxx[0] = 0.0
    Syy[0] = 0.0
    Sxy[0] = 0.0
    for i in range(N):
        xi = x[i] - x0
        yi = np.float64(y[i]) - y0
        Sx[i + 1] = Sx[i] + xi
        Sy[i + 1] = Sy[i] + yi
        Sxx[i + 1] = Sxx[i] + xi * xi
        Syy[i + 1] = Syy[i] + yi * yi
        Sxy[i + 1] = Sxy[i] + xi * yi
    nb = N - min_k
    if nb <= 1:
        return min_k
    for t in range(nb):
        k = min_k + t
        n = k - fit_from
        sx = Sx[k] - Sx[fit_from]
        sy = Sy[k] - Sy[fit_from]
        sxx = Sxx[k] - Sxx[fit_from]
        sxy = Sxy[k] - Sxy[fit_from]
        slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        if slope > -100:
            if slope == 0.0:
                b[t] = np.nan
            else:
                cxx = k * Sxx[k] - Sx[k] * Sx[k]
                cyy = k * Syy[k] - Sy[k] * Sy[k]
                cxy = k * Sxy[k] - Sx[k] * Sy[k]
                if cxx > 0.0 and cyy > 0.0:
                    b[t] = (cxy * cxy) / (cxx * cyy)
                else:
                    b[t] = np.nan
        else:
            b[t] = -10000.0
    j = _first_local_max(b, 1, nb)
    if j >= 0:
        l = j + min_k + 1
    else:
        m = b[1]
        for t in range(2, nb):
            if b[t] > m:
                m = b[t]
        idx = 0
        for t in range(nb):
            if b[t] == m:
                idx = t
                break
        l = idx + fb_add
    l0 = l
    for w in range(l0, N):
        sx = Sx[w]
        sy = Sy[w]
        slope = (w * Sxy[w] - sx * sy) / (w * Sxx[w] - sx * sx)
        if slope <= 0:
            l += 1
    if clamp_hi:
        if l > N - 1:
            l = N - 1
        if l < min_k:
            l = min_k
    return l


@njit(parallel=True)
def r2_onset_block(E3, x_full, sel, min_k, fit_from, clamp_hi, fb_add, out):
    """E3 (n_frames, rows, cols); writes full-series 1-based counts."""
    nr, nc = E3.shape[1], E3.shape[2]
    Ns = sel.shape[0]
    xs = np.empty(Ns)
    for s in range(Ns):
        xs[s] = x_full[sel[s]]
    for i in prange(nr):
        y = np.empty(Ns, dtype=np.float32)
        work = np.empty(6 * (Ns + 1))
        for j in range(nc):
            for s in range(Ns):
                y[s] = E3[sel[s], i, j]
            # ls is a 1-based count <= Ns (the unclamped Dataset 1 kernel
            # can return Ns itself, which maps to the last selected frame)
            ls = r2_onset_fast(y, xs, min_k, fit_from, clamp_hi, fb_add, work)
            out[i, j] = sel[ls - 1] + 1


@njit(parallel=True)
def offset_onset_block(E3, x_full, sel, fit_lo, fit_hi, min_k, offset,
                       out, e_el, rise_min=-1.0):
    """Strain-offset onset on the selected frames (see module docstring).

    fit_lo, fit_hi : 1-based frame numbers, inclusive (elastic window)
    min_k          : earliest admissible onset as a 0-based frame index
    out            : full-series count (production convention), NaN if the
                     elastic fit does not rise by more than `offset`
    e_el           : elastic-line strain at the frame where the yield
                     descriptor is read (index out - 1)
    """
    nr, nc = E3.shape[1], E3.shape[2]
    N = E3.shape[0]
    Ns = sel.shape[0]
    for i in prange(nr):
        for j in range(nc):
            n = 0
            sx = 0.0
            sy = 0.0
            sxx = 0.0
            sxy = 0.0
            for s in range(Ns):
                xv = x_full[sel[s]]
                if xv >= fit_lo and xv <= fit_hi:
                    yv = np.float64(E3[sel[s], i, j])
                    n += 1
                    sx += xv
                    sy += yv
                    sxx += xv * xv
                    sxy += xv * yv
            if n < 3:
                out[i, j] = np.nan
                e_el[i, j] = np.nan
                continue
            mx = sx / n
            my = sy / n
            slope = (sxy - n * my * mx) / (sxx - n * mx * mx)
            icpt = my - slope * mx
            rmin = offset if rise_min < 0 else rise_min
            if not (slope * (fit_hi - fit_lo + 1) > rmin):
                out[i, j] = np.nan
                e_el[i, j] = np.nan
                continue
            l = N - 1
            for s in range(Ns):
                k = sel[s]
                if k < min_k:
                    continue
                if E3[k, i, j] - (icpt + slope * x_full[k]) > offset:
                    l = k
                    break
            if l > N - 1:
                l = N - 1
            if l < min_k:
                l = min_k
            out[i, j] = l
            # descriptor read at index l-1, whose frame number is l
            e_el[i, j] = icpt + slope * l


@njit(parallel=True)
def offset_onset_stress_block(E3, s_full, sel, fit_lo, fit_hi, min_k, offset,
                              out, e_el, rise_min=-1.0):
    """Strain-offset onset with the elastic line e = a + b*s fitted against
    the machine stress s_full (one value per frame of the full series).

    fit_lo, fit_hi : 1-based frame numbers, inclusive (elastic window)
    min_k          : earliest admissible onset as a 0-based frame index
    out            : full-series count (production convention), NaN if the
                     elastic line does not rise by more than `offset`
    e_el           : elastic-line strain at the frame where the yield
                     descriptor is read (index out - 1)
    """
    nr, nc = E3.shape[1], E3.shape[2]
    N = E3.shape[0]
    Ns = sel.shape[0]
    s_lo = 1e300
    s_hi = -1e300
    for s in range(Ns):
        f = sel[s] + 1
        if f >= fit_lo and f <= fit_hi:
            v = s_full[sel[s]]
            if v < s_lo:
                s_lo = v
            if v > s_hi:
                s_hi = v
    for i in prange(nr):
        for j in range(nc):
            n = 0
            sx = 0.0
            sy = 0.0
            sxx = 0.0
            sxy = 0.0
            for s in range(Ns):
                f = sel[s] + 1
                if f >= fit_lo and f <= fit_hi:
                    xv = s_full[sel[s]]
                    yv = np.float64(E3[sel[s], i, j])
                    n += 1
                    sx += xv
                    sy += yv
                    sxx += xv * xv
                    sxy += xv * yv
            if n < 3:
                out[i, j] = np.nan
                e_el[i, j] = np.nan
                continue
            mx = sx / n
            my = sy / n
            slope = (sxy - n * my * mx) / (sxx - n * mx * mx)
            icpt = my - slope * mx
            rmin = offset if rise_min < 0 else rise_min
            if not (slope * (s_hi - s_lo) > rmin):
                out[i, j] = np.nan
                e_el[i, j] = np.nan
                continue
            l = N - 1
            for s in range(Ns):
                k = sel[s]
                if k < min_k:
                    continue
                if E3[k, i, j] - (icpt + slope * s_full[k]) > offset:
                    l = k
                    break
            if l > N - 1:
                l = N - 1
            if l < min_k:
                l = min_k
            out[i, j] = l
            # descriptor read at index l-1
            e_el[i, j] = icpt + slope * s_full[l - 1]
