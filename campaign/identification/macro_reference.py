"""Field-mean reference quantities shared by the Dataset 1 and Dataset 2
identification scripts.

The per-pixel rule (strain offset against an elastic line fitted over the
load window, plastic slope from 02_identify.fct_slope) is applied to the
field-mean equivalent-strain history.  The plastic slope of the field mean is
SLOP_REF, so a pixel that follows the field mean gets a hardening coefficient
of one.  The same history, read against the machine stress, gives the
macroscopic law used by Dataset 1 (no extensometer record exists for that
test): sigma0 at 0.02 % equivalent plastic strain with the nominal modulus,
and the additive Ludwik K with the exponent held fixed.
"""
import numpy as np

def equivalent_factor(nu):
    """eps_vM(deviatoric, plane stress) per unit uniaxial stress / E."""
    return 2.0 * (1.0 + nu) / 3.0


def estimate_coef(x, y):
    """Same least-squares line as 02_identify.estimate_coef."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    mx, my = x.mean(), y.mean()
    ssxy = np.sum(y * x) - n * my * mx
    ssxx = np.sum(x * x) - n * mx * mx
    slope = ssxy / ssxx
    return my - slope * mx, slope


def load_window(stress, sigma0, lo_frac, hi_frac):
    """First frame carrying >= lo_frac*sigma0 to the last frame carrying
    <= hi_frac*sigma0 (1-based, inclusive)."""
    s = np.asarray(stress, float)
    lo = int(np.argmax(s >= lo_frac * sigma0)) + 1
    hi = int(np.nonzero(s <= hi_frac * sigma0)[0].max()) + 1
    if not (1 <= lo < hi):
        raise SystemExit(f"no load window: lo={lo} hi={hi}")
    return lo, hi


def offset_onset_1d(e, abscissa, lo, hi, offset):
    """Onset count of one history (same convention as the block kernels):
    0-based index of the first frame past the window whose strain exceeds the
    elastic line by `offset`; the yield descriptor is read at index l-1."""
    e = np.asarray(e, float)
    x = np.asarray(abscissa, float)
    w = slice(lo - 1, hi)
    a, b = estimate_coef(x[w], e[w])
    for k in range(hi, e.size):
        if e[k] - (a + b * x[k]) > offset:
            return k, a, b
    return e.size - 1, a, b


def plastic_slope(e, l):
    """02_identify.fct_slope on one history with onset count l: elastic line
    through frames 1..l, then the slope of the residual from frame l+1 on,
    both against the frame number.  Falls back to fct_slope_negative."""
    e = np.asarray(e, float)
    x = np.arange(1, e.size + 1, dtype=float)
    a, b = estimate_coef(x[:l], e[:l])
    r = e - (a + b * x)
    _, s = estimate_coef(x[l:], r[l:])
    if s < 0:
        _, s = estimate_coef(x[l:], e[l:])
    return float(s)


def reference(emean, stress, lo, hi, offset, abscissa="stress"):
    """SLOP_REF and onset of the field-mean history by the pixel rule."""
    x = stress if abscissa == "stress" else np.arange(1, len(emean) + 1, dtype=float)
    l, a, b = offset_onset_1d(emean, x, lo, hi, offset)
    return dict(abscissa=abscissa, fit_lo=int(lo), fit_hi=int(hi), onset_count=int(l),
                yield_frame=int(l), yield_stress_mpa=float(np.asarray(stress)[l - 1]),
                slop_ref=plastic_slope(emean, l))


def macro_law(emean, stress, lo, hi, E, nu, n_exp, offset=2e-4):
    """Macroscopic additive Ludwik law from the field-mean history.

    E=None: the 0.02 % offset is taken from the elastic line fitted to the
    record itself (the pixel rule); otherwise from a line of slope E.

    Elastic line e = c0 + c1*s over the load window (its intercept removes the
    strain offset of the reference image); equivalent plastic strain
    ep = e - c0 - F*s/E with the nominal modulus E; sigma0 = stress at
    ep = offset (linear between the bracketing frames); K by least squares of
    s - sigma0 = K ep^n over the frames with ep > offset, n fixed."""
    e = np.asarray(emean, float)
    s = np.asarray(stress, float)
    F = equivalent_factor(nu)
    w = slice(lo - 1, hi)
    c0, c1 = estimate_coef(s[w], e[w])
    if E is None:                    # offset from the record's own fitted elastic line
        E = F / c1
    ep = e - c0 - F * s / E
    i = int(np.argmax(ep > offset))
    if i == 0:
        raise SystemExit("field-mean plastic strain never exceeds the offset")
    t = (offset - ep[i - 1]) / (ep[i] - ep[i - 1])
    sigma0 = float(s[i - 1] + t * (s[i] - s[i - 1]))
    pl = ep > offset
    g = ep[pl] ** n_exp
    K = float(np.sum((s[pl] - sigma0) * g) / np.sum(g * g))
    rms = float(np.sqrt(np.mean((sigma0 + K * g - s[pl]) ** 2)))
    return dict(E_app_mpa=float(F / c1), sigma0_mpa=sigma0, K_mpa=K, n_exp=float(n_exp),
                fit_rms_mpa=rms, plastic_frames=int(pl.sum()), ep_final=float(ep[-1]),
                window=[int(lo), int(hi)])


def self_consistent_law(emean, stress, sigma0_start, fracs, E, nu, n_exp, offset=2e-4, max_it=20):
    """Iterate window(sigma0) -> law -> sigma0 until the window is stable."""
    s0, seen = float(sigma0_start), []
    for _ in range(max_it):
        lo, hi = load_window(stress, s0, *fracs)
        law = macro_law(emean, stress, lo, hi, E, nu, n_exp, offset)
        seen.append((lo, hi, law["sigma0_mpa"]))
        lo2, hi2 = load_window(stress, law["sigma0_mpa"], *fracs)
        s0 = law["sigma0_mpa"]
        if (lo2, hi2) == (lo, hi):
            law["iterations"] = [list(x) for x in seen]
            return law
    raise SystemExit(f"macro law window did not settle: {seen}")
