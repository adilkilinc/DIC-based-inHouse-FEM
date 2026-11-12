#!/usr/bin/env python3
"""
fem_tri.py - in-house FEM replacing Abaqus at the mesoscale (CPS3 grain mesh)
=============================================================================
Element  : CPS3 constant-strain triangle, plane stress, 1 integration point
Material : von Mises, per-grain PIECEWISE-LINEAR hardening table - the same
           50-point *Plastic table Abaqus interpolates, perfectly plastic
           beyond the last knot
BCs      : prescribed (ux, uy) at every node on the outer boundary
Solver   : incremental loading with automatic cutback, Newton-Raphson with the
           CONSISTENT tangent

This is the unstructured-triangle sibling of Microscale/.../fem/1_codes/
fem_pixel.py.  The return mapping (`_rm`), the consistent tangent (`_cep`) and
the increment/cutback logic are the SAME algorithms, ported from the CPS4
version; the differences are only (a) CST kinematics instead of 2x2 Gauss CPS4,
(b) hardening looked up per grain from a table instead of an analytic Ludwik
law with per-pixel (sy0, K).

Outputs (element-level, one value per element - CPS3 has one integration point):
  U    (n_node, 2)
  RF   (n_node, 2)   nonzero only at prescribed dofs
  S    (n_elem, 3)   [s11, s22, s12]
  E    (n_elem, 3)   [e11, e22, gamma12]      gamma = ENGINEERING shear
  PE   (n_elem, 3)   [pe11, pe22, gamma_p12]
  PEEQ (n_elem,)

Usage:
    from fem_tri import run_fem_tri
    res = run_fem_tri(coords, elements, mat_id, table, ep_table,
                      bc_nodes, bc_u, E_mod=205000., nu=0.3, N_inc=20)
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve, splu
import os
import sys
import time

# ── run log ─────────────────────────────────────────────────────────────────
# Everything this module reports is mirrored to a log file so a run can be
# diagnosed AFTERWARDS - the Newton history, cutbacks and line-search steps are
# otherwise only on screen and lost. The trace is written to the log even when
# verbose=False, because a driver that runs quiet is exactly the one whose
# behaviour you cannot inspect later.
#
# ONE file per process, appended across calls, so a driver that solves many
# tiles produces one log rather than hundreds.
#
#   FEM_LOG=0            disable entirely
#   FEM_LOG_FILE=<path>  write exactly there
#   FEM_LOG_DIR=<dir>    write <dir>/fem_tri_<stamp>_pid<pid>.log
#   (default)            ./fem_logs/fem_tri_<stamp>_pid<pid>.log
_LOG_NAME = "fem_tri"
_LOG_FH = None
_LOG_OFF = False


def log_path():
    """Path this process is logging to, or None when logging is off."""
    return getattr(_LOG_FH, "name", None)


def _open_log():
    global _LOG_FH, _LOG_OFF
    if _LOG_FH is not None or _LOG_OFF:
        return
    if os.environ.get("FEM_LOG", "1") == "0":
        _LOG_OFF = True
        return
    p = os.environ.get("FEM_LOG_FILE")
    if not p:
        d = os.environ.get("FEM_LOG_DIR") or os.path.join(os.getcwd(), "fem_logs")
        p = os.path.join(d, "%s_%s_pid%d.log" % (
            _LOG_NAME, time.strftime("%Y%m%d_%H%M%S"), os.getpid()))
    try:
        d = os.path.dirname(os.path.abspath(p))
        if d:
            os.makedirs(d, exist_ok=True)
        _LOG_FH = open(p, "a", buffering=1)
    except OSError as ex:
        _LOG_OFF = True
        sys.stderr.write("[%s] logging disabled, cannot open %s (%s)\n"
                         % (_LOG_NAME, p, ex))


def print(*args, sep=" ", end="\n", flush=True):        # noqa: A001
    """stdout exactly as before, plus the run log when one is open."""
    msg = sep.join(str(a) for a in args) + end
    sys.stdout.write(msg)
    if flush:
        sys.stdout.flush()
    if _LOG_FH is not None:
        _LOG_FH.write(msg)


def _trace(msg, echo=True):
    """Always to the run log; to stdout only when the caller asked to see it."""
    if echo:
        print(msg)
    elif _LOG_FH is not None:
        _LOG_FH.write(msg + "\n")

# Optional fast multithreaded direct solver (MKL Pardiso):  pip install pypardiso
try:
    import pypardiso

    def _solve(A, b):
        return pypardiso.spsolve(A.tocsr(), b)
    _SOLVER_NAME = "pypardiso (MKL, multithreaded)"
except Exception:
    def _solve(A, b):
        return spsolve(A.tocsr(), b)
    _SOLVER_NAME = ("scipy SuperLU (single-threaded; "
                    "'pip install pypardiso' for a large speed-up)")

# von Mises plane-stress matrix:  svm^2 = s^T M s
_M = np.array([[1., -0.5, 0.], [-0.5, 1., 0.], [0., 0., 3.]])


def _vm(s):
    return np.sqrt(np.maximum(s[..., 0]**2 - s[..., 0]*s[..., 1]
                              + s[..., 1]**2 + 3*s[..., 2]**2, 0.))


def _Cps(E, nu):
    f = E / (1 - nu*nu)
    return f * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu)/2.]])


# ── Hardening: per-material piecewise-linear table ──────────────────────────
class TabularHardening:
    """sigma_y(material, ep), piecewise linear on a SHARED uniform ep grid.

    This is exactly how Abaqus evaluates a *Plastic table: linear interpolation
    between the tabulated points, and constant (perfectly plastic) beyond the
    last one.  Because every grain uses the same ep knots
    (linspace(0, 0.2, 50)), the segment index is a single division - no
    per-material searchsorted.
    """

    def __init__(self, table, ep_k):
        self.table = np.ascontiguousarray(table, dtype=np.float64)   # (n_mat, n_pts)
        self.ep_k = np.asarray(ep_k, dtype=np.float64)
        if self.table.shape[1] != len(self.ep_k):
            raise ValueError("table and ep_k length mismatch")
        d = np.diff(self.ep_k)
        if not np.allclose(d, d[0], rtol=1e-10, atol=1e-14):
            raise ValueError("fem_tri.TabularHardening needs a UNIFORM ep grid")
        if self.ep_k[0] != 0.0:
            raise ValueError("the *Plastic table must start at ep = 0")
        self.dep = float(d[0])
        self.slope = np.diff(self.table, axis=1) / self.dep          # (n_mat, n_seg)
        self.n_seg = self.table.shape[1] - 1
        self.ep_last = float(self.ep_k[-1])

    def _seg(self, ep):
        return np.clip((ep / self.dep).astype(np.int64), 0, self.n_seg - 1)

    def sy(self, m, ep):
        epc = np.minimum(ep, self.ep_last)
        i = self._seg(epc)
        return self.table[m, i] + self.slope[m, i] * (epc - self.ep_k[i])

    def H(self, m, ep):
        i = self._seg(np.minimum(ep, self.ep_last))
        return np.where(ep < self.ep_last, self.slope[m, i], 0.0)


class AnalyticLudwikHardening:
    """Per-material analytic Ludwik law, sigma_y = sy0 + K ep^n."""

    def __init__(self, sy0, K, n_exp):
        self.sy0 = np.asarray(sy0, dtype=np.float64)
        self.K = np.asarray(K, dtype=np.float64)
        self.n_exp = float(n_exp)
        if self.sy0.shape != self.K.shape:
            raise ValueError("sy0 and K must have identical shapes")

    def sy(self, m, ep):
        ep = np.asarray(ep, dtype=np.float64)
        hard = np.where(ep > 0.0, np.maximum(ep, 0.0)**self.n_exp, 0.0)
        return self.sy0[m] + self.K[m]*hard

    def H(self, m, ep):
        ep = np.asarray(ep, dtype=np.float64)
        slope = np.where(ep > 1e-15, ep**(self.n_exp - 1.0), 0.0)
        return self.K[m]*self.n_exp*slope


class GradedLudwikHardening:
    """Per-material Ludwik law with ep^n tabulated on graded knots,
    sigma_y = sy0 + K f(ep), f piecewise linear through ep_k^n at
    {0} U logspace(ep_first, ep_max, n_pts-1), perfectly plastic beyond
    ep_max.  Identical to fem_pixel.make_hardening(mode='graded'), the table
    used by the pixel-scale campaign."""

    def __init__(self, sy0, K, n_exp, ep_first=1e-6, ep_max=1.0, n_pts=200):
        self.sy0 = np.asarray(sy0, dtype=np.float64)
        self.K = np.asarray(K, dtype=np.float64)
        if self.sy0.shape != self.K.shape:
            raise ValueError("sy0 and K must have identical shapes")
        if not (0.0 < ep_first < ep_max) or n_pts < 3:
            raise ValueError("graded table needs 0 < ep_first < ep_max, n_pts >= 3")
        self.ep_k = np.concatenate([[0.0], np.logspace(np.log10(ep_first), np.log10(ep_max), n_pts - 1)])
        self.f_k = np.where(self.ep_k > 0, self.ep_k**float(n_exp), 0.0)
        self.sl = np.diff(self.f_k) / np.diff(self.ep_k)
        self.ep_last = float(self.ep_k[-1])

    def sy(self, m, ep):
        ep = np.asarray(ep, dtype=np.float64)
        return self.sy0[m] + self.K[m]*np.interp(np.clip(ep, 0.0, self.ep_last), self.ep_k, self.f_k)

    def H(self, m, ep):
        ep = np.asarray(ep, dtype=np.float64)
        i = np.clip(np.searchsorted(self.ep_k, ep, side="right") - 1, 0, len(self.sl) - 1)
        return self.K[m]*np.where(ep < self.ep_last, self.sl[i], 0.0)


# ── Return mapping ──────────────────────────────────────────────────────────
# A = I + r*CM is block-diagonal (cm13 = cm23 = 0) -> analytical inverse.
def _rm(s_tr, ep0, mat, hard, cm11, cm12, cm33, max_it=50, tol=1e-10):
    """Vectorised radial-return for plane-stress J2 with tabular hardening.

    Returns sigma (N,3), deps_p (N,3), dg (N,) equivalent-plastic increment.
    """
    phi0 = _vm(s_tr) - hard.sy(mat, ep0)
    pl = phi0 > 0
    sig = s_tr.copy()
    dp = np.zeros_like(s_tr)
    dg = np.zeros(len(ep0))
    if not pl.any():
        return sig, dp, dg

    idx = np.where(pl)[0]
    s0 = s_tr[idx]
    e0 = ep0[idx]
    m0 = mat[idx]

    def syp(g):
        return hard.sy(m0, e0 + g)

    def solp(g):
        sk = syp(g)
        r = g / np.where(sk > 1e-30, sk, 1e-30)
        a = 1 + r*cm11
        b = r*cm12
        c = 1 + r*cm33
        d = a*a - b*b
        return np.stack([(a*s0[:, 0] - b*s0[:, 1]) / d,
                         (a*s0[:, 1] - b*s0[:, 0]) / d,
                         s0[:, 2] / c], axis=1)

    def res(g):
        return _vm(solp(g)) - syp(g)

    # Guarded Newton: res(g) is monotone decreasing with res(0) > 0.  Bracket
    # [g_lo, g_hi] first, then Newton with a bisection fallback - unconditionally
    # stable, including in the perfectly-plastic clamp beyond the last table
    # point where H = 0 exactly.
    n_p = len(idx)
    g_lo = np.zeros(n_p)
    g_hi = np.full(n_p, 1e-4)
    for _ in range(80):
        open_ = res(g_hi) > 0
        if not open_.any():
            break
        g_hi = np.where(open_, g_hi*2., g_hi)
    g = 0.5*(g_lo + g_hi)
    for _ in range(max(max_it, 100)):
        r = res(g)
        conv = (np.abs(r) / (syp(g) + 1e-30)).max()
        if conv < tol:
            break
        g_lo = np.where(r > 0, g, g_lo)
        g_hi = np.where(r <= 0, g, g_hi)
        h = np.maximum(np.abs(g), 1e-8) * 1e-7
        dr = (res(g + h) - r) / h
        dr = np.where(np.abs(dr) < 1e-30, -1., dr)
        g_new = g - r/dr
        bad = ~np.isfinite(g_new) | (g_new <= g_lo) | (g_new >= g_hi)
        g = np.where(bad, 0.5*(g_lo + g_hi), g_new)

    sf = solp(g)
    sv = _vm(sf)
    nd = (sf @ _M.T) / np.maximum(sv, 1e-30)[:, None]
    sig[idx] = sf
    dp[idx] = g[:, None] * nd
    dg[idx] = g
    return sig, dp, dg


# ── Consistent tangent ──────────────────────────────────────────────────────
def _cep(sigma, dg, ep0, m0, hard, C_ps, cm11, cm12, cm33):
    """Analytical consistent tangent for the plastic points.

    C_ep = T @ C_ps  with  T = A^-1 - outer(w, p)/(1 + beta),
    w = A^-1 (alpha CM s),  p = A^-1 n,  alpha = (1 - r H)/(sy H).
    """
    sk = hard.sy(m0, ep0 + dg)
    Hk = hard.H(m0, ep0 + dg)
    r = dg / np.where(sk > 1e-30, sk, 1e-30)
    a = 1 + r*cm11
    b = r*cm12
    c = 1 + r*cm33
    d = a*a - b*b
    Hs = np.where(np.abs(Hk) > 1e-30, Hk, 1e-30)
    alpha = (1. - r*Hk) / (sk * Hs)

    s0, s1, s2 = sigma[:, 0], sigma[:, 1], sigma[:, 2]
    q0 = cm11*s0 + cm12*s1
    q1 = cm12*s0 + cm11*s1
    q2 = cm33*s2
    aq0, aq1, aq2 = alpha*q0, alpha*q1, alpha*q2
    w0 = (a*aq0 - b*aq1) / d
    w1 = (-b*aq0 + a*aq1) / d
    w2 = aq2 / c

    sv = _vm(sigma)
    sv_ = np.where(sv > 1e-30, sv, 1e-30)
    n0 = (s0 - .5*s1) / sv_
    n1 = (s1 - .5*s0) / sv_
    n2 = 3*s2 / sv_
    p0 = (a*n0 - b*n1) / d
    p1 = (-b*n0 + a*n1) / d
    p2 = n2 / c
    beta = n0*w0 + n1*w1 + n2*w2
    denom = 1. + beta

    z = np.zeros_like(a)
    Ai = np.stack([np.stack([a/d, -b/d, z], 1),
                   np.stack([-b/d, a/d, z], 1),
                   np.stack([z, z, 1/c], 1)], axis=1)      # (N,3,3)
    wv = np.stack([w0, w1, w2], 1)
    pv = np.stack([p0, p1, p2], 1)
    T = Ai - np.einsum('ni,nj->nij', wv, pv) / denom[:, None, None]
    return np.einsum('nij,jk->nik', T, C_ps)


def _with(u, dof, du):
    """u with `du` added at `dof`, without touching u (line-search trials)."""
    v = u.copy()
    v[dof] += du
    return v


# ── CST kinematics ──────────────────────────────────────────────────────────
def _cst_B(coords, elements):
    """B (n_e, 3, 6) and |A| (n_e,) for constant-strain triangles."""
    p = coords[elements]                       # (n_e, 3, 2)
    x0, x1, x2 = p[:, 0, 0], p[:, 1, 0], p[:, 2, 0]
    y0, y1, y2 = p[:, 0, 1], p[:, 1, 1], p[:, 2, 1]

    two_A = (x1 - x0)*(y2 - y0) - (x2 - x0)*(y1 - y0)      # SIGNED
    if np.any(np.abs(two_A) < 1e-20):
        bad = int(np.sum(np.abs(two_A) < 1e-20))
        raise ValueError(f"{bad} degenerate (zero-area) triangles in the mesh")

    b = np.stack([y1 - y2, y2 - y0, y0 - y1], axis=1) / two_A[:, None]
    c = np.stack([x2 - x1, x0 - x2, x1 - x0], axis=1) / two_A[:, None]

    n_e = len(elements)
    B = np.zeros((n_e, 3, 6))
    for k in range(3):
        B[:, 0, 2*k]     = b[:, k]
        B[:, 1, 2*k + 1] = c[:, k]
        B[:, 2, 2*k]     = c[:, k]
        B[:, 2, 2*k + 1] = b[:, k]
    return B, 0.5*np.abs(two_A)


# ── Main solver ─────────────────────────────────────────────────────────────
def run_fem_tri(coords, elements, mat_id, table, ep_table,
                bc_nodes, bc_u,
                E_mod=205000., nu=0.3, thickness=1.0,
                N_inc=20, max_nr=15, nr_tol=1e-6, line_search=6,
                snapshot_fractions=None, verbose=True,
                hardening_mode="tabular", sy0=None, K=None, n_exp=0.245,
                graded=None):
    """
    coords    (n_node, 2)  mm
    elements  (n_elem, 3)  node array indices
    mat_id    (n_elem,)    grain index into `table`
    table     (n_grain, n_pts)  *Plastic yield stress at `ep_table`
    ep_table  (n_pts,)     uniform, starting at 0
    bc_nodes  (n_bc,)      node indices with BOTH dofs prescribed
    bc_u      (n_bc, 2)    prescribed (ux, uy) in mm

    snapshot_fractions : load fractions in (0, 1]; the fields at those
        pseudo-times are recorded during ONE incremental solve and returned in
        result['frames'] = {fraction: {...}} - no re-solving per load level.
    """
    t0 = time.time()
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    elements = np.ascontiguousarray(elements, dtype=np.int64)
    mat_id = np.ascontiguousarray(mat_id, dtype=np.int64)
    n_node, n_e = len(coords), len(elements)
    n_dof = 2*n_node

    if hardening_mode == "tabular":
        hard = TabularHardening(table, ep_table)
        hardening_note = f"table {table.shape[1]} pts to ep={ep_table[-1]}"
    elif hardening_mode == "analytic":
        if sy0 is None or K is None:
            raise ValueError("analytic hardening requires sy0 and K")
        hard = AnalyticLudwikHardening(sy0, K, n_exp)
        hardening_note = f"analytic Ludwik n={float(n_exp):g}"
    elif hardening_mode == "graded":
        if sy0 is None or K is None:
            raise ValueError("graded hardening requires sy0 and K")
        g = dict(graded or {})
        hard = GradedLudwikHardening(sy0, K, n_exp, **g)
        hardening_note = (f"graded Ludwik table n={float(n_exp):g}, {hard.ep_k.size} knots "
                          f"0,{hard.ep_k[1]:g}..{hard.ep_last:g} log")
    else:
        raise ValueError(f"unknown hardening mode: {hardening_mode}")
    C_ps = _Cps(E_mod, nu)
    CM = C_ps @ _M
    cm11, cm12, cm33 = CM[0, 0], CM[0, 1], CM[2, 2]

    B, area = _cst_B(coords, elements)
    wgt = area * thickness                                  # (n_e,)

    ld = np.zeros((n_e, 6), dtype=np.int64)
    ld[:, 0::2] = 2*elements
    ld[:, 1::2] = 2*elements + 1

    dof_B = np.sort(np.concatenate([2*np.asarray(bc_nodes),
                                    2*np.asarray(bc_nodes) + 1]))
    dof_I = np.setdiff1d(np.arange(n_dof), dof_B)

    _open_log()
    # mesh quality: radius ratio 4*sqrt(3)*A / sum(L^2), 1 = equilateral.
    # Slivers wreck the conditioning of the tangent, so if convergence is bad
    # this line is the first place to look - hence it goes in the log, always.
    p3 = coords[elements]
    L2 = (np.sum((p3[:, 1] - p3[:, 0])**2, 1) +
          np.sum((p3[:, 2] - p3[:, 1])**2, 1) +
          np.sum((p3[:, 0] - p3[:, 2])**2, 1))
    q = 4*np.sqrt(3)*area/np.maximum(L2, 1e-300)
    _trace("\n" + "="*76, verbose)
    _trace(f"[run] fem_tri  {time.strftime('%Y-%m-%d %H:%M:%S')}", verbose)
    _trace(f"[run] mesh      {n_e} CPS3, {n_node} nodes, "
           f"{len(dof_I)} free of {n_dof} DOFs, {table.shape[0]} grains", verbose)
    _trace(f"[run] quality   radius ratio min {q.min():.3f}  "
           f"p1 {np.percentile(q, 1):.3f}  median {np.median(q):.3f}  "
           f"({int((q < 0.05).sum())} slivers q<0.05)", verbose)
    _trace(f"[run] area      min {area.min():.3e}  median {np.median(area):.3e}  "
           f"total {area.sum():.4f} mm2", verbose)
    _trace(f"[run] material  E={E_mod} nu={nu}  {hardening_note}", verbose)
    _trace(f"[run] bcs       {len(bc_nodes)} prescribed nodes  "
           f"ux {bc_u[:,0].min():+.5f}..{bc_u[:,0].max():+.5f}  "
           f"uy {bc_u[:,1].min():+.5f}..{bc_u[:,1].max():+.5f} mm", verbose)
    _trace(f"[run] stepping  N_inc={N_inc} max_nr={max_nr} nr_tol={nr_tol} "
           f"line_search={line_search}", verbose)
    _trace(f"[run] solver    {_SOLVER_NAME}", verbose)
    if _LOG_FH is not None and verbose:
        print(f"[run] log       {_LOG_FH.name}")
    _trace("="*76, verbose)

    _rows = np.repeat(ld, 6, axis=1).ravel()
    _cols = np.tile(ld, (1, 6)).ravel()

    def _assemble(Ke_arr):
        return coo_matrix((Ke_arr.ravel(), (_rows, _cols)),
                          shape=(n_dof, n_dof)).tocsr()

    def _Fint(sig):
        fe = np.einsum('e,eak,ea->ek', wgt, B, sig)
        F = np.zeros(n_dof)
        np.add.at(F, ld, fe)
        return F

    # Elastic stiffness - factorized ONCE, reused for every predictor
    Ke_el = np.einsum('e,eik,ij,ejl->ekl', wgt, B, C_ps, B)
    K_el = _assemble(Ke_el)
    KII_el = K_el[dof_I][:, dof_I].tocsr()
    KIB_el = K_el[dof_I][:, dof_B].tocsr()
    try:
        _lu_el = splu(KII_el.tocsc())
        solve_el = _lu_el.solve
    except Exception:
        def solve_el(b):
            return _solve(KII_el, b)

    u_bc = np.zeros(n_dof)
    u_bc[2*np.asarray(bc_nodes)]     = bc_u[:, 0]
    u_bc[2*np.asarray(bc_nodes) + 1] = bc_u[:, 1]

    # State
    u = np.zeros(n_dof)
    eps_p = np.zeros((n_e, 3))
    ep_bar = np.zeros(n_e)
    sig = np.zeros((n_e, 3))
    eps_tot = np.zeros((n_e, 3))

    def _snap():
        F = _Fint(sig)
        RF = np.zeros((n_node, 2))
        bcm = np.zeros(n_dof, dtype=bool)
        bcm[dof_B] = True
        RF[:, 0] = np.where(bcm[0::2], F[0::2], 0.)
        RF[:, 1] = np.where(bcm[1::2], F[1::2], 0.)
        return dict(U=np.stack([u[0::2], u[1::2]], axis=1),
                    RF=RF, S=sig.copy(), E=eps_tot.copy(),
                    PE=eps_p.copy(), PEEQ=ep_bar.copy())

    area_weight = area/area.sum()
    top_nodes = np.where(np.abs(coords[:, 1] - coords[:, 1].max()) < 1e-9)[0]
    history = {k: [0.0] for k in (
        "fraction", "mean_e22", "mean_evm", "mean_s22", "mean_svm",
        "mean_peeq", "rf2_top")}

    def _record_history(load_fraction):
        e11, e22, gamma12 = eps_tot[:, 0], eps_tot[:, 1], eps_tot[:, 2]
        e12 = 0.5*gamma12
        evm = np.sqrt(np.maximum(
            4.0/(9.0*(1.0 - nu + nu*nu))
            *((e11*e11 + e22*e22 - e11*e22)
              + 3.0*(1.0 - nu)**2*e12*e12), 0.0))
        F = _Fint(sig)
        values = {
            "fraction": float(load_fraction),
            "mean_e22": float(np.sum(area_weight*e22)),
            "mean_evm": float(np.sum(area_weight*evm)),
            "mean_s22": float(np.sum(area_weight*sig[:, 1])),
            "mean_svm": float(np.sum(area_weight*_vm(sig))),
            "mean_peeq": float(np.sum(area_weight*ep_bar)),
            "rf2_top": float(np.sum(F[2*top_nodes + 1])),
        }
        for key, value in values.items():
            history[key].append(value)

    # Incremental loading with automatic cutback (Abaqus-style).  dt_nat is the
    # NATURAL adaptive step; the step actually taken may be clipped short to
    # land exactly on a snapshot fraction, but dt_nat is never overwritten by
    # that clipped value (otherwise the 1.5x growth restarts from the remainder).
    t = 0.
    dt_nat = 1./N_inc
    dt_max = 1./N_inc
    dt_min = dt_max/1024.
    inc = 0
    n_nr = 0; n_cutback = 0; n_grow = 0; n_ls = 0   # logged, also returned
    snaps = {}
    pending = sorted(float(f) for f in snapshot_fractions) if snapshot_fractions else []

    while t < 1. - 1e-12:
        dt = min(dt_nat, 1. - t)
        snap_hit = False
        if pending and pending[0] - t < dt:
            dt = max(pending[0] - t, 1e-14)
            snap_hit = True
        inc += 1
        du_B = u_bc[dof_B]*dt
        u_save = u.copy()

        u[dof_B] += du_B
        u[dof_I] += solve_el(-KIB_el @ du_B)                # elastic predictor

        # State at the current u.  eps_p / ep_bar are frozen for this
        # increment, so this is a pure function of u.
        def _eval(uu):
            eps = np.einsum('eak,ek->ea', B, uu[ld])
            sf, dp, dg = _rm((eps - eps_p) @ C_ps.T, ep_bar, mat_id, hard,
                             cm11, cm12, cm33)
            R_I = _Fint(sf)[dof_I]
            r = np.linalg.norm(R_I)
            return dict(eps=eps, sf=sf, dp=dp, dg=dg, R_I=R_I,
                        res=r if np.isfinite(r) else np.inf)

        st = _eval(u)
        res0 = max(st["res"], 1e-30)
        converged = False
        diverged = False
        res_prev = None

        for nrit in range(max_nr):
            rel = st["res"]/res0
            n_nr += 1
            grew = res_prev is not None and st["res"] > res_prev
            if grew:
                n_grow += 1
            res_prev = st["res"]
            _trace(f"  inc {inc:3d} t={t+dt:.4f} NR {nrit+1:2d} "
                   f"|R|={st['res']:.3e} rel={rel:.3e} "
                   f"plastic={int((st['dg'] > 0).sum())}"
                   + ("  [residual grew]" if grew else ""), verbose)
            if st["res"] < 1e-9 or rel < nr_tol:
                converged = True
                break
            if not np.isfinite(st["res"]):
                diverged = True
                break

            C_ep = np.broadcast_to(C_ps, (n_e, 3, 3)).copy()
            pl_idx = np.where(st["dg"] > 0)[0]
            if len(pl_idx):
                C_ep[pl_idx] = _cep(st["sf"][pl_idx], st["dg"][pl_idx],
                                    ep_bar[pl_idx], mat_id[pl_idx], hard,
                                    C_ps, cm11, cm12, cm33)

            Ke = np.einsum('e,eik,eij,ejl->ekl', wgt, B, C_ep, B)
            KII = _assemble(Ke)[dof_I][:, dof_I].tocsr()
            du = _solve(KII, -st["R_I"])
            if not np.isfinite(du).all():
                diverged = True
                break                                       # singular tangent

            # Backtracking line search.  Full Newton on a step where the whole
            # domain yields at once overshoots badly, and the overshoot grows
            # with the mesh: at ~90k elements it can leave the basin entirely
            # and overflow within max_nr iterations.  Halving the step until
            # the residual actually drops costs one return mapping + one
            # internal-force pass per trial (no re-factorisation) and leaves
            # the quadratic rate untouched whenever s = 1 already works.
            s = 1.0
            trial = None
            for _ in range(line_search):
                cand = _eval(_with(u, dof_I, s*du))
                if cand["res"] < st["res"]:
                    trial = cand
                    break
                s *= 0.5
            if trial is None:
                diverged = True
                break                                       # no descent -> cutback
            u[dof_I] += s*du
            st = trial
            if s < 1.0:
                n_ls += 1
                _trace(f"        line search: step {s:.4f}", verbose)

        if not converged or diverged:
            u = u_save
            dt_nat = 0.5*dt
            n_cutback += 1
            _trace(f"  inc {inc:3d}: no convergence in {max_nr} iters, "
                   f"CUTBACK dt -> {dt_nat:.2e}", verbose)
            if dt_nat < dt_min:
                raise RuntimeError("run_fem_tri: increment cutback below minimum "
                                   f"({dt_nat:.2e}) - solution not converging")
            continue

        eps_p = eps_p + st["dp"]
        ep_bar = ep_bar + st["dg"]
        sig = st["sf"]
        eps_tot = st["eps"]
        t = pending[0] if snap_hit else t + dt
        dt_nat = min(dt_nat*1.5, dt_max)
        _record_history(t)

        while pending and t >= pending[0] - 1e-12:
            fsnap = pending.pop(0)
            snaps[fsnap] = _snap()
            _trace(f"  snapshot recorded at load fraction {fsnap:.4f}", verbose)

    out = _snap()
    out["frames"] = snaps
    out["history"] = {key: np.asarray(value, dtype=np.float64)
                      for key, value in history.items()}
    out["area"] = area
    wall = time.time() - t0
    out.update(n_inc=inc, n_nr=n_nr, n_cutback=n_cutback, n_res_growth=n_grow,
               n_line_search=n_ls, wall_s=wall, log_file=log_path())
    _trace("-"*76, verbose)
    _trace(f"[run] done {wall:.1f}s  increments={inc}  newton_iters={n_nr}  "
           f"cutbacks={n_cutback}  residual_growth={n_grow}  "
           f"line_search={n_ls}", verbose)
    _trace(f"[run] PEEQ_max={ep_bar.max():.6f}  svm_max={_vm(sig).max():.2f} MPa  "
           f"plastic={100.0*(ep_bar > 0).mean():.2f}%", verbose)
    _trace(f"[csv] fem_tri,{n_e},{n_node},{table.shape[0]},{N_inc},{inc},{n_nr},"
           f"{n_cutback},{n_grow},{n_ls},{wall:.2f}", verbose)
    _trace("="*76, verbose)
    return out


# ══════════════════════════════ verification ════════════════════════════════
def _verify():
    """Two checks, both with an exactly known answer.

    1. PATCH TEST (elastic): a random triangulation of a square, all boundary
       nodes driven by a uniform strain field.  A correct CST must reproduce
       that strain in EVERY element to machine precision, whatever the mesh.
    2. EQUAL BIAXIAL TENSION (plastic): homogeneous material, BCs from the
       analytic solution for sigma_vM = 400 MPa.  Same case fem_pixel._verify
       runs, so the two codes are checked against the same number.
    """
    from scipy.spatial import Delaunay

    E, nu = 205000., 0.3
    print("=" * 62)
    print("VERIFICATION 1: elastic patch test, random triangulation")
    print("=" * 62)
    rng = np.random.default_rng(0)
    m = 9
    gx, gy = np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, m), indexing='ij')
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    inner = (pts[:, 0] > 0) & (pts[:, 0] < 1) & (pts[:, 1] > 0) & (pts[:, 1] < 1)
    pts[inner] += rng.uniform(-0.35, 0.35, size=(inner.sum(), 2)) / (m - 1)
    tri = Delaunay(pts)
    elements = tri.simplices.astype(np.int64)

    exx, eyy, gxy = 3.0e-4, -1.1e-4, 7.0e-5
    bnd = np.where(~inner)[0]
    bu = np.stack([exx*pts[bnd, 0] + 0.5*gxy*pts[bnd, 1],
                   0.5*gxy*pts[bnd, 0] + eyy*pts[bnd, 1]], axis=1)

    huge = np.full((1, 50), 1e9)                      # never yields
    r = run_fem_tri(pts, elements, np.zeros(len(elements), dtype=np.int64),
                    huge, np.linspace(0, 0.2, 50), bnd, bu,
                    E_mod=E, nu=nu, N_inc=1, nr_tol=1e-12, verbose=False)
    err = np.abs(r["E"] - np.array([exx, eyy, gxy])).max()
    s_ex = _Cps(E, nu) @ np.array([exx, eyy, gxy])
    serr = np.abs(r["S"] - s_ex).max()
    print(f"  {len(elements)} elements, {len(pts)} nodes, {len(bnd)} constrained")
    print(f"  max |E_elem - E_exact| = {err:.3e}   (exact: {exx:.1e},{eyy:.1e},{gxy:.1e})")
    print(f"  max |S_elem - S_exact| = {serr:.3e} MPa")
    assert err < 1e-13, f"PATCH TEST FAILED: {err:.3e}"
    print("  PASSED")

    print()
    print("=" * 62)
    print("VERIFICATION 2: equal biaxial tension, plastic (vs fem_pixel)")
    print("=" * 62)
    sy0, K, nexp = 250., 500., 0.245
    sig_t = 400.
    ep_ref = ((sig_t - sy0)/K)**(1./nexp)
    eps_e = sig_t*(1. - nu)/E
    eps_tot_val = eps_e + ep_ref/2.

    m = 21
    gx, gy = np.meshgrid(np.linspace(0, 0.02, m), np.linspace(0, 0.02, m), indexing='ij')
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    inner = ((pts[:, 0] > 0) & (pts[:, 0] < 0.02) &
             (pts[:, 1] > 0) & (pts[:, 1] < 0.02))
    tri = Delaunay(pts)
    elements = tri.simplices.astype(np.int64)
    bnd = np.where(~inner)[0]
    bu = eps_tot_val*pts[bnd]

    # Ludwik sampled at the same 50 knots the pipeline writes into the .inp,
    # so this also exercises the tabular interpolation.
    ep_k = np.linspace(0., 0.2, 50)
    tbl = (sy0 + K*np.where(ep_k > 0, ep_k**nexp, 0.))[None, :]
    r = run_fem_tri(pts, elements, np.zeros(len(elements), dtype=np.int64),
                    tbl, ep_k, bnd, bu, E_mod=E, nu=nu,
                    N_inc=30, max_nr=12, nr_tol=1e-9, verbose=False)
    sv = _vm(r["S"]).mean()
    ep = r["PEEQ"].mean()
    print(f"  {len(elements)} elements")
    print(f"  sigma_vm = {sv:.4f} MPa   (target {sig_t:.1f})")
    print(f"  PEEQ     = {ep:.6f}      (Ludwik {ep_ref:.6f}; the tabular table")
    print(f"                            interpolates linearly between knots, so a")
    print(f"                            small offset here is CORRECT, not an error)")
    err = abs(sv - sig_t)/sig_t*100
    print(f"  sigma error = {err:.4f} %")
    assert err < 0.5, f"FAILED: {err:.3f}%"
    print("  PASSED")


if __name__ == "__main__":
    _verify()
