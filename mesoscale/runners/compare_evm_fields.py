# -*- coding: utf-8 -*-
"""
compare_evm_fields.py - DIC | Abaqus | in-house FEM field maps, mesoscale.

Mesoscale twin of Microscale/.../test_10x10/compare_evm_fields.py.  The two
solvers are rasterised onto the DIC pixel grid by fem_to_grid.py
(same routine for both), the DIC field is computed on that same grid, and all
panels share one colour scale and the mesh mask, so the three are directly
comparable pixel by pixel.

DIC    : U_<frame>/V_<frame> from the locked correlation setting's disflow
         tree (cfg.dic_u_path / cfg.dic_v_path), strains by np.gradient in
         pixel units, baseline (frames 1..DIC_BASELINE_N) subtracted.
Abaqus : grid_abaqus_{e11,e22,e12}.npy   (last frame)
FEM    : grid_fem_{e11,e22,e12}.npy      (last frame)

Outputs in results/<mesh>/compare:
    evm_fields_dic_abaqus_fem_<mesh>.pdf/.png
    u2_fields_dic_abaqus_fem_<mesh>.pdf/.png
    field_report_<mesh>.txt
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import meso_config as cfg

CMAP = os.environ.get("MESO_CMAP", "jet")   # field maps may use colour


# ── DIC ─────────────────────────────────────────────────────────────────────
def _pad_to(a, nr, nc):
    pr = max(0, nr - a.shape[0])
    pc = max(0, nc - a.shape[1])
    if pr or pc:
        a = np.pad(a, ((0, pr), (0, pc)), mode="edge")
    return a


def load_uv(k, nx, ny):
    """RAW orientation, no transpose: axis0 = mesh x, axis1 = mesh y.
    U carries the loading direction (mesh y), V the transverse (mesh x) -
    the same convention mesoscale_runner.py uses to write the BCs."""
    U = np.load(cfg.dic_u_path(k)).astype(float)
    V = np.load(cfg.dic_v_path(k)).astype(float)
    U = _pad_to(U, nx, ny)[:nx, :ny]
    V = _pad_to(V, nx, ny)[:nx, :ny]
    return U, V


def dic_strains(U, V):
    """exx, exy(TENSORIAL), eyy at pixel nodes; spacing = 1 px, dimensionless."""
    dU0, dU1 = np.gradient(U)
    dV0, dV1 = np.gradient(V)
    exx = dV0                      # d(u_x)/dx
    eyy = dU1                      # d(u_y)/dy
    exy = 0.5*(dV1 + dU0)
    return exx, exy, eyy


def _panel_grid(fields, mask, title_fmt, cbar_label, stem, scale=1.0,
                cmap=CMAP, vlim=None):
    n = len(fields)
    fig, axes = plt.subplots(1, n, figsize=(7.0*n, 6.6), constrained_layout=True)
    axes = np.atleast_1d(axes)
    if vlim is None:
        # robust limits on BOTH ends: pinning vmin at 0 for a field that never
        # goes near 0 throws away half the colour range and flattens the map
        vmax = max(np.nanpercentile(f[np.isfinite(f)], 99.5) for _, f in fields)*scale
        vmin = min(np.nanpercentile(f[np.isfinite(f)], 0.5) for _, f in fields)*scale
    else:
        vmin, vmax = vlim
    im = None
    for ax, (t, f) in zip(axes, fields):
        g = np.where(mask, f, np.nan)*scale
        im = ax.imshow(g.T, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(title_fmt.format(t), fontsize=19)
        ax.set_xlabel("x (px)", fontsize=14)
        ax.set_ylabel("y (px)", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.text(0.02, 0.98, f"mean = {np.nanmean(g):.4f}", transform=ax.transAxes,
                fontsize=13, va="top", ha="left",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    cb = fig.colorbar(im, ax=list(axes), shrink=0.85)
    cb.set_label(cbar_label, fontsize=16)
    cb.ax.tick_params(labelsize=12)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(cfg.CMP_OUT, f"{stem}.{ext}"), dpi=180)
    plt.close(fig)
    print(f"[fig ] {stem}.pdf/.png in {cfg.CMP_OUT}")


def _last(a):
    return a[-1] if a.ndim == 3 else a


def _grid(src, name):
    p = os.path.join(cfg.CMP_OUT, f"grid_{src}_{name}.npy")
    return _last(np.load(p)) if os.path.isfile(p) else None


def main():
    report = []

    have = {}
    for src in ("abaqus", "fem"):
        e = [_grid(src, k) for k in ("e11", "e22", "e12")]
        if all(x is not None for x in e):
            have[src] = e
    if not have:
        raise SystemExit("no rasterised fields found - run "
                         "fem_to_grid.py first")

    ref = next(iter(have.values()))[0]
    nx, ny = ref.shape
    mask = np.isfinite(ref)
    print(f"[fig ] pixel grid {nx} x {ny}, mesh covers "
          f"{100.0*mask.mean():.1f} % of it")

    # ── DIC ─────────────────────────────────────────────────────────────────
    base, n_base = None, 0
    for k in range(1, cfg.DIC_BASELINE_N + 1):
        if not (os.path.isfile(cfg.dic_u_path(k)) and
                os.path.isfile(cfg.dic_v_path(k))):
            continue
        U, V = load_uv(k, nx, ny)
        s = np.stack(dic_strains(U, V))
        base = s if base is None else base + s
        n_base += 1
    base = base/n_base if n_base else 0.0
    U, V = load_uv(cfg.DIC_FINAL_FRAME, nx, ny)
    exx_d, exy_d, eyy_d = dic_strains(U, V)
    if n_base:
        exx_d, exy_d, eyy_d = exx_d - base[0], exy_d - base[1], eyy_d - base[2]
        report.append(f"DIC baseline: mean of frames 1..{n_base} subtracted")
    evm_dic = cfg.evm_ps(exx_d, exy_d, eyy_d)

    # ── eps_vM panels ───────────────────────────────────────────────────────
    fields = [("DIC", evm_dic)]
    for src, label in (("abaqus", "Abaqus"), ("fem", "in-house FEM")):
        if src in have:
            e11, e22, e12 = have[src]
            fields.append((label, cfg.evm_ps(e11, e12/2.0, e22)))   # e12 is ENGINEERING
    _panel_grid(fields, mask, "{}", r"$\varepsilon_{vM}$ (%)",
                f"evm_fields_dic_abaqus_fem_{cfg.MESH_KEY}", scale=100.0)

    report.append("")
    report.append("eps_vM over the meshed region (%):")
    for t, f in fields:
        g = np.where(mask, f, np.nan)
        report.append(f"  {t:14s} mean {100*np.nanmean(g):9.5f}   "
                      f"p99.5 {100*np.nanpercentile(g[np.isfinite(g)], 99.5):9.5f}   "
                      f"max {100*np.nanmax(g):9.5f}")
    d = dict(fields)
    if "Abaqus" in d and "in-house FEM" in d:
        dd = np.where(mask, np.abs(d["Abaqus"] - d["in-house FEM"]), np.nan)
        report.append(f"  |Abaqus - FEM|  mean {100*np.nanmean(dd):.6f} %   "
                      f"max {100*np.nanmax(dd):.6f} %")

    # ── u2 panels ───────────────────────────────────────────────────────────
    u2_dic = _pad_to(np.load(cfg.dic_u_path(cfg.DIC_FINAL_FRAME)).astype(float),
                     nx, ny)[:nx, :ny]*cfg.PX_TO_MM
    u2 = [("DIC", u2_dic)]
    for src, label in (("abaqus", "Abaqus"), ("fem", "in-house FEM")):
        g = _grid(src, "u2")
        if g is not None:
            u2.append((label, g))
    if len(u2) > 1:
        vmin = min(np.nanmin(np.where(mask, f, np.nan)) for _, f in u2)
        vmax = max(np.nanmax(np.where(mask, f, np.nan)) for _, f in u2)
        _panel_grid(u2, mask, "{}  $u_2$", r"$u_2$ (mm)",
                    f"u2_fields_dic_abaqus_fem_{cfg.MESH_KEY}",
                    vlim=(vmin, vmax))
        du = dict(u2)
        report.append("")
        report.append("u2 over the meshed region (mm):")
        for t, f in u2:
            g = np.where(mask, f, np.nan)
            report.append(f"  {t:14s} mean {np.nanmean(g):12.6e}   "
                          f"min {np.nanmin(g):12.6e}   max {np.nanmax(g):12.6e}")
        if "Abaqus" in du and "in-house FEM" in du:
            dd = np.where(mask, np.abs(du["Abaqus"] - du["in-house FEM"]), np.nan)
            report.append(f"  |Abaqus - FEM| max {np.nanmax(dd):.4e} mm  "
                          f"rms {np.sqrt(np.nanmean(dd**2)):.4e} mm")
        for t in ("Abaqus", "in-house FEM"):
            if t in du:
                dd = np.where(mask, np.abs(du[t] - u2_dic), np.nan)
                report.append(f"  |{t} - DIC| max {np.nanmax(dd):.4e} mm  "
                              f"rms {np.sqrt(np.nanmean(dd**2)):.4e} mm")

    txt = "\n".join(report)
    print("\n" + txt)
    p = os.path.join(cfg.CMP_OUT, f"field_report_{cfg.MESH_KEY}.txt")
    with open(p, "w") as f:
        f.write(txt + "\n")
    print(f"\n[fig ] {p}")


if __name__ == "__main__":
    main()
