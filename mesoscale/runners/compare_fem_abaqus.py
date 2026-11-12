# -*- coding: utf-8 -*-
"""
compare_fem_abaqus.py - ELEMENT-LEVEL in-house-FEM vs Abaqus comparison.

This is the primary verification of the in-house solver.  Both sides solve the
same .inp on the same mesh, so the comparison is done element-for-element and
node-for-node, matched by Abaqus LABEL - no interpolation, no rasterisation,
nothing that could hide or invent a difference.

Reads
    results/<mesh>/fem/<fem_tag>_{S,E,PE,PEEQ,U,RF}.npy      (fem_meso_driver)
    results/<mesh>/abaqus/<job>_ab_*.npy + *_labels.npy       (odb_process_meso)

Writes
    results/<mesh>/compare/diff_table_<mesh>.txt
    results/<mesh>/compare/worst_grains_<mesh>.csv
    results/<mesh>/compare/fem_vs_abaqus_<mesh>.pdf
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import meso_config as cfg
import inp_io
from _style import fs

TOP_N = int(os.environ.get("MESO_TOP_N", 15))


def _perm(mesh_labels, odb_labels, what):
    pos = {int(L): i for i, L in enumerate(odb_labels)}
    try:
        return np.array([pos[int(L)] for L in mesh_labels], dtype=np.int64)
    except KeyError as ex:
        raise RuntimeError(f"{what} label {ex} present in the mesh but not in "
                           f"the ODB export - .inp and .odb disagree.")


def _stats(a, b, w=None):
    """max|diff|, rms(diff), rms(reference), relative error in %.

    `w` (element areas) makes the rms area-weighted, so a dense patch of tiny
    triangles cannot dominate the number.
    """
    d = np.asarray(a) - np.asarray(b)
    if w is None:
        rms_d = float(np.sqrt(np.nanmean(d**2)))
        rms_r = float(np.sqrt(np.nanmean(np.asarray(b)**2)))
    else:
        W = w/np.nansum(w)
        rms_d = float(np.sqrt(np.nansum(W*d**2)))
        rms_r = float(np.sqrt(np.nansum(W*np.asarray(b)**2)))
    mx = float(np.nanmax(np.abs(d)))
    rel = 100.0*rms_d/rms_r if rms_r > 0 else float("nan")
    return mx, rms_d, rms_r, rel


def main():
    prob = inp_io.build_problem(verbose=True)
    area = None

    # ── in-house FEM ────────────────────────────────────────────────────────
    need = ("S", "E", "PE", "PEEQ", "U", "RF")
    fem = {}
    for k in need:
        p = os.path.join(cfg.FEM_OUT, f"{cfg.FEM_TAG}_{k}.npy")
        if not os.path.isfile(p):
            raise SystemExit(f"missing {p}\nRun fem_meso_driver.py first.")
        fem[k] = np.load(p)

    # ── Abaqus, last frame, reordered into mesh order ───────────────────────
    ap = os.path.join(cfg.AB_OUT, f"{cfg.JOB_NAME}_ab_S.npy")
    if not os.path.isfile(ap):
        raise SystemExit(f"missing {ap}\nRun ../abaqus/odb_process_meso.py first.")
    lab_e = np.load(os.path.join(cfg.AB_OUT, f"{cfg.JOB_NAME}_elem_labels.npy"))
    lab_n = np.load(os.path.join(cfg.AB_OUT, f"{cfg.JOB_NAME}_node_labels.npy"))
    pe = _perm(prob["elem_ids"], lab_e, "element")
    pn = _perm(prob["node_ids"], lab_n, "node")

    ab = {}
    for k in need:
        p = os.path.join(cfg.AB_OUT, f"{cfg.JOB_NAME}_ab_{k}.npy")
        if not os.path.isfile(p):
            print(f"[cmp ] {os.path.basename(p)} not exported - skipping {k}")
            continue
        a = np.load(p, mmap_mode="r")[-1]
        ab[k] = np.asarray(a)[pe] if k in ("S", "E", "PE", "PEEQ") else np.asarray(a)[pn]
    times = np.load(os.path.join(cfg.AB_OUT, f"{cfg.JOB_NAME}_frametimes.npy"))

    area = inp_io.element_areas(prob["coords"], prob["elements"])

    # ── diff table ──────────────────────────────────────────────────────────
    rows = []
    comp = [("s11", "S", 0, "MPa"), ("s22", "S", 1, "MPa"), ("s12", "S", 2, "MPa"),
            ("e11", "E", 0, "-"),   ("e22", "E", 1, "-"),   ("g12", "E", 2, "-"),
            ("pe11", "PE", 0, "-"), ("pe22", "PE", 1, "-"), ("gp12", "PE", 2, "-")]
    for name, key, c, unit in comp:
        if key not in ab:
            continue
        rows.append((name, unit) + _stats(fem[key][:, c], ab[key][:, c], area))
    if "PEEQ" in ab:
        rows.append(("peeq", "-") + _stats(fem["PEEQ"], ab["PEEQ"], area))
    svm_f = cfg.svm_ps(fem["S"][:, 0], fem["S"][:, 1], fem["S"][:, 2])
    svm_a = cfg.svm_ps(ab["S"][:, 0], ab["S"][:, 1], ab["S"][:, 2])
    rows.append(("svm", "MPa") + _stats(svm_f, svm_a, area))
    evm_f = cfg.evm_ps(fem["E"][:, 0], fem["E"][:, 2]/2., fem["E"][:, 1])
    evm_a = cfg.evm_ps(ab["E"][:, 0], ab["E"][:, 2]/2., ab["E"][:, 1])
    rows.append(("evm", "-") + _stats(evm_f, evm_a, area))
    for name, key, c in (("u1", "U", 0), ("u2", "U", 1)):
        if "U" in ab:
            rows.append((name, "mm") + _stats(fem["U"][:, c], ab["U"][:, c]))

    hdr = (f"{'field':>6} {'unit':>5} {'max|diff|':>13} {'rms(diff)':>13} "
           f"{'rms(abaqus)':>13} {'rel %':>9}")
    lines = [f"mesh        : {cfg.MESH_KEY}   job {cfg.JOB_NAME}",
             f"elements    : {len(prob['elements'])}   nodes {len(prob['coords'])}"
             f"   grains {prob['table'].shape[0]}",
             f"abaqus frame: last of {len(times)}  (t = {times[-1]:.4f})",
             "",
             "element/nodal values matched by Abaqus label; rms is AREA-WEIGHTED",
             "for element fields so dense patches of small triangles do not",
             "dominate. rel % = rms(diff) / rms(abaqus).",
             "",
             hdr, "-"*len(hdr)]
    for r in rows:
        lines.append(f"{r[0]:>6} {r[1]:>5} {r[2]:13.5e} {r[3]:13.5e} "
                     f"{r[4]:13.5e} {r[5]:9.4f}")

    # boundary sanity: prescribed dofs must agree to round-off in both codes
    if "U" in ab:
        bn = prob["bc_nodes"]
        db = np.abs(fem["U"][bn] - ab["U"][bn]).max()
        di = np.abs(fem["U"] - ab["U"])
        mask = np.ones(len(fem["U"]), dtype=bool)
        mask[bn] = False
        lines += ["",
                  f"u at PRESCRIBED nodes : max|diff| = {db:.4e} mm "
                  f"(must be ~0 - both codes are given the same BCs)",
                  f"u at FREE nodes       : max|diff| = {di[mask].max():.4e} mm  "
                  f"rms = {np.sqrt((di[mask]**2).mean()):.4e} mm"]

    # domain means (area-weighted) - fixed by the BCs, so these must agree
    W = area/area.sum()
    lines += ["",
              "area-weighted domain means (set by the boundary displacements):",
              f"  {'':10s} {'FEM':>14s} {'Abaqus':>14s}"]
    for nm, fv, av in (("svm (MPa)", svm_f, svm_a), ("evm (%)", 100*evm_f, 100*evm_a),
                       ("PEEQ", fem["PEEQ"], ab.get("PEEQ", fem["PEEQ"]*np.nan)),
                       ("e11", fem["E"][:, 0], ab["E"][:, 0]),
                       ("e22", fem["E"][:, 1], ab["E"][:, 1])):
        lines.append(f"  {nm:10s} {np.nansum(W*fv):14.6f} {np.nansum(W*av):14.6f}")

    txt = "\n".join(lines)
    print("\n" + txt)
    out = os.path.join(cfg.CMP_OUT, f"diff_table_{cfg.MESH_KEY}.txt")
    with open(out, "w") as f:
        f.write(txt + "\n")
    print(f"\n[cmp ] {out}")

    # ── worst grains ────────────────────────────────────────────────────────
    mat = prob["mat_id"]
    ng = prob["table"].shape[0]
    d_svm = np.abs(svm_f - svm_a)
    num = np.bincount(mat, weights=d_svm*area, minlength=ng)
    den = np.bincount(mat, weights=area, minlength=ng)
    g_err = num/np.maximum(den, 1e-30)
    g_svm = np.bincount(mat, weights=svm_a*area, minlength=ng)/np.maximum(den, 1e-30)
    g_pq = np.bincount(mat, weights=ab.get("PEEQ", fem["PEEQ"])*area,
                       minlength=ng)/np.maximum(den, 1e-30)
    order = np.argsort(-g_err)
    csv = os.path.join(cfg.CMP_OUT, f"worst_grains_{cfg.MESH_KEY}.csv")
    with open(csv, "w") as f:
        f.write("rank,grain,face,n_elem,area_mm2,mean_svm_abaqus_MPa,"
                "mean_abs_dsvm_MPa,rel_pct,mean_peeq,sy0_MPa,K_MPa\n")
        for r, g in enumerate(order):
            f.write(f"{r+1},{g},{prob['grain_names'][g]},"
                    f"{int((mat == g).sum())},{den[g]:.6e},{g_svm[g]:.4f},"
                    f"{g_err[g]:.4f},{100*g_err[g]/max(g_svm[g], 1e-30):.4f},"
                    f"{g_pq[g]:.6f},{prob['sy0'][g]:.3f},{prob['K'][g]:.3f}\n")
    print(f"[cmp ] {csv}")
    print(f"\nworst {TOP_N} grains by area-weighted |d svm|:")
    print(f"  {'grain':>6} {'face':>10} {'svm_ab':>10} {'|dsvm|':>10} {'rel%':>8} {'PEEQ':>10}")
    for g in order[:TOP_N]:
        print(f"  {g:>6} {prob['grain_names'][g]:>10} {g_svm[g]:10.3f} "
              f"{g_err[g]:10.4f} {100*g_err[g]/max(g_svm[g],1e-30):8.3f} {g_pq[g]:10.5f}")

    _figure(prob, area, svm_f, svm_a, evm_f, evm_a, fem, ab, g_err, g_svm)


def _figure(prob, area, svm_f, svm_a, evm_f, evm_a, fem, ab, g_err, g_svm):
    """Four diagnostic panels. Correlation clouds and a difference histogram -
    monochrome, per the thesis line-chart rule; colour is reserved for the
    field maps in compare_evm_fields.py."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs.apply()
    s = getattr(fs, "CANVAS_SCALE", 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(6.69*s, 6.0*s),
                             constrained_layout=True)

    def cloud(ax, x, y, xl, yl):
        lo = float(min(np.nanmin(x), np.nanmin(y)))
        hi = float(max(np.nanmax(x), np.nanmax(y)))
        ax.plot([lo, hi], [lo, hi], color="0.55", ls=(0, (1, 2)), lw=0.6*s, zorder=0)
        ax.plot(x, y, ls="none", marker=".", ms=1.2*s, color="k",
                alpha=0.25, rasterized=True)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        fs.data_axes(ax, nticks=5)

    cloud(axes[0, 0], svm_a, svm_f, r"Abaqus  $\sigma_{vM}$ (MPa)",
          r"in-house  $\sigma_{vM}$ (MPa)")
    cloud(axes[0, 1], 100*evm_a, 100*evm_f, r"Abaqus  $\varepsilon_{vM}$ (%)",
          r"in-house  $\varepsilon_{vM}$ (%)")

    ax = axes[1, 0]
    d = svm_f - svm_a
    lim = float(np.percentile(np.abs(d[np.isfinite(d)]), 99.5)) or 1.0
    ax.hist(np.clip(d, -lim, lim), bins=80, weights=area,
            color="0.75", edgecolor="k", lw=0.6*s)
    ax.axvline(0, color="0.55", ls=(0, (1, 2)), lw=0.6*s)
    ax.set_xlabel(r"$\sigma_{vM}$: in-house $-$ Abaqus (MPa)")
    ax.set_ylabel("frequency (area-weighted)")
    fs.data_axes(ax, nticks=5)

    ax = axes[1, 1]
    ax.plot(g_svm, g_err, ls="none", marker="o", ms=2.5*s, color="k",
            mfc="white", mew=0.7*s)
    ax.set_xlabel(r"grain mean $\sigma_{vM}$, Abaqus (MPa)")
    ax.set_ylabel(r"grain mean $|\Delta\sigma_{vM}|$ (MPa)")
    fs.data_axes(ax, nticks=5)

    for i, ax in enumerate(axes.ravel()):
        try:
            fs.panel_label(ax, i)
        except Exception:
            pass

    stem = f"fem_vs_abaqus_{cfg.MESH_KEY}"
    try:
        fs.save(fig, cfg.CMP_OUT, stem, formats=("pdf", "png"))
    except TypeError:
        fs.save(fig, cfg.CMP_OUT, stem)
    plt.close(fig)
    print(f"[cmp ] {os.path.join(cfg.CMP_OUT, stem)}.pdf")


if __name__ == "__main__":
    main()
