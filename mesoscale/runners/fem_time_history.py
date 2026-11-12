# -*- coding: utf-8 -*-
"""
fem_time_history.py - domain-mean stress / strain / u2 against pseudo-time,
in-house FEM and Abaqus on the same axes.

Both solvers store one value per element per frame, so the domain means are
AREA-WEIGHTED - an unweighted mean over an unstructured mesh silently weights
by triangle count, and the mesh is much finer near grain boundaries.

Three stacked panels sharing the time axis (never a dual axis).  Two black
curves distinguished by dash + marker and direct-labelled, per the thesis
figure rules.

Reads
    results/<mesh>/fem/<fem_tag>_frames.npz
    results/<mesh>/abaqus/<job>_ab_*.npy   (optional; drawn if present)

Writes
    results/<mesh>/compare/time_history_<mesh>.npz
    results/<mesh>/compare/time_history_<mesh>.pdf/.png
"""
import os
import sys
import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
# Repository layout: the solver lives in ../solver, the runners and the
# configuration here. (Before the micro/meso split these sat together under
# fem/1_codes, so ROOT was two levels up - it is kept only for sys.path.)
ROOT = os.path.dirname(CODE_DIR)
SOLVER_DIR = os.path.join(ROOT, "solver")
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, SOLVER_DIR)
sys.path.insert(0, ROOT)

import meso_config as cfg
import inp_io
from _style import fs


def _series(S, E, U, area, bc_mask):
    """Area-weighted domain-mean svm, evm and mean free-node u2 per frame."""
    W = area/area.sum()
    svm = np.array([np.sum(W*cfg.svm_ps(S[k][:, 0], S[k][:, 1], S[k][:, 2]))
                    for k in range(len(S))])
    evm = np.array([np.sum(W*cfg.evm_ps(E[k][:, 0], E[k][:, 2]/2., E[k][:, 1]))
                    for k in range(len(E))])
    u2 = np.array([U[k][bc_mask, 1].mean() for k in range(len(U))])
    return svm, evm, u2


def main():
    prob = inp_io.build_problem(verbose=False)
    area = inp_io.element_areas(prob["coords"], prob["elements"])
    free = np.ones(len(prob["coords"]), dtype=bool)
    free[prob["bc_nodes"]] = False       # mean u2 over FREE nodes = the prediction

    out = {}

    fp = os.path.join(cfg.FEM_OUT, f"{cfg.FEM_TAG}_frames.npz")
    if os.path.isfile(fp):
        d = np.load(fp)
        out["in-house FEM"] = (d["time"],) + _series(d["S"], d["E"], d["U"],
                                                     area, free)
    else:
        print(f"[hist] {fp} not found - run fem_meso_driver.py "
              f"(it snapshots frames during the solve)")

    ap = os.path.join(cfg.AB_OUT, f"{cfg.JOB_NAME}_ab_S.npy")
    if os.path.isfile(ap):
        lab_e = np.load(os.path.join(cfg.AB_OUT, f"{cfg.JOB_NAME}_elem_labels.npy"))
        lab_n = np.load(os.path.join(cfg.AB_OUT, f"{cfg.JOB_NAME}_node_labels.npy"))
        pe = {int(L): i for i, L in enumerate(lab_e)}
        pn = {int(L): i for i, L in enumerate(lab_n)}
        pe = np.array([pe[int(L)] for L in prob["elem_ids"]])
        pn = np.array([pn[int(L)] for L in prob["node_ids"]])
        S = np.load(ap, mmap_mode="r")
        E = np.load(os.path.join(cfg.AB_OUT, f"{cfg.JOB_NAME}_ab_E.npy"), mmap_mode="r")
        U = np.load(os.path.join(cfg.AB_OUT, f"{cfg.JOB_NAME}_ab_U.npy"), mmap_mode="r")
        t = np.load(os.path.join(cfg.AB_OUT, f"{cfg.JOB_NAME}_frametimes.npy"))
        Sp = np.stack([np.asarray(S[k])[pe] for k in range(S.shape[0])])
        Ep = np.stack([np.asarray(E[k])[pe] for k in range(E.shape[0])])
        Up = np.stack([np.asarray(U[k])[pn] for k in range(U.shape[0])])
        out["Abaqus"] = (t,) + _series(Sp, Ep, Up, area, free)
    else:
        print(f"[hist] Abaqus export not found - FEM curve only")

    if not out:
        raise SystemExit("nothing to plot")

    np.savez(os.path.join(cfg.CMP_OUT, f"time_history_{cfg.MESH_KEY}.npz"),
             **{f"{k.replace(' ', '_')}_{n}": v
                for k, tup in out.items()
                for n, v in zip(("time", "svm", "evm", "u2"), tup)})

    for k, (t, svm, evm, u2) in out.items():
        print(f"[hist] {k:14s} final  svm {svm[-1]:9.3f} MPa   "
              f"evm {100*evm[-1]:8.5f} %   u2 {u2[-1]:.5e} mm   ({len(t)} frames)")
    if len(out) == 2:
        a, b = out["Abaqus"], out["in-house FEM"]
        if len(a[0]) == len(b[0]) and np.allclose(a[0], b[0], atol=1e-9):
            for n, i in (("svm", 1), ("evm", 2), ("u2", 3)):
                d = np.abs(np.asarray(a[i]) - np.asarray(b[i]))
                sc = np.sqrt(np.mean(np.asarray(a[i])**2))
                print(f"[hist] {n:>4} max|diff| over frames = {d.max():.5e}  "
                      f"({100*d.max()/sc if sc else float('nan'):.4f} % of rms)")
        else:
            print("[hist] frame times differ between the two - curves are "
                  "plotted but not differenced. Re-run fem_meso_driver.py "
                  "AFTER the Abaqus export so it snapshots the same times.")

    _plot(out)


def _plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs.apply()
    s = getattr(fs, "CANVAS_SCALE", 1.0)

    fig, axes = plt.subplots(3, 1, figsize=(5.2*s, 8.4*s), sharex=True,
                             constrained_layout=True)
    order = [k for k in ("Abaqus", "in-house FEM") if k in out]
    ylabels = [r"$\sigma_{vM}$ (MPa)", r"$\varepsilon_{vM}$ (%)", r"$u_2$ (mm)"]
    scales = [1.0, 100.0, 1.0]

    for row in range(3):
        ax = axes[row]
        for i, k in enumerate(order):
            t = out[k][0]
            y = np.asarray(out[k][row + 1])*scales[row]
            ax.plot(t, y, **fs.line(i))
        ax.set_ylabel(ylabels[row])
        fs.data_axes(ax, nticks=5)
    axes[-1].set_xlabel("pseudo-time")

    # Direct labels at the right end of every panel - no legend box.  The axes
    # share x, so the room for the labels is made ONCE.  Verifying a solver
    # means the two curves land on top of each other, which is exactly when
    # labels collide: they are nudged apart vertically when that happens, and
    # the offset is symmetric so neither curve is implied to be the reference.
    lo, hi = axes[0].get_xlim()
    axes[0].set_xlim(lo, hi + 0.34*(hi - lo))

    for row in range(3):
        ax = axes[row]
        ys = [np.asarray(out[k][row + 1])[-1]*scales[row] for k in order]
        y0, y1 = ax.get_ylim()
        span = (y1 - y0) or 1.0
        if len(ys) > 1 and abs(ys[0] - ys[1]) < 0.06*span:
            dys = [+0.045*span, -0.045*span]
        else:
            dys = [0.0]*len(ys)
        for i, k in enumerate(order):
            t = np.asarray(out[k][0])
            y = np.asarray(out[k][row + 1])*scales[row]
            fs.label_line(ax, t, y, k, side="right", dy=dys[i])

    for i, ax in enumerate(axes):
        try:
            fs.panel_label(ax, i)
        except Exception:
            pass

    stem = f"time_history_{cfg.MESH_KEY}"
    try:
        fs.save(fig, cfg.CMP_OUT, stem, formats=("pdf", "png"))
    except TypeError:
        fs.save(fig, cfg.CMP_OUT, stem)
    plt.close(fig)
    print(f"[hist] {os.path.join(cfg.CMP_OUT, stem)}.pdf")


if __name__ == "__main__":
    main()
