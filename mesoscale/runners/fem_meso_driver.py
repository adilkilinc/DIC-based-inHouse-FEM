# -*- coding: utf-8 -*-
"""
fem_meso_driver.py - solve the mesoscale full-domain problem with fem_tri.

Same problem definition as the Abaqus job: both this script and
../abaqus/gen_inp_meso.py call inp_io.build_problem(), so mesh, per-grain
*Plastic tables and prescribed boundary displacements are identical by
construction.

Snapshot load fractions are taken from the Abaqus frame times when the ODB has
already been exported (AB_OUT/<job>_frametimes.npy), so FEM and Abaqus frames
line up one-for-one for the time-history comparison.  Otherwise an even
1/N_FRAMES ladder is used.

Outputs, in results/<mesh>/fem:
    <tag>_U.npy      (n_node, 2)   final nodal displacement, mm
    <tag>_RF.npy     (n_node, 2)   final reaction force at prescribed dofs
    <tag>_S.npy      (n_elem, 3)   s11, s22, s12   MPa
    <tag>_E.npy      (n_elem, 3)   e11, e22, gamma12
    <tag>_PE.npy     (n_elem, 3)
    <tag>_PEEQ.npy   (n_elem,)
    <tag>_labels.npz node_ids / elem_ids, so fields can be matched to the ODB
    <tag>_frames.npz per-frame element + nodal fields and the frame times
"""
import os
import sys
import time
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
from fem_tri import run_fem_tri, _vm

N_INC     = int(os.environ.get("MESO_N_INC", 20))
N_FRAMES  = int(os.environ.get("MESO_N_FRAMES", 20))
MAX_NR    = int(os.environ.get("MESO_MAX_NR", 15))
NR_TOL    = float(os.environ.get("MESO_NR_TOL", 1e-6))
FORCE     = os.environ.get("MESO_FORCE_FEM", "0") == "1"

TAG = cfg.FEM_TAG
KEYS = ("U", "RF", "S", "E", "PE", "PEEQ")


def _out(name):
    return os.path.join(cfg.FEM_OUT, f"{TAG}_{name}.npy")


def _cached(n_node, n_elem):
    """Reuse an existing solve when it matches the current mesh."""
    if FORCE:
        return None
    d = {}
    for k in KEYS:
        p = _out(k)
        if not os.path.isfile(p):
            return None
        d[k] = np.load(p)
    if d["S"].shape[0] != n_elem or d["U"].shape[0] != n_node:
        return None
    return d


def snapshot_fractions():
    """Abaqus frame times if exported, else an even ladder."""
    p = os.path.join(cfg.AB_OUT, f"{cfg.JOB_NAME}_frametimes.npy")
    if os.path.isfile(p):
        ft = np.load(p).astype(float)
        ft = ft[(ft > 1e-12) & (ft <= 1.0 + 1e-12)]
        ft = np.clip(np.unique(np.round(ft, 12)), 0., 1.)
        if len(ft):
            print(f"[driver] snapshotting at the {len(ft)} Abaqus frame times "
                  f"({p})")
            return list(ft)
    print(f"[driver] no Abaqus frame times yet - even ladder of {N_FRAMES}")
    return list(np.linspace(1.0/N_FRAMES, 1.0, N_FRAMES))


def main():
    banner = getattr(cfg, "variant_banner", None)
    if banner is not None:
        print(banner())
    prob = inp_io.build_problem(verbose=True)
    n_node = len(prob["coords"])
    n_elem = len(prob["elements"])

    # sy0 / Kgrain are ADDITIVE keys (2026-08-23): the case-matrix figures plot
    # the per-grain material the solve actually used, and re-deriving it later
    # means re-running the whole preprocessing just to draw a histogram.
    # Nothing that reads this file by key is affected.
    np.savez(os.path.join(cfg.FEM_OUT, f"{TAG}_labels.npz"),
             node_ids=prob["node_ids"], elem_ids=prob["elem_ids"],
             mat_id=prob["mat_id"], coords=prob["coords"],
             elements=prob["elements"],
             sy0=prob["sy0"], Kgrain=prob["K"])

    res = _cached(n_node, n_elem)
    if res is not None:
        print(f"[driver] reusing saved fields (prefix {TAG}) - no re-solve. "
              f"Set MESO_FORCE_FEM=1 to force.")
    else:
        fr = snapshot_fractions()
        t0 = time.time()
        res = run_fem_tri(prob["coords"], prob["elements"], prob["mat_id"],
                          prob["table"], cfg.EP_TABLE,
                          prob["bc_nodes"], prob["bc_u"],
                          E_mod=cfg.E_MOD, nu=cfg.NU,
                          N_inc=N_INC, max_nr=MAX_NR, nr_tol=NR_TOL,
                          snapshot_fractions=fr, verbose=True)
        wall = time.time() - t0

        for k in KEYS:
            np.save(_out(k), res[k])

        frames = res["frames"]
        ft = np.array(sorted(frames))
        if len(ft):
            np.savez_compressed(
                os.path.join(cfg.FEM_OUT, f"{TAG}_frames.npz"),
                time=ft,
                S=np.stack([frames[f]["S"] for f in ft]),
                E=np.stack([frames[f]["E"] for f in ft]),
                PE=np.stack([frames[f]["PE"] for f in ft]),
                PEEQ=np.stack([frames[f]["PEEQ"] for f in ft]),
                U=np.stack([frames[f]["U"] for f in ft]),
                RF=np.stack([frames[f]["RF"] for f in ft]))
        with open(os.path.join(cfg.FEM_OUT, f"{TAG}_solve_report.txt"), "w") as f:
            f.write(f"mesh        : {cfg.MESH_KEY}\n")
            f.write(f"nodes/elems : {n_node} / {n_elem}\n")
            f.write(f"grains      : {prob['table'].shape[0]}\n")
            f.write(f"N_inc       : {N_INC} (actual increments {res['n_inc']})\n")
            f.write(f"frames      : {len(ft)}\n")
            f.write(f"wall time   : {wall:.1f} s ({wall/60:.2f} min)\n")
        print(f"[driver] fields saved with prefix {TAG} in {cfg.FEM_OUT}")

    # ── summary ─────────────────────────────────────────────────────────────
    S, E, PEEQ = res["S"], res["E"], res["PEEQ"]
    a = prob.get("area")
    svm = _vm(S)
    evm = cfg.evm_ps(E[:, 0], E[:, 2]/2.0, E[:, 1])
    print("\n=== in-house FEM, final state ===")
    print(f"  svm   mean {svm.mean():10.3f}  max {svm.max():10.3f} MPa")
    print(f"  evm   mean {100*evm.mean():10.5f}  max {100*evm.max():10.5f} %")
    print(f"  PEEQ  mean {PEEQ.mean():10.6f}  max {PEEQ.max():10.6f}")
    print(f"  plastic elements: {int((PEEQ > 0).sum())} / {n_elem} "
          f"({100.0*(PEEQ > 0).mean():.1f} %)")
    print(f"  u   x {res['U'][:,0].min():+.5f}..{res['U'][:,0].max():+.5f}  "
          f"y {res['U'][:,1].min():+.5f}..{res['U'][:,1].max():+.5f} mm")
    return res


if __name__ == "__main__":
    main()
