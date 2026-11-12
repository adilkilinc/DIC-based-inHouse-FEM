# -*- coding: utf-8 -*-
"""Run one Dataset 2 mesoscale case (locked meso solver, settings of the Dataset 1 cases).

    python run_case.py --case id_mt [--dry-run] [--threads 8]
    python run_case.py --index 2            (position in cases.json, for Slurm arrays)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

import common as C


def material(case, pb):
    mat_id = np.asarray(pb["mat_id"], np.int64)
    law = case["law"]
    if law == "pixel":
        return pb["sy0_pixel"], pb["K_pixel"], np.arange(len(pb["elements"]), dtype=np.int64)
    key = {"homogeneous": ("sy0_homogeneous", "K_homogeneous"), "dic": ("sy0_dic", "K_dic"),
           "dic_merged": ("sy0_dic_merged", "K_dic_merged"), "hp_kmacro": ("sy0_hp", "K_hp_macro"),
           "hp_kdic": ("sy0_hp", "K_hp_hybrid")}[law]
    return pb[key[0]], pb[key[1]], mat_id


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--case")
    g.add_argument("--index", type=int)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow-scipy", action="store_true", help="local tests only")
    ap.add_argument("--problems", default=str(C.PROBLEMS))
    ap.add_argument("--results", default=str(C.RESULTS))
    a = ap.parse_args()
    st = C.settings()
    pk = C.package()
    cases = pk["cases"]
    case = cases[a.index] if a.index is not None else {c["id"]: c for c in cases}.get(a.case)
    if case is None:
        raise SystemExit(f"unknown case {a.case}")
    table = st["tables"][case.get("table", st["table"])]
    n_inc = int(case.get("n_inc", st["n_inc"]))
    pf = Path(a.problems) / f"{case['mesh']}.npz"
    out = Path(a.results) / case["id"]
    mp = out / "run_meta.json"
    if mp.is_file() and not a.force and json.loads(mp.read_text()).get("completed_ok"):
        print(f"completed result exists: {mp}")
        return
    if not pf.is_file():
        raise SystemExit(f"missing problem {pf} (run build_problems.py)")
    with np.load(pf) as z:
        pb = {k: z[k] for k in z.files}
    coords = np.asarray(pb["coords"], np.float64)
    elements = np.asarray(pb["elements"], np.int64)
    bc_nodes = np.asarray(pb["bc_nodes"], np.int64)
    bc_u = np.asarray(pb["bc_u_dic" if case["bc"] == "dic" else "bc_u_edge_mean"], np.float64)
    sy0, K, mat_id = material(case, pb)
    sy0 = np.asarray(sy0, np.float64)
    K = np.asarray(K, np.float64)
    print(f"[prepared] {case['id']}: {len(coords):,} nodes, {len(elements):,} CPS3, {len(sy0):,} curves; "
          f"sy0 {sy0.min():.1f}..{sy0.max():.1f} (mean {sy0.mean():.1f}); K {K.min():.1f}..{K.max():.1f} MPa; "
          f"bc {case['bc']}; table {table}; n_inc {n_inc}", flush=True)
    if a.dry_run:
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    os.environ.update(FEM_LOG_DIR=str(out / "logs"), OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads),
                      OPENBLAS_NUM_THREADS="1", MKL_DYNAMIC="FALSE")
    sys.path.insert(0, str(C.SOLVER))
    import fem_tri
    if "pypardiso" not in fem_tri._SOLVER_NAME.lower() and not a.allow_scipy:
        raise SystemExit(f"production requires pypardiso, found {fem_tri._SOLVER_NAME}")
    t0 = time.time()
    r = fem_tri.run_fem_tri(coords, elements, mat_id, np.zeros((1, 2)), np.array([0.0, 1.0]), bc_nodes, bc_u,
                            E_mod=st["E"], nu=st["nu"], N_inc=n_inc, max_nr=st["max_nr"], nr_tol=st["nr_tol"],
                            snapshot_fractions=st["snapshot_fractions"], verbose=True,
                            hardening_mode="graded", sy0=sy0, K=K, n_exp=st["n_exp"],
                            graded=dict(ep_first=float(table["ep_first"]), ep_max=float(table["ep_max"]),
                                        n_pts=int(table["n_table"])))
    wall = time.time() - t0
    nu = st["nu"]
    for k in ("U", "RF", "S", "E", "PE", "PEEQ"):
        np.save(out / f"{k}.npy", np.asarray(r[k], np.float32))
    fr = np.array(sorted(r["frames"]), np.float64)
    np.savez_compressed(out / "snapshots.npz", fraction=fr,
                        E=np.stack([r["frames"][f]["E"] for f in fr]).astype(np.float32),
                        S=np.stack([r["frames"][f]["S"] for f in fr]).astype(np.float32),
                        PEEQ=np.stack([r["frames"][f]["PEEQ"] for f in fr]).astype(np.float32))
    h = r["history"]
    np.savez(out / "history_scalars.npz", **h)
    with (out / "history_scalars.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(list(h))
        w.writerows(zip(*(h[k] for k in h)))
    tri = coords[elements]
    area = 0.5 * np.abs((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
                        - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1]))
    w = area / area.sum()
    E = np.asarray(r["E"], np.float64)
    S = np.asarray(r["S"], np.float64)
    peeq_max = float(np.max(r["PEEQ"]))
    meta = dict(schema=1, package=pk["package"], case=case, table=table, n_inc_requested=n_inc,
                n_inc_accepted=int(r["n_inc"]), newton_iterations=int(r["n_nr"]), cutbacks=int(r["n_cutback"]),
                line_searches=int(r.get("n_line_search", 0)), nodes=int(len(coords)), elements=int(len(elements)),
                material_curves=int(len(sy0)), solver_wall_s=float(r["wall_s"]), total_wall_s=float(wall),
                final_mean_e22=float(np.sum(w * E[:, 1])), final_mean_evm=float(np.sum(w * C.evm_voigt(E, nu))),
                evm_definition="thesis Eq. (evmps), printed", final_mean_s22_mpa=float(np.sum(w * S[:, 1])),
                final_mean_svm_mpa=float(np.sum(w * C.vm_stress(S))), final_mean_peeq=float(np.sum(w * r["PEEQ"])),
                peeq_max=peeq_max, peeq_table_fraction=peeq_max / float(table["ep_max"]),
                rf2_top=float(h["rf2_top"][-1]) if "rf2_top" in h else float("nan"),
                boundary_error_max_mm=float(np.max(np.abs(r["U"][bc_nodes] - bc_u))),
                material_mean_yield_area_weighted=float(np.sum(w * sy0[mat_id])),
                material_mean_K_area_weighted=float(np.sum(w * K[mat_id])),
                inputs={"problem": str(pf)}, input_sha256={"problem": C.sha256(pf)},
                solver_sha256=C.sha256(C.SOLVER / "fem_tri.py"), solver=fem_tri._SOLVER_NAME,
                hostname=socket.gethostname(), threads=a.threads, slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                completed=time.strftime("%Y-%m-%d %H:%M:%S"), completed_ok=True)
    tmp = mp.with_name(mp.name + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2) + "\n")
    os.replace(tmp, mp)
    print(f"[complete] {mp}")


if __name__ == "__main__":
    main()
