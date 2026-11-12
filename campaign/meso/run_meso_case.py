"""Run one Dataset 1 mesoscale case of the final campaign.

Geometry and boundary data: 0_inputs/problems/<mesh>.npz (frozen).
Material: 1_materials/<mesh>.npz (build_materials.py, from the method-B maps).
Hardening: graded Ludwik table of the pixel-scale campaign (fem_tri 'graded').

  python meso/run_meso_case.py --case id_ft [--dry-run] [--threads 16]
"""
import argparse, csv, hashlib, json, os, socket, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOLVER = ROOT.parent / "mesoscale" / "solver"
sys.path.insert(0, str(SOLVER))


def sha256(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def vm_stress(S):
    return np.sqrt(np.maximum(S[:, 0]**2 - S[:, 0]*S[:, 1] + S[:, 1]**2 + 3*S[:, 2]**2, 0.0))


def vm_strain(E, nu):
    e11, e22, e12 = E[:, 0], E[:, 1], 0.5*E[:, 2]
    return np.sqrt(np.maximum(4.0/(9.0*(1.0 - nu + nu*nu))*((e11*e11 + e22*e22 - e11*e22) + 3.0*(1.0 - nu)**2*e12*e12), 0.0))


def meso_cases(cfg):
    return {c["id"]: c for c in cfg["meso"]["cases"]}


def material(case, problem, mats):
    mat_id = np.asarray(problem["mat_id"], np.int64)
    if case["law"] == "pixel":
        return mats["sy0_pixel"], mats["K_pixel"], np.arange(len(problem["elements"]), dtype=np.int64)
    if case["space"] == "homog":
        return mats["sy0_homogeneous"], mats["K_homogeneous"], mat_id
    key = {"dic": ("sy0_dic", "K_dic"), "hp_kmacro": ("sy0_hp", "K_hp_macro"),
           "hp_kdic": ("sy0_hp", "K_hp_hybrid")}[case["law"]]
    return mats[key[0]], mats[key[1]], mat_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow-scipy", action="store_true", help="local tests only")
    a = ap.parse_args()
    cfg = json.loads((ROOT / "campaign.json").read_text(encoding="utf-8"))
    mc = cfg["meso"]; cases = meso_cases(cfg)
    if a.case not in cases:
        raise SystemExit(f"unknown meso case {a.case}")
    case = cases[a.case]
    table = cfg["hardening_tables"][case.get("table", mc["table"])]
    n_inc = int(case.get("n_inc", mc["n_inc"]))
    pf = HERE / "0_inputs" / "problems" / f"{case['mesh']}.npz"
    mf = HERE / "1_materials" / f"{case['mesh']}.npz"
    out = HERE / "2_results" / a.case
    mp = out / "run_meta.json"
    if mp.is_file() and not a.force and json.loads(mp.read_text()).get("completed_ok"):
        print(f"completed result exists: {mp}"); return
    for p in (pf, mf):
        if not p.is_file():
            raise SystemExit(f"missing input {p}")
    with np.load(pf) as z:
        problem = {k: z[k] for k in z.files}
    with np.load(mf) as z:
        mats = {k: z[k] for k in z.files}
    coords = np.asarray(problem["coords"], np.float64)
    elements = np.asarray(problem["elements"], np.int64)
    bc_nodes = np.asarray(problem["bc_nodes"], np.int64)
    bc_u = np.asarray(problem["bc_u_dic" if case["bc"] == "dic" else "bc_u_edge_mean"], np.float64)
    sy0, K, mat_id = material(case, problem, mats)
    sy0 = np.asarray(sy0, np.float64); K = np.asarray(K, np.float64)
    print(f"[prepared] {a.case}: {len(coords):,} nodes, {len(elements):,} CPS3, {len(sy0):,} curves; "
          f"sy0 {sy0.min():.1f}..{sy0.max():.1f} (mean {sy0.mean():.1f}); K {K.min():.1f}..{K.max():.1f} MPa; "
          f"bc {case['bc']}; table {table}; n_inc {n_inc}", flush=True)
    if a.dry_run:
        return
    out.mkdir(parents=True, exist_ok=True)
    os.environ.update(FEM_LOG_DIR=str(out / "logs"), OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads),
                      OPENBLAS_NUM_THREADS="1", MKL_DYNAMIC="FALSE")
    (out / "logs").mkdir(exist_ok=True)
    import fem_tri
    if "pypardiso" not in fem_tri._SOLVER_NAME.lower() and not a.allow_scipy:
        raise SystemExit(f"production requires pypardiso, found {fem_tri._SOLVER_NAME}")
    t0 = time.time()
    r = fem_tri.run_fem_tri(coords, elements, mat_id, np.zeros((1, 2)), np.array([0.0, 1.0]), bc_nodes, bc_u,
                            E_mod=float(mc["E_mpa"]), nu=float(mc["nu"]), N_inc=n_inc,
                            max_nr=int(mc["max_nr"]), nr_tol=float(mc["nr_tol"]),
                            snapshot_fractions=[float(x) for x in mc["snapshot_fractions"]], verbose=True,
                            hardening_mode="graded", sy0=sy0, K=K, n_exp=float(mc["n_exp"]),
                            graded=dict(ep_first=float(table["ep_first"]), ep_max=float(table["ep_max"]),
                                        n_pts=int(table["n_table"])))
    wall = time.time() - t0
    nu = float(mc["nu"])
    for k in ("U", "RF", "S", "E", "PE", "PEEQ"):
        np.save(out / f"{k}.npy", np.asarray(r[k], np.float32))
    fr = np.array(sorted(r["frames"]), np.float64)
    np.savez_compressed(out / "snapshots.npz", fraction=fr,
                        E=np.stack([r["frames"][f]["E"] for f in fr]).astype(np.float32),
                        S=np.stack([r["frames"][f]["S"] for f in fr]).astype(np.float32),
                        PEEQ=np.stack([r["frames"][f]["PEEQ"] for f in fr]).astype(np.float32),
                        U=np.stack([r["frames"][f]["U"] for f in fr]).astype(np.float32),
                        RF=np.stack([r["frames"][f]["RF"] for f in fr]).astype(np.float32))
    h = r["history"]
    np.savez(out / "history_scalars.npz", **h)
    with (out / "history_scalars.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(list(h)); w.writerows(zip(*(h[k] for k in h)))
    tri = coords[elements]
    area = 0.5*np.abs((tri[:, 1, 0] - tri[:, 0, 0])*(tri[:, 2, 1] - tri[:, 0, 1])
                      - (tri[:, 2, 0] - tri[:, 0, 0])*(tri[:, 1, 1] - tri[:, 0, 1]))
    w = area/area.sum()
    peeq_max = float(np.max(r["PEEQ"]))
    meta = dict(schema=1, campaign=cfg["campaign"], case=case, table=table, n_inc_requested=n_inc,
                n_inc_accepted=int(r["n_inc"]), newton_iterations=int(r["n_nr"]), cutbacks=int(r["n_cutback"]),
                line_searches=int(r.get("n_line_search", 0)), nodes=int(len(coords)), elements=int(len(elements)),
                material_curves=int(len(sy0)), solver_wall_s=float(r["wall_s"]), total_wall_s=float(wall),
                final_mean_e22=float(np.sum(w*r["E"][:, 1])), final_mean_evm=float(np.sum(w*vm_strain(r["E"], nu))),
                final_mean_s22_mpa=float(np.sum(w*r["S"][:, 1])), final_mean_svm_mpa=float(np.sum(w*vm_stress(r["S"]))),
                final_mean_peeq=float(np.sum(w*r["PEEQ"])), peeq_max=peeq_max,
                peeq_table_fraction=peeq_max/float(table["ep_max"]),
                rf2_top=float(h["rf2_top"][-1]) if "rf2_top" in h else float("nan"),
                boundary_error_max_mm=float(np.max(np.abs(r["U"][bc_nodes] - bc_u))),
                material_mean_yield_area_weighted=float(np.sum(w*sy0[mat_id])),
                material_mean_K_area_weighted=float(np.sum(w*K[mat_id])),
                inputs={"problem": str(pf), "materials": str(mf)},
                input_sha256={"problem": sha256(pf), "materials": sha256(mf)},
                solver_sha256=sha256(SOLVER / "fem_tri.py"), solver=fem_tri._SOLVER_NAME,
                hostname=socket.gethostname(), threads=a.threads, slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                completed=time.strftime("%Y-%m-%d %H:%M:%S"), completed_ok=True)
    tmp = mp.with_name(mp.name + ".tmp"); tmp.write_text(json.dumps(meta, indent=2) + "\n"); os.replace(tmp, mp)
    print(f"[complete] {mp}")


if __name__ == "__main__":
    main()
