"""Run one case of the final locked campaign (graded Ludwik table, method B).

Method B (Dataset 1 and Dataset 2): elastic line per pixel fitted against the
machine stress over the frames carrying 0.11-0.80 sigma0, onset where the strain
exceeds it by 0.02 %, yield = machine stress at the onset frame,
K = SLOP_REF / plastic slope (bounds [0, 3], neighbour fill), K in MPa =
coefficient x macroscopic Ludwik K.

Method-B maps of both datasets are read from this campaign's own
0_identification/<dataset>/<key>/<variant> folders (written by
identification/d1_identify.py and d2_identify.py).  Material
variants used by the sensitivity checks are built from those maps here, at
run time, and every input file is hashed into run_meta.json.
"""
from __future__ import annotations

import argparse, contextlib, csv, hashlib, importlib.util, json, os, re, socket, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def config():
    return json.loads((HERE / "campaign.json").read_text(encoding="utf-8"))


def cases(cfg):
    out = {}
    for fam in cfg["case_families"]:
        for m in fam["materials"]:
            for f in fam["factors"]:
                cid = f"{fam['dataset']}_{m}_f{int(f):02d}" + (f"_{fam['suffix']}" if fam.get("suffix") else "")
                if cid in out:
                    raise ValueError(f"duplicate case id {cid}")
                geo = cfg["field_geometry"][fam["dataset"]]
                out[cid] = dict(fam, material=m, factor=int(f), case_id=cid, bc=fam.get("bc", "biaxial"),
                                field_shape=geo["field_shape"], crop_offset=geo["crop_offset"])
    return out


def id_key_of(cfg, case):
    return cfg["datasets"][case["dataset"]]["materials"][case["material"]].get("id_key")


def fractions_for(case):
    return [i / int(case["n_inc"]) for i in range(1, int(case["n_inc"]) + 1)]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def root_for(dataset, cfg, explicit=None):
    if explicit:
        return Path(explicit).resolve()
    info = cfg["roots"][dataset]
    env = os.environ.get(info["env"])
    if env:
        return Path(env).resolve()
    return Path(info["hpc"] if os.name != "nt" else info["local"])


def fill_nan_4nbr(a, name):
    a = np.asarray(a, dtype=np.float64).copy()
    if not np.isnan(a).any():
        return a
    print(f"[input] {name}: filling {int(np.isnan(a).sum())} NaNs", flush=True)
    for _ in range(max(a.shape)):
        bad = np.isnan(a)
        if not bad.any():
            break
        tot = np.zeros_like(a); cnt = np.zeros(a.shape, np.uint8)
        for q in (np.roll(a, 1, 0), np.roll(a, -1, 0), np.roll(a, 1, 1), np.roll(a, -1, 1)):
            g = np.isfinite(q); tot[g] += q[g]; cnt[g] += 1
        use = bad & (cnt > 0); a[use] = tot[use] / cnt[use]
    if np.isnan(a).any():
        raise ValueError(f"{name}: NaNs remain")
    return a


def block_mean(a, f):
    if f == 1:
        return np.asarray(a, np.float64)
    nx, ny = a.shape
    if nx % f or ny % f:
        raise ValueError(f"factor {f} does not divide {a.shape}")
    return a.reshape(nx // f, f, ny // f, f).mean(axis=(1, 3))


def material_paths(cfg, case, root):
    """Files that define the material of this case (hashed into run_meta)."""
    ds = case["dataset"]; d = cfg["datasets"][ds]; mat = d["materials"][case["material"]]
    p = {}
    if ds == "dataset1":
        p["U"], p["V"] = root / d["U"], root / d["V"]
        if "uv_key" in mat:
            # alpha sweep: boundary data from the same DIC solution as the maps
            n = int(cfg["identification"]["dataset1"]["final_frame"])
            k = mat["uv_key"]
            p["U"] = root / "2_data/disflow" / k / "U" / f"U_{n}.npy"
            p["V"] = root / "2_data/disflow" / k / "V" / f"V_{n}.npy"
    else:
        key = mat["source_key"]
        n = int(cfg["identification"]["dataset2"]["final_frame"])
        p["U"] = root / f"disflow/{key}/U/U_{n}.npy"
        p["V"] = root / f"disflow/{key}/V/V_{n}.npy"
    if "id_key" in mat and "yield_constant_mpa" not in mat:
        base = id_base(cfg, ds, mat)
        p["yield"] = base / "yield_macro_filled.npy"
        p["K_ratio"] = base / "K_coef_filled.npy"
        if "censored_yield_mpa" in mat:
            p["onset"] = base / "S_p.npy"
            p["frame_stress"] = base.parent / "frame_stress.npy"
        if mat.get("K_flagged_to_median") or "K_upper_bound" in mat:
            p["K_flag"] = base / "K_flag.npy"
        if "K_upper_bound" in mat:
            p["K_raw"] = base / "K_coef_raw.npy"
    return p


def id_base(cfg, ds, mat):
    """Method-B maps written by this campaign's identification jobs."""
    idc = cfg["identification"]
    return HERE / idc["out_dir"] / ds / mat["id_key"] / mat.get("variant_dir", idc["variant_dir"])


def build_material(cfg, case, paths, shape):
    """Yield (MPa) and K ratio maps on the source grid, with variant transforms."""
    mat = cfg["datasets"][case["dataset"]]["materials"][case["material"]]
    notes = {}
    if "yield_constant_mpa" in mat:
        Y = np.full(shape, float(mat["yield_constant_mpa"]))
        K = np.full(shape, float(mat["K_constant_mpa"]) / float(cfg["datasets"][case["dataset"]]["K_scale_mpa"]))
        return Y, K, notes
    Y = fill_nan_4nbr(np.load(paths["yield"], mmap_mode="r"), "yield")
    K = fill_nan_4nbr(np.load(paths["K_ratio"], mmap_mode="r"), "K_ratio")
    if "censored_yield_mpa" in mat:
        S = np.load(paths["onset"])
        n_frames = int(np.load(paths["frame_stress"]).size)
        # the offset kernel returns index N-1 when the offset is never crossed
        # (same convention as onset_last_frac in diagnostics.json)
        cens = np.isfinite(S) & (S >= n_frames - 1)
        Y[cens] = float(mat["censored_yield_mpa"])
        notes["censored_fraction"] = float(cens.mean())
        notes["censored_yield_mpa"] = float(mat["censored_yield_mpa"])
    if mat.get("K_flagged_to_median"):
        flag = np.load(paths["K_flag"])
        med = float(np.median(K[flag == 0]))
        K[flag > 0] = med
        notes["K_flagged_fraction"] = float((flag > 0).mean())
        notes["K_flagged_value"] = med
    if "K_upper_bound" in mat:
        flag = np.load(paths["K_flag"]); raw = np.load(paths["K_raw"])
        ub = float(mat["K_upper_bound"])
        above = (flag == 1)
        K[above] = np.minimum(raw[above], ub)
        notes["K_above3_fraction"] = float(above.mean())
        notes["K_above_new_bound_fraction"] = float((above & (raw > ub)).mean())
    return Y, K, notes


def load_inputs(cfg, case, root, smoke=False):
    d = cfg["datasets"][case["dataset"]]
    src = tuple(d["source_shape"]); shape = tuple(case["field_shape"]); ox, oy = map(int, case["crop_offset"]); f = case["factor"]
    paths = material_paths(cfg, case, root)
    missing = [str(p) for p in paths.values() if not Path(p).is_file()]
    if missing:
        raise SystemExit("missing inputs:\n  " + "\n  ".join(missing))
    U = np.load(paths["U"], mmap_mode="r"); V = np.load(paths["V"], mmap_mode="r")
    if U.shape != src or V.shape != src:
        raise ValueError(f"displacement shape {U.shape}, expected {src}")
    Y, K, notes = build_material(cfg, case, paths, src)
    if Y.shape != src or K.shape != src:
        raise ValueError("material shape mismatch")
    Y = Y[ox:ox + shape[0], oy:oy + shape[1]]
    K = K[ox:ox + shape[0], oy:oy + shape[1]] * float(d["K_scale_mpa"])
    sc = float(d["pixel_size_um"]) / 1000
    dx = (np.pad(np.asarray(V, np.float64), ((0, 1), (0, 1)), mode="edge") * sc)[ox:ox + shape[0] + 1, oy:oy + shape[1] + 1]
    dy = (np.pad(np.asarray(U, np.float64), ((0, 1), (0, 1)), mode="edge") * sc)[ox:ox + shape[0] + 1, oy:oy + shape[1] + 1]
    if smoke:
        n = int(cfg["smoke_coarse_elements"]) * f
        sx = ((shape[0] - n) // (2 * f)) * f; sy = ((shape[1] - n) // (2 * f)) * f
        Y, K = Y[sx:sx + n, sy:sy + n], K[sx:sx + n, sy:sy + n]
        dx, dy = dx[sx:sx + n + 1, sy:sy + n + 1], dy[sx:sx + n + 1, sy:sy + n + 1]
    Yc, Kc = block_mean(Y, f), block_mean(K, f)
    dxc, dyc = np.asarray(dx[::f, ::f]), np.asarray(dy[::f, ::f])
    if dxc.shape != (Yc.shape[0] + 1, Yc.shape[1] + 1):
        raise AssertionError((dxc.shape, Yc.shape))
    return dxc, dyc, Yc, Kc, paths, notes


def import_solver():
    p = REPO / "microscale/solver/fem_pixel.py"
    spec = importlib.util.spec_from_file_location("final_fem_pixel", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m, p


# ---- boundary-condition cases (same DOF sets as BC_study/1_codes/fem_bc.py) ----
def boundary_dofs(node_ids, mode):
    nyn = node_ids.shape[1]
    if mode == "biaxial":
        nodes = np.concatenate([node_ids[:, 0], node_ids[:, -1], node_ids[0, 1:-1], node_ids[-1, 1:-1]])
        dofs = np.union1d(2 * nodes, 2 * nodes + 1)
    elif mode == "semiaxial":
        nodes = np.concatenate([node_ids[:, 0], node_ids[:, -1]])
        dofs = np.union1d(2 * nodes, 2 * nodes + 1)
    elif mode == "uniaxial":
        nodes = np.concatenate([node_ids[:, 0], node_ids[:, -1]])
        dofs = np.union1d(2 * nodes + 1, np.array([2 * node_ids[0, nyn // 2]]))
    else:
        raise ValueError(mode)
    return np.unique(dofs).astype(int)


@contextlib.contextmanager
def bc_mode(solver, mode):
    if mode == "biaxial":
        yield; return
    base = solver._Mesh

    class _MeshBC(base):
        def __init__(self, xs, ys, el, sf):
            base.__init__(self, xs, ys, el, sf)
            d = boundary_dofs(self.node_ids, mode)
            self.dofs_bc = d
            self.dofs_free = np.setdiff1d(np.arange(self.n_dof), d).astype(int)
    solver._Mesh = _MeshBC
    try:
        yield
    finally:
        solver._Mesh = base


def vm_stress(S):
    return np.sqrt(np.maximum(S[..., 0]**2 - S[..., 0]*S[..., 1] + S[..., 1]**2 + 3*S[..., 2]**2, 0))


def vm_strain(E, nu):
    x, y, g = E[..., 0], E[..., 1], E[..., 2] / 2
    return np.sqrt(np.maximum(4/(9*(1-nu+nu**2))*((x*x+y*y-x*y)+3*(1-nu)**2*g*g), 0))


def history(frames, fractions, nu):
    keys = ["mean_s11", "mean_s22", "mean_s12", "mean_svm", "mean_e11", "mean_e22", "mean_e12", "mean_evm",
            "mean_peeq", "rf1_left", "rf1_right", "rf2_bottom", "rf2_top"]
    out = {k: [0.0] for k in keys}
    for f in fractions:
        q = frames[f]; S, E, RF = q["S"], q["E"], q["RF"]
        v = {"mean_s11": S[..., 0].mean(), "mean_s22": S[..., 1].mean(), "mean_s12": S[..., 2].mean(),
             "mean_svm": vm_stress(S).mean(), "mean_e11": E[..., 0].mean(), "mean_e22": E[..., 1].mean(),
             "mean_e12": E[..., 2].mean(), "mean_evm": vm_strain(E, nu).mean(), "mean_peeq": q["PEEQ"].mean(),
             "rf1_left": RF[0, :, 0].sum(), "rf1_right": RF[-1, :, 0].sum(), "rf2_bottom": RF[:, 0, 1].sum(),
             "rf2_top": RF[:, -1, 1].sum()}
        for k in keys:
            out[k].append(float(v[k]))
    out["fraction"] = [0.0] + list(fractions)
    return {k: np.asarray(v) for k, v in out.items()}


def atomic_npy(p, v):
    t = p.with_name(p.name + ".tmp.npy"); np.save(t, np.asarray(v, np.float32)); os.replace(t, p)


def atomic_npz(p, d):
    t = p.with_name(p.name + ".tmp.npz")
    with t.open("wb") as fh:
        np.savez(fh, **d)
    os.replace(t, p)


def save_outputs(r, out, tag, fractions, snapshots, nu):
    fields = {"u1": r["U"][..., 0], "u2": r["U"][..., 1], "rf1": r["RF"][..., 0], "rf2": r["RF"][..., 1],
              "s11": r["S"][..., 0], "s22": r["S"][..., 1], "s12": r["S"][..., 2],
              "e11": r["E"][..., 0], "e22": r["E"][..., 1], "e12": r["E"][..., 2],
              "pe11": r["PE"][..., 0], "pe22": r["PE"][..., 1], "pe12": r["PE"][..., 2], "peeq": r["PEEQ"]}
    paths = {}
    for k, v in fields.items():
        p = out / f"{tag}_{k}.npy"; atomic_npy(p, v); paths[k] = str(p)
    h = history(r["frames"], fractions, nu); atomic_npz(out / "history_scalars.npz", h)
    names = ["fraction"] + [k for k in h if k != "fraction"]
    tmp = out / "history_scalars.csv.tmp"
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(names)
        for i in range(len(h["fraction"])):
            w.writerow([f"{float(h[k][i]):.12g}" for k in names])
    os.replace(tmp, out / "history_scalars.csv")
    sd = out / "snapshots"; sd.mkdir(exist_ok=True); sp = []
    for f in snapshots:
        q = r["frames"][f]; p = sd / f"fields_f{round(100*f):03d}.npz"
        atomic_npz(p, {"fraction": np.float64(f), "u2": np.asarray(q["U"][..., 1], np.float32),
                       "e22": np.asarray(q["E"][..., 1], np.float32), "evm": np.asarray(vm_strain(q["E"], nu), np.float32),
                       "peeq": np.asarray(q["PEEQ"], np.float32), "s22": np.asarray(q["S"][..., 1], np.float32),
                       "svm": np.asarray(vm_stress(q["S"]), np.float32)})
        sp.append(str(p))
    return paths, sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--old-root"); ap.add_argument("--new-root")
    ap.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = config(); m = cases(cfg)
    if a.case not in m:
        raise SystemExit(f"unknown case {a.case}")
    case = m[a.case]; ds = case["dataset"]; d = cfg["datasets"][ds]
    root = root_for(ds, cfg, a.old_root if ds == "dataset1" else a.new_root)
    lock = HERE / "input_lock.json"
    if not lock.is_file():
        raise SystemExit("input_lock.json missing; run preflight.py --write-lock")
    if json.loads(lock.read_text())["campaign_config_sha256"] != sha256(HERE / "campaign.json"):
        raise SystemExit("campaign.json changed after the input lock")
    out = HERE / ("_smoke" if a.smoke else "2_data") / a.case; out.mkdir(parents=True, exist_ok=True)
    mp = out / "run_meta.json"
    if mp.is_file() and not a.force:
        if json.loads(mp.read_text()).get("completed_ok"):
            print(f"completed result exists: {mp}"); return
        raise SystemExit(f"incomplete output exists: {mp}; inspect or use --force")
    dx, dy, Y, K, paths, notes = load_inputs(cfg, case, root, a.smoke)
    print(f"[prepared] {a.case}: elements={Y.shape} bc={case['bc']} yield={Y.mean():.4g}+/-{Y.std():.4g} "
          f"K={K.mean():.4g}+/-{K.std():.4g} {notes}", flush=True)
    if a.dry_run:
        return
    solver, sp = import_solver()
    if "pypardiso" not in solver._SOLVER_NAME.lower() and not a.smoke:
        raise SystemExit(f"production requires pypardiso; got {solver._SOLVER_NAME}")
    os.environ.update(MKL_NUM_THREADS=str(a.threads), OMP_NUM_THREADS=str(a.threads), OPENBLAS_NUM_THREADS="1",
                      MKL_DYNAMIC="FALSE", FEM_LOG_DIR=str(HERE / ("_smoke/logs" if a.smoke else "logs") / a.case))
    Path(os.environ["FEM_LOG_DIR"]).mkdir(parents=True, exist_ok=True)
    tab = cfg["hardening_tables"][case["table"]]; fr = fractions_for(case)
    nx, ny = Y.shape; pitch = float(d["element_size_base"]) * case["factor"]; t0 = time.time()
    with bc_mode(solver, case["bc"]):
        r = solver.run_fem(dx, dy, Y, K, float(d["n_exp"]), nx * pitch, ny * pitch, pitch, float(d["pixel_size_um"]),
                           E_mod=float(d["E_mpa"]), nu=float(d["nu"]), N_inc=int(case["n_inc"]),
                           max_nr=int(cfg["max_nr"]), nr_tol=float(cfg["nr_tol"]), hardening=tab["mode"],
                           ep_table_max=float(tab["ep_max"]), n_table=int(tab["n_table"]),
                           ep_table_first=float(tab["ep_first"]), snapshot_fractions=fr, verbose=True,
                           **({"ref_force_tol": float(case["ref_force_tol"])} if case.get("ref_force_tol") else {}))
    outputs, snaps = save_outputs(r, out, a.case, fr, [float(x) for x in cfg["snapshot_fractions"]], float(d["nu"]))
    peeq = float("nan")
    try:
        v = re.findall(r"PEEQ_max=([0-9.eE+-]+)", Path(r["log_file"]).read_text(errors="replace")); peeq = float(v[-1]) if v else peeq
    except Exception:
        pass
    meta = {"schema": 2, "campaign": cfg["campaign"], "case_id": a.case, "stage": case["stage"], "dataset": ds,
            "material": case["material"], "material_definition": d["materials"][case["material"]], "material_notes": notes,
            "factor": case["factor"], "bc": case["bc"], "table": case["table"], "hardening": dict(tab),
            "n_inc": case["n_inc"], "max_nr": cfg["max_nr"], "nr_tol": cfg["nr_tol"],
            "ref_force_tol": float(case.get("ref_force_tol", 0.0)),
            "field_shape": case["field_shape"], "crop_offset": case["crop_offset"], "element_shape": list(Y.shape),
            "elements": int(Y.size), "E_mpa": d["E_mpa"], "nu": d["nu"], "K_scale_mpa": d["K_scale_mpa"], "n_exp": d["n_exp"],
            "prepared": {"yield_mean_mpa": float(Y.mean()), "yield_std_mpa": float(Y.std()),
                         "K_mean_mpa": float(K.mean()), "K_std_mpa": float(K.std())},
            "input_sha256": {k: sha256(p) for k, p in paths.items()}, "inputs": {k: str(p) for k, p in paths.items()},
            "input_lock_sha256": sha256(lock), "solver_sha256": sha256(sp), "solver": solver._SOLVER_NAME,
            "threads": a.threads, "hostname": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "solver_wall_s": float(r.get("wall_s", time.time() - t0)), "solver_n_inc": int(r["n_inc"]),
            "solver_n_nr": int(r["n_nr"]), "solver_n_cutback": int(r["n_cutback"]),
            "peeq_gp_max": peeq, "peeq_table_fraction": peeq / float(tab["ep_max"]),
            "history_fractions": fr, "outputs": outputs, "snapshots": snaps, "log_file": str(r.get("log_file", "")),
            "completed": time.strftime("%Y-%m-%d %H:%M:%S"), "completed_ok": True}
    tmp = mp.with_name(mp.name + ".tmp"); tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8"); os.replace(tmp, mp)
    print(f"[complete] {mp}")


if __name__ == "__main__":
    main()
