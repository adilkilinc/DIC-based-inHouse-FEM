"""Run one pixel-scale case of the follow-up studies with the locked solver and conventions.

The case, its material and its resources come from the frozen campaign.json plus
followup.json (common.campaign_config).  Inputs are prepared by the frozen
run_case.load_inputs, the solver is the frozen solver/fem_pixel.py, and the outputs
are written by the frozen run_case.save_outputs; only the output root (this
package's 2_data/) and the map root (this package's 0_identification/) differ.

    python run_fem.py --list [--with-optional]          case ids with their dependency stage
    python run_fem.py --resources <case>                cpus mem time partition fallback
    python run_fem.py --case <case> --dry-run           prepare the inputs only
    python run_fem.py --case <case> --threads 8
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import time
from pathlib import Path

import common as K


def all_cases(cfg, rc, with_optional=True):
    out = rc.cases(cfg)
    return {k: v for k, v in out.items() if with_optional or not v.get("optional")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--with-optional", action="store_true")
    ap.add_argument("--resources")
    ap.add_argument("--old-root"); ap.add_argument("--new-root")
    ap.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="64-element crop, any solver (local test)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = K.campaign_config()
    rc = K.frozen_run_case()
    cases = all_cases(cfg, rc, with_optional=True)
    if a.list:
        for cid, c in all_cases(cfg, rc, a.with_optional).items():
            done = (K.PKG / "2_data" / cid / "run_meta.json").is_file()
            print(f"{cid} {c['after']} {'done' if done else 'todo'}")
        return
    if a.resources:
        c = cases[a.resources]
        res = cfg["resources"][c["dataset"]][str(c["factor"])]
        print(res["cpus"], res["mem"], res["time"], res.get("partition", "-"), res.get("fallback_partition", "-"))
        return
    if a.case not in cases:
        raise SystemExit(f"unknown case {a.case}; known: {sorted(cases)}")
    case = cases[a.case]; ds = case["dataset"]; d = cfg["datasets"][ds]
    root = rc.root_for(ds, cfg, a.old_root if ds == "dataset1" else a.new_root)
    out = K.PKG / ("_smoke" if a.smoke else "2_data") / a.case
    out.mkdir(parents=True, exist_ok=True)
    mp = out / "run_meta.json"
    if mp.is_file() and not a.force:
        if json.loads(mp.read_text()).get("completed_ok"):
            print(f"completed result exists: {mp}"); return
        raise SystemExit(f"incomplete output exists: {mp}; inspect or use --force")
    dx, dy, Y, Kc, paths, notes = rc.load_inputs(cfg, case, root, a.smoke)
    print(f"[prepared] {a.case}: elements={Y.shape} bc={case['bc']} yield={Y.mean():.4g}+/-{Y.std():.4g} "
          f"K={Kc.mean():.4g}+/-{Kc.std():.4g} {notes}", flush=True)
    for k, p in paths.items():
        print(f"  input {k}: {p}", flush=True)
    if a.dry_run:
        return
    sp = K.REPO / "microscale" / "solver" / "fem_pixel.py"
    solver = K.load_module(sp, "frozen_fem_pixel")
    if "pypardiso" not in solver._SOLVER_NAME.lower() and not a.smoke:
        raise SystemExit(f"production requires pypardiso; got {solver._SOLVER_NAME}")
    os.environ.update(MKL_NUM_THREADS=str(a.threads), OMP_NUM_THREADS=str(a.threads), OPENBLAS_NUM_THREADS="1",
                      MKL_DYNAMIC="FALSE", FEM_LOG_DIR=str(K.PKG / ("_smoke/logs" if a.smoke else "logs") / a.case))
    Path(os.environ["FEM_LOG_DIR"]).mkdir(parents=True, exist_ok=True)
    tab = cfg["hardening_tables"][case["table"]]; fr = rc.fractions_for(case)
    nx, ny = Y.shape; pitch = float(d["element_size_base"]) * case["factor"]; t0 = time.time()
    with rc.bc_mode(solver, case["bc"]):
        r = solver.run_fem(dx, dy, Y, Kc, float(d["n_exp"]), nx * pitch, ny * pitch, pitch, float(d["pixel_size_um"]),
                           E_mod=float(d["E_mpa"]), nu=float(d["nu"]), N_inc=int(case["n_inc"]),
                           max_nr=int(cfg["max_nr"]), nr_tol=float(cfg["nr_tol"]), hardening=tab["mode"],
                           ep_table_max=float(tab["ep_max"]), n_table=int(tab["n_table"]),
                           ep_table_first=float(tab["ep_first"]), snapshot_fractions=fr, verbose=True,
                           **({"ref_force_tol": float(case["ref_force_tol"])} if case.get("ref_force_tol") else {}))
    outputs, snaps = rc.save_outputs(r, out, a.case, fr, [float(x) for x in cfg["snapshot_fractions"]], float(d["nu"]))
    peeq = float("nan")
    try:
        v = re.findall(r"PEEQ_max=([0-9.eE+-]+)", Path(r["log_file"]).read_text(errors="replace"))
        peeq = float(v[-1]) if v else peeq
    except Exception:
        pass
    meta = {"schema": 2, "campaign": cfg["campaign"], "case_id": a.case, "stage": case["stage"], "dataset": ds,
            "material": case["material"], "material_definition": d["materials"][case["material"]],
            "material_notes": notes, "factor": case["factor"], "bc": case["bc"], "table": case["table"],
            "hardening": dict(tab), "n_inc": case["n_inc"], "max_nr": cfg["max_nr"], "nr_tol": cfg["nr_tol"],
            "ref_force_tol": float(case.get("ref_force_tol", 0.0)),
            "field_shape": case["field_shape"], "crop_offset": case["crop_offset"], "element_shape": list(Y.shape),
            "elements": int(Y.size), "E_mpa": d["E_mpa"], "nu": d["nu"], "K_scale_mpa": d["K_scale_mpa"],
            "n_exp": d["n_exp"],
            "prepared": {"yield_mean_mpa": float(Y.mean()), "yield_std_mpa": float(Y.std()),
                         "K_mean_mpa": float(Kc.mean()), "K_std_mpa": float(Kc.std())},
            "input_sha256": {k: K.sha256(p) for k, p in paths.items()}, "inputs": {k: str(p) for k, p in paths.items()},
            "frozen_campaign_json_sha256": K.sha256(K.FROZEN / "campaign.json"),
            "followup_json_sha256": K.sha256(K.PKG / "followup.json"),
            "solver_sha256": K.sha256(sp), "solver": solver._SOLVER_NAME,
            "threads": a.threads, "hostname": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "solver_wall_s": float(r.get("wall_s", time.time() - t0)), "solver_n_inc": int(r["n_inc"]),
            "solver_n_nr": int(r["n_nr"]), "solver_n_cutback": int(r["n_cutback"]),
            "peeq_gp_max": peeq, "peeq_table_fraction": peeq / float(tab["ep_max"]),
            "history_fractions": fr, "outputs": outputs, "snapshots": snaps, "log_file": str(r.get("log_file", "")),
            "completed": time.strftime("%Y-%m-%d %H:%M:%S"), "completed_ok": True}
    K.write_json(mp, meta)
    print(f"[complete] {mp}")


if __name__ == "__main__":
    main()
