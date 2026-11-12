# -*- coding: utf-8 -*-
"""
run_all_abaqus.py - ONE-RUN Abaqus chain for the mesoscale full-domain test.

    python run_all_abaqus.py

Steps
  1  gen_inp_meso.py        -> <job>.inp        mesh + per-grain materials + DIC BCs
  2  job_submitter_meso.py  -> <job>.odb        Abaqus, blocking
  3  odb_process_meso.py    -> <job>_ab_*.npy   element/nodal export (abaqus python)
  4  fem_to_grid.py         -> grid_abaqus_*    rasterised onto the DIC pixel grid
  5  compare_evm_fields.py  -> field figures    (FEM panel appears once run_all_fem
                                                 has been run too)

Steps 1-3 all skip themselves when their output is already up to date, so
re-running the chain after an interruption is cheap.  Force with
MESO_FORCE_INP / MESO_SKIP_IF_ODB=0 / MESO_FORCE_ODB.

Which case runs is set by MESH_KEY in meso_config.py (or MESO_MESH_KEY).
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import meso_config as cfg
from _chain import run_step, write_report

RUN_GRID = os.environ.get("MESO_RUN_GRID", "1") == "1"

if __name__ == "__main__":
    print(f"mesh   : {cfg.MESH_KEY}")
    print(f"job    : {cfg.JOB_NAME}")
    print(f"inp    : {cfg.JOB_INP}")
    print(f"results: {cfg.RESULTS_DIR}")

    run_step("1/5 Generate .inp (mesh + materials + DIC BCs)",
             "../abaqus/gen_inp_meso.py")
    run_step("2/5 Run Abaqus job (blocking)",
             "../abaqus/job_submitter_meso.py")
    run_step("3/5 Export ODB to element/nodal npy",
             "../abaqus/odb_process_meso.py")
    if RUN_GRID:
        run_step("4/5 Rasterise onto the DIC pixel grid",
                 "fem_to_grid.py")
        run_step("5/5 DIC | Abaqus | FEM field maps",
                 "compare_evm_fields.py")

    write_report("abaqus", cfg)
    print("\nAbaqus chain complete. Outputs in:", cfg.RESULTS_DIR)
