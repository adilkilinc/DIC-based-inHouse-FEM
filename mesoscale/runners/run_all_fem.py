# -*- coding: utf-8 -*-
"""
run_all_fem.py - ONE-RUN in-house FEM chain for the mesoscale full-domain test.
No Abaqus needed for steps 1-2; steps 3-5 add the Abaqus comparison when the
Abaqus chain has already been run.

    python run_all_fem.py

Steps
  1  fem_meso_driver.py     -> <fem_tag>_{S,E,PE,PEEQ,U,RF}.npy + _frames.npz
                               (ONE incremental solve, snapshotting the Abaqus
                                frame times so the histories line up)
  2  fem_to_grid.py         -> grid_fem_* / grid_abaqus_* on the DIC pixel grid
  3  compare_fem_abaqus.py  -> diff_table / worst_grains / correlation figure
  4  fem_time_history.py    -> stress, strain and u2 against pseudo-time
  5  compare_evm_fields.py  -> DIC | Abaqus | FEM field maps

Knobs (environment, all optional):
    MESO_MESH_KEY     which case          (default fine_no_twins)
    MESO_N_INC        max increments      (default 20, halved on cutback)
    MESO_N_FRAMES     snapshots when no Abaqus frame times exist yet (20)
    MESO_FORCE_FEM=1  re-solve even if saved fields exist
    MESO_GRID_ALL_FRAMES=1  rasterise every frame, not just the last
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import meso_config as cfg
from _chain import run_step, write_report

if __name__ == "__main__":
    print(f"mesh   : {cfg.MESH_KEY}")
    print(f"job    : {cfg.JOB_NAME}")
    print(f"results: {cfg.RESULTS_DIR}")

    # Step 1 aborts the chain if the two codes are not on the same mesh or the
    # same grain partition - solving would just produce a comparison of two
    # different problems. Material/BC drift only warns.
    run_step("1/6 Grain mapping check (mesh, partition, materials, BCs)",
             "check_grain_mapping.py")
    run_step("2/6 In-house FEM solve (+ frame snapshots)",
             "fem_meso_driver.py")
    run_step("3/6 Rasterise onto the DIC pixel grid",
             "fem_to_grid.py")
    run_step("4/6 Element-level FEM vs Abaqus diff table",
             "compare_fem_abaqus.py")
    run_step("5/6 stress / strain / u2 - pseudo-time history",
             "fem_time_history.py")
    run_step("6/6 DIC | Abaqus | FEM field maps",
             "compare_evm_fields.py")

    write_report("fem", cfg)
    print("\nFEM chain complete. Outputs in:", cfg.RESULTS_DIR)
