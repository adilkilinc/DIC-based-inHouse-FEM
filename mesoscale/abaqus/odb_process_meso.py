# -*- coding: utf-8 -*-
"""
odb_process_meso.py - Python 3 wrapper that runs the element-level ODB export
inside Abaqus's own Python 2.7 interpreter.

    python odb_process_meso.py

Equivalent of 2_codes/Odb_process_2.py, but pointed at odb_extract_meso.py
(element values, ~40 MB) instead of odb_grid_4.py (pixel raster, ~1.8 GB per
field).  The pixel raster is produced later in Python 3 by
runners/fem_to_grid.py, using the same routine for Abaqus and FEM.

Skips itself when the export is already newer than the .odb; set
MESO_FORCE_ODB=1 to re-export.
"""
import os
import subprocess
import sys

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
# Repository layout: this folder is mesoscale/abaqus; meso_config and inp_io
# live in the sibling mesoscale/runners. (Before the micro/meso split these
# sat under abaqus/1_codes with the config two levels up.)
ROOT = os.path.dirname(CODE_DIR)
sys.path.insert(0, os.path.join(ROOT, "runners"))
sys.path.insert(0, ROOT)

import meso_config as cfg

ABAQUS_CMD = os.environ.get("MESO_ABAQUS_CMD", r"C:\SIMULIA\Commands\abaqus.bat")
SCRIPT = os.path.join(CODE_DIR, "odb_extract_meso.py")
STEP_NAME = os.environ.get("MESO_STEP_NAME", "Step-1")
INSTANCE_NAME = os.environ.get("MESO_INSTANCE_NAME", "PART-1-1")
FORCE = os.environ.get("MESO_FORCE_ODB", "0") == "1"


def main():
    if not os.path.isfile(cfg.ODB_PATH):
        raise FileNotFoundError(f"{cfg.ODB_PATH} - run job_submitter_meso.py first")

    stamp = os.path.join(cfg.AB_OUT, f"{cfg.JOB_NAME}_ab_S.npy")
    if os.path.isfile(stamp) and not FORCE:
        if os.path.getmtime(stamp) >= os.path.getmtime(cfg.ODB_PATH):
            print(f"[odb ] export in {cfg.AB_OUT} is up to date - skipping. "
                  f"Set MESO_FORCE_ODB=1 to re-export.")
            return

    cmd = [ABAQUS_CMD, "python", SCRIPT, cfg.ODB_PATH, cfg.AB_OUT,
           STEP_NAME, INSTANCE_NAME]
    print("[odb ] " + " ".join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         universal_newlines=True, shell=False)
    for line in iter(p.stdout.readline, ""):
        if not line:
            break
        print(line.rstrip())
    p.stdout.close()
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"abaqus python failed with return code {rc}")
    print(f"[odb ] export written to {cfg.AB_OUT}")


if __name__ == "__main__":
    main()
