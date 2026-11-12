# -*- coding: utf-8 -*-
"""
job_submitter_meso.py - run the Abaqus job for the configured case, BLOCKING.

Unlike 2_codes/job_submitter_2.py (which opens a detached cmd window) this
waits for the job to finish, so it can sit inside run_all_abaqus.py as one
step of a chain.  Abaqus's own log is streamed through to this console.

    python job_submitter_meso.py

Set MESO_ABAQUS_CPUS to use more cores (default 4).
Set MESO_SKIP_IF_ODB=0 to force a re-run even when the .odb already exists.
"""
import os
import subprocess
import sys
import time

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
# Repository layout: this folder is mesoscale/abaqus; meso_config and inp_io
# live in the sibling mesoscale/runners. (Before the micro/meso split these
# sat under abaqus/1_codes with the config two levels up.)
ROOT = os.path.dirname(CODE_DIR)
sys.path.insert(0, os.path.join(ROOT, "runners"))
sys.path.insert(0, ROOT)

import meso_config as cfg

ABAQUS_CMD = os.environ.get("MESO_ABAQUS_CMD", r"C:\SIMULIA\Commands\abaqus.bat")
CPUS = int(os.environ.get("MESO_ABAQUS_CPUS", 4))
SKIP_IF_ODB = os.environ.get("MESO_SKIP_IF_ODB", "1") == "1"


def main():
    if not os.path.isfile(cfg.JOB_INP):
        raise FileNotFoundError(f"{cfg.JOB_INP} - run gen_inp_meso.py first")

    if os.path.isfile(cfg.ODB_PATH) and SKIP_IF_ODB:
        if os.path.getmtime(cfg.ODB_PATH) >= os.path.getmtime(cfg.JOB_INP):
            print(f"[abq ] {cfg.ODB_PATH} is newer than the .inp - skipping the "
                  f"solve. Set MESO_SKIP_IF_ODB=0 to force a re-run.")
            return
        print("[abq ] .odb is OLDER than the .inp - re-running.")

    lck = os.path.join(cfg.INP_DIR, cfg.JOB_NAME + ".lck")
    if os.path.exists(lck):
        raise RuntimeError(f"{lck} exists - a job for {cfg.JOB_NAME} is already "
                           f"running. Wait for it or delete the lock.")

    cmd = [ABAQUS_CMD, f"job={cfg.JOB_NAME}", f"input={cfg.JOB_NAME}.inp",
           f"cpus={CPUS}", "interactive", "ask_delete=OFF"]
    print("[abq ] cwd:", cfg.INP_DIR)
    print("[abq ] " + " ".join(cmd))
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=cfg.INP_DIR, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, universal_newlines=True,
                         shell=False)
    for line in iter(p.stdout.readline, ""):
        if not line:
            break
        print(line.rstrip())
    p.stdout.close()
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"Abaqus exited with code {rc} - see "
                           f"{cfg.JOB_NAME}.msg / .dat in {cfg.INP_DIR}")
    if not os.path.isfile(cfg.ODB_PATH):
        raise RuntimeError(f"Abaqus finished but {cfg.ODB_PATH} was not created")
    print(f"[abq ] done in {time.time()-t0:.1f}s -> {cfg.ODB_PATH}")


if __name__ == "__main__":
    main()
