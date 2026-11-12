# -*- coding: utf-8 -*-
"""
gen_inp_meso.py - write the Abaqus .inp for the configured mesoscale case.

Mesoscale twin of Microscale/.../case5_test.py.  Everything
substantive lives in inp_io.build_problem(); this script only decides WHERE the
file goes and refuses to clobber an .inp whose Abaqus results already exist
unless you say so.

    python gen_inp_meso.py            # write only if missing
    MESO_FORCE_INP=1 python gen_inp_meso.py   # overwrite

The generated file is byte-compatible with the layout of
2_codes/mesoscale_mesh_triangular.py, so the existing Abaqus jobs and any new
one are the same kind of object.
"""
import os
import sys

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
# Repository layout: this folder is mesoscale/abaqus; meso_config and inp_io
# live in the sibling mesoscale/runners. (Before the micro/meso split these
# sat under abaqus/1_codes with the config two levels up.)
ROOT = os.path.dirname(CODE_DIR)
sys.path.insert(0, os.path.join(ROOT, "runners"))
sys.path.insert(0, ROOT)

import meso_config as cfg
import inp_io

FORCE = os.environ.get("MESO_FORCE_INP", "1") == "1"


def main():
    if os.path.isfile(cfg.JOB_INP) and not FORCE:
        print(f"[inp ] {cfg.JOB_INP} already exists - keeping it.")
        print("[inp ] set MESO_FORCE_INP=1 to regenerate (this invalidates the "
              "existing .odb and every export made from it).")
        return
    if os.path.isfile(cfg.ODB_PATH) and FORCE:
        print(f"[inp ] WARNING: {cfg.ODB_PATH} exists and will no longer match "
              f"the regenerated .inp. Re-run the Abaqus job.")
    prob = inp_io.build_problem(verbose=True)
    inp_io.write_inp(cfg.JOB_INP, prob, verbose=True)


if __name__ == "__main__":
    main()
