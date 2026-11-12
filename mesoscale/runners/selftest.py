# -*- coding: utf-8 -*-
"""
selftest.py - prove every script in this folder runs, without Abaqus and
without the real data.

    python selftest.py

What it does
  1  fem_tri._verify()   : elastic patch test on a random triangulation (the
                           CST must reproduce a uniform strain field exactly)
                           and equal-biaxial plasticity against the analytic
                           answer - the same case fem_pixel._verify uses
  2  builds a ~30-grain fixture Thesis tree in a temp folder
  3  runs gen_inp_meso -> fem_meso_driver -> fem_to_grid
  4  fabricates an Abaqus-shaped export from the FEM frames with the element
     and node rows SHUFFLED and 0.2 % noise added, so the label matching and
     the diff machinery are exercised against a KNOWN answer
  5  runs compare_fem_abaqus, fem_time_history, compare_evm_fields and checks
     the reported error really is ~0.2 %

Passing this does NOT say the mesoscale answer is right - it says the plumbing
is intact. The physics check is verification 1 plus the real Abaqus comparison.
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
NOISE = 2e-3


def run(script, env, title):
    print("\n" + "=" * 72)
    print("SELFTEST:", title)
    print("=" * 72)
    p = subprocess.run([PY, os.path.join(ROOT, script)], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    tail = "\n".join(p.stdout.strip().splitlines()[-14:])
    print(tail)
    if p.returncode != 0:
        print(p.stdout[-4000:])
        print(p.stderr[-4000:])
        raise SystemExit(f"FAILED: {script}")
    return p.stdout


FAKE_ABAQUS = r'''
import os, sys, numpy as np
sys.path.insert(0, sys.argv[1])
import meso_config as cfg, inp_io
prob = inp_io.build_problem(verbose=False)
d = np.load(os.path.join(cfg.FEM_OUT, cfg.FEM_TAG + "_frames.npz"))
rng = np.random.default_rng(1)
pe = rng.permutation(len(prob["elem_ids"])); pn = rng.permutation(len(prob["node_ids"]))
np.save(os.path.join(cfg.AB_OUT, cfg.JOB_NAME + "_elem_labels.npy"),
        prob["elem_ids"][pe].astype(np.int32))
np.save(os.path.join(cfg.AB_OUT, cfg.JOB_NAME + "_node_labels.npy"),
        prob["node_ids"][pn].astype(np.int32))
np.save(os.path.join(cfg.AB_OUT, cfg.JOB_NAME + "_frametimes.npy"), d["time"])
s = float(sys.argv[2])
for k in ("S", "E", "PE", "PEEQ", "U", "RF"):
    a = d[k]
    a = a[:, pe] if k in ("S", "E", "PE", "PEEQ") else a[:, pn]
    np.save(os.path.join(cfg.AB_OUT, cfg.JOB_NAME + "_ab_" + k + ".npy"),
            a*(1.0 + s*rng.standard_normal(a.shape)))
print("fake Abaqus export ->", cfg.AB_OUT)
'''


def main():
    tmp = tempfile.mkdtemp(prefix="meso_selftest_")
    fixture = os.path.join(tmp, "thesis")
    results = os.path.join(tmp, "results")
    print(f"[selftest] scratch: {tmp}")
    try:
        print("\n" + "=" * 72)
        print("SELFTEST: solver verification (fem_tri._verify)")
        print("=" * 72)
        p = subprocess.run([PY, os.path.join(os.path.dirname(ROOT), "solver", "fem_tri.py")],
                           capture_output=True, text=True)
        print(p.stdout)
        if p.returncode != 0:
            print(p.stderr[-3000:])
            raise SystemExit("FAILED: fem_tri verification")

        subprocess.run([PY, os.path.join(ROOT, "selftest_fixture.py"), fixture],
                       check=True)

        # The material maps and the kinematics no longer live under the
        # Thesis root, so MESO_THESIS_ROOT alone no longer redirects the whole
        # input tree - the two study roots have to be pointed at the fixture
        # as well, or the run would read the real D: data.
        env = dict(os.environ, MESO_THESIS_ROOT=fixture,
                   MESO_THRESH_STUDY_DIR=os.path.join(
                       fixture, "_studies/threshold_study/2_data"),
                   MESO_ALPHA_STUDY_DIR=os.path.join(
                       fixture, "_studies/alpha_study/2_data"),
                   MESO_RESULTS_DIR=results, MESO_MESH_KEY="fine_no_twins",
                   MESO_N_INC="10", MESO_N_FRAMES="6", PYTHONUNBUFFERED="1")

        run("../abaqus/gen_inp_meso.py", env, "generate .inp")
        out = run("check_grain_mapping.py", env, "grain mapping check")
        if "VERDICT: PASS" not in out:
            raise SystemExit("FAILED: grain mapping check did not pass on a "
                             "freshly generated .inp - the writer and the "
                             "solver disagree about the grains")
        run("fem_meso_driver.py", env, "in-house FEM solve")

        fk = os.path.join(tmp, "fake_abaqus.py")
        with open(fk, "w") as f:
            f.write(FAKE_ABAQUS)
        subprocess.run([PY, fk, ROOT, str(NOISE)], cwd=ROOT, env=env, check=True)

        run("fem_to_grid.py", env, "rasterise to the pixel grid")
        out = run("compare_fem_abaqus.py", env, "element-level diff table")
        run("fem_time_history.py", env, "time history")
        run("compare_evm_fields.py", env, "field maps")

        # the injected noise is 0.2 %, so the reported rel % must land there
        rel = None
        for line in out.splitlines():
            t = line.split()
            if len(t) == 6 and t[0] == "svm":
                rel = float(t[-1])
        if rel is None:
            raise SystemExit("FAILED: no svm row in the diff table")
        print(f"\n[selftest] injected noise {100*NOISE:.2f} %  ->  "
              f"reported svm rel {rel:.4f} %")
        if not (0.5*100*NOISE < rel < 2.0*100*NOISE):
            raise SystemExit("FAILED: the diff table does not recover the "
                             "injected error - the comparison logic is wrong")

        for f in ("diff_table_fine_no_twins.txt",
                  "worst_grains_fine_no_twins.csv",
                  "fem_vs_abaqus_fine_no_twins.pdf",
                  "time_history_fine_no_twins.pdf",
                  "evm_fields_dic_abaqus_fem_fine_no_twins.pdf",
                  "u2_fields_dic_abaqus_fem_fine_no_twins.pdf"):
            p = os.path.join(results, "compare", f)
            if not os.path.isfile(p):
                raise SystemExit(f"FAILED: {f} was not produced")

        print("\n" + "=" * 72)
        print("SELFTEST PASSED - every script ran and the comparison recovers "
              "a known error")
        print("=" * 72)
    finally:
        if os.environ.get("MESO_KEEP_SELFTEST", "0") != "1":
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f"[selftest] kept: {tmp}")


if __name__ == "__main__":
    main()
