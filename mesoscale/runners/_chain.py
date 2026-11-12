# -*- coding: utf-8 -*-
"""
_chain.py - step runner shared by run_all_abaqus.py and run_all_fem.py.

Each step is a separate process so a crash in one cannot leave stale module
state behind, and its stdout is streamed through ours so progress is visible in
Spyder/IPython (child console output is otherwise swallowed).
"""
import datetime
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

STEP_TIMES = []


def env(**extra):
    e = dict(os.environ, PYTHONUNBUFFERED="1")
    e.update({k: str(v) for k, v in extra.items()})
    return e


def run_step(title, script, environ=None):
    print("\n" + "=" * 72)
    print(f"STEP: {title}")
    print("=" * 72)
    t0 = time.time()
    p = subprocess.Popen([PY, os.path.join(ROOT, script)], cwd=ROOT,
                         env=environ or env(),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="")
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"Step '{title}' failed (rc={rc}) - aborting chain.")
    dt = time.time() - t0
    print(f"[ok] {title} ({dt:.1f}s)")
    STEP_TIMES.append((title, dt))
    return dt


def write_report(chain, cfg):
    total = sum(s for _, s in STEP_TIMES)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(cfg.RESULTS_DIR, f"run_report_{chain}_{cfg.MESH_KEY}_{ts}.txt")
    with open(path, "w") as f:
        f.write(f"chain    : {chain}\n")
        f.write(f"mesh     : {cfg.MESH_KEY}   job {cfg.JOB_NAME}\n")
        f.write(f"finished : {datetime.datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write("steps:\n")
        for t, s in STEP_TIMES:
            f.write(f"  {s:9.1f} s   {t}\n")
        f.write(f"\nTOTAL    : {total:.1f} s  ({total/60:.2f} min)\n")

    csvp = os.path.join(cfg.RESULTS_DIR, "timings_log.csv")
    new = not os.path.isfile(csvp)
    with open(csvp, "a") as f:
        if new:
            f.write("timestamp,chain,mesh,total_s,n_steps\n")
        f.write(f"{ts},{chain},{cfg.MESH_KEY},{total:.1f},{len(STEP_TIMES)}\n")

    print(f"\nTOTAL {chain} chain: {total:.1f} s ({total/60:.2f} min)")
    print(f"report:      {path}")
    print(f"appended to: {csvp}")
