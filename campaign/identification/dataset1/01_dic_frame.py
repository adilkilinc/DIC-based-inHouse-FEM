# -*- coding: utf-8 -*-
r"""One resumable shard of the Dataset 1 240-iteration alpha sweep.

Each array task takes a contiguous slice of the loading-window frames and
computes its U and V fields. There is no shared state between tasks: every
frame is an independent DIS solve writing two files, so the work parallelises
exactly and a failed task can simply be resubmitted.

Frame numbering, which is the one thing that must not drift
---------------------------------------------------------------------------
02 and 03 select frames by NUMBER, where k is the 1-based position of an
image in the sorted list of all 2551 acquisition frames. Only the 1537 frames
of the loading window are uploaded here, so that position cannot be recovered
from the directory listing. It is read instead from frames_manifest.txt,
written on the machine that holds the complete acquisition. Never regenerate
the manifest from a partial upload.

Parameters come from new_experiment_config.py so that this script cannot
drift from the local pipeline. The correlation is bit-for-bit identical to
the Windows build with opencv-python-headless 4.13.0, verified on the
delivered a100 fields.

Cores per task
---------------------------------------------------------------------------
A solve is single threaded and peaks near 3 GB, so the number of frames that
can run at once is min(cores, memory / 3 GB). On a general node memory binds
first and one core per task is right. On a 1.5 TB bigmem node memory never
binds: ask for as many cores as the node will give and set --procs to match,
because there 500 solves would fit in RAM and only the core count stops you.

    python hpc_dic_shard.py --shard 0 --nshards 100          # 1 core
    python hpc_dic_shard.py --shard 0 --nshards 4 --procs 32 # 32 cores
    python hpc_dic_shard.py --verify        # check every frame loads
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import old_experiment_config as cfg      # noqa: E402

cv2.setNumThreads(1)                     # one core per task; SLURM owns the rest
MANIFEST = os.environ.get("OLDEXP_MANIFEST", str(cfg.MANIFEST))


def load_manifest():
    rows = []
    with open(MANIFEST, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            k, name = line.rstrip("\n").split("\t")
            rows.append((int(k), name))
    if not rows:
        raise SystemExit(f"empty manifest: {MANIFEST}")
    return rows


_WORKER = {}


def _worker_setup(alpha):
    """Per-process state: the masked reference and a flow object. Built once
    on first use and kept, because rebuilding either per frame would cost
    more than it saves."""
    if _WORKER.get("alpha") != alpha:
        ref = cv2.imread(cfg.REF_IMG, 0)
        mask = cv2.imread(cfg.MASK_IMG, 0)
        if ref is None or mask is None:
            raise SystemExit(f"cannot read {cfg.REF_IMG} or {cfg.MASK_IMG}")
        mbin = (mask > 0).astype(np.uint8)
        _WORKER["ref"] = np.ascontiguousarray((ref * mbin).astype(np.uint8))
        _WORKER["mask"] = mbin
        _WORKER["flow"] = make_flow(alpha)
        _WORKER["alpha"] = alpha
        cv2.setNumThreads(1)
    return _WORKER["ref"], _WORKER["mask"], _WORKER["flow"]


def solve_frame(k, name, u_out, v_out, tag, alpha):
    """One frame, start to finish, with nothing captured from the caller."""
    Im1, mbin, flow = _worker_setup(alpha)
    Im2 = cv2.imread(os.path.join(cfg.IMG_DIR, name), 0)
    if Im2 is None:
        raise SystemExit(f"cannot read {os.path.join(cfg.IMG_DIR, name)}")
    Im2 = np.ascontiguousarray((Im2 * mbin).astype(np.uint8))
    res = flow.calc(Im1, Im2, None)
    # temporary name then rename: a task killed at the wall clock never
    # leaves a half-written file that a resubmission would treat as done
    for arr, out in ((res[:, :, 0][cfg.CROP_ROWS, cfg.CROP_COLS], u_out),
                     (res[:, :, 1][cfg.CROP_ROWS, cfg.CROP_COLS], v_out)):
        tmp = f"{out}.tmp{tag}"
        # np.save APPENDS .npy when given a filename that lacks it, which
        # would write <out>.tmpN.npy and leave the rename with nothing to
        # find. Handing it an open file object keeps the name exactly.
        with open(tmp, "wb") as fh:
            np.save(fh, arr)
        os.replace(tmp, out)
    return k


def make_flow(alpha):
    f = cv2.DISOpticalFlow_create()
    f.setFinestScale(cfg.DIS_FINEST_SCALE)
    f.setPatchSize(cfg.DIS_PATCH_SIZE)
    f.setPatchStride(cfg.DIS_PATCH_STRIDE)
    f.setGradientDescentIterations(16)
    f.setUseMeanNormalization(cfg.DIS_MEAN_NORMALIZATION)
    f.setUseSpatialPropagation(cfg.DIS_SPATIAL_PROPAGATION)
    f.setVariationalRefinementAlpha(float(alpha))
    f.setVariationalRefinementDelta(cfg.DIS_DELTA)
    f.setVariationalRefinementGamma(cfg.DIS_GAMMA)
    f.setVariationalRefinementEpsilon(cfg.DIS_EPSILON)
    f.setVariationalRefinementIterations(cfg.DIS_ITERATIONS)
    return f


def complete(u_out, v_out):
    """Done means both files exist AND load. A task killed at the wall clock
    can leave a truncated .npy behind, and a resubmission must redo it."""
    if not (os.path.isfile(u_out) and os.path.isfile(v_out)):
        return False
    for p in (u_out, v_out):
        try:
            np.load(p, mmap_mode="r").shape
        except Exception:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=int(
        os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--nshards", type=int, default=int(
        os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    ap.add_argument("--alpha", type=float, default=None,
                    help="regularisation weight; default is Dataset 1 production alpha")
    ap.add_argument("--procs", type=int, default=int(
        os.environ.get("SLURM_CPUS_PER_TASK", 1)),
        help="frames to solve at once inside this task; needs ~3 GB each")
    ap.add_argument("--verify", action="store_true",
                    help="report completeness over the whole manifest, compute nothing")
    a = ap.parse_args()

    rows = load_manifest()
    alpha = cfg.PROD_ALPHA if a.alpha is None else a.alpha
    key = cfg.alpha_key(int(alpha) if float(alpha).is_integer() else alpha)
    u_dir, v_dir = cfg.uv_dirs(key)
    os.makedirs(u_dir, exist_ok=True)
    os.makedirs(v_dir, exist_ok=True)

    if a.verify:
        miss = [k for k, _ in rows
                if not complete(os.path.join(u_dir, f"U_{k}.npy"),
                                os.path.join(v_dir, f"V_{k}.npy"))]
        print(f"key {key}: {len(rows)-len(miss)}/{len(rows)} frames complete")
        if miss:
            print(f"  missing or unreadable: {len(miss)}")
            print(f"  first few: {miss[:20]}")
            sys.exit(1)
        print("  all frames present and readable")
        return

    mine = rows[a.shard::a.nshards]          # round robin, so every task gets
    print(f"[shard {a.shard}/{a.nshards}] key={key}  "   # a mix of early and late frames
          f"{len(mine)} frames  {a.procs} core(s)  "
          f"alpha={alpha:g} iterations={cfg.DIS_ITERATIONS} "
          f"spatprop={cfg.DIS_SPATIAL_PROPAGATION} "
          f"cv2={cv2.__version__}", flush=True)

    todo = []
    skipped = 0
    for k, name in mine:
        u_out = os.path.join(u_dir, f"U_{k}.npy")
        v_out = os.path.join(v_dir, f"V_{k}.npy")
        if complete(u_out, v_out):
            skipped += 1
            continue
        todo.append((k, name, u_out, v_out))
    print(f"[shard {a.shard}] {len(todo)} to solve, {skipped} already present",
          flush=True)
    if not todo:
        return

    t0 = time.time()
    done = 0

    def progress():
        el = time.time() - t0
        print(f"[shard {a.shard}] {done}/{len(todo)}  {el/60:.1f} min  "
              f"{el/max(done,1):.0f} s/frame  "
              f"ETA {(len(todo)-done)*el/max(done,1)/60:.0f} min", flush=True)

    if a.procs <= 1:
        for item in todo:
            solve_frame(*item, a.shard, alpha)
            done += 1
            if done % 5 == 0:
                progress()
    else:
        with ProcessPoolExecutor(max_workers=a.procs) as pool:
            futs = [pool.submit(solve_frame, *item, a.shard, alpha)
                    for item in todo]
            for fut in as_completed(futs):
                fut.result()          # re-raise in the parent
                done += 1
                if done % max(a.procs, 5) == 0:
                    progress()

    wall = time.time() - t0
    print(f"[shard {a.shard}] DONE {done} solved, {skipped} already present, "
          f"{wall/60:.1f} min wall, {done*a.procs and wall*a.procs/3600:.2f} "
          f"core-hours", flush=True)


if __name__ == "__main__":
    main()
