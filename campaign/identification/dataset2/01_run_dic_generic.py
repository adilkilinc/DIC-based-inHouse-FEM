# -*- coding: utf-8 -*-
"""
01_run_dic_generic.py - run DIS optical flow (cv2.DISOpticalFlow, same call
as 1_generate_dic_corrected.py) at every alpha in new_experiment_config.ALPHAS,
over this specimen's own cam1_imgs frames, and save U/V npy files under
new_experiment_config.DISFLOW_DIR/<key>/{U,V}/U_<n>.npy, V_<n>.npy.

This is the generic, keyed generalisation of 1_generate_dic_corrected.py,
in the same style as the old experiment's 01_run_disflow_alpha_sweep.py:
only `alpha` (setVariationalRefinementAlpha) varies between runs - delta,
gamma, epsilon, patch size/stride, iterations and finestScale are held
fixed at 1_generate_dic_corrected.py's actual call-site values (see
new_experiment_config.py), because those also shape the smoothing and must
stay constant for the sweep to isolate alpha's effect alone.

SOURCE DATA - governing constraint
-----------------------------------------------------------------------
Every image read here comes from THIS specimen's own tree
(2_reidentification_new_experiment/2_Data/29_New_Tests_QI/...) - never
from D:\\WORK\\Thesis\\3_data or C:\\...\\3_data (the old experiment's
data, off limits per project constraint). See new_experiment_config.py's
module docstring for the exact justification.

Free production key
-----------------------------------------------------------------------
alpha=100 at the production iteration count (30) has ALREADY been computed
for the full 2551-frame set by 1_generate_dic_corrected.py, at
new_experiment_config.EXISTING_TRACTION_DIR (~3.9h of compute already
spent). Rather than redo that work, this script HARDLINKS those existing
U_k.npy/V_k.npy files into disflow/<a100 key>/{U,V}/ the first time that
key is requested, and only runs DIS optical flow for every OTHER alpha.
Hardlinks are same-volume, instant and use no extra disk; if the two paths
ever end up on different volumes/filesystems the link falls back to a
plain copy automatically.

Resume-safe: existing U_k.npy/V_k.npy are skipped unless NEWEXP_FORCE=1 is
set, so a crashed or interrupted sweep can just be re-run.

This is the expensive step for every alpha other than the production one -
N frames x DIS solves at full image resolution. Run it unattended.

Concurrency model
-----------------------------------------------------------------------
Two independent knobs, both env-overridable:
  NEWEXP_N_JOBS            total DIS-solve worker budget across EVERYTHING
                            running at once (default 8 - same as the
                            production script's joblib n_jobs).
  NEWEXP_ALPHA_CONCURRENCY how many alphas are processed at the same time
                            (default: min(3, number of alphas requested)).
Each concurrent alpha gets NEWEXP_N_JOBS // NEWEXP_ALPHA_CONCURRENCY frame
workers, so the total number of simultaneous DIS solves stays close to
NEWEXP_N_JOBS regardless of how it's split - running more alphas at once
does not multiply total CPU load, it re-divides the same budget so more
alphas make visible progress in parallel instead of one alpha hogging all
the cores while 9 others wait in line. Set NEWEXP_ALPHA_CONCURRENCY=1 to
get the old fully-sequential behaviour back.

Run:  python 01_run_dic_generic.py            (all alphas in cfg.ALPHAS)
      python 01_run_dic_generic.py 1 50 500   (only these alphas)
"""
import os
import sys
import re
import time
import shutil
import threading
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
import new_experiment_config as cfg

FORCE = os.environ.get("NEWEXP_FORCE", "0") == "1"
N_JOBS = int(os.environ.get("NEWEXP_N_JOBS", "8"))
# Correlate only the frames the identification actually uses
# (new_experiment_config.FRAME_START:FRAME_END, the pure-loading phase).
# The pre-load plateau and the unloading tail are never read by 02 or 03, so
# at a high iteration count they are 40 % of the cost for nothing. Downstream
# selects frames by NUMBER, so a windowed run stays consistent.
WINDOW_ONLY = os.environ.get("NEWEXP_WINDOW_ONLY", "0") == "1"

# joblib backend for the frame workers. This used to be "threads", which is
# why eight workers only ever bought a 1.5x speedup: DISOpticalFlow.calc holds
# the GIL for most of its run, and cv2.setNumThreads(1) removes OpenCV's own
# internal parallelism, so the threads queue behind one another. Separate
# processes have no GIL to share and scale with the cores. Each worker loads
# the masked reference once and keeps it (see _worker_reference), so the big
# array is not shipped per task.
PARALLEL_PREFER = os.environ.get("NEWEXP_PARALLEL", "processes")

# A single 240-iteration DIS solve on the 3008x4112 frame peaks at about
# 2.9 GB resident - the variational refinement keeps a pyramid of float
# buffers, and the cost is in the image size, not the iteration count. With
# threads that never mattered, because the GIL let only one or two solves
# actually run at a time. With processes every worker allocates its own, so
# eight of them want ~23 GB and the operating system starts killing them.
# Workers are therefore sized against free memory as well as cores, and the
# arithmetic is printed so a machine that can afford more can be given more.
MEM_PER_WORKER_GB = float(os.environ.get("NEWEXP_MEM_PER_WORKER_GB", "3.0"))
MEM_RESERVE_GB = float(os.environ.get("NEWEXP_MEM_RESERVE_GB", "2.5"))


def _memory_gb():
    """(available, total) physical memory in GB, or (None, None)."""
    try:
        if os.name == "nt":
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

            m = _MS()
            m.dwLength = ctypes.sizeof(_MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullAvailPhys / 2 ** 30, m.ullTotalPhys / 2 ** 30
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, v = line.partition(":")
                info[k] = int(v.split()[0])
        return info["MemAvailable"] / 2 ** 20, info["MemTotal"] / 2 ** 20
    except Exception:
        return None, None


def memory_safe_workers(requested):
    """Cap the worker count at what memory can actually hold."""
    if PARALLEL_PREFER != "processes":
        return requested
    avail, total = _memory_gb()
    if avail is None:
        _log("[01] could not read memory; leaving the worker count alone")
        return requested
    budget = max(avail - MEM_RESERVE_GB, MEM_PER_WORKER_GB)
    cap = max(1, int(budget // MEM_PER_WORKER_GB))
    n = min(requested, cap)
    _log(f"[01] memory: {avail:.1f} GB free of {total:.1f} GB; reserving "
         f"{MEM_RESERVE_GB:.1f} GB, {MEM_PER_WORKER_GB:.1f} GB per worker "
         f"-> room for {cap}")
    if n < requested:
        _log(f"[01] reducing frame workers {requested} -> {n} so the solves "
             f"fit in memory. Raise NEWEXP_N_JOBS on a larger machine, or "
             f"lower NEWEXP_MEM_PER_WORKER_GB if this proves conservative.")
    return n
_PRINT_LOCK = threading.Lock()


def _log(msg):
    with _PRINT_LOCK:
        print(msg, flush=True)

# ── cv2 import - same DLL-loading workaround 1_generate_dic_corrected.py and
# the old experiment's 01_run_disflow_alpha_sweep.py both need on this
# machine's custom opencv build. Tried first as a plain import so this script
# still works unmodified on a machine with a normal opencv-python install. ──
try:
    import cv2
except ImportError:
    cv2_dir = os.path.join(os.environ.get("OPENCV_BUILD", "opencv/build"), "lib", "python3")
    pyd_src = os.path.join(cv2_dir, "cv2.cp312-win_amd64.pyd")
    pyd_dst = os.path.join(cv2_dir, "cv2.pyd")
    if os.path.isfile(pyd_src) and not os.path.isfile(pyd_dst):
        shutil.copyfile(pyd_src, pyd_dst)
    cv2_bin = os.path.join(os.environ.get("OPENCV_BUILD", "opencv/build"), "bin")
    if os.path.isdir(cv2_bin):
        os.add_dll_directory(cv2_bin)
    sys.path.insert(0, cv2_dir)
    import cv2

import numpy as np
from joblib import Parallel, delayed

cv2.setNumThreads(1)   # joblib controls parallelism, not OpenCV internals


def DIC_eps(Im1, Im2, alpha):
    flow = cv2.DISOpticalFlow_create()
    flow.setFinestScale(cfg.DIS_FINEST_SCALE)
    flow.setVariationalRefinementAlpha(alpha)
    flow.setVariationalRefinementDelta(cfg.DIS_DELTA)
    flow.setVariationalRefinementGamma(cfg.DIS_GAMMA)
    flow.setVariationalRefinementEpsilon(cfg.DIS_EPSILON)
    flow.setVariationalRefinementIterations(cfg.DIS_ITERATIONS)
    # these two were never set and therefore ran at the OpenCV defaults;
    # they are the largest single lever on strain noise, see the rigid-body
    # study in Case_studies/dic_noise
    flow.setUseSpatialPropagation(cfg.DIS_SPATIAL_PROPAGATION)
    flow.setUseMeanNormalization(cfg.DIS_MEAN_NORMALIZATION)
    flow.setPatchSize(cfg.DIS_PATCH_SIZE)
    flow.setPatchStride(cfg.DIS_PATCH_STRIDE)
    res = flow.calc(Im1, Im2, None)
    return res[:, :, 0], res[:, :, 1]   # U (x-disp), V (y-disp)


_WORKER_REF = {}


def _worker_reference(ref_path, mask_path):
    """The masked reference image, loaded once per worker process and kept.
    Reading it per task would cost more than the correlation."""
    k = (ref_path, mask_path)
    if k not in _WORKER_REF:
        im = cv2.imread(ref_path, 0)
        mk = cv2.imread(mask_path, 0)
        if im is None or mk is None:
            raise FileNotFoundError(f"worker could not read {ref_path} or "
                                    f"{mask_path}")
        mb = (mk > 0).astype(np.uint8)
        _WORKER_REF[k] = ((im * mb).astype(np.uint8), mb)
    return _WORKER_REF[k]


def solve_frame(alpha, ref_path, mask_path, img_path, u_out, v_out):
    """One frame, start to finish, with nothing captured from the caller -
    so it survives being sent to a separate process."""
    Im1_masked, mask_bin = _worker_reference(ref_path, mask_path)
    Im2_full = cv2.imread(img_path, 0)
    if Im2_full is None:
        raise FileNotFoundError(f"could not read {img_path}")
    Im2_masked = (Im2_full * mask_bin).astype(np.uint8)
    U, V = DIC_eps(Im1_masked, Im2_masked, alpha=alpha)
    # crop the FLOW FIELD (not the images) - matches
    # 1_generate_dic_corrected.py and the old experiment's 01 script
    np.save(u_out, U[cfg.CROP_ROWS, cfg.CROP_COLS])
    np.save(v_out, V[cfg.CROP_ROWS, cfg.CROP_COLS])
    return 1


def _pair_is_complete(u_out, v_out):
    """A frame counts as done only if BOTH arrays exist AND load.

    Existence alone is not enough. Killing the run writes whatever the worker
    had flushed, so the frame it was in the middle of saving can be left as a
    truncated .npy. That file exists, so a resume would skip it, and the gap
    would only surface much later as a corrupt frame in the middle of the
    identification. Reading the header is cheap next to a DIS solve, so it is
    done for every candidate and the broken frame is simply recomputed.
    """
    if not (os.path.isfile(u_out) and os.path.isfile(v_out)):
        return False
    for f in (u_out, v_out):
        try:
            np.load(f, mmap_mode="r").shape
        except Exception:
            _log(f"[01] {os.path.basename(f)} is unreadable "
                 f"(likely truncated by an interrupted run) - recomputing "
                 f"this frame.")
            return False
    return True


def sorted_cam1_images(folder):
    """Sorted by the NUMERIC frame-index prefix in the filename (e.g.
    '1000_20.710.tiff' -> 1000), not lexicographically and not by the
    trailing time/load value - same convention 1_generate_dic_corrected.py
    uses."""
    paths = sorted(glob(os.path.join(folder, "*")))

    def key(p):
        nums = re.findall(r"\d+", os.path.basename(p))
        if not nums:
            return -1
        return int(nums[0])
    ordered = sorted(paths, key=key)
    keys = [key(p) for p in ordered]
    if keys != sorted(keys) or len(set(keys)) != len(keys):
        raise RuntimeError(
            "cam1_imgs filenames do not sort to a unique increasing frame "
            f"index - got {keys[:10]}...  Check the filenames in {folder}.")
    return ordered


def try_reuse_existing_production(key):
    """If `key` is the production alpha at the production iteration count
    and the already-computed traction/ data is available, hardlink its
    U_k.npy/V_k.npy files into disflow/<key>/{U,V}/ instead of recomputing.
    Returns True if the key is now (already, or newly) fully populated from
    the existing data, i.e. the caller can skip DIS entirely for this key.
    """
    alpha, iterations = cfg.parse_key(key)
    if not (alpha == cfg.PROD_ALPHA and iterations == cfg.PROD_DIS_ITERATIONS):
        return False
    if not os.path.isdir(cfg.EXISTING_TRACTION_DIR):
        return False

    u_dir, v_dir = cfg.uv_dirs(key)
    os.makedirs(u_dir, exist_ok=True)
    os.makedirs(v_dir, exist_ok=True)

    src_U = cfg.get_sorted_npy_files(cfg.EXISTING_TRACTION_DIR, "U")
    src_V = cfg.get_sorted_npy_files(cfg.EXISTING_TRACTION_DIR, "V")
    if not src_U or len(src_U) != len(src_V):
        _log(f"[01] {key}: EXISTING_TRACTION_DIR found but U/V file counts "
             f"don't match or are empty ({len(src_U)} vs {len(src_V)}) - "
             f"not reusing it, will run DIS normally.")
        return False

    n_linked = n_present = 0
    for fU, fV in zip(src_U, src_V):
        dst_u = os.path.join(u_dir, fU)
        dst_v = os.path.join(v_dir, fV)
        for src_name, dst in ((fU, dst_u), (fV, dst_v)):
            if os.path.isfile(dst) and not FORCE:
                n_present += 1
                continue
            src_path = os.path.join(cfg.EXISTING_TRACTION_DIR, src_name)
            try:
                if os.path.isfile(dst):
                    os.remove(dst)
                os.link(src_path, dst)   # hardlink: instant, no extra disk
            except OSError:
                shutil.copyfile(src_path, dst)   # cross-volume fallback
            n_linked += 1

    _log(f"[01] {key}: reused EXISTING production DIC data from "
         f"{cfg.EXISTING_TRACTION_DIR} ({len(src_U)} frame pairs, "
         f"{n_linked} newly linked/copied, {n_present} already present) "
         f"- no DIS solves needed for this key.")
    return True


def process_alpha(alpha, Im1_masked, mask_bin, images, per_alpha_jobs):
    """Everything needed to produce disflow/<key>/{U,V}/ for one alpha.
    Runs standalone in its own thread so several alphas can be in flight at
    once (see the concurrency model in the module docstring)."""
    key = cfg.alpha_key(alpha)

    if try_reuse_existing_production(key) and not FORCE:
        return

    u_dir, v_dir = cfg.uv_dirs(key)
    os.makedirs(u_dir, exist_ok=True)
    os.makedirs(v_dir, exist_ok=True)

    # Filter down to the frames that actually still need solving, so the
    # parallel dispatch below only ever does real work.
    todo = []
    n_skip = n_outside = 0
    for k, img_path in enumerate(images, start=1):
        if WINDOW_ONLY and not cfg.in_window(k):
            n_outside += 1
            continue
        u_out = os.path.join(u_dir, f"U_{k}.npy")
        v_out = os.path.join(v_dir, f"V_{k}.npy")
        if not FORCE and _pair_is_complete(u_out, v_out):
            n_skip += 1
            continue
        todo.append((k, img_path, u_out, v_out))

    _log(f"[01] alpha={alpha:>4d}  key={key}  "
         f"{len(todo)} to solve, {n_skip} already present, "
         f"{n_outside} outside the loading window  "
         f"(this alpha's frame workers: {per_alpha_jobs})")
    if not todo:
        return

    t0 = time.time()
    n_done = 0
    batch_size = max(per_alpha_jobs * 4, 1)
    for batch_start in range(0, len(todo), batch_size):
        batch = todo[batch_start:batch_start + batch_size]
        Parallel(n_jobs=per_alpha_jobs, prefer=PARALLEL_PREFER)(
            delayed(solve_frame)(alpha, cfg.REF_IMG, cfg.MASK_IMG, img_path,
                                 u_out, v_out)
            for (_k, img_path, u_out, v_out) in batch)
        n_done += len(batch)
        elapsed = time.time() - t0
        rate = n_done / max(elapsed, 1e-9)
        eta = (len(todo) - n_done) / max(rate, 1e-9)
        _log(f"[01] alpha={alpha:>4d}  {n_done}/{len(todo)} solved  "
             f"{elapsed/60:.1f} min elapsed  ETA {eta/60:.1f} min")

    _log(f"[01] alpha={alpha:>4d} DONE  {n_done} solved / {n_skip} "
         f"skipped  in {time.time()-t0:.1f}s  -> {u_dir}")


def main(alphas):
    if not os.path.isfile(cfg.REF_IMG):
        raise FileNotFoundError(f"reference image not found: {cfg.REF_IMG}")
    if not os.path.isfile(cfg.MASK_IMG):
        raise FileNotFoundError(f"mask image not found: {cfg.MASK_IMG}")
    if not os.path.isdir(cfg.IMG_DIR):
        raise FileNotFoundError(f"image source dir not found: {cfg.IMG_DIR}")

    Im1_full = cv2.imread(cfg.REF_IMG, 0)
    mask_full = cv2.imread(cfg.MASK_IMG, 0)
    if Im1_full is None:
        raise FileNotFoundError(f"could not read {cfg.REF_IMG}")
    if mask_full is None:
        raise FileNotFoundError(f"could not read {cfg.MASK_IMG}")

    # Defensive binarisation (same fix as 1_generate_dic_corrected.py): a
    # {0,255}-convention mask would silently wrap uint8*mask modulo 256. For
    # this specimen mask.png is already all-ones (masks nothing; the crop
    # window is the only spatial restriction) - this changes no numbers here,
    # it only guards against a future non-trivial mask using {0,255}.
    uniq = np.unique(mask_full)
    print(f"[01] mask.png unique values: {uniq[:8]}"
          + (" ..." if uniq.size > 8 else ""))
    if uniq.size == 1:
        print("[01] NOTE: mask has a single unique value -> masks NOTHING. "
              "The fixed crop window (new_experiment_config.CROP_ROWS/"
              "CROP_COLS) is the only spatial restriction actually applied.")
    mask_bin = (mask_full > 0).astype(np.uint8)

    Im1_masked = (Im1_full * mask_bin).astype(np.uint8)

    images = sorted_cam1_images(cfg.IMG_DIR)
    print(f"[01] {len(images)} frames from {cfg.IMG_DIR}")
    print(f"[01] first 5: {[os.path.basename(p) for p in images[:5]]}")
    print(f"[01] last 5 : {[os.path.basename(p) for p in images[-5:]]}")

    alpha_concurrency = min(
        int(os.environ.get("NEWEXP_ALPHA_CONCURRENCY", min(3, len(alphas)))),
        len(alphas))
    per_alpha_jobs = max(1, N_JOBS // max(alpha_concurrency, 1))
    print(f"[01] machine reports {os.cpu_count()} logical cores")
    per_alpha_jobs = memory_safe_workers(min(per_alpha_jobs,
                                             os.cpu_count() or per_alpha_jobs))
    print(f"[01] concurrency: {alpha_concurrency} alpha(s) at once, "
          f"{per_alpha_jobs} frame worker(s) each, backend {PARALLEL_PREFER}  "
          f"(NEWEXP_N_JOBS={N_JOBS}, NEWEXP_ALPHA_CONCURRENCY={alpha_concurrency})")

    t_all = time.time()
    errors = []
    with ThreadPoolExecutor(max_workers=alpha_concurrency) as pool:
        futures = {
            pool.submit(process_alpha, alpha, Im1_masked, mask_bin, images,
                        per_alpha_jobs): alpha
            for alpha in alphas
        }
        for fut in as_completed(futures):
            alpha = futures[fut]
            try:
                fut.result()
            except Exception as e:
                _log(f"[01] alpha={alpha:>4d} FAILED: {e!r}")
                errors.append((alpha, e))

    print(f"\n[01] TOTAL {time.time()-t_all:.1f}s ({(time.time()-t_all)/60:.1f} min)")
    if errors:
        failed = ", ".join(str(a) for a, _ in errors)
        raise RuntimeError(f"[01] {len(errors)} alpha(s) failed: {failed} - "
                           f"see the FAILED lines above; re-run this script "
                           f"to retry (already-solved frames are skipped).")
    print(f"[01] Now run:  python 02_yield_hardening_generic.py")


if __name__ == "__main__":
    alphas = [int(a) for a in sys.argv[1:]] or cfg.ALPHAS
    bad = [a for a in alphas if a not in cfg.ALPHAS]
    if bad:
        print(f"[01] warning: {bad} not in new_experiment_config.ALPHAS "
              f"({cfg.ALPHAS}) - running them anyway, but downstream scripts "
              f"iterate over ALPHAS, so add them there too if you want them "
              f"picked up automatically.")
    print(f"[01] alphas: {alphas}  (DIS_ITERATIONS={cfg.DIS_ITERATIONS})")
    main(alphas)
