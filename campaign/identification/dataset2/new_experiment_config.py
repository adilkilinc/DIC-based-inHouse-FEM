# -*- coding: utf-8 -*-
"""
new_experiment_config.py - single place to configure the new-experiment
DIC -> yield/hardening -> alpha-selection -> threshold+FEM chain, for
specimen rb_sp2 (session 26-03-26_11h26), mirroring alpha_study's
alpha_config.py / thresh_config.py in structure and naming so the same
downstream scripts (04_run_fem_*, 05_compare_fem_across_alpha, the
threshold-study scripts) can eventually be pointed at this tree with
minimal changes.

GOVERNING CONSTRAINT - READ THIS BEFORE CHANGING ANY PATH BELOW
-----------------------------------------------------------------------
No displacement, yield or hardening DATA from D:\\WORK\\Thesis\\3_data or
C:\\...\\3_data may be read as an input anywhere in this pipeline. That
excludes exactly one file, ``stress_strain.npy`` (macroscopic curve, no
recalculated counterpart) - see MACRO_CURVE below. Every displacement
field, strain history, yield map and hardening map used here is either
computed fresh by 01_run_dic_generic.py from this specimen's own raw
images, or is the ALREADY-COMPUTED production DIC run for this same
specimen (rb_sp2/traction, alpha=100 @ 30 iterations - see
EXISTING_TRACTION_DIR below), which lives entirely under
2_reidentification_new_experiment/2_Data/29_New_Tests_QI and has nothing
to do with the old experiment's 3_data tree. Reading METHODOLOGY
(algorithms, parameter conventions) from alpha_study/threshold_study's
code on D: is fine and is in fact where most of this file's structure
comes from; their DATA outputs are not used anywhere below.

Pipeline (run in order, from this folder):
    01_run_dic_generic.py         DIS optical flow at each alpha -> DIC fields
    02_yield_hardening_generic.py yield stress + hardening K per alpha/key
    (03) alpha/DIC-parameter selector           -- not yet built
    (04) threshold sweep + FEM                  -- not yet built

Physical/numerical constants below are the NEW experiment's own values,
decided this session:
  - E: 193397.86 MPa, measured on this specimen over the fixed elastic
    extensometer window by derive_macro_parameters.py.  nu remains 0.30.
    The earlier inherited 205000 MPa nominal value is no longer used for
    Dataset 2's constitutive FEM.
  - Pixel scale (PX_TO_MM/SCALE_FACTOR): calibrated optical-camera scale for
    rb_sp2, 3.45 um/px (provided 2026-09-12).  This replaces the earlier,
    incorrect 1.84 um/px assumption inherited from the old experiment.
    Downstream DIC-displacement-to-mm and FEM scripts import these constants
    from this module.
  - K_SCALE/N_EXP: this session's CORRECTED macro Ludwik fit for THIS
    specimen (4_ludwik_fit_corrected.py: K=324.0504 MPa, n=0.077767,
    R^2=0.9706, 0.02%-proportional-limit plastic-region filter), not the
    original script's stale hardcoded-yield-stress fit (K=340.12,
    n=0.0893). Per-pixel K in MPa = K_SCALE * K_hardening_coef_filled.
  - SLOP_REF: recomputed with the same 0.02% correction. The value
    hardcoded in 2_elastoplasticity_corrected.py (1.119e-05) was derived
    from the 0.2%-offset total_exx_rate (2.2085e-05, confirmed by
    2_yield_onset_02offset_corrected.py to be exactly this dataset's
    0.2%-offset value):
        slop_ref = 2*(1+nu)/3 * (total_exx_rate - elastic_exx_rate)
                 = 0.8667 * (2.2085e-05 - 9.177e-06) = 1.119e-05
    Using the corrected 0.02%-anchored total_exx_rate (2.2211e-05)
    instead:
        slop_ref_corrected = 0.8667 * (2.2211e-05 - 9.177e-06)
                            = 0.8667 * 1.3034e-05 = 1.130e-05
    a +0.94% change - flagged here for visibility, not because it moves
    any downstream number much.
"""
import os
import re
import numpy as np

# --------------------------------------------------------------------------
# Alpha sweep (identical list to the old experiment's alpha_study, so any
# self-convergence conclusions drawn there about coverage/spacing transfer)
# --------------------------------------------------------------------------
ALPHAS = [1, 10, 20, 50, 80, 100, 150, 200]
PROD_ALPHA = 100     # the value 1_generate_dic_corrected.py already used

# Defined here, above alpha_key, because the iteration count is part of the
# dataset key.
PROD_DIS_ITERATIONS = 30   # 1_generate_dic_corrected.py's DIC_eps default,
                           # never overridden at its call site -> what the
                           # existing rb_sp2/traction data was produced with.
DIS_ITERATIONS = int(os.environ.get("NEWEXP_DIS_ITERATIONS", PROD_DIS_ITERATIONS))


# Spatial propagation and mean normalisation are OpenCV defaults (both True)
# and were never set explicitly, so every field produced so far carries them
# on. The rigid-body sweep in 1_Codes/dis_sweep_results.json shows spatial
# propagation costs a factor of six in strain noise at 240 iterations while
# the photometric residual is, if anything, slightly better without it, so it
# is now an explicit switch rather than an unstated default.
DIS_SPATIAL_PROPAGATION = os.environ.get(
    "NEWEXP_DIS_SPATPROP", "1") not in ("0", "false", "False")
DIS_MEAN_NORMALIZATION = os.environ.get(
    "NEWEXP_DIS_MEANNORM", "1") not in ("0", "false", "False")


def alpha_key(a):
    """Dataset key for one alpha - same convention as alpha_config.alpha_key:
    the iteration count enters the key only when it differs from the
    production value, so a higher-iteration "locked" reference run (mirroring
    the old experiment's a100_it240) gets its own key rather than silently
    overwriting the production sweep."""
    key = f"a{a}" if DIS_ITERATIONS == PROD_DIS_ITERATIONS \
        else f"a{a}_it{DIS_ITERATIONS}"
    if not DIS_SPATIAL_PROPAGATION:
        key += "_nosp"       # never collides with an existing dataset
    if not DIS_MEAN_NORMALIZATION:
        key += "_nomn"
    return key


DATASET_KEYS = [alpha_key(a) for a in ALPHAS]
# No "REF" key here: unlike alpha_study (which compares against a colleague's
# separate reference dataset), this specimen has no external reference
# dataset - every key is this specimen's own images at a different alpha.

BAD_FRAMES = {}   # key -> [1-based frame indices]; empty until 02's outputs
                   # are inspected and a real artifact is found (mirrors
                   # alpha_config.BAD_FRAMES; 2_elastoplasticity_corrected.py
                   # found no bad frames needed interpolation for this
                   # specimen - temporal Gaussian smoothing was enough).

# --------------------------------------------------------------------------
# DIS optical flow settings - identical for every alpha (only alpha varies),
# copied from 1_generate_dic_corrected.py / DIC_eps's actual call-site values
# (NOT its unused function defaults alpha=10, epsilon=0.05).
# --------------------------------------------------------------------------
DIS_DELTA = 1
DIS_GAMMA = 0
DIS_EPSILON = 0.002
DIS_PATCH_SIZE = 4
DIS_PATCH_STRIDE = 1
DIS_FINEST_SCALE = 0

# --------------------------------------------------------------------------
# Source images - THIS specimen's own data, never the old experiment's
# --------------------------------------------------------------------------
NEW_EXP_ROOT = os.environ.get(
    "NEWEXP_SRC_ROOT",
    os.path.join("data", "new_experiment", "source"))
RB_SP2_DIR = os.path.join(NEW_EXP_ROOT, "rb_sp2")
IMG_DIR = os.path.join(NEW_EXP_ROOT, "data", "26-03-26_11h26", "cam1_imgs")
MACRO_RECORD = os.path.join(
    NEW_EXP_ROOT, "data", "26-03-26_11h26", "data_labjack.csv")
REF_IMG = os.path.join(RB_SP2_DIR, "ref_2.tif")
MASK_IMG = os.path.join(RB_SP2_DIR, "mask.png")

# Already-computed production DIC run for THIS specimen (alpha=100,
# iterations=30, from 1_generate_dic_corrected.py, 2551 frames, ~3.9h of
# compute already spent) - registered as the source for alpha_key(100) at
# PROD_DIS_ITERATIONS so it never needs recomputing. This is new-experiment
# data, not old-experiment data, and is therefore not the forbidden 3_data.
EXISTING_TRACTION_DIR = os.path.join(RB_SP2_DIR, "traction")

# Flow-field crop, applied to the OUTPUT of DIS (not the input images) -
# same fixed window 1_generate_dic_corrected.py already uses for this
# specimen, giving the same (1880, 1705) shape as the existing traction/ and
# results/ arrays.
CROP_ROWS = slice(560, 2440)
CROP_COLS = slice(1380, 3085)

# --------------------------------------------------------------------------
# Usable temporal window - loading phase only, EXCLUDING the pre-load
# plateau (frames 0-757) and the unloading tail (frames >=2295). Determined
# in 2_elastoplasticity_corrected.py from the mean-U/mean-V curves for THIS
# specimen. Applied as files_U[FRAME_START:FRAME_END] on the 0-based, sorted
# U_k.npy list (i.e. python slice semantics), giving N_frames = 1537.
# --------------------------------------------------------------------------
FRAME_START = 758
FRAME_END = 2295

# 01 names its outputs U_k.npy / V_k.npy with k the 1-BASED position of the
# image in the sorted cam1_imgs list, so frame number k sits at position
# k-1 and the loading window [FRAME_START:FRAME_END) in positions is exactly
# the frame NUMBERS FRAME_START+1 .. FRAME_END. Selecting by position is
# only correct when every frame of the acquisition has been correlated;
# selecting by number is correct either way and gives the identical answer
# when they are. Everything downstream selects by number, so a DIC run
# restricted to the loading window (NEWEXP_WINDOW_ONLY=1 in 01) stays
# readable by 02 and 03 instead of silently shifting every index.
WINDOW_FIRST_FRAME = FRAME_START + 1          # inclusive, 1-based
WINDOW_LAST_FRAME = FRAME_END                 # inclusive


def in_window(frame_number):
    return WINDOW_FIRST_FRAME <= frame_number <= WINDOW_LAST_FRAME


def frame_numbers_of(files):
    return [int(re.findall(r"\d+", f)[0]) for f in files]


def window_files(files):
    """Keep the U_k.npy / V_k.npy whose k is inside the loading window."""
    return [f for f, n in zip(files, frame_numbers_of(files)) if in_window(n)]

# --------------------------------------------------------------------------
# Yield / hardening identification constants (this specimen)
# --------------------------------------------------------------------------
# Physical elastic modulus measured from this specimen's extensometer record
# over 2e-4--6e-4 axial strain (derive_macro_parameters.py).  This is the
# modulus used by the constitutive FEM.  It is deliberately NOT replaced by
# the 176 GPa "effective E" inferred from DIC eps_vM: that number is a
# measurement-transfer calibration, not a material property, and varies with
# the DIC processing.
E_FEM = 193397.86038269
E_MOD = E_FEM             # compatibility alias for the existing FEM driver
NU = 0.3

# Yield is an effective onset descriptor: each pixel's DIC-detected onset
# frame is mapped to the simultaneously measured nominal machine stress.
# Applying 3G to the TOTAL DIC strain at a 0.02%-offset crossing incorrectly
# treats the imposed 2e-4 plastic offset as elastic and adds about 45 MPa.
YIELD_STRESS_METHOD = "macro_stress_at_dic_onset"
MACRO_WIDTH_MM = 7.00
MACRO_THICKNESS_MM = 1.93
MACRO_AREA_MM2 = MACRO_WIDTH_MM * MACRO_THICKNESS_MM
MACRO_SMOOTH_SIGMA = 20
SLOP_REF = 1.130e-05     # 0.02%-anchored already - CONSISTENT with the
                         # ONSET_METHOD='offset' rule below, so it is
                         # deliberately unchanged.  See module
                         # docstring for the full derivation and the
                         # +0.94% comparison against the original 1.119e-05
# ---------------------------------------------------------------------------
# Macro Ludwik pair, REPLACED 2026-09-14.  See the block comment below.
# ---------------------------------------------------------------------------
# WAS: K_SCALE = 324.0504, N_EXP = 0.077767
#   Those came from 4_ludwik_fit_corrected.py, which regressed
#       sigma = K * ep**n                      (pure Hollomon, NO yield term)
#   in log-log space against the total measured stress.  The solver, however,
#   evaluates
#       sigma_flow = sigma_y_local + K_local * ep**n_exp   (additive Ludwik)
#   and therefore added the yield level a second time.  Measured consequence:
#   at ep = 0.004 the law returned 381 MPa where the record says 211 MPa.
#   Documented in 5_Claude_Analysis/global_hardening_audit_20260913.
#
# NOW: the additive-Ludwik refit of the SAME record, with sigma0 held at the
#   0.02 % proportional limit so that the macro intercept and the per-pixel
#   yield map share one onset definition (see ONSET_METHOD below):
#
#       sigma = 172.3 + 332.9 * ep**0.4165        full 3 % record
#
#   Why sigma0 is fixed rather than fitted: with sigma0 free the intercept
#   lands near 140 MPa, which is a regression parameter and not a yield
#   stress.  Adding the macro hardening on top of a per-pixel PROPORTIONAL
#   LIMIT while the macro law was built on a free intercept is what made the
#   reconstructed microscale stress sit ~27 MPa high.
#
#   Why 0.02 % and not Rp0.2: refitting at each candidate sigma0 and reading
#   off the exponent,
#       sigma0   80 -> n 0.125     sigma0  155 -> n 0.303
#       sigma0  120 -> n 0.186     sigma0 172.3 -> n 0.4165   <- physical
#                                  sigma0 199.3 -> n 0.850    (Rp0.2)
#   only an onset near 170-180 MPa puts n in the 0.4-0.5 band that 316L
#   actually has, and the stress RMSE is best there too.  Rp0.2 = 199.3 MPa
#   remains the number to QUOTE as the engineering yield strength; it is not
#   the number that belongs in sigma0.
# All four come from derive_macro_parameters.py in this folder; re-run it to
# regenerate macro_parameters.json / .txt / .png and check these against it.
SIGMA0_MACRO_MPA = 172.3  # 0.02 % proportional limit of this record
K_SCALE = 332.8          # macro Ludwik K (MPa), additive fit at that sigma0
N_EXP = 0.4164           # macro Ludwik n,        additive fit at that sigma0
RP02_MPA = 199.3         # engineering yield strength, for reporting only

# ---------------------------------------------------------------------------
# Elastic-fit window and earliest admissible onset frame.  ADDED 2026-09-16.
# ---------------------------------------------------------------------------
# The offset detector needs two different frame indices, and until now this
# file supplied only one, MIN_ELASTIC_FRAMES = 30, which was used for both:
# as the END of the window over which each pixel's elastic line is fitted,
# and as the earliest frame at which onset may be declared.  Those are not
# the same quantity and one value served neither.
#
# The value 30 came from assuming the elastic branch lasts about 40 frames,
# obtained by spreading the 3.3 % total strain uniformly over the 1537 frames
# of the loading window.  The load record says otherwise.  Window frame 1 is
# position 759 of the sorted acquisition, image 1235_29.986.tiff, and the
# record reads:
#
#     window frame    t (s)      F (N)   sigma (MPa)   extensometer strain
#                1    29.99      -72.5          -5.4             -5.4e-05
#               15    30.54        4.0           0.3             -2.3e-05
#               27    31.01      261.8          19.4              6.8e-05
#              122    34.76     1867.2         138.2              7.0e-04
#              163    36.38     2314.6         171.3              1.1e-03
#
# The load train takes up its slack over frames 1-15; the cell reads -72 N
# there and the extensometer does not move.  The elastic branch runs from
# frame 15 to frame 163, where the stress reaches the 172.3 MPa proportional
# limit: 148 frames, not 40.  Fitting the elastic line over frames 3-30
# therefore fitted dead time.  Measured consequence on a100_it240: the median
# fitted slope was NEGATIVE, -6.6e-07 strain/frame; 71.7 % of pixels fell
# through the slope<=0 guard onto the floor; their yield stress was read at
# frame 30, where the specimen carries 23 MPa; and the identified yield
# median came out at 17.3 MPa against a macroscopic 172.3 MPa.
#
# The window below sits inside the elastic branch with margin at both ends.
# It opens once the slack is gone and the specimen carries 19 MPa, and closes
# at 138 MPa, 80 % of the proportional limit, so that pixels yielding earlier
# than the bulk do not contaminate their own elastic fit.  Across those 96
# frames the macroscopic strain gain is 6.3e-04 against a per-pixel temporal
# scatter of 1.3e-05, a signal-to-noise ratio of about 44; on the old window
# the same ratio was 0.07.
#
# Both bounds are 1-based window frame numbers and both are inclusive.
ELASTIC_FIT_FIRST = int(os.environ.get("NEWEXP_ELASTIC_FIT_FIRST", 27))
ELASTIC_FIT_LAST = int(os.environ.get("NEWEXP_ELASTIC_FIT_LAST", 122))

# Earliest frame at which onset may be declared.  It sits immediately past
# the fit window: a departure from a line cannot be sought among the frames
# the line was fitted through.  A pixel following the macroscopic curve gains
# the 2e-4 offset about 32 frames after the window closes, near frame 155,
# which is where the macroscopic proportional limit sits (frame 163).
MIN_ONSET_FRAME = int(os.environ.get("NEWEXP_MIN_ONSET_FRAME",
                                     ELASTIC_FIT_LAST + 1))

if not (3 <= ELASTIC_FIT_FIRST < ELASTIC_FIT_LAST < MIN_ONSET_FRAME):
    raise ValueError(
        "need 3 <= ELASTIC_FIT_FIRST < ELASTIC_FIT_LAST < MIN_ONSET_FRAME, "
        f"got {ELASTIC_FIT_FIRST}, {ELASTIC_FIT_LAST}, {MIN_ONSET_FRAME}")

# The R^2 detector's own onset floor, unchanged at 30 so that the results it
# produced before this date remain reproducible.  ONSET_METHOD selects which
# of the two floors 02 applies; see below.
MIN_ELASTIC_FRAMES = 30

# ---------------------------------------------------------------------------
# Yield-onset detector.  Added 2026-09-14.
# ---------------------------------------------------------------------------
# "r2"     the original R^2-peak detector (dectect_maximum_r2).  Kept so the
#          previous results stay reproducible.  On this specimen it fires very
#          early: the yield map it produced has a mean of 80.2 MPa against a
#          macroscopic onset of 155-172 MPa, and against 146 MPa for the
#          collaborator's independent identification of the same test.
#
# "offset" the plastic-strain offset rule, and the default from now on.  For
#          each pixel, fit the elastic line through its own strain history,
#          then declare onset at the first frame where the strain departs from
#          that line by more than ONSET_OFFSET.  This is the direct per-pixel
#          analogue of the macroscopic 0.02 % offset that now sets
#          SIGMA0_MACRO_MPA, so macro and micro finally share one definition -
#          which is the whole point, and was the thing that was wrong.
#
# ONSET_OFFSET is an EQUIVALENT PLASTIC strain offset, so 2.0e-4 = 0.02 %, the
# same number used on the macro curve.  No axial-to-equivalent conversion is
# applied: for the plastic part the two coincide to within the Evm formula's
# own plane-stress approximation.
ONSET_METHOD = os.environ.get("NEWEXP_ONSET_METHOD", "offset")
ONSET_OFFSET = float(os.environ.get("NEWEXP_ONSET_OFFSET", 2.0e-4))
if ONSET_METHOD not in ("offset", "r2"):
    raise ValueError(f"NEWEXP_ONSET_METHOD must be 'offset' or 'r2', got {ONSET_METHOD!r}")
ONSET_METHOD_CODE = 1 if ONSET_METHOD == "offset" else 0   # numba-friendly

# Which onset floor applies depends on the detector.  The offset rule uses
# MIN_ONSET_FRAME, immediately past its own elastic-fit window.  The R^2 rule
# keeps the 30 it was calibrated with, so its earlier results still reproduce.
ONSET_MIN_FRAME = (MIN_ONSET_FRAME if ONSET_METHOD == "offset"
                   else MIN_ELASTIC_FRAMES)

YIELD_SAFETY_FLOOR_MPA = float(os.environ.get("NEWEXP_YIELD_FLOOR", 1.0))
ONSET_CHECKPOINT_ROWS = int(os.environ.get("NEWEXP_ONSET_CHECKPOINT_ROWS", 100))
SAVE_EVM_CACHE = os.environ.get("NEWEXP_SAVE_EVM", "1") == "1"

# --------------------------------------------------------------------------
# FEM (in-house solver, fem_pixel.py) - same window/solver conventions as
# alpha_study, so results stay comparable in kind (not value) to the old
# experiment's own alpha/threshold studies.
# --------------------------------------------------------------------------
X_SIZE = float(os.environ.get("NEWEXP_X_SIZE", 0.5))
Y_SIZE = float(os.environ.get("NEWEXP_Y_SIZE", 0.5))
ELEMENT_SIZE = 0.001
SCALE_FACTOR = 3.45      # Calibrated rb_sp2 optical-camera scale, um/px.
PX_TO_MM = SCALE_FACTOR / 1000.0   # 0.00345 mm/px, same convention as alpha_config
ROW_OFFSET = int(os.environ.get("NEWEXP_ROW_OFFSET", 0))
COL_OFFSET = int(os.environ.get("NEWEXP_COL_OFFSET", 0))

NX = int(round(X_SIZE / ELEMENT_SIZE))
NY = int(round(Y_SIZE / ELEMENT_SIZE))

SOLVER = os.environ.get("NEWEXP_SOLVER", "fem_pixel")
N_INC = 20
HARDENING = "tabular"

# fem_pixel.py itself is generic (no dataset-specific data in it) - imported
# from the old experiment's alpha_study code tree rather than copied, so a
# fix there is picked up here too. This is CODE, not the forbidden DATA -
# see the governing-constraint note at the top of this file.
FEM_CODE_DIR = os.environ.get(
    "NEWEXP_FEM_CODE_DIR", r"D:/WORK/1_Active/Case_1_Pixel/alpha_study/1_codes")

HIST_N = 10
HIST_FRACS = [k / HIST_N for k in range(1, HIST_N + 1)]

# The one experimental input with no recalculated counterpart - the single
# permitted exception to the governing constraint.
MACRO_CURVE = os.environ.get(
    "NEWEXP_MACRO_CURVE",
    r"D:/WORK/2_Legacy/3_data/stress_strain.npy")

# --------------------------------------------------------------------------
# Output tree - explicitly on D:, per instruction, NOT derived as a sibling
# of CODE_DIR (code stays under the Claude/2_reidentification_new_experiment
# tree; data goes to the dedicated D: location).
# --------------------------------------------------------------------------
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get(
    "NEWEXP_DATA_DIR", os.path.join("data", "new_experiment"))
FIG_DIR = os.path.join(DATA_DIR, "3_figures")
DISFLOW_DIR = os.path.join(DATA_DIR, "disflow")
YH_DIR = os.path.join(DATA_DIR, "yield_hardening")
FEM_DIR = os.path.join(DATA_DIR, "fem")
for _d in (DATA_DIR, FIG_DIR, DISFLOW_DIR, YH_DIR, FEM_DIR):
    os.makedirs(_d, exist_ok=True)


def uv_dirs(key):
    """(U_dir, V_dir) for any dataset key."""
    parse_key(key)   # validate, fail loudly if malformed
    base = os.path.join(DISFLOW_DIR, key)
    return os.path.join(base, "U"), os.path.join(base, "V")


def yh_dir(key):
    d = os.path.join(YH_DIR, key)
    os.makedirs(d, exist_ok=True)
    return d


def fem_path(key):
    return os.path.join(FEM_DIR, f"fem_{key}_{SOLVER}_{NX}x{NY}.npz")


KEY_RE = re.compile(r"^a(\d+)(?:_it(\d+))?(?:_nosp)?(?:_nomn)?$")


def parse_key(key):
    """(alpha, iterations) from a dataset key."""
    m = KEY_RE.match(key)
    if not m:
        raise ValueError(f"unrecognised dataset key: {key!r}")
    return int(m.group(1)), int(m.group(2) or PROD_DIS_ITERATIONS)


# --------------------------------------------------------------------------
# Shared helpers - byte-identical to alpha_config.py / thresh_config.py
# where the same operation is needed, so nothing drifts between studies.
# --------------------------------------------------------------------------
def get_sorted_npy_files(folder, prefix):
    """Sorted by the NUMERIC frame index, not lexicographically."""
    files = [f for f in os.listdir(folder) if re.match(rf"{prefix}_\d+\.npy$", f)]
    return sorted(files, key=lambda x: int(re.findall(r"\d+", x)[0]))


def frame_indices(folder, prefix):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"dataset folder not found: {folder}")
    files = get_sorted_npy_files(folder, prefix)
    if not files:
        raise FileNotFoundError(f"no {prefix}_*.npy files in {folder}")
    return [int(re.findall(r"\d+", f)[0]) for f in files]


def crop_center(array, n_rows, n_cols, row_off=None, col_off=None):
    row_off = ROW_OFFSET if row_off is None else row_off
    col_off = COL_OFFSET if col_off is None else col_off
    start_row = (array.shape[0] - n_rows) // 2 + row_off
    start_col = (array.shape[1] - n_cols) // 2 + col_off
    if (start_row < 0 or start_col < 0 or
            start_row + n_rows > array.shape[0] or
            start_col + n_cols > array.shape[1]):
        raise ValueError(f"crop window out of bounds: offsets ({row_off},{col_off}) "
                         f"with size ({n_rows},{n_cols}) in array {array.shape}")
    return array[start_row:start_row + n_rows, start_col:start_col + n_cols]


def pad_to(a, n_rows, n_cols):
    pr = max(0, n_rows - a.shape[0])
    pc = max(0, n_cols - a.shape[1])
    if pr or pc:
        a = np.pad(a, ((0, pr), (0, pc)), mode="edge")
    return a


def fill_nan_iter_4nbr(a, max_iter=64):
    a = a.astype(float, copy=True)
    for _ in range(max_iter):
        nan = np.isnan(a)
        if not nan.any():
            break
        up    = np.pad(a, ((1, 0), (0, 0)), constant_values=np.nan)[:-1, :]
        down  = np.pad(a, ((0, 1), (0, 0)), constant_values=np.nan)[1:,  :]
        left  = np.pad(a, ((0, 0), (1, 0)), constant_values=np.nan)[:, :-1]
        right = np.pad(a, ((0, 0), (0, 1)), constant_values=np.nan)[:, 1:]
        stack  = np.stack([up, down, left, right], axis=0)
        sums   = np.nansum(stack, axis=0)
        counts = np.sum(np.isfinite(stack), axis=0)
        means  = sums / np.where(counts == 0, 1, counts)
        a[nan & (counts > 0)] = means[nan & (counts > 0)]
    return a


def replace_nan_with_neighbors(a, max_iter=64):
    """Iterative 8-neighbour NaN fill (same as el_K scripts / alpha_study's
    03_yield_hardening_alpha.py)."""
    a = a.astype(float, copy=True)
    for _ in range(max_iter):
        nan = np.isnan(a)
        if not nan.any():
            break
        neighbors = [
            np.pad(a, ((1, 0), (1, 0)), constant_values=np.nan)[:-1, :-1],
            np.pad(a, ((1, 0), (0, 0)), constant_values=np.nan)[:-1, :],
            np.pad(a, ((1, 0), (0, 1)), constant_values=np.nan)[:-1, 1:],
            np.pad(a, ((0, 0), (1, 0)), constant_values=np.nan)[:, :-1],
            np.pad(a, ((0, 0), (0, 1)), constant_values=np.nan)[:, 1:],
            np.pad(a, ((0, 1), (1, 0)), constant_values=np.nan)[1:, :-1],
            np.pad(a, ((0, 1), (0, 0)), constant_values=np.nan)[1:, :],
            np.pad(a, ((0, 1), (0, 1)), constant_values=np.nan)[1:, 1:],
        ]
        stack = np.stack(neighbors, axis=0)
        sums = np.nansum(stack, axis=0)
        counts = np.sum(np.isfinite(stack), axis=0)
        means = sums / np.where(counts == 0, 1, counts)
        a[nan & (counts > 0)] = means[nan & (counts > 0)]
    return a


def _evm_ps(exx, exy, eyy):
    """Plane-stress eps_vM from strain COMPONENTS."""
    return np.sqrt((4.0 / (9.0 * (1.0 - NU + NU**2))) *
                   ((exx**2 + eyy**2 - exx * eyy) + 3.0 * (1.0 - NU)**2 * exy**2))


def disp_strain_components(u_x_px, u_y_px):
    dU0, dU1 = np.gradient(u_y_px)
    dV0, dV1 = np.gradient(u_x_px)

    def n2c(a):
        return 0.25 * (a[:-1, :-1] + a[1:, :-1] + a[:-1, 1:] + a[1:, 1:])
    return n2c(dV0), n2c(0.5 * (dV1 + dU0)), n2c(dU1)


def evm_from_disp_px(u_x_px, u_y_px):
    exx, exy, eyy = disp_strain_components(u_x_px, u_y_px)
    return _evm_ps(exx, exy, eyy)


def deviatoric_strain_plane_stress(exx, eyy, exy, nu):
    ezz = -nu / (1.0 - nu) * (exx + eyy)
    trace = exx + eyy + ezz
    exx_dev = exx - trace / 3.0
    eyy_dev = eyy - trace / 3.0
    ezz_dev = ezz - trace / 3.0
    exy_dev = exy
    return exx_dev, eyy_dev, ezz_dev, exy_dev


def evm_from_edev(exx_dev, eyy_dev, ezz_dev, exy_dev):
    j2 = exx_dev**2 + eyy_dev**2 + ezz_dev**2 + 2.0 * exy_dev**2
    return np.sqrt((2.0 / 3.0) * j2)


def sigma_vm_from_evm(evm, E, nu):
    """Elastic conjugate of the deviatoric eps_vM: sigma_vm = 3*G*eps_vM.
    (Algebraically identical to the plane-stress Hooke-strain formula under
    the same plane-stress assumption - verified to ~1e-15 relative in the
    old experiment's alpha_study; only this branch is kept.)"""
    g = E / (2.0 * (1.0 + nu))
    return 3.0 * g * evm


if __name__ == "__main__":
    print(f"code dir        : {CODE_DIR}")
    print(f"data dir        : {DATA_DIR}")
    print(f"image source    : {IMG_DIR}")
    print(f"ref image       : {REF_IMG}  exists={os.path.isfile(REF_IMG)}")
    print(f"mask image      : {MASK_IMG}  exists={os.path.isfile(MASK_IMG)}")
    print(f"existing prod   : {EXISTING_TRACTION_DIR}  "
          f"exists={os.path.isdir(EXISTING_TRACTION_DIR)}")
    print(f"alphas          : {ALPHAS}")
    print(f"dataset keys    : {DATASET_KEYS}")
    print(f"frame window    : [{FRAME_START}:{FRAME_END}]  "
          f"(N={FRAME_END-FRAME_START})")
    print(f"onset detector  : {ONSET_METHOD!r}, offset {ONSET_OFFSET:.2e}")
    print(f"elastic fit     : window frames {ELASTIC_FIT_FIRST}-{ELASTIC_FIT_LAST} "
          f"(inclusive), onset floor {ONSET_MIN_FRAME}")
    print(f"crop            : rows {CROP_ROWS}, cols {CROP_COLS}")
    print(f"E_FEM={E_FEM} MPa, nu={NU}, SLOP_REF={SLOP_REF:.4e}, "
          f"K_SCALE={K_SCALE:.4f} MPa, N_EXP={N_EXP:.6f}")
    print(f"yield stress    : {YIELD_STRESS_METHOD} from {MACRO_RECORD}")
    print(f"pixel scale     : {SCALE_FACTOR} um/px  (calibrated rb_sp2 optical camera)")
