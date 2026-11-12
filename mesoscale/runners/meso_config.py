# -*- coding: utf-8 -*-
"""
meso_config.py - single place to configure the whole mesoscale full-domain test.

Mirrors Microscale/1_codes/test_10x10/test_config.py: every script in this
folder imports its paths, its material constants and its PREPROCESSING from
here, so the Abaqus chain and the in-house FEM chain are guaranteed to solve
the identical problem.

The preprocessing (`load_mesoscale_inputs`) is a verbatim re-implementation of
mesoscale_runner.py + mesoscale_mesh_triangular.py.  Two quirks of the original
are reproduced ON PURPOSE, because the Abaqus .inp that already exists was
written with them and any "fix" here would silently change the problem:

  Q1  `average_stress` writes float means back into a copy of the (integer)
      segmentation array, so the per-grain yield / K maps are TRUNCATED to the
      segmentation dtype.  Copied verbatim; see the note in that function.
  Q2  the original computes the *Plastic table with jax (`jnp`), whose default
      dtype is float32 - that is why the .inp tables read like
      153.86424255371094.  We build the table in float32 for the same reason,
      then widen to float64.  Set TABLE_FLOAT32 = False to opt out.
  Q3  `mesoscale_runner.stress_assign` samples the yield/K maps with
      `stress_field[round(cx) - 1, round(cy) - 1]` and NO lower clamp, so a
      grain whose first-element centroid rounds to pixel 0 gets index -1 and
      NUMPY WRAPS IT to the far edge of the image - that grain is assigned a
      completely unrelated grain's yield curve.  Measured on moderate_twins and
      fine_twins: 18 of 1447 grains.  Reproducing the wraparound (the default,
      CLIP_CENTROID = False) makes the in-house tables match the existing .inp
      for 1447/1447 grains; clamping instead matches only 1429/1447.  The
      no-twins meshes have no node at exactly 0, so 0 grains are affected there
      and the switch changes nothing for them.

      This one is a genuine BUG in the original, not a harmless convention.
      Set CLIP_CENTROID = True to sample correctly - but then the .inp must be
      regenerated (MESO_FORCE_INP=1) and Abaqus re-run, or the two codes are
      solving different problems.

And one thing that is NOT a quirk to reproduce but a per-mesh FACT (see BC_TOL):
`MeshCreator` selects boundary nodes by EXACT float equality against hard-coded
constants (`y == 0`, `y == 3.1*1.84`, `x == 3.6*1.84`).  That works on the twins
meshes, whose boundary nodes sit exactly on those values, and fails completely
on the repaired no-twins meshes, whose boundary nodes are scattered ~3e-4 mm
around them.  Hence a per-case `bc_tol` rather than one global tolerance.

Data source, changed 2026-08-19: the identified maps and the DIC kinematics
now come from the Chapter 3 studies - alpha_study `a100_it240` (the locked
correlation setting) and threshold_study `T075` (the selected yield threshold)
- not from the pre-study `3_data` maps. See the block below for what that
invalidates.

Switch case with MESH_KEY below.  Nothing else needs editing.
"""
import hashlib
import os
import re

import numpy as np

# ── which mesoscale case ─────────────────────────────────────────────────────
# override from the shell with MESO_MESH_KEY without touching this file
MESH_KEY = os.environ.get("MESO_MESH_KEY", "fine_no_twins")

# MESO_THESIS_ROOT lets the whole tree be pointed somewhere else (another
# drive, a copy on the HPC, the self-test fixture) without editing paths.
# No default: the full data tree (segmentation maps, DIC fields, extracted
# grids) is not shipped in this repository - see mesoscale/mesh/README.md.
# Point MESO_CASES_DIR at mesoscale/mesh/ to run the two meshes that ARE
# shipped, or MESO_THESIS_ROOT at a full local tree.
THESIS   = os.environ.get("MESO_THESIS_ROOT", "")
MESO     = os.path.join(THESIS, "Case_1_Pixel/Claude/2_old_experiment/Mesoscale")
# The case folders moved under `Case_studies/old/` on 2026-08-23. Both
# layouts are accepted, first match wins, because a hard-coded path that no
# longer exists fails at the FIRST file open inside build_problem() - several
# steps into a chain, with a FileNotFoundError naming a mesh file rather than
# the move that caused it. Override outright with MESO_CASES_DIR.
_CASE_CANDIDATES = (
    os.path.join(MESO, "Case_studies/Heterogeneous_material"),
    os.path.join(MESO, "Case_studies/old/Heterogeneous_material"),
)
CASES = os.environ.get("MESO_CASES_DIR") or next(
    (p for p in _CASE_CANDIDATES if os.path.isdir(p)), _CASE_CANDIDATES[0])

# Two .inp files, two different roles - do not confuse them:
#
#   <case>_base.inp   MESH ONLY: nodes (UNSCALED), CPS3 elements, Face_* elsets.
#                     Straight out of the CAD/meshing chain (CRISTALX -> Salome
#                     repair -> meshio). This is the ONLY source of the mesh and
#                     of the grain partition, and it is READ-ONLY here. Both
#                     chains take it, through inp_io.build_problem().
#
#   <job_name>.inp    MESH + DATA: the same mesh with per-grain materials and
#                     the DIC boundary conditions written in. This is an OUTPUT
#                     (gen_inp_meso.py writes it, Abaqus runs it). Nothing reads
#                     a mesh from it; check_grain_mapping.py reads it back only
#                     to verify it still agrees with the base mesh.
#
# base_inp   : the mesh-only file           -> BASE_INP
# job_name   : the .inp / .odb / npy prefix -> JOB_INP, ODB_PATH
# inp_dir    : where both live and where Abaqus runs
# seg_file   : segmentation map whose labels define the grains
# grid_data  : folder holding the already-extracted pixel-grid npy (may be None)
# bc_tol     : edge-node selection tolerance, in scaled mm. NOT a free knob -
#              see the BC note in the module docstring. It is fixed per mesh by
#              which meshing path produced the base .inp, and the correct value
#              is the one that reproduces the existing job .inp's constrained
#              node set NODE-FOR-NODE (check_grain_mapping layer D is the test).
#              Verified: twins 1e-6 -> 1314 / 2072 identical; no-twins 0.005 ->
#              840 identical. Swapping them loses 356 nodes / gains 1665.
MESHES = {
    "fine_no_twins": dict(
        base_inp=os.path.join(CASES, "2_fine_mesh_no_twins/1_inp/fine_no_twins_base.inp"),
        inp_dir =os.path.join(CASES, "2_fine_mesh_no_twins/1_inp"),
        grid_data=os.path.join(CASES, "2_fine_mesh_no_twins/2_data"),
        seg_file=os.path.join(THESIS, "4_seg/GB_without_twins_4.npy"),
        job_name="fine_no_twins_50",
        bc_tol=0.005,     # repaired mesh, boundary nodes scattered ~3e-4 mm
        twins=False,      # segmentation family - selects the Hall-Petch fit
    ),
    "moderate_no_twins": dict(
        base_inp=os.path.join(CASES, "4_moderate_mesh_no_twins/1_inp/moderate_no_twins_base.inp"),
        inp_dir =os.path.join(CASES, "4_moderate_mesh_no_twins/1_inp"),
        grid_data=os.path.join(CASES, "4_moderate_mesh_no_twins/2_data"),
        seg_file=os.path.join(THESIS, "4_seg/GB_without_twins_4.npy"),
        job_name="moderate_no_twins_50",
        bc_tol=0.005,     # same meshing path as fine_no_twins
        twins=False,
    ),
    "fine_twins": dict(
        base_inp=os.path.join(CASES, "1_fine_mesh_twins/1_inp/fine_base.inp"),
        inp_dir =os.path.join(CASES, "1_fine_mesh_twins/1_inp"),
        grid_data=os.path.join(MESO, "3_data"),
        seg_file=os.path.join(THESIS, "4_seg/segmented_image.npy"),
        job_name="fine_twins_50",
        bc_tol=1e-6,      # boundary nodes EXACTLY on 0 / 3.1*1.84 / 3.6*1.84
        twins=True,
    ),
    # Added for the heterogeneous-material case set (fem/2_data/1_cases). This
    # is the fourth corner of the mesh x twins grid and was the only one of
    # Case_studies/Heterogeneous_material/1..4 with no entry here. Purely
    # additive: no existing key is touched, so nothing already run changes.
    "moderate_twins": dict(
        base_inp=os.path.join(CASES, "3_moderate_mesh_twins/1_inp/moderate_base.inp"),
        inp_dir =os.path.join(CASES, "3_moderate_mesh_twins/1_inp"),
        grid_data=os.path.join(CASES, "3_moderate_mesh_twins/2_data"),
        seg_file=os.path.join(THESIS, "4_seg/segmented_image.npy"),
        job_name="moderate_twins_50",
        bc_tol=1e-6,      # same meshing path as fine_twins
        twins=True,
    ),
}
if MESH_KEY not in MESHES:
    raise ValueError(f"MESH_KEY '{MESH_KEY}' not in {sorted(MESHES)}")

CASE      = MESHES[MESH_KEY]
BASE_INP  = CASE["base_inp"]
INP_DIR   = CASE["inp_dir"]
SEG_FILE  = CASE["seg_file"]
JOB_NAME  = CASE["job_name"]
GRID_DATA = CASE["grid_data"]
JOB_INP  = os.path.join(INP_DIR, JOB_NAME + ".inp")
ODB_PATH  = os.path.join(INP_DIR, JOB_NAME + ".odb")

FEM_TAG = f"fem_{JOB_NAME}"

# ── physical / material constants (identical to mesoscale_runner.py) ─────────
SCALE_FACTOR = 1.84          # raw mesh unit -> mm  (1 raw unit = 1000 px)
PX_TO_MM     = 0.00184       # DIC pixel -> mm; also m_per_pixel for the raster
E_MOD        = 205000.0      # MPa
NU           = 0.30
N_EXP        = 0.245
K_SCALE      = 396.0
EP_MAX       = 0.2
N_TABLE      = 50            # E_p = linspace(0, EP_MAX, N_TABLE)
EP_TABLE     = np.linspace(0.0, EP_MAX, N_TABLE)
# Edge-node selection tolerance, in scaled mm. PER CASE - see Q3 in the module
# docstring. Override for a one-off experiment with MESO_BC_TOL.
BC_TOL       = float(os.environ.get("MESO_BC_TOL", CASE.get("bc_tol", 0.005)))
TABLE_FLOAT32 = True         # see Q2 in the module docstring
CLIP_CENTROID = False        # see Q3: False reproduces the original's -1 wrap

# ═══════════════════════════ CASE-MATRIX VARIANT AXES ════════════════════════
# ADDED 2026-08-23 for fem/2_data/4_case_matrix. Purely additive: with every
# variable at its default this block is a no-op and every code path below
# behaves exactly as it did before, byte for byte. selftest_matrix.py proves
# that (it builds the same problem with and without the block engaged).
#
# The old Case_studies/old tree ran three FAMILIES of mesoscale cases that
# differ only in how the material and the boundary condition are built from the
# same DIC data and the same four meshes:
#
#   Heterogeneous_material          per-grain material, per-node DIC BC
#   Homogeneous_material_hetero_BC  ONE global-mean material, per-node DIC BC
#   Homogeneous_material_homo_BC    ONE global-mean material, per-EDGE mean BC
#
# and each family in two material laws (identified-from-DIC, and Hall-Petch).
# Rather than three copies of the preprocessing, that is expressed here as two
# orthogonal axes plus the BC model.
#
#   MESO_MATERIAL        dic        per-grain mean of the identified yield map
#                                   and of the identified K map (x K_SCALE).
#                                   THE DEFAULT - today's behaviour.
#                        hp_kmacro  Hall-Petch yield, sigma_y,j = HP_SIGMA0 +
#                                   HP_K / sqrt(d_eq,j), with a UNIFORM
#                                   macroscopic hardening coefficient K_MACRO.
#                                   This is the standalone Hall-Petch
#                                   description the old 5..8 folders ran.
#                        hp_kdic    Hall-Petch yield, but the per-grain
#                                   identified K is kept. Isolates the yield
#                                   description alone; a hybrid, not a
#                                   standalone model. Report it as such.
#
#   MESO_MATERIAL_SPACE  hetero     keep the per-grain values. THE DEFAULT.
#                        homog      collapse the whole map to ONE global mean,
#                                   so every grain gets the same curve. This is
#                                   the "Homogeneous_material" control: the
#                                   grain partition still meshes the domain, it
#                                   just no longer carries any contrast.
#
#   MESO_BC_MODEL        dic        prescribe the measured DIC displacement at
#                                   every boundary node. THE DEFAULT.
#                        edge_mean  prescribe ONE (ux, uy) per edge, the mean of
#                                   the DIC values over that edge's nodes. This
#                                   is the "homogeneous BC" control.
#
# Hall-Petch parameters, 316L, Hu (2025) Fig. 3.10, as used by the old runners:
#
#     with twins      sigma_el = 86 + 538 d^-0.5  (d in um, R2 = 0.94)
#     without twins   sigma_el = 74 + 522 d^-0.5  (d in um, R2 = 0.90)
#
# converted to MPa*mm^0.5 by dividing the slope by sqrt(1000). WHICH FIT
# APPLIES IS SET BY THE SEGMENTATION, not by the mesh density: the twins meshes
# resolve 1447 grains and the no-twins meshes 244, and the fit was measured
# separately on each. That is why MESHES carries `twins`.
MATERIAL_MODEL   = os.environ.get("MESO_MATERIAL", "dic")
MATERIAL_SPACE   = os.environ.get("MESO_MATERIAL_SPACE", "hetero")
BC_MODEL         = os.environ.get("MESO_BC_MODEL", "dic")

_MATERIAL_MODELS = ("dic", "hp_kmacro", "hp_kdic")
_MATERIAL_SPACES = ("hetero", "homog")
_BC_MODELS       = ("dic", "edge_mean")
if MATERIAL_MODEL not in _MATERIAL_MODELS:
    raise ValueError(f"MESO_MATERIAL '{MATERIAL_MODEL}' not in {_MATERIAL_MODELS}")
if MATERIAL_SPACE not in _MATERIAL_SPACES:
    raise ValueError(f"MESO_MATERIAL_SPACE '{MATERIAL_SPACE}' not in {_MATERIAL_SPACES}")
if BC_MODEL not in _BC_MODELS:
    raise ValueError(f"MESO_BC_MODEL '{BC_MODEL}' not in {_BC_MODELS}")

TWINS = bool(CASE.get("twins", "no_twins" not in MESH_KEY))

# (sigma0 MPa, k MPa*mm^0.5, provenance) keyed by whether twins are resolved
HP_FIT = {
    True:  (86.0, 538.0 / np.sqrt(1000.0),
            "Hu (2025) Fig. 3.10, WITH twins: 86 + 538 d^-0.5, R2 = 0.94"),
    False: (74.0, 522.0 / np.sqrt(1000.0),
            "Hu (2025) Fig. 3.10, WITHOUT twins: 74 + 522 d^-0.5, R2 = 0.90"),
}
HP_SIGMA0 = float(os.environ.get("MESO_HP_SIGMA0", HP_FIT[TWINS][0]))
HP_K      = float(os.environ.get("MESO_HP_K",      HP_FIT[TWINS][1]))
HP_SOURCE = HP_FIT[TWINS][2]
K_MACRO   = float(os.environ.get("MESO_K_MACRO", 380.0))   # Ludwik-Hollomon K

# The old homogenising step wrote `np.where(map > 0, global_mean, 0.0)`, which
# would leave a zero-yield pixel wherever the map happened to be zero - and a
# grain whose sampled centroid landed there would get sigma_y = 0 and yield at
# the first increment. In practice the original never hit that (the DIC map is
# strictly positive and the Hall-Petch background is sigma0), so the `where` was
# a no-op. We fill the WHOLE map with the global mean instead: same result where
# the original was well defined, no trapdoor where it was not.
#
# What the mean is taken OVER does matter. Label 0 in the segmentation is the
# grain-boundary / background phase, not a grain; averaging it in pulls the
# homogenised value toward whatever the boundary pixels happen to hold. Default
# is to exclude it. Set MESO_HOMOG_INCLUDE_BG=1 to reproduce the old behaviour;
# load_material_maps() prints both numbers either way, so the choice is never
# invisible.
HOMOG_INCLUDE_BG = os.environ.get("MESO_HOMOG_INCLUDE_BG", "0") == "1"

VARIANT = f"{MATERIAL_MODEL}-{MATERIAL_SPACE}-bc_{BC_MODEL}"
IS_DEFAULT_VARIANT = (MATERIAL_MODEL == "dic" and MATERIAL_SPACE == "hetero"
                      and BC_MODEL == "dic")


def variant_banner():
    """One block describing the variant, for every script's header."""
    L = [f"variant        : {VARIANT}"
         + ("   (DEFAULT - identical to the pre-matrix chain)"
            if IS_DEFAULT_VARIANT else "")]
    if MATERIAL_MODEL == "dic":
        L.append(f"  material law : identified from DIC, per grain "
                 f"(threshold {T_KEY}, K x {K_SCALE:g})")
    else:
        L.append(f"  material law : Hall-Petch, sigma_y = {HP_SIGMA0:.1f} + "
                 f"{HP_K:.3f}/sqrt(d_eq)  [{HP_SOURCE}]")
        L.append("  hardening    : " + ("uniform K = %.1f MPa (macroscopic)"
                                        % K_MACRO if MATERIAL_MODEL == "hp_kmacro"
                                        else "per-grain identified K (hybrid)"))
    L.append("  material space: " + ("per grain" if MATERIAL_SPACE == "hetero"
                                     else "ONE global mean for every grain"))
    L.append("  BC model     : " + ("per-node DIC displacement"
                                    if BC_MODEL == "dic"
                                    else "one mean (ux, uy) per edge"))
    return "\n".join(L)

# ── identified material maps and DIC kinematics ──────────────────────────────
# BOTH now come from the Chapter 3 studies, not from the pre-study `3_data`
# maps this file used to read:
#
#   kinematics : alpha_study, key `a100_it240` - the LOCKED correlation
#                setting, alpha = 100 with 240 variational iterations (Sec 3.3).
#                Previously `3_data/22_U_V_again_`, which was 30 iterations.
#   materials  : threshold_study, key `T075` - the SELECTED hard yield
#                threshold, T = 75 MPa (Sec 3.4), chosen by C1/C2 as the
#                smallest admissible threshold whose agreement reaches 95 % of
#                the best admissible agreement. Previously `el_thresh50.npy`,
#                i.e. T = 50, which the threshold study calls the legacy value.
#
# The two studies live on D: in the production tree; point them elsewhere with
# MESO_ALPHA_STUDY_DIR / MESO_THRESH_STUDY_DIR (the self-test does exactly
# that). MESO_SRC_KEY / MESO_YIELD_T switch to another key or threshold
# without editing this file.
#
# CONSEQUENCE, read before comparing against Abaqus: every existing mesoscale
# `<job>.inp` and `.odb` was built from the OLD maps. Until each `.inp` is
# regenerated (MESO_FORCE_INP=1) and Abaqus re-run, check_grain_mapping's
# layers C and D will warn - correctly - that the material and the boundary
# conditions have drifted. Layers A and B (mesh, grain partition) are
# unaffected, because neither depends on the DIC or the yield maps.
# Identification sweeps live outside this repository; set these only if you
# have them locally.
ALPHA_STUDY_ROOT = os.environ.get("MESO_ALPHA_STUDY_DIR", "")
THRESH_STUDY_ROOT = os.environ.get("MESO_THRESH_STUDY_DIR", "")

SRC_KEY  = os.environ.get("MESO_SRC_KEY", "a100_it240")
YIELD_T  = int(os.environ.get("MESO_YIELD_T", "75"))
T_KEY    = f"T{YIELD_T:03d}"
DATA_SET = f"{SRC_KEY}_{T_KEY}"

DIC_DIR    = os.path.join(ALPHA_STUDY_ROOT, "disflow", SRC_KEY)
YIELD_FILE = os.path.join(THRESH_STUDY_ROOT, "yield_hardening", T_KEY,
                          "yield_vm_thresh.npy")
K_FILE     = os.path.join(THRESH_STUDY_ROOT, "yield_hardening", T_KEY,
                          "K_coef_filled.npy")
DIC_BASELINE_N = 5           # frames 1..5 averaged as the DIC strain baseline


# The alpha_study tree keeps the frames in U/ and V/ SUBDIRECTORIES, where
# `3_data/22_U_V_again_` had them side by side in one folder. Everything that
# opens a frame goes through these two helpers so the layout is stated once.
def dic_u_path(frame):
    return os.path.join(DIC_DIR, "U", f"U_{frame}.npy")


def dic_v_path(frame):
    return os.path.join(DIC_DIR, "V", f"V_{frame}.npy")


def dic_frames():
    """Frame indices present in BOTH U/ and V/, sorted. Empty if unreachable."""
    def _idx(sub, prefix):
        d = os.path.join(DIC_DIR, sub)
        if not os.path.isdir(d):
            return set()
        out = set()
        for fn in os.listdir(d):
            m = re.match(rf"{prefix}_(\d+)\.npy$", fn)
            if m:
                out.add(int(m.group(1)))
        return out
    return sorted(_idx("U", "U") & _idx("V", "V"))


# The final frame is DISCOVERED, not hardcoded. The old `3_data` sequence ended
# at frame 40; the re-correlated sequence need not, and silently solving the
# wrong load level is the sort of error that looks like a physics result. Pin
# it with MESO_DIC_FRAME when reproducing a specific run.
_dic_frame_env = os.environ.get("MESO_DIC_FRAME")
if _dic_frame_env:
    DIC_FINAL_FRAME = int(_dic_frame_env)
    DIC_FRAME_SOURCE = "pinned by MESO_DIC_FRAME"
else:
    _fr = dic_frames()
    if _fr:
        DIC_FINAL_FRAME = _fr[-1]
        DIC_FRAME_SOURCE = (f"last of {len(_fr)} frames found in {DIC_DIR} "
                            f"({_fr[0]}..{_fr[-1]})")
    else:
        DIC_FINAL_FRAME = 40
        DIC_FRAME_SOURCE = ("FALLBACK - no frames visible, the study tree is "
                            "not reachable from here")

# ── output tree (single source of truth for every script here) ───────────────
TEST_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.environ.get("MESO_RESULTS_DIR",
                             os.path.join(TEST_ROOT, "results", MESH_KEY))
AB_OUT  = os.path.join(RESULTS_DIR, "abaqus")   # ODB exports (element + grid)
FEM_OUT = os.path.join(RESULTS_DIR, "fem")      # in-house FEM fields
CMP_OUT = os.path.join(RESULTS_DIR, "compare")  # diff tables + figures
CACHE   = os.path.join(RESULTS_DIR, "cache")    # rasteriser / preprocessing cache
for _d in (RESULTS_DIR, AB_OUT, FEM_OUT, CMP_OUT, CACHE):
    os.makedirs(_d, exist_ok=True)


def tag():
    return MESH_KEY


# ═════════════════════════ preprocessing helpers ═════════════════════════════

def fill_nan_iter_4nbr(a, max_iter=64):
    """Iterative 4-neighbour NaN fill - verbatim from mesoscale_runner.py."""
    a = a.astype(float, copy=True)
    for _ in range(max_iter):
        nan = np.isnan(a)
        if not nan.any():
            break
        up    = np.pad(a, ((1, 0), (0, 0)), mode="constant", constant_values=np.nan)[:-1, :]
        down  = np.pad(a, ((0, 1), (0, 0)), mode="constant", constant_values=np.nan)[1:,  :]
        left  = np.pad(a, ((0, 0), (1, 0)), mode="constant", constant_values=np.nan)[:, :-1]
        right = np.pad(a, ((0, 0), (0, 1)), mode="constant", constant_values=np.nan)[:, 1:]
        stack  = np.stack([up, down, left, right], axis=0)
        sums   = np.nansum(stack, axis=0)
        counts = np.sum(np.isfinite(stack), axis=0)
        means  = sums / np.where(counts == 0, 1, counts)
        a[nan & (counts > 0)] = means[nan & (counts > 0)]
    return a


def average_stress(segmented_image, array):
    """Per-grain mean of `array`, painted back onto the segmentation.

    VERBATIM from mesoscale_runner.py, including the dtype quirk (Q1): the
    result keeps `segmented_image`'s dtype, so when the segmentation is an
    integer array the grain means are truncated to whole MPa before they reach
    the *Plastic table.  The existing Abaqus .inp was written this way; do not
    "fix" it here without regenerating and re-running the Abaqus job.
    """
    segmented_image_copy = segmented_image.copy()
    unique_labels = np.unique(segmented_image_copy)
    for label in unique_labels:
        grain_pixels = array[segmented_image_copy == label]
        average_yield_stress = np.mean(grain_pixels)
        segmented_image_copy[segmented_image == label] = average_yield_stress
    return segmented_image_copy


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def source_stamp():
    """Short hash of everything the grain maps depend on.

    It goes in the cache filename. Without it, repointing YIELD_FILE / K_FILE
    at a different study would silently reuse the cache built from the old
    maps - the new data would be configured and never actually read, which is
    the worst possible failure mode here because nothing about the run would
    look wrong. Covers the identity of the three input files (path, size,
    mtime), the K unit scale and the float32 table switch.
    """
    parts = [DATA_SET, f"K{K_SCALE:g}", f"F32{int(TABLE_FLOAT32)}"]
    for p in (SEG_FILE, YIELD_FILE, K_FILE):
        try:
            st = os.stat(p)
            parts.append(f"{os.path.basename(p)}:{st.st_size}:{int(st.st_mtime)}")
        except OSError:
            parts.append(f"{os.path.basename(p)}:missing")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def load_field_maps():
    """Grain-averaged yield and K maps, on the DIC pixel grid.

    Returns (yield_average, K_average), both shaped like the segmentation.
    Cached to RESULTS_DIR/cache because `average_stress` loops over every label.
    The cache key carries source_stamp(), so changing the study, the threshold
    or K_SCALE invalidates it instead of silently returning the old maps.
    """
    cpath = os.path.join(CACHE, f"grain_maps_{MESH_KEY}_{source_stamp()}.npz")
    if os.path.isfile(cpath):
        d = np.load(cpath)
        return d["yield_average"], d["K_average"]

    segmented_image = np.load(SEG_FILE)
    yield_pixel = fill_nan_iter_4nbr(np.load(YIELD_FILE))
    K_pixel     = fill_nan_iter_4nbr(np.load(K_FILE) * K_SCALE)

    yield_average = average_stress(segmented_image, yield_pixel)
    K_average     = average_stress(segmented_image, K_pixel)

    np.savez_compressed(cpath, yield_average=yield_average, K_average=K_average)
    print(f"[inputs] grain maps cached -> {cpath}")
    return yield_average, K_average


def load_dic_displacements():
    """Prescribed-displacement source maps, in mm, on the DIC pixel grid.

    Orientation follows mesoscale_runner.py exactly:
        disp_y  <- U_<frame>.npy   (loading direction, mesh y, DOF 2)
        disp_x  <- V_<frame>.npy   (transverse,        mesh x, DOF 1)
    Indexed as [x_pixel, y_pixel].
    """
    disp_y = np.load(dic_u_path(DIC_FINAL_FRAME)) * PX_TO_MM
    disp_x = np.load(dic_v_path(DIC_FINAL_FRAME)) * PX_TO_MM
    if disp_x.shape != disp_y.shape:
        raise ValueError(f"U and V disagree in shape at frame "
                         f"{DIC_FINAL_FRAME}: {disp_y.shape} vs {disp_x.shape}")
    return disp_x, disp_y


# ═══════════════════ variant material maps and BC model ═════════════════════
# Everything below is reached only through load_material_maps() and
# apply_bc_model(); with the default variant both are thin pass-throughs.

def grain_equivalent_diameter(segmented_image):
    """(labels, d_eq_mm) for every label present in the segmentation.

    d_eq is the equivalent-circle diameter of the grain's PIXEL AREA:
        d_eq = 2 sqrt(A / pi),  A = n_pixels * PX_TO_MM^2

    Same quantity `skimage.measure.regionprops(...).area` gives the old runner,
    computed with a bincount instead: regionprops groups by LABEL, not by
    connected component, so the two agree exactly, and this does not put skimage
    on the critical path of the solve.
    """
    lab = np.asarray(segmented_image)
    if lab.min() < 0:
        raise ValueError("segmentation holds negative labels - "
                         "grain areas cannot be counted")
    counts = np.bincount(lab.ravel().astype(np.int64))
    labels = np.nonzero(counts)[0]
    area_mm2 = counts[labels] * PX_TO_MM ** 2
    return labels, 2.0 * np.sqrt(area_mm2 / np.pi)


def seg_background_label(segmented_image, n_grain=None, verbose=True):
    """Is label 0 in this segmentation a GRAIN, or the background phase?

    It cannot be assumed. The old Hall-Petch runner hard-coded "label 0 is
    background" and filled it with sigma0; that is right if the labels run
    1..N and catastrophic if they run 0..N-1, because then one real grain
    silently loses its Hall-Petch value. The two cases are indistinguishable
    from the array alone - both give a contiguous label set - so the question
    is settled by counting against the number of Face_* elsets in the mesh,
    which is the only independent statement of how many grains there are.

    Returns 0 when label 0 is background, None when every label is a grain.
    Override with MESO_SEG_BG_LABEL = 0 | none.
    """
    forced = os.environ.get("MESO_SEG_BG_LABEL")
    if forced is not None:
        return None if forced.strip().lower() in ("none", "", "-1") else int(forced)

    lab = np.asarray(segmented_image)
    has_zero = bool((lab == 0).any())
    if not has_zero:
        return None
    n_nonzero = int(len(np.unique(lab[lab != 0])))
    if n_grain is None:
        if verbose:
            print("[mat ] label 0 is present and the grain count is unknown "
                  "here - assuming it is BACKGROUND, as the original runner "
                  "did. Pin it with MESO_SEG_BG_LABEL if that is wrong.")
        return 0
    if n_nonzero == n_grain:
        if verbose:
            print(f"[mat ] label 0 is BACKGROUND: {n_nonzero} non-zero labels "
                  f"for {n_grain} Face_* elsets")
        return 0
    if n_nonzero == n_grain - 1:
        if verbose:
            print(f"[mat ] label 0 IS A GRAIN: {n_nonzero} non-zero labels for "
                  f"{n_grain} Face_* elsets, so the labels run 0..{n_grain-1}")
        return None
    if verbose:
        print(f"[mat ] [WARN] {n_nonzero} non-zero labels but {n_grain} Face_* "
              f"elsets - neither convention fits. Treating label 0 as "
              f"background (the original's assumption); check the "
              f"segmentation, and pin MESO_SEG_BG_LABEL to silence this.")
    return 0


def hallpetch_yield_map(segmented_image, sigma0=None, k_mm=None, bg=0):
    """Pixel map of the Hall-Petch yield stress, sigma0 + k / sqrt(d_eq).

    Every pixel of grain j carries grain j's value, so the downstream
    first-element-centroid sampling in inp_io picks up that grain's own number
    through exactly the same code path the identified maps use.

    `bg` is the background label (0 in the original, None when every label is a
    grain - see seg_background_label). Background pixels get sigma0: they have
    no grain and therefore no diameter.

    Returned as float64. Unlike the identified maps this one never passes
    through `average_stress`, so the integer-truncation quirk Q1 does NOT apply
    and the Hall-Petch curves keep their fractional MPa. That is a real
    difference between the two material laws, not an oversight.
    """
    sigma0 = HP_SIGMA0 if sigma0 is None else float(sigma0)
    k_mm   = HP_K      if k_mm  is None else float(k_mm)
    lab = np.asarray(segmented_image).astype(np.int64)
    labels, d_eq = grain_equivalent_diameter(lab)
    lut = np.full(int(lab.max()) + 1, sigma0, dtype=np.float64)
    grain = labels != bg if bg is not None else np.ones(len(labels), bool)
    lut[labels[grain]] = sigma0 + k_mm / np.sqrt(d_eq[grain])
    return lut[lab]


def _homogenise(m, segmented_image, bg=0):
    """Collapse a pixel map to its single global mean. Returns (map, value).

    The old code wrote `np.where(map > 0, mean, 0.0)`, which would leave a
    zero-yield pixel wherever the map happened to be zero. We fill the WHOLE
    map: identical where the original was well defined, no trapdoor where it
    was not.
    """
    m = np.asarray(m, dtype=np.float64)
    seg = np.asarray(segmented_image)
    if bg is None or HOMOG_INCLUDE_BG or not (seg == bg).any():
        vals = m
    else:
        vals = m[seg != bg]
    g = float(np.nanmean(vals))
    return np.full(m.shape, g, dtype=np.float64), g


def load_material_maps(verbose=True, n_grain=None):
    """The (yield, K) pixel maps this VARIANT wants, ready for sampling.

    With the default variant this is `load_field_maps()` and nothing else, so
    the cached grain maps and the resulting .inp are unchanged, byte for byte.

    `n_grain` is the number of Face_* elsets; it is only used to decide whether
    label 0 is a grain or the background (seg_background_label).
    """
    if MATERIAL_MODEL == "dic" and MATERIAL_SPACE == "hetero":
        return load_field_maps()

    if verbose:
        print(f"[mat ] variant {VARIANT}")

    K_dic = None
    if MATERIAL_MODEL in ("dic", "hp_kdic"):
        yield_dic, K_dic = load_field_maps()

    seg = np.load(SEG_FILE)
    bg = seg_background_label(seg, n_grain, verbose=verbose)

    if MATERIAL_MODEL == "dic":
        yield_map, K_map = yield_dic, K_dic
    else:
        yield_map = hallpetch_yield_map(seg, bg=bg)
        K_map = (K_dic if MATERIAL_MODEL == "hp_kdic"
                 else np.full(seg.shape, K_MACRO, dtype=np.float64))
        if verbose:
            labels, d_eq = grain_equivalent_diameter(seg)
            g = labels != bg if bg is not None else np.ones(len(labels), bool)
            print(f"[mat ] Hall-Petch  sigma0 {HP_SIGMA0:.1f} MPa   "
                  f"k {HP_K:.3f} MPa*mm^0.5   ({HP_SOURCE})")
            print(f"[mat ] {int(g.sum())} grains, d_eq "
                  f"{1000*d_eq[g].min():.2f} .. {1000*d_eq[g].max():.2f} um "
                  f"(mean {1000*d_eq[g].mean():.2f} um)  ->  sigma_y "
                  f"{yield_map[seg != bg].min() if bg is not None else yield_map.min():.1f}"
                  f" .. "
                  f"{yield_map[seg != bg].max() if bg is not None else yield_map.max():.1f} MPa")

    if MATERIAL_SPACE == "homog":
        y_h, gy = _homogenise(yield_map, seg, bg)
        K_h, gK = _homogenise(K_map, seg, bg)
        if verbose:
            _, gy_all = _homogenise(yield_map, seg, None)
            print(f"[mat ] HOMOGENISED  sigma_y = {gy:.4f} MPa   "
                  f"K = {gK:.4f} MPa")
            print(f"[mat ]   background {'INCLUDED' if (bg is None or HOMOG_INCLUDE_BG) else 'excluded'} "
                  f"in the mean; including it would give sigma_y = {gy_all:.4f} MPa")
        yield_map, K_map = y_h, K_h

    return yield_map, K_map


def apply_bc_model(bc_u, edges, bc_nodes, verbose=True):
    """Rewrite the prescribed displacements for the chosen BC model.

    `dic` returns `bc_u` untouched.

    `edge_mean` replaces every node's value on an edge by that edge's MEAN,
    reproducing HomoBCMaterialInserter. Two deliberate departures from the old
    class, both so the BC contrast measures only the BC:

      * the CONSTRAINED NODE SET is left exactly as the DIC model chose it.
        The old class also re-sorted left/right by y before dropping the first
        and last node, which changes WHICH nodes are constrained. A control that
        changes two things at once measures neither.
      * edges are applied bottom, top, left, right, so where a node appears in
        two edge lists the later one wins - the same order Abaqus resolves
        duplicate *Boundary cards in, and the order inp_io.write_inp emits.
    """
    if BC_MODEL == "dic":
        return bc_u, "per-node DIC displacement"

    bc_u = np.array(bc_u, dtype=np.float64, copy=True)
    pos = {int(n): k for k, n in enumerate(np.asarray(bc_nodes))}
    parts = []
    for e in ("bottom", "top", "left", "right"):
        idx = np.array([pos[int(n)] for n in edges[e] if int(n) in pos],
                       dtype=np.int64)
        if not len(idx):
            raise ValueError(f"edge '{e}' has no constrained node - the "
                             f"edge-mean BC would be undefined there")
        ux = float(bc_u[idx, 0].mean())
        uy = float(bc_u[idx, 1].mean())
        bc_u[idx, 0] = ux
        bc_u[idx, 1] = uy
        parts.append(f"{e} ({len(idx)}) ux {ux:+.6f} uy {uy:+.6f}")
    note = "edge means: " + " | ".join(parts)
    if verbose:
        print(f"[bc  ] {note}")
    return bc_u, note


def grain_material_table(centroids_px, yield_average, K_average):
    """Per-grain (sy0, K) and the 50-point *Plastic table Abaqus interpolates.

    `centroids_px` is the (n_grain, 2) integer pixel index of each grain's
    FIRST element centroid - the same sampling point mesoscale_runner.py uses
    via find_single_element_centroid + stress_assign, including the `-1`.

    Returns dict with:
        sy0   (n_grain,)      table[:, 0]
        K     (n_grain,)      hardening coefficient at the same pixel
        table (n_grain, 50)   sigma_y at EP_TABLE, float64 (built in float32
                              when TABLE_FLOAT32, matching the jax-written .inp)
    """
    ix = centroids_px[:, 0]
    iy = centroids_px[:, 1]
    sy0 = np.asarray(yield_average, dtype=np.float64)[ix, iy]
    Kg  = np.asarray(K_average,     dtype=np.float64)[ix, iy]

    if TABLE_FLOAT32:
        ep = EP_TABLE.astype(np.float32)
        table = (sy0.astype(np.float32)[:, None]
                 + Kg.astype(np.float32)[:, None] * np.power(ep[None, :], np.float32(N_EXP)))
        table = table.astype(np.float64)
    else:
        table = sy0[:, None] + Kg[:, None] * np.power(EP_TABLE[None, :], N_EXP)

    return dict(sy0=table[:, 0].copy(), K=Kg, table=table)


def svm_ps(s11, s22, s12):
    """Plane-stress von Mises stress."""
    return np.sqrt(s11**2 + s22**2 - s11 * s22 + 3.0 * s12**2)


def evm_ps(exx, exy, eyy, nu=NU):
    """Plane-stress von Mises equivalent strain (exy TENSORIAL).

    Manuscript Eq. (evmps) -- the SINGLE canonical definition used by the
    framework, microscale and mesoscale chapters.  Obtained by closing the
    out-of-plane strain with the isotropic plane-stress relation
    ezz = -nu/(1-nu) (exx+eyy) and then forming sqrt(2/3 * J2') of the
    deviatoric strain.  Identical (to machine precision) to the form
    implemented in the microscale identification code
    ``el_K_recalculate_corrected.deviatoric_strain_plane_stress`` +
    ``evm_from_edev``, which produced the sigma_y and K maps this module
    loads.  Do not substitute another definition here: the two must agree
    or the identified maps and the reported fields are on different scales.

    Note this is a TOTAL-strain plane-stress *proxy*.  The elastic ezz
    closure is exact only before yielding; after yielding it is a
    convention, not an identity.  It is not PEEQ -- use the incrementally
    accumulated PEEQ from the solver for plastic state.
    """
    return (2.0 / (3.0 * (1.0 - nu))) * np.sqrt(
        (1.0 - nu)**2 * (exx**2 + eyy**2 - exx * eyy + 3.0 * exy**2)
        + nu * (exx + eyy)**2)


def evm_ps_legacy_archive(exx, exy, eyy, nu=NU):
    """SUPERSEDED definition -- retained only to reproduce pre-2026-09-03 figures.

    This is the form that was hard-coded here (and in visual28) before the
    equation audit.  It is NOT equivalent to ``evm_ps``: both are homogeneous
    of degree one, so the ratio legacy/thesis depends on the strain STATE --
    1.02 uniaxial elastic, 1.08 uniaxial plastic, 0.79 pure shear, 0.61
    equibiaxial (nu=0.3).  It therefore distorts shear-dominated regions
    relative to uniaxial ones, i.e. exactly the localisation bands.

    DO NOT call this in any active analysis.
    """
    return np.sqrt((4.0 / (9.0 * (1.0 - nu + nu**2))) *
                   ((exx**2 + eyy**2 - exx * eyy) + 3.0 * (1.0 - nu)**2 * exy**2))
