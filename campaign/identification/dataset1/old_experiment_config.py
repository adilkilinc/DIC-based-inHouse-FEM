"""Dataset 1 configuration shared by the HPC DIC and spatial-alpha stages."""
from pathlib import Path
import os
import re
import numpy as np

ROOT = Path(os.environ.get("OLDEXP_ROOT", "data/old_experiment"))
SOURCE_DIR = Path(os.environ.get("OLDEXP_SOURCE_DIR", ROOT / "1_experimental_data"))
DATA_DIR = Path(os.environ.get("OLDEXP_DATA_DIR", ROOT / "2_data"))
DISFLOW_DIR = DATA_DIR / "disflow"
YH_DIR = DATA_DIR / "yield_hardening"
REF_IMG = str(SOURCE_DIR / "ref.tif")
MASK_IMG = str(SOURCE_DIR / "mask.png")
IMG_DIR = str(SOURCE_DIR / "traction")
MANIFEST = Path(os.environ.get("OLDEXP_MANIFEST", ROOT / "4_hpc" / "frames_manifest.txt"))

ALPHAS = [10, 20, 50, 80, 100, 150, 200, 500]
PROD_ALPHA = 100
DIS_ITERATIONS = int(os.environ.get("OLDEXP_DIS_ITERATIONS", "240"))
DIS_FINEST_SCALE = 0
DIS_PATCH_SIZE = 4
DIS_PATCH_STRIDE = 1
DIS_DELTA = 1.0
DIS_GAMMA = 0.0
DIS_EPSILON = 0.002
DIS_MEAN_NORMALIZATION = True
DIS_SPATIAL_PROPAGATION = True
CROP_ROWS = slice(400, 4000)
CROP_COLS = slice(1211, 4311)
FIELD_SHAPE = (3600, 3100)

E_MOD = 205000.0
NU = 0.30
K_SCALE = 396.0
N_EXP = 0.245
PIXEL_UM = 1.84
PX_TO_MM = PIXEL_UM / 1000.0
NX = NY = 500

KEY_RE = re.compile(r"^a([0-9]+(?:\.[0-9]+)?)_it(\d+)$")


def alpha_key(alpha):
    a = int(alpha) if float(alpha).is_integer() else alpha
    return f"a{a}_it{DIS_ITERATIONS}"


def parse_key(key):
    m = KEY_RE.match(key)
    if not m:
        raise ValueError(f"invalid Dataset 1 alpha key: {key}")
    a = float(m.group(1))
    return (int(a) if a.is_integer() else a), int(m.group(2))


def uv_dirs(key):
    parse_key(key)
    base = DISFLOW_DIR / key
    return str(base / "U"), str(base / "V")


def crop_center(array, n_rows, n_cols):
    r0 = (array.shape[0] - n_rows) // 2
    c0 = (array.shape[1] - n_cols) // 2
    return array[r0:r0+n_rows, c0:c0+n_cols]


def evm_from_disp_px(u_px, v_px):
    """Dataset 1's established plane-stress equivalent-strain definition."""
    d_u0, d_u1 = np.gradient(v_px)
    d_v0, d_v1 = np.gradient(u_px)
    exx, eyy = d_v0, d_u1
    exy = 0.5 * (d_v1 + d_u0)
    return np.sqrt((4.0 / (9.0 * (1.0 - NU + NU**2))) *
                   ((exx**2 + eyy**2 - exx*eyy) +
                    3.0 * (1.0 - NU)**2 * exy**2))
