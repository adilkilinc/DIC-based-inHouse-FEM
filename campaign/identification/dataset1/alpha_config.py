"""Compatibility configuration for the established Dataset 1 identifier."""
from pathlib import Path
import os
import re
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import old_experiment_config as base

ALPHAS = base.ALPHAS
DIS_ITERATIONS = base.DIS_ITERATIONS
DATASET_KEYS = [base.alpha_key(a) for a in ALPHAS]
BAD_FRAMES = {key: [] for key in DATASET_KEYS}
E_MOD = base.E_MOD
NU = base.NU
SLOP_REF = 4.46e-05
K_SCALE = base.K_SCALE
YIELD_SAFETY_FLOOR_MPA = float(os.environ.get("OLDEXP_YIELD_FLOOR", "1.0"))
ONSET_CHECKPOINT_ROWS = int(os.environ.get("OLDEXP_ONSET_CHECKPOINT_ROWS", "100"))
SAVE_EVM_CACHE = os.environ.get("OLDEXP_SAVE_EVM", "0") == "1"
YH_DIR = base.YH_DIR


def uv_dirs(key):
    return base.uv_dirs(key)


def yh_dir(key):
    d = YH_DIR / key
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def get_sorted_npy_files(folder, prefix):
    files = [f for f in os.listdir(folder) if re.match(rf"{prefix}_[0-9]+[.]npy$", f)]
    return sorted(files, key=lambda x: int(re.findall(r"[0-9]+", x)[0]))
