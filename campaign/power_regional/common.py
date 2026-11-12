"""Isolated power-assignment extension of the locked Chapter 5 campaign."""
from pathlib import Path
import hashlib
import json
import os

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOCKED = Path(os.environ.get('PIXEL_CAMPAIGN', HERE.parent))
POWER = Path(os.environ.get('POWER_STUDY', LOCKED / 'power'))
MESHES = {'mnt': 'moderate_no_twins', 'mt': 'moderate_twins'}


def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write_json(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha256(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def fill4(a, max_iter=64):
    """Same residual-NaN fill as the locked regional material builders."""
    a = np.asarray(a, float).copy()
    for _ in range(max_iter):
        nan = np.isnan(a)
        if not nan.any():
            break
        stack = np.stack([
            np.pad(a, ((1, 0), (0, 0)), constant_values=np.nan)[:-1, :],
            np.pad(a, ((0, 1), (0, 0)), constant_values=np.nan)[1:, :],
            np.pad(a, ((0, 0), (1, 0)), constant_values=np.nan)[:, :-1],
            np.pad(a, ((0, 0), (0, 1)), constant_values=np.nan)[:, 1:]], axis=0)
        counts = np.isfinite(stack).sum(axis=0)
        means = np.nansum(stack, axis=0) / np.maximum(counts, 1)
        use = nan & (counts > 0)
        a[use] = means[use]
    if not np.isfinite(a).all():
        raise ValueError('Unresolved/nonfinite map entries after locked fill')
    return a


def label_means(labels, field):
    lab = np.asarray(labels, np.int64).ravel()
    cnt = np.bincount(lab)
    total = np.bincount(lab, weights=np.asarray(field, float).ravel(), minlength=len(cnt))
    return total / np.maximum(cnt, 1), cnt


def evm(E, nu=0.3):
    x, y, xy = E[..., 0], E[..., 1], 0.5 * E[..., 2]
    return 2 / (3 * (1 - nu)) * np.sqrt(np.maximum(
        (1 - nu)**2 * (x*x + y*y - x*y + 3*xy*xy) + nu*(x+y)**2, 0))


def svm(S):
    return np.sqrt(np.maximum(S[..., 0]**2 - S[..., 0]*S[..., 1] + S[..., 1]**2 + 3*S[..., 2]**2, 0))


def areas(coords, elements):
    t = coords[elements]
    return 0.5 * np.abs((t[:, 1, 0]-t[:, 0, 0])*(t[:, 2, 1]-t[:, 0, 1])
                        - (t[:, 2, 0]-t[:, 0, 0])*(t[:, 1, 1]-t[:, 0, 1]))


def baseline_dir(ds, short):
    base = LOCKED / ('meso' if ds == 'dataset1' else 'meso_d2')
    return base / '2_results' / ('id_' + short)
