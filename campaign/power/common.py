"""Paths, frozen-module access and field measures shared by the follow-up studies.

Roots (each can be overridden by the environment variable named):
  PIXEL_CAMPAIGN        pixel-scale campaign folder (read only)
  OLD_EXPERIMENT_ROOT   Dataset 1 tree (DIC folders, images, 2_codes)
  NEW_EXPERIMENT_ROOT   Dataset 2 output tree (new_work/out)

Equivalent strain: thesis Eq. (evmps), the printed definition of
evm_reconcile_20261005/evmfix.py, for DIC and finite-element fields alike.
Measures: Case_studies/_shared/field_measures.py (r, top-decile Dice, contrast,
relative RMS, with the RMS identity checked on every call).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent
FROZEN = Path(os.environ.get("PIXEL_CAMPAIGN", PKG.parent))
REPO = PKG.parents[1]
OLD = Path(os.environ.get("OLD_EXPERIMENT_ROOT", "data/old_experiment"))
NEW = Path(os.environ.get("NEW_EXPERIMENT_ROOT", "data/new_experiment"))
OUT = PKG / "out"
NU = 0.30


def followup() -> dict:
    return json.loads((PKG / "followup.json").read_text(encoding="utf-8"))


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def frozen_run_case():
    """run_case.py of the frozen campaign, with its output and map root moved to this package."""
    rc = load_module(FROZEN / "run_case.py", "frozen_run_case")
    rc.HERE = PKG
    return rc


def campaign_config(check_sha: bool = True) -> dict:
    """Frozen campaign.json with the materials and case families of followup.json."""
    p = FROZEN / "campaign.json"
    fu = followup()
    if check_sha and sha256(p) != fu["frozen_campaign_json_sha256"]:
        raise SystemExit(f"{p} differs from the file this package was built against")
    cfg = json.loads(p.read_text(encoding="utf-8"))
    for ds, mats in fu["materials"].items():
        clash = set(mats) & set(cfg["datasets"][ds]["materials"])
        if clash:
            raise SystemExit(f"material names already used by the frozen campaign: {sorted(clash)}")
        cfg["datasets"][ds]["materials"].update(mats)
    cfg["case_families"] = fu["case_families"]
    cfg["campaign"] = fu["package"]
    return cfg


RAW_FILES = ("K_coef_raw.npy", "K_flag.npy")


def locked_raw_available(dataset: str, key: str) -> bool:
    d = FROZEN / "0_identification" / dataset / key / "off_stress"
    return all((d / f).is_file() for f in RAW_FILES)


def rerun_raw_available(dataset: str, key: str) -> bool:
    d = PKG / "0_identification" / dataset / key / "off_stress"
    return all((d / f).is_file() for f in RAW_FILES + ("K_coef_filled.npy", "yield_macro_filled.npy"))


# ---------------------------------------------------------------- strain ---
def printed_tensor(exx, eyy, exy, nu=NU):
    """Thesis Eq. (evmps), tensor shear (evm_reconcile_20261005/evmfix.printed_tensor)."""
    s = exx + eyy
    q = (1.0 - nu) ** 2 * (exx * exx + eyy * eyy - exx * eyy + 3.0 * exy * exy) + nu * s * s
    return 2.0 / (3.0 * (1.0 - nu)) * np.sqrt(np.maximum(q, 0.0))


def block(a, f: int):
    a = np.asarray(a, np.float64)
    if f == 1:
        return a
    nx, ny = a.shape
    if nx % f or ny % f:
        raise ValueError(f"factor {f} does not divide {a.shape}")
    return a.reshape(nx // f, f, ny // f, f).mean(axis=(1, 3))


def dic_evm(uv_dir: Path, frame: int, shape, offset, factor: int, nu=NU):
    """Printed equivalent strain of one DIC frame on the analysis grid, block mean to the element size
    (field_metrics_20261014.dic_evm)."""
    U = np.asarray(np.load(Path(uv_dir) / "U" / f"U_{int(frame)}.npy", mmap_mode="r"), np.float64)
    V = np.asarray(np.load(Path(uv_dir) / "V" / f"V_{int(frame)}.npy", mmap_mode="r"), np.float64)
    e = printed_tensor(np.gradient(V, axis=0), np.gradient(U, axis=1),
                       (np.gradient(V, axis=1) + np.gradient(U, axis=0)) / 2.0, nu)
    ox, oy = offset
    nx, ny = shape
    return block(e[ox:ox + nx, oy:oy + ny], factor)


def fe_evm(case_dir: Path):
    """Printed equivalent strain of the final finite-element state (element strains; e12 is the
    engineering shear of the solver, evmfix.final_components)."""
    meta = json.loads((Path(case_dir) / "run_meta.json").read_text())
    outs = meta.get("outputs", {})
    comps = []
    for k in ("e11", "e22", "e12"):
        p = Path(outs.get(k, ""))
        if not p.is_file():
            p = Path(case_dir) / f"{meta['case_id']}_{k}.npy"
        comps.append(np.asarray(np.load(p), np.float64))
    return printed_tensor(comps[0], comps[1], 0.5 * comps[2], float(meta["nu"])), meta


# -------------------------------------------------------------- measures ---
def measures(a, b) -> dict:
    """Case_studies/_shared/field_measures.measures (standalone copy for the cluster)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    fa, fb = a.ravel(), b.ravel()
    r = float(np.corrcoef(fa, fb)[0, 1])
    aa, bb = a >= np.percentile(a, 90), b >= np.percentile(b, 90)
    dice = float(2 * np.count_nonzero(aa & bb) / max(np.count_nonzero(aa) + np.count_nonzero(bb), 1))
    c = float(a.std() / b.std())
    rms_b = float(np.sqrt(np.mean(fb * fb)))
    e = float(np.sqrt(np.mean((fa - fb) ** 2)) / rms_b)
    bias, sb, mb = float(a.mean() - b.mean()), float(b.std()), float(b.mean())
    ident = float(np.sqrt((bias * bias + sb * sb * (1 + c * c - 2 * r * c)) / (mb * mb + sb * sb)))
    if abs(ident - e) > 1e-9 * max(e, 1.0):
        raise SystemExit(f"[identity] RMS {e} differs from the decomposition {ident}")
    return dict(r=r, dice_top10=dice, contrast=c, rel_rms_pct=100 * e,
                centred_rms_over_sd=float(np.sqrt(max(1 + c * c - 2 * r * c, 0.0))),
                mean_ratio=float(a.mean() / mb), bias_over_dic_rms=bias / rms_b,
                uniform_rms_pct=100 * sb / rms_b)


def pearson(a, b) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def write_json(path: Path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path
