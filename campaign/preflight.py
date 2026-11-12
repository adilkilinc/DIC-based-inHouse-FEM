"""Check the case matrix and every static input, then write or verify the lock.

Method-B maps and mesoscale materials do not exist before their jobs run,
so they are not in this lock: each case hashes what it reads into its
run_meta.json, and freeze.py checks those hashes once everything is done.
The Dataset 2 strain-history caches (19.7 GB each) are checked for presence
only.
"""
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
from run_case import cases, root_for

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def sha256(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-root"); ap.add_argument("--new-root")
    ap.add_argument("--write-lock", action="store_true")
    a = ap.parse_args()
    cfg = json.loads((HERE / "campaign.json").read_text(encoding="utf-8"))
    m = cases(cfg); errors = []
    r1 = root_for("dataset1", cfg, a.old_root); r2 = root_for("dataset2", cfg, a.new_root)
    idc = cfg["identification"]
    for cid, c in m.items():
        d = cfg["datasets"][c["dataset"]]
        if c["table"] not in cfg["hardening_tables"]:
            errors.append(f"{cid}: unknown table")
        if str(c["factor"]) not in cfg["resources"][c["dataset"]]:
            errors.append(f"{cid}: no resources for factor {c['factor']}")
        sh = c["field_shape"]
        if sh[0] % c["factor"] or sh[1] % c["factor"]:
            errors.append(f"{cid}: factor does not divide {sh}")
        k = d["materials"][c["material"]].get("id_key")
        if k and k not in idc[c["dataset"]]["keys"]:
            errors.append(f"{cid}: id_key {k} is not identified")
    nw = r2.parent
    static = {"solver": REPO / "microscale/solver/fem_pixel.py", "meso_solver": REPO / "mesoscale/solver/fem_tri.py"}
    for f in ("d1_identify.py", "d2_identify.py", "identify_common.py", "onset_kernels.py", "macro_reference.py"):
        static[f"id:{f}"] = HERE / "identification" / f
    for f in ("build_materials.py", "run_meso_case.py", "analyse_meso.py"):
        static[f"meso:{f}"] = HERE / "meso" / f
    for f in sorted((HERE / "meso/0_inputs/problems").glob("*.npz")) + sorted((HERE / "meso/0_inputs/segmentation").glob("*.npy")):
        static[f"meso_input:{f.name}"] = f
    d1i = idc["dataset1"]
    static.update({"d1:02_identify": r1 / d1i["code_dir_rel"] / "02_identify.py",
                   "d1:alpha_config": r1 / d1i["code_dir_rel"] / "alpha_config.py",
                   "d1:old_experiment_config": r1 / "2_codes/old_experiment_config.py",
                   "d1:traction": r1 / d1i["traction_rel"],
                   "dataset1:U": r1 / cfg["datasets"]["dataset1"]["U"],
                   "dataset1:V": r1 / cfg["datasets"]["dataset1"]["V"],
                   "d2:new_experiment_config": nw / "code/new_experiment_config.py",
                   "d2:02_yield_hardening_generic": nw / "code/02_yield_hardening_generic.py",
                   "d2:frames_manifest": nw / "code/frames_manifest.txt",
                   "d2:data_labjack": nw / "src/29_New_Tests_QI/data/26-03-26_11h26/data_labjack.csv"})
    for key in idc["dataset2"]["keys"]:
        static[f"dataset2:{key}:U"] = r2 / f"disflow/{key}/U/U_2295.npy"
        static[f"dataset2:{key}:V"] = r2 / f"disflow/{key}/V/V_2295.npy"
    if len(list((HERE / "meso/0_inputs/problems").glob("*.npz"))) != 4:
        errors.append("meso/0_inputs/problems must hold the four frozen problems")
    for key in d1i["keys"]:
        for comp in ("U", "V"):
            dd = r1 / "2_data/disflow" / key / comp
            n = len(list(dd.glob(f"{comp}_*.npy"))) if dd.is_dir() else 0
            if n != 42:
                errors.append(f"dataset1 {key}/{comp}: {n} frames, expected 42")
    for key in idc["dataset2"]["keys"]:
        for comp in ("U", "V"):
            dd = r2 / "disflow" / key / comp
            n = sum(1 for p in dd.glob(f"{comp}_*.npy") if 759 <= int(p.stem.split("_")[1]) <= 2295) if dd.is_dir() else 0
            if n != 1537:
                errors.append(f"dataset2 {key}/{comp}: {n} loading-window frames, expected 1537")
        cache = r2 / "yield_hardening" / key / "Evm_history.npy"
        print(f"info: dataset2 {key} strain-history cache {'present (%.1f GB)' % (cache.stat().st_size / 1e9) if cache.is_file() else 'MISSING - the job recomputes it'}")
    for k, p in static.items():
        if not Path(p).is_file():
            errors.append(f"missing {k}: {p}")
    if errors:
        raise SystemExit("preflight failed:\n  " + "\n  ".join(errors))
    d1 = cfg["datasets"]["dataset1"]
    for c in ("U", "V"):
        ref = r1 / d1.get(f"reference_copy_{c}", "")
        if ref.is_file():
            same = sha256(ref) == sha256(r1 / d1[c])
            print(f"info: {d1[c]} {'is identical to' if same else 'DIFFERS from'} {ref.name}")
    for st in ("coarse", "full"):
        print(f"stage {st}: {sum(c['stage'] == st for c in m.values())} FEM cases")
    print(f"identification: dataset1 {len(idc['dataset1']['keys'])} keys, dataset2 {len(idc['dataset2']['keys'])} keys; "
          f"meso: {len(cfg['meso']['cases'])} cases")
    rec = {}
    for k, p in sorted(static.items()):
        rec[k] = {"sha256": sha256(p), "bytes": Path(p).stat().st_size, "path": str(p)}
        if str(p).endswith(".npy"):
            x = np.load(p, mmap_mode="r"); rec[k].update(shape=list(x.shape), dtype=str(x.dtype))
        print(f"ok {k}: {rec[k]['sha256'][:12]}")
    payload = {"schema": 2, "campaign_config_sha256": sha256(HERE / "campaign.json"), "inputs": rec}
    lock = HERE / "input_lock.json"
    if a.write_lock:
        lock.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8"); print(f"wrote {lock}"); return
    old = json.loads(lock.read_text())
    if old["campaign_config_sha256"] != payload["campaign_config_sha256"]:
        raise SystemExit("campaign.json does not match lock")
    bad = [k for k, v in rec.items() if old["inputs"].get(k, {}).get("sha256") != v["sha256"]]
    if bad:
        raise SystemExit("input lock mismatch: " + ", ".join(bad))
    print(f"verified {len(rec)} locked files")


if __name__ == "__main__":
    main()
