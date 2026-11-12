# -*- coding: utf-8 -*-
"""Boundary data and regional materials of the Dataset 2 mesoscale meshes.

Reads the shipped meshes (0_inputs/meshes, make_meshes.py) and partitions
(0_inputs/partitions_d2.npz), the production identification maps of Dataset 2
(0_identification/dataset2/a80_it240/off_stress) and the final DIC frame of
the same key; writes one self-contained problem per mesh to 1_problems/.

  boundary   every perimeter node; DIC data interpolated bilinearly at the node
             (pixel (i, j) of the field at raw position (i, j - column offset),
             the node convention of the pixel-scale model); edge-mean data
             as in the Dataset 1 meshes (one mean per edge, corners on the
             tensile ends)
  dic        region value = mean over the region's pixels of the 4-neighbour-
             filled yield map and of the filled K map in MPa
  dic_merged twin-resolved meshes only: the value of the twin-merged grain
  hp         Hall-Petch yield of Hu (2025), d_eq from the region's pixel area,
             with the uniform macroscopic K (hp_kmacro) or the identified K
             (hp_kdic)
  homog      macroscopic law of Dataset 2 (campaign.json)
  pixel      fine_twins only: maps sampled at every triangle centroid

    python build_problems.py [--meshes-dir 0_inputs/meshes] [--out 1_problems]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import common as C


def label_means(lab, field, n):
    cnt = np.bincount(lab.ravel(), minlength=n).astype(np.float64)
    tot = np.bincount(lab.ravel(), weights=np.asarray(field, np.float64).ravel(), minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return tot / cnt, cnt


def boundary_data(raw, bc, origin, offset, U, V, nr, nc):
    """DIC displacement (mm) at the boundary nodes, and the edge-mean data."""
    r = raw[bc, 0] + origin[0] + offset[0]
    c = raw[bc, 1] + origin[1] + offset[1]
    u = np.stack([C.bilinear(V, r, c), C.bilinear(U, r, c)], 1) * C.PX_MM
    x, y = raw[bc, 0], raw[bc, 1]
    end0, end1 = np.abs(y) < 1e-9, np.abs(y - nc) < 1e-9            # tensile ends, corners included
    side0 = (np.abs(x) < 1e-9) & ~end0 & ~end1
    side1 = (np.abs(x - nr) < 1e-9) & ~end0 & ~end1
    ue = np.zeros_like(u)
    for m in (end0, end1, side0, side1):
        ue[m] = u[m].mean(axis=0)
    if not np.all(end0 | end1 | side0 | side1):
        raise SystemExit("boundary node off the perimeter")
    return u, ue, dict(end0=int(end0.sum()), end1=int(end1.sum()), side0=int(side0.sum()), side1=int(side1.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meshes-dir", default=str(C.INPUTS / "meshes"))
    ap.add_argument("--partitions", default=str(C.INPUTS / "partitions_d2.npz"))
    ap.add_argument("--out", default=str(C.PROBLEMS))
    ap.add_argument("--meshes", default=",".join(C.MESHES))
    a = ap.parse_args()
    st = C.settings()
    pk = C.package()
    for p in (st["maps"] / "yield_macro_filled.npy", st["maps"] / "K_coef_filled.npy", st["U"], st["V"]):
        if not Path(p).is_file():
            raise SystemExit(f"missing input {p}")
    src, shape, off = st["source_shape"], st["field_shape"], st["crop_offset"]
    Y = C.fill_nan_iter_4nbr(np.load(st["maps"] / "yield_macro_filled.npy").astype(np.float64))
    K = C.fill_nan_iter_4nbr(np.load(st["maps"] / "K_coef_filled.npy").astype(np.float64) * st["k_scale"])
    if Y.shape != src or K.shape != src:
        raise SystemExit(f"map shape {Y.shape}, expected {src}")
    Y = Y[off[0]:off[0] + shape[0], off[1]:off[1] + shape[1]]
    K = K[off[0]:off[0] + shape[0], off[1]:off[1] + shape[1]]
    U = np.asarray(np.load(st["U"], mmap_mode="r"), np.float64)
    V = np.asarray(np.load(st["V"], mmap_mode="r"), np.float64)
    if U.shape != src or V.shape != src:
        raise SystemExit(f"displacement shape {U.shape}, expected {src}")
    with np.load(a.partitions) as z:
        part = {"twin_merged": z["twin_merged"].astype(np.int64), "twin_resolved": z["twin_resolved"].astype(np.int64)}
    for k, v in part.items():
        if v.shape != shape:
            raise SystemExit(f"{k} shape {v.shape}, expected {shape}")
    hom_y, hom_k = st["homogeneous"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(package=pk["package"], maps={str(st["maps"] / f): C.sha256(st["maps"] / f)
                                                  for f in ("yield_macro_filled.npy", "K_coef_filled.npy")},
                    dic={str(st["U"]): C.sha256(st["U"]), str(st["V"]): C.sha256(st["V"])},
                    partitions_sha256=C.sha256(a.partitions), k_scale_mpa=st["k_scale"],
                    homogeneous=[hom_y, hom_k], hp_fit=pk["hp_fit"],
                    map_means=dict(yield_mpa=float(Y.mean()), K_mpa=float(K.mean())), meshes={})
    for name in a.meshes.split(","):
        mf = Path(a.meshes_dir) / f"{name}.npz"
        with np.load(mf) as z:
            m = {k: z[k] for k in z.files}
        twins = str(m["partition"]) == "twin_resolved"
        nr, nc = map(int, m["grid_shape"])
        o0, o1 = map(int, m["grid_origin"])
        full = part[str(m["partition"])]
        lab = full[o0:o0 + nr, o1:o1 + nc]
        n = int(full.max()) + 1
        regions = m["segmentation_label"].astype(np.int64)
        ym, cnt = label_means(lab, Y[o0:o0 + nr, o1:o1 + nc], n)
        km, _ = label_means(lab, K[o0:o0 + nr, o1:o1 + nc], n)
        if np.any(cnt[regions] == 0):
            raise SystemExit(f"{name}: a mesh region has no pixel")
        cnt_full = np.bincount(full.ravel(), minlength=n).astype(np.float64)
        s0, k_um = pk["hp_fit"]["twins" if twins else "no_twins"]
        d_um = 2.0 * np.sqrt(cnt_full * (C.PX_MM * 1000.0) ** 2 / np.pi)
        with np.errstate(divide="ignore"):
            hp = s0 + k_um / np.sqrt(d_um)
        nreg = regions.size
        mat = dict(sy0_dic=ym[regions], K_dic=km[regions], sy0_hp=hp[regions],
                   K_hp_macro=np.full(nreg, st["k_scale"]), K_hp_hybrid=km[regions],
                   sy0_homogeneous=np.full(nreg, hom_y), K_homogeneous=np.full(nreg, hom_k),
                   region_pixels=cnt[regions], region_d_um=d_um[regions])
        info = dict(regions=int(nreg), elements=int(len(m["elements"])), nodes=int(len(m["coords"])),
                    yield_region_mean=float(ym[regions].mean()), yield_region_sd=float(ym[regions].std()),
                    K_region_mean=float(km[regions].mean()), K_region_sd=float(km[regions].std()),
                    yield_area_weighted=float(np.sum(ym[regions] * cnt[regions]) / cnt[regions].sum()),
                    K_area_weighted=float(np.sum(km[regions] * cnt[regions]) / cnt[regions].sum()),
                    hp_yield_mean=float(hp[regions].mean()), d_um_median=float(np.median(d_um[regions])))
        if twins:
            fm = part["twin_merged"]
            nm = int(fm.max()) + 1
            ymm, _ = label_means(fm[o0:o0 + nr, o1:o1 + nc], Y[o0:o0 + nr, o1:o1 + nc], nm)
            kmm, _ = label_means(fm[o0:o0 + nr, o1:o1 + nc], K[o0:o0 + nr, o1:o1 + nc], nm)
            g = m["merged_label_of_region"].astype(np.int64)
            mat["sy0_dic_merged"], mat["K_dic_merged"] = ymm[g], kmm[g]
            info["merged_on_twin_mesh_yield_sd"] = float(ymm[g].std())
        coords = np.asarray(m["coords"], np.float64)
        raw = coords / C.PX_MM
        el = np.asarray(m["elements"], np.int64)
        if name == "fine_twins":
            cen = np.floor(raw[el].mean(axis=1)).astype(np.int64)
            ci = np.clip(cen[:, 0], 0, nr - 1) + o0
            cj = np.clip(cen[:, 1], 0, nc - 1) + o1
            mat["sy0_pixel"], mat["K_pixel"] = Y[ci, cj], K[ci, cj]
            info.update(pixel_yield_mean=float(mat["sy0_pixel"].mean()), pixel_K_mean=float(mat["K_pixel"].mean()))
        bc = m["bc_nodes"].astype(np.int64)
        u, ue, edges = boundary_data(raw, bc, (o0, o1), off, U, V, nr, nc)
        # check against nearest-pixel sampling at the integer nodes
        ii = np.rint(raw[bc]).astype(np.int64)
        integer = np.all(np.abs(raw[bc] - ii) < 1e-9, axis=1)
        ri = np.clip(ii[integer, 0] + o0 + off[0], 0, src[0] - 1)
        cj = np.clip(ii[integer, 1] + o1 + off[1], 0, src[1] - 1)
        dev = np.max(np.abs(u[integer] - np.stack([V[ri, cj], U[ri, cj]], 1) * C.PX_MM))
        info.update(bc_nodes=int(bc.size), bc_edges=edges, bc_integer_node_dev_mm=float(dev),
                    bc_u_range_mm=[float(u[:, 0].min()), float(u[:, 0].max()), float(u[:, 1].min()), float(u[:, 1].max())],
                    mean_strain_from_edges=dict(
                        tensile=float((ue[np.abs(raw[bc, 1] - nc) < 1e-9, 1].mean() - ue[np.abs(raw[bc, 1]) < 1e-9, 1].mean())
                                      / (nc * C.PX_MM)),
                        lateral=float((u[np.abs(raw[bc, 0] - nr) < 1e-9, 0].mean() - u[np.abs(raw[bc, 0]) < 1e-9, 0].mean())
                                      / (nr * C.PX_MM))))
        dst = out / f"{name}.npz"
        np.savez(dst, coords=coords, elements=el.astype(np.int32), mat_id=m["mat_id"], bc_nodes=bc, bc_u_dic=u,
                 bc_u_edge_mean=ue, segmentation_label=m["segmentation_label"], mapping_purity=m["mapping_purity"],
                 grid_shape=m["grid_shape"], grid_origin=m["grid_origin"], partition=m["partition"], **mat)
        info.update(mesh_sha256=C.sha256(mf), problem_sha256=C.sha256(dst))
        manifest["meshes"][name] = info
        print(f"[problem] {name}: {nreg} regions, {info['elements']} CPS3; yield {info['yield_region_mean']:.1f} "
              f"+/- {info['yield_region_sd']:.1f} MPa, K {info['K_region_mean']:.1f} +/- {info['K_region_sd']:.1f} MPa, "
              f"HP {info['hp_yield_mean']:.1f} MPa; edge strain {info['mean_strain_from_edges']}; "
              f"bc dev {dev:.1e} mm", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
