# -*- coding: utf-8 -*-
"""
selftest_fixture.py - build a miniature stand-in for the Thesis data tree.

Creates a ~30-grain triangular mesh, a matching segmentation, yield/K maps and
DIC displacement fields, laid out exactly like the real
the real thesis tree, so that MESO_THESIS_ROOT can point the
whole pipeline at it.  Used by selftest.py to exercise every script in this
folder end to end without Abaqus and without the real data.

    python selftest_fixture.py <root>
"""
import os
import sys
import numpy as np


# Must match meso_config's defaults, or the fixture writes where the config
# does not look. selftest.py sets the two *_STUDY_DIR env vars to match.
SRC_KEY = "a100_it240"
T_KEY = "T075"
FINAL_FRAME = 40


def build(root, n_grain=30, nx_px=351, ny_px=301, seed=0):
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(seed)
    x_raw = (nx_px - 1)/1000.0          # pixel index = raw * 1000
    y_raw = (ny_px - 1)/1000.0

    meso = os.path.join(root, "Case_1_Pixel/Claude/2_old_experiment/Mesoscale")
    inp_dir = os.path.join(meso, "Case_studies/Heterogeneous_material",
                           "2_fine_mesh_no_twins/1_inp")
    # Mirror the real study layout, not the old flat 3_data one: the material
    # maps come from threshold_study/<T_KEY>/ and the kinematics from
    # alpha_study/disflow/<SRC_KEY>/{U,V}/. selftest.py points
    # MESO_THRESH_STUDY_DIR / MESO_ALPHA_STUDY_DIR at these.
    thresh_root = os.path.join(root, "_studies/threshold_study/2_data")
    alpha_root = os.path.join(root, "_studies/alpha_study/2_data")
    yh_dir = os.path.join(thresh_root, "yield_hardening", T_KEY)
    u_dir = os.path.join(alpha_root, "disflow", SRC_KEY, "U")
    v_dir = os.path.join(alpha_root, "disflow", SRC_KEY, "V")
    for d in (inp_dir, os.path.join(root, "4_seg"), yh_dir, u_dir, v_dir):
        os.makedirs(d, exist_ok=True)

    # ── mesh: structured points, interior jittered, Delaunay ────────────────
    m, n = 26, 23
    gx, gy = np.meshgrid(np.linspace(0, x_raw, m), np.linspace(0, y_raw, n),
                         indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    inner = ((pts[:, 0] > 0) & (pts[:, 0] < x_raw) &
             (pts[:, 1] > 0) & (pts[:, 1] < y_raw))
    pts[inner] += rng.uniform(-0.3, 0.3, (inner.sum(), 2))*np.array([x_raw/(m-1),
                                                                    y_raw/(n-1)])
    tri = Delaunay(pts).simplices

    # ── grains: Voronoi cells of random seeds ───────────────────────────────
    seeds = np.stack([rng.uniform(0, x_raw, n_grain),
                      rng.uniform(0, y_raw, n_grain)], axis=1)
    cent = pts[tri].mean(axis=1)
    lab_e = np.argmin(((cent[:, None, :] - seeds[None])**2).sum(-1), axis=1)
    used = np.unique(lab_e)
    remap = {int(g): k for k, g in enumerate(used)}
    lab_e = np.array([remap[int(g)] for g in lab_e])
    n_grain = len(used)
    seeds = seeds[used]

    # ── write the base .inp (nodes, CPS3, Face_* elsets) ────────────────────
    p = os.path.join(inp_dir, "fine_no_twins_base.inp")
    with open(p, "w") as f:
        f.write("*Node\n")
        for i, (x, y) in enumerate(pts):
            f.write(f"{i+1}, {x:.10e}, {y:.10e}\n")
        f.write("*Element, type=CPS3\n")
        for i, t in enumerate(tri):
            f.write(f"{i+1}, {t[0]+1}, {t[1]+1}, {t[2]+1}\n")
        for g in range(n_grain):
            ids = np.where(lab_e == g)[0] + 1
            f.write(f"*Elset, elset=Face_{g+1}\n")
            for k in range(0, len(ids), 16):
                f.write(", ".join(map(str, ids[k:k+16])) + "\n")

    # ── pixel maps on the same Voronoi partition ────────────────────────────
    px, py = np.meshgrid(np.arange(nx_px), np.arange(ny_px), indexing="ij")
    P = np.stack([px/1000.0, py/1000.0], axis=-1)
    seg = np.argmin(((P[..., None, :] - seeds[None, None])**2).sum(-1),
                    axis=-1).astype(np.int32)
    np.save(os.path.join(root, "4_seg/GB_without_twins_4.npy"), seg)

    yld = 120.0 + 60.0*rng.random((nx_px, ny_px)) + 20.0*np.sin(px/37.0)
    K = 0.8 + 0.5*rng.random((nx_px, ny_px))          # * K_SCALE = 396 later
    yld[rng.random(yld.shape) < 0.002] = np.nan       # exercise the NaN fill
    np.save(os.path.join(yh_dir, "yield_vm_thresh.npy"), yld)
    np.save(os.path.join(yh_dir, "K_coef_filled.npy"), K)

    # ── DIC: uniaxial stretch along mesh y (= U) plus a smooth perturbation ─
    for k in list(range(1, 6)) + [FINAL_FRAME]:
        s = 0.0 if k < 6 else 1.0
        eyy, exx = 0.020*s, -0.006*s
        wob = 0.35*s*np.sin(2*np.pi*px/nx_px)*np.cos(3*np.pi*py/ny_px)
        # stored in PIXEL units; the pipeline multiplies by PX_TO_MM
        np.save(os.path.join(u_dir, f"U_{k}.npy"), eyy*py + wob)
        np.save(os.path.join(v_dir, f"V_{k}.npy"), exx*px + 0.5*wob)

    print(f"[fixture] root      : {root}")
    print(f"[fixture] mesh      : {len(pts)} nodes, {len(tri)} CPS3, "
          f"{n_grain} grains -> {p}")
    print(f"[fixture] pixel grid: {nx_px} x {ny_px}")
    print(f"[fixture] materials : {yh_dir}")
    print(f"[fixture] kinematics: {os.path.dirname(u_dir)}  (final frame {FINAL_FRAME})")
    return p


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/thesis_fixture")
