# -*- coding: utf-8 -*-
"""
inp_io.py - Abaqus .inp reading and writing for the mesoscale test.

read_mesh()   : fast parser for the BASE mesh (*Node, *Element type=CPS3,
                *Elset elset=Face_*).  Replaces read_elsets.AbaqusInputReader
                so this folder has no dependency on Thesis/5_code.
build_problem(): the shared problem definition - scaled coordinates, element
                connectivity, per-element grain id, grain material tables and
                the prescribed boundary displacements.  BOTH chains call this,
                so the Abaqus .inp and the in-house FEM cannot drift apart.
write_inp()   : writes the Abaqus job, byte-for-byte in the layout of
                mesoscale_mesh_triangular.MeshCreator + MaterialInserter.
"""
import os
import numpy as np

import meso_config as cfg


# ═══════════════════════════════ reading ═════════════════════════════════════

def read_mesh(path):
    """Parse a base .inp.  Returns (nodes, elements, elsets, node_ids, elem_ids).

    nodes     (n_node, 2) float64, UNSCALED, ordered as written in the file
    elements  (n_elem, 3) int32, node ARRAY INDICES (not labels)
    elsets    dict {'Face_1': np.array of element ARRAY INDICES, ...}
    node_ids  (n_node,) int64 original labels
    elem_ids  (n_elem,) int64 original labels
    """
    nlab, ncoord, elab, econn = [], [], [], []
    esets, mode, cur = {}, None, None

    with open(path, "r", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("**"):
                continue
            if s.startswith("*"):
                u = s.upper()
                if u.startswith("*NODE") and "OUTPUT" not in u and "NSET" not in u:
                    mode = "n"
                elif u.startswith("*ELEMENT") and "OUTPUT" not in u:
                    mode = "e"
                elif u.startswith("*ELSET"):
                    mode = "s"
                    cur = s.split("=")[-1].strip()
                    esets.setdefault(cur, [])
                else:
                    mode = None
                continue
            p = [t for t in s.split(",") if t.strip()]
            if mode == "n":
                nlab.append(int(p[0])); ncoord.append((float(p[1]), float(p[2])))
            elif mode == "e":
                elab.append(int(p[0])); econn.append((int(p[1]), int(p[2]), int(p[3])))
            elif mode == "s":
                esets[cur].extend(int(t) for t in p)

    node_ids = np.array(nlab, dtype=np.int64)
    nodes    = np.array(ncoord, dtype=np.float64)
    elem_ids = np.array(elab, dtype=np.int64)
    conn_lab = np.array(econn, dtype=np.int64)

    # label -> array index, via a lookup table (labels are dense and 1-based
    # here).  Out-of-range labels are caught explicitly: a bare IndexError from
    # the fancy-index says nothing about WHICH elset is malformed, and a broken
    # grain partition is exactly the failure this folder exists to detect.
    def _lookup(labels, refs, what, where):
        tbl = np.full(int(labels.max()) + 1, -1, dtype=np.int64)
        tbl[labels] = np.arange(len(labels))
        refs = np.asarray(refs, dtype=np.int64)
        oob = (refs < 0) | (refs >= len(tbl))
        idx = np.where(oob, -1, tbl[np.clip(refs, 0, len(tbl) - 1)])
        if (idx < 0).any():
            bad = np.unique(refs[idx < 0])[:5]
            raise ValueError(f"{path}: {where} references {int((idx < 0).sum())} "
                             f"undefined {what} label(s), e.g. {list(bad)}")
        return idx

    elements = _lookup(node_ids, conn_lab.ravel(), "node",
                       "*Element").reshape(conn_lab.shape).astype(np.int32)

    elsets = {}
    for k, v in esets.items():
        elsets[k] = _lookup(elem_ids, v, "element",
                            f"elset {k}").astype(np.int32)

    return nodes, elements, elsets, node_ids, elem_ids


def element_areas(coords, elements):
    """Triangle areas (mm^2), used to area-weight every element-level average."""
    p = coords[elements]
    return 0.5*np.abs((p[:, 1, 0] - p[:, 0, 0])*(p[:, 2, 1] - p[:, 0, 1])
                      - (p[:, 2, 0] - p[:, 0, 0])*(p[:, 1, 1] - p[:, 0, 1]))


def _face_order(names):
    """Face_1, Face_2, ... in numeric order (dict insertion order is the file
    order, which is already numeric, but do not rely on it)."""
    def key(n):
        try:
            return int(n.split("_")[1])
        except (IndexError, ValueError):
            return 1 << 30
    return sorted(names, key=key)


# ═══════════════════════════ problem definition ══════════════════════════════

def build_problem(verbose=True):
    """The whole mesoscale boundary-value problem, derived from the DIC data.

    This is the mesoscale twin of test_config.load_case5_inputs(): the Abaqus
    .inp writer and the in-house FEM driver both consume exactly this dict.

    Returns dict with
        coords     (n_node, 2)  SCALED mm coordinates (x*1.84, y*1.84)
        elements   (n_elem, 3)  node array indices
        mat_id     (n_elem,)    grain index 0..n_grain-1
        grain_names            list of Face_* names, index-aligned with mat_id
        table      (n_grain, 50) *Plastic yield stress at cfg.EP_TABLE
        sy0, K     (n_grain,)
        bc_nodes   (n_bc,)      node array indices with BOTH dofs prescribed
        bc_u       (n_bc, 2)    prescribed (ux, uy) in mm
        node_ids, elem_ids      original Abaqus labels
        edges      dict of the four edge node-index arrays (file order)
    """
    nodes, elements, elsets, node_ids, elem_ids = read_mesh(cfg.BASE_INP)
    coords = nodes * cfg.SCALE_FACTOR
    n_node, n_elem = len(coords), len(elements)

    names = _face_order(elsets)
    mat_id = np.full(n_elem, -1, dtype=np.int32)
    for g, nm in enumerate(names):
        mat_id[elsets[nm]] = g
    if (mat_id < 0).any():
        n_orphan = int((mat_id < 0).sum())
        raise ValueError(f"{n_orphan} elements belong to no Face_* elset - "
                         f"they would have no material. Check {cfg.BASE_INP}.")

    if verbose:
        print(f"[mesh] {cfg.BASE_INP}")
        print(f"[mesh] {n_node} nodes | {n_elem} CPS3 | {len(names)} grains")
        print(f"[mesh] scaled domain: x 0..{coords[:,0].max():.6f}  "
              f"y 0..{coords[:,1].max():.6f} mm")

    # ── grain materials: first element of each elset, its centroid pixel ─────
    # verbatim mesoscale_runner.find_single_element_centroid + stress_assign,
    # INCLUDING the -1 offset (centroid pixel index, then minus one).
    # load_material_maps() is load_field_maps() for the default variant and the
    # Hall-Petch / homogenised map for the others (meso_config, VARIANT AXES).
    # getattr so an older meso_config without the block still works.
    _maps = getattr(cfg, "load_material_maps", None)
    if _maps is None:
        yield_average, K_average = cfg.load_field_maps()
    else:
        yield_average, K_average = _maps(verbose=verbose, n_grain=len(names))
    cx = np.empty(len(names), dtype=np.int64)
    cy = np.empty(len(names), dtype=np.int64)
    for g, nm in enumerate(names):
        e0 = int(elsets[nm][0])                 # FIRST element of the elset
        tri = nodes[elements[e0]]               # UNSCALED coords, as in the original
        cx[g] = int(round(float(tri[:, 0].mean()) * 1000)) - 1
        cy[g] = int(round(float(tri[:, 1].mean()) * 1000)) - 1

    # Q3 (see meso_config docstring). The original samples the maps with a bare
    # `stress_field[cx, cy]` and no lower clamp, so cx == -1 WRAPS to the far
    # edge of the image and that grain inherits an unrelated grain's yield
    # curve. Clamping to 0 is the correct thing to do and the wrong thing to
    # compare against: the existing .inp files - and the .odb results computed
    # from them - were written with the wraparound. Reproduce it by default.
    #
    # Only the LOWER bound is in question. The upper bound never triggers on
    # these meshes; the original would have raised IndexError, been caught, and
    # written a literal `None` into the .inp, which no .inp contains. Clamping
    # it here keeps a future mesh from failing silently.
    n_wrap = int((cx < 0).sum() + (cy < 0).sum())
    if getattr(cfg, "CLIP_CENTROID", False):
        cx = np.clip(cx, 0, yield_average.shape[0] - 1)
        cy = np.clip(cy, 0, yield_average.shape[1] - 1)
        if verbose and n_wrap:
            print(f"[mat ] CLIP_CENTROID=True: {n_wrap} centroid index(es) "
                  f"clamped to 0 that the original would have wrapped - the "
                  f"tables will NOT match the existing .inp. Regenerate it "
                  f"(MESO_FORCE_INP=1) and re-run Abaqus.")
    else:
        cx = np.minimum(cx, yield_average.shape[0] - 1)
        cy = np.minimum(cy, yield_average.shape[1] - 1)
        if verbose and n_wrap:
            print(f"[mat ] {n_wrap} grain centroid index(es) are -1 and wrap to "
                  f"the far edge, reproducing the original (Q3). Set "
                  f"cfg.CLIP_CENTROID=True to sample correctly instead.")
    mat = cfg.grain_material_table(np.stack([cx, cy], axis=1), yield_average, K_average)

    if verbose:
        print(f"[mat ] sy0  {mat['sy0'].min():.2f} .. {mat['sy0'].max():.2f} MPa")
        print(f"[mat ] K    {mat['K'].min():.2f} .. {mat['K'].max():.2f} MPa")

    # ── boundary nodes and their prescribed displacements ───────────────────
    disp_x, disp_y = cfg.load_dic_displacements()
    tol   = cfg.BC_TOL
    x_max = coords[:, 0].max()
    y_max = coords[:, 1].max()

    # file order within each edge, and edge order bottom/left/right/top - the
    # order mesoscale_mesh_triangular writes the *Nset blocks in, so the
    # generated .inp is identical to the existing one.
    bottom = np.where(coords[:, 1] < tol)[0]
    top    = np.where(np.abs(coords[:, 1] - y_max) < tol)[0]
    left   = np.where(coords[:, 0] < tol)[0]
    right  = np.where(np.abs(coords[:, 0] - x_max) < tol)[0]
    edges  = dict(bottom=bottom, top=top, left=left[1:-1], right=right[1:-1])
    # NOTE the [1:-1]: MaterialInserter drops the first and last LEFT and RIGHT
    # node (they are corner nodes already carried by bottom/top).

    bc_nodes = np.unique(np.concatenate([edges["bottom"], edges["top"],
                                         edges["left"], edges["right"]]))

    px = np.clip(np.round(coords[bc_nodes, 0] * 1000.0 / cfg.SCALE_FACTOR).astype(np.int64),
                 0, disp_y.shape[0] - 1)
    py = np.clip(np.round(coords[bc_nodes, 1] * 1000.0 / cfg.SCALE_FACTOR).astype(np.int64),
                 0, disp_y.shape[1] - 1)
    bc_u = np.stack([np.asarray(disp_x, dtype=np.float64)[px, py],
                     np.asarray(disp_y, dtype=np.float64)[px, py]], axis=1)

    # BC model hook (meso_config, VARIANT AXES). `dic` - the default - returns
    # bc_u unchanged; `edge_mean` replaces each edge's values by that edge's
    # mean, on the SAME constrained node set.
    bc_note = "per-node DIC displacement"
    _bcm = getattr(cfg, "apply_bc_model", None)
    if _bcm is not None:
        bc_u, bc_note = _bcm(bc_u, edges, bc_nodes, verbose=verbose)

    if verbose:
        print(f"[bc  ] {len(bc_nodes)} constrained nodes "
              f"(bottom {len(bottom)}, top {len(top)}, "
              f"left {len(edges['left'])}, right {len(edges['right'])}) tol={tol}")
        print(f"[bc  ] ux {bc_u[:,0].min():+.5f}..{bc_u[:,0].max():+.5f}  "
              f"uy {bc_u[:,1].min():+.5f}..{bc_u[:,1].max():+.5f} mm")
        if not np.isfinite(bc_u).all():
            raise ValueError("prescribed displacements contain NaN/inf - the DIC "
                             "field does not cover the whole mesh boundary")

    return dict(coords=coords, elements=elements, mat_id=mat_id,
                grain_names=names, elsets=elsets,
                table=mat["table"], sy0=mat["sy0"], K=mat["K"],
                bc_nodes=bc_nodes, bc_u=bc_u, edges=edges, bc_note=bc_note,
                node_ids=node_ids, elem_ids=elem_ids, nodes_raw=nodes)


# ═══════════════════════════════ writing ═════════════════════════════════════

_EDGE_KW = dict(bottom="BOTTOM", left="LEFT", right="RIGHT", top="TOP")


def write_inp(path, prob, verbose=True):
    """Write the Abaqus job.  Layout follows mesoscale_mesh_triangular.py."""
    coords   = prob["coords"]
    elements = prob["elements"]
    node_ids = prob["node_ids"]
    elem_ids = prob["elem_ids"]
    names    = prob["grain_names"]
    elsets   = prob["elsets"]
    edges    = prob["edges"]

    conn_lab = node_ids[elements]
    bc_of = {int(n): k for k, n in enumerate(prob["bc_nodes"])}

    with open(path, "w") as f:
        f.write("*Heading\n** PARTS\n*Part, name=PART-1\n*NODE\n")
        for k in range(len(coords)):
            f.write("{}, {}, {}\n".format(node_ids[k], coords[k, 0], coords[k, 1]))

        f.write("*Element, type=CPS3\n")
        for k in range(len(elements)):
            f.write("{}, {}, {}, {}\n".format(elem_ids[k], *conn_lab[k]))

        for nm in names:
            ID = nm.split("_")[1]
            f.write("*Elset, elset={}\n".format(nm))
            lab = elem_ids[elsets[nm]]
            for i in range(0, len(lab), 16):
                f.write(", ".join(map(str, lab[i:i + 16])) + "\n")
            f.write("** Section: Grain-{}\n".format(ID))
            f.write("*Solid Section, elset={}, material=Material-{}\n".format(nm, ID))

        f.write("*End Part\n** ASSEMBLY\n*Assembly, name=Assembly\n")
        f.write("*Instance, name=PART-1-1, part=PART-1\n*End Instance\n")

        for e in ("bottom", "left", "right", "top"):
            for ni in edges[e]:
                lab = node_ids[ni]
                f.write("*Nset, nset=NODE{}{}, instance=PART-1-1\n{}\n"
                        .format(_EDGE_KW[e], lab, lab))
        f.write("*End Assembly\n")

        # ── materials ────────────────────────────────────────────────────────
        for g, nm in enumerate(names):
            ID = nm.split("_")[1]
            f.write("** MATERIALS\n*Material, name=Material-{}\n".format(ID))
            f.write("*Elastic\n{}., {}\n".format(int(cfg.E_MOD), cfg.NU))
            f.write("*Plastic\n")
            for k in range(cfg.N_TABLE):
                f.write("{}, {}\n".format(prob["table"][g, k], cfg.EP_TABLE[k]))

        # ── step ─────────────────────────────────────────────────────────────
        f.write("** STEP: Step-1\n** \n")
        f.write("*Step, name=Step-1, nlgeom=NO, inc=1000\n")
        f.write("*Static\n0.001, 1., 1e-06, 1.\n")
        f.write("** BOUNDARY CONDITIONS\n")
        for e in ("bottom", "top", "left", "right"):
            kw = _EDGE_KW[e]
            for ni in edges[e]:
                lab = node_ids[ni]
                ux, uy = prob["bc_u"][bc_of[int(ni)]]
                f.write("** Name: {}{} Type: Displacement/Rotation\n*Boundary\n"
                        .format(e.capitalize(), lab))
                f.write("NODE{}{}, 2, 2, {}\n".format(kw, lab, float(uy)))
                f.write("NODE{}{}, 1, 1, {}\n".format(kw, lab, float(ux)))
        f.write("*Restart, write, frequency=0\n")
        f.write("*Output, field\n*Node Output\nRF, U\n")
        f.write("*Element Output, directions=YES\nE, PE, PEEQ, PEMAG, S\n")
        f.write("*Output, history, variable=PRESELECT\n*End Step\n")

    if verbose:
        print(f"[inp ] written: {path}  ({os.path.getsize(path)/1e6:.1f} MB)")
