# -*- coding: utf-8 -*-
"""
odb_extract_meso.py - ABAQUS PYTHON (2.7).  ELEMENT-LEVEL ODB export.

Run through `abaqus python`, normally via odb_process_meso.py.

    abaqus python odb_extract_meso.py <odb> <out_dir> [step] [instance]

Unlike odb_grid_4.py this does NOT rasterise: it writes the raw per-element and
per-node values, in ODB order, together with their labels.  That makes the
FEM-vs-Abaqus comparison exact and element-for-element - no interpolation error
mixed into the diff - and the files are ~40 MB instead of ~1.8 GB.
Rasterisation to the pixel grid is done afterwards in Python 3
(runners/fem_to_grid.py), with the SAME routine applied to both solvers.

CPS3 has a single integration point, so each element has exactly one value per
frame; if several are found they are averaged (and a warning is printed).

Tensor component order in a plane-stress ODB:
    data[0]=11  data[1]=22  data[2]=33 (=0)  data[3]=12
E and PE shear components are ENGINEERING shear strains, matching fem_tri.

Outputs (<base> = odb file stem):
    <base>_elem_labels.npy   (n_elem,)  int32
    <base>_node_labels.npy   (n_node,)  int32
    <base>_frametimes.npy    (n_fr,)    float64, step frameValue (0..1)
    <base>_ab_S.npy          (n_fr, n_elem, 3)  s11, s22, s12
    <base>_ab_E.npy          (n_fr, n_elem, 3)  e11, e22, gamma12
    <base>_ab_PE.npy         (n_fr, n_elem, 3)
    <base>_ab_PEEQ.npy       (n_fr, n_elem)
    <base>_ab_U.npy          (n_fr, n_node, 2)
    <base>_ab_RF.npy         (n_fr, n_node, 2)
"""
from odbAccess import openOdb
from abaqusConstants import NODAL, INTEGRATION_POINT
import numpy as np
import os
import sys


def _accum(field, n_ent, label_to_i, comps, is_elem):
    """Return (n_ent, len(comps)) averaged over whatever points are present."""
    out = np.zeros((n_ent, len(comps)), dtype=np.float64)
    cnt = np.zeros(n_ent, dtype=np.int32)

    used_bulk = False
    try:
        blocks = field.bulkDataBlocks
    except Exception:
        blocks = []
    for blk in blocks:
        try:
            lab = np.asarray(blk.elementLabels if is_elem else blk.nodeLabels)
            dat = np.asarray(blk.data)
        except Exception:
            blocks = []
            break
        if lab is None or dat is None or len(lab) == 0:
            continue
        idx = np.array([label_to_i.get(int(L), -1) for L in lab], dtype=np.int32)
        ok = idx >= 0
        if not ok.any():
            continue
        if dat.ndim == 1:
            dat = dat.reshape(-1, 1)
        for k, c in enumerate(comps):
            np.add.at(out[:, k], idx[ok], dat[ok, c])
        np.add.at(cnt, idx[ok], 1)
        used_bulk = True

    if not used_bulk:                     # fallback: value-by-value
        for v in field.values:
            L = int(v.elementLabel if is_elem else v.nodeLabel)
            i = label_to_i.get(L, -1)
            if i < 0:
                continue
            for k, c in enumerate(comps):
                out[i, k] += v.data[c] if hasattr(v.data, '__len__') else v.data
            cnt[i] += 1

    nz = cnt > 0
    for k in range(len(comps)):
        out[nz, k] /= cnt[nz]
    if (cnt > 1).any():
        print("  [warn] %d entities had >1 value per frame - averaged"
              % int((cnt > 1).sum()))
    if (cnt == 0).any():
        print("  [warn] %d entities had NO value in this frame" % int((cnt == 0).sum()))
    return out


def extract(odb_path, out_dir, step_name='Step-1', instance_name='PART-1-1'):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    odb = openOdb(path=odb_path, readOnly=True)
    step = odb.steps[step_name]
    inst = odb.rootAssembly.instances[instance_name]

    node_labels = np.array([n.label for n in inst.nodes], dtype=np.int32)
    elem_labels = np.array([e.label for e in inst.elements], dtype=np.int32)
    n_node, n_elem = len(node_labels), len(elem_labels)
    n_of = {}
    for i in range(n_node):
        n_of[int(node_labels[i])] = i
    e_of = {}
    for i in range(n_elem):
        e_of[int(elem_labels[i])] = i
    print("Nodes: %d   Elements: %d   Frames: %d"
          % (n_node, n_elem, len(step.frames)))

    base = os.path.splitext(os.path.basename(odb_path))[0]
    np.save(os.path.join(out_dir, base + '_node_labels.npy'), node_labels)
    np.save(os.path.join(out_dir, base + '_elem_labels.npy'), elem_labels)

    times, S, E, PE, PEEQ, U, RF = [], [], [], [], [], [], []

    for fr_idx, fr in enumerate(step.frames):
        print("Frame %d  (t = %g)" % (fr_idx, fr.frameValue))
        times.append(float(fr.frameValue))
        fo = fr.fieldOutputs

        fu = fo['U'].getSubset(region=inst, position=NODAL)
        U.append(_accum(fu, n_node, n_of, (0, 1), False))
        if 'RF' in fo.keys():
            frf = fo['RF'].getSubset(region=inst, position=NODAL)
            RF.append(_accum(frf, n_node, n_of, (0, 1), False))
        else:
            RF.append(np.zeros((n_node, 2)))

        fs = fo['S'].getSubset(region=inst, position=INTEGRATION_POINT)
        S.append(_accum(fs, n_elem, e_of, (0, 1, 3), True))
        fe = fo['E'].getSubset(region=inst, position=INTEGRATION_POINT)
        E.append(_accum(fe, n_elem, e_of, (0, 1, 3), True))
        if 'PE' in fo.keys():
            fp = fo['PE'].getSubset(region=inst, position=INTEGRATION_POINT)
            PE.append(_accum(fp, n_elem, e_of, (0, 1, 3), True))
        else:
            PE.append(np.zeros((n_elem, 3)))
        if 'PEEQ' in fo.keys():
            fq = fo['PEEQ'].getSubset(region=inst, position=INTEGRATION_POINT)
            PEEQ.append(_accum(fq, n_elem, e_of, (0,), True)[:, 0])
        else:
            PEEQ.append(np.zeros(n_elem))

    odb.close()

    np.save(os.path.join(out_dir, base + '_frametimes.npy'), np.array(times))
    np.save(os.path.join(out_dir, base + '_ab_S.npy'),    np.array(S))
    np.save(os.path.join(out_dir, base + '_ab_E.npy'),    np.array(E))
    np.save(os.path.join(out_dir, base + '_ab_PE.npy'),   np.array(PE))
    np.save(os.path.join(out_dir, base + '_ab_PEEQ.npy'), np.array(PEEQ))
    np.save(os.path.join(out_dir, base + '_ab_U.npy'),    np.array(U))
    np.save(os.path.join(out_dir, base + '_ab_RF.npy'),   np.array(RF))
    print("Saved element-level export to %s" % out_dir)


if __name__ == '__main__':
    odb_path = sys.argv[1]
    out_dir = sys.argv[2]
    step_name = sys.argv[3] if len(sys.argv) > 3 else 'Step-1'
    instance_name = sys.argv[4] if len(sys.argv) > 4 else 'PART-1-1'
    extract(odb_path, out_dir, step_name, instance_name)
