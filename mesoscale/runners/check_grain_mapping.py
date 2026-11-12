# -*- coding: utf-8 -*-
"""
check_grain_mapping.py - prove the grains are the SAME grains in both codes.

The mesh comes out of the CAD/meshing chain (CRISTALX -> Salome -> .inp) with
the grain partition already fixed as `Face_*` elsets.  Everything downstream
depends on one thing: that grain N in the in-house solver is grain N in the
Abaqus job, carrying the same elements and the same hardening curve.  Element
shape, triangle quality and how the properties were sampled are all beside the
point - this script checks only the grain identity.

It compares the .inp Abaqus ACTUALLY RAN (cfg.JOB_INP) against what
inp_io.build_problem() hands to fem_tri, in four layers:

  A  mesh        node labels + coordinates, element labels + connectivity
  B  partition   Face_g element-label sets, and that they tile the mesh
  C  material    Face_g -> Material-g -> *Elastic / *Plastic table, per grain
  D  boundary    constrained node set and the prescribed (ux, uy) per node

A or B failing means the two codes are not on the same mesh and no comparison
downstream means anything - exit code 1.  C or D failing means the mesh is
shared but the material or the loading drifted, usually because the .inp
predates a change in the preprocessing - loud warning, exit code 0, so a chain
that includes this step still runs and you can see the size of the drift.

    python check_grain_mapping.py

Writes results/<mesh>/compare/grain_mapping_<mesh>.txt
"""
import os
import re
import sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import meso_config as cfg
import inp_io

# Relative tolerance for the material tables. NOT 1e-9: the tables were written
# by jax in float32 and re-read from text, so an .inp produced by the ORIGINAL
# code differs from a freshly built table at ~1e-7 on essentially every grain -
# pure representation noise. At 1e-9 that noise reported "1447/1447 grains
# differ" and buried the 18 grains that differ for a real reason (Q3). 1e-6 sits
# an order of magnitude above the noise and well below anything meaningful; the
# max relative difference is printed regardless, so a drift between 1e-7 and
# 1e-6 is still visible in the report.
RTOL = 1e-6          # relative tolerance for the material tables
ATOL_U = 1e-12       # mm, prescribed displacements are written at full repr


# ── parse the job .inp that Abaqus ran ──────────────────────────────────────
def parse_job(path):
    nodes, elems, esets, sect = {}, {}, {}, {}
    mats, bc = {}, {}
    order = []
    mode = cur = curmat = None

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
                    order.append(cur)
                elif u.startswith("*SOLID SECTION"):
                    mode = None
                    m = re.search(r"ELSET\s*=\s*([^,]+).*MATERIAL\s*=\s*(\S+)", s,
                                  re.IGNORECASE)
                    if m:
                        sect[m.group(1).strip()] = m.group(2).strip()
                elif u.startswith("*MATERIAL"):
                    mode = None
                    curmat = s.split("=")[-1].strip()
                    mats[curmat] = dict(elastic=None, plastic=[])
                elif u.startswith("*ELASTIC"):
                    mode = "el"
                elif u.startswith("*PLASTIC"):
                    mode = "pl"
                elif u.startswith("*BOUNDARY"):
                    mode = "bc"
                else:
                    mode = None
                continue

            p = [t for t in s.split(",") if t.strip()]
            if mode == "n":
                nodes[int(p[0])] = (float(p[1]), float(p[2]))
            elif mode == "e":
                elems[int(p[0])] = tuple(int(t) for t in p[1:4])
            elif mode == "s":
                esets[cur] += [int(t) for t in p]
            elif mode == "el":
                mats[curmat]["elastic"] = (float(p[0]), float(p[1]))
            elif mode == "pl":
                mats[curmat]["plastic"].append((float(p[0]), float(p[1])))
            elif mode == "bc":
                m = re.match(r"NODE(?:BOTTOM|TOP|LEFT|RIGHT)(\d+)$", p[0].strip(),
                             re.IGNORECASE)
                if m and len(p) >= 4:
                    lab = int(m.group(1))
                    dof = int(p[1])
                    val = float(p[3])
                    e = bc.setdefault(lab, {})
                    e[dof] = val

    return dict(nodes=nodes, elems=elems, esets=esets, order=order,
                sect=sect, mats=mats, bc=bc)


# ── report helper ───────────────────────────────────────────────────────────
class Report:
    def __init__(self):
        self.lines = []
        self.fatal = False
        self.warn = False

    def say(self, s=""):
        print(s)
        self.lines.append(s)

    def check(self, ok, label, detail="", fatal=False):
        mark = "OK  " if ok else ("FAIL" if fatal else "WARN")
        self.say(f"  [{mark}] {label}" + (f"   {detail}" if detail else ""))
        if not ok:
            if fatal:
                self.fatal = True
            else:
                self.warn = True
        return ok


def main():
    if not os.path.isfile(cfg.JOB_INP):
        print(f"[check] {cfg.JOB_INP} does not exist yet - nothing to compare "
              f"against. Run ../abaqus/gen_inp_meso.py first.")
        return 0

    r = Report()
    r.say("=" * 72)
    r.say(f"GRAIN MAPPING CHECK   mesh {cfg.MESH_KEY}")
    r.say("=" * 72)
    r.say(f"  Abaqus job .inp : {cfg.JOB_INP}")
    r.say(f"  in-house source : {cfg.BASE_INP}")

    # Variant awareness (case matrix). Layers A and B - mesh and grain
    # partition - are what this script exists for and they are meaningful for
    # every variant: they depend on the base .inp alone. Layers C and D compare
    # the MATERIAL and the BOUNDARY CONDITION against the Abaqus job, which was
    # written for the DIC variant. For any other variant they are guaranteed to
    # differ BY CONSTRUCTION, so running them would print a WARN that means
    # "the study is doing what it was asked to" - the sort of alarm that
    # teaches people to ignore alarms. Skipped, with the reason stated.
    variant_ok = getattr(cfg, "IS_DEFAULT_VARIANT", True)
    if not variant_ok:
        r.say("")
        r.say(getattr(cfg, "variant_banner", lambda: "")())
        r.say("")
        r.say("  Layers C (material) and D (boundary conditions) are SKIPPED for")
        r.say("  this variant: the .inp on disk carries the DIC material and the")
        r.say("  per-node DIC BC, so a difference there is the point of the case,")
        r.say("  not a fault. Layers A and B still apply and still gate the run.")
    r.say("")

    job = parse_job(cfg.JOB_INP)
    prob = inp_io.build_problem(verbose=False)

    node_ids = prob["node_ids"]
    elem_ids = prob["elem_ids"]
    coords = prob["coords"]
    names = prob["grain_names"]

    # ── A. mesh ─────────────────────────────────────────────────────────────
    r.say("A. MESH")
    same_n = (len(job["nodes"]) == len(node_ids) and
              set(job["nodes"]) == set(int(x) for x in node_ids))
    r.check(same_n, "node label sets identical",
            f"{len(job['nodes'])} vs {len(node_ids)}", fatal=True)
    if same_n:
        jc = np.array([job["nodes"][int(L)] for L in node_ids])
        d = np.abs(jc - coords).max()
        r.check(d <= 1e-9, "node coordinates identical (base x SCALE_FACTOR)",
                f"max |diff| = {d:.3e} mm", fatal=True)

    same_e = (len(job["elems"]) == len(elem_ids) and
              set(job["elems"]) == set(int(x) for x in elem_ids))
    r.check(same_e, "element label sets identical",
            f"{len(job['elems'])} vs {len(elem_ids)}", fatal=True)
    if same_e:
        conn_lab = node_ids[prob["elements"]]
        bad = sum(1 for k, L in enumerate(elem_ids)
                  if job["elems"][int(L)] != tuple(int(x) for x in conn_lab[k]))
        r.check(bad == 0, "element connectivity identical",
                f"{bad} elements differ", fatal=True)

    # ── B. grain partition ──────────────────────────────────────────────────
    r.say("")
    r.say("B. GRAIN PARTITION")
    jset = {k: v for k, v in job["esets"].items() if k.upper().startswith("FACE")}
    r.check(set(jset) == set(names), "same set of Face_* elsets",
            f"{len(jset)} in .inp vs {len(names)} in the mesh", fatal=True)

    n_diff = 0
    worst = []
    for g, nm in enumerate(names):
        mine = set(int(x) for x in elem_ids[prob["elsets"][nm]])
        theirs = set(jset.get(nm, []))
        if mine != theirs:
            n_diff += 1
            if len(worst) < 5:
                worst.append(f"{nm}(+{len(mine-theirs)}/-{len(theirs-mine)})")
    r.check(n_diff == 0, "every Face_g holds the same elements in both",
            f"{n_diff} grains differ  {' '.join(worst)}", fatal=True)

    cover = sorted(e for v in jset.values() for e in v)
    r.check(len(cover) == len(set(cover)) == len(elem_ids),
            "grains tile the mesh exactly (no overlap, no orphan)",
            f"{len(cover)} listed, {len(set(cover))} unique, "
            f"{len(elem_ids)} elements", fatal=True)

    bad_sec = [(k, v) for k, v in job["sect"].items()
               if v.upper() != "MATERIAL-" + k.split("_")[-1]]
    r.check(not bad_sec, "Face_N -> Material-N for every grain",
            f"{len(bad_sec)} violations {bad_sec[:3]}", fatal=True)

    # ── C. per-grain material ───────────────────────────────────────────────
    if not variant_ok:
        r.say("")
        r.say("C/D. MATERIAL AND BOUNDARY CONDITIONS - not checked (see above)")
        _verdict(r)
        return _write(r)

    r.say("")
    r.say("C. PER-GRAIN MATERIAL")
    ep_ok = True
    tab_bad = []
    el_bad = []
    max_rel = 0.0
    for g, nm in enumerate(names):
        mat = job["sect"].get(nm)
        M = job["mats"].get(mat) or job["mats"].get(mat.replace("MATERIAL", "Material"))
        if M is None:
            tab_bad.append((nm, "material block missing"))
            continue
        if M["elastic"] is not None:
            if (abs(M["elastic"][0] - cfg.E_MOD) > 1e-6 or
                    abs(M["elastic"][1] - cfg.NU) > 1e-12):
                el_bad.append((nm, M["elastic"]))
        tab = np.array(M["plastic"])
        if len(tab) != cfg.N_TABLE:
            tab_bad.append((nm, f"{len(tab)} points, expected {cfg.N_TABLE}"))
            continue
        if np.abs(tab[:, 1] - cfg.EP_TABLE).max() > 1e-12:
            ep_ok = False
        mine = prob["table"][g]
        rel = np.abs(tab[:, 0] - mine)/np.maximum(np.abs(tab[:, 0]), 1e-30)
        max_rel = max(max_rel, float(rel.max()))
        if rel.max() > RTOL:
            tab_bad.append((nm, f"max rel {rel.max():.3e}  "
                                f".inp sy0 {tab[0,0]:.4f} vs mine {mine[0]:.4f}"))

    r.check(not el_bad, f"*Elastic is {cfg.E_MOD:.0f}, {cfg.NU} in every grain",
            f"{len(el_bad)} differ {el_bad[:3]}")
    r.check(ep_ok, "*Plastic strain knots match cfg.EP_TABLE")
    r.check(not tab_bad,
            "per-grain yield curve identical to the .inp",
            f"{len(tab_bad)}/{len(names)} grains differ, max rel {max_rel:.3e}"
            + ("  e.g. " + "; ".join(f"{a}: {b}" for a, b in tab_bad[:3])
               if tab_bad else ""))
    if tab_bad:
        r.say("        -> the mesh and the grain partition still agree, so grain")
        r.say("           N is grain N; only the CURVE assigned to it drifted.")
        r.say("           Usually the .inp predates a preprocessing change:")
        r.say("           either re-run Abaqus on a regenerated .inp")
        r.say("           (MESO_FORCE_INP=1) or find what changed in")
        r.say("           meso_config (TABLE_FLOAT32, K_SCALE, SEG_FILE, N_EXP).")

    # ── D. boundary conditions ──────────────────────────────────────────────
    r.say("")
    r.say("D. BOUNDARY CONDITIONS")
    mine_bc = {int(node_ids[n]): tuple(prob["bc_u"][k])
               for k, n in enumerate(prob["bc_nodes"])}
    r.check(set(job["bc"]) == set(mine_bc), "same constrained node set",
            f"{len(job['bc'])} in .inp vs {len(mine_bc)} here; "
            f"{len(set(job['bc']) ^ set(mine_bc))} symmetric difference")
    common = set(job["bc"]) & set(mine_bc)
    if common:
        d1 = max(abs(job["bc"][L].get(1, np.nan) - mine_bc[L][0]) for L in common)
        d2 = max(abs(job["bc"][L].get(2, np.nan) - mine_bc[L][1]) for L in common)
        r.check(d1 <= ATOL_U and d2 <= ATOL_U,
                "prescribed displacements identical",
                f"max |d ux| = {d1:.3e}  max |d uy| = {d2:.3e} mm")

    # ── verdict ─────────────────────────────────────────────────────────────
    _verdict(r)
    return _write(r)


def _verdict(r):
    r.say("")
    r.say("=" * 72)
    if r.fatal:
        r.say("VERDICT: FAIL - the two codes are NOT on the same mesh or the same")
        r.say("         grain partition. Nothing downstream is a valid comparison.")
    elif r.warn:
        r.say("VERDICT: MESH AND GRAINS MATCH, material or BCs drifted.")
        r.say("         Grain N is the same grain in both codes; see section C/D")
        r.say("         above for what differs and by how much.")
    else:
        r.say("VERDICT: PASS - identical mesh, identical grain partition,")
        r.say("         identical per-grain yield curve, identical BCs.")
        r.say("         Any difference in the results is a SOLVER difference.")
    r.say("=" * 72)


def _write(r):
    out = os.path.join(cfg.CMP_OUT, f"grain_mapping_{cfg.MESH_KEY}.txt")
    with open(out, "w") as f:
        f.write("\n".join(r.lines) + "\n")
    print(f"\n[check] {out}")
    return 1 if r.fatal else 0


if __name__ == "__main__":
    sys.exit(main())
