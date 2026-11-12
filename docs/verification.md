# Verification against Abaqus/Standard

Both solvers were verified against Abaqus/Standard on the same boundary-value
problem they are used for. This page carries the full record; the thesis quotes
only the headline table.

The question is not whether the two codes agree to machine precision — they do
not, and should not be expected to, since they use different increment
schedules and different convergence criteria. The question is whether the
difference between them is **smaller than the uncertainty already present in the
experimental input**. It is, by an order of magnitude.

## What makes the comparison fair

Any verification is only worth as much as the equality of its inputs. On both
paths these are identical:

- the same nodal coordinates and the same element connectivity
- the same element-wise $(\sigma_{y0}, K, n)$ map
- the same elastic constants
- the same DIC-measured boundary displacements
- **the same 50-point tabular hardening curve** — the in-house solver reads the
  same discrete table that is written into the Abaqus `*Plastic` card, so the
  constitutive input is bitwise identical rather than merely equivalent

Any remaining discrepancy can therefore only come from the numerical solution
procedure: the element implementation, the constitutive integration, or the
convergence tolerances.

Abaqus reports fields at integration points, the in-house solvers at element
centroids. Before comparison both are projected onto the same grid with one
routine applied to both.

## Microscale — CPS4 pixel mesh

### Field agreement

Heterogeneous $500\times500$ window (250,000 elements, 502,002 DOF), sampled at
full load. A demanding test: 88% of the area deforms plastically, the identified
yield stress spans 50–1380 MPa and the hardening coefficient 85–1188 MPa, so the
constitutive update is exercised across the full range of the local material
field.

| Quantity | Points | Rel. RMS (%) | RMSE | Max abs. | Corr. |
|---|---:|---:|---:|---:|---:|
| $u_2$ (mm) | 251,001 | 0.000 | 3.158e-7 | 1.165e-6 | 1.000000 |
| $\varepsilon_{vM}^{ps}$ | 250,000 | 0.438 | 1.105e-5 | 3.924e-4 | 0.999961 |
| $\sigma_{22}$ (MPa) | 250,000 | 0.339 | 3.826e-1 | 6.392 | 0.999964 |
| $\varepsilon_{p}$ | 250,000 | 0.678 | 1.075e-5 | 3.851e-4 | 0.999956 |

Quoted against Abaqus as reference. The RMS difference in vertical displacement
is four orders of magnitude below the displacement field itself, and far below
the precision of the DIC data used as boundary conditions. Derived quantities
read larger because differentiation and constitutive integration amplify small
displacement differences.

### Where the residual lives

The difference is **spatially structured rather than random**: it follows the
localization bands, is largest in plastically active regions and negligible in
the elastic regions between them. That pattern is consistent with a difference
in constitutive integration, not an error in the formulation — Abaqus uses 18
adaptive increments where the in-house solver uses 20 uniform ones, and the
plastic response is path dependent. A formulation error would instead produce a
systematic pattern tied to the mesh or the boundary, and none is observed.

### Agreement does not decay with problem size

| Window | Points | $u_2$ (%) | $\sigma_{22}$ (%) | $\varepsilon_{vM}^{ps}$ (%) |
|---|---:|---:|---:|---:|
| $100\times100$ | 10,201 | 0.000 | 0.595 | 0.476 |
| $250\times250$ | 63,001 | 0.000 | 0.371 | 0.427 |
| $500\times500$ | 251,001 | 0.000 | 0.339 | 0.438 |
| $1000\times500$ | 501,501 | 0.001 | 0.311 | 0.404 |
| $750\times750$ | 564,001 | 0.002 | 0.460 | 0.683 |

Five windows spanning a factor of 56 in element count. The relative RMS
difference stays between 0.40% and 0.68% in $\varepsilon_{vM}^{ps}$ and between
0.31% and 0.60% in $\sigma_{22}$, while the displacement difference stays below
0.002%.

The ordering is not monotone — the largest window is not the worst and the
smallest is not the best — so these five points establish the **absence of a
trend**, not a scaling law. Nor do they separate conditioning from size, since
the windows differ in aspect ratio as well as element count. What they do
establish is that the margin argued below holds at every size tested.

### The margin that matters

The identified parameter maps carry field-level scatter of **12–20%**. The
difference between the two solvers is 0.3–0.7%. The solver is therefore more
than an order of magnitude away from being a significant contributor to the
uncertainty of any reported result.

For scale on the other side: three nominally identical Abaqus runs on the same
workstation differed by 8.9% in elapsed time, and multithreaded direct solvers
accumulate round-off differently as thread scheduling changes. Bitwise identity
was never the target.

### Cost

Same $500\times500$ window, one workstation with 22 logical cores and 31.5 GB
usable memory.

| Stage | Abaqus path | In-house path |
|---|---:|---:|
| Write input deck (s) | 14.8–16.5 | — |
| Solver job (s) | 852.2–927.9 | 1472.9 |
| Fields into arrays (s) | 474.9–809.1 | — |
| **End to end (s)** | **1360–1754** | **1473** |

Normalised per Newton iteration — the only basis on which two different
increment schedules compare — Abaqus takes 8.8 s against the in-house solver's
13.0 s.

:::{note}
**No speed claim is made.** On the analysis alone the commercial code is
faster. What the in-house path removes is the deck-writing and ODB-extraction
stages: fields are written directly, at arbitrary load levels, with no
post-processing round trip. That is the reason it exists, not performance.
:::

## Mesoscale — CPS3 grain mesh

Unstructured SALOME grain mesh, `fine_no_twins`: 90,560 CPS3 elements, 45,658
nodes, **244 grains**, 840 constrained boundary nodes. Final frame:

| Quantity | Rel. difference |
|---|---:|
| $u$ at prescribed nodes | 4.2e-17 mm |
| $u$ at free nodes | 1.9e-6 mm rms (0.003%) |
| $\sigma_{vM}$ | 0.045% |
| $\varepsilon_{vM}$ | 0.15% |
| PEEQ | 0.21% |

Solve: 212.6 s over 20 increments, no cutbacks. Area-weighted domain means agree
to six decimals.

The per-grain material mapping was checked layer by layer against the existing
Abaqus deck, including a **per-grain yield-curve comparison over all 244 grains
with a maximum relative difference of exactly zero** — the replicated
preprocessing is bit-identical to the original.

### Reading the shear components

The shear components look worse ($\sigma_{12}$ 0.82%, $\gamma_{12}$ 1.48%) but
this is a normalisation artefact: their absolute RMS difference is *smaller*
than that of the normal components, divided by a reference roughly ten times
smaller. Nothing is shear-specific.

### Where the residual lives

Mean $|\Delta\sigma_{vM}|$ is 0.586 MPa in elastic elements against 0.081 MPa in
plastic ones — seven times higher, on only 678 of 90,560 elements, since 99.25%
of the mesh yields. Only 17 elements disagree about whether they yielded at all.

Grain boundaries are clean: 0.096 MPa at boundaries against 0.079 MPa in grain
interiors, so the 244 discrete materials cause no pathology. The residual is a
**load-path difference at the elastic/plastic frontier**, not a formulation
difference.

:::{warning}
A known contributor: on the first pass the in-house solver ran before the ODB
export existed, so it used an even 1/N increment ladder while Abaqus used its
own adaptive times, whose first increment is 50x smaller — right where nearly
every element yields. Running the Abaqus chain first, so the FEM picks up the
exported frame times, is expected to reduce this further. Untested as of
writing.
:::

## Reproducing it

```bash
# microscale: built-in analytic case
python microscale/solver/fem_pixel.py

# mesoscale: full chain on a synthetic fixture, no Abaqus and no real data
cd mesoscale/runners
python selftest.py             # exercises every script end to end
python check_grain_mapping.py  # proves grain N is grain N in both codes
```

`selftest.py` injects a known error and confirms the comparison machinery
recovers it, so a pass means the plumbing is sound as well as the solver.
