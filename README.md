# DIC-based in-house FEM

Finite-element reconstruction of DIC displacement fields with spatially varying Ludwik descriptors, at the pixel scale (CPS4) and on grain or grain–twin meshes (CPS3). The experimental data are not included.

## Layout

| Folder | Contents |
|---|---|
| `microscale/solver/` | `fem_pixel.py` (pixel solver), `seam_metrics.py` |
| `mesoscale/solver/` | `fem_tri.py` (triangle-mesh solver) |
| `microscale/abaqus/`, `mesoscale/abaqus/` | Abaqus reference path |
| `mesoscale/runners/`, `mesoscale/mesh/` | Mesoscale chain, self-test, base meshes |
| `campaign/` | Thesis cases, Dataset 1 and Dataset 2 |
| `legacy/` | Earlier scripts |
| `docs/` | Documentation sources |

`campaign/`:

| Path | Step |
|---|---|
| `identification/dataset1/`, `identification/dataset2/` | DIC (DIS optical flow) and original identifiers |
| `identification/d1_identify.py`, `d2_identify.py` | Yield and hardening maps |
| `power/d_kpow_maps.py` | Power-law K maps |
| `preflight.py`, `run_case.py`, `power/run_fem.py` | Pixel-scale cases (`campaign.json`, `power/followup.json`) |
| `meso/` | Dataset 1 regional materials and cases |
| `meso_d2/` | Dataset 2 meshes, problems and cases (`cases.json`) |
| `power_regional/` | Regional cases with power-law K |

## Install

```bash
pip install -r requirements.txt
pip install -r campaign/requirements.txt
```

## Run

```bash
cd microscale/solver && python fem_pixel.py
cd mesoscale/solver  && python fem_tri.py
cd mesoscale/runners && python selftest.py
```

Campaign data roots: `OLD_EXPERIMENT_ROOT` (Dataset 1), `NEW_EXPERIMENT_ROOT` (Dataset 2).

## Licence

GNU Lesser General Public License v2.1. See [`LICENSE`](LICENSE).
