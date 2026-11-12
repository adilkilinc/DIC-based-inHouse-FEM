# Workstation cost and scaling of the pixel solver

These measurements were removed from Chapter 4 of the thesis (former §4.6.2) and are kept here
as supporting documentation. They were made on a workstation before the full-domain HPC
simulations, to establish the scaling of monolithic solves.

## Runtime and memory

Timing the solver on square regions from 100 × 100 to 500 × 500 elements, at the retained
increment count, gives the power law

$$t = 8.648\times10^{-4}\,n^{1.159}\ \mathrm{s},$$

with a maximum residual of 16.5 % over the fitted range. Peak memory is linear in the
element count:

$$M = 417\ \mathrm{MB} + 21.51\ \mathrm{KB}\times n .$$

The fitted exponent describes total runtime over the tested range. It is lower than the
$n^{1.5}$ scaling expected for sparse-direct factorisation of a two-dimensional problem, so
other stages contribute substantially at these sizes; the fit does not separate them.

## Parallel throughput

200 × 200 region (40 000 elements, solo cost 188.6 s) on a 22-core workstation. Throughput
gain is the number of regions solved per unit time relative to one worker.

| Workers | Wall (s) | Per region (s) | Throughput gain | Efficiency (%) |
|--:|--:|--:|--:|--:|
| 1  | 185.5 | 180.5 | 1.02 | 101.7 |
| 2  | 180.0 | 179.6 | 2.10 | 104.8 |
| 4  | 206.8 | 205.2 | 3.65 | 91.2  |
| 8  | 348.0 | 342.3 | 4.34 | 54.2  |
| 16 | 764.1 | 703.7 | 3.95 | 24.7  |

Throughput peaks at a gain of 4.34 with eight workers and falls to 3.95 with sixteen;
additional concurrent solves stop improving throughput beyond this point. Efficiencies slightly
above 100 % at one and two workers are within the 8.9 % run-to-run variability of the platform.

The full-domain thesis simulations were run on single HPC nodes with 32 cores and the
multithreaded direct solver; their costs are reported in the thesis with the results.
