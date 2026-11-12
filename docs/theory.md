# Formulation

This page is the solver manual: the boundary-value problem, the constitutive
model, the return mapping and the consistent tangent, as implemented in
`microscale/solver/fem_pixel.py` (CPS4) and `mesoscale/solver/fem_tri.py`
(CPS3). Both solve the same physics on different discretisations.

## Problem statement

### Strong form

The body $\Omega$ is a two-dimensional domain in plane stress. Inertia and body
forces are neglected, so the governing equation is quasi-static equilibrium,

$$\nabla\cdot\boldsymbol{\sigma}=\mathbf{0}\qquad\text{in }\Omega,$$

closed by the small-strain kinematic relation and by boundary conditions. The
problem is displacement-driven on the **entire** boundary: the measured DIC
displacement field is prescribed at every node of all four edges,

$$\boldsymbol{\varepsilon}=\tfrac12\bigl(\nabla\mathbf{u}+\nabla\mathbf{u}^{\mathsf T}\bigr),
\qquad \mathbf{u}=\bar{\mathbf{u}}\quad\text{on }\partial\Omega .$$

There is no Neumann portion of the boundary and no external load vector. This is
a structural feature of the problem rather than a detail: the solution is driven
purely by the prescribed boundary motion, the interior is solved for
equilibrium, and **the reaction forces recovered at the boundary are an output
rather than an input**.

### Weak form

Multiplying by an admissible variation $\delta\mathbf{u}$ vanishing on
$\partial\Omega$ and integrating by parts gives the principle of virtual work.
With no body force and no prescribed traction it reduces to

$$\int_{\Omega}\delta\boldsymbol{\varepsilon}^{\mathsf T}\boldsymbol{\sigma}\;\mathrm{d}\Omega=0
\qquad\forall\,\delta\mathbf{u}\in V_{0}.$$

The discrete counterpart of this statement is the residual that the Newton
iteration drives to zero.

### Plane stress and Voigt notation

Plane stress assumes $\sigma_{33}=\sigma_{13}=\sigma_{23}=0$, with the
out-of-plane strain recovered afterwards from the constitutive law. Stresses and
strains are stored in Voigt form using **engineering** shear strain,

$$\boldsymbol{\sigma}=\begin{bmatrix}\sigma_{11}\\\sigma_{22}\\\sigma_{12}\end{bmatrix},
\qquad
\boldsymbol{\varepsilon}=\begin{bmatrix}\varepsilon_{11}\\\varepsilon_{22}\\\gamma_{12}\end{bmatrix},
\qquad \gamma_{12}=2\varepsilon_{12}.$$

:::{important}
The engineering convention is used consistently throughout, **including in the
output arrays**. That is why the `E` component written by the solver is directly
comparable with the `E` field exported from an Abaqus ODB — but it also means
that any post-processing formula expecting the *tensor* shear strain
$\varepsilon_{xy}$ must halve the stored `E[..., 2]` first.
:::

The thickness is taken as unity, so all forces reported by the solver are per
unit thickness.

## Constitutive model

### Elasticity

Isotropic linear elasticity specialised to plane stress,

$$\mathbf{C}=\frac{E}{1-\nu^{2}}
\begin{bmatrix}1&\nu&0\\ \nu&1&0\\ 0&0&\tfrac{1-\nu}{2}\end{bmatrix},$$

with $E$ and $\nu$ uniform over the domain. **Heterogeneity enters exclusively
through the plastic parameters.**

### Yield function

Yielding follows the von Mises criterion. In plane stress the equivalent stress
is a quadratic form in the Voigt stress vector,

$$\sigma_{vM}^{2}
=\sigma_{11}^{2}-\sigma_{11}\sigma_{22}+\sigma_{22}^{2}+3\sigma_{12}^{2}
=\boldsymbol{\sigma}^{\mathsf T}\mathbf{M}\boldsymbol{\sigma},
\qquad
\mathbf{M}=\begin{bmatrix}1&-\tfrac12&0\\ -\tfrac12&1&0\\ 0&0&3\end{bmatrix},$$

so that the yield function is

$$f(\boldsymbol{\sigma},\varepsilon_p)=\sigma_{vM}(\boldsymbol{\sigma})-\sigma_y(\varepsilon_p)\le0,$$

where $\varepsilon_p$ is the accumulated equivalent plastic strain. Writing the
criterion as a matrix quadratic form is what makes the plane-stress return
mapping tractable in closed form.

### Hardening

Isotropic hardening, with the flow stress a function of $\varepsilon_p$ alone,

$$\sigma_y(\varepsilon_p)=\sigma_{y0}+K\,h(\varepsilon_p),
\qquad H=\frac{\mathrm{d}\sigma_y}{\mathrm{d}\varepsilon_p}=K\,h'(\varepsilon_p).$$

Both $\sigma_{y0}$ and $K$ are supplied as **per-element maps**, so each pixel
(or each grain, at the mesoscale) carries its own yield stress and hardening
coefficient; $E$, $\nu$ and the exponent $n$ are global. Two forms of $h$ are
available:

`hardening='ludwik'`
: The analytic power law $h=\varepsilon_p^{\,n}$, $h'=n\,\varepsilon_p^{\,n-1}$.
  It hardens without bound and is the natural choice when the solver is used on
  its own.

`hardening='tabular'`
: A piecewise-linear interpolation of $\varepsilon_p^{\,n}$ through 50 equally
  spaced knots on $[0,\varepsilon_p^{\max}]$ with $\varepsilon_p^{\max}=0.2$,
  clamped to perfectly plastic beyond the last knot. This is not an
  approximation of convenience — it is exactly the discrete table written into
  the Abaqus `*Plastic` card, and therefore exactly what Abaqus integrates.
  Selecting it removes the constitutive input as a source of discrepancy in any
  Abaqus comparison, and it is the mode used for verification.

The clamp makes $H=0$ identically beyond the last knot. The return mapping and
the tangent are both written to stay stable in that perfectly plastic limit,
which a naive Newton scheme would not survive.

### Flow rule

Plastic flow is associative, so the flow direction is the gradient of the yield
surface,

$$\mathbf{n}=\frac{\partial f}{\partial\boldsymbol{\sigma}}
=\frac{\mathbf{M}\boldsymbol{\sigma}}{\sigma_{vM}},
\qquad \Delta\boldsymbol{\varepsilon}_{p}=\Delta\gamma\,\mathbf{n}.$$

With the engineering-shear Voigt convention this normalisation makes the plastic
multiplier $\Delta\gamma$ identical to the increment of equivalent plastic
strain, because the plastic work per unit volume is

$$\boldsymbol{\sigma}^{\mathsf T}\Delta\boldsymbol{\varepsilon}_{p}
=\Delta\gamma\,\frac{\boldsymbol{\sigma}^{\mathsf T}\mathbf{M}\boldsymbol{\sigma}}{\sigma_{vM}}
=\Delta\gamma\,\sigma_{vM}.$$

The implementation relies on this identity: the same scalar returned by the
return mapping is used both as the multiplier and as the increment added to
$\varepsilon_p$.

## Return mapping

### Elastic predictor

Freezing the plastic strain at its value at the start of the increment,

$$\boldsymbol{\sigma}^{\mathrm{tr}}
=\mathbf{C}\bigl(\boldsymbol{\varepsilon}_{n+1}-\boldsymbol{\varepsilon}_{p,n}\bigr).$$

If $f(\boldsymbol{\sigma}^{\mathrm{tr}},\varepsilon_{p,n})\le0$ the step is
elastic, $\Delta\gamma=0$ and the tangent is simply $\mathbf{C}$. The test is
evaluated for all integration points at once and only the yielding subset is
passed to the corrector.

### Plastic corrector and closed-form inversion

The backward-Euler stress update, with the consistency condition used to replace
$\sigma_{vM}$ by the flow stress, becomes a linear system in the unknown stress
for any given

$$r=\frac{\Delta\gamma}{\sigma_y(\varepsilon_{p,n}+\Delta\gamma)},
\qquad
\mathbf{A}\boldsymbol{\sigma}=\boldsymbol{\sigma}^{\mathrm{tr}},
\qquad \mathbf{A}=\mathbf{I}+r\,\mathbf{C}\mathbf{M}.$$

In plane stress $\mathbf{C}\mathbf{M}$ is symmetric, its two in-plane diagonal
entries are equal and its shear row and column are fully decoupled. $\mathbf{A}$
inherits that block structure and is inverted analytically. With
$a=1+r(\mathbf{CM})_{11}$, $b=r(\mathbf{CM})_{12}$,
$\tilde c=1+r(\mathbf{CM})_{33}$ and $d=a^{2}-b^{2}$,

$$\sigma_{11}=\frac{a\,\sigma^{\mathrm{tr}}_{11}-b\,\sigma^{\mathrm{tr}}_{22}}{d},
\qquad
\sigma_{22}=\frac{a\,\sigma^{\mathrm{tr}}_{22}-b\,\sigma^{\mathrm{tr}}_{11}}{d},
\qquad
\sigma_{12}=\frac{\sigma^{\mathrm{tr}}_{12}}{\tilde c}.$$

**No $3\times3$ linear solve is performed anywhere in the constitutive routine.**
That is what allows the return mapping to run as pure array arithmetic for all
million integration points of a $500\times500$ mesh simultaneously.

### The scalar residual

Everything above is a function of the single unknown $\Delta\gamma$.
Substituting the closed-form stress into the yield condition gives

$$R(\Delta\gamma)=\sigma_{vM}\bigl(\boldsymbol{\sigma}(\Delta\gamma)\bigr)
-\sigma_y(\varepsilon_{p,n}+\Delta\gamma)=0 .$$

On the plastic branch $R(0)>0$ by construction and $R$ decreases monotonically,
so the root exists and is unique. The implementation exploits that rather than
trusting Newton alone: a bracket is established (upper bound from $10^{-4}$,
doubled up to eighty times until $R$ changes sign), then Newton from the bracket
midpoint with a numerically differentiated slope, **each proposed step tested
against the bracket and replaced by bisection if it falls outside or is not
finite**. Convergence is $|R|/\sigma_y<10^{-10}$, capped at 100 iterations.

The guarded scheme costs little — the bisection fallback is rarely triggered —
but it makes the return mapping unconditionally robust, including in the
perfectly plastic clamp of the tabular law where $H=0$ and unguarded Newton
would take an unbounded step.

## Global solution procedure

The load is applied incrementally over a pseudo-time $t\in[0,1]$, exactly the
convention Abaqus uses for a single static step of period 1. Within each
increment the residual is driven to zero by Newton-Raphson using the
analytically derived **consistent (algorithmic) tangent** — not a numerical
one, and not an external material subroutine.

- **Convergence** — relative force residual, default $10^{-6}$. Abaqus applies
  roughly $5\times10^{-3}$ by default, so the in-house criterion is stricter.
- **Cutback** — an increment that fails to converge within `max_nr` iterations
  is halved and retried.
- **Adaptive stepping** — after a successful increment the step grows by 1.5x up
  to the nominal size. The *natural* step is tracked separately from the step
  actually taken, so that clipping an increment to land exactly on a requested
  snapshot fraction does not restart the growth from the clipped remainder.
- **Linear solver** — MKL Pardiso through `pypardiso` when available, SciPy's
  sparse LU otherwise. One factorisation per Newton iteration.
- **Line search** — the CPS3 solver (`fem_tri.py`) uses a backtracking line
  search, needed because full Newton overshoots on steps where the whole domain
  yields at once. `fem_pixel.py` does not: the behaviour was measured and does
  not escalate with mesh size, which is attributed to CPS4's four Gauss points
  averaging the constitutive response within an element.

### Run logs

Both solvers mirror their Newton history to a log file, **written even when
`verbose=False`** — a driver that runs quiet is exactly the one whose behaviour
you cannot inspect later. One file per process, appended across calls, so a
driver that solves many tiles produces one log rather than hundreds.

```
./fem_logs/<module>_<stamp>_pid<pid>.log   default, in the cwd
FEM_LOG_DIR=<dir> / FEM_LOG_FILE=<path>    relocate
FEM_LOG=0                                  off
```

The log holds a configuration header, every Newton iteration (marked
`[residual grew]`, `line search: step ...` or `CUTBACK` as applicable), a
summary, and one machine-readable `[csv]` line so a parameter sweep can be
grepped rather than parsed. The same counters come back in the result dict:
`n_inc`, `n_nr`, `n_cutback`, `n_res_growth`, `n_line_search` (CPS3 only),
`wall_s` and `log_file`.

## Built-in verification

Running either solver as a script executes its own verification and reports the
error against a closed-form answer:

```bash
python microscale/solver/fem_pixel.py   # equal-biaxial tension, analytic solution
python mesoscale/solver/fem_tri.py      # elastic patch test + equal-biaxial plasticity
```

See [Verification](verification.md) for the comparison against Abaqus/Standard.
