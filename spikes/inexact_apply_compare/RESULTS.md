# Exact vs projected apply: measured

Single-threaded, `-O3 -march=native`, Apple M-series laptop, best of 5 per
size. Throughput saturates from about 47k degrees of freedom upward, so the
rows at and below that are the honest ones.

| material | element | ndof | exact MDOF/s | projected MDOF/s | ratio | rel. difference |
|---|---|---|---|---|---|---|
| linear elasticity | TET4 |  46875 | 18.65 | 16.61 | 0.89x | 1.5e-16 |
| linear elasticity | TET4 | 206763 | 17.79 | 15.24 | 0.86x | 1.7e-16 |
| linear elasticity | HEX8 |  46875 | 26.11 | 17.01 | 0.65x | 4.8e-15 |
| linear elasticity | HEX8 | 206763 | 23.24 | 14.98 | 0.64x | 1.1e-14 |
| neohookean Ogden  | TET4 |  46875 |  5.26 |  4.83 | 0.92x | 1.8e-15 |
| neohookean Ogden  | TET4 | 206763 |  4.91 |  4.44 | 0.90x | 3.8e-15 |
| neohookean Ogden  | HEX8 |  46875 |  7.60 |  3.41 | 0.45x | 1.0e-04 |
| neohookean Ogden  | HEX8 | 206763 |  6.97 |  3.27 | 0.47x | 3.6e-05 |

Two things are confirmed and one is refuted.

**The projection is exact where it should be.** Agreement is at round-off
wherever the tangent does not vary over the element: both materials on TET4,
where there is one quadrature point, and linear elasticity on HEX8, where the
tangent does not depend on the state at all. That covers the quadrature
averaging as well as the algebra, since the HEX8 kernel really does average
over eight points.

**The approximation converges.** Neohookean on HEX8 is the only genuinely
inexact case, and its relative error falls with the mesh -- 8.9e-4, 2.3e-4,
1.0e-4, 5.7e-5, 3.6e-5 over the five refinements -- as a consistent projection
should.

**It is slower everywhere.** From 0.90x on TET4 to 0.45x on HEX8, and worst
exactly where the technique was supposed to pay.

## Why, and what it means for the design

The premise was that removing the quadrature loop saves the contraction: one
staged contraction against eight. The operation counts in
`docs/partial_assembly.tex` measure exactly that, and they are not wrong -- but
they count only the contraction.

They do not count forming the tangent. The projected kernel has to *materialise*
`Sbar` -- forty-five independent components after symmetry -- at every
quadrature point in order to average it. The exact kernel never forms the
tangent at all: it computes the directional linearisation `P'(F)[grad h]` at
each point, which is a vector-sized quantity rather than a rank-four one. On
HEX8 that is eight full tangent evaluations traded for eight cheap
linearisations, and the contraction saved does not come close to paying for it.

So the fused kernel -- the one wired here, which rebuilds the tangent on every
apply -- cannot win, and no amount of tuning the contraction will change that.

**The split form is the one that can.** Partial assembly pays when the tangent
is formed once and *reused*: in a Newton step the tangent is fixed while the
Krylov solve applies the operator tens or hundreds of times. Emitting `Sbar`
into storage once per Newton iteration and then applying it costs, per apply,
only the staged contraction -- the thing the operation counts do measure.

That is the measurement worth taking next, and it needs the split kernels: one
writing `Sbar` per element, one reading it. Its break-even is the number of
applies per tangent, which the fused form fixes at one and a Krylov solve makes
large.

# A material with more than one unit

`bench_mixed.cpp`.  Mooney-Rivlin elasticity plus Kelvin-Voigt viscosity is an
energy unit and a residual unit in one material.  The operator's action is the
sum of the two, so the split carries both -- two tangents assembled, two applies
summed -- and is compared against the sum of the two exact kernels.

It broke two assumptions that every single-unit material had quietly satisfied.

**The tangent's state is not only the current state.**  The viscous flux reads
`grad(old(u))`.  The flux form was folding those nine symbols in with `eta_s`
and `eta_b`, which would have emitted per-element field data as scalar uniforms.
`SfemSoAFluxForm` now carries `previous_gradient` separately, taken from the
residual field records that already model it.

**The tangent is not always symmetric.**  An energy's flux is a gradient, so its
tangent is a Hessian and `A[ijkl] == A[klij]` holds by equality of mixed
partials; that symmetry is what folds 81 stored numbers into 45.  A residual's
flux is not a gradient, its tangent is a Jacobian, and the Kelvin-Voigt viscous
tangent violates the symmetry by 0.25 in absolute terms.  Packing it into 45
averaged away its antisymmetric part and returned a different operator: 5.3e-2
relative error on TET4, an element where the projection is provably exact.  The
plan now decides symmetry symbolically and stores 45 or 81 accordingly, and it
defaults to *unsymmetric*, because storing 81 for a symmetric tangent wastes
memory while storing 45 for an unsymmetric one is silently wrong.

With that fixed both units reproduce their exact kernels to round-off on TET4:

| unit | front end | tangent | stored numbers | split vs exact |
|---|---|---|---|---|
| elastic (Mooney-Rivlin) | energy   | symmetric Hessian    | 45 | 7.27e-16 |
| viscous (Kelvin-Voigt)  | residual | unsymmetric Jacobian | 81 | 7.29e-16 |

The apply kernels are unchanged by any of this.  The viscous apply takes the
tangent and the increment -- no geometry, no state, no previous state, no
viscosity parameters.  However many state fields a material reads, they are all
absorbed into `Sbar` at assembly time, so extra state costs one argument on the
once-per-tangent kernel and nothing at all per Krylov iteration.

## Measured, single-threaded, 206763 dof

| | exact | st. f64 | st. f32 | st. f16 | assembly |
|---|---|---|---|---|---|
| MDOF/s | 2.39 | 6.80 | 8.07 | 9.92 | 1.32 |
| speed-up | -- | 2.85x | 3.38x | 4.15x | -- |

Correctness 4.44e-15 against the summed exact kernels.  Store 1008 / 504 / 260
bytes per element (45 elastic + 81 viscous), against 360 / 180 / 94 for a single
hyperelastic material.

The larger store costs bandwidth and the technique still wins by more here than
on any single-unit material, because the exact side has to evaluate *two*
expensive tangents per apply and the split evaluates neither.

## Threads, and what OpenMP costs the split

Same problem, 206763 dof, on 8 performance plus 2 efficiency cores.

| threads | exact | st. f64 | st. f32 | st. f16 | assembly | f16 speed-up |
|---|---|---|---|---|---|---|
|  1 |  2.52 |  4.89 |  5.87 |  7.01 | 1.38 | 2.78x |
|  2 |  5.25 |  9.79 | 11.44 | 13.50 | 2.76 | 2.57x |
|  4 | 10.48 | 19.46 | 24.17 | 28.84 | 5.58 | 2.75x |
|  8 | 13.04 | 23.00 | 27.11 | 32.58 | 6.26 | 2.50x |
| 10 | 13.53 | 24.05 | 29.65 | 36.05 | 7.92 | 2.66x |

Break-even stays near three applies per tangent throughout (2.7 to 3.5).

**The atomic scatter is what changes between the two builds, and it costs the
split disproportionately.**  Compare the single-thread rows above with the
single-threaded table before them, which was built without OpenMP: the exact
kernel is unaffected (2.39 against 2.52), while the f16 stored apply falls from
9.92 to 7.01, 29 per cent, purely from compiling with `-fopenmp`.  With OpenMP
the scatter is a real `#pragma omp atomic update` even on one thread; without it
the pragma disappears and the scatter is a plain `+=`.  The split does much less
arithmetic per scatter than the exact kernel does, so the same per-scatter cost
is a far larger share of its total, and the advantage falls from 4.15x to about
2.7x.

Parallel scaling itself is not the problem and is nearly identical for both:
5.4x for the exact apply on ten cores against 5.1x for the stored one.  The
split does not scale worse, it pays a constant tax.  That makes the scatter --
not the contraction, and not the store -- the first place to look for more
speed on this path.

## Reproducing

`bench.cpp` (TET4) and `bench_hex8.cpp` (HEX8) take the material through
`-DEXACT_APPLY=`, `-DPROJECTED_APPLY=`, `-DMATERIAL_INEXACT_HEADER=` and
`-DEXACT_TAKES_STATE` for a material whose exact apply takes one. Generate with
`inexact_apply=True`, then compile the bench against the element operator (and
the PROTEUS_HEX8 unit, which HEX8 aliases into).

# The split form: measured

`bench_split.cpp`, driven by `run_split.sh`, generates the split kernels, builds
`Sbar` once into a store, and applies it at three storage precisions.  Single
threaded, `-O3 -march=native`, Apple M-series laptop, best of 7 after one untimed
warm-up pass.  All figures MDOF/s.

At 206763 degrees of freedom:

| material | element | exact | st. f64 | st. f32 | st. f16 | assembly | speed-up (f16) | break-even f64/f32/f16 |
|---|---|---|---|---|---|---|---|---|
| neohookean Ogden  | TET4  |  5.14 | 23.25 | 26.45 | 30.55 |  4.25 | 5.94x | 1.6 / 1.5 / 1.5 |
| neohookean Ogden  | TET10 | 12.14 | 42.26 | 44.15 | 46.69 |  3.94 | 3.85x | 4.3 / 4.3 / 4.2 |
| neohookean Ogden  | HEX8  |  6.84 | 16.92 | 17.62 | 17.85 |  3.92 | 2.61x | 2.9 / 2.8 / 2.8 |
| linear elasticity | TET4  | 18.15 | 23.22 | 26.72 | 30.24 | 17.08 | 1.67x | 4.9 / 3.3 / 2.7 |
| linear elasticity | TET10 | 26.72 | 43.25 | 45.03 | 46.82 | 149.84 | 1.75x | 0.5 / 0.4 / 0.4 |
| linear elasticity | HEX8  | 24.80 | 17.26 | 17.58 | 17.98 | 112.90 | 0.72x | never |

Break-even is how many applies must share one tangent before the split has repaid
its assembly: `(1/a) / (1/e - 1/s)`.  Five of six pairs win, by 1.7x to 5.9x, at
between 0.4 and 4.3 applies per tangent -- well inside one Krylov solve.

**HEX8 linear elasticity never pays.**  Its break-even is negative at every
precision: the stored apply is slower than the exact one, so no amount of reuse
repays the assembly.  Twenty-four degrees of freedom per element against a
45-number tangent, and an exact apply that is already cheap and sum-factorizable,
make the store pure added bandwidth.  That is the boundary of the technique.

**Precision buys less than the byte count suggests.**  f64 to f16 is a 3.8x
reduction in the store for 1.07x to 1.31x throughput, and on HEX8 almost nothing.
These kernels are not purely bandwidth-bound on the tangent: the gather of the
increment, the scatter of the output and the contraction all cost.  f32 is the
sensible default -- half the bytes of f64, most of the speed of f16, and seven
digits of accuracy instead of three.

There is no fused variant.  An earlier revision emitted one, which rebuilt `Sbar`
on every apply; it was scaffolding for the comparison and is gone.  The stored
apply gates correctness now, and did so identically while both existed.

## Warp sweep: deviation against severity

`warp_sweep.cpp` / `run_warp.sh`.  The displacement is `u(x) = L x + w N(x)` with
`L` a fixed mild linear map and `N` smooth and nonlinear; `w` is the warp
severity.  The linear part has a constant deformation gradient, so at `w = 0` the
tangent does not vary over an element and the projection is *exact*.  Validity is
checked rather than assumed on every row: the tangent is built through
`log(det F)`, so a non-positive Jacobian anywhere makes an entry non-finite.  It
stayed finite at every severity below.

neohookean Ogden, n = 16, relative deviation of the stored-f64 apply from the exact:

| warp `w` | TET4 | TET10 | HEX8 |
|---|---|---|---|
| 0.000 | 8.9e-16 | 2.4e-14 | 2.7e-15 |
| 0.001 | 1.0e-15 | 2.5e-05 | 2.0e-07 |
| 0.010 | 1.0e-15 | 2.5e-04 | 2.0e-06 |
| 0.050 | 1.0e-15 | 1.2e-03 | 1.0e-05 |
| 0.100 | 1.0e-15 | 2.5e-03 | 2.0e-05 |
| 0.200 | 1.0e-15 | 5.0e-03 | 4.1e-05 |
| 0.400 | 1.0e-15 | 1.0e-02 | 8.3e-05 |
| 0.800 | 9.6e-16 | 2.1e-02 | 1.7e-04 |
| 1.200 | 9.4e-16 | 3.1e-02 | 2.7e-04 |
| 1.600 | 9.3e-16 | 4.3e-02 | 3.7e-04 |

TET4 is exact at every severity, because its tangent is constant over the element
whatever the deformation.  TET10 and HEX8 are both exactly first order in `w`,
which is the correct signature for a one-term projection: the error is
proportional to how much the tangent varies across the element.  TET10's constant
is about 120x HEX8's.

The `w = 0` row is worth more than it looks.  A quadratic element reproduces a
linear displacement exactly only if the nodal values land on the right nodes, so
round-off there clears the TET10 edge ordering, `Wbar`, the packed symmetry and
the contraction in one measurement -- and it does so for a *state-dependent*
material, which the two earlier controls could not.  Neither a zero state nor a
material whose tangent ignores the state can detect a scrambled state field.

## The TET10 "non-convergence" was the metric, not the kernel

Measured against a smooth increment, the TET10 deviation looked like a defect: it
sat at 1.0e-2 across an eightfold refinement while HEX8 fell like `h^2`.

| n | TET10, smooth | HEX8, smooth | TET10, random | HEX8, random |
|---|---|---|---|---|
|  4 | 1.07e-02 | 1.16e-03 | 2.73e-03 | 1.11e-03 |
|  8 | 1.04e-02 | 3.27e-04 | 1.30e-03 | 3.73e-04 |
| 16 | 1.01e-02 | 8.34e-05 | 6.08e-04 | 1.26e-04 |
| 24 | 9.99e-03 | 3.70e-05 | 3.96e-04 | 6.86e-05 |
| 32 | 9.93e-03 | 2.08e-05 | 2.95e-04 | 4.69e-05 |

With a white-noise increment TET10 converges at `O(h^1.07)` -- first order, which
is exactly what the projection is.  The kernel was never wrong.

The reason the smooth measurement misleads: for a smooth field the assembled
`(H h)_p` is a discrete second derivative, so contributions from neighbouring
elements largely cancel and the denominator collapses by an extra order, while
the per-element projection errors carry independent signs and do not cancel at
all.  A ratio of a non-cancelling numerator to a cancelling denominator can sit
flat while the approximation underneath converges perfectly well.  HEX8 hid this
because its symmetric element and symmetric Gauss rule let some of the error
cancel too.

Independently confirmed away from the mesh entirely: on a single TET10 element in
exact arithmetic, integrating the exact and projected actions with a high-order
rule, the projection converges at rate 0.92 as the element shrinks (8.5e-3 at
h = 0.4 down to 1.0e-3 at h = 0.025), and TET4 is exact to round-off.

So the practical statement about accuracy is: at a realistic deformation the
projected apply on TET10 differs from the exact one by one to three percent, it
is first order in both the mesh size and the warp, and whether that is acceptable
is a question about the solver that consumes it -- it is an inexact Newton
operator -- not about the kernel.

## Reproducing

    spikes/inexact_apply_compare/run_split.sh <material> <element> [repeats]
    spikes/inexact_apply_compare/run_warp.sh  <material> <element> [n]

Elements are TET4, HEX8 and TET10.  `WARP_EXTRA_FLAGS=-DRANDOM_INCREMENT` selects
the white-noise increment.  `SFEM_MAIN_CHECKOUT`, `SFEM_BUILD`, `SFEM_PYTHON` and
`SFEM_SPIKE_WORK` override the paths.  Both keep a full unfiltered log.
