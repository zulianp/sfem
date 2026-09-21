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


# The split form: measured

`bench_split.cpp`, driven by `run_split.sh`, generates the split kernels, builds
`Sbar` once into a store, and applies it at three storage precisions.

Single-threaded, `-O3 -march=native`, Apple M-series laptop, twenty repetitions
inside each timing window, at 206763 degrees of freedom.  All figures MDOF/s.
TET10 is swept over n = 4..20 rather than 8..40 because it carries `(2n+1)^3`
nodes; the dof counts are matched across elements.

| material | element | exact | st. f64 | st. f32 | st. f16 | assembly | speed-up (f16) | break-even f64/f32/f16 |
|---|---|---|---|---|---|---|---|---|
| neohookean Ogden  | TET4  |  5.12 | 22.79 | 26.15 | 30.26 |   4.22 | 5.91x | 1.6 / 1.5 / 1.5 |
| neohookean Ogden  | TET10 | 12.01 | 42.66 | 44.47 | 46.15 |   3.89 | 3.84x | 4.3 / 4.2 / 4.2 |
| neohookean Ogden  | HEX8  |  7.10 | 17.22 | 17.40 | 17.80 |   3.95 | 2.51x | 3.1 / 3.0 / 3.0 |
| linear elasticity | TET4  | 17.41 | 22.47 | 25.39 | 30.26 |  17.08 | 1.74x | 4.5 / 3.2 / 2.4 |
| linear elasticity | TET10 | 26.31 | 42.21 | 44.15 | 45.79 | 146.33 | 1.74x | 0.5 / 0.4 / 0.4 |
| linear elasticity | HEX8  | 23.94 | 16.95 | 17.15 | 17.59 | 110.22 | 0.73x | never |

Break-even is how many applies must share one tangent before the split has repaid
its assembly: `(1/a) / (1/e - 1/s)`.  Five of six win, by 1.7x to 5.9x, at between
0.4 and 4.3 applies per tangent -- well inside one Krylov solve.

**HEX8 linear elasticity never pays.**  Its break-even is negative at every
precision: the stored apply is slower than the exact one, so no amount of reuse
repays the assembly.  Twenty-four degrees of freedom per element against a
45-number tangent, and an exact apply that is already cheap and sum-factorizable,
make the store pure added bandwidth.  That is the boundary of the technique.

**Precision buys less than the byte count suggests.**  f64 to f16 is a 3.8x
reduction in the store for 1.03x to 1.33x throughput, and on HEX8 almost nothing.
These kernels are not purely bandwidth-bound on the tangent: the gather of the
increment, the scatter of the output and the contraction all cost.  f32 is the
sensible default -- half the bytes of f64, most of the speed of f16, and seven
digits of accuracy instead of three.

The projection is exact wherever the tangent does not vary over the element: both
materials on TET4, and linear elasticity everywhere, including on TET10 and HEX8
where the *basis* varies but the tangent does not.

There is no fused variant.  An earlier revision emitted one, which rebuilt `Sbar`
on every apply; it was scaffolding for the comparison and is gone.  The stored
apply gates correctness now, and did so identically while both existed.

## Accuracy of the store

| store | bytes/element | rel. difference from exact |
|---|---|---|
| f64                          | 360 | 3.8e-15 |
| f32 (`metric_tensor_t`)      | 180 | 1.7e-07 |
| f16 + scale (`compressed_t`) |  94 | 1.4e-03 |

For a single hyperelastic material, whose tangent is symmetric and takes 45
numbers.  The f64 store reproduces the exact apply to round-off, which is the
gate: on an affine simplex the projection loses nothing, so any difference would
be a bug in the split rather than an approximation.

The scale is applied to the outputs rather than to the stored components.  The
action is linear in `Sbar`, so it is the same number, reached with
`dim * n_nodes` multiplies instead of 45 and without a decompressed copy of the
tangent in registers.

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


# A material with more than one unit

`bench_mixed.cpp`.  Mooney-Rivlin elasticity plus Kelvin-Voigt viscosity is an
energy unit and a residual unit in one material.  The operator's action is the sum
of the two, so the split carries both -- two tangents assembled, two applies
summed -- and is compared against the sum of the two exact kernels.

It broke two assumptions that every single-unit material had quietly satisfied.

**The tangent's state is not only the current state.**  The viscous flux reads
`grad(old(u))`.  The flux form was folding those nine symbols in with `eta_s` and
`eta_b`, which would have emitted per-element field data as scalar uniforms.
`SfemSoAFluxForm` now carries `previous_gradient` separately, taken from the
residual field records that already model it.

**The tangent is not always symmetric.**  An energy's flux is a gradient, so its
tangent is a Hessian and `A[ijkl] == A[klij]` holds by equality of mixed partials;
that is what folds 81 stored numbers into 45.  A residual's flux is not a
gradient, its tangent is a Jacobian, and the Kelvin-Voigt viscous tangent violates
the symmetry by 0.25.  Packing it into 45 averaged away its antisymmetric part and
returned a different operator: 5.3e-2 relative error on TET4, an element where the
projection is provably exact.  The plan now decides symmetry symbolically and
stores 45 or 81 accordingly, defaulting to *unsymmetric*, because storing 81 for a
symmetric tangent wastes memory while storing 45 for an unsymmetric one is
silently wrong.

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

## The two-unit material on a curved element

TET4 was the only element this material had been measured on, and it is the one
where the projection is provably exact -- so it established correctness and
throughput and said nothing about accuracy.  `run_mixed.sh TET10` says the rest.
Single-threaded, best of three.

| elements | ndof | exact | st. f64 | st. f32 | st. f16 | assembly | rel diff |
|---|---|---|---|---|---|---|---|
|   3072 |   14739 | 5.66 | 24.03 | 24.09 | 24.37 | 1.33 | 3.9e-03 |
|  24576 |  107811 | 5.11 | 19.63 | 20.53 | 21.35 | 1.18 | 1.8e-03 |
|  82944 |  352947 | 4.88 | 17.30 | 19.21 | 19.96 | 1.15 | 1.2e-03 |
| 196608 |  823875 | 4.72 | 15.78 | 16.84 | 17.81 | 1.12 | 8.7e-04 |
| 384000 | 1594323 | 4.77 | 16.93 | 18.12 | 19.21 | 1.12 | 7.0e-04 |

MDOF/s, white-noise increment (`MIXED_EXTRA_FLAGS=-DRANDOM_INCREMENT`).

**The deviation is first order, at `O(h^1.07)`** across a fivefold refinement --
pairwise rates 1.12, 1.00, 1.12, 0.97.  That is the same rate the single-unit
TET10 measurement found, so carrying two units, one of them an unsymmetric
Jacobian stored in 81 numbers, does not degrade the projection.

With the smooth increment the same runs report a flat 2.9e-02 instead.  That is
the metric artefact documented above, not a property of this material; it is
recorded here only so that a reader who runs the default and sees 2.9e-02 knows
which number to believe.

See "What the store's precision actually costs" below for the f32 and f16
columns, which need a severity sweep to read and not a single row.

HEX8 says the same thing on a different element:

| elements | ndof | exact | st. f64 | st. f32 | st. f16 | assembly | rel diff |
|---|---|---|---|---|---|---|---|
|   512 |   2187 | 3.92 | 20.43 | 19.52 | 19.98 | 1.57 | 1.3e-03 |
|  4096 |  14739 | 3.40 | 17.12 | 17.01 | 17.43 | 1.31 | 5.9e-04 |
| 13824 |  46875 | 3.19 | 15.51 | 15.52 | 16.39 | 1.21 | 3.9e-04 |
| 32768 | 107811 | 3.07 | 14.01 | 14.46 | 15.09 | 1.16 | 2.9e-04 |
| 64000 | 206763 | 2.90 | 13.25 | 13.68 | 14.79 | 1.13 | 2.3e-04 |

`O(h^1.08)`, pairwise 1.14, 1.02, 1.03, 1.04, and the same collapse of the
precision columns onto one another.  Its deviation is about a third of TET10's
at equal dof, and its break-even is 3.3 rather than 5.9.

**Break-even**: 5.9 applies per tangent on TET10, 3.3 on HEX8, against 2.7 on
TET4.  The exact TET10 apply is only 1.9x more expensive per dof than the
exact TET4 one, while assembling the tangent costs about the same in all three,
so the fixed cost is amortised over more applies.  The speed-up once past it is
3.5x on TET10 and 4.6x on HEX8, against 2.8x on TET4.

## What the store's precision actually costs

The columns above invite a wrong reading.  At the deformation these benchmarks
use, f32 and f16 agree with f64 to two digits, which looks like "the store's
precision is free on a curved element".  It is not free; it is hidden, and a
severity sweep says by how much and until when.

`run_store_precision.sh <element>` sweeps the deformation amplitude at the
finest mesh.  Amplitude zero is the control: the deformation gradient is
constant there, so the projection is exact on *any* element and the whole
remaining difference is the store's own.  Without that row a store error and a
projection error are the same number and cannot be told apart.

TET10, 1.59 Mdof:

| amplitude | f64 (projection) | f32 | f16 | f16 inflation |
|---|---|---|---|---|
| 0      | 2.4e-16 | 2.4e-08 | 3.1e-04 | control |
| 0.0025 | 8.6e-05 | 8.6e-05 | 2.0e-04 | +133% |
| 0.005  | 1.7e-04 | 1.7e-04 | 2.5e-04 |  +47% |
| 0.01   | 3.5e-04 | 3.5e-04 | 3.9e-04 |  +11% |
| 0.02   | 7.0e-04 | 7.0e-04 | 7.2e-04 |   +3% |
| 0.04   | 1.4e-03 | 1.4e-03 | 1.4e-03 |   +0% |
| 0.08   | 2.9e-03 | 2.9e-03 | 2.9e-03 |   +0% |

HEX8, 206763 dof:

| amplitude | f64 (projection) | f32 | f16 | f16 inflation |
|---|---|---|---|---|
| 0     | 2.4e-16 | 2.6e-08 | 1.1e-05 | control |
| 0.005 | 5.6e-05 | 5.6e-05 | 1.0e-04 | +79% |
| 0.01  | 1.1e-04 | 1.1e-04 | 1.4e-04 | +27% |
| 0.02  | 2.3e-04 | 2.3e-04 | 2.4e-04 |  +4% |
| 0.08  | 1.0e-03 | 1.0e-03 | 1.0e-03 |  +0% |

Three things follow, and only the third is the headline.

**The projection error is first order in the deformation.**  Each doubling of
the amplitude doubles the f64 column, exactly, over five doublings.  That is the
same first order the mesh refinement shows, and it is what the projection is.

**The two errors combine in quadrature.**  Fitting `total^2 = P^2 + Q^2` to the
deformed rows gives a store term `Q` of 1.2e-04 on TET10 and 6.0e-05 on HEX8,
and the model then reproduces every measured f16 entry to one digit.  So the
store does not add to the projection error, it is absorbed by it -- which is why
the inflation column falls away rather than staying constant.

**f16 is within ten per cent of a perfect store once the projection error
exceeds about `2.2 Q`** -- amplitude 0.007 on TET10, 0.010 on HEX8.  Below that
it dominates and the store is what you are measuring.  At the amplitude these
tables use, 0.02, f16 costs 3% on TET10 and 4% on HEX8: small, but not the
"nothing measurable" that reading one row suggests.

f32 needs no such argument.  Its store term is 2.5e-08, four orders below the
projection error at any deformation worth applying, and it tracks f64 to two
digits from the first non-zero amplitude.  **f32 is unconditionally free here;
f16 is conditionally free**, and the condition is a deformation large enough to
hide it.

Two consequences worth stating plainly.  Refining the mesh moves *towards* f16
mattering, not away: the projection error falls as `O(h)` while `Q` does not
move, so the finest TET10 mesh here is already at +3% and four more refinements
would put the store back in charge.  And on TET4 the projection is exact, so
`P = 0` and f16's 1.1e-04 is the entire error -- there is nothing for it to hide
behind, which is what the Grace measurement was seeing when it concluded fp16
was not worth carrying.  That conclusion was right for the element it was drawn
on and does not generalise; this one does not generalise downwards either.

The undeformed control is not a store floor to quote on its own: it is 3.1e-04
on TET10 and 1.1e-05 on HEX8, a factor of 28 apart, because the quantisation
error depends on the dynamic range of the tangent being stored and the
undeformed tangent is a special, unusually uniform case.  `Q` fitted from the
deformed rows is the number that predicts.

## Threads

206763 dof, on 8 performance plus 2 efficiency cores, twenty repetitions inside
each timing window.  Store is 126 numbers per element (45 elastic + 81 viscous),
so 1008 / 504 / 260 bytes at f64 / f32 / f16.

| threads | exact | st. f64 | st. f32 | st. f16 | assembly | f16 speed-up |
|---|---|---|---|---|---|---|
|  1 |  2.68 |  5.32 |  6.37 |  7.61 |  1.45 | 2.84x |
|  2 |  5.40 | 10.50 | 12.63 | 15.19 |  2.94 | 2.81x |
|  4 | 10.70 | 19.63 | 23.91 | 28.95 |  5.59 | 2.71x |
|  8 | 18.75 | 34.00 | 42.35 | 55.15 | 10.61 | 2.94x |
| 10 | 14.84 | 27.09 | 31.76 | 38.72 |  7.79 | 2.61x |

Correctness 4.44e-15 against the summed exact kernels, at every thread count.
Both paths scale about sevenfold on eight threads -- 7.0x exact, 7.2x stored --
and the advantage holds near 2.8x throughout.  Break-even stays close to three.

**Eight threads is the operating point, not ten.**  Every column falls by about a
fifth when the two efficiency cores join: `schedule(static)` hands them chunks the
size the performance cores get, and the loop waits for them.  A schedule aware of
heterogeneous cores would recover it; nothing here does yet.

This is the strongest case for the split, and structurally so: the exact side
evaluates two expensive tangents per apply -- a Mooney-Rivlin Hessian and a
Kelvin-Voigt Jacobian that reconstructs `F`, `F^-1`, the velocity gradient and its
symmetric part from two states -- and the split evaluates neither.

## Grace

The same benchmark on one socket of a CSCS Alps GH200 node (`nid006544`): 72
Neoverse-V2 cores, one thread per core, 9 NUMA domains per socket, GCC 13.3 from
`prgenv-gnu/24.11`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`.  206763 dof,
twenty repetitions inside each timing window.

| threads | exact | st. f64 | st. f32 | st. f16 | assembly | f16 speed-up |
|---|---|---|---|---|---|---|
|  1 |  1.33 |   2.91 |   3.10 |   2.85 |  0.64 | 2.14x |
|  2 |  2.67 |   5.90 |   6.17 |   5.84 |  1.27 | 2.19x |
|  4 |  5.33 |  11.54 |  12.12 |  12.26 |  2.53 | 2.30x |
|  8 | 10.68 |  22.84 |  24.33 |  25.30 |  5.04 | 2.37x |
| 16 | 21.27 |  44.49 |  47.97 |  50.01 | 10.13 | 2.35x |
| 32 | 41.65 |  85.62 |  93.52 |  97.67 | 20.25 | 2.35x |
| 64 | 82.13 | 165.29 | 185.34 | 191.62 | 39.30 | 2.33x |
| 72 | 92.28 | 180.18 | 205.81 | 209.60 | 44.35 | 2.27x |

Correctness 1.78e-14 at every thread count -- round-off, differing from the
laptop's 4.44e-15 only in floating-point ordering and FMA contraction.

**Scaling is essentially linear all the way to 72 cores**: 69.4x for the exact
apply, 73.5x for the stored one, against a perfect 72x.  There is no ceiling in
the sweep and no NUMA cliff at 64 or 72 threads.

That retracts something the laptop measurements suggested.  On the laptop every
column fell by a fifth past eight threads, and the atomic scatter looked like the
scalability limit.  It is not: the laptop has eight performance cores and two
efficiency cores, `schedule(static)` gives them equal chunks, and the loop waits
for the slow ones.  On homogeneous cores the scatter costs a constant factor,
not scalability.  A conclusion about a kernel drawn from a heterogeneous laptop
needed a homogeneous machine to check, and did not survive it.

**fp16 is not worth carrying on this machine** -- on throughput.  That is a
separate question from what it costs in accuracy, which "What the store's
precision actually costs" above answers with a severity sweep; the two happen to
agree that f32 is the default to reach for, by different arguments.  At one thread it is *slower*
than both f32 and f64 (2.85 against 3.10 and 2.91), and at 72 threads it is
within two per cent of f32 (209.60 against 205.81) while costing four decimal
digits.  `half_t` is `_Float16` here and `__fp16` on the laptop, and the
conversion appears to cost more than the bandwidth it saves.  f32 is the default
on both machines, and on Grace it is the only sensible choice.

Per core Grace is about half the laptop -- 1.33 against 2.68 MDOF/s exact,
single-threaded, Neoverse-V2 against an Apple performance core -- and reaches
five times the aggregate because there are 72 of them rather than eight.

Reproducing: the generated tree has to keep its directory structure, because the
emitted headers include across it (`../../kernel_math.hpp`), and `half_t` must be
taken from `sfem_config.h` rather than declared locally, since it is `__fp16` on
one target and `_Float16` on the other.

## Store precision and throughput, all three elements on Grace

The accuracy question above is answered by a severity sweep.  The throughput
question is separate and answered here: what the three store widths *cost to
apply*, on one Grace GH200 socket (Neoverse-V2, GCC 13.3 from
`prgenv-gnu/24.11`, `OMP_PLACES=cores`), largest mesh only, MDOF/s.

| element | dof | threads | exact | st. f64 | st. f32 | st. f16 | assembly |
|---|---|---|---|---|---|---|---|
| TET4  |  206763 |  1 |   1.38 |   3.27 |   3.36 |   3.06 |  0.67 |
| TET4  |  206763 |  8 |  10.98 |  23.95 |  25.23 |  26.31 |  5.32 |
| TET4  |  206763 | 32 |  42.77 |  91.52 |  98.47 | 100.98 | 21.17 |
| TET4  |  206763 | 72 |  93.27 | 185.94 | 211.52 | 216.02 | 45.32 |
| HEX8  |  206763 |  1 |   1.47 |   5.52 |   5.42 |   5.01 |  0.45 |
| HEX8  |  206763 |  8 |  11.59 |  42.67 |  41.96 |  38.53 |  3.58 |
| HEX8  |  206763 | 32 |  44.55 | 148.69 | 147.24 | 137.87 | 14.21 |
| HEX8  |  206763 | 72 |  96.33 | 313.71 | 308.44 | 290.71 | 31.47 |
| TET10 | 1594323 |  1 |   2.58 |   6.59 |   6.64 |   6.92 |  0.24 |
| TET10 | 1594323 |  8 |  20.62 |  51.57 |  51.59 |  54.24 |  2.42 |
| TET10 | 1594323 | 32 |  80.48 | 195.77 | 190.25 | 198.44 | 11.47 |
| TET10 | 1594323 | 72 | 174.04 | 401.08 | 394.74 | 411.31 | 26.50 |

At 72 threads, f16 against f64: **+16.2% on TET4, -7.3% on HEX8, +2.6% on
TET10**.  f32 against f64: +13.8%, -1.7%, -1.6%.

**Narrowing the store is worth it only where the store is what the apply is
moving.**  The store is 126 numbers per element on every element, so what
differs is how many elements a dof is shared between: 234 stored numbers per dof
on TET4 against 39 on HEX8 and 30 on TET10.  TET4 moves six times the store
traffic per dof, its apply is bandwidth-bound, and both narrower widths pay
there.

That explains TET4 and nothing else.  HEX8 and TET10 are within 30% of each
other in store traffic per dof and f16 goes opposite ways on them, so traffic
alone does not decide it -- the `_Float16` conversion cost and the apply's own
arithmetic intensity both enter, and this measurement does not separate them.
What can be said without separating them is the practical rule: **f16 is worth
carrying on the low-order simplex and not elsewhere**, and f32 is never worse
than f64 by more than 2% anywhere.

Break-even rises steeply with element order, because assembling the tangent gets
dearer while the apply does not: 3.5 to 4.4 applies per tangent on TET4, 4.4 to
4.6 on HEX8, and 11.6 to 17.9 on TET10.

Two methodology notes.  `OMP_PROC_BIND=true` and `close` agree within noise at
every thread count on both TET4 and HEX8, which settles a discrepancy between
this file's scripts.  And an earlier attempt swept all five mesh sizes at every
thread count and produced erratic rows -- 44.9, 55.2, 148.1, 66.6, 158.2 down one
column -- because the four smaller meshes do not fill 72 cores.  Measuring the
largest mesh alone is reproducible to three digits, and `bench_mixed` now takes
an optional mesh size for exactly that.

## Neohookean Ogden on Grace, and what it says about f16

The same measurement for the single-unit material, on one Grace GH200 socket,
`OMP_PLACES=cores`, `OMP_PROC_BIND=true`, twenty repetitions, largest mesh only.
All three elements are at the same 206763 dof, so the rows compare directly.

| element | threads | exact | st. f64 | st. f32 | st. f16 | assembly |
|---|---|---|---|---|---|---|
| TET4  |  1 |   3.12 |   7.05 |   6.92 |   7.01 |   2.71 |
| TET4  |  8 |  24.96 |  53.28 |  56.51 |  56.11 |  21.68 |
| TET4  | 32 |  95.12 | 208.63 | 205.73 | 204.68 |  86.33 |
| TET4  | 72 | 202.03 | 431.92 | 421.19 | 413.35 | 192.98 |
| HEX8  |  1 |   3.50 |  10.63 |  10.48 |   9.82 |   1.65 |
| HEX8  |  8 |  27.54 |  80.71 |  78.56 |  74.64 |  13.19 |
| HEX8  | 32 | 104.19 | 271.35 | 267.54 | 255.94 |  52.50 |
| HEX8  | 72 | 220.74 | 530.63 | 518.11 | 517.26 | 116.59 |
| TET10 |  1 |   6.28 |  13.65 |  13.49 |  14.25 |   1.58 |
| TET10 |  8 |  48.86 | 100.33 | 100.05 | 104.56 |  12.61 |
| TET10 | 32 | 176.77 | 334.78 | 334.19 | 345.93 |  50.40 |
| TET10 | 72 | 360.22 | 641.08 | 638.55 | 652.87 | 110.20 |

Split against exact at 72 threads: 2.1x, 2.4x, 1.8x.  Break-even is 2.0, 3.2 and
7.4 applies per tangent -- lower than the two-unit material's 3.5, 4.4 and 11.6,
because there is one tangent to assemble rather than two.

**For neohookean, f16 is not a win on any element**: -4.3% on TET4, -2.5% on
HEX8, +1.8% on TET10, against f64 at 72 threads.  f32 is within 2.5% of f64
everywhere.

Put beside the two-unit material, that kills the bandwidth explanation offered
above.  Stored numbers per dof against what f16 buys, at 72 threads:

| | store#/dof | f16 vs f64 |
|---|---|---|
| neohookean TET4  |  83.6 |  -4.3% |
| neohookean HEX8  |  13.9 |  -2.5% |
| neohookean TET10 |  10.4 |  +1.8% |
| mooney TET4      | 234.0 | +16.2% |
| mooney HEX8      |  39.0 |  -7.3% |
| mooney TET10     |  30.3 |  +2.6% |

There is no monotone relation.  TET10 has the least store traffic of all six and
is the only element where f16 helps in both materials; HEX8 is negative in both;
TET4 swings from -4.3% to +16.2% depending on whether the store is 45 numbers or
126.  **The element decides the sign and the store size decides the size** --
mooney's TET4 is the one case bandwidth-bound enough for f16 to pay, at 2.8x the
store traffic per dof of any other row.

So the rule to carry is narrow and empirical: **f32 by default everywhere**, and
f16 only where it has been measured to pay, which so far is one case out of six.
This retires the earlier reading, drawn from mooney alone, that f16 is worth
carrying on the low-order simplex -- neohookean's TET4 is a low-order simplex and
f16 loses there.

Accuracy, for the record: the f64 store reproduces the exact apply to 1.4e-14 on
TET4, which is the gate, and the projection error is 3.6e-05 on HEX8.  The 2.7e-02
on TET10 is the smooth-increment metric artefact documented above, not the
approximation.

## What perf says is left, and it is not the store

`perf` on one Grace core, neohookean HEX8, and the answer is not where the store
precision work was looking.

    IPC                          3.76  (1 thread)   3.82  (72 threads)
    stalled cycles per insn      0.08
    cache-misses                 0.48% of references
    branch-misses                0.16% of branches

Nothing is stalling.  At 3.8 instructions per cycle on a machine about six wide,
with half a per cent of references missing cache, these kernels are compute-bound
and already issuing near the width of the core.  **There is no headroom in moving
data better; the only way to go faster is to execute fewer instructions.**

That also settles why narrowing the store bought so little.  It was never the
bottleneck.

### The inexact-apply kernels are the only ones in the generator that are not vectorised

Counting the instructions `perf annotate` attributes to each symbol:

| symbol | instructions | ldr | str | FP arith | `fmla` | `ld1`/`st1` |
|---|---|---|---|---|---|---|
| `inexact_apply_stored<double, float>`  |  3182 |   989 |  694 |  967 | 0 | 0 |
| `inexact_apply_tangent<double, float, double>` | 25078 | 12219 | 5542 | 6659 | 0 | 0 |

Scalar loads and stores are **53%** of the stored apply and **71%** of the
tangent assembly, and neither contains a single vector FMA or vector
load/store.  The same binary holds 2967 `fmla` and 157 `ld1` -- all of them in
the *exact* apply, which is lane-blocked and vectorised like every other kernel
this generator emits.

The reason is in the emitted source.  The inexact-apply kernels loop

    for (ptrdiff_t element = 0; element < nelements; ++element)

with no `VS` template parameter and no `#pragma omp simd` anywhere in the file,
against three simd pragmas in the tensor-product local header next to it.  This
kernel family was written one element at a time and never picked up the lane
blocking the rest of the framework uses.

### What that is worth, and what it changes about the store

Two things follow, and the second is why this matters more than it looks.

Half the stored apply's instructions are scalar memory operations that a lane
loop turns into one vector operation per two lanes at f64 and four at f32.  With
IPC already at the machine's limit and no stalls to recover, instruction count
is throughput, so the headroom is close to the reduction factor.

And **the store-precision question cannot be answered properly until this is
done.**  Scalar code loads one number per `ldr` whatever its width, so narrowing
f64 to f32 changes bytes moved and not instructions issued -- which is exactly
what the six measured rows show, a couple of per cent either way.  Vectorised, an
f32 vector load carries four lanes where f64 carries two, so the width goes
straight into the instruction count.  The conclusion "f32 by default, f16 only
where measured" is correct for the kernels as they exist, and should be
re-measured once they are lane-blocked, because the mechanism that would make
narrow stores pay is currently absent.

### Lane-blocking them: two failed attempts, then a win

Blocking these kernels over lanes like the rest of the framework does pay, but
the first two attempts said it did not, and both were measurement faults rather
than results.

**`run_split.sh` was not passing `-fopenmp`.** Every `#pragma omp simd` in the
generated kernels was therefore inert, and every comparison this file made of a
blocked kernel against a scalar one measured the blocking's overhead against
none of its benefit. Fixed; without it these benchmarks measure unvectorised
code whatever the kernels say.

**The one configuration that reached Grace read the increment inside the
arithmetic loop.** Those loads go through the mesh connectivity, there is no
gather instruction to put them in, and the compiler abandons the loop. That
shape was already diagnosed as unvectorisable here -- and then benchmarked
anyway, while the staged variant was only ever run on the laptop, where the
missing `-fopenmp` made it look worse too.

Staged and with OpenMP on, it wins.  The values reached through the connectivity
are gathered into lane-major scratch in one pass, so by the time the arithmetic
loop runs everything it touches is contiguous in the lane.  That is how the
exact apply has always got its vector code.  HEX8 at 206763 dof, stored apply,
one Grace socket:

| threads | scalar f64 | blocked f64 | | scalar f32 | blocked f32 |
|---|---|---|---|---|---|
|  1 |  10.86 |  12.59 | +16% |  10.69 |  10.79 | +1% |
|  8 |  82.30 |  94.34 | +15% |  80.20 |  81.43 | +2% |
| 32 | 277.97 | 312.69 | +13% | 273.16 | 278.12 | +2% |
| 72 | 548.29 | 617.29 | +13% | 529.77 | 553.77 | +5% |

The compressed kernel is not blocked and does not move at any thread count,
which is the control.  Answers are identical at 3.63e-05 and the binary gains
1193 vector instructions.  On the laptop, where clang vectorises this more
aggressively, both widths gain 34% to 42%.

Three things worth keeping.  The tangent is not staged: a component-major store,
which the ABI has always allowed, makes it contiguous across the lanes already,
and staging it costs a store and a reload per value for nothing -- an early
version did that and lost a third of the throughput.  The gathers go in **one**
loop with many statements, not one loop each; `tests/test_kernels_are_lean.py`
caught thirty-two of the latter.  And the f32 path, which is what the Op
actually uses, gains least on Grace -- 1 to 5% against f64's 13 to 16% -- so the
headline number is not the one that ships.

### The store's layout, and what it was costing

The blocked kernels addressed the store through two runtime strides, so that
"both layouts are expressible by the caller without a second kernel".  Nothing
ever used that: the only caller, the generated Op, passes `1, nelements` -- which
is the SoA layout -- and it is not free.  With a runtime element stride, the
address inside the lane loop is `btangent<k>[lane * tangent_element_stride]`, so
every one of the 45 accesses is an unknown-stride scatter as far as the
vectoriser is concerned.  One stride, and the lane indexes the component
directly.

The comparison below is between two trees whose *only* difference is that: both
have the blocked tangent, both have the staged gathers.  Grace, 206763 dof,
stored apply, MDOF/s.

| | | two strides | SoA | |
|---|---|---|---|---|
| HEX8 | f64, 72 threads | 625 | 652 | +4% |
| HEX8 | f32, 72 threads | 545 | 564 | +3% |
| TET4 | f64, 72 threads | 310 | 341 | +10% |
| TET4 | f32, 72 threads | 306 | 345 | **+13%** |
| TET4 | f32, 8 threads   | 39.1 | 43.7 | +12% |

f32 is what ships, TET4 gains most, and this is the kernel that runs every
iteration rather than once per Newton step.  Answers are unchanged to the digit
-- 3.6e-05 on HEX8, 1.7e-07 on TET4 -- because it is the same arithmetic reading
the same numbers from a different address.

### The tangent blocking is a clang result, not a Grace result

Blocking the assembly kernel the same way is worth +78% with clang and nothing
with gcc, and the vectorisation reports say why.  Clang vectorises the tangent's
1676-line CSE body; gcc vectorises only the gather loop above it and leaves the
body scalar, in every variant tried -- two strides, SoA, and SoA with an explicit
`simdlen(VS)` to rule out its cost model declining a loop it could have taken.

Assembly throughput, 206763 dof, MDOF/s:

| | scalar tangent | blocked tangent |
|---|---|---|
| HEX8, laptop, 1 thread | 3.88 | **6.93** |
| TET4, laptop, 1 thread | 4.18 | **7.32** |
| HEX8, Grace, 72 threads | 118.2 | 117.2 |
| TET4, Grace, 72 threads | 194.2 | 195.0 |

Repeatable to better than a per cent in both directions, so the Grace figures are
a wash rather than noise hiding a win: about -1% on HEX8 and +0.5% on TET4.  The
kernel ships blocked anyway -- the three kernels then have one shape, the clang
gain is large, and the assembly runs once against many applies -- but the
headline that belongs to this change is the store layout above it, not this.

`simdlen(VS)` is worth a further 2 to 3% on the HEX8 f32 apply and about -1.5% on
the assembly.  It is a property of the target's `vectorize_pragma`, so it would
change every generated kernel in the framework, and nothing here has measured
that; it stays an open question rather than a change.

### Default

`metric_tensor_t` is `float`, and the generated Op has always stored the tangent
in it -- `SharedBuffer<metric_tensor_t> inexact_tangent` -- and called
`inexact_apply_stored_*`.  f32 is the default and nothing needed changing; the
compressed f16 entry points are generated and no Op calls them.

## Roofline

The throughput tables above say how fast these kernels ran.  The roofline says
whether that was fast, by placing each against the two ceilings the machine
imposes at the kernel's own arithmetic intensity.  Both coordinates are read out
of the generated source by `codegen.framework.tools.roofline` -- the FLOPs from
the printed arithmetic, the bytes from the memory references and the widths the
`extern "C"` wrapper instantiates them with -- so the model follows the
generator rather than a table someone has to remember to update.

A mesh kernel does not have *one* intensity: how much of a gather is compulsory
is a property of the connectivity, not the kernel.  So two bounds are computed.
**Streamed** charges every reference its own bytes: no reuse at all.
**Compulsory** charges a value reached through the connectivity once per *node*
rather than once per element, scaled by `nnodes / nelements`.  The truth is
between them.

Neohookean Ogden, 206763 dof, Grace at 72 threads, morton3 ordering.  Peak 3571 GFLOP/s, or 1786
without SIMD; 500 GB/s; ridge at 7.1 FLOP/byte.  (Quoted peaks -- the tool
prints their provenance with every report.)

| kernel | FLOP/el | B/el streamed | B/el compulsory | I streamed | I compulsory | measured | of no-SIMD |
|---|---|---|---|---|---|---|---|
| HEX8 tangent    | 11171 | 444 | 278 | 25.2 | 40.2 | 407 GFLOP/s | 23% |
| HEX8 stored f32 |  1827 | 788 | 290 |  2.3 |  6.3 | 282 GFLOP/s | 16% |
| HEX8 compressed f16 | 1851 | 702 | 204 | 2.6 | 9.1 | 278 GFLOP/s | 16% |
| TET4 tangent    |  1314 | 332 | 240 |  4.0 |  5.5 | 475 GFLOP/s | 27% |
| TET4 stored f32 |   243 | 484 | 209 |  0.5 |  1.2 | 133 GFLOP/s | 7% |
| TET4 compressed f16 | 255 | 398 | 123 | 0.6 |  2.1 | 152 GFLOP/s | 9% |

Three things fall out of it that the throughput tables could only assert.  All
of it is a *ceiling* argument: none of these kernels is on either line, so read
the roofline as what they are under rather than what they are at.

**The dashed ceiling is the one that matters.**  Every kernel here sits between
7% and 27% of the *no-SIMD* ceiling -- the same issue rate with one lane per
operation -- and nowhere near the vector peak above it.  That is the same
finding as the vectorisation reports, arrived at from measured throughput rather
than from compiler diagnostics: on gcc these kernels are not vector code, so the
ceiling they are actually working against is half the headline one.  It is also
why lane-blocking the tangent bought nothing on Grace and +78% on clang.

**The precision inversion is a roofline result, not a mystery.**  Narrowing the
store raises a kernel's intensity, which only buys throughput if the kernel is
near the bandwidth ceiling.  TET4's stored apply is the nearest thing here to
one: at I = 0.5 the streamed ceiling is 250 GFLOP/s and it achieves 133, 53% of
it; narrowing to f16 moves it to I = 0.64, a ceiling of 320, and it achieves 152
-- 48% of the new one.  The model predicts a 1.28x gain and the measurement is
1.14x, so the direction is right and the size is not, which is about where a
two-bound model of a gather deserves to be believed.  HEX8's stored apply is not
near that ceiling at all: at I = 2.3 the ceiling is 1160 and it achieves 282,
24% of it.  There is no bandwidth being waited on, so the wider intensity buys
nothing and the conversion cost shows through instead -- f16 is level with f32
there, 278 against 282, where on TET4 it is 14% ahead.  One number, `I`, orders
the two elements correctly even though it does not predict either precisely.

**The packed layout crosses the streamed bound, which is the check.**  A kernel cannot
beat its own streamed ceiling -- that bound assumes every reference pays -- unless it
is getting reuse.  TET4's stored apply sits at 53% of its streamed ceiling on the
standard layout and at **134%** of it on the packed one, moving from 23% to 58% of
the compulsory ceiling.  That is the packed gather doing exactly what it claims:
fetching each node once per pack instead of once per element that reaches it.  HEX8,
which was never near the bandwidth ceiling, moves 24% to 41% of streamed and 16% to
27% of the no-SIMD ceiling -- a real gain, but there it comes from deleting the
atomics rather than from bandwidth.  The two elements improve for different reasons
and the model says which is which.

**The tangent is the intense kernel, and it is the one that is compute-bound.**
HEX8's assembly does 11171 FLOPs against 444 bytes; it is three and a half times
past the ridge and the only kernel in the set for which the memory ceiling is
irrelevant.  That agrees with the `perf` reading above -- 3.8 IPC, half a per
cent of references missing cache -- and it is why the store's precision was
never going to move the assembly.

Regenerate, with the plot:

    spikes/inexact_apply_compare/run_roofline.sh neohookean_ogden HEX8 grace
    spikes/inexact_apply_compare/run_roofline.sh neohookean_ogden TET4 m1max

It reuses the tree `run_split.sh` generates, so whichever is run first pays for
the generation.  The measured dots come from `measured_<machine>.json` -- one
file per machine, because a dof rate from a laptop and one from a Grace socket
are not comparable and a single file would invite mixing them.  A kernel with no
measurement is still modelled, it just has no dot.  The plot is written beside
the log rather than committed -- it is a generated artifact, and the numbers it
draws are the table above.  `--bind
tangent_t=double` models the f64 store, which is a benchmark-only instantiation:
the Op publishes f32.

## Mesh ordering

These benchmarks built their mesh lexicographically and nothing reordered it,
which is a gap against the rest of the repository: SFEM has space-filling-curve
ordering in `external/smesh/src/mesh/ordering/`, `smesh::SFC::create_from_env()`
defaults to `morton3`, and SFEM's own benchmark drivers reorder with it before
they measure (`drivers/bench/bench_hyperelasticity.exe.cpp:240`).  These did not,
so they were measuring a mesh the library does not run.

`element_mesh.inc` now takes `-DMESH_ORDER` and **defaults to `morton3`**,
reaching smesh's own encoders and applying SFEM's own algorithm: sort the
elements along the curve through their barycentres, then renumber the nodes in
the order the reordered elements first touch them.

Grace, 206763 dof, 72 threads, MDOF/s:

| element | ordering | exact | stored f64 | stored f32 | compressed f16 | assembly |
|---|---|---|---|---|---|---|
| TET4 | morton3  | 185.1 | 291.0 | 293.9 | 321.1 | 194.7 |
| TET4 | hilbert3 | 188.6 | 293.0 | 293.8 | 330.8 | 194.3 |
| TET4 | lex      | 205.6 | 337.6 | 342.2 | 407.6 | 195.2 |
| HEX8 | morton3  | 215.6 | 565.8 | 498.8 | 484.7 | 117.6 |
| HEX8 | hilbert3 | 217.0 | 573.4 | 509.2 | 460.8 | 117.2 |
| HEX8 | lex      | 222.2 | 648.4 | 430.9 | 508.2 | 117.5 |

**The two curves are indistinguishable from each other, and lexicographic beats
both on TET4.**  That is a statement about these meshes rather than about
space-filling curves: they are a structured cube grid, and lexicographic
numbering already *is* a good space-filling order for one.  On TET4 it is 16%
ahead of Morton, which is the cost of reordering something that did not need
reordering -- a curve through barycentres does not respect the six-tetrahedra
decomposition of a cube, and lexicographic does.

So the ordering is not, on these meshes, a lever: switching to what the library
runs moves the stored apply by at most 16% and the assembly not at all.  The
figures elsewhere in this file were taken under `lex` and the table above is the
correction to apply to them -- the f64 and f16 columns move down on TET4, the
HEX8 f32 column moves up, everything about the tangent kernel is unchanged.

What this cannot say is what an *unstructured* mesh does, which is the case SFC
ordering exists for and the one where these numbers might not transfer.  Every
mesh this spike builds is a structured grid, so every one of them is
well-ordered before anything reorders it.  Answering that needs a real
unstructured mesh, not a synthetic one.

`random3` is also accepted, and is `rand()` as the sort key.  It exists to
confirm that these measurements are sensitive to the ordering at all -- they are,
by about a factor of four -- and for nothing else: no tool produces such a mesh,
so no speed-up should be quoted against it.

Sweep it with:

    for o in morton3 hilbert3 lex; do
      MESH_ORDER=$o spikes/inexact_apply_compare/run_split.sh neohookean_ogden HEX8
    done

**A caveat on the f32 column.**  HEX8 stored f32 at 72 threads has come back
between 431 and 567 across runs on different nodes, while f64 on the same runs
stayed within 648-653.  The ordering comparison above is within one job on one
node, so it is internally consistent, but no single f32 figure at 72 threads in
this file should be trusted to better than 25% without repeats.

## The packed mesh layout, and why it dwarfs everything else here

The stored apply's cost was never the arithmetic.  The roofline puts 384 of its 788
bytes per element on HEX8 in the scatter alone, and that scatter is `dim * n_nodes`
separate `#pragma omp atomic update` into global arrays.  The packed mesh layout
exists to remove exactly that: it partitions elements into packs, renumbers nodes so
each pack owns a contiguous range, gathers each node **once per pack** into
thread-private scratch, and accumulates into that scratch with **no atomic at all**.
Only the pack boundary reaches global memory, and the two-pass form sends even that
through a buffer which a second, disjoint-destination pass reduces.

The generator does not emit a packed inexact kernel.  This is the reference
implementation, derived from the emitted standard one by
`make_packed_reference.py` -- a script rather than a pasted file, because the 900
lines of arithmetic it carries across are precisely the part that does *not* change,
and a transformation says that where a copy does not.

Grace, 206763 dof, 72 threads, morton3, stored f32 apply, MDOF/s:

| element | standard | packed | pack size | |
|---|---|---|---|---|
| TET4 | 295 | **744** | 256 | **+152%** |
| HEX8 | 498 | **848** | 128 | **+70%** |

The exact apply gains too, from the same layout and the kernel the generator already
publishes: TET4 187 to 323, HEX8 217 to 260.

Answers are identical -- 1.7e-07 on TET4 and 3.6e-05 on HEX8, the same figures the
standard store produces, because it is the same arithmetic reading the same store.
The tangent needs no transformation at all: packs are **contiguous element ranges**,
so `tangent + evb + k * tangent_component_stride` addresses what it always did, and
the store assembled on the standard mesh is valid for the packed one element for
element.

**Put beside everything else this file measures, the ordering is now clear.**  The SoA
store was worth 13%, the store's precision 3%, lane-blocking the assembly nothing at
all on Grace, and the choice between lexicographic and Morton ordering nothing
either.  The layout is worth 70 to 152%.  Every other number in this file is a
second-order effect on top of a kernel that was scattering through atomics.

**Pack size barely matters** -- 128 to 1024 spans about 10%, with TET4 preferring 256
and HEX8 128 -- which is worth knowing mostly because it means the knob does not have
to be tuned per machine.

**The generated kernel holds the reference's number.**  The emitter emits this shape
now, and on Grace at 72 threads it comes within 1% of the hand-written original:

| element | standard f32 | packed, generated | packed, reference | pack |
|---|---|---|---|---|
| TET4 | 295 | **739** | 744 | 256 |
| HEX8 | 499 | **845** | 848 | 128 |

Errors unchanged at 1.7e-07 and 3.6e-05, and the reproducibility harness now drives
both and finds their digests *identical* -- 0.310504988649, the same number, not a
tolerance.

One thing about how that was measured, because it nearly went wrong.  An earlier job
reported these figures from binaries a previous job had left in the same directory
under the same names; the build had in fact failed, and the numbers were the
reference's.  Every benchmark job here now deletes its binaries first and reports the
compiler's exit status, and the header line prints whether the kernel under test is
even present in the tree it compiled.  A benchmark that cannot fail loudly will
eventually report someone else's number.

This is the reference, not the deliverable: it is measured so that the emitter port
has a number to be held to.  `spikes/inexact_apply_compare/packed_mesh.inc` builds the
layout from `python/codegen/framework/tools/packed_layout.inc`, shared with
`tools/reproducibility.py` rather than written twice -- and that sharing is what makes
the layout trustworthy, because that harness drives the *generated* packed kernels
through this same builder and checks their answers against the unpacked ones.  The
first thing measured here was the generated packed **exact** apply on this layout: it
agreed with the standard apply to 3.4e-16, which is how the layout was known to be
right before any new kernel existed.

    PACKED=1 spikes/inexact_apply_compare/run_split.sh neohookean_ogden TET4
    PACKED=1 PACK_SIZE=256 spikes/inexact_apply_compare/run_split.sh neohookean_ogden HEX8

## In the BDF2 driver, which is what it was for

Everything above measures kernels. This measures a solve: `drivers/mech/hyperelasticity_bdf2.exe.cpp`
with `SFEM_LINEAR_OP_TYPE=INEXACT`, which assembles the tangent once per Newton
iteration and applies it for every CG iteration inside that step. Cubes clamped at one
face and pulled at the other, Newton to 1e-8, CG to 1e-3 -- the driver's own inner
tolerance, because this is an inexact Newton method and solving the linear system past
the accuracy of the direction is wasted work. Eight threads.

| element | dof | | Newton | CG | ms/apply | wall |
|---|---|---|---|---|---|---|
| HEX8 | 107811 | matrix-free | 4 | 92 | 3.54 | 0.67 s |
| HEX8 | 107811 | inexact | 4 | 92 | **0.73** | **0.39 s** |
| HEX8 | 352947 | matrix-free | 4 | 142 | 11.54 | 2.30 s |
| HEX8 | 352947 | inexact | 4 | 142 | **2.33** | **0.92 s** |
| TET4 | 206115 | matrix-free | 4 | 171 | 6.29 | 1.52 s |
| TET4 | 206115 | inexact | 4 | 171 | **4.57** | **1.25 s** |
| TET4 | 684723 | matrix-free | 4 | 261 | 20.28 | 3.99 s |
| TET4 | 684723 | inexact | 4 | 261 | **12.31** | **3.99 s** |

**The iteration counts are identical.** Same number of Newton steps, same number of CG
iterations, same final residual to within rounding -- on both elements and at both
sizes. The stored tangent is a good enough Newton direction that the outer solve does
not notice it, so the whole of the kernel speed-up reaches the solve: 1.72x on HEX8 at
107811 dof and 2.50x at 352947, 1.22x and 1.52x on TET4.

The speed-up *grows* with the problem, which is the shape to expect: the exact apply
recomputes the tangent from the geometry and the state on every call and is
compute-bound, while the stored apply reads a precomputed one. HEX8 gains most because
its exact apply is the expensive one -- 1827 FLOPs of contraction against TET4's 243.

**The store's stride has to be padded, and the Op was not padding it.** The store is
component-major, so the stride between one component's run over the elements and the
next decides whether the 45 components land in the same cache sets. At 393216 elements
the unpadded stride is exactly 1.5 MiB and they do. `bench_split.cpp` had known this
since the kernel work and padded by 64 elements; the generated Op passed `nelements`
and lost a third of its apply to conflict misses -- 6.13 ms against 4.57 on TET4, 1.06x
against 1.38x over matrix-free. The Op now pads, which is why the numbers above are
what they are.

### On Grace, where the driver was built for the purpose

The laptop could not answer this: its run-to-run drift was 40% against an effect of a
few per cent. SFEM builds on Grace in about three minutes (`cmake` + `make
hyperelasticity_bdf2`, OpenMP on, which the pre-existing Grace build did not have), so
the driver was measured there on the same mesh and the same boundary conditions.
mesh48, 352947 dof, three interleaved passes, medians; pass-to-pass spread under 2%.

| threads | | solve | ms/apply |
|---|---|---|---|
| 8 | matrix-free | 2.665 s | 13.09 |
| 8 | inexact | 1.360 s | 4.43 |
| 8 | inexact + packed | **1.041 s** | **2.65** |
| 72 | matrix-free | 0.813 s | 1.545 |
| 72 | inexact | 0.562 s | 0.575 |
| 72 | inexact + packed | 0.567 s | **0.332** |

All three produce the same answer, and the two inexact rows produce it bit-identically.

**Measure the linear solve, not the "solve".** Profiling the 72-thread run showed
`Output::write_time_step` at 23% of it, `Mesh::read` 16%, `Mesh::write` 14% and
operator `initialize` 14% -- more than half the wall time is file I/O and setup for a
one-step run, and all of CG is 12%. A number that dilutes a 5x kernel into 1.2x is
measuring the file system. Timing `ConjugateGradient::apply` instead, which is the
part partial assembly changes:

| threads | | CG | ms/apply | assembly | solve |
|---|---|---|---|---|---|
| 8 | matrix-free | 1.995 s | 13.33 | -- | 2.41 s |
| 8 | inexact | 0.688 s | 4.51 | 0.080 s | 1.18 s |
| 8 | inexact + packed | **0.421 s** | **2.70** | 0.080 s | 1.05 s |
| 72 | matrix-free | 0.257 s | 1.565 | -- | 0.603 s |
| 72 | inexact | 0.111 s | 0.580 | 0.012 s | 0.492 s |
| 72 | inexact + packed | **0.075 s** | **0.340** | 0.012 s | 0.508 s |

**The linear solve is 4.7x faster at eight threads and 3.4x at seventy-two**, packed;
2.9x and 2.3x without packing. The assembly is 19% of the linear solve at eight
threads and 16% at seventy-two -- real, and not the story.

The solve-level figures are in the table for honesty, not as the result: at 72 threads
CG is 43% of the matrix-free solve and 15% of the packed one, so what the remaining
speed-up is bounded by is I/O. On a run that wrote less and stepped more, the CG
column is what would show.

**The pack size has to be tuned, and the default is wrong for a many-core machine.**
`PackedMesh` derives it from the *index type's* ceiling -- `n_packs = ceil(n_elements *
nodes_per_element / 65536)` -- which says how many nodes a pack may address and nothing
about how many threads there are. The 48-cube comes out as **14 packs**, and the
generated kernels share the pack loop with `#pragma omp for`, so 58 of 72 threads get
nothing:

| pack size | packs | 8 threads | 72 threads |
|---|---|---|---|
| 64 | 1728 | 2.666 | 0.340 |
| 128 | 864 | 2.643 | 0.337 |
| 256 | 432 | 2.627 | **0.328** |
| 512 | 216 | **2.614** | 0.329 |
| 1024 | 108 | 2.705 | 0.442 |
| default | 14 | 2.980 | 1.573 |
| *unpacked* | -- | *4.377* | *0.575* |

At the default the packed apply is **2.7x slower than not packing at all** on 72
threads, and 1.75x faster once tuned. The answer never changes. Report the pack size
and the pack count with any packed measurement; the crossover moves with the mesh.

### A larger example, where the linear solve is the run

The 48-cube above is small enough that setup and I/O dominate at 72 threads. A 96-cube
of HEX8 -- **2738019 dof**, 884736 elements -- over three time steps puts CG at 87% of
the matrix-free solve, which is where a solver comparison belongs. Grace, 72 threads,
pack size 256. Every row takes the same 747 applies and agrees on the answer.

| | solve | CG | ms/apply | assembly | store |
|---|---|---|---|---|---|
| matrix-free | 10.94 s | 9.53 s | 12.40 | -- | -- |
| BSR | 10.10 s | 3.84 s | 4.75 | 2.87 s (x10) | 1.77 GB |
| inexact + packed | **4.19 s** | **2.37 s** | **2.76** | **0.23 s** (x9) | **159 MB** |

**Partial assembly wins on all three axes against the assembled matrix**: the apply is
1.7x cheaper, the assembly 11x cheaper -- 26 ms against 287 ms each -- and the store is
an eleventh of the memory. Against matrix-free the linear solve is 4.0x and the whole
solve 2.6x.

The shape is worth reading. BSR's *apply* is respectable, within a factor of two of the
stored tangent; what sinks it is that rebuilding 1.77 GB of matrix once per Newton
iteration costs almost as much as all of its CG. The stored tangent is 45 numbers per
element against a matrix row's worth, so its assembly is cheap enough to disappear:
0.23 s against a 2.37 s linear solve.

**BSR had to be fixed before it could be compared.** `sfem::hessian_bsr(f, u, es)`
assembles once, at construction, and the driver built its linear operator before the
Newton loop -- so `Function::hessian_bsr` was called *once* against twelve Newton
iterations and BSR was running a modified Newton on the Jacobian of the initial state.
It still converged, to a slightly different answer, in **11206 linear iterations against
747**, and nothing in the output said why. The driver now rebuilds an assembled
operator once per linearization; matrix-free and inexact need no rebuild, because they
read the current state through `Function::update` and `inexact_update`. With that, all
three take the same 747 applies and BSR reproduces the matrix-free answer exactly.

### Is the BSR implementation any good?

Two different answers for its two halves, and the measurement separates them cleanly.
96-cube HEX8, 2738019 dof, Grace at 72 threads. The matrix is 24.6 M 3x3 blocks, so one
apply moves 1.92 GB of values, indices and vectors.

| | apply | GB/s | % of 500 GB/s | CG | assembly | solve | memory |
|---|---|---|---|---|---|---|---|
| BSR, f64 values | 4.758 ms | 403 | **81%** | 3.96 s | 2.75 s | 10.12 s | 1.77 GB |
| BSR, f32 values | 2.661 ms | 387 | **77%** | 2.29 s | 2.75 s | 9.27 s | 0.88 GB |
| inexact + packed | 2.757 ms | -- | -- | 2.37 s | **0.23 s** | **4.19 s** | 159 MB |

**The apply is state of the art and has no headroom left in the kernel.** At 81% of
this machine's peak memory bandwidth an SpMV is doing as well as an SpMV can; the only
way to make it faster is to move fewer bytes, and that works exactly as predicted --
halving the value width with `SFEM_ENABLE_MIXED_PRECISION=4` gives 1.79x and lands BSR
*level with the packed stored tangent*, 2.66 ms against 2.76. Same iteration count, and
the answer agrees to twelve digits. Nobody should be looking for a better BSR kernel
here.

**The assembly is not state of the art, and the reason is not what it looks like.** It
writes the same 1.77 GB in 287 ms -- 6.7 GB/s, 1.3% of peak -- and the generated scatter
does `8 x 8 x 3 x 3 = 576` `#pragma omp atomic update` per element, 509 million over the
mesh, which is an inviting thing to blame. It is not the cost. Read the kernel and the
arithmetic says so plainly:

    for trial_component in 0..3:
      for trial_shape in 0..8:            # 24 unit basis vectors
        zero bh, bout; bh[this one] = 1
        tensor_product_apply_block(1 element, ...)   # a full apply
        copy out one column of the 24x24

**The element matrix is built by applying the operator to 24 unit vectors.** 287 ms
divided by 24 is 11.97 ms; one matrix-free apply over this mesh is 12.40 ms. The
assembly *is* twenty-four applies, to within 3%, and everything else -- the column
searches, the half-billion atomics -- fits in what is left.

That is also the whole of why BSR loses here. With f32 values its linear solve is 2.29 s
against partial assembly's 2.37 -- a dead heat -- and it still finishes in 9.27 s
against 4.19, because it spends 2.75 s rebuilding the matrix that partial assembly
rebuilds in 0.23.

#### Why it does that, and what would fix it

The generator **already emits a direct element-matrix kernel** --
`<material>_d<dim>_<family>_direct_hessian_<family>_element_matrix` -- and laplace and
linear elasticity call it. Neohookean does not, and the gate is one line,
`energy_codegen.py:_sfem_soa_direct_hessian_matrix_assembly_available`:

    return not _form_uses_current(form, default=True)

A direct element matrix is emitted only for a form that does **not** read the current
state. A linear material's Hessian is state-independent and gets one; a hyperelastic
material's depends on `u`, so it falls back to the generic 24-apply construction. That
is a scope boundary, not a mathematical one: the element matrix of a hyperelastic
operator at a given state is perfectly computable directly.

**Three routes, in increasing order of how much the tree already knows.**

*Lift the gate.* Emit a direct element-matrix kernel for state-dependent forms as well.
The 24 applies each recompute the geometry, the deformation gradient and the material
tangent at every quadrature point; a direct kernel computes them once and forms all 576
entries from them. The saving is whatever fraction of an apply is tangent evaluation
rather than contraction, which for a hyperelastic material is most of it.

*Assemble from the stored tangent -- and this repository has already done it by hand.*
`operators/sshex8/sshex8_neohookean_ogden.cpp:823` builds its element matrix with
`hex8_neohookean_hessian_from_S_ikmn(&partial_assembly[e * S_IKMN_SIZE], W, element_matrix)`:
`Sbar` once per element, then a contraction with the reference tensor. That is exactly
the object the inexact path already computes, at **26 ms for this whole mesh** against
the 287 ms the 24 applies cost, and the contraction that turns it into a column is what
`plans/inexact_apply.py`'s `action_stages` already expresses symbolically. Assembling
all 24 columns from one `Sbar` is the same contraction with 24 right-hand sides and
shares every load of `Sbar` between them.

*Then, and only then, the atomics.* `plans/matrix_formats.py` carries
`PackedAssemblyPass.ONE_PASS`/`TWO_PASS` and the packed layout accumulates into
thread-private storage with no atomic. It is the right technique and it is worth having
-- but on this evidence it is optimising the small term, and doing it first would have
produced a disappointing number and a wrong conclusion about why.

A cheaper lever exists independently: the elastic Hessian is symmetric and
`hessian_bcrs_sym` exists, but `neohookean_ogden` declares `matrix_formats=("bsr",
"block_diag_sym")` and publishes no `bcrs_sym` kernel, so `SFEM_LINEAR_OP_TYPE=BSR_SYM`
has nothing to dispatch to on this material. Adding it to the material's format list is
a one-line experiment.

#### The gate was lifted, the probing is gone, and it did not get faster

The first route was taken: `_sfem_soa_direct_hessian_matrix_assembly_available` now
returns `True`, the direct element-matrix kernel takes the state streams and rebuilds
`gu` the way the apply does, and both probing fallbacks are deleted along with the
`bh_data`/`bout_data` scratch that existed only to feed them. The assembled matrix is
now genuinely computed rather than recovered.

**The prediction above -- that the saving is whatever fraction of an apply is tangent
evaluation, "which for a hyperelastic material is most of it" -- was wrong.** Same
machine, same problem, same configuration: 287.0 ms before, 274.4 to 285.7 ms after.
Unchanged within run-to-run spread.

The arithmetic says why, and it was available before the measurement. Per quadrature
point of a HEX8 vector element, evaluating the material tangent once per trial degree
of freedom is 24 x 123 = 2952 operations; the double contraction over test and trial
degrees of freedom that follows it is 24 x 24 x 3 x 6 = 10368. The tangent is a quarter
of the kernel, not most of it. Hoisting it out of the trial loop -- and all 114 of its
CSE temporaries are loop-invariant, so gcc could in principle have done it and does not
-- is worth about 11%, and it costs the linear materials their folded constants by
turning an 81-entry tangent into a runtime-indexed array. That was built, measured
against this model, and reverted.

So forming a 24x24 element matrix really does cost about twenty-four applies' worth of
arithmetic. The 24-applies coincidence was not evidence of waste; it was the shape of
the problem. What is left is that **the assembly runs at `VS = 1`**, entirely scalar,
against an apply path the rest of this document spends pages vectorising, and that the
test/trial contraction has the same tensor-product structure the apply exploits through
sum factorisation and the assembly does not.

What the change did buy is correctness that can be checked. Probing was correct by
construction and so said nothing about the tangent, and the hyperelastic assembly had no
numerical gate at all; `sfem_MatrixFromatsTest` now requires the assembled BSR to
reproduce the matrix-free action at a non-zero state, which fails at 3.2e-07 against a
1e-12 tolerance if the matrix is assembled at the wrong state.

#### The assembly, given the benchmark treatment

Grace GH200, 72 cores, `OMP_PROC_BIND=true`, `OMP_PLACES=cores`, HEX8 neo-Hookean,
3 BDF2 steps, `SFEM_LSOLVE_RTOL=1e-3`, f64 values. `Function::hessian_bsr`, per call:

| dof | elements | 1 th | 4 th | 8 th | 18 th | 36 th | 72 th | 1 -> 72 |
|---|---|---|---|---|---|---|---|---|
| 107811 | 32768 | 716.4 ms | 178.2 | 89.2 | 40.3 | 20.6 | **10.7 ms** | 67.0x |
| 352947 | 110592 | 2431.4 ms | 604.9 | 304.6 | 135.2 | 68.6 | **35.1 ms** | 69.2x |
| 2738019 | 884736 | 19531.6 ms | -- | 2455.7 | 1080.1 | 543.7 | **274.4 ms** | 71.2x |

Throughput is flat at **3.1 to 3.2 Melements/s at 72 threads on all three sizes**, and
0.4 / 0.8 / 1.6 / 3.1 at 8 / 18 / 36 / 72. That is linear scaling and no size
dependence: the assembly is compute-bound and already saturated at 32768 elements, at
22.1 microseconds of core time per element.

The apply does the opposite. On 2738019 dof it goes 87.4 -> 12.6 -> 6.71 -> 4.82 ->
4.86 ms and **stops scaling at 36 threads**, which is the bandwidth ceiling this
document already measured at 81% of peak. The two halves of BSR are limited by
different resources, and only the assembly has cores left to give.

| dof | operator | solve | setup per call | apply | setup in applies | CG it |
|---|---|---|---|---|---|---|
| 107811 | matrix-free | 0.347 s | -- | 0.563 ms | -- | 224 |
| | BSR | 0.386 s | 10.79 ms | 0.041 ms | 266 | 224 |
| | inexact | **0.286 s** | 2.90 ms | 0.312 ms | 9.3 | 224 |
| 352947 | matrix-free | 0.927 s | -- | 1.690 ms | -- | 336 |
| | BSR | 1.019 s | 35.19 ms | 0.293 ms | 120 | 336 |
| | inexact | **0.627 s** | 3.38 ms | 0.662 ms | 5.1 | 336 |
| 2738019 | matrix-free | 10.534 s | -- | 12.071 ms | -- | 729 |
| | BSR | 10.164 s | 285.7 ms | 4.787 ms | 59.7 | 729 |
| | inexact | **5.114 s** | 24.3 ms | 4.400 ms | 5.5 | 729 |

Iteration counts are identical across all three operators at all three sizes, and
matrix-free and BSR agree bit for bit on the final displacement norm
(`3.032044587279e+00` at 2738019 dof). That is the correctness result: the assembled
matrix reproduces the matrix-free action exactly, at scale, on three meshes.

**BSR does not pay at a realistic inner tolerance.** One assembly costs 59.7 BSR applies
on the saturating problem and CG does 74.7 applies per Newton step at rtol 1e-3. The
margin is thin and it inverts on the smaller meshes -- 120 and 266 applies, where BSR is
*slower* than matrix-free. The inexact operator's setup costs 5.5 applies and builds at
36.4 Melements/s against the assembly's 3.1, an 11.7x difference, which is the whole of
why it wins 2.0x over both.

One loose end: about 2.8 s of the 10.16 s BSR solve is neither assembly nor applies. The
applies (3.58 s) plus the assembly (2.86 s) plus gradients, line search and output
account for roughly 7.3 s, against roughly 0.9 s unaccounted in the matrix-free run. The
driver rebuilds the linear operator every Newton step, which reallocates the 1.77 GB
value buffer ten times; that is the plausible cause and it is not confirmed.

#### The same treatment for Mooney-Rivlin Kelvin-Voigt Newmark, and what it found

This material had **no assembled matrix at all**: `matrix_formats` was empty, so
`GeneratedMooneyRivlinKelvinVoigtNewmark::hessian_bsr` was a stub returning
`SFEM_FAILURE` and `SFEM_LINEAR_OP_TYPE=BSR` had nothing to dispatch to. Enabling it
turned up two defects that had been invisible because nothing had ever asked.

**The coupled Op had no assembly dispatch.** This material is an energy (Mooney-Rivlin)
plus a residual (Kelvin-Voigt) and `_coupled_cases` built `gradient`, `apply`,
`objective` and `objective_steps` but no `hessian_bsr`, so both halves could produce
element matrices that nothing called. The dispatch now sequences the two units exactly
the way `apply` sequences them -- both accumulate into the same `values` -- and it is
emitted only when *both* units publish an assembly kernel, because a matrix holding
only the elastic tangent is a different problem from the one the apply solves and it
would converge quietly to it.

**The residual path's matrix assembly used the wrong stream order.** Its gather wrote
`b<role>[field * NS + shape]` and its scatter's `ROW_COMPONENT`/`ROW_SHAPE` tables were
indexed component-major, while every block kernel in the same file reads
`direction[shape * NC + field]` -- `direction[0]`, `[3]`, `[6]`, `[9]` for the first
component's gradient on a TET4. So the element matrix was permuted against the kernel
that filled it *and* the kernel was handed a permuted state, which is not even a clean
permutation of the right answer. The assembled viscous operator was off by 1.5e-1
against a reference of 1.4e-2 -- larger than the answer. `plans/layout.py` now derives
all three tables from the stream index the kernels actually use.

The gate that catches it: `sfem_MRKVHomogeneousDeformationValidation` checks the
assembled BSR against the matrix-free action at a non-zero state, **once per unit and
once for both** -- elastic only, viscous only, both. Isolating the units is what turned
"the matrix is wrong" into "the residual half is wrong" in one run. All three now agree
to 3e-16, 3e-17 and 3e-16 respectively.

Grace GH200, 72 cores, HEX8, 3 Newmark steps, BiCGStab at rtol 1e-3,
`eta_s = 0.1`, `eta_b = 0`. `Function::hessian_bsr`, per call:

| dof | elements | 1 th | 4 th | 8 th | 18 th | 36 th | 72 th | 1 -> 72 |
|---|---|---|---|---|---|---|---|---|
| 107811 | 32768 | 1668.2 ms | 417.4 | 209.0 | 94.6 | 47.8 | **26.0 ms** | 64.2x |
| 352947 | 110592 | 5707.7 ms | 1422.7 | 713.0 | 318.7 | 160.8 | **82.3 ms** | 69.3x |
| 2738019 | 884736 | 45843.9 ms | -- | 5766.2 | 2580.3 | 1288.0 | **668.2 ms** | 68.6x |

Same shape as neo-Hookean -- flat throughput across sizes, near-linear scaling to 72
threads -- at **1.3 Melements/s**, 2.4x slower per element, which is what assembling
two units instead of one costs. 51.8 microseconds of core time per element against 22.1.

| dof | operator | solve | setup per call | apply | setup in applies | BiCGStab it |
|---|---|---|---|---|---|---|
| 107811 | matrix-free | 0.775 s | -- | 1.224 ms | -- | 212 |
| | BSR | **0.646 s** | 25.0 ms | 0.042 ms | 603 | 212 |
| 352947 | matrix-free | 2.598 s | -- | 3.621 ms | -- | 302 |
| | BSR | **1.689 s** | 82.1 ms | 0.320 ms | 257 | 300 |
| 2738019 | matrix-free | 32.704 s | -- | 28.022 ms | -- | 539 |
| | BSR | **16.408 s** | 664.9 ms | 4.820 ms | 138 | 539 |

**And here BSR wins, where for neo-Hookean it did not.** 2.0x at 2738019 dof, against a
dead heat on the same machine and mesh for neo-Hookean. Nothing about the assembly
changed to cause that -- it got *slower*, 668 ms against 274 -- the matrix-free apply
did. This material's apply evaluates two units and one of them is a viscous Jacobian
action, so it costs 28.02 ms where neo-Hookean's costs 12.07, while the SpMV is the same
4.82 ms in both because an SpMV only knows about the sparsity pattern. The assembly
costs 138 applies and BiCGStab does about 98 per Newton step, so on the neo-Hookean
accounting BSR should lose; it wins anyway because each apply it replaces is 5.8x the
one it substitutes.

That is the general rule this pair of measurements gives: **assembly pays in proportion
to how expensive the matrix-free apply is relative to an SpMV, not in proportion to how
cheap the assembly is.** The break-even in applies is the wrong number to watch on its
own.

Two cautions on the table. The displacement norms agree between the two operators to
nine or ten digits rather than exactly, and the BiCGStab iteration counts wander by a
few per cent across the thread sweep (489, 546, 535, 474, 491 on the largest mesh),
because BiCGStab is far more sensitive to rounding than CG; the *setup* column is the
clean measurement and the solve column carries that noise. And the BSR apply saturates
at 36 threads here too -- 4.703 ms at 36 against 4.826 at 72 -- so the same split holds:
the assembly is compute-bound and scales, the apply is bandwidth-bound and does not.

One thing this did **not** fix: the residual path still builds its element matrix by
probing, 12 unit basis vectors through `jacobian_action` per TET4 element. The energy
path no longer does. That is the same construction removed above, in a second emitter.

#### The two-unit material's inexact apply on HEX8, standard and packed

TET4 established correctness and TET10 the accuracy under curvature; HEX8 is the element
the driver benchmarks above run on, and it had never been measured. Grace GH200 at 72
threads, `-O3 -march=native`, best of 7, pack size 128. MDOF/s for the apply, except
`assembly`, which is the once-per-tangent kernel:

| elements | ndof | exact | st. f64 | st. f32 | st. f16 | **pk. f64** | pk. f32 | assembly | f64 rel | pk rel |
|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 2187 | 30.29 | 36.20 | 35.72 | 54.88 | 28.25 | 24.07 | 13.43 | 6.3e-04 | 6.3e-04 |
| 4096 | 14739 | 74.11 | 146.54 | 135.74 | 130.20 | 181.40 | 155.06 | 26.62 | 1.5e-04 | 1.5e-04 |
| 13824 | 46875 | 88.11 | 231.64 | 209.48 | 199.81 | 399.65 | 324.11 | 29.78 | 6.6e-05 | 6.6e-05 |
| 32768 | 107811 | 96.09 | 287.08 | 249.37 | 231.89 | 565.06 | 443.47 | 30.66 | 3.7e-05 | 3.7e-05 |
| 64000 | **206763** | 94.78 | 320.08 | 278.30 | 281.05 | **672.02** | 520.24 | 31.61 | 2.3e-05 | 2.3e-05 |

**Packing is worth 2.10x on top of the stored apply, and 7.09x over the exact one.** The
packed answer is identical to the standard one at every row -- `pk rel` matches `f64 rel`
to every digit shown, so what is left is the projection error and nothing the layout did.

The pack size matters more here than anywhere else in this document, because it decides
how many packs there are to share out and the packed loop is `#pragma omp for` over them:

| pack size | packs at 64000 elements | pk. f64 at 206763 dof | packed vs standard |
|---|---|---|---|
| 128 | 500 | **672.02** | **2.10x** |
| 256 | 250 | 607.18 | 1.89x |
| 512 | 125 | 604.48 | 1.87x |
| 1024 | 63 | 600.91 | 1.89x |

At the largest size the spread is modest because even 63 packs is close to 72 threads.
Two rows up it is brutal: at 4096 elements a pack size of 128 gives 181.40 MDOF/s and
1024 gives 36.23, because 1024 leaves **four** packs for 72 threads. Smaller is safer,
and the number to watch is the pack count against the thread count, not the pack size.

**Narrowing the store does not pay on this material, in either layout.** Standard f32 is
278 against f64's 320; packed f32 is 520 against packed f64's 672. That inverts every
single-unit result in this document, where halving the store bought bandwidth. The cause
is the store's weight: the Kelvin-Voigt tangent is unsymmetric and cannot fold 81 numbers
into 45, so the pair costs **1008 bytes per element in f64** -- 45 elastic plus 81
viscous -- against 360 for a single symmetric unit. At that arithmetic-per-byte the apply
is not bandwidth-bound at these sizes, and the conversion on every load is a cost with
nothing to buy. f16 gives up accuracy for it as well, 5.8e-04 against 2.3e-05, so there is
no configuration of this material in which the narrow stores are the right choice.

The accuracy columns are the projection error rather than a store error, and they converge
under refinement -- 6.3e-04 to 2.3e-05 -- which is what an inexact quadrature on a
non-affine element should do. f64 and f32 agree to every digit shown, which says the same
thing from the other side.

Break-even rises to **4.3 applies per tangent** from the standard layout's 3.3, which is
the right direction and not a regression: the apply got twice as cheap while the assembly
did not, so it takes more of them to repay the same setup. A Newton step here does about
98.

**What it costs to reproduce, and why.** Generation is 1380 s for HEX8 alone, which is
why the material ships with `inexact_apply` off. The compile is worse: gcc 13.3 at
`-O3 -march=native` takes **38 minutes and over a gigabyte** on `bench_mixed.cpp`, which
overruns a 28-minute debug job. Compile it on the login node.

That number is a gcc pathology and not what this code costs to compile. **AppleClang 17
builds the same translation unit in 15.1 s and 381 MB**, so the 150x is the compiler, not
the source. Measured on the worst single kernel -- the viscous tangent, whose
`#pragma omp simd` body is 3855 straight-line statements -- gcc takes 239 s and 3.2 GB,
and `-ftime-report` says where:

| pass | time | share | memory |
|---|---|---|---|
| instruction scheduling | 51.5 s | 22% | 16 MB |
| combiner | 43.9 s | 19% | 35 MB |
| load CSE after reload | 35.0 s | 15% | -- |
| dead store elim2 | 10.1 s | 4% | **1473 MB** |
| tree SLP vectorization | 0.17 s | 0% | 5 MB |

It is the RTL back end, and the obvious suspect is not guilty: the SLP vectoriser costs
0.17 s and `-fno-tree-slp-vectorize` changes the total by less than one per cent, 237 s
against 239. Nor does lowering the optimisation level rescue it -- `-O2` is 205 s -- and
dropping `-march=native` for a generic `-O3` only reaches 145 s. The memory is the real
hazard: 3.2 GB for one kernel, `dead store elim2` alone holding 1.47 GB, which is what
would make a parallel build of several such kernels exhaust a node.

Every one of those passes is superlinear in basic-block size, and compiling the eight
kernels one at a time says which one and by how much:

| unit | kernel | wall | peak RSS | longest simd block |
|---|---|---|---|---|
| **viscous** | **`inexact_apply_tangent`** | **234.8 s** | **4.45 GB** | 3855 |
| elastic | `inexact_apply_tangent` | 83.0 s | 1.35 GB | 2654 |
| viscous | `..._stored_packed_two_pass` | 11.0 s | 295 MB | 769 |
| viscous | `..._stored` | 9.7 s | 221 MB | 683 |
| elastic | `..._stored_packed_two_pass` | 8.3 s | 258 MB | 733 |
| elastic | `..._stored` | 7.0 s | 147 MB | 647 |
| viscous | `..._compressed` | 0.57 s | 37 MB | -- |
| elastic | `..._compressed` | 0.48 s | 37 MB | -- |

The two tangents are 318 s of 355 s, and the viscous one alone is two thirds. Against the
stored apply of its own unit it is **5.6x the block size, 24x the compile time and 20x the
memory** -- close to quadratic, which is what those passes are.

The memory is the part to watch rather than the wall time: a `-j` build compiling both
tangents at once wants about 5.8 GB, and anything wider exhausts a node. Note also that
these eight sum to six minutes, not thirty-eight; the larger figure is a login-node build
of a bigger unit that also compiles the exact-apply operator sources in the same serial
invocation, and the two are not directly comparable.

So the fix that would help every compiler is on the generator's side, and it has an
address: `_tangent_lines` in `emitters/inexact_apply_codegen.py` emits the whole CSE chain
into one `#pragma omp simd` scope. Splitting that block should return roughly quadratic
savings and cut the memory more than the time. The cheap fix here is to split the two
units into separate translation units, which share nothing but the mesh, so they compile
concurrently.

### What the first version of these driver numbers got wrong

The first three versions of this table were measured on a mis-constrained problem, and
the mistake is worth keeping because none of the numbers looked wrong.

`smesh`'s `cube` driver SFC-reorders the mesh it writes. The Dirichlet node ids here
were derived from the lattice the cube was built on -- `(k*nn + j)*nn + i` -- which
after reordering names arbitrary interior nodes: of 1089 ids meant for the clamped
face, **37** were on it. The solve still ran, still converged, still printed a
displacement norm. It was solving a different problem: a bar with a scatter of pinned
points through its interior.

Three conclusions came out of that problem and all three were wrong. That partial
assembly costs Newton iterations -- ten against five -- when it costs none. That the
solve speed-up was 1.15x, when it is 1.72x to 2.5x. And that the technique "stops
converging under refinement", from a 48-cube run that hit every iteration cap: with
correct constraints the matrix-free operator fails there in exactly the same way, at
exactly the same load, because the failure was a load too large for one time step and
had nothing to do with the operator at all.

The lesson is not about `cube`. It is that a boundary condition derived from an
assumed node numbering is unverifiable from the output: a solve on the wrong nodes
converges to the wrong answer quietly. Deriving the nodeset from the coordinates costs
four lines and cannot be wrong in this way. Everything in this section is now built
that way.

Two driver defects surfaced while chasing it, both worth having:

* **`SFEM_ASSUME_AFFINE`**. A constant-P1 element publishes only affine kernels, so
  TET4 and TRI3 had nothing for this driver to dispatch to and every call failed.
* **A failed gradient is now fatal.** It was unchecked, so an operator with no kernel
  for the element left `rhs` zeroed and the Newton step read that as convergence:
  gnorm 0, zero iterations, and a printed solution that was never solved for.

## What the first version of these measurements got wrong

The throughput figures above replace an earlier set that was wrong, and the way it
was wrong is worth keeping, because the kernels were genuinely running in parallel
the whole time -- the pragmas were emitted, `libomp` was linked, CPU time exceeded
wall time -- and the *measurement* was still not a multithreaded result.

Two pieces of serial work sat inside the timed region.  The output arrays were
cleared with a serial `std::fill` on every timed call: several megabytes that are
no part of the operator and do not parallelise.  And each timing sample wrapped a
single call, charging it for OpenMP team startup and a cold cache -- a large share
of a kernel that runs a few milliseconds on ten cores.

The effect is invisible at one thread and grows with the thread count, which is
the worst possible shape for a scaling study: the single-thread figures barely
moved when it was fixed (2.52 to 2.68), while the eight-thread figures moved by
half (13.04 to 18.75, and 32.58 to 55.15).  Measured scaling was 5.4x where it is
really 7.0x, and the peak appeared to be at ten threads where it is really eight.
The single-threaded tables were understated too, most on TET4, whose kernels are
short enough for per-call overhead to dominate.

The speed-up *ratios* barely moved, because both paths paid the same overhead.
That is exactly why the flaw was easy to miss: the conclusion looked stable while
the numbers under it were not.

The timed region now holds the kernel and nothing else, over many repetitions.
The outputs are not cleared between them -- the apply accumulates, so the values
grow, which does not affect what is being timed, and correctness is checked
separately with clearing, outside any timed region.  `scale_probe.cpp` isolates a
single kernel with no harness at all and confirms the corrected figures.

## Reproducing

The supported way is CMake; see `README.md` under **Building**:

    cmake -S spikes/inexact_apply_compare -B build-spike \
          -DSPIKE_MATERIAL=<material> -DSPIKE_ELEMENT=<element>
    cmake --build build-spike -j
    PACK_SIZE=512 OMP_NUM_THREADS=72 ./build-spike/bench_split 5

The shell scripts below produced the numbers recorded above and still work.

    spikes/inexact_apply_compare/run_split.sh <material> <element> [repeats]
    spikes/inexact_apply_compare/run_mixed.sh <element> [repeats]
    spikes/inexact_apply_compare/run_store_precision.sh <element> [amplitudes...]
    spikes/inexact_apply_compare/run_warp.sh  <material> <element> [n]
    spikes/inexact_apply_compare/run_roofline.sh <material> <element> [machine]

`run_mixed.sh` drives the two-unit material and takes no material argument: it
generates one tree and links two sets of kernels out of it, because the energy
unit's exact entry point is `apply_a_msoa` and the residual unit's is
`jacobian_action_a_msoa`.  `MIXED_EXTRA_FLAGS=-DRANDOM_INCREMENT` selects the
white-noise increment, which is the one to use for anything about accuracy, and
`-DSTATE_AMPLITUDE=<a>` sets the deformation severity.  `run_store_precision.sh`
is those two together as a sweep, with the zero-amplitude control that separates
a store error from a projection error.

Elements are TET4, HEX8 and TET10.  `WARP_EXTRA_FLAGS=-DRANDOM_INCREMENT` selects
the white-noise increment.  `SFEM_MAIN_CHECKOUT`, `SFEM_BUILD`, `SFEM_PYTHON` and
`SFEM_SPIKE_WORK` override the paths.  Both keep a full unfiltered log.

For OpenMP on macOS, add
`-Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include -L$(brew --prefix libomp)/lib -lomp`.


## Grace again, after the codegen rework, and with the packed layout beside it

The same measurement as *Neohookean Ogden on Grace* above, rerun on one GH200
socket (`nid006549`, 72 Neoverse-V2 cores, GCC 13.3 from `prgenv-gnu/24.11`,
`OMP_PLACES=cores`, `OMP_PROC_BIND=true`, twenty repetitions, 206763 dof).
HEX8, and this time with the packed layout in the same run.

**These numbers are not comparable with the recorded run above, and the ratio
table this section first carried has been withdrawn.**

The recorded run predates not only the codegen rework but the benchmark harness
itself.  At the commit that recorded it, `element_mesh.inc` had no `MESH_ORDER`
at all: the mesh was built lexicographically and never reordered.  This run uses
`morton3`, which is what SFEM runs and what the harness has defaulted to since.
*Ordering the benchmark mesh* in this same file measures what that is worth to
this kernel -- a factor of four when gather locality is destroyed -- so a
difference of 23 to 28% between the two runs says nothing about the generator.

The exact apply held to within 2% across the two, which is what made the
comparison look safe; it is not a control across a change of mesh, only across a
change of kernel.  Three passes of one binary agree to better than 1%, so the
measurement itself is reproducible and the cross-run difference is real -- it is
simply not attributable to anything in the generator.

What this run does support, because it is all one run on one mesh:

| threads | exact | st. f64 | st. f32 | st. f16 | assembly |
|---|---|---|---|---|---|
| 1 | 3.58 | 13.65 | 11.27 | 10.33 | 1.85 |
| 8 | 27.80 | 99.52 | 82.73 | 76.09 | 14.87 |
| 32 | 105.79 | 336.21 | 286.04 | 271.50 | 59.48 |
| 72 | 215.79 | 568.55 | 500.92 | 494.40 | 132.16 |

**The f64 store beats the f32 one at every thread count**, by 13% at 72 threads
and 21% at 1.  That is the opposite of what halving the store traffic would
suggest, and it is the same sign in the recorded run, so it is a property of the
machine rather than of any change here: reading a `float` store into `double`
arithmetic costs 45 widening converts per element, and on Neoverse-V2 that
exceeds what the 180 saved bytes per element buy.  f32 remains the right default
for its memory footprint; it is not the faster one on this machine.

Break-even at 72 threads is 2.6 applies per tangent for f64, 2.9 for f32.

**The packed layout is the result worth having**, and the recorded run had no
packed column at all:

| threads | exact | packed exact | st. f32 | packed st. f32 | packed speed-up |
|---|---|---|---|---|---|
| 1 | 3.58 | 4.08 | 11.27 | 17.28 | 1.53x |
| 8 | 27.80 | 31.52 | 82.73 | 131.07 | 1.58x |
| 32 | 105.79 | 121.32 | 286.04 | 465.08 | 1.63x |
| 72 | 215.79 | 227.94 | 500.92 | 749.13 | 1.50x |

**749 MDOF/s against 501**, and against 216 for the exact matrix-free apply on
the same mesh: the projected apply on a packed mesh is 3.5x the exact one at 72
threads.  Packing helps the projected apply far more than the exact one (1.50x
against 1.06x), which is what the roofline predicts -- the stored apply moves
384 of its 788 bytes per element in the scatter, and packing is what removes it.

Every packed answer agrees with its standard counterpart to round-off (`pk diff`
1.2e-16 for the exact apply) or exactly (3.6e-05 for the projected one, the same
deviation the standard column reports), so the two layouts compute the same
thing.


## The delegation, measured properly

*Withdraw the cross-run ratios* above says what the previous comparison was not.
This is the one it should have been: two kernel trees, one harness, one mesh
(morton3), one node (`nid006546`), built in the same job and run interleaved,
three passes each, twenty repetitions per pass, 206763 dof.

    old   HEX8 publishes its own inexact micro-kernel
    new   HEX8 forwards to PROTEUS_HEX8 -- the same expression list in a
          different node numbering, with the connectivity permuted once per
          call instead of per element

Everything else is held: same generator otherwise, same flags, same compiler,
`OMP_PLACES=cores`, `OMP_PROC_BIND=true`.  Three passes of one binary agree to
0.24% at 72 threads, so this resolves about half a percent.

Means of three passes, MDOF/s:

| threads | | exact | st. f64 | st. f32 | st. f16 | assembly | packed st. f32 |
|---|---|---|---|---|---|---|---|
| 1 | old | 3.590 | 13.780 | 11.307 | 10.287 | 1.893 | 17.443 |
| 1 | new | 3.597 | 13.797 | 11.323 | 10.353 | 1.873 | 17.400 |
| 8 | old | 27.967 | 99.847 | 84.157 | 76.527 | 14.987 | 132.253 |
| 8 | new | 27.957 | 100.253 | 83.327 | 76.807 | 14.967 | 131.867 |
| 72 | old | 212.657 | 563.503 | 498.163 | 479.900 | 130.630 | 761.973 |
| 72 | new | 212.647 | 565.487 | 496.410 | 486.737 | 130.787 | 772.827 |

**Neutral.**  At 72 threads the ratios are 1.000, 1.004, 0.996, 1.014, 1.001 and
1.014; nothing anywhere in the table moves by more than 1.4%, and the packed
column's own pass-to-pass spread at 72 threads is 2.3%, wider than the
difference between the variants.

The exact apply is the control and agrees to **0.005%** at 72 threads, which is
what a control should look like when the two runs really are the same
measurement.

Two things this settles beyond the delegation:

**The 12 to 13% "assembly gain" withdrawn above was the mesh.**  Assembly at one
thread is 1.89 for the old kernel and 1.87 for the new, against 1.65 in the
recorded run.  Both kernels here are faster than the record by the same amount,
on the same ordered mesh, which is where that difference lives.

**f64 beating f32 is not new either.**  It holds in both variants at every
thread count, so it is the machine: 45 widening converts per element cost more
on Neoverse-V2 than the 180 bytes per element they save.
