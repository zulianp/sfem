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
