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

**fp16 is not worth carrying on this machine.**  At one thread it is *slower*
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
    spikes/inexact_apply_compare/run_warp.sh  <material> <element> [n]

Elements are TET4, HEX8 and TET10.  `WARP_EXTRA_FLAGS=-DRANDOM_INCREMENT` selects
the white-noise increment.  `SFEM_MAIN_CHECKOUT`, `SFEM_BUILD`, `SFEM_PYTHON` and
`SFEM_SPIKE_WORK` override the paths.  Both keep a full unfiltered log.

For OpenMP on macOS, add
`-Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include -L$(brew --prefix libomp)/lib -lomp`.
