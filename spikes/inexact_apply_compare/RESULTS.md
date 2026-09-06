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

## Reproducing

`bench.cpp` (TET4) and `bench_hex8.cpp` (HEX8) take the material through
`-DEXACT_APPLY=`, `-DPROJECTED_APPLY=`, `-DMATERIAL_INEXACT_HEADER=` and
`-DEXACT_TAKES_STATE` for a material whose exact apply takes one. Generate with
`inexact_apply=True`, then compile the bench against the element operator (and
the PROTEUS_HEX8 unit, which HEX8 aliases into).

# The split form: measured

The prediction above was that the split form is the one that can pay. It does.
`bench_split.cpp`, driven by `run_split.sh`, generates the split kernels, builds
`Sbar` once into a store, and then applies it — at three storage precisions, so
the store's cost and the apply's accuracy can be read against each other.

Single-threaded, `-O3 -march=native`, Apple M-series laptop, best of 7 after one
untimed warm-up pass, TET4.

**neohookean Ogden**

| ndof | exact | fused | stored f64 | stored f32 | stored f16 | assembly |
|---|---|---|---|---|---|---|
|   2187 | 7.22 | 6.23 | 45.25 | 42.78 | 44.86 |  6.29 |
|  14739 | 5.94 | 5.10 | 27.91 | 28.19 | 28.43 |  5.04 |
|  46875 | 5.58 | 4.77 | 30.31 | 31.85 | 34.97 |  4.71 |
| 107811 | 5.24 | 4.58 | 14.36 | 19.55 | 25.13 |  4.21 |
| 206763 | 5.13 | 4.49 | 22.71 | 26.21 | 29.63 |  4.21 |

**linear elasticity**

| ndof | exact | fused | stored f64 | stored f32 | stored f16 | assembly |
|---|---|---|---|---|---|---|
|   2187 | 24.14 | 22.96 | 31.89 | 30.36 | 31.60 | 22.70 |
|  14739 | 19.53 | 17.55 | 26.08 | 26.83 | 27.95 | 24.08 |
|  46875 | 18.72 | 17.11 | 28.43 | 31.84 | 35.07 | 20.40 |
| 107811 | 18.26 | 15.61 | 13.98 | 17.13 | 24.48 | 10.80 |
| 206763 | 17.70 | 16.19 | 22.29 | 26.16 | 29.88 | 17.33 |

All figures MDOF/s. At 206763 degrees of freedom:

| material | exact | stored f16 | speed-up | break-even applies/tangent |
|---|---|---|---|---|
| neohookean Ogden  |  5.13 | 29.63 | 5.78x | 1.5 |
| linear elasticity | 17.70 | 29.88 | 1.69x | 2.5 |

Break-even is the number of applies that must share one tangent before the split
has repaid its assembly: `(1/a) / (1/e - 1/s)` for assembly `a`, exact `e` and
stored `s`. Both are under three, and a Krylov solve does tens to hundreds, so
the technique pays from the first Newton step.

The speed-up is large exactly where the fused form was worst. That is the whole
argument: the fused kernel's cost is dominated by evaluating the material
tangent, and the split kernel does not evaluate it at all — the apply reads 45
numbers and contracts. Neohookean and linear elasticity converge to nearly the
same stored throughput (29.63 and 29.88) because at that point they are running
the *same* kernel on the same amount of data; only the assembly differs.

## Accuracy of the store

| store | bytes/element | rel. difference from exact |
|---|---|---|
| f64                     | 360 | 3.8e-15 (neohookean), 3.7e-15 (linear elasticity) |
| f32 (`metric_tensor_t`) | 180 | 1.7e-07, 1.5e-07 |
| f16 + scale (`compressed_t`) | 94 | 1.4e-03, 2.1e-04 |

The f64 store reproduces the exact apply to round-off, which is the gate: on an
affine simplex the projection loses nothing, so any difference would be a bug in
the split rather than an approximation. f32 costs seven digits and f16 costs
three, against a 3.8x reduction in the store.

The scale is applied to the outputs rather than to the 45 stored components. The
action is linear in `Sbar`, so it is the same number, reached with `dim *
n_nodes` multiplies instead of 45 and without a decompressed copy of the tangent
in registers.

## Two measurement artifacts, one explained and one not

**Warm-up matters more than expected.** Without an untimed first pass the
assembly kernel reads 3.5 MDOF/s; with one it reads 17-24. The store is 138 MB at
f64 and the untimed difference was page-fault cost, not kernel cost. Every figure
above is post-warm-up. The earlier fused-vs-exact tables in this file do not have
this problem — they touch no large store — but any future measurement of the
split must keep the warm-up.

**The 107811-dof row is systematically slow and the cause is not established.**
It is depressed for every stored variant, in both materials, across repeated
runs — not noise. The hypothesis was cache-set conflict: at 196608 elements the
unpadded component stride is exactly 1.5 MB, so all 45 streams alias. Padding the
stride by 64 elements did *not* remove it. The stride is a kernel parameter and
the padding is in the bench, but the explanation is wrong or incomplete, and the
row should be treated as unexplained rather than dismissed.

## Reproducing

    spikes/inexact_apply_compare/run_split.sh <material> [repeats]

Generates, compiles and runs, keeping a full unfiltered log. `SFEM_MAIN_CHECKOUT`,
`SFEM_BUILD`, `SFEM_PYTHON` and `SFEM_SPIKE_WORK` override the paths.
