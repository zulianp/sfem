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
