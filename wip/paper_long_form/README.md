# The operator, long form

The companion to `wip/paper`. The paper is hard-limited to ten pages and carries the performance
argument; it has no room for the discretisation. This document carries the discretisation: every
component of the CVFEM HEX8 Navier–Stokes operator in the form the code evaluates, the
micro-kernel that evaluates it for each convective scheme and layout, and the file and function
that implements it.

    make          build main.pdf
    make check    builds, no undefined refs, no LaTeX errors, no stub sections

## What is here

| section | contents |
|---|---|
| 1 | the CVFEM discretisation, the twelve sub-control surfaces, the residual's antisymmetry |
| 2 | geometry: the affine and isoparametric variants, and why the Jacobian determines the affine edge vectors |
| 3 | components: viscous, convective, upwind split, Rhie–Chow, transient, boundary |
| 4 | the five convective schemes, each limiter in closed form, and the bound-preservation table |
| 5 | **the micro-kernels**: staging, loop structure, compile-time dispatch, per-scheme generated kernels, what the layout changes |
| 6 | the Jacobian action: lagged vs exact, the coefficient's velocity sensitivity, limiter subgradients, partial assembly, the three oracles |
| 7 | the nodal gradient reconstruction, and the optimisations that were measured and rejected |

## Two rules this document keeps

**Every formula is the one the code evaluates**, not a textbook form it resembles. Where the
implementation departs from the usual presentation — the Rhie–Chow time scale, the
Darwish–Moukalled algebra, every limiter derivative — the departure is stated and the reason
given. Those are exactly the places where reading the equations and the source side by side would
otherwise mislead.

**No measured number is typed in.** Timings belong to the paper, which generates them from the
measurement files under `wip/paper/data`. What this document carries instead are *counts* —
loads per surface, live values, lines of generated kernel, the bound-preservation percentages —
which are properties of the written kernel or of an instrumented run, and are verifiable by
reading the source. If a count here disagrees with the source, the source is right and this is
stale.

## Why the rejected optimisations are in here

Sections 5 and 7 record changes that were built, measured and reverted, with their numbers. They
are kept because each one's static argument remains superficially convincing — a load count, a
spill count, an accumulator count — and a reader who re-derives it will rebuild the change. The
measurement is the artifact, not the code.
