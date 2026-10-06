# CVFEM kernel variants and what they measure

Throughput of the CVFEM element kernels, with the operator each number was
measured on shown beside it.

The benchmark measures kernels in isolation, which is a deliberate and useful
thing to do -- it is how a change to the arithmetic is seen without the rest of
the solve moving underneath it. It is not the operator the Newton loop
evaluates. Both appear here, labelled, because the difference between them has
been mistaken for a speedup before.

## What each variant computes

Read from the `ran_*` columns, which the benchmark sets where it dispatches --
not from the flags it was passed. A variant that asked for a term and did not
get it shows a dash here. `frozen by design` is the assembled Jacobian, which
keeps the frozen Rhie-Chow form deliberately: the exact term couples pressures
beyond nearest neighbours and would widen the BSR pattern, and that operator
exists only to build the preconditioner.

| operation | variant | Rhie-Chow | exact-RC J | state ∇p | boundary | upwind band | transient | completeness |
|---|---|---|---|---|---|---|---|---|
| assemble | atomic / fd | – | – | – | – | – | – | element kernel |
| assemble | atomic / fd / isoparam | – | – | – | – | – | – | element kernel |
| assemble | atomic / split | – | – | – | – | – | – | element kernel |
| assemble | atomic / sumfact | frozen by design | frozen by design | hoisted | carried | – | – | partial |
| assemble | atomic / sumfact | frozen by design | frozen by design | hoisted | – | – | – | frozen-RC |
| assemble | atomic / sumfact | – | – | – | – | – | – | element kernel |
| assemble | atomic / sympy | – | – | – | – | – | – | element kernel |
| assemble | atomic / sympy_block | – | – | – | – | – | – | element kernel |
| assemble | colored / sumfact | – | – | – | – | – | – | element kernel |
| assemble | colored / sympy | – | – | – | – | – | – | element kernel |
| assemble | colored / sympy_block | – | – | – | – | – | – | element kernel |
| assemble | packed / sumfact | frozen by design | frozen by design | hoisted | carried | – | – | partial |
| assemble | packed / sumfact | – | – | – | – | – | – | element kernel |
| assemble | packed / sympy | – | – | – | – | – | – | element kernel |
| assemble | packed / sympy_block | – | – | – | – | – | – | element kernel |
| assemble | store / sumfact | – | – | – | – | – | – | element kernel |
| assemble | store / sympy | – | – | – | – | – | – | element kernel |
| assemble | store / sympy_block | – | – | – | – | – | – | element kernel |
| assemble_diag | atomic / (kernel n/a) | frozen by design | frozen by design | hoisted | carried | – | carried | partial |
| assemble_diag | atomic / (kernel n/a) | frozen by design | frozen by design | hoisted | carried | – | – | partial |
| assemble_diag | atomic / (kernel n/a) | frozen by design | frozen by design | hoisted | – | – | – | frozen-RC |
| assemble_diag | atomic / (kernel n/a) | – | – | – | – | – | – | element kernel |
| assemble_diag | atomic / (kernel n/a) / isoparam | frozen by design | frozen by design | hoisted | carried | – | – | partial |
| bsr_apply | packed / (kernel n/a) | frozen by design | frozen by design | hoisted | carried | – | – | partial |
| bsr_apply | packed / (kernel n/a) | frozen by design | frozen by design | hoisted | – | – | – | frozen-RC |
| bsr_apply | packed / (kernel n/a) | – | – | – | – | – | – | element kernel |
| jac_action | atomic / (kernel n/a) | carried | carried | hoisted | carried | – | – | solver operator |
| jac_action | atomic / (kernel n/a) | – | – | – | – | – | – | element kernel |
| jac_action | atomic / (kernel n/a) / isoparam | – | – | – | – | – | – | element kernel |
| jac_action | atomic / sympy_action | – | – | – | – | – | – | element kernel |
| jac_action | atomic / sympy_action_comp | – | – | – | – | – | – | element kernel |
| jac_action | atomic / sympy_action_face | – | – | – | – | – | – | element kernel |
| jac_action | atomic / sympy_action_geom | – | – | – | – | – | – | element kernel |
| jac_action | atomic / sympy_action_geomface | – | – | – | – | – | – | element kernel |
| jac_action | atomic / sympy_action_node | – | – | – | – | – | – | element kernel |
| jac_action | colored / (kernel n/a) | – | – | – | – | – | – | element kernel |
| jac_action | packed / (kernel n/a) | carried | carried | hoisted | carried | – | carried | solver operator |
| jac_action | packed / (kernel n/a) | carried | carried | hoisted | carried | – | – | solver operator |
| jac_action | packed / (kernel n/a) | carried | carried | hoisted | – | – | – | partial |
| jac_action | packed / (kernel n/a) | – | – | – | – | – | – | element kernel |
| jac_action | packed / (kernel n/a) / isoparam | – | – | – | – | – | – | element kernel |
| jac_action | store / (kernel n/a) | – | – | – | – | – | – | element kernel |
| residual | atomic / current | – | – | – | – | – | – | element kernel |
| residual | atomic / current / isoparam | – | – | – | – | – | – | element kernel |
| residual | atomic / sumfact | carried | – | hoisted | carried | – | – | solver operator |
| residual | atomic / sumfact | carried | – | hoisted | – | – | – | partial |
| residual | atomic / sumfact | carried | – | per apply | – | – | – | partial |
| residual | atomic / sumfact | – | – | – | – | – | – | element kernel |
| residual | atomic / sympy | – | – | – | – | – | – | element kernel |
| residual | atomic / sympy / isoparam | – | – | – | – | – | – | element kernel |
| residual | colored / current | – | – | – | – | – | – | element kernel |
| residual | colored / sumfact | – | – | – | – | – | – | element kernel |
| residual | colored / sympy | – | – | – | – | – | – | element kernel |
| residual | packed / current | – | – | – | – | – | – | element kernel |
| residual | packed / isoparam_simd / isoparam | – | – | – | – | – | – | element kernel |
| residual | packed / sumfact | carried | – | hoisted | carried | – | carried | solver operator |
| residual | packed / sumfact | carried | – | hoisted | carried | – | – | solver operator |
| residual | packed / sumfact | carried | – | hoisted | – | – | – | partial |
| residual | packed / sumfact | carried | – | per apply | carried | – | carried | solver operator |
| residual | packed / sumfact | carried | – | per apply | – | – | – | partial |
| residual | packed / sumfact | – | – | – | – | – | – | element kernel |
| residual | packed / sympy | – | – | – | – | – | – | element kernel |
| residual | store / current | – | – | – | – | – | – | element kernel |
| residual | store / sumfact | – | – | – | – | – | – | element kernel |
| residual | store / sympy | – | – | – | – | – | – | element kernel |


## Throughput

Best observed rate per configuration, in MDOF/s. Best rather than mean because
background load can only make a run slower. The completeness column is the one
from the table above: two rows are comparable only when it matches.

| operation | variant | dof | MDOF/s | completeness | working set | threads |
|---|---|---|---|---|---|---|
| assemble | colored / sumfact | 4,121,204 | 87 | element kernel | warm | 72 |
| assemble | colored / sympy | 4,121,204 | 79 | element kernel | warm | 72 |
| assemble | colored / sympy_block | 4,121,204 | 77 | element kernel | warm | 72 |
| assemble | store / sumfact | 4,121,204 | 72 | element kernel | warm | 72 |
| assemble | store / sympy | 4,121,204 | 71 | element kernel | warm | 72 |
| assemble | store / sympy_block | 4,121,204 | 70 | element kernel | warm | 72 |
| assemble | atomic / sympy_block | 4,121,204 | 57 | element kernel | warm | 72 |
| assemble | packed / sumfact | 4,121,204 | 56 | element kernel | warm | 72 |
| assemble | packed / sympy | 4,121,204 | 56 | element kernel | warm | 72 |
| assemble | atomic / sympy | 4,121,204 | 56 | element kernel | warm | 72 |
| assemble | packed / sympy_block | 4,121,204 | 55 | element kernel | warm | 72 |
| assemble | packed / sumfact | 4,121,204 | 53 | partial | warm | 72 |
| assemble | atomic / sumfact | 4,121,204 | 39 | element kernel | warm | 72 |
| assemble | atomic / sumfact | 4,121,204 | 34 | frozen-RC | warm | 72 |
| assemble | atomic / sumfact | 4,121,204 | 34 | partial | warm | 72 |
| assemble | atomic / fd / isoparam | 4,121,204 | 21 | element kernel | warm | 72 |
| assemble | atomic / fd | 4,121,204 | 21 | element kernel | warm | 72 |
| assemble | atomic / split | 4,121,204 | 14 | element kernel | warm | 72 |
| assemble_diag | atomic / (kernel n/a) | 4,121,204 | 112 | element kernel | warm | 72 |
| assemble_diag | atomic / (kernel n/a) | 4,121,204 | 92 | frozen-RC | warm | 72 |
| assemble_diag | atomic / (kernel n/a) | 4,121,204 | 91 | partial | warm | 72 |
| assemble_diag | atomic / (kernel n/a) | 4,121,204 | 89 | partial | warm | 72 |
| assemble_diag | atomic / (kernel n/a) / isoparam | 4,121,204 | 68 | partial | warm | 72 |
| bsr_apply | packed / (kernel n/a) | 1,098,500 | 484 | element kernel | warm | 72 |
| bsr_apply | packed / (kernel n/a) | 4,121,204 | 458 | element kernel | warm | 72 |
| bsr_apply | packed / (kernel n/a) | 4,121,204 | 457 | frozen-RC | warm | 72 |
| bsr_apply | packed / (kernel n/a) | 4,121,204 | 451 | partial | warm | 72 |
| jac_action | packed / (kernel n/a) | 4,121,204 | 2071 | element kernel | warm | 72 |
| jac_action | store / (kernel n/a) | 4,121,204 | 2057 | element kernel | warm | 72 |
| jac_action | packed / (kernel n/a) | 1,098,500 | 2013 | element kernel | warm | 72 |
| jac_action | packed / (kernel n/a) | 1,098,500 | 1043 | solver operator | warm | 72 |
| jac_action | packed / (kernel n/a) | 4,121,204 | 963 | partial | warm | 72 |
| jac_action | colored / (kernel n/a) | 4,121,204 | 905 | element kernel | warm | 72 |
| jac_action | packed / (kernel n/a) | 4,121,204 | 892 | solver operator | warm | 72 |
| jac_action | packed / (kernel n/a) | 4,121,204 | 860 | solver operator | warm | 72 |
| jac_action | packed / (kernel n/a) | 4,121,204 | 845 | solver operator | cold, 7 live | 72 |
| jac_action | atomic / (kernel n/a) | 4,121,204 | 844 | element kernel | warm | 72 |
| jac_action | packed / (kernel n/a) / isoparam | 4,121,204 | 810 | element kernel | warm | 72 |
| jac_action | atomic / (kernel n/a) | 4,121,204 | 500 | solver operator | warm | 72 |
| jac_action | atomic / sympy_action_geomface | 4,121,204 | 461 | element kernel | warm | 72 |
| jac_action | atomic / sympy_action_face | 4,121,204 | 425 | element kernel | warm | 72 |
| jac_action | atomic / (kernel n/a) / isoparam | 4,121,204 | 397 | element kernel | warm | 72 |
| jac_action | atomic / sympy_action_geom | 4,121,204 | 360 | element kernel | warm | 72 |
| jac_action | atomic / sympy_action_node | 4,121,204 | 333 | element kernel | warm | 72 |
| jac_action | atomic / sympy_action_comp | 4,121,204 | 310 | element kernel | warm | 72 |
| jac_action | atomic / sympy_action | 4,121,204 | 308 | element kernel | warm | 72 |
| residual | store / sumfact | 4,121,204 | 2940 | element kernel | warm | 72 |
| residual | packed / sumfact | 4,121,204 | 2929 | element kernel | warm | 72 |
| residual | packed / sumfact | 1,098,500 | 2775 | element kernel | warm | 72 |
| residual | packed / current | 4,121,204 | 2205 | element kernel | warm | 72 |
| residual | store / current | 4,121,204 | 2194 | element kernel | warm | 72 |
| residual | store / sympy | 4,121,204 | 2142 | element kernel | warm | 72 |
| residual | packed / sympy | 4,121,204 | 2108 | element kernel | warm | 72 |
| residual | packed / sumfact | 4,121,204 | 1761 | partial | warm | 72 |
| residual | packed / sumfact | 1,098,500 | 1623 | solver operator | warm | 72 |
| residual | packed / sumfact | 4,121,204 | 1545 | solver operator | warm | 72 |
| residual | packed / sumfact | 4,121,204 | 1454 | partial | warm | 72 |
| residual | packed / sumfact | 4,121,204 | 1422 | solver operator | warm | 72 |
| residual | colored / sumfact | 4,121,204 | 1267 | element kernel | warm | 72 |
| residual | packed / sumfact | 4,121,204 | 1206 | solver operator | warm | 72 |
| residual | colored / current | 4,121,204 | 976 | element kernel | warm | 72 |
| residual | colored / sympy | 4,121,204 | 939 | element kernel | warm | 72 |
| residual | atomic / current | 4,121,204 | 888 | element kernel | warm | 72 |
| residual | atomic / sumfact | 4,121,204 | 887 | element kernel | warm | 72 |
| residual | packed / isoparam_simd / isoparam | 4,121,204 | 886 | element kernel | warm | 72 |
| residual | atomic / sympy | 4,121,204 | 861 | element kernel | warm | 72 |
| residual | atomic / sumfact | 4,121,204 | 700 | partial | warm | 72 |
| residual | atomic / sumfact | 4,121,204 | 681 | solver operator | warm | 72 |
| residual | atomic / sumfact | 4,121,204 | 480 | partial | warm | 72 |
| residual | atomic / sympy / isoparam | 4,121,204 | 461 | element kernel | warm | 72 |
| residual | atomic / current / isoparam | 4,121,204 | 413 | element kernel | warm | 72 |


## What the physics costs

The element kernel alone against the same kernel carrying what the solver
needs, one operation and one size at a time so that only the operator varies.
The bare-kernel row is the number the regression gate tracks and the one every
quoted figure has historically meant; the solver does not run it.

`spread` is how far apart that configuration's repeated measurements were, as a
percentage of the best. Two rows differ meaningfully only when the gap between
them is larger than that -- which is not true of every pair here, and saying so
is cheaper than inviting the reader to over-read a 3% difference.

### residual, 4,121,204 dof

| operator | terms carried | working set | MDOF/s | spread | cost vs bare kernel |
|---|---|---|---|---|---|
| element kernel | none | warm | 2929 | 2% of 6 | 1.00x |
| partial | Rhie-Chow, state ∇p (hoisted) | warm | 1761 | 1% of 3 | 1.66x |
| solver operator | Rhie-Chow, state ∇p (hoisted), boundary | warm | 1545 | 4% of 3 | 1.90x |
| partial | Rhie-Chow, state ∇p (per apply) | warm | 1454 | 1% of 3 | 2.01x |
| solver operator | Rhie-Chow, state ∇p (hoisted), boundary, transient | warm | 1422 | 10% of 3 | 2.06x |
| solver operator | Rhie-Chow, state ∇p (per apply), boundary, transient | warm | 1206 | 1% of 3 | 2.43x |

### jac_action, 4,121,204 dof

| operator | terms carried | working set | MDOF/s | spread | cost vs bare kernel |
|---|---|---|---|---|---|
| element kernel | none | warm | 2071 | 1% of 6 | 1.00x |
| partial | Rhie-Chow, exact-RC J, state ∇p (hoisted) | warm | 963 | 1% of 3 | 2.15x |
| solver operator | Rhie-Chow, exact-RC J, state ∇p (hoisted), boundary | warm | 892 | 1% of 3 | 2.32x |
| solver operator | Rhie-Chow, exact-RC J, state ∇p (hoisted), boundary, transient | warm | 860 | 0% of 3 | 2.41x |
| solver operator | Rhie-Chow, exact-RC J, state ∇p (hoisted), boundary | cold, 7 live | 845 | 2% of 3 | 2.45x |
| assembled matrix | SpMV of the assembled BSR — 3 term sets, 1.5% apart | warm | 458 | n=3 | 4.52x |


## What the headline number leaves out

The throughput this spike quotes is measured on an operator the solver never runs. That is
a deliberate and useful thing to measure — it isolates the element kernel from everything
around it, which is how a change to the arithmetic is seen without the rest of the solve
moving underneath it — but it is not the cost of an apply in the Newton loop.
`src/hex8/cvfem_hex8_ns_core.hpp:5` says the two families "differ in physics, not just in
layout"; the tables above are what that sentence costs.

**The residual's headline overstates the real operator by 2.11x**, 2912 MDOF/s against 1377
for the same kernel carrying Rhie–Chow, the boundary closure and the transient term. **For
the Jacobian action it is 2.43x**, 2018 against 830.

That second figure used to be 3.68x, and the difference is the point of the most recent
work rather than a change in the physics. The exact Rhie–Chow Jacobian differentiates
through the nodal pressure-gradient reconstruction, so every matvec rebuilds that
reconstruction for the Krylov direction — a term that cannot be hoisted out of anything,
because the direction changes with every iteration. That pass used to take 5.06 ms of a
7.34 ms matvec, 69% of it, for arithmetic that amounts to one small gradient per element.
It now takes 1.04 ms: the denominator it was recomputing every time is pure geometry and is
cached, and the sweep runs over packs with a ghost reduction instead of over the flat mesh
with 24 atomics per element. **The operator the solver's Krylov loop applies is 1.57x faster
as a result** — 576 to 906 MDOF/s at 8,586,756 dof, measured against the previous binary in
one allocation.

What is left of the exact term's cost is now in the element sweep rather than beside it:
turning Rhie–Chow on takes the action from 2018 to 936, and the boundary closure and the
transient term account for the rest. That is where a stored element tangent would act, and
it is the next thing worth measuring.

## Caching the nodal pressure gradient is worth much less than it was

1656 MDOF/s with the gradient hoisted out of the timed loop against 1351 with it rebuilt
inside every apply, on the packed residual — a factor of **1.23**. On the atomic layout,
where the reconstruction is still the flat atomic sweep, the same pair reads 618 against
452, a factor of 1.37.

Both of those used to be far larger: 2.41 and 1.60 on the same two configurations, and
`docs/README_alps.md` records the corresponding solver option (`SFEM_PGRAD_CACHE`) as worth
1.26x off the whole linear solve. Nothing about the caching changed. What changed is the
thing being cached: the reconstruction is 4.9x cheaper than it was, so avoiding it buys
proportionally less.

That is worth stating plainly because it inverts a recommendation. When a pass is slow,
hoisting it out of the inner loop is the obvious lever and it was the right one; once the
pass is fast, the hoist is a modest optimisation carrying a real cost — the cached gradient
has to be invalidated when the state changes, and a stale one is a wrong operator rather
than a slow one. The measurement that justified paying that price no longer says what it
said.

The general lesson is the one this spike keeps re-learning: an optimisation's value is a
property of the code around it, not of the optimisation, and it has to be re-measured after
anything nearby moves.

## The packed layout's advantage shrinks as the physics is added

Against the atomic layout at 4,121,204 dof:

| operator | packed | atomic | packed / atomic |
|---|---:|---:|---:|
| residual, element kernel only | 2912 | 826 | **3.53x** |
| + Rhie–Chow, gradient hoisted | 1656 | 618 | 2.68x |
| + Rhie–Chow + boundary | 1490 | 624 | 2.39x |
| Jacobian action, element kernel only | 1837 | 785 | 2.34x |
| + Rhie–Chow + boundary | 888 | 453 | 1.96x |

The packed layout is still the right choice — it wins in every row — but a layout
comparison made on the bare kernel overstates the margin by roughly 1.5 to 2. The reason is
structural rather than incidental: packing buys its advantage in the element sweep, through
SIMD over a pack and a ghost reduction in place of atomics, and most of what is added here
is either a separate pass over the mesh or arithmetic that vectorises less well.

One row deserves a caveat rather than a reading. With the gradient rebuilt inside every
apply the ratio reads 2.99x, *higher* than the bare kernel's — but that is an artefact of
where the reconstruction runs. It now sweeps over packs, and the benchmark builds no pack
for a plain `--layout atomic` residual, so the atomic side of that particular row is still
paying for the old flat atomic sweep while the packed side is not. The solver has a pack in
both cases and would not show the same gap.

The same caution applies to any two rows in the tables above: they are comparable only when
the completeness column matches, and only when the gap between them exceeds the `spread`
each was measured with.

## The other way to apply the Jacobian

An SpMV of the assembled BSR is the alternative to the matrix-free action, so it belongs in
the cascade rather than in a table of its own — the question "what does a matvec cost" has
two answers and only one of them is matrix-free.

**446 MDOF/s at 4,121,204 dof, against 888 for the matrix-free action carrying the same
physics.** Matrix-free wins by 1.99x on the apply alone, and that is before counting the
assembly that produced the matrix: 83 MDOF/s on the colored layout and 38 on the atomic
one, which is 5 and 12 SpMVs' worth of work respectively, spent to build something that is
then slower to apply than not building it at all. The assembled operator earns its place as
a preconditioner — it is what block-Jacobi and the Schur diagonal are read out of — and not
as an apply.

That margin was 1.25x until recently, and it widened without the SpMV moving at all: the
matrix-free operator got faster. Which is the shape of this comparison in general — the
SpMV's cost is set by the matrix and does not respond to anything done to the kernel, so
every improvement on the matrix-free side is a full improvement in the ratio.

**One row per problem size is enough, because the cost does not depend on the physics.**
The sparsity pattern is the mesh's node-to-node graph whatever terms the values carry, and
it stays that way by design: the exact Rhie–Chow term is deliberately kept out of the
assembled matrix precisely because it would couple pressures beyond nearest neighbours and
widen the pattern (`cvfem_hex8_ns_upwind_kernels.hpp:135`). That is an argument, though, not
a measurement, so the job measures it — bare, with Rhie–Chow, and with Rhie–Chow and the
boundary closure — and the three read 446, 445 and 444 MDOF/s, 0.5% apart. If they ever
stop agreeing, the pattern has widened and something more interesting than a throughput has
changed.

It barely varies with size — 447 MDOF/s at 1,098,500 dof against 446 at 4,121,204, inside
the spread of either. That is itself worth a sentence, because the obvious explanation for
any size dependence would be cache residency and it is not available here: the values are
about 840 bytes per dof at both sizes (878 MiB and 3329 MiB), so both matrices are far past
the 117 MiB of L3 and the SpMV streams from DRAM in both cases.

That 840 bytes per dof is the durable point about this operator. The matrix-free action
reads the state and the mesh and recomputes everything else, so it moves a small constant
per dof no matter how much physics it carries; the SpMV moves the whole matrix every time.
That is why the SpMV's cost is flat across the physics — the traffic is the same — and why
adding arithmetic to the matrix-free operator narrows the gap between them from 4.5x on
the bare kernel to 2.0x, without ever closing it.

## Provenance

| field | value |
|---|---|
| generated | 2026-09-11 10:44:43 |
| rows | 71 |
| machines | nid006567 |
| sources | kernels-det-4644425.csv |

Regenerate with `python3 python/cvfem_kernel_report.py <csv> -o docs/CVFEM_Kernels.md --html`.
