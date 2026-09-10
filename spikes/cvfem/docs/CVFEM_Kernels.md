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
| assemble | colored / sympy | 4,121,204 | 78 | element kernel | warm | 72 |
| assemble | colored / sympy_block | 4,121,204 | 77 | element kernel | warm | 72 |
| assemble | store / sumfact | 4,121,204 | 70 | element kernel | warm | 72 |
| assemble | store / sympy | 4,121,204 | 70 | element kernel | warm | 72 |
| assemble | store / sympy_block | 4,121,204 | 69 | element kernel | warm | 72 |
| assemble | atomic / sympy_block | 4,121,204 | 57 | element kernel | warm | 72 |
| assemble | atomic / sympy | 4,121,204 | 56 | element kernel | warm | 72 |
| assemble | packed / sumfact | 4,121,204 | 54 | element kernel | warm | 72 |
| assemble | packed / sympy | 4,121,204 | 54 | element kernel | warm | 72 |
| assemble | packed / sympy_block | 4,121,204 | 54 | element kernel | warm | 72 |
| assemble | packed / sumfact | 4,121,204 | 52 | partial | warm | 72 |
| assemble | atomic / sumfact | 4,121,204 | 39 | element kernel | warm | 72 |
| assemble | atomic / sumfact | 4,121,204 | 34 | frozen-RC | warm | 72 |
| assemble | atomic / sumfact | 4,121,204 | 34 | partial | warm | 72 |
| assemble | atomic / fd / isoparam | 4,121,204 | 21 | element kernel | warm | 72 |
| assemble | atomic / fd | 4,121,204 | 21 | element kernel | warm | 72 |
| assemble | atomic / split | 4,121,204 | 14 | element kernel | warm | 72 |
| assemble_diag | atomic / (kernel n/a) | 4,121,204 | 113 | element kernel | warm | 72 |
| assemble_diag | atomic / (kernel n/a) | 4,121,204 | 92 | frozen-RC | warm | 72 |
| assemble_diag | atomic / (kernel n/a) | 4,121,204 | 77 | partial | warm | 72 |
| assemble_diag | atomic / (kernel n/a) / isoparam | 4,121,204 | 68 | partial | warm | 72 |
| assemble_diag | atomic / (kernel n/a) | 4,121,204 | 63 | partial | warm | 72 |
| bsr_apply | packed / (kernel n/a) | 1,098,500 | 487 | element kernel | warm | 72 |
| bsr_apply | packed / (kernel n/a) | 4,121,204 | 462 | element kernel | warm | 72 |
| bsr_apply | packed / (kernel n/a) | 4,121,204 | 461 | frozen-RC | warm | 72 |
| bsr_apply | packed / (kernel n/a) | 4,121,204 | 457 | partial | warm | 72 |
| jac_action | store / (kernel n/a) | 4,121,204 | 2077 | element kernel | warm | 72 |
| jac_action | packed / (kernel n/a) | 4,121,204 | 2067 | element kernel | warm | 72 |
| jac_action | packed / (kernel n/a) | 1,098,500 | 2009 | element kernel | warm | 72 |
| jac_action | colored / (kernel n/a) | 4,121,204 | 910 | element kernel | warm | 72 |
| jac_action | atomic / (kernel n/a) | 4,121,204 | 835 | element kernel | warm | 72 |
| jac_action | packed / (kernel n/a) / isoparam | 4,121,204 | 809 | element kernel | warm | 72 |
| jac_action | packed / (kernel n/a) | 4,121,204 | 605 | partial | warm | 72 |
| jac_action | packed / (kernel n/a) | 4,121,204 | 579 | solver operator | warm | 72 |
| jac_action | packed / (kernel n/a) | 4,121,204 | 573 | solver operator | cold, 7 live | 72 |
| jac_action | packed / (kernel n/a) | 1,098,500 | 572 | solver operator | warm | 72 |
| jac_action | packed / (kernel n/a) | 4,121,204 | 561 | solver operator | warm | 72 |
| jac_action | atomic / sympy_action_geomface | 4,121,204 | 475 | element kernel | warm | 72 |
| jac_action | atomic / sympy_action_face | 4,121,204 | 422 | element kernel | warm | 72 |
| jac_action | atomic / (kernel n/a) / isoparam | 4,121,204 | 386 | element kernel | warm | 72 |
| jac_action | atomic / (kernel n/a) | 4,121,204 | 376 | solver operator | warm | 72 |
| jac_action | atomic / sympy_action_geom | 4,121,204 | 353 | element kernel | warm | 72 |
| jac_action | atomic / sympy_action_node | 4,121,204 | 333 | element kernel | warm | 72 |
| jac_action | atomic / sympy_action_comp | 4,121,204 | 328 | element kernel | warm | 72 |
| jac_action | atomic / sympy_action | 4,121,204 | 307 | element kernel | warm | 72 |
| residual | packed / sumfact | 4,121,204 | 2945 | element kernel | warm | 72 |
| residual | store / sumfact | 4,121,204 | 2940 | element kernel | warm | 72 |
| residual | packed / sumfact | 1,098,500 | 2745 | element kernel | warm | 72 |
| residual | packed / current | 4,121,204 | 2212 | element kernel | warm | 72 |
| residual | store / current | 4,121,204 | 2211 | element kernel | warm | 72 |
| residual | packed / sympy | 4,121,204 | 2147 | element kernel | warm | 72 |
| residual | store / sympy | 4,121,204 | 2142 | element kernel | warm | 72 |
| residual | packed / sumfact | 4,121,204 | 1759 | partial | warm | 72 |
| residual | packed / sumfact | 4,121,204 | 1559 | solver operator | warm | 72 |
| residual | packed / sumfact | 1,098,500 | 1516 | solver operator | warm | 72 |
| residual | packed / sumfact | 4,121,204 | 1436 | solver operator | warm | 72 |
| residual | colored / sumfact | 4,121,204 | 1283 | element kernel | warm | 72 |
| residual | colored / current | 4,121,204 | 983 | element kernel | warm | 72 |
| residual | colored / sympy | 4,121,204 | 949 | element kernel | warm | 72 |
| residual | atomic / current | 4,121,204 | 901 | element kernel | warm | 72 |
| residual | packed / isoparam_simd / isoparam | 4,121,204 | 884 | element kernel | warm | 72 |
| residual | atomic / sumfact | 4,121,204 | 883 | element kernel | warm | 72 |
| residual | atomic / sympy | 4,121,204 | 861 | element kernel | warm | 72 |
| residual | packed / sumfact | 4,121,204 | 729 | partial | warm | 72 |
| residual | atomic / sumfact | 4,121,204 | 703 | partial | warm | 72 |
| residual | atomic / sumfact | 4,121,204 | 675 | solver operator | warm | 72 |
| residual | packed / sumfact | 4,121,204 | 667 | solver operator | warm | 72 |
| residual | atomic / sympy / isoparam | 4,121,204 | 463 | element kernel | warm | 72 |
| residual | atomic / sumfact | 4,121,204 | 439 | partial | warm | 72 |
| residual | atomic / current / isoparam | 4,121,204 | 412 | element kernel | warm | 72 |


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
| element kernel | none | warm | 2945 | 2% of 6 | 1.00x |
| partial | Rhie-Chow, state ∇p (hoisted) | warm | 1759 | 1% of 3 | 1.67x |
| solver operator | Rhie-Chow, state ∇p (hoisted), boundary | warm | 1559 | 0% of 3 | 1.89x |
| solver operator | Rhie-Chow, state ∇p (hoisted), boundary, transient | warm | 1436 | 3% of 3 | 2.05x |
| partial | Rhie-Chow, state ∇p (per apply) | warm | 729 | 1% of 3 | 4.04x |
| solver operator | Rhie-Chow, state ∇p (per apply), boundary, transient | warm | 667 | 1% of 3 | 4.41x |

### jac_action, 4,121,204 dof

| operator | terms carried | working set | MDOF/s | spread | cost vs bare kernel |
|---|---|---|---|---|---|
| element kernel | none | warm | 2067 | 8% of 6 | 1.00x |
| partial | Rhie-Chow, exact-RC J, state ∇p (hoisted) | warm | 605 | 1% of 3 | 3.42x |
| solver operator | Rhie-Chow, exact-RC J, state ∇p (hoisted), boundary | warm | 579 | 1% of 3 | 3.57x |
| solver operator | Rhie-Chow, exact-RC J, state ∇p (hoisted), boundary | cold, 7 live | 573 | 1% of 3 | 3.60x |
| solver operator | Rhie-Chow, exact-RC J, state ∇p (hoisted), boundary, transient | warm | 561 | 1% of 3 | 3.68x |
| assembled matrix | SpMV of the assembled BSR — 3 term sets, 1.1% apart | warm | 462 | n=3 | 4.47x |


## What the headline number leaves out

The throughput this spike quotes is measured on an operator the solver never runs. That is
a deliberate and useful thing to measure — it isolates the element kernel from everything
around it, which is how a change to the arithmetic is seen without the rest of the solve
moving underneath it — but it is not the cost of an apply in the Newton loop.
`src/hex8/cvfem_hex8_ns_core.hpp:5` says the two families "differ in physics, not just in
layout"; the tables above are what that sentence costs.

**The residual's headline overstates the real operator by 2.05x.** 2945 MDOF/s is the
element kernel on the packed layout; the same kernel carrying Rhie–Chow, the boundary
closure and the transient term is 1436. Quoting the first as the operator's throughput is
not wrong, but it needs the qualifier, and this report is what the qualifier should point
at.

**For the Jacobian action the factor is 3.68x, not 2.05x**, and that asymmetry is the thing
worth taking away. 2066 bare against 561 for the operator the Krylov loop evaluates. The
action pays for something the residual has no analogue of: the *exact* Rhie–Chow Jacobian
differentiates through the nodal pressure-gradient reconstruction, so every matvec rebuilds
that reconstruction for the Krylov direction — a full element sweep that cannot be hoisted
out of anything, because the direction changes with every iteration. Turning the term on
alone takes the action from 2066 to 605; everything after that is comparatively cheap.

An earlier version of this analysis reported ~1.9x and measured the residual only, because
no benchmark row for the action with Rhie–Chow existed to measure. It does now, and the
number it produces is nearly twice as large. The residual figure it gave, 2579 against
1377, agrees with the 1.89x this run reads for the same two rows.

## Caching the nodal pressure gradient is worth 2.4x on the apply

1759 MDOF/s with the gradient hoisted out of the timed loop against 729 with it rebuilt
inside every apply, on the packed residual — a factor of 2.41. That is the apply alone;
`docs/README_alps.md` records the same option (`SFEM_PGRAD_CACHE`) as 1.26x off the whole
linear solve, which is consistent: a solve is more than its applies. Anything that forces
the gradient to be rebuilt per apply gives back more than half the operator.

The effect is a layout property as much as a physics one: on the atomic layout the same
pair reads 703 against 439, a factor of 1.60. The cache is worth most exactly where the
element sweep is fastest, because it is a fixed extra sweep and the kernel it is added to
is what varies.

This is the one term in the cascade that is a scheduling decision rather than a
discretisation choice. Rhie–Chow, the boundary closure and the transient term are all
things the operator either has or does not; where the gradient is computed is free to
choose, and choosing wrongly is the largest single number in this report.

## The packed layout's advantage shrinks as the physics is added

Against the atomic layout, on the residual at 4,121,204 dof:

| operator | packed | atomic | packed / atomic |
|---|---:|---:|---:|
| element kernel only | 2945 | 881 | **3.34x** |
| + Rhie–Chow, gradient hoisted | 1759 | 703 | 2.50x |
| + Rhie–Chow + boundary | 1559 | 675 | 2.31x |
| + Rhie–Chow, gradient per apply | 729 | 439 | 1.66x |

and on the Jacobian action, 2.47x on the bare kernel (2066 against 835) against 1.54x with
Rhie–Chow and the boundary closure (579 against 376).

The packed layout is still the right choice — it wins in every row — but a layout
comparison made on the bare kernel overstates the margin by about two. The reason is
structural rather than incidental: packing buys its advantage in the element sweep, through
SIMD over a pack and a ghost reduction in place of atomics, and every term added here is
either a separate pass over the mesh (the boundary closure, the transient term, the
gradient reconstruction) or arithmetic that does not vectorise as well (the Rhie–Chow
coefficient). None of those is helped by the pack, so each one dilutes what the pack is
for.

The same caution applies to any two rows in the tables above: they are comparable only when
the completeness column matches, and only when the gap between them exceeds the `spread`
each was measured with.

## The other way to apply the Jacobian

An SpMV of the assembled BSR is the alternative to the matrix-free action, so it belongs in
the cascade rather than in a table of its own — the question "what does a matvec cost" has
two answers and only one of them is matrix-free.

**462 MDOF/s at 4,121,204 dof, against 579 for the matrix-free action carrying the same
physics.** Matrix-free wins by 1.25x on the apply alone, and that is before counting the
assembly that produced the matrix: 87 MDOF/s on the colored layout and 39 on the atomic
one, which is 5 and 12 SpMVs' worth of work respectively, spent to build something that is
then slower to apply than not building it at all. The assembled operator earns its place as
a preconditioner — it is what block-Jacobi and the Schur diagonal are read out of — and not
as an apply.

**One row per problem size is enough, because the cost does not depend on the physics.**
The sparsity pattern is the mesh's node-to-node graph whatever terms the values carry, and
it stays that way by design: the exact Rhie–Chow term is deliberately kept out of the
assembled matrix precisely because it would couple pressures beyond nearest neighbours and
widen the pattern (`cvfem_hex8_ns_upwind_kernels.hpp:135`). That is an argument, though, not
a measurement, so the job measures it — bare, with Rhie–Chow, and with Rhie–Chow and the
boundary closure — and the three read 462, 461 and 457 MDOF/s, 1.1% apart. If they ever
stop agreeing, the pattern has widened and something more interesting than a throughput has
changed.

It varies mildly with size — 487 MDOF/s at 1,098,500 dof against 462 at 4,121,204 — and
that gap is 5%, only just outside the 3% these readings spread over, so it is worth
noticing and not worth explaining. What can be said is what it is *not*: the values are
about 840 bytes per dof at both sizes (878 MiB and 3329 MiB), so both matrices are far past
the 117 MiB of L3 and the SpMV is streaming from DRAM in both cases. Whatever the remaining
5% is, it is not a cache-residency effect.

That 840 bytes per dof is the durable point about this operator. The matrix-free action
reads the state and the mesh and recomputes everything else, so it moves a small constant
per dof no matter how much physics it carries; the SpMV moves the whole matrix every time.
That is why the SpMV's cost is flat across the physics — the traffic is the same — and why
adding arithmetic to the matrix-free operator narrowed the gap between them from 4.5x on
the bare kernel to 1.25x, without ever closing it.

## Provenance

| field | value |
|---|---|
| generated | 2026-09-10 14:02:39 |
| rows | 71 |
| machines | nid006548 |
| sources | kernels_grace.csv |

Regenerate with `python3 python/cvfem_kernel_report.py <csv> -o docs/CVFEM_Kernels.md --html`.
