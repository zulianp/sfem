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
| assemble | colored / sympy_block | 4,121,204 | 76 | element kernel | warm | 72 |
| assemble | store / sumfact | 4,121,204 | 72 | element kernel | warm | 72 |
| assemble | store / sympy | 4,121,204 | 71 | element kernel | warm | 72 |
| assemble | store / sympy_block | 4,121,204 | 71 | element kernel | warm | 72 |
| assemble | atomic / sympy_block | 4,121,204 | 57 | element kernel | warm | 72 |
| assemble | atomic / sympy | 4,121,204 | 56 | element kernel | warm | 72 |
| assemble | packed / sumfact | 4,121,204 | 55 | element kernel | warm | 72 |
| assemble | packed / sympy | 4,121,204 | 55 | element kernel | warm | 72 |
| assemble | packed / sumfact | 4,121,204 | 52 | partial | warm | 72 |
| assemble | atomic / sumfact | 4,121,204 | 39 | element kernel | warm | 72 |
| assemble | atomic / sumfact | 4,121,204 | 34 | frozen-RC | warm | 72 |
| assemble | atomic / sumfact | 4,121,204 | 33 | partial | warm | 72 |
| assemble | atomic / fd / isoparam | 4,121,204 | 21 | element kernel | warm | 72 |
| assemble | atomic / fd | 4,121,204 | 21 | element kernel | warm | 72 |
| assemble | packed / sympy_block | 4,121,204 | 20 | element kernel | warm | 72 |
| assemble | atomic / split | 4,121,204 | 14 | element kernel | warm | 72 |
| assemble_diag | atomic / (kernel n/a) | 4,121,204 | 113 | element kernel | warm | 72 |
| assemble_diag | atomic / (kernel n/a) | 4,121,204 | 93 | frozen-RC | warm | 72 |
| assemble_diag | atomic / (kernel n/a) | 4,121,204 | 92 | partial | warm | 72 |
| assemble_diag | atomic / (kernel n/a) | 4,121,204 | 90 | partial | warm | 72 |
| assemble_diag | atomic / (kernel n/a) / isoparam | 4,121,204 | 68 | partial | warm | 72 |
| bsr_apply | packed / (kernel n/a) | 4,121,204 | 449 | element kernel | warm | 72 |
| jac_action | store / (kernel n/a) | 4,121,204 | 2044 | element kernel | warm | 72 |
| jac_action | packed / (kernel n/a) | 4,121,204 | 2036 | element kernel | warm | 72 |
| jac_action | packed / (kernel n/a) | 1,098,500 | 1991 | element kernel | warm | 72 |
| jac_action | colored / (kernel n/a) | 4,121,204 | 925 | element kernel | warm | 72 |
| jac_action | atomic / (kernel n/a) | 4,121,204 | 840 | element kernel | warm | 72 |
| jac_action | packed / (kernel n/a) / isoparam | 4,121,204 | 807 | element kernel | warm | 72 |
| jac_action | packed / (kernel n/a) | 4,121,204 | 618 | partial | warm | 72 |
| jac_action | packed / (kernel n/a) | 4,121,204 | 584 | solver operator | warm | 72 |
| jac_action | packed / (kernel n/a) | 4,121,204 | 575 | solver operator | cold, 7 live | 72 |
| jac_action | packed / (kernel n/a) | 4,121,204 | 571 | solver operator | warm | 72 |
| jac_action | packed / (kernel n/a) | 1,098,500 | 570 | solver operator | warm | 72 |
| jac_action | atomic / sympy_action_geomface | 4,121,204 | 479 | element kernel | warm | 72 |
| jac_action | atomic / sympy_action_face | 4,121,204 | 429 | element kernel | warm | 72 |
| jac_action | atomic / (kernel n/a) / isoparam | 4,121,204 | 398 | element kernel | warm | 72 |
| jac_action | atomic / (kernel n/a) | 4,121,204 | 376 | solver operator | warm | 72 |
| jac_action | atomic / sympy_action_geom | 4,121,204 | 360 | element kernel | warm | 72 |
| jac_action | atomic / sympy_action_node | 4,121,204 | 335 | element kernel | warm | 72 |
| jac_action | atomic / sympy_action_comp | 4,121,204 | 329 | element kernel | warm | 72 |
| jac_action | atomic / sympy_action | 4,121,204 | 309 | element kernel | warm | 72 |
| residual | store / sumfact | 4,121,204 | 2931 | element kernel | warm | 72 |
| residual | packed / sumfact | 4,121,204 | 2916 | element kernel | warm | 72 |
| residual | packed / sumfact | 1,098,500 | 2664 | element kernel | warm | 72 |
| residual | packed / current | 4,121,204 | 2232 | element kernel | warm | 72 |
| residual | store / current | 4,121,204 | 2193 | element kernel | warm | 72 |
| residual | packed / sympy | 4,121,204 | 2132 | element kernel | warm | 72 |
| residual | store / sympy | 4,121,204 | 2116 | element kernel | warm | 72 |
| residual | packed / sumfact | 4,121,204 | 1759 | partial | warm | 72 |
| residual | packed / sumfact | 4,121,204 | 1556 | solver operator | warm | 72 |
| residual | packed / sumfact | 1,098,500 | 1490 | solver operator | warm | 72 |
| residual | packed / sumfact | 4,121,204 | 1445 | solver operator | warm | 72 |
| residual | colored / sumfact | 4,121,204 | 1285 | element kernel | warm | 72 |
| residual | colored / current | 4,121,204 | 990 | element kernel | warm | 72 |
| residual | colored / sympy | 4,121,204 | 950 | element kernel | warm | 72 |
| residual | atomic / current | 4,121,204 | 896 | element kernel | warm | 72 |
| residual | packed / isoparam_simd / isoparam | 4,121,204 | 888 | element kernel | warm | 72 |
| residual | atomic / sumfact | 4,121,204 | 881 | element kernel | warm | 72 |
| residual | atomic / sympy | 4,121,204 | 861 | element kernel | warm | 72 |
| residual | packed / sumfact | 4,121,204 | 741 | partial | warm | 72 |
| residual | atomic / sumfact | 4,121,204 | 706 | partial | warm | 72 |
| residual | atomic / sumfact | 4,121,204 | 677 | solver operator | warm | 72 |
| residual | packed / sumfact | 4,121,204 | 585 | solver operator | warm | 72 |
| residual | atomic / sympy / isoparam | 4,121,204 | 463 | element kernel | warm | 72 |
| residual | atomic / sumfact | 4,121,204 | 442 | partial | warm | 72 |
| residual | atomic / current / isoparam | 4,121,204 | 401 | element kernel | warm | 72 |


## What the physics costs

The element kernel alone against the same kernel carrying what the solver
needs, one operation and one size at a time so that only the operator varies.
The bare-kernel row is the number the regression gate tracks and the one every
quoted figure has historically meant; the solver does not run it.

### residual, 4,121,204 dof

| operator | terms carried | working set | MDOF/s | cost vs bare kernel |
|---|---|---|---|---|
| element kernel | none | warm | 2916 | 1.00x |
| partial | Rhie-Chow, state ∇p (hoisted) | warm | 1759 | 1.66x |
| solver operator | Rhie-Chow, state ∇p (hoisted), boundary | warm | 1556 | 1.87x |
| solver operator | Rhie-Chow, state ∇p (hoisted), boundary, transient | warm | 1445 | 2.02x |
| partial | Rhie-Chow, state ∇p (per apply) | warm | 741 | 3.94x |
| solver operator | Rhie-Chow, state ∇p (per apply), boundary, transient | warm | 585 | 4.98x |

### jac_action, 4,121,204 dof

| operator | terms carried | working set | MDOF/s | cost vs bare kernel |
|---|---|---|---|---|
| element kernel | none | warm | 2036 | 1.00x |
| partial | Rhie-Chow, exact-RC J, state ∇p (hoisted) | warm | 618 | 3.29x |
| solver operator | Rhie-Chow, exact-RC J, state ∇p (hoisted), boundary | warm | 584 | 3.49x |
| solver operator | Rhie-Chow, exact-RC J, state ∇p (hoisted), boundary | cold, 7 live | 575 | 3.54x |
| solver operator | Rhie-Chow, exact-RC J, state ∇p (hoisted), boundary, transient | warm | 571 | 3.57x |


## What the headline number leaves out

The throughput this spike quotes is measured on an operator the solver never runs. That is
a deliberate and useful thing to measure — it isolates the element kernel from everything
around it, which is how a change to the arithmetic is seen without the rest of the solve
moving underneath it — but it is not the cost of an apply in the Newton loop.
`src/hex8/cvfem_hex8_ns_core.hpp:5` says the two families "differ in physics, not just in
layout"; the tables above are what that sentence costs.

**The residual's headline overstates the real operator by 2.0x.** 2916 MDOF/s is the
element kernel on the packed layout; the same kernel carrying Rhie–Chow, the boundary
closure and the transient term is 1445. Quoting the first as the operator's throughput is
not wrong, but it needs the qualifier, and this report is what the qualifier should point
at.

**For the Jacobian action the factor is 3.6x, not 2.0x**, and that asymmetry is the thing
worth taking away. 2032 bare against 571 for the operator the Krylov loop evaluates. The
action pays for something the residual has no analogue of: the *exact* Rhie–Chow Jacobian
differentiates through the nodal pressure-gradient reconstruction, so every matvec rebuilds
that reconstruction for the Krylov direction — a full element sweep that cannot be hoisted
out of anything, because the direction changes with every iteration. Turning the term on
alone takes the action from 2032 to 618; everything after that is comparatively cheap.

An earlier version of this analysis reported ~1.9x and measured the residual only, because
no benchmark row for the action with Rhie–Chow existed to measure. It does now, and the
number it produces is nearly twice as large. The residual figure it gave, 2579 against
1377, agrees with the 1.87x this run reads for the same two rows.

**Where the assembled matrix sits.** The SpMV of the assembled Jacobian runs at 449 MDOF/s
against 584 for the matrix-free action carrying the same physics, so matrix-free wins by
1.3x on the apply alone — before counting the assembly that produced the matrix, which
costs 39 MDOF/s on the atomic layout. The assembled operator earns its place as a
preconditioner, not as an apply.

## Caching the nodal pressure gradient is worth 2.4x on the apply

1759 MDOF/s with the gradient hoisted out of the timed loop against 741 with it rebuilt
inside every apply, on the packed residual. That is the apply alone;
`docs/README_alps.md` records the same option (`SFEM_PGRAD_CACHE`) as 1.26x off the whole
linear solve, which is consistent — a solve is more than its applies. Anything that forces
the gradient to be rebuilt per apply gives back more than half the operator.

The effect is a layout property as much as a physics one: on the atomic layout the same
pair reads 706 against 442, a factor of 1.60. The cache is worth most exactly where the
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
| element kernel only | 2916 | 881 | **3.31x** |
| + Rhie–Chow, gradient hoisted | 1759 | 706 | 2.49x |
| + Rhie–Chow + boundary | 1556 | 677 | 2.30x |
| + Rhie–Chow, gradient per apply | 741 | 442 | 1.68x |

and on the Jacobian action, 2.42x on the bare kernel against 1.55x with Rhie–Chow and the
boundary closure.

The packed layout is still the right choice — it wins in every row — but a layout
comparison made on the bare kernel overstates the margin by about two. The reason is
structural rather than incidental: packing buys its advantage in the element sweep, through
SIMD over a pack and a ghost reduction in place of atomics, and every term added here is
either a separate pass over the mesh (the boundary closure, the transient term, the
gradient reconstruction) or arithmetic that does not vectorise as well (the Rhie–Chow
coefficient). None of those is helped by the pack, so each one dilutes what the pack is
for.

The same caution applies to any two rows in the tables above: they are comparable only when
the completeness column matches.

## Provenance

| field | value |
|---|---|
| generated | 2026-09-10 12:48:01 |
| rows | 68 |
| machines | nid006547 |
| sources | kernels_grace.csv |

Regenerate with `python3 python/cvfem_kernel_report.py <csv> -o docs/CVFEM_Kernels.md --html`.
