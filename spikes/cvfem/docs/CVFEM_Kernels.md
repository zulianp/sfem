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
| assemble | atomic / isoparam_generated / isoparam / bsr f64 | – | – | – | – | – | – | element kernel |
| assemble | atomic / sumfact / bsr f64 | frozen by design | frozen by design | hoisted | carried | – | – | partial |
| assemble | atomic / sumfact / bsr f64 | frozen by design | frozen by design | hoisted | – | – | – | frozen-RC |
| assemble | atomic / sumfact / bsr f64 | – | – | – | – | – | – | element kernel |
| assemble | colored / sumfact / bsr f64 | – | – | – | – | – | – | element kernel |
| assemble | packed / sumfact / bsr f64 | frozen by design | frozen by design | hoisted | carried | – | – | partial |
| assemble | packed / sumfact / bsr f64 | – | – | – | – | – | – | element kernel |
| assemble | store / sumfact / bsr f64 | – | – | – | – | – | – | element kernel |
| assemble_diag | atomic / isoparam_handwritten / isoparam | frozen by design | frozen by design | hoisted | carried | – | – | partial |
| assemble_diag | atomic / sumfact | frozen by design | frozen by design | hoisted | carried | – | carried | partial |
| assemble_diag | atomic / sumfact | frozen by design | frozen by design | hoisted | carried | – | – | partial |
| assemble_diag | atomic / sumfact | frozen by design | frozen by design | hoisted | – | – | – | frozen-RC |
| assemble_diag | atomic / sumfact | – | – | – | – | – | – | element kernel |
| bsr_apply | n/a / (kernel n/a) / bsr f64 | frozen by design | frozen by design | hoisted | carried | – | – | partial |
| bsr_apply | n/a / (kernel n/a) / bsr f64 | frozen by design | frozen by design | hoisted | – | – | – | frozen-RC |
| bsr_apply | n/a / (kernel n/a) / bsr f64 | – | – | – | – | – | – | element kernel |
| jac_action | atomic / isoparam_generated / isoparam | – | – | – | – | – | – | element kernel |
| jac_action | atomic / sumfact | carried | carried | hoisted | carried | – | – | solver operator |
| jac_action | atomic / sumfact | – | – | – | – | – | – | element kernel |
| jac_action | colored / sumfact | – | – | – | – | – | – | element kernel |
| jac_action | packed / isoparam_simd / isoparam | – | – | – | – | – | – | element kernel |
| jac_action | packed / sumfact | carried | carried | hoisted | carried | – | carried | solver operator |
| jac_action | packed / sumfact | carried | carried | hoisted | carried | – | – | solver operator |
| jac_action | packed / sumfact | carried | carried | hoisted | – | – | – | partial |
| jac_action | packed / sumfact | – | – | – | – | – | – | element kernel |
| residual | atomic / isoparam_generated / isoparam | – | – | – | – | – | – | element kernel |
| residual | atomic / sumfact | carried | – | hoisted | carried | – | – | solver operator |
| residual | atomic / sumfact | carried | – | hoisted | – | – | – | partial |
| residual | atomic / sumfact | carried | – | per apply | – | – | – | partial |
| residual | atomic / sumfact | – | – | – | – | – | – | element kernel |
| residual | colored / sumfact | – | – | – | – | – | – | element kernel |
| residual | packed / isoparam_simd / isoparam | – | – | – | – | – | – | element kernel |
| residual | packed / sumfact | carried | – | hoisted | carried | – | carried | solver operator |
| residual | packed / sumfact | carried | – | hoisted | carried | – | – | solver operator |
| residual | packed / sumfact | carried | – | hoisted | – | – | – | partial |
| residual | packed / sumfact | carried | – | per apply | carried | – | carried | solver operator |
| residual | packed / sumfact | carried | – | per apply | – | – | – | partial |
| residual | packed / sumfact | – | – | – | – | – | – | element kernel |


## Throughput

Best observed rate per configuration, in MDOF/s. Best rather than mean because
background load can only make a run slower. The completeness column is the one
from the table above: two rows are comparable only when it matches.

| operation | variant | dof | MDOF/s | completeness | working set | threads |
|---|---|---|---|---|---|---|
| assemble | colored / sumfact / bsr f64 | 4,121,204 | 103 | element kernel | warm | 72 |
| assemble | store / sumfact / bsr f64 | 4,121,204 | 88 | element kernel | warm | 72 |
| assemble | packed / sumfact / bsr f64 | 4,121,204 | 64 | element kernel | warm | 72 |
| assemble | packed / sumfact / bsr f64 | 4,121,204 | 60 | partial | warm | 72 |
| assemble | atomic / isoparam_generated / isoparam / bsr f64 | 4,121,204 | 46 | element kernel | warm | 72 |
| assemble | atomic / sumfact / bsr f64 | 4,121,204 | 39 | element kernel | warm | 72 |
| assemble | atomic / sumfact / bsr f64 | 4,121,204 | 34 | frozen-RC | warm | 72 |
| assemble | atomic / sumfact / bsr f64 | 4,121,204 | 33 | partial | warm | 72 |
| assemble_diag | atomic / sumfact | 4,121,204 | 112 | element kernel | warm | 72 |
| assemble_diag | atomic / sumfact | 4,121,204 | 89 | frozen-RC | warm | 72 |
| assemble_diag | atomic / sumfact | 4,121,204 | 88 | partial | warm | 72 |
| assemble_diag | atomic / sumfact | 4,121,204 | 86 | partial | warm | 72 |
| assemble_diag | atomic / isoparam_handwritten / isoparam | 4,121,204 | 66 | partial | warm | 72 |
| bsr_apply | n/a / (kernel n/a) / bsr f64 | 1,098,500 | 485 | element kernel | warm | 72 |
| bsr_apply | n/a / (kernel n/a) / bsr f64 | 4,121,204 | 455 | element kernel | warm | 72 |
| bsr_apply | n/a / (kernel n/a) / bsr f64 | 4,121,204 | 450 | frozen-RC | warm | 72 |
| bsr_apply | n/a / (kernel n/a) / bsr f64 | 4,121,204 | 448 | partial | warm | 72 |
| jac_action | packed / sumfact | 1,098,500 | 2119 | element kernel | warm | 72 |
| jac_action | packed / sumfact | 4,121,204 | 2028 | element kernel | warm | 72 |
| jac_action | colored / sumfact | 4,121,204 | 1416 | element kernel | warm | 72 |
| jac_action | packed / sumfact | 1,098,500 | 1053 | solver operator | warm | 72 |
| jac_action | atomic / sumfact | 4,121,204 | 962 | element kernel | warm | 72 |
| jac_action | packed / sumfact | 4,121,204 | 935 | partial | warm | 72 |
| jac_action | packed / sumfact | 4,121,204 | 862 | solver operator | warm | 72 |
| jac_action | packed / sumfact | 4,121,204 | 831 | solver operator | warm | 72 |
| jac_action | packed / sumfact | 4,121,204 | 816 | solver operator | cold, 7 live | 72 |
| jac_action | packed / isoparam_simd / isoparam | 4,121,204 | 795 | element kernel | warm | 72 |
| jac_action | atomic / sumfact | 4,121,204 | 408 | solver operator | warm | 72 |
| jac_action | atomic / isoparam_generated / isoparam | 4,121,204 | 390 | element kernel | warm | 72 |
| residual | packed / sumfact | 1,098,500 | 2892 | element kernel | warm | 72 |
| residual | packed / sumfact | 4,121,204 | 2838 | element kernel | warm | 72 |
| residual | colored / sumfact | 4,121,204 | 2013 | element kernel | warm | 72 |
| residual | packed / sumfact | 4,121,204 | 1821 | partial | warm | 72 |
| residual | packed / sumfact | 1,098,500 | 1747 | solver operator | warm | 72 |
| residual | packed / sumfact | 4,121,204 | 1637 | solver operator | warm | 72 |
| residual | packed / sumfact | 4,121,204 | 1523 | solver operator | warm | 72 |
| residual | packed / sumfact | 4,121,204 | 1492 | partial | warm | 72 |
| residual | packed / sumfact | 4,121,204 | 1265 | solver operator | warm | 72 |
| residual | atomic / sumfact | 4,121,204 | 1125 | element kernel | warm | 72 |
| residual | atomic / sumfact | 4,121,204 | 923 | partial | warm | 72 |
| residual | packed / isoparam_simd / isoparam | 4,121,204 | 893 | element kernel | warm | 72 |
| residual | atomic / sumfact | 4,121,204 | 885 | solver operator | warm | 72 |
| residual | atomic / sumfact | 4,121,204 | 618 | partial | warm | 72 |
| residual | atomic / isoparam_generated / isoparam | 4,121,204 | 447 | element kernel | warm | 72 |


## What the physics costs

The element kernel alone against the same kernel carrying what the solver
needs, one operation and one size at a time so that only the operator varies.
The bare-kernel row is the number the regression gate tracks and the one every
quoted figure has historically meant; the solver does not run it.

`spread` is how far apart that configuration's repeated measurements were, as a
percentage of the best. Two rows differ meaningfully only when the gap between
them is larger than that -- which is not true of every pair here, and saying so
is cheaper than inviting the reader to over-read a 3% difference.

### residual, packed layout, 4,121,204 dof

| operator | terms carried | working set | MDOF/s | spread | cost vs bare kernel |
|---|---|---|---|---|---|
| element kernel | none | warm | 2838 | 4% of 9 | 1.00x |
| partial | Rhie-Chow, state ∇p (hoisted) | warm | 1821 | 2% of 3 | 1.56x |
| solver operator | Rhie-Chow, state ∇p (hoisted), boundary | warm | 1637 | 0% of 3 | 1.73x |
| solver operator | Rhie-Chow, state ∇p (hoisted), boundary, transient | warm | 1523 | 1% of 3 | 1.86x |
| partial | Rhie-Chow, state ∇p (per apply) | warm | 1492 | 0% of 3 | 1.90x |
| solver operator | Rhie-Chow, state ∇p (per apply), boundary, transient | warm | 1265 | 1% of 3 | 2.24x |

### residual, atomic layout, 4,121,204 dof

| operator | terms carried | working set | MDOF/s | spread | cost vs bare kernel |
|---|---|---|---|---|---|
| element kernel | none | warm | 1125 | 21% of 6 | 1.00x |
| partial | Rhie-Chow, state ∇p (hoisted) | warm | 923 | 9% of 3 | 1.22x |
| solver operator | Rhie-Chow, state ∇p (hoisted), boundary | warm | 885 | 1% of 3 | 1.27x |
| partial | Rhie-Chow, state ∇p (per apply) | warm | 618 | 1% of 3 | 1.82x |

### residual, colored layout, 4,121,204 dof

| operator | terms carried | working set | MDOF/s | spread | cost vs bare kernel |
|---|---|---|---|---|---|
| element kernel | none | warm | 2013 | 0% of 3 | 1.00x |

### jac_action, packed layout, 4,121,204 dof

| operator | terms carried | working set | MDOF/s | spread | cost vs bare kernel |
|---|---|---|---|---|---|
| element kernel | none | warm | 2028 | 14% of 9 | 1.00x |
| partial | Rhie-Chow, exact-RC J, state ∇p (hoisted) | warm | 935 | 1% of 3 | 2.17x |
| solver operator | Rhie-Chow, exact-RC J, state ∇p (hoisted), boundary | warm | 862 | 1% of 3 | 2.35x |
| solver operator | Rhie-Chow, exact-RC J, state ∇p (hoisted), boundary, transient | warm | 831 | 1% of 3 | 2.44x |
| solver operator | Rhie-Chow, exact-RC J, state ∇p (hoisted), boundary | cold, 7 live | 816 | 2% of 3 | 2.48x |
| assembled matrix | SpMV of the assembled BSR, f64 values — 3 term sets, 1.5% apart | warm | 455 | n=3 | 4.46x |

### jac_action, atomic layout, 4,121,204 dof

| operator | terms carried | working set | MDOF/s | spread | cost vs bare kernel |
|---|---|---|---|---|---|
| element kernel | none | warm | 962 | 0% of 3 | 1.00x |
| solver operator | Rhie-Chow, exact-RC J, state ∇p (hoisted), boundary | warm | 408 | 4% of 3 | 2.36x |
| assembled matrix | SpMV of the assembled BSR, f64 values — 3 term sets, 1.5% apart | warm | 455 | n=3 | 2.12x |

### jac_action, colored layout, 4,121,204 dof

| operator | terms carried | working set | MDOF/s | spread | cost vs bare kernel |
|---|---|---|---|---|---|
| element kernel | none | warm | 1416 | 1% of 3 | 1.00x |
| assembled matrix | SpMV of the assembled BSR, f64 values — 3 term sets, 1.5% apart | warm | 455 | n=3 | 3.11x |


## Standard against packed

One operator per row, measured on both layouts. The ratio is packed over the
standard layout, so above 1.00x the packed sweep is the faster of the two.

`SpMV f64` and `SpMV f32` are the assembled matrix applied at the same size --
the same Jacobian, neither layout, and the only one of the three whose cost is
set by the sparsity pattern rather than by the element kernel. `f32 vs f64` is
what halving the value traffic bought; the arithmetic is identical, since the
SpMV up-converts each entry and accumulates in double either way.

The final column is the spread of the packed readings. A ratio nearer to 1.00x
than that spread is a tie and must be read as one.

| dof | operation | kernel | operator | terms carried | packed MDOF/s | standard | standard MDOF/s | packed vs standard | SpMV f64 | SpMV f32 | f32 vs f64 | spread (packed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4,121,204 | assemble | sumfact | element kernel | none | 64 | atomic | 39 | 1.64x | 455 | – | – | 1% of 3 |
| 4,121,204 | assemble | sumfact | element kernel | none | 64 | colored | 103 | 0.62x | 455 | – | – | 1% of 3 |
| 4,121,204 | assemble | sumfact | partial | Rhie-Chow (frozen by design), exact-RC J (frozen by design), state ∇p (hoisted), boundary | 60 | atomic | 33 | 1.81x | 455 | – | – | 2% of 3 |
| 4,121,204 | jac_action | sumfact | element kernel | none | 2028 | atomic | 962 | 2.11x | 455 | – | – | 14% of 9 |
| 4,121,204 | jac_action | sumfact | element kernel | none | 2028 | colored | 1416 | 1.43x | 455 | – | – | 14% of 9 |
| 4,121,204 | jac_action | sumfact | solver operator | Rhie-Chow, exact-RC J, state ∇p (hoisted), boundary | 862 | atomic | 408 | 2.11x | 455 | – | – | 1% of 3 |
| 4,121,204 | residual | sumfact | element kernel | none | 2838 | atomic | 1125 | 2.52x | 455 | – | – | 4% of 9 |
| 4,121,204 | residual | sumfact | element kernel | none | 2838 | colored | 2013 | 1.41x | 455 | – | – | 4% of 9 |
| 4,121,204 | residual | sumfact | partial | Rhie-Chow, state ∇p (hoisted) | 1821 | atomic | 923 | 1.97x | 455 | – | – | 2% of 3 |
| 4,121,204 | residual | sumfact | partial | Rhie-Chow, state ∇p (per apply) | 1492 | atomic | 618 | 2.41x | 455 | – | – | 0% of 3 |
| 4,121,204 | residual | sumfact | solver operator | Rhie-Chow, state ∇p (hoisted), boundary | 1637 | atomic | 885 | 1.85x | 455 | – | – | 0% of 3 |


## Provenance

| field | value |
|---|---|
| generated | 2026-10-05 23:06:33 |
| rows | 44 |
| machines | nid006544 |
| sources | kernels_4986020.csv |

Regenerate with:

```
python3 python/cvfem_kernel_report.py /private/tmp/claude-502/-Users-patrickzulian-Desktop-code-merge-git-repos-sfem-spikes-cvfem/5e7895c0-59d9-4ad1-b0a9-9f69607a4e16/scratchpad/kernels_4986020.csv -o docs/CVFEM_Kernels.md --html
```
