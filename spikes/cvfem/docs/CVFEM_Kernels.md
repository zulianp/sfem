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
| assemble | atomic / sumfact / bsr f64 | – | – | – | – | – | – | element kernel |
| assemble | atomic / sympy / bsr f64 | – | – | – | – | – | – | element kernel |
| assemble | colored / sumfact / bsr f64 | – | – | – | – | – | – | element kernel |
| assemble | colored / sympy / bsr f64 | – | – | – | – | – | – | element kernel |
| assemble | packed / sumfact / bsr f64 | – | – | – | – | – | – | element kernel |
| assemble | store / sumfact / bsr f64 | – | – | – | – | – | – | element kernel |
| assemble_diag | atomic / (kernel n/a) | – | – | – | – | – | – | element kernel |
| bsr_apply | n/a / (kernel n/a) / bsr f32 | – | – | – | – | – | – | element kernel |
| bsr_apply | n/a / (kernel n/a) / bsr f64 | – | – | – | – | – | – | element kernel |
| jac_action | atomic / (kernel n/a) | carried | carried | hoisted | carried | – | – | solver operator |
| jac_action | atomic / (kernel n/a) | carried | carried | hoisted | – | – | – | partial |
| jac_action | atomic / (kernel n/a) | – | – | – | – | – | – | element kernel |
| jac_action | packed / (kernel n/a) | carried | carried | hoisted | carried | – | – | solver operator |
| jac_action | packed / (kernel n/a) | carried | carried | hoisted | – | – | – | partial |
| jac_action | packed / (kernel n/a) | – | – | – | – | – | – | element kernel |
| residual | atomic / current | – | – | – | – | – | – | element kernel |
| residual | atomic / sumfact | carried | – | hoisted | carried | – | carried | solver operator |
| residual | atomic / sumfact | carried | – | hoisted | carried | – | – | solver operator |
| residual | atomic / sumfact | carried | – | hoisted | – | – | – | partial |
| residual | atomic / sumfact | carried | – | per apply | – | – | – | partial |
| residual | atomic / sumfact | – | – | – | – | – | – | element kernel |
| residual | packed / current | – | – | – | – | – | – | element kernel |
| residual | packed / sumfact | carried | – | hoisted | carried | – | carried | solver operator |
| residual | packed / sumfact | carried | – | hoisted | carried | – | – | solver operator |
| residual | packed / sumfact | carried | – | hoisted | – | – | – | partial |
| residual | packed / sumfact | carried | – | per apply | – | – | – | partial |
| residual | packed / sumfact | – | – | – | – | – | – | element kernel |


## Throughput

Best observed rate per configuration, in MDOF/s. Best rather than mean because
background load can only make a run slower. The completeness column is the one
from the table above: two rows are comparable only when it matches.

| operation | variant | dof | MDOF/s | completeness | working set | threads |
|---|---|---|---|---|---|---|
| assemble | colored / sumfact / bsr f64 | 28,756,228 | 113 | element kernel | warm | 72 |
| assemble | colored / sumfact / bsr f64 | 16,693,124 | 107 | element kernel | warm | 72 |
| assemble | colored / sympy / bsr f64 | 28,756,228 | 107 | element kernel | warm | 72 |
| assemble | colored / sympy / bsr f64 | 16,693,124 | 101 | element kernel | warm | 72 |
| assemble | colored / sumfact / bsr f64 | 8,586,756 | 94 | element kernel | warm | 72 |
| assemble | colored / sympy / bsr f64 | 8,586,756 | 87 | element kernel | warm | 72 |
| assemble | colored / sumfact / bsr f64 | 3,650,692 | 82 | element kernel | warm | 72 |
| assemble | store / sumfact / bsr f64 | 8,586,756 | 76 | element kernel | warm | 72 |
| assemble | store / sumfact / bsr f64 | 1,098,500 | 76 | element kernel | warm | 72 |
| assemble | store / sumfact / bsr f64 | 28,756,228 | 75 | element kernel | warm | 72 |
| assemble | store / sumfact / bsr f64 | 16,693,124 | 73 | element kernel | warm | 72 |
| assemble | store / sumfact / bsr f64 | 3,650,692 | 73 | element kernel | warm | 72 |
| assemble | colored / sympy / bsr f64 | 3,650,692 | 71 | element kernel | warm | 72 |
| assemble | packed / sumfact / bsr f64 | 1,098,500 | 57 | element kernel | warm | 72 |
| assemble | packed / sumfact / bsr f64 | 8,586,756 | 57 | element kernel | warm | 72 |
| assemble | packed / sumfact / bsr f64 | 16,693,124 | 57 | element kernel | warm | 72 |
| assemble | atomic / sympy / bsr f64 | 1,098,500 | 56 | element kernel | warm | 72 |
| assemble | packed / sumfact / bsr f64 | 28,756,228 | 56 | element kernel | warm | 72 |
| assemble | atomic / sympy / bsr f64 | 3,650,692 | 56 | element kernel | warm | 72 |
| assemble | atomic / sympy / bsr f64 | 16,693,124 | 55 | element kernel | warm | 72 |
| assemble | packed / sumfact / bsr f64 | 3,650,692 | 55 | element kernel | warm | 72 |
| assemble | atomic / sympy / bsr f64 | 28,756,228 | 55 | element kernel | warm | 72 |
| assemble | atomic / sympy / bsr f64 | 8,586,756 | 55 | element kernel | warm | 72 |
| assemble | colored / sumfact / bsr f64 | 1,098,500 | 48 | element kernel | warm | 72 |
| assemble | atomic / sumfact / bsr f64 | 1,098,500 | 39 | element kernel | warm | 72 |
| assemble | colored / sympy / bsr f64 | 1,098,500 | 39 | element kernel | warm | 72 |
| assemble | atomic / sumfact / bsr f64 | 3,650,692 | 39 | element kernel | warm | 72 |
| assemble | atomic / sumfact / bsr f64 | 8,586,756 | 39 | element kernel | warm | 72 |
| assemble | atomic / sumfact / bsr f64 | 16,693,124 | 39 | element kernel | warm | 72 |
| assemble | atomic / sumfact / bsr f64 | 28,756,228 | 39 | element kernel | warm | 72 |
| assemble_diag | atomic / (kernel n/a) | 16,693,124 | 114 | element kernel | warm | 72 |
| assemble_diag | atomic / (kernel n/a) | 28,756,228 | 114 | element kernel | warm | 72 |
| assemble_diag | atomic / (kernel n/a) | 8,586,756 | 113 | element kernel | warm | 72 |
| assemble_diag | atomic / (kernel n/a) | 3,650,692 | 112 | element kernel | warm | 72 |
| assemble_diag | atomic / (kernel n/a) | 1,098,500 | 107 | element kernel | warm | 72 |
| bsr_apply | n/a / (kernel n/a) / bsr f32 | 1,098,500 | 1076 | element kernel | warm | 72 |
| bsr_apply | n/a / (kernel n/a) / bsr f32 | 16,693,124 | 923 | element kernel | warm | 72 |
| bsr_apply | n/a / (kernel n/a) / bsr f32 | 3,650,692 | 858 | element kernel | warm | 72 |
| bsr_apply | n/a / (kernel n/a) / bsr f32 | 8,586,756 | 834 | element kernel | warm | 72 |
| bsr_apply | n/a / (kernel n/a) / bsr f64 | 1,098,500 | 511 | element kernel | warm | 72 |
| bsr_apply | n/a / (kernel n/a) / bsr f64 | 3,650,692 | 495 | element kernel | warm | 72 |
| bsr_apply | n/a / (kernel n/a) / bsr f64 | 16,693,124 | 461 | element kernel | warm | 72 |
| bsr_apply | n/a / (kernel n/a) / bsr f64 | 8,586,756 | 455 | element kernel | warm | 72 |
| jac_action | packed / (kernel n/a) | 3,650,692 | 2117 | element kernel | warm | 72 |
| jac_action | packed / (kernel n/a) | 1,098,500 | 2014 | element kernel | warm | 72 |
| jac_action | packed / (kernel n/a) | 28,756,228 | 1917 | element kernel | warm | 72 |
| jac_action | packed / (kernel n/a) | 8,586,756 | 1906 | element kernel | warm | 72 |
| jac_action | packed / (kernel n/a) | 16,693,124 | 1884 | element kernel | warm | 72 |
| jac_action | packed / (kernel n/a) | 1,098,500 | 1059 | partial | warm | 72 |
| jac_action | packed / (kernel n/a) | 1,098,500 | 975 | solver operator | warm | 72 |
| jac_action | packed / (kernel n/a) | 28,756,228 | 929 | partial | warm | 72 |
| jac_action | packed / (kernel n/a) | 3,650,692 | 926 | partial | warm | 72 |
| jac_action | packed / (kernel n/a) | 28,756,228 | 885 | solver operator | warm | 72 |
| jac_action | packed / (kernel n/a) | 16,693,124 | 876 | partial | warm | 72 |
| jac_action | packed / (kernel n/a) | 8,586,756 | 873 | partial | warm | 72 |
| jac_action | packed / (kernel n/a) | 3,650,692 | 857 | solver operator | warm | 72 |
| jac_action | atomic / (kernel n/a) | 28,756,228 | 829 | element kernel | warm | 72 |
| jac_action | atomic / (kernel n/a) | 3,650,692 | 828 | element kernel | warm | 72 |
| jac_action | atomic / (kernel n/a) | 16,693,124 | 828 | element kernel | warm | 72 |
| jac_action | packed / (kernel n/a) | 16,693,124 | 828 | solver operator | warm | 72 |
| jac_action | atomic / (kernel n/a) | 8,586,756 | 823 | element kernel | warm | 72 |
| jac_action | packed / (kernel n/a) | 8,586,756 | 821 | solver operator | warm | 72 |
| jac_action | atomic / (kernel n/a) | 1,098,500 | 764 | element kernel | warm | 72 |
| jac_action | atomic / (kernel n/a) | 1,098,500 | 443 | partial | warm | 72 |
| jac_action | atomic / (kernel n/a) | 8,586,756 | 436 | partial | warm | 72 |
| jac_action | atomic / (kernel n/a) | 1,098,500 | 432 | solver operator | warm | 72 |
| jac_action | atomic / (kernel n/a) | 3,650,692 | 432 | partial | warm | 72 |
| jac_action | atomic / (kernel n/a) | 16,693,124 | 425 | partial | warm | 72 |
| jac_action | atomic / (kernel n/a) | 28,756,228 | 425 | partial | warm | 72 |
| jac_action | atomic / (kernel n/a) | 8,586,756 | 424 | solver operator | warm | 72 |
| jac_action | atomic / (kernel n/a) | 3,650,692 | 421 | solver operator | warm | 72 |
| jac_action | atomic / (kernel n/a) | 28,756,228 | 415 | solver operator | warm | 72 |
| jac_action | atomic / (kernel n/a) | 16,693,124 | 414 | solver operator | warm | 72 |
| residual | packed / sumfact | 3,650,692 | 2947 | element kernel | warm | 72 |
| residual | packed / sumfact | 1,098,500 | 2740 | element kernel | warm | 72 |
| residual | packed / sumfact | 8,586,756 | 2624 | element kernel | warm | 72 |
| residual | packed / sumfact | 28,756,228 | 2596 | element kernel | warm | 72 |
| residual | packed / sumfact | 16,693,124 | 2568 | element kernel | warm | 72 |
| residual | packed / current | 3,650,692 | 2258 | element kernel | warm | 72 |
| residual | packed / current | 16,693,124 | 2148 | element kernel | warm | 72 |
| residual | packed / current | 28,756,228 | 2133 | element kernel | warm | 72 |
| residual | packed / current | 8,586,756 | 2128 | element kernel | warm | 72 |
| residual | packed / current | 1,098,500 | 2064 | element kernel | warm | 72 |
| residual | packed / sumfact | 3,650,692 | 1499 | partial | warm | 72 |
| residual | packed / sumfact | 1,098,500 | 1445 | partial | warm | 72 |
| residual | packed / sumfact | 28,756,228 | 1394 | partial | warm | 72 |
| residual | packed / sumfact | 8,586,756 | 1385 | partial | warm | 72 |
| residual | packed / sumfact | 16,693,124 | 1380 | partial | warm | 72 |
| residual | packed / sumfact | 3,650,692 | 1362 | solver operator | warm | 72 |
| residual | packed / sumfact | 1,098,500 | 1344 | solver operator | warm | 72 |
| residual | packed / sumfact | 28,756,228 | 1309 | solver operator | warm | 72 |
| residual | packed / sumfact | 1,098,500 | 1293 | partial | warm | 72 |
| residual | packed / sumfact | 3,650,692 | 1286 | partial | warm | 72 |
| residual | packed / sumfact | 1,098,500 | 1283 | solver operator | warm | 72 |
| residual | packed / sumfact | 16,693,124 | 1280 | solver operator | warm | 72 |
| residual | packed / sumfact | 3,650,692 | 1277 | solver operator | warm | 72 |
| residual | packed / sumfact | 8,586,756 | 1272 | solver operator | warm | 72 |
| residual | packed / sumfact | 28,756,228 | 1211 | solver operator | warm | 72 |
| residual | packed / sumfact | 8,586,756 | 1186 | partial | warm | 72 |
| residual | packed / sumfact | 28,756,228 | 1180 | partial | warm | 72 |
| residual | packed / sumfact | 8,586,756 | 1179 | solver operator | warm | 72 |
| residual | packed / sumfact | 16,693,124 | 1177 | solver operator | warm | 72 |
| residual | packed / sumfact | 16,693,124 | 1174 | partial | warm | 72 |
| residual | atomic / current | 3,650,692 | 891 | element kernel | warm | 72 |
| residual | atomic / current | 8,586,756 | 882 | element kernel | warm | 72 |
| residual | atomic / current | 28,756,228 | 866 | element kernel | warm | 72 |
| residual | atomic / current | 1,098,500 | 866 | element kernel | warm | 72 |
| residual | atomic / sumfact | 3,650,692 | 865 | element kernel | warm | 72 |
| residual | atomic / sumfact | 8,586,756 | 854 | element kernel | warm | 72 |
| residual | atomic / sumfact | 28,756,228 | 853 | element kernel | warm | 72 |
| residual | atomic / current | 16,693,124 | 838 | element kernel | warm | 72 |
| residual | atomic / sumfact | 1,098,500 | 837 | element kernel | warm | 72 |
| residual | atomic / sumfact | 16,693,124 | 814 | element kernel | warm | 72 |
| residual | atomic / sumfact | 1,098,500 | 603 | partial | warm | 72 |
| residual | atomic / sumfact | 8,586,756 | 595 | partial | warm | 72 |
| residual | atomic / sumfact | 1,098,500 | 587 | solver operator | warm | 72 |
| residual | atomic / sumfact | 3,650,692 | 586 | partial | warm | 72 |
| residual | atomic / sumfact | 28,756,228 | 583 | partial | warm | 72 |
| residual | atomic / sumfact | 8,586,756 | 579 | solver operator | warm | 72 |
| residual | atomic / sumfact | 1,098,500 | 575 | solver operator | warm | 72 |
| residual | atomic / sumfact | 3,650,692 | 574 | solver operator | warm | 72 |
| residual | atomic / sumfact | 28,756,228 | 572 | solver operator | warm | 72 |
| residual | atomic / sumfact | 16,693,124 | 568 | partial | warm | 72 |
| residual | atomic / sumfact | 3,650,692 | 565 | solver operator | warm | 72 |
| residual | atomic / sumfact | 8,586,756 | 562 | solver operator | warm | 72 |
| residual | atomic / sumfact | 16,693,124 | 556 | solver operator | warm | 72 |
| residual | atomic / sumfact | 28,756,228 | 550 | solver operator | warm | 72 |
| residual | atomic / sumfact | 16,693,124 | 538 | solver operator | warm | 72 |
| residual | atomic / sumfact | 8,586,756 | 435 | partial | warm | 72 |
| residual | atomic / sumfact | 28,756,228 | 432 | partial | warm | 72 |
| residual | atomic / sumfact | 3,650,692 | 420 | partial | warm | 72 |
| residual | atomic / sumfact | 16,693,124 | 420 | partial | warm | 72 |
| residual | atomic / sumfact | 1,098,500 | 410 | partial | warm | 72 |


## What the physics costs

The element kernel alone against the same kernel carrying what the solver
needs, one operation and one size at a time so that only the operator varies.
The bare-kernel row is the number the regression gate tracks and the one every
quoted figure has historically meant; the solver does not run it.

`spread` is how far apart that configuration's repeated measurements were, as a
percentage of the best. Two rows differ meaningfully only when the gap between
them is larger than that -- which is not true of every pair here, and saying so
is cheaper than inviting the reader to over-read a 3% difference.

### residual, packed layout, 3,650,692 dof

| operator | terms carried | working set | MDOF/s | spread | cost vs bare kernel |
|---|---|---|---|---|---|
| element kernel | none | warm | 2947 | 1% of 5 | 1.00x |
| partial | Rhie-Chow, state ∇p (hoisted) | warm | 1499 | 1% of 5 | 1.97x |
| solver operator | Rhie-Chow, state ∇p (hoisted), boundary | warm | 1362 | 1% of 5 | 2.16x |
| partial | Rhie-Chow, state ∇p (per apply) | warm | 1286 | 1% of 5 | 2.29x |
| solver operator | Rhie-Chow, state ∇p (hoisted), boundary, transient | warm | 1277 | 1% of 5 | 2.31x |

### residual, atomic layout, 3,650,692 dof

| operator | terms carried | working set | MDOF/s | spread | cost vs bare kernel |
|---|---|---|---|---|---|
| element kernel | none | warm | 865 | 4% of 5 | 1.00x |
| partial | Rhie-Chow, state ∇p (hoisted) | warm | 586 | 1% of 5 | 1.48x |
| solver operator | Rhie-Chow, state ∇p (hoisted), boundary | warm | 574 | 2% of 5 | 1.51x |
| solver operator | Rhie-Chow, state ∇p (hoisted), boundary, transient | warm | 565 | 13% of 5 | 1.53x |
| partial | Rhie-Chow, state ∇p (per apply) | warm | 420 | 10% of 5 | 2.06x |

### jac_action, packed layout, 3,650,692 dof

| operator | terms carried | working set | MDOF/s | spread | cost vs bare kernel |
|---|---|---|---|---|---|
| element kernel | none | warm | 2117 | 1% of 5 | 1.00x |
| partial | Rhie-Chow, exact-RC J, state ∇p (hoisted) | warm | 926 | 4% of 5 | 2.29x |
| solver operator | Rhie-Chow, exact-RC J, state ∇p (hoisted), boundary | warm | 857 | 1% of 5 | 2.47x |
| assembled matrix | SpMV of the assembled BSR, f64 values | warm | 495 | n=1 | 4.28x |
| assembled matrix | SpMV of the assembled BSR, f32 values | warm | 858 | n=1 | 2.47x |

### jac_action, atomic layout, 3,650,692 dof

| operator | terms carried | working set | MDOF/s | spread | cost vs bare kernel |
|---|---|---|---|---|---|
| element kernel | none | warm | 828 | 1% of 5 | 1.00x |
| partial | Rhie-Chow, exact-RC J, state ∇p (hoisted) | warm | 432 | 1% of 5 | 1.92x |
| solver operator | Rhie-Chow, exact-RC J, state ∇p (hoisted), boundary | warm | 421 | 3% of 5 | 1.97x |
| assembled matrix | SpMV of the assembled BSR, f64 values | warm | 495 | n=1 | 1.67x |
| assembled matrix | SpMV of the assembled BSR, f32 values | warm | 858 | n=1 | 0.97x |


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
| 1,098,500 | assemble | sumfact | element kernel | none | 57 | atomic | 39 | 1.44x | 511 | 1076 | 2.10x | 4% of 5 |
| 1,098,500 | assemble | sumfact | element kernel | none | 57 | colored | 48 | 1.19x | 511 | 1076 | 2.10x | 4% of 5 |
| 3,650,692 | assemble | sumfact | element kernel | none | 55 | atomic | 39 | 1.42x | 495 | 858 | 1.73x | 3% of 5 |
| 3,650,692 | assemble | sumfact | element kernel | none | 55 | colored | 82 | 0.68x | 495 | 858 | 1.73x | 3% of 5 |
| 8,586,756 | assemble | sumfact | element kernel | none | 57 | atomic | 39 | 1.46x | 455 | 834 | 1.83x | 2% of 5 |
| 8,586,756 | assemble | sumfact | element kernel | none | 57 | colored | 94 | 0.60x | 455 | 834 | 1.83x | 2% of 5 |
| 16,693,124 | assemble | sumfact | element kernel | none | 57 | atomic | 39 | 1.47x | 461 | 923 | 2.00x | 4% of 5 |
| 16,693,124 | assemble | sumfact | element kernel | none | 57 | colored | 107 | 0.53x | 461 | 923 | 2.00x | 4% of 5 |
| 28,756,228 | assemble | sumfact | element kernel | none | 56 | atomic | 39 | 1.45x | – | – | – | 54% of 5 |
| 28,756,228 | assemble | sumfact | element kernel | none | 56 | colored | 113 | 0.49x | – | – | – | 54% of 5 |
| 1,098,500 | jac_action | n/a | element kernel | none | 2014 | atomic | 764 | 2.64x | 511 | 1076 | 2.10x | 2% of 5 |
| 1,098,500 | jac_action | n/a | partial | Rhie-Chow, exact-RC J, state ∇p (hoisted) | 1059 | atomic | 443 | 2.39x | 511 | 1076 | 2.10x | 3% of 5 |
| 1,098,500 | jac_action | n/a | solver operator | Rhie-Chow, exact-RC J, state ∇p (hoisted), boundary | 975 | atomic | 432 | 2.26x | 511 | 1076 | 2.10x | 1% of 5 |
| 3,650,692 | jac_action | n/a | element kernel | none | 2117 | atomic | 828 | 2.56x | 495 | 858 | 1.73x | 1% of 5 |
| 3,650,692 | jac_action | n/a | partial | Rhie-Chow, exact-RC J, state ∇p (hoisted) | 926 | atomic | 432 | 2.14x | 495 | 858 | 1.73x | 4% of 5 |
| 3,650,692 | jac_action | n/a | solver operator | Rhie-Chow, exact-RC J, state ∇p (hoisted), boundary | 857 | atomic | 421 | 2.04x | 495 | 858 | 1.73x | 1% of 5 |
| 8,586,756 | jac_action | n/a | element kernel | none | 1906 | atomic | 823 | 2.32x | 455 | 834 | 1.83x | 1% of 5 |
| 8,586,756 | jac_action | n/a | partial | Rhie-Chow, exact-RC J, state ∇p (hoisted) | 873 | atomic | 436 | 2.00x | 455 | 834 | 1.83x | 1% of 5 |
| 8,586,756 | jac_action | n/a | solver operator | Rhie-Chow, exact-RC J, state ∇p (hoisted), boundary | 821 | atomic | 424 | 1.94x | 455 | 834 | 1.83x | 2% of 5 |
| 16,693,124 | jac_action | n/a | element kernel | none | 1884 | atomic | 828 | 2.28x | 461 | 923 | 2.00x | 1% of 5 |
| 16,693,124 | jac_action | n/a | partial | Rhie-Chow, exact-RC J, state ∇p (hoisted) | 876 | atomic | 425 | 2.06x | 461 | 923 | 2.00x | 5% of 5 |
| 16,693,124 | jac_action | n/a | solver operator | Rhie-Chow, exact-RC J, state ∇p (hoisted), boundary | 828 | atomic | 414 | 2.00x | 461 | 923 | 2.00x | 1% of 5 |
| 28,756,228 | jac_action | n/a | element kernel | none | 1917 | atomic | 829 | 2.31x | – | – | – | 1% of 5 |
| 28,756,228 | jac_action | n/a | partial | Rhie-Chow, exact-RC J, state ∇p (hoisted) | 929 | atomic | 425 | 2.19x | – | – | – | 2% of 5 |
| 28,756,228 | jac_action | n/a | solver operator | Rhie-Chow, exact-RC J, state ∇p (hoisted), boundary | 885 | atomic | 415 | 2.13x | – | – | – | 3% of 5 |
| 1,098,500 | residual | sumfact | element kernel | none | 2740 | atomic | 837 | 3.28x | 511 | 1076 | 2.10x | 2% of 5 |
| 1,098,500 | residual | current | element kernel | none | 2064 | atomic | 866 | 2.38x | 511 | 1076 | 2.10x | 2% of 5 |
| 1,098,500 | residual | sumfact | partial | Rhie-Chow, state ∇p (hoisted) | 1445 | atomic | 603 | 2.40x | 511 | 1076 | 2.10x | 3% of 5 |
| 1,098,500 | residual | sumfact | solver operator | Rhie-Chow, state ∇p (hoisted), boundary | 1344 | atomic | 587 | 2.29x | 511 | 1076 | 2.10x | 2% of 5 |
| 1,098,500 | residual | sumfact | solver operator | Rhie-Chow, state ∇p (hoisted), boundary, transient | 1283 | atomic | 575 | 2.23x | 511 | 1076 | 2.10x | 3% of 5 |
| 1,098,500 | residual | sumfact | partial | Rhie-Chow, state ∇p (per apply) | 1293 | atomic | 410 | 3.16x | 511 | 1076 | 2.10x | 4% of 5 |
| 3,650,692 | residual | sumfact | element kernel | none | 2947 | atomic | 865 | 3.41x | 495 | 858 | 1.73x | 1% of 5 |
| 3,650,692 | residual | current | element kernel | none | 2258 | atomic | 891 | 2.53x | 495 | 858 | 1.73x | 2% of 5 |
| 3,650,692 | residual | sumfact | partial | Rhie-Chow, state ∇p (hoisted) | 1499 | atomic | 586 | 2.56x | 495 | 858 | 1.73x | 1% of 5 |
| 3,650,692 | residual | sumfact | solver operator | Rhie-Chow, state ∇p (hoisted), boundary | 1362 | atomic | 574 | 2.37x | 495 | 858 | 1.73x | 1% of 5 |
| 3,650,692 | residual | sumfact | solver operator | Rhie-Chow, state ∇p (hoisted), boundary, transient | 1277 | atomic | 565 | 2.26x | 495 | 858 | 1.73x | 1% of 5 |
| 3,650,692 | residual | sumfact | partial | Rhie-Chow, state ∇p (per apply) | 1286 | atomic | 420 | 3.06x | 495 | 858 | 1.73x | 1% of 5 |
| 8,586,756 | residual | sumfact | element kernel | none | 2624 | atomic | 854 | 3.07x | 455 | 834 | 1.83x | 3% of 5 |
| 8,586,756 | residual | current | element kernel | none | 2128 | atomic | 882 | 2.41x | 455 | 834 | 1.83x | 2% of 5 |
| 8,586,756 | residual | sumfact | partial | Rhie-Chow, state ∇p (hoisted) | 1385 | atomic | 595 | 2.33x | 455 | 834 | 1.83x | 4% of 5 |
| 8,586,756 | residual | sumfact | solver operator | Rhie-Chow, state ∇p (hoisted), boundary | 1272 | atomic | 579 | 2.20x | 455 | 834 | 1.83x | 1% of 5 |
| 8,586,756 | residual | sumfact | solver operator | Rhie-Chow, state ∇p (hoisted), boundary, transient | 1179 | atomic | 562 | 2.10x | 455 | 834 | 1.83x | 12% of 5 |
| 8,586,756 | residual | sumfact | partial | Rhie-Chow, state ∇p (per apply) | 1186 | atomic | 435 | 2.73x | 455 | 834 | 1.83x | 1% of 5 |
| 16,693,124 | residual | sumfact | element kernel | none | 2568 | atomic | 814 | 3.15x | 461 | 923 | 2.00x | 2% of 5 |
| 16,693,124 | residual | current | element kernel | none | 2148 | atomic | 838 | 2.56x | 461 | 923 | 2.00x | 1% of 5 |
| 16,693,124 | residual | sumfact | partial | Rhie-Chow, state ∇p (hoisted) | 1380 | atomic | 568 | 2.43x | 461 | 923 | 2.00x | 2% of 5 |
| 16,693,124 | residual | sumfact | solver operator | Rhie-Chow, state ∇p (hoisted), boundary | 1280 | atomic | 556 | 2.30x | 461 | 923 | 2.00x | 4% of 5 |
| 16,693,124 | residual | sumfact | solver operator | Rhie-Chow, state ∇p (hoisted), boundary, transient | 1177 | atomic | 538 | 2.19x | 461 | 923 | 2.00x | 0% of 5 |
| 16,693,124 | residual | sumfact | partial | Rhie-Chow, state ∇p (per apply) | 1174 | atomic | 420 | 2.80x | 461 | 923 | 2.00x | 1% of 5 |
| 28,756,228 | residual | sumfact | element kernel | none | 2596 | atomic | 853 | 3.05x | – | – | – | 3% of 5 |
| 28,756,228 | residual | current | element kernel | none | 2133 | atomic | 866 | 2.46x | – | – | – | 1% of 5 |
| 28,756,228 | residual | sumfact | partial | Rhie-Chow, state ∇p (hoisted) | 1394 | atomic | 583 | 2.39x | – | – | – | 1% of 5 |
| 28,756,228 | residual | sumfact | solver operator | Rhie-Chow, state ∇p (hoisted), boundary | 1309 | atomic | 572 | 2.29x | – | – | – | 1% of 5 |
| 28,756,228 | residual | sumfact | solver operator | Rhie-Chow, state ∇p (hoisted), boundary, transient | 1211 | atomic | 550 | 2.20x | – | – | – | 1% of 5 |
| 28,756,228 | residual | sumfact | partial | Rhie-Chow, state ∇p (per apply) | 1180 | atomic | 432 | 2.73x | – | – | – | 1% of 5 |


## Reading the sections below

The tables above are regenerated from `perf/campaign_grace.csv` on every run. The sections
that follow are not: they are written by hand, they survive a regeneration through
`--prose`, and they quote the measurements that were in front of whoever wrote them.

That matters here because the two no longer share a problem size. The analyses below were
written against 4,121,204 and 8,586,756 dof on an earlier binary; the campaign above sweeps
1,098,500 to 28,756,228 dof and does not contain 4,121,204 at all. So a number in the prose
will often have no counterpart in the tables, and where both exist they were taken on
different binaries months apart.

Nothing below is contradicted by the tables above -- the campaign reproduces the recorded
baseline for the packed residual, 2624 MDOF/s against 2620.2 -- and the reasoning in each
section is about mechanism rather than about a particular figure. But a reader who tries to
find a prose number in a table above will not find it, and that is provenance rather than
disagreement.

One thing the campaign does add that the sections below predate: the layout comparison is
now swept across five sizes, and the assembly ranking INVERTS inside that range. Packed
assembly beats colored at 1,098,500 dof and loses to it by 2x at 28,756,228. Any statement
about which layout wins assembly is a statement about a size.

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
| generated | 2026-09-24 19:58:49 |
| rows | 133 |
| machines | nid006538 |
| sources | campaign_grace.csv |
| hand-written sections | 5, carried through --prose |

Regenerate with:

```
python3 python/cvfem_kernel_report.py perf/campaign_grace.csv -o docs/CVFEM_Kernels.md --html \
      --prose docs/kernel_prose/05_reading_the_prose.md \
      --prose docs/kernel_prose/10_headline_vs_operator.md \
      --prose docs/kernel_prose/20_gradient_cache.md \
      --prose docs/kernel_prose/30_layout_margin.md \
      --prose docs/kernel_prose/40_spmv.md
```
