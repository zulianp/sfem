All 5 final nonlinear targets met; spatial accuracy is reported separately.
The fixed maximum-velocity-error limit remains 0.01.
Adjacent-grid rates are observations, not a common asymptotic-order guarantee.

| Scheme | n | Velocity L2 | Shifted pressure L2 | Velocity max | Accuracy gate | Velocity L2 rate |
|---|---:|---:|---:|---:|---|---:|
| lo | 40 | 0.06757192 | 0.02384001 | 0.0605392 | FAIL | - |
| ho_lim0 | 40 | 0.003718764 | 0.001608478 | 0.008564342 | PASS | - |
| ho_lim1 | 40 | 0.003502292 | 0.001567528 | 0.008695268 | PASS | - |
| ho_lim2_k0 | 40 | 0.005266114 | 0.002227415 | 0.008723744 | PASS | - |
| ho_lim3 | 40 | 0.006247547 | 0.002553279 | 0.008738286 | PASS | - |

All logs and exit codes are retained, including failed accuracy gates.
Scheme labels come from runs.tsv; source, binary and launch provenance need separate review.
Intermediate residual-floor acceptance, where present, is recorded in summary.json.
This is an affine MMS study, not an unstructured-mesh or solver-performance result.
