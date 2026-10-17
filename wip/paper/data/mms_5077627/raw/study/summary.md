All 15 final nonlinear targets met; spatial accuracy is reported separately.
The fixed maximum-velocity-error limit remains 0.01.
Adjacent-grid rates are observations, not a common asymptotic-order guarantee.

| Scheme | n | Velocity L2 | Shifted pressure L2 | Velocity max | Accuracy gate | Velocity L2 rate |
|---|---:|---:|---:|---:|---|---:|
| lo | 16 | 0.1191008 | 0.04546217 | 0.1098873 | FAIL | - |
| lo | 24 | 0.09626501 | 0.03480003 | 0.08975286 | FAIL | 0.5250 |
| lo | 32 | 0.07950285 | 0.02825269 | 0.07281847 | FAIL | 0.6650 |
| ho_lim0 | 16 | 0.02422124 | 0.009988177 | 0.04150784 | FAIL | - |
| ho_lim0 | 24 | 0.01057835 | 0.004270332 | 0.02071203 | FAIL | 2.0431 |
| ho_lim0 | 32 | 0.005869032 | 0.002436545 | 0.01258695 | FAIL | 2.0478 |
| ho_lim1 | 16 | 0.02400928 | 0.009457753 | 0.044371 | FAIL | - |
| ho_lim1 | 24 | 0.01011878 | 0.004088535 | 0.0215461 | FAIL | 2.1310 |
| ho_lim1 | 32 | 0.005552846 | 0.002358197 | 0.01289726 | FAIL | 2.0859 |
| ho_lim2_k0 | 16 | 0.02553976 | 0.0104529 | 0.04448365 | FAIL | - |
| ho_lim2_k0 | 24 | 0.01186764 | 0.004977835 | 0.02156835 | FAIL | 1.8902 |
| ho_lim2_k0 | 32 | 0.007344536 | 0.00311574 | 0.01293553 | FAIL | 1.6680 |
| ho_lim3 | 16 | 0.02685859 | 0.01077593 | 0.0443178 | FAIL | - |
| ho_lim3 | 24 | 0.01310761 | 0.005372045 | 0.02157884 | FAIL | 1.7693 |
| ho_lim3 | 32 | 0.008453661 | 0.003480793 | 0.01295957 | FAIL | 1.5246 |

All logs and exit codes are retained, including failed accuracy gates.
Intermediate residual-floor acceptance, where present, is recorded in summary.json.
This is an affine MMS study, not an unstructured-mesh or solver-performance result.
