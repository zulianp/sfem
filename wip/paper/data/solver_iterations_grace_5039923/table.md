CPU regularized-cavity iteration study. Failed and capped runs are retained.

| Fine cells | Re | Warp | Preconditioner | Newton | Linear | Status |
|---:|---:|---:|---|---:|---:|---|
| 4 | 1 | 0 | bjacobi | 3 | 97 | converged |
| 4 | 1 | 0 | vanka | 3 | 66 | converged |
| 4 | 1 | 0.025 | vanka | 3 | 68 | converged |
| 4 | 1 | 0.025 | bjacobi | 3 | 106 | converged |
| 4 | 10 | 0 | bjacobi | 9 | 323 | converged |
| 4 | 10 | 0 | vanka | 9 | 208 | converged |
| 4 | 10 | 0.025 | vanka | 9 | 209 | converged |
| 4 | 10 | 0.025 | bjacobi | 9 | 335 | converged |
| 8 | 1 | 0 | bjacobi | 3 | 223 | converged |
| 8 | 1 | 0 | vanka | 3 | 145 | converged |
| 8 | 1 | 0.025 | vanka | 3 | 149 | converged |
| 8 | 1 | 0.025 | bjacobi | 3 | 252 | converged |
| 8 | 10 | 0 | bjacobi | 7 | 471 | converged |
| 8 | 10 | 0 | vanka | 7 | 298 | converged |
| 8 | 10 | 0.025 | vanka | 7 | 304 | converged |
| 8 | 10 | 0.025 | bjacobi | 7 | 509 | converged |
| 12 | 1 | 0 | bjacobi | 3 | 429 | converged |
| 12 | 1 | 0 | vanka | 3 | 298 | converged |
| 12 | 1 | 0.025 | vanka | 3 | 297 | converged |
| 12 | 1 | 0.025 | bjacobi | 3 | 468 | converged |
| 12 | 10 | 0 | bjacobi | 7 | 832 | converged |
| 12 | 10 | 0 | vanka | 7 | 549 | converged |
| 12 | 10 | 0.025 | vanka | 7 | 554 | converged |
| 12 | 10 | 0.025 | bjacobi | 7 | 893 | converged |

Counts include every attempted continuation stage. Failure counts are work spent, not converged iteration counts.
This measures the named semi-structured solver configuration, not unstructured packed solver performance or mesh-independent convergence.
