# Grace manufactured-solution refinement

Job 5077627 ran on Daint node `nid006551` on 2026-10-17 with GCC 14.3,
FP64 and eight configured OpenMP threads. It evaluates the steady affine
Farrell problem on `[0,2]^3` at 16, 24 and 32 cells per axis. The five schemes
share density 1, dynamic viscosity 0.01, exact Rhie--Chow derivatives,
manufactured velocity on all six faces and a zero-mean pressure gauge.
FGMRES uses block Jacobi, restart 480, linear tolerance `1e-6` and final
nonlinear relative tolerance `1e-10`. There is no transient term or frozen
convective correction.

All 15 final nonlinear targets are met; the largest final relative residual is
`7.593783e-11`. All 15 runs still fail the separate, unchanged maximum velocity
error target `0.01`. Every exit code remains 1. The clip scheme at n=16 accepted
an intermediate continuation residual floor, then met the final target.

| Scheme | Velocity L2 rate, 16 to 24 | Velocity L2 rate, 24 to 32 |
|---|---:|---:|
| First-order upwind | 0.5250 | 0.6650 |
| Unlimited reconstruction | 2.0431 | 2.0478 |
| Two-node clip | 2.1310 | 2.0859 |
| Venkatakrishnan, K=0 | 1.8902 | 1.6680 |
| Darwish | 1.7693 | 1.5246 |

All velocity L2 and maximum errors decrease. These adjacent-grid observations
support near-second-order behavior for unlimited reconstruction and clip in
this range; they do not establish a common asymptotic order for all limiters.
The lower Venkatakrishnan/Darwish rates do not by themselves identify a defect
or its cause. At n=32, reconstructed maximum velocity errors remain between
0.01259 and 0.01296. The later [n=40 follow-up](../mms_5077906/README.md)
checks the fixed target and extends this evidence; this archive retains only
the original 16/24/32 results.

`raw/` preserves all 29 retrieved files byte for byte. `RAW-SHA256.txt` also
covers the collection script, original report reader and stricter replay reader
in `sources/`. The returned source manifest and complete successful source-check
log bind the copied bundle; its missing Git metadata is not treated as a commit
identity. The binary hash is recorded, not independently rebuilt here. Scheme
labels rely on the source-pinned collector and launch settings; the driver log
does not echo limiter settings. No solution field files were retrieved.

Run `python3 verify.py` in this directory to check hashes, case coverage,
provenance, final residuals, retained failures and reported rates. The replay
does not modify the original reports or run a solver. This study provides
neither unstructured-solve evidence nor a solver-performance comparison.

The build completed with a GCC bound warning in the semi-structured
isoparametric gradient routine. These logs explicitly select flat affine
geometry (`semi_structured: 0`); this study does not exercise or validate that
warned path.
