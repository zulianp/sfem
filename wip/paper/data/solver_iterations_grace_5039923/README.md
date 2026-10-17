# Grace CPU solver iteration evidence

Campaign `validation-5039923-vfXOQo`, job 5039923, node `nid006545`, collected 2026-10-12 UTC with GCC 14.3.0. One OpenMP thread, first-order exact-Rhie--Chow matrix-free semi-structured regularized cavity on `[0,2]^3`; no multigrid and no timings accepted. The manifest and each result retain the controlled configuration.

All 24 public cases pass: 128 true linear residual checks meet the configured target, and final nonlinear residuals meet the strict configured target. The 12 deformed cases retain the boundary-preserving sine warp and positive trilinear-orientation certificate. Vanka setup calls equal Newton steps; the archived driver and `SFEM_VANKA_FREEZE=1` specify rebuilding each step. Node-block Jacobi is built directly each Newton step. The seven gate controls retain checked/unchecked count parity, Vanka stencil/BSR agreement, both warped preconditioners, and failing capped/nonfinite controls.

All 24 Newton and total FGMRES counts equal the preserved `../solver_iterations_local_20261013` archive. This is a bounded replication of iteration counts, not a runtime comparison, field parity test, mesh-independent convergence result, or unstructured packed solver benchmark.

Raw logs and result JSON are copied byte for byte. Selected collector/driver/FGMRES/Vanka sources, full source manifest, revisions, compiler, CMake cache and build log retain provenance. The manifest records an independent SHA256 readback of the retrieved solver executable matching its capture digest; the executable and unused trace files are omitted. The paper table generator reparses raw logs before accepting these counts.
