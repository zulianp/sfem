# Local CPU solver iteration evidence

These 24 runs use one thread on an Apple M2 Pro, macOS 26.6.2 arm64,
GCC 14.2.0, Release build. They were collected on 13 October 2026 local time.
They are separate from the Grace operator measurements. OpenMP reports that
affinity is unsupported on this machine; no performance-core binding is claimed.
The driver prints timings, but this dataset supports iteration counts only.

The public regularized cavity occupies `[0,2]^3`, with density and lid velocity
one and viscosity `2/Re`. Each pair uses the same semi-structured mesh, with
two microcells per macrocell direction. The table uses 4, 8 and 12 fine cells
per direction at Reynolds numbers 1 and 10. The regular series uses affine
geometry. The deformed series uses isoparametric geometry and the fixed
boundary-preserving displacement described in
`spikes/cvfem/docs/solver_iteration_study.md`, at fraction 0.025 of the box size.
Every deformed run records a positive-Jacobian certificate from its stored
coordinates and confirms that the curved geometry path is active.

Both preconditioners use the same first-order exact-Rhie--Chow matrix-free
Jacobian, FGMRES, zero-mean pressure gauge and continuation schedule. BJ in the
paper is the existing 3-by-3 velocity block plus scalar pressure inverse per
node, with damping one. V is the existing multiplicative diagonal Vanka method,
using FP64 32-dof microcell patches and damping 0.35. Factors are rebuilt for
every Newton iteration. This is a comparison of those configured methods,
including their stated damping choices.

The linear target is `max(1e-14, 1e-6 * ||b||)`, with restart 60 and at most
1000 iterations. Each correction is independently checked against the actual
operator and right-hand side before the line search. The nonlinear target is
`max(1e-12, 1e-8 * ||R_initial||)`, with at most 30 Newton steps per continuation
stage. Counts include all stages: one stage at Re=1 and Re=1,4,10 at Re=10.
The paired methods have equal Newton totals in all cases. All 24 solves meet
the strict linear and final nonlinear tests and return zero. Vanka uses fewer
linear iterations here, but both methods' counts grow with mesh size.
These results do not establish mesh-independent convergence, unstructured
packed solver performance or solver speedup.

`table.json` retains the complete controlled environment, every linear check,
final nonlinear summary, process return code and log hash for every run.
`logs/` contains the original logs byte for byte. Absolute command paths were
omitted from the published JSON. `manifest.json` records the binary hash,
selected source hashes, base revision, hardware and compiler. The driver patch
against that base is retained as `solver-driver.patch`; this is selected-source
provenance, not an archive of all compiler and dependency inputs.

After collection, the collector gained guards for invalid refinement levels
and malformed or missing geometry records. `collector-readback.patch` records
those changes. The manifest distinguishes the captured collector hash from
the current readback hash. The table generator rereads every raw log, checks
hashes, configuration, tolerances and geometry, and rejects changed counts or
missing cases. The changes did not alter any of the 24 recorded results.

Regenerate the paper table with `make figures` in `wip/paper`. The generator is
`wip/paper/python/solver_iterations.py`; its output is
`wip/paper/tables/solver_iterations.tex`. To collect a new dataset, use
`spikes/cvfem/scripts/run_solver_study.sh` with a build directory and a new result
directory. The numerical gate also checks a deliberately capped linear solve,
invalid tolerances, the opt-in check's effect on counts, and Vanka stencil
agreement with its assembled reference. Source-only bundle collection was
separately exercised with a supplied `SOURCE-REVISIONS` file and no Git metadata.
