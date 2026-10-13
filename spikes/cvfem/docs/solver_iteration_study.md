# CPU iteration study

This study compares the existing node-block Jacobi and diagonal Vanka preconditioners
with FGMRES on the public regularized cavity. It records Newton updates and all linear
iterations, including unsuccessful attempts. It makes no timing or mesh-independent
convergence claim.

Both arms use the same semi-structured HEX8 mesh, first-order convection, exact-RC
matrix-free Jacobian action, boundary values and zero-mean pressure gauge. The box is
`[0,2]^3`, density and velocity scale are one, and viscosity is `2/Re`. The macro
resolution is the fine cell count divided by the fixed internal refinement level.
This is a solver experiment on the semi-structured path; it does not establish solver
performance for the paper's unstructured packed layout.

Jacobi in this driver inverts a 3-by-3 velocity block and a separate pressure scalar
at each node. It is not a coupled 4-by-4 inverse. Vanka uses the existing multiplicative
microcell patch sweep, diagonal velocity approximation and 8-by-8 pressure Schur solve.
Its stored patch factors are double precision. The damping factors are the existing
no-multigrid defaults: one for Jacobi and 0.35 for Vanka. They are reported explicitly;
this is a comparison of those configurations, not equally tuned preconditioners.
Both preconditioners are rebuilt at every Newton step (`SFEM_VANKA_FREEZE=1`).

The deformed series applies a sine displacement to the interior nodes after
semi-structured refinement. Its amplitude is 0.025 times the shortest box length,
independent of mesh resolution, and its direction is `(1,0.7,-0.4)`. The sine factors
are `sin(2*pi*x/Lx) sin(pi*y/Ly) sin(pi*z/Lz)`. Boundary coordinates stay exactly
unchanged. `SFEM_CAVITY_WARP` is restricted to this public regularized cavity with
a semi-structured mesh and explicit `SFEM_GEOM=isoparam`; the existing curved-cell
dispatch supplies each microcell's geometry. The collector checks that this path
was actually selected.

The geometry check uses the stored coordinates, including their rounding. For each
fine cell, let `J0=diag(hx,hy,hz)`. Every entry of `J J0^-1-I` is a convex combination
of four normalized edge differences. Taking their maximum absolute values gives a
Frobenius-norm upper bound `q` everywhere in that trilinear cell. The fixture refuses
`q>=1`; otherwise `I+t*(J J0^-1-I)` is nonsingular for `0<=t<=1`, proving positive
orientation. Its determinant ratio is at least `(1-q)^3`. The reported certificate
uses the worst bound across all cells, rather than sampling determinant values.

## Residual acceptance

At Newton state `x`, the driver forms `b = -R(x)` after its existing constraint and
pressure-compatibility projections. Krylov solves `A dx = b`, where `A` is the actual
selected linear operator. The opt-in `SFEM_CHECK_LINEAR_RESIDUAL=1` evaluates `A dx`
again after the correction's existing pressure-gauge projection. The target is zeroed
before application because these operators accumulate into their output.

The independent test is

```
norm(b - A dx) < max(linear_atol, linear_rtol * norm(b)).
```

It uses the solver's unscaled Euclidean norm over owned rows, with the squared sums
reduced before taking the square root. Constrained rows and the pressure mean remain
in this check. Finite norms and successful solver/operator returns are required.
Rejected corrections stop that continuation stage before its line search. A solve
at its iteration limit is accepted only if the true residual passes; otherwise its
status is `iteration_limit`. Other statuses distinguish nonfinite values, solver
failure and failure of the true residual after an estimated Krylov convergence.
The generic FGMRES implementation is unchanged.

The collector also checks the final nonlinear residual against the configured
nonlinear tolerances, independently of `newton_converged`. Step-only convergence and
the driver's relaxed residual-floor acceptance are disabled. Reynolds continuation
uses a fixed factor of four, with retries and adaptation disabled. Newton and linear
totals cover all stages, so counts at different Reynolds numbers need not be monotone.

## Run and retain evidence

With a configured CVFEM build and an unused results path:

```bash
bash spikes/cvfem/scripts/run_solver_study.sh ../build/cvfem /tmp/solver-study
```

The wrapper builds `cvfem_hex8_ns_ssgmg`, runs the small residual gate and collects
the two preconditioners at fine resolutions 4, 8 and 12 and Reynolds numbers 1 and 10.
Regular and deformed meshes give 24 cases in total. Every solve uses one CPU thread.
The gate checks unchanged iteration counts when
the optional check is enabled, an enforced one-iteration failure, invalid tolerance
rejection, the existing Vanka stencil-versus-BSR comparison, and both preconditioners
on the deformed geometry with a positive-Jacobian certificate.

The Python collector can also be called directly:

```bash
PYTHONPATH=python python3 spikes/cvfem/scripts/solver_iteration_study.py \
  --binary ../build/cvfem/cvfem_hex8_ns_ssgmg --out /tmp/solver-study-direct \
  --sizes 4,8,12 --reynolds 1,10 --warps 0,0.025 --level 2 --threads 1
```

`table.json` retains configuration, process status, every linear check and final
nonlinear acceptance for every case. Each case has its raw `run.log` and its hash.
`manifest.json` records the binary hash, Git revision, host, selected source hashes
and driver patch hash; those selected sources and the driver patch are copied beside
it. Keep the build/dependency provenance with this directory for paper measurements.
For a source bundle without `.git`, pass `--provenance-file /path/to/SOURCE-REVISIONS`.
The wrapper detects that file beside the bundle's `sfem` directory, or accepts its
path through `SFEM_SOURCE_REVISIONS`. Source-only mode is recorded explicitly and
does not search parent directories for an unrelated Git repository.
The collector refuses an existing output directory and exits nonzero when any row
fails. Failed rows remain in both the JSON and Markdown tables. Their counts mean
work spent, not the number of iterations needed to converge.

Python 3.6 or newer is sufficient. Source and parser checks run locally; Grace
measurements must be collected separately with the Grace binary and preserved logs.
