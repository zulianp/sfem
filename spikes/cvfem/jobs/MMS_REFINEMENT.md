# Focused manufactured-solution refinement

This study separates final nonlinear convergence from the driver's fixed
maximum-velocity-error target of 0.01. A converged coarse-grid solution can fail
that target. Such failures stay visible in the logs and summary; they are not
converted into accuracy passes.

The steady equations are `rho*(u.grad)u - mu*laplacian(u) + grad(p) = f` and
`div(u)=0` on `[0,2]^3`. All six faces use the manufactured velocity and pressure
has a zero-mean gauge. The source is the generated Farrell polynomial fixture,
with density 1 and dynamic viscosity 0.01. Its pressure parameter is `1/mu=100`;
the driver's length-based Reynolds number is 200. Geometry is affine, the fields
are FP64, and there is no time term or frozen convective correction.

The five schemes are first-order upwind, unlimited reconstruction, two-node clip,
Venkatakrishnan with `K=0`, and Darwish. Each runs on 16, 24 and 32 cells per axis,
with pack size 1024 and eight threads. FGMRES uses block Jacobi, restart 480,
linear relative tolerance `1e-6` and nonlinear relative tolerance `1e-10`.
The existing continuation is retained. No solver-performance claim is intended.

The prepared bundle's `run-mms-on-grace.sh` builds the solver, verifies its source
manifest, records the binary hash and environment, and calls
`scripts/verify_report.sh` with only `VERIFY_GROUPS=conv`. Each case has a
300-second timeout; all fifteen logs and exits are retained. The summary refuses
missing cases, incorrect physical scope, nonfinite errors, unfinished continuation
and an unmet final nonlinear target. Intermediate residual-floor acceptance is
reported separately. It prints adjacent-grid rates using physical spacing `2/n`,
without asserting a common asymptotic order or accepting failed accuracy gates.
Each scheme/grid must name its own collector log. The reader checks all five
continuation stages, uses only the final stage's residual, rejects duplicate
records, and keeps validation active under Python optimization. Log hashes are
included in the summary. Scheme labels still come from the collector: review the
source manifest, binary identity and launch settings before accepting the study.

Rebuild the summary from a returned result directory with:

```sh
python3 spikes/cvfem/scripts/mms_refinement_summary.py /path/to/result/study
```

The reader can be checked without running a solver:

```sh
python3 spikes/cvfem/scripts/test_mms_refinement_summary.py
```

These synthetic checks exercise acceptance and refusal; they provide no numerical
accuracy evidence. Reader changes can be applied to retained logs without reruns.

The current local diagnostic on an Apple M2 Pro is separate from Grace: unlimited
reconstruction reproduces the historical 16-cell error, decreases from 0.02422124
to 0.005869032 to 0.003718764 in volume-weighted velocity L2 at 16/32/40 cells,
and first meets the unchanged maximum-error target at 40 cells (0.008564342).
First-order upwind at 32 cells still exceeds it (0.07281847). These observations
do not validate the limiter cases, an unstructured solve or a universal order.
The corresponding Grace evidence and follow-up are recorded below.

## Grace 16/24/32 result and n=40 follow-up

Job 5077627 on Daint node `nid006551` completed all fifteen cases on 2026-10-17.
The raw files and replay live in `wip/paper/data/mms_5077627/`. Every final
nonlinear target is met and every separate accuracy target still fails.
Unlimited and clip velocity L2 rates stay near two, while Venkatakrishnan and
Darwish rates decrease across the two intervals. This supersedes the pending
Grace status above, without establishing a common asymptotic order.

Run the prepared bundle wrapper with `--n40` for one additional grid across all
five schemes. It retains the same geometry, physical inputs, eight threads,
continuation, solver settings and fixed accuracy threshold. The five cases keep
their 300-second individual limits and write a new `mms-fine-*` result directory.
No earlier run is overwritten. This checks whether the reconstructed schemes
reach the target on Grace and extends the limiter evidence; it does not promise
that first-order upwind will pass at this resolution.

Replay the follow-up with `mms_refinement_summary.py /path/to/study --grids 40`.
One grid has no convergence rate by itself. Review its source/binary/settings
against job 5077627 before combining the retained results into a refinement curve.

## Completed follow-up

Job 5077906 on Daint node `nid006550` completed the five n=40 cases. Its solver
binary is byte-identical to job 5077627, and only the grid list and output path
differ in the recorded launch settings. `wip/paper/data/mms_5077906/` retains
the raw evidence and a combined replay of all twenty cases.

All final nonlinear targets pass. The four reconstructed schemes now pass the
unchanged 0.01 maximum velocity error target; first-order upwind still fails.
The n=32-to-40 velocity-L2 rates are about 2.04, 2.07, 1.49 and 1.36 for
unlimited, clip, Venkatakrishnan K=0 and Darwish respectively. The first-order
rate is about 0.73. This completes the bounded affine refinement assessment;
it does not establish a common second order or an unstructured full-solver result.
No further run is required to report these measured conclusions.
