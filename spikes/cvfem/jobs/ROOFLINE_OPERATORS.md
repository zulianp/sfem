# Grace residual counter measurements

Run `roofline_operator.sh /path/to/cvfem_hex8_ns_upwind_bench /new/results/directory`
in an exclusive one-task, 72-core Grace allocation with the GNU uenv active.
The transfer bundle's `run-roofline-operators-on-grace.sh` verifies source hashes,
rebuilds the FP64 benchmark and calls this job. The user launches the allocation.

This first operator campaign covers only the steady affine FP64 interior residual:
`n=128`, pack size 1024, Rhie--Chow scale 1, density 1, viscosity 0.01 and state
perturbation 0.017. Atomic and packed layouts each run first-order convection
and Darwish (limiter 3). State pressure/velocity gradients are cached before
measurement. Repeated sweeps include their usual accumulation and buffer handling.
No boundary or transient terms, live-vector churn or scatter-bypass probes enter
the window. These are fixed-state operator measurements, not a nonlinear solve.

The job first runs the fixed copy/triad reuse validation in the same allocation.
It must pass all existing gates. The operator collector verifies its hashes,
reproduces its report and requires identical topology/job metadata. Transported
bytes divided by the timed control loop provide new bandwidth references with
sample ranges, without converting a historical useful-byte STREAM result.
They are achieved rates on these controls, not an official STREAM score or a
proven upper bound. A new compute ceiling and Jacobian measurements are not part
of this first campaign; these residual data alone do not replace the full plot.

`--roofline-window session` retains the mesh, state, cached gradients and production
scratch through six captures in each scheme/layout/block. Only one worker is alive
at a time. Each request selects `normal`, `sweep` or `sweep-empty` and its repeat
count; all captures get 30 warmups. Worker PID and the velocity-x, pressure and
residual-x buffer addresses are checked on each response. These virtual addresses
do not establish fixed physical page placement. The single-window CLI remains
available for local regression checks.

Each request enters the checked residual-only path.
It calls the existing production sweep. All four equations are compared against
the atomic kernel, in the same node numbering, before warmup and after timing.
Every entry must be finite, the reference vector must be nonzero, and each field's
relative infinity error must be at most 1e-10. This checks layout and timing-path
equivalence; it does not independently validate the discretization. Temporary
verification vectors are freed before warmup. The state gradients are the same
stored gradients in the production and reference call.

The calibration and benchmark share the acknowledged perf control implementation.
Two channels enable SCF on its representative CPU and FP counters on all 72 bound
CPUs. Both replies arrive before the loop timer starts; both disable requests
follow its end. Setup, 30 warmups, verification and reporting are outside.
An empty window opens after warmup and closes before the uncounted loop. A normal
run has no perf process or counter switching. Numerical kernels are unchanged.

Six blocks use every permutation of 10/40/70 repeats and alternate scheme/layout
order. Each layout completes its six-capture block before the other starts.
Normal (40 repeats) runs alternate before/after each block's counted runs and use
the same worker with no active perf monitors. There are 72 counted sweeps,
48 empty captures and 24 normal captures in 24 workers. Each capture
has a 180-second timeout; partial logs survive failures and nested processes are
terminated on timeout. All samples are retained; no automatic retries, subtraction,
rescaling, outlier removal or tolerance adjustment takes place.

Acceptance screens, fixed before hardware execution:

- Existing strict event coverage (at least 99.99%), acknowledged windows, finite
  positive loop duration, correct 72-thread team and numerical checks must pass.
- Total write bytes equal 32 times write-back beats plus other write bytes.
- SCF/loop and average per-CPU FP/loop runtime ratios each lie in [0.99,1.10].
- Each empty byte/FP count is at most 1% of the minimum corresponding ten-sweep
  count for that scheme/layout. This screens nearby overhead without subtracting it.
- Read, write, total traffic and speculative FP counts each have at most 10%
  range/mean spread across direct per-sweep samples and, separately, adjacent-repeat
  slopes. Nonpositive slopes fail. No claim that layouts have equal FP work or
  memory traffic is built into this check.
- Mean paired timing overhead for counted versus normal 40-sweep runs is within
  +/-10% for each scheme/layout. Its full range is also reported. Passing this
  screen does not prove identical scheduling or cache behavior.

The report keeps ordinary timings separate from counter-window timings. Its
arithmetic intensity is speculative PMU FP operations per transported byte;
these operations include all precisions and must not be relabelled as a useful
FLOP model. Failed checks keep the campaign unaccepted. Passing these practical
screens does not establish absolute counter accuracy, a bottleneck, universal
performance fractions, or solver acceleration. Local mock-controller and parser
checks validate the measurement machinery; Grace execution is still required.

The reuse controls passed separately in Grace job 5038994 on nid006544. That
result supports this recipe; it is not substituted for the controls in the new
operator allocation. The old fresh-process campaign remains unaccepted.

Grace job 5039102 on nid006550 passed the same-allocation controls and residual
campaign. The archived raw evidence is in
`wip/paper/data/residual_counters_5039102/`. This validates the stated residual
scope; matching Jacobian and compute-reference measurements remain pending.
