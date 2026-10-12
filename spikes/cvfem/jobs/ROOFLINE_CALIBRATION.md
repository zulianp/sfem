# Grace counter check before new roofline measurements

Run `roofline_calibration.sh /new/results/directory` inside one exclusive allocation
with one task and 72 physical cores, using the GNU compiler uenv. The job builds a
standalone control; it does not rebuild or change the CVFEM operators. The Python
collector uses only the standard library. It also needs `perf` and `numactl`.

The controls perform copy (`a=b`, 16 useful bytes and zero FLOPs per entry) or triad
(`a=b+3*c`, 24 useful bytes and two FLOPs per entry), with 80 million doubles per
array. Each array is larger than the Grace shared cache. All outputs are checked.
These controls use STREAM's byte convention but are not official STREAM scores.

The collector verifies 72 bound CPUs in one package/NUMA node, selects the SCF PMU
whose representative CPU is in that set, and explicitly binds memory to that node.
It records topology, PMU aliases, memory maps, the OpenMP team, commands, compiler,
sources, binary hash, every raw event row, and capture hashes. It stops on missing
commands, permissions, ambiguous mapping, wrong teams, or unsupported/multiplexed
counters. It does not modify system settings or request root access.

SCF transported bytes are `32 * cmem_rd_data + cmem_wr_total_bytes`. FP operations
are `FP_SCALE_OPS_SPEC + FP_FIXED_OPS_SPEC` (raw events `r80c0`, `r80c1`), including
speculative operations of any precision. Fused multiply-add counts as two. The
[NVIDIA counter guide](https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/measuring-performance.html)
defines these conversions. [STREAM accounting](https://www.cs.virginia.edu/stream/ref.html)
explains why useful bytes and transported bytes can differ with write allocation.

The two counters of each type form a group; coverage must be at least 99.99%.
Two nested `perf stat` commands observe the same control process. The outer command
opens the socket-wide SCF group only on its representative CPU; the inner command
opens the FP group on all 72 bound CPUs. They write separate `.scf.perf` and
`.core.perf` captures. The representative CPU selects the shared SCF PMU; it does
not restrict the memory traffic counted to that CPU. Applying the 72-CPU mask to
SCF can open duplicate events on the same PMU and cause multiplexing.

Six blocks use all permutations of 10/40/70 repeats. Copy/triad order alternates by
block; each process performs three warmups. Adjacent-repeat slopes are reported
individually through their ranges. This reduces fixed setup contributions but does
not assert exact cancellation between independent processes. The SCF interval also
includes the inner `perf` startup and output; the two counter intervals are not
identical. These are whole-process count slopes, not counters limited to the timed
loop. Multiplexed captures remain invalid and are never repaired by scaling.

Diagnostic acceptance requires every FP slope to be within 2% of a triad's known
work, traffic slopes to be positive with at most 10% spread relative to their mean,
and mean transported/useful traffic to lie in [0.85, 1.8]. These are practical
screening tolerances, not hardware error bounds or a confidence interval. The
traffic window allows streaming stores and write allocation; inspect the separate
read/write slopes before using them. A failure preserves the logs and stops.

For unstable write counts, add `--diagnose-writes` after the output directory
(or after `run-roofline-on-grace.sh` in the transfer bundle). This collects a
diagnostic, not an accepted calibration. It adds `cmem_wb_data` and `cmem_wr_data`
to the same SCF group to check NVIDIA's documented identity:
`cmem_wr_total_bytes = 32 * cmem_wb_data + cmem_wr_data`.
There are four blocks, each with a 0.2-second idle control before and after copy
and triad controls at 10/40 repeats. Mode and repeat order alternate; the four
SCF events rotate order across blocks to expose possible counter-slot dependence.
All 24 captures retain the same CPU/memory binding, nested FP capture, raw rows
and coverage checks. The idle command is `sleep`, so it measures background and
launch traffic without allocating the control arrays. It does not isolate the
benchmark's allocation, initialization or teardown individually.

The diagnostic reports total and component write bytes and their signed difference
without correcting or replacing any count. Large disagreement would implicate
counter accounting; agreement with varying idle or workload traffic would require
further isolation. Event-order dependence is evidence to investigate, not proof
of a particular driver defect. A successful collection never prints the calibration
pass marker. The full copy/triad calibration must still pass before operator runs.

Use `--diagnose-phases` to locate excess writes within the control program. This
mode collects only the four SCF events and starts `perf` disabled. The control
uses perf's documented `--control=fd` enable/disable protocol and waits for each
acknowledgement. Both acknowledgements must complete; missing or malformed replies
fail within five seconds. The launcher also limits each capture to 60 seconds.
The receiver checks all five bytes of perf's `ack\n` reply, including its trailing
NUL, so no byte is left behind for the next acknowledgement.
The numerical sweeps are unchanged.

Each process still runs the entire control with three warmups and ten sweeps, but
counts only one region: allocation/initialization (`setup`), `warmup`, `sweep`,
result verification (`check`), freeing the arrays (`teardown`), or an `empty`
enable/disable pair. Three blocks reverse phase/mode order for 36 captures total.
Thread startup, NUMA-map printing, process exit and perf's own work remain outside
the selected regions. The
empty region reports control overhead; it is not subtracted. Delayed cache
writebacks can cross phase boundaries, and separate processes cannot be added up
as an exact decomposition of one run. The summary reports raw count ranges, not
bandwidth or accepted calibration. The default and write-component diagnostic
modes retain their previous measurement windows and acceptance rules.

The phase job first runs `roofline_phase_control_test.py` against its freshly built
binary. This checks 24 gated controls (12 with split replies), four normal controls
and five protocol failure cases: bad content, wrong terminator, truncated reply,
missing reply and unknown phase. Local checks use the same script with a
local control binary; only the following perf captures validate hardware behavior.

The [perf stat documentation](https://github.com/torvalds/linux/blob/master/tools/perf/Documentation/perf-stat.txt)
describes the control descriptors and `--delay=-1` behavior.

Use `--validate-sweeps` for the full calibration after phase isolation. Job 5031716
on nid006551 completed the phase diagnostic: setup writes ranged from about
2.8 to 13.1 GB, while ten-sweep writes ranged from 5.73 to 5.77 GB for copy and
5.83 to 5.98 GB for triad. This motivates excluding setup from the measured
window; it does not establish the underlying source of the setup traffic or
explain every earlier run on another node.

The new mode runs six blocks with all permutations of 10/40/70 sweeps and
alternating copy/triad order. Two independent acknowledged control channels
enable the outer SCF collector (one representative CPU) and inner FP collector
(all 72 CPUs). Both start disabled. The program sends both enable requests,
waits for both replies, times the sweeps, then sends both disable requests and
waits again. The windows bracket the same sweep but are not identical because
switching is asynchronous. All four SCF events and both FP events are retained.

Before and after each mode/block, an independent control process opens an empty
window immediately after its three warmups. Its later ten sweeps are uncounted.
These 24 controls screen handshake overhead and traffic near the warmup boundary;
they are neither subtracted nor treated as exact estimates of every sweep's
background. Together with the 36 sweep captures, the campaign has 60 captures.
Each capture has a 60-second timeout; failure terminates the nested process group
and closes both pairs of pipes. Partial records and raw logs are retained.

The acceptance screens are declared before the new hardware run:

- Every required event must have valid counts, positive runtime and at least
  99.99% coverage. Both controllers must acknowledge enable and disable, and the
  numerical control must pass. The write-component identity must hold exactly.
- Every direct count per sweep and every adjacent-repeat slope must pass the
  existing 2%-of-triad-work FLOP tolerance and transported/useful range [0.85,1.8].
  Writes must be positive. Read traffic must be between 85% of known input bytes
  and 110% of input bytes plus one output-array read, allowing write allocation.
- Read, write and total traffic must each have at most 10% range relative to
  their mean, separately over direct samples and adjacent slopes.
- Each empty control must be at most 1% of ten-sweep useful bytes and 1% of
  ten-sweep triad FLOPs (also used for copy, whose expected FLOPs are zero).
- In each sweep capture, SCF counter runtime and average per-CPU FP counter runtime
  must each be between 99% and 110% of the loop timer. The bounds allow small
  timer differences and switching overhead while rejecting a wider count window.

These are practical acceptance screens, not a guarantee of fixed absolute
counter accuracy or complete accounting of delayed writebacks. Missing or
duplicate captures fail. No rejected sample is dropped, adjusted or scaled.
The report prints SCF/loop and FP/loop ranges and identifies every failing timing
capture with its raw durations and ratios. The control also records the elapsed
enable and disable request/acknowledgement intervals, the inner interval between
completion of enable and the start of disable, and the outer interval covering
both commands. These timestamps use the same steady clock as the loop timer and
are printed after counting stops. They help locate excess time around the loop;
they do not identify an OS scheduling cause, correct counts, or relax any gate.
Older records can still be analyzed and explicitly report unavailable switch
timestamps. New collections require the timestamp record.
The report shows direct and slope ranges and all gate outcomes. A successful
run prints `ROOFLINE SWEEP COUNTER VALIDATION PASSED`; this accepts only this
configuration on these controls. The older default whole-process recipe and
diagnostic modes remain available with their original acceptance rules.

A pass supports the selected counter configuration on these controls only. The
next operator run still needs matching terms, state-gradient cache modes, measured
traffic and timed work in one allocation. Record each limiter separately; equal
source arrays do not establish equal DRAM traffic. Recompute a bandwidth reference
from transported bytes in that same allocation, rather than scaling an old STREAM
score by an assumed write-allocation factor. Report useful-work models separately
from speculative PMU operation counts. The historical paper plot remains qualified
until those measurements are available.


Use `--diagnose-warmup` for a paired 3-versus-30-warmup diagnostic. This is
not a calibration acceptance or an operator run. Job5036950 on nid006548 failed
only the triad write-slope stability screen (11.174175% versus10%). The saved
3-warmup campaigns on nid006544, nid006545 and nid006548 show a general decrease
in written bytes per sweep as the repeat count grows. Independent processes can
also differ in placement, cache state or background traffic, so the observations
do not establish a warmup cause. They justify testing warmup length explicitly.

The diagnostic keeps array size, numerical work, kernels, counter groups, binding,
acknowledged windows and all acceptance screens unchanged. Each existing capture
is paired with the other warmup count; pair order alternates. Six repeat-order
blocks yield120 captures, including48 empty controls. Every process records its
actual warmup count, and each capture retains the requested environment separately.
Both sets are analyzed with the original screens; the report also shows writes
and times at each repeat count, write-slope component extremes, and paired warmup
differences. It never promotes either set into an accepted calibration.

The control's `CVFEM_CALIBRATION_WARMUP` setting defaults to3. Standard collectors
explicitly set3 to preserve their recipe; only this diagnostic overrides it with
30. Numerical checks run after every control. More warmup flattening the
repeat-count trend would support a settling effect, but would not identify a
hardware cause or prove suitability for CVFEM. A persistent trend/spread leaves
process/allocation effects and counter behavior unresolved. No sample is removed,
no threshold is relaxed, and job5036950 remains rejected in either outcome.

The job runs local protocol tests for both warmup counts before collecting.
Success ends with `ROOFLINE WARMUP DIAGNOSTIC COLLECTED`, not a calibration pass.

## Process and array reuse diagnostic

Job 5038251 on nid006545 compares 3 and 30 warmups. More warmup reduces
the repeat-count trend: triad writes at 10/40/70 repeats change from
601.486/587.780/583.047 to 578.995/577.328/581.243 MB per sweep. Copy also
improves, but both arms fail copy write-slope stability (10.93% and 12.09%,
against 10%). The slowest copy capture is in a different block from the
30-warmup slope extremes. Neither arm accepts a calibration.

Use `run-roofline-on-grace.sh --diagnose-reuse` with the same exclusive
72-core allocation. This diagnostic compares fresh processes against one
process and allocation reused within each five-capture block (empty control,
three repeat counts, empty control). Both arms run the same session worker,
initialize all three arrays and perform 30 warmups before every capture.
All six repeat-order permutations are retained; fresh/reuse order alternates
by block. There are 120 captures, including 48 empty controls, in 72 worker
processes. Inactive workers are closed before the other arm starts.

Every capture opens fresh perf monitors with the existing CPU/event scopes,
strict coverage and acknowledged enable/disable controls. The worker runs
separately from perf; the events are CPU/socket-wide. A waiting `cat` keeps
the monitors alive until the worker finishes, then EOF closes them. Process
IDs and array addresses are checked on every capture. Worker and monitor
process groups are terminated on failure or timeout. Commands, requests,
raw counter files, source hashes and partial records are retained.

The original work, empty-control, timing and stability screens are reported
for both arms without changing their limits. Collection ends with
`ROOFLINE REUSE DIAGNOSTIC COLLECTED`, never a calibration pass or an operator
launch. A difference would support an effect of process/allocation history;
it would not identify the hardware mechanism. Reusing virtual addresses also
does not prove that physical pages never migrate. A separate validation is
still required before paper measurements.

## Validate the reuse configuration

Job5038857 on nid006566 passes every unchanged screen in the reuse arm.
Write-slope spread is4.364381% for copy and2.498354% for triad; the fresh arm
fails copy at10.544177%. The diagnostic retains its original status. It supports
testing the reuse configuration separately, without identifying a hardware cause.

Run `run-roofline-on-grace.sh --validate-reuse` in a new exclusive72-core
allocation. It runs only the predefined reuse arm:12 workers, each holding its
arrays for one five-capture block, for60 captures (36 sweep and24 empty).
Initialization,30 warmups, repeat permutations, binding, events, control protocol
and all acceptance limits are the same as in the diagnostic. Every required
screen must pass; a failure preserves logs, returns nonzero and prints no pass.
The old diagnostic is never relabelled as this new validation.

Completion is `ROOFLINE REUSE CONTROL VALIDATION PASSED`. This validates only
the controls in that allocation. It does not launch CVFEM. Job 5038994 on
nid006544 passed these screens, with write-slope spreads of 4.668549% for copy
and 5.958412% for triad. All 60 captures and their hashes were verified.

The operator job now runs these reuse controls in its own allocation, then
measures CVFEM with retained workers and arrays. It checks the operator outputs
and counter/timing screens separately; see `ROOFLINE_OPERATORS.md`.
Grace job 5039102 on nid006550 passed both these controls and the residual
campaign. Matching Jacobian and compute-reference measurements remain pending.
