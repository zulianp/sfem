# Matched action and compute measurements

Run `roofline_remaining.sh` inside one exclusive 72-core Grace allocation. It
requires the checked benchmark, the DGEMM reference binary and the actual LP64
OpenBLAS shared library selected for that binary. It repeats the copy/triad
controls in this allocation before collecting residual, action and DGEMM data.
Any failed control, correctness check or acceptance screen stops acceptance.

The seven action arms are first-order and Darwish exact Jacobian actions with
atomic and packed accumulation, first-order lagged actions with both layouts,
and the assembled first-order lagged sparse product. All use FP64, affine
interior cells, Rhie--Chow enabled, a fixed perturbed state and one fixed
direction keyed on coordinates, with order-independent fingerprints checked
across layouts. State gradients are cached; each exact action reconstructs the
direction gradients. Boundary and transient terms and basis-vector churn are
excluded. Only lagged action versus lagged SpMV applies the same linear operator.

Each action is checked before and after every capture against an atomic
reference that reconstructs direction gradients independently. Six workers per
arm retain their mesh, state and scratch arrays. Each worker performs normal,
empty and three counted windows, with 30 warmups before every window. All six
repeat-order permutations of 30, 60 and 90 are retained. Normal timing is paired
with the 60-apply capture. Empty counts are saved without subtraction.

After rejected job 5042788, the matched residual/action campaign uses these
longer windows to test the systematic short-window write-count deficit. In that
job, packed Darwish direct writes had a 10.499% spread, exceeding the unchanged
10% limit; its write slopes had a 5.351% spread. The hardware cause is not
established. This is a prospective protocol change, not a reclassification of
that failed run. The old 10/40/70 default remains available for reproducing old
records. Each new collection records its selected repeat counts, retains every
sample and prints component spreads. Calibration and DGEMM windows are unchanged.

The counter checks retain the residual campaign's scopes and practical limits:
full event coverage, the write-component identity, counter windows between 0.99
and 1.10 times the timed loop, empty counts below one percent of the shortest
capture, at most ten percent spread in direct and adjacent-repeat slopes, and
at most ten percent mean instrumentation overhead. These screens are not an
absolute counter-accuracy guarantee. Every sample, failed campaign and command
is retained; no sample deletion or counter adjustment is allowed.

The DGEMM result is an empirical compute reference for the selected library,
matrix size and thread configuration. Report useful `2*N^3` operations and
speculative PMU operations separately. It is not a strict compute ceiling for
the CVFEM instruction mix. Neither proximity to the copy/triad rates nor an
arithmetic-intensity plot alone identifies a bottleneck or proves solver speedup.

Use the same-allocation residual captures for comparison with these actions.
Keep the previously accepted residual campaign as independent evidence, rather
than silently combining different allocations into one measurement.

## SpMV timing diagnostic after job 5043387

The longer-window residual campaign passes on nid006549. The action campaign
passes all numerical and counter checks but fails the SpMV timing comparison:
the paired mean difference is -12.294%, outside the unchanged +/-10% limit.
Two ordinary SpMV windows take 29.434 and 29.292 ms/apply; the other four take
18.713--19.355 ms/apply. Counted windows take 18.562--19.384 ms/apply. An empty
control's uncounted loop also takes 36.283 ms/apply. The cause remains unresolved.
All samples remain recorded and the campaign remains rejected; the reported
3.468 ratio is not accepted performance evidence. DGEMM collection was not reached.

`roofline_spmv_timing.sh` collects a diagnostic on the same n=128 matrix and
coordinate-keyed direction. Eight workers retain their matrices and arrays;
each performs plain/traced normal/counted windows in a rotating order, then
the reverse order. Every capture retains 30 warmups, 60 applies and the original
before/after reference checks. There are 64 captures in total, 16 of each variant.

Traced windows save every warmup/apply duration after counters stop. The sum of
apply durations must equal the reported loop duration. These times include the
extra timestamp and storage work; plain windows retain their original loop.
Both variants are required to inspect tracing effects. Counts and controller
coverage are retained, but this diagnostic omits calibration and cannot accept
a performance campaign, change limits or establish a hardware cause. It can
show whether slow windows contain isolated long applies or broader slowdowns.
The existing bundle wrapper selects it with `--diagnose-spmv`; ordinary invocation
keeps the full campaign and all its original acceptance checks.

## Resource diagnostic after job 5044163

Job 5044163 on nid006555 retains all 64 captures and passes all 512 numerical
checks. Most traced applies take about 19 ms. Slow windows contain bursts of
three or four long applies, reaching 220--227 ms, in both normal and counted
variants. Similar bursts occur during warmup. This localizes the variation but
does not identify an operating-system, allocation or hardware cause. The failed
performance campaign remains rejected, and no samples are discarded.

The bundle wrapper's `--diagnose-spmv-resources` mode retains the same capture
order, arrays, reference checks and counter scopes. Traced variants additionally
sample `getrusage(RUSAGE_SELF)` around each apply: process user/system CPU time,
minor/major faults and voluntary/involuntary context switches. CPU time sums all
process threads, including any worker spinning, so it may exceed wall time.
Sampling overhead is included in
the trace; plain variants retain the ordinary loop with no per-apply usage calls.
Resource deltas are formed before converting CPU timestamps to floating point.
All records are saved, with intervals over 40 ms printed for inspection. That
threshold only controls the display and never selects performance samples.

Before and after each complete capture, the collector saves permitted process
status, NUMA mappings, system pressure and CPU-cgroup controls/statistics, including
visible ancestor groups. Missing or inaccessible files are explicitly recorded.
These snapshots cover verification, warmup and the timed loop together; ancestor
counters and system statistics may also include other work. Their changes cannot
be assigned to a particular slow apply or treated as proof of a cause. No system
setting is changed. See the [getrusage definitions](https://man7.org/linux/man-pages/man2/getrusage.2.html)
and [CPU cgroup documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html).

Both diagnostic modes leave ordinary full-campaign invocation and its acceptance
limits unchanged. A completed resource diagnostic does not validate performance;
it supplies observations for choosing a subsequent controlled experiment.

## Process THP comparison after job 5046131

Job 5046131 on nid006550 passes all 512 numerical checks. Across the 2,880
traced warmup/apply intervals, 15 exceed 40 ms. Ordinary intervals have a
19.301 ms median and 1,111.894 ms of aggregate process CPU time; slow intervals
have a 127.407 ms median but only 1,020.052 ms of CPU time. The added elapsed
time is consistent with waiting. No major faults were observed in those traces.
All 896 comparable CPU-throttling counter deltas are zero, the visible cgroups
have unlimited quotas, and automatic NUMA balancing is disabled.

Transparent huge pages (THP) are enabled with the system `always` policy.
The global `thp_collapse_alloc` counter increases in 13 whole captures, including
all seven whose measured average exceeds 23 ms/apply. Six other captures also
have increases. These are system-wide counters around verification, warmup and
measurement together: they neither identify a particular mapping nor establish
that a collapse caused an individual slow apply. The association motivates a
controlled policy experiment, not a correction of the accepted paper results.

`--diagnose-spmv-thp` compares the unchanged default policy with THP disabled
only for each benchmark worker using Linux `PR_SET_THP_DISABLE` before `exec`.
The [Linux interface](https://man7.org/linux/man-pages/man2/PR_SET_THP_DISABLE.2const.html)
preserves this setting across `exec`. The launcher checks the initial state and
readback; the collector also checks `THP_enabled` in the live worker before
and after every capture. Missing or unexpected states stop collection. No global
setting is changed. The parent collector and perf monitors keep their policy.

Eight paired blocks alternate which policy runs first. Each policy retains the
same eight plain/traced normal/counted windows per block, yielding 16 workers
and 128 captures. Numerical checks, mesh, direction, affinity, warmups, repeats
and counter scopes are unchanged. All captures are retained, including slow
ones. The two policies use separate processes and allocations; physical pages
are not fixed between arms. Process `smaps_rollup`, PMD huge-page size and
khugepaged progress are also saved where readable; missing observations remain
explicit. Whole-process huge-page totals include all mappings, not just SpMV.

If baseline bursts recur and disabling THP removes them, this supports a process
THP policy effect. It does not identify a specific kernel call. If neither arm
shows bursts, the result is inconclusive about their cause. No policy is adopted
automatically. Any change to a performance recipe would require a separate
same-policy calibration and matched operator campaign; no failed run is rescued.

## Matched campaign after the THP comparison, job 5046420

The paired run on nid006549 retains all 128 captures and passes all 1,024
numerical checks. With the default policy, seven of 960 traced warmups and
19 of 1,920 traced applies exceed 40 ms; the longest apply is 235.572 ms.
With THP disabled, none exceed 40 ms; the longest apply is 23.398 ms.
Ordinary disabled-policy windows take 19.112--19.357 ms/apply. Process
AnonHugePages is zero in every disabled-policy snapshot and nonzero in every
default-policy snapshot. This controlled comparison supports a process THP
policy effect in this allocation. It does not identify a particular kernel call
or establish a universal performance improvement. Every sample is retained.

The bundle wrapper's `--no-thp` mode now selects a separate full campaign.
Copy/triad calibration, atomic and packed residuals, exact and lagged actions,
SpMV and DGEMM all disable THP in the benchmark worker before allocation.
Controllers and system settings retain their existing policies. Each worker
must confirm the launch policy; live THP_enabled is checked at readiness and
before and after every capture. These records and memory-policy.json are saved
with the capture hashes. Missing checks, mismatched worker identity or
calibration under another policy stop acceptance.

The original invocation without this option is unchanged. The new mode keeps
all numerical, counter and timing screens, repeat orders and samples. It runs
fresh calibration in the same allocation and does not reuse the earlier default
policy reference. The diagnostic is not an accepted action/SpMV ratio: only a
passing full campaign can supply that comparison and the DGEMM reference.
