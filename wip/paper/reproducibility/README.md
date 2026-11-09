# Matched empirical roofline

The main figure uses job 5047137 on Grace nid006550: 72 cores, FP64 affine
interior operators, n=128, pack1024, Rhie–Chow on, fixed perturbed state and
coordinate-based direction. State gradients are cached; direction gradients
are included where needed. This campaign predates the Darwish coefficient
cache and must not be used as its counter measurement. It does not measure
the assembled consistent CSR/BSR matrices added in the later Jacobian study.

## Reproduce

Set `SFEM_MATCHED_COUNTER_ARCHIVE` to `pasc27-matched-counters-5047137.tar.gz`.
Its SHA256 is `edf02cf387612936773f32853e4c5d030cc68f0da2798c5314f47000f50f0389`.
From `wip/paper`, run `make empirical-roofline PY=python3`, then `make check`.
The pinned archive is also recoverable without an external download host:

```sh
git show fd4e81f1:wip/paper/data/matched_counters_5047137/captures.tar.gz > /path/to/pasc27-matched-counters-5047137.tar.gz
```

Ordinary builds use generated TeX and need no archive. The replay validates
the archive before importing captured analysis, checks 4,299 source-manifest
entries, and verifies all 492 captures (60 controls, 144 residual, 252 action,
36 DGEMM), 3,240 recorded numerical checks, raw counter files and saved reports.
It does not rerun the hardware measurements. The archive retains the exact
measurement sources, launch records, binaries' hashes, dependencies and controls;
the capture code is not installed in the production benchmark by this paper PR.

## Run identifiers and measurement protocol

Scheduler job and node identifiers are provenance, not hardware descriptions.
The roofline uses job 5047137 on nid006550. The separate accumulation table
uses job 5026072 on nid006544; its samples are not combined with the roofline.
The generated macros and archived launch records retain these identifiers.

The counter controller waits for acknowledgement that counting has started
before allowing the timed loop to run, then stops counting after the loop.
Both counter monitors bracket the same loop; their switching is asynchronous.
Empty-loop controls and the counter-runtime checks bound this overhead.
The processes and their arrays are reused across measurements so that setup
and allocation are outside the measured application windows.

Copy and triad are distinct streaming benchmarks, not repeated estimates from
one kernel. Their mean transported bandwidths are 401.2 and 411.4 GB/s,
respectively, in this allocation. The figure includes their sample variation.
DGEMM has two reported rates: 2448 GFLOP/s from speculative hardware counts
for the roofline, and 2453 GFLOP/s from useful work (2*N^3 per call) in the
platform table. Neither should be confused with the theoretical vendor peak.

## Definitions and limits

- Axes use FLOP/byte and GFLOP/s. FLOPs are speculative PMU floating-point
  operations across all precisions, not a source-level useful-work model.
  `FP_SCALE_OPS_SPEC` plus `FP_FIXED_OPS_SPEC` counts SVE, scalar and NEON
  arithmetic; fused multiply-add counts twice per active element. Integer
  indexing, loads, stores and synchronization are not part of the FLOP count.
  See the [Grace counter definitions](https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/measuring-performance.html#computational-intensity).
  Transported bytes are counted reads
  plus total writes, using the validated Grace event definitions.
- Application rates use the timed loop bracketed by both counter windows.
  Setup, assembly, validation and warmups lie outside those windows.
- Retained workers and allocations use 30 warmups before operator captures,
  repeat counts 30/60/90 in all six permutations and normal timing at 60.
  Worker-only THP disable is applied consistently, including the references.
- Copy/triad controls use 80 million doubles and repeat counts 10/40/70.
  OpenBLAS 0.3.30 DGEMM uses N=8192, three warmups and 1/2/3 calls. These
  achieved references are not strict hardware ceilings or official STREAM scores.
- The separate black solid roof uses vendor references: up to 500 GB/s and
  3.55 TFLOP/s per CPU die, the latter derived as half the advertised 7.1 TFLOP/s
  dual-die rate. Source: [NVIDIA Grace architecture specifications](https://developer.nvidia.com/blog/nvidia-grace-cpu-superchip-architecture-in-depth/).
  These values are not measurements of this allocation. The FP64 roof provides
  context rather than a strict ceiling for all-precision speculative counts;
  no percentage of vendor peak is claimed. It does not change any measured point.
- All samples remain. Counter coverage, empty controls, runtime windows,
  repeat-count stability and paired normal timing pass the original practical
  screens. No background subtraction or post-hoc sample filtering is applied.
- The main figure selects four residual arms and the equivalent lagged pair.
  Small points show every counted capture; large points are arithmetic means
  of per-capture coordinates. Reference bands show sample ranges, not confidence
  intervals. Each empirical roof follows min(B*I, P), using the copy or triad
  mean bandwidth B and the common hardware-counted DGEMM rate P. Slopes stop
  at their knees; the shared plateau starts at the first knee. The shaded
  envelope uses the smallest/largest reference bandwidth and DGEMM rates;
  it is a descriptive envelope, not a jointly measured bound.
  Full residual/action figures, sample coordinates and timing/traffic
  tables are regenerated under ignored `generated-roofline/`.
- The two lagged applications are equivalent operators. A residual and a
  Jacobian, or a consistent and lagged Jacobian, answer different questions.
  Counted GFLOP/s ratios are not automatically runtime speedups.

The matched results distinguish comparable traffic with cheaper accumulation
from avoiding matrix traffic altogether. They do not isolate a hardware
bottleneck, establish optimality, or show a full-solver speedup.

## Historical accounting

The old `figures/roofline.tex` and `roofline_macros.tex` are preserved as
historical generated inputs, but the main roofline no longer uses them.
Their captures combine separate runs and older accounting conventions; do
not combine their useful-work estimates, bandwidth references or inferred
phase costs with this matched campaign. Historical timing tables and the
separate newer precision diagnostic retain their own scope.
