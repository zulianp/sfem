# Companion counter analysis

This material accompanies the paper's matched timing, traffic and accumulation
results. It is stored with the manuscript and measurement archive, outside the
main paper's page count. It is not a separately submitted proceedings supplement.
The local review branch and this figure have not been published.

## Reproduce the figure

From `wip/paper`, with Python 3.6 or newer and a TeX installation providing
`latexmk`, `pgfplots` and Latin Modern:

```bash
make roofline PY=python3
```

On the reviewed Mac, `PY=/opt/homebrew/bin/python3` selects the approved local
interpreter. No NumPy, matplotlib, cluster access or new performance run is needed.
The Python generators use only the standard library.

The command replays the archived campaign, runs generator checks, generates the
plot sources and builds [roofline.pdf](roofline.pdf). The PDF is a build product;
the source, generated coordinates and documentation are kept in Git. Outputs are:

- [roofline-data.json](roofline-data.json): original per-capture counts and times,
  computed coordinates, reference samples, ordinary timing summaries and scope.
- `roofline-residuals.tex` and `roofline-actions.tex`: the generated plot panels.
- `roofline-values.tex`: generated numerical labels used by `roofline.tex`.

`make check PY=python3` checks the main manuscript and its 10-body-page limit.
The main paper retains the matched runtime/traffic tables and accumulation study;
it does not include this roofline as a figure.

## Evidence and measurement scope

The sole input is [job 5047137](../data/matched_counters_5047137/provenance.json)
and its hash-bound archive, `captures.tar.gz`. It contains all 492 captures and
the captured measurement/analysis sources. Generation verifies the archive hash,
source identities, raw counter files, numerical checks, CPU/event scopes, worker
lifetimes, memory policy and all four accepted reports before exporting samples.
The existing failure limits and numerical acceptance rules are unchanged.

All measurements use one allocation on `nid006550`, CPU cores 0--71 and NUMA0/SCF0.
Benchmark workers disable transparent huge pages before allocation; controllers
and global system policy are unchanged. Operators use `n=128`, pack size 1024,
FP64 arithmetic, affine geometry and Rhie--Chow coupling. State gradients are
cached. Each exact action reconstructs the direction gradients; directions are
keyed on coordinates so renumbered layouts apply the same physical direction.
Boundary/transient terms and assembly/setup cost are excluded.

Each operator has six retained worker blocks and all three repeat counts
30/60/90, giving 18 counted captures. Each capture follows 30 warmups and checks
the result before and after measurement. Copy/triad controls use six blocks with
10/40/70 repeats; DGEMM uses six workers with 1/2/3 calls. Each reference therefore
also has 18 counted captures. Ordinary timing has six independent captures per
operator with the matched middle repeat count; it is kept separate from the plot.

## Axes and reference lines

For each counted capture, define:

```
F = FP_SCALE_OPS_SPEC + FP_FIXED_OPS_SPEC
Q = 32 * cmem_rd_data + cmem_wr_total_bytes
t = the application timer's sweep-loop duration, in seconds
x = F / Q
y = F / t / 1e9
```

`F` counts speculative floating-point operations of all precisions, with fused
multiply-add counted as two operations. It is not a source-level useful-FLOP
model or a counter restricted to FP64. `Q` counts transported bytes in the selected
SCF scope, including memory-system effects; it is neither allocated storage nor
STREAM useful bytes. [NVIDIA documents the event formulas and scopes](https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/measuring-performance.html).

Both axes use the same capture's counts and timed loop. The counters' acknowledged
windows bracket that loop asynchronously; accepted runtime/window and empty-control
screens bound the discrepancy but do not eliminate it or establish absolute counter
accuracy. Empty counts are retained without subtraction. No normal-execution timer,
historical FLOP model or measurement from another allocation supplies either axis.

All 18 sample coordinates are shown as small marks. A large mark is the arithmetic
mean of the per-capture x and y coordinates. It is not a further measurement;
its y/x ratio need not equal the arithmetic mean bandwidth. The JSON retains all
coordinates and primitives. Sample spread is not a confidence interval.

For each reference capture, bandwidth is `Q/t/1e9` and DGEMM rate is `F/t/1e9`.
The diagonals are `y = mean_bandwidth * x`, one for copy and one for triad.
Their gray band spans the smallest through largest rate across both controls.
The horizontal is mean **counted** DGEMM rate, with its full sample range shaded.
The useful `2*N^3` DGEMM rate remains a separate platform-table measurement and
does not supply this line. References use the same counted-work and transported-byte
conventions as the operator points.

These are achieved references, not universal ceilings. The lines are drawn
independently; no memory/compute-bound regions, percentage-of-peak efficiency,
remaining-speedup estimate or hardware bottleneck is inferred from them. A broader
machine characterization such as the [Empirical Roofline Toolkit](https://docs.nersc.gov/tools/performance/roofline/)
has a different purpose from this single-configuration comparison.

## What the comparison supports

Packed versus atomic compares the same operator within each scheme. Only the
lagged action and its BSR product are mathematically equivalent across matrix-free
and assembled implementations. Exact Jacobians retain additional reconstruction
sensitivities and are not speedup comparisons against the lagged sparse matrix.

The runtime and traffic tables in the paper remain the evidence for performance.
In particular, the lagged matrix-free action does more counted arithmetic than
SpMV: its operation-rate ratio is larger than its time speedup. A point closer to
a reference line is not necessarily the faster way to apply an equivalent operator.
The scatter-bypass experiment supplies separate evidence for lower marginal
accumulation cost; this figure does not isolate that cost.

## Historical roofline accounting

The old [source manifest](../data/roofline_sources.json) and generated
`figures/roofline.tex` remain available for tracing the original analysis. That
figure is historical, is not included in the main manuscript and is not this
companion plot. It combines different nodes: campaign 5003779 used `nid005994`,
residual job 5006526 used `nid006545`, action job 5006527 used `nid005791`, and
STREAM 4789412 / DGEMM 4811812 used `nid006544`. Traffic archive 5005775 lacks
hostname, source/binary identity and CPU/PMU mapping; later hashing cannot recover
those missing records.

Historical first-order points omit Rhie--Chow; higher-order traffic covers Darwish
with coupling. The element-work models omit reconstruction and Rhie--Chow work,
and SpMV traffic was modelled. Two process-wide captures need not cancel setup,
warmup or background traffic. Historical intensity values and their matrix-free/
SpMV ratios therefore do not establish full-operator intensity, efficiency or
accumulation cost. Their data are preserved, but they do not enter the new plot.
Qualifications on the historical timing results that remain in the paper are
retained. See the [earlier audit](../README.md#roofline-audit-11-october-2026).
