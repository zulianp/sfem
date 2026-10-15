# PASC27 paper: a packed mesh format for deterministic, atomics-free matrix-free operators

Target: PASC27, **10 pages excluding references, ACM `sigconf`, deadline 12 December 2026 (no
extensions)**. Domain track: Computational Methods and Applied Mathematics. ORCID is mandatory
for every author and publication is open access.

```
make           build main.pdf
make figures   regenerate every figure and table input from the measurements
make selftest  run the generators' own checks
make check     the submission gates (clean build, no undefined refs, page limit)
make roofline  build the empirical roofline in the companion material
```

The [companion counter analysis](reproducibility/README.md) documents the current
roofline, its scope and reproduction commands. It is outside the main paper;
the manuscript retains its matched timing/traffic tables and accumulation study.

## The rule this directory is built around

**No measured number is typed into `main.tex`.** Everything arrives through `\input{}` of a
generated file, so a re-measurement propagates and a stale number cannot survive `make figures`.
If a number needs to appear in prose, it is emitted as a `\newcommand` by the generator that
owns it (see `figures/packed_counts.tex`).

## Scope, as agreed

- **Grace GH200 only.** One architecture; the paper says so rather than implying generality.
- **Operator application only** --- residual and Jacobian action. No solver-level section.
- **Single-node bitwise determinism, packed vs atomic.** No MPI decomposition invariance.

## Generators

| script | produces | depends on |
|---|---|---|
| `python/packed_format_figures.py` | `figures/packed_{decomposition,id_space,reduction,scatters}.tex`, `figures/scheme_passes.tex`, `figures/packed_counts.tex` | nothing --- it reimplements the format's documented rules and draws the result |
| `python/perf_figures.py` | `figures/{throughput_size,scaling,packsize,roofline}.tex`, `tables/{throughput,ladder,footprint,convho,jacfair}.tex` and their macro files | the campaign CSVs and per-measurement `.out` files below |
| `python/matched_counter_figures.py` and `python/matched_roofline.py` (called by `perf_figures.py`) | matched tables/macros and `reproducibility/roofline-{data.json,residuals.tex,actions.tex,values.tex}` | verified job 5047137 archive |

Both are stdlib-only and carry `--selftest`, following the repository's report generators: the
Alps uenv has no numpy or matplotlib, and a generator that cannot run where the measurements run
goes stale.

`packed_format_figures.py` is a *model*, not a drawing. It builds an **unstructured** mesh --
jittered points, Delaunay-triangulated by Bowyer-Watson, ordered along a Hilbert curve -- then
reimplements the ownership rules from `spikes/cvfem/docs/PACKED_FORMAT.md` on it. Unstructured and
Hilbert both matter to what the figure can show: on a grid the packs come out as rectangular blocks
and read as a decomposition somebody chose, when a pack boundary is really wherever a contiguous
range of the element order ends; and under a Morton order a pack is several disconnected shards,
which is true of Morton but is not what `smesh::SFC` does and is misleading as a picture of a
subdomain. The point set comes from an explicit LCG rather than `random`, so regeneration is
byte-identical without depending on the interpreter. Its selftest asserts the format's documented
invariants against that model --- owned ranges partitioning the index space, non-shared indices preceding
shared ones, ghost lists deduplicated, each reduction destination appearing in exactly one row,
every staged entry consumed exactly once. So the figures cannot disagree with the contract they
illustrate. It independently reproduces the reordering precondition too: mean nodes per pack
25.0 space-filling against 34.0 lexicographic, the same direction as the 2735 against 4373
measured on the real three-dimensional mesh.

## Measurement program

Run on one Grace node at 72 threads with `OMP_PROC_BIND=true`. **Every comparison lives inside a
single allocation**; node-to-node variation here is 5--11%, and a change already shown neutral
by a controlled comparison once read 5.7--6.5% slower when its two sides landed on different
nodes. `python/cvfem_kernel_report.py` now refuses to build a report from a CSV spanning
several hosts, and the campaign jobs stamp their output directory with the job id so a rerun
cannot append to another node's rows.

| # | measurement | job | status |
|---|---|---|---|
| M1 | throughput, 5 sizes x {packed, atomic, colored, store} x {residual, jac\_action, assemble, bsr\_apply f64/f32} | `jobs/camp_full.sbatch` | **done**, 4789238 (665 rows, single host nid006398) -> `data/campaign_grace_4789238.csv` |
| M2 | term-completeness ladder, both layouts | in the campaign config list | **done**, with M1 -> `tables/ladder.tex` (2.96x bare falling to 2.06x complete) |
| M3 | bitwise determinism by layout, repeats and thread sweep | `jobs/det_layout.sbatch` (new) | **done**, 4789239 -> `data/det_layout_4789239.out` -> `figures/determinism_macros.tex`; reported in prose, the table was dropped |
| M4 | thread scaling 1--72, three layouts | `jobs/thread_scaling.sbatch` (new) | **done**, 4789282 -> `data/tscale_4789282.out` -> `figures/scaling_res.tex` |
| M5 | pack-size sweep at two problem sizes, with the pack count | `jobs/packsize_paper.sbatch` (new) | **done**, 4789448 -> `data/packsize_4789448.out` -> `figures/packsize.tex` |
| M6 | STREAM triad, for the roofline bound | `jobs/stream.sbatch` (new) | **done**, 4789412 -> `data/stream_4789412.out` -> `figures/stream_macros.tex` |
| M7 | space-filling-order ablation (`--no-sfc`) | in `jobs/packsize_paper.sbatch` | with M5 |
| M8 | footprint: packed bytes/dof vs BSR | derived from the campaign CSV | **done** -> `tables/footprint.tex` |
| M12 | mesh footprint, standard vs packed, array by array, pack sizes 512--8192 | `cvfem_hex8_ns_upwind_bench --mesh-footprint` (new mode) | **done**, Grace login node -> `data/meshfoot_grace.out` -> `tables/footprint.tex`, `figures/meshfoot_macros.tex` |
| M9 | sustained fp64, the roofline's compute roof | `jobs/peak_fp64.sbatch` (new) | **done**, 4811812 -> `data/peak_4811812.out` -> `figures/roofline.tex` |
| M13 | **measured DRAM traffic** per apply, residual and Jacobian action, packed and atomic | `jobs/perf_hex8_alps.sbatch` with `PERF_GROUPS=scf`, at two repeat counts | **done**, 4816010 (repeat 10) + 4816011 (repeat 40) -> `data/dram_4816010_4816011.out` -> `figures/roofline.tex` |
| M10 | higher-order deferred-correction flux: four limiter arms x {generated, hand-written} x {packed, atomic}, plus Rhie--Chow and the hand-vectorised variant that lost | `jobs/conv_ho_bench.sbatch` (new) | **done**, 4814703 (nid006545) -> `data/convho_4814703.out` -> `tables/convho.tex` |
| M11 | like-for-like Jacobian action against the assembled matrix, exact and lagged | `jobs/jac_fair.sbatch` (new) | **done**, 4812322 -> `data/jacfair_4812322.out` -> `tables/jacfair.tex` |

### Re-measured on 2026-10-08, after the reconstruction moved onto the Jacobian

The deferred correction's reconstruction now takes its node-to-centroid vectors from the affine
element Jacobian's edge columns instead of staging node coordinates and summing twelve centroids
per lane (`cvfem_hex8_defcor_ref_increments`; the A/B is `spikes/cvfem/perf/ho_ref_increments_ab_grace.txt`:
every higher-order residual arm 16--26% faster, the Jacobian arms within the 5% band, every
first-order row unchanged). Every measurement that reaches a higher-order kernel was rerun on the
tree carrying it, and the generator picks the newest file per prefix, so these are the inputs now:

| measurement | job | file |
|---|---|---|
| M1 campaign | 5003779 (nid005994, 750 rows) | `data/campaign_grace_5003779.csv` |
| M4 thread scaling | 5003759 | `data/tscale_5003759.out` |
| M5/M7 pack size and ordering | 5005778 | `data/packsize_5005778.out` |
| M10 convective schemes, f64 and f32 | 5006526 | `data/convho_5006526.out` |
| M11 fair Jacobian comparison, f64 and f32 | 5006559 | `data/jacfair_5006559.out` |
| M13 DRAM traffic | 5005775 | `data/dram_5005775.out` |
| Jacobian per scheme (fig:jacho), f64 and f32 | 5006527 | `data/jacho_5006527.out` |
| f32 against f64, every layout (ladder f32 group) | 5006525 | `data/f32_5006525.out` |
| exact higher-order Jacobian breakdown | 5005769 | `data/hoexact_5005769.out` |
| instruction mix | 5003851 | `data/kmix_5003851.out` |
| residual accumulation diagnostic | 5026072 (nid006544) | `data/scatter_5026072/` |
| checked residual traffic and work | 5039102 (nid006550) | `data/residual_counters_5039102/` |
| matched residual/action/SpMV and compute references | 5047137 (nid006550) | `data/matched_counters_5047137/` |
| higher-order size sweep | 5005800 | `data/hosize_5005800.out` |

The rows naming jobs 5005763-5005800 were measured again after the limiter derivatives' rounding
band was fixed (spikes/cvfem/perf/f32_vs_f64_grace.txt): every job that runs the exact
higher-order Jacobian, which that fix costs 2-3%. Every sentence that reads those macros was
checked against the new values and still holds.

The rows marked "f64 and f32" run the double build and the single-precision build
(`cvfem_hex8_ns_upwind_bench_f32`, every field and kernel in float) in one allocation, order
alternating per pass, so the light f32 segments stacked on the bars of fig:convho and fig:jacho and
the f32 groups of tab:convho and tab:jacfair are each against their own f64 neighbour. The ladder's
f32 group cannot be: the campaign behind its f64 group runs 31 minutes, so its f32 group is the best
pass of `jobs/f32_vs_f64.sbatch` (5006525) and the caption says it is a separate allocation. A first
jac_fair run on the normal partition (5006528, nid005482) was discarded: its lagged action read
1182 MDOF/s while the four controls of the identical operator read 1232-1296, and its f64 rates sat
8-10% below every other run. The debug-partition rerun's lagged action and controls agree within 1%.

Three things the rerun turned up that are not about the kernels:

- `jobs/camp_full.sbatch` failed in its first seconds on a tree made with `git archive`, because
  `scripts/layout_campaign.sh` is tracked without its executable bit. Restoring the bit and
  resubmitting (5003779) is what produced the campaign above; the bit should be fixed in git.
- In the reviewed source, the instruction-mix job's `--kernel-only` flag has no effect
  on the residual path. The saved log does not identify its source revision, so its
  paired rates do not establish scatter cost. The job no longer runs that
  comparison, and the generator omits its row and macros. Historical raw logs are retained;
  their instruction counters and ordinary throughput remain separate measurements.
- The thread-scaling rows for the standard and element-coloured **Jacobian action with the
  Rhie--Chow term** fell from 510 to 335 and from 587 to 438 MDOF/s against the 2 October run
  (4967671). An interleaved check of the reference tree (9b62d097f, the commit the previous
  measurements were taken on) against the new one puts both at 340 and 437, so the fall
  predates this change and lies between 2 and 6 October; the throughput gate has no atomic
  Rhie--Chow Jacobian row, which is how it passed. Not chased here; recorded so the scaling
  figure's standard-layout Jacobian curve is read against the right cause.

### Provenance note, now resolved

`data/peak_4811812.out` was transcribed from the job's printed output during an Alps outage --
40 fetch attempts over ~35 minutes failed -- and the file carried a header saying so. Alps
returned, the file was fetched, and the measured lines match the transcription exactly
(PEAK_FP64_GFLOPS 2305.3). The file is now the fetched copy and the caveat is withdrawn.

## Roofline audit (11 October 2026)

This section records the historical audit. The replacement measurement campaign
is now accepted and supplies the [separate companion figure](reproducibility/README.md).
The old `figures/roofline.tex` remains historical and is not included in the paper.

The historical plot combines different nodes. `data/roofline_sources.json` now pins
six input files and their hashes; `tables/roofline_sources.tex` records the jobs and
hostnames. The traffic file has no hostname, raw event rows, source SHA, binary hash,
or CPU/PMU mapping. Hashing the archive now preserves it; it does not recover that
missing provenance. No older DRAM row is silently substituted for a missing new row.

The first-order points are bare kernels without Rhie--Chow. The higher-order traffic
was captured for Darwish with coupling, so only that limiter is plotted. Its action
predates the state-gradient cache correction. The FLOP model counts element work,
including a correction increment, but omits separate reconstruction and RC work.
Both axes now say modelled work; the plot cannot establish full-operator intensity.

NVIDIA documents read data beats times 32 plus write bytes. A 2:1 ratio of read
beats to accesses is not an independent calibration of the unit. Subtracting two
process-wide captures estimates a slope; setup and background traffic need not
cancel exactly. STREAM useful bytes and SCF transported bytes can differ because
of write allocation. The earlier percentages and bottleneck conclusions have been
removed. DGEMM remains a benchmark reference rather than a universal upper bound.
The documented GH200 nominal 500 GB/s is retained in the hardware table, while the
unsupported 3.55 TFLOP/s specification and dashed vendor roof are removed.

The counter-calibration job was the first required Grace gate before collecting
replacement operator points. Raw historical logs and ordinary throughput figures
remain unchanged. Remeasurement must record the selected node, CPU/memory binding,
PMU instance, event coverage, binary/source hashes, and each operator's dispatched
terms and cache mode. Use the same transported-byte convention for the bandwidth
reference and operator points; report PMU operations separately from the source
FLOP model, since the PMU events count speculative operations of all precisions.

## Accepted residual counter campaign (12 October 2026)

Job 5039102 on nid006550 passes the fixed reuse controls and the actual residual
campaign. `data/residual_counters_5039102/` contains a hash-bound archive of the
public raw counts, requests, outputs, source records and patch chain.
`python/residual_counter_figures.py`, called by `make figures` and `make selftest`,
checks the archive and captured analysis, reconstructs every count and numerical
check, and reproduces the acceptance summaries before emitting the table/macros.
It uses `python/roofline_operator_5039102.py` and the captured helpers in
`python/counter_sources_5039102/`, checked byte for byte against the archive,
so later measurement recipes cannot change this campaign's acceptance.
The source verification also checked 4,273 bundle entries, 7 operator and 11
calibration source hashes, 144 operator records and 60 controls. All 1,152
per-field before/after checks pass; maximum relative error is 4.145e-16.

Scope is fixed-state FP64 affine interior residuals at n=128, pack=1024, 72 threads,
Rhie--Chow on, first-order/Darwish, cached state gradients, without boundary or
transient terms. Each scheme/layout uses six worker blocks with retained arrays,
30 warmups before every capture and all six orders of 10/40/70 sweeps. A normal
40-sweep capture alternates before/after each block; empty controls bracket its
counted captures. Normal samples have no perf monitors. All acceptance limits
are unchanged; no samples or background counts are subtracted.

Normal atomic/packed times are 9.624/4.960 ms (1.940x) for first-order and
15.137/10.659 ms (1.420x) for Darwish. Mean transported bytes change by +3.75%
and -3.11%, respectively. Comparable traffic alongside these speedups rules out
a large reduction in transferred bytes as their explanation in this fixture;
it does not isolate the hardware bottleneck. The earlier controlled accumulation
experiment is independent evidence for the benefit of the accumulation method.
The reference rates use transported bytes; the work counts are speculative PMU
operations, not a useful-FLOP model. No Jacobian, compute ceiling, solver result
or complete replacement roofline is accepted by this campaign.

## Accepted matched campaign (13 October 2026)

Job 5047137 on nid006550 completes the same-allocation copy/triad, residual,
cached exact/lagged action, equivalent SpMV and DGEMM campaign. All 492 captures,
4,089 capture-file hashes, 3,240 numerical checks and 4,299 bundle source checks
are retained and verified. The archive includes captured measurement sources,
requests, raw counters, policy observations, summaries and build provenance.
The measurement sources match the reviewed checkout; every summary reproduces.

`python/matched_counter_figures.py` checks the archive and runs
`python/verify_matched_counters.py` in an isolated process using the captured
analysis. It reconstructs raw counts and numerical checks, verifies retained
worker identities, coordinate direction, CPU/PMU scopes and THP policy, and
reapplies every acceptance screen before generating tables/macros and exporting
the per-capture samples for the companion roofline. Corrupted
coverage, incomplete records, changed policy and wrong-direction controls are
rejected. The older accepted residual evidence remains independently checked.

The benchmark workers disable transparent huge pages before allocation. This
applies equally to copy/triad, both matrix-free layouts, SpMV and DGEMM; parent
collectors, perf monitors and global settings remain unchanged. Job 5046420's
paired diagnostic reproduced long pauses under the default policy and none with
THP disabled. It motivated this prospective, separately calibrated campaign;
earlier failed campaigns remain rejected. Every sample and the original limits
are retained. This does not identify a particular kernel call or establish a
universal benefit from disabling THP.

FP64 affine interior scope: n=128, pack=1024, 72 threads, RC on and a fixed
perturbed state. State gradients are cached; exact actions reconstruct direction
gradients on each apply. The common direction is keyed on coordinates. No boundary
or transient terms, assembly cost or solver speedup is measured. Six blocks per
arm retain all 30/60/90-apply permutations and 30 warmups; normal time uses 60
applies. Calibration retains 10/40/70 sweeps. DGEMM uses N=8192, three warmups,
six retained workers and all 1/2/3-call permutations, with normal time at two calls.

Ordinary packed speedups over atomic are 1.97x/1.44x for first-order/Darwish
residuals and 1.85x/1.86x for their exact Jacobian actions. Equivalent packed
lagged action/SpMV times are 6.503/19.368 ms: 2.978x. Their transported bytes are
1,277.5/7,720.7 MB per apply, not their storage footprints. SpMV's paired mean
instrumentation overhead is -0.030%; its ordinary range is 19.339--19.399 ms.
The rejected 3.468x ratio from job 5043387 is not used.

Copy and triad achieve 401.2/411.4 transported GB/s. Runtime-identified LP64
OpenBLAS 0.3.30 uses 72 threads and achieves 2,453 useful GFLOP/s in ordinary
DGEMM; counted speculative work is 2,448 Gop/s. These are empirical references,
not strict CVFEM ceilings or a proof of a bottleneck. Ordinary sample ranges are
reported separately from counted throughput; no confidence intervals are claimed.

The draft now uses this matched campaign for the main residual and action tables.
Historical FP32/colouring and rebuilt-state results remain in their original data
and generated artifacts, with their stated scope. The manuscript remains a local
proposal for Patrick's review; no author, submission or publication decision is
made by this update.

## Claims discipline

These are written down because each has already cost something.

- **Quote the completeness ladder, not the bare-kernel ratio.** The bare element kernel gives
  3.07x and the full solver operator 2.07x; a bare comparison overstates the margin by roughly
  1.5--2x. See `spikes/cvfem/docs/kernel_prose/30_layout_margin.md`, including its note that one
  row there is an artefact of the benchmark building no pack for the atomic layout.
- **The determinism claim is scoped to one-thread-per-pack execution** and the paper says so.
  A GPU implementation that splits a pack across threads and accumulates with atomic addition is
  *measured* to be bit-unstable; only an implementation that also orders the in-pack
  accumulation is reproducible.
- **Colouring measured bit-reproducible too (job 4789239), and the paper says so.** The
  determinism advantage is over *atomics*, not over colouring: both avoid atomics and both fix
  a summation order. What separates packed from colouring is throughput. Two caveats stay
  attached: a colouring's reproducibility depends on the colouring being constructed
  deterministically, which is a property of the construction rather than the layout; and
  agreement over seven runs is evidence, not proof.
- **Fingerprints compare only within a layout.** Packing renumbers the mesh, so cross-layout
  fingerprint comparison measures the renumbering. Operator equivalence is a separate question,
  established by matching nodes on coordinates.
- Every timing carries its degree-of-freedom count; standard and packed always appear side by
  side; a run that did not converge is a floor, not a result.
- **Assembly is reported as the case the format loses**: colouring beats packing 95 to 56 MDOF/s.
- **The mesh footprint is measured, and it is 85% of standard, not 60%.** An earlier arithmetic
  model said 81% and was wrong for a structural reason: it had to guess the two quantities that
  decide the answer -- how many ghost entries the packs produce and how many rows the reduction
  graph has -- and neither has a closed form, because both depend on how the pack boundaries
  happen to cut the mesh. The bench now reports each array's own `nbytes()`, the model is gone,
  and the generator's selftest pins the three totals. The measured decomposition at the default
  pack size, per dof: connectivity 3.91 against the standard mesh's 7.82 (the 16-bit index
  halving it exactly), ghost list 0.31, reduction graph 1.41, `node_map` 1.00 -> 6.64 total,
  **85%**. Two items in that are avoidable and the paper names both: `node_map` is never read by
  the apply (**72%** without it), and the reduction graph's two offset arrays are `ptrdiff_t`
  while indexing under a million entries, so narrowing them to 32 bits reaches **65%**, or
  **58%** at 4096 elements per pack where ghosts are proportionally fewer. So ~60% is what the
  format costs when its own index widths are used consistently; 85% is what this implementation
  costs today, and the table reports the implemented figure.
- **The narrowing was NOT done here, deliberately.** `ghost_reduce_ptr` and `ghost_reduce_idx`
  are public smesh API consumed by several hundred generated operator files across the codegen
  framework, plus the CUDA paths. Changing their width is a submodule API change belonging to
  that workset, not to a paper branch, and smesh is currently a clean `main`.
- **The paper compares PACKED against ATOMIC, and nothing else.** Which kernel each layout runs is
  an implementation question the paper does not litigate: each column is that layout's shipped
  kernel, which is the configuration a solver gets. Earlier drafts carried a hand-written-versus-
  generated comparison; it is out, deliberately, and must not come back. The code-generation work
  lives in `spikes/cvfem` with its own measurements.
- **All convective variants are presented together, first-order upwind among them.** One table and
  one figure, five rows each, no separate treatment and no withheld ratios -- the same shape the
  completeness ladder uses. The interesting result is the ordering: the layout is worth 3.06x on
  first-order upwind and falls monotonically to 1.34x on Venkatakrishnan as arithmetic per
  sub-control surface grows, which is the paper's thesis read off one table.

## Harvested from

| for | source |
|---|---|
| format spec | `spikes/cvfem/docs/PACKED_FORMAT.md` |
| prior draft of this paper (Elsevier CAS, HEX8 Laplacian) | `notebook/compressed/paper/main.tex` --- method section, Grace die table, roofline; does not build (missing figures and `references.bib`) |
| structural template and the only real bibliography in the tree | `docs/partial_assembly.tex` |
| related work | `spikes/cvfem/docs/CVFEM_SotA.tex` |
| the discretisation | `spikes/cvfem/docs/CVFEM_NSE.tex` |
| roofline script | `notebook/compressed/roofline.py` |

## Residual accumulation measurement

`--scatter-study` replaces the unsupported residual use of `--kernel-only` with
a separate FP64 affine diagnostic. Each sample times the normal residual, the
instrumented full residual, and the same instrumented sweep with accumulation
bypassed. Twelve samples use all six execution orders twice. The element kernel,
input staging and existing buffer clearing remain; state-gradient reconstruction
is outside the timed region. Boundary and transient terms are excluded.

The atomic bypass skips the four nodal atomic additions per element node. The
packed bypass skips pack-local additions, the owned/ghost drain and the final
ghost reduction. A compiler escape consumes the full SIMD result in both probe
arms, so bypassing its additions cannot delete the element calculations. Normal
calls instantiate the original loop without this instrumentation.

Before timing each case, the diagnostic captures every element's 32 contributions
in both arms and requires finite, nonzero, bit-identical arrays. Its full nodal
result must agree with the normal residual within `1e-11` relative infinity norm
in each equation, and the bypass must leave a zero output. Captures are freed
before timing; timed loops retain the compiler escape and accumulation switch.
The job stops on a failed check.

Run `spikes/cvfem/jobs/scatter_study.sh BENCH NEW_OUTPUT_DIRECTORY` inside one
72-core allocation. It records first-order and Darwish residuals with Rhie--Chow,
`n=128`, pack size 1024, at one and 72 threads. Analyze the eight logs with
`spikes/cvfem/scripts/analyze_scatter.py --expect-full OUTPUT_DIRECTORY/*.log`.
The summary gives mean times, signed paired differences, their range and the
full-probe overhead relative to normal execution. The atomic-minus-packed gap
is also split into the bypass gap and the difference in accumulation costs.

This measures the marginal cost of accumulation in the probe, including changes
in memory traffic and synchronization. It is not an additive instruction cost:
inspect instrumentation overhead and sample variation before relating the result
to production performance.

Grace job 5026072 on nid006544 (GNU 14.3.0) passes both CTest gates and all eight
timed cases. All 96 paired differences are positive. At 72 threads the mean costs
are 5.385 ms atomic versus 0.987 ms packed for first-order upwind, and 5.438 ms
versus 0.893 ms for Darwish. The difference in accumulation costs accounts for
92.9% and 95.4% of the corresponding full-probe gaps. Normal execution gives
2.02x and 1.43x packed speedups in the same allocation. Full-probe deviations from
normal range from -2.38% to +0.75% across all eight cases. These observations
support accumulation as the main contributor in this diagnostic; they do not
measure isolated atomic latency or establish a universal production fraction.

The raw logs are copied verbatim, with the suffix changed to `.out`, under
`data/scatter_5026072/`. Its `provenance.json` binds their hashes, source manifest,
source patch, binary hash and verification output. All 4250 source checks passed;
the largest full-probe/normal residual difference was 2.972e-16 per equation,
and every captured element contribution was bit-identical between probe modes.
`python/scatter_figures.py`, called by `perf_figures.py`, generates
`tables/scatter.tex` and `figures/scatter_macros.tex`. It rejects missing cases,
failed checks and altered input files. `make figures` now propagates generator
errors so it cannot quietly reuse a stale table after failed validation.

## Public irregular-mesh comparison (14 October 2026)

[Job 5052301](data/unstructured_5052301/README.md) adds two graded procedural
HEX8 timing meshes with irregular connectivity and non-affine cells. It compares
the first-order interior Rhie--Chow residual and exact action at 1/72 threads
and packs 256/1024. All twelve cases and 384 timing samples are retained.
The 72-thread table includes the medium-grid pack1024 action slowdown as well
as the speedups. Full absolute times, paired ranges, setup costs and the explicit
array ledger remain in the archived result. These are operator measurements,
not a spatial-accuracy or nonlinear-solver result.

`python/unstructured_figures.py`, called by `make figures`, checks every raw-file
hash, the captured parser identity, case coverage, commands, binary/policy records
and numerical/timing reports before generating `tables/unstructured.tex` and
`figures/unstructured_macros.tex`. The complete table must replay exactly from
the raw logs. `make check` also exercises archive rejection and preserves the
slower case in the generated table. No new cluster run is needed to regenerate it.

The [measurement protocol](../../spikes/cvfem/docs/unstructured_study.md) describes
the public generator and kernel correction. This dataset is separate from the
matched affine counter campaign; it supplies no new roofline points or traffic
attribution. The empirical roofline remains in the companion material.

## Open

- `refs.bib` holds one placeholder entry so the first build succeeds. Populate and delete it.
- The ORCID in `main.tex` is a placeholder; PASC requires a real one.
- `make check`'s page gate counts *all* pages, while the limit excludes references. It is
  therefore stricter than the rule --- safe, but it will need refining once the bibliography is
  real.

## A standing regression in the Rhie--Chow path (found while gating this work)

Not this paper's doing, but it affects numbers the paper reports, so it is recorded here.

`residual_packed_rc` lost **17.8%** and `residual_packed_rc_bnd` **17.2%** between commits
`735643efa` and `a3253433f` -- both predating the paper work. Established by an interleaved A/B
in one allocation between binaries built from those two commits, and corroborated across nodes:
the old code measures 1685.1 MDOF/s against the 2026-09-10 baseline's recorded 1681.4 (0.2%
apart, different node, five weeks later), while the current code measures 1385--1405 across
three nodes. A second A/B cleared this session's own changes at 0.6%.

A perf profile puts the whole of it in one symbol. Enabling Rhie--Chow does not add work to the
existing kernel -- it dispatches to `cvfem_hex8_conv_all_simd<RC=true, EPS=false>`, which
displaces `residual_sumfact_simd` (53.2% of the bare arm falls to 10.8%) and becomes the largest
cost at 45.6%. Two commits in the range change exactly that coefficient and are the suspects:
`c21b893a6` (the Jacobian treated the coefficient as frozen in the velocity) and `81c866966`
(the coefficient is a momentum time scale, not its viscous limit). Both are correctness fixes,
so this is most likely throughput paid for a more correct operator.

**Consequence for the paper.** The completeness ladder of \S\ref{sec:res:ladder} includes the
Rhie--Chow rungs, so its margins are measured against the current, slower RC path. That is the
right thing to report -- the paper describes the code as it is -- but the ladder's RC rungs
should not be compared with any number recorded before `a3253433f`.

`perf/baseline_grace.csv` has deliberately NOT been re-recorded: doing so would bake the 17.5%
in as the new normal before the cause is pinned down.

## Public numerical validation and storage-width probe (2026-10-13)

`data/mms_20261013/` preserves a local FP64 Farrell manufactured-solution ladder:
five convective schemes on affine n=4/8/16 grids over [0,2]^3, with Rhie--Chow,
unfrozen corrections and FGMRES/block-Jacobi. Physical-h velocity L2 fits range
from 0.747 for first-order convection to 1.879--1.916 for the reconstructed arms.
The local rates vary, so three coarse grids do not establish an asymptotic order
or a common order across limiter branches. Every final nonlinear residual meets
the requested 1e-10 relative tolerance. Every driver run also retains exit 1 from
its separate coarse-grid absolute velocity-error threshold. The clip n16 run's
intermediate residual-floor acceptance is recorded in the provenance.

The public packed/atomic gate retains its zero-flux box fixture as a localized
signed donor-derivative diagnostic and tests smooth agreement on both sides of
the switch. Boundary and transient gates test their stated individual terms;
they do not establish every composed limited, coupled or unsteady action.

`cvfem_storage_width_probe` and `src/tests/verification/cvfem_storage_width.sh`
prepare a bounded SFC-ordered graph-reduction and geometry-gather experiment.
The real reduction graph is copied to checked uint32 offsets/indices; the
FP32 geometry cache is widened into double scratch through the existing gather.
Original wide copies remain allocated. Clearing is excluded equally from graph
timing, conversion is included in geometry timing, and samples alternate after
warmups. These phase costs do not measure total footprint, operator speedup or
solver acceleration. Cube/sheared geometry checks and actual-reducer bitwise
parity precede and follow timing. Grace job 5039923 is now archived in
`data/storage_width_5039923/`. All checks pass at one and 72 threads. Narrow graph
timings depend strongly on which arm runs first, so they do not support an
index-width speedup claim. FP32 geometry gathering is faster in all 24 retained
pairs, including conversion, but its impact on a full operator remains unmeasured.

The historical RC slowdown above is associated with coefficient and linearization
changes; its exact cause has not been re-established here. This work does not
restore an old coefficient or treat it as an equivalent-operator optimization baseline.
