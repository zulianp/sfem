# PASC27 paper: a packed mesh format for deterministic, atomics-free matrix-free operators

Target: PASC27, **10 pages excluding references, ACM `sigconf`, deadline 12 December 2026 (no
extensions)**. Domain track: Computational Methods and Applied Mathematics. ORCID is mandatory
for every author and publication is open access.

```
make           build main.pdf
make figures   regenerate every figure and table input from the measurements
make selftest  run the generators' own checks
make check     the submission gates (clean build, no undefined refs, page limit)
```

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

The new counter-calibration job is the first required Grace gate before collecting
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
