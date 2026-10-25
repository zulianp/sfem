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

`make` uses the committed figure and table inputs. To regenerate the scatter
results with `make figures`, first supply the evidence archives described under
"Residual accumulation measurement", "Darwish coefficient cache" and "Equivalent consistent matrix" below. Missing evidence is an error.

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
| residual accumulation diagnostic | 5026072 (nid006544) | external `pasc27-scatter-5026072.zip` (instructions below) |
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

## The roofline is built on measured traffic, after a model that was wrong

The intensity of every matrix-free point used to come from a compulsory-traffic model, and the model
was wrong in a way that inverted a conclusion. It counted, per dof, 16 B of field vectors plus 6.8 B
of "mesh" (connectivity and nodal coordinates). But the bare affine kernel reads no nodal
coordinates -- they are staged only when Rhie--Chow or the boundary closure is on -- and it does read
the cached affine geometry, `jacobian_adjugate[9]` and `jacobian_determinant`, **ten doubles per
element**, which is 19.5 B/dof: larger than every term the model did count. Two further faults: the
atomic points were drawn with the *packed* connectivity width, so their intensity was identical to
the packed points' by construction and the prose then reasoned from that identity; and the mesh
figure was taken from the smallest problem size in the campaign rather than the n=128 the figure is
drawn at.

Rather than repair the model, the points are now placed on **measured DRAM traffic** from the Grace
SCF counters, differenced across two repeat counts so setup and warmup cancel exactly. Units were
established rather than assumed: `cmem_rd_access` is exactly half `cmem_rd_data` in every row, which
fixes `cmem_rd_data` as 32-byte flits, and the interpretation is corroborated by the atomic sweep
reading more than the packed one in both operators and writing more on the residual, as its wider
index, node re-reads, zeroing pass and read-modify-write updates require.

Effect on the conclusions, which is why this is recorded here rather than only in the commit:

| | old model | corrected model | measured |
|---|---:|---:|---:|
| traffic, packed residual | 22.8 B/dof | 39.5 | **51.5** |
| intensity, packed residual | 8.1 | 4.7 | **3.6** |
| intensity, atomic residual | 8.1 (identical by construction) | 4.2 | **3.2** |

Measured intensity is below the measured ridge of 5.1 for every matrix-free arm, so the memory roof
binds -- but each attains only 11--29% of it, while the assembled SpMV attains 92% of its own. The
format's contribution is to need less traffic, not to use bandwidth better; the matrix-free kernels
are limited by the element kernel's instruction mix and retain headroom, and the assembled operator
does not. An intermediate draft of this analysis said "bandwidth-bound", which the attained
fractions contradict; it does not.

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

The raw logs are preserved verbatim, with the suffix changed to `.out`, in the
separate `pasc27-scatter-5026072.zip` archive. It contains the original 15 files
under `scatter_5026072/`. Its `provenance.json` binds their hashes, source manifest,
source patch, binary hash and verification output. All 4250 source checks passed;
the largest full-probe/normal residual difference was 2.972e-16 per equation,
and every captured element contribution was bit-identical between probe modes.
The archived source patch describes the original measurement checkout, including
earlier Jacobian-only changes. It is retained unchanged for provenance; the
residual accumulation study can be reviewed independently of those changes.
`python/scatter_figures.py`, called by `perf_figures.py`, generates
`tables/scatter.tex` and `figures/scatter_macros.tex`. It rejects missing cases,
failed checks and altered input files. It also requires the exact archive checksum
and member list. `make figures` checks the archive before generating any figures,
so missing evidence cannot quietly reuse a stale scatter table.

### Obtaining and checking the scatter archive

The archive SHA256 is:

```
1b34fdb16838fc53d455d9d61b1cd74b813aae117f961b9f907ca53b0b14945e
```

Keep the archive outside Git. From `wip/paper`, point the generator to it:

```bash
export SFEM_SCATTER_ARCHIVE=/path/to/pasc27-scatter-5026072.zip
python3 python/scatter_figures.py
make figures
```

Alternatively, put it at the ignored path
`wip/paper/data/pasc27-scatter-5026072.zip`; no environment variable is then needed.
The generator reads it in a temporary directory and checks the original provenance.
The measured values and generated table are unchanged.

Until a supplementary archive download is published, the complete evidence remains
recoverable from the original review commit
`9ce344db3a1b08bea273f97c9f16d84aa553ae07`, also retained in this branch's history.
With that commit available locally, run from `wip/paper`:

```bash
python3 python/scatter_figures.py --recover-archive-from-git
make figures
```

Recovery uses local Git objects, verifies the archive checksum, and never overwrites
an existing archive. If a shallow checkout lacks the commit, fetch it first:

```bash
git fetch origin 9ce344db3a1b08bea273f97c9f16d84aa553ae07
```

This removes raw evidence from the review's final file diff; it does not remove the
original files from Git history or reduce the size of a full historical clone.

### File scope for subsequent review contributions

Keep implementation changes, focused regression tests, required analysis scripts,
paper source and generated paper inputs together. Supply raw captures, per-run
source snapshots and detailed environment records in separate checksummed evidence
archives. Each contribution must explain how to obtain its evidence and regenerate
its outputs; absent or changed evidence must fail regeneration. Keep temporary
builds, local handoff notes and duplicate run wrappers out of the review.

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

## Darwish coefficient cache

The new table is generated by `python/darwish_cache_figures.py` from
`pasc27-darwish-5108749.zip`, SHA256
`226de8ebd3dba370bdc132add5dfb2af447df10844157897a223fed0c9edf1d4`.
The archive is distributed separately from Git. Set `SFEM_DARWISH_ARCHIVE` to
its path and run `make darwish-cache PY=python3` to replay the retained logs,
check numerical results and regenerate the table/macros. Ordinary `make` uses
the committed generated inputs and does not need the archive.

Evidence is from Grace job 5108749, node nid006549, in
`pasc27-darwish-final/results/darwish-final-5108749-QoQCXv`. The archive contains
the exact reproduction kit (`darwish_5108749/kit/RUN.md`), source and binary
manifests, compiler flags, validation logs, commands and all 90 timing records.
The kit reconstructs both compared sources from base `9cfefc2a` and its pinned
patch, and uses an existing SFEM dependency installation. The archived shell
wrapper prepares isolated builds and runs the numerical checks before timing.
Raw full-vector binary dumps remain in the original result directory; their
hashes and per-field comparison records are retained in the archive. They were
independently checked before this table was generated. Regenerating the table
replays logs and recorded checks; it does not re-read omitted binary dumps.

The primary statistic uses the six 60-apply windows per arm. All 30/60/90 windows
remain available; the 60-apply length matches the prior paper normal-timing
protocol. Each arm retains one process and allocation per block, uses the same
coordinate-based direction, and rebuilds direction gradients on every apply
where required. Six repeat-order permutations and rotating/reversed arm order
are retained. Every capture has 30 warmups and pre/post reference checks outside
timing. THP is disabled for benchmark workers. No samples are removed and no
hardware counters were collected. Setup includes cache allocation, construction
and a finite scan; its reported payoff is against direct Darwish at unchanged
state. Matrix assembly is also excluded from SpMV application time.

Within that campaign, only lagged action versus SpMV compares equivalent operators. The original,
allocation-matched control and cached Darwish arms compare the same consistent
action on the checked perturbed state. Cache coefficients retain reconstruction
and Rhie–Chow sensitivities. The cache adds storage to existing state gradients;
it is not a total-footprint reduction. That campaign did not measure an assembled
consistent Jacobian or solver time. The separate consistent-matrix campaign below
addresses the former comparison.

The unperturbed `n=8` diagnostic is retained separately in `checks/`. On Grace,
original/control/cached all fail the independent packed/atomic check. On the
local platform, control/cached also fail their cache/direct check. This remains
unresolved and is not presented as a passing numerical gate. The performance
claim uses the perturbed state that passed all checks. The experimental cache
must be rebuilt after state, geometry, packing or physical-parameter changes;
its integration into a nonlinear solver is not part of this change.

## Equivalent consistent matrix

The consistent-matrix table and macros are generated by the same
`python/darwish_cache_figures.py` with `--consistent-matrix`. Set
`SFEM_CONSISTENT_MATRIX_ARCHIVE` to `pasc27-consistent-matrix-5109074.zip`
(SHA256 `497ec8ae48b1e13030836e84adf4be81089334ac36b0800103c5fe836078dda1`)
and run `make consistent-matrix PY=python3`. The external archive includes the
exact standalone reproduction kit in `kit/RUN.md`, the patched source archive,
source/dependency/binary manifests, compiler flags, detailed CTest output,
commands and all timing logs. Normal paper builds use the committed generated
inputs. Neither the archive nor its extracted files need to be added to Git.

Grace job 5109074 ran on nid006544, with GNU 14.3, 72 cores on NUMA 0 and THP
disabled for each benchmark worker. The result directory is
`pasc27-consistent-matrix/results/consistent-matrix-5109074-WlnwF4`.
The source is based on `14f9a086ad4491c703f4fa27e79b0b6167420cde`, with an
isolated driver entry point and matrix-construction header recorded in
`kit/SOURCE.json`. The wrapper builds that snapshot without editing a checkout.

At fixed state, differentiating reconstruction composes a nodal gradient with
an element action. This reaches at most two node-adjacency steps, contained in
a 5-by-5-by-5 node box on the checked Cartesian fixture. Coordinates modulo five
separate all possible columns in a row. The 125 node colours times four fields
therefore recover the consistent matrix by analytic action probing, without
finite differencing or dropping sensitivities. The code checks grid coordinates,
element locality, finite coefficients and absent-colour outputs. It drops only
exactly zero blocks in BSR, then exactly zero scalar entries in CSR. BSR uses
32-bit indices/offsets and CSR 32-bit columns with 64-bit row offsets in this
installation. Both use the existing SFEM host SpMV kernels.

Correctness fixtures cover n=4,8,9,16, one/72 threads, SFC and unchanged ordering,
and pack sizes 64/256. All 500 scalar basis columns are checked at n=4; larger
fixtures check sampled columns, mixed-sign directions and individual fields.
Both matrices, cached action and independently reconstructed atomic action are
compared with the direct action. The per-field error is scaled by
`||J_field||_infinity * ||direction||_infinity`, with a 5e-12 bound and finite-value
checks. The largest recorded scaled error is 3.472e-16. A control removing
couplings beyond immediate neighbours fails that same screen. Existing cache,
finite-difference and coupled-derivative CTests pass; the separate known
unperturbed-state discrepancy remains unresolved.

Performance uses n=128, pack 1024, FP64 affine interior, Rhie--Chow and consistent
Darwish, perturbation 0.017 and coordinate-based directions. Three independent
workers each retain the mesh, cache and both matrices. Eight four-arm orders
balance every arm in every position, including reversals. Every window has
30 warmups, 60 applications and checks before/after timing. All 96 windows
(24 per arm) remain; ranges are sample variation, not confidence intervals.
Direction gradients are rebuilt by matrix-free actions and encoded by the
matrices. Setup, validation and warmup are outside application timing.

The measured means are direct 22.928338 ms, cached 16.455394 ms, CSR 36.591476 ms
and BSR 53.396806 ms. The cache saves 28.2% against direct; the cached action is
2.22x faster than CSR and 3.24x faster than BSR, with all 24 paired comparisons
favouring it. CSR/cache paired ratios span 2.180830--2.243660, BSR/cache
3.196005--3.268701, and direct/cache 1.367689--1.408465. The faster assembled
format, CSR, is the primary assembled comparison. These results are distinct
from the earlier comparison against a different, lagged operator; both campaigns
remain reported separately rather than pooling their samples.

Cache allocation is 2,416,050,176 bytes, CSR arrays 15,479,122,856 bytes and BSR
arrays 22,450,444,660 bytes. These are representation-specific array ledgers,
not peak process memory or total solver footprints. Cache setup averages
0.290089 s. The 16.316952 s matrix construction and 0.204001 s CSR conversion
are costs of this probing reference, not an optimised direct assembler. No
hardware counters were collected: storage and timing alone do not establish a
bandwidth bottleneck. No full-solver speedup or universal comparison against
all sparse kernels is claimed.

## Fixed-state gradient reuse (PR #60)

Grace job `5114021` on `nid006553`, GNU 14.3.0, measures only state-gradient
reuse in commit `9cfefc2a67390cdf6e7aadc84c5db3244d2a5704`. Both arms use the
same FP64 executable; `--rebuild-state-gradients` is the only computational
switch. The Darwish coefficient-cache optimization from PR #61 is absent.

Scope: affine interior, n128, packs of 1024, Rhie–Chow enabled, perturbation 0.017,
72 cores bound to NUMA0, and transparent huge pages disabled per worker with
readback. Five schemes, six fresh-process pairs each, three in each launch
order. Each process performs 30 warmups and 70 timed applications, rotating seven
live direction vectors. Application times exclude the intervening vector churn,
setup, checks and output I/O; required direction reconstruction is included.
This represents application cost with a changing input/live-vector working set,
not complete Krylov or nonlinear solver time.

All 60 windows are retained. Time saved is `100*(1-mean_reuse/mean_rebuild)`;
paired ratios divide rebuild time by reuse time within the same block. Ranges
are full sample variation, not confidence intervals. No sample is rejected
because reuse is slower. The benefit is present in both launch-order groups.

<!-- BEGIN GENERATED STATE GRADIENT RESULTS -->

| Scheme | Rebuild mean ms [range] | Reuse mean ms [range] | Time saved | Paired time-ratio range | Pairs favor reuse |
|---|---|---|---:|---|---:|
| First-order | 9.332093 [9.273706, 9.430952] | 8.744212 [8.693610, 8.922551] | 6.30% | [1.056979, 1.075218] | 6/6 |
| Unlimited | 18.543087 [18.413451, 18.820626] | 15.769150 [15.723780, 15.891021] | 14.96% | [1.158733, 1.196798] | 6/6 |
| Barth–Jespersen | 21.174029 [21.070607, 21.232505] | 18.665304 [18.537026, 18.866264] | 11.85% | [1.122948, 1.141609] | 6/6 |
| Venkatakrishnan | 25.695058 [25.534954, 25.809657] | 23.028365 [22.909407, 23.095486] | 10.38% | [1.105625, 1.124806] | 6/6 |
| Darwish | 25.243216 [25.152728, 25.329357] | 22.633972 [22.539136, 22.720589] | 10.34% | [1.107192, 1.123188] | 6/6 |

<!-- END GENERATED STATE GRADIENT RESULTS -->

Both precision cache regressions, the limiter and coupled derivative regressions,
and the dump checker pass (five CTests). Every timing capture passes its existing
packed/atomic action check. Every pair compares complete final output vectors at
the original cache tolerance (64 FP64 epsilons in the scaled infinity norm).
All 30 pairs report zero difference. The first pair's full vectors per scheme are
retained and independently rechecked; later vectors were fully compared and
hashed before release. This does not check every intermediate direction in the
timed loop or resolve the known unperturbed-state discrepancy.

The paper's historical consistent-action tables remain rebuild-mode measurements. This separate
study quantifies the standalone benchmark correction; solver code already had
fixed-state reuse. Its rotating-vector, fresh-process recipe differs from the
later retained-worker coefficient-cache campaign, so multiplying the two
speedups would not establish a measured combined improvement.

### Evidence and regeneration

Raw evidence and the complete run kit stay outside Git in
`pasc27-state-gradients-5114021.zip` (about 622 MiB). SHA256:

```
436fd698025351559618b2c50bb9b6a38fcb269c52061f1dc2323d05e6104d6c
```

The author-held archive was made from the retained evidence directory
`/capstor/scratch/cscs/gandanie/git/pasc27-state-gradient-study/results/state-gradients-5114021-4n6mUq`;
request the pinned archive for the exact checksum (the zip container metadata is
part of that checksum). No public download is claimed at this stage. The archive
contains all logs/CSVs, records, retained vectors, source snapshot, recipe,
compiler/build/dependency hashes, binding metadata and detailed numerical gates.
The 151 source files were checked against the stated commit.

From `wip/paper`, set the archive path and regenerate:

```bash
export SFEM_STATE_GRADIENT_ARCHIVE=/path/to/pasc27-state-gradients-5114021.zip
make state-gradients
```

The generator checks the archive hash, source/binary identity, scope, commands,
launch order, complete numerical records and hashed logs/CSVs before producing
`figures/state_gradient_macros.tex` and the table above. Missing or altered evidence
fails regeneration. `make figures` includes this gate; `make selftest` requires no
external archive. No archived Python code is executed by the paper generator.

For a full-vector replay, extract the archive and use its checked recipe:

```bash
python3 state-gradients-5114021/kit/study.py replay state-gradients-5114021/captures
```

Full replay rechecks the retained vectors as well as every log/CSV. The remaining
pair checks retain their original errors and vector hashes, not the released arrays.


### Storage columns in the Jacobian comparisons

The two timing tables include matrix-array storage and additional Darwish
coefficient-cache storage as separate columns, in GiB (2^30 bytes). A dash means
that representation does not require that component. It does not mean zero
memory: mesh topology, geometry, fixed-state gradients, field vectors, scratch,
assembly maps and allocator overhead are outside these columns. The allocation
control retains the cache even though it does not read it. The consistent-matrix
timing workers retain both matrices and the cache in all arms; the columns are
representation costs, not each worker's live or peak memory.

The consistent CSR/BSR and coefficient-cache sizes come from the pinned logs.
The lagged timing session did not print matrix bytes. Its 7.02 GiB is instead
an exact array-ledger calculation for the n=128 Cartesian HEX8 fixture:
129^3 = 2,146,689 node rows and (3*128+1)^3 = 57,066,625 node-neighbour blocks.
The lagged BSR allocates all 16 FP64 entries per block, including structural
zeros, with 32-bit column indices and row offsets:
`57,066,625*(16*8+4) + (2,146,689+1)*4 = 7,541,381,260 bytes`.
This follows `make_bsr4` in `spikes/cvfem/src/frontend/op/cvfem_hex8_ns_core.hpp`
and the tensor-grid adjacency; it is labelled as calculated in the table.
The consistent BSR log's retained-block count and byte ledger independently
agree with these index widths. No lagged scalar-CSR footprint or performance
was measured in this campaign.

Printed Jacobian terminology uses "consistent" throughout. Historical source
flags, record keys, filenames and macro identifiers containing "exact" stay
unchanged so regeneration and source references remain valid.

## Matched Jacobian layouts and precisions (job 5138025)

Figure 9 now uses the complete Grace job 5138025 on nid006022, GNU 14.3.0,
72 cores on NUMA 0. It preserves the original five scheme groups, three layout
colours, solid FP64/light FP32 bars and two lagged SpMV reference lines.
Figure 8 remains the earlier residual measurement. Other historical Jacobian
throughput tables and roofline points retain their original measurement scope;
this run collected application times, not hardware counters.

The committed source is `177d423a3c32358d4b60f8bac8350a64e3679475` (155 source
and CMake files independently compared with Git); the later PR61 merge changes
only paper files. All five schemes use fixed-state gradient reuse. Packed
Darwish uses the checked coefficient cache, while its direct arm is retained
below. Thus its displayed advantage includes coefficient reuse as well as the
layout. The cached arm is faster than direct in every pair in both precisions;
no sample was excluded to choose it.

The fixture is affine interior, n=128, pack 1024, Rhie--Chow enabled,
perturbation 0.017, with THP disabled per worker. Six fresh-process blocks
rotate/reverse arm order; each precision pair runs three times in each order.
Every window uses 30 warmups, 70 applications and seven rotating live direction
vectors, initialized by local vector index. Packing changes node numbering,
so these are not a common coordinate-based direction across layouts; the
in-process checks compare each layout with its correctly mapped atomic action.
Direction gradients remain timed; setup, validation, output and intervening
vector churn are excluded. Both precisions use their own vector,
geometry and arithmetic types. SpMV uses the existing lagged BSR operator with
matching scalar precision; it is not equivalent to the consistent action.

All 204 windows and 102 precision comparisons are retained in the accounting.
Six CTests pass. Per-capture layout checks and twelve checks per cached Darwish
capture pass. All 34 retained block-zero vectors were retrieved and their
17 precision comparisons independently rerun. Later vectors were completely
compared and hashed on Grace before release. Smooth FP32 checks require a
maximum error below 2e-5 scaled by the FP64 maximum. Limited Jacobians retain
the existing criterion of fewer than 1% of entries exceeding 1e-4: precision-
dependent limiter switch bands allow exceptions. This is not entrywise close
agreement everywhere or resolution of the known unperturbed-state discrepancy.

Rates are 8,586,756 scalar degrees of freedom divided by the mean application
time, not element visits. Ranges contain every sample and are not confidence
intervals. Pairing is by block. Direct/cached Darwish compares the same checked
operator; Darwish/lagged SpMV compares different operators. The new run supports
an FP64 cached Darwish advantage against lagged SpMV, but FP32 cached Darwish
is slower than its FP32 SpMV reference in all six pairs. The earlier equivalent
consistent CSR/BSR campaign remains a separate result; samples and speedups
from different campaigns are not pooled.

<!-- BEGIN GENERATED JACOBIAN LAYOUT RESULTS -->

| Scheme | Layout | Coefficients | Precision | Mean ms [range] | MDOF/s from mean time |
|---|---|---|---|---|---:|
| upwind | packed | direct | f64 | 9.352746 [9.245038, 9.389627] | 918.100 |
| upwind | packed | direct | f32 | 5.871592 [5.788767, 5.968688] | 1462.424 |
| upwind | ecolor | direct | f64 | 17.024925 [16.781400, 17.255020] | 504.364 |
| upwind | ecolor | direct | f32 | 7.640448 [7.510660, 7.894579] | 1123.855 |
| upwind | atomic | direct | f64 | 17.323005 [17.101850, 17.535880] | 495.685 |
| upwind | atomic | direct | f32 | 14.303573 [14.130110, 14.670580] | 600.322 |
| unlimited | packed | direct | f64 | 16.746982 [16.577470, 16.904490] | 512.735 |
| unlimited | packed | direct | f32 | 10.544333 [10.405590, 10.680780] | 814.348 |
| unlimited | ecolor | direct | f64 | 40.085872 [39.666410, 40.451600] | 214.209 |
| unlimited | ecolor | direct | f32 | 18.635335 [18.320390, 18.903630] | 460.778 |
| unlimited | atomic | direct | f64 | 37.247128 [36.895080, 37.736030] | 230.535 |
| unlimited | atomic | direct | f32 | 31.446802 [31.033670, 31.812210] | 273.057 |
| barth | packed | direct | f64 | 19.993658 [19.660500, 20.589710] | 429.474 |
| barth | packed | direct | f32 | 12.159962 [11.989900, 12.378780] | 706.150 |
| barth | ecolor | direct | f64 | 42.391295 [41.904780, 42.669240] | 202.559 |
| barth | ecolor | direct | f32 | 19.889530 [19.682340, 20.244730] | 431.722 |
| barth | atomic | direct | f64 | 40.470652 [40.119910, 40.835170] | 212.172 |
| barth | atomic | direct | f32 | 32.937128 [32.798350, 33.161140] | 260.701 |
| venkat | packed | direct | f64 | 24.522590 [24.342720, 24.696630] | 350.157 |
| venkat | packed | direct | f32 | 14.315792 [14.170210, 14.411290] | 599.810 |
| venkat | ecolor | direct | f64 | 45.745985 [45.423620, 46.444310] | 187.705 |
| venkat | ecolor | direct | f32 | 21.546757 [21.366150, 21.731040] | 398.517 |
| venkat | atomic | direct | f64 | 44.903040 [44.501670, 45.538180] | 191.229 |
| venkat | atomic | direct | f32 | 34.969340 [34.753690, 35.230290] | 245.551 |
| darwish | packed | cached | f64 | 17.418922 [17.246620, 17.665900] | 492.956 |
| darwish | packed | cached | f32 | 11.053917 [10.809040, 11.256540] | 776.807 |
| darwish | ecolor | direct | f64 | 45.371560 [44.697250, 45.675610] | 189.254 |
| darwish | ecolor | direct | f32 | 21.759772 [21.413600, 22.114190] | 394.616 |
| darwish | atomic | direct | f64 | 44.555583 [44.336090, 44.938570] | 192.720 |
| darwish | atomic | direct | f32 | 34.862913 [34.481180, 35.260080] | 246.301 |
| darwish | packed | direct | f64 | 24.113418 [23.951310, 24.223830] | 356.099 |
| darwish | packed | direct | f32 | 14.330593 [14.143260, 14.465250] | 599.191 |
| spmv | packed | direct | f64 | 19.429913 [19.353260, 19.505740] | 441.935 |
| spmv | packed | direct | f32 | 10.250167 [10.162070, 10.323380] | 837.719 |

| Comparison (numerator / cached Darwish) | Precision | Ratio of mean times | Paired range | Pairs above one |
|---|---|---:|---|---:|
| Direct Darwish | f64 | 1.384323 | [1.366286, 1.397558] | 6/6 |
| Direct Darwish | f32 | 1.296427 | [1.263220, 1.335706] | 6/6 |
| Lagged SpMV (different operator) | f64 | 1.115449 | [1.100781, 1.126565] | 6/6 |
| Lagged SpMV (different operator) | f32 | 0.927288 | [0.910869, 0.947914] | 0/6 |

<!-- END GENERATED JACOBIAN LAYOUT RESULTS -->

The external evidence archive is `pasc27-jacobian-layouts-5138025.zip`, SHA256
`f6414f76657344d9d5b1e3041fd49010d3152355b22655237fc2e28eb3117e26`.
It contains the complete kit (`kit/RUN.md`), source archive, source/dependency/
binary hashes, binding/compiler metadata, CTest log, all captures and checks,
and independent verification record. Raw vectors stay in the retained result
directory and author-held local evidence copy; the archive retains their hashes
and comparison results. Table/figure regeneration checks the pinned archive,
complete coverage, commands, numerical records and log hashes; it does not
execute archived code or reread the omitted vectors.

```bash
export SFEM_JACOBIAN_LAYOUT_ARCHIVE=/path/to/pasc27-jacobian-layouts-5138025.zip
export SFEM_PRECISION_ARCHIVE=/path/to/pasc27-precision-5140975.zip
make jacobian-layouts PY=python3
```

Ordinary PDF builds use the committed generated figure/macros. `make figures`
regenerates this figure from the external archive after the historical plots.
The incomplete job 5130300 is retained separately and supplies no samples here.
Neither this run nor its figure establishes solver performance or a globally
optimal implementation.


## Precision diagnostic: matrix traffic and coefficient reuse

Job `5140975` on `nid005555` is a separate diagnostic, not a replacement for
Figure 9's layout campaign. It uses 72 Grace cores, affine interior n=128,
pack=1024, RC on, perturb=0.017, fixed-state gradient reuse and worker-only THP
disabling. Six blocks retain all 182 operator captures and 18 copy/triad
controls. Fresh processes use 30 warmups and 70 applications. Normal no-live
windows use a fixed direction; live7 windows retain the seven rotating vectors
and exclude intervening churn. SCF and core counters use separate no-live runs;
setup, assembly, cache construction, checks and warmups are excluded.

The SCF counters cover the socket once on its representative CPU; core cycles,
retired instructions and L1/L2 refill events cover the 72 allocated CPUs.
Read traffic is 32 times `cmem_rd_data`; writes use `cmem_wr_total_bytes`, with
the write-component identity checked. Counters are unscaled, fully scheduled
and retained without subtraction. All 84 operator screens pass: counter/loop
windows, empty traffic, mean instrumentation overhead and sample stability.
The copy/triad controls and raw counter definitions are in the external kit's
`RUN.md`; passing these practical screens does not establish absolute accuracy.

<!-- BEGIN GENERATED PRECISION DIAGNOSTIC -->

| Arm | Normal ms [range] | Live7 ms [range] | Transported GB/apply [range] | Transported GB/s [range] | Retired instructions, billions/apply [range] |
|---|---|---|---|---|---|
| cached64 | 16.225048 [16.125180, 16.511660] | 16.210203 [16.143290, 16.296890] | 5.355670 [5.345345, 5.364012] | 330.672009 [328.200230, 331.849879] | 7.419606 [7.401016, 7.431270] |
| cached32 | 10.191145 [9.960243, 10.513050] | 10.148408 [10.050070, 10.197860] | 2.674909 [2.665188, 2.681678] | 263.294529 [258.120384, 267.222616] | 4.824030 [4.817342, 4.834410] |
| direct64 | 22.840252 [22.547650, 23.423920] | 22.916803 [22.725590, 23.091240] | 2.972977 [2.953790, 2.989968] | 130.696400 [129.840112, 131.215283] | 16.384558 [16.372327, 16.400429] |
| direct32 | 13.037377 [12.951510, 13.192180] | 13.262225 [13.137170, 13.485900] | 1.485110 [1.482554, 1.486165] | 114.017755 [113.795649, 114.222150] | 9.433029 [9.427911, 9.438639] |
| spmv64 | 19.123137 [18.924260, 19.369090] | 19.224082 [19.162820, 19.315450] | 7.709718 [7.706338, 7.712195] | 403.410828 [398.702721, 408.084061] | 2.438383 [2.434212, 2.441687] |
| spmv32 | 9.851928 [9.821878, 9.884289] | 10.111917 [10.055550, 10.151480] | 3.915667 [3.911148, 3.918603] | 394.376149 [386.385822, 399.372216] | 2.192130 [2.189563, 2.194868] |
| spmv_mixed | 10.199885 [10.025760, 10.293400] | 10.045245 [9.906893, 10.285070] | 4.036181 [4.034643, 4.039437] | 398.190260 [393.778778, 408.559290] | 3.373748 [3.372589, 3.375114] |

<!-- END GENERATED PRECISION DIAGNOSTIC -->

Ranges are sample variation, not confidence intervals. Normal timings and
counter measurements above come from separate captures. `spmv_mixed` stores
rounded FP32 matrix values but retains FP64 vectors and arithmetic. It adds
conversion work and retains the original double matrix allocation; it is not a
pure bandwidth intervention or a total-memory saving. It reaches nearly pure
FP32 speed while executing more retired instructions than FP64 SpMV. Together
with approximately halved traffic and similar bandwidth, this supports reduced
matrix traffic as the main explanation for this fixture's SpMV precision gain.

Cached Darwish also halves traffic, but attains lower bandwidth in FP32.
Its additional instruction/gradient work does not identify a unique hardware
stall or vectorization bottleneck. Relative to direct coefficients, caching
moves about 1.80 times as many bytes while reducing instructions and time:
this is a measured storage/traffic versus recomputation tradeoff. Core counts
include runtime/background activity and are not useful floating-point work.

With live7 vectors, FP32 cached Darwish and lagged SpMV are nearly tied in this
run; the paired cached/SpMV ratios span 0.998030--1.008597, with cached faster
in two of six pairs. The earlier figure's FP32 deficit is therefore specific
to that campaign. These compare different operators; the equivalent consistent
CSR/BSR study remains separate. No solver speedup or universal ordering follows.

All 15 kit payload hashes and 155 reconstructed source hashes were verified,
as were the 182 capture records, 146 numerical/identity checks, 18 controls,
recorded binary identities and the four passing Grace CTests. The numerical
kernels match source177d423a; only driver counter-window hooks were added.
Binary files and large output arrays were not downloaded/recomputed locally;
numerical validation relies on the recorded comparisons and log hashes.
Block-zero arrays remain on the cluster. Limited FP32 checks retain their
existing subgradient-band exceptions, not entrywise or general PDE guarantees.

External archive: `pasc27-precision-5140975.zip`, SHA256
`71e9ea6fc5163313dda2d55ffcd63c7ae3b1ba130e3baa013c7d3115eff22a48`.
It contains the source/kit, commands, raw text counters, checks and review record,
not the omitted arrays. The figure reader checks this pinned archive and emits
these ranges and the prose macros without executing archived code. Both external
archives are required by `make jacobian-layouts`; ordinary builds use the
committed figure/macros. No measurements from the failed job5138345 are used.
