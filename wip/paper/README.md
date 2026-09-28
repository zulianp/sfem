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
| `python/packed_format_figures.py` | `figures/packed_{decomposition,id_space,reduction,scatters}.tex`, `figures/packed_counts.tex` | nothing --- it reimplements the format's documented rules and draws the result |
| `python/perf_figures.py` | `figures/{throughput_size,scaling,packsize,roofline}.tex`, `tables/{throughput,ladder,determinism,footprint,convho,jacfair}.tex` and their macro files | the campaign CSVs and per-measurement `.out` files below |

Both are stdlib-only and carry `--selftest`, following the repository's report generators: the
Alps uenv has no numpy or matplotlib, and a generator that cannot run where the measurements run
goes stale.

`packed_format_figures.py` is a *model*, not a drawing. It reimplements the ownership rules from
`spikes/cvfem/docs/PACKED_FORMAT.md` and its selftest asserts the format's documented invariants
against that model --- owned ranges partitioning the index space, non-shared indices preceding
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
| M3 | bitwise determinism by layout, repeats and thread sweep | `jobs/det_layout.sbatch` (new) | **done**, 4789239 -> `data/det_layout_4789239.out` -> `tables/determinism.tex` |
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
