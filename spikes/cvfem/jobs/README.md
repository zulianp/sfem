# `jobs/` — the paper's measurements and the verification gates

Slurm scripts for one Grace socket on Alps (`sbatch jobs/<name>.sbatch` from the spike root; the
environment comes from `scripts/alps_env.sh`). This folder holds only what is run again: the jobs
that produce the paper's data, the verification and validation gates, and the two Alps tools
`docs/README_alps.md` documents. Everything else is in `subpar/jobs/`, with what it measured and
where its answer is recorded.

**Rule.** A job whose output feeds `wip/paper` gets a row in the first table. A job written to
answer one question moves to `subpar/jobs/` once its answer is recorded in a perf note, a document
or the code.

## The paper (`wip/paper`)

`wip/paper/python/perf_figures.py` reads `wip/paper/data/` by file prefix and takes the newest file
for each, so every prefix has exactly one producing job. Copy a job's `.out` (or the campaign CSV)
into `wip/paper/data/` under its prefix and job id, then `make -C wip/paper figures`.

| job | data prefix | paper element |
|---|---|---|
| `camp_full.sbatch` | `campaign_grace_` | throughput figure, completeness ladder, footprint table, campaign macros |
| `det_layout.sbatch` | `det_layout_` | determinism macros |
| `thread_scaling.sbatch` | `tscale_` | thread-scaling figure |
| `packsize_paper.sbatch` | `packsize_` | pack-size figure, spatial-ordering macros |
| `stream.sbatch` | `stream_` | roofline bandwidth roof |
| `peak_fp64.sbatch` | `peak_` | roofline compute roof |
| `dram_traffic.sbatch` | `dram_` | roofline points, colouring traffic macros |
| `conv_ho_bench.sbatch` | `convho_` | convective-scheme table and figure (f64 and f32) |
| `jac_ho_bench.sbatch` | `jacho_` | Jacobian scheme figure (f64 and f32) |
| `jac_fair.sbatch` | `jacfair_` | Jacobian comparison table (f64 and f32) |
| `ho_exact_jac.sbatch` | `hoexact_` | exact higher-order Jacobian macros |
| `kernel_mix.sbatch` | `kmix_` | instruction-mix table |
| `ho_size_sweep.sbatch` | `hosize_` | higher-order series of the throughput figure |
| `nodal_grad_bench.sbatch` | `ngrad_` | nodal-gradient macros |
| `f32_vs_f64.sbatch` | `f32_` | f32 group of the ladder table; `perf/f32_vs_f64_grace.txt` |

`meshfoot_grace.out` has no job: it is `cvfem_hex8_ns_upwind_bench --n 128 --mesh-footprint` on
the login node, a byte count rather than a timing.

## Verification and validation

| job | what it checks |
|---|---|
| `verify_report.sbatch` | the verification matrix (`scripts/verify_report.sh`), which `docs/CVFEM_Verification_Report*.md` is built from |
| `nozzle_verify.sbatch` | the FDA nozzle: the operator on a curved mesh, flat and isoparametric |
| `verif_one.sbatch` | one verification case per allocation at ~20M dofs |
| `conv_limiter.sbatch` | limiter convergence on the step case, the evidence the verification report's limiter section points to |
| `determinism.sbatch` | solver bit-reproducibility at 1 and 72 threads |
| `ctest.sbatch` | the spike's ctest suite on Alps |
| `perf_regression.sbatch` | the performance-regression gate (`scripts/perf_regression.sh`) |
| `ab_refactor.sbatch` | packed throughput A/B against a reference build, interleaved in one allocation |
| `ab_semistructured.sbatch` | the same A/B for the semi-structured family (`CVFEM_REF=` a built reference) |

## Alps tools

| job | use |
|---|---|
| `bench_hex8_alps.sbatch` | HEX8 benchmark sweep (`docs/README_alps.md`) |
| `perf_hex8_alps.sbatch` | Linux `perf` profiling and memory counters of the HEX8 benchmark (`docs/README_alps.md`) |
