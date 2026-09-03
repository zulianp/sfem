# CVFEM HEX8 on CSCS Alps — benchmark, profile, report

Everything here targets **one Grace socket** (72 Neoverse V2 cores, ~500 GB/s of
LPDDR5X). The HEX8 Navier-Stokes assembly is memory-bound on a laptop; the point
of running it here is to find out what it is bound by when 72 cores share one
memory system, and what is left to fix.

## 1. Sweep

```bash
cd spikes/cvfem
sbatch bench_hex8_alps.sbatch
```

Writes `bench_alps_<timestamp>/cvfem_hex8_bench.csv`, one row per run:
every layout x operation, a pack-size sweep, thread scaling, problem-size
scaling, kernel variants, and a phase breakdown. Verification runs first and the
job aborts if any layout disagrees with the finite-difference Jacobian.

Useful overrides:

```bash
SIZES="48 64 96 128" sbatch bench_hex8_alps.sbatch     # add bigger meshes
SWEEPS="layout threads" sbatch bench_hex8_alps.sbatch  # just those two
TRIALS=5 sbatch bench_hex8_alps.sbatch                 # more interleaved passes
THREADS="1 2 4 9 18 36 72" sbatch bench_hex8_alps.sbatch
```

Trials are *interleaved*: the whole sweep runs `TRIALS` times rather than each
configuration running `TRIALS` times back to back. Drift over the job then hits
every configuration equally. This matters — measured back-to-back on a busy
machine, the colored layout looked 30% slower than packed; interleaved on the
same machine it is 60% faster. Barriers amplify external interference, so a
colored sweep is the configuration most likely to be misread on a shared node.

`--exclusive` is already in the header. Keep it.

## 2. Profile

```bash
sbatch perf_hex8_alps.sbatch                    # counters
PERF_RECORD=1 sbatch perf_hex8_alps.sbatch      # + sampled profile
```

Writes `perf_hex8_alps_<timestamp>/cvfem_hex8_perf.csv` (tidy:
`config,event,count`) plus the raw `perf stat` output and, with `PERF_RECORD=1`,
a `perf report` per configuration.

Counters come in six groups, each its own run rather than one multiplexed run —
Neoverse V2 has six programmable counters, and multiplexed ratios are not
trustworthy for a roofline argument:

| group    | answers                                                     |
|----------|-------------------------------------------------------------|
| `basic`  | IPC, branch behaviour                                       |
| `stalls` | frontend vs backend vs **backend-memory** stalls            |
| `cache`  | where the misses land: L1D, L2D, last level                 |
| `memory` | MEM_ACCESS read/write split, BUS_ACCESS, dTLB               |
| `simd`   | ASE / VFP / DP as a share of INST_SPEC — how vectorised     |
| `scf`    | Grace SCF `cmem` counters, i.e. real DRAM traffic           |

Every event is probed against `perf list` first and dropped if this kernel or PMU
does not expose it, so the script also runs unchanged on a laptop. What survived
is recorded in `available_events.txt` — check it before trusting a missing row.

**What to look for.** The hypothesis from the laptop is that assembly is
bandwidth-bound and the residual is not. On Grace that should show up as a high
`STALL_BACKEND_MEM / CPU_CYCLES` for assembly with low IPC, and SCF `cmem`
traffic close to the socket's achievable bandwidth. If instead the memory-stall
fraction is *low* and IPC is still poor, the limit has moved to the element
kernel — dependency chains and register spills in the generated code — and the
next move is SIMD across elements rather than more data-movement work.

## 3. Report and plots

```bash
python3 report_cvfem_bench.py bench_alps_*/cvfem_hex8_bench.csv \
        --perf perf_hex8_alps_*/cvfem_hex8_perf.csv \
        -o report.html --title "CVFEM HEX8 on Grace"
```

Standard library only — it runs inside a bare uenv. Produces one self-contained
page: headline rates, layout comparison, pack-size and thread and size scaling,
kernel variants, the phase budget, and the derived counter metrics. Every chart
is inline SVG that follows the reader's light/dark theme, and every chart has the
table it came from underneath it.

Add `--fragment` to emit body-only HTML suitable for publishing as a Claude Code
artifact.

For figures to drop into a paper or slides:

```bash
python3 plot_cvfem_bench.py bench_alps_*/cvfem_hex8_bench.csv -o plots/ \
        --format pdf --cache-mib 117      # Grace: 117 MiB L3
```

That one needs matplotlib. If it is not in the uenv, use the HTML report — it
needs nothing.

## Reading the numbers

**MDOF/s** is unique mesh degrees of freedom per second: four unknowns per node
(three velocity components and a pressure), divided by the time for one sweep.
That is the solver's unknown count, so it compares across element types. The
benchmark also prints `MDOF/s_element_visits`, which counts each node once per
adjacent element and reads about 8x higher for HEX8 — that one measures the
element kernel rather than the discretisation. Say which you mean.

`GFLOP/s_*_model` uses an idealised flop model that understates the SymPy
assembly kernel by roughly 2.4x. Compare kernels by MDOF/s.

## Expected shape of the answer

From the laptop, at n=64 (1.1 M dofs, 878 MiB of BSR values), 8 cores:

| operation        | atomic | packed | colored |
|------------------|--------|--------|---------|
| Residual         | 214    | **410**| 244     |
| Jacobian action  | 204    | **329**| 198     |
| Jacobian assemble| 19.5   | 24.6   | **40.0**|

Two things should carry to Grace and are worth checking first:

- The colored/packed assembly ratio grew with core count on the laptop
  (1.11x at 1 thread, 1.62x at 8). With 72 cores on one memory system it should
  be at least as large.
- Coloring needs each colour to hold at least as many packs as there are
  threads. At 72 threads that is a much stronger constraint than at 8 — start the
  pack-size sweep low and watch the `packs_per_color_min_max` line and the
  warning the benchmark prints.
