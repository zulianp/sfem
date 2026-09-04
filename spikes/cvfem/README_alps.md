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

## Operator throughput: Grace, Hopper, and a laptop

Baseline for the semi-structured work, all measuring the same thing -- the
matrix-free Jacobian action `y = J(u) v`, which T1 established is where the time
goes (roughly 800 linear iterations per Newton step, against one assembly).

| machine | cores/GPU | MDOF/s | ns/dof | vs Grace |
|---|---|---|---|---|
| Apple M-series | 10 cores | 74 | 13.5 | 0.17x |
| Grace | 72 cores | 424 | 2.36 | 1.0x |
| Hopper GH200 | 1 GPU | 10417 | 0.096 | 24.6x |

Saturation was swept, not assumed: Grace reaches it near 1.85M dofs, Hopper by
1.1M. Below that both mislead badly -- Grace reads 20 MDOF/s at 10k dofs, a
twentieth of its saturated figure.

Three things worth carrying forward.

**Grace scales, the laptop does not.** 46.5x on 72 cores (65% efficiency),
degrading smoothly from 98.6% at four threads. The laptop stalls at five of ten
cores and gains 5% for the second five. The conclusion first drawn there -- that
the memory system gives out at half the machine -- is a property of that machine
and does not reproduce here.

**Neither is bandwidth bound.** Compulsory traffic is 21.7 GB/s on Grace against
roughly 500 GB/s of LPDDR5X, and about 537 GB/s on Hopper against roughly 4 TB/s
of HBM3: 4% and 13% of peak. The traffic model is a floor, excluding
connectivity, coordinates and the nodal pressure gradient, so the real figure is
higher -- but not by the twenty-fold that would make either memory bound.

**Packing inverts on the GPU.** On CPU the packed layout is worth about 10% over
the atomic one. On Hopper it loses: 7351 MDOF/s packed against 10417 standard,
so the standard layout is 42% faster. Whatever packing buys on a CPU cache
hierarchy, it costs on the GPU.

And the case for matrix-free is much stronger on Hopper than on CPU. Assembly
runs at 248 MDOF/s there against 10417 for the action, so one assembly costs
about 42 applies -- but the decisive figure is memory: the assembled BSR is
7.4 GB at n=128 and grows with the mesh, against a few hundred megabytes for the
matrix-free path. At p=1 on CPU assembled BSR still wins on speed; on GPU it is
the resolution ceiling.

### Reproducing

```bash
rsync -az --delete --exclude 'build*/' spikes/cvfem/ alps:$SCRATCH/sfem/spikes/cvfem/
ssh alps
cd $SCRATCH/sfem/spikes/cvfem && source alps_env.sh

cvfem_configure && cvfem_build --target cvfem_ns_apply_bench
CVFEM_CPUS=72 cvfem_run env OMP_NUM_THREADS=72 OMP_PROC_BIND=close OMP_PLACES=cores \
    SFEM_BENCH_SIZES=8,16,32,48,64 ./build/cvfem_ns_apply_bench

cvfem_configure_cuda && cvfem_build_cuda --target cvfem_hex8_ns_cuda_verify
cvfem_run_cuda ./build_cuda/cvfem_hex8_ns_cuda_verify --n 128 --time-only --repeat 20
```

## T3: the macro-local gather, the invariants lifted out of it, and the block diagonal

The semi-structured kernel gathers a macro-element's `(L+1)^3` nodes once and runs its
`L^3` micro-elements against constant offsets, then lifts the affine-macro invariants out
of the loop: the direction areas, the node-separation vectors, and the twelve Rhie-Chow
coefficients, each of which costs a square root and a division and was recomputed
`12 * L^3` times per macro to produce the same twelve numbers. All variants agree with
the naive control to better than 5e-16.

One Grace socket, 4343300 dofs:

| L | naive | macro-local | + invariants | pgrad | apply+pgrad | blockdiag naive | blockdiag macro |
|---|---|---|---|---|---|---|---|
| 2 | 1.971 | 1.650 | 1.352 | 1.052 | 2.404 | 11.811 | 3.150 |
| 4 | 1.905 | 1.432 | 1.107 | 0.973 | 2.079 | 11.283 | 2.613 |
| 8 | 2.002 | 1.420 | **1.095** | 0.970 | **2.065** | 11.344 | **2.456** |
| 16 | 2.145 | 1.556 | 1.186 | 1.079 | 2.265 | 12.387 | 2.628 |

### Read the `apply+pgrad` column, not `+ invariants`

The flat operator recomputes the nodal pressure gradient inside every apply
(`cvfem_hex8_ns_op.cpp`, and `assemble_block_diag` does the same). This benchmark hoists
it out of the timed region. So the comparable figure against the flat kernel's 2.50
ns/dof is `apply+pgrad`, and the honest speedup is **1.21x**, not the 2.29x reported
before that discrepancy was noticed. The gains *within* the semi-structured kernel are
unaffected, since every variant there excludes the pass equally: the gather is worth about
1.41x and the invariants a further 1.30x, 1.83x together.

| | claimed | like-for-like |
|---|---|---|
| apply vs flat | 2.29x | **1.21x** |
| block diagonal vs flat | -- | **2.83x** |

### The block diagonal gains far more than the apply

The layout is worth **4.62x** there (11.344 to 2.456) against 1.83x for the apply, and
2.83x against the flat kernel's 9.70 ns/dof once the gradient pass is added to both.

The reason is that the semi-structured path can use the slot mask and the flat one cannot.
`cvfem_hex8_ns_upwind_jacobian_add_slots` writes exclusively through `cvfem_hex8_bsr_acc`,
which drops a negative slot, so the full element kernel produces the block diagonal with
none of the off-diagonal write traffic. The flat `assemble_block_diag` runs the SymPy
kernel, whose writes go straight to `values[...]` with no guard, so it has to assemble
each element into a 64-block scratch and discard seven eighths of it.

### The gradient pass is worth more than the layout

It is 0.970 ns/dof against an apply of 1.095, or 39% of the flat operator's per-apply
cost. In a Krylov solve the state is fixed for hundreds of applies, so computing it once
per Newton step instead of once per apply is a larger win than the entire layout change --
and `Op::update(x)` already exists as the place to do it.

L=8 is the optimum on both machines. The working set is `(L+1)^3` nodes times fifteen
arrays: 82 KB at L=8, 590 KB at L=16.

### The default

`sscvfem_apply` is the hoisted variant and `sscvfem_block_diag` its block diagonal. The
naive variants are the correctness controls the benchmark checks everything against, and
the intermediate ones are how the gather and invariant contributions are attributed. The
element-matrix and gemm variants lost and live in `subpar/cvfem_sshex8_em.hpp` under
`-DCVFEM_ENABLE_SUBPAR=ON`.

```bash
cvfem_build --target cvfem_sshex8_bench
CVFEM_CPUS=72 cvfem_run ./run_sshex8_sweep.sh
SFEM_BENCH_PROBE_DIAG=1 ./build/cvfem_sshex8_bench   # block diag against the operator
```

### The 2x2 field blocks

`sscvfem_apply_blocks` evaluates any subset of

```
       | A_uu  B^T |   momentum rows
  J =  |           |
       | B     C   |   continuity rows
```

with the unwanted terms compiled out rather than branched over. Cost as a share of the
full operator, M1, 561924 dofs, L=8:

| block | share of J |
|---|---|
| `pp` (C) | **36%** |
| `pu` (B) | **49%** |
| `con` rows (B and C) | 53% |
| `up` (B^T) | 69% |
| `uu` (A) | 94% |
| `mom` rows | 100% |

The blocks a Schur-complement scheme needs are the cheap ones. C alone costs about a
third of J, which matters for the pressure preconditioner: the standalone driver's
SFEM_PC_PSCALE work needed exactly this block and had no way to ask for it. A_uu is
barely cheaper than the whole operator, because the viscous and convective terms it keeps
are most of the cost.

One term does not go where the code's structure suggests. The convective flux contributes
to both A_uu and B^T, because the mass-flux derivative carries a velocity part and a
pressure part, `dmdot = rho/2 (v_i + v_j).A + c (q_i - q_j)`. Putting all of it in A_uu
would hide the Rhie-Chow coupling inside the momentum block and quietly wreck any Schur
approximation built on these blocks.

Two checks, because one is not enough. Each specialised kernel is compared against a
reference built by masking the inputs around the *unmodified* operator, which cannot
disagree with it by construction; and the four blocks must sum back to the full operator,
which is what catches a term landing in the wrong one. Both hold to 5.5e-16.

```bash
SFEM_BENCH_VERBOSE_BLOCKS=1 ./build/cvfem_sshex8_bench
```
