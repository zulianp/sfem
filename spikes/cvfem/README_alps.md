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

## Plan, revised against what has been measured

The original list was written before any of this was measured. Five items are done, two
were answered as side effects, two were measured and rejected, and the largest remaining
item was not on the list at all -- it surfaced from an error in how the benchmarks were
being compared.

### Done

| item | outcome |
|---|---|
| T1 assembled vs matrix-free | Assembled BSR is 1.48x cheaper at p=1 on CPU; assembly is under 1% of runtime. Matrix-free is the default anyway, because on Hopper the assembled BSR is 7.4 GB at n=128 and is a resolution ceiling, not a speed question. |
| T2 baseline at saturation | Flat apply saturates at 2.36 ns/dof on Grace, 13.5 on an M1. Neither is bandwidth bound: compulsory traffic is 4% of peak on Grace, 13% on Hopper. |
| T3 macro-local gather | 1.41x within the semi-structured kernel. |
| T4 hoist affine-macro invariants | A further 1.30x, and larger than the gather on the laptop. The twelve Rhie-Chow coefficients were the cost. |
| T10 level sweep | L=8 on both machines. |
| T12 mixed precision | 25% off the BSR SpMV at saturation, for an environment variable. Not yet tried on the block diagonal. |
| block diagonal (added) | Both layouts, 4.62x from the layout on Grace, and the sub-block split that a Schur scheme needs. |

### Measured and rejected

| item | why |
|---|---|
| T5 hoist the boundary-face test | Exactly zero on Grace, interleaved A/B in one binary. An interior micro-element costs six plane tests and a return; there was nothing to skip. |
| T13 element matrix applied as a gemm | 10% behind direct evaluation on Grace after three revisions. In `subpar/`. |
| T14 face-based flux dedup | Ruled out analytically: CVFEM sub-control-surfaces are interior to an element, not shared. |
| T9 block-diagonal scratch cost | Real (3x an apply) but 0.08% of a solve. Superseded by the semi-structured block diagonal. |

### Next, in order

**1. Stop recomputing the nodal pressure gradient per apply.** Done, and it is the largest
single win of the campaign: **1.26x off the whole linear solve**, 3636 to 2881 us per
linear iteration in the frontend driver at N=12, matrix-free, interleaved. Larger than
everything T3 and T4 won together.

It was not on the original list. It surfaced only because the flat operator recomputed the
gradient inside the timed region while the semi-structured benchmark hoisted it out, which
is what made every cross-comparison wrong until the discrepancy was found -- the error
pointed at the optimisation.

It is opt-in, `SFEM_PGRAD_CACHE`, on by default in the driver and off in the operator.
Switching it on is a promise about the caller's loop: after any change to the state,
`update()` or `gradient()` must run before the next `apply()`. A Newton loop satisfies that,
since the residual is evaluated right after the step and before the linear solve, but
nothing enforces it and a caller who breaks it gets a stale gradient and a wrong answer
rather than a failure. The operator caches per state pointer, so pointing it at a new
vector is safe; changing the contents behind the same pointer is not, and that is why it
is not the default. The gate checks the cached and uncached paths agree, at 3.5e-16.

**2. Specialise the gather.** On Grace the gather-and-scatter floor is 35% of the operator
and C costs 46%, so eleven points is all that is left for the pressure block without
touching it. Every block currently loads all fourteen arrays regardless of need.

**3. Wire the semi-structured kernels into the Op and the driver.** Everything measured so
far is kernel-level. The end-to-end claim, and the multigrid work behind it, needs this.

**4. Hopper.** Nothing semi-structured has run on a GPU, and layout conclusions have already
inverted once between Grace and Hopper: packing is worth 10% on CPU and loses by 42% there.

**5. Retire the pack machinery.** Semi-structured meshes give node contiguity by
construction, which is what `PackedMesh` renumbering manufactures -- and that renumbering
was the cause of a real segfault earlier in this work.

Lower down: fusing residual and Jacobian into one sweep, mixed precision on the block
diagonal, and SIMD strategy at macro granularity.


## Performance assessment: the 2x2 field blocks

`sscvfem_apply_blocks` evaluates any subset of

```
       | A_uu  B^T |   momentum rows
  J =  |           |
       | B     C   |   continuity rows
```

with the unwanted terms compiled out. A scheme can then ask for the block it needs
instead of evaluating J and discarding three quarters of it: a Schur approximation needs
B and B^T to form `B A^-1 B^T`, a segregated scheme solves the momentum rows alone, and
the pressure preconditioner explored in the standalone driver needs C by itself.

### Method

Matched problem size -- `macros * level` held constant, so every row solves the same
number of dofs -- swept over the macro-element level, on both machines. 4343300 dofs on
one Grace socket, 561924 on an M1. `gather only` is a `Blocks = 0` sweep: it gathers the
macro-element, computes nothing, and scatters zeros, which measures the floor any block
specialisation can reach rather than leaving it to be inferred.

### Cost as a share of the full operator

| block | Grace L=4 | L=8 | L=16 | M1 L=8 | what wants it |
|---|---|---|---|---|---|
| **gather only (floor)** | 34.5% | **35.3%** | 33.4% | **14.9%** | -- |
| `pp` (C) | 48.6% | **46.4%** | 46.5% | 38.1% | pressure preconditioner, Schur |
| `pu` (B) | 54.2% | 53.0% | 53.2% | 52.4% | `B A^-1 B^T` |
| `con` rows | 54.9% | 54.9% | 54.9% | 53.9% | segregated pressure solve |
| `up` (B^T) | 68.3% | 66.9% | 67.1% | 73.2% | `B A^-1 B^T` |
| `uu` (A) | 89.1% | 88.9% | 88.8% | 98.2% | momentum solve |
| `mom` rows | 98.7% | 98.2% | 98.3% | 104.5% | -- |

Grace is stable to within a point across L=4..16. L=2 is worse across the board -- the
floor alone is 45% there -- because `(L+1)^3 / L^3` is 3.375, so a macro-element gathers
more than three nodes for every micro-element it runs.

### What the numbers say

**The blocks a Schur scheme needs are the cheap half.** C costs 46% of J on Grace and B
53%, against 89% for A_uu. A_uu is barely cheaper than the whole operator, because the
viscous and convective terms it keeps are most of the cost.

**Asking for the momentum rows is not worth it.** At 98% of J it is within noise of just
evaluating the operator, and on the M1 it is slower. Use the full apply for that.

**The floor is the gather, and how much that matters depends on the machine.** On Grace it
is 35% of the operator, so C at 46% sits only eleven points above it and there is little
left to win without specialising the gather itself. On the M1 the floor is 15%, because
its kernels are roughly ten times slower per dof so the same fixed cost is a much smaller
share. The two machines disagree about where the remaining headroom is, and Grace is the
one to believe.

**Two hypotheses of mine were wrong, in opposite directions.** I had written into the
kernel that the upwind switch "cannot be specialised away". It can -- the continuity row
is `dmdot_v + dmdot_q` with no `sgn` in it -- and removing it from the pressure rows was
worth about 1%, not the large win expected. I then predicted the gather dominated, which
the M1 flatly contradicted at a 15% floor, and Grace then confirmed at 35%. The
measurement was right both times and the reasoning was not.

### Measured and rejected: hoisting the boundary term

`boundary_scs_add_jacobian_action` runs on every micro-element and tests six faces before
finding, in the interior, that it has nothing to do. A macro-element with no node on a
domain plane contains no micro-element with a face on one, so the call can be skipped
outright -- exactly, not approximately. It looked like the obvious next optimisation and
it is worth **nothing**.

Interleaved A/B in a single binary, one Grace socket, 4343300 dofs, apply in ns/dof:

| | trial 1 | trial 2 | trial 3 |
|---|---|---|---|
| L=4 hoist on | 1.188 | 1.191 | 1.185 |
| L=4 hoist off | 1.189 | 1.191 | 1.185 |
| L=8 hoist on | 1.162 | 1.174 | 1.161 |
| L=8 hoist off | 1.162 | 1.166 | 1.173 |

Identical to within 0.1%. The reason is visible once looked at rather than assumed: for
an interior micro-element the boundary kernel does six plane tests and returns, so there
was never much to skip. It was reverted rather than kept behind a flag, because the cheap
version of the test reads the eight macro corners and that is only valid for a box --
a latent trap for the curved macro-elements the hierarchy will eventually want, bought for
no measured gain.

Three things about how this was measured are worth keeping, since two earlier readings of
the same change were wrong.

**Across builds is not an A/B.** The first comparison put the hoisted kernel at 1.133
against a 1.095 recorded before it, and concluded a 4% regression. That 1.095 predated the
block split and the upwind specialisation as well, so it measured three changes at once.
A runtime switch inside one binary is what settled it.

**Interleave the arms.** Alternating hoist-on and hoist-off across trials, rather than
running each arm back to back, is the same discipline `bench_hex8_alps.sbatch` already
applies -- on a busy node the colored layout once looked 30% slower than packed measured
back to back and 60% faster interleaved.

**Check a control column.** `bd_nv` never touches the guard and held at 11.45-11.49
throughout the Grace job, which is what makes the 0.1% agreement believable. On the M1 the
same control swung 24 to 31, so its apparent 7% gain carries no weight -- the machine was
not quiet enough to measure a 7% effect.

### Where the remaining headroom is

Specialise the gather. Every block currently loads all fourteen arrays and scatters all
four components regardless of what it needs; C needs the coordinates and the pressure
direction, and little else. On Grace that is the only change with room left in it, since
the floor is most of what C costs.

The boundary term was the other candidate and it has since been tried and rejected; see
above. That leaves the gather as the only identified headroom on Grace.

### Correctness

Two checks, since either alone is insufficient. Each specialised kernel is compared
against a reference built by masking the inputs around the *unmodified* operator, which
cannot disagree with it by construction; and the four blocks must sum back to the full
operator, which is what catches a term landing in the wrong block -- the convective flux
contributes to both A_uu and B^T, and putting all of it in A_uu would still sum correctly
overall while burying the Rhie-Chow coupling in the momentum block. Both hold to 5.5e-16
and the benchmark fails on either.

```bash
CVFEM_CPUS=72 cvfem_run ./run_block_assess.sh          # the sweep above
SFEM_BENCH_VERBOSE_BLOCKS=1 ./build/cvfem_sshex8_bench # one size
```
