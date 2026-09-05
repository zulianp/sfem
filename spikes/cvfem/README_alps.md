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
| Hopper GH200 | 1 GPU | 10417 | 0.096 | 24.6x* |

\* The Hopper figure is a kernel *without* the Rhie-Chow term, which every host figure
includes; see the Hopper section below. Corrected for it the ratio is nearer 18x.

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

**2. Specialise the gather.** Done. Each block now gathers only the arrays it reads and
scatters only the rows it writes, and on Grace the blocks a Schur scheme wants got about a
quarter cheaper:

| | before | after |
|---|---|---|
| floor | 35.3% | 12.2% |
| `pp` (C) | 46.4% | **36.1%** |
| `pu` (B) | 53.0% | **43.2%** |
| `con` rows | 54.9% | **45.5%** |

`uu` and `mom` barely moved, 2 points, which is right: they read almost everything anyway.
Two constraints cap what C can save. The boundary term takes the state velocity whatever
is masked, and the macro geometry needs the coordinates, so C still gathers seven of the
fourteen arrays rather than the three its own arithmetic uses.

**3. Wire the semi-structured kernels into the Op and the driver.** Done. The operator
picks the path from what the space carries -- `has_semi_structured_mesh()` -- rather than
being configured, and the driver turns a mesh semi-structured with
`SFEM_ELEMENT_REFINE_LEVEL`. The same problem decomposed three ways:

| | nodes | elements | newton | lin_it | u_linf |
|---|---|---|---|---|---|
| flat, N=8 | 2673 | 2048 | 19 | 15420 | 9.706443e-10 |
| N=4, level 2 | 2673 | 512 macros | 19 | 15409 | 9.706011e-10 |
| N=2, level 4 | 2673 | 32 macros | 19 | 15440 | 9.706217e-10 |

Identical discrete problem, same Newton count, `u_linf` agreeing to six figures, solved
through 32 macro-elements instead of 2048 flat ones.

Writing this needed the residual, which the semi-structured path did not have -- it had
the Jacobian action, the block diagonal and the block split, none of which Newton can
start from. It is implemented in the same two layouts as everything else so the naive one
gates the macro-local one, and agrees at 2.6e-15.

Two limits are deliberate. The path is affine-macro only: one Jacobian per macro-element,
reused across its lattice, which is exact for a box and wrong for a curved macro-element,
and it ignores `SFEM_GEOM` for the same reason. And it refuses `hessian_bsr`, because an
assembled matrix per level is the memory a hierarchy exists to avoid; refusing beats
returning a zero matrix.

**4. Hopper.** Measured, and it says do not port the gather. Two things came out of it.

*Every device figure previously in this file was for a kernel without the Rhie-Chow term.*
None of the `cvfem_cuda_time_*` entry points takes `rc_scale`; `cvfem_cuda_residual_rc`
existed but was only ever verified, never timed, and the timing path attached neither the
coordinates nor the nodal gradient that it needs. Every host figure includes the term. The
comparison was therefore between a device kernel missing a term and host kernels that have
it -- the third comparability error of this work, after the pressure gradient and the
cross-build boundary A/B, and the same shape each time: two sides doing different work with
nothing in the harness to notice. `cvfem_cuda_time_residual_rc` now exists.

| Hopper, packed residual, n=128 | MDOF/s |
|---|---|
| without Rhie-Chow | 7872 |
| with Rhie-Chow | **5899** |

The term costs 1.33x, so it is a quarter of the device kernel -- close to its share on the
host, where hoisting its coefficients was worth 1.28x. Applying that factor to the apply
figure puts Hopper nearer **18x Grace than the 24.6x** reported before; that scaling is an
inference from the residual, since there is no timed apply with the term.

*The macro-local gather is already on the GPU, as packing, and it loses.* The packed
kernel is block-per-pack with shared-memory staging, which is the same transformation, and
the plain global layout beats it by 1.41x for the Jacobian action -- 10420 against 7297
MDOF/s -- confirmed twice in separate builds.

So a semi-structured CUDA port would buy nothing from the half that wins on CPU and
everything from the half the GPU lacks: the device takes `adj` and `det` as precomputed
inputs, so geometry is already hoisted there, but `mdot_coeff` still runs per
sub-control-surface per element. Hoisting those coefficients needs elements sharing a
Jacobian, which packs cannot give and macro-elements can. The ceiling on that is the 25%
above, and realistically less, since the Rhie-Chow term is more than its coefficients.
Worth doing only if a quarter of the device kernel is worth a port.

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

| block | Grace L=4 | L=8 | L=16 | what wants it |
|---|---|---|---|---|
| gather only (floor) | 13.0% | **12.2%** | 12.2% | -- |
| `pp` (C) | 38.0% | **36.1%** | 34.1% | pressure preconditioner, Schur |
| `pu` (B) | 44.8% | 43.2% | 40.5% | `B A^-1 B^T` |
| `con` rows | 46.4% | 45.5% | 43.6% | segregated pressure solve |
| `up` (B^T) | 66.3% | 64.6% | 65.1% | `B A^-1 B^T` |
| `uu` (A) | 87.7% | 86.8% | 87.0% | momentum solve |
| `mom` rows | 95.7% | 95.9% | 96.0% | -- |

Shares are against the full operator measured through the same block kernel in the same
run. `SSBLOCK_ALL` sets every flag, so it gathers everything and the denominator is
unaffected by the specialisation below.

Grace is stable to within a point across L=4..16. L=2 is worse across the board -- the
floor alone is 45% there -- because `(L+1)^3 / L^3` is 3.375, so a macro-element gathers
more than three nodes for every micro-element it runs.

### What the numbers say

**The blocks a Schur scheme needs are the cheap half.** C costs 46% of J on Grace and B
53%, against 89% for A_uu. A_uu is barely cheaper than the whole operator, because the
viscous and convective terms it keeps are most of the cost.

**Asking for the momentum rows is not worth it.** At 98% of J it is within noise of just
evaluating the operator, and on the M1 it is slower. Use the full apply for that.

**The floor was the gather, and specialising it was worth a quarter on the pressure
blocks.** Before, on Grace, the gather and scatter were 35% of the operator and C cost 46%,
leaving eleven points; the M1 put the same floor at 15%, because its kernels are about ten
times slower per dof so the same fixed cost is a smaller share of them. Grace was the one
to believe, and gathering only what each block reads took C to 36% and B to 43%. Note that
the floor figure is now block-dependent by construction -- a `Blocks = 0` sweep gathers
only the coordinates and the state velocity -- so 12% is the floor for a block that reads
nothing, not a bound shared by all of them.

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

## Semi-structured geometric multigrid: a running V-cycle, and why it is not yet a win

`create_gmg_data` is wired up and a V-cycle runs, preconditioning BiCGStab inside the
Newton loop (`SFEM_GMG=1` in `cvfem_hex8_ns_ssgmg`). Getting it to run at all turned on one
parameter, and the result it produces says the smoother is the wrong one.

### The wiring

`Function` owns the coarse `Function`s that `create_gmg_data` derefines but does not hand
their operators back, and every level here needs two things a linear problem would not
need: the state to linearise about, and its own block diagonal. So `CVFEMNavierStokes`
records the operator it produced in `derefine_op` and exposes it as `coarser()`, and the
driver walks that chain from the finest level. Per level it holds a state buffer, restricted
from the fine state with the averaging restriction; a matrix-free operator bound to that
buffer; and a 4x4 block-Jacobi smoother built from `hessian_block_diag`. The coarse level is
solved with BiCGStab. `build_gmg` runs once, `refresh_gmg` per Newton step -- rebuilding the
whole hierarchy per step instead made the first run appear to hang.

Three pieces of the default GMG path are deliberately not used. `create_gmg_operators`
passes `nullptr` as the state, which is fatal for a nonlinear operator.
`create_gmg_default_smoothers_and_solver` computes `sym_block_size = (block_size == 3 ? 6 :
3)`, silently yielding 3 for block size 4, and reaches for `hessian_block_diag_sym`, whose
packing assumes a symmetry a Navier-Stokes block does not have. Its CG coarse solver wants
an SPD system.

### Damping is what made it converge

Undamped, the V-cycle was not merely ineffective but actively harmful: BiCGStab sat on its
1000-iteration cap on every Newton step. Newton still crawled forward on the truncated
steps, which is what made this slow to spot -- the residual fell 2.6e-2 -> 6.4e-6 and only
the iteration counts showed anything wrong.

The cause is that block-Jacobi is being asked to do a different job than elsewhere in this
driver. As a Krylov preconditioner it is applied once and undamped is fine; as a smoother it
is a stationary iteration, and undamped on this saddle-point system it does not converge.
SFEM's own multigrid damps its block-Jacobi by `1/block_size` for exactly this reason.
Measured (N=1, L=4, first four Newton steps):

| omega | lin_it per Newton step |
|-------|------------------------|
| 1.0   | 1000, 1000 (capped)    |
| 0.8   | 1000, 785, 334, 710    |
| 0.7   | 391, 131, 122, 352     |
| 0.6   | 164, 24, 548, 119      |
| 0.5   | 31, 16, 154, 31        |
| 0.4   | 41, 23, 143, 19        |
| 0.25  | 50, 29, 376            |

`SFEM_GMG_OMEGA` defaults to 0.5. The damping applies to the smoothers only; the coarse
solve and the flat block-Jacobi preconditioner are left undamped.

### It is not level-independent, which is the result that matters

Total linear iterations over four Newton steps, V-cycle against the flat block-Jacobi
preconditioner:

| level | V-cycle | block-Jacobi |
|-------|---------|--------------|
| 2     | 48      | 123          |
| 4     | 247     | 303          |
| 8     | 2140    | 1082         |

A working V-cycle holds iteration counts roughly flat as the lattice deepens. These grow
faster than the flat preconditioner's and overtake it by L=8, where the V-cycle is *worse*
than the smoother it is built from.

That first reading -- that the smoother was at fault -- was wrong, and the reasoning behind
it was wrong in a way worth recording. It rested on smoothing steps at L=8 reducing
iterations monotonically (3113, 2140, 1402, 480 for 1, 3, 6 and 12), read as evidence that
the coarse-grid correction was sound and only the smoother was weak. But a damped smoother
is a convergent iteration by itself, so a cycle whose coarse correction contributed nothing
whatever would improve with smoothing count in exactly the same way. Counted in operator
applies rather than iterations the same numbers say the opposite: 7114, 14671, 19223, 13162
against block-Jacobi's 1082. More smoothing was buying less, not more.

### The cost bar a V-cycle has to clear

A V-cycle with three pre- and three post-smoothing steps costs roughly sixteen operator
applies; the flat preconditioner costs one. So the V-cycle has to cut iteration counts by
more than about 16x merely to break even on wall time, not the 2-3x it currently manages.
That is not out of reach -- at L=8 block-Jacobi needs 1082 iterations and an effective
V-cycle would need well under 50, comfortably past the bar -- but it does mean an
almost-working smoother is worth nothing, and the smoother has to be most of the way to
level-independent before the machinery pays for itself.

Wall-clock numbers are not quoted here as a comparison. These runs are at N=1, far below
saturation, where per-apply overhead dominates and the measured 2.1 ms for a 425-node apply
is overhead rather than work. The iteration counts and their growth with level are the
meaningful signal at this size; a wall-clock claim needs a saturated problem and will be
worth making once the cycle is fixed.


## What is actually wrong with the V-cycle

Chasing the above produced a diagnosis, one real bug fixed, and a clear statement of what
still blocks the cycle. The instruments are in the driver behind `SFEM_GMG_CHECK`.

### A control arm that never ran

`SFEM_GMG=2` runs the same damped block-Jacobi as a stationary iteration on the fine level
for the same number of sweeps a V-cycle spends smoothing, with no hierarchy under it. It
exists because iteration counts cannot otherwise distinguish a weak smoother from a broken
coarse correction.

Its first results showed V-cycle and control agreeing to the digit -- 48 against 48, 470
against 470 -- which was not a finding but a bug: the hierarchy was built under `if
(use_gmg)`, so `SFEM_GMG=2` took the `if (gmg)` branch and ran the V-cycle. The control was
unreachable. It now builds only for `SFEM_GMG == 1`.

### The bug: the state was restricted with the residual's operator

Every coarse operator is linearised about a state restricted from the level above, and that
restriction was `create_hierarchical_restriction`. The adjoint test in `check_transfers`
shows that operator is exactly the transpose of the prolongation -- ratio 1.000000 on every
level, once the probe vectors respect the constraints that both transfers impose on their
output. (Probing with unconstrained noise reports a spurious mismatch; the first version of
this test did exactly that and produced ratios of 0.41 and 1.12, which read convincingly as
a broken transfer and were nothing of the kind.)

Being the adjoint is precisely right for the residual and precisely wrong for a state. `P^T`
sums where a state transfer must average, inflating each coarse state by the number of fine
nodes feeding a coarse node -- a measured factor of about 3.8 per level. Every coarse
operator was therefore linearised about a field several times too large. Normalising by `R`
applied to the constant 1 recovers the partition-of-unity average. The effect on the cycle's
own convergence rate at L=8 was the difference between diverging and converging:

| cycle | before | after |
|-------|--------|-------|
| 1     | 5.83   | 0.185 |
| 2     | 1.17   | 0.626 |

### What still blocks it: Rhie-Chow does not survive coarsening

The cycle still turns divergent after the second cycle, settling at about 1.34 per cycle at
L=8, and the V-cycle remains the worst of the three preconditioners:

| level | V-cycle | fine smoother, no hierarchy | block-Jacobi |
|-------|---------|-----------------------------|--------------|
| 2     | 48      | 42                          | 123          |
| 4     | 470     | 84                          | 304          |
| 8     | 2407    | 574                         | 918          |

The coarse-operator consistency check applies `A_c` and `R A_f P` to the same smooth coarse
vector and compares them per component. The rediscretised coarse operator disagrees with
the Galerkin operator the transfers imply by a factor of about six, and the disagreement is
almost entirely in the pressure rows:

| level pair | ux   | uy   | uz   | p    |
|------------|------|------|------|------|
| 0->1       | 0.79 | 0.72 | 0.76 | 6.59 |
| 1->2       | 1.59 | 1.30 | 2.42 | 5.24 |
| 2->3       | 0.00 | 0.00 | 0.00 | 6.01 |

That localises it to the stabilisation. `Df = rc_scale * h^2 / (2 mu)` is the one term that
depends on the lattice spacing outright, so each level stabilises a different equation, and
rediscretisation hands the cycle a coarse pressure operator that is not a coarse version of
the fine one. Holding `Df` at the fine level's value (`SFEM_GMG_RC_DECAY=0.25`) confirms the
mechanism -- the pressure inconsistency falls from about 6 to between 0.6 and 1.2.

The awkward part is that the same change makes the cycle *worse*, taking the L=8 rates to
0.41, 1.30, 1.52. A coarse operator stabilised for the fine level's `h` is closer to the
Galerkin operator and simultaneously under-stabilised on its own mesh, where it is near
enough singular that solving it amplifies what it returns. The two requirements point in
opposite directions, which is the real obstacle: consistency with the fine operator and
stability on the coarse mesh cannot both come from rediscretising with an h-dependent
stabilisation.

Nor is it a scalar. `SFEM_GMG_CGC` scales the prolonged correction; swept over 0.125 to 8 at
L=8, every value diverges eventually -- values below 1 delay it, values above accelerate it
sharply (4 gives 4.7 per cycle, 8 gives 17). A single factor per level cannot repair a
coarse operator that differs in what it does rather than by how much.

### Ruled out

Recorded so they are not re-investigated: the transfer pair (exact adjoints, ratio
1.000000); the pressure null space (every level carries exactly one pressure pin, and
filtering the constant pressure mode out of each prolonged correction with
`SFEM_GMG_PFILTER=1` changes the rates in the fourth decimal); hierarchy depth (capping at
two levels with `SFEM_GMG_MAX_LEVELS`, so the coarse level is the well-resolved L=4 mesh,
diverges at the same 1.33); the nodal pressure-gradient cache (`SFEM_PGRAD_CACHE=0`
reproduces the rates bit for bit); and smoother damping (swept; 0.5 is best and is the
default).

### Where this leaves the preconditioner

Block-Jacobi is still the one to beat, and in work rather than iterations it is not close.
At L=8 it spends about 918 operator applies against roughly 3400 for the no-hierarchy
smoother arm and some 19000 for the V-cycle. The fine-level stationary smoother wins on
iteration count at every level and loses on work at every level.

The next step is not a better smoother -- the evidence points away from that. It is the
coarse pressure operator: either a stabilisation that coarsens consistently, or a coarse
level built as a genuine Galerkin product for the pressure block instead of rediscretised.

## Independent evaluation of the null-space treatment

`nullspace_eval.py` is a standalone study of whether our constant-pressure null space is
what limits the V-cycle, and whether the hybrid matrix-free elimination from the
self-contact rigid-body-modes work helps if applied to it. It models a stabilised
colocated Navier-Stokes system in 2D with the same constant-pressure null space and the
same `Df = rc h^2 / (2 mu)` stabilisation, small enough to solve exactly.

It is gated rather than merely run. Stage 1 requires the model to reproduce the driver's
symptom before anything else is believed; stage 1b requires the smoother to converge at
all; stage 1c requires the condensed operator to solve the problem to round-off before its
cycle rate is quoted. All three gates fired during development and each caught a real
error: a symmetric-indefinite model whose smoother diverged at every damping, a pure Stokes
model missing the convective diagonal that makes our smoother work, a truncated inter-level
transfer, and a right-hand side that double-counted `B_tilde C_lam^-1 g_tilde` by adding
both of the paper's two equivalent forms for it.

The model reproduces our failure closely. Coarse-operator consistency is about 0.5 in the
velocity rows and 5.2 in the pressure rows, against 0.7 and 6.6 in the driver.

**The gauge does not matter.** Pinning the same node on every level, pinning a
level-dependent node, and projecting the constant mode out per level are
indistinguishable, and none is far from the smoother alone:

| treatment                  | rate (n=24) | n=16 | n=32 |
|----------------------------|-------------|------|------|
| pin, shared node           | 0.948       | 0.919| 0.972|
| pin, level-dependent node  | 0.939       | 0.779| 1.143|
| projection, per level      | 0.936       | 0.993| 0.944|
| condensation, per level    | 12.2        | 0.922| 23.3 |

No treatment wins consistently across sizes, which is itself the result: the differences
are noise around a cycle that is limited by something else. The condensation is the
exception in the wrong direction -- its operator is verified correct to 1e-12, so its
divergence is a real property of the scheme here and not an implementation fault, and it
worsens with problem size. That is not a mark against the method in its own setting: it
changes the gauge, and a gauge is not what ails us. It also has to coarsen a dense global
rank-one term on top of a stabilisation that already fails to coarsen.

**The stabilisation is the lever**, and it is non-monotone:

| rc scaled per level | pressure consistency | V-cycle rate |
|---------------------|----------------------|--------------|
| 1.0 (as now)        | 5.18                 | 0.948        |
| 0.5                 | 2.21                 | 0.905        |
| 0.25                | 0.75                 | **0.719**    |
| 0.125               | 0.29                 | 9.81         |

Consistency improves monotonically all the way down while the rate has an optimum at 0.25
-- exactly the value that holds `Df` at the fine level's value -- and then diverges. This
is the tension stated earlier made quantitative: consistency with the fine operator and
stability on the coarse mesh are competing requirements, and the optimum is interior.

One discrepancy to resolve rather than explain away: in the model `rc_decay = 0.25`
improves the cycle (0.948 to 0.719), while in the driver the same setting made it worse
(rates 0.41, 1.30, 1.52 against 0.185, 0.63, 1.05). The exponent is dimension-independent,
since `Df ~ h^2` either way, so 0.25 should be right in 3D too. Candidate causes are the
driver's Reynolds regime, its hierarchy depth, or something still wrong in the driver that
the model does not carry. That is the next thing to chase, and it is a much narrower
question than the one this evaluation started with.

## The smoother was divergent, and the iteration counts were noise

Two findings that overturn parts of the account above.

### The default damping made the smoother diverge

`SFEM_GMG_CHECK=3` runs the smoother standalone as the stationary iteration it actually is
inside a cycle. Its good showing as a BiCGStab preconditioner proved nothing: a Krylov
method tolerates a preconditioner that would diverge if iterated, and inside a V-cycle it
is iterated.

At the then-default `omega = 0.5` the residual falls for about twenty sweeps, bottoms out
near 4.2e-3, and then grows; the per-sweep rate rises monotonically through 1 at around
sweep 27 and reaches 1.038 by sweep 39. An earlier reading of this same measurement stopped
at eight sweeps, saw 0.88 to 0.95, and called the smoother convergent. The rate was still
rising at the point it was cut off.

Asymptotic rates over sweeps 35-39: `omega` 0.5 gives 1.038 and rising, 0.3 gives 0.9717
and flat, 0.15 gives 0.9855, 0.05 gives 0.9915. The default is now 0.35.

This is what the earlier `SFEM_GMG_CGC=0` test was pointing at and what nothing else
explained: with the coarse-grid correction switched off entirely the cycle still diverged
(0.68, 0.80, 0.87, 0.98, 1.11, 1.21), while the smoother allegedly converged. A cycle that
diverges with no coarse correction has nothing to do with its coarse grid.

With a convergent smoother the V-cycle converges as an iteration for the first time.
Cycle rates at L=8: `omega` 0.35 gives 0.25, 0.30, 0.35, 0.57, 0.51, 0.48; 0.3 and 0.25 are
similar; 0.5 still diverges to 1.32.

So the plan's P5 is back, and this time on direct evidence rather than on the inference
that was withdrawn: the smoother is genuinely inadequate here, and damping only moves it
from divergent to barely convergent at 0.97 per sweep.

### The iteration counts in this report carry about a factor of two of noise

Four runs of one identical configuration (L=8, `omega` 0.35, V-cycle) gave 3084, 2354, 2793
and 1485 total linear iterations. Block-Jacobi under the same treatment gave 993 and 979.

The V-cycle path performs far more operator applications, each carrying OpenMP atomic
rounding non-determinism, and an outer BiCGStab that is close to stagnating amplifies the
difference. The consequence is that any single-shot comparison of V-cycle iteration counts
in this document is unreliable at better than a factor of two, which covers the
`SFEM_GMG_PSCALE` sweep, the `omega` 0.35 against 0.5 comparison, and the earlier
level-independence tables. Differences of that size were read as signal and were not.

What survives is what was measured as a rate rather than a count -- the standalone smoother
and cycle rates, which are monotone and reproducible -- and the block-Jacobi-against-V-cycle
gap, which is larger than the spread. Block-Jacobi at about 985 still beats the V-cycle at
1485 to 3084, so the cycle is still not competitive; it has merely stopped diverging.

### What the independent evaluation does and does not transfer

`nullspace_eval.py` predicted that scaling the coarse continuity rows by the measured
pressure/velocity ratio of the best-fit block scales would fix the cycle, and in the model
it does: the predicted beta is optimal at n = 16, 24 and 32 without tuning. The driver
reports the same pathology -- best-fit scales of about 0.63 on velocity and 0.12 on
pressure, a ratio of 0.196 -- but `SFEM_GMG_PSCALE` at that value does not help, before or
after the damping fix, and the differences are inside the noise quantified above.

The model's own stage 1b gate required a convergent smoother before reporting anything.
The driver had no such gate until now, which is precisely how a divergent smoother survived
several rounds of coarse-grid investigation.

## P5 answered: a saddle-point smoother is not the fix

SIMPLE is implemented (`SimpleSmoother`, `SFEM_SMOOTHER=simple`) on the 2x2 block split,
which is what that split was built for. Following the rule the previous section learned the
hard way, it was measured as a standalone smoother with no coarse space before being allowed
anywhere near a cycle. Standalone rates at L=8 over sweeps 35-39:

| smoother | omega | rate |
|----------|-------|------|
| block-Jacobi | 0.35 | 0.9669 |
| SIMPLE       | 0.35 | 0.9667 |
| SIMPLE       | 0.7  | 1.850  |
| SIMPLE       | 1.0  | 3.059  |

SIMPLE matches block-Jacobi to four digits and diverges sooner as damping is relaxed.
Neither more inner sweeps nor rescaling the Schur diagonal changes it.

The block-split gate under `SFEM_GMG_CHECK=1` explains why, and is the reason the null
result is trustworthy rather than a suspected bug. The four blocks sum to the full Jacobian
action to 1.4e-16, so the split is exact, and their norms are `uu` 0.284, `up` 2.231,
`pu` 0.552, `pp` 23.748. The pressure-pressure block -- the Rhie-Chow stabilisation --
is about eighty-five times the momentum block, and the divergence coupling `pu` that SIMPLE
uses to build its pressure correction is a two percent perturbation on it. SIMPLE's Schur
complement `S = Dpp - C Du^-1 B` is therefore `Dpp` to within a couple of percent, its
pressure update reduces to block-Jacobi's, and its velocity correction is negligible.

So this system is not coupling-limited and a saddle-point smoother has nothing to work with.
That is a different diagnosis from the one P5 was written under: the difficulty is not that
velocity and pressure are strongly coupled, it is that the stabilisation dominates the
operator outright.

It is also worth correcting an impression left by the previous section. An asymptotic
smoother rate near 0.97 is not by itself a bad smoother -- a smoother's asymptotic rate is
set by the smoothest mode, which is precisely what the coarse grid exists to remove, and
good multigrid smoothers routinely look terrible measured this way. What is fatal is a rate
above 1, which is what the old default damping produced. With that fixed the smoother is
doing its job, and the remaining weakness is in the coarse correction, where the velocity and
pressure rows still coarsen with different best-fit scales (0.63 against 0.12).

## Why the model's fix did not transfer: rediscretisation is the whole fault

`SFEM_GMG_CHECK=4` applies the two-level correction operator `P A_c^-1 R A` to a chosen
error mode and reports what fraction survives. The mode is built as `P` applied to a coarse
field, so it is exactly representable on the coarse grid and a correct correction must
remove essentially all of it. `SFEM_GMG_GALERKIN=1` swaps the rediscretised coarse operator
for `R A P`, composed matrix-free, which is far too expensive for production and is exactly
the right thing for a diagnostic: with it the surviving fraction is zero by construction if
the transfers are sound. `SFEM_GMG_CGC_SMOOTH=1` seeds two levels down instead of one, so
the mode is smooth relative to the coarse grid rather than oscillatory on it.

| coarse operator | mode | velocity | pressure |
|-----------------|------|----------|----------|
| rediscretised   | coarse-oscillatory | 5.52 | 0.786 |
| rediscretised   | coarse-smooth      | 0.312 | 0.601 |
| Galerkin `R A P`| either             | 0.000 | 0.000 |

The Galerkin correction is exact, which validates the transfers and the test at once. The
rediscretised operator amplifies a coarse-representable velocity error more than fivefold,
and removes only forty percent of a coarse-smooth pressure error. Rediscretisation is the
entire fault; nothing else in the cycle is.

Two checks close off the alternatives. The derefined coarse operator is bit-identical to one
built directly on the coarse space -- `derefine_op` is not the problem. And the disagreement
is not a scaling: removing each component's own best-fit scale still leaves 0.68, 0.52, 0.52
and 0.40 relative error in ux, uy, uz and p. That is why `SFEM_GMG_PSCALE`, `SFEM_GMG_CGC`
and `SFEM_GMG_RC_DECAY` all failed -- the entire family of scaling knobs was addressing a
component of the error that is a minority of it.

This is also the answer to why the independent evaluation's prediction did not transfer. In
the model the mismatch between the coarse and Galerkin operators really was close to a pure
per-block scaling, so scaling the coarse continuity rows by the measured ratio fixed it. In
the driver it is not, so no scaling can. The model was right about itself and about the
method; it was wrong about the driver because the two operators fail in different ways, and
only measuring the after-scale residual in both revealed that.

The V-cycle's behaviour follows exactly. Run long enough its rate climbs to 0.963, against
the smoother's own 0.967: the coarse correction helps for a few cycles, then contributes
nothing, and the residual left behind is pressure, reduced fifty times less than velocity.

### What this means for the next step

Galerkin coarsening works and rediscretisation does not, at least for the pressure block.
Composing `R A P` at solve time is what the diagnostic does and is not an option here --
it makes every coarse application cost fine-level work, which defeats the hierarchy. So the
choice is between assembling the coarse levels once per Newton step and finding a coarse
discretisation that behaves like the Galerkin operator without being it. The measurements
above are the gate either way: any candidate coarse operator should be required to bring the
surviving fraction near zero on coarse-representable modes before it is put into a cycle.

## Galerkin coarse operators, assembled once per Newton step

`SFEM_GMG_GALERKIN=2` assembles `A_c = R A P` into BSR once per Newton step and applies it
as a sparse matrix, so no coarse level reaches back up to a finer one during the solve.
(`=1` keeps the matrix-free composition, which is the diagnostic, not a solver: it puts
fine-level work under every coarse application.)

Assembly does two jobs. It removes the fine-level dependency, and it supplies the coarse
smoother with the diagonal of the matrix it actually smooths -- the matrix-free composite
cannot, and using the rediscretised diagonal instead mismatches the Galerkin operator by the
per-block scale factors (about 1.6 in velocity, 8 in pressure), which alone made the coarse
levels diverge.

The entries are recovered by probing under a distance-2 colouring of the coarse node graph,
so no row ever sees two neighbours of one colour and a whole set of blocks falls out per
application. That is colours x 4 applications instead of one per coarse degree of freedom:
41 colours and 164 applications for 425 nodes, against 1700 for column-by-column.

The pattern is self-correcting, and needs to be. Probing does not drop a non-zero that lies
outside the pattern -- it folds it into the wrong entry, so too narrow a pattern yields a
wrong matrix rather than an approximate one. The coarse mesh graph is right while the mesh
is fine enough that `R A P` does not reach past it, and is wrong on the coarsest levels,
where a handful of nodes are all within reach of each other. The gate caught exactly that:
levels 1 and 2 assembled to 2e-16 while the 20-node coarsest level came out at 3.8e-1. It
now widens to the squared adjacency and then to a dense pattern, and all levels assemble
exactly:

```
gate 2.1236e-16  OK   425 nodes, 8281 blocks, 41 colours, 164 applications
gate 2.7612e-16  OK    81 nodes, 1225 blocks, 31 colours, 124 applications
gate 1.5981e-16  OK    20 nodes,  400 blocks, 20 colours,  80 applications  (dense pattern)
```

### It works two-level and fails multi-level, for a specific reason

Cycle rates at L=8, first three and last three of twelve:

| hierarchy | coarse handling | rates |
|-----------|-----------------|-------|
| rediscretised, 2 levels | solved | 0.096, 0.546, 0.850 ... 0.964 |
| Galerkin, 2 levels      | solved | 0.021, 0.207, 0.238 ... 0.861 |
| Galerkin, 4 levels      | smoothed | 0.433, 5.319, 5.659 ... 5.671 |

Two-level Galerkin is a clear improvement and behaves as the correction-operator measurement
predicted. Four-level Galerkin diverges, and not for want of damping: omega 0.35, 0.2, 0.1
and 0.05 give 5.67, 5.23, 3.09 and 1.32, improving steadily and never reaching 1.

The difference between the two rows is not the number of levels but what happens on the
intermediate ones: solved in the first case, smoothed in the second. The Galerkin operator is
a much better approximation of the fine operator and a much worse candidate for block-Jacobi
smoothing -- it is denser, and its diagonal is not dominant in the way the rediscretised
operator's is. That is the standard trade between the two coarsenings, and it is now the
binding constraint rather than a suspicion.

### Where that leaves it

The two coarse operators fail in opposite directions. Rediscretisation is smoothable and
approximates badly enough that its correction is worthless; Galerkin approximates well and
cannot be smoothed by the smoother available. Two-level Galerkin with a solved coarse level
sidesteps the conflict and is the best cycle measured so far, at 0.861 against 0.964.

The next thing to try is therefore not another coarse operator but a stronger coarse-level
solver: a few Krylov iterations per level in place of the stationary smoother, which
tolerates an operator that block-Jacobi cannot smooth. That carries a consequence worth
stating before it is measured -- a Krylov smoother makes the preconditioner vary between
applications, which BiCGStab does not admit, so the outer solver would have to become
flexible (FGMRES) at the same time.

## Krylov smoothing and a flexible outer solver: the V-cycle finally works

Two changes that had to land together. `SFEM_GMG_KSMOOTH=n` replaces the stationary smoother
with n BiCGStab iterations per level, which does not need the diagonal dominance the
Galerkin operators lack. That makes the cycle vary between applications, and BiCGStab
assumes its preconditioner does not -- it does not fail loudly when that is violated, it
stagnates -- so `cvfem_fgmres.hpp` adds flexible GMRES, selected automatically whenever the
smoother is Krylov. SFEM had no GMRES of any kind.

FGMRES was gated before use: with a fixed block-Jacobi preconditioner it reaches the same
solution as BiCGStab (u_linf 1.6569e-03 against 1.6558e-03). It needs more iterations there,
which is expected -- restarted GMRES discards information at each restart and BiCGStab
performs two operator applications per iteration -- and is beside the point, since it exists
for the case BiCGStab cannot handle at all.

### Total linear iterations over four Newton steps

| refine level | dofs   | block-Jacobi | Galerkin + Krylov smoothing + FGMRES |
|--------------|--------|--------------|--------------------------------------|
| 2            | 324    | 123          | 20                                   |
| 4            | 1700   | 305          | 29                                   |
| 8            | 10692  | 959          | 59   (ksmooth 8)                     |
| 16           | 75140  | 2954         | 82   (ksmooth 16)                    |

Block-Jacobi grows by about a factor of three per refinement; this grows by about half that.
At the largest size measured it is a factor of thirty-six fewer iterations, and unlike the
rediscretised V-cycle it is stable run to run -- 89 and 102 on repeats, against 3084 and
1485 for the arm that was being read as signal earlier.

This is the first configuration in which the V-cycle does what it was built for.

### What it costs, and what is not yet shown

Wall time, L=16, two repeats: block-Jacobi 9.8 and 13.2 seconds, this 25.5 and 29.3. Thirty-six
times fewer iterations and still about twice the time, because each iteration now carries a
cycle whose levels each run sixteen preconditioned BiCGStab iterations, plus the per-Newton
assembly. Smoothing strength is near optimal at that setting: at L=16, ksmooth 10, 12, 16 and
24 give 77.1, 72.9, 23.5 and 27.1 seconds.

Two honest limits. First, smoothing strength has to grow with depth -- eight iterations
suffice at four levels and give 1194 iterations at five, where sixteen give 99. That the
cycle needs more smoothing as it deepens says the smoother is still the weak component, and
it eats into the iteration gain because the cost per cycle rises with it. Second, the
crossover in wall time was not demonstrated: the iteration counts diverge fast enough that
one should exist, but L=32 exceeded the time available here, so that remains a projection
rather than a measurement, and projections of exactly this kind have been wrong twice
already in this document.

## State of the code, and where the solver is matrix-free

### The shape of the method

The solver is matrix-free where it is large and matrix-based where it is small, and that
split is not a compromise -- each half was forced by a measurement.

**Matrix-free, and staying that way: everything on the fine level.** The CVFEM
Navier-Stokes operator over the semi-structured `sshex8` lattice is never assembled. The
residual, the Jacobian action, the 4x4 block diagonal, and the 2x2 (velocity, pressure)
block split are all element sweeps over macro-elements. The grid transfers are matrix-free
lattice operations from `smesh`. The fine-level smoother's block-Jacobi is built from
`hessian_block_diag`, which is `n_nodes x 16` values -- O(n) storage, not a matrix. This is
the reason the semi-structured hierarchy exists and none of it changed.

**Matrix-based, once per Newton step: the coarse levels.** Each coarse operator is an
assembled BSR matrix formed by Galerkin coarsening, `A_c = R A P`, with entries recovered by
probing under a distance-2 colouring. During the solve the coarse levels apply a sparse
matrix and never touch a finer level.

### Why the boundary sits there

Three measurements put it there, in order.

The rediscretised coarse operator -- the matrix-free choice, and the one the hierarchy was
built around -- does not work. Applying the two-level correction operator to a mode that is
exactly representable on the coarse grid leaves 5.52 of a velocity mode (it amplifies the
error) and 0.79 of a pressure mode, where the Galerkin operator leaves 0.000. The
disagreement is not a scaling: removing each component's own best-fit scale still leaves
0.68, 0.52, 0.52 and 0.40 in ux, uy, uz and p, which is why every scaling knob tried
(`PSCALE`, `CGC`, `RC_DECAY`) did nothing.

Galerkin cannot be applied matrix-free. Composing `R A P` at solve time works and is kept
as `SFEM_GMG_GALERKIN=1`, but it puts a fine-level application under every coarse
application, so cost stops falling geometrically with depth. That is the one thing a
hierarchy must not do.

Assembling also fixes a second problem that has nothing to do with cost. A coarse smoother
needs the diagonal of the operator it smooths; a matrix-free composite cannot supply one,
and substituting the rediscretised diagonal mismatches the Galerkin operator by the
per-block scale factors -- about 1.6 in velocity, 8 in pressure -- which by itself made the
coarse levels diverge. An assembled matrix hands over its own diagonal.

### What being matrix-based actually costs

The fine matrix is still never formed, and that is the whole point: the assembled hierarchy
is the coarse levels only.

| | blocks | memory |
|---|--------|--------|
| assembled coarse hierarchy (L=16, N=1) | 70531 | 8.6 MiB |
| the fine BSR, which is never formed | 507195 | 61.9 MiB |

The hierarchy costs about 14% of what assembling the fine level would, because each level in
3D is roughly eight times smaller than the one above it and the sum is dominated by the
first coarse level rather than the fine one.

Assembly costs 548 probe applications per Newton step at that size, of which only the 180 at
the finest transfer involve a fine-level operator application; the rest are sparse
applications on already-assembled coarse levels. Probing is what makes this affordable at
all -- a distance-2 colouring means one application reveals a whole set of blocks, so the
count scales with the stencil rather than with the number of coarse unknowns (41 colours and
164 applications for 425 nodes, against 425 column by column).

### The rest of the algorithmic configuration

The coarse levels are smoothed with BiCGStab rather than a stationary iteration, because the
Galerkin operators are denser and lack the diagonal dominance block-Jacobi needs -- under
block-Jacobi they diverge at every damping. That makes the cycle vary between applications,
so the outer solver is FGMRES rather than BiCGStab; the two changes are one change, and the
driver selects FGMRES automatically whenever the smoother is Krylov.

Recommended configuration as measured:

```
SFEM_GMG=1  SFEM_GMG_GALERKIN=2  SFEM_GMG_KSMOOTH=16  SFEM_GMG_OMEGA=0.35
```

### What is settled and what is not

Settled: the V-cycle reduces iterations by a factor of twenty-three to thirty-six against
block-Jacobi and, unlike every earlier configuration, does so reproducibly. On Grace at
L=16, 60 iterations against 1408, reaching the same solution.

Not settled: it is not yet faster. Same Grace run, 7.59 seconds against 4.23. Thirty-six
times fewer iterations and 1.8 times the wall clock, because each iteration carries sixteen
preconditioned BiCGStab iterations per level plus the per-Newton assembly. The gap is
narrower on Grace than on the M1 (1.8 against 2.4), which is the direction one would expect
from sparse coarse work vectorising better there, but a single pair of runs is not evidence
of a trend.

Also unsettled, and the reason the crossover has not been demonstrated: refine level 32 does
not exist. It aborts in `smesh` with "Invalid element setup for proteus hex: 32", so the
larger problem has to come from more macro-elements at a valid level rather than a deeper
lattice. That sweep (N = 2 and 3 at L = 16) is running.

### Instrumentation

All of it is behind `SFEM_GMG_CHECK`, and all of it exists because something got past its
absence.

- `=1` transfers and their adjointness, a constraint census per level, the block-split sum
  against the full operator, and coarse-operator consistency reported three ways: raw, the
  per-component best-fit scale, and the residual left after removing that scale.
- `=2` the cycle's own convergence rate standalone, plus the stalled residual split by
  component.
- `=3` the smoother alone, with no coarse space. This is the first thing to run when a cycle
  misbehaves, and running it long enough to see the asymptote is part of the check: a rate
  that is still moving when the measurement stops has not been measured.
- `=4` the two-level correction operator applied to a prescribed mode, rediscretised against
  Galerkin, on coarse-oscillatory or (with `SFEM_GMG_CGC_SMOOTH=1`) coarse-smooth modes.
- The Galerkin assembly gates itself against the composite it was probed from and widens its
  sparsity pattern until it agrees, because probing folds a non-zero lying outside the
  pattern into the wrong entry rather than dropping it.

## Where the time actually goes

Cost had been inferred from operator-application counts, which is a model rather than a
measurement: it assumes every application costs the same and ignores the sparse coarse work,
the transfers and the assembly. The driver now times each phase directly and prints a
breakdown (SFEM's own tracing needs `SMESH_ENABLE_TRACE` compiled into smesh, which the
installed one lacks). Note that `precond_total` contains the `smooth[*]` rows, so the
"accounted" total double counts it.

### The coarse levels were paying for threads they could not use

The first breakdown showed the coarse smoothers costing 22.5, 21.3 and 21.2 ms per call on
levels of 2673, 425 and 81 nodes -- flat, where work should fall roughly eightfold per
level. It is not arithmetic: 324 unknowns cannot take 21 ms. It is the cost of starting a
thread team for each vector operation on a level with nothing to distribute.

Per smoother application on the 81-node level, and the whole solve:

| threads | smooth[L3] | total wall |
|---------|------------|------------|
| 1       | 0.156 ms   | 14.87 s    |
| 4       | 4.64 ms    | 4.85 s     |
| 8       | 14.20 ms   | 13.00 s    |

Ninety times slower for having eight cores instead of one, and the whole solve is fastest at
four threads and slower at eight. Each level now runs with a thread count matched to its own
size rather than the machine's (`SFEM_GMG_DOFS_PER_THREAD`, default 20000). After that the
coarse levels decay as they should -- 5752, 778 and 161 us per call across the three -- the
thread-count penalty is gone, and L=16 N=1 goes from 23.5 s to 18.3 s.

This also disposes of the large-case results reported above. Those Grace runs used 72
threads, where this penalty is far worse than at eight, so the 12x to 80x slowdowns at N=2
and N=3 are not a property of the method and those numbers should not be read as one. They
need re-running.

### What remains is fine-level smoothing, and it is the whole story

With the coarse levels fixed, the breakdown at L=16, N=1, eight threads is:

| phase | seconds | share |
|-------|---------|-------|
| smooth[L0] (fine) | 10.36 | 42% |
| galerkin_assembly | 1.47 | 6% |
| all coarse levels together | 0.78 | 3% |
| transfers, coarse solve, everything else | <0.4 | <2% |

A fine operator application costs 2.43 ms. The baseline spends 959 iterations x 2 = 1918 of
them. The cycle spends 86 iterations x 64 -- two smoothing applications per cycle, sixteen
BiCGStab iterations each, two applications per iteration -- which is 5504.

That is the arithmetic of the whole problem, and it is not about the hierarchy at all. To
break even the cycle may spend at most about 22 fine applications per cycle and it spends
64; equivalently, iteration count would have to fall by 32x and it falls by 11x. The
assembly, the transfers and every coarse level together account for under 10% and are not
where the decision lies.

The lever is therefore the fine-level smoother: it has to become roughly three times cheaper
per cycle without giving back the iteration count. Sixteen BiCGStab iterations there is
strong smoothing bought at two operator applications each; the alternatives worth measuring
are fewer Krylov iterations, a stationary sweep at one application each, or a Chebyshev
smoother, which would need an eigenvalue estimate but costs one application per sweep.

### Correction: the breakdown percentages above were normalised wrongly

`precond_total` is a container -- it wraps the whole V-cycle, so the smoother, operator,
transfer and coarse-solve rows sit inside it. Summing every row double counts, and shares
taken against that sum understate everything. The table in the previous section put
fine-level smoothing at 42% on that basis, and there appeared to be half the runtime
missing. There is not. Against wall time, with containers excluded from the denominator,
top-level phases account for 99.6% of the run:

| phase | seconds | share of wall |
|-------|---------|---------------|
| the V-cycle (`precond_total`) | 12.782 | 87.7% |
| `galerkin_assembly` | 1.609 | 11.0% |
| outer Krylov operator applications | 0.115 | 0.8% |
| Newton residual, block diagonals | 0.012 | 0.1% |

and inside the V-cycle:

| phase | seconds | share of wall |
|-------|---------|---------------|
| `smooth[L0]`, the fine level | 11.495 | 78.9% |
| `smooth[L1]` | 0.732 | 5.0% |
| transfers | 0.172 | 1.2% |
| `op[L0]` | 0.164 | 1.1% |
| `smooth[L2]`, `smooth[L3]`, coarse solve | 0.124 | 0.9% |

So fine-level smoothing is 79% of the solve, not 42%, and the conclusion drawn from the
wrong normalisation is strengthened rather than changed: the fine smoother is the only thing
worth optimising, the assembly is a real but secondary 11%, and everything below the fine
level together is under 8%. The reporter now excludes containers from its denominator and
labels them, so the table cannot be read this way again.

## Why it was slower, and the configuration that is not

The V-cycle's fine-level smoother was itself BiCGStab preconditioned by block-Jacobi -- the
same solver the whole cycle is competing against. So the cycle was running the baseline
solver as a subroutine, sixteen iterations at a time, twice per cycle.

Counted in fine operator applications per outer iteration:

| | applications per outer iteration |
|---|---|
| baseline BiCGStab + block-Jacobi | 2 |
| V-cycle with `ksmooth` 16 on the fine level | 64 = (pre + post) x 16 iterations x 2 |

Thirty-two times the work per iteration, against an eleven-fold reduction in iterations. The
cycle needed the iteration count to fall by 32x to break even and it fell by 11x, which is
the entire explanation for a method that was measurably better per iteration and measurably
worse per second. Nothing about the hierarchy was involved.

The fix is to stop smoothing the fine level like a solver. Coarse levels keep sixteen
BiCGStab iterations, because the assembled Galerkin operators genuinely need them and are
cheap; the fine level takes two. That is `SFEM_GMG_KSMOOTH_FINE`, now defaulting to 2.

Three repeats each, L=16, N=1, eight threads, two Newton steps:

| arm | t_solve (s) | iterations |
|-----|-------------|------------|
| baseline BiCGStab + block-Jacobi | 6.98, 7.07, 4.64 | 1625, 1625, 1064 |
| V-cycle, fine 1 | 5.25, 7.09, 6.87 | 133, 179, 174 |
| **V-cycle, fine 2** | **3.25, 4.84, 2.99** | 65, 97, 60 |
| V-cycle, fine 4 | 8.09, 11.37, 6.01 | 112, 157, 83 |
| V-cycle, fine 16 | 27.12 | 66 |

All reach the same solution (4.29148e-03 against the baseline's 4.29289e-03). At two
fine iterations the cycle is about twice as fast as the baseline on median, having been four
times slower at sixteen. This is the first configuration in this document that is faster
rather than merely fewer-iterations, and it took the phase measurement to find, because the
whole cost was in one line of the breakdown.

The two fixes compound: clamping the coarse levels' thread count made coarse smoothing cheap
enough that spending on it is affordable, which is what makes a weak fine smoother viable.
Measured before the clamp, cheap fine smoothing looked worse, and that reading is what
delayed this by a round.

## The 2x speedup does not survive to a large problem

The configuration that beat the baseline at L=16, N=1 on a laptop fails at N=3 on Grace.
Measured at 1,853,572 unknowns, 72 threads, two Newton steps:

| arm | iterations | t_solve | u_linf |
|-----|-----------|---------|--------|
| baseline BiCGStab + block-Jacobi | 1702 | 12.42 s | 1.30e-03 |
| baseline, repeat | 1888 | 13.78 s | 5.80e-03 |
| V-cycle, fine smoothing 2 | 2000 (cap) | 382.6 s | 1.99e-01 |
| V-cycle, fine smoothing 16 | 518 | 201.7 s | 1.49e-03 |

Weak fine smoothing does not merely lose here, it fails: the linear solve exhausts its
iteration cap and returns an answer two orders of magnitude off. Strong fine smoothing
converges to the right answer and takes fifteen times as long as the baseline. There is no
setting at this size that is both correct and competitive.

The pattern across every measurement in this document is now consistent: the smoothing
strength this cycle needs grows with the problem, and the cost of that smoothing is what
sinks it. It needed 8 iterations per level at four levels and 16 at five; it works at 2 on
the fine level at 75k unknowns and fails at 2 at 1.85M. Each time the requirement rises the
per-cycle cost rises with it, and the iteration count does not fall fast enough to pay.

So the honest state is that the V-cycle is faster than block-Jacobi on one problem size on
one machine, and slower or wrong everywhere else that has been measured. The earlier
sections reporting the 2x win should be read with this one attached.

### Two implementation notes attached to the same runs

The thread clamp is off by default. Resizing the OpenMP team per operator application costs
more than it saves once the team is large: at 72 threads it made the coarse smoothers slower
than leaving them alone (33.3 ms against 14.0 ms per call on level 1), having made them
faster at eight threads. The underlying problem -- coarse levels cannot use 72 threads -- is
real and unsolved; the clamp moved the cost rather than removing it.

An attempt to fix that by giving small levels a hand-written serial apply was reverted. It
was slower (30.5 s against 3.0 s at L=16, N=1) and, more seriously, it changed the iteration
count from 60 to 220 consistently, which is the signature of a wrong operator rather than a
slow one -- most likely a block-layout mismatch, since the assembly gate only ever validated
the values against `h_bsr_spmv` and not against a second reader of the same array. Any
future serial path needs its own gate against the parallel one before it is trusted.

## There is no threading bug, but threading has been flattering the baseline

Running one thread against many, on a small case, settles several things at once.

**No race.** Every deterministic check is bit-identical at 1, 2, 4 and 8 threads: the
Galerkin assembly gates (2.3509e-16, 1.7047e-16, 1.3501e-16), the block-split norms and
their sum against the full operator, and the transfer adjointness. Where both solvers
converge they agree to seven digits (u_linf 7.167056e-03 against 7.167062e-03). The
operators, transfers and assembly are thread-independent.

**Single-threaded runs are exactly reproducible and multithreaded ones are not.** At L=16,
N=1 the V-cycle gives 102, 102, 102 iterations on one thread and 70, 56, 75 on eight; the
baseline gives 2000, 2000 and 1125, 1625. That is reduction order in the Krylov dot
products, not a defect, but it is worth knowing that every multithreaded iteration count in
this document carries roughly twenty percent of noise.

**And the noise helps the baseline, which invalidates the comparisons above.** On one thread
at L=16, N=1:

| arm | iterations | t_solve | u_linf | p_linf |
|-----|-----------|---------|--------|--------|
| baseline BiCGStab + block-Jacobi | 2000 (cap) | 18.98 s | 6.275e-03 | 1.605e-02 |
| V-cycle | 102 | 7.35 s | 4.291e-03 | 2.257e-03 |

The baseline does not converge single-threaded. It stagnates and exhausts its cap, and the
answer it returns is wrong -- 4.29e-03 is the value every converged run in this document
produces, and its pressure error is seven times worse. On eight threads the same solver
converges in 1125 to 1625 iterations. Rounding noise from threaded reductions is perturbing
a stagnating BiCGStab enough to break it out, which is a known behaviour and is pure luck.

So the baseline this method has been measured against was being helped by an accident of
parallel reduction order, and every multithreaded comparison reported above is a V-cycle
against a baseline that is quietly getting a free restart. Single-threaded, where both
solvers are deterministic and the comparison is honest, the V-cycle takes 102 iterations and
7.35 seconds and is correct, while the baseline takes 2000, nineteen seconds, and is wrong.

**What remains true is the parallel efficiency gap.** From one to eight threads, t_solve goes
19.08, 6.50, 6.79, 7.08 for the baseline and 7.44, 8.41, 3.08, 3.42 for the V-cycle. Neither
scales past two to four threads on this machine -- the problem is memory bound -- and much of
the baseline's apparent gain is its iteration count falling rather than its work speeding up.
The V-cycle scales worse in the sense that matters, because its coarse levels cannot use the
threads at all, and at 72 threads on Grace that is the difference between the two results.

This does not rescue the Grace numbers, and the large-case conclusion stands: at 1.85M
unknowns the V-cycle is still far slower there. But it does mean the laptop comparisons were
measuring the wrong thing, and that the correct single-threaded comparison at 75k unknowns
favours the V-cycle by more than the earlier multithreaded one suggested.

## With an assembled fine operator: the baseline's stagnation was an artefact

`SFEM_ASSEMBLE_FINE=1` replaces the matrix-free fine operator with a BSR one, probed by the
same coloured probing the Galerkin levels use (null transfers assemble A rather than R A P).
The motivation is determinism: a matrix-free apply accumulates through atomics, so its
summation order follows the thread schedule, while a BSR apply accumulates each row in one
thread.

It does what it should, and it corrects the previous section.

**The operator becomes thread-independent.** The assembly gate is identical at 1 and 8
threads (1.7650e-16, 2.3080e-16), and at L=8 the V-cycle takes exactly 36 iterations at both
thread counts where matrix-free gave 36 and 37.

**The solver does not.** At L=16 the assembled baseline gives 1862 and 2000 iterations on
two runs at 8 threads. Removing the operator's non-determinism leaves the Krylov method's
own: the dot products are OpenMP reductions and their order still follows the schedule. A
deterministic operator is necessary for a reproducible parallel solve and not sufficient.

**And the baseline's single-threaded stagnation was specific to the matrix-free operator.**
The previous section reported that the baseline fails to converge on one thread -- 2000
iterations, capped, u_linf 6.28e-03 against the correct 4.29e-03 -- and concluded that the
V-cycle wins by 2.6x once the comparison is made deterministically. With the assembled
operator the baseline converges on one thread in 1873 iterations to u_linf 4.292937e-03. The
stagnation was an accident of the matrix-free operator's rounding, not a property of
BiCGStab on this problem, and the conclusion drawn from it is withdrawn.

The honest comparison at L=16, N=1, with a deterministic operator on both sides:

| arm | threads | iterations | t_solve | u_linf |
|-----|---------|-----------|---------|--------|
| baseline | 1 | 1873 | 5.05 s | 4.292937e-03 |
| V-cycle | 1 | 112 | 7.78 s | 4.291494e-03 |
| baseline | 8 | 1210 | 3.48 s | 4.223545e-03 |
| V-cycle | 8 | 86 | 4.55 s | 4.291483e-03 |

The V-cycle uses seventeen times fewer iterations and is about 1.4x slower, at both thread
counts, and the ranking no longer depends on how many threads are used or on which operator
the baseline happens to get. That is the first comparison in this document that is stable
under both, and it says the V-cycle is not yet competitive at this size -- by a much smaller
margin than the multithreaded matrix-free numbers suggested, and in the opposite direction
from the single-threaded ones.

Assembling the fine operator costs 1.85 s single-threaded and 0.90 s on eight, which is
already counted in the timings above.

## The packed layout, brought to the semi-structured path

The flat HEX8 path won on CPU with a packed mesh: each pack writes its exclusively owned
nodes straight out, stages the shared ones, and a second pass gathers each shared node's
contributions in a fixed order. The semi-structured path never had it. `initialize()`
returns early for a semi-structured mesh, before the packing block, so those kernels
accumulated locally within a macro-element and then scattered every node with `atomic_add`.

That is where the irreproducibility came from. Atomic ordering follows thread timing and
floating-point addition is not associative, so with 27 macro-elements on 8 threads the same
operator application gave 0.080346978588455187 and 0.080346695382904176, while one thread
gave the same value every time. The earlier check that appeared to show determinism was run
at N=1, where there is a single macro-element and the atomics never contend -- a test that
could not have failed.

`SFEM_SS_SCATTER=1` (default) applies the packed layout's structure. It is simpler here
because the split is geometric: every lattice node strictly inside a macro-element belongs to
it alone, and only the skin is shared -- (L+1)^3 - (L-1)^3 nodes, 37% at L=4 falling to 12%
at L=16. Interior nodes are written directly, skin nodes are staged, and a second pass sums
each shared node's contributions in element order. It covers the Jacobian action and the
nodal pressure gradient, which had its own set of atomics and fed the apply, so fixing only
the first left the operator non-reproducible.

Measured at N=3, L=4, 8 threads, with the pressure-gradient cache off so both kernels are
exercised:

| | repeat-diff within a run | two runs, same thread count |
|---|---|---|
| atomic scatter | 5.551e-17 | 0.080346978588455187, 0.080346695382904176 |
| two-pass scatter | 0.000e+00 | 0.080346967363645994, 0.080346967363645994 |

The operator is now bit-reproducible run to run at a fixed thread count, and it is faster:
at N=3, L=8 the solve takes 8.79 to 9.73 s against 9.80 to 11.26 s, about 11%.

Two things it does not do. Results still differ between one thread and eight, but that
difference is in the initial state (206.55554994212025 against 206.55554942713621), which is
a setup-phase reduction and not these kernels. And the full solve is still not reproducible
run to run -- 1284, 1228, 1355 iterations -- because the Krylov method's dot products are
OpenMP reductions whose order still follows the schedule. That is the same conclusion the
assembled-operator experiment reached from the other direction: a deterministic operator is
necessary for a reproducible parallel solve and not sufficient. Closing it needs a
fixed-order reduction in the BLAS layer.

Still atomic, and worth the same treatment: the residual, the block diagonal, and the
2x2 block-split kernels, none of which are on the path measured above.

### The remaining kernels

The residual, the block diagonal and the 2x2 block-split applies now use the same two-pass
scatter. The helpers are templated on the number of values per node, since the block diagonal
carries sixteen rather than four, and the tables are shared across all of them.

One thing changed rather than being preserved. The block-split scatter previously wrote only
the components its block selection touches -- "a continuity-row block touches one component
of four, and the atomics are the expensive half of the scatter". That saving existed because
atomics were expensive; with a plain write it is worth nothing, and the untouched components
are zero in the element buffer anyway, so the two-pass path moves all four.

Determinism, N=3 L=4 on 8 threads, repeat-diff within a run:

| kernel | atomic | two-pass |
|--------|--------|----------|
| residual | 8.674e-19 | 0.000e+00 |
| block diagonal | 5.551e-17 | 0.000e+00 |
| block split | 1.735e-18 | 0.000e+00 |

And they are faster, in line with the Jacobian action's 11%: the residual goes from 3918 to
3528 us per call and the block diagonal from 7524 to 6952, about 10% and 8%.

Correctness is unchanged -- `cvfem_ns_op_gate` passes with the scatter on and off, at the
same tolerances (grad 1.5e-15, bsr 1.2e-16, apply 4.3e-16, blockdiag 5.9e-16, diag 2.4e-17,
asm_vs_mf 1.7e-14). `cvfem_sshex8_bench` reports three disagreeing configurations both with
and without these changes, and with the kernel edits stashed, so that failure is pre-existing
and not caused here. It is worth chasing separately.

What is left is not in these kernels. With the scatter on, the residual output tracks the
input state exactly: two runs sharing a state checksum of 206.55554942713621 both give
-0.6316666704417313, and the run that produced 206.55554979906918 is the only one that
differs. The state itself still varies between runs at 8 threads and is stable at one, so the
remaining non-determinism is in the setup that builds the initial field, not in the operator.
That and the Krylov reductions are what stand between this and a reproducible parallel solve.

## Measures for the remaining non-determinism

Four candidate sources were checked, not assumed. Two are clean, one is fixed, and two need
work outside this spike.

| source | status | evidence |
|--------|--------|----------|
| element scatter (operator) | **fixed** | repeat-diff 0.000e+00 across all five kernels |
| grid transfers | clean | R and P checksums bit-identical over three runs |
| mesh coordinates | **broken** | x varies at 8 threads, stable at 1; y and z exact everywhere |
| Krylov reductions | broken by construction | `#pragma omp parallel for reduction(+ : ret)` |

### 1. Mesh coordinates, in `smesh::to_semistructured`

The x coordinate checksum varies between runs at 8 threads (16562.000025525689,
16562.000030174851) and is stable at one (16562.000023022294), while y and z are identical
everywhere. Only x, which fits shared lattice nodes being written by more than one
macro-element: y and z land on exactly representable values so the order cannot matter, and
x does not. Everything downstream inherits it -- the analytic pressure seed varies by 20%
in its (near-cancelling) checksum, so the initial state differs before any solver runs.

The measure is the one already applied to the element scatter: give each shared lattice node
a single canonical writer. Better still, compute a node's coordinate as a pure function of
the macro corners and its lattice index, evaluated once per node rather than once per
incident macro-element, which removes the question rather than ordering it. This is in
`smesh`, not here.

### 2. Krylov reductions, in `algebra/openmp/sfem_openmp_blas.hpp`

`dot` and `norm2` use `reduction(+ : ret)`, which combines partial sums in an unspecified
order over chunks whose boundaries follow the thread count. `SFEM_DETERMINISTIC_BLAS=1` now
selects a fixed 256-chunk decomposition, independent of the thread count, summed serially
per chunk and combined in index order. Off by default, since it changes results in the last
bits.

It is **unverified in this build**: the flag never fires, because the spike links a prebuilt
SFEM whose instantiation of these templates comes from the library rather than from the
edited header. Confirming it needs SFEM itself rebuilt. The code is written and gated; the
measurement is owed.

### 3. What is already done

The two-pass scatter, on by default (`SFEM_SS_SCATTER=1`), covering the Jacobian action, the
nodal pressure gradient, the residual, the block diagonal and the block split.

### 4. Measurement discipline, independent of the above

Until the two remaining sources are closed, comparisons should pin the thread count and
report medians of repeats rather than single runs, and prefer quantities measured as rates
over iteration counts. `SFEM_ASSEMBLE_FINE=1` gives a deterministic operator for A/B work
where the memory is affordable. It would be worth adding a determinism check to the test
suite -- one checksum at one thread against the same at N -- so that a regression in any of
this is caught rather than rediscovered.

## Reproducibility, verified end to end

With the smesh coordinate patch built in, all three measures are live and the chain is
closed. Everything below is measured, not projected.

**Mesh coordinates.** N=3 on 8 threads, three runs: 16562.000023022294 every time with
`SMESH_DETERMINISTIC_COORDS=1`, against 16562.000030174851, 16562.000029459596,
16562.000029459596 with it off. The deterministic value is exactly what a single-threaded
run produced before the patch, which is what the ownership rule promised -- the lowest
element was already winning there.

**Operator.** Bit-identical at 1, 2, 4 and 8 threads: the Jacobian action checksums
0.08034669538290462 at every one, as do the residual (-0.63166661236032529) and the initial
state (206.55554994212321). Thread-count independent, not merely run-to-run stable.

**Solve.** N=3, L=8, across thread counts:

| | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| `SFEM_DETERMINISTIC_BLAS=0` | 2000 (cap) | 2000 (cap) | 1688 | 1223 |
| `SFEM_DETERMINISTIC_BLAS=1` | 1030 | 1030 | 1030 | 1030 |

Identical iteration counts and identical u_linf (2.982674e-03) at every thread count. The
~20% iteration noise that made every comparison in this document unreliable is gone.

### One thing worth noticing

The deterministic run is not merely reproducible, it is better: 1030 iterations against 1223
at best and two outright failures to converge. That is not luck. The fixed 256-chunk sum is
partially pairwise, so it is *more accurate* than the serial accumulation a single thread
performs -- which is why the deterministic single-threaded run converges where the
non-deterministic single-threaded run stagnates at its cap. Determinism here costs nothing
and buys accuracy.

It also explains a result reported earlier in this document and never satisfactorily
accounted for: the baseline that "failed to converge single-threaded and succeeded on eight".
That was never about thread count. It was a solver sitting close enough to stagnation that
the accumulated error in a long serial sum decided the outcome, and the thread count only
changed how that sum was grouped.

### The three measures

| measure | where | default |
|---------|-------|---------|
| `SMESH_DETERMINISTIC_COORDS` | smesh, `sshex8_fill_points*` | on |
| `SFEM_SS_SCATTER` | this spike, five sshex8 kernels | on |
| `SFEM_DETERMINISTIC_BLAS` | `algebra/openmp/sfem_openmp_blas.hpp` | on |

All three are on by default, each opting out with `=0`.

The reduction was switched on after measuring what it costs, which is nothing: per-iteration
time went from 8509 to 7672 microseconds at N=3 and 1880 to 1811 at N=1. The `reduction`
clause privatises and combines per thread; a flat chunk array summed once is cheaper. So it
is reproducible, more accurate and faster, and there was no trade to weigh.

The chunk count is a function of the length alone -- serial below 4096 elements, and above
that growing so a chunk stays near 8192. Depending only on the length is what keeps the
result identical across thread counts. A fixed 256 was wrong at both ends: it spawned 256
chunks over almost no work for short vectors, which is the same thread-team overhead that
made small multigrid levels slower on more cores, and for long ones it left each chunk a
serial sum whose error grew with the problem.

Verified with nothing set, N=3, L=8:

| threads | 1 | 2 | 4 | 8 |
|---------|---|---|---|---|
| iterations | 1178 | 1178 | 1178 | 1178 |
| u_linf | 2.985426e-03 | 2.985426e-03 | 2.985426e-03 | 2.985426e-03 |

`cvfem_ns_op_gate` passes.

One practical note, learned three times over in this document: the spike compiles the
*installed* SFEM headers, not the ones in this tree. Editing `algebra/` here changes nothing
until `build64` is rebuilt and installed, and the symptom is a measurement that silently
matches the old behaviour. Check a changed default against its own opt-out before believing
it took effect.

## The GMG comparison, rerun with everything deterministic

Every earlier comparison in this document carried about twenty percent of iteration-count
noise and should be read as indicative at best. With the mesh, the operator and the
reductions all deterministic, iteration counts now repeat exactly -- 526/526, 40/40,
1834/1834, 104/104 on repeated runs -- so these numbers mean what they say.

Two Newton steps, 8 threads, baseline is block-Jacobi + BiCGStab, V-cycle is assembled
Galerkin with `SFEM_GMG_KSMOOTH=16` and the default fine smoothing of 2.

| case | dofs | baseline | V-cycle | verdict |
|------|------|----------|---------|---------|
| N=1, L=8  |  10,692 |  526 its, 0.76 s |  40 its, 0.37 s | **2.1x faster** |
| N=1, L=16 |  75,140 | 1834 its, 7.51 s | 104 its, 4.96 s | **1.5x faster** |
| N=2, L=8  |  75,140 | 1061 its, 2.89 s | 230 its, 10.35 s | 3.6x slower |
| N=3, L=8  | 242,500 | 1178 its, 6.99 s | 430 its, 51.6 s | 7.4x slower |
| N=2, L=16 | 561,924 | 1369 its, 25.7 s | 2000 capped, wrong answer | fails |

Tuning recovers a good deal of that -- at N=3, L=8, three levels with the coarse solve capped
at 30 iterations and `KSMOOTH=8` gives 120 iterations in 8.5 s against 430 in 51.6 s, and at
N=2, L=16 it turns a wrong answer into 307 iterations and a correct one in 48.2 s -- but in
neither case does it overtake the baseline.

### The result is about lattice depth, not problem size

The third and second rows are the same problem size, 75,140 unknowns, decomposed differently:
one macro-element with a level-16 lattice against eight macro-elements with a level-8 one.
The V-cycle is 1.5x faster on the first and 3.6x slower on the second. Size is not the
variable; how much of the mesh is lattice rather than macro-elements is.

The reason is structural. `create_gmg_data` derefines the lattice and stops at the macro
mesh, which it never coarsens. At N=1 the coarsest level is 20 nodes and free. At N=3 it is
208 nodes whose Galerkin operator needs the dense fallback -- 43,264 blocks, 832 probe
applications -- and the coarse solve alone is 21% of the run at 26.7 ms per application,
with the two smoothed levels above it taking another 59%. The hierarchy runs out of levels
while the problem is still big.

That also explains why the baseline moves in the opposite direction: it takes 1834 iterations
on the deep lattice and 1061 on the shallow one at equal size. A deep lattice is the hard
case for a pointwise preconditioner, and it is exactly where the V-cycle pays off.

So the honest summary is narrower and better supported than any earlier one here: **the
V-cycle wins where the mesh is mostly lattice, and loses where it is mostly macro-elements**,
and it wins by more the deeper the lattice. For the semi-structured meshes this work exists
to exploit -- few macro-elements, deep lattices -- that is the favourable regime.

The next step is not more tuning. It is to keep coarsening below the macro mesh, so the
hierarchy does not terminate on a problem that is still large and dense; that is a change in
`create_gmg_data` rather than in this spike.

## Why it capped, and what fixing that revealed

The N=2, L=16 failure above was not a weak preconditioner. The coarsest level's solve was
diverging, and the cycle was faithfully prolonging the result.

`SFEM_GMG_COARSE_VERBOSE=1` on that configuration shows BiCGStab on the 81-node coarsest
operator taking its residual from 1.577 to 36066 over a hundred iterations, from 17.26 to
82130, from 0.914 to 273. It then exhausts its 200-iteration cap and returns that amplified
vector as the coarse-grid correction. The V-cycle amplified by 1e6 to 1e9 per cycle, FGMRES
could not converge against such a preconditioner, the linear solve hit its own cap, and
Newton stepped from a badly solved system to an answer three orders of magnitude wrong.
Depth confirms the localisation: cycle rates are 0.06, 0.20 and 0.07 for two, three and four
levels, and 2.8e6 at five.

The tuning that appeared to rescue it was not addressing the cause. `MAX_LEVELS=3` removed
the offending level and `COARSE_MAX_IT=30` limited how far the divergence could run.

### The fix, and what it says about everything above

A few hundred unknowns should not be handed to an iterative solver at all. The coarsest
level is already stored dense, so it is now factorised: `DenseLU`, recovered by applying the
operator to unit vectors, exact and incapable of diverging. `SFEM_GMG_DENSE_LU_BELOW`
(default 4096 unknowns) selects it.

With that in place, every case improves and the V-cycle wins everywhere:

| case | dofs | baseline | V-cycle | speedup |
|------|------|----------|---------|---------|
| N=1, L=8  |  10,692 |  526 its, 0.72 s | 36 its, 0.32 s | 2.2x |
| N=1, L=16 |  75,140 | 1834 its, 6.88 s | 80 its, 3.76 s | 1.8x |
| N=2, L=8  |  75,140 | 1061 its, 2.69 s | 55 its, 1.79 s | 1.5x |
| N=3, L=8  | 242,500 | 1178 its, 6.03 s | 71 its, 5.71 s | 1.06x |
| N=2, L=16 | 561,924 | 1369 its, 25.7 s | 84 its, 15.8 s | 1.6x |

Against the previous section, N=2 L=8 goes from 230 iterations and 10.35 s to 55 and 1.79 s,
N=3 L=8 from 430 and 51.6 s to 71 and 5.71 s, and N=2 L=16 from a wrong answer to the
fastest arm in the table.

**This retracts the conclusion of the previous section.** That section read the N=1 wins and
the N>1 losses as evidence that the method depends on lattice depth, and blamed the
hierarchy terminating at the macro mesh. That was wrong. The variable was the coarsest
level's solve, which fails harder at larger N because the coarsest operator is bigger and
worse conditioned there; the correlation with macro-element count was real and the causal
story attached to it was not.

The iteration counts now say something the noisy measurements never could: 36, 80, 55, 71
and 84 across 10,692 to 561,924 unknowns. Fifty times the problem for a bit over twice the
iterations is close to the level independence a multigrid method is supposed to deliver, and
the baseline over the same range goes from 526 to 1369.

`cvfem_ns_op_gate` passes.

### On the coarsening work that was proposed next

It was proposed on the strength of the retracted conclusion, so its justification is gone
rather than merely weakened. Coarsening below the macro mesh may still be worth doing -- at
N=3 the margin is only 1.06x, and a deeper hierarchy is the obvious way to widen it -- but it
should be argued from measurements taken with the coarse solve working, not from those above.

## Phase 0: the gate says build Phase 1 and drop Phase 2

Before building coarsening below the macro mesh, two measurements were taken to check that a
deeper hierarchy would help at all. Neither supports it.

### M1 — assembly is now the dominant cost

With the coarse solve fixed by the dense LU, the probing assembly is the largest single term
in the cycle: 32.7% of the run at N=3 L=8 (3.43 s of 10.49 s, 571 ms per call) and 67.4% at
N=1 L=8. The coarse solve it used to hide behind is now 0.2%.

### M2 — more levels is worse, not better

The macro mesh is a cube refined to level L, so the same fine discretisation is reachable at
several macro/lattice splits. At 64x16x16 (75,140 unknowns), varying only the split -- which
is exactly what a macro-mesh coarsening would do, executed by hand:

| macro / L | levels | iterations | t_solve | t_prec | total |
|-----------|--------|-----------|---------|--------|-------|
| 16x4x4 / L=4  | 3 | 51 | 1.66 | 4.88 | 6.55 |
| 8x2x2 / L=8   | 4 | 54 | 1.83 | 1.60 | **3.43** |
| 4x1x1 / L=16  | 5 | 80 | 3.83 | 1.44 | 5.27 |
| baseline      | - | 1834 | 7.20 | 0.01 | 7.20 |

Five levels is worse than three on iterations (80 against 51) and worse on total time. Adding
levels below the macro mesh would extend the hierarchy in exactly the direction that measures
worse. **The Phase 2 gate fails; the AMG should not be built for this problem.**

A measurement error is worth recording, because it briefly pointed the other way. `t_solve`
covers only the Krylov solve; `refresh_gmg`, and therefore the whole assembly, is counted in
`t_precond`. Comparing arms on `t_solve` alone credited the shallow-lattice arm with 1.54 s
while ignoring its 4.88 s of assembly. Only the total is meaningful when the arms have
different level counts.

### What the same numbers say about Phase 1

The two effects run against each other. Iterations improve as the lattice gets shallower and
the macro mesh finer (51, 54, 80), because the coarse levels are then better resolved. But
the probing assembly gets sharply worse in that direction (4.88, 1.60, 1.44 s), because the
coarsest level has more nodes and the pattern guess falls back to dense -- 425 nodes and 1700
probe applications in the best-iteration arm, against 20 nodes and 80 in the worst.

So the configuration that converges best is the one probing punishes hardest. Removing the
probing does not merely save its own 30-70%; it unlocks the split that wins on iterations.
That is the case for Phase 1, and it is stronger than the one the plan was written on.

### Correction: the Phase 2 gate was run in the wrong regime

The conclusion above -- that coarsening below the macro mesh is not worth building -- is
withdrawn. It was drawn from experiments whose coarsest level was 20 to 425 nodes, where the
exact coarse solve is free and extra levels can therefore only add cost. The gate could not
have returned anything else.

The cost that coarsening removes is the *terminal* problem's size, and a dense factorisation
is O(n^3) in time and O(n^2) in memory:

| macro nodes | dofs | LU storage | factor flops | |
|---|---|---|---|---|
| 425 | 1,700 | 22 MiB | 1.6e9 | free |
| 1,024 | 4,096 | 128 MiB | 2.3e10 | marginal (per Newton step) |
| 4,096 | 16,384 | 2.0 GiB | 1.5e12 | impossible |
| 65,536 | 262,144 | 512 GiB | 6.0e15 | impossible |

`SFEM_GMG_DENSE_LU_BELOW` is 4096 dofs, i.e. 1024 macro nodes. Beyond it the driver falls
back to the BiCGStab that was measured to diverge on this operator. So today there is a hard
ceiling at about a thousand macro elements, above which there is neither a working exact
coarse solve nor any way to make the coarse problem smaller.

Extending the M2 ladder one rung into that regime, same fine mesh of 64x16x16 throughout:

| macro / L | coarsest | assembly | t_precond | t_solve | total |
|-----------|----------|----------|-----------|---------|-------|
| 8x2x2 / L=8   | 81 nodes | 6,561 blocks, 324 applications | 1.60 | 1.83 | 3.43 |
| 16x4x4 / L=4  | 425 nodes | 180,625 blocks, 1,700 applications | 4.88 | 1.66 | 6.55 |
| **32x8x8 / L=2** | **2,673 nodes** | **7,144,929 blocks (915 MB), 10,692 applications** | **144.4** | **386.1** | **~530** |
| baseline | - | - | 0.01 | 7.20 | 7.20 |

Seventy-four times slower than the baseline on the same discretisation. Both failure modes
appear together: the probing assembly goes fully dense at 7.1 million blocks, and the coarse
solve drops to a Krylov method that cannot be trusted on this operator.

This is the regime that matters for a real macro mesh, and it needs both phases rather than
one:

- **Phase 1** removes the dense-pattern probing, which is what produced the 915 MB operator
  and the 10,692 applications per Newton step. A sparse triple product yields the exact
  pattern, which for a 2,673-node level is a normal sparse matrix rather than a dense one.
- **Phase 2 is reinstated**, but its purpose is not the one the plan gave it. It is not there
  to add levels for faster convergence -- M2 correctly showed that does not help when the
  coarse problem is already small. It is there to **bound the size of the terminal problem**,
  so the exact coarse solve stays affordable as the macro mesh grows. That is invisible below
  the factorisation knee and decisive above it.

The gate should be re-run above the knee once Phase 1 lands, since Phase 1 changes the
assembly cost that currently dominates this measurement.

## Can the coarsest level be assembled directly instead of probed?

`hessian_bsr` refuses on the semi-structured path but works on an unstructured level, so the
coarsest level could be assembled outright and the last remaining probe dropped. That trades
a Galerkin operator for a rediscretised one, and the proposal was that projecting the
velocity and pressure properly -- an L2 projection rather than the partition-of-unity average
used now -- would make the rediscretised operator good enough.

The gap narrows as the coarse mesh resolves, which is the right trend:

| coarsest | raw gap (ux) | best-fit scale | after-scale (ux) | after-scale (p) |
|----------|--------------|----------------|------------------|-----------------|
| 81 nodes   | 13.86 | -0.007 | 0.995 | 0.870 |
| 425 nodes  |  3.49 |  0.074 | 0.964 | 0.634 |
| 2673 nodes |  1.42 |  0.336 | 0.805 | 0.628 |

At 81 nodes the two operators are nearly orthogonal, which is no surprise: a rediscretised
Navier-Stokes operator on 81 nodes is a different problem, not a coarse version of the same
one. By 2673 nodes the raw gap has fallen tenfold and the correlation has risen to a third.

### The state is not what separates them

`SFEM_GMG_CONST_STATE=1` gives every level the same constant field. Averaging and an L2
projection reproduce a constant identically, so the two operators are then evaluated at
genuinely the same state and the state is eliminated as a variable:

| coarsest | after-scale ux (real state -> constant) | after-scale p (real -> constant) |
|----------|------------------------------------------|----------------------------------|
| 81 nodes   | 0.995 -> 0.955 | 0.870 -> 0.870 |
| 425 nodes  | 0.964 -> 0.602 | 0.634 -> 0.634 |
| 2673 nodes | 0.805 -> 0.735 | 0.628 -> 0.628 |

**The pressure figures do not move at all.** They are identical to four digits, which is what
they must be if the cause is the discretisation: Rhie-Chow's `Df = rc h^2 / (2 mu)`, the
pressure Laplacian and the divergence block are all state-free, and only momentum convection
depends on the state. That the numbers are bit-identical is also a check that the diagnostic
is measuring what it claims.

Velocity does improve -- 0.964 to 0.602 at 425 nodes -- so part of that gap really is the
state, and a better projection would recover it. But even with a perfect, exactly
representable state the velocity operators still differ by 60 to 90 percent after optimal
per-component rescaling.

So the answer is no, on this evidence. An L2 projection is worth having on its own merits,
since it would sharpen the coarse operator's convective coefficients, but it cannot make
direct assembly a substitute for Galerkin here: most of the disagreement, and all of the
pressure disagreement, is in the discretisation rather than in the state. The probe at the
first coarse level stays, and its cost is now 108 operator applications rather than 10,692.

## Element-wise Galerkin: assembling the coarse operator inside the macro element

The last probe survives only because the fine operator has no matrix form. It does not need
one. The fine operator is a sum of macro-element contributions and the prolongation's support
is local -- a fine node interpolates only from coarse nodes of the sub-cell containing it, and
a face node gets the same contributors from either side -- so

    R A P  =  sum_e  P_e^T A_e P_e

element by element, with no global matrix ever formed and nothing reaching outside a macro
element.

**The enabling property is verified, not assumed.** Element-wise Galerkin is exact only if A
really is a sum of element operators, and Rhie-Chow couples through a *nodal* pressure
gradient, which is not element-local -- unless it is frozen from the state rather than
recomputed from the direction. Putting a direction on one macro element's interior and
measuring the response outside it gives exactly zero (max |out| inside 4.2480e-02, outside
0.0000e+00 at L=8; likewise at L=4). The frozen gradient is what makes this work, and it is
worth knowing that changing Rhie-Chow to differentiate the pressure gradient would silently
invalidate the construction.

### Cost

Per macro element, in element-kernel evaluations:

| L (hop) | fine nodes | coarse/elem | global probe | local probe | local matrix |
|---------|-----------|-------------|--------------|-------------|--------------|
| 2  |   27 |   8 | 108 |   32 | ~1-4 |
| 4  |  125 |  27 | 176 |  108 | ~1-4 |
| 8  |  729 | 125 | 176 |  500 | ~1-4 |
| 16 | 4913 | 729 | 192 | 2916 | ~1-4 |

Two variants. *Local probing* -- applying the element kernel to each local coarse basis
function -- only beats global probing for shallow lattices, and is three times worse at L=8.
*Local matrix* -- assembling the macro element's own sparse operator once on its lattice
stencil and then doing a small triple product -- wins everywhere, and turns assembly from
108-192 global operator applications into something comparable to a single one.

The local matrix is transient, one per element or per thread: 729 blocks and 0.09 MiB at
L=2, 19,683 and 2.40 MiB at L=8, 132,651 and 16.19 MiB at L=16.

### What it would need

Only one new kernel: a macro-element-local assembly for the semi-structured CVFEM operator.
The per-micro-cell Jacobian entries already exist -- the unstructured path assembles them
into a global BSR -- so the work is scattering them into a lattice-local structure instead of
into the global matrix. Everything else is in place: the structured prolongation gives P_e
directly, and the deterministic two-pass scatter already built for the operator kernels is
exactly what accumulates the local contributions into the coarse BSR.

It also removes the last of the machinery this section has been dismantling: no probing, no
colouring, no sparsity pattern to derive or guess, and no global fine matrix.

### Element matrices all the way down, BSR only at the coarsest level

Element-locality does more than remove the probe. If the coarse operator is built inside the
macro element, it can also be *kept* there: each coarse level becomes a set of Galerkin
element matrices applied with GEMM, and only the coarsest level needs a global sparse matrix,
because that is the only level that is factorised.

This is SFEM's existing idiom rather than a new one --
`frontend/ops/sfem_SemiStructuredEMLinearElasticity.hpp` assembles the element matrix on the
fly and applies it with GEMM -- and the pieces are already in this spike:
`subpar/cvfem_sshex8_em.hpp`, the `em24`/`em32` bench columns, and
`packed_elements_matmul_sym` / `_nonsym` in `operators/packed_elements.hpp`.

| coarse lattice | nodes/elem | EM dofs | EM MiB/elem | assembled BSR, 32 elems |
|----------------|-----------|---------|-------------|-------------------------|
| level 4 | 125 | 500 | 1.91 | 8.81 |
| level 2 |  27 | 108 | 0.09 | 1.40 |
| level 1 |   8 |  32 | 0.01 | 0.27 |

The last hop is a 32x32 element matrix, which is exactly the `em32` shape the bench already
measures.

**Correction, from building it.** The table above compares a *dense* element matrix against
the BSR, and on that basis the storage runs the other way at the shallow end -- level 4 costs
about seven times the assembled BSR. The kernel that was actually written does not store a
dense element matrix. A coarse node couples only to its 3x3x3 lattice neighbourhood, so the
local operator is a 27-point stencil of `(Lc+1)^3 * 27` blocks, and the only excess over the
assembled BSR is duplication at shared macro-element faces:

| coarse lattice | dense EM vs BSR | 27-stencil EM vs BSR |
|----------------|-----------------|----------------------|
| level 2 | 1.0x | 3.38x |
| level 4 | 4.6x | 1.95x |
| level 8 | 27.0x | 1.42x |

So the storage objection to keeping intermediate levels as element matrices is much weaker
than the first estimate suggested, and it *improves* with lattice depth rather than worsening.
That reopens "BSR only at the coarsest" as a real option rather than something the numbers
argue against; what settles it is a measurement of the two applies, not the storage.

One property makes this cleaner than it first appears. The prolongation composed within a
macro element is itself a trilinear interpolation, so any level's element matrix can be
formed directly from the fine element operator as `(P_e^{L->l})^T A_e P_e^{L->l}` rather than
by chaining level-to-level products. Each coarse level is then independent of the others: no
error accumulates through repeated Galerkin products, and a level can be rebuilt without
touching its neighbours.

The resulting architecture drops nearly everything this section has been repairing:

- fine level: matrix-free, unchanged
- intermediate coarse levels: Galerkin element matrices, GEMM apply, assembled on the fly
  where storage warrants it
- coarsest level only: assembled BSR, for the dense LU that must stay exact

No probing, no colouring, no sparsity pattern derived or guessed, no SpGEMM, and no global
sparse matrix above the coarsest level -- so the unsorted-column hazard, the `mm` workspace
sizing and the host-only serial transpose all stop applying. The coarse block diagonals the
smoothers need come from summing element contributions, which the deterministic two-pass
scatter already does.

### Element-wise Galerkin, implemented

`SFEM_GMG_EGAL=1` (default) builds the level-1 coarse operator as `sum_e P_e^T A_e P_e`
straight from the fine macro-elements. That is where the probe was -- level 1 is the only
level whose operator above it is matrix-free and has no matrix form; below it the level above
IS a matrix and `rap` is already exact and cheap. It does not turn Galerkin coarsening on:
that is still `SFEM_GMG_GALERKIN=2`, off by default, so a run setting neither still gets
rediscretised coarse operators.

**What made it cheap.** Three structural facts, none of which needed new physics:

1. *The micro-cell matrix was already reachable.* Passing identity slots (`sl[k] = k`) to
   `cvfem_hex8_ns_upwind_jacobian_add_slots` writes a dense 8x8-block cell matrix into a local
   buffer -- the trick `assemble_block_diag` already used. So the entries come out directly
   and nothing is probed.
2. *A micro-cell's coarse support is exactly eight nodes.* The cell spans fine indices
   `[xi, xi+1]`, whose coarse floors differ by at most one, so per axis it reaches coarse
   indices `{ax, ax+1}` and no more -- for any ratio `q`, not only 2:1. The triple product is
   a fixed 8x8 -> 8x8 contraction rather than something growing with `L`.
3. *The weights depend only on the offset class* `(xi%q, yi%q, zi%q)`. There are `q^3` of them,
   shared by every cell and macro-element, so they are a table built once and never an array
   indexed per entry -- the principle `cvfem_ss_transfer.hpp` applies to the prolongation.

**Cost.** Per micro-cell the two contraction stages are `8*27` and `27*8` block
multiply-accumulates, so 432 against the 1024 a dense 8x8 -> 8x8 contraction would need. The
27 is the average row count of the prolongation restricted to a cell: one corner interpolates
from 1 coarse node, three from 2, three from 4, one from 8, and `1+6+12+8 = 27`. Stage 1
contracts through a precomputed transpose of that map so each output block is stored once
rather than zeroed and accumulated into, saving 1024 scalars of zeroing per cell for the same
arithmetic. Assembly runs in chunks of macro-elements sized to keep the staging buffer near
32 MiB.

**The pattern is derived, not guessed.** An entry exists exactly where two coarse nodes share
a macro-element and sit within one lattice step of each other, which is the true Galerkin
pattern. That removes the probe's worst failure mode outright: an entry outside a too-narrow
guess was not dropped but folded into the wrong slot, so a bad guess gave a wrong matrix
rather than an approximate one, and the retry loop that widened it is what produced the
7,144,929-block dense coarse operator. The derived pattern is also tighter than the probe's --
112 blocks at the coarsest level where the probe padded to 144, which `rap` independently
confirms is the true count.

**Determinism** comes for free: accumulation runs over destinations rather than sources, each
block summing its own contributions in a fixed order with chunks in fixed order too, so there
are no atomics and the matrix is the same bits on any thread count. This is the two-pass
packed idea from the apply path, applied to assembly.

### Levels chain after all, once constraints are in play

The design note above claimed a level at ratio `q` could be built straight from the fine
element matrices, with no chaining, because piecewise-linear interpolation on nested uniform
lattices composes to the direct map. That is true of the **raw** operator, and the gates
measure it at 2.2e-16 and 4.4e-16 through the last hop.

It is false for the hierarchy the driver actually builds. Its transfers zero constrained
degrees of freedom at *every* hop, so the composite is `R2 Z1 (R1 Z0 A Z0 P1) Z1 P2` -- a `Z`
at each level, not only at the fine end. Level 2 built by chaining therefore coarsens a
level-1 matrix that already carries identity rows, and those rows contribute to the product.
A level built directly from level 0 cannot see them.

This did not show up in any operator comparison, because the operators agree to 2e-16 on
unconstrained columns. It showed up in the **block diagonal**, which differed by 1.2e-2 at
level 2 and 5.7e-2 at level 3 -- and the block diagonal is what the smoother inverts. Those
are not round-off; the smoother was being handed wrong diagonals at two of the three coarse
levels.

Masking the constrained coarse columns as well is *not* the fix; it makes level 1 disagree too
(4.4e-18 -> 7.3e-3), because the probe does not mask them. The fix is to leave the chaining
alone: element-wise Galerkin replaces the probe at level 1, and `rap` continues to build the
levels below from the level above, which is the already-validated path. With that, every gate
reads machine precision at every Newton step -- the level-1 operator at 4e-16 and its block
diagonal at 1.5e-17.

**The gate that missed it** was copied from the `rap` check, which zeroes constrained rows on
both sides with the comment that "those rows are not part of what is being tested". That is
right for testing a triple product and wrong for testing a replacement construction: it makes
the comparison blind to precisely the rows where two constraint treatments differ. Comparing
the block diagonals, which no operator gate covers, is what located it.

### Gates

On a 2x1x1 macro mesh at L=8 with `SFEM_GMG_CHECK=1 SFEM_GMG_GALERKIN=2`:

| gate | what it settles | result |
|------|-----------------|--------|
| `egal identity (q=1)` | the assembly reproduces `A` itself -- cell matrix, geometry, Rhie-Chow, pattern, scatter, at once | 1.540e-16 |
| `egal galerkin (0->1)` | the coarsening reproduces `P^T A P` against the matrix-free composite | 2.199e-16 |
| `egal galerkin (0->2,3)` | direct equals chained for the raw operator | 2.210e-16, 4.444e-16 |
| `egal level 1` | the constrained operator equals the probed composite it replaces | 2.1e-16 |
| `egal diag 1` | the block diagonal the smoother inverts equals the probed one | 4.4e-18 |
| `rap level 2,3` | the levels below, unchanged | 1.8e-16, 1.8e-16 |

The identity gate is the one worth keeping. At `q = 1` the prolongation is the identity, so
`P^T A P` is `A`, and one comparison covers everything the construction rests on: that
identity slots really do yield the micro-cell matrix, that the hoisted geometry and Rhie-Chow
struct fed to the assembly are the ones the apply uses, that the derived pattern holds every
entry, and that the inverted-index accumulation lands each block where it belongs.

### The whole hierarchy element-wise: the coarse constraints already existed

The section above concluded that extending the element-wise construction below level 1 was
blocked by the constraint treatment. It was not. `create_gmg_data` derefines the `Function` at
every level (`f_prev->derefine(fs_next, true)`, via `DirichletConditions::derefine`), so each
level already carries its own constraints, and the transfers apply them as `R = Z_coarse Rhat`
and `P = Z_fine Phat`. The composite at a hop is therefore

    Z_i Rhat A_{i-1} Z_{i-1} Phat

-- mask the source level's columns with that level's own mask, contract with the plain
interpolation, patch identity rows at the target. That is the recipe the `rap` branch was
already using (`mask_block_columns`, `rap`, `patch_identity_rows`), one level up.

So the hierarchy is built by chaining, but the chaining happens *inside the macro-element*:
`galerkin_hop` coarsens a level's 27-point stencil element matrices to the next level's,
masking between hops, with no global matrix at any point. Because the mask acts on the matrix
rather than on the transfer, the interpolation stays scalar -- no per-component prolongation is
needed even though the constraints are per component. The coarse stencil stays 27-point for any
ratio: a source node and its 27-neighbour land on coarse nodes at most one step apart, since
reaching two would need their coarse floors to differ while the upper one is off-lattice, and a
differing floor forces it to be on-lattice.

With that, every level matches the construction it replaces, block diagonals included:

| level | operator | block diagonal, before -> after |
|-------|----------|--------------------------------|
| 1 | 2.1e-16 | 4.4e-18 -> 4.4e-18 |
| 2 | 1.2e-17 | **1.2e-2 -> 4.6e-18** |
| 3 | 7.1e-17 | **5.7e-2 -> 0 (exact)** |

Nothing is probed at any level, and no triple product touches a global sparse matrix.

**A latent bug this exposed.** Moving the assembly ahead of the level loop segfaulted, because
it read the operator's cached state fields -- whatever the last `apply()`, `gradient()` or
`update()` happened to leave there. It had been correct only by accident of call order, which
is the kind of dependency that is invisible while it holds. `assemble_hierarchy` now takes the
fine state explicitly and calls `update()` itself.

### BSR only at the coarsest level

`SFEM_GMG_EGAL_EM=1` keeps every level but the coarsest as element matrices, applied directly:
gather a macro-element's coarse nodes, run the 27-point stencil over its local matrix, reduce.
Nothing above the level that is factorised is assembled at all -- no pattern, no scatter index,
no sparse structure. It costs no extra construction either, since `galerkin_hop` coarsens
element matrices *to* element matrices; the element form is what the hops already produce, and
this simply stops there instead of going on to a BSR.

Two things made it cheaper to reach than expected. `make_dense_lu` densifies by *applying* the
operator, so the coarsest level never needed matrix structure for the direct solve -- it is
assembled because a matrix is the natural thing to hand a factorisation, not because anything
demands one. And with element-wise Galerkin on, `g.Amat` turned out to be written but never
read: the only consumer was the `rap` branch, which no longer runs.

Both forms pass every gate on 2x1x1 at L=8:

| level | assembled: operator / diagonal | element matrices: operator / diagonal |
|-------|-------------------------------|---------------------------------------|
| 1 | 2.1e-16 / 4.4e-18 | 3.2e-16 / 4.4e-18 |
| 2 | 1.2e-17 / 4.6e-18 | 1.8e-16 / 5.4e-18 |
| 3 (coarsest, assembled in both) | 7.1e-17 / 0 | 7.1e-17 / 0 |

Storage and block multiplies exceed the assembled form only by duplication at shared
macro-element faces -- 3.38x at a level-2 coarse lattice, 1.95x at level 4, 1.42x at level 8 --
improving as the lattice deepens rather than worsening. What is bought is contiguous 4x4 blocks
with no column indirection, and no global sparse structure above the coarsest level.

**Measured on Grace, and it loses.** 108 macro-elements at L=8 -- 60,625 nodes, **242,500
dofs** -- on 72 Grace cores, two Newton steps so all three arms do the same work:

| | probe | element-wise, assembled | element-wise, element matrices |
|-----------------------|---------|---------|---------|
| assembly, total       | 1.114 s | 0.030 s | 0.015 s |
| assembly, share       | 2.9%    | 0.1%    | 0.0%    |
| linear iterations     | 1040    | 1040    | 1040    |
| us per linear iteration | 44077 | 43121   | 63326   |
| `t_solve`             | 45.84 s | 44.85 s | 65.86 s |

Identical iteration counts across all three confirm the preconditioners are equivalent, so this
is a pure cost comparison. Keeping the levels as element matrices gives the cheapest assembly
of the three -- half the assembled form's, since it stops before building a pattern -- and then
loses all of it and more on the apply: 47% more time per linear iteration, 65.86 s against
44.85 s.

That is the duplication showing up exactly where the estimate said it would. The coarse levels
here are at ratios where the element form does 1.95x and 3.38x the block multiplies, and the
apply is memory-bound, so the contiguity and the absent column indirection do not come close to
paying for the extra traffic. It is not an implementation defect to be tuned away; it is what
storing a shared node once per incident macro-element costs.

So `SFEM_GMG_EGAL_EM` stays off. It remains worth having built: it is the form the hops
naturally produce, it is what a device port would want (no indirection, no gather), and the
comparison above is the reason to keep assembling rather than an assumption that we should.

**Verified on Grace.** The gates hold on aarch64 under the alps toolchain at the same machine
precision they reach on the development machine -- level 1 at 2.7e-16 with its block diagonal
at 4.1e-18, level 2 at 1.6e-16 and 5.2e-18, the coarsest at 1.1e-16 and exactly zero -- with
levels 1 and 2 kept as element matrices and only the coarsest assembled.

**A mistake worth recording, because it was made twice.** Patching the constrained rows of the
block diagonal by writing identity into both the row *and* the column of the 4x4 block gives a
7.3e-3 disagreement with the reference diagonal, at every level. `patch_identity_rows` replaces
the row and leaves the column alone; clearing the column as well zeroes real off-diagonal
entries of the diagonal block. The same 7.3e-3 had appeared earlier from masking constrained
coarse columns, and it was not recognised as the same error the second time. The diagonal gate
caught it both times -- which is the argument for having added it.

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

**2. Specialise the gather.** Done. Each block now gathers only the arrays it reads and
scatters only the rows it writes, and on Grace the blocks a Schur scheme wants got about a
quarter cheaper:

| | before | after |
|---|---|---|
| floor | 35.3% | 12.2% |
| `pp` (C) | 46.4% | **36.1%** |
| `pu` (B) | 53.0% | **43.2%** |
| `con` rows | 54.9% | **45.5%** |

`uu` and `mom` barely moved, 2 points, which is right: they read almost everything anyway.
Two constraints cap what C can save. The boundary term takes the state velocity whatever
is masked, and the macro geometry needs the coordinates, so C still gathers seven of the
fourteen arrays rather than the three its own arithmetic uses.

**3. Wire the semi-structured kernels into the Op and the driver.** Done. The operator
picks the path from what the space carries -- `has_semi_structured_mesh()` -- rather than
being configured, and the driver turns a mesh semi-structured with
`SFEM_ELEMENT_REFINE_LEVEL`. The same problem decomposed three ways:

| | nodes | elements | newton | lin_it | u_linf |
|---|---|---|---|---|---|
| flat, N=8 | 2673 | 2048 | 19 | 15420 | 9.706443e-10 |
| N=4, level 2 | 2673 | 512 macros | 19 | 15409 | 9.706011e-10 |
| N=2, level 4 | 2673 | 32 macros | 19 | 15440 | 9.706217e-10 |

Identical discrete problem, same Newton count, `u_linf` agreeing to six figures, solved
through 32 macro-elements instead of 2048 flat ones.

Writing this needed the residual, which the semi-structured path did not have -- it had
the Jacobian action, the block diagonal and the block split, none of which Newton can
start from. It is implemented in the same two layouts as everything else so the naive one
gates the macro-local one, and agrees at 2.6e-15.

Two limits are deliberate. The path is affine-macro only: one Jacobian per macro-element,
reused across its lattice, which is exact for a box and wrong for a curved macro-element,
and it ignores `SFEM_GEOM` for the same reason. And it refuses `hessian_bsr`, because an
assembled matrix per level is the memory a hierarchy exists to avoid; refusing beats
returning a zero matrix.

**4. Hopper.** Measured, and it says do not port the gather. Two things came out of it.

*Every device figure previously in this file was for a kernel without the Rhie-Chow term.*
None of the `cvfem_cuda_time_*` entry points takes `rc_scale`; `cvfem_cuda_residual_rc`
existed but was only ever verified, never timed, and the timing path attached neither the
coordinates nor the nodal gradient that it needs. Every host figure includes the term. The
comparison was therefore between a device kernel missing a term and host kernels that have
it -- the third comparability error of this work, after the pressure gradient and the
cross-build boundary A/B, and the same shape each time: two sides doing different work with
nothing in the harness to notice. `cvfem_cuda_time_residual_rc` now exists.

| Hopper, packed residual, n=128 | MDOF/s |
|---|---|
| without Rhie-Chow | 7872 |
| with Rhie-Chow | **5899** |

The term costs 1.33x, so it is a quarter of the device kernel -- close to its share on the
host, where hoisting its coefficients was worth 1.28x. Applying that factor to the apply
figure puts Hopper nearer **18x Grace than the 24.6x** reported before; that scaling is an
inference from the residual, since there is no timed apply with the term.

*The macro-local gather is already on the GPU, as packing, and it loses.* The packed
kernel is block-per-pack with shared-memory staging, which is the same transformation, and
the plain global layout beats it by 1.41x for the Jacobian action -- 10420 against 7297
MDOF/s -- confirmed twice in separate builds.

So a semi-structured CUDA port would buy nothing from the half that wins on CPU and
everything from the half the GPU lacks: the device takes `adj` and `det` as precomputed
inputs, so geometry is already hoisted there, but `mdot_coeff` still runs per
sub-control-surface per element. Hoisting those coefficients needs elements sharing a
Jacobian, which packs cannot give and macro-elements can. The ceiling on that is the 25%
above, and realistically less, since the Rhie-Chow term is more than its coefficients.
Worth doing only if a quarter of the device kernel is worth a port.

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

| block | Grace L=4 | L=8 | L=16 | what wants it |
|---|---|---|---|---|
| gather only (floor) | 13.0% | **12.2%** | 12.2% | -- |
| `pp` (C) | 38.0% | **36.1%** | 34.1% | pressure preconditioner, Schur |
| `pu` (B) | 44.8% | 43.2% | 40.5% | `B A^-1 B^T` |
| `con` rows | 46.4% | 45.5% | 43.6% | segregated pressure solve |
| `up` (B^T) | 66.3% | 64.6% | 65.1% | `B A^-1 B^T` |
| `uu` (A) | 87.7% | 86.8% | 87.0% | momentum solve |
| `mom` rows | 95.7% | 95.9% | 96.0% | -- |

Shares are against the full operator measured through the same block kernel in the same
run. `SSBLOCK_ALL` sets every flag, so it gathers everything and the denominator is
unaffected by the specialisation below.

Grace is stable to within a point across L=4..16. L=2 is worse across the board -- the
floor alone is 45% there -- because `(L+1)^3 / L^3` is 3.375, so a macro-element gathers
more than three nodes for every micro-element it runs.

### What the numbers say

**The blocks a Schur scheme needs are the cheap half.** C costs 46% of J on Grace and B
53%, against 89% for A_uu. A_uu is barely cheaper than the whole operator, because the
viscous and convective terms it keeps are most of the cost.

**Asking for the momentum rows is not worth it.** At 98% of J it is within noise of just
evaluating the operator, and on the M1 it is slower. Use the full apply for that.

**The floor was the gather, and specialising it was worth a quarter on the pressure
blocks.** Before, on Grace, the gather and scatter were 35% of the operator and C cost 46%,
leaving eleven points; the M1 put the same floor at 15%, because its kernels are about ten
times slower per dof so the same fixed cost is a smaller share of them. Grace was the one
to believe, and gathering only what each block reads took C to 36% and B to 43%. Note that
the floor figure is now block-dependent by construction -- a `Blocks = 0` sweep gathers
only the coordinates and the state velocity -- so 12% is the floor for a block that reads
nothing, not a bound shared by all of them.

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

## Semi-structured geometric multigrid: a running V-cycle, and why it is not yet a win

`create_gmg_data` is wired up and a V-cycle runs, preconditioning BiCGStab inside the
Newton loop (`SFEM_GMG=1` in `cvfem_hex8_ns_ssgmg`). Getting it to run at all turned on one
parameter, and the result it produces says the smoother is the wrong one.

### The wiring

`Function` owns the coarse `Function`s that `create_gmg_data` derefines but does not hand
their operators back, and every level here needs two things a linear problem would not
need: the state to linearise about, and its own block diagonal. So `CVFEMNavierStokes`
records the operator it produced in `derefine_op` and exposes it as `coarser()`, and the
driver walks that chain from the finest level. Per level it holds a state buffer, restricted
from the fine state with the averaging restriction; a matrix-free operator bound to that
buffer; and a 4x4 block-Jacobi smoother built from `hessian_block_diag`. The coarse level is
solved with BiCGStab. `build_gmg` runs once, `refresh_gmg` per Newton step -- rebuilding the
whole hierarchy per step instead made the first run appear to hang.

Three pieces of the default GMG path are deliberately not used. `create_gmg_operators`
passes `nullptr` as the state, which is fatal for a nonlinear operator.
`create_gmg_default_smoothers_and_solver` computes `sym_block_size = (block_size == 3 ? 6 :
3)`, silently yielding 3 for block size 4, and reaches for `hessian_block_diag_sym`, whose
packing assumes a symmetry a Navier-Stokes block does not have. Its CG coarse solver wants
an SPD system.

### Damping is what made it converge

Undamped, the V-cycle was not merely ineffective but actively harmful: BiCGStab sat on its
1000-iteration cap on every Newton step. Newton still crawled forward on the truncated
steps, which is what made this slow to spot -- the residual fell 2.6e-2 -> 6.4e-6 and only
the iteration counts showed anything wrong.

The cause is that block-Jacobi is being asked to do a different job than elsewhere in this
driver. As a Krylov preconditioner it is applied once and undamped is fine; as a smoother it
is a stationary iteration, and undamped on this saddle-point system it does not converge.
SFEM's own multigrid damps its block-Jacobi by `1/block_size` for exactly this reason.
Measured (N=1, L=4, first four Newton steps):

| omega | lin_it per Newton step |
|-------|------------------------|
| 1.0   | 1000, 1000 (capped)    |
| 0.8   | 1000, 785, 334, 710    |
| 0.7   | 391, 131, 122, 352     |
| 0.6   | 164, 24, 548, 119      |
| 0.5   | 31, 16, 154, 31        |
| 0.4   | 41, 23, 143, 19        |
| 0.25  | 50, 29, 376            |

`SFEM_GMG_OMEGA` defaults to 0.5. The damping applies to the smoothers only; the coarse
solve and the flat block-Jacobi preconditioner are left undamped.

### It is not level-independent, which is the result that matters

Total linear iterations over four Newton steps, V-cycle against the flat block-Jacobi
preconditioner:

| level | V-cycle | block-Jacobi |
|-------|---------|--------------|
| 2     | 48      | 123          |
| 4     | 247     | 303          |
| 8     | 2140    | 1082         |

A working V-cycle holds iteration counts roughly flat as the lattice deepens. These grow
faster than the flat preconditioner's and overtake it by L=8, where the V-cycle is *worse*
than the smoother it is built from.

That first reading -- that the smoother was at fault -- was wrong, and the reasoning behind
it was wrong in a way worth recording. It rested on smoothing steps at L=8 reducing
iterations monotonically (3113, 2140, 1402, 480 for 1, 3, 6 and 12), read as evidence that
the coarse-grid correction was sound and only the smoother was weak. But a damped smoother
is a convergent iteration by itself, so a cycle whose coarse correction contributed nothing
whatever would improve with smoothing count in exactly the same way. Counted in operator
applies rather than iterations the same numbers say the opposite: 7114, 14671, 19223, 13162
against block-Jacobi's 1082. More smoothing was buying less, not more.

### The cost bar a V-cycle has to clear

A V-cycle with three pre- and three post-smoothing steps costs roughly sixteen operator
applies; the flat preconditioner costs one. So the V-cycle has to cut iteration counts by
more than about 16x merely to break even on wall time, not the 2-3x it currently manages.
That is not out of reach -- at L=8 block-Jacobi needs 1082 iterations and an effective
V-cycle would need well under 50, comfortably past the bar -- but it does mean an
almost-working smoother is worth nothing, and the smoother has to be most of the way to
level-independent before the machinery pays for itself.

Wall-clock numbers are not quoted here as a comparison. These runs are at N=1, far below
saturation, where per-apply overhead dominates and the measured 2.1 ms for a 425-node apply
is overhead rather than work. The iteration counts and their growth with level are the
meaningful signal at this size; a wall-clock claim needs a saturated problem and will be
worth making once the cycle is fixed.


## What is actually wrong with the V-cycle

Chasing the above produced a diagnosis, one real bug fixed, and a clear statement of what
still blocks the cycle. The instruments are in the driver behind `SFEM_GMG_CHECK`.

### A control arm that never ran

`SFEM_GMG=2` runs the same damped block-Jacobi as a stationary iteration on the fine level
for the same number of sweeps a V-cycle spends smoothing, with no hierarchy under it. It
exists because iteration counts cannot otherwise distinguish a weak smoother from a broken
coarse correction.

Its first results showed V-cycle and control agreeing to the digit -- 48 against 48, 470
against 470 -- which was not a finding but a bug: the hierarchy was built under `if
(use_gmg)`, so `SFEM_GMG=2` took the `if (gmg)` branch and ran the V-cycle. The control was
unreachable. It now builds only for `SFEM_GMG == 1`.

### The bug: the state was restricted with the residual's operator

Every coarse operator is linearised about a state restricted from the level above, and that
restriction was `create_hierarchical_restriction`. The adjoint test in `check_transfers`
shows that operator is exactly the transpose of the prolongation -- ratio 1.000000 on every
level, once the probe vectors respect the constraints that both transfers impose on their
output. (Probing with unconstrained noise reports a spurious mismatch; the first version of
this test did exactly that and produced ratios of 0.41 and 1.12, which read convincingly as
a broken transfer and were nothing of the kind.)

Being the adjoint is precisely right for the residual and precisely wrong for a state. `P^T`
sums where a state transfer must average, inflating each coarse state by the number of fine
nodes feeding a coarse node -- a measured factor of about 3.8 per level. Every coarse
operator was therefore linearised about a field several times too large. Normalising by `R`
applied to the constant 1 recovers the partition-of-unity average. The effect on the cycle's
own convergence rate at L=8 was the difference between diverging and converging:

| cycle | before | after |
|-------|--------|-------|
| 1     | 5.83   | 0.185 |
| 2     | 1.17   | 0.626 |

### What still blocks it: Rhie-Chow does not survive coarsening

The cycle still turns divergent after the second cycle, settling at about 1.34 per cycle at
L=8, and the V-cycle remains the worst of the three preconditioners:

| level | V-cycle | fine smoother, no hierarchy | block-Jacobi |
|-------|---------|-----------------------------|--------------|
| 2     | 48      | 42                          | 123          |
| 4     | 470     | 84                          | 304          |
| 8     | 2407    | 574                         | 918          |

The coarse-operator consistency check applies `A_c` and `R A_f P` to the same smooth coarse
vector and compares them per component. The rediscretised coarse operator disagrees with
the Galerkin operator the transfers imply by a factor of about six, and the disagreement is
almost entirely in the pressure rows:

| level pair | ux   | uy   | uz   | p    |
|------------|------|------|------|------|
| 0->1       | 0.79 | 0.72 | 0.76 | 6.59 |
| 1->2       | 1.59 | 1.30 | 2.42 | 5.24 |
| 2->3       | 0.00 | 0.00 | 0.00 | 6.01 |

That localises it to the stabilisation. `Df = rc_scale * h^2 / (2 mu)` is the one term that
depends on the lattice spacing outright, so each level stabilises a different equation, and
rediscretisation hands the cycle a coarse pressure operator that is not a coarse version of
the fine one. Holding `Df` at the fine level's value (`SFEM_GMG_RC_DECAY=0.25`) confirms the
mechanism -- the pressure inconsistency falls from about 6 to between 0.6 and 1.2.

The awkward part is that the same change makes the cycle *worse*, taking the L=8 rates to
0.41, 1.30, 1.52. A coarse operator stabilised for the fine level's `h` is closer to the
Galerkin operator and simultaneously under-stabilised on its own mesh, where it is near
enough singular that solving it amplifies what it returns. The two requirements point in
opposite directions, which is the real obstacle: consistency with the fine operator and
stability on the coarse mesh cannot both come from rediscretising with an h-dependent
stabilisation.

Nor is it a scalar. `SFEM_GMG_CGC` scales the prolonged correction; swept over 0.125 to 8 at
L=8, every value diverges eventually -- values below 1 delay it, values above accelerate it
sharply (4 gives 4.7 per cycle, 8 gives 17). A single factor per level cannot repair a
coarse operator that differs in what it does rather than by how much.

### Ruled out

Recorded so they are not re-investigated: the transfer pair (exact adjoints, ratio
1.000000); the pressure null space (every level carries exactly one pressure pin, and
filtering the constant pressure mode out of each prolonged correction with
`SFEM_GMG_PFILTER=1` changes the rates in the fourth decimal); hierarchy depth (capping at
two levels with `SFEM_GMG_MAX_LEVELS`, so the coarse level is the well-resolved L=4 mesh,
diverges at the same 1.33); the nodal pressure-gradient cache (`SFEM_PGRAD_CACHE=0`
reproduces the rates bit for bit); and smoother damping (swept; 0.5 is best and is the
default).

### Where this leaves the preconditioner

Block-Jacobi is still the one to beat, and in work rather than iterations it is not close.
At L=8 it spends about 918 operator applies against roughly 3400 for the no-hierarchy
smoother arm and some 19000 for the V-cycle. The fine-level stationary smoother wins on
iteration count at every level and loses on work at every level.

The next step is not a better smoother -- the evidence points away from that. It is the
coarse pressure operator: either a stabilisation that coarsens consistently, or a coarse
level built as a genuine Galerkin product for the pressure block instead of rediscretised.

## Independent evaluation of the null-space treatment

`nullspace_eval.py` is a standalone study of whether our constant-pressure null space is
what limits the V-cycle, and whether the hybrid matrix-free elimination from the
self-contact rigid-body-modes work helps if applied to it. It models a stabilised
colocated Navier-Stokes system in 2D with the same constant-pressure null space and the
same `Df = rc h^2 / (2 mu)` stabilisation, small enough to solve exactly.

It is gated rather than merely run. Stage 1 requires the model to reproduce the driver's
symptom before anything else is believed; stage 1b requires the smoother to converge at
all; stage 1c requires the condensed operator to solve the problem to round-off before its
cycle rate is quoted. All three gates fired during development and each caught a real
error: a symmetric-indefinite model whose smoother diverged at every damping, a pure Stokes
model missing the convective diagonal that makes our smoother work, a truncated inter-level
transfer, and a right-hand side that double-counted `B_tilde C_lam^-1 g_tilde` by adding
both of the paper's two equivalent forms for it.

The model reproduces our failure closely. Coarse-operator consistency is about 0.5 in the
velocity rows and 5.2 in the pressure rows, against 0.7 and 6.6 in the driver.

**The gauge does not matter.** Pinning the same node on every level, pinning a
level-dependent node, and projecting the constant mode out per level are
indistinguishable, and none is far from the smoother alone:

| treatment                  | rate (n=24) | n=16 | n=32 |
|----------------------------|-------------|------|------|
| pin, shared node           | 0.948       | 0.919| 0.972|
| pin, level-dependent node  | 0.939       | 0.779| 1.143|
| projection, per level      | 0.936       | 0.993| 0.944|
| condensation, per level    | 12.2        | 0.922| 23.3 |

No treatment wins consistently across sizes, which is itself the result: the differences
are noise around a cycle that is limited by something else. The condensation is the
exception in the wrong direction -- its operator is verified correct to 1e-12, so its
divergence is a real property of the scheme here and not an implementation fault, and it
worsens with problem size. That is not a mark against the method in its own setting: it
changes the gauge, and a gauge is not what ails us. It also has to coarsen a dense global
rank-one term on top of a stabilisation that already fails to coarsen.

**The stabilisation is the lever**, and it is non-monotone:

| rc scaled per level | pressure consistency | V-cycle rate |
|---------------------|----------------------|--------------|
| 1.0 (as now)        | 5.18                 | 0.948        |
| 0.5                 | 2.21                 | 0.905        |
| 0.25                | 0.75                 | **0.719**    |
| 0.125               | 0.29                 | 9.81         |

Consistency improves monotonically all the way down while the rate has an optimum at 0.25
-- exactly the value that holds `Df` at the fine level's value -- and then diverges. This
is the tension stated earlier made quantitative: consistency with the fine operator and
stability on the coarse mesh are competing requirements, and the optimum is interior.

One discrepancy to resolve rather than explain away: in the model `rc_decay = 0.25`
improves the cycle (0.948 to 0.719), while in the driver the same setting made it worse
(rates 0.41, 1.30, 1.52 against 0.185, 0.63, 1.05). The exponent is dimension-independent,
since `Df ~ h^2` either way, so 0.25 should be right in 3D too. Candidate causes are the
driver's Reynolds regime, its hierarchy depth, or something still wrong in the driver that
the model does not carry. That is the next thing to chase, and it is a much narrower
question than the one this evaluation started with.

## The smoother was divergent, and the iteration counts were noise

Two findings that overturn parts of the account above.

### The default damping made the smoother diverge

`SFEM_GMG_CHECK=3` runs the smoother standalone as the stationary iteration it actually is
inside a cycle. Its good showing as a BiCGStab preconditioner proved nothing: a Krylov
method tolerates a preconditioner that would diverge if iterated, and inside a V-cycle it
is iterated.

At the then-default `omega = 0.5` the residual falls for about twenty sweeps, bottoms out
near 4.2e-3, and then grows; the per-sweep rate rises monotonically through 1 at around
sweep 27 and reaches 1.038 by sweep 39. An earlier reading of this same measurement stopped
at eight sweeps, saw 0.88 to 0.95, and called the smoother convergent. The rate was still
rising at the point it was cut off.

Asymptotic rates over sweeps 35-39: `omega` 0.5 gives 1.038 and rising, 0.3 gives 0.9717
and flat, 0.15 gives 0.9855, 0.05 gives 0.9915. The default is now 0.35.

This is what the earlier `SFEM_GMG_CGC=0` test was pointing at and what nothing else
explained: with the coarse-grid correction switched off entirely the cycle still diverged
(0.68, 0.80, 0.87, 0.98, 1.11, 1.21), while the smoother allegedly converged. A cycle that
diverges with no coarse correction has nothing to do with its coarse grid.

With a convergent smoother the V-cycle converges as an iteration for the first time.
Cycle rates at L=8: `omega` 0.35 gives 0.25, 0.30, 0.35, 0.57, 0.51, 0.48; 0.3 and 0.25 are
similar; 0.5 still diverges to 1.32.

So the plan's P5 is back, and this time on direct evidence rather than on the inference
that was withdrawn: the smoother is genuinely inadequate here, and damping only moves it
from divergent to barely convergent at 0.97 per sweep.

### The iteration counts in this report carry about a factor of two of noise

Four runs of one identical configuration (L=8, `omega` 0.35, V-cycle) gave 3084, 2354, 2793
and 1485 total linear iterations. Block-Jacobi under the same treatment gave 993 and 979.

The V-cycle path performs far more operator applications, each carrying OpenMP atomic
rounding non-determinism, and an outer BiCGStab that is close to stagnating amplifies the
difference. The consequence is that any single-shot comparison of V-cycle iteration counts
in this document is unreliable at better than a factor of two, which covers the
`SFEM_GMG_PSCALE` sweep, the `omega` 0.35 against 0.5 comparison, and the earlier
level-independence tables. Differences of that size were read as signal and were not.

What survives is what was measured as a rate rather than a count -- the standalone smoother
and cycle rates, which are monotone and reproducible -- and the block-Jacobi-against-V-cycle
gap, which is larger than the spread. Block-Jacobi at about 985 still beats the V-cycle at
1485 to 3084, so the cycle is still not competitive; it has merely stopped diverging.

### What the independent evaluation does and does not transfer

`nullspace_eval.py` predicted that scaling the coarse continuity rows by the measured
pressure/velocity ratio of the best-fit block scales would fix the cycle, and in the model
it does: the predicted beta is optimal at n = 16, 24 and 32 without tuning. The driver
reports the same pathology -- best-fit scales of about 0.63 on velocity and 0.12 on
pressure, a ratio of 0.196 -- but `SFEM_GMG_PSCALE` at that value does not help, before or
after the damping fix, and the differences are inside the noise quantified above.

The model's own stage 1b gate required a convergent smoother before reporting anything.
The driver had no such gate until now, which is precisely how a divergent smoother survived
several rounds of coarse-grid investigation.

## P5 answered: a saddle-point smoother is not the fix

SIMPLE is implemented (`SimpleSmoother`, `SFEM_SMOOTHER=simple`) on the 2x2 block split,
which is what that split was built for. Following the rule the previous section learned the
hard way, it was measured as a standalone smoother with no coarse space before being allowed
anywhere near a cycle. Standalone rates at L=8 over sweeps 35-39:

| smoother | omega | rate |
|----------|-------|------|
| block-Jacobi | 0.35 | 0.9669 |
| SIMPLE       | 0.35 | 0.9667 |
| SIMPLE       | 0.7  | 1.850  |
| SIMPLE       | 1.0  | 3.059  |

SIMPLE matches block-Jacobi to four digits and diverges sooner as damping is relaxed.
Neither more inner sweeps nor rescaling the Schur diagonal changes it.

The block-split gate under `SFEM_GMG_CHECK=1` explains why, and is the reason the null
result is trustworthy rather than a suspected bug. The four blocks sum to the full Jacobian
action to 1.4e-16, so the split is exact, and their norms are `uu` 0.284, `up` 2.231,
`pu` 0.552, `pp` 23.748. The pressure-pressure block -- the Rhie-Chow stabilisation --
is about eighty-five times the momentum block, and the divergence coupling `pu` that SIMPLE
uses to build its pressure correction is a two percent perturbation on it. SIMPLE's Schur
complement `S = Dpp - C Du^-1 B` is therefore `Dpp` to within a couple of percent, its
pressure update reduces to block-Jacobi's, and its velocity correction is negligible.

So this system is not coupling-limited and a saddle-point smoother has nothing to work with.
That is a different diagnosis from the one P5 was written under: the difficulty is not that
velocity and pressure are strongly coupled, it is that the stabilisation dominates the
operator outright.

It is also worth correcting an impression left by the previous section. An asymptotic
smoother rate near 0.97 is not by itself a bad smoother -- a smoother's asymptotic rate is
set by the smoothest mode, which is precisely what the coarse grid exists to remove, and
good multigrid smoothers routinely look terrible measured this way. What is fatal is a rate
above 1, which is what the old default damping produced. With that fixed the smoother is
doing its job, and the remaining weakness is in the coarse correction, where the velocity and
pressure rows still coarsen with different best-fit scales (0.63 against 0.12).

## Why the model's fix did not transfer: rediscretisation is the whole fault

`SFEM_GMG_CHECK=4` applies the two-level correction operator `P A_c^-1 R A` to a chosen
error mode and reports what fraction survives. The mode is built as `P` applied to a coarse
field, so it is exactly representable on the coarse grid and a correct correction must
remove essentially all of it. `SFEM_GMG_GALERKIN=1` swaps the rediscretised coarse operator
for `R A P`, composed matrix-free, which is far too expensive for production and is exactly
the right thing for a diagnostic: with it the surviving fraction is zero by construction if
the transfers are sound. `SFEM_GMG_CGC_SMOOTH=1` seeds two levels down instead of one, so
the mode is smooth relative to the coarse grid rather than oscillatory on it.

| coarse operator | mode | velocity | pressure |
|-----------------|------|----------|----------|
| rediscretised   | coarse-oscillatory | 5.52 | 0.786 |
| rediscretised   | coarse-smooth      | 0.312 | 0.601 |
| Galerkin `R A P`| either             | 0.000 | 0.000 |

The Galerkin correction is exact, which validates the transfers and the test at once. The
rediscretised operator amplifies a coarse-representable velocity error more than fivefold,
and removes only forty percent of a coarse-smooth pressure error. Rediscretisation is the
entire fault; nothing else in the cycle is.

Two checks close off the alternatives. The derefined coarse operator is bit-identical to one
built directly on the coarse space -- `derefine_op` is not the problem. And the disagreement
is not a scaling: removing each component's own best-fit scale still leaves 0.68, 0.52, 0.52
and 0.40 relative error in ux, uy, uz and p. That is why `SFEM_GMG_PSCALE`, `SFEM_GMG_CGC`
and `SFEM_GMG_RC_DECAY` all failed -- the entire family of scaling knobs was addressing a
component of the error that is a minority of it.

This is also the answer to why the independent evaluation's prediction did not transfer. In
the model the mismatch between the coarse and Galerkin operators really was close to a pure
per-block scaling, so scaling the coarse continuity rows by the measured ratio fixed it. In
the driver it is not, so no scaling can. The model was right about itself and about the
method; it was wrong about the driver because the two operators fail in different ways, and
only measuring the after-scale residual in both revealed that.

The V-cycle's behaviour follows exactly. Run long enough its rate climbs to 0.963, against
the smoother's own 0.967: the coarse correction helps for a few cycles, then contributes
nothing, and the residual left behind is pressure, reduced fifty times less than velocity.

### What this means for the next step

Galerkin coarsening works and rediscretisation does not, at least for the pressure block.
Composing `R A P` at solve time is what the diagnostic does and is not an option here --
it makes every coarse application cost fine-level work, which defeats the hierarchy. So the
choice is between assembling the coarse levels once per Newton step and finding a coarse
discretisation that behaves like the Galerkin operator without being it. The measurements
above are the gate either way: any candidate coarse operator should be required to bring the
surviving fraction near zero on coarse-representable modes before it is put into a cycle.

## Galerkin coarse operators, assembled once per Newton step

`SFEM_GMG_GALERKIN=2` assembles `A_c = R A P` into BSR once per Newton step and applies it
as a sparse matrix, so no coarse level reaches back up to a finer one during the solve.
(`=1` keeps the matrix-free composition, which is the diagnostic, not a solver: it puts
fine-level work under every coarse application.)

Assembly does two jobs. It removes the fine-level dependency, and it supplies the coarse
smoother with the diagonal of the matrix it actually smooths -- the matrix-free composite
cannot, and using the rediscretised diagonal instead mismatches the Galerkin operator by the
per-block scale factors (about 1.6 in velocity, 8 in pressure), which alone made the coarse
levels diverge.

The entries are recovered by probing under a distance-2 colouring of the coarse node graph,
so no row ever sees two neighbours of one colour and a whole set of blocks falls out per
application. That is colours x 4 applications instead of one per coarse degree of freedom:
41 colours and 164 applications for 425 nodes, against 1700 for column-by-column.

The pattern is self-correcting, and needs to be. Probing does not drop a non-zero that lies
outside the pattern -- it folds it into the wrong entry, so too narrow a pattern yields a
wrong matrix rather than an approximate one. The coarse mesh graph is right while the mesh
is fine enough that `R A P` does not reach past it, and is wrong on the coarsest levels,
where a handful of nodes are all within reach of each other. The gate caught exactly that:
levels 1 and 2 assembled to 2e-16 while the 20-node coarsest level came out at 3.8e-1. It
now widens to the squared adjacency and then to a dense pattern, and all levels assemble
exactly:

```
gate 2.1236e-16  OK   425 nodes, 8281 blocks, 41 colours, 164 applications
gate 2.7612e-16  OK    81 nodes, 1225 blocks, 31 colours, 124 applications
gate 1.5981e-16  OK    20 nodes,  400 blocks, 20 colours,  80 applications  (dense pattern)
```

### It works two-level and fails multi-level, for a specific reason

Cycle rates at L=8, first three and last three of twelve:

| hierarchy | coarse handling | rates |
|-----------|-----------------|-------|
| rediscretised, 2 levels | solved | 0.096, 0.546, 0.850 ... 0.964 |
| Galerkin, 2 levels      | solved | 0.021, 0.207, 0.238 ... 0.861 |
| Galerkin, 4 levels      | smoothed | 0.433, 5.319, 5.659 ... 5.671 |

Two-level Galerkin is a clear improvement and behaves as the correction-operator measurement
predicted. Four-level Galerkin diverges, and not for want of damping: omega 0.35, 0.2, 0.1
and 0.05 give 5.67, 5.23, 3.09 and 1.32, improving steadily and never reaching 1.

The difference between the two rows is not the number of levels but what happens on the
intermediate ones: solved in the first case, smoothed in the second. The Galerkin operator is
a much better approximation of the fine operator and a much worse candidate for block-Jacobi
smoothing -- it is denser, and its diagonal is not dominant in the way the rediscretised
operator's is. That is the standard trade between the two coarsenings, and it is now the
binding constraint rather than a suspicion.

### Where that leaves it

The two coarse operators fail in opposite directions. Rediscretisation is smoothable and
approximates badly enough that its correction is worthless; Galerkin approximates well and
cannot be smoothed by the smoother available. Two-level Galerkin with a solved coarse level
sidesteps the conflict and is the best cycle measured so far, at 0.861 against 0.964.

The next thing to try is therefore not another coarse operator but a stronger coarse-level
solver: a few Krylov iterations per level in place of the stationary smoother, which
tolerates an operator that block-Jacobi cannot smooth. That carries a consequence worth
stating before it is measured -- a Krylov smoother makes the preconditioner vary between
applications, which BiCGStab does not admit, so the outer solver would have to become
flexible (FGMRES) at the same time.

## Krylov smoothing and a flexible outer solver: the V-cycle finally works

Two changes that had to land together. `SFEM_GMG_KSMOOTH=n` replaces the stationary smoother
with n BiCGStab iterations per level, which does not need the diagonal dominance the
Galerkin operators lack. That makes the cycle vary between applications, and BiCGStab
assumes its preconditioner does not -- it does not fail loudly when that is violated, it
stagnates -- so `cvfem_fgmres.hpp` adds flexible GMRES, selected automatically whenever the
smoother is Krylov. SFEM had no GMRES of any kind.

FGMRES was gated before use: with a fixed block-Jacobi preconditioner it reaches the same
solution as BiCGStab (u_linf 1.6569e-03 against 1.6558e-03). It needs more iterations there,
which is expected -- restarted GMRES discards information at each restart and BiCGStab
performs two operator applications per iteration -- and is beside the point, since it exists
for the case BiCGStab cannot handle at all.

### Total linear iterations over four Newton steps

| refine level | dofs   | block-Jacobi | Galerkin + Krylov smoothing + FGMRES |
|--------------|--------|--------------|--------------------------------------|
| 2            | 324    | 123          | 20                                   |
| 4            | 1700   | 305          | 29                                   |
| 8            | 10692  | 959          | 59   (ksmooth 8)                     |
| 16           | 75140  | 2954         | 82   (ksmooth 16)                    |

Block-Jacobi grows by about a factor of three per refinement; this grows by about half that.
At the largest size measured it is a factor of thirty-six fewer iterations, and unlike the
rediscretised V-cycle it is stable run to run -- 89 and 102 on repeats, against 3084 and
1485 for the arm that was being read as signal earlier.

This is the first configuration in which the V-cycle does what it was built for.

### What it costs, and what is not yet shown

Wall time, L=16, two repeats: block-Jacobi 9.8 and 13.2 seconds, this 25.5 and 29.3. Thirty-six
times fewer iterations and still about twice the time, because each iteration now carries a
cycle whose levels each run sixteen preconditioned BiCGStab iterations, plus the per-Newton
assembly. Smoothing strength is near optimal at that setting: at L=16, ksmooth 10, 12, 16 and
24 give 77.1, 72.9, 23.5 and 27.1 seconds.

Two honest limits. First, smoothing strength has to grow with depth -- eight iterations
suffice at four levels and give 1194 iterations at five, where sixteen give 99. That the
cycle needs more smoothing as it deepens says the smoother is still the weak component, and
it eats into the iteration gain because the cost per cycle rises with it. Second, the
crossover in wall time was not demonstrated: the iteration counts diverge fast enough that
one should exist, but L=32 exceeded the time available here, so that remains a projection
rather than a measurement, and projections of exactly this kind have been wrong twice
already in this document.

## State of the code, and where the solver is matrix-free

### The shape of the method

The solver is matrix-free where it is large and matrix-based where it is small, and that
split is not a compromise -- each half was forced by a measurement.

**Matrix-free, and staying that way: everything on the fine level.** The CVFEM
Navier-Stokes operator over the semi-structured `sshex8` lattice is never assembled. The
residual, the Jacobian action, the 4x4 block diagonal, and the 2x2 (velocity, pressure)
block split are all element sweeps over macro-elements. The grid transfers are matrix-free
lattice operations from `smesh`. The fine-level smoother's block-Jacobi is built from
`hessian_block_diag`, which is `n_nodes x 16` values -- O(n) storage, not a matrix. This is
the reason the semi-structured hierarchy exists and none of it changed.

**Matrix-based, once per Newton step: the coarse levels.** Each coarse operator is an
assembled BSR matrix formed by Galerkin coarsening, `A_c = R A P`, with entries recovered by
probing under a distance-2 colouring. During the solve the coarse levels apply a sparse
matrix and never touch a finer level.

### Why the boundary sits there

Three measurements put it there, in order.

The rediscretised coarse operator -- the matrix-free choice, and the one the hierarchy was
built around -- does not work. Applying the two-level correction operator to a mode that is
exactly representable on the coarse grid leaves 5.52 of a velocity mode (it amplifies the
error) and 0.79 of a pressure mode, where the Galerkin operator leaves 0.000. The
disagreement is not a scaling: removing each component's own best-fit scale still leaves
0.68, 0.52, 0.52 and 0.40 in ux, uy, uz and p, which is why every scaling knob tried
(`PSCALE`, `CGC`, `RC_DECAY`) did nothing.

Galerkin cannot be applied matrix-free. Composing `R A P` at solve time works and is kept
as `SFEM_GMG_GALERKIN=1`, but it puts a fine-level application under every coarse
application, so cost stops falling geometrically with depth. That is the one thing a
hierarchy must not do.

Assembling also fixes a second problem that has nothing to do with cost. A coarse smoother
needs the diagonal of the operator it smooths; a matrix-free composite cannot supply one,
and substituting the rediscretised diagonal mismatches the Galerkin operator by the
per-block scale factors -- about 1.6 in velocity, 8 in pressure -- which by itself made the
coarse levels diverge. An assembled matrix hands over its own diagonal.

### What being matrix-based actually costs

The fine matrix is still never formed, and that is the whole point: the assembled hierarchy
is the coarse levels only.

| | blocks | memory |
|---|--------|--------|
| assembled coarse hierarchy (L=16, N=1) | 70531 | 8.6 MiB |
| the fine BSR, which is never formed | 507195 | 61.9 MiB |

The hierarchy costs about 14% of what assembling the fine level would, because each level in
3D is roughly eight times smaller than the one above it and the sum is dominated by the
first coarse level rather than the fine one.

Assembly costs 548 probe applications per Newton step at that size, of which only the 180 at
the finest transfer involve a fine-level operator application; the rest are sparse
applications on already-assembled coarse levels. Probing is what makes this affordable at
all -- a distance-2 colouring means one application reveals a whole set of blocks, so the
count scales with the stencil rather than with the number of coarse unknowns (41 colours and
164 applications for 425 nodes, against 425 column by column).

### The rest of the algorithmic configuration

The coarse levels are smoothed with BiCGStab rather than a stationary iteration, because the
Galerkin operators are denser and lack the diagonal dominance block-Jacobi needs -- under
block-Jacobi they diverge at every damping. That makes the cycle vary between applications,
so the outer solver is FGMRES rather than BiCGStab; the two changes are one change, and the
driver selects FGMRES automatically whenever the smoother is Krylov.

Recommended configuration as measured:

```
SFEM_GMG=1  SFEM_GMG_GALERKIN=2  SFEM_GMG_KSMOOTH=16  SFEM_GMG_OMEGA=0.35
```

### What is settled and what is not

Settled: the V-cycle reduces iterations by a factor of twenty-three to thirty-six against
block-Jacobi and, unlike every earlier configuration, does so reproducibly. On Grace at
L=16, 60 iterations against 1408, reaching the same solution.

Not settled: it is not yet faster. Same Grace run, 7.59 seconds against 4.23. Thirty-six
times fewer iterations and 1.8 times the wall clock, because each iteration carries sixteen
preconditioned BiCGStab iterations per level plus the per-Newton assembly. The gap is
narrower on Grace than on the M1 (1.8 against 2.4), which is the direction one would expect
from sparse coarse work vectorising better there, but a single pair of runs is not evidence
of a trend.

Also unsettled, and the reason the crossover has not been demonstrated: refine level 32 does
not exist. It aborts in `smesh` with "Invalid element setup for proteus hex: 32", so the
larger problem has to come from more macro-elements at a valid level rather than a deeper
lattice. That sweep (N = 2 and 3 at L = 16) is running.

### Instrumentation

All of it is behind `SFEM_GMG_CHECK`, and all of it exists because something got past its
absence.

- `=1` transfers and their adjointness, a constraint census per level, the block-split sum
  against the full operator, and coarse-operator consistency reported three ways: raw, the
  per-component best-fit scale, and the residual left after removing that scale.
- `=2` the cycle's own convergence rate standalone, plus the stalled residual split by
  component.
- `=3` the smoother alone, with no coarse space. This is the first thing to run when a cycle
  misbehaves, and running it long enough to see the asymptote is part of the check: a rate
  that is still moving when the measurement stops has not been measured.
- `=4` the two-level correction operator applied to a prescribed mode, rediscretised against
  Galerkin, on coarse-oscillatory or (with `SFEM_GMG_CGC_SMOOTH=1`) coarse-smooth modes.
- The Galerkin assembly gates itself against the composite it was probed from and widens its
  sparsity pattern until it agrees, because probing folds a non-zero lying outside the
  pattern into the wrong entry rather than dropping it.

## Where the time actually goes

Cost had been inferred from operator-application counts, which is a model rather than a
measurement: it assumes every application costs the same and ignores the sparse coarse work,
the transfers and the assembly. The driver now times each phase directly and prints a
breakdown (SFEM's own tracing needs `SMESH_ENABLE_TRACE` compiled into smesh, which the
installed one lacks). Note that `precond_total` contains the `smooth[*]` rows, so the
"accounted" total double counts it.

### The coarse levels were paying for threads they could not use

The first breakdown showed the coarse smoothers costing 22.5, 21.3 and 21.2 ms per call on
levels of 2673, 425 and 81 nodes -- flat, where work should fall roughly eightfold per
level. It is not arithmetic: 324 unknowns cannot take 21 ms. It is the cost of starting a
thread team for each vector operation on a level with nothing to distribute.

Per smoother application on the 81-node level, and the whole solve:

| threads | smooth[L3] | total wall |
|---------|------------|------------|
| 1       | 0.156 ms   | 14.87 s    |
| 4       | 4.64 ms    | 4.85 s     |
| 8       | 14.20 ms   | 13.00 s    |

Ninety times slower for having eight cores instead of one, and the whole solve is fastest at
four threads and slower at eight. Each level now runs with a thread count matched to its own
size rather than the machine's (`SFEM_GMG_DOFS_PER_THREAD`, default 20000). After that the
coarse levels decay as they should -- 5752, 778 and 161 us per call across the three -- the
thread-count penalty is gone, and L=16 N=1 goes from 23.5 s to 18.3 s.

This also disposes of the large-case results reported above. Those Grace runs used 72
threads, where this penalty is far worse than at eight, so the 12x to 80x slowdowns at N=2
and N=3 are not a property of the method and those numbers should not be read as one. They
need re-running.

### What remains is fine-level smoothing, and it is the whole story

With the coarse levels fixed, the breakdown at L=16, N=1, eight threads is:

| phase | seconds | share |
|-------|---------|-------|
| smooth[L0] (fine) | 10.36 | 42% |
| galerkin_assembly | 1.47 | 6% |
| all coarse levels together | 0.78 | 3% |
| transfers, coarse solve, everything else | <0.4 | <2% |

A fine operator application costs 2.43 ms. The baseline spends 959 iterations x 2 = 1918 of
them. The cycle spends 86 iterations x 64 -- two smoothing applications per cycle, sixteen
BiCGStab iterations each, two applications per iteration -- which is 5504.

That is the arithmetic of the whole problem, and it is not about the hierarchy at all. To
break even the cycle may spend at most about 22 fine applications per cycle and it spends
64; equivalently, iteration count would have to fall by 32x and it falls by 11x. The
assembly, the transfers and every coarse level together account for under 10% and are not
where the decision lies.

The lever is therefore the fine-level smoother: it has to become roughly three times cheaper
per cycle without giving back the iteration count. Sixteen BiCGStab iterations there is
strong smoothing bought at two operator applications each; the alternatives worth measuring
are fewer Krylov iterations, a stationary sweep at one application each, or a Chebyshev
smoother, which would need an eigenvalue estimate but costs one application per sweep.

### Correction: the breakdown percentages above were normalised wrongly

`precond_total` is a container -- it wraps the whole V-cycle, so the smoother, operator,
transfer and coarse-solve rows sit inside it. Summing every row double counts, and shares
taken against that sum understate everything. The table in the previous section put
fine-level smoothing at 42% on that basis, and there appeared to be half the runtime
missing. There is not. Against wall time, with containers excluded from the denominator,
top-level phases account for 99.6% of the run:

| phase | seconds | share of wall |
|-------|---------|---------------|
| the V-cycle (`precond_total`) | 12.782 | 87.7% |
| `galerkin_assembly` | 1.609 | 11.0% |
| outer Krylov operator applications | 0.115 | 0.8% |
| Newton residual, block diagonals | 0.012 | 0.1% |

and inside the V-cycle:

| phase | seconds | share of wall |
|-------|---------|---------------|
| `smooth[L0]`, the fine level | 11.495 | 78.9% |
| `smooth[L1]` | 0.732 | 5.0% |
| transfers | 0.172 | 1.2% |
| `op[L0]` | 0.164 | 1.1% |
| `smooth[L2]`, `smooth[L3]`, coarse solve | 0.124 | 0.9% |

So fine-level smoothing is 79% of the solve, not 42%, and the conclusion drawn from the
wrong normalisation is strengthened rather than changed: the fine smoother is the only thing
worth optimising, the assembly is a real but secondary 11%, and everything below the fine
level together is under 8%. The reporter now excludes containers from its denominator and
labels them, so the table cannot be read this way again.

## Why it was slower, and the configuration that is not

The V-cycle's fine-level smoother was itself BiCGStab preconditioned by block-Jacobi -- the
same solver the whole cycle is competing against. So the cycle was running the baseline
solver as a subroutine, sixteen iterations at a time, twice per cycle.

Counted in fine operator applications per outer iteration:

| | applications per outer iteration |
|---|---|
| baseline BiCGStab + block-Jacobi | 2 |
| V-cycle with `ksmooth` 16 on the fine level | 64 = (pre + post) x 16 iterations x 2 |

Thirty-two times the work per iteration, against an eleven-fold reduction in iterations. The
cycle needed the iteration count to fall by 32x to break even and it fell by 11x, which is
the entire explanation for a method that was measurably better per iteration and measurably
worse per second. Nothing about the hierarchy was involved.

The fix is to stop smoothing the fine level like a solver. Coarse levels keep sixteen
BiCGStab iterations, because the assembled Galerkin operators genuinely need them and are
cheap; the fine level takes two. That is `SFEM_GMG_KSMOOTH_FINE`, now defaulting to 2.

Three repeats each, L=16, N=1, eight threads, two Newton steps:

| arm | t_solve (s) | iterations |
|-----|-------------|------------|
| baseline BiCGStab + block-Jacobi | 6.98, 7.07, 4.64 | 1625, 1625, 1064 |
| V-cycle, fine 1 | 5.25, 7.09, 6.87 | 133, 179, 174 |
| **V-cycle, fine 2** | **3.25, 4.84, 2.99** | 65, 97, 60 |
| V-cycle, fine 4 | 8.09, 11.37, 6.01 | 112, 157, 83 |
| V-cycle, fine 16 | 27.12 | 66 |

All reach the same solution (4.29148e-03 against the baseline's 4.29289e-03). At two
fine iterations the cycle is about twice as fast as the baseline on median, having been four
times slower at sixteen. This is the first configuration in this document that is faster
rather than merely fewer-iterations, and it took the phase measurement to find, because the
whole cost was in one line of the breakdown.

The two fixes compound: clamping the coarse levels' thread count made coarse smoothing cheap
enough that spending on it is affordable, which is what makes a weak fine smoother viable.
Measured before the clamp, cheap fine smoothing looked worse, and that reading is what
delayed this by a round.

## The 2x speedup does not survive to a large problem

The configuration that beat the baseline at L=16, N=1 on a laptop fails at N=3 on Grace.
Measured at 1,853,572 unknowns, 72 threads, two Newton steps:

| arm | iterations | t_solve | u_linf |
|-----|-----------|---------|--------|
| baseline BiCGStab + block-Jacobi | 1702 | 12.42 s | 1.30e-03 |
| baseline, repeat | 1888 | 13.78 s | 5.80e-03 |
| V-cycle, fine smoothing 2 | 2000 (cap) | 382.6 s | 1.99e-01 |
| V-cycle, fine smoothing 16 | 518 | 201.7 s | 1.49e-03 |

Weak fine smoothing does not merely lose here, it fails: the linear solve exhausts its
iteration cap and returns an answer two orders of magnitude off. Strong fine smoothing
converges to the right answer and takes fifteen times as long as the baseline. There is no
setting at this size that is both correct and competitive.

The pattern across every measurement in this document is now consistent: the smoothing
strength this cycle needs grows with the problem, and the cost of that smoothing is what
sinks it. It needed 8 iterations per level at four levels and 16 at five; it works at 2 on
the fine level at 75k unknowns and fails at 2 at 1.85M. Each time the requirement rises the
per-cycle cost rises with it, and the iteration count does not fall fast enough to pay.

So the honest state is that the V-cycle is faster than block-Jacobi on one problem size on
one machine, and slower or wrong everywhere else that has been measured. The earlier
sections reporting the 2x win should be read with this one attached.

### Two implementation notes attached to the same runs

The thread clamp is off by default. Resizing the OpenMP team per operator application costs
more than it saves once the team is large: at 72 threads it made the coarse smoothers slower
than leaving them alone (33.3 ms against 14.0 ms per call on level 1), having made them
faster at eight threads. The underlying problem -- coarse levels cannot use 72 threads -- is
real and unsolved; the clamp moved the cost rather than removing it.

An attempt to fix that by giving small levels a hand-written serial apply was reverted. It
was slower (30.5 s against 3.0 s at L=16, N=1) and, more seriously, it changed the iteration
count from 60 to 220 consistently, which is the signature of a wrong operator rather than a
slow one -- most likely a block-layout mismatch, since the assembly gate only ever validated
the values against `h_bsr_spmv` and not against a second reader of the same array. Any
future serial path needs its own gate against the parallel one before it is trusted.

## There is no threading bug, but threading has been flattering the baseline

Running one thread against many, on a small case, settles several things at once.

**No race.** Every deterministic check is bit-identical at 1, 2, 4 and 8 threads: the
Galerkin assembly gates (2.3509e-16, 1.7047e-16, 1.3501e-16), the block-split norms and
their sum against the full operator, and the transfer adjointness. Where both solvers
converge they agree to seven digits (u_linf 7.167056e-03 against 7.167062e-03). The
operators, transfers and assembly are thread-independent.

**Single-threaded runs are exactly reproducible and multithreaded ones are not.** At L=16,
N=1 the V-cycle gives 102, 102, 102 iterations on one thread and 70, 56, 75 on eight; the
baseline gives 2000, 2000 and 1125, 1625. That is reduction order in the Krylov dot
products, not a defect, but it is worth knowing that every multithreaded iteration count in
this document carries roughly twenty percent of noise.

**And the noise helps the baseline, which invalidates the comparisons above.** On one thread
at L=16, N=1:

| arm | iterations | t_solve | u_linf | p_linf |
|-----|-----------|---------|--------|--------|
| baseline BiCGStab + block-Jacobi | 2000 (cap) | 18.98 s | 6.275e-03 | 1.605e-02 |
| V-cycle | 102 | 7.35 s | 4.291e-03 | 2.257e-03 |

The baseline does not converge single-threaded. It stagnates and exhausts its cap, and the
answer it returns is wrong -- 4.29e-03 is the value every converged run in this document
produces, and its pressure error is seven times worse. On eight threads the same solver
converges in 1125 to 1625 iterations. Rounding noise from threaded reductions is perturbing
a stagnating BiCGStab enough to break it out, which is a known behaviour and is pure luck.

So the baseline this method has been measured against was being helped by an accident of
parallel reduction order, and every multithreaded comparison reported above is a V-cycle
against a baseline that is quietly getting a free restart. Single-threaded, where both
solvers are deterministic and the comparison is honest, the V-cycle takes 102 iterations and
7.35 seconds and is correct, while the baseline takes 2000, nineteen seconds, and is wrong.

**What remains true is the parallel efficiency gap.** From one to eight threads, t_solve goes
19.08, 6.50, 6.79, 7.08 for the baseline and 7.44, 8.41, 3.08, 3.42 for the V-cycle. Neither
scales past two to four threads on this machine -- the problem is memory bound -- and much of
the baseline's apparent gain is its iteration count falling rather than its work speeding up.
The V-cycle scales worse in the sense that matters, because its coarse levels cannot use the
threads at all, and at 72 threads on Grace that is the difference between the two results.

This does not rescue the Grace numbers, and the large-case conclusion stands: at 1.85M
unknowns the V-cycle is still far slower there. But it does mean the laptop comparisons were
measuring the wrong thing, and that the correct single-threaded comparison at 75k unknowns
favours the V-cycle by more than the earlier multithreaded one suggested.

## With an assembled fine operator: the baseline's stagnation was an artefact

`SFEM_ASSEMBLE_FINE=1` replaces the matrix-free fine operator with a BSR one, probed by the
same coloured probing the Galerkin levels use (null transfers assemble A rather than R A P).
The motivation is determinism: a matrix-free apply accumulates through atomics, so its
summation order follows the thread schedule, while a BSR apply accumulates each row in one
thread.

It does what it should, and it corrects the previous section.

**The operator becomes thread-independent.** The assembly gate is identical at 1 and 8
threads (1.7650e-16, 2.3080e-16), and at L=8 the V-cycle takes exactly 36 iterations at both
thread counts where matrix-free gave 36 and 37.

**The solver does not.** At L=16 the assembled baseline gives 1862 and 2000 iterations on
two runs at 8 threads. Removing the operator's non-determinism leaves the Krylov method's
own: the dot products are OpenMP reductions and their order still follows the schedule. A
deterministic operator is necessary for a reproducible parallel solve and not sufficient.

**And the baseline's single-threaded stagnation was specific to the matrix-free operator.**
The previous section reported that the baseline fails to converge on one thread -- 2000
iterations, capped, u_linf 6.28e-03 against the correct 4.29e-03 -- and concluded that the
V-cycle wins by 2.6x once the comparison is made deterministically. With the assembled
operator the baseline converges on one thread in 1873 iterations to u_linf 4.292937e-03. The
stagnation was an accident of the matrix-free operator's rounding, not a property of
BiCGStab on this problem, and the conclusion drawn from it is withdrawn.

The honest comparison at L=16, N=1, with a deterministic operator on both sides:

| arm | threads | iterations | t_solve | u_linf |
|-----|---------|-----------|---------|--------|
| baseline | 1 | 1873 | 5.05 s | 4.292937e-03 |
| V-cycle | 1 | 112 | 7.78 s | 4.291494e-03 |
| baseline | 8 | 1210 | 3.48 s | 4.223545e-03 |
| V-cycle | 8 | 86 | 4.55 s | 4.291483e-03 |

The V-cycle uses seventeen times fewer iterations and is about 1.4x slower, at both thread
counts, and the ranking no longer depends on how many threads are used or on which operator
the baseline happens to get. That is the first comparison in this document that is stable
under both, and it says the V-cycle is not yet competitive at this size -- by a much smaller
margin than the multithreaded matrix-free numbers suggested, and in the opposite direction
from the single-threaded ones.

Assembling the fine operator costs 1.85 s single-threaded and 0.90 s on eight, which is
already counted in the timings above.

## The packed layout, brought to the semi-structured path

The flat HEX8 path won on CPU with a packed mesh: each pack writes its exclusively owned
nodes straight out, stages the shared ones, and a second pass gathers each shared node's
contributions in a fixed order. The semi-structured path never had it. `initialize()`
returns early for a semi-structured mesh, before the packing block, so those kernels
accumulated locally within a macro-element and then scattered every node with `atomic_add`.

That is where the irreproducibility came from. Atomic ordering follows thread timing and
floating-point addition is not associative, so with 27 macro-elements on 8 threads the same
operator application gave 0.080346978588455187 and 0.080346695382904176, while one thread
gave the same value every time. The earlier check that appeared to show determinism was run
at N=1, where there is a single macro-element and the atomics never contend -- a test that
could not have failed.

`SFEM_SS_SCATTER=1` (default) applies the packed layout's structure. It is simpler here
because the split is geometric: every lattice node strictly inside a macro-element belongs to
it alone, and only the skin is shared -- (L+1)^3 - (L-1)^3 nodes, 37% at L=4 falling to 12%
at L=16. Interior nodes are written directly, skin nodes are staged, and a second pass sums
each shared node's contributions in element order. It covers the Jacobian action and the
nodal pressure gradient, which had its own set of atomics and fed the apply, so fixing only
the first left the operator non-reproducible.

Measured at N=3, L=4, 8 threads, with the pressure-gradient cache off so both kernels are
exercised:

| | repeat-diff within a run | two runs, same thread count |
|---|---|---|
| atomic scatter | 5.551e-17 | 0.080346978588455187, 0.080346695382904176 |
| two-pass scatter | 0.000e+00 | 0.080346967363645994, 0.080346967363645994 |

The operator is now bit-reproducible run to run at a fixed thread count, and it is faster:
at N=3, L=8 the solve takes 8.79 to 9.73 s against 9.80 to 11.26 s, about 11%.

Two things it does not do. Results still differ between one thread and eight, but that
difference is in the initial state (206.55554994212025 against 206.55554942713621), which is
a setup-phase reduction and not these kernels. And the full solve is still not reproducible
run to run -- 1284, 1228, 1355 iterations -- because the Krylov method's dot products are
OpenMP reductions whose order still follows the schedule. That is the same conclusion the
assembled-operator experiment reached from the other direction: a deterministic operator is
necessary for a reproducible parallel solve and not sufficient. Closing it needs a
fixed-order reduction in the BLAS layer.

Still atomic, and worth the same treatment: the residual, the block diagonal, and the
2x2 block-split kernels, none of which are on the path measured above.

### The remaining kernels

The residual, the block diagonal and the 2x2 block-split applies now use the same two-pass
scatter. The helpers are templated on the number of values per node, since the block diagonal
carries sixteen rather than four, and the tables are shared across all of them.

One thing changed rather than being preserved. The block-split scatter previously wrote only
the components its block selection touches -- "a continuity-row block touches one component
of four, and the atomics are the expensive half of the scatter". That saving existed because
atomics were expensive; with a plain write it is worth nothing, and the untouched components
are zero in the element buffer anyway, so the two-pass path moves all four.

Determinism, N=3 L=4 on 8 threads, repeat-diff within a run:

| kernel | atomic | two-pass |
|--------|--------|----------|
| residual | 8.674e-19 | 0.000e+00 |
| block diagonal | 5.551e-17 | 0.000e+00 |
| block split | 1.735e-18 | 0.000e+00 |

And they are faster, in line with the Jacobian action's 11%: the residual goes from 3918 to
3528 us per call and the block diagonal from 7524 to 6952, about 10% and 8%.

Correctness is unchanged -- `cvfem_ns_op_gate` passes with the scatter on and off, at the
same tolerances (grad 1.5e-15, bsr 1.2e-16, apply 4.3e-16, blockdiag 5.9e-16, diag 2.4e-17,
asm_vs_mf 1.7e-14). `cvfem_sshex8_bench` reports three disagreeing configurations both with
and without these changes, and with the kernel edits stashed, so that failure is pre-existing
and not caused here. It is worth chasing separately.

What is left is not in these kernels. With the scatter on, the residual output tracks the
input state exactly: two runs sharing a state checksum of 206.55554942713621 both give
-0.6316666704417313, and the run that produced 206.55554979906918 is the only one that
differs. The state itself still varies between runs at 8 threads and is stable at one, so the
remaining non-determinism is in the setup that builds the initial field, not in the operator.
That and the Krylov reductions are what stand between this and a reproducible parallel solve.

## Measures for the remaining non-determinism

Four candidate sources were checked, not assumed. Two are clean, one is fixed, and two need
work outside this spike.

| source | status | evidence |
|--------|--------|----------|
| element scatter (operator) | **fixed** | repeat-diff 0.000e+00 across all five kernels |
| grid transfers | clean | R and P checksums bit-identical over three runs |
| mesh coordinates | **broken** | x varies at 8 threads, stable at 1; y and z exact everywhere |
| Krylov reductions | broken by construction | `#pragma omp parallel for reduction(+ : ret)` |

### 1. Mesh coordinates, in `smesh::to_semistructured`

The x coordinate checksum varies between runs at 8 threads (16562.000025525689,
16562.000030174851) and is stable at one (16562.000023022294), while y and z are identical
everywhere. Only x, which fits shared lattice nodes being written by more than one
macro-element: y and z land on exactly representable values so the order cannot matter, and
x does not. Everything downstream inherits it -- the analytic pressure seed varies by 20%
in its (near-cancelling) checksum, so the initial state differs before any solver runs.

The measure is the one already applied to the element scatter: give each shared lattice node
a single canonical writer. Better still, compute a node's coordinate as a pure function of
the macro corners and its lattice index, evaluated once per node rather than once per
incident macro-element, which removes the question rather than ordering it. This is in
`smesh`, not here.

### 2. Krylov reductions, in `algebra/openmp/sfem_openmp_blas.hpp`

`dot` and `norm2` use `reduction(+ : ret)`, which combines partial sums in an unspecified
order over chunks whose boundaries follow the thread count. `SFEM_DETERMINISTIC_BLAS=1` now
selects a fixed 256-chunk decomposition, independent of the thread count, summed serially
per chunk and combined in index order. Off by default, since it changes results in the last
bits.

It is **unverified in this build**: the flag never fires, because the spike links a prebuilt
SFEM whose instantiation of these templates comes from the library rather than from the
edited header. Confirming it needs SFEM itself rebuilt. The code is written and gated; the
measurement is owed.

### 3. What is already done

The two-pass scatter, on by default (`SFEM_SS_SCATTER=1`), covering the Jacobian action, the
nodal pressure gradient, the residual, the block diagonal and the block split.

### 4. Measurement discipline, independent of the above

Until the two remaining sources are closed, comparisons should pin the thread count and
report medians of repeats rather than single runs, and prefer quantities measured as rates
over iteration counts. `SFEM_ASSEMBLE_FINE=1` gives a deterministic operator for A/B work
where the memory is affordable. It would be worth adding a determinism check to the test
suite -- one checksum at one thread against the same at N -- so that a regression in any of
this is caught rather than rediscovered.

## Reproducibility, verified end to end

With the smesh coordinate patch built in, all three measures are live and the chain is
closed. Everything below is measured, not projected.

**Mesh coordinates.** N=3 on 8 threads, three runs: 16562.000023022294 every time with
`SMESH_DETERMINISTIC_COORDS=1`, against 16562.000030174851, 16562.000029459596,
16562.000029459596 with it off. The deterministic value is exactly what a single-threaded
run produced before the patch, which is what the ownership rule promised -- the lowest
element was already winning there.

**Operator.** Bit-identical at 1, 2, 4 and 8 threads: the Jacobian action checksums
0.08034669538290462 at every one, as do the residual (-0.63166661236032529) and the initial
state (206.55554994212321). Thread-count independent, not merely run-to-run stable.

**Solve.** N=3, L=8, across thread counts:

| | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| `SFEM_DETERMINISTIC_BLAS=0` | 2000 (cap) | 2000 (cap) | 1688 | 1223 |
| `SFEM_DETERMINISTIC_BLAS=1` | 1030 | 1030 | 1030 | 1030 |

Identical iteration counts and identical u_linf (2.982674e-03) at every thread count. The
~20% iteration noise that made every comparison in this document unreliable is gone.

### One thing worth noticing

The deterministic run is not merely reproducible, it is better: 1030 iterations against 1223
at best and two outright failures to converge. That is not luck. The fixed 256-chunk sum is
partially pairwise, so it is *more accurate* than the serial accumulation a single thread
performs -- which is why the deterministic single-threaded run converges where the
non-deterministic single-threaded run stagnates at its cap. Determinism here costs nothing
and buys accuracy.

It also explains a result reported earlier in this document and never satisfactorily
accounted for: the baseline that "failed to converge single-threaded and succeeded on eight".
That was never about thread count. It was a solver sitting close enough to stagnation that
the accumulated error in a long serial sum decided the outcome, and the thread count only
changed how that sum was grouped.

### The three measures

| measure | where | default |
|---------|-------|---------|
| `SMESH_DETERMINISTIC_COORDS` | smesh, `sshex8_fill_points*` | on |
| `SFEM_SS_SCATTER` | this spike, five sshex8 kernels | on |
| `SFEM_DETERMINISTIC_BLAS` | `algebra/openmp/sfem_openmp_blas.hpp` | on |

All three are on by default, each opting out with `=0`.

The reduction was switched on after measuring what it costs, which is nothing: per-iteration
time went from 8509 to 7672 microseconds at N=3 and 1880 to 1811 at N=1. The `reduction`
clause privatises and combines per thread; a flat chunk array summed once is cheaper. So it
is reproducible, more accurate and faster, and there was no trade to weigh.

The chunk count is a function of the length alone -- serial below 4096 elements, and above
that growing so a chunk stays near 8192. Depending only on the length is what keeps the
result identical across thread counts. A fixed 256 was wrong at both ends: it spawned 256
chunks over almost no work for short vectors, which is the same thread-team overhead that
made small multigrid levels slower on more cores, and for long ones it left each chunk a
serial sum whose error grew with the problem.

Verified with nothing set, N=3, L=8:

| threads | 1 | 2 | 4 | 8 |
|---------|---|---|---|---|
| iterations | 1178 | 1178 | 1178 | 1178 |
| u_linf | 2.985426e-03 | 2.985426e-03 | 2.985426e-03 | 2.985426e-03 |

`cvfem_ns_op_gate` passes.

One practical note, learned three times over in this document: the spike compiles the
*installed* SFEM headers, not the ones in this tree. Editing `algebra/` here changes nothing
until `build64` is rebuilt and installed, and the symptom is a measurement that silently
matches the old behaviour. Check a changed default against its own opt-out before believing
it took effect.

## The GMG comparison, rerun with everything deterministic

Every earlier comparison in this document carried about twenty percent of iteration-count
noise and should be read as indicative at best. With the mesh, the operator and the
reductions all deterministic, iteration counts now repeat exactly -- 526/526, 40/40,
1834/1834, 104/104 on repeated runs -- so these numbers mean what they say.

Two Newton steps, 8 threads, baseline is block-Jacobi + BiCGStab, V-cycle is assembled
Galerkin with `SFEM_GMG_KSMOOTH=16` and the default fine smoothing of 2.

| case | dofs | baseline | V-cycle | verdict |
|------|------|----------|---------|---------|
| N=1, L=8  |  10,692 |  526 its, 0.76 s |  40 its, 0.37 s | **2.1x faster** |
| N=1, L=16 |  75,140 | 1834 its, 7.51 s | 104 its, 4.96 s | **1.5x faster** |
| N=2, L=8  |  75,140 | 1061 its, 2.89 s | 230 its, 10.35 s | 3.6x slower |
| N=3, L=8  | 242,500 | 1178 its, 6.99 s | 430 its, 51.6 s | 7.4x slower |
| N=2, L=16 | 561,924 | 1369 its, 25.7 s | 2000 capped, wrong answer | fails |

Tuning recovers a good deal of that -- at N=3, L=8, three levels with the coarse solve capped
at 30 iterations and `KSMOOTH=8` gives 120 iterations in 8.5 s against 430 in 51.6 s, and at
N=2, L=16 it turns a wrong answer into 307 iterations and a correct one in 48.2 s -- but in
neither case does it overtake the baseline.

### The result is about lattice depth, not problem size

The third and second rows are the same problem size, 75,140 unknowns, decomposed differently:
one macro-element with a level-16 lattice against eight macro-elements with a level-8 one.
The V-cycle is 1.5x faster on the first and 3.6x slower on the second. Size is not the
variable; how much of the mesh is lattice rather than macro-elements is.

The reason is structural. `create_gmg_data` derefines the lattice and stops at the macro
mesh, which it never coarsens. At N=1 the coarsest level is 20 nodes and free. At N=3 it is
208 nodes whose Galerkin operator needs the dense fallback -- 43,264 blocks, 832 probe
applications -- and the coarse solve alone is 21% of the run at 26.7 ms per application,
with the two smoothed levels above it taking another 59%. The hierarchy runs out of levels
while the problem is still big.

That also explains why the baseline moves in the opposite direction: it takes 1834 iterations
on the deep lattice and 1061 on the shallow one at equal size. A deep lattice is the hard
case for a pointwise preconditioner, and it is exactly where the V-cycle pays off.

So the honest summary is narrower and better supported than any earlier one here: **the
V-cycle wins where the mesh is mostly lattice, and loses where it is mostly macro-elements**,
and it wins by more the deeper the lattice. For the semi-structured meshes this work exists
to exploit -- few macro-elements, deep lattices -- that is the favourable regime.

The next step is not more tuning. It is to keep coarsening below the macro mesh, so the
hierarchy does not terminate on a problem that is still large and dense; that is a change in
`create_gmg_data` rather than in this spike.

## Why it capped, and what fixing that revealed

The N=2, L=16 failure above was not a weak preconditioner. The coarsest level's solve was
diverging, and the cycle was faithfully prolonging the result.

`SFEM_GMG_COARSE_VERBOSE=1` on that configuration shows BiCGStab on the 81-node coarsest
operator taking its residual from 1.577 to 36066 over a hundred iterations, from 17.26 to
82130, from 0.914 to 273. It then exhausts its 200-iteration cap and returns that amplified
vector as the coarse-grid correction. The V-cycle amplified by 1e6 to 1e9 per cycle, FGMRES
could not converge against such a preconditioner, the linear solve hit its own cap, and
Newton stepped from a badly solved system to an answer three orders of magnitude wrong.
Depth confirms the localisation: cycle rates are 0.06, 0.20 and 0.07 for two, three and four
levels, and 2.8e6 at five.

The tuning that appeared to rescue it was not addressing the cause. `MAX_LEVELS=3` removed
the offending level and `COARSE_MAX_IT=30` limited how far the divergence could run.

### The fix, and what it says about everything above

A few hundred unknowns should not be handed to an iterative solver at all. The coarsest
level is already stored dense, so it is now factorised: `DenseLU`, recovered by applying the
operator to unit vectors, exact and incapable of diverging. `SFEM_GMG_DENSE_LU_BELOW`
(default 4096 unknowns) selects it.

With that in place, every case improves and the V-cycle wins everywhere:

| case | dofs | baseline | V-cycle | speedup |
|------|------|----------|---------|---------|
| N=1, L=8  |  10,692 |  526 its, 0.72 s | 36 its, 0.32 s | 2.2x |
| N=1, L=16 |  75,140 | 1834 its, 6.88 s | 80 its, 3.76 s | 1.8x |
| N=2, L=8  |  75,140 | 1061 its, 2.69 s | 55 its, 1.79 s | 1.5x |
| N=3, L=8  | 242,500 | 1178 its, 6.03 s | 71 its, 5.71 s | 1.06x |
| N=2, L=16 | 561,924 | 1369 its, 25.7 s | 84 its, 15.8 s | 1.6x |

Against the previous section, N=2 L=8 goes from 230 iterations and 10.35 s to 55 and 1.79 s,
N=3 L=8 from 430 and 51.6 s to 71 and 5.71 s, and N=2 L=16 from a wrong answer to the
fastest arm in the table.

**This retracts the conclusion of the previous section.** That section read the N=1 wins and
the N>1 losses as evidence that the method depends on lattice depth, and blamed the
hierarchy terminating at the macro mesh. That was wrong. The variable was the coarsest
level's solve, which fails harder at larger N because the coarsest operator is bigger and
worse conditioned there; the correlation with macro-element count was real and the causal
story attached to it was not.

The iteration counts now say something the noisy measurements never could: 36, 80, 55, 71
and 84 across 10,692 to 561,924 unknowns. Fifty times the problem for a bit over twice the
iterations is close to the level independence a multigrid method is supposed to deliver, and
the baseline over the same range goes from 526 to 1369.

`cvfem_ns_op_gate` passes.

### On the coarsening work that was proposed next

It was proposed on the strength of the retracted conclusion, so its justification is gone
rather than merely weakened. Coarsening below the macro mesh may still be worth doing -- at
N=3 the margin is only 1.06x, and a deeper hierarchy is the obvious way to widen it -- but it
should be argued from measurements taken with the coarse solve working, not from those above.

## Phase 0: the gate says build Phase 1 and drop Phase 2

Before building coarsening below the macro mesh, two measurements were taken to check that a
deeper hierarchy would help at all. Neither supports it.

### M1 — assembly is now the dominant cost

With the coarse solve fixed by the dense LU, the probing assembly is the largest single term
in the cycle: 32.7% of the run at N=3 L=8 (3.43 s of 10.49 s, 571 ms per call) and 67.4% at
N=1 L=8. The coarse solve it used to hide behind is now 0.2%.

### M2 — more levels is worse, not better

The macro mesh is a cube refined to level L, so the same fine discretisation is reachable at
several macro/lattice splits. At 64x16x16 (75,140 unknowns), varying only the split -- which
is exactly what a macro-mesh coarsening would do, executed by hand:

| macro / L | levels | iterations | t_solve | t_prec | total |
|-----------|--------|-----------|---------|--------|-------|
| 16x4x4 / L=4  | 3 | 51 | 1.66 | 4.88 | 6.55 |
| 8x2x2 / L=8   | 4 | 54 | 1.83 | 1.60 | **3.43** |
| 4x1x1 / L=16  | 5 | 80 | 3.83 | 1.44 | 5.27 |
| baseline      | - | 1834 | 7.20 | 0.01 | 7.20 |

Five levels is worse than three on iterations (80 against 51) and worse on total time. Adding
levels below the macro mesh would extend the hierarchy in exactly the direction that measures
worse. **The Phase 2 gate fails; the AMG should not be built for this problem.**

A measurement error is worth recording, because it briefly pointed the other way. `t_solve`
covers only the Krylov solve; `refresh_gmg`, and therefore the whole assembly, is counted in
`t_precond`. Comparing arms on `t_solve` alone credited the shallow-lattice arm with 1.54 s
while ignoring its 4.88 s of assembly. Only the total is meaningful when the arms have
different level counts.

### What the same numbers say about Phase 1

The two effects run against each other. Iterations improve as the lattice gets shallower and
the macro mesh finer (51, 54, 80), because the coarse levels are then better resolved. But
the probing assembly gets sharply worse in that direction (4.88, 1.60, 1.44 s), because the
coarsest level has more nodes and the pattern guess falls back to dense -- 425 nodes and 1700
probe applications in the best-iteration arm, against 20 nodes and 80 in the worst.

So the configuration that converges best is the one probing punishes hardest. Removing the
probing does not merely save its own 30-70%; it unlocks the split that wins on iterations.
That is the case for Phase 1, and it is stronger than the one the plan was written on.

### Correction: the Phase 2 gate was run in the wrong regime

The conclusion above -- that coarsening below the macro mesh is not worth building -- is
withdrawn. It was drawn from experiments whose coarsest level was 20 to 425 nodes, where the
exact coarse solve is free and extra levels can therefore only add cost. The gate could not
have returned anything else.

The cost that coarsening removes is the *terminal* problem's size, and a dense factorisation
is O(n^3) in time and O(n^2) in memory:

| macro nodes | dofs | LU storage | factor flops | |
|---|---|---|---|---|
| 425 | 1,700 | 22 MiB | 1.6e9 | free |
| 1,024 | 4,096 | 128 MiB | 2.3e10 | marginal (per Newton step) |
| 4,096 | 16,384 | 2.0 GiB | 1.5e12 | impossible |
| 65,536 | 262,144 | 512 GiB | 6.0e15 | impossible |

`SFEM_GMG_DENSE_LU_BELOW` is 4096 dofs, i.e. 1024 macro nodes. Beyond it the driver falls
back to the BiCGStab that was measured to diverge on this operator. So today there is a hard
ceiling at about a thousand macro elements, above which there is neither a working exact
coarse solve nor any way to make the coarse problem smaller.

Extending the M2 ladder one rung into that regime, same fine mesh of 64x16x16 throughout:

| macro / L | coarsest | assembly | t_precond | t_solve | total |
|-----------|----------|----------|-----------|---------|-------|
| 8x2x2 / L=8   | 81 nodes | 6,561 blocks, 324 applications | 1.60 | 1.83 | 3.43 |
| 16x4x4 / L=4  | 425 nodes | 180,625 blocks, 1,700 applications | 4.88 | 1.66 | 6.55 |
| **32x8x8 / L=2** | **2,673 nodes** | **7,144,929 blocks (915 MB), 10,692 applications** | **144.4** | **386.1** | **~530** |
| baseline | - | - | 0.01 | 7.20 | 7.20 |

Seventy-four times slower than the baseline on the same discretisation. Both failure modes
appear together: the probing assembly goes fully dense at 7.1 million blocks, and the coarse
solve drops to a Krylov method that cannot be trusted on this operator.

This is the regime that matters for a real macro mesh, and it needs both phases rather than
one:

- **Phase 1** removes the dense-pattern probing, which is what produced the 915 MB operator
  and the 10,692 applications per Newton step. A sparse triple product yields the exact
  pattern, which for a 2,673-node level is a normal sparse matrix rather than a dense one.
- **Phase 2 is reinstated**, but its purpose is not the one the plan gave it. It is not there
  to add levels for faster convergence -- M2 correctly showed that does not help when the
  coarse problem is already small. It is there to **bound the size of the terminal problem**,
  so the exact coarse solve stays affordable as the macro mesh grows. That is invisible below
  the factorisation knee and decisive above it.

The gate should be re-run above the knee once Phase 1 lands, since Phase 1 changes the
assembly cost that currently dominates this measurement.

## Can the coarsest level be assembled directly instead of probed?

`hessian_bsr` refuses on the semi-structured path but works on an unstructured level, so the
coarsest level could be assembled outright and the last remaining probe dropped. That trades
a Galerkin operator for a rediscretised one, and the proposal was that projecting the
velocity and pressure properly -- an L2 projection rather than the partition-of-unity average
used now -- would make the rediscretised operator good enough.

The gap narrows as the coarse mesh resolves, which is the right trend:

| coarsest | raw gap (ux) | best-fit scale | after-scale (ux) | after-scale (p) |
|----------|--------------|----------------|------------------|-----------------|
| 81 nodes   | 13.86 | -0.007 | 0.995 | 0.870 |
| 425 nodes  |  3.49 |  0.074 | 0.964 | 0.634 |
| 2673 nodes |  1.42 |  0.336 | 0.805 | 0.628 |

At 81 nodes the two operators are nearly orthogonal, which is no surprise: a rediscretised
Navier-Stokes operator on 81 nodes is a different problem, not a coarse version of the same
one. By 2673 nodes the raw gap has fallen tenfold and the correlation has risen to a third.

### The state is not what separates them

`SFEM_GMG_CONST_STATE=1` gives every level the same constant field. Averaging and an L2
projection reproduce a constant identically, so the two operators are then evaluated at
genuinely the same state and the state is eliminated as a variable:

| coarsest | after-scale ux (real state -> constant) | after-scale p (real -> constant) |
|----------|------------------------------------------|----------------------------------|
| 81 nodes   | 0.995 -> 0.955 | 0.870 -> 0.870 |
| 425 nodes  | 0.964 -> 0.602 | 0.634 -> 0.634 |
| 2673 nodes | 0.805 -> 0.735 | 0.628 -> 0.628 |

**The pressure figures do not move at all.** They are identical to four digits, which is what
they must be if the cause is the discretisation: Rhie-Chow's `Df = rc h^2 / (2 mu)`, the
pressure Laplacian and the divergence block are all state-free, and only momentum convection
depends on the state. That the numbers are bit-identical is also a check that the diagnostic
is measuring what it claims.

Velocity does improve -- 0.964 to 0.602 at 425 nodes -- so part of that gap really is the
state, and a better projection would recover it. But even with a perfect, exactly
representable state the velocity operators still differ by 60 to 90 percent after optimal
per-component rescaling.

So the answer is no, on this evidence. An L2 projection is worth having on its own merits,
since it would sharpen the coarse operator's convective coefficients, but it cannot make
direct assembly a substitute for Galerkin here: most of the disagreement, and all of the
pressure disagreement, is in the discretisation rather than in the state. The probe at the
first coarse level stays, and its cost is now 108 operator applications rather than 10,692.

## Element-wise Galerkin: assembling the coarse operator inside the macro element

The last probe survives only because the fine operator has no matrix form. It does not need
one. The fine operator is a sum of macro-element contributions and the prolongation's support
is local -- a fine node interpolates only from coarse nodes of the sub-cell containing it, and
a face node gets the same contributors from either side -- so

    R A P  =  sum_e  P_e^T A_e P_e

element by element, with no global matrix ever formed and nothing reaching outside a macro
element.

**The enabling property is verified, not assumed.** Element-wise Galerkin is exact only if A
really is a sum of element operators, and Rhie-Chow couples through a *nodal* pressure
gradient, which is not element-local -- unless it is frozen from the state rather than
recomputed from the direction. Putting a direction on one macro element's interior and
measuring the response outside it gives exactly zero (max |out| inside 4.2480e-02, outside
0.0000e+00 at L=8; likewise at L=4). The frozen gradient is what makes this work, and it is
worth knowing that changing Rhie-Chow to differentiate the pressure gradient would silently
invalidate the construction.

### Cost

Per macro element, in element-kernel evaluations:

| L (hop) | fine nodes | coarse/elem | global probe | local probe | local matrix |
|---------|-----------|-------------|--------------|-------------|--------------|
| 2  |   27 |   8 | 108 |   32 | ~1-4 |
| 4  |  125 |  27 | 176 |  108 | ~1-4 |
| 8  |  729 | 125 | 176 |  500 | ~1-4 |
| 16 | 4913 | 729 | 192 | 2916 | ~1-4 |

Two variants. *Local probing* -- applying the element kernel to each local coarse basis
function -- only beats global probing for shallow lattices, and is three times worse at L=8.
*Local matrix* -- assembling the macro element's own sparse operator once on its lattice
stencil and then doing a small triple product -- wins everywhere, and turns assembly from
108-192 global operator applications into something comparable to a single one.

The local matrix is transient, one per element or per thread: 729 blocks and 0.09 MiB at
L=2, 19,683 and 2.40 MiB at L=8, 132,651 and 16.19 MiB at L=16.

### What it would need

Only one new kernel: a macro-element-local assembly for the semi-structured CVFEM operator.
The per-micro-cell Jacobian entries already exist -- the unstructured path assembles them
into a global BSR -- so the work is scattering them into a lattice-local structure instead of
into the global matrix. Everything else is in place: the structured prolongation gives P_e
directly, and the deterministic two-pass scatter already built for the operator kernels is
exactly what accumulates the local contributions into the coarse BSR.

It also removes the last of the machinery this section has been dismantling: no probing, no
colouring, no sparsity pattern to derive or guess, and no global fine matrix.

### Element matrices all the way down, BSR only at the coarsest level

Element-locality does more than remove the probe. If the coarse operator is built inside the
macro element, it can also be *kept* there: each coarse level becomes a set of Galerkin
element matrices applied with GEMM, and only the coarsest level needs a global sparse matrix,
because that is the only level that is factorised.

This is SFEM's existing idiom rather than a new one --
`frontend/ops/sfem_SemiStructuredEMLinearElasticity.hpp` assembles the element matrix on the
fly and applies it with GEMM -- and the pieces are already in this spike:
`subpar/cvfem_sshex8_em.hpp`, the `em24`/`em32` bench columns, and
`packed_elements_matmul_sym` / `_nonsym` in `operators/packed_elements.hpp`.

| coarse lattice | nodes/elem | EM dofs | EM MiB/elem | assembled BSR, 32 elems |
|----------------|-----------|---------|-------------|-------------------------|
| level 4 | 125 | 500 | 1.91 | 8.81 |
| level 2 |  27 | 108 | 0.09 | 1.40 |
| level 1 |   8 |  32 | 0.01 | 0.27 |

The last hop is a 32x32 element matrix, which is exactly the `em32` shape the bench already
measures.

**Correction, from building it.** The table above compares a *dense* element matrix against
the BSR, and on that basis the storage runs the other way at the shallow end -- level 4 costs
about seven times the assembled BSR. The kernel that was actually written does not store a
dense element matrix. A coarse node couples only to its 3x3x3 lattice neighbourhood, so the
local operator is a 27-point stencil of `(Lc+1)^3 * 27` blocks, and the only excess over the
assembled BSR is duplication at shared macro-element faces:

| coarse lattice | dense EM vs BSR | 27-stencil EM vs BSR |
|----------------|-----------------|----------------------|
| level 2 | 1.0x | 3.38x |
| level 4 | 4.6x | 1.95x |
| level 8 | 27.0x | 1.42x |

So the storage objection to keeping intermediate levels as element matrices is much weaker
than the first estimate suggested, and it *improves* with lattice depth rather than worsening.
That reopens "BSR only at the coarsest" as a real option rather than something the numbers
argue against; what settles it is a measurement of the two applies, not the storage.

One property makes this cleaner than it first appears. The prolongation composed within a
macro element is itself a trilinear interpolation, so any level's element matrix can be
formed directly from the fine element operator as `(P_e^{L->l})^T A_e P_e^{L->l}` rather than
by chaining level-to-level products. Each coarse level is then independent of the others: no
error accumulates through repeated Galerkin products, and a level can be rebuilt without
touching its neighbours.

The resulting architecture drops nearly everything this section has been repairing:

- fine level: matrix-free, unchanged
- intermediate coarse levels: Galerkin element matrices, GEMM apply, assembled on the fly
  where storage warrants it
- coarsest level only: assembled BSR, for the dense LU that must stay exact

No probing, no colouring, no sparsity pattern derived or guessed, no SpGEMM, and no global
sparse matrix above the coarsest level -- so the unsorted-column hazard, the `mm` workspace
sizing and the host-only serial transpose all stop applying. The coarse block diagonals the
smoothers need come from summing element contributions, which the deterministic two-pass
scatter already does.

### Element-wise Galerkin, implemented

`SFEM_GMG_EGAL=1` (default) builds the level-1 coarse operator as `sum_e P_e^T A_e P_e`
straight from the fine macro-elements. That is where the probe was -- level 1 is the only
level whose operator above it is matrix-free and has no matrix form; below it the level above
IS a matrix and `rap` is already exact and cheap. It does not turn Galerkin coarsening on:
that is still `SFEM_GMG_GALERKIN=2`, off by default, so a run setting neither still gets
rediscretised coarse operators.

**What made it cheap.** Three structural facts, none of which needed new physics:

1. *The micro-cell matrix was already reachable.* Passing identity slots (`sl[k] = k`) to
   `cvfem_hex8_ns_upwind_jacobian_add_slots` writes a dense 8x8-block cell matrix into a local
   buffer -- the trick `assemble_block_diag` already used. So the entries come out directly
   and nothing is probed.
2. *A micro-cell's coarse support is exactly eight nodes.* The cell spans fine indices
   `[xi, xi+1]`, whose coarse floors differ by at most one, so per axis it reaches coarse
   indices `{ax, ax+1}` and no more -- for any ratio `q`, not only 2:1. The triple product is
   a fixed 8x8 -> 8x8 contraction rather than something growing with `L`.
3. *The weights depend only on the offset class* `(xi%q, yi%q, zi%q)`. There are `q^3` of them,
   shared by every cell and macro-element, so they are a table built once and never an array
   indexed per entry -- the principle `cvfem_ss_transfer.hpp` applies to the prolongation.

**Cost.** Per micro-cell the two contraction stages are `8*27` and `27*8` block
multiply-accumulates, so 432 against the 1024 a dense 8x8 -> 8x8 contraction would need. The
27 is the average row count of the prolongation restricted to a cell: one corner interpolates
from 1 coarse node, three from 2, three from 4, one from 8, and `1+6+12+8 = 27`. Stage 1
contracts through a precomputed transpose of that map so each output block is stored once
rather than zeroed and accumulated into, saving 1024 scalars of zeroing per cell for the same
arithmetic. Assembly runs in chunks of macro-elements sized to keep the staging buffer near
32 MiB.

**The pattern is derived, not guessed.** An entry exists exactly where two coarse nodes share
a macro-element and sit within one lattice step of each other, which is the true Galerkin
pattern. That removes the probe's worst failure mode outright: an entry outside a too-narrow
guess was not dropped but folded into the wrong slot, so a bad guess gave a wrong matrix
rather than an approximate one, and the retry loop that widened it is what produced the
7,144,929-block dense coarse operator. The derived pattern is also tighter than the probe's --
112 blocks at the coarsest level where the probe padded to 144, which `rap` independently
confirms is the true count.

**Determinism** comes for free: accumulation runs over destinations rather than sources, each
block summing its own contributions in a fixed order with chunks in fixed order too, so there
are no atomics and the matrix is the same bits on any thread count. This is the two-pass
packed idea from the apply path, applied to assembly.

### Levels chain after all, once constraints are in play

The design note above claimed a level at ratio `q` could be built straight from the fine
element matrices, with no chaining, because piecewise-linear interpolation on nested uniform
lattices composes to the direct map. That is true of the **raw** operator, and the gates
measure it at 2.2e-16 and 4.4e-16 through the last hop.

It is false for the hierarchy the driver actually builds. Its transfers zero constrained
degrees of freedom at *every* hop, so the composite is `R2 Z1 (R1 Z0 A Z0 P1) Z1 P2` -- a `Z`
at each level, not only at the fine end. Level 2 built by chaining therefore coarsens a
level-1 matrix that already carries identity rows, and those rows contribute to the product.
A level built directly from level 0 cannot see them.

This did not show up in any operator comparison, because the operators agree to 2e-16 on
unconstrained columns. It showed up in the **block diagonal**, which differed by 1.2e-2 at
level 2 and 5.7e-2 at level 3 -- and the block diagonal is what the smoother inverts. Those
are not round-off; the smoother was being handed wrong diagonals at two of the three coarse
levels.

Masking the constrained coarse columns as well is *not* the fix; it makes level 1 disagree too
(4.4e-18 -> 7.3e-3), because the probe does not mask them. The fix is to leave the chaining
alone: element-wise Galerkin replaces the probe at level 1, and `rap` continues to build the
levels below from the level above, which is the already-validated path. With that, every gate
reads machine precision at every Newton step -- the level-1 operator at 4e-16 and its block
diagonal at 1.5e-17.

**The gate that missed it** was copied from the `rap` check, which zeroes constrained rows on
both sides with the comment that "those rows are not part of what is being tested". That is
right for testing a triple product and wrong for testing a replacement construction: it makes
the comparison blind to precisely the rows where two constraint treatments differ. Comparing
the block diagonals, which no operator gate covers, is what located it.

### Gates

On a 2x1x1 macro mesh at L=8 with `SFEM_GMG_CHECK=1 SFEM_GMG_GALERKIN=2`:

| gate | what it settles | result |
|------|-----------------|--------|
| `egal identity (q=1)` | the assembly reproduces `A` itself -- cell matrix, geometry, Rhie-Chow, pattern, scatter, at once | 1.540e-16 |
| `egal galerkin (0->1)` | the coarsening reproduces `P^T A P` against the matrix-free composite | 2.199e-16 |
| `egal galerkin (0->2,3)` | direct equals chained for the raw operator | 2.210e-16, 4.444e-16 |
| `egal level 1` | the constrained operator equals the probed composite it replaces | 2.1e-16 |
| `egal diag 1` | the block diagonal the smoother inverts equals the probed one | 4.4e-18 |
| `rap level 2,3` | the levels below, unchanged | 1.8e-16, 1.8e-16 |

The identity gate is the one worth keeping. At `q = 1` the prolongation is the identity, so
`P^T A P` is `A`, and one comparison covers everything the construction rests on: that
identity slots really do yield the micro-cell matrix, that the hoisted geometry and Rhie-Chow
struct fed to the assembly are the ones the apply uses, that the derived pattern holds every
entry, and that the inverted-index accumulation lands each block where it belongs.

### The whole hierarchy element-wise: the coarse constraints already existed

The section above concluded that extending the element-wise construction below level 1 was
blocked by the constraint treatment. It was not. `create_gmg_data` derefines the `Function` at
every level (`f_prev->derefine(fs_next, true)`, via `DirichletConditions::derefine`), so each
level already carries its own constraints, and the transfers apply them as `R = Z_coarse Rhat`
and `P = Z_fine Phat`. The composite at a hop is therefore

    Z_i Rhat A_{i-1} Z_{i-1} Phat

-- mask the source level's columns with that level's own mask, contract with the plain
interpolation, patch identity rows at the target. That is the recipe the `rap` branch was
already using (`mask_block_columns`, `rap`, `patch_identity_rows`), one level up.

So the hierarchy is built by chaining, but the chaining happens *inside the macro-element*:
`galerkin_hop` coarsens a level's 27-point stencil element matrices to the next level's,
masking between hops, with no global matrix at any point. Because the mask acts on the matrix
rather than on the transfer, the interpolation stays scalar -- no per-component prolongation is
needed even though the constraints are per component. The coarse stencil stays 27-point for any
ratio: a source node and its 27-neighbour land on coarse nodes at most one step apart, since
reaching two would need their coarse floors to differ while the upper one is off-lattice, and a
differing floor forces it to be on-lattice.

With that, every level matches the construction it replaces, block diagonals included:

| level | operator | block diagonal, before -> after |
|-------|----------|--------------------------------|
| 1 | 2.1e-16 | 4.4e-18 -> 4.4e-18 |
| 2 | 1.2e-17 | **1.2e-2 -> 4.6e-18** |
| 3 | 7.1e-17 | **5.7e-2 -> 0 (exact)** |

Nothing is probed at any level, and no triple product touches a global sparse matrix.

**A latent bug this exposed.** Moving the assembly ahead of the level loop segfaulted, because
it read the operator's cached state fields -- whatever the last `apply()`, `gradient()` or
`update()` happened to leave there. It had been correct only by accident of call order, which
is the kind of dependency that is invisible while it holds. `assemble_hierarchy` now takes the
fine state explicitly and calls `update()` itself.

### Keeping levels as element matrices

The element matrices can also be kept and applied directly rather than assembled: gather a
macro-element's coarse nodes, run the 27-point stencil over its local matrix, reduce.
`make_element_matrix_level` does that and `SFEM_GMG_CHECK=1` compares it against the assembled
operator it replaces.

The storage objection is weaker than the design note above estimated, because that estimate was
for a *dense* element matrix and the kernel builds a stencil one. Both storage and block
multiplies exceed the assembled form only by duplication at shared macro-element faces -- 3.38x
at a level-2 coarse lattice, 1.95x at level 4, 1.42x at level 8 -- improving as the lattice
deepens rather than worsening. What is bought is contiguous 4x4 blocks with no column
indirection.

With `galerkin_hop` this is now genuinely reachable: the hop coarsens element matrices to
element matrices, so a whole hierarchy can exist without any level being assembled, and only
the coarsest -- the one that is factorised -- needs a BSR. What still stands in the way is not
the construction but the driver, which takes each level's operator as a matrix (`g.Amat[i]`)
and hands it to the smoothers and the dense solve. That is a plumbing change, not an
algorithmic one, and it is wired out rather than wired wrong.

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

**2. Specialise the gather.** Done. Each block now gathers only the arrays it reads and
scatters only the rows it writes, and on Grace the blocks a Schur scheme wants got about a
quarter cheaper:

| | before | after |
|---|---|---|
| floor | 35.3% | 12.2% |
| `pp` (C) | 46.4% | **36.1%** |
| `pu` (B) | 53.0% | **43.2%** |
| `con` rows | 54.9% | **45.5%** |

`uu` and `mom` barely moved, 2 points, which is right: they read almost everything anyway.
Two constraints cap what C can save. The boundary term takes the state velocity whatever
is masked, and the macro geometry needs the coordinates, so C still gathers seven of the
fourteen arrays rather than the three its own arithmetic uses.

**3. Wire the semi-structured kernels into the Op and the driver.** Done. The operator
picks the path from what the space carries -- `has_semi_structured_mesh()` -- rather than
being configured, and the driver turns a mesh semi-structured with
`SFEM_ELEMENT_REFINE_LEVEL`. The same problem decomposed three ways:

| | nodes | elements | newton | lin_it | u_linf |
|---|---|---|---|---|---|
| flat, N=8 | 2673 | 2048 | 19 | 15420 | 9.706443e-10 |
| N=4, level 2 | 2673 | 512 macros | 19 | 15409 | 9.706011e-10 |
| N=2, level 4 | 2673 | 32 macros | 19 | 15440 | 9.706217e-10 |

Identical discrete problem, same Newton count, `u_linf` agreeing to six figures, solved
through 32 macro-elements instead of 2048 flat ones.

Writing this needed the residual, which the semi-structured path did not have -- it had
the Jacobian action, the block diagonal and the block split, none of which Newton can
start from. It is implemented in the same two layouts as everything else so the naive one
gates the macro-local one, and agrees at 2.6e-15.

Two limits are deliberate. The path is affine-macro only: one Jacobian per macro-element,
reused across its lattice, which is exact for a box and wrong for a curved macro-element,
and it ignores `SFEM_GEOM` for the same reason. And it refuses `hessian_bsr`, because an
assembled matrix per level is the memory a hierarchy exists to avoid; refusing beats
returning a zero matrix.

**4. Hopper.** Measured, and it says do not port the gather. Two things came out of it.

*Every device figure previously in this file was for a kernel without the Rhie-Chow term.*
None of the `cvfem_cuda_time_*` entry points takes `rc_scale`; `cvfem_cuda_residual_rc`
existed but was only ever verified, never timed, and the timing path attached neither the
coordinates nor the nodal gradient that it needs. Every host figure includes the term. The
comparison was therefore between a device kernel missing a term and host kernels that have
it -- the third comparability error of this work, after the pressure gradient and the
cross-build boundary A/B, and the same shape each time: two sides doing different work with
nothing in the harness to notice. `cvfem_cuda_time_residual_rc` now exists.

| Hopper, packed residual, n=128 | MDOF/s |
|---|---|
| without Rhie-Chow | 7872 |
| with Rhie-Chow | **5899** |

The term costs 1.33x, so it is a quarter of the device kernel -- close to its share on the
host, where hoisting its coefficients was worth 1.28x. Applying that factor to the apply
figure puts Hopper nearer **18x Grace than the 24.6x** reported before; that scaling is an
inference from the residual, since there is no timed apply with the term.

*The macro-local gather is already on the GPU, as packing, and it loses.* The packed
kernel is block-per-pack with shared-memory staging, which is the same transformation, and
the plain global layout beats it by 1.41x for the Jacobian action -- 10420 against 7297
MDOF/s -- confirmed twice in separate builds.

So a semi-structured CUDA port would buy nothing from the half that wins on CPU and
everything from the half the GPU lacks: the device takes `adj` and `det` as precomputed
inputs, so geometry is already hoisted there, but `mdot_coeff` still runs per
sub-control-surface per element. Hoisting those coefficients needs elements sharing a
Jacobian, which packs cannot give and macro-elements can. The ceiling on that is the 25%
above, and realistically less, since the Rhie-Chow term is more than its coefficients.
Worth doing only if a quarter of the device kernel is worth a port.

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

| block | Grace L=4 | L=8 | L=16 | what wants it |
|---|---|---|---|---|
| gather only (floor) | 13.0% | **12.2%** | 12.2% | -- |
| `pp` (C) | 38.0% | **36.1%** | 34.1% | pressure preconditioner, Schur |
| `pu` (B) | 44.8% | 43.2% | 40.5% | `B A^-1 B^T` |
| `con` rows | 46.4% | 45.5% | 43.6% | segregated pressure solve |
| `up` (B^T) | 66.3% | 64.6% | 65.1% | `B A^-1 B^T` |
| `uu` (A) | 87.7% | 86.8% | 87.0% | momentum solve |
| `mom` rows | 95.7% | 95.9% | 96.0% | -- |

Shares are against the full operator measured through the same block kernel in the same
run. `SSBLOCK_ALL` sets every flag, so it gathers everything and the denominator is
unaffected by the specialisation below.

Grace is stable to within a point across L=4..16. L=2 is worse across the board -- the
floor alone is 45% there -- because `(L+1)^3 / L^3` is 3.375, so a macro-element gathers
more than three nodes for every micro-element it runs.

### What the numbers say

**The blocks a Schur scheme needs are the cheap half.** C costs 46% of J on Grace and B
53%, against 89% for A_uu. A_uu is barely cheaper than the whole operator, because the
viscous and convective terms it keeps are most of the cost.

**Asking for the momentum rows is not worth it.** At 98% of J it is within noise of just
evaluating the operator, and on the M1 it is slower. Use the full apply for that.

**The floor was the gather, and specialising it was worth a quarter on the pressure
blocks.** Before, on Grace, the gather and scatter were 35% of the operator and C cost 46%,
leaving eleven points; the M1 put the same floor at 15%, because its kernels are about ten
times slower per dof so the same fixed cost is a smaller share of them. Grace was the one
to believe, and gathering only what each block reads took C to 36% and B to 43%. Note that
the floor figure is now block-dependent by construction -- a `Blocks = 0` sweep gathers
only the coordinates and the state velocity -- so 12% is the floor for a block that reads
nothing, not a bound shared by all of them.

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

## Semi-structured geometric multigrid: a running V-cycle, and why it is not yet a win

`create_gmg_data` is wired up and a V-cycle runs, preconditioning BiCGStab inside the
Newton loop (`SFEM_GMG=1` in `cvfem_hex8_ns_ssgmg`). Getting it to run at all turned on one
parameter, and the result it produces says the smoother is the wrong one.

### The wiring

`Function` owns the coarse `Function`s that `create_gmg_data` derefines but does not hand
their operators back, and every level here needs two things a linear problem would not
need: the state to linearise about, and its own block diagonal. So `CVFEMNavierStokes`
records the operator it produced in `derefine_op` and exposes it as `coarser()`, and the
driver walks that chain from the finest level. Per level it holds a state buffer, restricted
from the fine state with the averaging restriction; a matrix-free operator bound to that
buffer; and a 4x4 block-Jacobi smoother built from `hessian_block_diag`. The coarse level is
solved with BiCGStab. `build_gmg` runs once, `refresh_gmg` per Newton step -- rebuilding the
whole hierarchy per step instead made the first run appear to hang.

Three pieces of the default GMG path are deliberately not used. `create_gmg_operators`
passes `nullptr` as the state, which is fatal for a nonlinear operator.
`create_gmg_default_smoothers_and_solver` computes `sym_block_size = (block_size == 3 ? 6 :
3)`, silently yielding 3 for block size 4, and reaches for `hessian_block_diag_sym`, whose
packing assumes a symmetry a Navier-Stokes block does not have. Its CG coarse solver wants
an SPD system.

### Damping is what made it converge

Undamped, the V-cycle was not merely ineffective but actively harmful: BiCGStab sat on its
1000-iteration cap on every Newton step. Newton still crawled forward on the truncated
steps, which is what made this slow to spot -- the residual fell 2.6e-2 -> 6.4e-6 and only
the iteration counts showed anything wrong.

The cause is that block-Jacobi is being asked to do a different job than elsewhere in this
driver. As a Krylov preconditioner it is applied once and undamped is fine; as a smoother it
is a stationary iteration, and undamped on this saddle-point system it does not converge.
SFEM's own multigrid damps its block-Jacobi by `1/block_size` for exactly this reason.
Measured (N=1, L=4, first four Newton steps):

| omega | lin_it per Newton step |
|-------|------------------------|
| 1.0   | 1000, 1000 (capped)    |
| 0.8   | 1000, 785, 334, 710    |
| 0.7   | 391, 131, 122, 352     |
| 0.6   | 164, 24, 548, 119      |
| 0.5   | 31, 16, 154, 31        |
| 0.4   | 41, 23, 143, 19        |
| 0.25  | 50, 29, 376            |

`SFEM_GMG_OMEGA` defaults to 0.5. The damping applies to the smoothers only; the coarse
solve and the flat block-Jacobi preconditioner are left undamped.

### It is not level-independent, which is the result that matters

Total linear iterations over four Newton steps, V-cycle against the flat block-Jacobi
preconditioner:

| level | V-cycle | block-Jacobi |
|-------|---------|--------------|
| 2     | 48      | 123          |
| 4     | 247     | 303          |
| 8     | 2140    | 1082         |

A working V-cycle holds iteration counts roughly flat as the lattice deepens. These grow
faster than the flat preconditioner's and overtake it by L=8, where the V-cycle is *worse*
than the smoother it is built from.

That first reading -- that the smoother was at fault -- was wrong, and the reasoning behind
it was wrong in a way worth recording. It rested on smoothing steps at L=8 reducing
iterations monotonically (3113, 2140, 1402, 480 for 1, 3, 6 and 12), read as evidence that
the coarse-grid correction was sound and only the smoother was weak. But a damped smoother
is a convergent iteration by itself, so a cycle whose coarse correction contributed nothing
whatever would improve with smoothing count in exactly the same way. Counted in operator
applies rather than iterations the same numbers say the opposite: 7114, 14671, 19223, 13162
against block-Jacobi's 1082. More smoothing was buying less, not more.

### The cost bar a V-cycle has to clear

A V-cycle with three pre- and three post-smoothing steps costs roughly sixteen operator
applies; the flat preconditioner costs one. So the V-cycle has to cut iteration counts by
more than about 16x merely to break even on wall time, not the 2-3x it currently manages.
That is not out of reach -- at L=8 block-Jacobi needs 1082 iterations and an effective
V-cycle would need well under 50, comfortably past the bar -- but it does mean an
almost-working smoother is worth nothing, and the smoother has to be most of the way to
level-independent before the machinery pays for itself.

Wall-clock numbers are not quoted here as a comparison. These runs are at N=1, far below
saturation, where per-apply overhead dominates and the measured 2.1 ms for a 425-node apply
is overhead rather than work. The iteration counts and their growth with level are the
meaningful signal at this size; a wall-clock claim needs a saturated problem and will be
worth making once the cycle is fixed.


## What is actually wrong with the V-cycle

Chasing the above produced a diagnosis, one real bug fixed, and a clear statement of what
still blocks the cycle. The instruments are in the driver behind `SFEM_GMG_CHECK`.

### A control arm that never ran

`SFEM_GMG=2` runs the same damped block-Jacobi as a stationary iteration on the fine level
for the same number of sweeps a V-cycle spends smoothing, with no hierarchy under it. It
exists because iteration counts cannot otherwise distinguish a weak smoother from a broken
coarse correction.

Its first results showed V-cycle and control agreeing to the digit -- 48 against 48, 470
against 470 -- which was not a finding but a bug: the hierarchy was built under `if
(use_gmg)`, so `SFEM_GMG=2` took the `if (gmg)` branch and ran the V-cycle. The control was
unreachable. It now builds only for `SFEM_GMG == 1`.

### The bug: the state was restricted with the residual's operator

Every coarse operator is linearised about a state restricted from the level above, and that
restriction was `create_hierarchical_restriction`. The adjoint test in `check_transfers`
shows that operator is exactly the transpose of the prolongation -- ratio 1.000000 on every
level, once the probe vectors respect the constraints that both transfers impose on their
output. (Probing with unconstrained noise reports a spurious mismatch; the first version of
this test did exactly that and produced ratios of 0.41 and 1.12, which read convincingly as
a broken transfer and were nothing of the kind.)

Being the adjoint is precisely right for the residual and precisely wrong for a state. `P^T`
sums where a state transfer must average, inflating each coarse state by the number of fine
nodes feeding a coarse node -- a measured factor of about 3.8 per level. Every coarse
operator was therefore linearised about a field several times too large. Normalising by `R`
applied to the constant 1 recovers the partition-of-unity average. The effect on the cycle's
own convergence rate at L=8 was the difference between diverging and converging:

| cycle | before | after |
|-------|--------|-------|
| 1     | 5.83   | 0.185 |
| 2     | 1.17   | 0.626 |

### What still blocks it: Rhie-Chow does not survive coarsening

The cycle still turns divergent after the second cycle, settling at about 1.34 per cycle at
L=8, and the V-cycle remains the worst of the three preconditioners:

| level | V-cycle | fine smoother, no hierarchy | block-Jacobi |
|-------|---------|-----------------------------|--------------|
| 2     | 48      | 42                          | 123          |
| 4     | 470     | 84                          | 304          |
| 8     | 2407    | 574                         | 918          |

The coarse-operator consistency check applies `A_c` and `R A_f P` to the same smooth coarse
vector and compares them per component. The rediscretised coarse operator disagrees with
the Galerkin operator the transfers imply by a factor of about six, and the disagreement is
almost entirely in the pressure rows:

| level pair | ux   | uy   | uz   | p    |
|------------|------|------|------|------|
| 0->1       | 0.79 | 0.72 | 0.76 | 6.59 |
| 1->2       | 1.59 | 1.30 | 2.42 | 5.24 |
| 2->3       | 0.00 | 0.00 | 0.00 | 6.01 |

That localises it to the stabilisation. `Df = rc_scale * h^2 / (2 mu)` is the one term that
depends on the lattice spacing outright, so each level stabilises a different equation, and
rediscretisation hands the cycle a coarse pressure operator that is not a coarse version of
the fine one. Holding `Df` at the fine level's value (`SFEM_GMG_RC_DECAY=0.25`) confirms the
mechanism -- the pressure inconsistency falls from about 6 to between 0.6 and 1.2.

The awkward part is that the same change makes the cycle *worse*, taking the L=8 rates to
0.41, 1.30, 1.52. A coarse operator stabilised for the fine level's `h` is closer to the
Galerkin operator and simultaneously under-stabilised on its own mesh, where it is near
enough singular that solving it amplifies what it returns. The two requirements point in
opposite directions, which is the real obstacle: consistency with the fine operator and
stability on the coarse mesh cannot both come from rediscretising with an h-dependent
stabilisation.

Nor is it a scalar. `SFEM_GMG_CGC` scales the prolonged correction; swept over 0.125 to 8 at
L=8, every value diverges eventually -- values below 1 delay it, values above accelerate it
sharply (4 gives 4.7 per cycle, 8 gives 17). A single factor per level cannot repair a
coarse operator that differs in what it does rather than by how much.

### Ruled out

Recorded so they are not re-investigated: the transfer pair (exact adjoints, ratio
1.000000); the pressure null space (every level carries exactly one pressure pin, and
filtering the constant pressure mode out of each prolonged correction with
`SFEM_GMG_PFILTER=1` changes the rates in the fourth decimal); hierarchy depth (capping at
two levels with `SFEM_GMG_MAX_LEVELS`, so the coarse level is the well-resolved L=4 mesh,
diverges at the same 1.33); the nodal pressure-gradient cache (`SFEM_PGRAD_CACHE=0`
reproduces the rates bit for bit); and smoother damping (swept; 0.5 is best and is the
default).

### Where this leaves the preconditioner

Block-Jacobi is still the one to beat, and in work rather than iterations it is not close.
At L=8 it spends about 918 operator applies against roughly 3400 for the no-hierarchy
smoother arm and some 19000 for the V-cycle. The fine-level stationary smoother wins on
iteration count at every level and loses on work at every level.

The next step is not a better smoother -- the evidence points away from that. It is the
coarse pressure operator: either a stabilisation that coarsens consistently, or a coarse
level built as a genuine Galerkin product for the pressure block instead of rediscretised.

## Independent evaluation of the null-space treatment

`nullspace_eval.py` is a standalone study of whether our constant-pressure null space is
what limits the V-cycle, and whether the hybrid matrix-free elimination from the
self-contact rigid-body-modes work helps if applied to it. It models a stabilised
colocated Navier-Stokes system in 2D with the same constant-pressure null space and the
same `Df = rc h^2 / (2 mu)` stabilisation, small enough to solve exactly.

It is gated rather than merely run. Stage 1 requires the model to reproduce the driver's
symptom before anything else is believed; stage 1b requires the smoother to converge at
all; stage 1c requires the condensed operator to solve the problem to round-off before its
cycle rate is quoted. All three gates fired during development and each caught a real
error: a symmetric-indefinite model whose smoother diverged at every damping, a pure Stokes
model missing the convective diagonal that makes our smoother work, a truncated inter-level
transfer, and a right-hand side that double-counted `B_tilde C_lam^-1 g_tilde` by adding
both of the paper's two equivalent forms for it.

The model reproduces our failure closely. Coarse-operator consistency is about 0.5 in the
velocity rows and 5.2 in the pressure rows, against 0.7 and 6.6 in the driver.

**The gauge does not matter.** Pinning the same node on every level, pinning a
level-dependent node, and projecting the constant mode out per level are
indistinguishable, and none is far from the smoother alone:

| treatment                  | rate (n=24) | n=16 | n=32 |
|----------------------------|-------------|------|------|
| pin, shared node           | 0.948       | 0.919| 0.972|
| pin, level-dependent node  | 0.939       | 0.779| 1.143|
| projection, per level      | 0.936       | 0.993| 0.944|
| condensation, per level    | 12.2        | 0.922| 23.3 |

No treatment wins consistently across sizes, which is itself the result: the differences
are noise around a cycle that is limited by something else. The condensation is the
exception in the wrong direction -- its operator is verified correct to 1e-12, so its
divergence is a real property of the scheme here and not an implementation fault, and it
worsens with problem size. That is not a mark against the method in its own setting: it
changes the gauge, and a gauge is not what ails us. It also has to coarsen a dense global
rank-one term on top of a stabilisation that already fails to coarsen.

**The stabilisation is the lever**, and it is non-monotone:

| rc scaled per level | pressure consistency | V-cycle rate |
|---------------------|----------------------|--------------|
| 1.0 (as now)        | 5.18                 | 0.948        |
| 0.5                 | 2.21                 | 0.905        |
| 0.25                | 0.75                 | **0.719**    |
| 0.125               | 0.29                 | 9.81         |

Consistency improves monotonically all the way down while the rate has an optimum at 0.25
-- exactly the value that holds `Df` at the fine level's value -- and then diverges. This
is the tension stated earlier made quantitative: consistency with the fine operator and
stability on the coarse mesh are competing requirements, and the optimum is interior.

One discrepancy to resolve rather than explain away: in the model `rc_decay = 0.25`
improves the cycle (0.948 to 0.719), while in the driver the same setting made it worse
(rates 0.41, 1.30, 1.52 against 0.185, 0.63, 1.05). The exponent is dimension-independent,
since `Df ~ h^2` either way, so 0.25 should be right in 3D too. Candidate causes are the
driver's Reynolds regime, its hierarchy depth, or something still wrong in the driver that
the model does not carry. That is the next thing to chase, and it is a much narrower
question than the one this evaluation started with.

## The smoother was divergent, and the iteration counts were noise

Two findings that overturn parts of the account above.

### The default damping made the smoother diverge

`SFEM_GMG_CHECK=3` runs the smoother standalone as the stationary iteration it actually is
inside a cycle. Its good showing as a BiCGStab preconditioner proved nothing: a Krylov
method tolerates a preconditioner that would diverge if iterated, and inside a V-cycle it
is iterated.

At the then-default `omega = 0.5` the residual falls for about twenty sweeps, bottoms out
near 4.2e-3, and then grows; the per-sweep rate rises monotonically through 1 at around
sweep 27 and reaches 1.038 by sweep 39. An earlier reading of this same measurement stopped
at eight sweeps, saw 0.88 to 0.95, and called the smoother convergent. The rate was still
rising at the point it was cut off.

Asymptotic rates over sweeps 35-39: `omega` 0.5 gives 1.038 and rising, 0.3 gives 0.9717
and flat, 0.15 gives 0.9855, 0.05 gives 0.9915. The default is now 0.35.

This is what the earlier `SFEM_GMG_CGC=0` test was pointing at and what nothing else
explained: with the coarse-grid correction switched off entirely the cycle still diverged
(0.68, 0.80, 0.87, 0.98, 1.11, 1.21), while the smoother allegedly converged. A cycle that
diverges with no coarse correction has nothing to do with its coarse grid.

With a convergent smoother the V-cycle converges as an iteration for the first time.
Cycle rates at L=8: `omega` 0.35 gives 0.25, 0.30, 0.35, 0.57, 0.51, 0.48; 0.3 and 0.25 are
similar; 0.5 still diverges to 1.32.

So the plan's P5 is back, and this time on direct evidence rather than on the inference
that was withdrawn: the smoother is genuinely inadequate here, and damping only moves it
from divergent to barely convergent at 0.97 per sweep.

### The iteration counts in this report carry about a factor of two of noise

Four runs of one identical configuration (L=8, `omega` 0.35, V-cycle) gave 3084, 2354, 2793
and 1485 total linear iterations. Block-Jacobi under the same treatment gave 993 and 979.

The V-cycle path performs far more operator applications, each carrying OpenMP atomic
rounding non-determinism, and an outer BiCGStab that is close to stagnating amplifies the
difference. The consequence is that any single-shot comparison of V-cycle iteration counts
in this document is unreliable at better than a factor of two, which covers the
`SFEM_GMG_PSCALE` sweep, the `omega` 0.35 against 0.5 comparison, and the earlier
level-independence tables. Differences of that size were read as signal and were not.

What survives is what was measured as a rate rather than a count -- the standalone smoother
and cycle rates, which are monotone and reproducible -- and the block-Jacobi-against-V-cycle
gap, which is larger than the spread. Block-Jacobi at about 985 still beats the V-cycle at
1485 to 3084, so the cycle is still not competitive; it has merely stopped diverging.

### What the independent evaluation does and does not transfer

`nullspace_eval.py` predicted that scaling the coarse continuity rows by the measured
pressure/velocity ratio of the best-fit block scales would fix the cycle, and in the model
it does: the predicted beta is optimal at n = 16, 24 and 32 without tuning. The driver
reports the same pathology -- best-fit scales of about 0.63 on velocity and 0.12 on
pressure, a ratio of 0.196 -- but `SFEM_GMG_PSCALE` at that value does not help, before or
after the damping fix, and the differences are inside the noise quantified above.

The model's own stage 1b gate required a convergent smoother before reporting anything.
The driver had no such gate until now, which is precisely how a divergent smoother survived
several rounds of coarse-grid investigation.

## P5 answered: a saddle-point smoother is not the fix

SIMPLE is implemented (`SimpleSmoother`, `SFEM_SMOOTHER=simple`) on the 2x2 block split,
which is what that split was built for. Following the rule the previous section learned the
hard way, it was measured as a standalone smoother with no coarse space before being allowed
anywhere near a cycle. Standalone rates at L=8 over sweeps 35-39:

| smoother | omega | rate |
|----------|-------|------|
| block-Jacobi | 0.35 | 0.9669 |
| SIMPLE       | 0.35 | 0.9667 |
| SIMPLE       | 0.7  | 1.850  |
| SIMPLE       | 1.0  | 3.059  |

SIMPLE matches block-Jacobi to four digits and diverges sooner as damping is relaxed.
Neither more inner sweeps nor rescaling the Schur diagonal changes it.

The block-split gate under `SFEM_GMG_CHECK=1` explains why, and is the reason the null
result is trustworthy rather than a suspected bug. The four blocks sum to the full Jacobian
action to 1.4e-16, so the split is exact, and their norms are `uu` 0.284, `up` 2.231,
`pu` 0.552, `pp` 23.748. The pressure-pressure block -- the Rhie-Chow stabilisation --
is about eighty-five times the momentum block, and the divergence coupling `pu` that SIMPLE
uses to build its pressure correction is a two percent perturbation on it. SIMPLE's Schur
complement `S = Dpp - C Du^-1 B` is therefore `Dpp` to within a couple of percent, its
pressure update reduces to block-Jacobi's, and its velocity correction is negligible.

So this system is not coupling-limited and a saddle-point smoother has nothing to work with.
That is a different diagnosis from the one P5 was written under: the difficulty is not that
velocity and pressure are strongly coupled, it is that the stabilisation dominates the
operator outright.

It is also worth correcting an impression left by the previous section. An asymptotic
smoother rate near 0.97 is not by itself a bad smoother -- a smoother's asymptotic rate is
set by the smoothest mode, which is precisely what the coarse grid exists to remove, and
good multigrid smoothers routinely look terrible measured this way. What is fatal is a rate
above 1, which is what the old default damping produced. With that fixed the smoother is
doing its job, and the remaining weakness is in the coarse correction, where the velocity and
pressure rows still coarsen with different best-fit scales (0.63 against 0.12).

## Why the model's fix did not transfer: rediscretisation is the whole fault

`SFEM_GMG_CHECK=4` applies the two-level correction operator `P A_c^-1 R A` to a chosen
error mode and reports what fraction survives. The mode is built as `P` applied to a coarse
field, so it is exactly representable on the coarse grid and a correct correction must
remove essentially all of it. `SFEM_GMG_GALERKIN=1` swaps the rediscretised coarse operator
for `R A P`, composed matrix-free, which is far too expensive for production and is exactly
the right thing for a diagnostic: with it the surviving fraction is zero by construction if
the transfers are sound. `SFEM_GMG_CGC_SMOOTH=1` seeds two levels down instead of one, so
the mode is smooth relative to the coarse grid rather than oscillatory on it.

| coarse operator | mode | velocity | pressure |
|-----------------|------|----------|----------|
| rediscretised   | coarse-oscillatory | 5.52 | 0.786 |
| rediscretised   | coarse-smooth      | 0.312 | 0.601 |
| Galerkin `R A P`| either             | 0.000 | 0.000 |

The Galerkin correction is exact, which validates the transfers and the test at once. The
rediscretised operator amplifies a coarse-representable velocity error more than fivefold,
and removes only forty percent of a coarse-smooth pressure error. Rediscretisation is the
entire fault; nothing else in the cycle is.

Two checks close off the alternatives. The derefined coarse operator is bit-identical to one
built directly on the coarse space -- `derefine_op` is not the problem. And the disagreement
is not a scaling: removing each component's own best-fit scale still leaves 0.68, 0.52, 0.52
and 0.40 relative error in ux, uy, uz and p. That is why `SFEM_GMG_PSCALE`, `SFEM_GMG_CGC`
and `SFEM_GMG_RC_DECAY` all failed -- the entire family of scaling knobs was addressing a
component of the error that is a minority of it.

This is also the answer to why the independent evaluation's prediction did not transfer. In
the model the mismatch between the coarse and Galerkin operators really was close to a pure
per-block scaling, so scaling the coarse continuity rows by the measured ratio fixed it. In
the driver it is not, so no scaling can. The model was right about itself and about the
method; it was wrong about the driver because the two operators fail in different ways, and
only measuring the after-scale residual in both revealed that.

The V-cycle's behaviour follows exactly. Run long enough its rate climbs to 0.963, against
the smoother's own 0.967: the coarse correction helps for a few cycles, then contributes
nothing, and the residual left behind is pressure, reduced fifty times less than velocity.

### What this means for the next step

Galerkin coarsening works and rediscretisation does not, at least for the pressure block.
Composing `R A P` at solve time is what the diagnostic does and is not an option here --
it makes every coarse application cost fine-level work, which defeats the hierarchy. So the
choice is between assembling the coarse levels once per Newton step and finding a coarse
discretisation that behaves like the Galerkin operator without being it. The measurements
above are the gate either way: any candidate coarse operator should be required to bring the
surviving fraction near zero on coarse-representable modes before it is put into a cycle.

## Galerkin coarse operators, assembled once per Newton step

`SFEM_GMG_GALERKIN=2` assembles `A_c = R A P` into BSR once per Newton step and applies it
as a sparse matrix, so no coarse level reaches back up to a finer one during the solve.
(`=1` keeps the matrix-free composition, which is the diagnostic, not a solver: it puts
fine-level work under every coarse application.)

Assembly does two jobs. It removes the fine-level dependency, and it supplies the coarse
smoother with the diagonal of the matrix it actually smooths -- the matrix-free composite
cannot, and using the rediscretised diagonal instead mismatches the Galerkin operator by the
per-block scale factors (about 1.6 in velocity, 8 in pressure), which alone made the coarse
levels diverge.

The entries are recovered by probing under a distance-2 colouring of the coarse node graph,
so no row ever sees two neighbours of one colour and a whole set of blocks falls out per
application. That is colours x 4 applications instead of one per coarse degree of freedom:
41 colours and 164 applications for 425 nodes, against 1700 for column-by-column.

The pattern is self-correcting, and needs to be. Probing does not drop a non-zero that lies
outside the pattern -- it folds it into the wrong entry, so too narrow a pattern yields a
wrong matrix rather than an approximate one. The coarse mesh graph is right while the mesh
is fine enough that `R A P` does not reach past it, and is wrong on the coarsest levels,
where a handful of nodes are all within reach of each other. The gate caught exactly that:
levels 1 and 2 assembled to 2e-16 while the 20-node coarsest level came out at 3.8e-1. It
now widens to the squared adjacency and then to a dense pattern, and all levels assemble
exactly:

```
gate 2.1236e-16  OK   425 nodes, 8281 blocks, 41 colours, 164 applications
gate 2.7612e-16  OK    81 nodes, 1225 blocks, 31 colours, 124 applications
gate 1.5981e-16  OK    20 nodes,  400 blocks, 20 colours,  80 applications  (dense pattern)
```

### It works two-level and fails multi-level, for a specific reason

Cycle rates at L=8, first three and last three of twelve:

| hierarchy | coarse handling | rates |
|-----------|-----------------|-------|
| rediscretised, 2 levels | solved | 0.096, 0.546, 0.850 ... 0.964 |
| Galerkin, 2 levels      | solved | 0.021, 0.207, 0.238 ... 0.861 |
| Galerkin, 4 levels      | smoothed | 0.433, 5.319, 5.659 ... 5.671 |

Two-level Galerkin is a clear improvement and behaves as the correction-operator measurement
predicted. Four-level Galerkin diverges, and not for want of damping: omega 0.35, 0.2, 0.1
and 0.05 give 5.67, 5.23, 3.09 and 1.32, improving steadily and never reaching 1.

The difference between the two rows is not the number of levels but what happens on the
intermediate ones: solved in the first case, smoothed in the second. The Galerkin operator is
a much better approximation of the fine operator and a much worse candidate for block-Jacobi
smoothing -- it is denser, and its diagonal is not dominant in the way the rediscretised
operator's is. That is the standard trade between the two coarsenings, and it is now the
binding constraint rather than a suspicion.

### Where that leaves it

The two coarse operators fail in opposite directions. Rediscretisation is smoothable and
approximates badly enough that its correction is worthless; Galerkin approximates well and
cannot be smoothed by the smoother available. Two-level Galerkin with a solved coarse level
sidesteps the conflict and is the best cycle measured so far, at 0.861 against 0.964.

The next thing to try is therefore not another coarse operator but a stronger coarse-level
solver: a few Krylov iterations per level in place of the stationary smoother, which
tolerates an operator that block-Jacobi cannot smooth. That carries a consequence worth
stating before it is measured -- a Krylov smoother makes the preconditioner vary between
applications, which BiCGStab does not admit, so the outer solver would have to become
flexible (FGMRES) at the same time.

## Krylov smoothing and a flexible outer solver: the V-cycle finally works

Two changes that had to land together. `SFEM_GMG_KSMOOTH=n` replaces the stationary smoother
with n BiCGStab iterations per level, which does not need the diagonal dominance the
Galerkin operators lack. That makes the cycle vary between applications, and BiCGStab
assumes its preconditioner does not -- it does not fail loudly when that is violated, it
stagnates -- so `cvfem_fgmres.hpp` adds flexible GMRES, selected automatically whenever the
smoother is Krylov. SFEM had no GMRES of any kind.

FGMRES was gated before use: with a fixed block-Jacobi preconditioner it reaches the same
solution as BiCGStab (u_linf 1.6569e-03 against 1.6558e-03). It needs more iterations there,
which is expected -- restarted GMRES discards information at each restart and BiCGStab
performs two operator applications per iteration -- and is beside the point, since it exists
for the case BiCGStab cannot handle at all.

### Total linear iterations over four Newton steps

| refine level | dofs   | block-Jacobi | Galerkin + Krylov smoothing + FGMRES |
|--------------|--------|--------------|--------------------------------------|
| 2            | 324    | 123          | 20                                   |
| 4            | 1700   | 305          | 29                                   |
| 8            | 10692  | 959          | 59   (ksmooth 8)                     |
| 16           | 75140  | 2954         | 82   (ksmooth 16)                    |

Block-Jacobi grows by about a factor of three per refinement; this grows by about half that.
At the largest size measured it is a factor of thirty-six fewer iterations, and unlike the
rediscretised V-cycle it is stable run to run -- 89 and 102 on repeats, against 3084 and
1485 for the arm that was being read as signal earlier.

This is the first configuration in which the V-cycle does what it was built for.

### What it costs, and what is not yet shown

Wall time, L=16, two repeats: block-Jacobi 9.8 and 13.2 seconds, this 25.5 and 29.3. Thirty-six
times fewer iterations and still about twice the time, because each iteration now carries a
cycle whose levels each run sixteen preconditioned BiCGStab iterations, plus the per-Newton
assembly. Smoothing strength is near optimal at that setting: at L=16, ksmooth 10, 12, 16 and
24 give 77.1, 72.9, 23.5 and 27.1 seconds.

Two honest limits. First, smoothing strength has to grow with depth -- eight iterations
suffice at four levels and give 1194 iterations at five, where sixteen give 99. That the
cycle needs more smoothing as it deepens says the smoother is still the weak component, and
it eats into the iteration gain because the cost per cycle rises with it. Second, the
crossover in wall time was not demonstrated: the iteration counts diverge fast enough that
one should exist, but L=32 exceeded the time available here, so that remains a projection
rather than a measurement, and projections of exactly this kind have been wrong twice
already in this document.

## State of the code, and where the solver is matrix-free

### The shape of the method

The solver is matrix-free where it is large and matrix-based where it is small, and that
split is not a compromise -- each half was forced by a measurement.

**Matrix-free, and staying that way: everything on the fine level.** The CVFEM
Navier-Stokes operator over the semi-structured `sshex8` lattice is never assembled. The
residual, the Jacobian action, the 4x4 block diagonal, and the 2x2 (velocity, pressure)
block split are all element sweeps over macro-elements. The grid transfers are matrix-free
lattice operations from `smesh`. The fine-level smoother's block-Jacobi is built from
`hessian_block_diag`, which is `n_nodes x 16` values -- O(n) storage, not a matrix. This is
the reason the semi-structured hierarchy exists and none of it changed.

**Matrix-based, once per Newton step: the coarse levels.** Each coarse operator is an
assembled BSR matrix formed by Galerkin coarsening, `A_c = R A P`, with entries recovered by
probing under a distance-2 colouring. During the solve the coarse levels apply a sparse
matrix and never touch a finer level.

### Why the boundary sits there

Three measurements put it there, in order.

The rediscretised coarse operator -- the matrix-free choice, and the one the hierarchy was
built around -- does not work. Applying the two-level correction operator to a mode that is
exactly representable on the coarse grid leaves 5.52 of a velocity mode (it amplifies the
error) and 0.79 of a pressure mode, where the Galerkin operator leaves 0.000. The
disagreement is not a scaling: removing each component's own best-fit scale still leaves
0.68, 0.52, 0.52 and 0.40 in ux, uy, uz and p, which is why every scaling knob tried
(`PSCALE`, `CGC`, `RC_DECAY`) did nothing.

Galerkin cannot be applied matrix-free. Composing `R A P` at solve time works and is kept
as `SFEM_GMG_GALERKIN=1`, but it puts a fine-level application under every coarse
application, so cost stops falling geometrically with depth. That is the one thing a
hierarchy must not do.

Assembling also fixes a second problem that has nothing to do with cost. A coarse smoother
needs the diagonal of the operator it smooths; a matrix-free composite cannot supply one,
and substituting the rediscretised diagonal mismatches the Galerkin operator by the
per-block scale factors -- about 1.6 in velocity, 8 in pressure -- which by itself made the
coarse levels diverge. An assembled matrix hands over its own diagonal.

### What being matrix-based actually costs

The fine matrix is still never formed, and that is the whole point: the assembled hierarchy
is the coarse levels only.

| | blocks | memory |
|---|--------|--------|
| assembled coarse hierarchy (L=16, N=1) | 70531 | 8.6 MiB |
| the fine BSR, which is never formed | 507195 | 61.9 MiB |

The hierarchy costs about 14% of what assembling the fine level would, because each level in
3D is roughly eight times smaller than the one above it and the sum is dominated by the
first coarse level rather than the fine one.

Assembly costs 548 probe applications per Newton step at that size, of which only the 180 at
the finest transfer involve a fine-level operator application; the rest are sparse
applications on already-assembled coarse levels. Probing is what makes this affordable at
all -- a distance-2 colouring means one application reveals a whole set of blocks, so the
count scales with the stencil rather than with the number of coarse unknowns (41 colours and
164 applications for 425 nodes, against 425 column by column).

### The rest of the algorithmic configuration

The coarse levels are smoothed with BiCGStab rather than a stationary iteration, because the
Galerkin operators are denser and lack the diagonal dominance block-Jacobi needs -- under
block-Jacobi they diverge at every damping. That makes the cycle vary between applications,
so the outer solver is FGMRES rather than BiCGStab; the two changes are one change, and the
driver selects FGMRES automatically whenever the smoother is Krylov.

Recommended configuration as measured:

```
SFEM_GMG=1  SFEM_GMG_GALERKIN=2  SFEM_GMG_KSMOOTH=16  SFEM_GMG_OMEGA=0.35
```

### What is settled and what is not

Settled: the V-cycle reduces iterations by a factor of twenty-three to thirty-six against
block-Jacobi and, unlike every earlier configuration, does so reproducibly. On Grace at
L=16, 60 iterations against 1408, reaching the same solution.

Not settled: it is not yet faster. Same Grace run, 7.59 seconds against 4.23. Thirty-six
times fewer iterations and 1.8 times the wall clock, because each iteration carries sixteen
preconditioned BiCGStab iterations per level plus the per-Newton assembly. The gap is
narrower on Grace than on the M1 (1.8 against 2.4), which is the direction one would expect
from sparse coarse work vectorising better there, but a single pair of runs is not evidence
of a trend.

Also unsettled, and the reason the crossover has not been demonstrated: refine level 32 does
not exist. It aborts in `smesh` with "Invalid element setup for proteus hex: 32", so the
larger problem has to come from more macro-elements at a valid level rather than a deeper
lattice. That sweep (N = 2 and 3 at L = 16) is running.

### Instrumentation

All of it is behind `SFEM_GMG_CHECK`, and all of it exists because something got past its
absence.

- `=1` transfers and their adjointness, a constraint census per level, the block-split sum
  against the full operator, and coarse-operator consistency reported three ways: raw, the
  per-component best-fit scale, and the residual left after removing that scale.
- `=2` the cycle's own convergence rate standalone, plus the stalled residual split by
  component.
- `=3` the smoother alone, with no coarse space. This is the first thing to run when a cycle
  misbehaves, and running it long enough to see the asymptote is part of the check: a rate
  that is still moving when the measurement stops has not been measured.
- `=4` the two-level correction operator applied to a prescribed mode, rediscretised against
  Galerkin, on coarse-oscillatory or (with `SFEM_GMG_CGC_SMOOTH=1`) coarse-smooth modes.
- The Galerkin assembly gates itself against the composite it was probed from and widens its
  sparsity pattern until it agrees, because probing folds a non-zero lying outside the
  pattern into the wrong entry rather than dropping it.

## Where the time actually goes

Cost had been inferred from operator-application counts, which is a model rather than a
measurement: it assumes every application costs the same and ignores the sparse coarse work,
the transfers and the assembly. The driver now times each phase directly and prints a
breakdown (SFEM's own tracing needs `SMESH_ENABLE_TRACE` compiled into smesh, which the
installed one lacks). Note that `precond_total` contains the `smooth[*]` rows, so the
"accounted" total double counts it.

### The coarse levels were paying for threads they could not use

The first breakdown showed the coarse smoothers costing 22.5, 21.3 and 21.2 ms per call on
levels of 2673, 425 and 81 nodes -- flat, where work should fall roughly eightfold per
level. It is not arithmetic: 324 unknowns cannot take 21 ms. It is the cost of starting a
thread team for each vector operation on a level with nothing to distribute.

Per smoother application on the 81-node level, and the whole solve:

| threads | smooth[L3] | total wall |
|---------|------------|------------|
| 1       | 0.156 ms   | 14.87 s    |
| 4       | 4.64 ms    | 4.85 s     |
| 8       | 14.20 ms   | 13.00 s    |

Ninety times slower for having eight cores instead of one, and the whole solve is fastest at
four threads and slower at eight. Each level now runs with a thread count matched to its own
size rather than the machine's (`SFEM_GMG_DOFS_PER_THREAD`, default 20000). After that the
coarse levels decay as they should -- 5752, 778 and 161 us per call across the three -- the
thread-count penalty is gone, and L=16 N=1 goes from 23.5 s to 18.3 s.

This also disposes of the large-case results reported above. Those Grace runs used 72
threads, where this penalty is far worse than at eight, so the 12x to 80x slowdowns at N=2
and N=3 are not a property of the method and those numbers should not be read as one. They
need re-running.

### What remains is fine-level smoothing, and it is the whole story

With the coarse levels fixed, the breakdown at L=16, N=1, eight threads is:

| phase | seconds | share |
|-------|---------|-------|
| smooth[L0] (fine) | 10.36 | 42% |
| galerkin_assembly | 1.47 | 6% |
| all coarse levels together | 0.78 | 3% |
| transfers, coarse solve, everything else | <0.4 | <2% |

A fine operator application costs 2.43 ms. The baseline spends 959 iterations x 2 = 1918 of
them. The cycle spends 86 iterations x 64 -- two smoothing applications per cycle, sixteen
BiCGStab iterations each, two applications per iteration -- which is 5504.

That is the arithmetic of the whole problem, and it is not about the hierarchy at all. To
break even the cycle may spend at most about 22 fine applications per cycle and it spends
64; equivalently, iteration count would have to fall by 32x and it falls by 11x. The
assembly, the transfers and every coarse level together account for under 10% and are not
where the decision lies.

The lever is therefore the fine-level smoother: it has to become roughly three times cheaper
per cycle without giving back the iteration count. Sixteen BiCGStab iterations there is
strong smoothing bought at two operator applications each; the alternatives worth measuring
are fewer Krylov iterations, a stationary sweep at one application each, or a Chebyshev
smoother, which would need an eigenvalue estimate but costs one application per sweep.

### Correction: the breakdown percentages above were normalised wrongly

`precond_total` is a container -- it wraps the whole V-cycle, so the smoother, operator,
transfer and coarse-solve rows sit inside it. Summing every row double counts, and shares
taken against that sum understate everything. The table in the previous section put
fine-level smoothing at 42% on that basis, and there appeared to be half the runtime
missing. There is not. Against wall time, with containers excluded from the denominator,
top-level phases account for 99.6% of the run:

| phase | seconds | share of wall |
|-------|---------|---------------|
| the V-cycle (`precond_total`) | 12.782 | 87.7% |
| `galerkin_assembly` | 1.609 | 11.0% |
| outer Krylov operator applications | 0.115 | 0.8% |
| Newton residual, block diagonals | 0.012 | 0.1% |

and inside the V-cycle:

| phase | seconds | share of wall |
|-------|---------|---------------|
| `smooth[L0]`, the fine level | 11.495 | 78.9% |
| `smooth[L1]` | 0.732 | 5.0% |
| transfers | 0.172 | 1.2% |
| `op[L0]` | 0.164 | 1.1% |
| `smooth[L2]`, `smooth[L3]`, coarse solve | 0.124 | 0.9% |

So fine-level smoothing is 79% of the solve, not 42%, and the conclusion drawn from the
wrong normalisation is strengthened rather than changed: the fine smoother is the only thing
worth optimising, the assembly is a real but secondary 11%, and everything below the fine
level together is under 8%. The reporter now excludes containers from its denominator and
labels them, so the table cannot be read this way again.

## Why it was slower, and the configuration that is not

The V-cycle's fine-level smoother was itself BiCGStab preconditioned by block-Jacobi -- the
same solver the whole cycle is competing against. So the cycle was running the baseline
solver as a subroutine, sixteen iterations at a time, twice per cycle.

Counted in fine operator applications per outer iteration:

| | applications per outer iteration |
|---|---|
| baseline BiCGStab + block-Jacobi | 2 |
| V-cycle with `ksmooth` 16 on the fine level | 64 = (pre + post) x 16 iterations x 2 |

Thirty-two times the work per iteration, against an eleven-fold reduction in iterations. The
cycle needed the iteration count to fall by 32x to break even and it fell by 11x, which is
the entire explanation for a method that was measurably better per iteration and measurably
worse per second. Nothing about the hierarchy was involved.

The fix is to stop smoothing the fine level like a solver. Coarse levels keep sixteen
BiCGStab iterations, because the assembled Galerkin operators genuinely need them and are
cheap; the fine level takes two. That is `SFEM_GMG_KSMOOTH_FINE`, now defaulting to 2.

Three repeats each, L=16, N=1, eight threads, two Newton steps:

| arm | t_solve (s) | iterations |
|-----|-------------|------------|
| baseline BiCGStab + block-Jacobi | 6.98, 7.07, 4.64 | 1625, 1625, 1064 |
| V-cycle, fine 1 | 5.25, 7.09, 6.87 | 133, 179, 174 |
| **V-cycle, fine 2** | **3.25, 4.84, 2.99** | 65, 97, 60 |
| V-cycle, fine 4 | 8.09, 11.37, 6.01 | 112, 157, 83 |
| V-cycle, fine 16 | 27.12 | 66 |

All reach the same solution (4.29148e-03 against the baseline's 4.29289e-03). At two
fine iterations the cycle is about twice as fast as the baseline on median, having been four
times slower at sixteen. This is the first configuration in this document that is faster
rather than merely fewer-iterations, and it took the phase measurement to find, because the
whole cost was in one line of the breakdown.

The two fixes compound: clamping the coarse levels' thread count made coarse smoothing cheap
enough that spending on it is affordable, which is what makes a weak fine smoother viable.
Measured before the clamp, cheap fine smoothing looked worse, and that reading is what
delayed this by a round.

## The 2x speedup does not survive to a large problem

The configuration that beat the baseline at L=16, N=1 on a laptop fails at N=3 on Grace.
Measured at 1,853,572 unknowns, 72 threads, two Newton steps:

| arm | iterations | t_solve | u_linf |
|-----|-----------|---------|--------|
| baseline BiCGStab + block-Jacobi | 1702 | 12.42 s | 1.30e-03 |
| baseline, repeat | 1888 | 13.78 s | 5.80e-03 |
| V-cycle, fine smoothing 2 | 2000 (cap) | 382.6 s | 1.99e-01 |
| V-cycle, fine smoothing 16 | 518 | 201.7 s | 1.49e-03 |

Weak fine smoothing does not merely lose here, it fails: the linear solve exhausts its
iteration cap and returns an answer two orders of magnitude off. Strong fine smoothing
converges to the right answer and takes fifteen times as long as the baseline. There is no
setting at this size that is both correct and competitive.

The pattern across every measurement in this document is now consistent: the smoothing
strength this cycle needs grows with the problem, and the cost of that smoothing is what
sinks it. It needed 8 iterations per level at four levels and 16 at five; it works at 2 on
the fine level at 75k unknowns and fails at 2 at 1.85M. Each time the requirement rises the
per-cycle cost rises with it, and the iteration count does not fall fast enough to pay.

So the honest state is that the V-cycle is faster than block-Jacobi on one problem size on
one machine, and slower or wrong everywhere else that has been measured. The earlier
sections reporting the 2x win should be read with this one attached.

### Two implementation notes attached to the same runs

The thread clamp is off by default. Resizing the OpenMP team per operator application costs
more than it saves once the team is large: at 72 threads it made the coarse smoothers slower
than leaving them alone (33.3 ms against 14.0 ms per call on level 1), having made them
faster at eight threads. The underlying problem -- coarse levels cannot use 72 threads -- is
real and unsolved; the clamp moved the cost rather than removing it.

An attempt to fix that by giving small levels a hand-written serial apply was reverted. It
was slower (30.5 s against 3.0 s at L=16, N=1) and, more seriously, it changed the iteration
count from 60 to 220 consistently, which is the signature of a wrong operator rather than a
slow one -- most likely a block-layout mismatch, since the assembly gate only ever validated
the values against `h_bsr_spmv` and not against a second reader of the same array. Any
future serial path needs its own gate against the parallel one before it is trusted.

## There is no threading bug, but threading has been flattering the baseline

Running one thread against many, on a small case, settles several things at once.

**No race.** Every deterministic check is bit-identical at 1, 2, 4 and 8 threads: the
Galerkin assembly gates (2.3509e-16, 1.7047e-16, 1.3501e-16), the block-split norms and
their sum against the full operator, and the transfer adjointness. Where both solvers
converge they agree to seven digits (u_linf 7.167056e-03 against 7.167062e-03). The
operators, transfers and assembly are thread-independent.

**Single-threaded runs are exactly reproducible and multithreaded ones are not.** At L=16,
N=1 the V-cycle gives 102, 102, 102 iterations on one thread and 70, 56, 75 on eight; the
baseline gives 2000, 2000 and 1125, 1625. That is reduction order in the Krylov dot
products, not a defect, but it is worth knowing that every multithreaded iteration count in
this document carries roughly twenty percent of noise.

**And the noise helps the baseline, which invalidates the comparisons above.** On one thread
at L=16, N=1:

| arm | iterations | t_solve | u_linf | p_linf |
|-----|-----------|---------|--------|--------|
| baseline BiCGStab + block-Jacobi | 2000 (cap) | 18.98 s | 6.275e-03 | 1.605e-02 |
| V-cycle | 102 | 7.35 s | 4.291e-03 | 2.257e-03 |

The baseline does not converge single-threaded. It stagnates and exhausts its cap, and the
answer it returns is wrong -- 4.29e-03 is the value every converged run in this document
produces, and its pressure error is seven times worse. On eight threads the same solver
converges in 1125 to 1625 iterations. Rounding noise from threaded reductions is perturbing
a stagnating BiCGStab enough to break it out, which is a known behaviour and is pure luck.

So the baseline this method has been measured against was being helped by an accident of
parallel reduction order, and every multithreaded comparison reported above is a V-cycle
against a baseline that is quietly getting a free restart. Single-threaded, where both
solvers are deterministic and the comparison is honest, the V-cycle takes 102 iterations and
7.35 seconds and is correct, while the baseline takes 2000, nineteen seconds, and is wrong.

**What remains true is the parallel efficiency gap.** From one to eight threads, t_solve goes
19.08, 6.50, 6.79, 7.08 for the baseline and 7.44, 8.41, 3.08, 3.42 for the V-cycle. Neither
scales past two to four threads on this machine -- the problem is memory bound -- and much of
the baseline's apparent gain is its iteration count falling rather than its work speeding up.
The V-cycle scales worse in the sense that matters, because its coarse levels cannot use the
threads at all, and at 72 threads on Grace that is the difference between the two results.

This does not rescue the Grace numbers, and the large-case conclusion stands: at 1.85M
unknowns the V-cycle is still far slower there. But it does mean the laptop comparisons were
measuring the wrong thing, and that the correct single-threaded comparison at 75k unknowns
favours the V-cycle by more than the earlier multithreaded one suggested.

## With an assembled fine operator: the baseline's stagnation was an artefact

`SFEM_ASSEMBLE_FINE=1` replaces the matrix-free fine operator with a BSR one, probed by the
same coloured probing the Galerkin levels use (null transfers assemble A rather than R A P).
The motivation is determinism: a matrix-free apply accumulates through atomics, so its
summation order follows the thread schedule, while a BSR apply accumulates each row in one
thread.

It does what it should, and it corrects the previous section.

**The operator becomes thread-independent.** The assembly gate is identical at 1 and 8
threads (1.7650e-16, 2.3080e-16), and at L=8 the V-cycle takes exactly 36 iterations at both
thread counts where matrix-free gave 36 and 37.

**The solver does not.** At L=16 the assembled baseline gives 1862 and 2000 iterations on
two runs at 8 threads. Removing the operator's non-determinism leaves the Krylov method's
own: the dot products are OpenMP reductions and their order still follows the schedule. A
deterministic operator is necessary for a reproducible parallel solve and not sufficient.

**And the baseline's single-threaded stagnation was specific to the matrix-free operator.**
The previous section reported that the baseline fails to converge on one thread -- 2000
iterations, capped, u_linf 6.28e-03 against the correct 4.29e-03 -- and concluded that the
V-cycle wins by 2.6x once the comparison is made deterministically. With the assembled
operator the baseline converges on one thread in 1873 iterations to u_linf 4.292937e-03. The
stagnation was an accident of the matrix-free operator's rounding, not a property of
BiCGStab on this problem, and the conclusion drawn from it is withdrawn.

The honest comparison at L=16, N=1, with a deterministic operator on both sides:

| arm | threads | iterations | t_solve | u_linf |
|-----|---------|-----------|---------|--------|
| baseline | 1 | 1873 | 5.05 s | 4.292937e-03 |
| V-cycle | 1 | 112 | 7.78 s | 4.291494e-03 |
| baseline | 8 | 1210 | 3.48 s | 4.223545e-03 |
| V-cycle | 8 | 86 | 4.55 s | 4.291483e-03 |

The V-cycle uses seventeen times fewer iterations and is about 1.4x slower, at both thread
counts, and the ranking no longer depends on how many threads are used or on which operator
the baseline happens to get. That is the first comparison in this document that is stable
under both, and it says the V-cycle is not yet competitive at this size -- by a much smaller
margin than the multithreaded matrix-free numbers suggested, and in the opposite direction
from the single-threaded ones.

Assembling the fine operator costs 1.85 s single-threaded and 0.90 s on eight, which is
already counted in the timings above.

## The packed layout, brought to the semi-structured path

The flat HEX8 path won on CPU with a packed mesh: each pack writes its exclusively owned
nodes straight out, stages the shared ones, and a second pass gathers each shared node's
contributions in a fixed order. The semi-structured path never had it. `initialize()`
returns early for a semi-structured mesh, before the packing block, so those kernels
accumulated locally within a macro-element and then scattered every node with `atomic_add`.

That is where the irreproducibility came from. Atomic ordering follows thread timing and
floating-point addition is not associative, so with 27 macro-elements on 8 threads the same
operator application gave 0.080346978588455187 and 0.080346695382904176, while one thread
gave the same value every time. The earlier check that appeared to show determinism was run
at N=1, where there is a single macro-element and the atomics never contend -- a test that
could not have failed.

`SFEM_SS_SCATTER=1` (default) applies the packed layout's structure. It is simpler here
because the split is geometric: every lattice node strictly inside a macro-element belongs to
it alone, and only the skin is shared -- (L+1)^3 - (L-1)^3 nodes, 37% at L=4 falling to 12%
at L=16. Interior nodes are written directly, skin nodes are staged, and a second pass sums
each shared node's contributions in element order. It covers the Jacobian action and the
nodal pressure gradient, which had its own set of atomics and fed the apply, so fixing only
the first left the operator non-reproducible.

Measured at N=3, L=4, 8 threads, with the pressure-gradient cache off so both kernels are
exercised:

| | repeat-diff within a run | two runs, same thread count |
|---|---|---|
| atomic scatter | 5.551e-17 | 0.080346978588455187, 0.080346695382904176 |
| two-pass scatter | 0.000e+00 | 0.080346967363645994, 0.080346967363645994 |

The operator is now bit-reproducible run to run at a fixed thread count, and it is faster:
at N=3, L=8 the solve takes 8.79 to 9.73 s against 9.80 to 11.26 s, about 11%.

Two things it does not do. Results still differ between one thread and eight, but that
difference is in the initial state (206.55554994212025 against 206.55554942713621), which is
a setup-phase reduction and not these kernels. And the full solve is still not reproducible
run to run -- 1284, 1228, 1355 iterations -- because the Krylov method's dot products are
OpenMP reductions whose order still follows the schedule. That is the same conclusion the
assembled-operator experiment reached from the other direction: a deterministic operator is
necessary for a reproducible parallel solve and not sufficient. Closing it needs a
fixed-order reduction in the BLAS layer.

Still atomic, and worth the same treatment: the residual, the block diagonal, and the
2x2 block-split kernels, none of which are on the path measured above.

### The remaining kernels

The residual, the block diagonal and the 2x2 block-split applies now use the same two-pass
scatter. The helpers are templated on the number of values per node, since the block diagonal
carries sixteen rather than four, and the tables are shared across all of them.

One thing changed rather than being preserved. The block-split scatter previously wrote only
the components its block selection touches -- "a continuity-row block touches one component
of four, and the atomics are the expensive half of the scatter". That saving existed because
atomics were expensive; with a plain write it is worth nothing, and the untouched components
are zero in the element buffer anyway, so the two-pass path moves all four.

Determinism, N=3 L=4 on 8 threads, repeat-diff within a run:

| kernel | atomic | two-pass |
|--------|--------|----------|
| residual | 8.674e-19 | 0.000e+00 |
| block diagonal | 5.551e-17 | 0.000e+00 |
| block split | 1.735e-18 | 0.000e+00 |

And they are faster, in line with the Jacobian action's 11%: the residual goes from 3918 to
3528 us per call and the block diagonal from 7524 to 6952, about 10% and 8%.

Correctness is unchanged -- `cvfem_ns_op_gate` passes with the scatter on and off, at the
same tolerances (grad 1.5e-15, bsr 1.2e-16, apply 4.3e-16, blockdiag 5.9e-16, diag 2.4e-17,
asm_vs_mf 1.7e-14). `cvfem_sshex8_bench` reports three disagreeing configurations both with
and without these changes, and with the kernel edits stashed, so that failure is pre-existing
and not caused here. It is worth chasing separately.

What is left is not in these kernels. With the scatter on, the residual output tracks the
input state exactly: two runs sharing a state checksum of 206.55554942713621 both give
-0.6316666704417313, and the run that produced 206.55554979906918 is the only one that
differs. The state itself still varies between runs at 8 threads and is stable at one, so the
remaining non-determinism is in the setup that builds the initial field, not in the operator.
That and the Krylov reductions are what stand between this and a reproducible parallel solve.

## Measures for the remaining non-determinism

Four candidate sources were checked, not assumed. Two are clean, one is fixed, and two need
work outside this spike.

| source | status | evidence |
|--------|--------|----------|
| element scatter (operator) | **fixed** | repeat-diff 0.000e+00 across all five kernels |
| grid transfers | clean | R and P checksums bit-identical over three runs |
| mesh coordinates | **broken** | x varies at 8 threads, stable at 1; y and z exact everywhere |
| Krylov reductions | broken by construction | `#pragma omp parallel for reduction(+ : ret)` |

### 1. Mesh coordinates, in `smesh::to_semistructured`

The x coordinate checksum varies between runs at 8 threads (16562.000025525689,
16562.000030174851) and is stable at one (16562.000023022294), while y and z are identical
everywhere. Only x, which fits shared lattice nodes being written by more than one
macro-element: y and z land on exactly representable values so the order cannot matter, and
x does not. Everything downstream inherits it -- the analytic pressure seed varies by 20%
in its (near-cancelling) checksum, so the initial state differs before any solver runs.

The measure is the one already applied to the element scatter: give each shared lattice node
a single canonical writer. Better still, compute a node's coordinate as a pure function of
the macro corners and its lattice index, evaluated once per node rather than once per
incident macro-element, which removes the question rather than ordering it. This is in
`smesh`, not here.

### 2. Krylov reductions, in `algebra/openmp/sfem_openmp_blas.hpp`

`dot` and `norm2` use `reduction(+ : ret)`, which combines partial sums in an unspecified
order over chunks whose boundaries follow the thread count. `SFEM_DETERMINISTIC_BLAS=1` now
selects a fixed 256-chunk decomposition, independent of the thread count, summed serially
per chunk and combined in index order. Off by default, since it changes results in the last
bits.

It is **unverified in this build**: the flag never fires, because the spike links a prebuilt
SFEM whose instantiation of these templates comes from the library rather than from the
edited header. Confirming it needs SFEM itself rebuilt. The code is written and gated; the
measurement is owed.

### 3. What is already done

The two-pass scatter, on by default (`SFEM_SS_SCATTER=1`), covering the Jacobian action, the
nodal pressure gradient, the residual, the block diagonal and the block split.

### 4. Measurement discipline, independent of the above

Until the two remaining sources are closed, comparisons should pin the thread count and
report medians of repeats rather than single runs, and prefer quantities measured as rates
over iteration counts. `SFEM_ASSEMBLE_FINE=1` gives a deterministic operator for A/B work
where the memory is affordable. It would be worth adding a determinism check to the test
suite -- one checksum at one thread against the same at N -- so that a regression in any of
this is caught rather than rediscovered.

## Reproducibility, verified end to end

With the smesh coordinate patch built in, all three measures are live and the chain is
closed. Everything below is measured, not projected.

**Mesh coordinates.** N=3 on 8 threads, three runs: 16562.000023022294 every time with
`SMESH_DETERMINISTIC_COORDS=1`, against 16562.000030174851, 16562.000029459596,
16562.000029459596 with it off. The deterministic value is exactly what a single-threaded
run produced before the patch, which is what the ownership rule promised -- the lowest
element was already winning there.

**Operator.** Bit-identical at 1, 2, 4 and 8 threads: the Jacobian action checksums
0.08034669538290462 at every one, as do the residual (-0.63166661236032529) and the initial
state (206.55554994212321). Thread-count independent, not merely run-to-run stable.

**Solve.** N=3, L=8, across thread counts:

| | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| `SFEM_DETERMINISTIC_BLAS=0` | 2000 (cap) | 2000 (cap) | 1688 | 1223 |
| `SFEM_DETERMINISTIC_BLAS=1` | 1030 | 1030 | 1030 | 1030 |

Identical iteration counts and identical u_linf (2.982674e-03) at every thread count. The
~20% iteration noise that made every comparison in this document unreliable is gone.

### One thing worth noticing

The deterministic run is not merely reproducible, it is better: 1030 iterations against 1223
at best and two outright failures to converge. That is not luck. The fixed 256-chunk sum is
partially pairwise, so it is *more accurate* than the serial accumulation a single thread
performs -- which is why the deterministic single-threaded run converges where the
non-deterministic single-threaded run stagnates at its cap. Determinism here costs nothing
and buys accuracy.

It also explains a result reported earlier in this document and never satisfactorily
accounted for: the baseline that "failed to converge single-threaded and succeeded on eight".
That was never about thread count. It was a solver sitting close enough to stagnation that
the accumulated error in a long serial sum decided the outcome, and the thread count only
changed how that sum was grouped.

### The three measures

| measure | where | default |
|---------|-------|---------|
| `SMESH_DETERMINISTIC_COORDS` | smesh, `sshex8_fill_points*` | on |
| `SFEM_SS_SCATTER` | this spike, five sshex8 kernels | on |
| `SFEM_DETERMINISTIC_BLAS` | `algebra/openmp/sfem_openmp_blas.hpp` | on |

All three are on by default, each opting out with `=0`.

The reduction was switched on after measuring what it costs, which is nothing: per-iteration
time went from 8509 to 7672 microseconds at N=3 and 1880 to 1811 at N=1. The `reduction`
clause privatises and combines per thread; a flat chunk array summed once is cheaper. So it
is reproducible, more accurate and faster, and there was no trade to weigh.

The chunk count is a function of the length alone -- serial below 4096 elements, and above
that growing so a chunk stays near 8192. Depending only on the length is what keeps the
result identical across thread counts. A fixed 256 was wrong at both ends: it spawned 256
chunks over almost no work for short vectors, which is the same thread-team overhead that
made small multigrid levels slower on more cores, and for long ones it left each chunk a
serial sum whose error grew with the problem.

Verified with nothing set, N=3, L=8:

| threads | 1 | 2 | 4 | 8 |
|---------|---|---|---|---|
| iterations | 1178 | 1178 | 1178 | 1178 |
| u_linf | 2.985426e-03 | 2.985426e-03 | 2.985426e-03 | 2.985426e-03 |

`cvfem_ns_op_gate` passes.

One practical note, learned three times over in this document: the spike compiles the
*installed* SFEM headers, not the ones in this tree. Editing `algebra/` here changes nothing
until `build64` is rebuilt and installed, and the symptom is a measurement that silently
matches the old behaviour. Check a changed default against its own opt-out before believing
it took effect.

## The GMG comparison, rerun with everything deterministic

Every earlier comparison in this document carried about twenty percent of iteration-count
noise and should be read as indicative at best. With the mesh, the operator and the
reductions all deterministic, iteration counts now repeat exactly -- 526/526, 40/40,
1834/1834, 104/104 on repeated runs -- so these numbers mean what they say.

Two Newton steps, 8 threads, baseline is block-Jacobi + BiCGStab, V-cycle is assembled
Galerkin with `SFEM_GMG_KSMOOTH=16` and the default fine smoothing of 2.

| case | dofs | baseline | V-cycle | verdict |
|------|------|----------|---------|---------|
| N=1, L=8  |  10,692 |  526 its, 0.76 s |  40 its, 0.37 s | **2.1x faster** |
| N=1, L=16 |  75,140 | 1834 its, 7.51 s | 104 its, 4.96 s | **1.5x faster** |
| N=2, L=8  |  75,140 | 1061 its, 2.89 s | 230 its, 10.35 s | 3.6x slower |
| N=3, L=8  | 242,500 | 1178 its, 6.99 s | 430 its, 51.6 s | 7.4x slower |
| N=2, L=16 | 561,924 | 1369 its, 25.7 s | 2000 capped, wrong answer | fails |

Tuning recovers a good deal of that -- at N=3, L=8, three levels with the coarse solve capped
at 30 iterations and `KSMOOTH=8` gives 120 iterations in 8.5 s against 430 in 51.6 s, and at
N=2, L=16 it turns a wrong answer into 307 iterations and a correct one in 48.2 s -- but in
neither case does it overtake the baseline.

### The result is about lattice depth, not problem size

The third and second rows are the same problem size, 75,140 unknowns, decomposed differently:
one macro-element with a level-16 lattice against eight macro-elements with a level-8 one.
The V-cycle is 1.5x faster on the first and 3.6x slower on the second. Size is not the
variable; how much of the mesh is lattice rather than macro-elements is.

The reason is structural. `create_gmg_data` derefines the lattice and stops at the macro
mesh, which it never coarsens. At N=1 the coarsest level is 20 nodes and free. At N=3 it is
208 nodes whose Galerkin operator needs the dense fallback -- 43,264 blocks, 832 probe
applications -- and the coarse solve alone is 21% of the run at 26.7 ms per application,
with the two smoothed levels above it taking another 59%. The hierarchy runs out of levels
while the problem is still big.

That also explains why the baseline moves in the opposite direction: it takes 1834 iterations
on the deep lattice and 1061 on the shallow one at equal size. A deep lattice is the hard
case for a pointwise preconditioner, and it is exactly where the V-cycle pays off.

So the honest summary is narrower and better supported than any earlier one here: **the
V-cycle wins where the mesh is mostly lattice, and loses where it is mostly macro-elements**,
and it wins by more the deeper the lattice. For the semi-structured meshes this work exists
to exploit -- few macro-elements, deep lattices -- that is the favourable regime.

The next step is not more tuning. It is to keep coarsening below the macro mesh, so the
hierarchy does not terminate on a problem that is still large and dense; that is a change in
`create_gmg_data` rather than in this spike.

## Why it capped, and what fixing that revealed

The N=2, L=16 failure above was not a weak preconditioner. The coarsest level's solve was
diverging, and the cycle was faithfully prolonging the result.

`SFEM_GMG_COARSE_VERBOSE=1` on that configuration shows BiCGStab on the 81-node coarsest
operator taking its residual from 1.577 to 36066 over a hundred iterations, from 17.26 to
82130, from 0.914 to 273. It then exhausts its 200-iteration cap and returns that amplified
vector as the coarse-grid correction. The V-cycle amplified by 1e6 to 1e9 per cycle, FGMRES
could not converge against such a preconditioner, the linear solve hit its own cap, and
Newton stepped from a badly solved system to an answer three orders of magnitude wrong.
Depth confirms the localisation: cycle rates are 0.06, 0.20 and 0.07 for two, three and four
levels, and 2.8e6 at five.

The tuning that appeared to rescue it was not addressing the cause. `MAX_LEVELS=3` removed
the offending level and `COARSE_MAX_IT=30` limited how far the divergence could run.

### The fix, and what it says about everything above

A few hundred unknowns should not be handed to an iterative solver at all. The coarsest
level is already stored dense, so it is now factorised: `DenseLU`, recovered by applying the
operator to unit vectors, exact and incapable of diverging. `SFEM_GMG_DENSE_LU_BELOW`
(default 4096 unknowns) selects it.

With that in place, every case improves and the V-cycle wins everywhere:

| case | dofs | baseline | V-cycle | speedup |
|------|------|----------|---------|---------|
| N=1, L=8  |  10,692 |  526 its, 0.72 s | 36 its, 0.32 s | 2.2x |
| N=1, L=16 |  75,140 | 1834 its, 6.88 s | 80 its, 3.76 s | 1.8x |
| N=2, L=8  |  75,140 | 1061 its, 2.69 s | 55 its, 1.79 s | 1.5x |
| N=3, L=8  | 242,500 | 1178 its, 6.03 s | 71 its, 5.71 s | 1.06x |
| N=2, L=16 | 561,924 | 1369 its, 25.7 s | 84 its, 15.8 s | 1.6x |

Against the previous section, N=2 L=8 goes from 230 iterations and 10.35 s to 55 and 1.79 s,
N=3 L=8 from 430 and 51.6 s to 71 and 5.71 s, and N=2 L=16 from a wrong answer to the
fastest arm in the table.

**This retracts the conclusion of the previous section.** That section read the N=1 wins and
the N>1 losses as evidence that the method depends on lattice depth, and blamed the
hierarchy terminating at the macro mesh. That was wrong. The variable was the coarsest
level's solve, which fails harder at larger N because the coarsest operator is bigger and
worse conditioned there; the correlation with macro-element count was real and the causal
story attached to it was not.

The iteration counts now say something the noisy measurements never could: 36, 80, 55, 71
and 84 across 10,692 to 561,924 unknowns. Fifty times the problem for a bit over twice the
iterations is close to the level independence a multigrid method is supposed to deliver, and
the baseline over the same range goes from 526 to 1369.

`cvfem_ns_op_gate` passes.

### On the coarsening work that was proposed next

It was proposed on the strength of the retracted conclusion, so its justification is gone
rather than merely weakened. Coarsening below the macro mesh may still be worth doing -- at
N=3 the margin is only 1.06x, and a deeper hierarchy is the obvious way to widen it -- but it
should be argued from measurements taken with the coarse solve working, not from those above.

## Phase 0: the gate says build Phase 1 and drop Phase 2

Before building coarsening below the macro mesh, two measurements were taken to check that a
deeper hierarchy would help at all. Neither supports it.

### M1 — assembly is now the dominant cost

With the coarse solve fixed by the dense LU, the probing assembly is the largest single term
in the cycle: 32.7% of the run at N=3 L=8 (3.43 s of 10.49 s, 571 ms per call) and 67.4% at
N=1 L=8. The coarse solve it used to hide behind is now 0.2%.

### M2 — more levels is worse, not better

The macro mesh is a cube refined to level L, so the same fine discretisation is reachable at
several macro/lattice splits. At 64x16x16 (75,140 unknowns), varying only the split -- which
is exactly what a macro-mesh coarsening would do, executed by hand:

| macro / L | levels | iterations | t_solve | t_prec | total |
|-----------|--------|-----------|---------|--------|-------|
| 16x4x4 / L=4  | 3 | 51 | 1.66 | 4.88 | 6.55 |
| 8x2x2 / L=8   | 4 | 54 | 1.83 | 1.60 | **3.43** |
| 4x1x1 / L=16  | 5 | 80 | 3.83 | 1.44 | 5.27 |
| baseline      | - | 1834 | 7.20 | 0.01 | 7.20 |

Five levels is worse than three on iterations (80 against 51) and worse on total time. Adding
levels below the macro mesh would extend the hierarchy in exactly the direction that measures
worse. **The Phase 2 gate fails; the AMG should not be built for this problem.**

A measurement error is worth recording, because it briefly pointed the other way. `t_solve`
covers only the Krylov solve; `refresh_gmg`, and therefore the whole assembly, is counted in
`t_precond`. Comparing arms on `t_solve` alone credited the shallow-lattice arm with 1.54 s
while ignoring its 4.88 s of assembly. Only the total is meaningful when the arms have
different level counts.

### What the same numbers say about Phase 1

The two effects run against each other. Iterations improve as the lattice gets shallower and
the macro mesh finer (51, 54, 80), because the coarse levels are then better resolved. But
the probing assembly gets sharply worse in that direction (4.88, 1.60, 1.44 s), because the
coarsest level has more nodes and the pattern guess falls back to dense -- 425 nodes and 1700
probe applications in the best-iteration arm, against 20 nodes and 80 in the worst.

So the configuration that converges best is the one probing punishes hardest. Removing the
probing does not merely save its own 30-70%; it unlocks the split that wins on iterations.
That is the case for Phase 1, and it is stronger than the one the plan was written on.

### Correction: the Phase 2 gate was run in the wrong regime

The conclusion above -- that coarsening below the macro mesh is not worth building -- is
withdrawn. It was drawn from experiments whose coarsest level was 20 to 425 nodes, where the
exact coarse solve is free and extra levels can therefore only add cost. The gate could not
have returned anything else.

The cost that coarsening removes is the *terminal* problem's size, and a dense factorisation
is O(n^3) in time and O(n^2) in memory:

| macro nodes | dofs | LU storage | factor flops | |
|---|---|---|---|---|
| 425 | 1,700 | 22 MiB | 1.6e9 | free |
| 1,024 | 4,096 | 128 MiB | 2.3e10 | marginal (per Newton step) |
| 4,096 | 16,384 | 2.0 GiB | 1.5e12 | impossible |
| 65,536 | 262,144 | 512 GiB | 6.0e15 | impossible |

`SFEM_GMG_DENSE_LU_BELOW` is 4096 dofs, i.e. 1024 macro nodes. Beyond it the driver falls
back to the BiCGStab that was measured to diverge on this operator. So today there is a hard
ceiling at about a thousand macro elements, above which there is neither a working exact
coarse solve nor any way to make the coarse problem smaller.

Extending the M2 ladder one rung into that regime, same fine mesh of 64x16x16 throughout:

| macro / L | coarsest | assembly | t_precond | t_solve | total |
|-----------|----------|----------|-----------|---------|-------|
| 8x2x2 / L=8   | 81 nodes | 6,561 blocks, 324 applications | 1.60 | 1.83 | 3.43 |
| 16x4x4 / L=4  | 425 nodes | 180,625 blocks, 1,700 applications | 4.88 | 1.66 | 6.55 |
| **32x8x8 / L=2** | **2,673 nodes** | **7,144,929 blocks (915 MB), 10,692 applications** | **144.4** | **386.1** | **~530** |
| baseline | - | - | 0.01 | 7.20 | 7.20 |

Seventy-four times slower than the baseline on the same discretisation. Both failure modes
appear together: the probing assembly goes fully dense at 7.1 million blocks, and the coarse
solve drops to a Krylov method that cannot be trusted on this operator.

This is the regime that matters for a real macro mesh, and it needs both phases rather than
one:

- **Phase 1** removes the dense-pattern probing, which is what produced the 915 MB operator
  and the 10,692 applications per Newton step. A sparse triple product yields the exact
  pattern, which for a 2,673-node level is a normal sparse matrix rather than a dense one.
- **Phase 2 is reinstated**, but its purpose is not the one the plan gave it. It is not there
  to add levels for faster convergence -- M2 correctly showed that does not help when the
  coarse problem is already small. It is there to **bound the size of the terminal problem**,
  so the exact coarse solve stays affordable as the macro mesh grows. That is invisible below
  the factorisation knee and decisive above it.

The gate should be re-run above the knee once Phase 1 lands, since Phase 1 changes the
assembly cost that currently dominates this measurement.

## Can the coarsest level be assembled directly instead of probed?

`hessian_bsr` refuses on the semi-structured path but works on an unstructured level, so the
coarsest level could be assembled outright and the last remaining probe dropped. That trades
a Galerkin operator for a rediscretised one, and the proposal was that projecting the
velocity and pressure properly -- an L2 projection rather than the partition-of-unity average
used now -- would make the rediscretised operator good enough.

The gap narrows as the coarse mesh resolves, which is the right trend:

| coarsest | raw gap (ux) | best-fit scale | after-scale (ux) | after-scale (p) |
|----------|--------------|----------------|------------------|-----------------|
| 81 nodes   | 13.86 | -0.007 | 0.995 | 0.870 |
| 425 nodes  |  3.49 |  0.074 | 0.964 | 0.634 |
| 2673 nodes |  1.42 |  0.336 | 0.805 | 0.628 |

At 81 nodes the two operators are nearly orthogonal, which is no surprise: a rediscretised
Navier-Stokes operator on 81 nodes is a different problem, not a coarse version of the same
one. By 2673 nodes the raw gap has fallen tenfold and the correlation has risen to a third.

### The state is not what separates them

`SFEM_GMG_CONST_STATE=1` gives every level the same constant field. Averaging and an L2
projection reproduce a constant identically, so the two operators are then evaluated at
genuinely the same state and the state is eliminated as a variable:

| coarsest | after-scale ux (real state -> constant) | after-scale p (real -> constant) |
|----------|------------------------------------------|----------------------------------|
| 81 nodes   | 0.995 -> 0.955 | 0.870 -> 0.870 |
| 425 nodes  | 0.964 -> 0.602 | 0.634 -> 0.634 |
| 2673 nodes | 0.805 -> 0.735 | 0.628 -> 0.628 |

**The pressure figures do not move at all.** They are identical to four digits, which is what
they must be if the cause is the discretisation: Rhie-Chow's `Df = rc h^2 / (2 mu)`, the
pressure Laplacian and the divergence block are all state-free, and only momentum convection
depends on the state. That the numbers are bit-identical is also a check that the diagnostic
is measuring what it claims.

Velocity does improve -- 0.964 to 0.602 at 425 nodes -- so part of that gap really is the
state, and a better projection would recover it. But even with a perfect, exactly
representable state the velocity operators still differ by 60 to 90 percent after optimal
per-component rescaling.

So the answer is no, on this evidence. An L2 projection is worth having on its own merits,
since it would sharpen the coarse operator's convective coefficients, but it cannot make
direct assembly a substitute for Galerkin here: most of the disagreement, and all of the
pressure disagreement, is in the discretisation rather than in the state. The probe at the
first coarse level stays, and its cost is now 108 operator applications rather than 10,692.

## Element-wise Galerkin: assembling the coarse operator inside the macro element

The last probe survives only because the fine operator has no matrix form. It does not need
one. The fine operator is a sum of macro-element contributions and the prolongation's support
is local -- a fine node interpolates only from coarse nodes of the sub-cell containing it, and
a face node gets the same contributors from either side -- so

    R A P  =  sum_e  P_e^T A_e P_e

element by element, with no global matrix ever formed and nothing reaching outside a macro
element.

**The enabling property is verified, not assumed.** Element-wise Galerkin is exact only if A
really is a sum of element operators, and Rhie-Chow couples through a *nodal* pressure
gradient, which is not element-local -- unless it is frozen from the state rather than
recomputed from the direction. Putting a direction on one macro element's interior and
measuring the response outside it gives exactly zero (max |out| inside 4.2480e-02, outside
0.0000e+00 at L=8; likewise at L=4). The frozen gradient is what makes this work, and it is
worth knowing that changing Rhie-Chow to differentiate the pressure gradient would silently
invalidate the construction.

### Cost

Per macro element, in element-kernel evaluations:

| L (hop) | fine nodes | coarse/elem | global probe | local probe | local matrix |
|---------|-----------|-------------|--------------|-------------|--------------|
| 2  |   27 |   8 | 108 |   32 | ~1-4 |
| 4  |  125 |  27 | 176 |  108 | ~1-4 |
| 8  |  729 | 125 | 176 |  500 | ~1-4 |
| 16 | 4913 | 729 | 192 | 2916 | ~1-4 |

Two variants. *Local probing* -- applying the element kernel to each local coarse basis
function -- only beats global probing for shallow lattices, and is three times worse at L=8.
*Local matrix* -- assembling the macro element's own sparse operator once on its lattice
stencil and then doing a small triple product -- wins everywhere, and turns assembly from
108-192 global operator applications into something comparable to a single one.

The local matrix is transient, one per element or per thread: 729 blocks and 0.09 MiB at
L=2, 19,683 and 2.40 MiB at L=8, 132,651 and 16.19 MiB at L=16.

### What it would need

Only one new kernel: a macro-element-local assembly for the semi-structured CVFEM operator.
The per-micro-cell Jacobian entries already exist -- the unstructured path assembles them
into a global BSR -- so the work is scattering them into a lattice-local structure instead of
into the global matrix. Everything else is in place: the structured prolongation gives P_e
directly, and the deterministic two-pass scatter already built for the operator kernels is
exactly what accumulates the local contributions into the coarse BSR.

It also removes the last of the machinery this section has been dismantling: no probing, no
colouring, no sparsity pattern to derive or guess, and no global fine matrix.

### Element matrices all the way down, BSR only at the coarsest level

Element-locality does more than remove the probe. If the coarse operator is built inside the
macro element, it can also be *kept* there: each coarse level becomes a set of Galerkin
element matrices applied with GEMM, and only the coarsest level needs a global sparse matrix,
because that is the only level that is factorised.

This is SFEM's existing idiom rather than a new one --
`frontend/ops/sfem_SemiStructuredEMLinearElasticity.hpp` assembles the element matrix on the
fly and applies it with GEMM -- and the pieces are already in this spike:
`subpar/cvfem_sshex8_em.hpp`, the `em24`/`em32` bench columns, and
`packed_elements_matmul_sym` / `_nonsym` in `operators/packed_elements.hpp`.

| coarse lattice | nodes/elem | EM dofs | EM MiB/elem | assembled BSR, 32 elems |
|----------------|-----------|---------|-------------|-------------------------|
| level 4 | 125 | 500 | 1.91 | 8.81 |
| level 2 |  27 | 108 | 0.09 | 1.40 |
| level 1 |   8 |  32 | 0.01 | 0.27 |

The last hop is a 32x32 element matrix, which is exactly the `em32` shape the bench already
measures.

**Correction, from building it.** The table above compares a *dense* element matrix against
the BSR, and on that basis the storage runs the other way at the shallow end -- level 4 costs
about seven times the assembled BSR. The kernel that was actually written does not store a
dense element matrix. A coarse node couples only to its 3x3x3 lattice neighbourhood, so the
local operator is a 27-point stencil of `(Lc+1)^3 * 27` blocks, and the only excess over the
assembled BSR is duplication at shared macro-element faces:

| coarse lattice | dense EM vs BSR | 27-stencil EM vs BSR |
|----------------|-----------------|----------------------|
| level 2 | 1.0x | 3.38x |
| level 4 | 4.6x | 1.95x |
| level 8 | 27.0x | 1.42x |

So the storage objection to keeping intermediate levels as element matrices is much weaker
than the first estimate suggested, and it *improves* with lattice depth rather than worsening.
That reopens "BSR only at the coarsest" as a real option rather than something the numbers
argue against; what settles it is a measurement of the two applies, not the storage.

One property makes this cleaner than it first appears. The prolongation composed within a
macro element is itself a trilinear interpolation, so any level's element matrix can be
formed directly from the fine element operator as `(P_e^{L->l})^T A_e P_e^{L->l}` rather than
by chaining level-to-level products. Each coarse level is then independent of the others: no
error accumulates through repeated Galerkin products, and a level can be rebuilt without
touching its neighbours.

The resulting architecture drops nearly everything this section has been repairing:

- fine level: matrix-free, unchanged
- intermediate coarse levels: Galerkin element matrices, GEMM apply, assembled on the fly
  where storage warrants it
- coarsest level only: assembled BSR, for the dense LU that must stay exact

No probing, no colouring, no sparsity pattern derived or guessed, no SpGEMM, and no global
sparse matrix above the coarsest level -- so the unsorted-column hazard, the `mm` workspace
sizing and the host-only serial transpose all stop applying. The coarse block diagonals the
smoothers need come from summing element contributions, which the deterministic two-pass
scatter already does.

### Element-wise Galerkin, implemented

`SFEM_GMG_EGAL=1` (default) builds the level-1 coarse operator as `sum_e P_e^T A_e P_e`
straight from the fine macro-elements. That is where the probe was -- level 1 is the only
level whose operator above it is matrix-free and has no matrix form; below it the level above
IS a matrix and `rap` is already exact and cheap. It does not turn Galerkin coarsening on:
that is still `SFEM_GMG_GALERKIN=2`, off by default, so a run setting neither still gets
rediscretised coarse operators.

**What made it cheap.** Three structural facts, none of which needed new physics:

1. *The micro-cell matrix was already reachable.* Passing identity slots (`sl[k] = k`) to
   `cvfem_hex8_ns_upwind_jacobian_add_slots` writes a dense 8x8-block cell matrix into a local
   buffer -- the trick `assemble_block_diag` already used. So the entries come out directly
   and nothing is probed.
2. *A micro-cell's coarse support is exactly eight nodes.* The cell spans fine indices
   `[xi, xi+1]`, whose coarse floors differ by at most one, so per axis it reaches coarse
   indices `{ax, ax+1}` and no more -- for any ratio `q`, not only 2:1. The triple product is
   a fixed 8x8 -> 8x8 contraction rather than something growing with `L`.
3. *The weights depend only on the offset class* `(xi%q, yi%q, zi%q)`. There are `q^3` of them,
   shared by every cell and macro-element, so they are a table built once and never an array
   indexed per entry -- the principle `cvfem_ss_transfer.hpp` applies to the prolongation.

**Cost.** Per micro-cell the two contraction stages are `8*27` and `27*8` block
multiply-accumulates, so 432 against the 1024 a dense 8x8 -> 8x8 contraction would need. The
27 is the average row count of the prolongation restricted to a cell: one corner interpolates
from 1 coarse node, three from 2, three from 4, one from 8, and `1+6+12+8 = 27`. Stage 1
contracts through a precomputed transpose of that map so each output block is stored once
rather than zeroed and accumulated into, saving 1024 scalars of zeroing per cell for the same
arithmetic. Assembly runs in chunks of macro-elements sized to keep the staging buffer near
32 MiB.

**The pattern is derived, not guessed.** An entry exists exactly where two coarse nodes share
a macro-element and sit within one lattice step of each other, which is the true Galerkin
pattern. That removes the probe's worst failure mode outright: an entry outside a too-narrow
guess was not dropped but folded into the wrong slot, so a bad guess gave a wrong matrix
rather than an approximate one, and the retry loop that widened it is what produced the
7,144,929-block dense coarse operator. The derived pattern is also tighter than the probe's --
112 blocks at the coarsest level where the probe padded to 144, which `rap` independently
confirms is the true count.

**Determinism** comes for free: accumulation runs over destinations rather than sources, each
block summing its own contributions in a fixed order with chunks in fixed order too, so there
are no atomics and the matrix is the same bits on any thread count. This is the two-pass
packed idea from the apply path, applied to assembly.

### Levels chain after all, once constraints are in play

The design note above claimed a level at ratio `q` could be built straight from the fine
element matrices, with no chaining, because piecewise-linear interpolation on nested uniform
lattices composes to the direct map. That is true of the **raw** operator, and the gates
measure it at 2.2e-16 and 4.4e-16 through the last hop.

It is false for the hierarchy the driver actually builds. Its transfers zero constrained
degrees of freedom at *every* hop, so the composite is `R2 Z1 (R1 Z0 A Z0 P1) Z1 P2` -- a `Z`
at each level, not only at the fine end. Level 2 built by chaining therefore coarsens a
level-1 matrix that already carries identity rows, and those rows contribute to the product.
A level built directly from level 0 cannot see them.

This did not show up in any operator comparison, because the operators agree to 2e-16 on
unconstrained columns. It showed up in the **block diagonal**, which differed by 1.2e-2 at
level 2 and 5.7e-2 at level 3 -- and the block diagonal is what the smoother inverts. Those
are not round-off; the smoother was being handed wrong diagonals at two of the three coarse
levels.

Masking the constrained coarse columns as well is *not* the fix; it makes level 1 disagree too
(4.4e-18 -> 7.3e-3), because the probe does not mask them. The fix is to leave the chaining
alone: element-wise Galerkin replaces the probe at level 1, and `rap` continues to build the
levels below from the level above, which is the already-validated path. With that, every gate
reads machine precision at every Newton step -- the level-1 operator at 4e-16 and its block
diagonal at 1.5e-17.

**The gate that missed it** was copied from the `rap` check, which zeroes constrained rows on
both sides with the comment that "those rows are not part of what is being tested". That is
right for testing a triple product and wrong for testing a replacement construction: it makes
the comparison blind to precisely the rows where two constraint treatments differ. Comparing
the block diagonals, which no operator gate covers, is what located it.

### Gates

On a 2x1x1 macro mesh at L=8 with `SFEM_GMG_CHECK=1 SFEM_GMG_GALERKIN=2`:

| gate | what it settles | result |
|------|-----------------|--------|
| `egal identity (q=1)` | the assembly reproduces `A` itself -- cell matrix, geometry, Rhie-Chow, pattern, scatter, at once | 1.540e-16 |
| `egal galerkin (0->1)` | the coarsening reproduces `P^T A P` against the matrix-free composite | 2.199e-16 |
| `egal galerkin (0->2,3)` | direct equals chained for the raw operator | 2.210e-16, 4.444e-16 |
| `egal level 1` | the constrained operator equals the probed composite it replaces | 2.1e-16 |
| `egal diag 1` | the block diagonal the smoother inverts equals the probed one | 4.4e-18 |
| `rap level 2,3` | the levels below, unchanged | 1.8e-16, 1.8e-16 |

The identity gate is the one worth keeping. At `q = 1` the prolongation is the identity, so
`P^T A P` is `A`, and one comparison covers everything the construction rests on: that
identity slots really do yield the micro-cell matrix, that the hoisted geometry and Rhie-Chow
struct fed to the assembly are the ones the apply uses, that the derived pattern holds every
entry, and that the inverted-index accumulation lands each block where it belongs.

### The whole hierarchy element-wise: the coarse constraints already existed

The section above concluded that extending the element-wise construction below level 1 was
blocked by the constraint treatment. It was not. `create_gmg_data` derefines the `Function` at
every level (`f_prev->derefine(fs_next, true)`, via `DirichletConditions::derefine`), so each
level already carries its own constraints, and the transfers apply them as `R = Z_coarse Rhat`
and `P = Z_fine Phat`. The composite at a hop is therefore

    Z_i Rhat A_{i-1} Z_{i-1} Phat

-- mask the source level's columns with that level's own mask, contract with the plain
interpolation, patch identity rows at the target. That is the recipe the `rap` branch was
already using (`mask_block_columns`, `rap`, `patch_identity_rows`), one level up.

So the hierarchy is built by chaining, but the chaining happens *inside the macro-element*:
`galerkin_hop` coarsens a level's 27-point stencil element matrices to the next level's,
masking between hops, with no global matrix at any point. Because the mask acts on the matrix
rather than on the transfer, the interpolation stays scalar -- no per-component prolongation is
needed even though the constraints are per component. The coarse stencil stays 27-point for any
ratio: a source node and its 27-neighbour land on coarse nodes at most one step apart, since
reaching two would need their coarse floors to differ while the upper one is off-lattice, and a
differing floor forces it to be on-lattice.

With that, every level matches the construction it replaces, block diagonals included:

| level | operator | block diagonal, before -> after |
|-------|----------|--------------------------------|
| 1 | 2.1e-16 | 4.4e-18 -> 4.4e-18 |
| 2 | 1.2e-17 | **1.2e-2 -> 4.6e-18** |
| 3 | 7.1e-17 | **5.7e-2 -> 0 (exact)** |

Nothing is probed at any level, and no triple product touches a global sparse matrix.

**A latent bug this exposed.** Moving the assembly ahead of the level loop segfaulted, because
it read the operator's cached state fields -- whatever the last `apply()`, `gradient()` or
`update()` happened to leave there. It had been correct only by accident of call order, which
is the kind of dependency that is invisible while it holds. `assemble_hierarchy` now takes the
fine state explicitly and calls `update()` itself.

### Keeping levels as element matrices

### BSR only at the coarsest level

The element matrices can also be kept and applied directly rather than assembled: gather a
macro-element's coarse nodes, run the 27-point stencil over its local matrix, reduce.
`SFEM_GMG_EGAL_EM=1` selects that, and `SFEM_GMG_CHECK=1` compares it against the assembled
operator it replaces, same construction and same constraint treatment.

The storage objection is weaker than the design note above estimated, because that estimate
was for a *dense* element matrix and the kernel builds a stencil one. Both storage and block
multiplies exceed the assembled form only by duplication at shared macro-element faces --
3.38x at a level-2 coarse lattice, 1.95x at level 4, 1.42x at level 8 -- improving as the
lattice deepens rather than worsening. What is bought is contiguous 4x4 blocks with no column
indirection.

It is off by default for a structural reason rather than a performance one. Level 1 is the
only level element-wise Galerkin builds, and the levels below it are built by `rap` *from*
level 1's matrix. Leaving level 1 as element matrices removes that matrix, so level 2 falls
back to probing -- trading one probe for another, further down. Making this pay needs the
whole hierarchy built element-wise, which is what the constraint chaining above currently
blocks: the transfers zero constrained dofs at each hop, and an element-wise level has no way
to see the identity rows the level above it carries.

So the honest position is that the apply is written and correct, and the path to "BSR only at
the coarsest" runs through the constraint treatment, not through the assembly kernel. The
options are to stop zeroing at intermediate levels, or to carry the identity rows down the
element-wise construction explicitly. Neither is measured yet.

### Measured

`SFEM_GMG_GALERKIN=2`, N=2x1x1 at L=8, 8 threads, same binary for both arms:

| | probe | element-wise |
|---|-------|--------------|
| `galerkin_assembly`, total | 9.286 s | **0.181 s** |
| per call | 49.9 ms | **0.78 ms** |
| share of measured phases | 4.0% | 0.1% |

So the assembly itself is about **64x cheaper per call**. What that is worth end to end is
smaller than it sounds: the symbolic-pattern fix earlier in this section had already removed
the catastrophic part of probing, leaving assembly at 4% of the solve at this size. The case
for the element-wise construction is therefore mostly the other three properties -- an exact
derived pattern that cannot be too narrow, a tighter one than the probe's guess, and bitwise
reproducibility -- with the speed following at larger coarse levels, where probing costs
colours and this costs one sweep.

That first table is a latency and not a throughput: N=2x1x1 is two macro-elements and the
assembly parallelises over macro-elements, so six of eight threads sit idle. Filling them, and
giving the coarse levels enough nodes to matter, changes the picture substantially:

| macro mesh, L=4 | probe, per call | element-wise, per call | speedup | probe share of measured phases |
|-----------------|-----------------|------------------------|---------|-------------------------------|
| 2x2x2, 8 elements  | 66.6 ms | **0.56 ms** | 120x | 45.5% -> 0.2% |
| 4x2x2, 16 elements | 72.0 ms | **0.79 ms** |  92x | 45.6% -> 0.8% |

Both of those are still on 8 threads with 8 and 16 macro-elements -- 2,916 dofs -- so they are
latencies on an under-filled machine. At saturation on Grace -- 108 macro-elements at L=8,
**242,500 dofs**, 72 cores -- the assembly total falls from 1.114 s to 0.030 s, a factor of 37,
and its share of the solve from 2.9% to 0.1%, with the linear iteration count unchanged at 1040. The per-call figures are not comparable between the two paths any more,
since the element-wise path times one hierarchy build where the probe times one level.

At these configurations probing is **45% of the measured phase time** and the element-wise
construction takes it under 1%. That is the scaling the two costs predict: probing pays a
colouring plus an operator application per colour per component, so it grows with the coarse
level; this pays one sweep over the macro-elements whatever the level looks like. The 4% share
seen at N=2x1x1 L=8 was the small end of that, not the typical case.

Parallelising inside a macro-element would race on its local matrix, so the answer if the
per-element latency ever matters is more macro-elements, not a different loop.

**The Newton step count is not the right measurement, and chasing it wasted a round.** The
element arm first appeared not to converge at N=2 -- 40 Newton steps against the probe's 27 --
which looked like a defect every gate had missed. Three measurements settle it, in increasing
order of sharpness.

First, the missing control: the probe does not converge at N=1 L=8 either (40 Newton steps,
31269 linear iterations, 84.9 s), where the element-wise arm is slightly *better* (30361
iterations, 71.1 s). A difference whose sign flips with problem size is not evidence about
either construction.

Second, raising the Newton cap shows both converge, at 27 steps against 62, and shows why the
count is meaningless here: both plateau at an absolute residual of 2 to 4e-10 while the 1e-8
*relative* tolerance sits inside that plateau, so the step at which a run first dips under the
threshold is decided by noise rather than by the preconditioner.

Third, and decisive, `SFEM_GMG_CHECK=2` measures the cycle itself rather than the Newton loop
around it. Both constructions take **36 linear iterations** on the first Newton step and land
on residuals agreeing to eight digits (1.4453e-10 against 1.2294e-10). The preconditioners are
equivalent, which is what the operator and block-diagonal gates already said and what the
Newton count was never going to show.

The lesson for this section's own gates: when a change is meant to be exactly equivalent,
measure the object that is supposed to be equivalent -- here the cycle -- not an outer loop
whose stopping test sits in its own noise. The block-diagonal defect above was real and is
fixed on its own evidence (1.5e-17 now against 1.2e-2 before); attributing the Newton counts to
it, as this note first did, was a conclusion drawn from the wrong instrument.


### Converged solve, element-wise against the probe

Full Newton to convergence on Grace: 108 macro-elements at L=4 -- 8,281 nodes, **33,124 dofs**
-- on 72 cores. This is the first end-to-end comparison in a regime that actually converges;
every earlier one either stagnated or was a fixed-work run stopped after two Newton steps.

| | probe | element-wise |
|---|---|---|
| converged | yes | yes |
| Newton steps | 27 | 32 |
| linear iterations | 29340 | 32100 |
| us per linear iteration | 23694 | 24239 |
| `galerkin_assembly` | 5.008 s (0.7%) | **0.103 s (0.0%)** |
| `t_solve` | 695.19 s | 778.07 s |
| `u_linf` | 2.553635e-07 | 2.554991e-07 |
| `p_linf` | 1.174712e-07 | 1.175410e-07 |

Both reach the same solution, agreeing to four digits in both velocity and pressure. Assembly
is 49x cheaper, consistent with the 37x measured at 242,500 dofs. The cost per linear iteration
is unchanged within 2.3%, which is what the gates and the fixed-work run predicted: the
preconditioners are equivalent, so the apply costs the same.

The Newton counts differ, 32 against 27, making this solve 12% slower and swamping the 4.9 s
the assembly saved. That is most likely trajectory divergence rather than a worse
preconditioner -- the fixed-work run at 242,500 dofs gave *identical* iteration counts (1040
for both), the operators agree to 2e-16 and their block diagonals to 4e-18, so a preconditioner
difference would have shown there. Over 27-plus Newton steps at a tight tolerance two operators
differing at round-off diverge in trajectory, and the step where the convergence test trips
becomes somewhat arbitrary. But this is one run per arm and cannot establish that on its own;
repeats would be needed to call the 12% noise rather than a regression.

At 33,124 dofs the assembly is 0.7% of the solve, so 49x buys little end to end here. Its value
is at the configurations where probing reached 45% of measured phases, together with the exact
derived pattern and the bitwise reproducibility.


### Reynolds robustness

Element-wise Galerkin, 108 macro-elements at L=4 -- 8,281 nodes, **33,124 dofs** -- on 72 Grace
cores, Poiseuille so the exact solution stays parabolic at every Re and `u_linf` remains a real
correctness check.

| Re | converged | Newton | linear its | per step | `u_linf` | verdict |
|------|-----|----|-------|------|--------------|---------|
| 100  | yes | 32 | 32100 | 1003 | 2.553635e-07 | correct |
| 200  | no  | 40 | 51550 | 1289 | 2.900010e-07 | **correct solution, Newton test not met** |
| 400  | no  | 40 | 39530 |  988 | 1.777071e+13 | diverged |
| 800  | no  | 40 | 28832 |  721 | 5.218090e+112 | diverged |
| 3200 | no  | 40 | 26514 |  663 | 2.637724e+87 | diverged |

**The breakdown is between Re=200 and Re=400**, and the two failures are of different kinds.
At Re=200 the solver reaches the right answer -- 2.90e-07 against the converged Re=100 run's
2.55e-07, both at discretisation accuracy -- and only fails the Newton residual test within 40
steps, at the highest linear cost of any run. At Re=400 and above the iteration diverges
outright, to 1e+13 and beyond.

**It is the continuation, not the preconditioner.** Every failing run reaches
`stage: navier-stokes`, so the Re=1 stage always succeeds and the blow-up is always on the
single jump to physical density. The linear solves stay healthy throughout: 663 to 1289
iterations per Newton step across the whole range, with no upward trend as Re rises -- if the
V-cycle were degrading under convection dominance that number would climb, and it does not. The
blow-up magnitudes are non-monotonic (1e+13, 1e+112, 1e+87), the signature of unbounded
divergence rather than graded degradation: the exponent is wherever the iteration happened to
be when it hit the Newton cap.

The driver ramps in exactly two stages, Re=1 then physical Re. That jump is a factor of 100 at
Re=100 and survives; at 400 it does not. The two-stage scheme exists because Newton from a zero
state diverged at Re=100, so the mechanism is already known to be load-bearing -- it is simply
under-resolved above its original design point, which was the only Re ever tested.

The obvious next step is geometric ramping over several stages, each starting from the previous
solution, rather than one bound. Re=200 additionally suggests the Newton cap and the residual
tolerance want revisiting: a run that reaches discretisation accuracy and still reports failure
is measuring the wrong thing.
