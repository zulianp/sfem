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
