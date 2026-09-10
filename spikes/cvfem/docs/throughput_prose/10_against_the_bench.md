## Against the bench: what the missing factor was

perf/baseline_grace.csv records `jac_action_packed_sumfact` at 1933 MDOF/s, and
re-measured on one allocation with three repeats it is 1695 (spread 3%). The scope
above read 607-625 when this was first investigated, and reads 851-911 now that the
Rhie-Chow coefficient is hoisted. These are the same kernel, and the trace figure
already EXCLUDES the boundary pass and the nodal gradient -- they are sibling
scopes, listed separately in the tables below -- so the cascade in
docs/CVFEM_Kernels.md accounts for the difference.

The bench could not measure the configuration the solver runs: it refused
`--rhie-chow` with `--jac-action` on `--layout packed`, because the bench's packed
staging never carried the term. That staging now exists (`--rhie-chow` with
`--jac-action`, layout packed or store, geom affine), and it agrees with the atomic
reference to 7.9e-16 relative, so the two measurements can finally be put on the
same axis.

Grace, one node, 72 threads, OMP_PROC_BIND=true, --exclusive, 11,212,884 dof
(n=140), 20 repeats after 5 warmup:

| packed Jacobian action | MDOF/s | vs. the plain kernel |
|---|---:|---:|
| element kernel, no Rhie-Chow (what the baseline quotes) | 1912 | 1.00x |
| element kernel, with Rhie-Chow, coefficient computed in the face loop | 882 | 2.17x slower |
| element kernel, with Rhie-Chow, coefficient hoisted | 1460 | 1.31x slower |
| the whole matvec: hoisted kernel + the direction's nodal gradient | 602 | 3.18x slower |

Rhie-Chow appeared to cost the packed element kernel 2.17x, and most of that turned
out not to be the term at all. The mass-flux coefficient carries a
degenerate-geometry guard, and with that guard inside the twelve sub-control-surface
face loops GCC 13.3 refuses to vectorise them -- `-fopt-info-vec-all` says
"vectorization is not profitable" -- while without it all twelve vectorise. The
guard's arithmetic is not the cost: measured against builds differing only in that
function's body, its sqrt is worth 1.3% and its two divides 0.7%, and rewriting the
early returns branchlessly recovers only 9%. Its presence in the loop is the cost.

The coefficient is pure geometry, so it is now built once per element per Newton step
and read in the face loop -- the arrangement the semi-structured path has always used
(`SSMacroGeom::coeff`), which is why that path never paid this. The guard still runs,
at build time, so the value is unchanged: the packed action still agrees with the
atomic reference at 7.9e-16, exactly as before. **That leaves the true cost of
Rhie-Chow in the element kernel at 1.31x, not 2.17x.**

The hoist is applied to the Jacobian action and deliberately not to the residual. The
residual's face loop is smaller, its cost model lands on the other side of the same
threshold, and its checksum is bit-identical either way -- so its vectorisation never
changed and it would pay the gather for nothing. Measured, it did: 1614 -> 1545
MDOF/s. With the residual left alone it is 1614 -> 1605, inside the spread.

What now dominates a matvec is the other half of Rhie-Chow: the direction's nodal
gradient reconstruction, which cannot be hoisted out of a Krylov solve because the
direction changes every iteration. At 1460 MDOF/s for the kernel and 602 for the
matvec it is **59% of every matvec** -- the largest single item on the flat path, and
the next thing worth attacking.

The residual discrepancy is 871 (bench) against 607-625 (solver), a factor of 1.4
rather than 3.1, and the two are no longer in different configurations -- they run
on different meshes (a plain cube against the constrained channel) and at different
sizes. That is an ordinary gap; the factor of three was not.

Two further things this settles:

  * Scaling is not the problem. With Rhie-Chow the sweep goes 7.53 -> 463.8 MDOF/s
    from 1 to 72 threads (61.6x); without it, 30.8 -> 1881 (61.1x). The Rhie-Chow
    path parallelises just as well. The cost is arithmetic and a second sweep.
  * The thread binding was never the explanation. OMP_PROC_BIND=true and close
    measure 1711 and 1718 MDOF/s, equal within a 3% spread; spread is the slow one
    at 1490.

Quoting ~2000 MDOF/s for this kernel still means the version without Rhie-Chow,
and the solver does not run that version. The number to quote for what the solver
runs is 602 MDOF/s at 11.2M dof on 72 Grace cores.

The configuration without Rhie-Chow -- every row in perf/baseline_grace.csv -- is
untouched by all of this, checked in the same runs: 1910 -> 1912 MDOF/s for the
Jacobian action and 2642 -> 2640 for the residual, with bit-identical checksums.

Reproduce with `jobs/fused_rc.sbatch` (the cascade) and `jobs/rc_hoist.sbatch` (the
hoist, against the binary from the parent commit).

### What the remaining bench-to-solver gap is, and is not

The bench's packed Jacobian action reads 1482 MDOF/s at 4,121,204 dof and the solver's
own scope reads 851 at 4,343,300 -- 1.74x apart on the same kernel, the same pack size
(2048 elements) and near-identical element counts. Two candidates are now settled.

**It is not problem size.** The bench is flat: 1798 at 1.1M dof, 1458 at 4.1M, 1476 at
11.2M. Measuring at the solver's size changes nothing.

**It is not the Krylov working set.** That was the leading hypothesis -- the bench
replays one apply on one direction with everything warm, while the solver evaluates the
same kernel with its basis live and a fresh direction each iteration. `--live-vectors N`
makes the bench do exactly that: N vectors of the solution size, churned between applies
and taken in rotation as the direction, with only the apply on the clock. It makes no
difference at all.

| live vectors | working set | kernel_only MDOF/s |
|---:|---:|---:|
| 0 | -- | 1482 |
| 3 | 94 MiB | 1498 |
| 7 | 220 MiB | 1486 |
| 14 | 440 MiB | 1482 |

Flat from nothing to 440 MiB, well past this socket's 117 MiB of L3. The hypothesis was
wrong and the instrument built to test it says so cleanly.

**A quarter of it is pack locality.** The bench space-fills its element order by default
and the driver never does. Turning the bench's off reproduces a good part of the gap:

| | mean nodes per pack | kernel_only MDOF/s |
|---|---:|---:|
| bench, SFC ordered | 2735 | 1483 |
| bench, `--no-sfc` | 4373 | 1183 |
| solver, no SFC | -- | 851 |

A pack of 2048 elements holds 2735 nodes when the order is space-filling and 4373 when
it is lexicographic, and the sweep pays 1.25x for the difference. The solver's packed
mesh costs 1.34x the original storage (37,254 KB against 27,869) where the bench's costs
about 0.7x, which is the same signature read off memory rather than throughput -- and the
solver's domain is a 256x64x64 channel, so a contiguous run of 2048 elements there is a
far worse slab than the cube measured here.

So **space-filling the solver's element order is worth about 1.25x on the flat element
sweep**, which is more than either change made to the kernel itself today, and it is a
mesh-setup change rather than a kernel one. The residual 1.39x after that is not yet
attributed.

Reproduce with `jobs/live_vectors.sbatch`.

### Space-filling the driver's element order

The driver never ordered its mesh; the benchmark always has. Applying the same
reordering to the driver, both settings back to back in one allocation so only the order
differs, element sweep from the tracer:

| run | dof | SFEM_SFC=0 | SFEM_SFC=1 | |
|---|---:|---:|---:|---:|
| flat_N32 | 561,924 | 1033 | 1249 | 1.21x |
| flat_N48 | 1,853,572 | 976 | 1120 | 1.15x |
| flat_N64 | 4,343,300 | 921 | 1072 | 1.16x |
| ss_L2_N16 | 561,924 | 878 | 894 | 1.02x |
| ss_L2_N24 | 1,853,572 | 827 | 870 | 1.05x |
| ss_L2_N32 | 4,343,300 | 767 | 788 | 1.03x |

So about 1.17x on the flat element sweep, and 1.02-1.05x on the semi-structured one,
which is the smaller gain the macro-element layout would predict: a macro-element
already carries its own lattice, so a pack of them is far less sensitive to the order
they are numbered in.

Two things to be honest about. The gain is a little below the 1.25x the benchmark
measured for the same switch, and below what a 256x64x64 channel packing into slabs
would have suggested -- the expectation that the driver would gain MORE than the cube
was wrong. And within each pair the SFEM_SFC=0 run goes first, so any warm-up favouring
the second run biases these ratios up; the throughput A/B on unchanged code put that
effect at about 2%.

Against the benchmark, the flat sweep at 4.3M dof now reads 1072 where the bench reads
1482 at 4.1M, so the unattributed gap is 1.38x rather than the 1.74x it was.

Reproduce with `jobs/sfc_driver.sbatch`.

### perf: the remaining gap is not in the kernel

Hardware counters separate the two possibilities cleanly. If the solver executed more
instructions per element, the two would not be running the same code; if it executed the
same instructions in more cycles, the kernel would be stalling. Exact totals from `perf
stat` (counting mode) multiplied by per-symbol shares from `perf record`, at matched size
-- the solver 1,048,576 elements x 903 applies, the benchmark 1,000,000 x 20:

| element sweep | instructions / element | cycles / element | IPC |
|---|---:|---:|---:|
| benchmark | 1707 | 721 | 2.37 |
| solver | 1716 | 693 | 2.48 |

Neither. The instruction counts agree to 0.5%, so it is the same code doing the same
work, and the solver's kernel runs it in slightly FEWER cycles at slightly HIGHER IPC.
Per element of work the solver's sweep is not slower at all.

So the 1.38x is wall-clock inside the traced scope during which threads are not executing
kernel instructions. The OpenMP runtime's share is consistent with that and is the
leading candidate: libgomp is 10.4% of the solver's cycles against 5.5% of the
benchmark's, and 27% of its instructions against 18% -- barrier and spin-wait, which is
what idle threads at an unbalanced `schedule(static)` loop look like.

Two traps this went through, recorded because both produced confident wrong numbers:

  * `perf record` wrapped around `uenv run` profiles the launcher and loses the process
    at exec. The first attempt attributed 99.99% of its samples to
    ld-linux-aarch64.so.1. perf has to go INSIDE the uenv.
  * `cvfem_hex8_conv_all_jv_simd` is a separate symbol, not inlined into the sweep, and
    the solver additionally keeps `cvfem_hex8_ns_upwind_jacobian_action_simd` separate
    where the benchmark inlines it. Attributing only the enclosing function undercounts
    the sweep by 40% in one binary and 47% in the other, in opposite proportion.

The whole-run cache and stall counters are deliberately NOT quoted here. Both runs were
measured, but the solver's instruction mix is dominated by the Krylov solve rather than
the sweep, so per-instruction miss rates compare the composition of two different
programs and say nothing about the kernel.

What this does not yet establish is WHY the threads are idle -- imbalance across packs
against barrier cost per call. 512 packs over 72 threads is 7.1 each, so static
scheduling gives some threads 8 and others 7 before any variation in per-pack work is
counted. Running at a thread count that divides the pack count exactly would separate
the two.

Reproduce with `jobs/perf_sweep.sbatch` (per-symbol) and `jobs/perf_stat.sbatch` (totals).



