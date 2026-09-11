# CVFEM solver throughput and bottlenecks

Where the time goes in the CVFEM Navier-Stokes solve, on both the flat HEX8 and
the semi-structured operator, measured by the scopes compiled into the solver
rather than by a sampling profiler -- so every row is a named routine and the
shares add up.

Throughput is `calls * dof / seconds`, in MDOF/s. For an element sweep that is the
rate the operator processes unknowns; for a setup routine called once it is not a
rate at all and is there only for scale.

## Summary

The element sweep that dominates the solve, in MDOF/s -- `calls * dof / seconds`
on the scope the Krylov iteration actually spends its time in.

| run | dof | sweep | MDOF/s | sweep | nodal grad | boundary | constraints |
|---|---|---|---|---|---|---|---|
| flat_N16 | 75,140 | apply_jacobian_action_packed | 163 | 70% | 18% | 5% | 7% |
| flat_N24 | 242,500 | apply_jacobian_action_packed | 530 | 72% | 13% | 6% | 6% |
| flat_N32 | 561,924 | apply_jacobian_action_packed | 1240 | 56% | 22% | 8% | 5% |
| flat_N48 | 1,853,572 | apply_jacobian_action_packed | 1114 | 72% | 14% | 8% | 2% |
| flat_N64 | 4,343,300 | apply_jacobian_action_packed | 1022 | 74% | 14% | 7% | 1% |
| ss_L2_N8 | 75,140 | apply_macro_local_hoisted | 674 | 37% | 23% | 0% | 14% |
| ss_L2_N12 | 242,500 | apply_macro_local_hoisted | 855 | 53% | 24% | 0% | 8% |
| ss_L2_N16 | 561,924 | apply_macro_local_hoisted | 890 | 68% | 25% | 0% | 5% |
| ss_L2_N24 | 1,853,572 | apply_macro_local_hoisted | 868 | 69% | 27% | 0% | 2% |
| ss_L2_N32 | 4,343,300 | apply_macro_local_hoisted | 794 | 67% | 30% | 0% | 1% |
| ss_L4_N4 | 75,140 | apply_macro_local_hoisted | 734 | 47% | 31% | 0% | 20% |
| ss_L4_N6 | 242,500 | apply_macro_local_hoisted | 932 | 53% | 24% | 0% | 9% |
| ss_L4_N8 | 561,924 | apply_macro_local_hoisted | 973 | 62% | 23% | 0% | 5% |
| ss_L4_N12 | 1,853,572 | apply_macro_local_hoisted | 941 | 70% | 24% | 0% | 2% |
| ss_L4_N16 | 4,343,300 | apply_macro_local_hoisted | 921 | 71% | 27% | 0% | 1% |

The boundary closure is a separate pass on the flat operator and is FUSED into
the macro-element sweep on the semi-structured one, where it is inside the micro
loop and cannot carry a scope of its own without paying for one per element. So
its column is blank for the semi-structured rows and its cost is inside theirs;
the two boundary shares are not comparable and the sweep shares are not either.


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

## The boundary closure was bound by its own load imbalance

The flat operator closes its boundary control volumes in a second sweep after the
element kernel, and that sweep walked the WHOLE mesh -- gathering coordinates, fields
and, for the Jacobian, the direction, 88 doubles per element -- in order to do work on
the shell alone. At N=96 the shell is about 9% of the elements, so 91% of the gathers
bought nothing.

The obvious fix is to test the face mask and skip. It bought nothing at all, and that
is the interesting part:

| packed Jacobian action, boundary scope | N=48, 1,853,572 dof | N=96, 14,489,860 dof |
|---|---:|---:|
| as it was | 1059 us/call | 7331 us/call |
| skipping elements with no boundary face | 1070 us/call | 6212 us/call |
| iterating a compacted list of them | **180 us/call** | **714 us/call** |
| | **5.9x** | **10.3x** |

The skip removes ~91% of the work and leaves the wall time where it was, because the
boundary elements are the shell of the mesh: under `schedule(static)` they fall into a
few threads' chunks, and the pass is bound by whichever thread owns them, not by the
total. Compacting the elements into a list first restores the balance -- every thread
gets an equal share of the elements that do something -- and only then does the work
reduction show up as time.

The share of a matvec spent closing the boundary goes from 23.5% to 5.0% at N=48 and
from 22.2% to 2.7% at N=96. The element sweep and the gradient reconstruction are
unchanged in the same runs (13.13 -> 13.26 s and 10.05 -> 10.23 s at N=96), so nothing
was moved around.

The skip is exact rather than an approximation. `fmask` is read in exactly one place in
each of the three boundary kernels, the per-face inclusion test, so an element with no
boundary face contributes nothing -- which tests/cvfem_boundary_mask_test.cpp asserts
directly ("fmask 0 contributes nothing"). Where no sideset mask exists, and that is the
default since one is only compiled under SFEM_BOUNDARY_MASK=1, the list is built from
the same bounding-box test the kernels would have run face by face.

The benchmark could not have found this: its own `--boundary` pass already skipped
zero-mask elements, so it had the work reduction and the imbalance together and showed
neither. The measurement had to come from the solver's trace.

Reproduce with `jobs/bnd_skip.sbatch`.

## Per configuration

### flat_N16

75,140 dof, 72 threads, nid006558. 0.620 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `cvfem_hex8_ns_steady::apply_jacobian_action_packed` | element sweep | 903 | 0.417 | 462.3 | 162.5 | 67.3% |
| `cvfem_hex8_ns_steady::nodal_grad_strided` | nodal gradient | 913 | 0.109 | 119.6 | 628.3 | 17.6% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.040 | 44.1 | 1702.9 | 6.4% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action` | boundary | 903 | 0.027 | 29.6 | 2540.0 | 4.3% |
| `cvfem_hex8_ns_steady::assemble_block_diag` | element sweep | 3 | 0.016 | 5218.7 | 14.4 | 2.5% |
| `cvfem_hex8_ns_steady::apply_residual_packed` | element sweep | 7 | 0.004 | 509.9 | 147.3 | 0.6% |
| `SFC::reorder` | other | 1 | 0.003 | 3122.6 | 24.1 | 0.5% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_residual` | boundary | 7 | 0.002 | 218.9 | 343.3 | 0.2% |
| `Mesh::initialize_node_to_node_graph` | setup | 1 | 0.001 | 783.0 | 96.0 | 0.1% |
| `create_crs_graph_mem_conservative` | setup | 1 | 0.001 | 735.8 | 102.1 | 0.1% |
| `cvfem_hex8_ns_steady::precompute_element_bsr_slots` | setup | 1 | 0.000 | 382.7 | 196.4 | 0.1% |
| `create_n2e` | other | 1 | 0.000 | 303.3 | 247.8 | 0.0% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 43.3 | 1735.7 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 42.7 | 1759.3 | 0.0% |
| `cvfem_hex8_ns_steady::build_rc_coeff` | other | 903 | 0.000 | 0.2 | 429893.7 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 89.4 | 840.4 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.437 | 70.4% |
| nodal gradient | 0.109 | 17.6% |
| constraints | 0.041 | 6.6% |
| boundary | 0.028 | 4.6% |
| other | 0.004 | 0.6% |
| setup | 0.002 | 0.3% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 0.920 | 306695.0 |
| `Function::apply` | 903 | 0.551 | 610.0 |
| `CVFEMNavierStokes::apply` | 903 | 0.510 | 565.2 |
| `cvfem_hex8_ns_steady::apply_jacobian_action_accumulate` | 903 | 0.500 | 553.4 |
| `Function::gradient` | 7 | 0.061 | 8675.4 |
| `CVFEMNavierStokes::gradient` | 7 | 0.060 | 8630.8 |
| `cvfem_hex8_ns_steady::apply_residual` | 7 | 0.060 | 8568.0 |
| `cvfem_hex8_ns_steady::assemble_nodal_p_grad` | 10 | 0.055 | 5498.9 |


### flat_N24

242,500 dof, 72 threads, nid006558. 0.603 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `cvfem_hex8_ns_steady::apply_jacobian_action_packed` | element sweep | 903 | 0.413 | 457.1 | 530.5 | 68.4% |
| `cvfem_hex8_ns_steady::nodal_grad_strided` | nodal gradient | 913 | 0.081 | 88.6 | 2736.5 | 13.4% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.038 | 42.3 | 5731.4 | 6.3% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action` | boundary | 903 | 0.034 | 38.1 | 6361.5 | 5.7% |
| `cvfem_hex8_ns_steady::assemble_block_diag` | element sweep | 3 | 0.017 | 5800.3 | 41.8 | 2.9% |
| `SFC::reorder` | other | 1 | 0.005 | 5034.0 | 48.2 | 0.8% |
| `cvfem_hex8_ns_steady::apply_residual_packed` | element sweep | 7 | 0.003 | 474.7 | 510.9 | 0.6% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_residual` | boundary | 7 | 0.003 | 456.2 | 531.6 | 0.5% |
| `Mesh::initialize_node_to_node_graph` | setup | 1 | 0.002 | 2154.1 | 112.6 | 0.4% |
| `create_crs_graph_mem_conservative` | setup | 1 | 0.002 | 2142.4 | 113.2 | 0.4% |
| `create_n2e` | other | 1 | 0.001 | 1223.8 | 198.2 | 0.2% |
| `cvfem_hex8_ns_steady::precompute_element_bsr_slots` | setup | 1 | 0.001 | 1030.2 | 235.4 | 0.2% |
| `cvfem_hex8_ns_steady::build_rc_coeff` | other | 903 | 0.000 | 0.5 | 528457.6 | 0.1% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 50.8 | 4772.0 | 0.1% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 40.8 | 5943.1 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 102.3 | 2370.9 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.434 | 71.9% |
| nodal gradient | 0.081 | 13.4% |
| constraints | 0.039 | 6.5% |
| boundary | 0.038 | 6.2% |
| other | 0.007 | 1.1% |
| setup | 0.005 | 0.9% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 1.458 | 485916.7 |
| `Function::apply` | 903 | 0.550 | 608.6 |
| `CVFEMNavierStokes::apply` | 903 | 0.511 | 565.6 |
| `cvfem_hex8_ns_steady::apply_jacobian_action_accumulate` | 903 | 0.499 | 553.0 |
| `cvfem_hex8_ns_steady::assemble_nodal_q_grad` | 903 | 0.051 | 56.2 |
| `Function::copy_constrained_dofs` | 903 | 0.038 | 42.6 |
| `Function::gradient` | 7 | 0.038 | 5377.4 |
| `CVFEMNavierStokes::gradient` | 7 | 0.037 | 5325.2 |


### flat_N32

561,924 dof, 72 threads, nid006558. 0.285 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `cvfem_hex8_ns_steady::apply_jacobian_action_packed` | element sweep | 326 | 0.148 | 453.1 | 1240.3 | 51.8% |
| `cvfem_hex8_ns_steady::nodal_grad_strided` | nodal gradient | 331 | 0.062 | 187.8 | 2992.8 | 21.8% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action` | boundary | 326 | 0.018 | 55.0 | 10208.4 | 6.3% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 326 | 0.015 | 44.7 | 12568.8 | 5.1% |
| `cvfem_hex8_ns_steady::assemble_block_diag` | element sweep | 2 | 0.010 | 5118.8 | 109.8 | 3.6% |
| `SFC::reorder` | other | 1 | 0.008 | 8441.4 | 66.6 | 3.0% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_residual` | boundary | 3 | 0.005 | 1817.5 | 309.2 | 1.9% |
| `Mesh::initialize_node_to_node_graph` | setup | 1 | 0.005 | 4884.7 | 115.0 | 1.7% |
| `create_crs_graph_mem_conservative` | setup | 1 | 0.005 | 4868.0 | 115.4 | 1.7% |
| `create_n2e` | other | 1 | 0.003 | 3160.7 | 177.8 | 1.1% |
| `cvfem_hex8_ns_steady::precompute_element_bsr_slots` | setup | 1 | 0.002 | 2244.9 | 250.3 | 0.8% |
| `cvfem_hex8_ns_steady::apply_residual_packed` | element sweep | 3 | 0.002 | 741.7 | 757.6 | 0.8% |
| `cvfem_hex8_ns_steady::build_rc_coeff` | other | 326 | 0.001 | 2.2 | 250437.8 | 0.3% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 182.6 | 3076.9 | 0.1% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 181.9 | 3089.0 | 0.1% |
| `DirichletConditions::gradient` | constraints | 3 | 0.000 | 46.8 | 12004.4 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.160 | 56.1% |
| nodal gradient | 0.062 | 21.8% |
| boundary | 0.023 | 8.2% |
| constraints | 0.015 | 5.4% |
| other | 0.012 | 4.3% |
| setup | 0.012 | 4.2% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 2 | 0.934 | 466771.0 |
| `Function::apply` | 326 | 0.210 | 642.8 |
| `CVFEMNavierStokes::apply` | 326 | 0.195 | 597.1 |
| `cvfem_hex8_ns_steady::apply_jacobian_action_accumulate` | 326 | 0.189 | 580.1 |
| `Function::gradient` | 3 | 0.048 | 15980.6 |
| `CVFEMNavierStokes::gradient` | 3 | 0.048 | 15931.0 |
| `cvfem_hex8_ns_steady::apply_residual` | 3 | 0.048 | 15841.0 |
| `cvfem_hex8_ns_steady::assemble_nodal_p_grad` | 5 | 0.040 | 8002.6 |


### flat_N48

1,853,572 dof, 72 threads, nid006558. 2.146 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `cvfem_hex8_ns_steady::apply_jacobian_action_packed` | element sweep | 903 | 1.502 | 1663.8 | 1114.1 | 70.0% |
| `cvfem_hex8_ns_steady::nodal_grad_strided` | nodal gradient | 917 | 0.293 | 319.3 | 5804.4 | 13.6% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action` | boundary | 903 | 0.163 | 180.1 | 10291.7 | 7.6% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.046 | 51.2 | 36205.6 | 2.2% |
| `cvfem_hex8_ns_steady::assemble_block_diag` | element sweep | 3 | 0.032 | 10518.9 | 176.2 | 1.5% |
| `SFC::reorder` | other | 1 | 0.023 | 22904.6 | 80.9 | 1.1% |
| `Mesh::initialize_node_to_node_graph` | setup | 1 | 0.018 | 17736.4 | 104.5 | 0.8% |
| `create_crs_graph_mem_conservative` | setup | 1 | 0.018 | 17705.9 | 104.7 | 0.8% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_residual` | boundary | 11 | 0.014 | 1246.1 | 1487.5 | 0.6% |
| `cvfem_hex8_ns_steady::apply_residual_packed` | element sweep | 11 | 0.013 | 1197.7 | 1547.6 | 0.6% |
| `create_n2e` | other | 1 | 0.013 | 12665.5 | 146.3 | 0.6% |
| `cvfem_hex8_ns_steady::precompute_element_bsr_slots` | setup | 1 | 0.008 | 7722.8 | 240.0 | 0.4% |
| `cvfem_hex8_ns_steady::build_rc_coeff` | other | 903 | 0.003 | 3.1 | 593484.8 | 0.1% |
| `DirichletConditions::gradient` | constraints | 11 | 0.001 | 56.2 | 32980.7 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 11 | 0.001 | 48.9 | 37890.5 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 419.4 | 4419.8 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 1.547 | 72.1% |
| nodal gradient | 0.293 | 13.6% |
| boundary | 0.176 | 8.2% |
| constraints | 0.048 | 2.3% |
| setup | 0.043 | 2.0% |
| other | 0.038 | 1.8% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 8.475 | 2825050.0 |
| `Function::apply` | 903 | 2.049 | 2269.4 |
| `CVFEMNavierStokes::apply` | 903 | 2.000 | 2214.6 |
| `cvfem_hex8_ns_steady::apply_jacobian_action_accumulate` | 903 | 1.928 | 2134.6 |
| `cvfem_hex8_ns_steady::assemble_nodal_q_grad` | 903 | 0.256 | 283.5 |
| `CVFEMNavierStokes::initialize` | 1 | 0.105 | 104988.0 |
| `Function::gradient` | 11 | 0.068 | 6186.6 |
| `CVFEMNavierStokes::gradient` | 11 | 0.067 | 6126.4 |


### flat_N64

4,343,300 dof, 72 threads, nid006558. 5.276 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `cvfem_hex8_ns_steady::apply_jacobian_action_packed` | element sweep | 903 | 3.836 | 4247.7 | 1022.5 | 72.7% |
| `cvfem_hex8_ns_steady::nodal_grad_strided` | nodal gradient | 913 | 0.734 | 804.4 | 5399.1 | 13.9% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action` | boundary | 903 | 0.337 | 372.9 | 11646.1 | 6.4% |
| `cvfem_hex8_ns_steady::assemble_block_diag` | element sweep | 3 | 0.069 | 23012.9 | 188.7 | 1.3% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.054 | 59.6 | 72897.6 | 1.0% |
| `SFC::reorder` | other | 1 | 0.053 | 52538.2 | 82.7 | 1.0% |
| `Mesh::initialize_node_to_node_graph` | setup | 1 | 0.044 | 44003.5 | 98.7 | 0.8% |
| `create_crs_graph_mem_conservative` | setup | 1 | 0.044 | 43966.1 | 98.8 | 0.8% |
| `create_n2e` | other | 1 | 0.034 | 33717.6 | 128.8 | 0.6% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_residual` | boundary | 7 | 0.024 | 3433.8 | 1264.9 | 0.5% |
| `cvfem_hex8_ns_steady::apply_residual_packed` | element sweep | 7 | 0.021 | 2953.9 | 1470.4 | 0.4% |
| `cvfem_hex8_ns_steady::precompute_element_bsr_slots` | setup | 1 | 0.018 | 18349.2 | 236.7 | 0.3% |
| `cvfem_hex8_ns_steady::build_rc_coeff` | other | 903 | 0.006 | 6.7 | 643737.0 | 0.1% |
| `Function::constraints_mask` | constraints | 1 | 0.001 | 769.4 | 5645.2 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.001 | 766.8 | 5664.5 | 0.0% |
| `DirichletConditions::gradient` | constraints | 7 | 0.001 | 85.1 | 51048.7 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 3.925 | 74.4% |
| nodal gradient | 0.734 | 13.9% |
| boundary | 0.361 | 6.8% |
| setup | 0.106 | 2.0% |
| other | 0.092 | 1.8% |
| constraints | 0.056 | 1.1% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 18.636 | 6211833.3 |
| `Function::apply` | 903 | 5.118 | 5668.3 |
| `CVFEMNavierStokes::apply` | 903 | 5.061 | 5604.4 |
| `cvfem_hex8_ns_steady::apply_jacobian_action_accumulate` | 903 | 4.876 | 5400.1 |
| `cvfem_hex8_ns_steady::assemble_nodal_q_grad` | 903 | 0.691 | 765.6 |
| `CVFEMNavierStokes::initialize` | 1 | 0.225 | 225290.0 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.103 | 34204.0 |
| `Function::gradient` | 7 | 0.094 | 13392.4 |


### ss_L2_N8

75,140 dof, 72 threads, nid006558. 0.277 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 0.101 | 111.4 | 674.4 | 36.3% |
| `to_semistructured` | other | 1 | 0.068 | 67883.5 | 1.1 | 24.5% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 913 | 0.065 | 71.3 | 1054.5 | 23.5% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.039 | 42.8 | 1756.9 | 13.9% |
| `SFC::reorder` | other | 1 | 0.002 | 2295.7 | 32.7 | 0.8% |
| `sscvfem::block_diag` | element sweep | 3 | 0.001 | 308.8 | 243.3 | 0.3% |
| `sscvfem::build_scatter` | setup | 1 | 0.001 | 715.7 | 105.0 | 0.3% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 43.3 | 1737.1 | 0.1% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 42.8 | 1753.7 | 0.1% |
| `create_dual_graph` | other | 1 | 0.000 | 255.3 | 294.3 | 0.1% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 95.8 | 784.0 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 26.6 | 2826.5 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 46.5 | 1616.2 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 45.3 | 1658.7 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.0 | 1962685.7 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.0 | 0.0 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.102 | 36.6% |
| other | 0.070 | 25.4% |
| nodal gradient | 0.065 | 23.5% |
| constraints | 0.039 | 14.2% |
| setup | 0.001 | 0.3% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 0.577 | 192222.7 |
| `Function::apply` | 903 | 0.215 | 238.0 |
| `CVFEMNavierStokes::apply` | 903 | 0.176 | 194.6 |
| `sscvfem::apply` | 903 | 0.166 | 183.4 |
| `sscvfem::nodal_q_grad` | 903 | 0.065 | 71.5 |
| `Function::copy_constrained_dofs` | 903 | 0.039 | 43.0 |
| `Function::gradient` | 7 | 0.002 | 278.6 |
| `CVFEMNavierStokes::gradient` | 7 | 0.002 | 234.5 |


### ss_L2_N12

242,500 dof, 72 threads, nid006558. 0.487 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 0.256 | 283.5 | 855.5 | 52.6% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 913 | 0.117 | 128.1 | 1893.3 | 24.0% |
| `to_semistructured` | other | 1 | 0.066 | 66166.9 | 3.7 | 13.6% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.039 | 43.2 | 5616.3 | 8.0% |
| `SFC::reorder` | other | 1 | 0.002 | 2417.1 | 100.3 | 0.5% |
| `sscvfem::block_diag` | element sweep | 3 | 0.002 | 759.0 | 319.5 | 0.5% |
| `sscvfem::build_scatter` | setup | 1 | 0.002 | 2070.2 | 117.1 | 0.4% |
| `create_dual_graph` | other | 1 | 0.001 | 850.2 | 285.2 | 0.2% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 42.7 | 5682.2 | 0.1% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 41.0 | 5913.5 | 0.1% |
| `create_n2e` | other | 2 | 0.000 | 80.9 | 2995.9 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 122.8 | 1975.0 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 103.2 | 2349.0 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 101.6 | 2387.6 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.0 | 6607609.5 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.0 | 7119818.5 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.258 | 53.1% |
| nodal gradient | 0.117 | 24.0% |
| other | 0.070 | 14.3% |
| constraints | 0.040 | 8.2% |
| setup | 0.002 | 0.4% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 1.329 | 443083.3 |
| `Function::apply` | 903 | 0.425 | 470.3 |
| `CVFEMNavierStokes::apply` | 903 | 0.385 | 426.5 |
| `sscvfem::apply` | 903 | 0.372 | 412.3 |
| `sscvfem::nodal_q_grad` | 903 | 0.116 | 128.3 |
| `Function::copy_constrained_dofs` | 903 | 0.039 | 43.4 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.004 | 1286.5 |
| `Function::gradient` | 7 | 0.004 | 547.2 |


### ss_L2_N16

561,924 dof, 72 threads, nid006558. 0.851 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 0.570 | 631.2 | 890.2 | 67.0% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 913 | 0.216 | 236.5 | 2375.8 | 25.4% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.039 | 43.5 | 12907.6 | 4.6% |
| `to_semistructured` | other | 1 | 0.009 | 9289.7 | 60.5 | 1.1% |
| `sscvfem::block_diag` | element sweep | 3 | 0.006 | 1849.9 | 303.8 | 0.7% |
| `sscvfem::build_scatter` | setup | 1 | 0.005 | 4911.2 | 114.4 | 0.6% |
| `SFC::reorder` | other | 1 | 0.003 | 2683.6 | 209.4 | 0.3% |
| `create_dual_graph` | other | 1 | 0.002 | 2120.0 | 265.1 | 0.2% |
| `create_n2e` | other | 2 | 0.000 | 195.7 | 2870.7 | 0.0% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 43.5 | 12909.4 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 40.4 | 13910.8 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 182.6 | 3076.9 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 180.7 | 3109.3 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 108.7 | 5168.6 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.0 | 14283629.6 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.0 | 0.0 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.576 | 67.6% |
| nodal gradient | 0.216 | 25.4% |
| constraints | 0.040 | 4.7% |
| other | 0.014 | 1.7% |
| setup | 0.005 | 0.6% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 2.802 | 933983.3 |
| `Function::apply` | 903 | 0.839 | 929.2 |
| `CVFEMNavierStokes::apply` | 903 | 0.799 | 885.0 |
| `sscvfem::apply` | 903 | 0.784 | 868.1 |
| `sscvfem::nodal_q_grad` | 903 | 0.213 | 236.0 |
| `Function::copy_constrained_dofs` | 903 | 0.040 | 43.8 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.009 | 3155.4 |
| `Function::gradient` | 7 | 0.008 | 1124.6 |


### ss_L2_N24

1,853,572 dof, 72 threads, nid006558. 2.831 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 1.929 | 2135.7 | 867.9 | 68.1% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 916 | 0.773 | 844.4 | 2195.1 | 27.3% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.048 | 53.1 | 34931.5 | 1.7% |
| `to_semistructured` | other | 1 | 0.026 | 26235.8 | 70.7 | 0.9% |
| `sscvfem::block_diag` | element sweep | 3 | 0.020 | 6560.9 | 282.5 | 0.7% |
| `sscvfem::build_scatter` | setup | 1 | 0.019 | 18735.6 | 98.9 | 0.7% |
| `create_dual_graph` | other | 1 | 0.008 | 7845.4 | 236.3 | 0.3% |
| `SFC::reorder` | other | 1 | 0.005 | 4745.0 | 390.6 | 0.2% |
| `create_n2e` | other | 2 | 0.002 | 916.2 | 2023.0 | 0.1% |
| `DirichletConditions::gradient` | constraints | 10 | 0.001 | 66.7 | 27805.6 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 10 | 0.000 | 43.9 | 42252.4 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 416.3 | 4452.7 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 413.4 | 4483.5 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 114.2 | 16230.6 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.1 | 26592135.9 | 0.0% |
| `sscvfem::apply_transient` | transient | 10 | 0.000 | 0.0 | 38872235.2 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 1.948 | 68.8% |
| nodal gradient | 0.773 | 27.3% |
| constraints | 0.050 | 1.8% |
| other | 0.041 | 1.4% |
| setup | 0.019 | 0.7% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 9.157 | 3052396.7 |
| `Function::apply` | 903 | 2.810 | 3111.5 |
| `CVFEMNavierStokes::apply` | 903 | 2.759 | 3055.9 |
| `sscvfem::apply` | 903 | 2.695 | 2983.9 |
| `sscvfem::nodal_q_grad` | 903 | 0.764 | 845.7 |
| `Function::copy_constrained_dofs` | 903 | 0.049 | 53.9 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.035 | 11682.5 |
| `Function::gradient` | 10 | 0.031 | 3118.8 |


### ss_L2_N32

4,343,300 dof, 72 threads, nid006558. 7.451 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 4.942 | 5473.0 | 793.6 | 66.3% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 919 | 2.198 | 2391.9 | 1815.9 | 29.5% |
| `to_semistructured` | other | 1 | 0.096 | 96220.5 | 45.1 | 1.3% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.087 | 95.8 | 45335.7 | 1.2% |
| `sscvfem::block_diag` | element sweep | 3 | 0.046 | 15445.2 | 281.2 | 0.6% |
| `sscvfem::build_scatter` | setup | 1 | 0.045 | 45487.2 | 95.5 | 0.6% |
| `create_dual_graph` | other | 1 | 0.019 | 19135.5 | 227.0 | 0.3% |
| `SFC::reorder` | other | 1 | 0.008 | 8268.8 | 525.3 | 0.1% |
| `create_n2e` | other | 2 | 0.005 | 2299.5 | 1888.8 | 0.1% |
| `DirichletConditions::gradient` | constraints | 13 | 0.002 | 125.1 | 34709.4 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.001 | 769.6 | 5643.5 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.001 | 766.5 | 5666.3 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 13 | 0.001 | 48.5 | 89637.6 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 141.1 | 30772.1 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.1 | 47134833.0 | 0.0% |
| `sscvfem::apply_transient` | transient | 13 | 0.000 | 0.2 | 18217129.5 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 4.988 | 67.0% |
| nodal gradient | 2.198 | 29.5% |
| other | 0.128 | 1.7% |
| constraints | 0.090 | 1.2% |
| setup | 0.045 | 0.6% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 20.931 | 6977066.7 |
| `Function::apply` | 903 | 7.378 | 8171.0 |
| `CVFEMNavierStokes::apply` | 903 | 7.289 | 8071.5 |
| `sscvfem::apply` | 903 | 7.111 | 7874.9 |
| `sscvfem::nodal_q_grad` | 903 | 2.164 | 2396.7 |
| `Function::gradient` | 13 | 0.100 | 7720.5 |
| `CVFEMNavierStokes::gradient` | 13 | 0.099 | 7591.6 |
| `Function::copy_constrained_dofs` | 903 | 0.088 | 96.9 |


### ss_L4_N4

75,140 dof, 72 threads, nid006558. 0.327 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 1505 | 0.154 | 102.4 | 733.7 | 47.1% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 1525 | 0.101 | 66.4 | 1132.3 | 30.9% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 1505 | 0.065 | 43.1 | 1742.3 | 19.8% |
| `SFC::reorder` | other | 1 | 0.002 | 2283.1 | 32.9 | 0.7% |
| `to_semistructured` | other | 1 | 0.002 | 1807.2 | 41.6 | 0.6% |
| `sscvfem::block_diag` | element sweep | 5 | 0.001 | 233.9 | 321.3 | 0.4% |
| `DirichletConditions::gradient` | constraints | 15 | 0.001 | 44.3 | 1695.0 | 0.2% |
| `DirichletConditions::apply_value` | constraints | 15 | 0.001 | 41.7 | 1800.2 | 0.2% |
| `sscvfem::build_scatter` | setup | 1 | 0.000 | 383.6 | 195.9 | 0.1% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 97.0 | 774.3 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 1505 | 0.000 | 0.0 | 2001332.6 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 46.5 | 1616.2 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 45.8 | 1641.5 | 0.0% |
| `create_dual_graph` | other | 1 | 0.000 | 32.4 | 2317.4 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 3.6 | 21010.7 | 0.0% |
| `sscvfem::apply_transient` | transient | 15 | 0.000 | 0.0 | 0.0 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.155 | 47.4% |
| nodal gradient | 0.101 | 30.9% |
| constraints | 0.066 | 20.3% |
| other | 0.004 | 1.3% |
| setup | 0.000 | 0.1% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 5 | 0.960 | 191919.8 |
| `Function::apply` | 1505 | 0.338 | 224.9 |
| `CVFEMNavierStokes::apply` | 1505 | 0.273 | 181.3 |
| `sscvfem::apply` | 1505 | 0.255 | 169.4 |
| `sscvfem::nodal_q_grad` | 1505 | 0.100 | 66.5 |
| `Function::copy_constrained_dofs` | 1505 | 0.065 | 43.3 |
| `Function::gradient` | 15 | 0.004 | 283.7 |
| `CVFEMNavierStokes::gradient` | 15 | 0.004 | 238.4 |


### ss_L4_N6

242,500 dof, 72 threads, nid006558. 0.446 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 0.235 | 260.2 | 931.9 | 52.7% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 913 | 0.105 | 115.4 | 2102.1 | 23.6% |
| `to_semistructured` | other | 1 | 0.059 | 59188.8 | 4.1 | 13.3% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.040 | 44.3 | 5479.7 | 9.0% |
| `sscvfem::block_diag` | element sweep | 3 | 0.002 | 716.1 | 338.6 | 0.5% |
| `SFC::reorder` | other | 1 | 0.002 | 2106.9 | 115.1 | 0.5% |
| `sscvfem::build_scatter` | setup | 1 | 0.001 | 989.9 | 245.0 | 0.2% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 44.5 | 5447.5 | 0.1% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 41.7 | 5821.6 | 0.1% |
| `create_dual_graph` | other | 1 | 0.000 | 107.3 | 2260.3 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 103.2 | 2349.0 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 101.8 | 2382.0 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 79.4 | 3054.4 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.0 | 6042492.2 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 10.5 | 23116.4 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.0 | 7119818.5 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.237 | 53.2% |
| nodal gradient | 0.105 | 23.6% |
| other | 0.061 | 13.8% |
| constraints | 0.041 | 9.2% |
| setup | 0.001 | 0.2% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 1.304 | 434740.0 |
| `Function::apply` | 903 | 0.393 | 435.6 |
| `CVFEMNavierStokes::apply` | 903 | 0.353 | 390.7 |
| `sscvfem::apply` | 903 | 0.340 | 376.2 |
| `sscvfem::nodal_q_grad` | 903 | 0.104 | 115.5 |
| `Function::copy_constrained_dofs` | 903 | 0.040 | 44.5 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.004 | 1249.6 |
| `Function::gradient` | 7 | 0.004 | 505.1 |


### ss_L4_N8

561,924 dof, 72 threads, nid006558. 0.845 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 0.522 | 577.6 | 972.9 | 61.8% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 913 | 0.195 | 213.6 | 2631.2 | 23.1% |
| `to_semistructured` | other | 1 | 0.077 | 77184.2 | 7.3 | 9.1% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.040 | 43.8 | 12833.5 | 4.7% |
| `sscvfem::block_diag` | element sweep | 3 | 0.005 | 1696.8 | 331.2 | 0.6% |
| `sscvfem::build_scatter` | setup | 1 | 0.003 | 2527.7 | 222.3 | 0.3% |
| `SFC::reorder` | other | 1 | 0.002 | 2229.0 | 252.1 | 0.3% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 44.0 | 12769.5 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 40.8 | 13759.9 | 0.0% |
| `create_dual_graph` | other | 1 | 0.000 | 256.3 | 2192.4 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 181.9 | 3089.0 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 180.0 | 3121.7 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 102.8 | 5468.4 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 25.4 | 22130.3 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.0 | 11823693.9 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.0 | 16498131.4 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.527 | 62.4% |
| nodal gradient | 0.195 | 23.1% |
| other | 0.080 | 9.4% |
| constraints | 0.041 | 4.8% |
| setup | 0.003 | 0.3% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 2.732 | 910626.7 |
| `Function::apply` | 903 | 0.770 | 852.9 |
| `CVFEMNavierStokes::apply` | 903 | 0.730 | 808.3 |
| `sscvfem::apply` | 903 | 0.715 | 791.3 |
| `sscvfem::nodal_q_grad` | 903 | 0.192 | 213.1 |
| `Function::copy_constrained_dofs` | 903 | 0.040 | 44.0 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.009 | 2884.6 |
| `Function::gradient` | 7 | 0.007 | 1007.5 |


### ss_L4_N12

1,853,572 dof, 72 threads, nid006558. 2.583 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 1.779 | 1970.0 | 940.9 | 68.9% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 913 | 0.630 | 689.5 | 2688.1 | 24.4% |
| `to_semistructured` | other | 1 | 0.096 | 96316.3 | 19.2 | 3.7% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.045 | 49.7 | 37282.8 | 1.7% |
| `sscvfem::block_diag` | element sweep | 3 | 0.018 | 6022.5 | 307.8 | 0.7% |
| `sscvfem::build_scatter` | setup | 1 | 0.010 | 9560.8 | 193.9 | 0.4% |
| `SFC::reorder` | other | 1 | 0.003 | 2576.8 | 719.3 | 0.1% |
| `create_dual_graph` | other | 1 | 0.001 | 867.1 | 2137.6 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 460.4 | 4026.1 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 458.0 | 4047.1 | 0.0% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 53.2 | 34840.6 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 42.0 | 44101.3 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 82.6 | 22437.1 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 119.9 | 15456.1 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.1 | 23093184.2 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.1 | 27210564.6 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 1.797 | 69.6% |
| nodal gradient | 0.630 | 24.4% |
| other | 0.100 | 3.9% |
| constraints | 0.047 | 1.8% |
| setup | 0.010 | 0.4% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 8.825 | 2941723.3 |
| `Function::apply` | 903 | 2.516 | 2786.0 |
| `CVFEMNavierStokes::apply` | 903 | 2.469 | 2734.0 |
| `sscvfem::apply` | 903 | 2.404 | 2662.5 |
| `sscvfem::nodal_q_grad` | 903 | 0.623 | 690.1 |
| `Function::copy_constrained_dofs` | 903 | 0.046 | 50.5 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.033 | 10971.7 |
| `Function::gradient` | 7 | 0.020 | 2823.0 |


### ss_L4_N16

4,343,300 dof, 72 threads, nid006558. 6.089 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 4.259 | 4716.4 | 920.9 | 69.9% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 916 | 1.630 | 1779.0 | 2441.4 | 26.8% |
| `to_semistructured` | other | 1 | 0.070 | 70184.5 | 61.9 | 1.2% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.055 | 61.0 | 71230.2 | 0.9% |
| `sscvfem::block_diag` | element sweep | 3 | 0.042 | 13980.7 | 310.7 | 0.7% |
| `sscvfem::build_scatter` | setup | 1 | 0.025 | 24791.7 | 175.2 | 0.4% |
| `SFC::reorder` | other | 1 | 0.003 | 2902.8 | 1496.3 | 0.0% |
| `create_dual_graph` | other | 1 | 0.002 | 2110.0 | 2058.4 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.001 | 775.8 | 5598.4 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.001 | 773.4 | 5615.6 | 0.0% |
| `DirichletConditions::gradient` | constraints | 10 | 0.001 | 71.7 | 60602.6 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 10 | 0.000 | 41.8 | 103979.1 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 203.3 | 21369.0 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 149.5 | 29054.5 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.1 | 56529420.7 | 0.0% |
| `sscvfem::apply_transient` | transient | 10 | 0.000 | 0.2 | 26024458.8 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 4.301 | 70.6% |
| nodal gradient | 1.630 | 26.8% |
| other | 0.076 | 1.2% |
| constraints | 0.058 | 1.0% |
| setup | 0.025 | 0.4% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 19.608 | 6535933.3 |
| `Function::apply` | 903 | 6.111 | 6767.9 |
| `CVFEMNavierStokes::apply` | 903 | 6.052 | 6702.6 |
| `sscvfem::apply` | 903 | 5.874 | 6504.5 |
| `sscvfem::nodal_q_grad` | 903 | 1.610 | 1783.4 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.079 | 26266.7 |
| `Function::gradient` | 10 | 0.065 | 6524.1 |
| `CVFEMNavierStokes::gradient` | 10 | 0.064 | 6448.6 |


## Throughput against problem size

MDOF/s per scope, the same scope across every configuration. A kernel that is
memory bound flattens; one that is not keeps climbing with the problem until it
does. A number measured below saturation is not a throughput, so the smallest
sizes are here to show where that begins rather than to be quoted.

### `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action`

| run | dof | calls | seconds | MDOF/s |
|---|---|---|---|---|
| flat_N16 | 75,140 | 903 | 0.027 | 2540.0 |
| flat_N24 | 242,500 | 903 | 0.034 | 6361.5 |
| flat_N32 | 561,924 | 326 | 0.018 | 10208.4 |
| flat_N48 | 1,853,572 | 903 | 0.163 | 10291.7 |
| flat_N64 | 4,343,300 | 903 | 0.337 | 11646.1 |
| ss_L2_N8 | 75,140 | -- | -- | -- |
| ss_L2_N12 | 242,500 | -- | -- | -- |
| ss_L2_N16 | 561,924 | -- | -- | -- |
| ss_L2_N24 | 1,853,572 | -- | -- | -- |
| ss_L2_N32 | 4,343,300 | -- | -- | -- |
| ss_L4_N4 | 75,140 | -- | -- | -- |
| ss_L4_N6 | 242,500 | -- | -- | -- |
| ss_L4_N8 | 561,924 | -- | -- | -- |
| ss_L4_N12 | 1,853,572 | -- | -- | -- |
| ss_L4_N16 | 4,343,300 | -- | -- | -- |

### `cvfem_hex8_ns_steady::apply_boundary_scs_residual`

| run | dof | calls | seconds | MDOF/s |
|---|---|---|---|---|
| flat_N16 | 75,140 | 7 | 0.002 | 343.3 |
| flat_N24 | 242,500 | 7 | 0.003 | 531.6 |
| flat_N32 | 561,924 | 3 | 0.005 | 309.2 |
| flat_N48 | 1,853,572 | 11 | 0.014 | 1487.5 |
| flat_N64 | 4,343,300 | 7 | 0.024 | 1264.9 |
| ss_L2_N8 | 75,140 | -- | -- | -- |
| ss_L2_N12 | 242,500 | -- | -- | -- |
| ss_L2_N16 | 561,924 | -- | -- | -- |
| ss_L2_N24 | 1,853,572 | -- | -- | -- |
| ss_L2_N32 | 4,343,300 | -- | -- | -- |
| ss_L4_N4 | 75,140 | -- | -- | -- |
| ss_L4_N6 | 242,500 | -- | -- | -- |
| ss_L4_N8 | 561,924 | -- | -- | -- |
| ss_L4_N12 | 1,853,572 | -- | -- | -- |
| ss_L4_N16 | 4,343,300 | -- | -- | -- |

### `cvfem_hex8_ns_steady::apply_jacobian_action_packed`

| run | dof | calls | seconds | MDOF/s |
|---|---|---|---|---|
| flat_N16 | 75,140 | 903 | 0.417 | 162.5 |
| flat_N24 | 242,500 | 903 | 0.413 | 530.5 |
| flat_N32 | 561,924 | 326 | 0.148 | 1240.3 |
| flat_N48 | 1,853,572 | 903 | 1.502 | 1114.1 |
| flat_N64 | 4,343,300 | 903 | 3.836 | 1022.5 |
| ss_L2_N8 | 75,140 | -- | -- | -- |
| ss_L2_N12 | 242,500 | -- | -- | -- |
| ss_L2_N16 | 561,924 | -- | -- | -- |
| ss_L2_N24 | 1,853,572 | -- | -- | -- |
| ss_L2_N32 | 4,343,300 | -- | -- | -- |
| ss_L4_N4 | 75,140 | -- | -- | -- |
| ss_L4_N6 | 242,500 | -- | -- | -- |
| ss_L4_N8 | 561,924 | -- | -- | -- |
| ss_L4_N12 | 1,853,572 | -- | -- | -- |
| ss_L4_N16 | 4,343,300 | -- | -- | -- |

### `cvfem_hex8_ns_steady::apply_residual_packed`

| run | dof | calls | seconds | MDOF/s |
|---|---|---|---|---|
| flat_N16 | 75,140 | 7 | 0.004 | 147.3 |
| flat_N24 | 242,500 | 7 | 0.003 | 510.9 |
| flat_N32 | 561,924 | 3 | 0.002 | 757.6 |
| flat_N48 | 1,853,572 | 11 | 0.013 | 1547.6 |
| flat_N64 | 4,343,300 | 7 | 0.021 | 1470.4 |
| ss_L2_N8 | 75,140 | -- | -- | -- |
| ss_L2_N12 | 242,500 | -- | -- | -- |
| ss_L2_N16 | 561,924 | -- | -- | -- |
| ss_L2_N24 | 1,853,572 | -- | -- | -- |
| ss_L2_N32 | 4,343,300 | -- | -- | -- |
| ss_L4_N4 | 75,140 | -- | -- | -- |
| ss_L4_N6 | 242,500 | -- | -- | -- |
| ss_L4_N8 | 561,924 | -- | -- | -- |
| ss_L4_N12 | 1,853,572 | -- | -- | -- |
| ss_L4_N16 | 4,343,300 | -- | -- | -- |

### `cvfem_hex8_ns_steady::assemble_block_diag`

| run | dof | calls | seconds | MDOF/s |
|---|---|---|---|---|
| flat_N16 | 75,140 | 3 | 0.016 | 14.4 |
| flat_N24 | 242,500 | 3 | 0.017 | 41.8 |
| flat_N32 | 561,924 | 2 | 0.010 | 109.8 |
| flat_N48 | 1,853,572 | 3 | 0.032 | 176.2 |
| flat_N64 | 4,343,300 | 3 | 0.069 | 188.7 |
| ss_L2_N8 | 75,140 | -- | -- | -- |
| ss_L2_N12 | 242,500 | -- | -- | -- |
| ss_L2_N16 | 561,924 | -- | -- | -- |
| ss_L2_N24 | 1,853,572 | -- | -- | -- |
| ss_L2_N32 | 4,343,300 | -- | -- | -- |
| ss_L4_N4 | 75,140 | -- | -- | -- |
| ss_L4_N6 | 242,500 | -- | -- | -- |
| ss_L4_N8 | 561,924 | -- | -- | -- |
| ss_L4_N12 | 1,853,572 | -- | -- | -- |
| ss_L4_N16 | 4,343,300 | -- | -- | -- |

### `cvfem_hex8_ns_steady::nodal_grad_strided`

| run | dof | calls | seconds | MDOF/s |
|---|---|---|---|---|
| flat_N16 | 75,140 | 913 | 0.109 | 628.3 |
| flat_N24 | 242,500 | 913 | 0.081 | 2736.5 |
| flat_N32 | 561,924 | 331 | 0.062 | 2992.8 |
| flat_N48 | 1,853,572 | 917 | 0.293 | 5804.4 |
| flat_N64 | 4,343,300 | 913 | 0.734 | 5399.1 |
| ss_L2_N8 | 75,140 | -- | -- | -- |
| ss_L2_N12 | 242,500 | -- | -- | -- |
| ss_L2_N16 | 561,924 | -- | -- | -- |
| ss_L2_N24 | 1,853,572 | -- | -- | -- |
| ss_L2_N32 | 4,343,300 | -- | -- | -- |
| ss_L4_N4 | 75,140 | -- | -- | -- |
| ss_L4_N6 | 242,500 | -- | -- | -- |
| ss_L4_N8 | 561,924 | -- | -- | -- |
| ss_L4_N12 | 1,853,572 | -- | -- | -- |
| ss_L4_N16 | 4,343,300 | -- | -- | -- |

### `sscvfem::apply_macro_local_hoisted`

| run | dof | calls | seconds | MDOF/s |
|---|---|---|---|---|
| flat_N16 | 75,140 | -- | -- | -- |
| flat_N24 | 242,500 | -- | -- | -- |
| flat_N32 | 561,924 | -- | -- | -- |
| flat_N48 | 1,853,572 | -- | -- | -- |
| flat_N64 | 4,343,300 | -- | -- | -- |
| ss_L2_N8 | 75,140 | 903 | 0.101 | 674.4 |
| ss_L2_N12 | 242,500 | 903 | 0.256 | 855.5 |
| ss_L2_N16 | 561,924 | 903 | 0.570 | 890.2 |
| ss_L2_N24 | 1,853,572 | 903 | 1.929 | 867.9 |
| ss_L2_N32 | 4,343,300 | 903 | 4.942 | 793.6 |
| ss_L4_N4 | 75,140 | 1505 | 0.154 | 733.7 |
| ss_L4_N6 | 242,500 | 903 | 0.235 | 931.9 |
| ss_L4_N8 | 561,924 | 903 | 0.522 | 972.9 |
| ss_L4_N12 | 1,853,572 | 903 | 1.779 | 940.9 |
| ss_L4_N16 | 4,343,300 | 903 | 4.259 | 920.9 |

### `sscvfem::block_diag`

| run | dof | calls | seconds | MDOF/s |
|---|---|---|---|---|
| flat_N16 | 75,140 | -- | -- | -- |
| flat_N24 | 242,500 | -- | -- | -- |
| flat_N32 | 561,924 | -- | -- | -- |
| flat_N48 | 1,853,572 | -- | -- | -- |
| flat_N64 | 4,343,300 | -- | -- | -- |
| ss_L2_N8 | 75,140 | 3 | 0.001 | 243.3 |
| ss_L2_N12 | 242,500 | 3 | 0.002 | 319.5 |
| ss_L2_N16 | 561,924 | 3 | 0.006 | 303.8 |
| ss_L2_N24 | 1,853,572 | 3 | 0.020 | 282.5 |
| ss_L2_N32 | 4,343,300 | 3 | 0.046 | 281.2 |
| ss_L4_N4 | 75,140 | 5 | 0.001 | 321.3 |
| ss_L4_N6 | 242,500 | 3 | 0.002 | 338.6 |
| ss_L4_N8 | 561,924 | 3 | 0.005 | 331.2 |
| ss_L4_N12 | 1,853,572 | 3 | 0.018 | 307.8 |
| ss_L4_N16 | 4,343,300 | 3 | 0.042 | 310.7 |

### `sscvfem::nodal_grad_strided`

| run | dof | calls | seconds | MDOF/s |
|---|---|---|---|---|
| flat_N16 | 75,140 | -- | -- | -- |
| flat_N24 | 242,500 | -- | -- | -- |
| flat_N32 | 561,924 | -- | -- | -- |
| flat_N48 | 1,853,572 | -- | -- | -- |
| flat_N64 | 4,343,300 | -- | -- | -- |
| ss_L2_N8 | 75,140 | 913 | 0.065 | 1054.5 |
| ss_L2_N12 | 242,500 | 913 | 0.117 | 1893.3 |
| ss_L2_N16 | 561,924 | 913 | 0.216 | 2375.8 |
| ss_L2_N24 | 1,853,572 | 916 | 0.773 | 2195.1 |
| ss_L2_N32 | 4,343,300 | 919 | 2.198 | 1815.9 |
| ss_L4_N4 | 75,140 | 1525 | 0.101 | 1132.3 |
| ss_L4_N6 | 242,500 | 913 | 0.105 | 2102.1 |
| ss_L4_N8 | 561,924 | 913 | 0.195 | 2631.2 |
| ss_L4_N12 | 1,853,572 | 913 | 0.630 | 2688.1 |
| ss_L4_N16 | 4,343,300 | 916 | 1.630 | 2441.4 |


## Provenance

| field | value |
|---|---|
| generated | 2026-09-11 10:33:36 |
| configurations | 15 |
| machines | nid006558 |
