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
| flat_N16 | 75,140 | apply_jacobian_action_packed | 164 | 77% | 10% | 5% | 8% |
| flat_N24 | 242,500 | apply_jacobian_action_packed | 529 | 71% | 13% | 8% | 7% |
| flat_N32 | 561,924 | apply_jacobian_action_packed | 1227 | 62% | 18% | 11% | 6% |
| flat_N48 | 1,853,572 | apply_jacobian_action_packed | 1107 | 71% | 16% | 7% | 2% |
| flat_N64 | 4,343,300 | apply_jacobian_action_packed | 1035 | 75% | 15% | 5% | 1% |
| ss_L2_N8 | 75,140 | apply_macro_local_hoisted | 682 | 39% | 25% | 0% | 15% |
| ss_L2_N12 | 242,500 | apply_macro_local_hoisted | 783 | 57% | 24% | 0% | 9% |
| ss_L2_N16 | 561,924 | apply_macro_local_hoisted | 888 | 68% | 25% | 0% | 5% |
| ss_L2_N24 | 1,853,572 | apply_macro_local_hoisted | 860 | 68% | 27% | 0% | 2% |
| ss_L2_N32 | 4,343,300 | apply_macro_local_hoisted | 788 | 66% | 30% | 0% | 1% |
| ss_L4_N4 | 75,140 | apply_macro_local_hoisted | 724 | 42% | 28% | 0% | 19% |
| ss_L4_N6 | 242,500 | apply_macro_local_hoisted | 924 | 53% | 24% | 0% | 10% |
| ss_L4_N8 | 561,924 | apply_macro_local_hoisted | 971 | 64% | 23% | 0% | 5% |
| ss_L4_N12 | 1,853,572 | apply_macro_local_hoisted | 946 | 70% | 26% | 0% | 2% |
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

75,140 dof, 72 threads, nid006547. 1.131 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `cvfem_hex8_ns_steady::apply_jacobian_action_packed` | element sweep | 1875 | 0.857 | 456.9 | 164.4 | 75.7% |
| `cvfem_hex8_ns_steady::nodal_grad_strided` | nodal gradient | 1900 | 0.114 | 60.0 | 1253.3 | 10.1% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 1875 | 0.085 | 45.2 | 1662.6 | 7.5% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action` | boundary | 1875 | 0.056 | 29.9 | 2513.8 | 5.0% |
| `cvfem_hex8_ns_steady::apply_residual_packed` | element sweep | 18 | 0.007 | 411.3 | 182.7 | 0.7% |
| `cvfem_hex8_ns_steady::assemble_block_diag` | element sweep | 7 | 0.004 | 636.3 | 118.1 | 0.4% |
| `SFC::reorder` | other | 1 | 0.003 | 2822.4 | 26.6 | 0.2% |
| `DirichletConditions::gradient` | constraints | 18 | 0.001 | 44.7 | 1679.9 | 0.1% |
| `Mesh::initialize_node_to_node_graph` | setup | 1 | 0.001 | 769.4 | 97.7 | 0.1% |
| `DirichletConditions::apply_value` | constraints | 18 | 0.001 | 42.3 | 1776.1 | 0.1% |
| `create_crs_graph_mem_conservative` | setup | 1 | 0.001 | 726.0 | 103.5 | 0.1% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_residual` | boundary | 18 | 0.001 | 39.4 | 1905.6 | 0.1% |
| `cvfem_hex8_ns_steady::precompute_element_bsr_slots` | setup | 1 | 0.000 | 360.0 | 208.7 | 0.0% |
| `create_n2e` | other | 1 | 0.000 | 291.8 | 257.5 | 0.0% |
| `cvfem_hex8_ns_steady::build_rc_coeff` | other | 1875 | 0.000 | 0.1 | 554339.1 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 93.5 | 804.0 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.869 | 76.8% |
| nodal gradient | 0.114 | 10.1% |
| constraints | 0.086 | 7.6% |
| boundary | 0.057 | 5.0% |
| other | 0.003 | 0.3% |
| setup | 0.002 | 0.2% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 7 | 1.925 | 275001.4 |
| `Function::apply` | 1875 | 1.137 | 606.4 |
| `CVFEMNavierStokes::apply` | 1875 | 1.051 | 560.4 |
| `cvfem_hex8_ns_steady::apply_jacobian_action_accumulate` | 1875 | 1.027 | 547.6 |
| `cvfem_hex8_ns_steady::assemble_nodal_q_grad` | 1875 | 0.111 | 59.4 |
| `Function::copy_constrained_dofs` | 1875 | 0.085 | 45.5 |
| `Function::gradient` | 18 | 0.013 | 722.8 |
| `CVFEMNavierStokes::gradient` | 18 | 0.012 | 677.0 |


### flat_N24

242,500 dof, 72 threads, nid006547. 0.598 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `cvfem_hex8_ns_steady::apply_jacobian_action_packed` | element sweep | 903 | 0.414 | 458.1 | 529.3 | 69.2% |
| `cvfem_hex8_ns_steady::nodal_grad_strided` | nodal gradient | 913 | 0.077 | 84.8 | 2861.3 | 12.9% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action` | boundary | 903 | 0.045 | 49.9 | 4861.0 | 7.5% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.040 | 44.1 | 5500.6 | 6.7% |
| `cvfem_hex8_ns_steady::assemble_block_diag` | element sweep | 3 | 0.005 | 1781.8 | 136.1 | 0.9% |
| `SFC::reorder` | other | 1 | 0.005 | 4814.6 | 50.4 | 0.8% |
| `cvfem_hex8_ns_steady::apply_residual_packed` | element sweep | 7 | 0.004 | 520.0 | 466.3 | 0.6% |
| `Mesh::initialize_node_to_node_graph` | setup | 1 | 0.002 | 2095.7 | 115.7 | 0.4% |
| `create_crs_graph_mem_conservative` | setup | 1 | 0.002 | 2084.0 | 116.4 | 0.3% |
| `create_n2e` | other | 1 | 0.001 | 1158.5 | 209.3 | 0.2% |
| `cvfem_hex8_ns_steady::precompute_element_bsr_slots` | setup | 1 | 0.001 | 968.5 | 250.4 | 0.2% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_residual` | boundary | 7 | 0.001 | 75.4 | 3217.3 | 0.1% |
| `cvfem_hex8_ns_steady::build_rc_coeff` | other | 903 | 0.000 | 0.4 | 603851.0 | 0.1% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 44.3 | 5472.6 | 0.1% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 40.8 | 5943.1 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 105.1 | 2306.4 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.423 | 70.7% |
| nodal gradient | 0.077 | 12.9% |
| boundary | 0.046 | 7.6% |
| constraints | 0.041 | 6.8% |
| other | 0.006 | 1.1% |
| setup | 0.005 | 0.9% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 1.480 | 493203.3 |
| `Function::apply` | 903 | 0.565 | 626.1 |
| `CVFEMNavierStokes::apply` | 903 | 0.525 | 581.3 |
| `cvfem_hex8_ns_steady::apply_jacobian_action_accumulate` | 903 | 0.512 | 567.5 |
| `cvfem_hex8_ns_steady::assemble_nodal_q_grad` | 903 | 0.052 | 58.0 |
| `Function::copy_constrained_dofs` | 903 | 0.040 | 44.3 |
| `Function::gradient` | 7 | 0.030 | 4304.0 |
| `CVFEMNavierStokes::gradient` | 7 | 0.030 | 4258.6 |


### flat_N32

561,924 dof, 72 threads, nid006547. 0.691 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `cvfem_hex8_ns_steady::apply_jacobian_action_packed` | element sweep | 903 | 0.413 | 457.8 | 1227.4 | 59.9% |
| `cvfem_hex8_ns_steady::nodal_grad_strided` | nodal gradient | 913 | 0.124 | 136.1 | 4130.1 | 18.0% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action` | boundary | 903 | 0.073 | 80.3 | 6998.3 | 10.5% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.041 | 45.8 | 12274.6 | 6.0% |
| `cvfem_hex8_ns_steady::assemble_block_diag` | element sweep | 3 | 0.011 | 3770.5 | 149.0 | 1.6% |
| `SFC::reorder` | other | 1 | 0.008 | 8146.3 | 69.0 | 1.2% |
| `Mesh::initialize_node_to_node_graph` | setup | 1 | 0.004 | 4436.5 | 126.7 | 0.6% |
| `create_crs_graph_mem_conservative` | setup | 1 | 0.004 | 4421.5 | 127.1 | 0.6% |
| `cvfem_hex8_ns_steady::apply_residual_packed` | element sweep | 7 | 0.003 | 481.6 | 1166.7 | 0.5% |
| `create_n2e` | other | 1 | 0.003 | 2775.2 | 202.5 | 0.4% |
| `cvfem_hex8_ns_steady::precompute_element_bsr_slots` | setup | 1 | 0.002 | 2059.7 | 272.8 | 0.3% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_residual` | boundary | 7 | 0.001 | 105.4 | 5330.6 | 0.1% |
| `cvfem_hex8_ns_steady::build_rc_coeff` | other | 903 | 0.001 | 0.8 | 688535.2 | 0.1% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 46.0 | 12202.8 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 41.9 | 13424.1 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 184.1 | 3053.0 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.428 | 62.0% |
| nodal gradient | 0.124 | 18.0% |
| boundary | 0.073 | 10.6% |
| constraints | 0.042 | 6.1% |
| other | 0.012 | 1.7% |
| setup | 0.011 | 1.6% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 2.596 | 865376.7 |
| `Function::apply` | 903 | 0.604 | 669.0 |
| `CVFEMNavierStokes::apply` | 903 | 0.562 | 622.4 |
| `cvfem_hex8_ns_steady::apply_jacobian_action_accumulate` | 903 | 0.547 | 606.1 |
| `Function::gradient` | 7 | 0.070 | 10008.8 |
| `CVFEMNavierStokes::gradient` | 7 | 0.070 | 9961.2 |
| `cvfem_hex8_ns_steady::apply_residual` | 7 | 0.069 | 9859.4 |
| `cvfem_hex8_ns_steady::assemble_nodal_p_grad` | 10 | 0.065 | 6510.3 |


### flat_N48

1,853,572 dof, 72 threads, nid006547. 1.968 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `cvfem_hex8_ns_steady::apply_jacobian_action_packed` | element sweep | 809 | 1.355 | 1674.4 | 1107.0 | 68.8% |
| `cvfem_hex8_ns_steady::nodal_grad_strided` | nodal gradient | 820 | 0.316 | 384.9 | 4816.3 | 16.0% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action` | boundary | 809 | 0.130 | 160.8 | 11527.8 | 6.6% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 809 | 0.041 | 50.2 | 36888.0 | 2.1% |
| `cvfem_hex8_ns_steady::assemble_block_diag` | element sweep | 3 | 0.036 | 12061.1 | 153.7 | 1.8% |
| `SFC::reorder` | other | 1 | 0.023 | 22776.8 | 81.4 | 1.2% |
| `Mesh::initialize_node_to_node_graph` | setup | 1 | 0.015 | 15436.2 | 120.1 | 0.8% |
| `create_crs_graph_mem_conservative` | setup | 1 | 0.015 | 15414.7 | 120.2 | 0.8% |
| `cvfem_hex8_ns_steady::apply_residual_packed` | element sweep | 8 | 0.013 | 1686.6 | 1099.0 | 0.7% |
| `create_n2e` | other | 1 | 0.010 | 10472.5 | 177.0 | 0.5% |
| `cvfem_hex8_ns_steady::precompute_element_bsr_slots` | setup | 1 | 0.007 | 6957.3 | 266.4 | 0.4% |
| `cvfem_hex8_ns_steady::build_rc_coeff` | other | 809 | 0.002 | 3.0 | 611465.5 | 0.1% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_residual` | boundary | 8 | 0.002 | 205.3 | 9028.2 | 0.1% |
| `DirichletConditions::gradient` | constraints | 8 | 0.001 | 70.9 | 26154.6 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 420.1 | 4412.3 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 418.2 | 4432.4 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 1.404 | 71.4% |
| nodal gradient | 0.316 | 16.0% |
| boundary | 0.132 | 6.7% |
| constraints | 0.043 | 2.2% |
| setup | 0.038 | 1.9% |
| other | 0.036 | 1.8% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 7.624 | 2541203.3 |
| `Function::apply` | 809 | 1.839 | 2272.7 |
| `CVFEMNavierStokes::apply` | 809 | 1.796 | 2219.6 |
| `cvfem_hex8_ns_steady::apply_jacobian_action_accumulate` | 809 | 1.727 | 2135.2 |
| `cvfem_hex8_ns_steady::assemble_nodal_q_grad` | 809 | 0.237 | 292.7 |
| `CVFEMNavierStokes::initialize` | 1 | 0.102 | 101844.0 |
| `Function::gradient` | 8 | 0.097 | 12177.6 |
| `CVFEMNavierStokes::gradient` | 8 | 0.097 | 12103.0 |


### flat_N64

4,343,300 dof, 72 threads, nid006547. 5.205 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `cvfem_hex8_ns_steady::apply_jacobian_action_packed` | element sweep | 903 | 3.788 | 4195.4 | 1035.3 | 72.8% |
| `cvfem_hex8_ns_steady::nodal_grad_strided` | nodal gradient | 918 | 0.757 | 824.3 | 5269.0 | 14.5% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action` | boundary | 903 | 0.279 | 308.5 | 14077.5 | 5.4% |
| `cvfem_hex8_ns_steady::assemble_block_diag` | element sweep | 3 | 0.092 | 30555.4 | 142.1 | 1.8% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.054 | 60.2 | 72143.7 | 1.0% |
| `SFC::reorder` | other | 1 | 0.052 | 52347.9 | 83.0 | 1.0% |
| `Mesh::initialize_node_to_node_graph` | setup | 1 | 0.043 | 42880.8 | 101.3 | 0.8% |
| `create_crs_graph_mem_conservative` | setup | 1 | 0.043 | 42847.6 | 101.4 | 0.8% |
| `cvfem_hex8_ns_steady::apply_residual_packed` | element sweep | 12 | 0.035 | 2911.7 | 1491.7 | 0.7% |
| `create_n2e` | other | 1 | 0.033 | 32956.8 | 131.8 | 0.6% |
| `cvfem_hex8_ns_steady::precompute_element_bsr_slots` | setup | 1 | 0.016 | 15919.9 | 272.8 | 0.3% |
| `cvfem_hex8_ns_steady::build_rc_coeff` | other | 903 | 0.005 | 5.9 | 730496.6 | 0.1% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_residual` | boundary | 12 | 0.004 | 360.3 | 12055.0 | 0.1% |
| `DirichletConditions::gradient` | constraints | 12 | 0.001 | 73.0 | 59516.9 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.001 | 819.4 | 5300.3 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.001 | 816.8 | 5317.3 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 3.915 | 75.2% |
| nodal gradient | 0.757 | 14.5% |
| boundary | 0.283 | 5.4% |
| setup | 0.102 | 2.0% |
| other | 0.091 | 1.7% |
| constraints | 0.058 | 1.1% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 18.540 | 6179900.0 |
| `Function::apply` | 903 | 5.038 | 5579.3 |
| `CVFEMNavierStokes::apply` | 903 | 4.979 | 5514.2 |
| `cvfem_hex8_ns_steady::apply_jacobian_action_accumulate` | 903 | 4.796 | 5311.5 |
| `cvfem_hex8_ns_steady::assemble_nodal_q_grad` | 903 | 0.718 | 795.0 |
| `CVFEMNavierStokes::initialize` | 1 | 0.223 | 223398.0 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.124 | 41365.3 |
| `Function::gradient` | 12 | 0.086 | 7137.5 |


### ss_L2_N8

75,140 dof, 72 threads, nid006547. 0.258 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 0.099 | 110.2 | 682.1 | 38.5% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 913 | 0.063 | 69.5 | 1081.6 | 24.6% |
| `to_semistructured` | other | 1 | 0.052 | 51724.2 | 1.5 | 20.0% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.038 | 42.6 | 1763.2 | 14.9% |
| `SFC::reorder` | other | 1 | 0.002 | 2412.1 | 31.2 | 0.9% |
| `sscvfem::block_diag` | element sweep | 3 | 0.001 | 255.5 | 294.1 | 0.3% |
| `sscvfem::build_scatter` | setup | 1 | 0.001 | 668.5 | 112.4 | 0.3% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 42.4 | 1773.4 | 0.1% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 40.3 | 1866.4 | 0.1% |
| `create_dual_graph` | other | 1 | 0.000 | 255.8 | 293.7 | 0.1% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 100.4 | 748.6 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 27.2 | 2764.6 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 46.7 | 1608.0 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 45.8 | 1641.5 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.0 | 2047405.3 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.0 | 2206116.1 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.100 | 38.8% |
| nodal gradient | 0.063 | 24.6% |
| other | 0.054 | 21.1% |
| constraints | 0.039 | 15.2% |
| setup | 0.001 | 0.3% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 0.584 | 194561.0 |
| `Function::apply` | 903 | 0.215 | 238.2 |
| `CVFEMNavierStokes::apply` | 903 | 0.176 | 195.0 |
| `sscvfem::apply` | 903 | 0.163 | 180.2 |
| `sscvfem::nodal_q_grad` | 903 | 0.063 | 69.7 |
| `Function::copy_constrained_dofs` | 903 | 0.039 | 42.8 |
| `Function::gradient` | 7 | 0.002 | 274.1 |
| `CVFEMNavierStokes::gradient` | 7 | 0.002 | 230.4 |


### ss_L2_N12

242,500 dof, 72 threads, nid006547. 0.493 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 0.280 | 309.6 | 783.3 | 56.7% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 913 | 0.117 | 128.3 | 1890.1 | 23.7% |
| `to_semistructured` | other | 1 | 0.045 | 45455.2 | 5.3 | 9.2% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.042 | 47.0 | 5160.7 | 8.6% |
| `sscvfem::block_diag` | element sweep | 3 | 0.002 | 794.5 | 305.2 | 0.5% |
| `SFC::reorder` | other | 1 | 0.002 | 2368.0 | 102.4 | 0.5% |
| `sscvfem::build_scatter` | setup | 1 | 0.002 | 1952.4 | 124.2 | 0.4% |
| `create_dual_graph` | other | 1 | 0.001 | 858.1 | 282.6 | 0.2% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 49.4 | 4906.8 | 0.1% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 46.4 | 5231.3 | 0.1% |
| `create_n2e` | other | 2 | 0.000 | 93.9 | 2581.5 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 112.1 | 2164.1 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 103.0 | 2354.4 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 101.6 | 2387.6 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.0 | 7467136.1 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.1 | 2373276.1 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.282 | 57.1% |
| nodal gradient | 0.117 | 23.7% |
| other | 0.049 | 9.9% |
| constraints | 0.043 | 8.8% |
| setup | 0.002 | 0.4% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 1.366 | 455433.3 |
| `Function::apply` | 903 | 0.453 | 501.5 |
| `CVFEMNavierStokes::apply` | 903 | 0.410 | 453.9 |
| `sscvfem::apply` | 903 | 0.396 | 438.5 |
| `sscvfem::nodal_q_grad` | 903 | 0.116 | 128.4 |
| `Function::copy_constrained_dofs` | 903 | 0.043 | 47.2 |
| `Function::gradient` | 7 | 0.006 | 886.6 |
| `CVFEMNavierStokes::gradient` | 7 | 0.006 | 839.4 |


### ss_L2_N16

561,924 dof, 72 threads, nid006547. 0.853 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 0.571 | 632.8 | 888.1 | 67.0% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 913 | 0.213 | 233.8 | 2403.1 | 25.0% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.042 | 46.7 | 12030.6 | 4.9% |
| `to_semistructured` | other | 1 | 0.009 | 9383.4 | 59.9 | 1.1% |
| `sscvfem::block_diag` | element sweep | 3 | 0.006 | 1894.1 | 296.7 | 0.7% |
| `sscvfem::build_scatter` | setup | 1 | 0.005 | 4780.5 | 117.5 | 0.6% |
| `SFC::reorder` | other | 1 | 0.003 | 2866.3 | 196.0 | 0.3% |
| `create_dual_graph` | other | 1 | 0.002 | 2131.2 | 263.7 | 0.2% |
| `create_n2e` | other | 2 | 0.000 | 194.7 | 2886.6 | 0.0% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 45.1 | 12451.4 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 42.5 | 13230.3 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 184.3 | 3049.0 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 183.1 | 3068.9 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 114.7 | 4900.0 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.0 | 13819873.1 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.0 | 16498131.4 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.577 | 67.6% |
| nodal gradient | 0.213 | 25.0% |
| constraints | 0.043 | 5.1% |
| other | 0.015 | 1.7% |
| setup | 0.005 | 0.6% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 2.804 | 934663.3 |
| `Function::apply` | 903 | 0.841 | 931.8 |
| `CVFEMNavierStokes::apply` | 903 | 0.798 | 884.1 |
| `sscvfem::apply` | 903 | 0.783 | 866.9 |
| `sscvfem::nodal_q_grad` | 903 | 0.211 | 233.3 |
| `Function::copy_constrained_dofs` | 903 | 0.042 | 47.0 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.010 | 3212.8 |
| `Function::gradient` | 7 | 0.007 | 1069.1 |


### ss_L2_N24

1,853,572 dof, 72 threads, nid006547. 2.883 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 1.946 | 2155.2 | 860.1 | 67.5% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 916 | 0.781 | 852.3 | 2174.8 | 27.1% |
| `to_semistructured` | other | 1 | 0.055 | 55008.9 | 33.7 | 1.9% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.047 | 52.4 | 35389.5 | 1.6% |
| `sscvfem::block_diag` | element sweep | 3 | 0.020 | 6506.7 | 284.9 | 0.7% |
| `sscvfem::build_scatter` | setup | 1 | 0.018 | 18305.5 | 101.3 | 0.6% |
| `create_dual_graph` | other | 1 | 0.008 | 7955.6 | 233.0 | 0.3% |
| `SFC::reorder` | other | 1 | 0.005 | 4770.3 | 388.6 | 0.2% |
| `create_n2e` | other | 2 | 0.002 | 895.7 | 2069.3 | 0.1% |
| `DirichletConditions::gradient` | constraints | 10 | 0.001 | 56.8 | 32610.9 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 423.7 | 4375.0 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 10 | 0.000 | 42.2 | 43973.1 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 419.1 | 4422.3 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 125.2 | 14808.4 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.1 | 27316422.0 | 0.0% |
| `sscvfem::apply_transient` | transient | 10 | 0.000 | 0.0 | 77744307.3 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 1.966 | 68.2% |
| nodal gradient | 0.781 | 27.1% |
| other | 0.070 | 2.4% |
| constraints | 0.049 | 1.7% |
| setup | 0.018 | 0.6% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 9.253 | 3084363.3 |
| `Function::apply` | 903 | 2.834 | 3138.5 |
| `CVFEMNavierStokes::apply` | 903 | 2.784 | 3083.3 |
| `sscvfem::apply` | 903 | 2.719 | 3011.6 |
| `sscvfem::nodal_q_grad` | 903 | 0.771 | 854.0 |
| `Function::copy_constrained_dofs` | 903 | 0.048 | 53.4 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.035 | 11596.4 |
| `Function::gradient` | 10 | 0.032 | 3171.9 |


### ss_L2_N32

4,343,300 dof, 72 threads, nid006547. 7.600 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 4.975 | 5509.2 | 788.4 | 65.5% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 919 | 2.295 | 2497.6 | 1739.0 | 30.2% |
| `to_semistructured` | other | 1 | 0.113 | 113277.0 | 38.3 | 1.5% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.093 | 103.4 | 42005.8 | 1.2% |
| `sscvfem::block_diag` | element sweep | 3 | 0.045 | 15051.3 | 288.6 | 0.6% |
| `sscvfem::build_scatter` | setup | 1 | 0.043 | 42634.5 | 101.9 | 0.6% |
| `create_dual_graph` | other | 1 | 0.019 | 18973.1 | 228.9 | 0.2% |
| `SFC::reorder` | other | 1 | 0.009 | 8503.7 | 510.8 | 0.1% |
| `create_n2e` | other | 2 | 0.004 | 2226.4 | 1950.9 | 0.1% |
| `DirichletConditions::gradient` | constraints | 13 | 0.002 | 119.7 | 36283.5 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.001 | 773.9 | 5612.2 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.001 | 771.5 | 5629.5 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 13 | 0.001 | 43.2 | 100647.1 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 142.3 | 30514.4 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.1 | 48099563.2 | 0.0% |
| `sscvfem::apply_transient` | transient | 13 | 0.000 | 0.1 | 59205661.5 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 5.020 | 66.0% |
| nodal gradient | 2.295 | 30.2% |
| other | 0.145 | 1.9% |
| constraints | 0.097 | 1.3% |
| setup | 0.043 | 0.6% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 21.055 | 7018400.0 |
| `Function::apply` | 903 | 7.515 | 8322.0 |
| `CVFEMNavierStokes::apply` | 903 | 7.418 | 8214.7 |
| `sscvfem::apply` | 903 | 7.238 | 8015.7 |
| `sscvfem::nodal_q_grad` | 903 | 2.259 | 2501.6 |
| `Function::gradient` | 13 | 0.104 | 7983.8 |
| `CVFEMNavierStokes::gradient` | 13 | 0.102 | 7860.2 |
| `Function::copy_constrained_dofs` | 903 | 0.094 | 104.6 |


### ss_L4_N4

75,140 dof, 72 threads, nid006547. 0.373 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 1505 | 0.156 | 103.7 | 724.5 | 41.9% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 1525 | 0.103 | 67.6 | 1111.0 | 27.7% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 1505 | 0.069 | 45.6 | 1646.9 | 18.4% |
| `to_semistructured` | other | 1 | 0.039 | 39144.5 | 1.9 | 10.5% |
| `SFC::reorder` | other | 1 | 0.002 | 2379.2 | 31.6 | 0.6% |
| `sscvfem::block_diag` | element sweep | 5 | 0.001 | 242.3 | 310.1 | 0.3% |
| `DirichletConditions::gradient` | constraints | 15 | 0.001 | 44.8 | 1676.4 | 0.2% |
| `DirichletConditions::apply_value` | constraints | 15 | 0.001 | 42.9 | 1750.2 | 0.2% |
| `sscvfem::build_scatter` | setup | 1 | 0.000 | 341.9 | 219.8 | 0.1% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 80.1 | 938.0 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 1505 | 0.000 | 0.0 | 1943918.3 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 46.5 | 1616.2 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 45.5 | 1650.1 | 0.0% |
| `create_dual_graph` | other | 1 | 0.000 | 32.4 | 2317.4 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 3.7 | 20332.9 | 0.0% |
| `sscvfem::apply_transient` | transient | 15 | 0.000 | 0.0 | 2363700.8 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.157 | 42.2% |
| nodal gradient | 0.103 | 27.7% |
| constraints | 0.070 | 18.8% |
| other | 0.042 | 11.2% |
| setup | 0.000 | 0.1% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 5 | 0.970 | 193939.0 |
| `Function::apply` | 1505 | 0.352 | 233.9 |
| `CVFEMNavierStokes::apply` | 1505 | 0.283 | 187.8 |
| `sscvfem::apply` | 1505 | 0.259 | 172.0 |
| `sscvfem::nodal_q_grad` | 1505 | 0.102 | 67.8 |
| `Function::copy_constrained_dofs` | 1505 | 0.069 | 45.8 |
| `Function::gradient` | 15 | 0.004 | 261.7 |
| `CVFEMNavierStokes::gradient` | 15 | 0.003 | 216.0 |


### ss_L4_N6

242,500 dof, 72 threads, nid006547. 0.449 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 0.237 | 262.5 | 924.0 | 52.7% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 913 | 0.106 | 116.3 | 2085.1 | 23.6% |
| `to_semistructured` | other | 1 | 0.056 | 55713.9 | 4.4 | 12.4% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.044 | 48.9 | 4962.6 | 9.8% |
| `SFC::reorder` | other | 1 | 0.002 | 2274.3 | 106.6 | 0.5% |
| `sscvfem::block_diag` | element sweep | 3 | 0.002 | 702.6 | 345.1 | 0.5% |
| `sscvfem::build_scatter` | setup | 1 | 0.001 | 974.9 | 248.7 | 0.2% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 44.9 | 5402.0 | 0.1% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 41.6 | 5835.9 | 0.1% |
| `create_dual_graph` | other | 1 | 0.000 | 105.9 | 2290.8 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 102.3 | 2370.9 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 101.3 | 2393.2 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 75.8 | 3198.5 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.0 | 6513891.8 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 10.7 | 22602.6 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.1 | 3559916.7 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.239 | 53.2% |
| nodal gradient | 0.106 | 23.6% |
| other | 0.058 | 12.9% |
| constraints | 0.045 | 10.0% |
| setup | 0.001 | 0.2% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 1.312 | 437200.0 |
| `Function::apply` | 903 | 0.401 | 444.3 |
| `CVFEMNavierStokes::apply` | 903 | 0.356 | 394.7 |
| `sscvfem::apply` | 903 | 0.343 | 379.4 |
| `sscvfem::nodal_q_grad` | 903 | 0.105 | 116.4 |
| `Function::copy_constrained_dofs` | 903 | 0.044 | 49.1 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.004 | 1229.0 |
| `Function::gradient` | 7 | 0.004 | 500.7 |


### ss_L4_N8

561,924 dof, 72 threads, nid006547. 0.830 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 0.522 | 578.6 | 971.2 | 62.9% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 913 | 0.188 | 205.4 | 2735.3 | 22.6% |
| `to_semistructured` | other | 1 | 0.068 | 67841.1 | 8.3 | 8.2% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.041 | 45.5 | 12357.7 | 4.9% |
| `sscvfem::block_diag` | element sweep | 3 | 0.005 | 1706.9 | 329.2 | 0.6% |
| `sscvfem::build_scatter` | setup | 1 | 0.002 | 2452.8 | 229.1 | 0.3% |
| `SFC::reorder` | other | 1 | 0.002 | 2294.1 | 244.9 | 0.3% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 46.0 | 12211.8 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 40.6 | 13840.7 | 0.0% |
| `create_dual_graph` | other | 1 | 0.000 | 255.3 | 2200.6 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 182.9 | 3072.9 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 180.7 | 3109.3 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 106.3 | 5284.5 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 25.3 | 22234.7 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.0 | 16890996.9 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.0 | 16498131.4 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.528 | 63.5% |
| nodal gradient | 0.188 | 22.6% |
| other | 0.070 | 8.5% |
| constraints | 0.042 | 5.1% |
| setup | 0.002 | 0.3% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 2.725 | 908490.0 |
| `Function::apply` | 903 | 0.765 | 846.9 |
| `CVFEMNavierStokes::apply` | 903 | 0.723 | 800.6 |
| `sscvfem::apply` | 903 | 0.708 | 784.1 |
| `sscvfem::nodal_q_grad` | 903 | 0.185 | 204.9 |
| `Function::copy_constrained_dofs` | 903 | 0.041 | 45.8 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.009 | 2896.7 |
| `Function::gradient` | 7 | 0.007 | 997.2 |


### ss_L4_N12

1,853,572 dof, 72 threads, nid006547. 2.547 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 1.770 | 1959.7 | 945.8 | 69.5% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 913 | 0.657 | 719.5 | 2576.0 | 25.8% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.045 | 50.0 | 37079.7 | 1.8% |
| `to_semistructured` | other | 1 | 0.042 | 42471.9 | 43.6 | 1.7% |
| `sscvfem::block_diag` | element sweep | 3 | 0.018 | 6163.3 | 300.7 | 0.7% |
| `sscvfem::build_scatter` | setup | 1 | 0.009 | 9165.3 | 202.2 | 0.4% |
| `SFC::reorder` | other | 1 | 0.003 | 2531.3 | 732.3 | 0.1% |
| `create_dual_graph` | other | 1 | 0.001 | 860.0 | 2155.4 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 428.4 | 4326.3 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 426.3 | 4348.1 | 0.0% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 53.7 | 34487.4 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 42.1 | 44065.7 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 83.2 | 22276.4 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 120.4 | 15395.0 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.1 | 34413342.7 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.0 | 54421015.1 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 1.788 | 70.2% |
| nodal gradient | 0.657 | 25.8% |
| constraints | 0.047 | 1.8% |
| other | 0.046 | 1.8% |
| setup | 0.009 | 0.4% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 8.822 | 2940626.7 |
| `Function::apply` | 903 | 2.532 | 2803.5 |
| `CVFEMNavierStokes::apply` | 903 | 2.484 | 2751.0 |
| `sscvfem::apply` | 903 | 2.421 | 2681.2 |
| `sscvfem::nodal_q_grad` | 903 | 0.650 | 719.7 |
| `Function::copy_constrained_dofs` | 903 | 0.046 | 50.9 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.033 | 11104.5 |
| `Function::gradient` | 7 | 0.020 | 2849.7 |


### ss_L4_N16

4,343,300 dof, 72 threads, nid006547. 6.093 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 4.258 | 4715.0 | 921.2 | 69.9% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 916 | 1.653 | 1805.0 | 2406.3 | 27.1% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.056 | 61.6 | 70480.1 | 0.9% |
| `to_semistructured` | other | 1 | 0.053 | 52839.0 | 82.2 | 0.9% |
| `sscvfem::block_diag` | element sweep | 3 | 0.041 | 13522.1 | 321.2 | 0.7% |
| `sscvfem::build_scatter` | setup | 1 | 0.025 | 24596.0 | 176.6 | 0.4% |
| `SFC::reorder` | other | 1 | 0.003 | 2887.7 | 1504.1 | 0.0% |
| `create_dual_graph` | other | 1 | 0.002 | 2190.4 | 1982.9 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.001 | 774.9 | 5605.3 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.001 | 771.5 | 5629.5 | 0.0% |
| `DirichletConditions::gradient` | constraints | 10 | 0.001 | 74.1 | 58594.8 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 10 | 0.001 | 50.2 | 86542.2 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 229.6 | 18917.1 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 127.6 | 34050.7 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.1 | 57517703.3 | 0.0% |
| `sscvfem::apply_transient` | transient | 10 | 0.000 | 0.1 | 36434329.6 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 4.298 | 70.5% |
| nodal gradient | 1.653 | 27.1% |
| constraints | 0.059 | 1.0% |
| other | 0.058 | 1.0% |
| setup | 0.025 | 0.4% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 19.575 | 6525133.3 |
| `Function::apply` | 903 | 6.133 | 6791.6 |
| `CVFEMNavierStokes::apply` | 903 | 6.073 | 6725.6 |
| `sscvfem::apply` | 903 | 5.895 | 6528.0 |
| `sscvfem::nodal_q_grad` | 903 | 1.633 | 1808.8 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.077 | 25811.4 |
| `Function::gradient` | 10 | 0.063 | 6304.9 |
| `CVFEMNavierStokes::gradient` | 10 | 0.062 | 6226.7 |


## Throughput against problem size

MDOF/s per scope, the same scope across every configuration. A kernel that is
memory bound flattens; one that is not keeps climbing with the problem until it
does. A number measured below saturation is not a throughput, so the smallest
sizes are here to show where that begins rather than to be quoted.

### `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action`

| run | dof | calls | seconds | MDOF/s |
|---|---|---|---|---|
| flat_N16 | 75,140 | 1875 | 0.056 | 2513.8 |
| flat_N24 | 242,500 | 903 | 0.045 | 4861.0 |
| flat_N32 | 561,924 | 903 | 0.073 | 6998.3 |
| flat_N48 | 1,853,572 | 809 | 0.130 | 11527.8 |
| flat_N64 | 4,343,300 | 903 | 0.279 | 14077.5 |
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
| flat_N16 | 75,140 | 18 | 0.001 | 1905.6 |
| flat_N24 | 242,500 | 7 | 0.001 | 3217.3 |
| flat_N32 | 561,924 | 7 | 0.001 | 5330.6 |
| flat_N48 | 1,853,572 | 8 | 0.002 | 9028.2 |
| flat_N64 | 4,343,300 | 12 | 0.004 | 12055.0 |
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
| flat_N16 | 75,140 | 1875 | 0.857 | 164.4 |
| flat_N24 | 242,500 | 903 | 0.414 | 529.3 |
| flat_N32 | 561,924 | 903 | 0.413 | 1227.4 |
| flat_N48 | 1,853,572 | 809 | 1.355 | 1107.0 |
| flat_N64 | 4,343,300 | 903 | 3.788 | 1035.3 |
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
| flat_N16 | 75,140 | 18 | 0.007 | 182.7 |
| flat_N24 | 242,500 | 7 | 0.004 | 466.3 |
| flat_N32 | 561,924 | 7 | 0.003 | 1166.7 |
| flat_N48 | 1,853,572 | 8 | 0.013 | 1099.0 |
| flat_N64 | 4,343,300 | 12 | 0.035 | 1491.7 |
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
| flat_N16 | 75,140 | 7 | 0.004 | 118.1 |
| flat_N24 | 242,500 | 3 | 0.005 | 136.1 |
| flat_N32 | 561,924 | 3 | 0.011 | 149.0 |
| flat_N48 | 1,853,572 | 3 | 0.036 | 153.7 |
| flat_N64 | 4,343,300 | 3 | 0.092 | 142.1 |
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
| flat_N16 | 75,140 | 1900 | 0.114 | 1253.3 |
| flat_N24 | 242,500 | 913 | 0.077 | 2861.3 |
| flat_N32 | 561,924 | 913 | 0.124 | 4130.1 |
| flat_N48 | 1,853,572 | 820 | 0.316 | 4816.3 |
| flat_N64 | 4,343,300 | 918 | 0.757 | 5269.0 |
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
| ss_L2_N8 | 75,140 | 903 | 0.099 | 682.1 |
| ss_L2_N12 | 242,500 | 903 | 0.280 | 783.3 |
| ss_L2_N16 | 561,924 | 903 | 0.571 | 888.1 |
| ss_L2_N24 | 1,853,572 | 903 | 1.946 | 860.1 |
| ss_L2_N32 | 4,343,300 | 903 | 4.975 | 788.4 |
| ss_L4_N4 | 75,140 | 1505 | 0.156 | 724.5 |
| ss_L4_N6 | 242,500 | 903 | 0.237 | 924.0 |
| ss_L4_N8 | 561,924 | 903 | 0.522 | 971.2 |
| ss_L4_N12 | 1,853,572 | 903 | 1.770 | 945.8 |
| ss_L4_N16 | 4,343,300 | 903 | 4.258 | 921.2 |

### `sscvfem::block_diag`

| run | dof | calls | seconds | MDOF/s |
|---|---|---|---|---|
| flat_N16 | 75,140 | -- | -- | -- |
| flat_N24 | 242,500 | -- | -- | -- |
| flat_N32 | 561,924 | -- | -- | -- |
| flat_N48 | 1,853,572 | -- | -- | -- |
| flat_N64 | 4,343,300 | -- | -- | -- |
| ss_L2_N8 | 75,140 | 3 | 0.001 | 294.1 |
| ss_L2_N12 | 242,500 | 3 | 0.002 | 305.2 |
| ss_L2_N16 | 561,924 | 3 | 0.006 | 296.7 |
| ss_L2_N24 | 1,853,572 | 3 | 0.020 | 284.9 |
| ss_L2_N32 | 4,343,300 | 3 | 0.045 | 288.6 |
| ss_L4_N4 | 75,140 | 5 | 0.001 | 310.1 |
| ss_L4_N6 | 242,500 | 3 | 0.002 | 345.1 |
| ss_L4_N8 | 561,924 | 3 | 0.005 | 329.2 |
| ss_L4_N12 | 1,853,572 | 3 | 0.018 | 300.7 |
| ss_L4_N16 | 4,343,300 | 3 | 0.041 | 321.2 |

### `sscvfem::nodal_grad_strided`

| run | dof | calls | seconds | MDOF/s |
|---|---|---|---|---|
| flat_N16 | 75,140 | -- | -- | -- |
| flat_N24 | 242,500 | -- | -- | -- |
| flat_N32 | 561,924 | -- | -- | -- |
| flat_N48 | 1,853,572 | -- | -- | -- |
| flat_N64 | 4,343,300 | -- | -- | -- |
| ss_L2_N8 | 75,140 | 913 | 0.063 | 1081.6 |
| ss_L2_N12 | 242,500 | 913 | 0.117 | 1890.1 |
| ss_L2_N16 | 561,924 | 913 | 0.213 | 2403.1 |
| ss_L2_N24 | 1,853,572 | 916 | 0.781 | 2174.8 |
| ss_L2_N32 | 4,343,300 | 919 | 2.295 | 1739.0 |
| ss_L4_N4 | 75,140 | 1525 | 0.103 | 1111.0 |
| ss_L4_N6 | 242,500 | 913 | 0.106 | 2085.1 |
| ss_L4_N8 | 561,924 | 913 | 0.188 | 2735.3 |
| ss_L4_N12 | 1,853,572 | 913 | 0.657 | 2576.0 |
| ss_L4_N16 | 4,343,300 | 916 | 1.653 | 2406.3 |


## Provenance

| field | value |
|---|---|
| generated | 2026-09-11 08:17:22 |
| configurations | 15 |
| machines | nid006547 |
