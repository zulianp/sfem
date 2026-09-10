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
| flat_N16 | 75,140 | apply_jacobian_action_packed | 157 | 66% | 20% | 6% | 7% |
| flat_N24 | 242,500 | apply_jacobian_action_packed | 466 | 39% | 52% | 5% | 4% |
| flat_N32 | 561,924 | apply_jacobian_action_packed | 820 | 49% | 40% | 6% | 4% |
| flat_N48 | 1,853,572 | apply_jacobian_action_packed | 911 | 51% | 42% | 5% | 1% |
| flat_N64 | 4,343,300 | apply_jacobian_action_packed | 851 | 54% | 40% | 4% | 1% |
| ss_L2_N8 | 75,140 | apply_macro_local_hoisted | 678 | 43% | 29% | 0% | 17% |
| ss_L2_N12 | 242,500 | apply_macro_local_hoisted | 766 | 62% | 26% | 0% | 10% |
| ss_L2_N16 | 561,924 | apply_macro_local_hoisted | 790 | 62% | 25% | 0% | 5% |
| ss_L2_N24 | 1,853,572 | apply_macro_local_hoisted | 752 | 68% | 28% | 0% | 2% |
| ss_L2_N32 | 4,343,300 | apply_macro_local_hoisted | 710 | 66% | 31% | 0% | 1% |
| ss_L4_N4 | 75,140 | apply_macro_local_hoisted | 702 | 47% | 31% | 0% | 20% |
| ss_L4_N6 | 242,500 | apply_macro_local_hoisted | 853 | 50% | 23% | 0% | 9% |
| ss_L4_N8 | 561,924 | apply_macro_local_hoisted | 876 | 59% | 24% | 0% | 5% |
| ss_L4_N12 | 1,853,572 | apply_macro_local_hoisted | 821 | 72% | 25% | 0% | 2% |
| ss_L4_N16 | 4,343,300 | apply_macro_local_hoisted | 843 | 71% | 27% | 0% | 1% |

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

75,140 dof, 72 threads, nid006546. 0.662 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `cvfem_hex8_ns_steady::apply_jacobian_action_packed` | element sweep | 903 | 0.432 | 478.4 | 157.1 | 65.3% |
| `cvfem_hex8_ns_steady::nodal_grad_strided` | nodal gradient | 913 | 0.135 | 147.4 | 509.9 | 20.3% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.045 | 49.4 | 1519.5 | 6.7% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action` | boundary | 903 | 0.039 | 43.5 | 1728.0 | 5.9% |
| `cvfem_hex8_ns_steady::apply_residual_packed` | element sweep | 7 | 0.004 | 597.3 | 125.8 | 0.6% |
| `cvfem_hex8_ns_steady::assemble_block_diag` | element sweep | 3 | 0.002 | 697.8 | 107.7 | 0.3% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_residual` | boundary | 7 | 0.002 | 297.9 | 252.3 | 0.3% |
| `Mesh::initialize_node_to_node_graph` | setup | 1 | 0.001 | 724.1 | 103.8 | 0.1% |
| `create_crs_graph_mem_conservative` | setup | 1 | 0.001 | 674.7 | 111.4 | 0.1% |
| `create_n2e` | other | 1 | 0.000 | 295.9 | 254.0 | 0.0% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 41.8 | 1796.5 | 0.0% |
| `cvfem_hex8_ns_steady::precompute_element_bsr_slots` | setup | 1 | 0.000 | 288.5 | 260.5 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 40.7 | 1844.6 | 0.0% |
| `cvfem_hex8_ns_steady::build_rc_coeff` | other | 903 | 0.000 | 0.2 | 494937.1 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 75.8 | 991.1 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 46.3 | 1624.5 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.438 | 66.2% |
| nodal gradient | 0.135 | 20.3% |
| constraints | 0.045 | 6.9% |
| boundary | 0.041 | 6.2% |
| setup | 0.002 | 0.3% |
| other | 0.000 | 0.1% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 1.060 | 353183.3 |
| `Function::apply` | 903 | 0.664 | 735.3 |
| `CVFEMNavierStokes::apply` | 903 | 0.619 | 685.2 |
| `cvfem_hex8_ns_steady::apply_jacobian_action` | 903 | 0.606 | 670.9 |
| `cvfem_hex8_ns_steady::assemble_nodal_q_grad` | 903 | 0.133 | 147.8 |
| `Function::copy_constrained_dofs` | 903 | 0.045 | 49.7 |
| `Function::gradient` | 7 | 0.008 | 1145.4 |
| `CVFEMNavierStokes::gradient` | 7 | 0.008 | 1102.3 |


### flat_N24

242,500 dof, 72 threads, nid006546. 1.265 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `cvfem_hex8_ns_steady::nodal_grad_strided` | nodal gradient | 918 | 0.660 | 719.2 | 337.2 | 52.2% |
| `cvfem_hex8_ns_steady::apply_jacobian_action_packed` | element sweep | 903 | 0.470 | 520.2 | 466.1 | 37.1% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action` | boundary | 903 | 0.063 | 70.0 | 3463.1 | 5.0% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.042 | 46.5 | 5215.5 | 3.3% |
| `cvfem_hex8_ns_steady::assemble_block_diag` | element sweep | 3 | 0.011 | 3671.2 | 66.1 | 0.9% |
| `cvfem_hex8_ns_steady::apply_residual_packed` | element sweep | 12 | 0.006 | 536.0 | 452.4 | 0.5% |
| `Mesh::initialize_node_to_node_graph` | setup | 1 | 0.002 | 2433.8 | 99.6 | 0.2% |
| `create_crs_graph_mem_conservative` | setup | 1 | 0.002 | 2424.2 | 100.0 | 0.2% |
| `DirichletConditions::gradient` | constraints | 12 | 0.002 | 169.2 | 1433.1 | 0.2% |
| `create_n2e` | other | 1 | 0.002 | 1678.0 | 144.5 | 0.1% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_residual` | boundary | 12 | 0.001 | 97.7 | 2483.3 | 0.1% |
| `cvfem_hex8_ns_steady::precompute_element_bsr_slots` | setup | 1 | 0.001 | 797.3 | 304.2 | 0.1% |
| `DirichletConditions::apply_value` | constraints | 12 | 0.001 | 54.2 | 4477.4 | 0.1% |
| `cvfem_hex8_ns_steady::build_rc_coeff` | other | 903 | 0.000 | 0.5 | 458084.1 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 103.7 | 2338.2 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 102.3 | 2370.9 | 0.0% |

| kind | seconds | share |
|---|---|---|
| nodal gradient | 0.660 | 52.2% |
| element sweep | 0.487 | 38.5% |
| boundary | 0.064 | 5.1% |
| constraints | 0.045 | 3.6% |
| setup | 0.006 | 0.4% |
| other | 0.002 | 0.2% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 2.231 | 743786.7 |
| `Function::apply` | 903 | 1.243 | 1376.9 |
| `CVFEMNavierStokes::apply` | 903 | 1.201 | 1329.6 |
| `cvfem_hex8_ns_steady::apply_jacobian_action` | 903 | 1.185 | 1312.1 |
| `cvfem_hex8_ns_steady::assemble_nodal_q_grad` | 903 | 0.650 | 720.1 |
| `Function::copy_constrained_dofs` | 903 | 0.042 | 46.8 |
| `CVFEMNavierStokes::initialize` | 1 | 0.020 | 20372.6 |
| `Function::gradient` | 12 | 0.019 | 1583.3 |


### flat_N32

561,924 dof, 72 threads, nid006546. 1.312 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `cvfem_hex8_ns_steady::apply_jacobian_action_packed` | element sweep | 903 | 0.619 | 685.5 | 819.7 | 47.2% |
| `cvfem_hex8_ns_steady::nodal_grad_strided` | nodal gradient | 913 | 0.524 | 574.0 | 979.0 | 39.9% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action` | boundary | 903 | 0.083 | 92.4 | 6080.2 | 6.4% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.046 | 51.4 | 10924.6 | 3.5% |
| `cvfem_hex8_ns_steady::assemble_block_diag` | element sweep | 3 | 0.013 | 4428.1 | 126.9 | 1.0% |
| `cvfem_hex8_ns_steady::apply_residual_packed` | element sweep | 7 | 0.009 | 1285.2 | 437.2 | 0.7% |
| `Mesh::initialize_node_to_node_graph` | setup | 1 | 0.004 | 4477.5 | 125.5 | 0.3% |
| `create_crs_graph_mem_conservative` | setup | 1 | 0.004 | 4465.8 | 125.8 | 0.3% |
| `create_n2e` | other | 1 | 0.003 | 3155.0 | 178.1 | 0.2% |
| `cvfem_hex8_ns_steady::precompute_element_bsr_slots` | setup | 1 | 0.002 | 1856.3 | 302.7 | 0.1% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_residual` | boundary | 7 | 0.001 | 165.7 | 3391.2 | 0.1% |
| `cvfem_hex8_ns_steady::build_rc_coeff` | other | 903 | 0.001 | 0.8 | 732872.4 | 0.1% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 45.5 | 12358.2 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 41.9 | 13424.1 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 181.9 | 3089.0 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 180.2 | 3117.6 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.641 | 48.9% |
| nodal gradient | 0.524 | 39.9% |
| boundary | 0.085 | 6.4% |
| constraints | 0.048 | 3.6% |
| setup | 0.011 | 0.8% |
| other | 0.004 | 0.3% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 3.419 | 1139780.0 |
| `Function::apply` | 903 | 1.286 | 1423.7 |
| `CVFEMNavierStokes::apply` | 903 | 1.238 | 1371.2 |
| `cvfem_hex8_ns_steady::apply_jacobian_action` | 903 | 1.223 | 1354.4 |
| `cvfem_hex8_ns_steady::assemble_nodal_q_grad` | 903 | 0.518 | 574.2 |
| `Function::copy_constrained_dofs` | 903 | 0.047 | 51.8 |
| `CVFEMNavierStokes::initialize` | 1 | 0.042 | 42409.7 |
| `Function::gradient` | 7 | 0.016 | 2312.8 |


### flat_N48

1,853,572 dof, 72 threads, nid006546. 3.730 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `cvfem_hex8_ns_steady::apply_jacobian_action_packed` | element sweep | 903 | 1.837 | 2033.9 | 911.3 | 49.2% |
| `cvfem_hex8_ns_steady::nodal_grad_strided` | nodal gradient | 917 | 1.550 | 1690.0 | 1096.8 | 41.5% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action` | boundary | 903 | 0.172 | 190.0 | 9754.3 | 4.6% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.053 | 58.3 | 31802.4 | 1.4% |
| `cvfem_hex8_ns_steady::assemble_block_diag` | element sweep | 3 | 0.043 | 14490.4 | 127.9 | 1.2% |
| `cvfem_hex8_ns_steady::apply_residual_packed` | element sweep | 11 | 0.017 | 1583.4 | 1170.6 | 0.5% |
| `Mesh::initialize_node_to_node_graph` | setup | 1 | 0.016 | 15951.2 | 116.2 | 0.4% |
| `create_crs_graph_mem_conservative` | setup | 1 | 0.016 | 15934.0 | 116.3 | 0.4% |
| `create_n2e` | other | 1 | 0.012 | 12156.7 | 152.5 | 0.3% |
| `cvfem_hex8_ns_steady::precompute_element_bsr_slots` | setup | 1 | 0.007 | 7425.6 | 249.6 | 0.2% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_residual` | boundary | 11 | 0.003 | 247.1 | 7501.0 | 0.1% |
| `cvfem_hex8_ns_steady::build_rc_coeff` | other | 903 | 0.002 | 2.5 | 729005.8 | 0.1% |
| `DirichletConditions::gradient` | constraints | 11 | 0.001 | 50.8 | 36468.6 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 11 | 0.000 | 41.3 | 44915.4 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 438.9 | 4222.9 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 435.8 | 4253.0 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 1.898 | 50.9% |
| nodal gradient | 1.550 | 41.5% |
| boundary | 0.174 | 4.7% |
| constraints | 0.055 | 1.5% |
| setup | 0.039 | 1.1% |
| other | 0.014 | 0.4% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 10.448 | 3482566.7 |
| `Function::apply` | 903 | 3.732 | 4132.8 |
| `CVFEMNavierStokes::apply` | 903 | 3.677 | 4071.9 |
| `cvfem_hex8_ns_steady::apply_jacobian_action` | 903 | 3.543 | 3923.4 |
| `cvfem_hex8_ns_steady::assemble_nodal_q_grad` | 903 | 1.529 | 1693.1 |
| `CVFEMNavierStokes::initialize` | 1 | 0.152 | 151570.0 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.055 | 18367.1 |
| `Function::copy_constrained_dofs` | 903 | 0.053 | 59.1 |


### flat_N64

4,343,300 dof, 72 threads, nid006546. 8.839 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `cvfem_hex8_ns_steady::apply_jacobian_action_packed` | element sweep | 903 | 4.608 | 5103.2 | 851.1 | 52.1% |
| `cvfem_hex8_ns_steady::nodal_grad_strided` | nodal gradient | 916 | 3.557 | 3883.6 | 1118.4 | 40.2% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action` | boundary | 903 | 0.344 | 381.4 | 11387.2 | 3.9% |
| `cvfem_hex8_ns_steady::assemble_block_diag` | element sweep | 3 | 0.098 | 32547.4 | 133.4 | 1.1% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.068 | 75.8 | 57330.5 | 0.8% |
| `Mesh::initialize_node_to_node_graph` | setup | 1 | 0.036 | 36273.7 | 119.7 | 0.4% |
| `create_crs_graph_mem_conservative` | setup | 1 | 0.036 | 36252.0 | 119.8 | 0.4% |
| `cvfem_hex8_ns_steady::apply_residual_packed` | element sweep | 10 | 0.035 | 3459.9 | 1255.3 | 0.4% |
| `create_n2e` | other | 1 | 0.026 | 26243.7 | 165.5 | 0.3% |
| `cvfem_hex8_ns_steady::precompute_element_bsr_slots` | setup | 1 | 0.015 | 15315.8 | 283.6 | 0.2% |
| `cvfem_hex8_ns_steady::build_rc_coeff` | other | 903 | 0.005 | 5.7 | 767046.3 | 0.1% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_residual` | boundary | 10 | 0.005 | 470.2 | 9237.4 | 0.1% |
| `DirichletConditions::gradient` | constraints | 10 | 0.001 | 115.1 | 37739.9 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 10 | 0.001 | 108.2 | 40125.8 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.001 | 753.4 | 5764.9 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.001 | 750.5 | 5786.9 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 4.740 | 53.6% |
| nodal gradient | 3.557 | 40.2% |
| boundary | 0.349 | 4.0% |
| setup | 0.088 | 1.0% |
| constraints | 0.072 | 0.8% |
| other | 0.031 | 0.4% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 22.736 | 7578833.3 |
| `Function::apply` | 903 | 8.753 | 9693.7 |
| `CVFEMNavierStokes::apply` | 903 | 8.682 | 9614.9 |
| `cvfem_hex8_ns_steady::apply_jacobian_action` | 903 | 8.473 | 9383.2 |
| `cvfem_hex8_ns_steady::assemble_nodal_q_grad` | 903 | 3.511 | 3888.2 |
| `CVFEMNavierStokes::initialize` | 1 | 0.360 | 360097.0 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.130 | 43220.0 |
| `Function::gradient` | 10 | 0.083 | 8266.4 |


### ss_L2_N8

75,140 dof, 72 threads, nid006546. 0.314 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 1204 | 0.133 | 110.9 | 677.8 | 42.5% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 1221 | 0.090 | 73.9 | 1016.4 | 28.7% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 1204 | 0.054 | 44.5 | 1687.8 | 17.0% |
| `to_semistructured` | other | 1 | 0.034 | 33620.1 | 2.2 | 10.7% |
| `sscvfem::block_diag` | element sweep | 4 | 0.001 | 271.9 | 276.3 | 0.3% |
| `sscvfem::build_scatter` | setup | 1 | 0.001 | 716.9 | 104.8 | 0.2% |
| `DirichletConditions::gradient` | constraints | 13 | 0.001 | 41.7 | 1800.9 | 0.2% |
| `DirichletConditions::apply_value` | constraints | 13 | 0.001 | 40.1 | 1874.2 | 0.2% |
| `create_dual_graph` | other | 1 | 0.000 | 253.4 | 296.5 | 0.1% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 95.1 | 789.9 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 28.8 | 2604.6 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 1204 | 0.000 | 0.0 | 1567986.0 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 46.7 | 1608.0 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 45.8 | 1641.5 | 0.0% |
| `sscvfem::apply_transient` | transient | 13 | 0.000 | 0.0 | 2048540.7 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.135 | 42.8% |
| nodal gradient | 0.090 | 28.7% |
| constraints | 0.055 | 17.4% |
| other | 0.034 | 10.8% |
| setup | 0.001 | 0.2% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 4 | 0.822 | 205472.2 |
| `Function::apply` | 1204 | 0.294 | 244.2 |
| `CVFEMNavierStokes::apply` | 1204 | 0.240 | 199.2 |
| `sscvfem::apply` | 1204 | 0.223 | 185.5 |
| `sscvfem::nodal_q_grad` | 1204 | 0.089 | 74.2 |
| `Function::copy_constrained_dofs` | 1204 | 0.054 | 44.7 |
| `CVFEMNavierStokes::hessian_block_diag` | 4 | 0.004 | 1048.0 |
| `Function::gradient` | 13 | 0.003 | 256.2 |


### ss_L2_N12

242,500 dof, 72 threads, nid006546. 0.467 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 0.286 | 316.4 | 766.5 | 61.2% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 913 | 0.121 | 132.9 | 1825.0 | 26.0% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.047 | 52.1 | 4651.5 | 10.1% |
| `to_semistructured` | other | 1 | 0.006 | 6316.7 | 38.4 | 1.4% |
| `sscvfem::block_diag` | element sweep | 3 | 0.003 | 834.9 | 290.5 | 0.5% |
| `sscvfem::build_scatter` | setup | 1 | 0.002 | 2030.1 | 119.5 | 0.4% |
| `create_dual_graph` | other | 1 | 0.001 | 839.0 | 289.0 | 0.2% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 40.7 | 5953.0 | 0.1% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 39.2 | 6185.8 | 0.1% |
| `create_n2e` | other | 2 | 0.000 | 90.1 | 2690.8 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 102.8 | 2359.9 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 101.8 | 2382.0 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 79.4 | 3054.4 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.0 | 5634717.0 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.0 | 0.0 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.288 | 61.7% |
| nodal gradient | 0.121 | 26.0% |
| constraints | 0.048 | 10.3% |
| other | 0.007 | 1.6% |
| setup | 0.002 | 0.4% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 1.452 | 483900.0 |
| `Function::apply` | 903 | 0.468 | 518.1 |
| `CVFEMNavierStokes::apply` | 903 | 0.420 | 465.3 |
| `sscvfem::apply` | 903 | 0.406 | 450.0 |
| `sscvfem::nodal_q_grad` | 903 | 0.120 | 133.1 |
| `Function::copy_constrained_dofs` | 903 | 0.047 | 52.3 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.007 | 2167.3 |
| `Function::gradient` | 7 | 0.005 | 654.1 |


### ss_L2_N16

561,924 dof, 72 threads, nid006546. 1.042 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 0.642 | 711.5 | 789.8 | 61.6% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 913 | 0.259 | 284.0 | 1978.6 | 24.9% |
| `to_semistructured` | other | 1 | 0.079 | 79472.1 | 7.1 | 7.6% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.046 | 51.3 | 10943.5 | 4.4% |
| `sscvfem::block_diag` | element sweep | 3 | 0.006 | 2104.2 | 267.0 | 0.6% |
| `sscvfem::build_scatter` | setup | 1 | 0.005 | 4908.1 | 114.5 | 0.5% |
| `create_dual_graph` | other | 1 | 0.002 | 2080.4 | 270.1 | 0.2% |
| `create_n2e` | other | 2 | 0.000 | 195.7 | 2870.7 | 0.0% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 42.3 | 13294.3 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 40.1 | 14029.1 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 217.0 | 2590.0 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 215.5 | 2607.2 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 104.2 | 5393.3 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.1 | 10432658.2 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.1 | 8249083.0 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.649 | 62.2% |
| nodal gradient | 0.259 | 24.9% |
| other | 0.082 | 7.9% |
| constraints | 0.047 | 4.6% |
| setup | 0.005 | 0.5% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 3.046 | 1015173.3 |
| `Function::apply` | 903 | 0.963 | 1066.8 |
| `CVFEMNavierStokes::apply` | 903 | 0.916 | 1014.7 |
| `sscvfem::apply` | 903 | 0.895 | 990.8 |
| `sscvfem::nodal_q_grad` | 903 | 0.251 | 278.5 |
| `Function::copy_constrained_dofs` | 903 | 0.047 | 51.6 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.013 | 4206.0 |
| `Function::gradient` | 7 | 0.010 | 1437.9 |


### ss_L2_N24

1,853,572 dof, 72 threads, nid006546. 3.302 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 2.227 | 2465.8 | 751.7 | 67.4% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 914 | 0.926 | 1012.7 | 1830.3 | 28.0% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.063 | 69.7 | 26608.7 | 1.9% |
| `to_semistructured` | other | 1 | 0.030 | 30027.9 | 61.7 | 0.9% |
| `sscvfem::block_diag` | element sweep | 3 | 0.026 | 8804.2 | 210.5 | 0.8% |
| `sscvfem::build_scatter` | setup | 1 | 0.019 | 19291.9 | 96.1 | 0.6% |
| `create_dual_graph` | other | 1 | 0.008 | 7622.7 | 243.2 | 0.2% |
| `create_n2e` | other | 2 | 0.002 | 882.1 | 2101.2 | 0.1% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 441.1 | 4202.4 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 438.9 | 4222.9 | 0.0% |
| `DirichletConditions::gradient` | constraints | 8 | 0.000 | 53.4 | 34707.3 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 8 | 0.000 | 43.2 | 42893.5 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 114.0 | 16264.5 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.1 | 25435964.9 | 0.0% |
| `sscvfem::apply_transient` | transient | 8 | 0.000 | 0.1 | 12439141.3 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 2.253 | 68.2% |
| nodal gradient | 0.926 | 28.0% |
| constraints | 0.065 | 2.0% |
| other | 0.039 | 1.2% |
| setup | 0.019 | 0.6% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 9.906 | 3301916.7 |
| `Function::apply` | 903 | 3.295 | 3648.5 |
| `CVFEMNavierStokes::apply` | 903 | 3.230 | 3576.7 |
| `sscvfem::apply` | 903 | 3.145 | 3483.2 |
| `sscvfem::nodal_q_grad` | 903 | 0.917 | 1015.1 |
| `Function::copy_constrained_dofs` | 903 | 0.063 | 70.3 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.042 | 14019.3 |
| `Function::gradient` | 8 | 0.029 | 3620.7 |


### ss_L2_N32

4,343,300 dof, 72 threads, nid006546. 8.405 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 5.524 | 6116.9 | 710.0 | 65.7% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 914 | 2.594 | 2838.0 | 1530.4 | 30.9% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.098 | 108.7 | 39974.7 | 1.2% |
| `to_semistructured` | other | 1 | 0.067 | 66530.2 | 65.3 | 0.8% |
| `sscvfem::block_diag` | element sweep | 3 | 0.049 | 16189.4 | 268.3 | 0.6% |
| `sscvfem::build_scatter` | setup | 1 | 0.046 | 45583.7 | 95.3 | 0.5% |
| `create_dual_graph` | other | 1 | 0.019 | 19270.9 | 225.4 | 0.2% |
| `create_n2e` | other | 2 | 0.005 | 2510.1 | 1730.4 | 0.1% |
| `DirichletConditions::apply` | constraints | 1 | 0.002 | 1506.3 | 2883.4 | 0.0% |
| `DirichletConditions::gradient` | constraints | 8 | 0.001 | 100.1 | 43374.1 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.001 | 770.6 | 5636.5 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.001 | 767.7 | 5657.5 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 8 | 0.000 | 39.6 | 109741.6 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.1 | 50615593.8 | 0.0% |
| `sscvfem::apply_transient` | transient | 8 | 0.000 | 0.2 | 20819567.0 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 5.572 | 66.3% |
| nodal gradient | 2.594 | 30.9% |
| constraints | 0.102 | 1.2% |
| other | 0.091 | 1.1% |
| setup | 0.046 | 0.5% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 22.310 | 7436700.0 |
| `Function::apply` | 903 | 8.390 | 9291.6 |
| `CVFEMNavierStokes::apply` | 903 | 8.289 | 9179.8 |
| `sscvfem::apply` | 903 | 8.094 | 8963.9 |
| `sscvfem::nodal_q_grad` | 903 | 2.568 | 2843.5 |
| `Function::copy_constrained_dofs` | 903 | 0.099 | 109.6 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.088 | 29221.6 |
| `Function::gradient` | 8 | 0.067 | 8352.6 |


### ss_L4_N4

75,140 dof, 72 threads, nid006546. 0.209 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 0.097 | 107.0 | 702.4 | 46.1% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 913 | 0.066 | 71.8 | 1045.9 | 31.3% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.042 | 46.2 | 1627.9 | 19.9% |
| `to_semistructured` | other | 1 | 0.004 | 3588.2 | 20.9 | 1.7% |
| `sscvfem::block_diag` | element sweep | 3 | 0.001 | 269.7 | 278.7 | 0.4% |
| `sscvfem::build_scatter` | setup | 1 | 0.000 | 340.0 | 221.0 | 0.2% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 41.2 | 1821.7 | 0.1% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 39.1 | 1920.0 | 0.1% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 74.6 | 1006.9 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 46.5 | 1616.2 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 45.5 | 1650.1 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.0 | 1949243.7 | 0.0% |
| `create_dual_graph` | other | 1 | 0.000 | 31.9 | 2351.9 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 3.2 | 23345.2 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.1 | 1103060.4 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.097 | 46.5% |
| nodal gradient | 0.066 | 31.3% |
| constraints | 0.042 | 20.3% |
| other | 0.004 | 1.7% |
| setup | 0.000 | 0.2% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 0.614 | 204592.7 |
| `Function::apply` | 903 | 0.220 | 243.7 |
| `CVFEMNavierStokes::apply` | 903 | 0.178 | 197.0 |
| `sscvfem::apply` | 903 | 0.162 | 179.5 |
| `sscvfem::nodal_q_grad` | 903 | 0.065 | 72.1 |
| `Function::copy_constrained_dofs` | 903 | 0.042 | 46.4 |
| `Function::gradient` | 7 | 0.002 | 246.5 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.001 | 477.1 |


### ss_L4_N6

242,500 dof, 72 threads, nid006546. 0.522 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 0.257 | 284.3 | 852.9 | 49.2% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 913 | 0.123 | 134.2 | 1807.4 | 23.5% |
| `to_semistructured` | other | 1 | 0.094 | 94475.0 | 2.6 | 18.1% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.044 | 48.4 | 5010.2 | 8.4% |
| `sscvfem::block_diag` | element sweep | 3 | 0.002 | 729.9 | 332.2 | 0.4% |
| `sscvfem::build_scatter` | setup | 1 | 0.001 | 1063.3 | 228.1 | 0.2% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 42.2 | 5741.8 | 0.1% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 40.7 | 5953.0 | 0.1% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 102.8 | 2359.9 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 101.8 | 2382.0 | 0.0% |
| `create_dual_graph` | other | 1 | 0.000 | 99.9 | 2427.5 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 93.9 | 2581.5 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.1 | 4373610.6 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 8.6 | 28253.4 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.0 | 0.0 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.259 | 49.6% |
| nodal gradient | 0.123 | 23.5% |
| other | 0.095 | 18.1% |
| constraints | 0.045 | 8.5% |
| setup | 0.001 | 0.2% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 1.394 | 464760.0 |
| `Function::apply` | 903 | 0.440 | 487.2 |
| `CVFEMNavierStokes::apply` | 903 | 0.396 | 438.1 |
| `sscvfem::apply` | 903 | 0.379 | 419.4 |
| `sscvfem::nodal_q_grad` | 903 | 0.121 | 134.5 |
| `Function::copy_constrained_dofs` | 903 | 0.044 | 48.6 |
| `Function::gradient` | 7 | 0.005 | 704.4 |
| `CVFEMNavierStokes::gradient` | 7 | 0.005 | 660.8 |


### ss_L4_N8

561,924 dof, 72 threads, nid006546. 0.955 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 875 | 0.561 | 641.6 | 875.8 | 58.8% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 885 | 0.225 | 253.9 | 2213.2 | 23.5% |
| `to_semistructured` | other | 1 | 0.112 | 112474.0 | 5.0 | 11.8% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 875 | 0.047 | 53.6 | 10479.4 | 4.9% |
| `sscvfem::block_diag` | element sweep | 3 | 0.005 | 1714.2 | 327.8 | 0.5% |
| `sscvfem::build_scatter` | setup | 1 | 0.003 | 2527.5 | 222.3 | 0.3% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 42.8 | 13114.6 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 40.1 | 14005.2 | 0.0% |
| `create_dual_graph` | other | 1 | 0.000 | 248.2 | 2264.1 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 183.1 | 3068.9 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 181.7 | 3093.0 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 103.0 | 5455.7 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 24.6 | 22882.3 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 875 | 0.000 | 0.0 | 15505775.8 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.1 | 4124541.5 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.567 | 59.4% |
| nodal gradient | 0.225 | 23.5% |
| other | 0.113 | 11.8% |
| constraints | 0.048 | 5.0% |
| setup | 0.003 | 0.3% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 2.841 | 946990.0 |
| `Function::apply` | 875 | 0.856 | 977.9 |
| `CVFEMNavierStokes::apply` | 875 | 0.808 | 923.4 |
| `sscvfem::apply` | 875 | 0.784 | 896.0 |
| `sscvfem::nodal_q_grad` | 875 | 0.222 | 253.8 |
| `Function::copy_constrained_dofs` | 875 | 0.047 | 54.0 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.009 | 2920.9 |
| `Function::gradient` | 7 | 0.007 | 929.2 |


### ss_L4_N12

1,853,572 dof, 72 threads, nid006546. 2.852 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 2.038 | 2257.3 | 821.2 | 71.5% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 920 | 0.718 | 780.1 | 2376.0 | 25.2% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.052 | 57.6 | 32160.1 | 1.8% |
| `sscvfem::block_diag` | element sweep | 3 | 0.022 | 7202.5 | 257.4 | 0.8% |
| `to_semistructured` | other | 1 | 0.010 | 10222.7 | 181.3 | 0.4% |
| `sscvfem::build_scatter` | setup | 1 | 0.009 | 8874.6 | 208.9 | 0.3% |
| `create_dual_graph` | other | 1 | 0.001 | 860.2 | 2154.8 | 0.0% |
| `DirichletConditions::gradient` | constraints | 14 | 0.001 | 53.5 | 34641.1 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 14 | 0.001 | 39.4 | 47097.5 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 417.5 | 4440.0 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 415.8 | 4457.8 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 82.0 | 22600.1 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 118.0 | 15706.0 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.1 | 29497168.2 | 0.0% |
| `sscvfem::apply_transient` | transient | 14 | 0.000 | 0.1 | 36280727.5 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 2.060 | 72.2% |
| nodal gradient | 0.718 | 25.2% |
| constraints | 0.054 | 1.9% |
| other | 0.011 | 0.4% |
| setup | 0.009 | 0.3% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 9.341 | 3113640.0 |
| `Function::apply` | 903 | 2.878 | 3186.7 |
| `CVFEMNavierStokes::apply` | 903 | 2.824 | 3127.1 |
| `sscvfem::apply` | 903 | 2.745 | 3039.8 |
| `sscvfem::nodal_q_grad` | 903 | 0.705 | 780.7 |
| `Function::copy_constrained_dofs` | 903 | 0.053 | 58.2 |
| `Function::gradient` | 14 | 0.043 | 3052.2 |
| `CVFEMNavierStokes::gradient` | 14 | 0.042 | 2996.8 |


### ss_L4_N16

4,343,300 dof, 72 threads, nid006546. 6.655 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 4.654 | 5154.2 | 842.7 | 69.9% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 918 | 1.792 | 1952.5 | 2224.5 | 26.9% |
| `to_semistructured` | other | 1 | 0.069 | 68604.5 | 63.3 | 1.0% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.067 | 74.2 | 58547.4 | 1.0% |
| `sscvfem::block_diag` | element sweep | 3 | 0.044 | 14718.6 | 295.1 | 0.7% |
| `sscvfem::build_scatter` | setup | 1 | 0.023 | 22525.8 | 192.8 | 0.3% |
| `create_dual_graph` | other | 1 | 0.002 | 2096.4 | 2071.8 | 0.0% |
| `DirichletConditions::gradient` | constraints | 12 | 0.001 | 87.1 | 49864.2 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.001 | 775.8 | 5598.4 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.001 | 773.7 | 5613.9 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 12 | 0.000 | 40.8 | 106532.9 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 201.8 | 21520.5 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 112.5 | 38595.4 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.1 | 56335841.8 | 0.0% |
| `sscvfem::apply_transient` | transient | 12 | 0.000 | 0.2 | 19873255.5 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 4.698 | 70.6% |
| nodal gradient | 1.792 | 26.9% |
| other | 0.071 | 1.1% |
| constraints | 0.070 | 1.1% |
| setup | 0.023 | 0.3% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 20.404 | 6801266.7 |
| `Function::apply` | 903 | 6.706 | 7426.0 |
| `CVFEMNavierStokes::apply` | 903 | 6.636 | 7349.1 |
| `sscvfem::apply` | 903 | 6.425 | 7115.7 |
| `sscvfem::nodal_q_grad` | 903 | 1.769 | 1958.8 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.081 | 27050.3 |
| `Function::gradient` | 12 | 0.077 | 6447.6 |
| `CVFEMNavierStokes::gradient` | 12 | 0.076 | 6357.2 |


## Throughput against problem size

MDOF/s per scope, the same scope across every configuration. A kernel that is
memory bound flattens; one that is not keeps climbing with the problem until it
does. A number measured below saturation is not a throughput, so the smallest
sizes are here to show where that begins rather than to be quoted.

### `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action`

| run | dof | calls | seconds | MDOF/s |
|---|---|---|---|---|
| flat_N16 | 75,140 | 903 | 0.039 | 1728.0 |
| flat_N24 | 242,500 | 903 | 0.063 | 3463.1 |
| flat_N32 | 561,924 | 903 | 0.083 | 6080.2 |
| flat_N48 | 1,853,572 | 903 | 0.172 | 9754.3 |
| flat_N64 | 4,343,300 | 903 | 0.344 | 11387.2 |
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
| flat_N16 | 75,140 | 7 | 0.002 | 252.3 |
| flat_N24 | 242,500 | 12 | 0.001 | 2483.3 |
| flat_N32 | 561,924 | 7 | 0.001 | 3391.2 |
| flat_N48 | 1,853,572 | 11 | 0.003 | 7501.0 |
| flat_N64 | 4,343,300 | 10 | 0.005 | 9237.4 |
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
| flat_N16 | 75,140 | 903 | 0.432 | 157.1 |
| flat_N24 | 242,500 | 903 | 0.470 | 466.1 |
| flat_N32 | 561,924 | 903 | 0.619 | 819.7 |
| flat_N48 | 1,853,572 | 903 | 1.837 | 911.3 |
| flat_N64 | 4,343,300 | 903 | 4.608 | 851.1 |
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
| flat_N16 | 75,140 | 7 | 0.004 | 125.8 |
| flat_N24 | 242,500 | 12 | 0.006 | 452.4 |
| flat_N32 | 561,924 | 7 | 0.009 | 437.2 |
| flat_N48 | 1,853,572 | 11 | 0.017 | 1170.6 |
| flat_N64 | 4,343,300 | 10 | 0.035 | 1255.3 |
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
| flat_N16 | 75,140 | 3 | 0.002 | 107.7 |
| flat_N24 | 242,500 | 3 | 0.011 | 66.1 |
| flat_N32 | 561,924 | 3 | 0.013 | 126.9 |
| flat_N48 | 1,853,572 | 3 | 0.043 | 127.9 |
| flat_N64 | 4,343,300 | 3 | 0.098 | 133.4 |
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
| flat_N16 | 75,140 | 913 | 0.135 | 509.9 |
| flat_N24 | 242,500 | 918 | 0.660 | 337.2 |
| flat_N32 | 561,924 | 913 | 0.524 | 979.0 |
| flat_N48 | 1,853,572 | 917 | 1.550 | 1096.8 |
| flat_N64 | 4,343,300 | 916 | 3.557 | 1118.4 |
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
| ss_L2_N8 | 75,140 | 1204 | 0.133 | 677.8 |
| ss_L2_N12 | 242,500 | 903 | 0.286 | 766.5 |
| ss_L2_N16 | 561,924 | 903 | 0.642 | 789.8 |
| ss_L2_N24 | 1,853,572 | 903 | 2.227 | 751.7 |
| ss_L2_N32 | 4,343,300 | 903 | 5.524 | 710.0 |
| ss_L4_N4 | 75,140 | 903 | 0.097 | 702.4 |
| ss_L4_N6 | 242,500 | 903 | 0.257 | 852.9 |
| ss_L4_N8 | 561,924 | 875 | 0.561 | 875.8 |
| ss_L4_N12 | 1,853,572 | 903 | 2.038 | 821.2 |
| ss_L4_N16 | 4,343,300 | 903 | 4.654 | 842.7 |

### `sscvfem::block_diag`

| run | dof | calls | seconds | MDOF/s |
|---|---|---|---|---|
| flat_N16 | 75,140 | -- | -- | -- |
| flat_N24 | 242,500 | -- | -- | -- |
| flat_N32 | 561,924 | -- | -- | -- |
| flat_N48 | 1,853,572 | -- | -- | -- |
| flat_N64 | 4,343,300 | -- | -- | -- |
| ss_L2_N8 | 75,140 | 4 | 0.001 | 276.3 |
| ss_L2_N12 | 242,500 | 3 | 0.003 | 290.5 |
| ss_L2_N16 | 561,924 | 3 | 0.006 | 267.0 |
| ss_L2_N24 | 1,853,572 | 3 | 0.026 | 210.5 |
| ss_L2_N32 | 4,343,300 | 3 | 0.049 | 268.3 |
| ss_L4_N4 | 75,140 | 3 | 0.001 | 278.7 |
| ss_L4_N6 | 242,500 | 3 | 0.002 | 332.2 |
| ss_L4_N8 | 561,924 | 3 | 0.005 | 327.8 |
| ss_L4_N12 | 1,853,572 | 3 | 0.022 | 257.4 |
| ss_L4_N16 | 4,343,300 | 3 | 0.044 | 295.1 |

### `sscvfem::nodal_grad_strided`

| run | dof | calls | seconds | MDOF/s |
|---|---|---|---|---|
| flat_N16 | 75,140 | -- | -- | -- |
| flat_N24 | 242,500 | -- | -- | -- |
| flat_N32 | 561,924 | -- | -- | -- |
| flat_N48 | 1,853,572 | -- | -- | -- |
| flat_N64 | 4,343,300 | -- | -- | -- |
| ss_L2_N8 | 75,140 | 1221 | 0.090 | 1016.4 |
| ss_L2_N12 | 242,500 | 913 | 0.121 | 1825.0 |
| ss_L2_N16 | 561,924 | 913 | 0.259 | 1978.6 |
| ss_L2_N24 | 1,853,572 | 914 | 0.926 | 1830.3 |
| ss_L2_N32 | 4,343,300 | 914 | 2.594 | 1530.4 |
| ss_L4_N4 | 75,140 | 913 | 0.066 | 1045.9 |
| ss_L4_N6 | 242,500 | 913 | 0.123 | 1807.4 |
| ss_L4_N8 | 561,924 | 885 | 0.225 | 2213.2 |
| ss_L4_N12 | 1,853,572 | 920 | 0.718 | 2376.0 |
| ss_L4_N16 | 4,343,300 | 918 | 1.792 | 2224.5 |


## Provenance

| field | value |
|---|---|
| generated | 2026-09-10 07:45:50 |
| configurations | 15 |
| machines | nid006546 |
