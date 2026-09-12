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
| flat_N16 | 75,140 | apply_jacobian_action_packed | 163 | 76% | 10% | 5% | 7% |
| flat_N24 | 242,500 | apply_jacobian_action_packed | 519 | 75% | 10% | 7% | 7% |
| flat_N32 | 561,924 | apply_jacobian_action_packed | 1205 | 53% | 26% | 8% | 5% |
| flat_N48 | 1,853,572 | apply_jacobian_action_packed | 1100 | 70% | 15% | 8% | 2% |
| flat_N64 | 4,343,300 | apply_jacobian_action_packed | 1019 | 74% | 14% | 7% | 1% |
| ss_L2_N8 | 75,140 | apply_macro_local_hoisted | 666 | 39% | 22% | 0% | 15% |
| ss_L2_N12 | 242,500 | apply_macro_local_hoisted | 834 | 64% | 23% | 0% | 11% |
| ss_L2_N16 | 561,924 | apply_macro_local_hoisted | 868 | 64% | 19% | 0% | 5% |
| ss_L2_N24 | 1,853,572 | apply_macro_local_hoisted | 831 | 73% | 21% | 0% | 2% |
| ss_L2_N32 | 4,343,300 | apply_macro_local_hoisted | 773 | 71% | 25% | 0% | 1% |
| ss_L4_N4 | 75,140 | apply_macro_local_hoisted | 699 | 39% | 22% | 0% | 17% |
| ss_L4_N6 | 242,500 | apply_macro_local_hoisted | 921 | 49% | 18% | 0% | 8% |
| ss_L4_N8 | 561,924 | apply_macro_local_hoisted | 950 | 67% | 19% | 0% | 5% |
| ss_L4_N12 | 1,853,572 | apply_macro_local_hoisted | 911 | 72% | 19% | 0% | 2% |
| ss_L4_N16 | 4,343,300 | apply_macro_local_hoisted | 894 | 75% | 22% | 0% | 1% |
| ss_L8p_N2 | 75,140 | apply_macro_local_hoisted | 386 | 60% | 25% | 0% | 14% |
| ss_L8p_N3 | 242,500 | apply_macro_local_hoisted | 702 | 64% | 16% | 0% | 8% |
| ss_L8p_N4 | 561,924 | apply_macro_local_hoisted | 898 | 81% | 12% | 0% | 6% |
| ss_L8p_N6 | 1,853,572 | apply_macro_local_hoisted | 957 | 84% | 11% | 0% | 2% |
| ss_L8p_N8 | 4,343,300 | apply_macro_local_hoisted | 959 | 87% | 10% | 0% | 1% |

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

75,140 dof, 72 threads, nid006544. 0.570 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `cvfem_hex8_ns_steady::apply_jacobian_action_packed` | element sweep | 903 | 0.417 | 461.3 | 162.9 | 73.1% |
| `cvfem_hex8_ns_steady::nodal_grad_strided` | nodal gradient | 913 | 0.057 | 62.9 | 1193.8 | 10.1% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.041 | 45.3 | 1657.8 | 7.2% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action` | boundary | 903 | 0.027 | 30.3 | 2478.1 | 4.8% |
| `cvfem_hex8_ns_steady::assemble_block_diag` | element sweep | 3 | 0.016 | 5248.9 | 14.3 | 2.8% |
| `SFC::reorder` | other | 1 | 0.003 | 2916.8 | 25.8 | 0.5% |
| `cvfem_hex8_ns_steady::apply_residual_packed` | element sweep | 7 | 0.003 | 412.6 | 182.1 | 0.5% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_residual` | boundary | 7 | 0.002 | 283.0 | 265.5 | 0.3% |
| `Mesh::initialize_node_to_node_graph` | setup | 1 | 0.001 | 1283.4 | 58.5 | 0.2% |
| `create_crs_graph_mem_conservative` | setup | 1 | 0.001 | 1261.0 | 59.6 | 0.2% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 49.8 | 1509.0 | 0.1% |
| `cvfem_hex8_ns_steady::precompute_element_bsr_slots` | setup | 1 | 0.000 | 339.3 | 221.5 | 0.1% |
| `create_n2e` | other | 1 | 0.000 | 337.4 | 222.7 | 0.1% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 42.9 | 1752.3 | 0.1% |
| `cvfem_hex8_ns_steady::build_rc_coeff` | other | 903 | 0.000 | 0.2 | 453892.1 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 98.7 | 761.3 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.435 | 76.3% |
| nodal gradient | 0.057 | 10.1% |
| constraints | 0.042 | 7.3% |
| boundary | 0.029 | 5.2% |
| other | 0.003 | 0.6% |
| setup | 0.003 | 0.5% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 0.927 | 309038.0 |
| `Function::apply` | 903 | 0.553 | 611.9 |
| `CVFEMNavierStokes::apply` | 903 | 0.511 | 565.8 |
| `cvfem_hex8_ns_steady::apply_jacobian_action_accumulate` | 903 | 0.500 | 553.8 |
| `cvfem_hex8_ns_steady::assemble_nodal_q_grad` | 903 | 0.055 | 60.8 |
| `Function::copy_constrained_dofs` | 903 | 0.041 | 45.6 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.016 | 5444.7 |
| `Function::gradient` | 7 | 0.008 | 1207.7 |


### flat_N24

242,500 dof, 72 threads, nid006544. 0.592 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `cvfem_hex8_ns_steady::apply_jacobian_action_packed` | element sweep | 903 | 0.422 | 466.8 | 519.4 | 71.2% |
| `cvfem_hex8_ns_steady::nodal_grad_strided` | nodal gradient | 913 | 0.057 | 62.9 | 3855.9 | 9.7% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.041 | 45.5 | 5328.4 | 6.9% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action` | boundary | 903 | 0.036 | 39.6 | 6123.5 | 6.0% |
| `cvfem_hex8_ns_steady::assemble_block_diag` | element sweep | 3 | 0.017 | 5504.8 | 44.1 | 2.8% |
| `SFC::reorder` | other | 1 | 0.005 | 4955.5 | 48.9 | 0.8% |
| `cvfem_hex8_ns_steady::apply_residual_packed` | element sweep | 7 | 0.003 | 467.9 | 518.3 | 0.6% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_residual` | boundary | 7 | 0.003 | 463.0 | 523.8 | 0.5% |
| `Mesh::initialize_node_to_node_graph` | setup | 1 | 0.002 | 2229.2 | 108.8 | 0.4% |
| `create_crs_graph_mem_conservative` | setup | 1 | 0.002 | 2217.5 | 109.4 | 0.4% |
| `create_n2e` | other | 1 | 0.001 | 1277.9 | 189.8 | 0.2% |
| `cvfem_hex8_ns_steady::precompute_element_bsr_slots` | setup | 1 | 0.001 | 1046.2 | 231.8 | 0.2% |
| `cvfem_hex8_ns_steady::build_rc_coeff` | other | 903 | 0.000 | 0.5 | 538369.5 | 0.1% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 46.2 | 5254.5 | 0.1% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 43.4 | 5593.0 | 0.1% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 104.9 | 2311.6 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.441 | 74.6% |
| nodal gradient | 0.057 | 9.7% |
| constraints | 0.042 | 7.1% |
| boundary | 0.039 | 6.6% |
| other | 0.007 | 1.1% |
| setup | 0.005 | 0.9% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 1.502 | 500506.7 |
| `Function::apply` | 903 | 0.567 | 628.3 |
| `CVFEMNavierStokes::apply` | 903 | 0.525 | 581.8 |
| `cvfem_hex8_ns_steady::apply_jacobian_action_accumulate` | 903 | 0.513 | 567.9 |
| `cvfem_hex8_ns_steady::assemble_nodal_q_grad` | 903 | 0.054 | 59.8 |
| `Function::copy_constrained_dofs` | 903 | 0.041 | 45.8 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.018 | 6060.2 |
| `CVFEMNavierStokes::initialize` | 1 | 0.014 | 13529.3 |


### flat_N32

561,924 dof, 72 threads, nid006544. 0.309 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `cvfem_hex8_ns_steady::apply_jacobian_action_packed` | element sweep | 326 | 0.152 | 466.1 | 1205.5 | 49.2% |
| `cvfem_hex8_ns_steady::nodal_grad_strided` | nodal gradient | 331 | 0.081 | 246.0 | 2284.4 | 26.4% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action` | boundary | 326 | 0.018 | 56.7 | 9911.0 | 6.0% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 326 | 0.015 | 46.3 | 12149.6 | 4.9% |
| `cvfem_hex8_ns_steady::assemble_block_diag` | element sweep | 2 | 0.009 | 4557.7 | 123.3 | 3.0% |
| `SFC::reorder` | other | 1 | 0.008 | 8453.6 | 66.5 | 2.7% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_residual` | boundary | 3 | 0.006 | 1867.8 | 300.8 | 1.8% |
| `Mesh::initialize_node_to_node_graph` | setup | 1 | 0.005 | 5159.8 | 108.9 | 1.7% |
| `create_crs_graph_mem_conservative` | setup | 1 | 0.005 | 5150.1 | 109.1 | 1.7% |
| `create_n2e` | other | 1 | 0.003 | 3181.9 | 176.6 | 1.0% |
| `cvfem_hex8_ns_steady::precompute_element_bsr_slots` | setup | 1 | 0.002 | 2239.5 | 250.9 | 0.7% |
| `cvfem_hex8_ns_steady::apply_residual_packed` | element sweep | 3 | 0.002 | 518.1 | 1084.6 | 0.5% |
| `cvfem_hex8_ns_steady::build_rc_coeff` | other | 326 | 0.001 | 2.5 | 222772.6 | 0.3% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 186.4 | 3013.9 | 0.1% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 184.8 | 3041.1 | 0.1% |
| `DirichletConditions::gradient` | constraints | 3 | 0.000 | 46.9 | 11984.1 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.163 | 52.6% |
| nodal gradient | 0.081 | 26.4% |
| boundary | 0.024 | 7.8% |
| constraints | 0.016 | 5.1% |
| setup | 0.013 | 4.1% |
| other | 0.012 | 4.0% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 2 | 0.960 | 479992.5 |
| `Function::apply` | 326 | 0.216 | 662.9 |
| `CVFEMNavierStokes::apply` | 326 | 0.201 | 615.4 |
| `cvfem_hex8_ns_steady::apply_jacobian_action_accumulate` | 326 | 0.195 | 598.0 |
| `Function::gradient` | 3 | 0.066 | 21958.0 |
| `CVFEMNavierStokes::gradient` | 3 | 0.066 | 21908.9 |
| `cvfem_hex8_ns_steady::apply_residual` | 3 | 0.065 | 21813.5 |
| `cvfem_hex8_ns_steady::assemble_nodal_p_grad` | 5 | 0.058 | 11685.6 |


### flat_N48

1,853,572 dof, 72 threads, nid006544. 2.239 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `cvfem_hex8_ns_steady::apply_jacobian_action_packed` | element sweep | 903 | 1.521 | 1684.7 | 1100.2 | 67.9% |
| `cvfem_hex8_ns_steady::nodal_grad_strided` | nodal gradient | 917 | 0.330 | 359.9 | 5150.3 | 14.7% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action` | boundary | 903 | 0.168 | 185.9 | 9969.0 | 7.5% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.048 | 53.7 | 34511.3 | 2.2% |
| `Mesh::initialize_node_to_node_graph` | setup | 1 | 0.034 | 33693.1 | 55.0 | 1.5% |
| `create_crs_graph_mem_conservative` | setup | 1 | 0.034 | 33678.8 | 55.0 | 1.5% |
| `cvfem_hex8_ns_steady::assemble_block_diag` | element sweep | 3 | 0.030 | 9993.9 | 185.5 | 1.3% |
| `SFC::reorder` | other | 1 | 0.024 | 23525.2 | 78.8 | 1.1% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_residual` | boundary | 11 | 0.014 | 1291.4 | 1435.4 | 0.6% |
| `cvfem_hex8_ns_steady::apply_residual_packed` | element sweep | 11 | 0.013 | 1182.9 | 1566.9 | 0.6% |
| `create_n2e` | other | 1 | 0.011 | 11285.5 | 164.2 | 0.5% |
| `cvfem_hex8_ns_steady::precompute_element_bsr_slots` | setup | 1 | 0.007 | 7113.0 | 260.6 | 0.3% |
| `cvfem_hex8_ns_steady::build_rc_coeff` | other | 903 | 0.003 | 2.9 | 649487.8 | 0.1% |
| `DirichletConditions::gradient` | constraints | 11 | 0.001 | 58.3 | 31779.6 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 11 | 0.000 | 43.3 | 42845.2 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 425.6 | 4355.4 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 1.564 | 69.9% |
| nodal gradient | 0.330 | 14.7% |
| boundary | 0.182 | 8.1% |
| setup | 0.074 | 3.3% |
| constraints | 0.051 | 2.3% |
| other | 0.037 | 1.7% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 8.699 | 2899616.7 |
| `Function::apply` | 903 | 2.093 | 2317.4 |
| `CVFEMNavierStokes::apply` | 903 | 2.041 | 2260.3 |
| `cvfem_hex8_ns_steady::apply_jacobian_action_accumulate` | 903 | 1.967 | 2178.3 |
| `cvfem_hex8_ns_steady::assemble_nodal_q_grad` | 903 | 0.271 | 300.2 |
| `CVFEMNavierStokes::initialize` | 1 | 0.124 | 124138.0 |
| `Function::gradient` | 11 | 0.091 | 8302.0 |
| `CVFEMNavierStokes::gradient` | 11 | 0.091 | 8240.0 |


### flat_N64

4,343,300 dof, 72 threads, nid006544. 5.319 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `cvfem_hex8_ns_steady::apply_jacobian_action_packed` | element sweep | 903 | 3.851 | 4264.3 | 1018.5 | 72.4% |
| `cvfem_hex8_ns_steady::nodal_grad_strided` | nodal gradient | 913 | 0.754 | 825.3 | 5262.6 | 14.2% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action` | boundary | 903 | 0.350 | 388.0 | 11192.8 | 6.6% |
| `cvfem_hex8_ns_steady::assemble_block_diag` | element sweep | 3 | 0.067 | 22438.7 | 193.6 | 1.3% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.053 | 58.8 | 73807.4 | 1.0% |
| `SFC::reorder` | other | 1 | 0.053 | 53015.2 | 81.9 | 1.0% |
| `Mesh::initialize_node_to_node_graph` | setup | 1 | 0.044 | 43811.3 | 99.1 | 0.8% |
| `create_crs_graph_mem_conservative` | setup | 1 | 0.044 | 43788.4 | 99.2 | 0.8% |
| `create_n2e` | other | 1 | 0.033 | 33249.4 | 130.6 | 0.6% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_residual` | boundary | 7 | 0.025 | 3502.1 | 1240.2 | 0.5% |
| `cvfem_hex8_ns_steady::apply_residual_packed` | element sweep | 7 | 0.020 | 2895.0 | 1500.3 | 0.4% |
| `cvfem_hex8_ns_steady::precompute_element_bsr_slots` | setup | 1 | 0.017 | 16757.5 | 259.2 | 0.3% |
| `cvfem_hex8_ns_steady::build_rc_coeff` | other | 903 | 0.005 | 6.1 | 716872.1 | 0.1% |
| `Function::constraints_mask` | constraints | 1 | 0.001 | 818.3 | 5308.0 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.001 | 814.0 | 5336.0 | 0.0% |
| `DirichletConditions::gradient` | constraints | 7 | 0.001 | 72.9 | 59560.9 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 3.938 | 74.0% |
| nodal gradient | 0.754 | 14.2% |
| boundary | 0.375 | 7.0% |
| setup | 0.104 | 2.0% |
| other | 0.092 | 1.7% |
| constraints | 0.056 | 1.0% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 19.088 | 6362833.3 |
| `Function::apply` | 903 | 5.198 | 5756.9 |
| `CVFEMNavierStokes::apply` | 903 | 5.141 | 5693.4 |
| `cvfem_hex8_ns_steady::apply_jacobian_action_accumulate` | 903 | 4.954 | 5485.9 |
| `cvfem_hex8_ns_steady::assemble_nodal_q_grad` | 903 | 0.740 | 820.0 |
| `CVFEMNavierStokes::initialize` | 1 | 0.228 | 227950.0 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.102 | 33849.3 |
| `Function::gradient` | 7 | 0.064 | 9205.8 |


### ss_L2_N8

75,140 dof, 72 threads, nid006544. 0.260 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 0.102 | 112.8 | 665.9 | 39.2% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 913 | 0.057 | 62.8 | 1197.0 | 22.0% |
| `to_semistructured` | other | 1 | 0.057 | 57129.9 | 1.3 | 22.0% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.039 | 43.2 | 1740.6 | 15.0% |
| `SFC::reorder` | other | 1 | 0.002 | 2312.9 | 32.5 | 0.9% |
| `sscvfem::block_diag` | element sweep | 3 | 0.001 | 267.3 | 281.1 | 0.3% |
| `sscvfem::build_scatter` | setup | 1 | 0.001 | 649.7 | 115.7 | 0.2% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 44.6 | 1684.1 | 0.1% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 40.8 | 1843.0 | 0.1% |
| `create_dual_graph` | other | 1 | 0.000 | 258.4 | 290.7 | 0.1% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 97.8 | 768.7 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 26.3 | 2852.1 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 47.4 | 1583.7 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 46.5 | 1616.2 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.0 | 1884698.8 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.0 | 0.0 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.103 | 39.5% |
| other | 0.060 | 23.0% |
| nodal gradient | 0.057 | 22.0% |
| constraints | 0.040 | 15.3% |
| setup | 0.001 | 0.2% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 0.583 | 194357.7 |
| `Function::apply` | 903 | 0.209 | 231.5 |
| `CVFEMNavierStokes::apply` | 903 | 0.170 | 187.8 |
| `sscvfem::apply` | 903 | 0.159 | 176.2 |
| `sscvfem::nodal_q_grad` | 903 | 0.057 | 62.9 |
| `Function::copy_constrained_dofs` | 903 | 0.039 | 43.4 |
| `Function::gradient` | 7 | 0.002 | 313.7 |
| `CVFEMNavierStokes::gradient` | 7 | 0.002 | 267.9 |


### ss_L2_N12

242,500 dof, 72 threads, nid006544. 0.414 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 0.263 | 290.8 | 834.0 | 63.4% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 913 | 0.095 | 103.6 | 2341.2 | 22.8% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.044 | 48.4 | 5011.8 | 10.5% |
| `to_semistructured` | other | 1 | 0.005 | 5002.3 | 48.5 | 1.2% |
| `SFC::reorder` | other | 1 | 0.002 | 2399.7 | 101.1 | 0.6% |
| `sscvfem::block_diag` | element sweep | 3 | 0.002 | 734.2 | 330.3 | 0.5% |
| `sscvfem::build_scatter` | setup | 1 | 0.002 | 1981.7 | 122.4 | 0.5% |
| `create_dual_graph` | other | 1 | 0.001 | 873.6 | 277.6 | 0.2% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 56.1 | 4322.9 | 0.1% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 42.7 | 5673.2 | 0.1% |
| `create_n2e` | other | 2 | 0.000 | 81.2 | 2987.1 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 104.2 | 2327.5 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 103.0 | 2354.4 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 76.5 | 3168.6 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.0 | 6248020.5 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.1 | 3559916.7 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.265 | 63.9% |
| nodal gradient | 0.095 | 22.8% |
| constraints | 0.045 | 10.8% |
| other | 0.008 | 2.0% |
| setup | 0.002 | 0.5% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 1.358 | 452520.0 |
| `Function::apply` | 903 | 0.413 | 457.8 |
| `CVFEMNavierStokes::apply` | 903 | 0.369 | 408.7 |
| `sscvfem::apply` | 903 | 0.356 | 394.6 |
| `sscvfem::nodal_q_grad` | 903 | 0.093 | 103.2 |
| `Function::copy_constrained_dofs` | 903 | 0.044 | 48.7 |
| `Function::gradient` | 7 | 0.004 | 584.4 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.004 | 1265.4 |


### ss_L2_N16

561,924 dof, 72 threads, nid006544. 0.919 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 0.585 | 647.7 | 867.5 | 63.7% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 913 | 0.173 | 189.2 | 2969.9 | 18.8% |
| `to_semistructured` | other | 1 | 0.102 | 102323.0 | 5.5 | 11.1% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.042 | 46.3 | 12128.2 | 4.6% |
| `sscvfem::block_diag` | element sweep | 3 | 0.006 | 1847.7 | 304.1 | 0.6% |
| `sscvfem::build_scatter` | setup | 1 | 0.005 | 4863.3 | 115.5 | 0.5% |
| `SFC::reorder` | other | 1 | 0.003 | 2816.0 | 199.5 | 0.3% |
| `create_dual_graph` | other | 1 | 0.002 | 2176.3 | 258.2 | 0.2% |
| `create_n2e` | other | 2 | 0.000 | 219.9 | 2554.9 | 0.0% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 52.0 | 10804.3 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 43.2 | 13000.9 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 184.1 | 3053.0 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 182.9 | 3072.9 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 113.2 | 4961.8 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.0 | 16627040.5 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.0 | 0.0 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.590 | 64.3% |
| nodal gradient | 0.173 | 18.8% |
| other | 0.108 | 11.7% |
| constraints | 0.043 | 4.7% |
| setup | 0.005 | 0.5% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 2.823 | 940853.3 |
| `Function::apply` | 903 | 0.814 | 901.2 |
| `CVFEMNavierStokes::apply` | 903 | 0.771 | 853.9 |
| `sscvfem::apply` | 903 | 0.755 | 836.5 |
| `sscvfem::nodal_q_grad` | 903 | 0.170 | 188.0 |
| `Function::copy_constrained_dofs` | 903 | 0.042 | 46.7 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.009 | 3068.4 |
| `Function::gradient` | 7 | 0.008 | 1211.0 |


### ss_L2_N24

1,853,572 dof, 72 threads, nid006544. 2.779 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 2.015 | 2231.2 | 830.7 | 72.5% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 914 | 0.582 | 636.3 | 2912.9 | 20.9% |
| `to_semistructured` | other | 1 | 0.080 | 79556.7 | 23.3 | 2.9% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.049 | 54.2 | 34176.5 | 1.8% |
| `sscvfem::block_diag` | element sweep | 3 | 0.019 | 6478.2 | 286.1 | 0.7% |
| `sscvfem::build_scatter` | setup | 1 | 0.018 | 18083.1 | 102.5 | 0.7% |
| `create_dual_graph` | other | 1 | 0.008 | 8065.7 | 229.8 | 0.3% |
| `SFC::reorder` | other | 1 | 0.005 | 4955.5 | 374.0 | 0.2% |
| `create_n2e` | other | 2 | 0.002 | 902.2 | 2054.6 | 0.1% |
| `DirichletConditions::gradient` | constraints | 8 | 0.000 | 55.6 | 33313.1 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 441.8 | 4195.6 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 439.9 | 4213.8 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 8 | 0.000 | 42.7 | 43432.7 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 123.0 | 15066.8 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.1 | 20231490.9 | 0.0% |
| `sscvfem::apply_transient` | transient | 8 | 0.000 | 0.1 | 31097788.1 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 2.034 | 73.2% |
| nodal gradient | 0.582 | 20.9% |
| other | 0.094 | 3.4% |
| constraints | 0.051 | 1.8% |
| setup | 0.018 | 0.7% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 9.239 | 3079800.0 |
| `Function::apply` | 903 | 2.708 | 2999.3 |
| `CVFEMNavierStokes::apply` | 903 | 2.657 | 2942.1 |
| `sscvfem::apply` | 903 | 2.590 | 2867.9 |
| `sscvfem::nodal_q_grad` | 903 | 0.573 | 634.1 |
| `Function::copy_constrained_dofs` | 903 | 0.050 | 55.4 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.034 | 11445.7 |
| `Function::gradient` | 8 | 0.027 | 3336.6 |


### ss_L2_N32

4,343,300 dof, 72 threads, nid006544. 7.170 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 5.077 | 5622.3 | 772.5 | 70.8% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 919 | 1.757 | 1911.5 | 2272.2 | 24.5% |
| `to_semistructured` | other | 1 | 0.129 | 129339.0 | 33.6 | 1.8% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.080 | 88.5 | 49067.1 | 1.1% |
| `sscvfem::block_diag` | element sweep | 3 | 0.048 | 15854.8 | 273.9 | 0.7% |
| `sscvfem::build_scatter` | setup | 1 | 0.043 | 42918.2 | 101.2 | 0.6% |
| `create_dual_graph` | other | 1 | 0.020 | 19644.7 | 221.1 | 0.3% |
| `SFC::reorder` | other | 1 | 0.008 | 8488.9 | 511.6 | 0.1% |
| `create_n2e` | other | 2 | 0.005 | 2327.4 | 1866.1 | 0.1% |
| `DirichletConditions::gradient` | constraints | 13 | 0.001 | 110.5 | 39293.6 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.001 | 771.8 | 5627.8 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.001 | 767.9 | 5655.7 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 13 | 0.001 | 44.8 | 96979.0 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 134.9 | 32185.7 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.1 | 51567603.9 | 0.0% |
| `sscvfem::apply_transient` | transient | 13 | 0.000 | 0.1 | 47364628.5 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 5.125 | 71.5% |
| nodal gradient | 1.757 | 24.5% |
| other | 0.162 | 2.3% |
| constraints | 0.084 | 1.2% |
| setup | 0.043 | 0.6% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 21.012 | 7003900.0 |
| `Function::apply` | 903 | 7.074 | 7833.6 |
| `CVFEMNavierStokes::apply` | 903 | 6.990 | 7740.8 |
| `sscvfem::apply` | 903 | 6.806 | 7537.6 |
| `sscvfem::nodal_q_grad` | 903 | 1.725 | 1909.8 |
| `Function::gradient` | 13 | 0.102 | 7822.2 |
| `CVFEMNavierStokes::gradient` | 13 | 0.100 | 7707.6 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.085 | 28413.9 |


### ss_L4_N4

75,140 dof, 72 threads, nid006544. 0.417 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 1505 | 0.162 | 107.5 | 699.1 | 38.8% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 1525 | 0.093 | 61.2 | 1228.2 | 22.4% |
| `to_semistructured` | other | 1 | 0.087 | 86678.7 | 0.9 | 20.8% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 1505 | 0.069 | 46.1 | 1629.5 | 16.7% |
| `SFC::reorder` | other | 1 | 0.002 | 2363.7 | 31.8 | 0.6% |
| `sscvfem::block_diag` | element sweep | 5 | 0.001 | 234.9 | 319.9 | 0.3% |
| `DirichletConditions::apply_value` | constraints | 15 | 0.001 | 44.1 | 1703.6 | 0.2% |
| `DirichletConditions::gradient` | constraints | 15 | 0.001 | 43.7 | 1719.7 | 0.2% |
| `sscvfem::build_scatter` | setup | 1 | 0.000 | 351.9 | 213.5 | 0.1% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 83.4 | 900.5 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 1505 | 0.000 | 0.0 | 1817303.4 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 47.7 | 1575.8 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 46.3 | 1624.5 | 0.0% |
| `create_dual_graph` | other | 1 | 0.000 | 33.9 | 2219.4 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 3.9 | 19100.6 | 0.0% |
| `sscvfem::apply_transient` | transient | 15 | 0.000 | 0.0 | 2363700.8 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.163 | 39.1% |
| nodal gradient | 0.093 | 22.4% |
| other | 0.089 | 21.4% |
| constraints | 0.071 | 17.0% |
| setup | 0.000 | 0.1% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 5 | 0.970 | 193909.0 |
| `Function::apply` | 1505 | 0.343 | 227.8 |
| `CVFEMNavierStokes::apply` | 1505 | 0.273 | 181.2 |
| `sscvfem::apply` | 1505 | 0.255 | 169.1 |
| `sscvfem::nodal_q_grad` | 1505 | 0.092 | 61.2 |
| `Function::copy_constrained_dofs` | 1505 | 0.070 | 46.3 |
| `Function::gradient` | 15 | 0.005 | 303.3 |
| `CVFEMNavierStokes::gradient` | 15 | 0.004 | 258.6 |


### ss_L4_N6

242,500 dof, 72 threads, nid006544. 0.486 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 0.238 | 263.4 | 920.6 | 48.9% |
| `to_semistructured` | other | 1 | 0.112 | 112135.0 | 2.2 | 23.1% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 913 | 0.090 | 98.3 | 2466.9 | 18.5% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.040 | 44.7 | 5423.6 | 8.3% |
| `SFC::reorder` | other | 1 | 0.002 | 2131.0 | 113.8 | 0.4% |
| `sscvfem::block_diag` | element sweep | 3 | 0.002 | 670.4 | 361.7 | 0.4% |
| `sscvfem::build_scatter` | setup | 1 | 0.001 | 1016.1 | 238.6 | 0.2% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 43.3 | 5601.8 | 0.1% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 42.1 | 5765.0 | 0.1% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 140.9 | 1721.0 | 0.0% |
| `create_dual_graph` | other | 1 | 0.000 | 108.0 | 2245.3 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 104.2 | 2327.5 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 103.0 | 2354.4 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.0 | 5669496.5 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 10.7 | 22602.6 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.0 | 7119818.5 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.240 | 49.3% |
| other | 0.114 | 23.5% |
| nodal gradient | 0.090 | 18.5% |
| constraints | 0.041 | 8.5% |
| setup | 0.001 | 0.2% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 1.314 | 437993.3 |
| `Function::apply` | 903 | 0.381 | 421.9 |
| `CVFEMNavierStokes::apply` | 903 | 0.340 | 376.5 |
| `sscvfem::apply` | 903 | 0.327 | 362.0 |
| `sscvfem::nodal_q_grad` | 903 | 0.089 | 98.0 |
| `Function::copy_constrained_dofs` | 903 | 0.041 | 45.0 |
| `Function::gradient` | 7 | 0.004 | 586.6 |
| `CVFEMNavierStokes::gradient` | 7 | 0.004 | 542.2 |


### ss_L4_N8

561,924 dof, 72 threads, nid006544. 0.807 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 0.534 | 591.4 | 950.1 | 66.2% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 913 | 0.157 | 172.1 | 3264.6 | 19.5% |
| `to_semistructured` | other | 1 | 0.063 | 63309.4 | 8.9 | 7.8% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.041 | 45.6 | 12333.7 | 5.1% |
| `sscvfem::block_diag` | element sweep | 3 | 0.005 | 1737.4 | 323.4 | 0.6% |
| `sscvfem::build_scatter` | setup | 1 | 0.003 | 2548.2 | 220.5 | 0.3% |
| `SFC::reorder` | other | 1 | 0.002 | 2231.6 | 251.8 | 0.3% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 45.7 | 12302.9 | 0.0% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 45.2 | 12442.1 | 0.0% |
| `create_dual_graph` | other | 1 | 0.000 | 258.7 | 2172.2 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 186.4 | 3013.9 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 185.3 | 3033.3 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 112.8 | 4982.8 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 25.7 | 21823.0 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.0 | 12898550.6 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.0 | 16498131.4 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.539 | 66.8% |
| nodal gradient | 0.157 | 19.5% |
| other | 0.066 | 8.2% |
| constraints | 0.042 | 5.2% |
| setup | 0.003 | 0.3% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 2.760 | 919950.0 |
| `Function::apply` | 903 | 0.747 | 827.1 |
| `CVFEMNavierStokes::apply` | 903 | 0.705 | 780.6 |
| `sscvfem::apply` | 903 | 0.689 | 763.0 |
| `sscvfem::nodal_q_grad` | 903 | 0.154 | 170.9 |
| `Function::copy_constrained_dofs` | 903 | 0.041 | 45.9 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.009 | 2870.8 |
| `Function::gradient` | 7 | 0.007 | 1039.5 |


### ss_L4_N12

1,853,572 dof, 72 threads, nid006544. 0.892 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 310 | 0.630 | 2033.6 | 911.5 | 70.7% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 315 | 0.170 | 540.1 | 3431.8 | 19.1% |
| `to_semistructured` | other | 1 | 0.046 | 46132.3 | 40.2 | 5.2% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 310 | 0.018 | 58.6 | 31642.1 | 2.0% |
| `sscvfem::block_diag` | element sweep | 2 | 0.012 | 6092.6 | 304.2 | 1.4% |
| `sscvfem::build_scatter` | setup | 1 | 0.009 | 9438.5 | 196.4 | 1.1% |
| `SFC::reorder` | other | 1 | 0.003 | 2612.3 | 709.5 | 0.3% |
| `create_dual_graph` | other | 1 | 0.001 | 962.3 | 1926.3 | 0.1% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 431.5 | 4295.3 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 428.7 | 4323.9 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 84.5 | 21930.7 | 0.0% |
| `DirichletConditions::gradient` | constraints | 3 | 0.000 | 52.5 | 35284.9 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 3 | 0.000 | 41.8 | 44341.0 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 116.6 | 15898.6 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 310 | 0.000 | 0.1 | 23862132.9 | 0.0% |
| `sscvfem::apply_transient` | transient | 3 | 0.000 | 0.0 | 0.0 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.643 | 72.1% |
| nodal gradient | 0.170 | 19.1% |
| other | 0.050 | 5.6% |
| constraints | 0.019 | 2.2% |
| setup | 0.009 | 1.1% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 2 | 3.059 | 1529570.0 |
| `Function::apply` | 310 | 0.836 | 2696.9 |
| `CVFEMNavierStokes::apply` | 310 | 0.817 | 2635.8 |
| `sscvfem::apply` | 310 | 0.795 | 2566.0 |
| `sscvfem::nodal_q_grad` | 310 | 0.164 | 530.1 |
| `CVFEMNavierStokes::hessian_block_diag` | 2 | 0.022 | 10896.8 |
| `Function::copy_constrained_dofs` | 310 | 0.019 | 59.7 |
| `Function::gradient` | 3 | 0.011 | 3739.0 |


### ss_L4_N16

4,343,300 dof, 72 threads, nid006544. 5.872 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 4.389 | 4860.1 | 893.7 | 74.7% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 916 | 1.287 | 1405.4 | 3090.4 | 21.9% |
| `to_semistructured` | other | 1 | 0.066 | 66133.5 | 65.7 | 1.1% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.056 | 62.5 | 69525.9 | 1.0% |
| `sscvfem::block_diag` | element sweep | 3 | 0.041 | 13544.7 | 320.7 | 0.7% |
| `sscvfem::build_scatter` | setup | 1 | 0.024 | 24306.3 | 178.7 | 0.4% |
| `SFC::reorder` | other | 1 | 0.003 | 3034.8 | 1431.2 | 0.1% |
| `create_dual_graph` | other | 1 | 0.002 | 2166.5 | 2004.7 | 0.0% |
| `DirichletConditions::gradient` | constraints | 10 | 0.001 | 78.9 | 55069.9 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.001 | 776.3 | 5594.9 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.001 | 774.9 | 5605.3 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 237.9 | 18253.6 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 10 | 0.000 | 43.0 | 101094.0 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 126.8 | 34242.6 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.1 | 50000190.0 | 0.0% |
| `sscvfem::apply_transient` | transient | 10 | 0.000 | 0.1 | 45542816.5 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 4.429 | 75.4% |
| nodal gradient | 1.287 | 21.9% |
| other | 0.072 | 1.2% |
| constraints | 0.059 | 1.0% |
| setup | 0.024 | 0.4% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 19.718 | 6572766.7 |
| `Function::apply` | 903 | 5.897 | 6530.7 |
| `CVFEMNavierStokes::apply` | 903 | 5.837 | 6463.6 |
| `sscvfem::apply` | 903 | 5.657 | 6264.1 |
| `sscvfem::nodal_q_grad` | 903 | 1.264 | 1399.3 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.077 | 25618.0 |
| `Function::gradient` | 10 | 0.070 | 6975.0 |
| `CVFEMNavierStokes::gradient` | 10 | 0.069 | 6892.1 |


### ss_L8p_N2

75,140 dof, 72 threads, nid006544. 0.296 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 0.176 | 194.8 | 385.8 | 59.3% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 917 | 0.073 | 79.9 | 940.3 | 24.7% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.040 | 43.8 | 1715.4 | 13.3% |
| `to_semistructured` | other | 1 | 0.003 | 2599.0 | 28.9 | 0.9% |
| `SFC::reorder` | other | 1 | 0.002 | 2242.1 | 33.5 | 0.8% |
| `sscvfem::block_diag` | element sweep | 3 | 0.001 | 444.4 | 169.1 | 0.4% |
| `DirichletConditions::gradient` | constraints | 11 | 0.000 | 42.9 | 1753.5 | 0.2% |
| `DirichletConditions::apply_value` | constraints | 11 | 0.000 | 41.1 | 1828.5 | 0.2% |
| `sscvfem::build_scatter` | setup | 2 | 0.000 | 165.8 | 453.1 | 0.1% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 110.6 | 679.2 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 47.0 | 1599.8 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 46.0 | 1633.0 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.0 | 1990133.7 | 0.0% |
| `create_dual_graph` | other | 1 | 0.000 | 6.2 | 12121.5 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 0.7 | 105053.4 | 0.0% |
| `sscvfem::apply_transient` | transient | 11 | 0.000 | 0.0 | 3466753.9 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.177 | 59.8% |
| nodal gradient | 0.073 | 24.7% |
| constraints | 0.041 | 13.7% |
| other | 0.005 | 1.6% |
| setup | 0.000 | 0.1% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 0.674 | 224648.0 |
| `Function::apply` | 903 | 0.300 | 331.7 |
| `CVFEMNavierStokes::apply` | 903 | 0.259 | 287.2 |
| `sscvfem::apply` | 903 | 0.248 | 275.0 |
| `sscvfem::nodal_q_grad` | 903 | 0.072 | 79.7 |
| `Function::copy_constrained_dofs` | 903 | 0.040 | 44.1 |
| `Function::gradient` | 11 | 0.005 | 431.8 |
| `CVFEMNavierStokes::gradient` | 11 | 0.004 | 387.8 |


### ss_L8p_N3

242,500 dof, 72 threads, nid006544. 0.490 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 0.312 | 345.5 | 701.9 | 63.7% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 913 | 0.076 | 83.3 | 2912.5 | 15.5% |
| `to_semistructured` | other | 1 | 0.055 | 55229.9 | 4.4 | 11.3% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.040 | 43.8 | 5531.8 | 8.1% |
| `sscvfem::block_diag` | element sweep | 3 | 0.003 | 889.9 | 272.5 | 0.5% |
| `SFC::reorder` | other | 1 | 0.002 | 2030.1 | 119.5 | 0.4% |
| `sscvfem::build_scatter` | setup | 2 | 0.001 | 532.9 | 455.1 | 0.2% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 44.5 | 5451.6 | 0.1% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 41.6 | 5835.9 | 0.1% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 104.4 | 2322.2 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 103.0 | 2354.4 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 98.5 | 2462.8 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.0 | 5634717.0 | 0.0% |
| `create_dual_graph` | other | 1 | 0.000 | 15.7 | 15410.9 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 1.9 | 127139.7 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.0 | 7119818.5 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.315 | 64.3% |
| nodal gradient | 0.076 | 15.5% |
| other | 0.057 | 11.7% |
| constraints | 0.040 | 8.3% |
| setup | 0.001 | 0.2% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 1.358 | 452640.0 |
| `Function::apply` | 903 | 0.440 | 487.5 |
| `CVFEMNavierStokes::apply` | 903 | 0.400 | 442.9 |
| `sscvfem::apply` | 903 | 0.388 | 429.1 |
| `sscvfem::nodal_q_grad` | 903 | 0.075 | 83.0 |
| `Function::copy_constrained_dofs` | 903 | 0.040 | 44.1 |
| `CVFEMNavierStokes::initialize` | 1 | 0.004 | 4443.9 |
| `Function::gradient` | 7 | 0.004 | 617.0 |


### ss_L8p_N4

561,924 dof, 72 threads, nid006544. 0.702 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 0.565 | 625.8 | 897.9 | 80.5% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 914 | 0.082 | 89.3 | 6289.8 | 11.6% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.040 | 44.7 | 12565.2 | 5.8% |
| `sscvfem::block_diag` | element sweep | 3 | 0.005 | 1805.6 | 311.2 | 0.8% |
| `to_semistructured` | other | 1 | 0.003 | 3222.2 | 174.4 | 0.5% |
| `sscvfem::build_scatter` | setup | 2 | 0.003 | 1354.5 | 414.9 | 0.4% |
| `SFC::reorder` | other | 1 | 0.002 | 2219.9 | 253.1 | 0.3% |
| `DirichletConditions::gradient` | constraints | 8 | 0.000 | 44.1 | 12739.9 | 0.1% |
| `DirichletConditions::apply_value` | constraints | 8 | 0.000 | 42.1 | 13362.9 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 185.3 | 3033.3 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 184.3 | 3049.0 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 122.8 | 4576.5 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.0 | 12302100.8 | 0.0% |
| `create_dual_graph` | other | 1 | 0.000 | 32.7 | 17203.5 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 5.0 | 112232.2 | 0.0% |
| `sscvfem::apply_transient` | transient | 8 | 0.000 | 0.1 | 9427523.5 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.571 | 81.3% |
| nodal gradient | 0.082 | 11.6% |
| constraints | 0.042 | 5.9% |
| other | 0.005 | 0.8% |
| setup | 0.003 | 0.4% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 2.681 | 893753.3 |
| `Function::apply` | 903 | 0.702 | 777.0 |
| `CVFEMNavierStokes::apply` | 903 | 0.660 | 731.3 |
| `sscvfem::apply` | 903 | 0.646 | 715.2 |
| `sscvfem::nodal_q_grad` | 903 | 0.080 | 88.7 |
| `Function::copy_constrained_dofs` | 903 | 0.041 | 45.0 |
| `CVFEMNavierStokes::initialize` | 1 | 0.010 | 10219.1 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.008 | 2792.2 |


### ss_L8p_N6

1,853,572 dof, 72 threads, nid006544. 2.106 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 1.749 | 1937.3 | 956.8 | 83.1% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 919 | 0.224 | 243.8 | 7602.9 | 10.6% |
| `to_semistructured` | other | 1 | 0.053 | 53266.8 | 34.8 | 2.5% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.046 | 50.4 | 36751.5 | 2.2% |
| `sscvfem::block_diag` | element sweep | 3 | 0.019 | 6386.0 | 290.3 | 0.9% |
| `sscvfem::build_scatter` | setup | 2 | 0.010 | 5101.3 | 363.4 | 0.5% |
| `SFC::reorder` | other | 1 | 0.002 | 2216.8 | 836.1 | 0.1% |
| `DirichletConditions::gradient` | constraints | 13 | 0.001 | 51.3 | 36108.5 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 13 | 0.001 | 42.0 | 44134.4 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 417.9 | 4434.9 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 417.0 | 4445.1 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 115.6 | 16029.8 | 0.0% |
| `create_dual_graph` | other | 1 | 0.000 | 112.5 | 16471.2 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.1 | 24719439.8 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 10.5 | 176692.2 | 0.0% |
| `sscvfem::apply_transient` | transient | 13 | 0.000 | 0.0 | 0.0 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 1.769 | 84.0% |
| nodal gradient | 0.224 | 10.6% |
| other | 0.056 | 2.6% |
| constraints | 0.048 | 2.3% |
| setup | 0.010 | 0.5% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 8.532 | 2843850.0 |
| `Function::apply` | 903 | 2.090 | 2315.0 |
| `CVFEMNavierStokes::apply` | 903 | 2.043 | 2262.3 |
| `sscvfem::apply` | 903 | 1.971 | 2182.3 |
| `sscvfem::nodal_q_grad` | 903 | 0.219 | 242.4 |
| `Function::copy_constrained_dofs` | 903 | 0.046 | 51.3 |
| `CVFEMNavierStokes::initialize` | 1 | 0.035 | 35189.9 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.032 | 10776.5 |


### ss_L8p_N8

4,343,300 dof, 72 threads, nid006544. 4.751 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 4.088 | 4527.0 | 959.4 | 86.0% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 913 | 0.484 | 530.2 | 8191.1 | 10.2% |
| `to_semistructured` | other | 1 | 0.055 | 54652.9 | 79.5 | 1.2% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.055 | 60.5 | 71768.1 | 1.2% |
| `sscvfem::block_diag` | element sweep | 3 | 0.038 | 12642.5 | 343.5 | 0.8% |
| `sscvfem::build_scatter` | setup | 2 | 0.027 | 13494.8 | 321.9 | 0.6% |
| `SFC::reorder` | other | 1 | 0.002 | 2309.8 | 1880.4 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.001 | 761.5 | 5703.5 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.001 | 759.4 | 5719.7 | 0.0% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 69.8 | 62204.8 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 43.3 | 100330.3 | 0.0% |
| `create_dual_graph` | other | 1 | 0.000 | 258.2 | 16821.0 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 130.7 | 33243.0 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.1 | 43750152.3 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 26.8 | 161929.9 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.0 | 127519618.8 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 4.126 | 86.8% |
| nodal gradient | 0.484 | 10.2% |
| other | 0.057 | 1.2% |
| constraints | 0.057 | 1.2% |
| setup | 0.027 | 0.6% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 18.650 | 6216600.0 |
| `Function::apply` | 903 | 4.811 | 5327.8 |
| `CVFEMNavierStokes::apply` | 903 | 4.753 | 5263.1 |
| `sscvfem::apply` | 903 | 4.565 | 5055.4 |
| `sscvfem::nodal_q_grad` | 903 | 0.474 | 524.4 |
| `CVFEMNavierStokes::initialize` | 1 | 0.088 | 88442.8 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.073 | 24279.2 |
| `Function::copy_constrained_dofs` | 903 | 0.056 | 61.7 |


## Throughput against problem size

MDOF/s per scope, the same scope across every configuration. A kernel that is
memory bound flattens; one that is not keeps climbing with the problem until it
does. A number measured below saturation is not a throughput, so the smallest
sizes are here to show where that begins rather than to be quoted.

### `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action`

| run | dof | calls | seconds | MDOF/s |
|---|---|---|---|---|
| flat_N16 | 75,140 | 903 | 0.027 | 2478.1 |
| flat_N24 | 242,500 | 903 | 0.036 | 6123.5 |
| flat_N32 | 561,924 | 326 | 0.018 | 9911.0 |
| flat_N48 | 1,853,572 | 903 | 0.168 | 9969.0 |
| flat_N64 | 4,343,300 | 903 | 0.350 | 11192.8 |
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
| ss_L8p_N2 | 75,140 | -- | -- | -- |
| ss_L8p_N3 | 242,500 | -- | -- | -- |
| ss_L8p_N4 | 561,924 | -- | -- | -- |
| ss_L8p_N6 | 1,853,572 | -- | -- | -- |
| ss_L8p_N8 | 4,343,300 | -- | -- | -- |

### `cvfem_hex8_ns_steady::apply_boundary_scs_residual`

| run | dof | calls | seconds | MDOF/s |
|---|---|---|---|---|
| flat_N16 | 75,140 | 7 | 0.002 | 265.5 |
| flat_N24 | 242,500 | 7 | 0.003 | 523.8 |
| flat_N32 | 561,924 | 3 | 0.006 | 300.8 |
| flat_N48 | 1,853,572 | 11 | 0.014 | 1435.4 |
| flat_N64 | 4,343,300 | 7 | 0.025 | 1240.2 |
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
| ss_L8p_N2 | 75,140 | -- | -- | -- |
| ss_L8p_N3 | 242,500 | -- | -- | -- |
| ss_L8p_N4 | 561,924 | -- | -- | -- |
| ss_L8p_N6 | 1,853,572 | -- | -- | -- |
| ss_L8p_N8 | 4,343,300 | -- | -- | -- |

### `cvfem_hex8_ns_steady::apply_jacobian_action_packed`

| run | dof | calls | seconds | MDOF/s |
|---|---|---|---|---|
| flat_N16 | 75,140 | 903 | 0.417 | 162.9 |
| flat_N24 | 242,500 | 903 | 0.422 | 519.4 |
| flat_N32 | 561,924 | 326 | 0.152 | 1205.5 |
| flat_N48 | 1,853,572 | 903 | 1.521 | 1100.2 |
| flat_N64 | 4,343,300 | 903 | 3.851 | 1018.5 |
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
| ss_L8p_N2 | 75,140 | -- | -- | -- |
| ss_L8p_N3 | 242,500 | -- | -- | -- |
| ss_L8p_N4 | 561,924 | -- | -- | -- |
| ss_L8p_N6 | 1,853,572 | -- | -- | -- |
| ss_L8p_N8 | 4,343,300 | -- | -- | -- |

### `cvfem_hex8_ns_steady::apply_residual_packed`

| run | dof | calls | seconds | MDOF/s |
|---|---|---|---|---|
| flat_N16 | 75,140 | 7 | 0.003 | 182.1 |
| flat_N24 | 242,500 | 7 | 0.003 | 518.3 |
| flat_N32 | 561,924 | 3 | 0.002 | 1084.6 |
| flat_N48 | 1,853,572 | 11 | 0.013 | 1566.9 |
| flat_N64 | 4,343,300 | 7 | 0.020 | 1500.3 |
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
| ss_L8p_N2 | 75,140 | -- | -- | -- |
| ss_L8p_N3 | 242,500 | -- | -- | -- |
| ss_L8p_N4 | 561,924 | -- | -- | -- |
| ss_L8p_N6 | 1,853,572 | -- | -- | -- |
| ss_L8p_N8 | 4,343,300 | -- | -- | -- |

### `cvfem_hex8_ns_steady::assemble_block_diag`

| run | dof | calls | seconds | MDOF/s |
|---|---|---|---|---|
| flat_N16 | 75,140 | 3 | 0.016 | 14.3 |
| flat_N24 | 242,500 | 3 | 0.017 | 44.1 |
| flat_N32 | 561,924 | 2 | 0.009 | 123.3 |
| flat_N48 | 1,853,572 | 3 | 0.030 | 185.5 |
| flat_N64 | 4,343,300 | 3 | 0.067 | 193.6 |
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
| ss_L8p_N2 | 75,140 | -- | -- | -- |
| ss_L8p_N3 | 242,500 | -- | -- | -- |
| ss_L8p_N4 | 561,924 | -- | -- | -- |
| ss_L8p_N6 | 1,853,572 | -- | -- | -- |
| ss_L8p_N8 | 4,343,300 | -- | -- | -- |

### `cvfem_hex8_ns_steady::nodal_grad_strided`

| run | dof | calls | seconds | MDOF/s |
|---|---|---|---|---|
| flat_N16 | 75,140 | 913 | 0.057 | 1193.8 |
| flat_N24 | 242,500 | 913 | 0.057 | 3855.9 |
| flat_N32 | 561,924 | 331 | 0.081 | 2284.4 |
| flat_N48 | 1,853,572 | 917 | 0.330 | 5150.3 |
| flat_N64 | 4,343,300 | 913 | 0.754 | 5262.6 |
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
| ss_L8p_N2 | 75,140 | -- | -- | -- |
| ss_L8p_N3 | 242,500 | -- | -- | -- |
| ss_L8p_N4 | 561,924 | -- | -- | -- |
| ss_L8p_N6 | 1,853,572 | -- | -- | -- |
| ss_L8p_N8 | 4,343,300 | -- | -- | -- |

### `sscvfem::apply_macro_local_hoisted`

| run | dof | calls | seconds | MDOF/s |
|---|---|---|---|---|
| flat_N16 | 75,140 | -- | -- | -- |
| flat_N24 | 242,500 | -- | -- | -- |
| flat_N32 | 561,924 | -- | -- | -- |
| flat_N48 | 1,853,572 | -- | -- | -- |
| flat_N64 | 4,343,300 | -- | -- | -- |
| ss_L2_N8 | 75,140 | 903 | 0.102 | 665.9 |
| ss_L2_N12 | 242,500 | 903 | 0.263 | 834.0 |
| ss_L2_N16 | 561,924 | 903 | 0.585 | 867.5 |
| ss_L2_N24 | 1,853,572 | 903 | 2.015 | 830.7 |
| ss_L2_N32 | 4,343,300 | 903 | 5.077 | 772.5 |
| ss_L4_N4 | 75,140 | 1505 | 0.162 | 699.1 |
| ss_L4_N6 | 242,500 | 903 | 0.238 | 920.6 |
| ss_L4_N8 | 561,924 | 903 | 0.534 | 950.1 |
| ss_L4_N12 | 1,853,572 | 310 | 0.630 | 911.5 |
| ss_L4_N16 | 4,343,300 | 903 | 4.389 | 893.7 |
| ss_L8p_N2 | 75,140 | 903 | 0.176 | 385.8 |
| ss_L8p_N3 | 242,500 | 903 | 0.312 | 701.9 |
| ss_L8p_N4 | 561,924 | 903 | 0.565 | 897.9 |
| ss_L8p_N6 | 1,853,572 | 903 | 1.749 | 956.8 |
| ss_L8p_N8 | 4,343,300 | 903 | 4.088 | 959.4 |

### `sscvfem::block_diag`

| run | dof | calls | seconds | MDOF/s |
|---|---|---|---|---|
| flat_N16 | 75,140 | -- | -- | -- |
| flat_N24 | 242,500 | -- | -- | -- |
| flat_N32 | 561,924 | -- | -- | -- |
| flat_N48 | 1,853,572 | -- | -- | -- |
| flat_N64 | 4,343,300 | -- | -- | -- |
| ss_L2_N8 | 75,140 | 3 | 0.001 | 281.1 |
| ss_L2_N12 | 242,500 | 3 | 0.002 | 330.3 |
| ss_L2_N16 | 561,924 | 3 | 0.006 | 304.1 |
| ss_L2_N24 | 1,853,572 | 3 | 0.019 | 286.1 |
| ss_L2_N32 | 4,343,300 | 3 | 0.048 | 273.9 |
| ss_L4_N4 | 75,140 | 5 | 0.001 | 319.9 |
| ss_L4_N6 | 242,500 | 3 | 0.002 | 361.7 |
| ss_L4_N8 | 561,924 | 3 | 0.005 | 323.4 |
| ss_L4_N12 | 1,853,572 | 2 | 0.012 | 304.2 |
| ss_L4_N16 | 4,343,300 | 3 | 0.041 | 320.7 |
| ss_L8p_N2 | 75,140 | 3 | 0.001 | 169.1 |
| ss_L8p_N3 | 242,500 | 3 | 0.003 | 272.5 |
| ss_L8p_N4 | 561,924 | 3 | 0.005 | 311.2 |
| ss_L8p_N6 | 1,853,572 | 3 | 0.019 | 290.3 |
| ss_L8p_N8 | 4,343,300 | 3 | 0.038 | 343.5 |

### `sscvfem::nodal_grad_strided`

| run | dof | calls | seconds | MDOF/s |
|---|---|---|---|---|
| flat_N16 | 75,140 | -- | -- | -- |
| flat_N24 | 242,500 | -- | -- | -- |
| flat_N32 | 561,924 | -- | -- | -- |
| flat_N48 | 1,853,572 | -- | -- | -- |
| flat_N64 | 4,343,300 | -- | -- | -- |
| ss_L2_N8 | 75,140 | 913 | 0.057 | 1197.0 |
| ss_L2_N12 | 242,500 | 913 | 0.095 | 2341.2 |
| ss_L2_N16 | 561,924 | 913 | 0.173 | 2969.9 |
| ss_L2_N24 | 1,853,572 | 914 | 0.582 | 2912.9 |
| ss_L2_N32 | 4,343,300 | 919 | 1.757 | 2272.2 |
| ss_L4_N4 | 75,140 | 1525 | 0.093 | 1228.2 |
| ss_L4_N6 | 242,500 | 913 | 0.090 | 2466.9 |
| ss_L4_N8 | 561,924 | 913 | 0.157 | 3264.6 |
| ss_L4_N12 | 1,853,572 | 315 | 0.170 | 3431.8 |
| ss_L4_N16 | 4,343,300 | 916 | 1.287 | 3090.4 |
| ss_L8p_N2 | 75,140 | 917 | 0.073 | 940.3 |
| ss_L8p_N3 | 242,500 | 913 | 0.076 | 2912.5 |
| ss_L8p_N4 | 561,924 | 914 | 0.082 | 6289.8 |
| ss_L8p_N6 | 1,853,572 | 919 | 0.224 | 7602.9 |
| ss_L8p_N8 | 4,343,300 | 913 | 0.484 | 8191.1 |


## Provenance

| field | value |
|---|---|
| generated | 2026-09-12 09:12:59 |
| configurations | 20 |
| machines | nid006544 |
