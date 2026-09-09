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
| flat_N16 | 75,140 | apply_jacobian_action_packed | 100 | 75% | 14% | 7% | 4% |
| flat_N24 | 242,500 | apply_jacobian_action_packed | 305 | 46% | 40% | 11% | 4% |
| flat_N32 | 561,924 | apply_jacobian_action_packed | 698 | 46% | 28% | 21% | 4% |
| flat_N48 | 1,853,572 | apply_jacobian_action_packed | 657 | 51% | 28% | 19% | 1% |
| flat_N64 | 4,343,300 | apply_jacobian_action_packed | 625 | 53% | 27% | 19% | 1% |
| ss_L2_N8 | 75,140 | apply_macro_local_hoisted | 721 | 35% | 24% | 0% | 15% |
| ss_L2_N12 | 242,500 | apply_macro_local_hoisted | 842 | 61% | 28% | 0% | 9% |
| ss_L2_N16 | 561,924 | apply_macro_local_hoisted | 874 | 67% | 26% | 0% | 5% |
| ss_L2_N24 | 1,853,572 | apply_macro_local_hoisted | 821 | 67% | 27% | 0% | 2% |
| ss_L2_N32 | 4,343,300 | apply_macro_local_hoisted | 759 | 65% | 32% | 0% | 1% |
| ss_L4_N4 | 75,140 | apply_macro_local_hoisted | 731 | 47% | 31% | 0% | 20% |
| ss_L4_N6 | 242,500 | apply_macro_local_hoisted | 933 | 53% | 24% | 0% | 9% |
| ss_L4_N8 | 561,924 | apply_macro_local_hoisted | 962 | 69% | 25% | 0% | 5% |
| ss_L4_N12 | 1,853,572 | apply_macro_local_hoisted | 916 | 72% | 25% | 0% | 2% |
| ss_L4_N16 | 4,343,300 | apply_macro_local_hoisted | 900 | 69% | 28% | 0% | 1% |

The boundary closure is a separate pass on the flat operator and is FUSED into
the macro-element sweep on the semi-structured one, where it is inside the micro
loop and cannot carry a scope of its own without paying for one per element. So
its column is blank for the semi-structured rows and its cost is inside theirs;
the two boundary shares are not comparable and the sweep shares are not either.

## Against the bench: what the missing factor was

perf/baseline_grace.csv records `jac_action_packed_sumfact` at 1933 MDOF/s, and
re-measured on one allocation with three repeats it is 1695 (spread 3%). The scope
above reads 607-625. These are the same kernel, and the trace figure already
EXCLUDES the boundary pass and the nodal gradient -- they are sibling scopes,
listed separately in the tables below -- so the cascade in
docs/CVFEM_Operator_Cascade.md did not account for the difference.

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
| element kernel, no Rhie-Chow (what the baseline quotes) | 1878 | 1.00x |
| element kernel, with Rhie-Chow | 871 | 2.16x slower |
| the whole matvec: kernel + the direction's nodal gradient | 467 | 4.02x slower |

**Rhie-Chow costs the packed element kernel 2.16x**, and the direction's gradient
reconstruction costs another 1.86x on top of it -- 46% of every matvec, measured
directly (`frac_jac_action_qgrad`, 0.41 at one thread rising to 0.47 at 72 as the
element sweep parallelises slightly better than the reconstruction's atomic
scatter). So of the ~3x that was unattributed, 2.16x is the Rhie-Chow term inside
the kernel and the rest of the gap is a sibling scope that was never in the bench
at all.

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
runs is 467 MDOF/s at 11.2M dof on 72 Grace cores.

Reproduce with `jobs/fused_rc.sbatch`.


## Per configuration

### flat_N16

75,140 dof, 72 threads, nid006544. 0.916 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `cvfem_hex8_ns_steady::apply_jacobian_action_packed` | element sweep | 903 | 0.675 | 748.1 | 100.4 | 73.7% |
| `cvfem_hex8_ns_steady::nodal_grad_strided` | nodal gradient | 917 | 0.127 | 138.0 | 544.4 | 13.8% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action` | boundary | 903 | 0.062 | 68.9 | 1090.1 | 6.8% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.038 | 42.5 | 1766.8 | 4.2% |
| `cvfem_hex8_ns_steady::apply_residual_packed` | element sweep | 11 | 0.007 | 624.8 | 120.3 | 0.8% |
| `cvfem_hex8_ns_steady::assemble_block_diag` | element sweep | 3 | 0.002 | 706.2 | 106.4 | 0.2% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_residual` | boundary | 11 | 0.001 | 106.5 | 705.5 | 0.1% |
| `Mesh::initialize_node_to_node_graph` | setup | 1 | 0.001 | 755.1 | 99.5 | 0.1% |
| `create_crs_graph_mem_conservative` | setup | 1 | 0.001 | 689.7 | 108.9 | 0.1% |
| `DirichletConditions::gradient` | constraints | 11 | 0.000 | 41.1 | 1826.5 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 11 | 0.000 | 40.8 | 1843.0 | 0.0% |
| `create_n2e` | other | 1 | 0.000 | 309.7 | 242.6 | 0.0% |
| `cvfem_hex8_ns_steady::precompute_element_bsr_slots` | setup | 1 | 0.000 | 305.7 | 245.8 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 76.3 | 984.9 | 0.0% |
| `cvfem_hex8_ns_steady::apply_transient_action` | transient | 903 | 0.000 | 0.1 | 1276187.1 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 47.2 | 1591.7 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.684 | 74.7% |
| nodal gradient | 0.127 | 13.8% |
| boundary | 0.063 | 6.9% |
| constraints | 0.039 | 4.3% |
| setup | 0.002 | 0.2% |
| other | 0.000 | 0.0% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 1.304 | 434610.0 |
| `Function::apply` | 903 | 0.913 | 1010.9 |
| `CVFEMNavierStokes::apply` | 903 | 0.874 | 967.6 |
| `cvfem_hex8_ns_steady::apply_jacobian_action` | 903 | 0.863 | 956.1 |
| `cvfem_hex8_ns_steady::assemble_nodal_q_grad` | 903 | 0.125 | 138.2 |
| `Function::copy_constrained_dofs` | 903 | 0.039 | 42.8 |
| `Function::gradient` | 11 | 0.011 | 979.0 |
| `CVFEMNavierStokes::gradient` | 11 | 0.010 | 936.6 |


### flat_N24

242,500 dof, 72 threads, nid006544. 1.605 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `cvfem_hex8_ns_steady::apply_jacobian_action_packed` | element sweep | 903 | 0.717 | 794.3 | 305.3 | 44.7% |
| `cvfem_hex8_ns_steady::nodal_grad_strided` | nodal gradient | 913 | 0.638 | 699.1 | 346.9 | 39.8% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action` | boundary | 903 | 0.167 | 185.2 | 1309.2 | 10.4% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.059 | 65.1 | 3726.3 | 3.7% |
| `cvfem_hex8_ns_steady::assemble_block_diag` | element sweep | 3 | 0.010 | 3170.8 | 76.5 | 0.6% |
| `cvfem_hex8_ns_steady::apply_residual_packed` | element sweep | 7 | 0.005 | 753.7 | 321.7 | 0.3% |
| `Mesh::initialize_node_to_node_graph` | setup | 1 | 0.002 | 1815.3 | 133.6 | 0.1% |
| `create_crs_graph_mem_conservative` | setup | 1 | 0.002 | 1804.6 | 134.4 | 0.1% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_residual` | boundary | 7 | 0.002 | 218.2 | 1111.4 | 0.1% |
| `create_n2e` | other | 1 | 0.001 | 1139.4 | 212.8 | 0.1% |
| `cvfem_hex8_ns_steady::precompute_element_bsr_slots` | setup | 1 | 0.001 | 859.3 | 282.2 | 0.1% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 49.9 | 4863.3 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 41.7 | 5821.6 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 142.6 | 1700.9 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 104.4 | 2322.2 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 104.0 | 2332.8 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.732 | 45.6% |
| nodal gradient | 0.638 | 39.8% |
| boundary | 0.169 | 10.5% |
| constraints | 0.060 | 3.7% |
| setup | 0.004 | 0.3% |
| other | 0.001 | 0.1% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 2.545 | 848193.3 |
| `Function::apply` | 903 | 1.589 | 1759.7 |
| `CVFEMNavierStokes::apply` | 903 | 1.529 | 1693.8 |
| `cvfem_hex8_ns_steady::apply_jacobian_action` | 903 | 1.517 | 1680.3 |
| `cvfem_hex8_ns_steady::assemble_nodal_q_grad` | 903 | 0.632 | 699.6 |
| `Function::copy_constrained_dofs` | 903 | 0.059 | 65.4 |
| `CVFEMNavierStokes::initialize` | 1 | 0.021 | 20547.2 |
| `Function::gradient` | 7 | 0.013 | 1820.7 |


### flat_N32

561,924 dof, 72 threads, nid006544. 1.638 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `cvfem_hex8_ns_steady::apply_jacobian_action_packed` | element sweep | 903 | 0.727 | 805.5 | 697.6 | 44.4% |
| `cvfem_hex8_ns_steady::nodal_grad_strided` | nodal gradient | 919 | 0.459 | 499.6 | 1124.8 | 28.0% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action` | boundary | 903 | 0.343 | 380.0 | 1478.8 | 20.9% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.068 | 75.3 | 7459.9 | 4.2% |
| `cvfem_hex8_ns_steady::assemble_block_diag` | element sweep | 3 | 0.013 | 4468.6 | 125.7 | 0.8% |
| `cvfem_hex8_ns_steady::apply_residual_packed` | element sweep | 13 | 0.008 | 588.9 | 954.2 | 0.5% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_residual` | boundary | 13 | 0.004 | 336.3 | 1670.8 | 0.3% |
| `Mesh::initialize_node_to_node_graph` | setup | 1 | 0.004 | 4348.3 | 129.2 | 0.3% |
| `create_crs_graph_mem_conservative` | setup | 1 | 0.004 | 4334.7 | 129.6 | 0.3% |
| `create_n2e` | other | 1 | 0.003 | 2920.9 | 192.4 | 0.2% |
| `cvfem_hex8_ns_steady::precompute_element_bsr_slots` | setup | 1 | 0.002 | 1991.3 | 282.2 | 0.1% |
| `DirichletConditions::gradient` | constraints | 13 | 0.001 | 43.9 | 12798.4 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 13 | 0.001 | 41.0 | 13721.2 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 185.0 | 3037.2 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 183.8 | 3056.9 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 84.4 | 6657.9 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.748 | 45.7% |
| nodal gradient | 0.459 | 28.0% |
| boundary | 0.348 | 21.2% |
| constraints | 0.070 | 4.2% |
| setup | 0.011 | 0.7% |
| other | 0.003 | 0.2% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 3.682 | 1227310.0 |
| `Function::apply` | 903 | 1.609 | 1781.6 |
| `CVFEMNavierStokes::apply` | 903 | 1.540 | 1705.3 |
| `cvfem_hex8_ns_steady::apply_jacobian_action` | 903 | 1.523 | 1686.9 |
| `cvfem_hex8_ns_steady::assemble_nodal_q_grad` | 903 | 0.451 | 499.8 |
| `Function::copy_constrained_dofs` | 903 | 0.068 | 75.7 |
| `CVFEMNavierStokes::initialize` | 1 | 0.043 | 43080.1 |
| `Function::gradient` | 13 | 0.021 | 1611.9 |


### flat_N48

1,853,572 dof, 72 threads, nid006544. 5.103 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `cvfem_hex8_ns_steady::apply_jacobian_action_packed` | element sweep | 903 | 2.546 | 2819.6 | 657.4 | 49.9% |
| `cvfem_hex8_ns_steady::nodal_grad_strided` | nodal gradient | 917 | 1.409 | 1536.6 | 1206.3 | 27.6% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action` | boundary | 903 | 0.964 | 1068.1 | 1735.4 | 18.9% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.054 | 59.3 | 31268.2 | 1.0% |
| `cvfem_hex8_ns_steady::assemble_block_diag` | element sweep | 3 | 0.043 | 14378.0 | 128.9 | 0.8% |
| `Mesh::initialize_node_to_node_graph` | setup | 1 | 0.019 | 18648.1 | 99.4 | 0.4% |
| `create_crs_graph_mem_conservative` | setup | 1 | 0.019 | 18625.7 | 99.5 | 0.4% |
| `cvfem_hex8_ns_steady::apply_residual_packed` | element sweep | 11 | 0.017 | 1537.2 | 1205.8 | 0.3% |
| `create_n2e` | other | 1 | 0.012 | 12174.4 | 152.3 | 0.2% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_residual` | boundary | 11 | 0.011 | 996.6 | 1859.8 | 0.2% |
| `cvfem_hex8_ns_steady::precompute_element_bsr_slots` | setup | 1 | 0.007 | 7264.8 | 255.1 | 0.1% |
| `DirichletConditions::gradient` | constraints | 11 | 0.001 | 57.3 | 32332.3 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 11 | 0.000 | 42.2 | 43923.4 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 446.6 | 4150.8 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 444.9 | 4166.4 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 132.1 | 14033.3 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 2.606 | 51.1% |
| nodal gradient | 1.409 | 27.6% |
| boundary | 0.975 | 19.1% |
| constraints | 0.056 | 1.1% |
| setup | 0.045 | 0.9% |
| other | 0.012 | 0.2% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 11.683 | 3894366.7 |
| `Function::apply` | 903 | 5.023 | 5562.7 |
| `CVFEMNavierStokes::apply` | 903 | 4.966 | 5500.0 |
| `cvfem_hex8_ns_steady::apply_jacobian_action` | 903 | 4.903 | 5430.2 |
| `cvfem_hex8_ns_steady::assemble_nodal_q_grad` | 903 | 1.388 | 1537.4 |
| `CVFEMNavierStokes::initialize` | 1 | 0.155 | 155168.0 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.055 | 18227.6 |
| `Function::copy_constrained_dofs` | 903 | 0.055 | 60.5 |


### flat_N64

4,343,300 dof, 72 threads, nid006544. 12.138 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `cvfem_hex8_ns_steady::apply_jacobian_action_packed` | element sweep | 903 | 6.271 | 6944.9 | 625.4 | 51.7% |
| `cvfem_hex8_ns_steady::nodal_grad_strided` | nodal gradient | 913 | 3.265 | 3575.7 | 1214.7 | 26.9% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action` | boundary | 903 | 2.254 | 2496.5 | 1739.8 | 18.6% |
| `cvfem_hex8_ns_steady::assemble_block_diag` | element sweep | 3 | 0.099 | 32842.4 | 132.2 | 0.8% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.087 | 96.3 | 45089.9 | 0.7% |
| `Mesh::initialize_node_to_node_graph` | setup | 1 | 0.036 | 36152.4 | 120.1 | 0.3% |
| `create_crs_graph_mem_conservative` | setup | 1 | 0.036 | 36124.0 | 120.2 | 0.3% |
| `create_n2e` | other | 1 | 0.027 | 27251.5 | 159.4 | 0.2% |
| `cvfem_hex8_ns_steady::apply_residual_packed` | element sweep | 7 | 0.026 | 3708.4 | 1171.2 | 0.2% |
| `cvfem_hex8_ns_steady::precompute_element_bsr_slots` | setup | 1 | 0.018 | 17808.9 | 243.9 | 0.1% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_residual` | boundary | 7 | 0.016 | 2322.5 | 1870.1 | 0.1% |
| `Function::constraints_mask` | constraints | 1 | 0.001 | 853.8 | 5087.2 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.001 | 848.5 | 5118.6 | 0.0% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 59.2 | 73413.9 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 43.2 | 100647.2 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 138.3 | 31408.8 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 6.396 | 52.7% |
| nodal gradient | 3.265 | 26.9% |
| boundary | 2.271 | 18.7% |
| setup | 0.090 | 0.7% |
| constraints | 0.090 | 0.7% |
| other | 0.027 | 0.2% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 26.052 | 8684000.0 |
| `Function::apply` | 903 | 12.042 | 13335.5 |
| `CVFEMNavierStokes::apply` | 903 | 11.951 | 13234.9 |
| `cvfem_hex8_ns_steady::apply_jacobian_action` | 903 | 11.764 | 13028.1 |
| `cvfem_hex8_ns_steady::assemble_nodal_q_grad` | 903 | 3.231 | 3578.1 |
| `CVFEMNavierStokes::initialize` | 1 | 0.371 | 370729.0 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.131 | 43612.7 |
| `Function::copy_constrained_dofs` | 903 | 0.088 | 97.9 |


### ss_L2_N8

75,140 dof, 72 threads, nid006544. 0.357 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 1204 | 0.126 | 104.2 | 720.8 | 35.1% |
| `to_semistructured` | other | 1 | 0.090 | 89955.3 | 0.8 | 25.2% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 1221 | 0.087 | 70.9 | 1059.6 | 24.2% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 1204 | 0.052 | 42.9 | 1753.3 | 14.4% |
| `sscvfem::block_diag` | element sweep | 4 | 0.001 | 321.8 | 233.5 | 0.4% |
| `sscvfem::build_scatter` | setup | 1 | 0.001 | 716.0 | 104.9 | 0.2% |
| `DirichletConditions::gradient` | constraints | 13 | 0.001 | 42.4 | 1772.9 | 0.2% |
| `DirichletConditions::apply_value` | constraints | 13 | 0.001 | 40.9 | 1837.3 | 0.1% |
| `create_dual_graph` | other | 1 | 0.000 | 258.0 | 291.3 | 0.1% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 77.0 | 975.7 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 31.1 | 2415.0 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 47.2 | 1591.7 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 46.0 | 1633.0 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 1204 | 0.000 | 0.0 | 1997120.5 | 0.0% |
| `sscvfem::apply_transient` | transient | 13 | 0.000 | 0.1 | 1365692.8 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.127 | 35.5% |
| other | 0.090 | 25.3% |
| nodal gradient | 0.087 | 24.2% |
| constraints | 0.053 | 14.8% |
| setup | 0.001 | 0.2% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 4 | 0.780 | 194903.0 |
| `Function::apply` | 1204 | 0.278 | 231.0 |
| `CVFEMNavierStokes::apply` | 1204 | 0.226 | 187.6 |
| `sscvfem::apply` | 1204 | 0.212 | 175.8 |
| `sscvfem::nodal_q_grad` | 1204 | 0.086 | 71.1 |
| `Function::copy_constrained_dofs` | 1204 | 0.052 | 43.1 |
| `Function::gradient` | 13 | 0.003 | 259.0 |
| `CVFEMNavierStokes::gradient` | 13 | 0.003 | 215.6 |


### ss_L2_N12

242,500 dof, 72 threads, nid006544. 0.431 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 0.260 | 288.1 | 841.7 | 60.3% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 913 | 0.120 | 131.3 | 1847.0 | 27.8% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.038 | 42.0 | 5769.3 | 8.8% |
| `to_semistructured` | other | 1 | 0.006 | 6259.4 | 38.7 | 1.5% |
| `sscvfem::block_diag` | element sweep | 3 | 0.003 | 931.7 | 260.3 | 0.6% |
| `sscvfem::build_scatter` | setup | 1 | 0.002 | 2047.1 | 118.5 | 0.5% |
| `create_dual_graph` | other | 1 | 0.001 | 863.6 | 280.8 | 0.2% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 51.1 | 4746.6 | 0.1% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 40.3 | 6023.5 | 0.1% |
| `create_n2e` | other | 2 | 0.000 | 88.7 | 2734.2 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 103.7 | 2338.2 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 103.0 | 2354.4 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 98.9 | 2450.9 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.0 | 5887552.3 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.1 | 1779958.4 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.263 | 61.0% |
| nodal gradient | 0.120 | 27.8% |
| constraints | 0.039 | 9.0% |
| other | 0.007 | 1.7% |
| setup | 0.002 | 0.5% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 1.369 | 456463.3 |
| `Function::apply` | 903 | 0.430 | 476.6 |
| `CVFEMNavierStokes::apply` | 903 | 0.392 | 433.9 |
| `sscvfem::apply` | 903 | 0.379 | 420.0 |
| `sscvfem::nodal_q_grad` | 903 | 0.119 | 131.4 |
| `Function::copy_constrained_dofs` | 903 | 0.038 | 42.3 |
| `Function::gradient` | 7 | 0.007 | 1054.2 |
| `CVFEMNavierStokes::gradient` | 7 | 0.007 | 1002.1 |


### ss_L2_N16

561,924 dof, 72 threads, nid006544. 0.877 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 0.581 | 643.1 | 873.8 | 66.2% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 913 | 0.229 | 250.6 | 2242.6 | 26.1% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.042 | 46.2 | 12171.1 | 4.8% |
| `to_semistructured` | other | 1 | 0.011 | 11216.2 | 50.1 | 1.3% |
| `sscvfem::block_diag` | element sweep | 3 | 0.006 | 1972.3 | 284.9 | 0.7% |
| `sscvfem::build_scatter` | setup | 1 | 0.005 | 5031.1 | 111.7 | 0.6% |
| `create_dual_graph` | other | 1 | 0.002 | 2131.7 | 263.6 | 0.2% |
| `create_n2e` | other | 2 | 0.000 | 193.2 | 2907.9 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 54.3 | 10350.2 | 0.0% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 47.5 | 11826.6 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 187.2 | 3002.4 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 185.0 | 3037.2 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 108.2 | 5191.4 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.0 | 13056836.0 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.1 | 4124541.5 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.587 | 66.9% |
| nodal gradient | 0.229 | 26.1% |
| constraints | 0.043 | 4.9% |
| other | 0.014 | 1.6% |
| setup | 0.005 | 0.6% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 2.906 | 968566.7 |
| `Function::apply` | 903 | 0.865 | 958.4 |
| `CVFEMNavierStokes::apply` | 903 | 0.823 | 911.5 |
| `sscvfem::apply` | 903 | 0.807 | 893.7 |
| `sscvfem::nodal_q_grad` | 903 | 0.226 | 249.9 |
| `Function::copy_constrained_dofs` | 903 | 0.042 | 46.4 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.010 | 3268.6 |
| `Function::gradient` | 7 | 0.008 | 1110.0 |


### ss_L2_N24

1,853,572 dof, 72 threads, nid006544. 3.058 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 2.039 | 2257.9 | 820.9 | 66.7% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 914 | 0.823 | 900.8 | 2057.7 | 26.9% |
| `to_semistructured` | other | 1 | 0.096 | 96491.6 | 19.2 | 3.2% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.046 | 51.2 | 36199.4 | 1.5% |
| `sscvfem::build_scatter` | setup | 1 | 0.021 | 20866.4 | 88.8 | 0.7% |
| `sscvfem::block_diag` | element sweep | 3 | 0.021 | 6940.4 | 267.1 | 0.7% |
| `create_dual_graph` | other | 1 | 0.008 | 7880.2 | 235.2 | 0.3% |
| `create_n2e` | other | 2 | 0.002 | 868.9 | 2133.2 | 0.1% |
| `DirichletConditions::gradient` | constraints | 8 | 0.000 | 57.5 | 32242.4 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 449.9 | 4120.0 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 447.5 | 4141.9 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 8 | 0.000 | 46.3 | 39997.1 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 125.6 | 14752.2 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.1 | 26001204.2 | 0.0% |
| `sscvfem::apply_transient` | transient | 8 | 0.000 | 0.1 | 20731844.3 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 2.060 | 67.4% |
| nodal gradient | 0.823 | 26.9% |
| other | 0.106 | 3.5% |
| constraints | 0.048 | 1.6% |
| setup | 0.021 | 0.7% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 9.510 | 3170113.3 |
| `Function::apply` | 903 | 2.970 | 3289.5 |
| `CVFEMNavierStokes::apply` | 903 | 2.922 | 3235.4 |
| `sscvfem::apply` | 903 | 2.855 | 3162.0 |
| `sscvfem::nodal_q_grad` | 903 | 0.814 | 901.6 |
| `Function::copy_constrained_dofs` | 903 | 0.047 | 52.4 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.037 | 12199.2 |
| `Function::gradient` | 8 | 0.028 | 3543.9 |


### ss_L2_N32

4,343,300 dof, 72 threads, nid006544. 8.082 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 5.165 | 5720.3 | 759.3 | 63.9% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 914 | 2.569 | 2810.8 | 1545.2 | 31.8% |
| `to_semistructured` | other | 1 | 0.133 | 133134.0 | 32.6 | 1.6% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.088 | 97.4 | 44610.2 | 1.1% |
| `sscvfem::build_scatter` | setup | 1 | 0.050 | 50403.6 | 86.2 | 0.6% |
| `sscvfem::block_diag` | element sweep | 3 | 0.049 | 16195.2 | 268.2 | 0.6% |
| `create_dual_graph` | other | 1 | 0.019 | 19471.6 | 223.1 | 0.2% |
| `create_n2e` | other | 2 | 0.005 | 2551.7 | 1702.1 | 0.1% |
| `DirichletConditions::gradient` | constraints | 8 | 0.001 | 107.2 | 40505.0 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.001 | 778.4 | 5579.5 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.001 | 776.3 | 5594.9 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 8 | 0.000 | 41.2 | 105530.0 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 138.0 | 31463.2 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.1 | 60478117.9 | 0.0% |
| `sscvfem::apply_transient` | transient | 8 | 0.000 | 0.1 | 72868506.4 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 5.214 | 64.5% |
| nodal gradient | 2.569 | 31.8% |
| other | 0.158 | 2.0% |
| constraints | 0.091 | 1.1% |
| setup | 0.050 | 0.6% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 21.891 | 7297100.0 |
| `Function::apply` | 903 | 7.994 | 8853.1 |
| `CVFEMNavierStokes::apply` | 903 | 7.903 | 8751.6 |
| `sscvfem::apply` | 903 | 7.711 | 8539.7 |
| `sscvfem::nodal_q_grad` | 903 | 2.542 | 2814.7 |
| `Function::copy_constrained_dofs` | 903 | 0.089 | 98.6 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.088 | 29498.4 |
| `Function::gradient` | 8 | 0.067 | 8355.5 |


### ss_L4_N4

75,140 dof, 72 threads, nid006544. 0.200 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 0.093 | 102.8 | 730.9 | 46.3% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 913 | 0.061 | 67.3 | 1116.4 | 30.7% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.039 | 43.6 | 1723.6 | 19.6% |
| `to_semistructured` | other | 1 | 0.004 | 4346.4 | 17.3 | 2.2% |
| `sscvfem::block_diag` | element sweep | 3 | 0.001 | 394.3 | 190.6 | 0.6% |
| `sscvfem::build_scatter` | setup | 1 | 0.000 | 348.1 | 215.9 | 0.2% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 45.0 | 1671.3 | 0.2% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 41.2 | 1823.2 | 0.1% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 94.9 | 791.9 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 47.2 | 1591.7 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 46.3 | 1624.5 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.0 | 1872300.5 | 0.0% |
| `create_dual_graph` | other | 1 | 0.000 | 32.7 | 2300.4 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 3.7 | 20332.9 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.0 | 2206116.1 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.094 | 46.9% |
| nodal gradient | 0.061 | 30.7% |
| constraints | 0.040 | 20.0% |
| other | 0.004 | 2.2% |
| setup | 0.000 | 0.2% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 0.583 | 194491.0 |
| `Function::apply` | 903 | 0.205 | 226.9 |
| `CVFEMNavierStokes::apply` | 903 | 0.165 | 182.7 |
| `sscvfem::apply` | 903 | 0.154 | 170.7 |
| `sscvfem::nodal_q_grad` | 903 | 0.061 | 67.5 |
| `Function::copy_constrained_dofs` | 903 | 0.040 | 43.8 |
| `Function::gradient` | 7 | 0.002 | 264.4 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.002 | 612.6 |


### ss_L4_N6

242,500 dof, 72 threads, nid006544. 0.448 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 0.235 | 259.9 | 933.1 | 52.4% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 913 | 0.109 | 119.5 | 2029.7 | 24.3% |
| `to_semistructured` | other | 1 | 0.060 | 59780.6 | 4.1 | 13.3% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.040 | 44.2 | 5489.7 | 8.9% |
| `sscvfem::block_diag` | element sweep | 3 | 0.003 | 847.3 | 286.2 | 0.6% |
| `sscvfem::build_scatter` | setup | 1 | 0.001 | 982.0 | 246.9 | 0.2% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 43.3 | 5606.2 | 0.1% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 42.2 | 5741.8 | 0.1% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 145.2 | 1670.1 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 104.4 | 2322.2 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 104.0 | 2332.8 | 0.0% |
| `create_dual_graph` | other | 1 | 0.000 | 101.6 | 2387.6 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.0 | 6334193.4 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 8.7 | 27866.2 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.1 | 3559916.7 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.237 | 52.9% |
| nodal gradient | 0.109 | 24.3% |
| other | 0.060 | 13.4% |
| constraints | 0.041 | 9.1% |
| setup | 0.001 | 0.2% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 1.331 | 443530.0 |
| `Function::apply` | 903 | 0.396 | 439.0 |
| `CVFEMNavierStokes::apply` | 903 | 0.356 | 394.2 |
| `sscvfem::apply` | 903 | 0.343 | 379.9 |
| `sscvfem::nodal_q_grad` | 903 | 0.108 | 119.5 |
| `Function::copy_constrained_dofs` | 903 | 0.040 | 44.4 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.004 | 1390.1 |
| `Function::gradient` | 7 | 0.003 | 484.2 |


### ss_L4_N8

561,924 dof, 72 threads, nid006544. 0.754 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 875 | 0.511 | 584.2 | 961.9 | 67.8% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 885 | 0.189 | 213.5 | 2631.7 | 25.0% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 875 | 0.039 | 45.0 | 12477.6 | 5.2% |
| `sscvfem::block_diag` | element sweep | 3 | 0.006 | 1899.9 | 295.8 | 0.8% |
| `to_semistructured` | other | 1 | 0.005 | 5307.9 | 105.9 | 0.7% |
| `sscvfem::build_scatter` | setup | 1 | 0.002 | 2462.9 | 228.2 | 0.3% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 45.9 | 12239.0 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 41.8 | 13445.9 | 0.0% |
| `create_dual_graph` | other | 1 | 0.000 | 252.0 | 2229.8 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 185.5 | 3029.4 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 183.8 | 3056.9 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 104.4 | 5381.0 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 24.9 | 22553.9 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 875 | 0.000 | 0.0 | 12970236.3 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.0 | 16498131.4 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.517 | 68.5% |
| nodal gradient | 0.189 | 25.0% |
| constraints | 0.040 | 5.4% |
| other | 0.006 | 0.7% |
| setup | 0.002 | 0.3% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 2.744 | 914653.3 |
| `Function::apply` | 875 | 0.757 | 865.1 |
| `CVFEMNavierStokes::apply` | 875 | 0.717 | 819.4 |
| `sscvfem::apply` | 875 | 0.698 | 798.0 |
| `sscvfem::nodal_q_grad` | 875 | 0.187 | 213.3 |
| `Function::copy_constrained_dofs` | 875 | 0.040 | 45.3 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.009 | 3078.5 |
| `Function::gradient` | 7 | 0.006 | 919.8 |


### ss_L4_N12

1,853,572 dof, 72 threads, nid006544. 2.574 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 1.826 | 2022.6 | 916.4 | 71.0% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 915 | 0.644 | 703.4 | 2635.1 | 25.0% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.044 | 48.5 | 38248.3 | 1.7% |
| `to_semistructured` | other | 1 | 0.029 | 28959.3 | 64.0 | 1.1% |
| `sscvfem::block_diag` | element sweep | 3 | 0.019 | 6199.8 | 299.0 | 0.7% |
| `sscvfem::build_scatter` | setup | 1 | 0.009 | 9208.2 | 201.3 | 0.4% |
| `create_dual_graph` | other | 1 | 0.001 | 866.2 | 2140.0 | 0.0% |
| `DirichletConditions::gradient` | constraints | 9 | 0.000 | 50.8 | 36480.7 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 439.6 | 4216.1 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 438.2 | 4229.8 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 9 | 0.000 | 47.2 | 39287.0 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 84.2 | 22023.9 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 124.2 | 14922.2 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.1 | 22719505.6 | 0.0% |
| `sscvfem::apply_transient` | transient | 9 | 0.000 | 0.0 | 69969876.6 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 1.845 | 71.7% |
| nodal gradient | 0.644 | 25.0% |
| constraints | 0.046 | 1.8% |
| other | 0.030 | 1.2% |
| setup | 0.009 | 0.4% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 9.004 | 3001266.7 |
| `Function::apply` | 903 | 2.570 | 2845.5 |
| `CVFEMNavierStokes::apply` | 903 | 2.523 | 2794.4 |
| `sscvfem::apply` | 903 | 2.463 | 2727.8 |
| `sscvfem::nodal_q_grad` | 903 | 0.635 | 703.4 |
| `Function::copy_constrained_dofs` | 903 | 0.045 | 49.6 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.034 | 11205.7 |
| `Function::gradient` | 9 | 0.026 | 2906.3 |


### ss_L4_N16

4,343,300 dof, 72 threads, nid006544. 6.339 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 4.356 | 4824.2 | 900.3 | 68.7% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 918 | 1.774 | 1931.9 | 2248.2 | 28.0% |
| `to_semistructured` | other | 1 | 0.081 | 81078.1 | 53.6 | 1.3% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.054 | 60.0 | 72408.9 | 0.9% |
| `sscvfem::block_diag` | element sweep | 3 | 0.043 | 14408.0 | 301.5 | 0.7% |
| `sscvfem::build_scatter` | setup | 1 | 0.025 | 24814.6 | 175.0 | 0.4% |
| `create_dual_graph` | other | 1 | 0.002 | 2109.3 | 2059.1 | 0.0% |
| `DirichletConditions::gradient` | constraints | 12 | 0.001 | 74.2 | 58513.2 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.001 | 783.4 | 5543.9 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.001 | 781.8 | 5555.7 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 12 | 0.001 | 42.6 | 101961.6 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 197.9 | 21948.3 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 143.1 | 30361.9 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.1 | 50152617.7 | 0.0% |
| `sscvfem::apply_transient` | transient | 12 | 0.000 | 0.1 | 31229350.5 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 4.399 | 69.4% |
| nodal gradient | 1.774 | 28.0% |
| other | 0.084 | 1.3% |
| constraints | 0.057 | 0.9% |
| setup | 0.025 | 0.4% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 20.190 | 6729933.3 |
| `Function::apply` | 903 | 6.358 | 7040.5 |
| `CVFEMNavierStokes::apply` | 903 | 6.300 | 6976.3 |
| `sscvfem::apply` | 903 | 6.107 | 6763.1 |
| `sscvfem::nodal_q_grad` | 903 | 1.747 | 1934.5 |
| `Function::gradient` | 12 | 0.098 | 8130.6 |
| `CVFEMNavierStokes::gradient` | 12 | 0.097 | 8052.3 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.080 | 26804.7 |


## Throughput against problem size

MDOF/s per scope, the same scope across every configuration. A kernel that is
memory bound flattens; one that is not keeps climbing with the problem until it
does. A number measured below saturation is not a throughput, so the smallest
sizes are here to show where that begins rather than to be quoted.

### `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action`

| run | dof | calls | seconds | MDOF/s |
|---|---|---|---|---|
| flat_N16 | 75,140 | 903 | 0.062 | 1090.1 |
| flat_N24 | 242,500 | 903 | 0.167 | 1309.2 |
| flat_N32 | 561,924 | 903 | 0.343 | 1478.8 |
| flat_N48 | 1,853,572 | 903 | 0.964 | 1735.4 |
| flat_N64 | 4,343,300 | 903 | 2.254 | 1739.8 |
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
| flat_N16 | 75,140 | 11 | 0.001 | 705.5 |
| flat_N24 | 242,500 | 7 | 0.002 | 1111.4 |
| flat_N32 | 561,924 | 13 | 0.004 | 1670.8 |
| flat_N48 | 1,853,572 | 11 | 0.011 | 1859.8 |
| flat_N64 | 4,343,300 | 7 | 0.016 | 1870.1 |
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
| flat_N16 | 75,140 | 903 | 0.675 | 100.4 |
| flat_N24 | 242,500 | 903 | 0.717 | 305.3 |
| flat_N32 | 561,924 | 903 | 0.727 | 697.6 |
| flat_N48 | 1,853,572 | 903 | 2.546 | 657.4 |
| flat_N64 | 4,343,300 | 903 | 6.271 | 625.4 |
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
| flat_N16 | 75,140 | 11 | 0.007 | 120.3 |
| flat_N24 | 242,500 | 7 | 0.005 | 321.7 |
| flat_N32 | 561,924 | 13 | 0.008 | 954.2 |
| flat_N48 | 1,853,572 | 11 | 0.017 | 1205.8 |
| flat_N64 | 4,343,300 | 7 | 0.026 | 1171.2 |
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
| flat_N16 | 75,140 | 3 | 0.002 | 106.4 |
| flat_N24 | 242,500 | 3 | 0.010 | 76.5 |
| flat_N32 | 561,924 | 3 | 0.013 | 125.7 |
| flat_N48 | 1,853,572 | 3 | 0.043 | 128.9 |
| flat_N64 | 4,343,300 | 3 | 0.099 | 132.2 |
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
| flat_N16 | 75,140 | 917 | 0.127 | 544.4 |
| flat_N24 | 242,500 | 913 | 0.638 | 346.9 |
| flat_N32 | 561,924 | 919 | 0.459 | 1124.8 |
| flat_N48 | 1,853,572 | 917 | 1.409 | 1206.3 |
| flat_N64 | 4,343,300 | 913 | 3.265 | 1214.7 |
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
| ss_L2_N8 | 75,140 | 1204 | 0.126 | 720.8 |
| ss_L2_N12 | 242,500 | 903 | 0.260 | 841.7 |
| ss_L2_N16 | 561,924 | 903 | 0.581 | 873.8 |
| ss_L2_N24 | 1,853,572 | 903 | 2.039 | 820.9 |
| ss_L2_N32 | 4,343,300 | 903 | 5.165 | 759.3 |
| ss_L4_N4 | 75,140 | 903 | 0.093 | 730.9 |
| ss_L4_N6 | 242,500 | 903 | 0.235 | 933.1 |
| ss_L4_N8 | 561,924 | 875 | 0.511 | 961.9 |
| ss_L4_N12 | 1,853,572 | 903 | 1.826 | 916.4 |
| ss_L4_N16 | 4,343,300 | 903 | 4.356 | 900.3 |

### `sscvfem::block_diag`

| run | dof | calls | seconds | MDOF/s |
|---|---|---|---|---|
| flat_N16 | 75,140 | -- | -- | -- |
| flat_N24 | 242,500 | -- | -- | -- |
| flat_N32 | 561,924 | -- | -- | -- |
| flat_N48 | 1,853,572 | -- | -- | -- |
| flat_N64 | 4,343,300 | -- | -- | -- |
| ss_L2_N8 | 75,140 | 4 | 0.001 | 233.5 |
| ss_L2_N12 | 242,500 | 3 | 0.003 | 260.3 |
| ss_L2_N16 | 561,924 | 3 | 0.006 | 284.9 |
| ss_L2_N24 | 1,853,572 | 3 | 0.021 | 267.1 |
| ss_L2_N32 | 4,343,300 | 3 | 0.049 | 268.2 |
| ss_L4_N4 | 75,140 | 3 | 0.001 | 190.6 |
| ss_L4_N6 | 242,500 | 3 | 0.003 | 286.2 |
| ss_L4_N8 | 561,924 | 3 | 0.006 | 295.8 |
| ss_L4_N12 | 1,853,572 | 3 | 0.019 | 299.0 |
| ss_L4_N16 | 4,343,300 | 3 | 0.043 | 301.5 |

### `sscvfem::nodal_grad_strided`

| run | dof | calls | seconds | MDOF/s |
|---|---|---|---|---|
| flat_N16 | 75,140 | -- | -- | -- |
| flat_N24 | 242,500 | -- | -- | -- |
| flat_N32 | 561,924 | -- | -- | -- |
| flat_N48 | 1,853,572 | -- | -- | -- |
| flat_N64 | 4,343,300 | -- | -- | -- |
| ss_L2_N8 | 75,140 | 1221 | 0.087 | 1059.6 |
| ss_L2_N12 | 242,500 | 913 | 0.120 | 1847.0 |
| ss_L2_N16 | 561,924 | 913 | 0.229 | 2242.6 |
| ss_L2_N24 | 1,853,572 | 914 | 0.823 | 2057.7 |
| ss_L2_N32 | 4,343,300 | 914 | 2.569 | 1545.2 |
| ss_L4_N4 | 75,140 | 913 | 0.061 | 1116.4 |
| ss_L4_N6 | 242,500 | 913 | 0.109 | 2029.7 |
| ss_L4_N8 | 561,924 | 885 | 0.189 | 2631.7 |
| ss_L4_N12 | 1,853,572 | 915 | 0.644 | 2635.1 |
| ss_L4_N16 | 4,343,300 | 918 | 1.774 | 2248.2 |


## Provenance

| field | value |
|---|---|
| generated | 2026-09-09 22:07:31 |
| configurations | 15 |
| machines | nid006544 |
