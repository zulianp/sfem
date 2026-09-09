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
| flat_N16 | 75,140 | apply_jacobian_action_packed | 99 | 72% | 15% | 7% | 5% |
| flat_N24 | 242,500 | apply_jacobian_action_packed | 284 | 46% | 39% | 11% | 4% |
| flat_N32 | 561,924 | apply_jacobian_action_packed | 687 | 45% | 28% | 21% | 5% |
| flat_N48 | 1,853,572 | apply_jacobian_action_packed | 642 | 49% | 29% | 19% | 2% |
| flat_N64 | 4,343,300 | apply_jacobian_action_packed | 607 | 52% | 27% | 18% | 1% |
| ss_L2_N8 | 75,140 | apply_macro_local_hoisted | 661 | 39% | 26% | 0% | 17% |
| ss_L2_N12 | 242,500 | apply_macro_local_hoisted | 778 | 54% | 25% | 0% | 8% |
| ss_L2_N16 | 561,924 | apply_macro_local_hoisted | 753 | 65% | 26% | 0% | 6% |
| ss_L2_N24 | 1,853,572 | apply_macro_local_hoisted | 756 | 68% | 27% | 0% | 2% |
| ss_L2_N32 | 4,343,300 | apply_macro_local_hoisted | 709 | 66% | 31% | 0% | 1% |
| ss_L4_N4 | 75,140 | apply_macro_local_hoisted | 685 | 39% | 27% | 0% | 16% |
| ss_L4_N6 | 242,500 | apply_macro_local_hoisted | 859 | 55% | 27% | 0% | 11% |
| ss_L4_N8 | 561,924 | apply_macro_local_hoisted | 866 | 65% | 24% | 0% | 6% |
| ss_L4_N12 | 1,853,572 | apply_macro_local_hoisted | 872 | 70% | 26% | 0% | 2% |
| ss_L4_N16 | 4,343,300 | apply_macro_local_hoisted | 833 | 71% | 27% | 0% | 1% |

The boundary closure is a separate pass on the flat operator and is FUSED into
the macro-element sweep on the semi-structured one, where it is inside the micro
loop and cannot carry a scope of its own without paying for one per element. So
its column is blank for the semi-structured rows and its cost is inside theirs;
the two boundary shares are not comparable and the sweep shares are not either.

For scale against the numbers already recorded: perf/baseline_grace.csv quotes
2574 MDOF/s for the packed residual KERNEL alone, and
docs/CVFEM_Operator_Cascade.md walks that down to 719.9 once the boundary closure
and a per-apply nodal gradient are added. What is measured here is the whole
Jacobian action with both, so it belongs beside the 719.9 and not beside the 2574.


## Per configuration

### flat_N16

75,140 dof, 72 threads, nid006555. 0.959 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `cvfem_hex8_ns_steady::apply_jacobian_action_packed` | element sweep | 903 | 0.686 | 759.2 | 99.0 | 71.5% |
| `cvfem_hex8_ns_steady::nodal_grad_strided` | nodal gradient | 913 | 0.147 | 160.9 | 467.1 | 15.3% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action` | boundary | 903 | 0.067 | 74.4 | 1010.2 | 7.0% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.048 | 53.0 | 1416.5 | 5.0% |
| `cvfem_hex8_ns_steady::apply_residual_packed` | element sweep | 7 | 0.006 | 787.3 | 95.4 | 0.6% |
| `cvfem_hex8_ns_steady::assemble_block_diag` | element sweep | 3 | 0.002 | 692.6 | 108.5 | 0.2% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_residual` | boundary | 7 | 0.001 | 130.1 | 577.4 | 0.1% |
| `Mesh::initialize_node_to_node_graph` | setup | 1 | 0.001 | 722.6 | 104.0 | 0.1% |
| `create_crs_graph_mem_conservative` | setup | 1 | 0.001 | 667.1 | 112.6 | 0.1% |
| `cvfem_hex8_ns_steady::precompute_element_bsr_slots` | setup | 1 | 0.000 | 322.6 | 232.9 | 0.0% |
| `create_n2e` | other | 1 | 0.000 | 302.1 | 248.7 | 0.0% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 40.9 | 1838.4 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 40.6 | 1850.8 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 77.2 | 972.7 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 46.7 | 1608.0 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 46.0 | 1633.0 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.693 | 72.3% |
| nodal gradient | 0.147 | 15.3% |
| boundary | 0.068 | 7.1% |
| constraints | 0.049 | 5.1% |
| setup | 0.002 | 0.2% |
| other | 0.000 | 0.0% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 1.366 | 455363.3 |
| `Function::apply` | 903 | 0.962 | 1064.9 |
| `CVFEMNavierStokes::apply` | 903 | 0.913 | 1011.0 |
| `cvfem_hex8_ns_steady::apply_jacobian_action` | 903 | 0.899 | 995.8 |
| `cvfem_hex8_ns_steady::assemble_nodal_q_grad` | 903 | 0.146 | 161.3 |
| `Function::copy_constrained_dofs` | 903 | 0.048 | 53.3 |
| `Function::gradient` | 7 | 0.009 | 1280.1 |
| `CVFEMNavierStokes::gradient` | 7 | 0.009 | 1237.9 |


### flat_N24

242,500 dof, 72 threads, nid006555. 1.728 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `cvfem_hex8_ns_steady::apply_jacobian_action_packed` | element sweep | 903 | 0.771 | 854.2 | 283.9 | 44.6% |
| `cvfem_hex8_ns_steady::nodal_grad_strided` | nodal gradient | 913 | 0.675 | 739.1 | 328.1 | 39.1% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action` | boundary | 903 | 0.182 | 201.8 | 1201.8 | 10.5% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.075 | 82.8 | 2928.1 | 4.3% |
| `cvfem_hex8_ns_steady::assemble_block_diag` | element sweep | 3 | 0.009 | 3123.8 | 77.6 | 0.5% |
| `cvfem_hex8_ns_steady::apply_residual_packed` | element sweep | 7 | 0.006 | 883.1 | 274.6 | 0.4% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_residual` | boundary | 7 | 0.002 | 303.2 | 799.8 | 0.1% |
| `Mesh::initialize_node_to_node_graph` | setup | 1 | 0.002 | 1775.5 | 136.6 | 0.1% |
| `create_crs_graph_mem_conservative` | setup | 1 | 0.002 | 1765.0 | 137.4 | 0.1% |
| `create_n2e` | other | 1 | 0.001 | 1074.3 | 225.7 | 0.1% |
| `cvfem_hex8_ns_steady::precompute_element_bsr_slots` | setup | 1 | 0.001 | 843.5 | 287.5 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.001 | 106.0 | 2287.1 | 0.0% |
| `DirichletConditions::gradient` | constraints | 7 | 0.001 | 104.4 | 2322.9 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 104.0 | 2332.8 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 102.3 | 2370.9 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 95.8 | 2530.1 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.787 | 45.5% |
| nodal gradient | 0.675 | 39.1% |
| boundary | 0.184 | 10.7% |
| constraints | 0.077 | 4.4% |
| setup | 0.004 | 0.3% |
| other | 0.001 | 0.1% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 2.688 | 896043.3 |
| `Function::apply` | 903 | 1.715 | 1898.8 |
| `CVFEMNavierStokes::apply` | 903 | 1.639 | 1815.1 |
| `cvfem_hex8_ns_steady::apply_jacobian_action` | 903 | 1.622 | 1796.2 |
| `cvfem_hex8_ns_steady::assemble_nodal_q_grad` | 903 | 0.667 | 739.1 |
| `Function::copy_constrained_dofs` | 903 | 0.075 | 83.1 |
| `CVFEMNavierStokes::initialize` | 1 | 0.020 | 20005.0 |
| `Function::gradient` | 7 | 0.016 | 2276.1 |


### flat_N32

561,924 dof, 72 threads, nid006555. 1.675 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `cvfem_hex8_ns_steady::apply_jacobian_action_packed` | element sweep | 903 | 0.738 | 817.6 | 687.3 | 44.1% |
| `cvfem_hex8_ns_steady::nodal_grad_strided` | nodal gradient | 913 | 0.473 | 518.3 | 1084.1 | 28.3% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action` | boundary | 903 | 0.345 | 382.6 | 1468.7 | 20.6% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.081 | 89.8 | 6258.1 | 4.8% |
| `cvfem_hex8_ns_steady::assemble_block_diag` | element sweep | 3 | 0.014 | 4718.1 | 119.1 | 0.8% |
| `cvfem_hex8_ns_steady::apply_residual_packed` | element sweep | 7 | 0.005 | 777.3 | 722.9 | 0.3% |
| `Mesh::initialize_node_to_node_graph` | setup | 1 | 0.004 | 4354.2 | 129.1 | 0.3% |
| `create_crs_graph_mem_conservative` | setup | 1 | 0.004 | 4338.5 | 129.5 | 0.3% |
| `create_n2e` | other | 1 | 0.003 | 3032.7 | 185.3 | 0.2% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_residual` | boundary | 7 | 0.002 | 352.8 | 1592.8 | 0.1% |
| `cvfem_hex8_ns_steady::precompute_element_bsr_slots` | setup | 1 | 0.002 | 2024.9 | 277.5 | 0.1% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 44.4 | 12642.3 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 42.0 | 13369.6 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 185.0 | 3037.2 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 183.8 | 3056.9 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 102.3 | 5493.9 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.758 | 45.2% |
| nodal gradient | 0.473 | 28.3% |
| boundary | 0.348 | 20.8% |
| constraints | 0.082 | 4.9% |
| setup | 0.011 | 0.6% |
| other | 0.003 | 0.2% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 3.744 | 1248043.3 |
| `Function::apply` | 903 | 1.655 | 1833.2 |
| `CVFEMNavierStokes::apply` | 903 | 1.573 | 1742.3 |
| `cvfem_hex8_ns_steady::apply_jacobian_action` | 903 | 1.553 | 1719.5 |
| `cvfem_hex8_ns_steady::assemble_nodal_q_grad` | 903 | 0.467 | 517.7 |
| `Function::copy_constrained_dofs` | 903 | 0.082 | 90.3 |
| `CVFEMNavierStokes::initialize` | 1 | 0.043 | 43175.9 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.017 | 5664.0 |


### flat_N48

1,853,572 dof, 72 threads, nid006555. 5.392 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `cvfem_hex8_ns_steady::apply_jacobian_action_packed` | element sweep | 903 | 2.607 | 2887.5 | 641.9 | 48.4% |
| `cvfem_hex8_ns_steady::nodal_grad_strided` | nodal gradient | 914 | 1.590 | 1740.1 | 1065.2 | 29.5% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action` | boundary | 903 | 0.991 | 1097.6 | 1688.8 | 18.4% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.081 | 89.4 | 20730.8 | 1.5% |
| `cvfem_hex8_ns_steady::assemble_block_diag` | element sweep | 3 | 0.045 | 15033.3 | 123.3 | 0.8% |
| `Mesh::initialize_node_to_node_graph` | setup | 1 | 0.017 | 16599.7 | 111.7 | 0.3% |
| `create_crs_graph_mem_conservative` | setup | 1 | 0.017 | 16575.6 | 111.8 | 0.3% |
| `cvfem_hex8_ns_steady::apply_residual_packed` | element sweep | 8 | 0.015 | 1812.5 | 1022.7 | 0.3% |
| `create_n2e` | other | 1 | 0.013 | 12674.8 | 146.2 | 0.2% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_residual` | boundary | 8 | 0.008 | 960.3 | 1930.2 | 0.1% |
| `cvfem_hex8_ns_steady::precompute_element_bsr_slots` | setup | 1 | 0.007 | 7301.3 | 253.9 | 0.1% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 440.4 | 4209.2 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 437.7 | 4234.4 | 0.0% |
| `DirichletConditions::gradient` | constraints | 8 | 0.000 | 50.5 | 36736.9 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 8 | 0.000 | 40.5 | 45765.7 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 113.0 | 16401.8 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 2.667 | 49.5% |
| nodal gradient | 1.590 | 29.5% |
| boundary | 0.999 | 18.5% |
| constraints | 0.082 | 1.5% |
| setup | 0.040 | 0.8% |
| other | 0.013 | 0.2% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 12.068 | 4022666.7 |
| `Function::apply` | 903 | 5.368 | 5944.8 |
| `CVFEMNavierStokes::apply` | 903 | 5.284 | 5852.0 |
| `cvfem_hex8_ns_steady::apply_jacobian_action` | 903 | 5.178 | 5734.2 |
| `cvfem_hex8_ns_steady::assemble_nodal_q_grad` | 903 | 1.575 | 1743.9 |
| `CVFEMNavierStokes::initialize` | 1 | 0.151 | 151457.0 |
| `Function::copy_constrained_dofs` | 903 | 0.082 | 90.6 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.057 | 18877.7 |


### flat_N64

4,343,300 dof, 72 threads, nid006555. 12.665 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `cvfem_hex8_ns_steady::apply_jacobian_action_packed` | element sweep | 903 | 6.465 | 7159.7 | 606.6 | 51.0% |
| `cvfem_hex8_ns_steady::nodal_grad_strided` | nodal gradient | 918 | 3.468 | 3778.1 | 1149.6 | 27.4% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action` | boundary | 903 | 2.281 | 2526.1 | 1719.3 | 18.0% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.143 | 158.2 | 27449.2 | 1.1% |
| `cvfem_hex8_ns_steady::assemble_block_diag` | element sweep | 3 | 0.099 | 33014.1 | 131.6 | 0.8% |
| `cvfem_hex8_ns_steady::apply_residual_packed` | element sweep | 12 | 0.047 | 3878.7 | 1119.8 | 0.4% |
| `Mesh::initialize_node_to_node_graph` | setup | 1 | 0.041 | 40658.2 | 106.8 | 0.3% |
| `create_crs_graph_mem_conservative` | setup | 1 | 0.041 | 40628.7 | 106.9 | 0.3% |
| `create_n2e` | other | 1 | 0.032 | 31574.2 | 137.6 | 0.2% |
| `cvfem_hex8_ns_steady::apply_boundary_scs_residual` | boundary | 12 | 0.028 | 2300.1 | 1888.3 | 0.2% |
| `cvfem_hex8_ns_steady::precompute_element_bsr_slots` | setup | 1 | 0.018 | 18338.9 | 236.8 | 0.1% |
| `Function::constraints_mask` | constraints | 1 | 0.001 | 799.4 | 5433.1 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.001 | 795.1 | 5462.4 | 0.0% |
| `DirichletConditions::gradient` | constraints | 12 | 0.001 | 56.4 | 77055.1 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 12 | 0.000 | 40.9 | 106119.2 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 129.5 | 33549.1 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 6.611 | 52.2% |
| nodal gradient | 3.468 | 27.4% |
| boundary | 2.309 | 18.2% |
| constraints | 0.146 | 1.2% |
| setup | 0.100 | 0.8% |
| other | 0.032 | 0.2% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 26.602 | 8867200.0 |
| `Function::apply` | 903 | 12.521 | 13866.2 |
| `CVFEMNavierStokes::apply` | 903 | 12.374 | 13703.8 |
| `cvfem_hex8_ns_steady::apply_jacobian_action` | 903 | 12.165 | 13472.3 |
| `cvfem_hex8_ns_steady::assemble_nodal_q_grad` | 903 | 3.411 | 3777.2 |
| `CVFEMNavierStokes::initialize` | 1 | 0.372 | 371765.0 |
| `Function::copy_constrained_dofs` | 903 | 0.144 | 159.6 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.131 | 43790.3 |


### ss_L2_N8

75,140 dof, 72 threads, nid006555. 0.360 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 1204 | 0.137 | 113.7 | 661.0 | 38.0% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 1221 | 0.092 | 75.4 | 996.4 | 25.6% |
| `to_semistructured` | other | 1 | 0.065 | 65069.2 | 1.2 | 18.1% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 1204 | 0.061 | 50.9 | 1477.3 | 17.0% |
| `sscvfem::block_diag` | element sweep | 4 | 0.003 | 628.2 | 119.6 | 0.7% |
| `sscvfem::build_scatter` | setup | 1 | 0.001 | 728.4 | 103.2 | 0.2% |
| `DirichletConditions::apply_value` | constraints | 13 | 0.001 | 46.6 | 1611.1 | 0.2% |
| `DirichletConditions::gradient` | constraints | 13 | 0.001 | 42.7 | 1761.4 | 0.2% |
| `create_dual_graph` | other | 1 | 0.000 | 256.1 | 293.4 | 0.1% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 76.5 | 981.8 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 29.4 | 2551.9 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 47.0 | 1599.8 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 45.8 | 1641.5 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 1204 | 0.000 | 0.0 | 2143799.7 | 0.0% |
| `sscvfem::apply_transient` | transient | 13 | 0.000 | 0.0 | 2048540.7 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.139 | 38.7% |
| nodal gradient | 0.092 | 25.6% |
| other | 0.065 | 18.2% |
| constraints | 0.063 | 17.4% |
| setup | 0.001 | 0.2% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 4 | 0.837 | 209167.0 |
| `Function::apply` | 1204 | 0.304 | 252.7 |
| `CVFEMNavierStokes::apply` | 1204 | 0.242 | 201.3 |
| `sscvfem::apply` | 1204 | 0.228 | 189.7 |
| `sscvfem::nodal_q_grad` | 1204 | 0.091 | 75.6 |
| `Function::copy_constrained_dofs` | 1204 | 0.061 | 51.0 |
| `Function::gradient` | 13 | 0.003 | 263.2 |
| `CVFEMNavierStokes::hessian_block_diag` | 4 | 0.003 | 838.8 |


### ss_L2_N12

242,500 dof, 72 threads, nid006555. 0.523 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 0.281 | 311.5 | 778.5 | 53.8% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 913 | 0.132 | 144.8 | 1674.2 | 25.3% |
| `to_semistructured` | other | 1 | 0.061 | 61327.5 | 4.0 | 11.7% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.042 | 46.0 | 5275.8 | 7.9% |
| `sscvfem::block_diag` | element sweep | 3 | 0.003 | 868.1 | 279.4 | 0.5% |
| `sscvfem::build_scatter` | setup | 1 | 0.002 | 2076.9 | 116.8 | 0.4% |
| `create_dual_graph` | other | 1 | 0.001 | 849.5 | 285.5 | 0.2% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 42.4 | 5723.3 | 0.1% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 40.3 | 6023.5 | 0.1% |
| `create_n2e` | other | 2 | 0.000 | 86.3 | 2809.7 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 103.2 | 2349.0 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 102.0 | 2376.4 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 82.3 | 2948.2 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.0 | 6655486.9 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.0 | 7119818.5 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.284 | 54.3% |
| nodal gradient | 0.132 | 25.3% |
| other | 0.062 | 11.9% |
| constraints | 0.042 | 8.1% |
| setup | 0.002 | 0.4% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 1.448 | 482576.7 |
| `Function::apply` | 903 | 0.469 | 519.2 |
| `CVFEMNavierStokes::apply` | 903 | 0.427 | 472.7 |
| `sscvfem::apply` | 903 | 0.413 | 457.1 |
| `sscvfem::nodal_q_grad` | 903 | 0.131 | 145.1 |
| `Function::copy_constrained_dofs` | 903 | 0.042 | 46.2 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.004 | 1485.2 |
| `Function::gradient` | 7 | 0.004 | 547.3 |


### ss_L2_N16

561,924 dof, 72 threads, nid006555. 1.052 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 0.674 | 745.9 | 753.4 | 64.0% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 913 | 0.273 | 299.3 | 1877.6 | 26.0% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.060 | 66.7 | 8427.2 | 5.7% |
| `to_semistructured` | other | 1 | 0.030 | 29561.3 | 19.0 | 2.8% |
| `sscvfem::block_diag` | element sweep | 3 | 0.007 | 2363.8 | 237.7 | 0.7% |
| `sscvfem::build_scatter` | setup | 1 | 0.005 | 5020.6 | 111.9 | 0.5% |
| `create_dual_graph` | other | 1 | 0.002 | 2118.8 | 265.2 | 0.2% |
| `create_n2e` | other | 2 | 0.000 | 206.8 | 2716.9 | 0.0% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 45.4 | 12367.5 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 42.6 | 13188.0 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 184.3 | 3049.0 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 182.4 | 3080.9 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 107.0 | 5249.2 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.0 | 13642709.1 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.1 | 5499384.8 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.681 | 64.7% |
| nodal gradient | 0.273 | 26.0% |
| constraints | 0.061 | 5.8% |
| other | 0.032 | 3.0% |
| setup | 0.005 | 0.5% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 3.083 | 1027636.7 |
| `Function::apply` | 903 | 1.025 | 1135.5 |
| `CVFEMNavierStokes::apply` | 903 | 0.964 | 1068.1 |
| `sscvfem::apply` | 903 | 0.944 | 1045.8 |
| `sscvfem::nodal_q_grad` | 903 | 0.270 | 299.3 |
| `Function::copy_constrained_dofs` | 903 | 0.060 | 66.9 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.011 | 3727.4 |
| `Function::gradient` | 7 | 0.008 | 1136.7 |


### ss_L2_N24

1,853,572 dof, 72 threads, nid006555. 3.309 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 2.213 | 2450.6 | 756.4 | 66.9% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 914 | 0.909 | 994.7 | 1863.5 | 27.5% |
| `to_semistructured` | other | 1 | 0.081 | 81472.4 | 22.8 | 2.5% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.050 | 55.3 | 33497.6 | 1.5% |
| `sscvfem::block_diag` | element sweep | 3 | 0.023 | 7752.2 | 239.1 | 0.7% |
| `sscvfem::build_scatter` | setup | 1 | 0.021 | 20793.0 | 89.1 | 0.6% |
| `create_dual_graph` | other | 1 | 0.008 | 7674.0 | 241.5 | 0.2% |
| `create_n2e` | other | 2 | 0.002 | 874.6 | 2119.2 | 0.1% |
| `DirichletConditions::gradient` | constraints | 8 | 0.001 | 78.2 | 23711.6 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 437.7 | 4234.4 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 434.4 | 4267.0 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 8 | 0.000 | 41.0 | 45233.1 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 114.9 | 16129.5 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.1 | 26392210.5 | 0.0% |
| `sscvfem::apply_transient` | transient | 8 | 0.000 | 0.0 | 62195445.8 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 2.236 | 67.6% |
| nodal gradient | 0.909 | 27.5% |
| other | 0.091 | 2.7% |
| constraints | 0.052 | 1.6% |
| setup | 0.021 | 0.6% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 9.751 | 3250173.3 |
| `Function::apply` | 903 | 3.253 | 3602.8 |
| `CVFEMNavierStokes::apply` | 903 | 3.201 | 3545.1 |
| `sscvfem::apply` | 903 | 3.114 | 3448.5 |
| `sscvfem::nodal_q_grad` | 903 | 0.899 | 995.3 |
| `Function::copy_constrained_dofs` | 903 | 0.051 | 56.2 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.040 | 13484.6 |
| `Function::gradient` | 8 | 0.028 | 3464.1 |


### ss_L2_N32

4,343,300 dof, 72 threads, nid006555. 8.493 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 5.529 | 6123.3 | 709.3 | 65.1% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 914 | 2.632 | 2879.4 | 1508.4 | 31.0% |
| `to_semistructured` | other | 1 | 0.104 | 103992.0 | 41.8 | 1.2% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.099 | 109.6 | 39643.3 | 1.2% |
| `sscvfem::block_diag` | element sweep | 3 | 0.051 | 16842.0 | 257.9 | 0.6% |
| `sscvfem::build_scatter` | setup | 1 | 0.050 | 50070.8 | 86.7 | 0.6% |
| `create_dual_graph` | other | 1 | 0.019 | 19421.3 | 223.6 | 0.2% |
| `create_n2e` | other | 2 | 0.005 | 2695.1 | 1611.6 | 0.1% |
| `DirichletConditions::gradient` | constraints | 8 | 0.001 | 114.2 | 38041.5 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.001 | 774.6 | 5607.0 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.001 | 772.2 | 5624.3 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 8 | 0.000 | 42.4 | 102415.2 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 150.0 | 28962.1 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.1 | 51892926.2 | 0.0% |
| `sscvfem::apply_transient` | transient | 8 | 0.000 | 0.2 | 18217107.5 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 5.580 | 65.7% |
| nodal gradient | 2.632 | 31.0% |
| other | 0.129 | 1.5% |
| constraints | 0.102 | 1.2% |
| setup | 0.050 | 0.6% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 22.384 | 7461233.3 |
| `Function::apply` | 903 | 8.456 | 9364.5 |
| `CVFEMNavierStokes::apply` | 903 | 8.354 | 9251.0 |
| `sscvfem::apply` | 903 | 8.141 | 9015.7 |
| `sscvfem::nodal_q_grad` | 903 | 2.607 | 2887.1 |
| `Function::copy_constrained_dofs` | 903 | 0.100 | 110.8 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.090 | 29842.0 |
| `Function::gradient` | 8 | 0.068 | 8473.0 |


### ss_L4_N4

75,140 dof, 72 threads, nid006555. 0.259 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 0.099 | 109.8 | 684.5 | 38.3% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 913 | 0.070 | 76.2 | 986.4 | 26.9% |
| `to_semistructured` | other | 1 | 0.046 | 46489.2 | 1.6 | 18.0% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.041 | 45.6 | 1646.5 | 15.9% |
| `sscvfem::block_diag` | element sweep | 3 | 0.001 | 400.5 | 187.6 | 0.5% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 53.4 | 1407.0 | 0.1% |
| `sscvfem::build_scatter` | setup | 1 | 0.000 | 345.5 | 217.5 | 0.1% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 42.7 | 1760.7 | 0.1% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 77.0 | 975.7 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 47.0 | 1599.8 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 46.3 | 1624.5 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.0 | 2092571.4 | 0.0% |
| `create_dual_graph` | other | 1 | 0.000 | 32.2 | 2334.5 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 3.7 | 20332.9 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.1 | 1103060.4 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.100 | 38.8% |
| nodal gradient | 0.070 | 26.9% |
| other | 0.047 | 18.0% |
| constraints | 0.042 | 16.2% |
| setup | 0.000 | 0.1% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 0.630 | 209943.0 |
| `Function::apply` | 903 | 0.226 | 249.9 |
| `CVFEMNavierStokes::apply` | 903 | 0.184 | 203.7 |
| `sscvfem::apply` | 903 | 0.168 | 186.6 |
| `sscvfem::nodal_q_grad` | 903 | 0.069 | 76.4 |
| `Function::copy_constrained_dofs` | 903 | 0.041 | 45.8 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.002 | 613.3 |
| `Function::gradient` | 7 | 0.002 | 259.0 |


### ss_L4_N6

242,500 dof, 72 threads, nid006555. 0.470 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 0.255 | 282.5 | 858.5 | 54.3% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 913 | 0.128 | 139.8 | 1734.2 | 27.2% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.048 | 53.3 | 4547.0 | 10.2% |
| `to_semistructured` | other | 1 | 0.034 | 33711.2 | 7.2 | 7.2% |
| `sscvfem::block_diag` | element sweep | 3 | 0.002 | 796.4 | 304.5 | 0.5% |
| `DirichletConditions::gradient` | constraints | 7 | 0.001 | 202.3 | 1198.8 | 0.3% |
| `sscvfem::build_scatter` | setup | 1 | 0.001 | 986.6 | 245.8 | 0.2% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 42.6 | 5691.3 | 0.1% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 103.5 | 2343.6 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 103.2 | 2349.0 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 102.0 | 2376.4 | 0.0% |
| `create_dual_graph` | other | 1 | 0.000 | 100.9 | 2404.5 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.0 | 7231944.7 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 9.1 | 26766.3 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.0 | 0.0 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.257 | 54.8% |
| nodal gradient | 0.128 | 27.2% |
| constraints | 0.050 | 10.7% |
| other | 0.034 | 7.2% |
| setup | 0.001 | 0.2% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 1.444 | 481210.0 |
| `Function::apply` | 903 | 0.449 | 497.5 |
| `CVFEMNavierStokes::apply` | 903 | 0.401 | 443.6 |
| `sscvfem::apply` | 903 | 0.382 | 423.1 |
| `sscvfem::nodal_q_grad` | 903 | 0.127 | 140.2 |
| `Function::copy_constrained_dofs` | 903 | 0.048 | 53.6 |
| `Function::gradient` | 7 | 0.006 | 915.2 |
| `CVFEMNavierStokes::gradient` | 7 | 0.005 | 711.6 |


### ss_L4_N8

561,924 dof, 72 threads, nid006555. 0.876 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 875 | 0.568 | 649.2 | 865.6 | 64.8% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 885 | 0.212 | 239.8 | 2343.6 | 24.2% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 875 | 0.048 | 55.0 | 10219.6 | 5.5% |
| `to_semistructured` | other | 1 | 0.039 | 38722.5 | 14.5 | 4.4% |
| `sscvfem::block_diag` | element sweep | 3 | 0.005 | 1788.8 | 314.1 | 0.6% |
| `sscvfem::build_scatter` | setup | 1 | 0.003 | 2521.5 | 222.9 | 0.3% |
| `DirichletConditions::gradient` | constraints | 7 | 0.000 | 44.0 | 12769.5 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 7 | 0.000 | 40.3 | 13946.0 | 0.0% |
| `create_dual_graph` | other | 1 | 0.000 | 253.4 | 2217.2 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 182.9 | 3072.9 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 181.4 | 3097.1 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 80.8 | 6952.4 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 24.6 | 22882.3 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 875 | 0.000 | 0.0 | 12651972.9 | 0.0% |
| `sscvfem::apply_transient` | transient | 7 | 0.000 | 0.0 | 0.0 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 0.573 | 65.4% |
| nodal gradient | 0.212 | 24.2% |
| constraints | 0.049 | 5.6% |
| other | 0.039 | 4.5% |
| setup | 0.003 | 0.3% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 2.852 | 950813.3 |
| `Function::apply` | 875 | 0.846 | 966.8 |
| `CVFEMNavierStokes::apply` | 875 | 0.797 | 911.1 |
| `sscvfem::apply` | 875 | 0.778 | 889.2 |
| `sscvfem::nodal_q_grad` | 875 | 0.209 | 239.4 |
| `Function::copy_constrained_dofs` | 875 | 0.048 | 55.2 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.009 | 3029.3 |
| `Function::gradient` | 7 | 0.007 | 930.9 |


### ss_L4_N12

1,853,572 dof, 72 threads, nid006555. 2.755 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 1.919 | 2125.7 | 872.0 | 69.7% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 914 | 0.720 | 788.1 | 2352.0 | 26.1% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.044 | 48.7 | 38052.6 | 1.6% |
| `to_semistructured` | other | 1 | 0.041 | 40561.0 | 45.7 | 1.5% |
| `sscvfem::block_diag` | element sweep | 3 | 0.018 | 6158.8 | 301.0 | 0.7% |
| `sscvfem::build_scatter` | setup | 1 | 0.009 | 9405.6 | 197.1 | 0.3% |
| `create_dual_graph` | other | 1 | 0.001 | 858.5 | 2159.0 | 0.0% |
| `DirichletConditions::gradient` | constraints | 8 | 0.000 | 62.4 | 29701.8 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.000 | 417.0 | 4445.1 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.000 | 415.1 | 4465.5 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 8 | 0.000 | 40.4 | 45866.9 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 82.3 | 22534.6 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 115.9 | 15996.9 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.1 | 14748610.1 | 0.0% |
| `sscvfem::apply_transient` | transient | 8 | 0.000 | 0.0 | 62195445.8 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 1.938 | 70.3% |
| nodal gradient | 0.720 | 26.1% |
| constraints | 0.046 | 1.7% |
| other | 0.042 | 1.5% |
| setup | 0.009 | 0.3% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 9.347 | 3115596.7 |
| `Function::apply` | 903 | 2.783 | 3081.7 |
| `CVFEMNavierStokes::apply` | 903 | 2.736 | 3030.3 |
| `sscvfem::apply` | 903 | 2.634 | 2917.1 |
| `sscvfem::nodal_q_grad` | 903 | 0.713 | 789.3 |
| `Function::copy_constrained_dofs` | 903 | 0.045 | 49.8 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.033 | 11113.3 |
| `Function::gradient` | 8 | 0.024 | 2947.6 |


### ss_L4_N16

4,343,300 dof, 72 threads, nid006555. 6.698 s in non-container scopes.

| scope | kind | calls | seconds | us/call | MDOF/s | share |
|---|---|---|---|---|---|---|
| `sscvfem::apply_macro_local_hoisted` | element sweep | 903 | 4.708 | 5213.3 | 833.1 | 70.3% |
| `sscvfem::nodal_grad_strided` | nodal gradient | 918 | 1.813 | 1975.2 | 2198.9 | 27.1% |
| `DirichletConditions::copy_constrained_dofs` | constraints | 903 | 0.077 | 85.1 | 51061.8 | 1.1% |
| `sscvfem::block_diag` | element sweep | 3 | 0.045 | 14869.6 | 292.1 | 0.7% |
| `to_semistructured` | other | 1 | 0.025 | 25055.2 | 173.3 | 0.4% |
| `sscvfem::build_scatter` | setup | 1 | 0.025 | 24713.8 | 175.7 | 0.4% |
| `create_dual_graph` | other | 1 | 0.002 | 2132.6 | 2036.6 | 0.0% |
| `DirichletConditions::gradient` | constraints | 12 | 0.001 | 72.8 | 59646.8 | 0.0% |
| `Function::constraints_mask` | constraints | 1 | 0.001 | 767.2 | 5661.0 | 0.0% |
| `DirichletConditions::mask` | constraints | 1 | 0.001 | 764.6 | 5680.4 | 0.0% |
| `DirichletConditions::apply_value` | constraints | 12 | 0.001 | 43.4 | 100048.2 | 0.0% |
| `create_n2e` | other | 2 | 0.000 | 194.9 | 22284.0 | 0.0% |
| `DirichletConditions::apply` | constraints | 1 | 0.000 | 133.0 | 32647.1 | 0.0% |
| `sscvfem::apply_transient_action` | transient | 903 | 0.000 | 0.1 | 49848686.0 | 0.0% |
| `sscvfem::apply_transient` | transient | 12 | 0.000 | 0.1 | 36434278.7 | 0.0% |

| kind | seconds | share |
|---|---|---|
| element sweep | 4.752 | 71.0% |
| nodal gradient | 1.813 | 27.1% |
| constraints | 0.080 | 1.2% |
| other | 0.028 | 0.4% |
| setup | 0.025 | 0.4% |
| transient | 0.000 | 0.0% |

Containers, listed apart because the rows above are inside them and adding
both would count the same seconds twice:

| scope | calls | seconds | us/call |
|---|---|---|---|
| `BiCGStab::apply` | 3 | 20.564 | 6854833.3 |
| `Function::apply` | 903 | 6.787 | 7516.1 |
| `CVFEMNavierStokes::apply` | 903 | 6.707 | 7427.3 |
| `sscvfem::apply` | 903 | 6.501 | 7199.1 |
| `sscvfem::nodal_q_grad` | 903 | 1.789 | 1981.7 |
| `CVFEMNavierStokes::hessian_block_diag` | 3 | 0.082 | 27319.6 |
| `Function::gradient` | 12 | 0.080 | 6676.2 |
| `CVFEMNavierStokes::gradient` | 12 | 0.079 | 6599.9 |


## Throughput against problem size

MDOF/s per scope, the same scope across every configuration. A kernel that is
memory bound flattens; one that is not keeps climbing with the problem until it
does. A number measured below saturation is not a throughput, so the smallest
sizes are here to show where that begins rather than to be quoted.

### `cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action`

| run | dof | calls | seconds | MDOF/s |
|---|---|---|---|---|
| flat_N16 | 75,140 | 903 | 0.067 | 1010.2 |
| flat_N24 | 242,500 | 903 | 0.182 | 1201.8 |
| flat_N32 | 561,924 | 903 | 0.345 | 1468.7 |
| flat_N48 | 1,853,572 | 903 | 0.991 | 1688.8 |
| flat_N64 | 4,343,300 | 903 | 2.281 | 1719.3 |
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
| flat_N16 | 75,140 | 7 | 0.001 | 577.4 |
| flat_N24 | 242,500 | 7 | 0.002 | 799.8 |
| flat_N32 | 561,924 | 7 | 0.002 | 1592.8 |
| flat_N48 | 1,853,572 | 8 | 0.008 | 1930.2 |
| flat_N64 | 4,343,300 | 12 | 0.028 | 1888.3 |
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
| flat_N16 | 75,140 | 903 | 0.686 | 99.0 |
| flat_N24 | 242,500 | 903 | 0.771 | 283.9 |
| flat_N32 | 561,924 | 903 | 0.738 | 687.3 |
| flat_N48 | 1,853,572 | 903 | 2.607 | 641.9 |
| flat_N64 | 4,343,300 | 903 | 6.465 | 606.6 |
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
| flat_N16 | 75,140 | 7 | 0.006 | 95.4 |
| flat_N24 | 242,500 | 7 | 0.006 | 274.6 |
| flat_N32 | 561,924 | 7 | 0.005 | 722.9 |
| flat_N48 | 1,853,572 | 8 | 0.015 | 1022.7 |
| flat_N64 | 4,343,300 | 12 | 0.047 | 1119.8 |
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
| flat_N16 | 75,140 | 3 | 0.002 | 108.5 |
| flat_N24 | 242,500 | 3 | 0.009 | 77.6 |
| flat_N32 | 561,924 | 3 | 0.014 | 119.1 |
| flat_N48 | 1,853,572 | 3 | 0.045 | 123.3 |
| flat_N64 | 4,343,300 | 3 | 0.099 | 131.6 |
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
| flat_N16 | 75,140 | 913 | 0.147 | 467.1 |
| flat_N24 | 242,500 | 913 | 0.675 | 328.1 |
| flat_N32 | 561,924 | 913 | 0.473 | 1084.1 |
| flat_N48 | 1,853,572 | 914 | 1.590 | 1065.2 |
| flat_N64 | 4,343,300 | 918 | 3.468 | 1149.6 |
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
| ss_L2_N8 | 75,140 | 1204 | 0.137 | 661.0 |
| ss_L2_N12 | 242,500 | 903 | 0.281 | 778.5 |
| ss_L2_N16 | 561,924 | 903 | 0.674 | 753.4 |
| ss_L2_N24 | 1,853,572 | 903 | 2.213 | 756.4 |
| ss_L2_N32 | 4,343,300 | 903 | 5.529 | 709.3 |
| ss_L4_N4 | 75,140 | 903 | 0.099 | 684.5 |
| ss_L4_N6 | 242,500 | 903 | 0.255 | 858.5 |
| ss_L4_N8 | 561,924 | 875 | 0.568 | 865.6 |
| ss_L4_N12 | 1,853,572 | 903 | 1.919 | 872.0 |
| ss_L4_N16 | 4,343,300 | 903 | 4.708 | 833.1 |

### `sscvfem::block_diag`

| run | dof | calls | seconds | MDOF/s |
|---|---|---|---|---|
| flat_N16 | 75,140 | -- | -- | -- |
| flat_N24 | 242,500 | -- | -- | -- |
| flat_N32 | 561,924 | -- | -- | -- |
| flat_N48 | 1,853,572 | -- | -- | -- |
| flat_N64 | 4,343,300 | -- | -- | -- |
| ss_L2_N8 | 75,140 | 4 | 0.003 | 119.6 |
| ss_L2_N12 | 242,500 | 3 | 0.003 | 279.4 |
| ss_L2_N16 | 561,924 | 3 | 0.007 | 237.7 |
| ss_L2_N24 | 1,853,572 | 3 | 0.023 | 239.1 |
| ss_L2_N32 | 4,343,300 | 3 | 0.051 | 257.9 |
| ss_L4_N4 | 75,140 | 3 | 0.001 | 187.6 |
| ss_L4_N6 | 242,500 | 3 | 0.002 | 304.5 |
| ss_L4_N8 | 561,924 | 3 | 0.005 | 314.1 |
| ss_L4_N12 | 1,853,572 | 3 | 0.018 | 301.0 |
| ss_L4_N16 | 4,343,300 | 3 | 0.045 | 292.1 |

### `sscvfem::nodal_grad_strided`

| run | dof | calls | seconds | MDOF/s |
|---|---|---|---|---|
| flat_N16 | 75,140 | -- | -- | -- |
| flat_N24 | 242,500 | -- | -- | -- |
| flat_N32 | 561,924 | -- | -- | -- |
| flat_N48 | 1,853,572 | -- | -- | -- |
| flat_N64 | 4,343,300 | -- | -- | -- |
| ss_L2_N8 | 75,140 | 1221 | 0.092 | 996.4 |
| ss_L2_N12 | 242,500 | 913 | 0.132 | 1674.2 |
| ss_L2_N16 | 561,924 | 913 | 0.273 | 1877.6 |
| ss_L2_N24 | 1,853,572 | 914 | 0.909 | 1863.5 |
| ss_L2_N32 | 4,343,300 | 914 | 2.632 | 1508.4 |
| ss_L4_N4 | 75,140 | 913 | 0.070 | 986.4 |
| ss_L4_N6 | 242,500 | 913 | 0.128 | 1734.2 |
| ss_L4_N8 | 561,924 | 885 | 0.212 | 2343.6 |
| ss_L4_N12 | 1,853,572 | 914 | 0.720 | 2352.0 |
| ss_L4_N16 | 4,343,300 | 918 | 1.813 | 2198.9 |


## Provenance

| field | value |
|---|---|
| generated | 2026-09-09 21:49:11 |
| configurations | 15 |
| machines | nid006555 |
