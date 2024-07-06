# SOLVERS

## Basic 2-level method (32 OpenMP treads on Desktop)

```yaml
SFEM_MATRIX_FREE: 1
SFEM_COARSE_MATRIX_FREE: 1
SFEM_OPERATOR: LinearElasticity
SFEM_BLOCK_SIZE: 3
SFEM_USE_PRECONDITIONER: 1
SFEM_USE_CHEB: 1
SFEM_DEBUG: 0
SFEM_MG: 1
SFEM_CHEB_EIG_MAX_SCALE: 1.020000
SFEM_TOL: 0.000000
SFEM_SMOOTHER_SWEEPS: 30
SFEM_CHEB_EIG_TOL: 0.000000
```

PowerMethod (2718): lambda = 0.25154
crs_graph.c: build_n2e		4.08965 seconds
crs_graph.c: overestimate nnz	4.03965 seconds
crs_graph.c: build nz structure	8.64792 seconds
Multigrid
iter	abs		rel		rate
0	0.152666	-		-
1	0.0044969	0.0294559	0.0294559
2	0.000377827	0.00247486	0.0840193
3	7.305e-05	0.000478497	0.193342
4	1.93513e-05	0.000126756	0.264906
5	5.99078e-06	3.92412e-05	0.309579
6	2.04898e-06	1.34214e-05	0.342023
7	7.52794e-07	4.931e-06	0.367399
8	2.92075e-07	1.91317e-06	0.387988
9	1.18306e-07	7.74937e-07	0.405055
10	4.9623e-08	3.25043e-07	0.419445
11	2.1426e-08	1.40346e-07	0.431775
12	9.48121e-09	6.21044e-08	0.44251
13	4.28558e-09	2.80717e-08	0.452008
14	1.97369e-09	1.29282e-08	0.460543
15	9.24332e-10	6.05462e-09	0.468326
----------------------------------------
mgsolve (MACRO_TET4):
----------------------------------------
#elements 236191744 #nodes 316693569 #dofs 950080707
TTS:		66314.5 [s], compute 66299.9 [s] (solve: 56863.2 [s], init: 9436.72 [s])
residual:	9.24332e-10


## perf_950080707.csv (fine level)
```
function,seconds
create_crs_graph,0
destroy_crs_graph,0
hessian_crs,0
hessian_diag,0
gradient,0
apply,10136
value,0
apply_constraints,0.0666279
constraints_gradient,0
apply_zero_constraints,0.0668811
copy_constrained_dofs,28.9852
report_solution,0
```

## perf_119425635.csv (coarse level)
```
function,seconds
create_crs_graph,0
destroy_crs_graph,0
hessian_crs,0
hessian_diag,0.511496
gradient,0
apply,27801
value,0
apply_constraints,0
constraints_gradient,0
apply_zero_constraints,0
copy_constrained_dofs,100.477
report_solution,0
initial_guess,0
```