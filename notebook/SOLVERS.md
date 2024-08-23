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


```c
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
```	

## perf_950080707.csv (fine level)
```c
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
```c
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


## Basic two-level method on NVIDIA P100 GPU (Piz Daint)

```yaml

SFEM_MATRIX_FREE: 1
SFEM_COARSE_MATRIX_FREE: 1
SFEM_OPERATOR: LinearElasticity
SFEM_BLOCK_SIZE: 3
SFEM_USE_PRECONDITIONER: 0
SFEM_USE_CHEB: 1
SFEM_DEBUG: 0
SFEM_MG: 1
SFEM_CHEB_EIG_MAX_SCALE: 1.020000
SFEM_TOL: 1e-6
SFEM_SMOOTHER_SWEEPS: 30
SFEM_CHEB_EIG_TOL: 1e-7
```

```c
PowerMethod (1344): lambda = 0.55367
build_n2e: allocating 0.0402536 GB
build_n2e: allocating 0.472383 GB
crs_graph.c: build_n2e		0.809775 seconds
crs_graph.c: build nz (mem conservative) structure	3.50567 seconds
Multigrid
iter	abs		rel		rate
0	0.165836	-		-
1	0.00667081	0.0402253	0.0402253
2	0.000985268	0.00594122	0.147698
3	0.000259423	0.00156434	0.263302
4	8.21721e-05	0.000495503	0.316749
5	2.87756e-05	0.000173518	0.350186
6	1.07654e-05	6.49158e-05	0.374115
7	4.21896e-06	2.54406e-05	0.391901
8	1.70991e-06	1.03109e-05	0.405293
9	7.10327e-07	4.28331e-06	0.415417
----------------------------------------
mgsolve (MACRO_TET4):
----------------------------------------
#elements 29523968 #nodes 39808545 #dofs 119425635
TTS:		1036.09 [s], compute 1028.83 [s] (solve: 949.825 [s], init: 78.9262 [s])
residual:	7.10327e-07
```

## Two-level laplacian with coarse grid solver using SpMV vs MF

Note that the fine level CRS does not fit on the P100 for this problem size.

Coarse with CRS SpMV
```c
#elements 31031296 #nodes 41828417 #dofs 41828417
TTS:		110.587 [s], compute 104.27 [s] (solve: 88.2348 [s], init: 15.9743 [s])
residual:	3.88119e-10
```

MF

```c
#elements 31031296 #nodes 41828417 #dofs 41828417
TTS:		139.182 [s], compute 132.298 [s] (solve: 116.557 [s], init: 15.6775 [s])
residual:	3.88119e-10
```

For basic TET4 as expected CRS is faster. Matrix-based AMG or equivalent may be used for the coarse level.

