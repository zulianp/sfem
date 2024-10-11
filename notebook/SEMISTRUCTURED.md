# HEX8 semistructured discretization

Basic implementation of laplace operator using semistructured HEX8 meshes

## Performance on ARM M1 CPU with 8 threads

### Plain conjugate gradient

```c
SFEM_HEX8_ASSUME_AFFINE=1 
SFEM_ELEMENT_REFINE_LEVEL=8
SFEM_REPEAT=666

#elements 59319 #microelements 30371328 #nodes 30664297
#nodexelement 729 #microelementsxelement 512
Operator TTS:			0.0721					[s]
Operator throughput:	0.8						[ME/s]
Operator throughput:	421.0					[MmicroE/s]
Operator throughput:	425.1					[MDOF/s]
Operator memory 0.490629 (2 x coeffs) + 0.00142366 (FFFs) + 0.172974 (index) = 0.665027 [GB]
Total:			48.3685	[s]
```


Solving Poisson problem in a 3D arc with affine HEX8 discretization with plain CG in approx. 3 hours
```c
SFEM_HEX8_ASSUME_AFFINE=1 
SFEM_ELEMENT_REFINE_LEVEL=8

Iterations 5882 
residual abs: 1.34694e-11 
residual rel: 9.89148e-11
----------------------------------------
obstacle (PROTEUS_HEX8):
----------------------------------------
#elements 591552 #nodes 304390625 #dofs 304390625
TTS:		10136.8 [s], solve: 10130.4 [s]
```

### Two-level geometric multigrid

**Experiment 1**

295555 [Solve DOFS/s]

```c
#elements 59319 #nodes 64000 #dofs 91992891
PowerMethod (74): lambda = 0.0377093
Multigrid
iter	abs		rel		rate
0	0.00987967	-		-
1	0.0010982	0.111157	0.111157
2	0.000132116	0.0133725	0.120303
3	1.77319e-05	0.00179479	0.134215
4	2.72229e-06	0.000275545	0.153525
5	4.73179e-07	4.78943e-05	0.173817
6	8.97762e-08	9.08697e-06	0.18973
7	1.79979e-08	1.82171e-06	0.200475
8	3.74158e-09	3.78715e-07	0.20789
9	7.98989e-10	8.08721e-08	0.213543
----------------------------------------
mgsolve (PROTEUS_HEX8):
----------------------------------------
#elements 59319 #nodes 64000 #dofs 91992891
TTS:		312.95 [s], compute 311.254 [s] (solve: 259.869 [s], init: 51.3746 [s])
residual:	7.98989e-10
```

**Experiment 2**

286782 [Solve DOFS/s]. The dof coarsening factor is `490`. Fine/coarse operator cost ratio = `957.922 [s] / 39.9906 [s] = 23`

```c
#elements 205379 #nodes 216000 #dofs 317471451
PowerMethod (60): lambda = 0.0248176
Multigrid
iter	abs		rel		rate
0	0.00984173	-		-
1	0.00110032	0.111802	0.111802
2	0.000129651	0.0131735	0.11783
3	1.66335e-05	0.0016901	0.128295
4	2.40446e-06	0.000244312	0.144555
5	3.95783e-07	4.02147e-05	0.164604
6	7.22104e-08	7.33716e-06	0.182449
7	1.40711e-08	1.42974e-06	0.194863
8	2.85613e-09	2.90206e-07	0.202978
9	5.96252e-10	6.05841e-08	0.208762
----------------------------------------
mgsolve (PROTEUS_HEX8):
----------------------------------------
#elements 205379 #nodes 216000 #dofs 317471451
TTS:		1131.7 [s], compute 1107.01 [s] (solve: 964.925 [s], init: 142.05 [s])
residual:	5.96252e-10
```

## Solving contact for linear elasticity with MPRGP

```c
SFEM_HEX8_ASSUME_AFFINE=1 
SFEM_ELEMENT_REFINE_LEVEL=8

// Example 1
Iterations 10011
residual abs: 9.76926e-09
residual rel: ~2.2e-14
#cg_steps				9631
#expansion_steps		362
#proportioning_steps	19
----------------------------------------
obstacle (PROTEUS_HEX8):
----------------------------------------
#elements 1872 #nodes 1030625 #dofs 3091875
TTS:		480.354 [s], solve: 480.285 [s]


// Example 2
Iterations 16501 
residual abs: 9.97225e-09
residual rel: ~2.2e-14
#cg_steps				15809
#expansion_steps		672
#proportioning_steps	21
----------------------------------------
obstacle (PROTEUS_HEX8):
----------------------------------------
#elements 1872 #nodes 1030625 #dofs 3091875
TTS:		451.011 [s], solve: 450.914 [s]
```


# Laplacian -- GPU implementation on P100 (level 8)

On ARM latop TTS is `174.152 [s]`

## 1 Thread x Macro-element 

**Baseline**

`1.26x` speed-up over CPU implementation (bad)

```c
#elements 328509 #nodes 343000 #dofs 169112377
TTS:		145.841 [s], compute 137.394 [s] (solve: 124.731 [s], init: 12.6104 [s])
residual:	1.28746e-10

```

**Fixed size implementation (LEVEL=8 === BLOCK_SIZE=9)**

`2.8x` speed-up over GPU baseline, and `3.5x` speed-up over CPU implementation 

```c
#elements 328509 #nodes 343000 #dofs 169112377
TTS:		57.0432 [s], compute 48.9466 [s] (solve: 43.8373 [s], init: 5.05854 [s])
residual:	1.28746e-10
```
