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

On ARM latop TTS `compute` is `174.152 [s]` with `OMP_NUM_THREADS=8`

Level 8 has the highest throughput on the ARM but very low on P100. Warp level solution might be required

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


## Effects of SFEM_ELEMENT_REFINE_LEVEL on P100

This study should be redone with same `#dofs`

| SFEM_ELEMENT_REFINE_LEVEL | #dofs | TTS [ms] | TP [GDOF/s] |
|---|---|---|---|
| 2 | 115501303 | 116.10 | 0.999 |
| 4 | 108531333 | 74.177 | 1.4   |
| 6 | 107171875 | 97.887 | 1.09  |
| 8 | 105823817	| 97.971 | 1.08  |
| 10| 118370771 | 108.08 | 1.09	 |

### With interior idx gen
Here we assume that the interior dofs are order implicitly

| SFEM_ELEMENT_REFINE_LEVEL | #dofs | TTS [ms] | TP [GDOF/s] |
|---|---|---|---|
| 4 | 119823157 | 71.512 | 1.67 |
| 8 | 105823817 | 64.467 | 1.64 |

### With warp code

| SFEM_ELEMENT_REFINE_LEVEL | #dofs | TTS [ms] | TP [GDOF/s] |
|---|---|---|---|
| 4 | 119823157 | 41.371 | 2.89 |
| 8 | 116930169 | 27.207 | **4.29** |

Single precision 8 levels: `116930169` `22.995 [ms]`  `5 [GDOF/s]`


# P100 vs GH200 (Poisson's problem, 3D mesh/1D solution)

**P100**
```c
Multigrid
iter	abs		rel		rate
0	0.0432236	-		-
1	0.00136906	0.031674	0.031674
2	5.06484e-05	0.00117178	0.0369949
3	1.96645e-06	4.54948e-05	0.0388255
4	7.82113e-08	1.80946e-06	0.0397729
5	3.15638e-09	7.30244e-08	0.0403571
6	1.28658e-10	2.97656e-09	0.0407612
----------------------------------------
mgsolve (PROTEUS_HEX8):
----------------------------------------
#elements 226981 #nodes 238328 #dofs 116930169
TTS:		22.1533 [s], compute 16.1473 [s] (solve: 13.8048 [s], init: 2.29316 [s])
residual:	1.28658e-10
```

**GH200 (CPU/Grace only)**
```c
Multigrid
iter	abs		rel		rate
0	0.0432236	-		-
1	0.00136907	0.0316742	0.0316742
2	5.06491e-05	0.00117179	0.0369952
3	1.96648e-06	4.54956e-05	0.0388257
4	7.82131e-08	1.8095e-06	0.0397731
5	3.15647e-09	7.30265e-08	0.0403572
6	1.28662e-10	2.97665e-09	0.0407613
----------------------------------------
mgsolve (PROTEUS_HEX8):
----------------------------------------
#elements 226981 #nodes 238328 #dofs 116930169
TTS:		28.9183 [s], compute 26.0561 [s] (solve: 24.0579 [s], init: 1.99194 [s])
residual:	1.28662e-10
```

**GH200 (GPU/Hopper only)**
```c
Multigrid
iter	abs		rel		rate
0	0.0432236	-		-
1	0.00136906	0.031674	0.031674
2	5.06484e-05	0.00117178	0.0369949
3	1.96645e-06	4.54948e-05	0.0388255
4	7.82113e-08	1.80946e-06	0.0397729
5	3.15638e-09	7.30244e-08	0.0403571
6	1.28658e-10	2.97656e-09	0.0407612
----------------------------------------
mgsolve (PROTEUS_HEX8):
----------------------------------------
#elements 226981 #nodes 238328 #dofs 116930169
TTS:		17.8547 [s], compute 3.46191 [s] (solve: 2.55559 [s], init: 0.899413 [s])
residual:	1.28658e-10
```


```c
Multigrid
iter	abs		rel		rate
0	0.0431658	-		-
1	0.00562527	0.130318	0.130318
2	0.000763037	0.0176769	0.135644
3	0.000105342	0.0024404	0.138056
4	1.47101e-05	0.000340782	0.139642
5	2.07e-06	4.79545e-05	0.140719
6	2.9292e-07	6.78593e-06	0.141508
7	4.16311e-08	9.64444e-07	0.142124
8	5.9379e-09	1.3756e-07	0.142631
9	8.49459e-10	1.9679e-08	0.143057
----------------------------------------
mgsolve (PROTEUS_HEX8):
----------------------------------------
#elements 59319 #nodes 64000 #dofs 201230056
TTS:		55.1387 [s], compute 24.7028 [s] (solve: 20.3692 [s], init: 4.32541 [s])
residual:	8.49459e-10
```

```c
Multigrid
iter	abs		rel		rate
0	0.0431432	-		-
1	0.0056922	0.131937	0.131937
2	0.000777642	0.0180247	0.136615
3	0.000107773	0.00249803	0.13859
4	1.50805e-05	0.000349544	0.139928
5	2.12378e-06	4.92264e-05	0.14083
6	3.00477e-07	6.96465e-06	0.141482
7	4.26636e-08	9.88884e-07	0.141986
8	6.07515e-09	1.40814e-07	0.142397
9	8.67185e-10	2.01002e-08	0.142743
----------------------------------------
mgsolve (PROTEUS_HEX8):
----------------------------------------
#elements 205379 #nodes 216000 #dofs 695506456
TTS:		176.452 [s], compute 92.5454 [s] (solve: 84.8508 [s], init: 7.67918 [s])
residual:	8.67185e-10
```


# Linear elasticity in 3D (Joint mesh)

Aggressive coarsening with `SFEM_ELEMENT_REFINE_LEVEL=6`, large deformation

## ARM M1
```c
iter	abs		rel		rate
0	0.0206412	-		-
1	0.00631359	0.305873	0.305873
2	0.00316598	0.153381	0.501454
3	0.00186495	0.0903505	0.589058
4	0.00117958	0.057147	0.632503
5	0.000774669	0.0375301	0.65673
6	0.00052111	0.0252461	0.672688
7	0.000356762	0.0172839	0.684619
8	0.00024768	0.0119993	0.694245
9	0.00017396	0.0084278	0.702358
10	0.000123404	0.00597852	0.709381
11	8.83041e-05	0.00427804	0.715569
12	6.36757e-05	0.00308488	0.721096
13	4.62341e-05	0.00223989	0.726087
14	3.37803e-05	0.00163654	0.730635
15	2.48221e-05	0.00120255	0.734811
16	1.83354e-05	0.000888289	0.738671
17	1.36096e-05	0.00065934	0.742259
18	1.01474e-05	0.000491609	0.745608
19	7.59783e-06	0.00036809	0.748745
20	5.71123e-06	0.00027669	0.751692
21	4.30894e-06	0.000208754	0.754467
22	3.26223e-06	0.000158044	0.757085
23	2.47785e-06	0.000120044	0.759557
24	1.88786e-06	9.14605e-05	0.761893
25	1.44252e-06	6.98853e-05	0.764104
26	1.10525e-06	5.35459e-05	0.766197
27	8.49031e-07	4.11328e-05	0.768178
28	6.538e-07	3.16745e-05	0.770054
29	5.04622e-07	2.44473e-05	0.77183
30	3.90331e-07	1.89103e-05	0.773512
31	3.02547e-07	1.46574e-05	0.775103
32	2.34961e-07	1.13831e-05	0.776609
33	1.82807e-07	8.85641e-06	0.778033
34	1.42476e-07	6.9025e-06	0.779379
35	1.11224e-07	5.38844e-06	0.780651
36	8.69607e-08	4.21296e-06	0.781851
37	6.80887e-08	3.29868e-06	0.782983
38	5.3385e-08	2.58633e-06	0.78405
39	4.19102e-08	2.03041e-06	0.785056
----------------------------------------
mgsolve (PROTEUS_HEX8):
----------------------------------------
#elements 3785 #nodes 5181 #dofs 2599413
TTS:		61.2121 [s], compute 61.125 [s] (solve: 60.9462 [s], init: 0.177892 [s])
residual:	3.72987e-08

```

## P100

**Baseline**

Throughput of fine-op is very bad (`0.22 [GDOF/s]`) with this implementation, with only `2.6x` speed-up over ARM M1 version.

```c
...
39	4.18898e-08	2.02951e-06	0.785051
----------------------------------------
mgsolve (PROTEUS_HEX8):
----------------------------------------
#elements 3785 #nodes 5181 #dofs 2599413
TTS:		23.5485 [s], compute 23.0922 [s] (solve: 22.3401 [s], init: 0.501535 [s])
residual:	3.72803e-08
```
__Operator__

| SFEM_ELEMENT_REFINE_LEVEL | #macro-elems | #dofs | TTS [ms] | TP [GDOF/s] |
|---|---|---|---|---|
| 2 | 473125 |	11761677	| 13.831 | **0.85**  |
| 4 | 473125 |	92463957	| 135.35 | 0.68 |

With new micro-code

| SFEM_ELEMENT_REFINE_LEVEL | #macro-elems | #dofs | TTS [ms] | TP [GDOF/s] |
|---|---|---|---|---|
| 0 | 473125 |	 1521387	| 1.7585| **0.86** |
| 4 | 473125 |	92463957	| 130.61| 0.7  |

**Segmented matrix-based**

Throughput of fine-op is still bad (`0.4 [GDOF/s]`) with this implementation, with only `3.6x` speed-up over ARM M1 version.
```c
39	4.18898e-08	2.02951e-06	0.785051
----------------------------------------
mgsolve (PROTEUS_HEX8):
#elements 3785 #nodes 5181 #dofs 2599413
TTS:		17.4191 [s], compute 16.5649 [s] (solve: 15.7444 [s], init: 0.664405 [s]) 
residual:	3.72803e-08
```

__Operator__
With numerical quadrature and with symbolic integration

| SFEM_ELEMENT_REFINE_LEVEL | #macro-elems | #dofs | TTS [ms] | TP [GDOF/s] |
|---|---|---|---|---|
| 5 | 473125 | 179958447	| 233.74 | 0.77 |
| 6 | 3785 	 |  20206725  	| 33.160 | 0.6  |
| 6 | 102195 |  67538013	| 97.344 | 0.69 |
| 4 | 242240 |  47549853  	| 56.735 | 0.83 |
| 4 | 473125 |  92463957  	| 108.57 | 0.85 |
| 3 | 473125 |  39237117  	| 39.411 | **1** |
| 2 | 473125 |	11761677	| 22.997 | 0.51 |
|---|---|---|---|---|
| 6 | 242240 | 159309357	| 225.9  | 0.7 |
| 5 | 242240 | 92463957	    | 127.62 | 0.72 |
| 5 | 473125 |179958447		| 245.04 | 0.73	|

**Warp level**

| SFEM_ELEMENT_REFINE_LEVEL | #macro-elems | #dofs | TTS [ms] | TP [GDOF/s] |
| 4 | 473125 | 92463957  | 38.223 | 2.41 |
| 5 | 473125 | 179958447 | 78.153 |	2.3	 |

