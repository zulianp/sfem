# Grace Perf analysis HEX8

#elements 2197000
#nodes    2248091
#dofs 	  6744273

1558  FLOPs  Micro-Kernel
8 * 3 FLOPs  atomicAdd

45 * 4 READS (float)
8  * 3 READS (double)
8  * 3 WRITE (double, atomicAdd)
8      READS (int16)


AI = `(1558 + 8 * 3) / ((45) * 4 + (8 * 3 + 8 * 3) * 8 + (8) * 2) = 2.72` FLOPS/Byte (memory bound)
Grace Ridge Point: 6.93 FLOPS/Byte


## Single core

NOMINAL  peak: `55 		GFLOP/s`
MEASURED peak: `51.2318 GFLOP/s`

`1e-9 * (2197000 * 1558 + 8 * 3) / ( 2.788e-01 )   = 12.2773530273 GLOP/s`
Percentage of Peak: `12.2773530273/51.2318  * 100% = 23.9%`

## Multicore (72 OpenMP threads)

NOMINAL peak: `3960 GFLOP/s`
MEASURE peak: `3688 GFLOP/s`

Achievable peak: `2.72 * 512 = 1392.64 GFLOP/s`

OpenMP (performance degradation)
`1e-9 * (2197000 * 1558 + 8 * 3) / ( 7.188e-03 ) = 476.2 GLOPS/s`

Percentage of nominal peak:    `476.2 / 3960 * 100% 		  = 12.0%`
Percentage of measured peak:   `476.2 / (72 * 51.2318) * 100% = 12.9%`
Percentage of achievable peak: `476.2 / 1392.64 * 100%        = 34.19%`


# Larger

```c++
OMP_NUM_THREADS=72 OMP_PROC_BIND=true SFEM_USE_SFC=1 SFEM_REPEAT=20 SFEM_BASE_RESOLUTION=400 SFEM_NEOHOOKEAN_OGDEN_USE_AOS=1 SFEM_USE_PARTIAL_ASSEMBLY=1   SFEM_ELEMENTS_PER_PACK=4096  ./bench_op
element_type HEX8
--------------------
| Packed |
n_packs: 15625
elements_per_pack: 4096
Memory Packed: 905490 KB
Original:      1.701e+06 KB
--------------------
[Warning]Skipping CRS ops for large meshes #nodes 64481201 #dim 3
[Warning] Skipping BSR ops for large meshes #nodes 64481201 #dim 3
#elements 64000000
#nodes 64481201
Operation                 Type           Time [s]    Rate [MDOF/s]    BW [MDOF/s]    Setup [s]    Dimensions
-----------------------   ------------   ---------   -------------    -----------    ---------    ------------
Laplacian                 MF             5.205e-02        1238.849       2477.698    4.224e-06    (64481201, 64481201)
PackedLaplacian           MF             3.159e-02        2040.972       4081.944    3.104e-06    (64481201, 64481201)
Mass                      MF             3.140e-01         205.342        410.685    2.976e-06    (64481201, 64481201)
LinearElasticity          MF             4.056e-01         476.929        953.858    7.263e-06    (193443603, 193443603)
NeoHookeanOgden           MF             3.742e-01         516.998       1033.996    1.070e+00    (193443603, 193443603)
NeoHookeanOgdenPacked     MF             2.030e-01         952.984       1905.968    1.068e+00    (193443603, 193443603)
```


`1e-9 * (64000000 * 1558 + 8 * 3) / (   1.859e-01   ) = 536.37 	GLOPS/s`
Percentage of achievable peak: `536.37 / 1392.64 * 100%         = 38.5%`

<!-- cmake .. -DCMAKE_CXX_FLAGS="-fopenmp-simd -ffp-contract=fast" -->


# TET10

2060
