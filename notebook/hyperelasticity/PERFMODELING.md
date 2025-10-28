# Grace Perf analysis

1558 FLOPs   Micro-Kernel
8 * 3 FLOPs  atomicAdd

45 * 4 READS
8 * 3 READS
8 * 3 WRITE (atomicAdd)

AI = (1558 + 8 * 3) / ((45) * 4 + (4 + 8 * 3 + 8 * 3) * 8) = 2.654 FLOPS/Byte (memory bound)
Grace Ridge Point: 6.93 FLOPS/Byte


## Single core

NOMINAL  peak: 55 		GFLOP/s
MEASURED peak: 51.2318  GFLOP/s

1e-9 * (2197000 * 1558 + 8 * 3) / ( 2.788e-01 )   = 12.2773530273 GLOP/s
Percentage of Peak: 12.2773530273/51.2318  * 100% = 23.9%

## Multicore

NOMINAL peak: 3960 GFLOP/s
MEASURE peak: 3688 GFLOP/s

Achievable peak: 2.654 * 512 = 1358.8 GFLOP/s

OpenMP (performance degradation)
1e-9 * (2197000 * 1558 + 8 * 3) / ( 7.188e-03 ) = 476.2 GLOPS/s

Percentage of nominal peak:    476.2 / 3960 * 100% 			 = 12.0%
Percentage of measured peak:   476.2 / (72 * 51.2318) * 100% = 12.9%
Percentage of achievable peak: 476.2 / 1358.8 * 100%         = 35.04%