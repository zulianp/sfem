<!-- README.md -->


# M1 Max Pro 8 threads

## Small example 6M nodes

```c++
3 x 3 spheres 
#microelements 6553600, #micronodes 6697665
n_dofs: 20092995

TTS: 217.314 [s] (i.e., 3 [m] and 36 [s])
10|33|990) [lagr++ 10] norm_pen 7.052373e-12, norm_rpen 5.608164e-13

LinearElasticity::apply called 279 times. Total: 0.012408 [s], Avg: 4.44731e-05 [s], TP 4.85687 [MDOF/s]
SemiStructuredLinearElasticity[2]::apply(affine) called 693 times. Total: 0.058612 [s], Avg: 8.45772e-05 [s], TP 12.8758 [MDOF/s]
SemiStructuredLinearElasticity[4]::apply(affine) called 693 times. Total: 0.088383 [s], Avg: 0.000127537 [s], TP 51.8674 [MDOF/s]
SemiStructuredLinearElasticity[8]::apply(affine) called 693 times. Total: 0.240119 [s], Avg: 0.000346492 [s], TP 130.99 [MDOF/s]
SemiStructuredLinearElasticity[16]::apply(affine) called 693 times. Total: 1.6883 [s], Avg: 0.00243622 [s], TP 137.348 [MDOF/s]
SemiStructuredLinearElasticity[32]::apply(affine) called 693 times. Total: 12.3578 [s], Avg: 0.0178323 [s], TP 143.906 [MDOF/s]
SemiStructuredLinearElasticity[64]::apply(affine) called 1298 times. Total: 178.365 [s], Avg: 0.137415 [s], TP 146.221 [MDOF/s]
```

```csv
count_iter,count_mg_cycles,count_nl_smooth,count_smooth,norm_penetration,norm_residual,energy_norm_correction,penalty_param,omega,rate
1,2,6,60,7.80912,0.00016983,3.09875,10,0.1,1
2,4,12,120,1.84516,0.000224483,0.0958868,100,0.01,0.0309438
3,6,18,180,0.0266599,4.89927e-05,0.0307723,1000,0.001,0.320923
4,9,27,270,0,5.11038e-07,0.00279591,1000,1e-06,0.090858
5,15,45,450,0.000149453,6.25471e-10,0.000268467,1000,1e-09,0.0960214
6,17,51,510,2.06026e-05,2.78835e-08,8.13816e-05,10000,0.0001,0.303134
7,19,57,570,1.82375e-06,4.67829e-09,4.33828e-06,10000,1e-08,0.0533078
8,29,87,870,1.42201e-07,7.32334e-12,5.16329e-07,10000,1e-11,0.119017
9,31,93,930,7.91497e-10,3.56753e-12,1.331e-08,100000,1e-05,0.0257782
10,33,99,990,7.05237e-12,5.60816e-13,2.26525e-10,100000,1e-10,0.0170191
```

## Medium size example 53M nodes

```c++
3 x 3 spheres 
#microelements 52428800, #micronodes 53003649
n_dofs: 159010947

TBA
11|36|1080) [lagr++ 11] norm_pen 7.218486e-12, norm_rpen 9.550587e-13, penetration_tol 1.000000e-14, penalty_param 1.000000e+05


LinearElasticity::apply called 306 times. Total: 0.016008 [s], Avg: 5.23137e-05 [s], TP 4.12894 [MDOF/s]
SemiStructuredLinearElasticity[2]::apply(affine) called 756 times. Total: 0.063469 [s], Avg: 8.39537e-05 [s], TP 12.9714 [MDOF/s]
SemiStructuredLinearElasticity[4]::apply(affine) called 756 times. Total: 0.092006 [s], Avg: 0.000121701 [s], TP 54.3545 [MDOF/s]
SemiStructuredLinearElasticity[8]::apply(affine) called 756 times. Total: 0.263747 [s], Avg: 0.000348872 [s], TP 130.097 [MDOF/s]
SemiStructuredLinearElasticity[16]::apply(affine) called 756 times. Total: 1.9344 [s], Avg: 0.00255873 [s], TP 130.772 [MDOF/s]
SemiStructuredLinearElasticity[32]::apply(affine) called 756 times. Total: 122.984 [s], Avg: 0.162678 [s], TP 15.7746 [MDOF/s]
SemiStructuredLinearElasticity[64]::apply(affine) called 756 times. Total: 160.593 [s], Avg: 0.212424 [s], TP 94.5889 [MDOF/s]
SemiStructuredLinearElasticity[128]::apply(affine) called 1416 times. Total: 3510.72 [s], Avg: 2.47932 [s], TP 64.1349 [MDOF/s]

```

```csv
count_iter,count_mg_cycles,count_nl_smooth,count_smooth,norm_penetration,norm_residual,energy_norm_correction,penalty_param,omega,rate
1,2,6,60,15.8636,0.000123298,4.38407,10,0.1,1
2,4,12,120,3.90934,0.0001353,0.0963879,100,0.01,0.0219859
3,6,18,180,0.0822152,9.17072e-05,0.0330794,1000,0.001,0.343191
4,9,27,270,0,8.99437e-07,0.00491116,1000,1e-06,0.148466
5,16,48,480,2.27162e-05,5.8364e-10,0.000577702,1000,1e-09,0.11763
6,18,54,540,5.22362e-05,6.36316e-08,0.000115943,10000,0.0001,0.200697
7,20,60,600,9.52962e-06,6.18319e-09,9.82044e-06,10000,1e-08,0.0847006
8,30,90,900,1.17158e-06,9.28873e-12,1.30131e-06,10000,1e-11,0.13251
9,32,96,960,1.36286e-08,1.04206e-11,8.67241e-08,100000,1e-05,0.066644
10,34,102,1020,2.14521e-10,2.21501e-12,1.69421e-09,100000,1e-10,0.0195356
11,36,108,1080,7.21849e-12,9.55059e-13,2.60857e-10,100000,1e-11,0.15397
```
