# Performance of operators on Apple M1 Max 

## SDACRS with SFC

SDACRS vs CRS reduces memory footprint by a factor of 2.3 (T=half), 1.6 (T=float), and 1.17 (T=double)

### Half precision (no overflow protection)

CRS KB: 558239
SDACRS KB: 236657
DIAG NNZ: 54332672
OFFDIAG NNZ: 1805658
CRS NNZ: 47045881
SDACRS/CRS: 0.423935


### Single precision

CRS KB: 558239
SDACRS KB: 346302
DIAG NNZ: 54332672
OFFDIAG NNZ: 1805658
CRS NNZ: 47045881
SDACRS/CRS: 0.620348

## SDACRS with Cuthill McKee

### Half precision (no overflow protection)

CRS KB: 121915
SDACRS KB: 52800
DIAG NNZ: 12130032
OFFDIAG NNZ: 0
CRS NNZ: 10172313
SDACRS/CRS: 0.433088

### Single precision

CRS KB: 121915
SDACRS KB: 76491
DIAG NNZ: 12130032
OFFDIAG NNZ: 0
CRS NNZ: 10172313
SDACRS/CRS: 0.627415


## Packed mesh format reduces the memory footprint by a factor 1.6x

Example: 

n_packs: 422
elements_per_pack: 4096
Memory Packed: 27875.9 KB
Original:      45926.9 KB


## Performance on HEX8 mesh with SFC

#elements 1728000
#nodes    1771561

### Half precision matrices

Operation          Type           Time [s]    Rate [MDOF/s]    BW [MDOF/s]    Setup [s]    Dimensions
----------------   ------------   ---------   -------------    -----------    ---------    ------------
Laplacian          MF             1.366e-02         129.682        259.364    2.000e-06    (1771561, 1771561)
LinearElasticity   MF             9.523e-02          55.807        111.613    3.000e-06    (5314683, 5314683)
LinearElasticity   BSR            3.056e-02         173.890        347.781    9.434e-01    (5314683, 5314683)
Laplacian          CRS            4.126e-03         429.365        858.730    7.839e-02    (1771561, 1771561)
Laplacian          SPLITCRS       3.156e-03         561.367       1122.733    1.539e-01    (1771561, 1771561)
Laplacian          ALIGNEDCRS     1.187e-02         149.290        298.579    1.365e-01    (1771561, 1771561)
Laplacian          SPLITDACRS     3.505e-03         505.496       1010.992    1.302e-01    (1771561, 1771561)
PackedLaplacian    MF             1.262e-02         140.337        280.674    2.000e-06    (1771561, 1771561)
Mass               MF             6.273e-02          28.240         56.481    1.000e-06    (1771561, 1771561)
LinearElasticity   BSR_SYM        3.234e-02         164.353        328.706    2.980e+00    (5314683, 5314683)
NeoHookeanOgden    MF             4.218e-02         125.987        251.974    3.922e-02    (5314683, 5314683)

### Single precision matrices

Operation          Type           Time [s]    Rate [MDOF/s]    BW [MDOF/s]    Setup [s]    Dimensions
----------------   ------------   ---------   -------------    -----------    ---------    ------------
Laplacian          MF             1.399e-02         126.609        253.218    1.000e-06    (1771561, 1771561)
LinearElasticity   MF             1.008e-01          52.745        105.490    4.000e-06    (5314683, 5314683)
LinearElasticity   BSR            4.127e-02         128.788        257.577    7.530e-01    (5314683, 5314683)
Laplacian          CRS            6.303e-03         281.048        562.097    6.924e-02    (1771561, 1771561)
Laplacian          SPLITCRS       5.637e-03         314.296        628.592    1.691e-01    (1771561, 1771561)
Laplacian          ALIGNEDCRS     1.205e-02         147.030        294.059    1.500e-01    (1771561, 1771561)
Laplacian          SPLITDACRS     4.051e-03         437.358        874.715    1.747e-01    (1771561, 1771561)
PackedLaplacian    MF             1.323e-02         133.947        267.895    9.000e-06    (1771561, 1771561)
Mass               MF             7.177e-02          24.685         49.370    2.000e-06    (1771561, 1771561)
LinearElasticity   BSR_SYM        3.418e-02         155.494        310.987    3.244e+00    (5314683, 5314683)
NeoHookeanOgden    MF             4.403e-02         120.706        241.412    9.413e-02    (5314683, 5314683)


## Performance on TET4 mesh with Cuthill McKee


#elements 3989504
#nodes 	   693385

### Half precision matrices

Operation          Type           Time [s]    Rate [MDOF/s]    BW [MDOF/s]    Setup [s]    Dimensions
----------------   ------------   ---------   -------------    -----------    ---------    ------------
Laplacian          MF             6.187e-03         112.079        224.157    1.700e-05    (693385, 693385)
LinearElasticity   MF             1.957e-02         106.308        212.617    3.000e-06    (2080155, 2080155)
LinearElasticity   BSR            8.010e-03         259.701        519.402    3.075e-01    (2080155, 2080155)
Laplacian          CRS            1.502e-03         461.580        923.159    5.524e-02    (693385, 693385)
Laplacian          SPLITCRS       1.267e-03         547.438       1094.876    8.137e-02    (693385, 693385)
Laplacian          ALIGNEDCRS     3.290e-03         210.730        421.459    8.555e-02    (693385, 693385)
Laplacian          SPLITDACRS     1.043e-03         664.926       1329.852    7.385e-02    (693385, 693385)
PackedLaplacian    MF             5.829e-03         118.946        237.892    3.000e-06    (693385, 693385)
NeoHookeanOgden    MF             1.863e-02         111.645        223.291    7.725e-02    (2080155, 2080155)


### Single precision matrices

Operation          Type           Time [s]    Rate [MDOF/s]    BW [MDOF/s]    Setup [s]    Dimensions
----------------   ------------   ---------   -------------    -----------    ---------    ------------
Laplacian          MF             5.761e-03         120.354        240.709    8.000e-06    (693385, 693385)
LinearElasticity   MF             2.032e-02         102.381        204.762    1.000e-06    (2080155, 2080155)
LinearElasticity   BSR            1.004e-02         207.232        414.464    2.482e-01    (2080155, 2080155)
Laplacian          CRS            1.826e-03         379.771        759.541    5.486e-02    (693385, 693385)
Laplacian          SPLITCRS       1.230e-03         563.636       1127.272    8.366e-02    (693385, 693385)
Laplacian          ALIGNEDCRS     3.268e-03         212.200        424.400    8.223e-02    (693385, 693385)
Laplacian          SPLITDACRS     1.047e-03         662.259       1324.518    7.426e-02    (693385, 693385)
PackedLaplacian    MF             5.756e-03         120.459        240.918    5.000e-06    (693385, 693385)
NeoHookeanOgden    MF             1.773e-02         117.304        234.608    7.351e-02    (2080155, 2080155)


## Peformance on Semistructured HEX8 meshes (L=8)

#microelements 4096000 
#micronodes    4173281

Operation          Type           Time [s]    Rate [MDOF/s]    BW [MDOF/s]    Setup [s]    Dimensions
----------------   ------------   ---------   -------------    -----------    ---------    ------------
Laplacian          MF             5.988e-03         696.917       1393.835    1.000e-06    (4173281, 4173281)
LinearElasticity   MF             3.474e-02         360.356        720.712    1.000e-06    (12519843, 12519843)
LinearElasticity   BSR            9.854e-02         127.055        254.110    1.381e+00    (12519843, 12519843)
em:Laplacian       MF             5.783e-03         721.596       1443.193    3.000e-06    (4173281, 4173281)
NeoHookeanOgden    MF             3.333e-02         375.662        751.324    5.025e-01    (12519843, 12519843)
