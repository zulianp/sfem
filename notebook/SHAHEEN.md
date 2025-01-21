<!-- SHAHEEN.md -->
# SHAHEEN Cluster

`#microelements 65536000, #micronodes 66049281`

sfem_PoissonTest 

```bash
OMP_NUM_THREADS=192
SemiStructuredLaplacian[16]::apply(affine) called 326 times. Total: 6.03169 [s], Avg: 0.0185021 [s], TP 3569.82 [MDOF/s]
OMP_NUM_THREADS=384 # (Hyper-threading)
SemiStructuredLaplacian[16]::apply(affine) called 326 times. Total: 5.96174 [s], Avg: 0.0182875 [s], TP 3611.71 [MDOF/s]
```
