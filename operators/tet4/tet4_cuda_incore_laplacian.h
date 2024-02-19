#ifndef TET4_CUDA_INCORE_LAPLACIAN_H
#define TET4_CUDA_INCORE_LAPLACIAN_H

typedef struct {
    geom_t* d_fff;
    idx_t* d_elems;
} cuda_incore_laplacian_t;

#endif  // TET4_CUDA_INCORE_LAPLACIAN_H
