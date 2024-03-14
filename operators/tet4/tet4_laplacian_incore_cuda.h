#ifndef TET4_CUDA_INCORE_LAPLACIAN_H
#define TET4_CUDA_INCORE_LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    ptrdiff_t nelements;
    jacobian_t *d_fff;
    idx_t *d_elems;
} cuda_incore_laplacian_t;

int tet4_cuda_incore_laplacian_init(cuda_incore_laplacian_t *ctx,
                                    const ptrdiff_t nelements,
                                    idx_t **const SFEM_RESTRICT elements,
                                    geom_t **const SFEM_RESTRICT points);

int tet4_cuda_incore_laplacian_destroy(cuda_incore_laplacian_t *ctx);

int tet4_cuda_incore_laplacian_apply(cuda_incore_laplacian_t *ctx,
                                     const real_t *const d_x,
                                     real_t *const d_y);

#ifdef __cplusplus
}
#endif
#endif  // TET4_CUDA_INCORE_LAPLACIAN_H
