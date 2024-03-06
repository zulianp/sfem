#ifndef MACRO_TET4_CUDA_INCORE_LAPLACIAN_H
#define MACRO_TET4_CUDA_INCORE_LAPLACIAN_H

#include "tet4_laplacian_incore_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

int macro_tet4_cuda_incore_laplacian_init(cuda_incore_laplacian_t *ctx,
                                          const ptrdiff_t nelements,
                                          idx_t **const SFEM_RESTRICT elements,
                                          geom_t **const SFEM_RESTRICT points);

int macro_tet4_cuda_incore_laplacian_destroy(cuda_incore_laplacian_t *ctx);
int macro_tet4_cuda_incore_laplacian_apply(cuda_incore_laplacian_t *ctx,
                                           const real_t *const d_x,
                                           real_t *const d_y);

#ifdef __cplusplus
}
#endif
#endif  // MACRO_TET4_CUDA_INCORE_LAPLACIAN_H
