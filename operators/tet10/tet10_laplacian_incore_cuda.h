#ifndef TET10_CUDA_INCORE_LAPLACIAN_H
#define TET10_CUDA_INCORE_LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.h"

#include "tet4_laplacian_incore_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

int tet10_cuda_incore_laplacian_init(cuda_incore_laplacian_t *ctx,
                                    const ptrdiff_t nelements,
                                    idx_t **const SFEM_RESTRICT elements,
                                    geom_t **const SFEM_RESTRICT points);

int tet10_cuda_incore_laplacian_destroy(cuda_incore_laplacian_t *ctx);

int tet10_cuda_incore_laplacian_apply(cuda_incore_laplacian_t *ctx,
                                     const real_t *const d_x,
                                     real_t *const d_y);

#ifdef __cplusplus
}
#endif
#endif  // TET10_CUDA_INCORE_LAPLACIAN_H
