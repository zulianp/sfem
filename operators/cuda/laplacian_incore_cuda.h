#ifndef LAPLACIAN_INCORE_CUDA_H
#define LAPLACIAN_INCORE_CUDA_H

#include <stddef.h>

#include "boundary_condition.h"
#include "sfem_base.h"

#include "tet4_laplacian_incore_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_incore_laplacian_init(const enum ElemType element_type,
                               cuda_incore_laplacian_t *ctx,
                               const ptrdiff_t nelements,
                               idx_t **const SFEM_RESTRICT elements,
                               geom_t **const SFEM_RESTRICT points);

int cuda_incore_laplacian_destroy(cuda_incore_laplacian_t *ctx);
int cuda_incore_laplacian_apply(cuda_incore_laplacian_t *ctx,
                                const real_t *const d_x,
                                real_t *const d_y);

int cuda_incore_laplacian_diag(cuda_incore_laplacian_t *ctx, real_t *const d_t);

#ifdef __cplusplus
}
#endif
#endif  // LAPLACIAN_INCORE_CUDA_H
