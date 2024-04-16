#ifndef LINEAR_ELASTICITY_INCORE_CUDA_H
#define LINEAR_ELASTICITY_INCORE_CUDA_H

#include <stddef.h>

#include "boundary_condition.h"
#include "sfem_base.h"

#include "tet4_linear_elasticity_incore_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_incore_linear_elasticity_init(const enum ElemType element_type,
                               cuda_incore_linear_elasticity_t *ctx,
                               const real_t mu,
                               const real_t lambda,
                               const ptrdiff_t nelements,
                               idx_t **const SFEM_RESTRICT elements,
                               geom_t **const SFEM_RESTRICT points);

int cuda_incore_linear_elasticity_destroy(cuda_incore_linear_elasticity_t *ctx);
int cuda_incore_linear_elasticity_apply(cuda_incore_linear_elasticity_t *ctx,
                                const real_t *const d_x,
                                real_t *const d_y);

int cuda_incore_linear_elasticity_diag(cuda_incore_linear_elasticity_t *ctx, real_t *const d_t);

#ifdef __cplusplus
}
#endif
#endif  // LINEAR_ELASTICITY_INCORE_CUDA_H
