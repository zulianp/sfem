#ifndef LINEAR_ELASTICITY_INCORE_CUDA_H
#define LINEAR_ELASTICITY_INCORE_CUDA_H

#include <stddef.h>

#include "boundary_condition.h"
#include "sfem_base.h"

#include "cu_tet4_linear_elasticity.h"

#ifdef __cplusplus
extern "C" {
#endif

int cu_linear_elasticity_apply(const enum ElemType element_type,
                               const ptrdiff_t nelements,
                               const ptrdiff_t stride,  // Stride for elements and fff
                               const idx_t *const SFEM_RESTRICT elements,
                               const void *const SFEM_RESTRICT jacobian_adjugate,
                               const void *const SFEM_RESTRICT jacobian_determinant,
                               const real_t mu,
                               const real_t lambda,
                               const enum RealType real_type,
                               const real_t *const d_x,
                               real_t *const d_y,
                               void *stream);

int cu_linear_elasticity_diag(const enum ElemType element_type,
                              const ptrdiff_t nelements,
                              const ptrdiff_t stride,  // Stride for elements and fff
                              const idx_t *const SFEM_RESTRICT elements,
                              const void *const SFEM_RESTRICT jacobian_adjugate,
                              const void *const SFEM_RESTRICT jacobian_determinant,
                              const real_t mu,
                              const real_t lambda,
                              const enum RealType real_type,
                              real_t *const d_t,
                              void *stream);

// Block sparse row (BSR) https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-storage-formats
int cu_linear_elasticity_bsr(const enum ElemType element_type,
                             const ptrdiff_t nelements,
                             const ptrdiff_t stride,
                             const idx_t *const SFEM_RESTRICT elements,
                             const void *const SFEM_RESTRICT jacobian_adjugate,
                             const void *const SFEM_RESTRICT jacobian_determinant,
                             const real_t mu,
                             const real_t lambda,
                             const enum RealType real_type,
                             const count_t *const SFEM_RESTRICT rowptr,
                             const idx_t *const SFEM_RESTRICT colidx,
                             void *const SFEM_RESTRICT values,
                             void *stream);

#ifdef __cplusplus
}
#endif
#endif  // LINEAR_ELASTICITY_INCORE_CUDA_H
