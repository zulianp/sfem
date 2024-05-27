#ifndef TET4_LINEAR_ELASTICITY_INCORE_CUDA_H
#define TET4_LINEAR_ELASTICITY_INCORE_CUDA_H

#include <stddef.h>

#include "boundary_condition.h"
#include "sfem_base.h"

#include "tet4_linear_elasticity.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    real_t mu;
    real_t lambda;
    enum ElemType element_type;
    ptrdiff_t nelements;
    void *jacobian_determinant;
    void *jacobian_adjugate;
    idx_t *elements;
} cuda_incore_linear_elasticity_t;

int tet4_cuda_incore_linear_elasticity_init(cuda_incore_linear_elasticity_t *const ctx,
                                            const real_t mu,
                                            const real_t lambda,
                                            const ptrdiff_t nelements,
                                            idx_t **const SFEM_RESTRICT elements,
                                            geom_t **const SFEM_RESTRICT points);

int tet4_cuda_incore_linear_elasticity_destroy(cuda_incore_linear_elasticity_t *ctx);
int tet4_cuda_incore_linear_elasticity_apply(const cuda_incore_linear_elasticity_t *const ctx,
                                             const real_t *const SFEM_RESTRICT u,
                                             real_t *const SFEM_RESTRICT values);

int tet4_cuda_incore_linear_elasticity_diag(cuda_incore_linear_elasticity_t *ctx,
                                            real_t *const SFEM_RESTRICT d_t);

#ifdef __cplusplus
}
#endif
#endif  // TET4_LINEAR_ELASTICITY_INCORE_CUDA_H
