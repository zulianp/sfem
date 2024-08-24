#ifndef MACRO_TET4_LINEAR_ELASTICITY_INCORE_CUDA_H
#define MACRO_TET4_LINEAR_ELASTICITY_INCORE_CUDA_H

#include <stddef.h>
#include "cu_tet4_linear_elasticity.h"
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int cu_macro_tet4_linear_elasticity_init(cuda_incore_linear_elasticity_t *const ctx,
                                         const real_t mu,
                                         const real_t lambda,
                                         const ptrdiff_t nelements,
                                         idx_t **const SFEM_RESTRICT elements,
                                         geom_t **const SFEM_RESTRICT points);

int cu_macro_tet4_linear_elasticity_destroy(cuda_incore_linear_elasticity_t *const ctx);

int cu_macro_tet4_linear_elasticity_apply(const cuda_incore_linear_elasticity_t *const ctx,
                                          const real_t *const SFEM_RESTRICT u,
                                          real_t *const SFEM_RESTRICT values);

int cu_macro_tet4_linear_elasticity_diag(const cuda_incore_linear_elasticity_t *const ctx,
                                         real_t *const SFEM_RESTRICT diag);

#ifdef __cplusplus
}

#endif
#endif  // MACRO_TET4_LINEAR_ELASTICITY_INCORE_CUDA_H
