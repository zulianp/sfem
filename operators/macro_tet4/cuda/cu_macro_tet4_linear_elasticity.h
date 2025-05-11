#ifndef MACRO_TET4_LINEAR_ELASTICITY_INCORE_CUDA_H
#define MACRO_TET4_LINEAR_ELASTICITY_INCORE_CUDA_H

#include <stddef.h>
#include "cu_tet4_linear_elasticity.h"
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int cu_macro_tet4_linear_elasticity_apply(const ptrdiff_t                 nelements,
                                          idx_t **const SFEM_RESTRICT     elements,
                                          const ptrdiff_t                 jacobian_stride,
                                          const void *const SFEM_RESTRICT jacobian_adjugate,
                                          const void *const SFEM_RESTRICT jacobian_determinant,
                                          const real_t                    mu,
                                          const real_t                    lambda,
                                          const enum RealType             real_type,
                                          const ptrdiff_t                 u_stride,
                                          const void *const SFEM_RESTRICT ux,
                                          const void *const SFEM_RESTRICT uy,
                                          const void *const SFEM_RESTRICT uz,
                                          const ptrdiff_t                 out_stride,
                                          void *const SFEM_RESTRICT       outx,
                                          void *const SFEM_RESTRICT       outy,
                                          void *const SFEM_RESTRICT       outz,
                                          void                           *stream);

int cu_macro_tet4_linear_elasticity_diag(const ptrdiff_t                 nelements,
                                         idx_t **const SFEM_RESTRICT     elements,
                                         const ptrdiff_t                 jacobian_stride,
                                         const void *const SFEM_RESTRICT jacobian_adjugate,
                                         const void *const SFEM_RESTRICT jacobian_determinant,
                                         const real_t                    mu,
                                         const real_t                    lambda,
                                         const enum RealType             real_type,
                                         const ptrdiff_t                 diag_stride,
                                         void *const SFEM_RESTRICT       diagx,
                                         void *const SFEM_RESTRICT       diagy,
                                         void *const SFEM_RESTRICT       diagz,
                                         void                           *stream);

#ifdef __cplusplus
}

#endif
#endif  // MACRO_TET4_LINEAR_ELASTICITY_INCORE_CUDA_H
