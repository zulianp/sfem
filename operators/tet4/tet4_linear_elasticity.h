#ifndef TET4_LINEAR_ELASTICITY_H
#define TET4_LINEAR_ELASTICITY_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

int tet4_linear_elasticity_value(const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elements,
                                 geom_t **const SFEM_RESTRICT points,
                                 const real_t mu,
                                 const real_t lambda,
                                 const ptrdiff_t u_stride,
                                 const real_t *const ux,
                                 const real_t *const uy,
                                 const real_t *const uz,
                                 real_t *const SFEM_RESTRICT value);

int tet4_linear_elasticity_apply(const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elements,
                                 geom_t **const SFEM_RESTRICT points,
                                 const real_t mu,
                                 const real_t lambda,
                                 const ptrdiff_t u_stride,
                                 const real_t *const ux,
                                 const real_t *const uy,
                                 const real_t *const uz,
                                 const ptrdiff_t out_stride,
                                 real_t *const outx,
                                 real_t *const outy,
                                 real_t *const outz);

int tet4_linear_elasticity_apply_opt(const ptrdiff_t nelements,
                                 idx_t **const SFEM_RESTRICT elements,
                                 const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate,
                                 const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant,
                                 const real_t mu,
                                 const real_t lambda,
                                 const ptrdiff_t u_stride,
                                 const real_t *const ux,
                                 const real_t *const uy,
                                 const real_t *const uz,
                                 const ptrdiff_t out_stride,
                                 real_t *const outx,
                                 real_t *const outy,
                                 real_t *const outz);

int tet4_linear_elasticity_diag(const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const SFEM_RESTRICT elements,
                                geom_t **const SFEM_RESTRICT points,
                                const real_t mu,
                                const real_t lambda,
                                const ptrdiff_t out_stride,
                                real_t *const outx,
                                real_t *const outy,
                                real_t *const outz);

int tet4_linear_elasticity_diag_opt(const ptrdiff_t nelements,
                                    idx_t **const SFEM_RESTRICT elements,
                                    const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate,
                                    const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant,
                                    const real_t mu,
                                    const real_t lambda,
                                    const ptrdiff_t out_stride,
                                    real_t *const outx,
                                    real_t *const outy,
                                    real_t *const outz);

int tet4_linear_elasticity_hessian(const ptrdiff_t nelements,
                                   const ptrdiff_t nnodes,
                                   idx_t **const SFEM_RESTRICT elements,
                                   geom_t **const SFEM_RESTRICT points,
                                   const real_t mu,
                                   const real_t lambda,
                                   const count_t *const SFEM_RESTRICT rowptr,
                                   const idx_t *const SFEM_RESTRICT colidx,
                                   real_t *const SFEM_RESTRICT values);

#ifdef __cplusplus
}
#endif
#endif  // TET4_LINEAR_ELASTICITY_H
