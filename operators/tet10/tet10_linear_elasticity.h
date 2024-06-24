#ifndef TET10_LINEAR_ELASTICITY_H
#define TET10_LINEAR_ELASTICITY_H

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"
#include "tet4_linear_elasticity.h"

#ifdef __cplusplus
extern "C" {
#endif

int tet10_linear_elasticity_apply(const ptrdiff_t nelements,
                                  const ptrdiff_t nnodes,
                                  idx_t **const SFEM_RESTRICT elements,
                                  geom_t **const SFEM_RESTRICT points,
                                  const real_t mu,
                                  const real_t lambda,
                                  const ptrdiff_t u_stride,
                                  const real_t *const SFEM_RESTRICT ux,
                                  const real_t *const SFEM_RESTRICT uy,
                                  const real_t *const SFEM_RESTRICT uz,
                                  const ptrdiff_t out_stride,
                                  real_t *const SFEM_RESTRICT outx,
                                  real_t *const SFEM_RESTRICT outy,
                                  real_t *const SFEM_RESTRICT outz);

int tet10_linear_elasticity_diag(const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elements,
                                 geom_t **const SFEM_RESTRICT points,
                                 const real_t mu,
                                 const real_t lambda,
                                 const ptrdiff_t out_stride,
                                 real_t *const SFEM_RESTRICT outx,
                                 real_t *const SFEM_RESTRICT outy,
                                 real_t *const SFEM_RESTRICT outz);

int tet10_linear_elasticity_apply_opt(const ptrdiff_t nelements,
                                      idx_t **const SFEM_RESTRICT elements,
                                      const jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
                                      const jacobian_t *const SFEM_RESTRICT jacobian_determinant,
                                      const real_t mu,
                                      const real_t lambda,
                                      const ptrdiff_t u_stride,
                                      const real_t *const SFEM_RESTRICT ux,
                                      const real_t *const SFEM_RESTRICT uy,
                                      const real_t *const SFEM_RESTRICT uz,
                                      const ptrdiff_t out_stride,
                                      real_t *const SFEM_RESTRICT outx,
                                      real_t *const SFEM_RESTRICT outy,
                                      real_t *const SFEM_RESTRICT outz);

int tet10_linear_elasticity_diag_opt(const ptrdiff_t nelements,
                                     idx_t **const SFEM_RESTRICT elements,
                                     const jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
                                     const jacobian_t *const SFEM_RESTRICT jacobian_determinant,
                                     const real_t mu,
                                     const real_t lambda,
                                     const ptrdiff_t out_stride,
                                     real_t *const SFEM_RESTRICT outx,
                                     real_t *const SFEM_RESTRICT outy,
                                     real_t *const SFEM_RESTRICT outz);

int tet10_linear_elasticity_hessian(const ptrdiff_t nelements,
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
#endif  // TET10_LINEAR_ELASTICITY_H
