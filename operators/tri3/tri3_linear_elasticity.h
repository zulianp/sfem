#ifndef TRI3_LINEAR_ELASTICITY_H
#define TRI3_LINEAR_ELASTICITY_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

int tri3_linear_elasticity_value(const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elements,
                                 geom_t **const SFEM_RESTRICT points,
                                 const real_t mu,
                                 const real_t lambda,
                                 const ptrdiff_t u_stride,
                                 const real_t *const ux,
                                 const real_t *const uy,
                                 real_t *const SFEM_RESTRICT value);

int tri3_linear_elasticity_apply(const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elements,
                                 geom_t **const SFEM_RESTRICT points,
                                 const real_t mu,
                                 const real_t lambda,
                                 const ptrdiff_t u_stride,
                                 const real_t *const ux,
                                 const real_t *const uy,
                                 const ptrdiff_t out_stride,
                                 real_t *const outx,
                                 real_t *const outy);

int tri3_linear_elasticity_diag(const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const SFEM_RESTRICT elements,
                                geom_t **const SFEM_RESTRICT points,
                                const real_t mu,
                                const real_t lambda,
                                const ptrdiff_t out_stride,
                                real_t *const outx,
                                real_t *const outy);

int tri3_linear_elasticity_hessian_aos(const ptrdiff_t nelements,
                                   const ptrdiff_t nnodes,
                                   idx_t **const SFEM_RESTRICT elements,
                                   geom_t **const SFEM_RESTRICT points,
                                   const real_t mu,
                                   const real_t lambda,
                                   const count_t *const SFEM_RESTRICT rowptr,
                                   const idx_t *const SFEM_RESTRICT colidx,
                                   real_t *const SFEM_RESTRICT values);

int tri3_linear_elasticity_assemble_hessian_soa(const ptrdiff_t nelements,
                                                 const ptrdiff_t nnodes,
                                                 idx_t **const SFEM_RESTRICT elements,
                                                 geom_t **const SFEM_RESTRICT points,
                                                 const real_t mu,
                                                 const real_t lambda,
                                                 const count_t *const SFEM_RESTRICT rowptr,
                                                 const idx_t *const SFEM_RESTRICT colidx,
                                                 real_t **const SFEM_RESTRICT values);

#ifdef __cplusplus
}
#endif
#endif  // TRI3_LINEAR_ELASTICITY_H
