#ifndef HEX8_LINEAR_ELASTICITY_H
#define HEX8_LINEAR_ELASTICITY_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

int hex8_linear_elasticity_apply(const ptrdiff_t nelements,
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

int affine_hex8_linear_elasticity_apply(const ptrdiff_t nelements,
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
#ifdef __cplusplus
}
#endif
#endif  // HEX8_LINEAR_ELASTICITY_H
