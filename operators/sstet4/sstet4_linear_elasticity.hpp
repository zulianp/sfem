#ifndef SSTET4_LINEAR_ELASTICITY_H
#define SSTET4_LINEAR_ELASTICITY_H

#include <stddef.h>
#include "sfem_base.hpp"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sstet4_linear_elasticity_stencil sstet4_linear_elasticity_stencil_t;

int sstet4_linear_elasticity_apply_points(const int                         level,
                                          const ptrdiff_t                   nelements,
                                          idx_t **const SFEM_RESTRICT       elements,
                                          geom_t **const SFEM_RESTRICT      points,
                                          const real_t                      mu,
                                          const real_t                      lambda,
                                          const real_t *const SFEM_RESTRICT u,
                                          real_t *const SFEM_RESTRICT       values);

int sstet4_linear_elasticity_stencil_create_from_points(const int                                  level,
                                                        const ptrdiff_t                            nelements,
                                                        idx_t **const SFEM_RESTRICT                elements,
                                                        geom_t **const SFEM_RESTRICT               points,
                                                        const real_t                               mu,
                                                        const real_t                               lambda,
                                                        sstet4_linear_elasticity_stencil_t **const stencil);

void sstet4_linear_elasticity_stencil_destroy(sstet4_linear_elasticity_stencil_t *stencil);

int sstet4_linear_elasticity_stencil_nrows(const sstet4_linear_elasticity_stencil_t *const stencil);
int sstet4_linear_elasticity_stencil_max_row_len(const sstet4_linear_elasticity_stencil_t *const stencil);
int sstet4_linear_elasticity_stencil_n_unique_stencils(const sstet4_linear_elasticity_stencil_t *const stencil);

int sstet4_linear_elasticity_apply_stencil(const sstet4_linear_elasticity_stencil_t *const stencil,
                                           const ptrdiff_t                                  nelements,
                                           const real_t *const SFEM_RESTRICT                u,
                                           real_t *const SFEM_RESTRICT                      values);

int sstet4_linear_elasticity_apply_stencil_global_vectorized(
        const sstet4_linear_elasticity_stencil_t *const stencil,
        const ptrdiff_t                                  nelements,
        idx_t **const SFEM_RESTRICT                      elements,
        const real_t *const SFEM_RESTRICT                u,
        real_t *const SFEM_RESTRICT                      values);

int sstet4_linear_elasticity_block_diag_sym_stencil(
        const sstet4_linear_elasticity_stencil_t *const stencil,
        const ptrdiff_t                                  nelements,
        idx_t **const SFEM_RESTRICT                      elements,
        const ptrdiff_t                                  out_stride,
        real_t *const SFEM_RESTRICT                      out0,
        real_t *const SFEM_RESTRICT                      out1,
        real_t *const SFEM_RESTRICT                      out2,
        real_t *const SFEM_RESTRICT                      out3,
        real_t *const SFEM_RESTRICT                      out4,
        real_t *const SFEM_RESTRICT                      out5);

int sstet4_linear_elasticity_diag_stencil(const sstet4_linear_elasticity_stencil_t *const stencil,
                                          const ptrdiff_t                                  nelements,
                                          idx_t **const SFEM_RESTRICT                      elements,
                                          const ptrdiff_t                                  out_stride,
                                          real_t *const SFEM_RESTRICT                      outx,
                                          real_t *const SFEM_RESTRICT                      outy,
                                          real_t *const SFEM_RESTRICT                      outz);

#ifdef __cplusplus
}
#endif

#endif  // SSTET4_LINEAR_ELASTICITY_H
