#ifndef SSTET4_LAPLACIAN_H
#define SSTET4_LAPLACIAN_H

// Adapted from Bole Ma implementation in https://github.com/zulianp/hpcfem

#include <stddef.h>
#include "sfem_base.hpp"

#ifdef __cplusplus
extern "C" {
#endif

int sstet4_nxe(int level);
int sstet4_txe(int level);

int sstet4_laplacian_apply(const int                             level,
                           const ptrdiff_t                       nelements,
                           const jacobian_t *const SFEM_RESTRICT fff,
                           const real_t *const SFEM_RESTRICT     u,
                           real_t *const SFEM_RESTRICT           values);

int sstet4_laplacian_apply_elementwise(const int                             level,
                                       const ptrdiff_t                       nelements,
                                       const jacobian_t *const SFEM_RESTRICT fff,
                                       const real_t *const SFEM_RESTRICT     u,
                                       real_t *const SFEM_RESTRICT           values);

typedef struct sstet4_laplacian_stencil sstet4_laplacian_stencil_t;

int sstet4_laplacian_stencil_create(const int                             level,
                                    const ptrdiff_t                       nelements,
                                    const jacobian_t *const SFEM_RESTRICT fff,
                                    sstet4_laplacian_stencil_t **const    stencil);

int sstet4_laplacian_stencil_create_from_points(const int                         level,
                                                const ptrdiff_t                   nelements,
                                                idx_t **const SFEM_RESTRICT       elements,
                                                geom_t **const SFEM_RESTRICT      points,
                                                sstet4_laplacian_stencil_t **const stencil);

void sstet4_laplacian_stencil_destroy(sstet4_laplacian_stencil_t *stencil);

int sstet4_laplacian_stencil_nrows(const sstet4_laplacian_stencil_t *const stencil);
int sstet4_laplacian_stencil_max_row_len(const sstet4_laplacian_stencil_t *const stencil);
int sstet4_laplacian_stencil_max_slot_terms(const sstet4_laplacian_stencil_t *const stencil);
int sstet4_laplacian_stencil_n_unique_stencils(const sstet4_laplacian_stencil_t *const stencil);

int sstet4_laplacian_apply_stencil(const sstet4_laplacian_stencil_t *const stencil,
                                   const ptrdiff_t                         nelements,
                                   const real_t *const SFEM_RESTRICT       u,
                                   real_t *const SFEM_RESTRICT             values);

int sstet4_laplacian_apply_stencil_global(const sstet4_laplacian_stencil_t *const stencil,
                                          const ptrdiff_t                         nelements,
                                          idx_t **const SFEM_RESTRICT             elements,
                                          const real_t *const SFEM_RESTRICT       u,
                                          real_t *const SFEM_RESTRICT             values);

int sstet4_laplacian_apply_stencil_global_range(const sstet4_laplacian_stencil_t *const stencil,
                                                const ptrdiff_t                         element_offset,
                                                const ptrdiff_t                         nelements,
                                                idx_t **const SFEM_RESTRICT             elements,
                                                const real_t *const SFEM_RESTRICT       u,
                                                real_t *const SFEM_RESTRICT             values);

int sstet4_laplacian_apply_points(const int                         level,
                                  const ptrdiff_t                   nelements,
                                  idx_t **const SFEM_RESTRICT       elements,
                                  geom_t **const SFEM_RESTRICT      points,
                                  const real_t *const SFEM_RESTRICT u,
                                  real_t *const SFEM_RESTRICT       values);

int sstet4_laplacian_diag_points(const int                    level,
                                 const ptrdiff_t              nelements,
                                 idx_t **const SFEM_RESTRICT  elements,
                                 geom_t **const SFEM_RESTRICT points,
                                 real_t *const SFEM_RESTRICT  values);

#ifdef __cplusplus
}
#endif
#endif  // SSTET4_LAPLACIAN_H
