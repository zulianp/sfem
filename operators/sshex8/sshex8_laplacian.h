#ifndef SFEM_SSHEX8_LAPLACIAN_H
#define SFEM_SSHEX8_LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

int sshex8_laplacian_apply(const int                         level,
                           const ptrdiff_t                   nelements,
                           ptrdiff_t                         interior_start,
                           idx_t **const SFEM_RESTRICT       elements,
                           geom_t **const SFEM_RESTRICT      points,
                           const real_t *const SFEM_RESTRICT u,
                           real_t *const SFEM_RESTRICT       values);

int affine_sshex8_laplacian_apply(const int                         level,
                                  const ptrdiff_t                   nelements,
                                  ptrdiff_t                         interior_start,
                                  idx_t **const SFEM_RESTRICT       elements,
                                  geom_t **const SFEM_RESTRICT      points,
                                  const real_t *const SFEM_RESTRICT u,
                                  real_t *const SFEM_RESTRICT       values);

int affine_sshex8_laplacian_diag(const int                    level,
                                 const ptrdiff_t              nelements,
                                 ptrdiff_t                    interior_start,
                                 idx_t **const SFEM_RESTRICT  elements,
                                 geom_t **const SFEM_RESTRICT std_hex8_points,
                                 real_t *const SFEM_RESTRICT  diag);

int affine_sshex8_laplacian_stencil_apply(const int                         level,
                                          const ptrdiff_t                   nelements,
                                          ptrdiff_t                         interior_start,
                                          idx_t **const SFEM_RESTRICT       elements,
                                          geom_t **const SFEM_RESTRICT      std_hex8_points,
                                          const real_t *const SFEM_RESTRICT u,
                                          real_t *const SFEM_RESTRICT       values);

int affine_sshex8_laplacian_stencil_apply_fff(const int                             level,
                                              const ptrdiff_t                       nelements,
                                              idx_t **const SFEM_RESTRICT           elements,
                                              const jacobian_t *const SFEM_RESTRICT fff,
                                              const real_t *const SFEM_RESTRICT     u,
                                              real_t *const SFEM_RESTRICT           values);

int sshex8_laplacian_element_matrix(int                           level,
                                    const ptrdiff_t               nelements,
                                    const ptrdiff_t               nnodes,
                                    idx_t **const SFEM_RESTRICT   elements,
                                    geom_t **const SFEM_RESTRICT  points,
                                    scalar_t *const SFEM_RESTRICT values);

int sshex8_laplacian_element_matrix_cartesian(int                           level,
                                              const ptrdiff_t               nelements,
                                              const ptrdiff_t               nnodes,
                                              idx_t **const SFEM_RESTRICT   elements,
                                              geom_t **const SFEM_RESTRICT  points,
                                              scalar_t *const SFEM_RESTRICT values);

int affine_sshex8_laplacian_substructuring_inner_fff(const int                             level,
                                                     const ptrdiff_t                       nelements,
                                                     idx_t **const SFEM_RESTRICT           elements,
                                                     const jacobian_t *const SFEM_RESTRICT fff,
                                                     const real_t *const SFEM_RESTRICT     rhs,
                                                     real_t *const SFEM_RESTRICT           u);

int affine_sshex8_laplacian_bjacobi_fff(const int                             level,
                                        const ptrdiff_t                       nelements,
                                        idx_t **const SFEM_RESTRICT           elements,
                                        const jacobian_t *const SFEM_RESTRICT g_fff,
                                        const uint16_t *const                 count,
                                        const mask_t *const                   mask,
                                        const real_t *const SFEM_RESTRICT     diag,
                                        const real_t *const SFEM_RESTRICT     rhs,
                                        real_t *const SFEM_RESTRICT           u);
#ifdef __cplusplus
}
#endif
#endif  // SFEM_SSHEX8_LAPLACIAN_H
