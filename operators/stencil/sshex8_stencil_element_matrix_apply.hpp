#ifndef SSHEX8_STENCIL_ELEMENT_MATRIX_APPLY_H
#define SSHEX8_STENCIL_ELEMENT_MATRIX_APPLY_H

#include <stddef.h>
#include "sfem_base.hpp"

#ifdef __cplusplus
extern "C" {
#endif

int sshex8_stencil_element_matrix_apply(const int                           level,
                                        const ptrdiff_t                     nelements,
                                        idx_t **const SFEM_RESTRICT         elements,
                                        const scalar_t *const SFEM_RESTRICT g_element_matrix,
                                        const real_t *const SFEM_RESTRICT   u,
                                        real_t *const SFEM_RESTRICT         values);

int sshex8_stencil_element_matrix_apply3(const int                           level,
                                         const ptrdiff_t                     nelements,
                                         idx_t **const SFEM_RESTRICT         elements,
                                         const scalar_t *const SFEM_RESTRICT g_element_matrix,
                                         const ptrdiff_t                     u_stride,
                                         const real_t *const SFEM_RESTRICT   ux,
                                         const real_t *const SFEM_RESTRICT   uy,
                                         const real_t *const SFEM_RESTRICT   uz,
                                         const ptrdiff_t                     out_stride,
                                         real_t *const SFEM_RESTRICT         outx,
                                         real_t *const SFEM_RESTRICT         outy,
                                         real_t *const SFEM_RESTRICT         outz);

#ifdef __cplusplus
}
#endif

#endif  // SSHEX8_STENCIL_ELEMENT_MATRIX_APPLY_H
