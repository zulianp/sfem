#ifndef SSHEX8_STENCIL_ELEMENT_MATRIX_APPLY_H
#define SSHEX8_STENCIL_ELEMENT_MATRIX_APPLY_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

int sshex8_stencil_element_matrix_apply(const int                           level,
                                        const ptrdiff_t                     nelements,
                                        idx_t **const SFEM_RESTRICT         elements,
                                        const scalar_t *const SFEM_RESTRICT g_element_matrix,
                                        const real_t *const SFEM_RESTRICT   u,
                                        real_t *const SFEM_RESTRICT         values);

#ifdef __cplusplus
}
#endif

#endif  // SSHEX8_STENCIL_ELEMENT_MATRIX_APPLY_H
