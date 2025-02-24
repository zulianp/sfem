#ifndef SSHEX8_STENCIL_ELEMENT_MATRIX_APPLY_H
#define SSHEX8_STENCIL_ELEMENT_MATRIX_APPLY_H

#include "sfem_base.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int sshex8_stencil_element_matrix_apply(const int                         level,
                                        const ptrdiff_t                   nelements,
                                        ptrdiff_t                         interior_start,
                                        idx_t **const SFEM_RESTRICT       elements,
                                        scalar_t *const SFEM_RESTRICT    g_element_matrix,
                                        const real_t *const SFEM_RESTRICT u,
                                        real_t *const SFEM_RESTRICT       values);

#ifdef __cplusplus
}
#endif

#endif  // SSHEX8_STENCIL_ELEMENT_MATRIX_APPLY_H
