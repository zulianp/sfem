#ifndef SSHEX8_VECTOR_LAPLACIAN_H
#define SSHEX8_VECTOR_LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

int affine_sshex8_vector_laplacian_apply_fff(const int                             level,
                                             const ptrdiff_t                       nelements,
                                             idx_t **const SFEM_RESTRICT           elements,
                                             const jacobian_t *const SFEM_RESTRICT fff,
                                             const int                             vector_size,
                                             const ptrdiff_t                       stride,
                                             real_t **const SFEM_RESTRICT          u,
                                             real_t **const SFEM_RESTRICT          values);

#ifdef __cplusplus
}
#endif
#endif  // SSHEX8_VECTOR_LAPLACIAN_H
