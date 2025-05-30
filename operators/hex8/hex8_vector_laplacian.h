#ifndef HEX8_VECTOR_LAPLACIAN_H
#define HEX8_VECTOR_LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

int affine_hex8_vector_laplacian_apply(const ptrdiff_t              nelements,
                                       const ptrdiff_t              nnodes,
                                       idx_t **const SFEM_RESTRICT  elements,
                                       geom_t **const SFEM_RESTRICT points,
                                       const int                    vector_size,
                                       const ptrdiff_t              stride,
                                       real_t **const SFEM_RESTRICT u,
                                       real_t **const SFEM_RESTRICT values);

int affine_hex8_vector_laplacian_apply_fff(const ptrdiff_t                       nelements,
                                           idx_t **const SFEM_RESTRICT           elements,
                                           const jacobian_t *const SFEM_RESTRICT fff,
                                           const int                             vector_size,
                                           const ptrdiff_t                       stride,
                                           real_t **const SFEM_RESTRICT          u,
                                           real_t **const SFEM_RESTRICT          values);

#ifdef __cplusplus
}
#endif
#endif  // HEX8_VECTOR_LAPLACIAN_H
