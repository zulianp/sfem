#ifndef QUAD4_LAPLACIAN_H
#define QUAD4_LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.hpp"

#ifdef __cplusplus
extern "C" {
#endif

int quad4_laplacian_apply(const ptrdiff_t                   nelements,
                          const ptrdiff_t                   nnodes,
                          idx_t **const SFEM_RESTRICT       elements,
                          geom_t **const SFEM_RESTRICT      points,
                          const real_t *const SFEM_RESTRICT u,
                          real_t *const SFEM_RESTRICT       values);

int quad4_laplacian_diag(const ptrdiff_t              nelements,
                         const ptrdiff_t              nnodes,
                         idx_t **const SFEM_RESTRICT  elements,
                         geom_t **const SFEM_RESTRICT points,
                         real_t *const SFEM_RESTRICT  diag);

#ifdef __cplusplus
}
#endif

#endif  // QUAD4_LAPLACIAN_H
