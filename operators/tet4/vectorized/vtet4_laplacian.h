#ifndef VTET4_LAPLACIAN_H
#define VTET4_LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.h"

int vtet4_laplacian_apply(const ptrdiff_t nelements,
                          const ptrdiff_t nnodes,
                          idx_t **const SFEM_RESTRICT elements,
                          geom_t **const SFEM_RESTRICT points,
                          const real_t *const SFEM_RESTRICT u,
                          real_t *const SFEM_RESTRICT values);

int vtet4_laplacian_apply_opt(const ptrdiff_t nelements,
                             idx_t **const SFEM_RESTRICT elements,
                             const jacobian_t *const SFEM_RESTRICT fff,
                             const real_t *const SFEM_RESTRICT u,
                             real_t *const SFEM_RESTRICT values);

#endif  // VTET4_LAPLACIAN_H
