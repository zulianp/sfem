#ifndef HEX8_LAPLACIAN_H
#define HEX8_LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.h"

int hex8_laplacian_apply(const ptrdiff_t nelements,
                         const ptrdiff_t nnodes,
                         idx_t **const SFEM_RESTRICT elements,
                         geom_t **const SFEM_RESTRICT points,
                         const real_t *const SFEM_RESTRICT u,
                         real_t *const SFEM_RESTRICT values);


#endif  // HEX8_LAPLACIAN_H
