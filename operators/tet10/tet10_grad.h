#ifndef SFEM_TET10_GRAD_H
#define SFEM_TET10_GRAD_H

#include "sfem_base.h"
#include <stddef.h>

// Returns P1 coefficients per element
void tet10_grad(const ptrdiff_t nelements,
                const ptrdiff_t nnodes,
                idx_t **const SFEM_RESTRICT elems,
                geom_t **SFEM_RESTRICT xyz,
                const real_t *const SFEM_RESTRICT f,
                real_t *const SFEM_RESTRICT dfdx,
                real_t *const SFEM_RESTRICT dfdy,
                real_t *const SFEM_RESTRICT dfdz);

#endif  // SFEM_TET10_GRAD_H
