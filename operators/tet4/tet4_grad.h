#ifndef SFEM_GRAD_P1_H
#define SFEM_GRAD_P1_H

#include "sfem_base.h"
#include <stddef.h>


void tet4_grad(const ptrdiff_t nelements,
              const ptrdiff_t nnodes,
              idx_t **const SFEM_RESTRICT elems,
              geom_t **SFEM_RESTRICT xyz,
              const real_t *const SFEM_RESTRICT f,
              real_t *const SFEM_RESTRICT dfdx,
              real_t *const SFEM_RESTRICT dfdy,
              real_t *const SFEM_RESTRICT dfdz);

#endif  // SFEM_GRAD_P1_H
