#ifndef MACRO_TRI3_LAPLACIAN_H
#define MACRO_TRI3_LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.h"

void macro_tri3_laplacian_apply(const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const SFEM_RESTRICT elems,
                                geom_t **const SFEM_RESTRICT xyz,
                                const real_t *const SFEM_RESTRICT u,
                                real_t *const SFEM_RESTRICT values);

#endif  // MACRO_TRI3_LAPLACIAN_H
