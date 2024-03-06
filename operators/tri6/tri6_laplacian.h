#ifndef TRI6_LAPLACIAN_H
#define TRI6_LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.h"

void tri6_laplacian_assemble_hessian(const ptrdiff_t nelements,
                                     const ptrdiff_t nnodes,
                                     idx_t **const SFEM_RESTRICT elems,
                                     geom_t **const SFEM_RESTRICT xyz,
                                     const count_t *const SFEM_RESTRICT rowptr,
                                     const idx_t *const SFEM_RESTRICT colidx,
                                     real_t *const SFEM_RESTRICT values);

#endif  // TRI6_LAPLACIAN_H
