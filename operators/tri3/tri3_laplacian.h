#ifndef TRI3_LAPLACIAN_H
#define TRI3_LAPLACIAN_H

#include "sfem_base.h"
#include <stddef.h>

void tri3_laplacian_assemble_hessian(const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const SFEM_RESTRICT elems,
                                geom_t **const SFEM_RESTRICT xyz,
                                const count_t *const SFEM_RESTRICT rowptr,
                                const idx_t *const SFEM_RESTRICT colidx,
                                real_t *const SFEM_RESTRICT values);

#endif //TRI3_LAPLACIAN_H
