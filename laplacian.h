#ifndef LAPLACIAN_H
#define LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.h"

void assemble_laplacian(const ptrdiff_t nelements,
                        const ptrdiff_t nnodes,
                        idx_t **const elems,
                        geom_t **const xyz,
                        const idx_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values);

#endif  // LAPLACIAN_H
