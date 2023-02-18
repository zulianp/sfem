#ifndef LAPLACIAN_H
#define LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.h"

void laplacian_assemble_value(const ptrdiff_t nelements,
                              const ptrdiff_t nnodes,
                              idx_t **const elems,
                              geom_t **const xyz,
                              const real_t *const u,
                              real_t *const value);

void laplacian_assemble_gradient(const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const elems,
                                 geom_t **const xyz,
                                 const real_t *const u,
                                 real_t *const values);

void laplacian_assemble_hessian(const ptrdiff_t nelements,
                        const ptrdiff_t nnodes,
                        idx_t **const elems,
                        geom_t **const xyz,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values);

#endif  // LAPLACIAN_H
