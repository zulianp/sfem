#ifndef LAPLACIAN_H
#define LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.h"

void neohookean_assemble_hessian(const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t *const elems[4],
                                 geom_t *const xyz[3],
                                 const real_t mu,
                                 const real_t lambda,
                                 const real_t *const displacement,
                                 count_t *const rowptr,
                                 idx_t *const colidx,
                                 real_t *const values);

#endif  // LAPLACIAN_H
