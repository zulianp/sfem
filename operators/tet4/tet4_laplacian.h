#ifndef TET4_LAPLACIAN_H
#define TET4_LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.h"

void tet4_laplacian_assemble_value(const ptrdiff_t nelements,
                              const ptrdiff_t nnodes,
                              idx_t **const SFEM_RESTRICT elems,
                              geom_t **const SFEM_RESTRICT xyz,
                              const real_t *const SFEM_RESTRICT u,
                              real_t *const SFEM_RESTRICT value);

void tet4_laplacian_assemble_gradient(const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elems,
                                 geom_t **const SFEM_RESTRICT xyz,
                                 const real_t *const SFEM_RESTRICT u,
                                 real_t *const SFEM_RESTRICT values);

void tet4_laplacian_assemble_hessian(const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const SFEM_RESTRICT elems,
                                geom_t **const SFEM_RESTRICT xyz,
                                const count_t *const SFEM_RESTRICT rowptr,
                                const idx_t *const SFEM_RESTRICT colidx,
                                real_t *const SFEM_RESTRICT values);

void tet4_laplacian_apply(const ptrdiff_t nelements,
                     const ptrdiff_t nnodes,
                     idx_t **const SFEM_RESTRICT elems,
                     geom_t **const SFEM_RESTRICT xyz,
                     const real_t *const SFEM_RESTRICT u,
                     real_t *const SFEM_RESTRICT values);

#endif  // TET4_LAPLACIAN_H
