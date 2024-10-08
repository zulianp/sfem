#ifndef TRI6_LAPLACIAN_H
#define TRI6_LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.h"

// Code generated with laplace_op.py (mixed symbolic and numerical integration)

int tri6_laplacian_assemble_value(const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elems,
                                 geom_t **const SFEM_RESTRICT xyz,
                                 const real_t *const SFEM_RESTRICT u,
                                 real_t *const SFEM_RESTRICT value);

int tri6_laplacian_apply(const ptrdiff_t nelements,
                        const ptrdiff_t nnodes,
                        idx_t **const SFEM_RESTRICT elems,
                        geom_t **const SFEM_RESTRICT xyz,
                        const real_t *const SFEM_RESTRICT u,
                        real_t *const SFEM_RESTRICT values);

int tri6_laplacian_crs(const ptrdiff_t nelements,
                                   const ptrdiff_t nnodes,
                                   idx_t **const SFEM_RESTRICT elems,
                                   geom_t **const SFEM_RESTRICT xyz,
                                   const count_t *const SFEM_RESTRICT rowptr,
                                   const idx_t *const SFEM_RESTRICT colidx,
                                   real_t *const SFEM_RESTRICT values);

int tri6_laplacian_diag(const ptrdiff_t nelements,
                           const ptrdiff_t nnodes,
                           idx_t **const SFEM_RESTRICT elements,
                           geom_t **const SFEM_RESTRICT points,
                           real_t *const SFEM_RESTRICT diag);

int tri6_laplacian_apply_opt(const ptrdiff_t nelements,
                             idx_t **const SFEM_RESTRICT elements,
                             const jacobian_t *const SFEM_RESTRICT fff,
                             const real_t *const SFEM_RESTRICT u,
                             real_t *const SFEM_RESTRICT values);

int tri6_laplacian_diag_opt(const ptrdiff_t nelements,
                            idx_t **const SFEM_RESTRICT elements,
                            const jacobian_t *const SFEM_RESTRICT fff,
                            real_t *const SFEM_RESTRICT diag);

#endif  // TRI6_LAPLACIAN_H
