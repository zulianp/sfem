#ifndef TRI3_LAPLACIAN_H
#define TRI3_LAPLACIAN_H

#include "sfem_base.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif


void tri3_laplacian_assemble_hessian(const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const SFEM_RESTRICT elems,
                                geom_t **const SFEM_RESTRICT xyz,
                                const count_t *const SFEM_RESTRICT rowptr,
                                const idx_t *const SFEM_RESTRICT colidx,
                                real_t *const SFEM_RESTRICT values);

void tri3_laplacian_apply(const ptrdiff_t nelements,
                          const ptrdiff_t nnodes,
                          idx_t **const SFEM_RESTRICT elems,
                          geom_t **const SFEM_RESTRICT xyz,
                          const real_t *const SFEM_RESTRICT u,
                          real_t *const SFEM_RESTRICT values);

#ifdef __cplusplus
}
#endif

#endif //TRI3_LAPLACIAN_H
