#ifndef SFEM_CVFEM_TET4_LAPLACIAN_H
#define SFEM_CVFEM_TET4_LAPLACIAN_H

#include "sfem_base.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void cvfem_tet4_laplacian_assemble_hessian(const ptrdiff_t nelements,
                                           const ptrdiff_t nnodes,
                                           idx_t **const SFEM_RESTRICT elems,
                                           geom_t **const SFEM_RESTRICT xyz,
                                           const count_t *const SFEM_RESTRICT rowptr,
                                           const idx_t *const SFEM_RESTRICT colidx,
                                           real_t *const SFEM_RESTRICT values);

#ifdef __cplusplus
}
#endif
#endif //SFEM_CVFEM_TET4_LAPLACIAN_H
