#ifndef SFEM_CVFEM_TET4_LAPLACIAN_H
#define SFEM_CVFEM_TET4_LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

void cvfem_tet4_convection_assemble_hessian(const ptrdiff_t nelements,
                                            const ptrdiff_t nnodes,
                                            idx_t **const SFEM_RESTRICT elems,
                                            geom_t **const SFEM_RESTRICT xyz,
                                            real_t **const SFEM_RESTRICT velocity,
                                            const count_t *const SFEM_RESTRICT rowptr,
                                            const idx_t *const SFEM_RESTRICT colidx,
                                            real_t *const SFEM_RESTRICT values);

void cvfem_tet4_convection_apply(const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elems,
                                 geom_t **const SFEM_RESTRICT xyz,
                                 real_t **const SFEM_RESTRICT velocity,
                                 const real_t *const SFEM_RESTRICT u,
                                 real_t *const SFEM_RESTRICT values);

void cvfem_tet4_cv_volumes(const ptrdiff_t nelements,
                           const ptrdiff_t nnodes,
                           idx_t **const SFEM_RESTRICT elems,
                           geom_t **const SFEM_RESTRICT xyz,
                           real_t *const SFEM_RESTRICT values);

#ifdef __cplusplus
}
#endif
#endif  // SFEM_CVFEM_TET4_LAPLACIAN_H
