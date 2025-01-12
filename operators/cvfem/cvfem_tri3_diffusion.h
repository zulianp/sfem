#ifndef SFEM_CVFEM_TRI3_DIFFUSION_H
#define SFEM_CVFEM_TRI3_DIFFUSION_H

#include "sfem_base.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void cvfem_tri3_diffusion_assemble_hessian(const ptrdiff_t nelements,
                                           const ptrdiff_t nnodes,
                                           idx_t **const SFEM_RESTRICT elems,
                                           geom_t **const SFEM_RESTRICT xyz,
                                           const count_t *const SFEM_RESTRICT rowptr,
                                           const idx_t *const SFEM_RESTRICT colidx,
                                           real_t *const SFEM_RESTRICT values);
#ifdef __cplusplus
}
#endif

#endif //SFEM_CVFEM_TRI3_DIFFUSION_H
