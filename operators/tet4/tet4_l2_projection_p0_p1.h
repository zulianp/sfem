#ifndef SFEM_TET4_L2_PROJECTION_P0_P1_H
#define SFEM_TET4_L2_PROJECTION_P0_P1_H

#include "sfem_base.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void tet4_p0_p1_l2_projection_apply(const ptrdiff_t nelements,
                       const ptrdiff_t nnodes,
                       idx_t **const SFEM_RESTRICT elems,
                       geom_t **const SFEM_RESTRICT xyz,
                       const real_t *const SFEM_RESTRICT p0,
                       real_t *const SFEM_RESTRICT p1);

void tet4_p0_p1_projection_coeffs(const ptrdiff_t nelements,
                         const ptrdiff_t nnodes,
                         idx_t **const SFEM_RESTRICT elems,
                         geom_t **const SFEM_RESTRICT xyz,
                         const real_t *const SFEM_RESTRICT p0,
                         real_t *const SFEM_RESTRICT p1);


#ifdef __cplusplus
}
#endif

#endif //SFEM_TET4_L2_PROJECTION_P0_P1_H
