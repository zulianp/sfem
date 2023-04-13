#ifndef SFEM_TET4_L2_PROJECTION_P0_P1_H
#define SFEM_TET4_L2_PROJECTION_P0_P1_H

#include "sfem_base.h"
#include <stddef.h>

void tet10_ep1_p2_l2_projection_apply(const ptrdiff_t nelements,
                       const ptrdiff_t nnodes,
                       idx_t **const SFEM_RESTRICT elems,
                       geom_t **const SFEM_RESTRICT xyz,
                       const real_t *const SFEM_RESTRICT element_wise_p1,
                       real_t *const SFEM_RESTRICT p2);

void tet10_ep1_p2_projection_coeffs(const ptrdiff_t nelements,
                         const ptrdiff_t nnodes,
                         idx_t **const SFEM_RESTRICT elems,
                         geom_t **const SFEM_RESTRICT xyz,
                         const real_t *const SFEM_RESTRICT element_wise_p1,
                         real_t *const SFEM_RESTRICT p2);

#endif //SFEM_TET4_L2_PROJECTION_P0_P1_H
