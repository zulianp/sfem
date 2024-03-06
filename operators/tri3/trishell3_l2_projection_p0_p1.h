#ifndef TRISHELL3_L2_PROJECTION_P0_P1_H
#define TRISHELL3_L2_PROJECTION_P0_P1_H

#include <stddef.h>

#include "sfem_defs.h"


void trishell3_p0_p1_l2_projection_apply(const ptrdiff_t nelements,
                                         const ptrdiff_t nnodes,
                                         idx_t **const SFEM_RESTRICT elems,
                                         geom_t **const SFEM_RESTRICT xyz,
                                         const real_t *const SFEM_RESTRICT p0,
                                         real_t *const SFEM_RESTRICT p1);

void trishell3_p0_p1_projection_coeffs(const ptrdiff_t nelements,
                                       const ptrdiff_t nnodes,
                                       idx_t **const SFEM_RESTRICT elems,
                                       geom_t **const SFEM_RESTRICT xyz,
                                       const real_t *const SFEM_RESTRICT p0,
                                       real_t *const SFEM_RESTRICT p1);

#endif  // TRISHELL3_L2_PROJECTION_P0_P1_H
