#ifndef SURFACE_L2_PROJECTION_H
#define SURFACE_L2_PROJECTION_H

#include <stddef.h>

#include "sfem_base.h"

void surface_e_projection_apply(const int element_type,
                                const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const SFEM_RESTRICT elems,
                                geom_t **const SFEM_RESTRICT xyz,
                                const real_t *const SFEM_RESTRICT element_wise_u,
                                real_t *const SFEM_RESTRICT u);

void surface_e_projection_coeffs(const int element_type,
                                 const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elems,
                                 geom_t **const SFEM_RESTRICT xyz,
                                 const real_t *const SFEM_RESTRICT element_wise_u,
                                 real_t *const SFEM_RESTRICT u);

#endif  // SURFACE_L2_PROJECTION_H
