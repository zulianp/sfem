#ifndef SFEM_STRAIN_H
#define SFEM_STRAIN_H

#include <stddef.h>

#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif


void strain(const ptrdiff_t nelements,
           const ptrdiff_t nnodes,
           idx_t **const SFEM_RESTRICT elems,
           geom_t **const SFEM_RESTRICT xyz,
           const real_t *const SFEM_RESTRICT ux,
           const real_t *const SFEM_RESTRICT uy,
           const real_t *const SFEM_RESTRICT uz,
           real_t *const SFEM_RESTRICT strain_xx,
           real_t *const SFEM_RESTRICT strain_xy,
           real_t *const SFEM_RESTRICT strain_xz,
           real_t *const SFEM_RESTRICT strain_yy,
           real_t *const SFEM_RESTRICT strain_yz,
           real_t *const SFEM_RESTRICT strain_zz);

// See principal_strain.cpp
void principal_strains(const ptrdiff_t nelements,
           const ptrdiff_t nnodes,
           idx_t **const SFEM_RESTRICT elems,
           geom_t **const SFEM_RESTRICT xyz,
           const real_t *const SFEM_RESTRICT ux,
           const real_t *const SFEM_RESTRICT uy,
           const real_t *const SFEM_RESTRICT uz,
           real_t *const SFEM_RESTRICT strain_e0,
           real_t *const SFEM_RESTRICT strain_e1,
           real_t *const SFEM_RESTRICT strain_e2);


#ifdef __cplusplus
}
#endif


#endif //SFEM_STRAIN_H
