#ifndef SFE_TET_10_MASS_H
#define SFE_TET_10_MASS_H

#include <stddef.h>

#include "sfem_base.h"

void tet10_assemble_mass(const ptrdiff_t nelements,
                         const ptrdiff_t nnodes,
                         idx_t **const SFEM_RESTRICT elems,
                         geom_t **const SFEM_RESTRICT xyz,
                         const count_t *const SFEM_RESTRICT rowptr,
                         const idx_t *const SFEM_RESTRICT colidx,
                         real_t *const SFEM_RESTRICT values);

void tet10_apply_inv_lumped_mass(const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elems,
                                 geom_t **const SFEM_RESTRICT xyz,
                                 const real_t *const SFEM_RESTRICT x,
                                 real_t *const SFEM_RESTRICT values);

void tet10_assemble_lumped_mass(const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const SFEM_RESTRICT elems,
                                geom_t **const SFEM_RESTRICT xyz,
                                real_t *const SFEM_RESTRICT values);

#endif  // SFE_TET_10_MASS_H
