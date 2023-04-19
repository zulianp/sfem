#ifndef SFE_TET_10_MASS_H
#define SFE_TET_10_MASS_H

#include <stddef.h>

#include "sfem_base.h"

void tet10_assemble_mass(const ptrdiff_t nelements,
                   const ptrdiff_t nnodes,
                   idx_t **const elems,
                   geom_t **const xyz,
                   count_t *const rowptr,
                   idx_t *const colidx,
                   real_t *const values);

void tet_10_apply_inv_lumped_mass(const ptrdiff_t nelements,
                          const ptrdiff_t nnodes,
                          idx_t **const SFEM_RESTRICT elems,
                          geom_t **const SFEM_RESTRICT xyz,
                          const real_t*const x,
                          real_t *const values);

#endif //SFE_TET_10_MASS_H
