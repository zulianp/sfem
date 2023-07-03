#ifndef SFEM_TET4_MASS_H
#define SFEM_TET4_MASS_H

#include <stddef.h>
#include "sfem_base.h"

void tet4_assemble_mass(const ptrdiff_t nelements,
                        const ptrdiff_t nnodes,
                        idx_t **const SFEM_RESTRICT elems,
                        geom_t **const SFEM_RESTRICT xyz,
                        count_t *const rowptr,
                        idx_t *const colidx,
                        real_t *const values);

void tet4_assemble_lumped_mass(const ptrdiff_t nelements,
                               const ptrdiff_t nnodes,
                               idx_t **const SFEM_RESTRICT elems,
                               geom_t **const SFEM_RESTRICT xyz,
                               real_t *const SFEM_RESTRICT values);

void tet4_apply_inv_lumped_mass(const ptrdiff_t nelements,
                          const ptrdiff_t nnodes,
                          idx_t **const SFEM_RESTRICT elems,
                          geom_t **const SFEM_RESTRICT xyz,
                          const real_t*const x,
                          real_t *const values);

#endif  // SFEM_TET4_MASS_H
