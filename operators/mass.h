#ifndef MASS_H
#define MASS_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

void assemble_mass(const int element_type,
                   const ptrdiff_t nelements,
                   const ptrdiff_t nnodes,
                   idx_t **const SFEM_RESTRICT elems,
                   geom_t **const SFEM_RESTRICT xyz,
                   count_t *const SFEM_RESTRICT rowptr,
                   idx_t *const SFEM_RESTRICT colidx,
                   real_t *const SFEM_RESTRICT values);

void assemble_lumped_mass(const int element_type,
                          const ptrdiff_t nelements,
                          const ptrdiff_t nnodes,
                          idx_t **const SFEM_RESTRICT elems,
                          geom_t **const SFEM_RESTRICT xyz,
                          real_t *const SFEM_RESTRICT values);

void apply_inv_lumped_mass(const int element_type,
                           const ptrdiff_t nelements,
                           const ptrdiff_t nnodes,
                           idx_t **const SFEM_RESTRICT elems,
                           geom_t **const SFEM_RESTRICT xyz,
                           const real_t *const x,
                           real_t *const values);

void apply_mass(const int element_type,
                const ptrdiff_t nelements,
                const ptrdiff_t nnodes,
                idx_t **const SFEM_RESTRICT elems,
                geom_t **const SFEM_RESTRICT xyz,
                const real_t *const x,
                real_t *const values);

#ifdef __cplusplus
}
#endif

#endif  // MASS_H
