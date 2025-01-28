#ifndef SFEM_BOUNDARY_MASS_H
#define SFEM_BOUNDARY_MASS_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

void assemble_boundary_mass(const ptrdiff_t nelements,
                   const ptrdiff_t nnodes,
                   idx_t *const elems[3],
                   geom_t *const xyz[3],
                   count_t *const rowptr,
                   idx_t *const colidx,
                   real_t *const values);

void assemble_lumped_boundary_mass(const ptrdiff_t nelements,
                          const ptrdiff_t nnodes,
                          idx_t *const elems[3],
                          geom_t *const xyz[3],
                          real_t *const values);

#ifdef __cplusplus
}
#endif

#endif  // SFEM_BOUNDARY_MASS_H
