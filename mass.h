#ifndef MASS_H
#define MASS_H

#include "sfem_base.h"
#include <stddef.h>

void assemble_mass(const ptrdiff_t nelements,
                   const ptrdiff_t nnodes,
                   idx_t *const elems[4],
                   geom_t *const xyz[3],
                   idx_t *const rowptr,
                   idx_t *const colidx,
                   real_t *const values);

#endif  // MASS_H
