#ifndef SFEM_DIV_H
#define SFEM_DIV_H

#include "sfem_base.h"
#include <stddef.h>

void div_apply(const ptrdiff_t nelements,
               const ptrdiff_t nnodes,
               idx_t **const elems,
               geom_t **const xyz,
               const real_t *const ux,
               const real_t *const uy,
               const real_t *const uz,
               real_t *const values);

#endif  // SFEM_DIV_H
