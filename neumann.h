#ifndef SFEM_NEUMANN_H
#define SFEM_NEUMANN_H

#include <stddef.h>
#include "sfem_base.h"

void surface_forcing_function(const ptrdiff_t nfaces,
                              const idx_t *faces_neumann,
                              geom_t **const xyz,
                              const real_t value,
                              real_t *output);

#endif  // SFEM_NEUMANN_H
