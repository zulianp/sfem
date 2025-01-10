#ifndef SFEM_NEUMANN_H
#define SFEM_NEUMANN_H

#include <stddef.h>
#include "sfem_base.h"


#ifdef __cplusplus
extern "C" {
#endif

void surface_forcing_function(const int element_type,
                              const ptrdiff_t nfaces,
                              const idx_t *SFEM_RESTRICT faces_neumann,
                              geom_t **const SFEM_RESTRICT xyz,
                              const real_t value,
                              real_t * SFEM_RESTRICT output);

void surface_forcing_function_vec(
                            const int element_type,
                              const ptrdiff_t nfaces,
                              const idx_t *faces_neumann,
                              geom_t **const xyz,
                              const real_t value,
                              const int block_size,
                              const int component,
                              real_t *output);


#ifdef __cplusplus
}
#endif

#endif  // SFEM_NEUMANN_H
