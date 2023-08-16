#ifndef TRI6_MASS_H
#define TRI6_MASS_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

void tri6_apply_inv_lumped_mass(const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const SFEM_RESTRICT elems,
                                geom_t **const SFEM_RESTRICT points,
                                const real_t *const x,
                                real_t *const values);

#ifdef __cplusplus
}
#endif

#endif
