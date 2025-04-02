#ifndef SPECTRAL_HEX_ADVECTION_H
#define SPECTRAL_HEX_ADVECTION_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

int spectral_hex_advection_apply(const int                         order,
                                 const ptrdiff_t                   nelements,
                                 const ptrdiff_t                   nnodes,
                                 idx_t** const SFEM_RESTRICT       elements,
                                 geom_t** const SFEM_RESTRICT      points,
                                 const real_t* const SFEM_RESTRICT vx,
                                 const real_t* const SFEM_RESTRICT vy,
                                 const real_t* const SFEM_RESTRICT vz,
                                 const real_t* const SFEM_RESTRICT c,
                                 real_t* const SFEM_RESTRICT       values);

#ifdef __cplusplus
}
#endif
#endif  // SPECTRAL_HEX_ADVECTION_H
