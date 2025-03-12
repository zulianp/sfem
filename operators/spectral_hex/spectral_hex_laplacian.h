#ifndef SPECTRAL_LAPLACIAN_H
#define SPECTRAL_LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

int spectral_hex_laplacian_apply(const int                         order,
                                 const ptrdiff_t                   nelements,
                                 const ptrdiff_t                   nnodes,
                                 idx_t** const SFEM_RESTRICT       elements,
                                 geom_t** const SFEM_RESTRICT      points,
                                 const real_t* const SFEM_RESTRICT u,
                                 real_t* const SFEM_RESTRICT       values);

#ifdef __cplusplus
}
#endif
#endif  // SPECTRAL_LAPLACIAN_H
