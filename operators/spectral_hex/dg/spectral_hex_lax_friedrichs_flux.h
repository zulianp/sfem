#ifndef SPECTRAL_HEX_LAX_FRIEDRICHS_FLUX_H
#define SPECTRAL_HEX_LAX_FRIEDRICHS_FLUX_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif


// < v n, {{c u}} + | c n |/w [[ u ]]>
int spectral_hex_lax_friedrichs_flux(const int                         order,
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
#endif  // SPECTRAL_HEX_LAX_FRIEDRICHS_FLUX_H
