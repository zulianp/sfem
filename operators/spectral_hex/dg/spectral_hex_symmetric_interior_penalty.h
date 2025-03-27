#ifndef SPECTRAL_HEX_LAX_FRIEDRICHS_FLUX_H
#define SPECTRAL_HEX_LAX_FRIEDRICHS_FLUX_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

// - < v n, {{ grad u }}> - < grad v, [[u]]/2> + <v n, tau [[u]]>>
int spectral_hex_symmetric_interior_penalty_apply(const int                                order,
                                                  const ptrdiff_t                          nelements,
                                                  const ptrdiff_t                          nnodes,
                                                  idx_t** const SFEM_RESTRICT              elements,
                                                  const element_idx_t* const SFEM_RESTRICT adj_table,
                                                  const sshex_side_code_t* const           side_code,
                                                  geom_t** const SFEM_RESTRICT             points,
                                                  const real_t                             tau,
                                                  const real_t* const SFEM_RESTRICT        u,
                                                  real_t* const SFEM_RESTRICT              values);

#ifdef __cplusplus
}
#endif
#endif  // SPECTRAL_HEX_LAX_FRIEDRICHS_FLUX_H
