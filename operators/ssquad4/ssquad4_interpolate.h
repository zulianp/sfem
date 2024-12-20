#ifndef SSQUAD4_INTERPOLATE_H
#define SSQUAD4_INTERPOLATE_H

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int ssquad4_restrict(const int                           level,
                     const int                           from_level,
                     const int                           to_level,
                     const ptrdiff_t                     nelements,
                     idx_t **const SFEM_RESTRICT         elements,
                     const uint16_t *const SFEM_RESTRICT element_to_node_incidence_count,
                     const int                           vec_size,
                     const real_t *const SFEM_RESTRICT   from,
                     real_t *const SFEM_RESTRICT         to);

int ssquad4_prolongate(const ptrdiff_t                   nelements,
                       const int                         from_level,
                       const int                         from_level_stride,
                       idx_t **const SFEM_RESTRICT       from_elements,
                       const int                         to_level,
                       const int                         to_level_stride,
                       idx_t **const SFEM_RESTRICT       to_elements,
                       const int                         vec_size,
                       const real_t *const SFEM_RESTRICT from,
                       real_t *const SFEM_RESTRICT       to) ;

#ifdef __cplusplus
}
#endif

#endif  // SSQUAD4_INTERPOLATE_H