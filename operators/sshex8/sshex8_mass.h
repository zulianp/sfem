#ifndef SFEM_SSHEX8_MASS_H
#define SFEM_SSHEX8_MASS_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

// int sshex8_mass_apply(const int                         level,
//                       const ptrdiff_t                   nelements,
//                       ptrdiff_t                         interior_start,
//                       idx_t **const SFEM_RESTRICT       elements,
//                       geom_t **const SFEM_RESTRICT      points,
//                       const real_t *const SFEM_RESTRICT u,
//                       real_t *const SFEM_RESTRICT       values);

int affine_sshex8_mass_lumped(const int                    level,
                              const ptrdiff_t              nelements,
                              ptrdiff_t                    interior_start,
                              idx_t **const SFEM_RESTRICT  elements,
                              geom_t **const SFEM_RESTRICT std_hex8_points,
                              real_t *const SFEM_RESTRICT  diag);

#ifdef __cplusplus
}
#endif
#endif  // SFEM_SSHEX8_MASS_H
