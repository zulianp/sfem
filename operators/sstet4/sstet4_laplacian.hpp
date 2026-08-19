#ifndef SSTET4_LAPLACIAN_H
#define SSTET4_LAPLACIAN_H

// Adapted from Bole Ma implementation in https://github.com/zulianp/hpcfem

#include <stddef.h>
#include "sfem_base.hpp"

#ifdef __cplusplus
extern "C" {
#endif

int sstet4_nxe(int level);
int sstet4_txe(int level);

int sstet4_laplacian_apply(const int                             level,
                           const ptrdiff_t                       nelements,
                           const jacobian_t *const SFEM_RESTRICT fff,
                           const real_t *const SFEM_RESTRICT     u,
                           real_t *const SFEM_RESTRICT           values);

int sstet4_laplacian_apply_points(const int                         level,
                                  const ptrdiff_t                   nelements,
                                  idx_t **const SFEM_RESTRICT       elements,
                                  geom_t **const SFEM_RESTRICT      points,
                                  const real_t *const SFEM_RESTRICT u,
                                  real_t *const SFEM_RESTRICT       values);

int sstet4_laplacian_diag_points(const int                    level,
                                 const ptrdiff_t              nelements,
                                 idx_t **const SFEM_RESTRICT  elements,
                                 geom_t **const SFEM_RESTRICT points,
                                 real_t *const SFEM_RESTRICT  values);

#ifdef __cplusplus
}
#endif
#endif  // SSTET4_LAPLACIAN_H
