#ifndef TET10_LINEAR_ELASTICITY_H
#define TET10_LINEAR_ELASTICITY_H

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"
#include "tet4_linear_elasticity.h"

#ifdef __cplusplus
extern "C" {
#endif

void tet10_linear_elasticity_init(linear_elasticity_t *const ctx,
                                  const real_t mu,
                                  const real_t lambda,
                                  const ptrdiff_t nelements,
                                  idx_t **const SFEM_RESTRICT elements,
                                  geom_t **const SFEM_RESTRICT points);

void tet10_linear_elasticity_destroy(linear_elasticity_t *const ctx);

void tet10_linear_elasticity_apply_opt(const linear_elasticity_t *const ctx,
                                       const real_t *const SFEM_RESTRICT u,
                                       real_t *const SFEM_RESTRICT values);

void tet10_linear_elasticity_diag(const linear_elasticity_t *const ctx,
                                  real_t *const SFEM_RESTRICT diag);

void tet10_linear_elasticity_apply_aos(const ptrdiff_t nelements,
                                       const ptrdiff_t nnodes,
                                       idx_t **const SFEM_RESTRICT elements,
                                       geom_t **const SFEM_RESTRICT points,
                                       const real_t mu,
                                       const real_t lambda,
                                       const real_t *const SFEM_RESTRICT u,
                                       real_t *const SFEM_RESTRICT values);

#ifdef __cplusplus
}
#endif
#endif  // TET10_LINEAR_ELASTICITY_H
