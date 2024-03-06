#ifndef TET4_VISOUS_POWER_DENSITY_CURNIER_HPP
#define TET4_VISOUS_POWER_DENSITY_CURNIER_HPP

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

void tet4_viscous_power_density_curiner_assemble_hessian(const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t *const SFEM_RESTRICT elems[4],
                                 geom_t *const SFEM_RESTRICT xyz[3],
                                 const real_t mu,
                                 const real_t lambda,
                                 const real_t *const SFEM_RESTRICT displacement,
                                 count_t *const SFEM_RESTRICT rowptr,
                                 idx_t *const SFEM_RESTRICT colidx,
                                 real_t *const SFEM_RESTRICT values);

void tet4_viscous_power_density_curiner_assemble_gradient(const ptrdiff_t nelements,
                                  const ptrdiff_t nnodes,
                                  idx_t *const SFEM_RESTRICT elems[4],
                                  geom_t *const SFEM_RESTRICT xyz[3],
                                  const real_t mu,
                                  const real_t lambda,
                                  const real_t *const SFEM_RESTRICT displacement,
                                  real_t *const SFEM_RESTRICT values);


void tet4_viscous_power_density_curiner_assemble_value(const ptrdiff_t nelements,
                               const ptrdiff_t nnodes,
                               idx_t *const SFEM_RESTRICT elems[4],
                               geom_t *const SFEM_RESTRICT xyz[3],
                               const real_t mu,
                               const real_t lambda,
                               const real_t *const SFEM_RESTRICT displacement,
                               real_t *const SFEM_RESTRICT value);

#ifdef __cplusplus
}
#endif
#endif  // TET4_VISOUS_POWER_DENSITY_CURNIER_HPP
