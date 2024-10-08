#ifndef SFEM_NEOHOOKEAN_H
#define SFEM_NEOHOOKEAN_H

#include <stddef.h>
#include "sfem_base.h"

// FIXME all functionalities to be revisited and moved to tet4_neohookean_ogden.{h,c}

#ifdef __cplusplus
extern "C" {
#endif

void neohookean_assemble_hessian(const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t *const SFEM_RESTRICT elems[4],
                                 geom_t *const SFEM_RESTRICT xyz[3],
                                 const real_t mu,
                                 const real_t lambda,
                                 const real_t *const SFEM_RESTRICT displacement,
                                 count_t *const SFEM_RESTRICT rowptr,
                                 idx_t *const SFEM_RESTRICT colidx,
                                 real_t *const SFEM_RESTRICT values);

void neohookean_assemble_gradient(const ptrdiff_t nelements,
                                  const ptrdiff_t nnodes,
                                  idx_t *const SFEM_RESTRICT elems[4],
                                  geom_t *const SFEM_RESTRICT xyz[3],
                                  const real_t mu,
                                  const real_t lambda,
                                  const real_t *const SFEM_RESTRICT displacement,
                                  real_t *const SFEM_RESTRICT values);


void neohookean_assemble_value(const ptrdiff_t nelements,
                               const ptrdiff_t nnodes,
                               idx_t *const SFEM_RESTRICT elems[4],
                               geom_t *const SFEM_RESTRICT xyz[3],
                               const real_t mu,
                               const real_t lambda,
                               const real_t *const SFEM_RESTRICT displacement,
                               real_t *const SFEM_RESTRICT value);

void neohookean_cauchy_stress_aos(const ptrdiff_t nelements,
                                  const ptrdiff_t nnodes,
                                  idx_t *const SFEM_RESTRICT elems[4],
                                  geom_t *const SFEM_RESTRICT xyz[3],
                                  const real_t mu,
                                  const real_t lambda,
                                  const real_t *const SFEM_RESTRICT displacement,
                                  real_t *const SFEM_RESTRICT out[6]);

void neohookean_cauchy_stress_soa(const ptrdiff_t nelements,
                                  const ptrdiff_t nnodes,
                                  idx_t *const SFEM_RESTRICT elems[4],
                                  geom_t *const SFEM_RESTRICT xyz[3],
                                  const real_t mu,
                                  const real_t lambda,
                                  real_t **const SFEM_RESTRICT u,
                                  real_t *const SFEM_RESTRICT out[6]);

void neohookean_vonmises_soa(const ptrdiff_t nelements,
                             const ptrdiff_t nnodes,
                             idx_t *const SFEM_RESTRICT elems[4],
                             geom_t *const SFEM_RESTRICT xyz[3],
                             const real_t mu,
                             const real_t lambda,
                             real_t **const SFEM_RESTRICT u,
                             real_t *const SFEM_RESTRICT out);

void neohookean_principal_stresses_aos(const ptrdiff_t nelements,
                                       const ptrdiff_t nnodes,
                                       idx_t **const SFEM_RESTRICT elems,
                                       geom_t **const SFEM_RESTRICT xyz,
                                       const real_t mu,
                                       const real_t lambda,
                                       real_t *const SFEM_RESTRICT u,
                                       real_t **const SFEM_RESTRICT stress);

void neohookean_principal_stresses_soa(const ptrdiff_t nelements,
                                       const ptrdiff_t nnodes,
                                       idx_t **const SFEM_RESTRICT elems,
                                       geom_t **const SFEM_RESTRICT xyz,
                                       const real_t mu,
                                       const real_t lambda,
                                       real_t **const SFEM_RESTRICT u,
                                       real_t **const SFEM_RESTRICT stress);

void neohookean_assemble_hessian_soa(const ptrdiff_t nelements,
                                     const ptrdiff_t nnodes,
                                     idx_t **const SFEM_RESTRICT elems,
                                     geom_t **const SFEM_RESTRICT xyz,
                                     const real_t mu,
                                     const real_t lambda,
                                     real_t **const SFEM_RESTRICT displacement,
                                     count_t *const SFEM_RESTRICT rowptr,
                                     idx_t *const SFEM_RESTRICT colidx,
                                     real_t **const SFEM_RESTRICT values);

#ifdef __cplusplus
}
#endif
#endif  // SFEM_NEOHOOKEAN_H
