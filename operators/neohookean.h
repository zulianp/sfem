#ifndef LAPLACIAN_H
#define LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.h"

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

void neohookean_cauchy_stress_aos(const ptrdiff_t nelements,
                              const ptrdiff_t nnodes,
                              idx_t *const SFEM_RESTRICT elems[4],
                              geom_t *const SFEM_RESTRICT xyz[3],
                              const real_t mu,
                              const real_t lambda,
                              const real_t *const SFEM_RESTRICT displacement,
                              real_t *const SFEM_RESTRICT out[9]);

void neohookean_cauchy_stress_soa(const ptrdiff_t nelements,
                                  const ptrdiff_t nnodes,
                                  idx_t *const SFEM_RESTRICT elems[4],
                                  geom_t *const SFEM_RESTRICT xyz[3],
                                  const real_t mu,
                                  const real_t lambda,
                                  real_t **const SFEM_RESTRICT u,
                                  real_t *const SFEM_RESTRICT out[9]);

#endif  // LAPLACIAN_H
