#ifndef TET10_NAVIER_STOKES_H
#define TET10_NAVIER_STOKES_H

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"
#include "crs_graph.h"
#include "sortreduce.h"

#ifdef __cplusplus
extern "C" {
#endif

void tet10_explict_momentum_tentative(const ptrdiff_t nelements,
                                     const ptrdiff_t nnodes,
                                     idx_t **const elems,
                                     geom_t **const points,
                                     const real_t dt,
                                     const real_t nu,
                                     const real_t convonoff,
                                     real_t **const SFEM_RESTRICT vel,
                                     real_t **const SFEM_RESTRICT f);

void tet4_tet10_divergence(const ptrdiff_t nelements,
                          const ptrdiff_t nnodes,
                          idx_t **const elems,
                          geom_t **const points,
                          const real_t dt,
                          const real_t rho,
                          const real_t nu,
                          real_t **const SFEM_RESTRICT vel,
                          real_t *const SFEM_RESTRICT f);

void tet10_tet4_correction(const ptrdiff_t nelements,
                          const ptrdiff_t nnodes,
                          idx_t **const elems,
                          geom_t **const points,
                          const real_t dt,
                          const real_t rho,
                          real_t *const SFEM_RESTRICT p,
                          real_t **const SFEM_RESTRICT values);


void tet10_momentum_lhs_scalar_crs(const ptrdiff_t nelements,
                                  const ptrdiff_t nnodes,
                                  idx_t **const elems,
                                  geom_t **const points,
                                  const real_t dt,
                                  const real_t nu,
                                  const count_t *const SFEM_RESTRICT rowptr,
                                  const idx_t *const SFEM_RESTRICT colidx,
                                  real_t *const SFEM_RESTRICT values);

#ifdef __cplusplus
}
#endif
#endif  // TET10_NAVIER_STOKES_H
