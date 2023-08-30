#ifndef NAVIER_STOKES_H
#define NAVIER_STOKES_H

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

//////////////////////////
// Structure of arrays
//////////////////////////

// void navier_stokes_assemble_value_soa(const enum ElemType element_type,
//                                           const ptrdiff_t nelements,
//                                           const ptrdiff_t nnodes,
//                                           idx_t **const SFEM_RESTRICT elems,
//                                           geom_t **const SFEM_RESTRICT xyz,
//                                           const real_t nu,
//                                           const real_t rho,
//                                           const real_t **const SFEM_RESTRICT u,
//                                           real_t *const SFEM_RESTRICT value);

// void navier_stokes_assemble_gradient_soa(const enum ElemType element_type,
//                                              const ptrdiff_t nelements,
//                                              const ptrdiff_t nnodes,
//                                              idx_t **const SFEM_RESTRICT elems,
//                                              geom_t **const SFEM_RESTRICT xyz,
//                                              const real_t nu,
//                                              const real_t rho,
//                                              const real_t **const SFEM_RESTRICT u,
//                                              real_t **const SFEM_RESTRICT values);

// void navier_stokes_assemble_hessian_soa(const enum ElemType element_type,
//                                             const ptrdiff_t nelements,
//                                             const ptrdiff_t nnodes,
//                                             idx_t **const SFEM_RESTRICT elems,
//                                             geom_t **const SFEM_RESTRICT xyz,
//                                             const real_t nu,
//                                             const real_t rho,
//                                             const count_t *const SFEM_RESTRICT rowptr,
//                                             const idx_t *const SFEM_RESTRICT colidx,
//                                             real_t **const SFEM_RESTRICT values);

// void navier_stokes_apply_soa(const enum ElemType element_type,
//                                  const ptrdiff_t nelements,
//                                  const ptrdiff_t nnodes,
//                                  idx_t **const SFEM_RESTRICT elems,
//                                  geom_t **const SFEM_RESTRICT xyz,
//                                  const real_t nu,
//                                  const real_t rho,
//                                  const real_t **const SFEM_RESTRICT u,
//                                  real_t **const SFEM_RESTRICT values);


//////////////////////////
// Array of structures
//////////////////////////

void navier_stokes_assemble_value_aos(const enum ElemType element_type,
                                          const ptrdiff_t nelements,
                                          const ptrdiff_t nnodes,
                                          idx_t **const SFEM_RESTRICT elems,
                                          geom_t **const SFEM_RESTRICT xyz,
                                          const real_t nu,
                                          const real_t rho,
                                          const real_t *const SFEM_RESTRICT u,
                                          real_t *const SFEM_RESTRICT value);

void navier_stokes_assemble_gradient_aos(const enum ElemType element_type,
                                             const ptrdiff_t nelements,
                                             const ptrdiff_t nnodes,
                                             idx_t **const SFEM_RESTRICT elems,
                                             geom_t **const SFEM_RESTRICT xyz,
                                             const real_t nu,
                                             const real_t rho,
                                             const real_t *const SFEM_RESTRICT u,
                                             real_t *const SFEM_RESTRICT values);

void navier_stokes_assemble_hessian_aos(const enum ElemType element_type,
                                            const ptrdiff_t nelements,
                                            const ptrdiff_t nnodes,
                                            idx_t **const SFEM_RESTRICT elems,
                                            geom_t **const SFEM_RESTRICT xyz,
                                            const real_t nu,
                                            const real_t rho,
                                            const count_t *const SFEM_RESTRICT rowptr,
                                            const idx_t *const SFEM_RESTRICT colidx,
                                            real_t *const SFEM_RESTRICT values);

void navier_stokes_apply_aos(const enum ElemType element_type,
                                 const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elems,
                                 geom_t **const SFEM_RESTRICT xyz,
                                 const real_t nu,
                                 const real_t rho,
                                 const real_t *const SFEM_RESTRICT u,
                                 real_t *const SFEM_RESTRICT values);

void tri6_explict_momentum_tentative(const ptrdiff_t nelements,
                                     const ptrdiff_t nnodes,
                                     idx_t **const elems,
                                     geom_t **const points,
                                     const real_t dt,
                                     const real_t nu,
                                     const real_t convonoff,
                                     real_t **const SFEM_RESTRICT vel,
                                     real_t **const SFEM_RESTRICT f);

void tri3_tri6_divergence(const ptrdiff_t nelements,
                          const ptrdiff_t nnodes,
                          idx_t **const elems,
                          geom_t **const points,
                          const real_t dt,
                          const real_t rho,
                          const real_t nu,
                          real_t **const SFEM_RESTRICT vel,
                          real_t *const SFEM_RESTRICT f);

void tri6_tri3_correction(const ptrdiff_t nelements,
                          const ptrdiff_t nnodes,
                          idx_t **const elems,
                          geom_t **const points,
                          const real_t dt,
                          const real_t rho,
                          real_t **const SFEM_RESTRICT vel,
                          real_t *const SFEM_RESTRICT p,
                          real_t **const SFEM_RESTRICT values);


void tri6_momentum_lhs_scalar_crs(const ptrdiff_t nelements,
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
#endif  // NAVIER_STOKES_H
