#ifndef STOKES_MINI_H
#define STOKES_MINI_H

#include <stddef.h>

#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

void stokes_mini_assemble_hessian_soa(enum ElemType element_type,
                                      const ptrdiff_t nelements,
                                      const ptrdiff_t nnodes,
                                      idx_t **const elems,
                                      geom_t **const points,
                                      const real_t mu,
                                      const count_t *const rowptr,
                                      const idx_t *const colidx,
                                      real_t **const values);



void stokes_mini_assemble_hessian_aos(enum ElemType element_type,
                                      const ptrdiff_t nelements,
                                      const ptrdiff_t nnodes,
                                      idx_t **const elems,
                                      geom_t **const points,
                                      const real_t mu,
                                      const count_t *const rowptr,
                                      const idx_t *const colidx,
                                      real_t *const values);

void stokes_mini_assemble_rhs_soa(enum ElemType element_type,
                                       const ptrdiff_t nelements,
                                       const ptrdiff_t nnodes,
                                       idx_t **const elems,
                                       geom_t **const points,
                                       const real_t mu,
                                       const real_t rho,
                                       real_t **SFEM_RESTRICT forcing,
                                       real_t **const SFEM_RESTRICT rhs);

void stokes_mini_assemble_rhs_aos(enum ElemType element_type,
                                       const ptrdiff_t nelements,
                                       const ptrdiff_t nnodes,
                                       idx_t **const elems,
                                       geom_t **const points,
                                       const real_t mu,
                                       const real_t rho,
                                       real_t **SFEM_RESTRICT forcing,
                                       real_t *const SFEM_RESTRICT rhs);

#ifdef __cplusplus
}
#endif

#endif  // STOKES_MINI_H
