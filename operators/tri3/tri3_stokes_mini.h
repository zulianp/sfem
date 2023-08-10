#ifndef TRI3_STOKES_MINI_H
#define TRI3_STOKES_MINI_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

void tri3_stokes_mini_assemble_hessian_soa(const ptrdiff_t nelements,
                                           const ptrdiff_t nnodes,
                                           idx_t **const elems,
                                           geom_t **const points,
                                           const real_t mu,
                                           const count_t *const rowptr,
                                           const idx_t *const colidx,
                                           real_t **const values);

void tri3_stokes_mini_assemble_hessian_aos(const ptrdiff_t nelements,
                                           const ptrdiff_t nnodes,
                                           idx_t **const elems,
                                           geom_t **const points,
                                           const real_t mu,
                                           const count_t *const rowptr,
                                           const idx_t *const colidx,
                                           real_t *const values);

void tri3_stokes_mini_assemble_rhs_soa(const ptrdiff_t nelements,
                                       const ptrdiff_t nnodes,
                                       idx_t **const elems,
                                       geom_t **const points,
                                       const real_t mu,
                                       const real_t rho,
                                       real_t **SFEM_RESTRICT forcing,
                                       real_t **const SFEM_RESTRICT rhs);

void tri3_stokes_mini_assemble_rhs_aos(const ptrdiff_t nelements,
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

#endif  // TRI3_STOKES_MINI_H
