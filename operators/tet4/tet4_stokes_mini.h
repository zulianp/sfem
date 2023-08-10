#ifndef TET4_STOKES_MINI_H
#define TET4_STOKES_MINI_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

void tet4_stokes_mini_assemble_hessian_soa(const ptrdiff_t nelements,
                                           const ptrdiff_t nnodes,
                                           idx_t **const elems,
                                           geom_t **const points,
                                           const real_t mu,
                                           const count_t *const rowptr,
                                           const idx_t *const colidx,
                                           real_t **const values);

void tet4_stokes_mini_assemble_hessian_aos(const ptrdiff_t nelements,
                                           const ptrdiff_t nnodes,
                                           idx_t **const elems,
                                           geom_t **const points,
                                           const real_t mu,
                                           const count_t *const rowptr,
                                           const idx_t *const colidx,
                                           real_t *const values);

#ifdef __cplusplus
}
#endif

#endif  // TET4_STOKES_MINI_H
