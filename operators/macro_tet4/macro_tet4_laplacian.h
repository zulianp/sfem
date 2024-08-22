#ifndef MACRO_TET4_LAPLACIAN_H
#define MACRO_TET4_LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

int macro_tet4_laplacian_apply(const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const SFEM_RESTRICT elems,
                                geom_t **const SFEM_RESTRICT xyz,
                                const real_t *const SFEM_RESTRICT u,
                                real_t *const SFEM_RESTRICT values);

int macro_tet4_laplacian_crs(const ptrdiff_t nelements,
                                           const ptrdiff_t nnodes,
                                           idx_t **const SFEM_RESTRICT elems,
                                           geom_t **const SFEM_RESTRICT xyz,
                                           const count_t *const SFEM_RESTRICT rowptr,
                                           const idx_t *const SFEM_RESTRICT colidx,
                                           real_t *const SFEM_RESTRICT values);

int macro_tet4_laplacian_diag(const ptrdiff_t nelements,
                               const ptrdiff_t nnodes,
                               idx_t **const SFEM_RESTRICT elements,
                               geom_t **const SFEM_RESTRICT points,
                               real_t *const SFEM_RESTRICT diag);

// Optimized for matrix-free
int macro_tet4_laplacian_crs_opt(const ptrdiff_t nelements,
                                               idx_t **const SFEM_RESTRICT elements,
                                               const jacobian_t *const SFEM_RESTRICT fff,
                                               const count_t *const SFEM_RESTRICT rowptr,
                                               const idx_t *const SFEM_RESTRICT colidx,
                                               real_t *const SFEM_RESTRICT values);

int macro_tet4_laplacian_apply_opt(const ptrdiff_t nelements,
                                   idx_t **const SFEM_RESTRICT elements,
                                   const jacobian_t *const SFEM_RESTRICT fff,
                                   const real_t *const SFEM_RESTRICT u,
                                   real_t *const SFEM_RESTRICT values);

int macro_tet4_laplacian_diag_opt(const ptrdiff_t nelements,
                                  idx_t **const SFEM_RESTRICT elements,
                                  const jacobian_t *const SFEM_RESTRICT fff,
                                  real_t *const SFEM_RESTRICT diag);

#ifdef __cplusplus
}
#endif
#endif  // MACRO_TET4_LAPLACIAN_H
