#ifndef MACRO_TET4_LAPLACIAN_H
#define MACRO_TET4_LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef struct {
    ptrdiff_t nelements;
    jacobian_t *fff;
    idx_t **elements;
} macro_tet4_laplacian_t;

void macro_tet4_laplacian_init(macro_tet4_laplacian_t *const ctx,
                               const ptrdiff_t nelements,
                               idx_t **const SFEM_RESTRICT elems,
                               geom_t **const SFEM_RESTRICT xyz);

void macro_tet4_laplacian_destroy(macro_tet4_laplacian_t *const ctx);

void macro_tet4_laplacian_apply_opt(const macro_tet4_laplacian_t *const ctx,
                                    const real_t *const SFEM_RESTRICT u,
                                    real_t *const SFEM_RESTRICT values);

void macro_tet4_laplacian_diag(const macro_tet4_laplacian_t *const ctx,
                               real_t *const SFEM_RESTRICT diag);

void macro_tet4_laplacian_apply(const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const SFEM_RESTRICT elems,
                                geom_t **const SFEM_RESTRICT xyz,
                                const real_t *const SFEM_RESTRICT u,
                                real_t *const SFEM_RESTRICT values);

#ifdef __cplusplus
}
#endif
#endif  // MACRO_TET4_LAPLACIAN_H
