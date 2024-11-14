#ifndef LAPLACIAN_H
#define LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.h"
#include "tet4_fff.h"

#ifdef __cplusplus
extern "C" {
#endif

int laplacian_is_opt(int element_type);

int laplacian_assemble_value(int element_type,
                             const ptrdiff_t nelements,
                             const ptrdiff_t nnodes,
                             idx_t **const SFEM_RESTRICT elems,
                             geom_t **const SFEM_RESTRICT xyz,
                             const real_t *const SFEM_RESTRICT u,
                             real_t *const SFEM_RESTRICT value);

int laplacian_apply(int element_type,
                    const ptrdiff_t nelements,
                    const ptrdiff_t nnodes,
                    idx_t **const SFEM_RESTRICT elements,
                    geom_t **const SFEM_RESTRICT points,
                    const real_t *const SFEM_RESTRICT u,
                    real_t *const SFEM_RESTRICT values);

int laplacian_assemble_gradient(int element_type,
                                const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const SFEM_RESTRICT elements,
                                geom_t **const SFEM_RESTRICT points,
                                const real_t *const SFEM_RESTRICT u,
                                real_t *const SFEM_RESTRICT values);

int laplacian_crs(int element_type,
                  const ptrdiff_t nelements,
                  const ptrdiff_t nnodes,
                  idx_t **const SFEM_RESTRICT elems,
                  geom_t **const SFEM_RESTRICT xyz,
                  const count_t *const SFEM_RESTRICT rowptr,
                  const idx_t *const SFEM_RESTRICT colidx,
                  real_t *const SFEM_RESTRICT values);

int laplacian_diag(int element_type,
                   const ptrdiff_t nelements,
                   const ptrdiff_t nnodes,
                   idx_t **const SFEM_RESTRICT elements,
                   geom_t **const SFEM_RESTRICT points,
                   real_t *const SFEM_RESTRICT values);

// Optimized for matrix-free
int laplacian_apply_opt(int element_type,
                        const ptrdiff_t nelements,
                        idx_t **const SFEM_RESTRICT elements,
                        const jacobian_t *const SFEM_RESTRICT fff,
                        const real_t *const SFEM_RESTRICT u,
                        real_t *const SFEM_RESTRICT values);

#ifdef __cplusplus
}
#endif

#endif  // LAPLACIAN_H
