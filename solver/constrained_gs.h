#ifndef CONSTRAINED_GS_H
#define CONSTRAINED_GS_H
#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int constrained_gs(const ptrdiff_t nnodes,
                   const count_t *const SFEM_RESTRICT rowptr,
                   const idx_t *const SFEM_RESTRICT colidx,
                   real_t *const SFEM_RESTRICT values,
                   real_t *const SFEM_RESTRICT inv_diag,
                   real_t *const SFEM_RESTRICT rhs,
                   real_t *const SFEM_RESTRICT x,
                   const real_t *const SFEM_RESTRICT weights,
                   const real_t sum_weights,
                   real_t *const SFEM_RESTRICT lagrange_multiplier,
                   const int num_sweeps);

int constrained_gs_init(const ptrdiff_t nnodes,
                        const count_t *const SFEM_RESTRICT rowptr,
                        const idx_t *const SFEM_RESTRICT colidx,
                        real_t *const SFEM_RESTRICT values,
                        real_t *const SFEM_RESTRICT off,
                        real_t *const inv_diag);

int constrained_gs_residual(const ptrdiff_t nnodes,
             const count_t *const SFEM_RESTRICT rowptr,
             const idx_t *const SFEM_RESTRICT colidx,
             real_t *const SFEM_RESTRICT values,
             real_t *const SFEM_RESTRICT rhs,
             real_t *const SFEM_RESTRICT x,
             real_t *res);

#ifdef __cplusplus
}
#endif

#endif  // CONSTRAINED_GS_H
