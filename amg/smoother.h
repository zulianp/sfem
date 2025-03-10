#ifndef SMOOTHER_H
#define SMOOTHER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include "sfem_config.h"

int l2_smoother(const ptrdiff_t dim, const mask_t *const bdy_dofs, const count_t offdiag_nnz,
                const real_t *const diag, const real_t *const offdiag_values,
                const idx_t *const offdiag_row_indices, const idx_t *const offdiag_col_indices,
                real_t *smoother);

int l1_smoother(const ptrdiff_t dim, const mask_t *const bdy_dofs, const count_t offdiag_nnz,
                const real_t *const diag, const real_t *const offdiag_values,
                const idx_t *const offdiag_row_indices, const idx_t *const offdiag_col_indices,
                real_t *smoother);
#ifdef __cplusplus
}
#endif
#endif  // SMOOTHER_H
