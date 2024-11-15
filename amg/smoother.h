#ifndef SMOOTHER_H
#define SMOOTHER_H

#include "sparse.h"

ptrdiff_t dim;  // Number of rows / columns

count_t offdiag_nnz;         // Length of 3 below arrays, or half the actualy number of
                             // off diagonal non-zero entries
idx_t *offdiag_row_indices;  // Row indices of off diagonal non-zero entries
idx_t *offdiag_col_indices;  // Column indices of off diagonal non-zero entries
real_t *offdiag_values;      // Off diagonal non-zero values

real_t *diag;  // Diagonal values

int l2_smoother(const ptrdiff_t dim,
                const count_t offdiag_nnz,
                const real_t *const diag,
                const idx_t *const offdiag_row_indices,
                const idx_t *const offdiag_col_indices,
                real_t *smoother);

int l1_smoother(const ptrdiff_t dim,
                const count_t offdiag_nnz,
                const real_t *const diag,
                const idx_t *const offdiag_row_indices,
                const idx_t *const offdiag_col_indices,
                real_t *smoother);
#endif  // SMOOTHER_H
