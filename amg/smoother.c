#include "smoother.h"
#include <math.h>
#include "sfem_mask.h"

int l2_smoother(const ptrdiff_t dim, const mask_t *const bdy_dofs, const count_t offdiag_nnz,
                const real_t *const diag, const real_t *const offdiag_values,
                const idx_t *const offdiag_row_indices, const idx_t *const offdiag_col_indices,
                real_t *smoother) {
#pragma omp parallel for
    for (idx_t i = 0; i < dim; i++) {
        smoother[i] = diag[i];
    }

    for (idx_t k = 0; k < offdiag_nnz; k++) {
        idx_t i = offdiag_row_indices[k];
        idx_t j = offdiag_col_indices[k];
        if ((!bdy_dofs) || !(mask_get(i, bdy_dofs) || mask_get(j, bdy_dofs))) {
            real_t val = fabs(offdiag_values[k]);

            // Could avoid repeated sqrt with workspace
            real_t sqrt_i = sqrt(diag[i]);
            real_t sqrt_j = sqrt(diag[j]);
            smoother[i] += val * sqrt_i / sqrt_j;
            smoother[j] += val * sqrt_j / sqrt_i;
        }
    }

#pragma omp parallel for
    for (idx_t i = 0; i < dim; i++) {
        if ((!bdy_dofs) || !mask_get(i, bdy_dofs)) {
            // smoother[i] = 1.0 / smoother[i];
        } else {
            smoother[i] = 0.0;
        }
    }

    return 0;
}

int l1_smoother(const ptrdiff_t dim, const mask_t *const bdy_dofs, const count_t offdiag_nnz,
                const real_t *const diag, const real_t *const offdiag_values,
                const idx_t *const offdiag_row_indices, const idx_t *const offdiag_col_indices,
                real_t *smoother) {
#pragma omp parallel for
    for (idx_t i = 0; i < dim; i++) {
        smoother[i] = diag[i];
    }

    for (idx_t k = 0; k < offdiag_nnz; k++) {
        idx_t i = offdiag_row_indices[k];
        idx_t j = offdiag_col_indices[k];
        if ((!bdy_dofs) || !(mask_get(i, bdy_dofs) || mask_get(j, bdy_dofs))) {
            real_t val = fabs(offdiag_values[k]);

            smoother[i] += val;
            smoother[j] += val;
        }
    }

#pragma omp parallel for
    for (idx_t i = 0; i < dim; i++) {
        if ((!bdy_dofs) || !mask_get(i, bdy_dofs)) {
            // smoother[i] = 1.0 / smoother[i];
        } else {
            smoother[i] = 0.0;
        }
    }

    return 0;
}
