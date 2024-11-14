#include "smoother.h"
#include "sparse.h"
#include <math.h>

int l2_smoother(const SymmCOOMatrix *mat, real_t *smoother) {

#pragma omp parallel for
  for (idx_t i = 0; i < mat->dim; i++) {
    smoother[i] = mat->diag[i];
  }

  for (idx_t k = 0; k < mat->offdiag_nnz; k++) {
    idx_t i = mat->offdiag_row_indices[k];
    idx_t j = mat->offdiag_col_indices[k];
    real_t val = fabs(mat->offdiag_values[k]);

    // Could avoid repeated sqrt with workspace
    real_t sqrt_i = sqrt(mat->diag[i]);
    real_t sqrt_j = sqrt(mat->diag[j]);
    smoother[i] += val * sqrt_i / sqrt_j;
    smoother[j] += val * sqrt_j / sqrt_i;
  }

#pragma omp parallel for
  for (idx_t i = 0; i < mat->dim; i++) {
    smoother[i] = 1.0 / smoother[i];
  }

  return 0;
}

int l1_smoother(const SymmCOOMatrix *mat, real_t *smoother) {

#pragma omp parallel for
  for (idx_t i = 0; i < mat->dim; i++) {
    smoother[i] = mat->diag[i];
  }

  for (idx_t k = 0; k < mat->offdiag_nnz; k++) {
    idx_t i = mat->offdiag_row_indices[k];
    idx_t j = mat->offdiag_col_indices[k];
    real_t val = fabs(mat->offdiag_values[k]);

    smoother[i] += val;
    smoother[j] += val;
  }

#pragma omp parallel for
  for (idx_t i = 0; i < mat->dim; i++) {
    smoother[i] = 1.0 / smoother[i];
  }

  return 0;
}
