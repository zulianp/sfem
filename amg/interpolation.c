#include "interpolation.h"

/*
void piecewise_constant(const idx_t *partition, const real_t *near_null,
                        COOMatrix *p) {
  for (idx_t i = 0; i < p->nrows; i++) {
    p->row_indices[i] = i;
    p->col_indices[i] = partition[i];
    p->values[i] = near_null[i];
  }
}
*/

void coarsen(const SymmCOOMatrix *a, const PiecewiseConstantTransfer *p, SymmCOOMatrix *a_coarse) {
    count_t fine_nnz = a->offdiag_nnz;

    idx_t *row_indices = (idx_t *)malloc(fine_nnz * sizeof(idx_t));
    idx_t *col_indices = (idx_t *)malloc(fine_nnz * sizeof(idx_t));
    real_t *values = (real_t *)malloc(fine_nnz * sizeof(real_t));
    real_t *acoarse_diag = (real_t *)calloc(p->coarse_dim, sizeof(real_t));

    idx_t *sort_indices = (idx_t *)malloc(fine_nnz * sizeof(idx_t));
    count_t new_nnz = fine_nnz;

    pwc_restrict(p, a->diag, acoarse_diag);

    idx_t write_pos = 0;
    for (idx_t k = 0; k < fine_nnz; k++) {
        idx_t i = a->offdiag_row_indices[k];
        idx_t j = a->offdiag_col_indices[k];
        real_t val = a->offdiag_values[k];

        idx_t coarse_i = p->partition[i];
        idx_t coarse_j = p->partition[j];
        real_t coarse_val = p->weights[i] * val * p->weights[j];
        if (coarse_j > coarse_i) {
            row_indices[write_pos] = coarse_i;
            col_indices[write_pos] = coarse_j;
            values[write_pos] = coarse_val;
            write_pos++;
        } else if (coarse_i > coarse_j) {
            row_indices[write_pos] = coarse_j;
            col_indices[write_pos] = coarse_i;
            values[write_pos] = coarse_val;
            write_pos++;
        } else {
            if (coarse_i >= 0) {
                acoarse_diag[coarse_i] += coarse_val;
            }
        }
    }

    sum_duplicates(row_indices, col_indices, values, sort_indices, &new_nnz);

    a_coarse->diag = acoarse_diag;
    a_coarse->dim = p->coarse_dim;
    a_coarse->offdiag_nnz = new_nnz;
    a_coarse->offdiag_row_indices = (idx_t *)malloc(new_nnz * sizeof(idx_t));
    a_coarse->offdiag_col_indices = (idx_t *)malloc(new_nnz * sizeof(idx_t));
    a_coarse->offdiag_values = (real_t *)malloc(new_nnz * sizeof(real_t));

#pragma omp parallel for
    for (idx_t k = 0; k < new_nnz; k++) {
        a_coarse->offdiag_row_indices[k] = row_indices[k];
        a_coarse->offdiag_col_indices[k] = col_indices[k];
        a_coarse->offdiag_values[k] = values[k];
    }

    free(values);
    free(col_indices);
    free(row_indices);
    free(sort_indices);
}

void pwc_interpolate(const PiecewiseConstantTransfer *p, const real_t *v_coarse, real_t *v) {
#pragma omp parallel for
    for (idx_t k = 0; k < p->fine_dim; k++) {
        idx_t coarse_idx = p->partition[k];
        if (coarse_idx >= 0) {
            v[k] = v_coarse[coarse_idx] * p->weights[k];
        }
    }
}

void pwc_restrict(const PiecewiseConstantTransfer *p, const real_t *v, real_t *v_coarse) {
#pragma omp parallel for
    for (idx_t k = 0; k < p->coarse_dim; k++) {
        v_coarse[k] = 0.0;
    }

    for (idx_t k = 0; k < p->fine_dim; k++) {
        idx_t coarse_idx = p->partition[k];
        if (coarse_idx >= 0) {
            v_coarse[coarse_idx] += v[k] * p->weights[k];
        }
    }
}
