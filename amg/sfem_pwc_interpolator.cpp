#include "sfem_pwc_interpolator.hpp"

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

namespace sfem {
    template <typename R, typename T>
    std::shared_ptr<sfem::CooSymSpMV<R, T>> sfem::PiecewiseConstantInterpolator<R, T>::coarsen(
            const std::shared_ptr<sfem::CooSymSpMV<R, T>> &a) {
        count_t fine_nnz = a->values->size();
        R *partition = partition_->data();
        T *weights = weights_->data();

        R *row_indices = (R *)malloc(fine_nnz * sizeof(R));
        R *col_indices = (R *)malloc(fine_nnz * sizeof(R));
        T *values = (T *)malloc(fine_nnz * sizeof(T));
        T *acoarse_diag = (T *)calloc(coarse_dim, sizeof(T));

        R *sort_indices = (R *)malloc(fine_nnz * sizeof(R));
        count_t new_nnz = fine_nnz;

        pwc_restrict(a->diag_values->data(), acoarse_diag);

        R write_pos = 0;
        for (R k = 0; k < fine_nnz; k++) {
            R i = a->offdiag_rowidx->data()[k];
            R j = a->offdiag_colidx->data()[k];
            T val = a->values->data()[k];

            R coarse_i = partition[i];
            R coarse_j = partition[j];
            T coarse_val = weights[i] * val * weights[j];
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

        R *offdiag_row_indices = (R *)malloc(new_nnz * sizeof(R));
        R *offdiag_col_indices = (R *)malloc(new_nnz * sizeof(R));
        T *offdiag_values = (T *)malloc(new_nnz * sizeof(T));

#pragma omp parallel for
        for (R k = 0; k < new_nnz; k++) {
            offdiag_row_indices[k] = row_indices[k];
            offdiag_col_indices[k] = col_indices[k];
            offdiag_values[k] = values[k];
        }

        free(values);
        free(col_indices);
        free(row_indices);
        free(sort_indices);

        return h_coosym(Buffer<R>::own(new_nnz, offdiag_row_indices, free, MEMORY_SPACE_HOST),
                        Buffer<R>::own(new_nnz, offdiag_col_indices, free, MEMORY_SPACE_HOST),
                        Buffer<T>::own(new_nnz, offdiag_values, free, MEMORY_SPACE_HOST),
                        Buffer<T>::own(coarse_dim, acoarse_diag, free, MEMORY_SPACE_HOST)

        );
    }

    template class PiecewiseConstantInterpolator<idx_t, real_t>;
}  // namespace sfem
