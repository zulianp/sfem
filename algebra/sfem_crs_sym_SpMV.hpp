#ifndef SFEM_CRS_SYM_SPMV_HPP
#define SFEM_CRS_SYM_SPMV_HPP

#include <iostream>

namespace sfem {
    template <typename R, typename C, typename T>
    class CRSSymSpMV : public Operator<T> {
    public:
        // TODO
        // std::function<void(const T* const, T* const)> apply_;

        int apply(const T* const x, T* const y) override {
            apply_(x, y);
            return 0;
        }

        std::ptrdiff_t rows() const override { return rows_; }
        std::ptrdiff_t cols() const override { return cols_; }

        std::shared_ptr<Buffer<R>> rowptr;
        std::shared_ptr<Buffer<C>> colidx;
        std::shared_ptr<Buffer<T>> diag_values;
        std::shared_ptr<Buffer<T>> off_diag_values;
        T scale_output{0};

        ptrdiff_t rows_{0};
        ptrdiff_t cols_{0};

        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};
        ExecutionSpace execution_space() const override { return execution_space_; }

        void apply_(const T* const x, T* const y) const {
            const ptrdiff_t nnz = this->colidx->size();
            const ptrdiff_t rows = this->rows_;

            // Check preconditions
            assert(this->colidx->size() == nnz);
            assert(this->off_diag_values->size() == nnz);
            assert(this->diag_values->size() == rows);

            // Get raw pointers
            auto rowptr = this->rowptr->data();
            auto colidx = this->colidx->data();
            auto diag_values = this->diag_values->data();
            auto off_diag_values = this->off_diag_values->data();

#pragma omp parallel for
            for (ptrdiff_t i = 0; i < rows; ++i) {
                y[i] = scale_output * y[i] + (diag_values[i] * x[i]);
            }

            // Apply off-diagonal blocks
#pragma omp parallel for
            for (ptrdiff_t row = 0; row < rows; row++) {
                const count_t lenrow = rowptr[row + 1] - rowptr[row];
                const idx_t* const cols = &colidx[rowptr[row]];
                const real_t* const values = &off_diag_values[rowptr[row]];

                T y_local = 0;
                for (count_t k = 0; k < lenrow; k++) {
                    const auto col = cols[k];
                    y_local += values[k] * x[col];
                }
                y[row] += y_local;

                const T x_local = x[row];
                for (count_t k = 0; k < lenrow; k++) {
                    const idx_t col = cols[k];
                    y[col] += values[k] * x_local;
                }
            }
        }
    };

    template <typename R, typename C, typename T>
    std::shared_ptr<CRSSymSpMV<R, C, T>> h_crs_sym_spmv(
            const ptrdiff_t rows,
            const ptrdiff_t cols,
            const std::shared_ptr<Buffer<R>>& rowptr,
            const std::shared_ptr<Buffer<C>>& colidx,
            const std::shared_ptr<Buffer<T>>& diag_values,
            const std::shared_ptr<Buffer<T>>& off_diag_values,
            const T scale_output) {
        auto ret = std::make_shared<CRSSymSpMV<R, C, T>>();
        ret->rowptr = rowptr;
        ret->colidx = colidx;
        ret->diag_values = diag_values;
        ret->off_diag_values = off_diag_values;
        ret->scale_output = scale_output;
        ret->rows_ = rows;
        ret->cols_ = cols;
        ret->execution_space_ = EXECUTION_SPACE_HOST;
        return ret;
    }
}  // namespace sfem

#endif  // SFEM_BC_SYM_SPMV_HPP
