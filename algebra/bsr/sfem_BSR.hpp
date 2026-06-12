#ifndef SFEM_BSR_SPMV_HPP
#define SFEM_BSR_SPMV_HPP

#include <cstddef>
#include <memory>

#include "sfem_MatrixFreeLinearSolver.hpp"
#include "sfem_aliases.hpp"

namespace sfem {
    template <typename R, typename C, typename TStorage>
    int bsr_transpose_sym(const ptrdiff_t                     block_rows,
                          const ptrdiff_t                     block_cols,
                          const int                           row_block_size,
                          const int                           col_block_size,
                          const R* const SFEM_RESTRICT        a_rowptr,
                          const C* const SFEM_RESTRICT        a_colidx,
                          const TStorage* const SFEM_RESTRICT a_values,
                          R* const SFEM_RESTRICT              b_rowptr) {
        (void)row_block_size;
        (void)col_block_size;
        (void)a_values;

        for (ptrdiff_t i = 0; i <= block_cols; i++) {
            b_rowptr[i] = 0;
        }

        for (ptrdiff_t i = 0; i < block_rows; i++) {
            const R                      a_begin = a_rowptr[i];
            const R                      a_end   = a_rowptr[i + 1];
            const C* const SFEM_RESTRICT a_cols  = &a_colidx[a_begin];

            for (R a_k = 0, a_len = a_end - a_begin; a_k < a_len; a_k++) {
                b_rowptr[a_cols[a_k] + 1]++;
            }
        }

        for (ptrdiff_t i = 0; i < block_cols; i++) {
            b_rowptr[i + 1] += b_rowptr[i];
        }

        return SFEM_SUCCESS;
    }

    template <int RowBlockSize, int ColBlockSize, typename R, typename C, typename TStorage>
    void bsr_transpose_apply_static(const ptrdiff_t                     block_rows,
                                    const ptrdiff_t                     block_cols,
                                    const R* const SFEM_RESTRICT        a_rowptr,
                                    const C* const SFEM_RESTRICT        a_colidx,
                                    const TStorage* const SFEM_RESTRICT a_values,
                                    const R* const SFEM_RESTRICT        b_rowptr,
                                    C* const SFEM_RESTRICT              b_colidx,
                                    TStorage* const SFEM_RESTRICT       b_values,
                                    R* const SFEM_RESTRICT              next_workspace) {
        constexpr int block_matrix_size = RowBlockSize * ColBlockSize;

        for (ptrdiff_t i = 0; i < block_cols; i++) {
            next_workspace[i] = b_rowptr[i];
        }

        for (ptrdiff_t i = 0; i < block_rows; i++) {
            const C                      b_j     = static_cast<C>(i);
            const R                      a_begin = a_rowptr[i];
            const R                      a_end   = a_rowptr[i + 1];
            const C* const SFEM_RESTRICT a_cols  = &a_colidx[a_begin];

            for (R a_k = 0, a_len = a_end - a_begin; a_k < a_len; a_k++) {
                const C b_i    = a_cols[a_k];
                const R offset = next_workspace[b_i]++;

                b_colidx[offset] = b_j;

                const TStorage* const SFEM_RESTRICT a_block = &a_values[(a_begin + a_k) * block_matrix_size];
                TStorage* const SFEM_RESTRICT       b_block = &b_values[offset * block_matrix_size];

                for (int d2 = 0; d2 < ColBlockSize; d2++) {
                    TStorage* const SFEM_RESTRICT b_row = &b_block[d2 * RowBlockSize];
                    for (int d1 = 0; d1 < RowBlockSize; d1++) {
                        b_row[d1] = a_block[d1 * ColBlockSize + d2];
                    }
                }
            }
        }
    }

    template <typename R, typename C, typename TStorage>
    void bsr_transpose_apply_dynamic(const ptrdiff_t                     block_rows,
                                     const ptrdiff_t                     block_cols,
                                     const int                           row_block_size,
                                     const int                           col_block_size,
                                     const R* const SFEM_RESTRICT        a_rowptr,
                                     const C* const SFEM_RESTRICT        a_colidx,
                                     const TStorage* const SFEM_RESTRICT a_values,
                                     const R* const SFEM_RESTRICT        b_rowptr,
                                     C* const SFEM_RESTRICT              b_colidx,
                                     TStorage* const SFEM_RESTRICT       b_values,
                                     R* const SFEM_RESTRICT              next_workspace) {
        const int block_matrix_size = row_block_size * col_block_size;

        for (ptrdiff_t i = 0; i < block_cols; i++) {
            next_workspace[i] = b_rowptr[i];
        }

        for (ptrdiff_t i = 0; i < block_rows; i++) {
            const C                      b_j     = static_cast<C>(i);
            const R                      a_begin = a_rowptr[i];
            const R                      a_end   = a_rowptr[i + 1];
            const C* const SFEM_RESTRICT a_cols  = &a_colidx[a_begin];

            for (R a_k = 0, a_len = a_end - a_begin; a_k < a_len; a_k++) {
                const C b_i    = a_cols[a_k];
                const R offset = next_workspace[b_i]++;

                b_colidx[offset] = b_j;

                const TStorage* const SFEM_RESTRICT a_block = &a_values[(a_begin + a_k) * block_matrix_size];
                TStorage* const SFEM_RESTRICT       b_block = &b_values[offset * block_matrix_size];

                for (int d2 = 0; d2 < col_block_size; d2++) {
                    TStorage* const SFEM_RESTRICT b_row = &b_block[d2 * row_block_size];
                    for (int d1 = 0; d1 < row_block_size; d1++) {
                        b_row[d1] = a_block[d1 * col_block_size + d2];
                    }
                }
            }
        }
    }

    template <typename R, typename C, typename TStorage>
    void bsr_transpose_apply(const ptrdiff_t                     block_rows,
                             const ptrdiff_t                     block_cols,
                             const int                           row_block_size,
                             const int                           col_block_size,
                             const R* const SFEM_RESTRICT        a_rowptr,
                             const C* const SFEM_RESTRICT        a_colidx,
                             const TStorage* const SFEM_RESTRICT a_values,
                             const R* const SFEM_RESTRICT        b_rowptr,
                             C* const SFEM_RESTRICT              b_colidx,
                             TStorage* const SFEM_RESTRICT       b_values,
                             R* const SFEM_RESTRICT              next_workspace) {
        if (row_block_size == 3 && col_block_size == 3) {
            bsr_transpose_apply_static<3, 3>(
                    block_rows, block_cols, a_rowptr, a_colidx, a_values, b_rowptr, b_colidx, b_values, next_workspace);
        } else if (row_block_size == 6 && col_block_size == 6) {
            bsr_transpose_apply_static<6, 6>(
                    block_rows, block_cols, a_rowptr, a_colidx, a_values, b_rowptr, b_colidx, b_values, next_workspace);
        } else if (row_block_size == 3 && col_block_size == 6) {
            bsr_transpose_apply_static<3, 6>(
                    block_rows, block_cols, a_rowptr, a_colidx, a_values, b_rowptr, b_colidx, b_values, next_workspace);
        } else if (row_block_size == 6 && col_block_size == 3) {
            bsr_transpose_apply_static<6, 3>(
                    block_rows, block_cols, a_rowptr, a_colidx, a_values, b_rowptr, b_colidx, b_values, next_workspace);
        } else {
            bsr_transpose_apply_dynamic(block_rows,
                                        block_cols,
                                        row_block_size,
                                        col_block_size,
                                        a_rowptr,
                                        a_colidx,
                                        a_values,
                                        b_rowptr,
                                        b_colidx,
                                        b_values,
                                        next_workspace);
        }
    }

    // TODO: implement bsr_mm_sym and bsr_mm_apply (see CRS as a reference)

    template <typename T>
    void bsr_scale_output(const ptrdiff_t rows, const T scale_output, T* const SFEM_RESTRICT y) {
        if (scale_output == 0) {
#pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < rows; i++) {
                y[i] = 0;
            }
        } else if (scale_output != 1) {
#pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < rows; i++) {
                y[i] *= scale_output;
            }
        }
    }

    template <int RowBlockSize, int ColBlockSize, typename R, typename C, typename TStorage, typename T = TStorage>
    void bsr_spmv_static(const ptrdiff_t                     block_rows,
                         const R* const SFEM_RESTRICT        rowptr,
                         const C* const SFEM_RESTRICT        colidx,
                         const TStorage* const SFEM_RESTRICT values,
                         const T* const SFEM_RESTRICT        x,
                         T* const SFEM_RESTRICT              y) {
        static_assert(RowBlockSize > 0, "RowBlockSize must be positive");
        static_assert(ColBlockSize > 0, "ColBlockSize must be positive");

        constexpr int block_matrix_size = RowBlockSize * ColBlockSize;

#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < block_rows; i++) {
            const R                   row_begin = rowptr[i];
            const R                   row_end   = rowptr[i + 1];
            auto* const SFEM_RESTRICT block_y   = &y[i * RowBlockSize];

            for (R k = row_begin; k < row_end; k++) {
                const C                         j       = colidx[k];
                const auto* const SFEM_RESTRICT block_x = &x[j * ColBlockSize];
                const auto* const SFEM_RESTRICT aij     = &values[k * block_matrix_size];

                for (int d1 = 0; d1 < RowBlockSize; d1++) {
                    const auto* const SFEM_RESTRICT row = &aij[d1 * ColBlockSize];
                    for (int d2 = 0; d2 < ColBlockSize; d2++) {
                        block_y[d1] += row[d2] * block_x[d2];
                    }
                }
            }
        }
    }

    template <typename R, typename C, typename TStorage, typename T = TStorage>
    void bsr_spmv_dynamic(const ptrdiff_t                     block_rows,
                          const int                           row_block_size,
                          const int                           col_block_size,
                          const R* const SFEM_RESTRICT        rowptr,
                          const C* const SFEM_RESTRICT        colidx,
                          const TStorage* const SFEM_RESTRICT values,
                          const T* const SFEM_RESTRICT        x,
                          T* const SFEM_RESTRICT              y) {
        const int block_matrix_size = row_block_size * col_block_size;

#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < block_rows; i++) {
            const R                   row_begin = rowptr[i];
            const R                   row_end   = rowptr[i + 1];
            auto* const SFEM_RESTRICT block_y   = &y[i * row_block_size];

            for (R k = row_begin; k < row_end; k++) {
                const C                         j       = colidx[k];
                const auto* const SFEM_RESTRICT block_x = &x[j * col_block_size];
                const auto* const SFEM_RESTRICT aij     = &values[k * block_matrix_size];

                for (int d1 = 0; d1 < row_block_size; d1++) {
                    const auto* const SFEM_RESTRICT row = &aij[d1 * col_block_size];
                    for (int d2 = 0; d2 < col_block_size; d2++) {
                        block_y[d1] += row[d2] * block_x[d2];
                    }
                }
            }
        }
    }

    template <typename R, typename C, typename TStorage, typename T = TStorage>
    void bsr_spmv(const ptrdiff_t                     block_rows,
                  const ptrdiff_t                     block_cols,
                  const int                           row_block_size,
                  const int                           col_block_size,
                  const R* const SFEM_RESTRICT        rowptr,
                  const C* const SFEM_RESTRICT        colidx,
                  const TStorage* const SFEM_RESTRICT values,
                  const T                             scale_output,
                  const T* const SFEM_RESTRICT        x,
                  T* const SFEM_RESTRICT              y) {
        (void)block_cols;

        bsr_scale_output(block_rows * row_block_size, scale_output, y);

        if (row_block_size == 3 && col_block_size == 3) {
            bsr_spmv_static<3, 3>(block_rows, rowptr, colidx, values, x, y);
        } else if (row_block_size == 6 && col_block_size == 6) {
            bsr_spmv_static<6, 6>(block_rows, rowptr, colidx, values, x, y);
        } else if (row_block_size == 3 && col_block_size == 6) {
            bsr_spmv_static<3, 6>(block_rows, rowptr, colidx, values, x, y);
        } else if (row_block_size == 6 && col_block_size == 3) {
            bsr_spmv_static<6, 3>(block_rows, rowptr, colidx, values, x, y);
        } else {
            bsr_spmv_dynamic(block_rows, row_block_size, col_block_size, rowptr, colidx, values, x, y);
        }
    }

    template <typename R, typename C, typename TStorage, typename T = TStorage>
    void bsr_spmv(const ptrdiff_t                     block_rows,
                  const ptrdiff_t                     block_cols,
                  const int                           block_size,
                  const R* const SFEM_RESTRICT        rowptr,
                  const C* const SFEM_RESTRICT        colidx,
                  const TStorage* const SFEM_RESTRICT values,
                  const T                             scale_output,
                  const T* const SFEM_RESTRICT        x,
                  T* const SFEM_RESTRICT              y) {
        bsr_spmv(block_rows, block_cols, block_size, block_size, rowptr, colidx, values, scale_output, x, y);
    }

    template <typename R, typename C, typename TStorage, typename T = TStorage>
    class BSR : public Operator<T> {
    public:
        std::function<void(const T* const, T* const)> apply_;

        int apply(const T* const x, T* const y) override {
            SFEM_TRACE_SCOPE("BSR::apply");

            apply_(x, y);
            return 0;
        }

        std::ptrdiff_t rows() const override { return row_block_size_ * (row_ptr->size() - 1); }
        std::ptrdiff_t cols() const override { return col_block_size_ * block_cols_; }
        inline int     row_block_size() const { return row_block_size_; }
        inline int     col_block_size() const { return col_block_size_; }
        inline int     block_size() const {
            assert(row_block_size_ == col_block_size_);
            return row_block_size_;
        }

        SharedBuffer<R>        row_ptr;
        SharedBuffer<C>        col_idx;
        SharedBuffer<TStorage> values;

        int       row_block_size_{0};
        int       col_block_size_{0};
        ptrdiff_t block_cols_{0};

        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};

        ExecutionSpace execution_space() const override { return execution_space_; }

        void print(std::ostream& os) const {
            os << "BSR" << std::endl;

            os << "row_block_size: " << row_block_size_ << std::endl;
            os << "col_block_size: " << col_block_size_ << std::endl;
            os << "block_cols: " << block_cols_ << std::endl;

            const ptrdiff_t n = (row_ptr->size() - 1);
            for (ptrdiff_t i = 0; i < n; i++) {
                for (ptrdiff_t j = row_ptr->data()[i]; j < row_ptr->data()[i + 1]; j++) {
                    const auto* const block = &values->data()[j * row_block_size_ * col_block_size_];
                    idx_t             col   = col_idx->data()[j];
                    os << "(" << i << ", " << col << "): ";
                    os << "\n";
                    for (int d1 = 0; d1 < row_block_size_; d1++) {
                        for (int d2 = 0; d2 < col_block_size_; d2++) {
                            os << block[d1 * col_block_size_ + d2] << " ";
                        }
                        os << "\n";
                    }
                    os << "\n";
                }
            }

            os << std::endl;
        }
    };

    template <typename R, typename C, typename TStorage, typename T = TStorage>
    std::shared_ptr<BSR<R, C, TStorage, T>> h_bsr_spmv(const ptrdiff_t               block_rows,
                                                       const ptrdiff_t               block_cols,
                                                       const int                     row_block_size,
                                                       const int                     col_block_size,
                                                       const SharedBuffer<R>&        rowptr,
                                                       const SharedBuffer<C>&        colidx,
                                                       const SharedBuffer<TStorage>& values,
                                                       const T                       scale_output) {
        auto ret             = std::make_shared<BSR<R, C, TStorage, T>>();
        ret->row_ptr         = rowptr;
        ret->col_idx         = colidx;
        ret->values          = values;
        ret->block_cols_     = block_cols;
        ret->row_block_size_ = row_block_size;
        ret->col_block_size_ = col_block_size;

        ret->execution_space_ = EXECUTION_SPACE_HOST;

        ret->apply_ = [=](const T* const x, T* const y) {
            bsr_spmv(block_rows,
                     block_cols,
                     row_block_size,
                     col_block_size,
                     rowptr->data(),
                     colidx->data(),
                     values->data(),
                     scale_output,
                     x,
                     y);
        };

        return ret;
    }

    template <typename R, typename C, typename TStorage, typename T = TStorage>
    std::shared_ptr<BSR<R, C, TStorage, T>> h_bsr_spmv(const ptrdiff_t               block_rows,
                                                       const ptrdiff_t               block_cols,
                                                       const int                     block_size,
                                                       const SharedBuffer<R>&        rowptr,
                                                       const SharedBuffer<C>&        colidx,
                                                       const SharedBuffer<TStorage>& values,
                                                       const T                       scale_output) {
        return h_bsr_spmv(block_rows, block_cols, block_size, block_size, rowptr, colidx, values, scale_output);
    }
}  // namespace sfem

#endif  // SFEM_BSR_SPMV_HPP
