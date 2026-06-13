#ifndef SFEM_BSRSOA_HPP
#define SFEM_BSRSOA_HPP

#include <cassert>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "sfem_MatrixFreeLinearSolver.hpp"
#include "sfem_aliases.hpp"
#include "smesh_types.hpp"

namespace sfem {

    template <typename R, typename C, typename TStorage, typename T = TStorage>
    static inline T bsr_soa_dot(const R                            extent,
                                const C* const SFEM_RESTRICT        cols,
                                const TStorage* const SFEM_RESTRICT vals,
                                const T* const SFEM_RESTRICT        x) {
        T acc = 0;
#pragma omp simd reduction(+ : acc)
        for (R k = 0; k < extent; k++) {
            acc += vals[k] * x[cols[k]];
        }

        return acc;
    }

    template <int RowBlockSize, int ColBlockSize, int ScaleMode, typename R, typename C, typename TStorage, typename T = TStorage>
    void bsr_soa_spmv_static(const ptrdiff_t                              block_rows,
                             const ptrdiff_t                              block_cols,
                             const R* const SFEM_RESTRICT                 rowptr,
                             const C* const SFEM_RESTRICT                 colidx,
                             TStorage* const* const SFEM_RESTRICT         values,
                             const T                                      scale_output,
                             const T* const SFEM_RESTRICT                 x,
                             T* const SFEM_RESTRICT                       y) {
        static_assert(RowBlockSize > 0, "RowBlockSize must be positive");
        static_assert(ColBlockSize > 0, "ColBlockSize must be positive");

#pragma omp parallel for collapse(2) schedule(static)
        for (int d1 = 0; d1 < RowBlockSize; d1++) {
            for (ptrdiff_t i = 0; i < block_rows; i++) {
                const R                     row_begin = rowptr[i];
                const R                     extent    = rowptr[i + 1] - row_begin;
                const C* const SFEM_RESTRICT cols      = &colidx[row_begin];
                T* const SFEM_RESTRICT       y_comp    = &y[d1 * block_rows];

                T val;
                if constexpr (ScaleMode == 0) {
                    val = 0;
                } else if constexpr (ScaleMode == 1) {
                    val = y_comp[i];
                } else {
                    val = scale_output * y_comp[i];
                }

#pragma unroll(ColBlockSize)
                for (int d2 = 0; d2 < ColBlockSize; d2++) {
                    const int                              bb     = d1 * ColBlockSize + d2;
                    const TStorage* const SFEM_RESTRICT    vals   = &values[bb][row_begin];
                    const T* const SFEM_RESTRICT           x_comp = &x[d2 * block_cols];
                    val += bsr_soa_dot<R, C, TStorage, T>(extent, cols, vals, x_comp);
                }

                y_comp[i] = val;
            }
        }
    }

    template <int ScaleMode, typename R, typename C, typename TStorage, typename T = TStorage>
    void bsr_soa_spmv_dynamic(const ptrdiff_t                              block_rows,
                              const ptrdiff_t                              block_cols,
                              const int                                    row_block_size,
                              const int                                    col_block_size,
                              const R* const SFEM_RESTRICT                 rowptr,
                              const C* const SFEM_RESTRICT                 colidx,
                              TStorage* const* const SFEM_RESTRICT         values,
                              const T                                      scale_output,
                              const T* const SFEM_RESTRICT                 x,
                              T* const SFEM_RESTRICT                       y) {
#pragma omp parallel for collapse(2) schedule(static)
        for (int d1 = 0; d1 < row_block_size; d1++) {
            for (ptrdiff_t i = 0; i < block_rows; i++) {
                const R                      row_begin = rowptr[i];
                const R                      extent    = rowptr[i + 1] - row_begin;
                const C* const SFEM_RESTRICT cols      = &colidx[row_begin];
                T* const SFEM_RESTRICT       y_comp    = &y[d1 * block_rows];

                T val;
                if constexpr (ScaleMode == 0) {
                    val = 0;
                } else if constexpr (ScaleMode == 1) {
                    val = y_comp[i];
                } else {
                    val = scale_output * y_comp[i];
                }

                for (int d2 = 0; d2 < col_block_size; d2++) {
                    const int                           bb     = d1 * col_block_size + d2;
                    const TStorage* const SFEM_RESTRICT vals   = &values[bb][row_begin];
                    const T* const SFEM_RESTRICT        x_comp = &x[d2 * block_cols];
                    val += bsr_soa_dot<R, C, TStorage, T>(extent, cols, vals, x_comp);
                }

                y_comp[i] = val;
            }
        }
    }

    template <int ScaleMode, typename R, typename C, typename TStorage, typename T = TStorage>
    void bsr_soa_spmv_dispatch(const ptrdiff_t                              block_rows,
                               const ptrdiff_t                              block_cols,
                               const int                                    row_block_size,
                               const int                                    col_block_size,
                               const R* const SFEM_RESTRICT                 rowptr,
                               const C* const SFEM_RESTRICT                 colidx,
                               TStorage* const* const SFEM_RESTRICT         values,
                               const T                                      scale_output,
                               const T* const SFEM_RESTRICT                 x,
                               T* const SFEM_RESTRICT                       y) {
        if (row_block_size == 3 && col_block_size == 3) {
            bsr_soa_spmv_static<3, 3, ScaleMode>(block_rows, block_cols, rowptr, colidx, values, scale_output, x, y);
        } else if (row_block_size == 6 && col_block_size == 6) {
            bsr_soa_spmv_static<6, 6, ScaleMode>(block_rows, block_cols, rowptr, colidx, values, scale_output, x, y);
        } else if (row_block_size == 3 && col_block_size == 6) {
            bsr_soa_spmv_static<3, 6, ScaleMode>(block_rows, block_cols, rowptr, colidx, values, scale_output, x, y);
        } else if (row_block_size == 6 && col_block_size == 3) {
            bsr_soa_spmv_static<6, 3, ScaleMode>(block_rows, block_cols, rowptr, colidx, values, scale_output, x, y);
        } else {
            bsr_soa_spmv_dynamic<ScaleMode>(
                    block_rows, block_cols, row_block_size, col_block_size, rowptr, colidx, values, scale_output, x, y);
        }
    }

    template <typename R, typename C, typename TStorage, typename T = TStorage>
    void bsr_soa_spmv(const ptrdiff_t                              block_rows,
                      const ptrdiff_t                              block_cols,
                      const int                                    row_block_size,
                      const int                                    col_block_size,
                      const R* const SFEM_RESTRICT                 rowptr,
                      const C* const SFEM_RESTRICT                 colidx,
                      TStorage* const* const SFEM_RESTRICT         values,
                      const T                                      scale_output,
                      const T* const SFEM_RESTRICT                 x,
                      T* const SFEM_RESTRICT                       y) {
        if (scale_output == 0) {
            bsr_soa_spmv_dispatch<0>(block_rows, block_cols, row_block_size, col_block_size, rowptr, colidx, values, scale_output, x, y);
        } else if (scale_output == 1) {
            bsr_soa_spmv_dispatch<1>(block_rows, block_cols, row_block_size, col_block_size, rowptr, colidx, values, scale_output, x, y);
        } else {
            bsr_soa_spmv_dispatch<2>(block_rows, block_cols, row_block_size, col_block_size, rowptr, colidx, values, scale_output, x, y);
        }
    }

    template <typename R, typename C, typename TStorage, typename T = TStorage>
    void bsr_soa_spmv(const ptrdiff_t                              block_rows,
                      const ptrdiff_t                              block_cols,
                      const int                                    block_size,
                      const R* const SFEM_RESTRICT                 rowptr,
                      const C* const SFEM_RESTRICT                 colidx,
                      TStorage* const* const SFEM_RESTRICT         values,
                      const T                                      scale_output,
                      const T* const SFEM_RESTRICT                 x,
                      T* const SFEM_RESTRICT                       y) {
        bsr_soa_spmv(block_rows, block_cols, block_size, block_size, rowptr, colidx, values, scale_output, x, y);
    }

    template <typename R, typename C, typename TStorage, typename T = TStorage>
    class BSRSoA : public Operator<T> {
    public:
        std::function<void(const T* const, T* const)> apply_;

        int apply(const T* const x, T* const y) override {
            SFEM_TRACE_SCOPE("BSRSoA::apply");

            apply_(x, y);
            return 0;
        }

        std::ptrdiff_t rows() const override { return row_block_size_ * block_rows_; }
        std::ptrdiff_t cols() const override { return col_block_size_ * block_cols_; }

        size_t     nbytes() const { return row_ptr->nbytes() + col_idx->nbytes() + values->nbytes(); }
        inline int row_block_size() const { return row_block_size_; }
        inline int col_block_size() const { return col_block_size_; }
        inline int block_size() const {
            assert(row_block_size_ == col_block_size_);
            return row_block_size_;
        }

        SharedBuffer<R>         row_ptr;
        SharedBuffer<C>         col_idx;
        SharedBuffer<TStorage*> values;

        ptrdiff_t block_rows_{0};
        ptrdiff_t block_cols_{0};
        int       row_block_size_{0};
        int       col_block_size_{0};
        T         uniform_pre_output_scaling{0};

        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};

        ExecutionSpace execution_space() const override { return execution_space_; }

        void print(std::ostream& os = std::cout) const {
            os << "BSRSoA" << std::endl;

            os << "row_block_size: " << row_block_size_ << std::endl;
            os << "col_block_size: " << col_block_size_ << std::endl;
            os << "block_rows: " << block_rows_ << std::endl;
            os << "block_cols: " << block_cols_ << std::endl;

            TStorage* const* const vals = values->data();
            for (ptrdiff_t i = 0; i < block_rows_; i++) {
                for (ptrdiff_t j = row_ptr->data()[i]; j < row_ptr->data()[i + 1]; j++) {
                    const C col = col_idx->data()[j];
                    os << "(" << i << ", " << col << "):\n";
                    for (int d1 = 0; d1 < row_block_size_; d1++) {
                        for (int d2 = 0; d2 < col_block_size_; d2++) {
                            os << vals[d1 * col_block_size_ + d2][j] << " ";
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
    std::shared_ptr<BSRSoA<R, C, TStorage, T>> h_bsr_soa_spmv(const ptrdiff_t                block_rows,
                                                              const ptrdiff_t                block_cols,
                                                              const int                      row_block_size,
                                                              const int                      col_block_size,
                                                              const SharedBuffer<R>&         rowptr,
                                                              const SharedBuffer<C>&         colidx,
                                                              const SharedBuffer<TStorage*>& values,
                                                              const T                        scale_output) {
        auto ret                        = std::make_shared<BSRSoA<R, C, TStorage, T>>();
        ret->row_ptr                    = rowptr;
        ret->col_idx                    = colidx;
        ret->values                     = values;
        ret->block_rows_                = block_rows;
        ret->block_cols_                = block_cols;
        ret->row_block_size_            = row_block_size;
        ret->col_block_size_            = col_block_size;
        ret->uniform_pre_output_scaling = scale_output;

        ret->execution_space_ = EXECUTION_SPACE_HOST;

        ret->apply_ = [=](const T* const x, T* const y) {
            bsr_soa_spmv(block_rows,
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
    std::shared_ptr<BSRSoA<R, C, TStorage, T>> h_bsr_soa_spmv(const ptrdiff_t                block_rows,
                                                              const ptrdiff_t                block_cols,
                                                              const int                      block_size,
                                                              const SharedBuffer<R>&         rowptr,
                                                              const SharedBuffer<C>&         colidx,
                                                              const SharedBuffer<TStorage*>& values,
                                                              const T                        scale_output) {
        return h_bsr_soa_spmv(block_rows, block_cols, block_size, block_size, rowptr, colidx, values, scale_output);
    }

}  // namespace sfem

#endif
