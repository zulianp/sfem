#ifndef SFEM_CRS_SPMV_HPP
#define SFEM_CRS_SPMV_HPP

#include <cstddef>
#include <memory>

#include "sfem_Buffer.hpp"
#include "sfem_MatrixFreeLinearSolver.hpp"

#include "sfem_Tracer.hpp"

namespace sfem {

    template <typename R, typename C, typename T>
    class CRSSpMV : public Operator<T> {
    public:
        std::function<void(const T* const, T* const)> apply_;

        int apply(const T* const x, T* const y) override {
            SFEM_TRACE_SCOPE("CRSSpMV::apply");

            apply_(x, y);
            return 0;
        }
        std::ptrdiff_t rows() const override { return row_ptr->size() - 1; }
        std::ptrdiff_t cols() const override { return cols_; }

        SharedBuffer<R> row_ptr;
        SharedBuffer<C> col_idx;
        SharedBuffer<T> values;
        ptrdiff_t       cols_{0};

        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};

        ExecutionSpace execution_space() const override { return execution_space_; }
    };

    template <typename R, typename C, typename T>
    std::shared_ptr<CRSSpMV<R, C, T>> h_crs_spmv(const ptrdiff_t        rows,
                                                 const ptrdiff_t        cols,
                                                 const SharedBuffer<R>& rowptr,
                                                 const SharedBuffer<C>& colidx,
                                                 const SharedBuffer<T>& values,
                                                 const T                scale_output) {
        auto ret     = std::make_shared<CRSSpMV<R, C, T>>();
        ret->row_ptr = rowptr;
        ret->col_idx = colidx;
        ret->values  = values;
        ret->cols_   = cols;

        ret->execution_space_ = EXECUTION_SPACE_HOST;

        ret->apply_ = [=](const T* const x, T* const y) {
            auto rowptr_ = ret->row_ptr->data();
            auto colidx_ = ret->col_idx->data();
            auto values_ = ret->values->data();

            if (scale_output == 0) {

#pragma omp parallel for  // nowait
                for (ptrdiff_t i = 0; i < rows; i++) {
                    const R row_begin = rowptr_[i];
                    const R row_end   = rowptr_[i + 1];

                    T val = 0;
                    for (R k = row_begin; k < row_end; k++) {
                        const C j   = colidx_[k];
                        const T aij = values_[k];

                        val += aij * x[j];
                    }

                    y[i] = val;
                }
            } else if (scale_output == 1) {
#if 0
#pragma omp parallel for  // nowait
                for (ptrdiff_t i = 0; i < rows; i++) {
                    const R row_begin = rowptr_[i];
                    const R row_end   = rowptr_[i + 1];

                    T val = y[i];
                    for (R k = row_begin; k < row_end; k++) {
                        const C j   = colidx_[k];
                        const T aij = values_[k];

                        val += aij * x[j];
                    }

                    y[i] = val;
                }
#else // 20-27% faster on M1
#pragma omp parallel for  // nowait
                for (ptrdiff_t i = 0; i < rows; i++) {
                    const R row_begin = rowptr_[i];
                    const R extent    = rowptr_[i + 1] - row_begin;

                    const auto* const SFEM_RESTRICT cols = &colidx_[row_begin];
                    const auto* const SFEM_RESTRICT vals = &values_[row_begin];

                    T val = y[i];

                    const static int BLOCK_SIZE       = 8;
                    const R          n_blocks         = extent / BLOCK_SIZE;
                    const R          b_extent         = n_blocks * BLOCK_SIZE;
                    T                buff[BLOCK_SIZE] = {0};

                    for (R k = 0; k < b_extent; k += BLOCK_SIZE) {
#pragma unroll(BLOCK_SIZE)
                        for (int b = 0; b < BLOCK_SIZE; b++) {
                            buff[b] += vals[k + b] * x[cols[k + b]];
                        }
                    }

                    if (b_extent) {
                        for (int b = 0; b < BLOCK_SIZE; b++) {
                            val += buff[b];
                        }
                    }

                    for (R k = b_extent; k < extent; k++) {
                        const C j   = cols[k];
                        const T aij = vals[k];

                        val += aij * x[j];
                    }

                    y[i] = val;
                }
#endif
            } else {
#pragma omp parallel for  // nowait
                for (ptrdiff_t i = 0; i < rows; i++) {
                    const R row_begin = rowptr_[i];
                    const R row_end   = rowptr_[i + 1];

                    T val = scale_output * y[i];
                    for (R k = row_begin; k < row_end; k++) {
                        const C j   = colidx_[k];
                        const T aij = values_[k];

                        val += aij * x[j];
                    }

                    y[i] = val;
                }
            }
        };

        return ret;
    }

}  // namespace sfem

#endif  // SFEM_CRS_SPMV_HPP
