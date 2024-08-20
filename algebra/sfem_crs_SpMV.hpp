#ifndef SFEM_CRS_SPMV_HPP
#define SFEM_CRS_SPMV_HPP

#include <cstddef>
#include <memory>

#include "sfem_Buffer.hpp"
#include "sfem_MatrixFreeLinearSolver.hpp"

namespace sfem {

    template <typename R, typename C, typename T>
    class CRSSpMV : public Operator<T> {
    public:
        std::function<void(const T* const, T* const)> apply_;

        int apply(const T* const x, T* const y) override {
            apply_(x, y);
            return 0;
        }
        std::ptrdiff_t rows() const override { return row_ptr->size() - 1; }
        std::ptrdiff_t cols() const override { return cols_; }

        std::shared_ptr<Buffer<R>> row_ptr;
        std::shared_ptr<Buffer<C>> col_idx;
        std::shared_ptr<Buffer<T>> values;
        ptrdiff_t cols_{0};
    };

    template <typename R, typename C, typename T>
    std::shared_ptr<CRSSpMV<R, C, T>> h_crs_spmv(const ptrdiff_t rows,
                                                 const ptrdiff_t cols,
                                                 const std::shared_ptr<Buffer<R>> &rowptr,
                                                 const std::shared_ptr<Buffer<C>> &colidx,
                                                 const std::shared_ptr<Buffer<T>> &values,
                                                 const T scale_output) {
        auto ret = std::make_shared<CRSSpMV<R, C, T>>();
        ret->row_ptr = rowptr;
        ret->col_idx = colidx;
        ret->values = values;
        ret->cols_ = cols;

        ret->apply_ = [=](const T* const x, T* const y) {
            auto rowptr_ = ret->row_ptr->data();
            auto colidx_ = ret->col_idx->data();
            auto values_ = ret->values->data();

            if (scale_output == 0) {

#pragma omp parallel for  // nowait
                for (ptrdiff_t i = 0; i < rows; i++) {
                    const R row_begin = rowptr_[i];
                    const R row_end = rowptr_[i + 1];

                    T val = 0;
                    for (R k = row_begin; k < row_end; k++) {
                        const C j = colidx_[k];
                        const T aij = values_[k];

                        val += aij * x[j];
                    }

                    y[i] = val;
                }
            } else if (scale_output == 1) {
#pragma omp parallel for  // nowait
                for (ptrdiff_t i = 0; i < rows; i++) {
                    const R row_begin = rowptr_[i];
                    const R row_end = rowptr_[i + 1];

                    T val = 0;
                    for (R k = row_begin; k < row_end; k++) {
                        const C j = colidx_[k];
                        const T aij = values_[k];

                        val += aij * x[j];
                    }

                    y[i] += val;
                }
            } else {
#pragma omp parallel for  // nowait
                for (ptrdiff_t i = 0; i < rows; i++) {
                    const R row_begin = rowptr_[i];
                    const R row_end = rowptr_[i + 1];

                    T val = 0;
                    for (R k = row_begin; k < row_end; k++) {
                        const C j = colidx_[k];
                        const T aij = values_[k];

                        val += aij * x[j];
                    }

                    y[i] = scale_output * y[i] + val;
                }
            }
        };

        return ret;
    }

}  // namespace sfem

#endif  // SFEM_CRS_SPMV_HPP
