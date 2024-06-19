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
                                                 R* row_ptr,
                                                 C* col_idx,
                                                 T* values,
                                                 const T scale_output) {
        auto ret = std::make_shared<CRSSpMV<R, C, T>>();
        ret->row_ptr = Buffer<R>::wrap(rows + 1, row_ptr, MEMORY_SPACE_HOST);
        ret->col_idx = Buffer<C>::wrap(row_ptr[rows], col_idx, MEMORY_SPACE_HOST);
        ret->values = Buffer<T>::wrap(row_ptr[rows], values, MEMORY_SPACE_HOST);
        ret->cols_ = cols;

        ret->apply_ = [=](const T* const x, T* const y) {
            if (scale_output == 0) {

#pragma omp parallel for  // nowait
                for (ptrdiff_t i = 0; i < rows; i++) {
                    const R row_begin = row_ptr[i];
                    const R row_end = row_ptr[i + 1];

                    real_t val = 0;
                    for (R k = row_begin; k < row_end; k++) {
                        const C j = col_idx[k];
                        const T aij = values[k];

                        val += aij * x[j];
                    }

                    y[i] = val;
                }
            } else if (scale_output == 1) {
#pragma omp parallel for  // nowait
                for (ptrdiff_t i = 0; i < rows; i++) {
                    const R row_begin = row_ptr[i];
                    const R row_end = row_ptr[i + 1];

                    real_t val = 0;
                    for (R k = row_begin; k < row_end; k++) {
                        const C j = col_idx[k];
                        const T aij = values[k];

                        val += aij * x[j];
                    }

                    y[i] += val;
                }
            } else {
#pragma omp parallel for  // nowait
                for (ptrdiff_t i = 0; i < rows; i++) {
                    const R row_begin = row_ptr[i];
                    const R row_end = row_ptr[i + 1];

                    real_t val = 0;
                    for (R k = row_begin; k < row_end; k++) {
                        const C j = col_idx[k];
                        const T aij = values[k];

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
