#ifndef SFEM_BSR_SPMV_HPP
#define SFEM_BSR_SPMV_HPP

#include <cstddef>
#include <memory>

#include "sfem_Buffer.hpp"
#include "sfem_MatrixFreeLinearSolver.hpp"

namespace sfem {

    template <typename R, typename C, typename T>
    class BSRSpMV : public Operator<T> {
    public:
        std::function<void(const T* const, T* const)> apply_;

        int apply(const T* const x, T* const y) override {
            apply_(x, y);
            return 0;
        }

        std::ptrdiff_t rows() const override { return block_size_ * (row_ptr->size() - 1); }
        std::ptrdiff_t cols() const override { return block_size_ * block_cols_; }
        inline int block_size() const { return block_size_; }

        std::shared_ptr<Buffer<R>> row_ptr;
        std::shared_ptr<Buffer<C>> col_idx;
        std::shared_ptr<Buffer<T>> values;

        int block_size_{0};
        ptrdiff_t block_cols_{0};

        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};

        ExecutionSpace execution_space() const override { return execution_space_; }


        void print(std::ostream& os) const  {
            os << "BSRSpMV" << std::endl;

            os << "block_size: " << block_size_ << std::endl;
            os << "block_cols: " << block_cols_ << std::endl;   

            const ptrdiff_t n = (row_ptr->size() - 1);
            for (ptrdiff_t i = 0; i < n; i++) {
                for (ptrdiff_t j = row_ptr->data()[i]; j < row_ptr->data()[i + 1]; j++) {
                    const auto *const block = &values->data()[j * block_size_ * block_size_];
                    idx_t col = col_idx->data()[j];
                    os << "(" << i << ", " << col << "): ";
                    os << "\n";
                    for (int d1 = 0; d1 < block_size_; d1++) {
                        for (int d2 = 0; d2 < block_size_; d2++) {
                            os << block[d1 * block_size_ + d2] << " ";
                        }
                        os << "\n";
                    }
                    os << "\n";
                }
            }

            os << std::endl;
        }
    };

    template <typename R, typename C, typename T>
    std::shared_ptr<BSRSpMV<R, C, T>> h_bsr_spmv(const ptrdiff_t block_rows,
                                                 const ptrdiff_t block_cols,
                                                 const int block_size,
                                                 const std::shared_ptr<Buffer<R>>& rowptr,
                                                 const std::shared_ptr<Buffer<C>>& colidx,
                                                 const std::shared_ptr<Buffer<T>>& values,
                                                 const T scale_output) {
        auto ret = std::make_shared<BSRSpMV<R, C, T>>();
        ret->row_ptr = rowptr;
        ret->col_idx = colidx;
        ret->values = values;
        ret->block_cols_ = block_cols;
        ret->block_size_ = block_size;

        ret->execution_space_ = EXECUTION_SPACE_HOST;

        ret->apply_ = [=](const T* const x, T* const y) {
            auto rowptr_ = ret->row_ptr->data();
            auto colidx_ = ret->col_idx->data();
            auto values_ = ret->values->data();

            const int block_size = ret->block_size();
            const int block_matrix_size = block_size * block_size;

            if (scale_output != 1) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < block_rows * block_size; i++) {
                    y[i] *= scale_output;
                }
            }

            if (block_size == 3) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < block_rows; i++) {
                    const R row_begin = rowptr_[i];
                    const R row_end = rowptr_[i + 1];
                    T* const block_y = &y[i * 3];

                    for (R k = row_begin; k < row_end; k++) {
                        const C j = colidx_[k];

                        const T* const block_x = &x[j * 3];
                        const T* const aij = &values_[k * block_matrix_size];
#pragma unroll(3)
                        for (int d1 = 0; d1 < 3; d1++) {
#pragma unroll(3)
                            for (int d2 = 0; d2 < 3; d2++) {
                                block_y[d1] += aij[d1 * 3 + d2] * block_x[d2];
                            }
                        }
                    }
                }

            } else {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < block_rows; i++) {
                    const R row_begin = rowptr_[i];
                    const R row_end = rowptr_[i + 1];
                    T* const block_y = &y[i * block_size];

                    for (R k = row_begin; k < row_end; k++) {
                        const C j = colidx_[k];

                        const T* const block_x = &x[j * block_size];
                        const T* const aij = &values_[k * block_matrix_size];

#pragma unroll
                        for (int d1 = 0; d1 < block_size; d1++) {
#pragma unroll
                            for (int d2 = 0; d2 < block_size; d2++) {
                                block_y[d1] += aij[d1 * block_size + d2] * block_x[d2];
                            }
                        }
                    }
                }
            }
        };

        // ret->print(std::cout);

        return ret;
    }

}  // namespace sfem

#endif  // SFEM_BSR_SPMV_HPP
