#ifndef SFEM_DIA_HPP
#define SFEM_DIA_HPP

#include <cstddef>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "sfem_MatrixFreeLinearSolver.hpp"
#include "sfem_aliases.hpp"

namespace sfem {

    template <typename TStorage, typename T = TStorage>
    void dia_spmv(const ptrdiff_t                 block_rows,
                  const ptrdiff_t                 block_cols,
                  const int                       block_size,
                  const int *const SFEM_RESTRICT  diagonal_offsets,
                  const ptrdiff_t                 ndiag,
                  const TStorage *const SFEM_RESTRICT values,
                  const T                         scale_output,
                  const T *const SFEM_RESTRICT    x,
                  T *const SFEM_RESTRICT          y) {
        const ptrdiff_t rows = block_rows * block_size;

        if (scale_output == 0) {
#pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < rows; ++i) {
                y[i] = 0;
            }
        } else if (scale_output != 1) {
#pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < rows; ++i) {
                y[i] *= scale_output;
            }
        }

        if (block_size == 1) {
#pragma omp parallel for schedule(static)
            for (ptrdiff_t row = 0; row < block_rows; ++row) {
                T acc = 0;
                for (ptrdiff_t diagonal = 0; diagonal < ndiag; ++diagonal) {
                    const ptrdiff_t offset = diagonal_offsets[diagonal];
                    const ptrdiff_t col = row + offset;
                    if (col >= 0 && col < block_cols) {
                        acc += values[diagonal * block_rows + row] * x[col];
                    }
                }
                y[row] += acc;
            }
        } else {
            const ptrdiff_t block_matrix_size = ptrdiff_t(block_size) * block_size;
#pragma omp parallel for schedule(static)
            for (ptrdiff_t block_i = 0; block_i < block_rows; ++block_i) {
                for (int bi = 0; bi < block_size; ++bi) {
                    T acc = 0;
                    for (ptrdiff_t diagonal = 0; diagonal < ndiag; ++diagonal) {
                        const ptrdiff_t offset = diagonal_offsets[diagonal];
                        const ptrdiff_t block_j = block_i + offset;
                        if (block_j >= 0 && block_j < block_cols) {
                            const TStorage *const SFEM_RESTRICT block =
                                    &values[(diagonal * block_rows + block_i) * block_matrix_size];
                            const T *const SFEM_RESTRICT block_x = &x[block_j * block_size];
                            const TStorage *const SFEM_RESTRICT block_row = &block[bi * block_size];
                            for (int bj = 0; bj < block_size; ++bj) {
                                acc += block_row[bj] * block_x[bj];
                            }
                        }
                    }
                    y[block_i * block_size + bi] += acc;
                }
            }
        }
    }

    template <typename TStorage, typename T = TStorage>
    class DIA final : public Operator<T> {
    public:
        int apply(const T *const x, T *const y) override {
            SFEM_TRACE_SCOPE("DIA::apply");
            dia_spmv(block_rows_,
                     block_cols_,
                     block_size_,
                     diagonal_offsets->data(),
                     diagonal_offsets->size(),
                     values->data(),
                     uniform_pre_output_scaling,
                     x,
                     y);
            return SFEM_SUCCESS;
        }

        std::ptrdiff_t rows() const override { return block_rows_ * block_size_; }
        std::ptrdiff_t cols() const override { return block_cols_ * block_size_; }

        ExecutionSpace execution_space() const override { return execution_space_; }

        SharedBuffer<int>      diagonal_offsets;
        SharedBuffer<TStorage> values;

        ptrdiff_t      block_rows_{0};
        ptrdiff_t      block_cols_{0};
        int            block_size_{1};
        T              uniform_pre_output_scaling{0};
        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};
    };

    template <typename TStorage, typename T = TStorage>
    std::shared_ptr<DIA<TStorage, T>> h_dia_spmv(const ptrdiff_t                 block_rows,
                                                 const ptrdiff_t                 block_cols,
                                                 const int                       block_size,
                                                 const SharedBuffer<int>        &diagonal_offsets,
                                                 const SharedBuffer<TStorage>   &values,
                                                 const T                         uniform_pre_output_scaling) {
        auto ret = std::make_shared<DIA<TStorage, T>>();
        ret->block_rows_ = block_rows;
        ret->block_cols_ = block_cols;
        ret->block_size_ = block_size;
        ret->diagonal_offsets = diagonal_offsets;
        ret->values = values;
        ret->uniform_pre_output_scaling = uniform_pre_output_scaling;
        ret->execution_space_ = EXECUTION_SPACE_HOST;
        return ret;
    }

}  // namespace sfem

#endif  // SFEM_DIA_HPP
