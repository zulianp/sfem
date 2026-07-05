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
#include "smesh_alloc.hpp"
#include "smesh_types.hpp"

namespace sfem {

    template <typename R, typename TStorage, typename T = TStorage>
    static inline T bsr_soa_dot(const R extent, const TStorage* const SFEM_RESTRICT vals, const T* const SFEM_RESTRICT x) {
        T acc = 0;

        const static int BLOCK_SIZE       = 16;
        const R          n_blocks         = extent / BLOCK_SIZE;
        const R          b_extent         = n_blocks * BLOCK_SIZE;
        T                buff[BLOCK_SIZE] = {0};

        for (R k = 0; k < b_extent; k += BLOCK_SIZE) {
            auto* v  = &vals[k];
            auto* xx = &x[k];
#pragma omp simd
            for (int b = 0; b < BLOCK_SIZE; b++) {
                buff[b] += v[b] * xx[b];
            }
        }

        if (b_extent) {
#pragma omp simd reduction(+ : acc)
            for (int b = 0; b < BLOCK_SIZE; b++) {
                acc += buff[b];
            }
        }

        for (R k = b_extent; k < extent; k++) {
            const T aij = vals[k];

            acc += aij * x[k];
        }

        return acc;
    }

    template <typename T>
    void bsr_soa_scale_output(const ptrdiff_t rows, const T scale_output, T* const SFEM_RESTRICT y) {
        if (scale_output == 0) {
#pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < rows; i++) {
                y[i] = 0;
            }
        } else if (scale_output != 1) {
#pragma omp      parallel for schedule(static)
            for (ptrdiff_t i = 0; i < rows; i++) {
                y[i] *= scale_output;
            }
        }
    }

    template <int RowBlockSize, int ColBlockSize, typename R, typename C, typename TStorage, typename T = TStorage>
    void bsr_soa_spmv_static(const ptrdiff_t                      block_rows,
                                  const ptrdiff_t                      block_cols,
                                  const R* const SFEM_RESTRICT         rowptr,
                                  const C* const SFEM_RESTRICT         colidx,
                                  TStorage* const* const SFEM_RESTRICT values,
                                  const T* const SFEM_RESTRICT         x,
                                  T* const* const SFEM_RESTRICT        x_workspace,
                                  T* const SFEM_RESTRICT               y) {
        static_assert(RowBlockSize > 0, "RowBlockSize must be positive");
        static_assert(ColBlockSize > 0, "ColBlockSize must be positive");
        (void)block_cols;

#pragma omp parallel
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            T* const SFEM_RESTRICT x_row = x_workspace[tid];

#pragma omp for schedule(static)
            for (ptrdiff_t i = 0; i < block_rows; i++) {
                const R row_begin = rowptr[i];
                const R row_end   = rowptr[i + 1];
                const R row_nnz   = row_end - row_begin;

                for (R k = 0; k < row_nnz; k++) {
                    const C                col     = colidx[row_begin + k];
                    T* const SFEM_RESTRICT x_block = &x_row[k * ColBlockSize];
#pragma unroll(ColBlockSize)
                    for (int d2 = 0; d2 < ColBlockSize; d2++) {
                        x_block[d2] = x[col * ColBlockSize + d2];
                    }
                }

                const R scalar_extent = row_nnz * ColBlockSize;

#pragma unroll(RowBlockSize)
                for (int d1 = 0; d1 < RowBlockSize; d1++) {
                    T* const SFEM_RESTRICT              y_comp = &y[d1 * block_rows];
                    const TStorage* const SFEM_RESTRICT vals   = &values[d1][row_begin * ColBlockSize];
                    T                                   val    = y_comp[i];

                    val += bsr_soa_dot<R, TStorage, T>(scalar_extent, vals, x_row);

                    y_comp[i] = val;
                }
            }
        }
    }

    template <typename R, typename C, typename TStorage, typename T = TStorage>
    void bsr_soa_spmv_dynamic(const ptrdiff_t                      block_rows,
                              const ptrdiff_t                      block_cols,
                              const int                            row_block_size,
                              const int                            col_block_size,
                              const R* const SFEM_RESTRICT         rowptr,
                              const C* const SFEM_RESTRICT         colidx,
                              TStorage* const* const SFEM_RESTRICT values,
                              const T* const SFEM_RESTRICT         x,
                              T* const* const SFEM_RESTRICT        x_workspace,
                              T* const SFEM_RESTRICT               y) {
        (void)block_cols;

#pragma omp parallel
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            T* const SFEM_RESTRICT x_row = x_workspace[tid];

#pragma omp for schedule(static)
            for (ptrdiff_t i = 0; i < block_rows; i++) {
                const R row_begin = rowptr[i];
                const R row_end   = rowptr[i + 1];
                const R row_nnz   = row_end - row_begin;

                const C* const SFEM_RESTRICT cols = &colidx[row_begin];

                for (R k = 0; k < row_nnz; k++) {
                    T* const SFEM_RESTRICT       x_block = &x_row[k * col_block_size];
                    const T* const SFEM_RESTRICT xx      = &x[cols[k] * col_block_size];
                    for (int d2 = 0; d2 < col_block_size; d2++) {
                        x_block[d2] = xx[d2];
                    }
                }
                const R scalar_extent = row_nnz * col_block_size;

                for (int d1 = 0; d1 < row_block_size; d1++) {
                    T* const SFEM_RESTRICT              y_comp = &y[d1 * block_rows];
                    const TStorage* const SFEM_RESTRICT vals   = &values[d1][row_begin * col_block_size];
                    const T                             val    = bsr_soa_dot<R, TStorage, T>(scalar_extent, vals, x_row);

                    y_comp[i] += val;
                }
            }
        }
    }

    template <typename R, typename C, typename TStorage, typename T = TStorage>
    void bsr_soa_spmv_dispatch(const ptrdiff_t                      block_rows,
                               const ptrdiff_t                      block_cols,
                               const int                            row_block_size,
                               const int                            col_block_size,
                               const R* const SFEM_RESTRICT         rowptr,
                               const C* const SFEM_RESTRICT         colidx,
                               TStorage* const* const SFEM_RESTRICT values,
                               const T* const SFEM_RESTRICT         x,
                               T* const* const SFEM_RESTRICT        x_workspace,
                               T* const SFEM_RESTRICT               y) {
        if (row_block_size == 3 && col_block_size == 3) {
            bsr_soa_spmv_static<3, 3>(block_rows, block_cols, rowptr, colidx, values, x, x_workspace, y);
        } else if (row_block_size == 6 && col_block_size == 6) {
            bsr_soa_spmv_static<6, 6>(block_rows, block_cols, rowptr, colidx, values, x, x_workspace, y);
        } else if (row_block_size == 3 && col_block_size == 6) {
            bsr_soa_spmv_static<3, 6>(block_rows, block_cols, rowptr, colidx, values, x, x_workspace, y);
        } else if (row_block_size == 6 && col_block_size == 3) {
            bsr_soa_spmv_static<6, 3>(block_rows, block_cols, rowptr, colidx, values, x, x_workspace, y);
        } else {
            bsr_soa_spmv_dynamic(
                    block_rows, block_cols, row_block_size, col_block_size, rowptr, colidx, values, x, x_workspace, y);
        }
    }

    template <typename R, typename C, typename TStorage, typename T = TStorage>
    void bsr_soa_spmv(const ptrdiff_t                      block_rows,
                      const ptrdiff_t                      block_cols,
                      const int                            row_block_size,
                      const int                            col_block_size,
                      const R* const SFEM_RESTRICT         rowptr,
                      const C* const SFEM_RESTRICT         colidx,
                      TStorage* const* const SFEM_RESTRICT values,
                      const T                              scale_output,
                      const T* const SFEM_RESTRICT         x,
                      T* const* const SFEM_RESTRICT        x_workspace,
                      T* const SFEM_RESTRICT               y) {
        bsr_soa_scale_output(block_rows * row_block_size, scale_output, y);
        bsr_soa_spmv_dispatch(block_rows, block_cols, row_block_size, col_block_size, rowptr, colidx, values, x, x_workspace, y);
    }

    template <typename R, typename C, typename TStorage, typename T = TStorage>
    void bsr_soa_spmv(const ptrdiff_t                      block_rows,
                      const ptrdiff_t                      block_cols,
                      const int                            block_size,
                      const R* const SFEM_RESTRICT         rowptr,
                      const C* const SFEM_RESTRICT         colidx,
                      TStorage* const* const SFEM_RESTRICT values,
                      const T                              scale_output,
                      const T* const SFEM_RESTRICT         x,
                      T* const* const SFEM_RESTRICT        x_workspace,
                      T* const SFEM_RESTRICT               y) {
        bsr_soa_spmv(block_rows, block_cols, block_size, block_size, rowptr, colidx, values, scale_output, x, x_workspace, y);
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
        SharedBuffer<T*>        x_workspace;

        ptrdiff_t block_rows_{0};
        ptrdiff_t block_cols_{0};
        ptrdiff_t x_workspace_stride_{0};
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
                            os << vals[d1][j * col_block_size_ + d2] << " ";
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
        const ptrdiff_t nnz = rowptr->data()[block_rows];
        assert(values->extent(0) == static_cast<size_t>(row_block_size));
        assert(values->extent(1) == static_cast<size_t>(nnz * col_block_size));

        ptrdiff_t max_row_nnz = 0;
        for (ptrdiff_t i = 0; i < block_rows; i++) {
            const ptrdiff_t row_nnz = rowptr->data()[i + 1] - rowptr->data()[i];
            if (row_nnz > max_row_nnz) {
                max_row_nnz = row_nnz;
            }
        }

        int nthreads = 1;
#ifdef _OPENMP
        nthreads = omp_get_max_threads();
#endif

        auto ret                 = std::make_shared<BSRSoA<R, C, TStorage, T>>();
        ret->row_ptr             = rowptr;
        ret->col_idx             = colidx;
        ret->values              = values;
        ret->x_workspace_stride_ = max_row_nnz ? max_row_nnz * col_block_size : col_block_size;

        const size_t workspace_stride = static_cast<size_t>(ret->x_workspace_stride_);
        T** const    x_workspace_data = static_cast<T**>(SMESH_ALLOC(static_cast<size_t>(nthreads) * sizeof(T*)));

#ifdef _OPENMP
#pragma omp parallel num_threads(nthreads)
        {
            const int tid         = omp_get_thread_num();
            x_workspace_data[tid] = static_cast<T*>(SMESH_CALLOC(workspace_stride, sizeof(T)));
        }
#else
        x_workspace_data[0] = static_cast<T*>(SMESH_CALLOC(workspace_stride, sizeof(T)));
#endif

        ret->x_workspace     = manage_host_buffer<T>(static_cast<size_t>(nthreads), workspace_stride, x_workspace_data);
        ret->block_rows_     = block_rows;
        ret->block_cols_     = block_cols;
        ret->row_block_size_ = row_block_size;
        ret->col_block_size_ = col_block_size;
        ret->uniform_pre_output_scaling = scale_output;

        ret->execution_space_ = EXECUTION_SPACE_HOST;

        auto x_workspace = ret->x_workspace;

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
                         x_workspace->data(),
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
