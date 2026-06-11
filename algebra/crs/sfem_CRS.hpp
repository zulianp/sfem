#ifndef SFEM_CRS_SPMV_HPP
#define SFEM_CRS_SPMV_HPP

#include <cstddef>
#include <limits>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "sfem_MatrixFreeLinearSolver.hpp"
#include "sfem_aliases.hpp"
#include "smesh_types.hpp"

// TODO: have a look at this paper for the MM https://dl.acm.org/doi/10.1145/3721145.3725773

namespace sfem {

    template <typename R, typename C, typename T>
    int crs_transpose_sym(const ptrdiff_t              rows,
                          const ptrdiff_t              columns,
                          const R* const SFEM_RESTRICT a_rowptr,
                          const C* const SFEM_RESTRICT a_colidx,
                          const T* const SFEM_RESTRICT a_values,
                          R* const SFEM_RESTRICT       b_rowptr) {
        (void)a_values;

        for (ptrdiff_t i = 0; i <= columns; i++) {
            b_rowptr[i] = 0;
        }

        for (ptrdiff_t i = 0; i < rows; i++) {
            const R                      a_begin = a_rowptr[i];
            const R                      a_end   = a_rowptr[i + 1];
            const C* const SFEM_RESTRICT a_cols  = &a_colidx[a_begin];

            for (R a_k = 0, a_len = a_end - a_begin; a_k < a_len; a_k++) {
                b_rowptr[a_cols[a_k] + 1]++;
            }
        }

        for (ptrdiff_t i = 0; i < columns; i++) {
            b_rowptr[i + 1] += b_rowptr[i];
        }

        return SFEM_SUCCESS;
    }

    template <typename R, typename C, typename T>
    void crs_transpose_apply(const ptrdiff_t              rows,
                             const ptrdiff_t              columns,
                             const R* const SFEM_RESTRICT a_rowptr,
                             const C* const SFEM_RESTRICT a_colidx,
                             const T* const SFEM_RESTRICT a_values,
                             const R* const SFEM_RESTRICT b_rowptr,
                             C* const SFEM_RESTRICT       b_colidx,
                             T* const SFEM_RESTRICT       b_values,
                             R* const SFEM_RESTRICT       next_workspace) {
        for (ptrdiff_t i = 0; i < columns; i++) {
            next_workspace[i] = b_rowptr[i];
        }

        for (ptrdiff_t i = 0; i < rows; i++) {
            const C                      b_j     = static_cast<C>(i);
            const R                      a_begin = a_rowptr[i];
            const R                      a_end   = a_rowptr[i + 1];
            const C* const SFEM_RESTRICT a_cols  = &a_colidx[a_begin];
            const T* const SFEM_RESTRICT a_vals  = &a_values[a_begin];

            for (R a_k = 0, a_len = a_end - a_begin; a_k < a_len; a_k++) {
                const C b_i    = a_cols[a_k];
                const R offset = next_workspace[b_i]++;

                b_colidx[offset] = b_j;
                b_values[offset] = a_vals[a_k];
            }
        }
    }

    template <typename R, typename C>
    void crs_mm_sym(const ptrdiff_t              rows,
                    const ptrdiff_t              c_columns,
                    const R* const SFEM_RESTRICT a_rowptr,
                    const C* const SFEM_RESTRICT a_colidx,
                    const R* const SFEM_RESTRICT b_rowptr,
                    const C* const SFEM_RESTRICT b_colidx,
                    R* const SFEM_RESTRICT       c_rowptr,
                    R* const SFEM_RESTRICT       mask_workspace,
                    const int                    n_workspaces) {
        const R unseen = smesh::invalid_idx<R>();

        c_rowptr[0] = 0;

#pragma omp parallel num_threads(n_workspaces)
        {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            R* const SFEM_RESTRICT mask = &mask_workspace[tid * c_columns];

            for (ptrdiff_t i = 0; i < c_columns; i++) {
                mask[i] = unseen;
            }

#pragma omp for schedule(static)
            for (ptrdiff_t i = 0; i < rows; i++) {
                R nnz = 0;

                const R                      a_begin = a_rowptr[i];
                const R                      a_end   = a_rowptr[i + 1];
                const C* const SFEM_RESTRICT a_cols  = &a_colidx[a_begin];

                for (R a_k = 0, a_len = a_end - a_begin; a_k < a_len; a_k++) {
                    const C a_j = a_cols[a_k];

                    const R                      b_begin = b_rowptr[a_j];
                    const R                      b_end   = b_rowptr[a_j + 1];
                    const C* const SFEM_RESTRICT b_cols  = &b_colidx[b_begin];

                    for (R b_k = 0, b_len = b_end - b_begin; b_k < b_len; b_k++) {
                        const C b_j = b_cols[b_k];

                        if (mask[b_j] != i) {
                            mask[b_j] = i;
                            nnz++;
                        }
                    }
                }

                c_rowptr[i + 1] = nnz;
            }
        }

        for (ptrdiff_t c_i = 0; c_i < rows; c_i++) {
            c_rowptr[c_i + 1] += c_rowptr[c_i];
        }
    }

    template <typename R, typename C, typename T>
    void crs_mm_apply(const ptrdiff_t              rows,
                      const ptrdiff_t              c_columns,
                      const R* const SFEM_RESTRICT a_rowptr,
                      const C* const SFEM_RESTRICT a_colidx,
                      const T* const SFEM_RESTRICT a_values,
                      const R* const SFEM_RESTRICT b_rowptr,
                      const C* const SFEM_RESTRICT b_colidx,
                      const T* const SFEM_RESTRICT b_values,
                      const R* const SFEM_RESTRICT c_rowptr,
                      C* const SFEM_RESTRICT       c_colidx,
                      T* const SFEM_RESTRICT       c_values,
                      R* const SFEM_RESTRICT       next_workspace,
                      T* const SFEM_RESTRICT       acc_workspace,
                      const int                    n_workspaces) {
        const R init   = std::numeric_limits<R>::max();
        const R unseen = smesh::invalid_idx<R>();

#pragma omp parallel num_threads(n_workspaces)
        {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            R* const SFEM_RESTRICT next = &next_workspace[tid * c_columns];
            T* const SFEM_RESTRICT acc  = &acc_workspace[tid * c_columns];

            for (ptrdiff_t i = 0; i < c_columns; i++) {
                next[i] = unseen;
            }

#pragma omp for schedule(static)
            for (ptrdiff_t i = 0; i < rows; i++) {
                R head = init;
                R len  = 0;

                const R                      a_begin = a_rowptr[i];
                const R                      a_end   = a_rowptr[i + 1];
                const C* const SFEM_RESTRICT a_cols  = &a_colidx[a_begin];
                const T* const SFEM_RESTRICT a_vals  = &a_values[a_begin];

                for (R a_k = 0, a_len = a_end - a_begin; a_k < a_len; a_k++) {
                    const C a_j = a_cols[a_k];
                    const T aij = a_vals[a_k];

                    const R                      b_begin = b_rowptr[a_j];
                    const R                      b_end   = b_rowptr[a_j + 1];
                    const C* const SFEM_RESTRICT b_cols  = &b_colidx[b_begin];
                    const T* const SFEM_RESTRICT b_vals  = &b_values[b_begin];

                    for (R b_k = 0, b_len = b_end - b_begin; b_k < b_len; b_k++) {
                        const C b_j = b_cols[b_k];
                        const T bij = b_vals[b_k];

                        if (next[b_j] == unseen) {
                            next[b_j] = head;
                            head      = b_j;
                            acc[b_j]  = aij * bij;
                            len++;
                        } else {
                            acc[b_j] += aij * bij;
                        }
                    }
                }

                R offset = c_rowptr[i];
                for (R k = 0; k < len; k++) {
                    c_colidx[offset] = head;
                    c_values[offset] = acc[head];
                    offset++;

                    const R temp = head;
                    head         = next[head];
                    next[temp]   = unseen;
                }
            }
        }
    }

    template <typename R, typename C, typename T>
    int crs_transpose(const ptrdiff_t        columns,
                      const SharedBuffer<R>& a_rowptr,
                      const SharedBuffer<C>& a_colidx,
                      const SharedBuffer<T>& a_values,
                      SharedBuffer<R>&       b_rowptr,
                      SharedBuffer<C>&       b_colidx,
                      SharedBuffer<T>&       b_values) {
        const ptrdiff_t rows = a_rowptr->size() - 1;

        if (b_rowptr->size() != columns + 1) {
            b_rowptr = create_host_buffer<R>(columns + 1);
        }

        const R* const SFEM_RESTRICT d_a_rowptr = a_rowptr->data();
        const C* const SFEM_RESTRICT d_a_colidx = a_colidx->data();
        const T* const SFEM_RESTRICT d_a_values = a_values->data();

        R* const SFEM_RESTRICT d_b_rowptr = b_rowptr->data();

        crs_transpose_sym(rows, columns, d_a_rowptr, d_a_colidx, d_a_values, d_b_rowptr);

        if (b_colidx->size() != d_b_rowptr[columns]) {
            b_colidx = create_host_buffer<C>(d_b_rowptr[columns]);
        }

        if (b_values->size() != d_b_rowptr[columns]) {
            b_values = create_host_buffer<T>(d_b_rowptr[columns]);
        }

        C* const SFEM_RESTRICT d_b_colidx = b_colidx->data();
        T* const SFEM_RESTRICT d_b_values = b_values->data();

        auto next_workspace = create_host_buffer<R>(columns);

        crs_transpose_apply(
                rows, columns, d_a_rowptr, d_a_colidx, d_a_values, d_b_rowptr, d_b_colidx, d_b_values, next_workspace->data());

        return SFEM_SUCCESS;
    }

    template <typename R, typename C, typename T>
    int crs_mm(const ptrdiff_t        c_columns,
               const SharedBuffer<R>& a_rowptr,
               const SharedBuffer<C>& a_colidx,
               const SharedBuffer<T>& a_values,
               const SharedBuffer<R>& b_rowptr,
               const SharedBuffer<C>& b_colidx,
               const SharedBuffer<T>& b_values,
               SharedBuffer<R>&       c_rowptr,
               SharedBuffer<C>&       c_colidx,
               SharedBuffer<T>&       c_values) {
        const ptrdiff_t rows = a_rowptr->size() - 1;

        if (c_rowptr->size() != rows + 1) {
            c_rowptr = create_host_buffer<R>(rows + 1);
        }

        const R* const SFEM_RESTRICT d_a_rowptr = a_rowptr->data();
        const C* const SFEM_RESTRICT d_a_colidx = a_colidx->data();
        const T* const SFEM_RESTRICT d_a_values = a_values->data();
        const R* const SFEM_RESTRICT d_b_rowptr = b_rowptr->data();
        const C* const SFEM_RESTRICT d_b_colidx = b_colidx->data();
        const T* const SFEM_RESTRICT d_b_values = b_values->data();

        R* const SFEM_RESTRICT d_c_rowptr = c_rowptr->data();

#ifdef _OPENMP
        const int n_workspaces = omp_get_max_threads();
#else
        const int n_workspaces = 1;
#endif

        auto mask_workspace = create_host_buffer<R>(n_workspaces * c_columns);

        crs_mm_sym(rows,
                   c_columns,
                   d_a_rowptr,
                   d_a_colidx,
                   d_b_rowptr,
                   d_b_colidx,
                   d_c_rowptr,
                   mask_workspace->data(),
                   n_workspaces);

        // Allocate column indices
        if (c_colidx->size() != d_c_rowptr[rows]) {
            c_colidx = create_host_buffer<C>(d_c_rowptr[rows]);
        }

        if (c_values->size() != d_c_rowptr[rows]) {
            c_values = create_host_buffer<T>(d_c_rowptr[rows]);
        }

        C* const SFEM_RESTRICT d_c_colidx = c_colidx->data();
        T* const SFEM_RESTRICT d_c_values = c_values->data();

        auto next_workspace = create_host_buffer<R>(n_workspaces * c_columns);
        auto acc_workspace  = create_host_buffer<T>(n_workspaces * c_columns);

        crs_mm_apply(rows,
                     c_columns,
                     d_a_rowptr,
                     d_a_colidx,
                     d_a_values,
                     d_b_rowptr,
                     d_b_colidx,
                     d_b_values,
                     d_c_rowptr,
                     d_c_colidx,
                     d_c_values,
                     next_workspace->data(),
                     acc_workspace->data(),
                     n_workspaces);

        return SFEM_SUCCESS;
    }

    template <typename R, typename C, typename TStorage, typename T = TStorage>
    void crs_mv(const ptrdiff_t                     rows,
                const R* const SFEM_RESTRICT        rowptr,
                const C* const SFEM_RESTRICT        colidx,
                const TStorage* const SFEM_RESTRICT values,
                const T* const SFEM_RESTRICT        x,
                T* const SFEM_RESTRICT              y,
                const T                             scale_output) {
        if (scale_output == 0) {
#pragma omp parallel for  // nowait
            for (ptrdiff_t i = 0; i < rows; i++) {
                const R row_begin = rowptr[i];
                const R row_end   = rowptr[i + 1];

                T val = 0;
                for (R k = row_begin; k < row_end; k++) {
                    const C j   = colidx[k];
                    const T aij = values[k];

                    val += aij * x[j];
                }

                y[i] = val;
            }
        } else if (scale_output == 1) {
#if 0
#pragma omp parallel for  // nowait
            for (ptrdiff_t i = 0; i < rows; i++) {
                const R row_begin = rowptr[i];
                const R row_end   = rowptr[i + 1];

                T val = y[i];
                for (R k = row_begin; k < row_end; k++) {
                    const C j   = colidx[k];
                    const T aij = values[k];

                    val += aij * x[j];
                }

                y[i] = val;
            }
#else                     // 20-27% faster on M1
#pragma omp parallel for  // nowait
            for (ptrdiff_t i = 0; i < rows; i++) {
                const R row_begin = rowptr[i];
                const R extent    = rowptr[i + 1] - row_begin;

                const auto* const SFEM_RESTRICT cols = &colidx[row_begin];
                const auto* const SFEM_RESTRICT vals = &values[row_begin];

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
                const R row_begin = rowptr[i];
                const R row_end   = rowptr[i + 1];

                T val = scale_output * y[i];
                for (R k = row_begin; k < row_end; k++) {
                    const C j   = colidx[k];
                    const T aij = values[k];

                    val += aij * x[j];
                }

                y[i] = val;
            }
        }
    }

    template <typename R, typename C, typename TStorage, typename T = TStorage>
    class CRS : public Operator<T> {
    public:
        std::function<void(const T* const, T* const)> apply_;

        std::shared_ptr<CRS<R, C, TStorage, T>> transpose() const {
            if (execution_space() != EXECUTION_SPACE_HOST) {
                // TODO: Implement device version
                SFEM_ERROR("Transpose is not supported for non-host execution space");
                return nullptr;
            }

            auto ret     = std::make_shared<CRS<R, C, TStorage, T>>();
            ret->row_ptr = create_host_buffer<R>(0);
            ret->col_idx = create_host_buffer<C>(0);
            ret->values  = create_host_buffer<TStorage>(0);

            crs_transpose(cols_, row_ptr, col_idx, values, ret->row_ptr, ret->col_idx, ret->values);

            const ptrdiff_t ret_rows        = cols_;
            ret->cols_                      = rows();
            ret->uniform_pre_output_scaling = uniform_pre_output_scaling;
            ret->execution_space_           = EXECUTION_SPACE_HOST;

            ret->apply_ = [=](const T* const x, T* const y) {
                auto rowptr_ = ret->row_ptr->data();
                auto colidx_ = ret->col_idx->data();
                auto values_ = ret->values->data();

                crs_mv(ret_rows, rowptr_, colidx_, values_, x, y, ret->uniform_pre_output_scaling);
            };

            return ret;
        }

        std::shared_ptr<CRS<R, C, TStorage, T>> mm(const std::shared_ptr<CRS<R, C, TStorage, T>>& other) const {
            if (execution_space() != EXECUTION_SPACE_HOST || other->execution_space() != EXECUTION_SPACE_HOST) {
                // TODO: Implement device version
                SFEM_ERROR("Matrix multiplication is not supported for non-host execution space");
                return nullptr;
            }

            auto ret     = std::make_shared<CRS<R, C, TStorage, T>>();
            ret->row_ptr = create_host_buffer<R>(0);
            ret->col_idx = create_host_buffer<C>(0);
            ret->values  = create_host_buffer<TStorage>(0);

            crs_mm(other->cols_,
                   row_ptr,
                   col_idx,
                   values,
                   other->row_ptr,
                   other->col_idx,
                   other->values,
                   ret->row_ptr,
                   ret->col_idx,
                   ret->values);

            const ptrdiff_t ret_rows        = rows();
            ret->cols_                      = other->cols_;
            ret->uniform_pre_output_scaling = 0;
            ret->execution_space_           = EXECUTION_SPACE_HOST;

            ret->apply_ = [=](const T* const x, T* const y) {
                auto rowptr_ = ret->row_ptr->data();
                auto colidx_ = ret->col_idx->data();
                auto values_ = ret->values->data();

                crs_mv(ret_rows, rowptr_, colidx_, values_, x, y, ret->uniform_pre_output_scaling);
            };

            return ret;
        }

        int apply(const T* const x, T* const y) override {
            SFEM_TRACE_SCOPE("CRS::apply");
            apply_(x, y);
            return 0;
        }
        std::ptrdiff_t rows() const override { return row_ptr->size() - 1; }
        std::ptrdiff_t cols() const override { return cols_; }

        size_t nbytes() const { return row_ptr->nbytes() + col_idx->nbytes() + values->nbytes(); }

        SharedBuffer<R>        row_ptr;
        SharedBuffer<C>        col_idx;
        SharedBuffer<TStorage> values;
        ptrdiff_t              cols_{0};
        T                      uniform_pre_output_scaling{0};

        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};

        ExecutionSpace execution_space() const override { return execution_space_; }

        void print(std::ostream& os = std::cout) const {
            if (execution_space_ == EXECUTION_SPACE_HOST) {
                os << "CRS (" << rows() << " rows, " << cols() << " cols)\n";

                const ptrdiff_t nrows = row_ptr->size() - 1;
                for (ptrdiff_t i = 0; i < nrows; i++) {
                    os << i << ") ";
                    for (ptrdiff_t j = row_ptr->data()[i]; j < row_ptr->data()[i + 1]; j++) {
                        os << col_idx->data()[j] << " -> (" << values->data()[j] << "), ";
                    }
                    os << "\n";
                }

                os << "\n";
            }
        }
    };

    // CRS matrix product here: https://github.com/zhen-xie/IA-SpGEMM/blob/master/IA-SPGEMM-CPU_release/detail/csr/common_csr.h
    // https://dl.acm.org/doi/pdf/10.1145/3330345.3330354

    template <typename R, typename C, typename TStorage, typename T = TStorage>
    std::shared_ptr<CRS<R, C, TStorage, T>> h_crs_spmv(const ptrdiff_t               rows,
                                                       const ptrdiff_t               cols,
                                                       const SharedBuffer<R>&        rowptr,
                                                       const SharedBuffer<C>&        colidx,
                                                       const SharedBuffer<TStorage>& values,
                                                       const T                       uniform_pre_output_scaling) {
        auto ret                        = std::make_shared<CRS<R, C, TStorage, T>>();
        ret->row_ptr                    = rowptr;
        ret->col_idx                    = colidx;
        ret->values                     = values;
        ret->cols_                      = cols;
        ret->uniform_pre_output_scaling = uniform_pre_output_scaling;

        ret->execution_space_ = EXECUTION_SPACE_HOST;

        ret->apply_ = [=](const T* const x, T* const y) {
            auto rowptr_ = ret->row_ptr->data();
            auto colidx_ = ret->col_idx->data();
            auto values_ = ret->values->data();

            crs_mv(rows, rowptr_, colidx_, values_, x, y, ret->uniform_pre_output_scaling);
        };

        return ret;
    }

    template <typename R, typename C, typename TStorage, typename T = TStorage>
    std::shared_ptr<CRS<R, C, TStorage, T>> rap(const std::shared_ptr<CRS<R, C, TStorage, T>>& r,
                                                const std::shared_ptr<CRS<R, C, TStorage, T>>& a,
                                                const std::shared_ptr<CRS<R, C, TStorage, T>>& p) {
        // Compute D = A P
        auto d = a->mm(p);

        // Compute G = R D
        auto g = r->mm(d);
        return g;
    }

}  // namespace sfem

#endif  // SFEM_CRS_SPMV_HPP
