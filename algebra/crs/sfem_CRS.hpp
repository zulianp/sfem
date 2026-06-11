#ifndef SFEM_CRS_SPMV_HPP
#define SFEM_CRS_SPMV_HPP

#include <cstddef>
#include <limits>
#include <memory>

#include "sfem_MatrixFreeLinearSolver.hpp"
#include "sfem_aliases.hpp"
#include "smesh_types.hpp"

namespace sfem {

    // TODO: Refactor this into two function crs_mm_sym (count and fill the rowptr of C, no inner allocations), and crs_mm_apply
    // where the matrix product is applied. call these functions from crs_mm. crs_mm_sym and crs_mm_apply should have a pure array
    // interface and no inner allocations
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

        d_c_rowptr[0] = 0;

#pragma omp parallel
        {
            auto mask_buff = create_host_buffer<R>(c_columns);
            auto mask      = mask_buff->data();

            for (ptrdiff_t i = 0; i < c_columns; i++) {
                mask[i] = smesh::invalid_idx<R>();
            }

#pragma omp for schedule(static)
            for (ptrdiff_t i = 0; i < rows; i++) {
                const R                      a_len  = d_a_rowptr[i + 1] - d_a_rowptr[i];
                const C* const SFEM_RESTRICT a_cols = &d_a_colidx[d_a_rowptr[i]];

                R nnz = 0;

                for (R a_k = 0; a_k < a_len; a_k++) {
                    const C a_j = a_cols[a_k];

                    const R                      b_len  = d_b_rowptr[a_j + 1] - d_b_rowptr[a_j];
                    const C* const SFEM_RESTRICT b_cols = &d_b_colidx[d_b_rowptr[a_j]];

                    for (R b_k = 0; b_k < b_len; b_k++) {
                        const C b_j = b_cols[b_k];

                        if (mask[b_j] != i) {
                            mask[b_j] = i;
                            nnz++;
                        }
                    }
                }

                d_c_rowptr[i + 1] = nnz;
            }
        }

        // cumulative sum
        for (ptrdiff_t c_i = 0; c_i < rows; c_i++) {
            d_c_rowptr[c_i + 1] += d_c_rowptr[c_i];
        }

        // Allocate column indices
        if (c_colidx->size() != d_c_rowptr[rows]) {
            c_colidx = create_host_buffer<C>(d_c_rowptr[rows]);
        }

        if (c_values->size() != d_c_rowptr[rows]) {
            c_values = create_host_buffer<T>(d_c_rowptr[rows]);
        }

        C* const SFEM_RESTRICT d_c_colidx = c_colidx->data();
        T* const SFEM_RESTRICT d_c_values = c_values->data();

        const R init   = std::numeric_limits<R>::max();
        const R unseen = smesh::invalid_idx<R>();

#pragma omp parallel
        {
            auto                   next_buff = create_host_buffer<R>(c_columns);
            R* const SFEM_RESTRICT next      = next_buff->data();
            auto                   acc_buff  = create_host_buffer<T>(c_columns);
            T* const SFEM_RESTRICT acc       = acc_buff->data();

            for (ptrdiff_t i = 0; i < c_columns; i++) {
                next[i] = unseen;
            }

#pragma omp for schedule(static)
            for (ptrdiff_t i = 0; i < rows; i++) {
                R head = init;
                R len  = 0;

                const R                      a_len  = d_a_rowptr[i + 1] - d_a_rowptr[i];
                const C* const SFEM_RESTRICT a_cols = &d_a_colidx[d_a_rowptr[i]];
                const T* const SFEM_RESTRICT a_vals = &d_a_values[d_a_rowptr[i]];

                for (R a_k = 0; a_k < a_len; a_k++) {
                    const C a_j = a_cols[a_k];
                    const T aij = a_vals[a_k];

                    const R                      b_len  = d_b_rowptr[a_j + 1] - d_b_rowptr[a_j];
                    const C* const SFEM_RESTRICT b_cols = &d_b_colidx[d_b_rowptr[a_j]];
                    const T* const SFEM_RESTRICT b_vals = &d_b_values[d_b_rowptr[a_j]];

                    for (R b_k = 0; b_k < b_len; b_k++) {
                        const C b_j = b_cols[b_k];
                        const T bij = b_vals[b_k];

                        acc[b_j] += aij * bij;

                        if (next[b_j] == unseen) {
                            next[b_j] = head;
                            head      = b_j;
                            len++;
                        }
                    }
                }

                R offset = d_c_rowptr[i];
                for (R k = 0; k < len; k++) {
                    d_c_colidx[offset] = head;
                    d_c_values[offset] = acc[head];
                    offset++;

                    R temp = head;
                    head   = next[head];

                    // Clear
                    next[temp] = unseen;
                    acc[temp]  = 0;
                }
            }
        }

        return SFEM_SUCCESS;
    }

    template <typename R, typename C, typename TStorage, typename T = TStorage>
    class CRS : public Operator<T> {
    public:
        std::function<void(const T* const, T* const)> apply_;

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
                                                       const T                       scale_output) {
        auto ret     = std::make_shared<CRS<R, C, TStorage, T>>();
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
#else                     // 20-27% faster on M1
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
