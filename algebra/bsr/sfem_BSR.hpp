#ifndef SFEM_BSR_SPMV_HPP
#define SFEM_BSR_SPMV_HPP

#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "sfem_MatrixFreeLinearSolver.hpp"
#include "sfem_aliases.hpp"
#include "smesh_types.hpp"

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

    template <typename R, typename C, typename TStorage>
    int bsr_transpose(const ptrdiff_t               block_rows,
                      const ptrdiff_t               block_cols,
                      const int                     row_block_size,
                      const int                     col_block_size,
                      const SharedBuffer<R>&        a_rowptr,
                      const SharedBuffer<C>&        a_colidx,
                      const SharedBuffer<TStorage>& a_values,
                      SharedBuffer<R>&              b_rowptr,
                      SharedBuffer<C>&              b_colidx,
                      SharedBuffer<TStorage>&       b_values) {
        const ptrdiff_t block_matrix_size = (ptrdiff_t)row_block_size * col_block_size;

        if (b_rowptr->size() != block_cols + 1) {
            b_rowptr = create_host_buffer<R>(block_cols + 1);
        }

        const R* const SFEM_RESTRICT        d_a_rowptr = a_rowptr->data();
        const C* const SFEM_RESTRICT        d_a_colidx = a_colidx->data();
        const TStorage* const SFEM_RESTRICT d_a_values = a_values->data();

        R* const SFEM_RESTRICT d_b_rowptr = b_rowptr->data();

        bsr_transpose_sym(block_rows, block_cols, row_block_size, col_block_size, d_a_rowptr, d_a_colidx, d_a_values, d_b_rowptr);

        const ptrdiff_t nblocks = d_b_rowptr[block_cols];

        if (b_colidx->size() != nblocks) {
            b_colidx = create_host_buffer<C>(nblocks);
        }

        if (b_values->size() != nblocks * block_matrix_size) {
            b_values = create_host_buffer<TStorage>(nblocks * block_matrix_size);
        }

        C* const SFEM_RESTRICT        d_b_colidx = b_colidx->data();
        TStorage* const SFEM_RESTRICT d_b_values = b_values->data();

        auto next_workspace = create_host_buffer<R>(block_cols);

        bsr_transpose_apply(block_rows,
                            block_cols,
                            row_block_size,
                            col_block_size,
                            d_a_rowptr,
                            d_a_colidx,
                            d_a_values,
                            d_b_rowptr,
                            d_b_colidx,
                            d_b_values,
                            next_workspace->data());

        return SFEM_SUCCESS;
    }

    template <typename R, typename C>
    void bsr_mm_sym(const ptrdiff_t              block_rows,
                    const ptrdiff_t              c_block_cols,
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
            R* const SFEM_RESTRICT mask = &mask_workspace[tid * c_block_cols];

            for (ptrdiff_t i = 0; i < c_block_cols; i++) {
                mask[i] = unseen;
            }

#pragma omp for schedule(static)
            for (ptrdiff_t i = 0; i < block_rows; i++) {
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

        for (ptrdiff_t c_i = 0; c_i < block_rows; c_i++) {
            c_rowptr[c_i + 1] += c_rowptr[c_i];
        }
    }

    template <typename TStorage>
    void bsr_block_mm_acc(const TStorage* const SFEM_RESTRICT a_block,
                          const TStorage* const SFEM_RESTRICT b_block,
                          TStorage* const SFEM_RESTRICT       c_block,
                          const int                           row_block_size,
                          const int                           inner_block_size,
                          const int                           col_block_size,
                          const bool                          init) {
        if (init) {
            for (int d1 = 0; d1 < row_block_size; d1++) {
                TStorage* const SFEM_RESTRICT c_row = &c_block[d1 * col_block_size];
                for (int d2 = 0; d2 < col_block_size; d2++) {
                    TStorage sum = 0;
                    for (int d_inner = 0; d_inner < inner_block_size; d_inner++) {
                        sum += a_block[d1 * inner_block_size + d_inner] * b_block[d_inner * col_block_size + d2];
                    }
                    c_row[d2] = sum;
                }
            }
        } else {
            for (int d1 = 0; d1 < row_block_size; d1++) {
                TStorage* const SFEM_RESTRICT c_row = &c_block[d1 * col_block_size];
                for (int d2 = 0; d2 < col_block_size; d2++) {
                    TStorage sum = 0;
                    for (int d_inner = 0; d_inner < inner_block_size; d_inner++) {
                        sum += a_block[d1 * inner_block_size + d_inner] * b_block[d_inner * col_block_size + d2];
                    }
                    c_row[d2] += sum;
                }
            }
        }
    }

    template <typename R, typename C, typename TStorage>
    void bsr_mm_apply(const ptrdiff_t                     block_rows,
                      const ptrdiff_t                     c_block_cols,
                      const int                           a_row_block_size,
                      const int                           a_col_block_size,
                      const int                           b_col_block_size,
                      const R* const SFEM_RESTRICT        a_rowptr,
                      const C* const SFEM_RESTRICT        a_colidx,
                      const TStorage* const SFEM_RESTRICT a_values,
                      const R* const SFEM_RESTRICT        b_rowptr,
                      const C* const SFEM_RESTRICT        b_colidx,
                      const TStorage* const SFEM_RESTRICT b_values,
                      const R* const SFEM_RESTRICT        c_rowptr,
                      C* const SFEM_RESTRICT              c_colidx,
                      TStorage* const SFEM_RESTRICT       c_values,
                      R* const SFEM_RESTRICT              next_workspace,
                      TStorage* const SFEM_RESTRICT       acc_workspace,
                      const int                           n_workspaces) {
        const R init   = std::numeric_limits<R>::max();
        const R unseen = smesh::invalid_idx<R>();

        const int a_block_matrix_size = a_row_block_size * a_col_block_size;
        const int b_block_matrix_size = a_col_block_size * b_col_block_size;
        const int c_block_matrix_size = a_row_block_size * b_col_block_size;

#pragma omp parallel num_threads(n_workspaces)
        {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            R* const SFEM_RESTRICT        next = &next_workspace[tid * c_block_cols];
            TStorage* const SFEM_RESTRICT acc  = &acc_workspace[tid * c_block_cols * c_block_matrix_size];

            for (ptrdiff_t i = 0; i < c_block_cols; i++) {
                next[i] = unseen;
            }

#pragma omp for schedule(static)
            for (ptrdiff_t i = 0; i < block_rows; i++) {
                R head = init;
                R len  = 0;

                const R                      a_begin = a_rowptr[i];
                const R                      a_end   = a_rowptr[i + 1];
                const C* const SFEM_RESTRICT a_cols  = &a_colidx[a_begin];

                for (R a_k = 0, a_len = a_end - a_begin; a_k < a_len; a_k++) {
                    const C a_j = a_cols[a_k];

                    const R                      b_begin = b_rowptr[a_j];
                    const R                      b_end   = b_rowptr[a_j + 1];
                    const C* const SFEM_RESTRICT b_cols  = &b_colidx[b_begin];

                    const TStorage* const SFEM_RESTRICT a_block = &a_values[(a_begin + a_k) * a_block_matrix_size];

                    for (R b_k = 0, b_len = b_end - b_begin; b_k < b_len; b_k++) {
                        const C b_j = b_cols[b_k];

                        const TStorage* const SFEM_RESTRICT b_block = &b_values[(b_begin + b_k) * b_block_matrix_size];

                        if (next[b_j] == unseen) {
                            next[b_j] = head;
                            head      = b_j;
                            bsr_block_mm_acc(a_block,
                                             b_block,
                                             &acc[b_j * c_block_matrix_size],
                                             a_row_block_size,
                                             a_col_block_size,
                                             b_col_block_size,
                                             true);
                            len++;
                        } else {
                            bsr_block_mm_acc(a_block,
                                             b_block,
                                             &acc[b_j * c_block_matrix_size],
                                             a_row_block_size,
                                             a_col_block_size,
                                             b_col_block_size,
                                             false);
                        }
                    }
                }

                R offset = c_rowptr[i];
                for (R k = 0; k < len; k++) {
                    c_colidx[offset] = head;

                    TStorage* const SFEM_RESTRICT       c_block   = &c_values[offset * c_block_matrix_size];
                    const TStorage* const SFEM_RESTRICT acc_block = &acc[head * c_block_matrix_size];
                    for (int d = 0; d < c_block_matrix_size; d++) {
                        c_block[d] = acc_block[d];
                    }

                    offset++;

                    const R temp = head;
                    head         = next[head];
                    next[temp]   = unseen;
                }
            }
        }
    }

    template <typename R, typename C, typename TStorage>
    int bsr_mm(const ptrdiff_t               c_block_cols,
               const int                     a_row_block_size,
               const int                     a_col_block_size,
               const int                     b_col_block_size,
               const SharedBuffer<R>&        a_rowptr,
               const SharedBuffer<C>&        a_colidx,
               const SharedBuffer<TStorage>& a_values,
               const SharedBuffer<R>&        b_rowptr,
               const SharedBuffer<C>&        b_colidx,
               const SharedBuffer<TStorage>& b_values,
               SharedBuffer<R>&              c_rowptr,
               SharedBuffer<C>&              c_colidx,
               SharedBuffer<TStorage>&       c_values) {
        const ptrdiff_t block_rows = a_rowptr->size() - 1;

        if (c_rowptr->size() != block_rows + 1) {
            c_rowptr = create_host_buffer<R>(block_rows + 1);
        }

        const R* const SFEM_RESTRICT        d_a_rowptr = a_rowptr->data();
        const C* const SFEM_RESTRICT        d_a_colidx = a_colidx->data();
        const TStorage* const SFEM_RESTRICT d_a_values = a_values->data();
        const R* const SFEM_RESTRICT        d_b_rowptr = b_rowptr->data();
        const C* const SFEM_RESTRICT        d_b_colidx = b_colidx->data();
        const TStorage* const SFEM_RESTRICT d_b_values = b_values->data();

        R* const SFEM_RESTRICT d_c_rowptr = c_rowptr->data();

#ifdef _OPENMP
        const int n_workspaces = omp_get_max_threads();
#else
        const int n_workspaces = 1;
#endif

        auto mask_workspace = create_host_buffer<R>(n_workspaces * c_block_cols);

        bsr_mm_sym(block_rows,
                   c_block_cols,
                   d_a_rowptr,
                   d_a_colidx,
                   d_b_rowptr,
                   d_b_colidx,
                   d_c_rowptr,
                   mask_workspace->data(),
                   n_workspaces);

        const ptrdiff_t nblocks             = d_c_rowptr[block_rows];
        const ptrdiff_t c_block_matrix_size = (ptrdiff_t)a_row_block_size * b_col_block_size;

        if (c_colidx->size() != nblocks) {
            c_colidx = create_host_buffer<C>(nblocks);
        }

        if (c_values->size() != nblocks * c_block_matrix_size) {
            c_values = create_host_buffer<TStorage>(nblocks * c_block_matrix_size);
        }

        C* const SFEM_RESTRICT        d_c_colidx = c_colidx->data();
        TStorage* const SFEM_RESTRICT d_c_values = c_values->data();

        auto next_workspace = create_host_buffer<R>(n_workspaces * c_block_cols);
        auto acc_workspace  = create_host_buffer<TStorage>(n_workspaces * c_block_cols * c_block_matrix_size);

        bsr_mm_apply(block_rows,
                     c_block_cols,
                     a_row_block_size,
                     a_col_block_size,
                     b_col_block_size,
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

        std::shared_ptr<BSR<R, C, TStorage, T>> transpose() const {
            SFEM_TRACE_SCOPE("BSR::transpose");
            if (execution_space() != EXECUTION_SPACE_HOST) {
                // TODO: Implement device version
                SFEM_ERROR("Transpose is not supported for non-host execution space");
                return nullptr;
            }

            const ptrdiff_t block_rows = row_ptr->size() - 1;

            auto ret     = std::make_shared<BSR<R, C, TStorage, T>>();
            ret->row_ptr = create_host_buffer<R>(0);
            ret->col_idx = create_host_buffer<C>(0);
            ret->values  = create_host_buffer<TStorage>(0);

            bsr_transpose(block_rows,
                          block_cols_,
                          row_block_size_,
                          col_block_size_,
                          row_ptr,
                          col_idx,
                          values,
                          ret->row_ptr,
                          ret->col_idx,
                          ret->values);

            // The transposed block has swapped row/column block sizes
            ret->block_cols_                = block_rows;
            ret->row_block_size_            = col_block_size_;
            ret->col_block_size_            = row_block_size_;
            ret->uniform_pre_output_scaling = uniform_pre_output_scaling;
            ret->execution_space_           = EXECUTION_SPACE_HOST;

            const ptrdiff_t ret_block_rows = block_cols_;
            const ptrdiff_t ret_block_cols = block_rows;
            const int       ret_row_bs     = col_block_size_;
            const int       ret_col_bs     = row_block_size_;
            const T         scaling        = uniform_pre_output_scaling;

            auto t_rowptr = ret->row_ptr;
            auto t_colidx = ret->col_idx;
            auto t_values = ret->values;

            ret->apply_ = [=](const T* const x, T* const y) {
                bsr_spmv(ret_block_rows,
                         ret_block_cols,
                         ret_row_bs,
                         ret_col_bs,
                         t_rowptr->data(),
                         t_colidx->data(),
                         t_values->data(),
                         scaling,
                         x,
                         y);
            };

            return ret;
        }

        std::shared_ptr<BSR<R, C, TStorage, T>> mm(const std::shared_ptr<BSR<R, C, TStorage, T>>& other) const {
            SFEM_TRACE_SCOPE("BSR::mm");
            if (execution_space() != EXECUTION_SPACE_HOST || other->execution_space() != EXECUTION_SPACE_HOST) {
                // TODO: Implement device version
                SFEM_ERROR("Matrix multiplication is not supported for non-host execution space");
                return nullptr;
            }

            const ptrdiff_t a_block_rows = row_ptr->size() - 1;
            const ptrdiff_t b_block_rows = other->row_ptr->size() - 1;
            if (block_cols_ != b_block_rows) {
                SFEM_ERROR("BSR::mm incompatible block dimensions: A has %td block cols, B has %td block rows",
                           block_cols_,
                           b_block_rows);
                return nullptr;
            }
            if (col_block_size_ != other->row_block_size_) {
                SFEM_ERROR("BSR::mm incompatible block sizes: A col block size %d, B row block size %d",
                           col_block_size_,
                           other->row_block_size_);
                return nullptr;
            }

            auto ret     = std::make_shared<BSR<R, C, TStorage, T>>();
            ret->row_ptr = create_host_buffer<R>(0);
            ret->col_idx = create_host_buffer<C>(0);
            ret->values  = create_host_buffer<TStorage>(0);

            bsr_mm(other->block_cols_,
                   row_block_size_,
                   col_block_size_,
                   other->col_block_size_,
                   row_ptr,
                   col_idx,
                   values,
                   other->row_ptr,
                   other->col_idx,
                   other->values,
                   ret->row_ptr,
                   ret->col_idx,
                   ret->values);

            const ptrdiff_t ret_block_rows = a_block_rows;
            const ptrdiff_t ret_block_cols = other->block_cols_;
            const int       ret_row_bs     = row_block_size_;
            const int       ret_col_bs     = other->col_block_size_;

            ret->block_cols_                = ret_block_cols;
            ret->row_block_size_            = ret_row_bs;
            ret->col_block_size_            = ret_col_bs;
            ret->uniform_pre_output_scaling = 0;
            ret->execution_space_           = EXECUTION_SPACE_HOST;

            auto c_rowptr = ret->row_ptr;
            auto c_colidx = ret->col_idx;
            auto c_values = ret->values;

            ret->apply_ = [=](const T* const x, T* const y) {
                bsr_spmv(ret_block_rows,
                         ret_block_cols,
                         ret_row_bs,
                         ret_col_bs,
                         c_rowptr->data(),
                         c_colidx->data(),
                         c_values->data(),
                         static_cast<T>(0),
                         x,
                         y);
            };

            return ret;
        }

        int apply(const T* const x, T* const y) override {
            SFEM_TRACE_SCOPE("BSR::apply");

            apply_(x, y);
            return 0;
        }

        std::ptrdiff_t rows() const override { return row_block_size_ * (row_ptr->size() - 1); }
        std::ptrdiff_t cols() const override { return col_block_size_ * block_cols_; }

        size_t     nbytes() const { return row_ptr->nbytes() + col_idx->nbytes() + values->nbytes(); }
        inline int row_block_size() const { return row_block_size_; }
        inline int col_block_size() const { return col_block_size_; }
        inline int block_size() const {
            assert(row_block_size_ == col_block_size_);
            return row_block_size_;
        }

        SharedBuffer<R>        row_ptr;
        SharedBuffer<C>        col_idx;
        SharedBuffer<TStorage> values;

        int       row_block_size_{0};
        int       col_block_size_{0};
        ptrdiff_t block_cols_{0};
        T         uniform_pre_output_scaling{0};

        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};

        ExecutionSpace execution_space() const override { return execution_space_; }

        void print(std::ostream& os = std::cout) const {
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
        auto ret                        = std::make_shared<BSR<R, C, TStorage, T>>();
        ret->row_ptr                    = rowptr;
        ret->col_idx                    = colidx;
        ret->values                     = values;
        ret->block_cols_                = block_cols;
        ret->row_block_size_            = row_block_size;
        ret->col_block_size_            = col_block_size;
        ret->uniform_pre_output_scaling = scale_output;

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

    template <typename R, typename C, typename TStorage, typename T = TStorage>
    std::shared_ptr<BSR<R, C, TStorage, T>> rap(const std::shared_ptr<BSR<R, C, TStorage, T>>& r,
                                                const std::shared_ptr<BSR<R, C, TStorage, T>>& a,
                                                const std::shared_ptr<BSR<R, C, TStorage, T>>& p) {
        // Compute D = A P
        auto d = a->mm(p);

        // Compute G = R D
        auto g = r->mm(d);
        return g;
    }
}  // namespace sfem

#endif  // SFEM_BSR_SPMV_HPP
