#ifndef SFEM_CRS_X_BSR_HPP
#define SFEM_CRS_X_BSR_HPP

#include "sfem_Operator.hpp"

#include "sfem_BSR.hpp"
#include "sfem_CRS.hpp"

namespace sfem {

    template <typename R, typename C>
    void csr_x_bsr_mm_sym(const ptrdiff_t              block_rows,
                          const ptrdiff_t              c_block_cols,
                          const R* const SFEM_RESTRICT crs_rowptr,
                          const C* const SFEM_RESTRICT crs_colidx,
                          const R* const SFEM_RESTRICT bsr_rowptr,
                          const C* const SFEM_RESTRICT bsr_colidx,
                          R* const SFEM_RESTRICT       c_rowptr,
                          R* const SFEM_RESTRICT       mask_workspace,
                          const int                    n_workspaces) {
        crs_mm_sym(
                block_rows, c_block_cols, crs_rowptr, crs_colidx, bsr_rowptr, bsr_colidx, c_rowptr, mask_workspace, n_workspaces);
    }

    template <typename TStorage>
    inline void csr_x_bsr_scaled_block(const TStorage                      scale,
                                       const TStorage* const SFEM_RESTRICT b_block,
                                       TStorage* const SFEM_RESTRICT       c_block,
                                       const int                           block_matrix_size) {
        for (int d = 0; d < block_matrix_size; d++) {
            c_block[d] = scale * b_block[d];
        }
    }

    template <typename TStorage>
    inline void csr_x_bsr_scaled_block_add(const TStorage                      scale,
                                           const TStorage* const SFEM_RESTRICT b_block,
                                           TStorage* const SFEM_RESTRICT       c_block,
                                           const int                           block_matrix_size) {
        for (int d = 0; d < block_matrix_size; d++) {
            c_block[d] += scale * b_block[d];
        }
    }

    template <typename R, typename C, typename TStorage>
    void csr_x_bsr_mm_apply(const ptrdiff_t                     block_rows,
                            const ptrdiff_t                     c_block_cols,
                            const int                           block_matrix_size,
                            const R* const SFEM_RESTRICT        crs_rowptr,
                            const C* const SFEM_RESTRICT        crs_colidx,
                            const TStorage* const SFEM_RESTRICT crs_values,
                            const R* const SFEM_RESTRICT        bsr_rowptr,
                            const C* const SFEM_RESTRICT        bsr_colidx,
                            const TStorage* const SFEM_RESTRICT bsr_values,
                            const R* const SFEM_RESTRICT        c_rowptr,
                            C* const SFEM_RESTRICT              c_colidx,
                            TStorage* const SFEM_RESTRICT       c_values,
                            R* const SFEM_RESTRICT              next_workspace,
                            TStorage* const SFEM_RESTRICT       acc_workspace,
                            const int                           n_workspaces) {
        const R init   = std::numeric_limits<R>::max();
        const R unseen = smesh::invalid_idx<R>();

#pragma omp parallel num_threads(n_workspaces)
        {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            R* const SFEM_RESTRICT        next = &next_workspace[tid * c_block_cols];
            TStorage* const SFEM_RESTRICT acc  = &acc_workspace[tid * c_block_cols * block_matrix_size];

            for (ptrdiff_t i = 0; i < c_block_cols; i++) {
                next[i] = unseen;
            }

#pragma omp for schedule(static)
            for (ptrdiff_t i = 0; i < block_rows; i++) {
                R head = init;
                R len  = 0;

                const R                             a_begin = crs_rowptr[i];
                const R                             a_end   = crs_rowptr[i + 1];
                const C* const SFEM_RESTRICT        a_cols  = &crs_colidx[a_begin];
                const TStorage* const SFEM_RESTRICT a_vals  = &crs_values[a_begin];

                for (R a_k = 0, a_len = a_end - a_begin; a_k < a_len; a_k++) {
                    const C        a_j = a_cols[a_k];
                    const TStorage aij = a_vals[a_k];

                    const R                      b_begin = bsr_rowptr[a_j];
                    const R                      b_end   = bsr_rowptr[a_j + 1];
                    const C* const SFEM_RESTRICT b_cols  = &bsr_colidx[b_begin];

                    for (R b_k = 0, b_len = b_end - b_begin; b_k < b_len; b_k++) {
                        const C b_j = b_cols[b_k];

                        const TStorage* const SFEM_RESTRICT b_block = &bsr_values[(b_begin + b_k) * block_matrix_size];
                        TStorage* const SFEM_RESTRICT       c_block = &acc[b_j * block_matrix_size];

                        if (next[b_j] == unseen) {
                            next[b_j] = head;
                            head      = b_j;
                            csr_x_bsr_scaled_block(aij, b_block, c_block, block_matrix_size);
                            len++;
                        } else {
                            csr_x_bsr_scaled_block_add(aij, b_block, c_block, block_matrix_size);
                        }
                    }
                }

                R offset = c_rowptr[i];
                for (R k = 0; k < len; k++) {
                    c_colidx[offset] = head;

                    TStorage* const SFEM_RESTRICT       c_block   = &c_values[offset * block_matrix_size];
                    const TStorage* const SFEM_RESTRICT acc_block = &acc[head * block_matrix_size];
                    for (int d = 0; d < block_matrix_size; d++) {
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
    int csr_x_bsr_mm(const ptrdiff_t               c_block_cols,
                     const int                     row_block_size,
                     const int                     col_block_size,
                     const SharedBuffer<R>&        crs_rowptr,
                     const SharedBuffer<C>&        crs_colidx,
                     const SharedBuffer<TStorage>& crs_values,
                     const SharedBuffer<R>&        bsr_rowptr,
                     const SharedBuffer<C>&        bsr_colidx,
                     const SharedBuffer<TStorage>& bsr_values,
                     SharedBuffer<R>&              c_rowptr,
                     SharedBuffer<C>&              c_colidx,
                     SharedBuffer<TStorage>&       c_values) {
        const ptrdiff_t block_rows = crs_rowptr->size() - 1;

        if (c_rowptr->size() != block_rows + 1) {
            c_rowptr = create_host_buffer<R>(block_rows + 1);
        }

        const R* const SFEM_RESTRICT        d_crs_rowptr = crs_rowptr->data();
        const C* const SFEM_RESTRICT        d_crs_colidx = crs_colidx->data();
        const TStorage* const SFEM_RESTRICT d_crs_values = crs_values->data();
        const R* const SFEM_RESTRICT        d_bsr_rowptr = bsr_rowptr->data();
        const C* const SFEM_RESTRICT        d_bsr_colidx = bsr_colidx->data();
        const TStorage* const SFEM_RESTRICT d_bsr_values = bsr_values->data();

        R* const SFEM_RESTRICT d_c_rowptr = c_rowptr->data();

#ifdef _OPENMP
        const int n_workspaces = omp_get_max_threads();
#else
        const int n_workspaces = 1;
#endif

        auto mask_workspace = create_host_buffer<R>(n_workspaces * c_block_cols);

        csr_x_bsr_mm_sym(block_rows,
                         c_block_cols,
                         d_crs_rowptr,
                         d_crs_colidx,
                         d_bsr_rowptr,
                         d_bsr_colidx,
                         d_c_rowptr,
                         mask_workspace->data(),
                         n_workspaces);

        const ptrdiff_t nblocks           = d_c_rowptr[block_rows];
        const int       block_matrix_size = row_block_size * col_block_size;

        if (c_colidx->size() != nblocks) {
            c_colidx = create_host_buffer<C>(nblocks);
        }

        if (c_values->size() != nblocks * block_matrix_size) {
            c_values = create_host_buffer<TStorage>(nblocks * block_matrix_size);
        }

        C* const SFEM_RESTRICT        d_c_colidx = c_colidx->data();
        TStorage* const SFEM_RESTRICT d_c_values = c_values->data();

        auto next_workspace = create_host_buffer<R>(n_workspaces * c_block_cols);
        auto acc_workspace  = create_host_buffer<TStorage>(n_workspaces * c_block_cols * block_matrix_size);

        csr_x_bsr_mm_apply(block_rows,
                           c_block_cols,
                           block_matrix_size,
                           d_crs_rowptr,
                           d_crs_colidx,
                           d_crs_values,
                           d_bsr_rowptr,
                           d_bsr_colidx,
                           d_bsr_values,
                           d_c_rowptr,
                           d_c_colidx,
                           d_c_values,
                           next_workspace->data(),
                           acc_workspace->data(),
                           n_workspaces);

        return SFEM_SUCCESS;
    }

    template <typename R, typename C>
    void bsr_x_crs_mm_sym(const ptrdiff_t              block_rows,
                          const ptrdiff_t              c_block_cols,
                          const R* const SFEM_RESTRICT bsr_rowptr,
                          const C* const SFEM_RESTRICT bsr_colidx,
                          const R* const SFEM_RESTRICT crs_rowptr,
                          const C* const SFEM_RESTRICT crs_colidx,
                          R* const SFEM_RESTRICT       c_rowptr,
                          R* const SFEM_RESTRICT       mask_workspace,
                          const int                    n_workspaces) {
        bsr_mm_sym(
                block_rows, c_block_cols, bsr_rowptr, bsr_colidx, crs_rowptr, crs_colidx, c_rowptr, mask_workspace, n_workspaces);
    }

    template <typename R, typename C, typename TStorage>
    void bsr_x_crs_mm_apply(const ptrdiff_t                     block_rows,
                            const ptrdiff_t                     c_block_cols,
                            const int                           block_matrix_size,
                            const R* const SFEM_RESTRICT        bsr_rowptr,
                            const C* const SFEM_RESTRICT        bsr_colidx,
                            const TStorage* const SFEM_RESTRICT bsr_values,
                            const R* const SFEM_RESTRICT        crs_rowptr,
                            const C* const SFEM_RESTRICT        crs_colidx,
                            const TStorage* const SFEM_RESTRICT crs_values,
                            const R* const SFEM_RESTRICT        c_rowptr,
                            C* const SFEM_RESTRICT              c_colidx,
                            TStorage* const SFEM_RESTRICT       c_values,
                            R* const SFEM_RESTRICT              next_workspace,
                            TStorage* const SFEM_RESTRICT       acc_workspace,
                            const int                           n_workspaces) {
        const R init   = std::numeric_limits<R>::max();
        const R unseen = smesh::invalid_idx<R>();

#pragma omp parallel num_threads(n_workspaces)
        {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            R* const SFEM_RESTRICT        next = &next_workspace[tid * c_block_cols];
            TStorage* const SFEM_RESTRICT acc  = &acc_workspace[tid * c_block_cols * block_matrix_size];

            for (ptrdiff_t i = 0; i < c_block_cols; i++) {
                next[i] = unseen;
            }

#pragma omp for schedule(static)
            for (ptrdiff_t i = 0; i < block_rows; i++) {
                R head = init;
                R len  = 0;

                const R                      a_begin = bsr_rowptr[i];
                const R                      a_end   = bsr_rowptr[i + 1];
                const C* const SFEM_RESTRICT a_cols  = &bsr_colidx[a_begin];

                for (R a_k = 0, a_len = a_end - a_begin; a_k < a_len; a_k++) {
                    const C a_j = a_cols[a_k];

                    const TStorage* const SFEM_RESTRICT a_block = &bsr_values[(a_begin + a_k) * block_matrix_size];

                    const R                             b_begin = crs_rowptr[a_j];
                    const R                             b_end   = crs_rowptr[a_j + 1];
                    const C* const SFEM_RESTRICT        b_cols  = &crs_colidx[b_begin];
                    const TStorage* const SFEM_RESTRICT b_vals  = &crs_values[b_begin];

                    for (R b_k = 0, b_len = b_end - b_begin; b_k < b_len; b_k++) {
                        const C        b_j = b_cols[b_k];
                        const TStorage bij = b_vals[b_k];

                        TStorage* const SFEM_RESTRICT c_block = &acc[b_j * block_matrix_size];

                        if (next[b_j] == unseen) {
                            next[b_j] = head;
                            head      = b_j;
                            csr_x_bsr_scaled_block(bij, a_block, c_block, block_matrix_size);
                            len++;
                        } else {
                            csr_x_bsr_scaled_block_add(bij, a_block, c_block, block_matrix_size);
                        }
                    }
                }

                R offset = c_rowptr[i];
                for (R k = 0; k < len; k++) {
                    c_colidx[offset] = head;

                    TStorage* const SFEM_RESTRICT       c_block   = &c_values[offset * block_matrix_size];
                    const TStorage* const SFEM_RESTRICT acc_block = &acc[head * block_matrix_size];
                    for (int d = 0; d < block_matrix_size; d++) {
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
    int bsr_x_crs_mm(const ptrdiff_t               c_block_cols,
                     const int                     row_block_size,
                     const int                     col_block_size,
                     const SharedBuffer<R>&        bsr_rowptr,
                     const SharedBuffer<C>&        bsr_colidx,
                     const SharedBuffer<TStorage>& bsr_values,
                     const SharedBuffer<R>&        crs_rowptr,
                     const SharedBuffer<C>&        crs_colidx,
                     const SharedBuffer<TStorage>& crs_values,
                     SharedBuffer<R>&              c_rowptr,
                     SharedBuffer<C>&              c_colidx,
                     SharedBuffer<TStorage>&       c_values) {
        const ptrdiff_t block_rows = bsr_rowptr->size() - 1;

        if (c_rowptr->size() != block_rows + 1) {
            c_rowptr = create_host_buffer<R>(block_rows + 1);
        }

        const R* const SFEM_RESTRICT        d_bsr_rowptr = bsr_rowptr->data();
        const C* const SFEM_RESTRICT        d_bsr_colidx = bsr_colidx->data();
        const TStorage* const SFEM_RESTRICT d_bsr_values = bsr_values->data();
        const R* const SFEM_RESTRICT        d_crs_rowptr = crs_rowptr->data();
        const C* const SFEM_RESTRICT        d_crs_colidx = crs_colidx->data();
        const TStorage* const SFEM_RESTRICT d_crs_values = crs_values->data();

        R* const SFEM_RESTRICT d_c_rowptr = c_rowptr->data();

#ifdef _OPENMP
        const int n_workspaces = omp_get_max_threads();
#else
        const int n_workspaces = 1;
#endif

        auto mask_workspace = create_host_buffer<R>(n_workspaces * c_block_cols);

        bsr_x_crs_mm_sym(block_rows,
                         c_block_cols,
                         d_bsr_rowptr,
                         d_bsr_colidx,
                         d_crs_rowptr,
                         d_crs_colidx,
                         d_c_rowptr,
                         mask_workspace->data(),
                         n_workspaces);

        const ptrdiff_t nblocks           = d_c_rowptr[block_rows];
        const int       block_matrix_size = row_block_size * col_block_size;

        if (c_colidx->size() != nblocks) {
            c_colidx = create_host_buffer<C>(nblocks);
        }

        if (c_values->size() != nblocks * block_matrix_size) {
            c_values = create_host_buffer<TStorage>(nblocks * block_matrix_size);
        }

        C* const SFEM_RESTRICT        d_c_colidx = c_colidx->data();
        TStorage* const SFEM_RESTRICT d_c_values = c_values->data();

        auto next_workspace = create_host_buffer<R>(n_workspaces * c_block_cols);
        auto acc_workspace  = create_host_buffer<TStorage>(n_workspaces * c_block_cols * block_matrix_size);

        bsr_x_crs_mm_apply(block_rows,
                           c_block_cols,
                           block_matrix_size,
                           d_bsr_rowptr,
                           d_bsr_colidx,
                           d_bsr_values,
                           d_crs_rowptr,
                           d_crs_colidx,
                           d_crs_values,
                           d_c_rowptr,
                           d_c_colidx,
                           d_c_values,
                           next_workspace->data(),
                           acc_workspace->data(),
                           n_workspaces);

        return SFEM_SUCCESS;
    }

    template <typename R, typename C, typename TStorage, typename T = TStorage>
    std::shared_ptr<BSR<R, C, TStorage, T>> mm(const std::shared_ptr<CRS<R, C, TStorage, T>>& crs,
                                               const std::shared_ptr<BSR<R, C, TStorage, T>>& bsr) {
        SFEM_TRACE_SCOPE("CRS_X_BSR::mm");

        if (crs->execution_space() != EXECUTION_SPACE_HOST || bsr->execution_space() != EXECUTION_SPACE_HOST) {
            SFEM_ERROR("CRS_X_BSR::mm is not supported for non-host execution space");
            return nullptr;
        }

        const ptrdiff_t bsr_block_rows = bsr->row_ptr->size() - 1;
        if (crs->cols() != bsr_block_rows) {
            SFEM_ERROR("CRS_X_BSR::mm incompatible dimensions: CRS has %td cols, BSR has %td block rows",
                       crs->cols(),
                       bsr_block_rows);
            return nullptr;
        }

        auto ret     = std::make_shared<BSR<R, C, TStorage, T>>();
        ret->row_ptr = create_host_buffer<R>(0);
        ret->col_idx = create_host_buffer<C>(0);
        ret->values  = create_host_buffer<TStorage>(0);

        csr_x_bsr_mm(bsr->block_cols_,
                     bsr->row_block_size_,
                     bsr->col_block_size_,
                     crs->row_ptr,
                     crs->col_idx,
                     crs->values,
                     bsr->row_ptr,
                     bsr->col_idx,
                     bsr->values,
                     ret->row_ptr,
                     ret->col_idx,
                     ret->values);

        const ptrdiff_t ret_block_rows = crs->rows();
        const ptrdiff_t ret_block_cols = bsr->block_cols_;
        const int       ret_row_bs     = bsr->row_block_size_;
        const int       ret_col_bs     = bsr->col_block_size_;

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

    template <typename R, typename C, typename TStorage, typename T = TStorage>
    std::shared_ptr<BSR<R, C, TStorage, T>> mm(const std::shared_ptr<BSR<R, C, TStorage, T>>& bsr,
                                               const std::shared_ptr<CRS<R, C, TStorage, T>>& crs) {
        SFEM_TRACE_SCOPE("BSR_X_CRS::mm");

        if (bsr->execution_space() != EXECUTION_SPACE_HOST || crs->execution_space() != EXECUTION_SPACE_HOST) {
            SFEM_ERROR("BSR_X_CRS::mm is not supported for non-host execution space");
            return nullptr;
        }

        if (bsr->block_cols_ != crs->rows()) {
            SFEM_ERROR("BSR_X_CRS::mm incompatible dimensions: BSR has %td block cols, CRS has %td rows",
                       bsr->block_cols_,
                       crs->rows());
            return nullptr;
        }

        auto ret     = std::make_shared<BSR<R, C, TStorage, T>>();
        ret->row_ptr = create_host_buffer<R>(0);
        ret->col_idx = create_host_buffer<C>(0);
        ret->values  = create_host_buffer<TStorage>(0);

        bsr_x_crs_mm(crs->cols_,
                     bsr->row_block_size_,
                     bsr->col_block_size_,
                     bsr->row_ptr,
                     bsr->col_idx,
                     bsr->values,
                     crs->row_ptr,
                     crs->col_idx,
                     crs->values,
                     ret->row_ptr,
                     ret->col_idx,
                     ret->values);

        const ptrdiff_t ret_block_rows = bsr->row_ptr->size() - 1;
        const ptrdiff_t ret_block_cols = crs->cols_;
        const int       ret_row_bs     = bsr->row_block_size_;
        const int       ret_col_bs     = bsr->col_block_size_;

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

    template <typename R, typename C, typename TStorage, typename T = TStorage>
    std::shared_ptr<BSR<R, C, TStorage, T>> rap(const std::shared_ptr<CRS<R, C, TStorage, T>>& r,
                                                const std::shared_ptr<BSR<R, C, TStorage, T>>& a,
                                                const std::shared_ptr<CRS<R, C, TStorage, T>>& p) {
        // Compute D = A P
        auto d = mm(a, p);

        // Compute G = R D
        auto g = mm(r, d);
        return g;
    }
}  // namespace sfem

#endif  // SFEM_CRS_X_BSR_HPP
