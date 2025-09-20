#ifndef SCRS_HPP
#define SCRS_HPP

#include "sfem_Buffer.hpp"
#include "sfem_Operator.hpp"

#include <algorithm>
#include <cassert>
#include <limits>
#include <type_traits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sfem {
    template <int VEC_SIZE, typename IDX, typename VAL, typename TOp>
    static SFEM_FORCE_INLINE TOp sdot(const ptrdiff_t                n,
                                      const ptrdiff_t                col_offset,
                                      const IDX *const SFEM_RESTRICT cols,
                                      const VAL *const SFEM_RESTRICT vals,
                                      const TOp *const SFEM_RESTRICT x) {
        TOp                    ret            = 0;
        static const ptrdiff_t n_blocks       = n / VEC_SIZE;
        const ptrdiff_t        b_extent       = n_blocks * VEC_SIZE;
        TOp                    buff[VEC_SIZE] = {0};

        for (ptrdiff_t k = 0; k < b_extent; k += VEC_SIZE) {
#pragma unroll(VEC_SIZE)
            for (ptrdiff_t b = 0; b < VEC_SIZE; b++) {
                buff[b] += vals[k + b] * x[col_offset + cols[k + b]];
            }
        }

        if (b_extent) {
            for (ptrdiff_t b = 0; b < VEC_SIZE; b++) {
                ret += buff[b];
            }
        }

        for (ptrdiff_t k = b_extent; k < n; k++) {
            ret += vals[k] * x[col_offset + cols[k]];
        }

        return ret;
    }

    template <int VEC_SIZE, typename IDX, typename VAL, typename TOp>
    static SFEM_FORCE_INLINE TOp sdot_padded(const ptrdiff_t                n,
                                             const ptrdiff_t                col_offset,
                                             const IDX *const SFEM_RESTRICT cols,
                                             const VAL *const SFEM_RESTRICT vals,
                                             const TOp *const SFEM_RESTRICT x) {
        TOp                    ret            = 0;
        static const ptrdiff_t n_blocks       = n / VEC_SIZE;
        const ptrdiff_t        b_extent       = n_blocks * VEC_SIZE;
        TOp                    buff[VEC_SIZE] = {0};

        for (ptrdiff_t k = 0; k < b_extent; k += VEC_SIZE) {
#pragma unroll(VEC_SIZE)
            for (ptrdiff_t b = 0; b < VEC_SIZE; b++) {
                buff[b] += vals[k + b] * x[col_offset + cols[k + b]];
            }
        }

        for (ptrdiff_t b = 0; b < VEC_SIZE; b++) {
            ret += buff[b];
        }

        return ret;
    }
    
    template <typename R, typename C, typename T, typename S, typename TOp = T>
    class SCRS : public Operator<TOp> {
    public:
        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};

        SharedBuffer<R> diag_rowptr;
        SharedBuffer<S> diag_colidx;
        SharedBuffer<T> diag_values;

        SharedBuffer<R> offdiag_rowptr;
        SharedBuffer<C> offdiag_colidx;
        SharedBuffer<T> offdiag_values;

        static const int  BLOCK_SPAN = std::numeric_limits<S>::max() + 1;
        static const int  VEC_SIZE   = 8;  // 8 * sizeof(double) / sizeof(T);
        static const bool PAD        = true;

        size_t nbytes() const {
            size_t ret = 0;
            if (diag_rowptr) ret += diag_rowptr->nbytes();
            if (diag_colidx) ret += diag_colidx->nbytes();
            if (diag_values) ret += diag_values->nbytes();
            if (offdiag_rowptr) ret += offdiag_rowptr->nbytes();
            if (offdiag_colidx) ret += offdiag_colidx->nbytes();
            if (offdiag_values) ret += offdiag_values->nbytes();
            return ret;
        }

        int apply(const TOp *const x, TOp *const y) override {
            SFEM_TRACE_SCOPE("SCRS::apply");

            assert(BLOCK_SPAN > 0);
            if (!diag_rowptr || !offdiag_rowptr) {
                SFEM_ERROR("SCRS in invalid state!\n");
                return SFEM_FAILURE;
            }

            auto d_diag_rowptr    = diag_rowptr->data();
            auto d_offdiag_rowptr = offdiag_rowptr->data();
            auto d_diag_colidx    = diag_colidx->data();
            auto d_offdiag_colidx = offdiag_colidx->data();
            auto d_diag_values    = diag_values->data();
            auto d_offdiag_values = offdiag_values->data();

            const ptrdiff_t nrows   = diag_rowptr->size() - 1;
            const ptrdiff_t nblocks = (nrows + BLOCK_SPAN - 1) / BLOCK_SPAN;

#pragma omp parallel for
            for (ptrdiff_t b = 0; b < nblocks; b++) {
                const ptrdiff_t block_base = b * BLOCK_SPAN;
                const ptrdiff_t block_end  = MIN(block_base + BLOCK_SPAN, nrows);

                for (ptrdiff_t row = block_base; row < block_end; ++row) {
                    TOp acc = y[row];
                    {
                        auto            cols   = &d_diag_colidx[d_diag_rowptr[row]];
                        auto            vals   = &d_diag_values[d_diag_rowptr[row]];
                        const ptrdiff_t extent = d_diag_rowptr[row + 1] - d_diag_rowptr[row];
                        if (PAD) {
                            acc += sdot_padded<VEC_SIZE>(extent, block_base, cols, vals, x);
                        } else {
                            acc += sdot<VEC_SIZE>(extent, block_base, cols, vals, x);
                        }
                    }
                    {
                        auto            cols   = &d_offdiag_colidx[d_offdiag_rowptr[row]];
                        auto            vals   = &d_offdiag_values[d_offdiag_rowptr[row]];
                        const ptrdiff_t extent = d_offdiag_rowptr[row + 1] - d_offdiag_rowptr[row];
                        if (PAD) {
                            acc += sdot_padded<VEC_SIZE>(extent, 0, cols, vals, x);
                        } else {
                            acc += sdot<VEC_SIZE>(extent, 0, cols, vals, x);
                        }
                    }
                    y[row] = acc;
                }
            }

            return SFEM_SUCCESS;
        }

        std::ptrdiff_t rows() const override { return diag_rowptr ? (diag_rowptr->size() - 1) : 0; }
        std::ptrdiff_t cols() const override { return rows(); }
        ExecutionSpace execution_space() const override { return execution_space_; }
    };

    template <typename R, typename C, typename T, typename S = uint16_t, typename TOp = T>
    std::shared_ptr<SCRS<R, C, T, S, TOp>> scrs_from_crs(const SharedBuffer<R> &rowptr,
                                                         const SharedBuffer<C> &colidx,
                                                         const SharedBuffer<T> &values,
                                                         const ExecutionSpace   es) {
        assert(es == EXECUTION_SPACE_HOST);
        using SCRS_t = SCRS<R, C, T, S, TOp>;

        static const bool PAD        = SCRS_t::PAD;
        static const int  VEC_SIZE   = SCRS_t::VEC_SIZE;
        static const int  BLOCK_SPAN = SCRS_t::BLOCK_SPAN;

        const auto nrows = rowptr->size() - 1;

        auto diag_rowptr = sfem::create_host_buffer<R>(nrows + 1);
        auto off_rowptr  = sfem::create_host_buffer<R>(nrows + 1);

        auto d_rowptr = rowptr->data();
        auto d_colidx = colidx->data();
        auto d_values = values->data();
        auto d_drp    = diag_rowptr->data();
        auto d_orp    = off_rowptr->data();

        d_drp[0] = 0;
        d_orp[0] = 0;

        const ptrdiff_t nblocks = (nrows + BLOCK_SPAN - 1) / BLOCK_SPAN;

        for (ptrdiff_t b = 0; b < nblocks; b++) {
            const ptrdiff_t block_base = b * BLOCK_SPAN;
            const ptrdiff_t block_end  = MIN(block_base + BLOCK_SPAN, nrows);

            for (ptrdiff_t row = block_base; row < block_end; ++row) {
                const auto row_begin = d_rowptr[row];
                const auto row_end   = d_rowptr[row + 1];

                ptrdiff_t diag_count    = 0;
                ptrdiff_t offdiag_count = 0;

                for (auto idx = row_begin; idx < row_end; ++idx) {
                    const auto col = static_cast<ptrdiff_t>(d_colidx[idx]);
                    if (col >= block_base && col < block_end) {
                        ++diag_count;
                    } else {
                        ++offdiag_count;
                    }
                }

                if (PAD) {
                    diag_count    = (diag_count + VEC_SIZE - 1) / VEC_SIZE * VEC_SIZE;
                    offdiag_count = (offdiag_count + VEC_SIZE - 1) / VEC_SIZE * VEC_SIZE;
                }

                d_drp[row + 1] = d_drp[row] + static_cast<R>(diag_count);
                d_orp[row + 1] = d_orp[row] + static_cast<R>(offdiag_count);
            }
        }

        const ptrdiff_t diag_nnz    = static_cast<ptrdiff_t>(d_drp[nrows]);
        const ptrdiff_t offdiag_nnz = static_cast<ptrdiff_t>(d_orp[nrows]);

        auto diag_colidx = sfem::create_host_buffer<S>(diag_nnz);
        auto diag_values = sfem::create_host_buffer<T>(diag_nnz);
        auto off_colidx  = sfem::create_host_buffer<C>(offdiag_nnz);
        auto off_values  = sfem::create_host_buffer<T>(offdiag_nnz);

        auto d_diag_colidx = diag_colidx->data();
        auto d_diag_values = diag_values->data();
        auto d_off_colidx  = off_colidx->data();
        auto d_off_values  = off_values->data();

        for (ptrdiff_t b = 0; b < nblocks; b++) {
            const ptrdiff_t block_base = b * BLOCK_SPAN;
            const ptrdiff_t block_end  = MIN(block_base + BLOCK_SPAN, nrows);

            for (ptrdiff_t row = block_base; row < block_end; ++row) {
                const auto row_begin = d_rowptr[row];
                const auto row_end   = d_rowptr[row + 1];

                auto diag_write = d_drp[row];
                auto off_write  = d_orp[row];

                for (auto idx = row_begin; idx < row_end; ++idx) {
                    const auto col = static_cast<ptrdiff_t>(d_colidx[idx]);
                    const auto val = d_values[idx];

                    if (col >= block_base && col < block_end) {
                        const auto local = col - block_base;
                        assert(local >= 0 && local <= max_local_index);
                        d_diag_colidx[diag_write] = static_cast<S>(local);
                        d_diag_values[diag_write] = val;
                        ++diag_write;
                    } else {
                        d_off_colidx[off_write] = static_cast<C>(col);
                        d_off_values[off_write] = val;
                        ++off_write;
                    }
                }

                if (PAD) {
                    for (int i = diag_write; i < d_drp[row + 1]; i++) {
                        d_diag_colidx[i] = 0;
                        d_diag_values[i] = 0;
                    }

                    for (int i = off_write; i < d_orp[row + 1]; i++) {
                        d_off_colidx[i] = 0;
                        d_off_values[i] = 0;
                    }
                }

                assert(diag_write == d_drp[row + 1]);
                assert(off_write == d_orp[row + 1]);
            }
        }

        auto scrs              = std::make_shared<SCRS_t>();
        scrs->diag_rowptr      = diag_rowptr;
        scrs->diag_colidx      = diag_colidx;
        scrs->diag_values      = diag_values;
        scrs->offdiag_rowptr   = off_rowptr;
        scrs->offdiag_colidx   = off_colidx;
        scrs->offdiag_values   = off_values;
        scrs->execution_space_ = es;

        size_t crs_bytes  = rowptr->nbytes() + colidx->nbytes() + values->nbytes();
        size_t scrs_bytes = scrs->nbytes();

        if (0) {
            printf("CRS KB: %zu\n", crs_bytes / 1024);
            printf("SCRS KB: %zu\n", scrs_bytes / 1024);
            printf("DIAG NNZ: %zu\n", diag_values->size());
            printf("OFFDIAG NNZ: %zu\n", off_values->size());
            printf("CRS NNZ: %zu\n", values->size());
            printf("SCRS/CRS: %f\n", static_cast<double>(scrs_bytes) / crs_bytes);
        }
        return scrs;
    }
}  // namespace sfem

#endif  // SCRS_HPP
