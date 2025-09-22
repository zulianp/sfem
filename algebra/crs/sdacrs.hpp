#ifndef SDACRS_HPP
#define SDACRS_HPP

#include "sfem_Buffer.hpp"
#include "sfem_Operator.hpp"
#include "sfem_Tracer.hpp"

#include <algorithm>
#include <cassert>
#include <limits>
#include <type_traits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sfem {
    template <int VEC_SIZE, typename IDX, typename VAL, typename TOp>
    static SFEM_FORCE_INLINE TOp da_sdot(
                                      const ptrdiff_t                n,
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
                buff[b] += vals[k + b] * x[cols[k + b]];
            }
        }

        if (b_extent) {
            for (ptrdiff_t b = 0; b < VEC_SIZE; b++) {
                ret += buff[b];
            }
        }

        for (ptrdiff_t k = b_extent; k < n; k++) {
            ret += vals[k] * x[cols[k]];
        }

        return ret;
    }

    template <typename R, typename C, typename T, typename S, typename TOp = T>
    class SDACRS : public Operator<TOp> {
    public:
        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};

        SharedBuffer<R> diag_rowptr;
        SharedBuffer<S> diag_colidx;
        SharedBuffer<T> diag_values;

        SharedBuffer<R> offdiag_rowptr;
        SharedBuffer<C> offdiag_colidx;
        SharedBuffer<T> offdiag_values;

        static const int  VEC_SIZE   = 8;  // 8 * sizeof(double) / sizeof(T);

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
            SFEM_TRACE_SCOPE("SDACRS::apply");

            if (!diag_rowptr || !offdiag_rowptr) {
                SFEM_ERROR("SDACRS in invalid state!\n");
                return SFEM_FAILURE;
            }

            auto d_diag_rowptr    = diag_rowptr->data();
            auto d_offdiag_rowptr = offdiag_rowptr->data();
            auto d_diag_colidx    = diag_colidx->data();
            auto d_offdiag_colidx = offdiag_colidx->data();
            auto d_diag_values    = diag_values->data();
            auto d_offdiag_values = offdiag_values->data();

            const ptrdiff_t nrows = diag_rowptr->size() - 1;

            if(offdiag_values->size() > 0) {
#pragma omp parallel for
                for (ptrdiff_t row = 0; row < nrows; ++row) {
                    TOp acc = y[row];
                    {
                        auto            cols   = &d_diag_colidx[d_diag_rowptr[row]];
                        auto            vals   = &d_diag_values[d_diag_rowptr[row]];
                        const ptrdiff_t extent = d_diag_rowptr[row + 1] - d_diag_rowptr[row];
                        acc += da_sdot<VEC_SIZE>(extent, cols, vals, &x[row]);
                    }
                    {
                        auto            cols   = &d_offdiag_colidx[d_offdiag_rowptr[row]];
                        auto            vals   = &d_offdiag_values[d_offdiag_rowptr[row]];
                        const ptrdiff_t extent = d_offdiag_rowptr[row + 1] - d_offdiag_rowptr[row];
                        acc += da_sdot<VEC_SIZE>(extent, cols, vals, x);
                    }
                    y[row] = acc;
                }
            } else {
#pragma omp parallel for
                for (ptrdiff_t row = 0; row < nrows; ++row) {
                    TOp acc = y[row];
                    {
                        auto            cols   = &d_diag_colidx[d_diag_rowptr[row]];
                        auto            vals   = &d_diag_values[d_diag_rowptr[row]];
                        const ptrdiff_t extent = d_diag_rowptr[row + 1] - d_diag_rowptr[row];
                        acc += da_sdot<VEC_SIZE>(extent, cols, vals, &x[row]);
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

    template <typename R, typename C, typename T, typename S = int16_t, typename TOp = T, typename TStorage = T>
    std::shared_ptr<SDACRS<R, C, TStorage, S, TOp>> sdacrs_from_crs(const SharedBuffer<R> &rowptr,
                                                                const SharedBuffer<C> &colidx,
                                                                const SharedBuffer<T> &values,
                                                                const ExecutionSpace   es) {
        assert(es == EXECUTION_SPACE_HOST);
        using SDACRS_t = SDACRS<R, C, TStorage, S, TOp>;

        const auto nrows = rowptr->size() - 1;

        auto diag_rowptr = sfem::create_host_buffer<R>(nrows + 1);
        auto off_rowptr  = sfem::create_host_buffer<R>(nrows + 1);

        static const S ub_span = std::numeric_limits<S>::max();
        static const S lb_span = -std::numeric_limits<S>::max();
        printf("UB SPAN: %d\n", ub_span);
        printf("LB SPAN: %d\n", lb_span);

        auto d_rowptr = rowptr->data();
        auto d_colidx = colidx->data();
        auto d_values = values->data();
        auto d_drp    = diag_rowptr->data();
        auto d_orp    = off_rowptr->data();

        d_drp[0] = 0;
        d_orp[0] = 0;

        for (ptrdiff_t row = 0; row < nrows; ++row) {
            const auto row_begin = d_rowptr[row];
            const auto row_end   = d_rowptr[row + 1];

            ptrdiff_t diag_count    = 0;
            ptrdiff_t offdiag_count = 0;

            for (auto idx = row_begin; idx < row_end; ++idx) {
                const auto col = static_cast<ptrdiff_t>(d_colidx[idx]);
                const auto local = col - row;

                if (local >= lb_span && local <= ub_span) {
                    ++diag_count;
                } else {
                    ++offdiag_count;
                }
            }

            d_drp[row + 1] = d_drp[row] + static_cast<R>(diag_count);
            d_orp[row + 1] = d_orp[row] + static_cast<R>(offdiag_count);
        }
        

        const ptrdiff_t diag_nnz    = static_cast<ptrdiff_t>(d_drp[nrows]);
        const ptrdiff_t offdiag_nnz = static_cast<ptrdiff_t>(d_orp[nrows]);

        auto diag_colidx = sfem::create_host_buffer<S>(diag_nnz);
        auto diag_values = sfem::create_host_buffer<TStorage>(diag_nnz);
        auto off_colidx  = sfem::create_host_buffer<C>(offdiag_nnz);
        auto off_values  = sfem::create_host_buffer<TStorage>(offdiag_nnz);

        auto d_diag_colidx = diag_colidx->data();
        auto d_diag_values = diag_values->data();
        auto d_off_colidx  = off_colidx->data();
        auto d_off_values  = off_values->data();

        for (ptrdiff_t row = 0; row < nrows; ++row) {
            const auto row_begin = d_rowptr[row];
            const auto row_end   = d_rowptr[row + 1];

            auto diag_write = d_drp[row];
            auto off_write  = d_orp[row];

            for (auto idx = row_begin; idx < row_end; ++idx) {
                const auto col = static_cast<ptrdiff_t>(d_colidx[idx]);
                const auto val = d_values[idx];
                const auto local = col - row;

                if (local >= lb_span && local <= ub_span) {
                    d_diag_colidx[diag_write] = static_cast<S>(local);
                    d_diag_values[diag_write] = val;
                    ++diag_write;
                } else {
                    d_off_colidx[off_write] = static_cast<C>(col);
                    d_off_values[off_write] = val;
                    ++off_write;
                }
            }
        }

        auto sdacrs              = std::make_shared<SDACRS_t>();
        sdacrs->diag_rowptr      = diag_rowptr;
        sdacrs->diag_colidx      = diag_colidx;
        sdacrs->diag_values      = diag_values;
        sdacrs->offdiag_rowptr   = off_rowptr;
        sdacrs->offdiag_colidx   = off_colidx;
        sdacrs->offdiag_values   = off_values;
        sdacrs->execution_space_ = es;

        size_t crs_bytes  = rowptr->nbytes() + colidx->nbytes() + values->nbytes();
        size_t sdacrs_bytes = sdacrs->nbytes();

        if (false) {
            printf("CRS KB: %zu\n", crs_bytes / 1024);
            printf("SDACRS KB: %zu\n", sdacrs_bytes / 1024);
            printf("DIAG NNZ: %zu\n", diag_values->size());
            printf("OFFDIAG NNZ: %zu\n", off_values->size());
            printf("CRS NNZ: %zu\n", values->size());
            printf("SDACRS/CRS: %f\n", static_cast<double>(sdacrs_bytes) / crs_bytes);
        }

        return sdacrs;
    }
}  // namespace sfem

#endif  // SDACRS_HPP
