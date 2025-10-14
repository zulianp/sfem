#ifndef SFEM_ELL_SELL_HPP
#define SFEM_ELL_SELL_HPP

#include "sfem_Buffer.hpp"
#include "sfem_Operator.hpp"
#include "sfem_Tracer.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sfem {

    template <typename R, typename C, typename T, typename TOp = T>
    class SlicedEllpack : public Operator<TOp> {
    public:
        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};

        // SELL-C format (column-major within a slice):
        // - Rows are grouped into slices of height slice_height_
        // - Each slice has a (padded) width equal to the maximum nnz per row in the slice
        // - Entries are stored column-major per slice to improve memory locality

        SharedBuffer<R> slice_ptr;  // size: n_slices + 1, prefix-sum of per-slice widths
        SharedBuffer<C> colidx;     // size: slice_ptr[n_slices] * slice_height_
        SharedBuffer<T> values;     // size: slice_ptr[n_slices] * slice_height_

        static const int DEFAULT_SLICE_HEIGHT = 32;

        // Dimensions
        std::ptrdiff_t n_rows{0};
        int            slice_height_{DEFAULT_SLICE_HEIGHT};

        size_t nbytes() const {
            size_t ret = 0;
            if (slice_ptr) ret += slice_ptr->nbytes();
            if (colidx) ret += colidx->nbytes();
            if (values) ret += values->nbytes();
            return ret;
        }

        int apply(const TOp *const x, TOp *const y) override {
            SFEM_TRACE_SCOPE("SlicedEllpack::apply");

            if (!slice_ptr || !colidx || !values) {
                SFEM_ERROR("SELL in invalid state!\n");
                return SFEM_FAILURE;
            }

            auto d_slice_ptr = slice_ptr->data();
            auto d_colidx    = colidx->data();
            auto d_values    = values->data();

            const ptrdiff_t nrows  = n_rows;
            const int       height = slice_height_;

#pragma omp parallel for
            for (ptrdiff_t row = 0; row < nrows; ++row) {
                const ptrdiff_t s         = row / height;                        // slice id
                const int       r_in_s    = static_cast<int>(row - s * height);  // local row within slice
                const ptrdiff_t width     = static_cast<ptrdiff_t>(d_slice_ptr[s + 1] - d_slice_ptr[s]);
                const ptrdiff_t base_cols = static_cast<ptrdiff_t>(d_slice_ptr[s]) * height;

                TOp acc = y[row];
                for (ptrdiff_t j = 0; j < width; ++j) {
                    const ptrdiff_t idx = base_cols + j * height + r_in_s;
                    const C         c   = d_colidx[idx];
                    const TOp       v   = static_cast<TOp>(d_values[idx]);
                    acc += v * x[c];
                }
                y[row] = acc;
            }

            return SFEM_SUCCESS;
        }

        std::ptrdiff_t rows() const override { return n_rows; }
        std::ptrdiff_t cols() const override { return rows(); }
        ExecutionSpace execution_space() const override { return execution_space_; }
    };

    // Converter from CRS to SELL-C (no row reordering within slices, i.e., sigma=0)
    template <typename R, typename C, typename T, typename TOp = T, typename TStorage = T>
    std::shared_ptr<SlicedEllpack<R, C, TStorage, TOp>> sell_from_crs(
            const SharedBuffer<R> &rowptr,
            const SharedBuffer<C> &colidx,
            const SharedBuffer<T> &values,
            const ExecutionSpace   es,
            const int              slice_height = SlicedEllpack<R, C, TStorage, TOp>::DEFAULT_SLICE_HEIGHT) {
        assert(es == EXECUTION_SPACE_HOST);
        using SELL_t = SlicedEllpack<R, C, TStorage, TOp>;

        const ptrdiff_t nrows = rowptr->size() - 1;
        if (nrows <= 0) {
            auto sell              = std::make_shared<SELL_t>();
            sell->n_rows           = 0;
            sell->slice_height_    = slice_height;
            sell->execution_space_ = es;
            return sell;
        }

        auto d_rowptr = rowptr->data();
        auto d_colidx = colidx->data();
        auto d_values = values->data();

        const int       H        = slice_height;
        const ptrdiff_t n_slices = (nrows + H - 1) / H;

        // Compute per-slice padded widths (max row length in slice)
        auto slice_ptr = sfem::create_host_buffer<R>(n_slices + 1);
        auto d_sp      = slice_ptr->data();
        d_sp[0]        = 0;

        for (ptrdiff_t s = 0; s < n_slices; ++s) {
            const ptrdiff_t row_begin = s * H;
            const ptrdiff_t row_end   = std::min<ptrdiff_t>(row_begin + H, nrows);

            ptrdiff_t max_width = 0;
            for (ptrdiff_t row = row_begin; row < row_end; ++row) {
                const ptrdiff_t extent = static_cast<ptrdiff_t>(d_rowptr[row + 1] - d_rowptr[row]);
                if (extent > max_width) max_width = extent;
            }

            d_sp[s + 1] = static_cast<R>(d_sp[s] + max_width);
        }

        const ptrdiff_t total_columns = static_cast<ptrdiff_t>(d_sp[n_slices]);
        const ptrdiff_t storage_size  = total_columns * H;  // column-major per slice, padded to slice height

        auto sell_colidx = sfem::create_host_buffer<C>(storage_size);
        auto sell_values = sfem::create_host_buffer<TStorage>(storage_size);
        auto d_sc        = sell_colidx->data();
        auto d_sv        = sell_values->data();

        // Initialize to zero in case of padding or incomplete last slice
        // (buffers are calloc'ed, so already zeroed)

        for (ptrdiff_t s = 0; s < n_slices; ++s) {
            const ptrdiff_t row_begin = s * H;
            const ptrdiff_t row_end   = std::min<ptrdiff_t>(row_begin + H, nrows);
            const ptrdiff_t width     = static_cast<ptrdiff_t>(d_sp[s + 1] - d_sp[s]);
            const ptrdiff_t base      = static_cast<ptrdiff_t>(d_sp[s]) * H;

            for (ptrdiff_t r = 0; r < H; ++r) {
                const ptrdiff_t row       = row_begin + r;
                const bool      valid_row = (row < row_end);

                const ptrdiff_t row_start = valid_row ? static_cast<ptrdiff_t>(d_rowptr[row]) : 0;
                const ptrdiff_t row_stop  = valid_row ? static_cast<ptrdiff_t>(d_rowptr[row + 1]) : 0;
                const ptrdiff_t row_nnz   = valid_row ? (row_stop - row_start) : 0;

                for (ptrdiff_t j = 0; j < width; ++j) {
                    const ptrdiff_t idx = base + j * H + r;
                    if (j < row_nnz) {
                        d_sc[idx] = d_colidx[row_start + j];
                        d_sv[idx] = static_cast<TStorage>(d_values[row_start + j]);
                    } else {
                        d_sc[idx] = 0;
                        d_sv[idx] = static_cast<TStorage>(0);
                    }
                }
            }
        }

        auto sell              = std::make_shared<SELL_t>();
        sell->slice_ptr        = slice_ptr;
        sell->colidx           = sell_colidx;
        sell->values           = sell_values;
        sell->n_rows           = nrows;
        sell->slice_height_    = H;
        sell->execution_space_ = es;

        // Optional stats
        if (false) {
            const size_t crs_bytes  = rowptr->nbytes() + colidx->nbytes() + values->nbytes();
            const size_t sell_bytes = sell->nbytes();
            printf("CRS KB: %zu\n", crs_bytes / 1024);
            printf("SELL KB: %zu\n", sell_bytes / 1024);
            printf("SELL/CRS: %f\n", static_cast<double>(sell_bytes) / crs_bytes);
        }

        return sell;
    }
}  // namespace sfem

#endif