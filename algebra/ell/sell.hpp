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

    // Stateless SELL-C SpMV kernel with compile-time slice height
    // Layout semantics:
    // - slice_ptr has size n_slices+1, cumulative per-slice widths (max nnz per row in slice)
    // - colidx/values are column-major within each slice with contiguous stride SLICE_HEIGHT
    // - storage is padded to complete columns within a slice
    // Row-major traversal (one row per thread)
    // Pointer-stride across column-major storage to remove muls in hot loop.
    // Optional compile-time prefetch distance in columns.
    template <int SLICE_HEIGHT, int PREFETCH_DIST = 0, typename R, typename C, typename V, typename TOp>
    static inline int sell_spmv_rowmajor(const std::ptrdiff_t           nrows,
                                         const R *const SFEM_RESTRICT   slice_ptr,
                                         const C *const SFEM_RESTRICT   colidx,
                                         const V *const SFEM_RESTRICT   values,
                                         const TOp *const SFEM_RESTRICT x,
                                         TOp *const SFEM_RESTRICT       y) {
        if (!slice_ptr || !colidx || !values || !x || !y) {
            SFEM_ERROR("sell_spmv: invalid pointers!\n");
            return SFEM_FAILURE;
        }

#pragma omp parallel for schedule(static, SLICE_HEIGHT)
        for (std::ptrdiff_t row = 0; row < nrows; ++row) {
            const std::ptrdiff_t s         = row / SLICE_HEIGHT;                        // slice id
            const int            r_in_s    = static_cast<int>(row - s * SLICE_HEIGHT);  // local row in slice
            const std::ptrdiff_t elems     = static_cast<std::ptrdiff_t>(slice_ptr[s + 1] - slice_ptr[s]);
            const std::ptrdiff_t width     = elems / SLICE_HEIGHT;                       // columns in this slice
            const std::ptrdiff_t base_cols = static_cast<std::ptrdiff_t>(slice_ptr[s]);  // element offset base

            if (width == 0) {
                // Nothing stored in this slice; no valid base to stride from
                y[row] = y[row];
                continue;
            }

            const C *SFEM_RESTRICT colp = colidx + base_cols + r_in_s;
            const V *SFEM_RESTRICT valp = values + base_cols + r_in_s;

            TOp acc = y[row];
            for (std::ptrdiff_t j = 0; j < width; ++j) {
#if (PREFETCH_DIST > 0)
#if defined(__GNUC__)
                if (j + PREFETCH_DIST < width) {
                    __builtin_prefetch(valp + SLICE_HEIGHT * PREFETCH_DIST, 0, 1);
                    __builtin_prefetch(colp + SLICE_HEIGHT * PREFETCH_DIST, 0, 1);
                    const C pf_c = colp[SLICE_HEIGHT * PREFETCH_DIST];
                    __builtin_prefetch(x + pf_c, 0, 1);
                }
#endif
#endif
                if (colp[0] != (C)-1) acc += static_cast<TOp>(valp[0]) * x[colp[0]];
                colp += SLICE_HEIGHT;
                valp += SLICE_HEIGHT;
            }
            y[row] = acc;
        }

        return SFEM_SUCCESS;
    }

    // Slice-major traversal (one slice per thread). Improves locality for values/colidx/y.
    // The last slice may be partial; guard rows beyond nrows.
    template <int SLICE_HEIGHT, typename R, typename C, typename V, typename TOp>
    static inline int sell_spmv_slicemajor(const std::ptrdiff_t           nrows,
                                           const R *const SFEM_RESTRICT   slice_ptr,
                                           const C *const SFEM_RESTRICT   colidx,
                                           const V *const SFEM_RESTRICT   values,
                                           const TOp *const SFEM_RESTRICT x,
                                           TOp *const SFEM_RESTRICT       y) {
        if (!slice_ptr || !colidx || !values || !x || !y) {
            SFEM_ERROR("sell_spmv: invalid pointers!\n");
            return SFEM_FAILURE;
        }

        const std::ptrdiff_t n_slices = (nrows + SLICE_HEIGHT - 1) / SLICE_HEIGHT;

#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t s = 0; s < n_slices; ++s) {
            const std::ptrdiff_t row_begin = s * SLICE_HEIGHT;
            const std::ptrdiff_t row_end   = std::min<std::ptrdiff_t>(row_begin + SLICE_HEIGHT, nrows);
            const std::ptrdiff_t elems     = static_cast<std::ptrdiff_t>(slice_ptr[s + 1] - slice_ptr[s]);
            const std::ptrdiff_t width     = elems / SLICE_HEIGHT;
            const std::ptrdiff_t base      = static_cast<std::ptrdiff_t>(slice_ptr[s]);

            // Process one column of the slice at a time for better contiguous accesses
            for (std::ptrdiff_t j = 0; j < width; ++j) {
                const std::ptrdiff_t col_base = base + j * SLICE_HEIGHT;

#pragma unroll(4)
                for (std::ptrdiff_t row = row_begin; row < row_end; ++row) {
                    const int            r_in_s = static_cast<int>(row - row_begin);
                    const std::ptrdiff_t idx    = col_base + r_in_s;
                    if (colidx[idx] != (C)-1) y[row] += static_cast<TOp>(values[idx]) * x[colidx[idx]];
                }
            }
        }

        return SFEM_SUCCESS;
    }

    // Backward-compatible wrapper: defaults to row-major without prefetch
    template <int SLICE_HEIGHT, typename R, typename C, typename V, typename TOp>
    static inline int sell_spmv(const std::ptrdiff_t           nrows,
                                const R *const SFEM_RESTRICT   slice_ptr,
                                const C *const SFEM_RESTRICT   colidx,
                                const V *const SFEM_RESTRICT   values,
                                const TOp *const SFEM_RESTRICT x,
                                TOp *const SFEM_RESTRICT       y) {
        return sell_spmv_rowmajor<SLICE_HEIGHT, 0, R, C, V, TOp>(nrows, slice_ptr, colidx, values, x, y);
    }

    template <int SLICE_HEIGHT, typename R, typename C, typename V, typename TOp>
    static inline int sell_spmv_simd(const std::ptrdiff_t           nrows,
                                     const R *const SFEM_RESTRICT   slice_ptr,
                                     const C *const SFEM_RESTRICT   colidx,
                                     const V *const SFEM_RESTRICT   values,
                                     const TOp *const SFEM_RESTRICT x,
                                     TOp *const SFEM_RESTRICT       y) {
        // TODO
        ptrdiff_t n_slices = nrows / SLICE_HEIGHT;
        ptrdiff_t reminder = nrows - n_slices * SLICE_HEIGHT;

        // Vectorized loop
        for (ptrdiff_t s = 0; s < n_slices; ++s) {
            const ptrdiff_t row_begin = s * SLICE_HEIGHT;
            const ptrdiff_t row_end   = row_begin + SLICE_HEIGHT;
            const ptrdiff_t elems     = slice_ptr[s + 1] - slice_ptr[s];
            const ptrdiff_t width     = elems / SLICE_HEIGHT;
            const ptrdiff_t base      = slice_ptr[s];

            TOp * const SFEM_RESTRICT acc = &y[row_begin];

            for(ptrdiff_t j = 0; j < width; ++j) {
                const ptrdiff_t col_base = base + j * SLICE_HEIGHT;
                const C * const SFEM_RESTRICT colp = &colidx[col_base];
                const V * const SFEM_RESTRICT valp = &values[col_base];

                TOp in[SLICE_HEIGHT] = {0};
                for(ptrdiff_t r = 0; r < SLICE_HEIGHT; ++r) {
                    if(colp[r] != -1) {
                        in[r] = x[colp[r]];
                    }
                }

#pragma omp simd safelen(SLICE_HEIGHT)
                for(ptrdiff_t r = 0; r < SLICE_HEIGHT; ++r) {
                    acc[r] += in[r] * valp[r];
                }
            }
        }

        // Clean-up loop for the last slice
        for (ptrdiff_t r = 0; r < reminder; ++r) {
            const ptrdiff_t idx = nrows - reminder + r;
            if (colidx[idx] != -1) y[idx] += static_cast<TOp>(values[idx]) * x[colidx[idx]];
        }

        return SFEM_SUCCESS;
    }

    template <typename R, typename C, typename T, typename TOp = T>
    class SlicedEllpack : public Operator<TOp> {
    public:
        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};

        // SELL-C format (column-major within a slice):
        // - Rows are grouped into slices of height slice_height_
        // - Each slice has a (padded) width equal to the maximum nnz per row in the slice
        // - Entries are stored column-major per slice to improve memory locality

        // CuSPARSE-compatible SELL:
        // - slice_ptr holds element offsets (prefix-sum of width[s] * slice_height_)
        // - base element offset for slice s is slice_ptr[s]
        // - width[s] = (slice_ptr[s+1] - slice_ptr[s]) / slice_height_
        SharedBuffer<R> slice_ptr;  // size: n_slices + 1, prefix-sum of per-slice element counts
        SharedBuffer<C> colidx;     // size: slice_ptr[n_slices]
        SharedBuffer<T> values;     // size: slice_ptr[n_slices]

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

            const auto      d_slice_ptr = slice_ptr->data();
            const auto      d_colidx    = colidx->data();
            const auto      d_values    = values->data();
            const ptrdiff_t nrows       = n_rows;

            switch (slice_height_) {
            case 8:
                return sell_spmv_simd<8>(nrows, d_slice_ptr, d_colidx, d_values, x, y);
            case 16:
                return sell_spmv_simd<16>(nrows, d_slice_ptr, d_colidx, d_values, x, y);
            case 32:
                return sell_spmv_simd<32>(nrows, d_slice_ptr, d_colidx, d_values, x, y);
            case 64:
                return sell_spmv_simd<64>(nrows, d_slice_ptr, d_colidx, d_values, x, y);
            default:
                break;
            }

            // Fallback generic path for unusual slice heights: iterate by slices
            {
                const int       height   = slice_height_;
                const ptrdiff_t n_slices = (nrows + height - 1) / height;

#pragma omp parallel for schedule(static)
                for (ptrdiff_t s = 0; s < n_slices; ++s) {
                    const ptrdiff_t row_begin = s * height;
                    const ptrdiff_t row_end   = std::min<ptrdiff_t>(row_begin + height, nrows);
                    const ptrdiff_t elems     = static_cast<ptrdiff_t>(d_slice_ptr[s + 1] - d_slice_ptr[s]);
                    const ptrdiff_t width     = elems / height;
                    const ptrdiff_t base      = static_cast<ptrdiff_t>(d_slice_ptr[s]);
                    if (width == 0) continue;

                    for (ptrdiff_t j = 0; j < width; ++j) {
                        const ptrdiff_t col_base = base + j * height;

#pragma omp simd
                        for (ptrdiff_t row = row_begin; row < row_end; ++row) {
                            const int       r_in_s = static_cast<int>(row - row_begin);
                            const ptrdiff_t idx    = col_base + r_in_s;

                            if (d_colidx[idx] != -1) y[row] += static_cast<TOp>(d_values[idx]) * x[d_colidx[idx]];
                        }
                    }
                }
            }

            //             const ptrdiff_t num_slices = nrows / slice_height_;
            // #pragma omp parallel for schedule(static)
            //             for (ptrdiff_t s = 0; s < num_slices; ++s) {
            //                 ptrdiff_t start = d_slice_ptr[s];
            //                 ptrdiff_t end = d_slice_ptr[s+1];
            //                 int slice_width = (end - start) / slice_height_;
            //                 for (ptrdiff_t j = 0; j < slice_width; ++j) {
            // #pragma omp simd
            //                     for (ptrdiff_t r = 0; r < slice_height_; ++r) {
            //                         ptrdiff_t idx = start + j * slice_height_ + r;
            //                         if(d_colidx[idx] != -1)
            //                             y[s*slice_height_+ r] += d_values[idx] * x[d_colidx[idx]];
            //                     }
            //                 }
            //             }

            //             // Clean-up loop for the last slice
            //             const ptrdiff_t last_slice_height = n_rows % slice_height_;
            //             for (ptrdiff_t r = 0; r < last_slice_height; ++r) {
            //                 ptrdiff_t idx = n_rows - last_slice_height + r;
            //                 y[idx] += d_values[idx] * x[d_colidx[idx]];
            //             }

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

        // Compute per-slice padded element counts (width * H) to match cuSPARSE layout
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

            // element count for this slice (column-major, padded with -1)
            d_sp[s + 1] = static_cast<R>(d_sp[s] + max_width * H);
        }

        const ptrdiff_t storage_size = static_cast<ptrdiff_t>(d_sp[n_slices]);

        auto sell_colidx = sfem::create_host_buffer<C>(storage_size);
        auto sell_values = sfem::create_host_buffer<TStorage>(storage_size);
        auto d_sc        = sell_colidx->data();
        auto d_sv        = sell_values->data();

        // Initialize to zero in case of padding or incomplete last slice
        // (buffers are calloc'ed, so already zeroed)

        for (ptrdiff_t s = 0; s < n_slices; ++s) {
            const ptrdiff_t row_begin = s * H;
            const ptrdiff_t row_end   = std::min<ptrdiff_t>(row_begin + H, nrows);
            const ptrdiff_t elems     = static_cast<ptrdiff_t>(d_sp[s + 1] - d_sp[s]);
            const ptrdiff_t width     = elems / H;
            const ptrdiff_t base      = static_cast<ptrdiff_t>(d_sp[s]);
            if (width == 0) {
                continue;
            }

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
                        d_sc[idx] = (C)-1;  // cuSPARSE padding sentinel
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