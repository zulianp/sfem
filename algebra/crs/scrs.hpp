#ifndef SCRS_HPP
#define SCRS_HPP

#include "sfem_Buffer.hpp"
#include "sfem_Operator.hpp"

#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sfem {

    template <typename R, typename C, typename T, typename S, typename TOp = T>
    class SCRS : public Operator<TOp> {
    public:
        ExecutionSpace                 execution_space_{EXECUTION_SPACE_INVALID};
        SharedBuffer<R>                rowptr;
        SharedBuffer<R>                mapptr;
        SharedBuffer<C>                mapidx;
        SharedBuffer<S>                colidx;
        SharedBuffer<T>                values;
        std::vector<SharedBuffer<TOp>> gather;

        void init_gather() {
            auto d_mapptr    = mapptr->data();
            int  num_threads = 1;
#ifdef _OPENMP
#pragma omp parallel
            {
#pragma omp single
                { num_threads = omp_get_num_threads(); }
            }
#endif
            gather.resize(num_threads);
            const ptrdiff_t n_blocks = mapptr->size() - 1;

#pragma omp parallel
            {
                ptrdiff_t max_size = 0;
#pragma omp for
                for (ptrdiff_t b = 0; b < n_blocks; b++) {
                    const ptrdiff_t offset  = d_mapptr[b];
                    const ptrdiff_t ngather = d_mapptr[b + 1] - offset;
                    max_size                = MAX(ngather, max_size);
                }

                int thread_id = 0;
#ifdef _OPENMP
                thread_id = omp_get_thread_num();
#endif

                gather[thread_id] = sfem::create_host_buffer<TOp>(max_size);
            }
        }

        int apply(const TOp *const x, TOp *const y) override {
            SFEM_TRACE_SCOPE("SCRS::apply");
            auto d_rowptr = rowptr->data();
            auto d_mapptr = mapptr->data();
            auto d_mapidx = mapidx->data();
            auto d_colidx = colidx->data();
            auto d_values = values->data();

            const ptrdiff_t        nrows     = rowptr->size() - 1;
            const ptrdiff_t        n_blocks  = mapptr->size() - 1;
            static const ptrdiff_t small_max = std::numeric_limits<S>::max() - 1;

#pragma omp parallel
            {
#ifdef _OPENMP
                const int thread_id = omp_get_thread_num();
#else
                const int thread_id = 0;
#endif
                auto d_gather = gather[thread_id]->data();

#pragma omp for
                for (ptrdiff_t b = 0; b < n_blocks; b++) {
                    const ptrdiff_t offset  = d_mapptr[b];
                    const ptrdiff_t ngather = d_mapptr[b + 1] - offset;
                    const auto      d_mi    = &d_mapidx[offset];

                    // Gather from global vector
                    for (ptrdiff_t i = 0; i < ngather; i++) {
                        d_gather[i] = x[d_mi[i]];
                    }

                    const ptrdiff_t rs     = b * small_max;
                    const ptrdiff_t extent = MIN(small_max, nrows - rs);
                    const auto      rpi    = &d_rowptr[rs];
                    auto            yi     = &y[rs];

                    for (ptrdiff_t i = 0; i < extent; i++) {
                        const ptrdiff_t ncols = rpi[i + 1] - rpi[i];
                        const auto      row_vals = &d_values[rpi[i]];
                        const auto      cols     = &d_colidx[rpi[i]];

                        const static int VECTOR_SIZE       = 8;
                        const auto       nvecs             = ncols / VECTOR_SIZE;
                        const auto       bextent           = nvecs * VECTOR_SIZE;
                        TOp              buff[VECTOR_SIZE] = {0};

                        for (ptrdiff_t k = 0; k < bextent; k += VECTOR_SIZE) {
#pragma unroll(VECTOR_SIZE)
                            for (int v = 0; v < VECTOR_SIZE; v++) {
                                buff[v] += static_cast<TOp>(row_vals[k + v]) * d_gather[cols[k + v]];
                            }
                        }

                        TOp acc = yi[i];
                        if (bextent) {
                            for (int b = 0; b < VECTOR_SIZE; b++) {
                                acc += buff[b];
                            }
                        }

                        for (ptrdiff_t k = bextent; k < ncols; k++) {
                            acc += static_cast<TOp>(row_vals[k]) * d_gather[cols[k]];
                        }

                        yi[i] = acc;
                    }
                }
            }

            return SFEM_SUCCESS;
        }

        std::ptrdiff_t rows() const override { return rowptr->size() - 1; }
        std::ptrdiff_t cols() const override { return rows(); }
        ExecutionSpace execution_space() const override { return execution_space_; }
    };

    template <typename R, typename C, typename T, typename S = uint16_t, typename TOp = T>
    std::shared_ptr<SCRS<R, C, T, S, TOp>> scrs_from_crs(const SharedBuffer<R> &rowptr,
                                                         const SharedBuffer<C> &colidx,
                                                         const SharedBuffer<T> &values,
                                                         const ExecutionSpace   es) {
        assert(es == EXECUTION_SPACE_HOST);

        auto                   scrs      = std::make_shared<SCRS<R, C, T, S, TOp>>();
        static const ptrdiff_t small_max = std::numeric_limits<S>::max() - 1;
        const ptrdiff_t        nrows     = rowptr->size() - 1;
        const ptrdiff_t        n_blocks  = (nrows + small_max - 1) / small_max;

        SharedBuffer<S> inv_map = sfem::create_host_buffer<S>(nrows);
        SharedBuffer<S> scolidx = sfem::create_host_buffer<S>(colidx->size());
        SharedBuffer<R> mapptr  = sfem::create_host_buffer<R>(n_blocks + 1);

        auto d_inv_map = inv_map->data();
        auto d_rowptr  = rowptr->data();
        auto d_colidx  = colidx->data();
        auto d_mapptr  = mapptr->data();
        auto d_scolidx = scolidx->data();

        for (ptrdiff_t b = 0; b < n_blocks; b++) {
            const ptrdiff_t rs     = b * small_max;
            const ptrdiff_t extent = MIN(small_max, nrows - rs);
            const auto      rpi    = &d_rowptr[rs];

            S nmaps = 0;
            for (ptrdiff_t i = 0; i < extent; i++) {
                const ptrdiff_t ncols = rpi[i + 1] - rpi[i];
                const auto      cols  = &d_colidx[rpi[i]];

                for (ptrdiff_t k = 0; k < ncols; k++) {
                    const C col = cols[k];
                    if (!d_inv_map[col]) {
                        d_inv_map[col] = ++nmaps;
                    }
                }
            }

            d_mapptr[b + 1] = nmaps + d_mapptr[b];
        }

        auto mapidx    = sfem::create_host_buffer<C>(d_mapptr[n_blocks]);
        auto d_map_idx = mapidx->data();

        for (ptrdiff_t b = 0; b < n_blocks; b++) {
            const ptrdiff_t rs     = b * small_max;
            const ptrdiff_t extent = MIN(small_max, nrows - rs);
            const auto      rpi    = &d_rowptr[rs];
            const ptrdiff_t offset = d_mapptr[b];

            S nmaps = 0;
            for (ptrdiff_t i = 0; i < extent; i++) {
                const ptrdiff_t ncols = rpi[i + 1] - rpi[i];
                const auto      cols  = &d_colidx[rpi[i]];
                auto            scols = &d_scolidx[rpi[i]];

                for (ptrdiff_t k = 0; k < ncols; k++) {
                    const C col = cols[k];
                    if (!d_inv_map[col]) {
                        d_map_idx[offset + nmaps] = col;
                        d_inv_map[col]            = ++nmaps;
                    }

                    scols[k] = d_inv_map[col] - 1;
                }
            }

            // Clean-up
            for (ptrdiff_t i = offset; i < d_mapptr[b + 1]; i++) {
                d_inv_map[d_map_idx[i]] = 0;
            }
        }

        scrs->rowptr           = rowptr;
        scrs->values           = values;
        scrs->colidx           = scolidx;
        scrs->mapidx           = mapidx;
        scrs->mapptr           = mapptr;
        scrs->execution_space_ = es;
        scrs->init_gather();
        return scrs;
    }
}  // namespace sfem

#endif  // SCRS_HPP
