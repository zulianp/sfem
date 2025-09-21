#ifndef ACRS_HPP
#define ACRS_HPP

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
//     template <int VEC_SIZE, typename IDX, typename VAL, typename TOp>
//     static SFEM_FORCE_INLINE TOp sdot(const ptrdiff_t                n,
//                                       const ptrdiff_t                col_offset,
//                                       const IDX *const SFEM_RESTRICT cols,
//                                       const VAL *const SFEM_RESTRICT vals,
//                                       const TOp *const SFEM_RESTRICT x) {
//         TOp                    ret            = 0;
//         static const ptrdiff_t n_blocks       = n / VEC_SIZE;
//         const ptrdiff_t        b_extent       = n_blocks * VEC_SIZE;
//         TOp                    buff[VEC_SIZE] = {0};

//         for (ptrdiff_t k = 0; k < b_extent; k += VEC_SIZE) {
// #pragma unroll(VEC_SIZE)
//             for (ptrdiff_t b = 0; b < VEC_SIZE; b++) {
//                 buff[b] += vals[k + b] * x[col_offset + cols[k + b]];
//             }
//         }

//         if (b_extent) {
//             for (ptrdiff_t b = 0; b < VEC_SIZE; b++) {
//                 ret += buff[b];
//             }
//         }

//         for (ptrdiff_t k = b_extent; k < n; k++) {
//             ret += vals[k] * x[col_offset + cols[k]];
//         }

//         return ret;
//     }

//     template <int VEC_SIZE, typename IDX, typename VAL, typename TOp>
//     static SFEM_FORCE_INLINE TOp sdot_padded(const ptrdiff_t                n,
//                                              const ptrdiff_t                col_offset,
//                                              const IDX *const SFEM_RESTRICT cols,
//                                              const VAL *const SFEM_RESTRICT vals,
//                                              const TOp *const SFEM_RESTRICT x) {
//         TOp                    ret            = 0;
//         static const ptrdiff_t n_blocks       = n / VEC_SIZE;
//         const ptrdiff_t        b_extent       = n_blocks * VEC_SIZE;
//         TOp                    buff[VEC_SIZE] = {0};

//         for (ptrdiff_t k = 0; k < b_extent; k += VEC_SIZE) {
// #pragma unroll(VEC_SIZE)
//             for (ptrdiff_t b = 0; b < VEC_SIZE; b++) {
//                 buff[b] += vals[k + b] * x[col_offset + cols[k + b]];
//             }
//         }

//         for (ptrdiff_t b = 0; b < VEC_SIZE; b++) {
//             ret += buff[b];
//         }

//         return ret;
//     }

    template <typename R, typename C, typename T, typename TOp = T>
    class ACRS : public Operator<TOp> {
    public:
        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};

        SharedBuffer<R> rowptr;
        SharedBuffer<C> colidx;
        SharedBuffer<T> values;

        static const int VEC_SIZE   = 2;  // 8 * sizeof(double) / sizeof(T);

        size_t nbytes() const {
            size_t ret = 0;
            if (rowptr) ret += rowptr->nbytes();
            if (colidx) ret += colidx->nbytes();
            if (values) ret += values->nbytes();
            return ret;
        }

        int apply(const TOp *const x, TOp *const y) override {
            SFEM_TRACE_SCOPE("ACRS::apply");

            auto d_rowptr = rowptr->data();
            auto d_colidx = colidx->data();
            auto d_values = values->data();

            const ptrdiff_t nrows = rowptr->size() - 1;
            const ptrdiff_t ncols = this->cols();

#pragma omp parallel for
            for(ptrdiff_t i = 0; i < nrows; i++) {
                auto cols = &d_colidx[d_rowptr[i]];
                const ptrdiff_t ncols = d_rowptr[i + 1] - d_rowptr[i];

                TOp acc = y[i];
                for(ptrdiff_t j = 0; j < ncols; j++) {
                    const idx_t bucket = cols[j];
                    const ptrdiff_t boffset = bucket * VEC_SIZE;
                    const auto vals = &d_values[d_rowptr[i] + boffset];
                    const int vec_size = MIN(VEC_SIZE, ncols - boffset);
                    const auto xb = &x[boffset];

                    for(ptrdiff_t k = 0; k < vec_size; k++) {
                        acc += vals[k] * xb[k];
                    }
                }

                y[i] = acc;
            }
      
            return SFEM_SUCCESS;
        }

        std::ptrdiff_t rows() const override { return rowptr ? (rowptr->size() - 1) : 0; }
        std::ptrdiff_t cols() const override { return rows(); }
        ExecutionSpace execution_space() const override { return execution_space_; }
    };

    template <typename R, typename C, typename T, typename TOp = T, typename TStorage = T>
    std::shared_ptr<ACRS<R, C, TStorage, TOp>> acrs_from_crs(const SharedBuffer<R> &rowptr,
                                                      const SharedBuffer<C> &colidx,
                                                      const SharedBuffer<T> &values,
                                                      const ExecutionSpace   es) {
        assert(es == EXECUTION_SPACE_HOST);
        using ACRS_t = ACRS<R, C, TStorage, TOp>;
        static const int VEC_SIZE = ACRS_t::VEC_SIZE;

        const ptrdiff_t nrows = rowptr->size() - 1;
        auto d_rowptr = rowptr->data();
        auto d_colidx = colidx->data();
        auto d_values = values->data();

        auto acrs_rowptr = sfem::create_host_buffer<R>(nrows + 1);
        auto tag = sfem::create_host_buffer<mask_t>(mask_count(nrows));
        auto d_tag = tag->data();
        auto d_acrs_rowptr = acrs_rowptr->data();

        for(ptrdiff_t i = 0; i < nrows; i++) {
            auto cols = &d_colidx[d_rowptr[i]];
            const ptrdiff_t ncols = d_rowptr[i + 1] - d_rowptr[i];

            count_t nbuckets = 0;
            for(ptrdiff_t j = 0; j < ncols; j++) {
                const idx_t bucket = cols[j] / VEC_SIZE;
                if(mask_get(bucket, d_tag) == 0) {
                    mask_set(bucket, d_tag);
                    nbuckets++;
                }
            }
            d_acrs_rowptr[i+1] = d_acrs_rowptr[i] + nbuckets;

            for(ptrdiff_t j = 0; j < ncols; j++) {
                const idx_t bucket = cols[j] / VEC_SIZE;
                mask_unset(bucket, d_tag);
            }
        }

        auto acrs_colidx = sfem::create_host_buffer<C>(d_acrs_rowptr[nrows]);
        auto acrs_values = sfem::create_host_buffer<TStorage>(d_acrs_rowptr[nrows] * VEC_SIZE);
        auto d_acrs_colidx = acrs_colidx->data();
        auto d_acrs_values = acrs_values->data();

        for(ptrdiff_t i = 0; i < nrows; i++) {
            auto cols = &d_colidx[d_rowptr[i]];
            const ptrdiff_t ncols = d_rowptr[i + 1] - d_rowptr[i];

            auto buckets = &d_tag[d_acrs_rowptr[i]];
            const ptrdiff_t nbuckets = d_acrs_rowptr[i + 1] - d_acrs_rowptr[i];

            idx_t prev_bucket = 0;
            if(ncols > 0) {
                prev_bucket = cols[0] / VEC_SIZE;
            }

            ptrdiff_t bucket_offset = 0;
            for(ptrdiff_t k = 0; k < ncols; k++) {
                const idx_t bucket = cols[k] / VEC_SIZE;
                assert(bucket < nbuckets);

                if(bucket != prev_bucket) {
                    assert(prev_bucket < bucket);
                    prev_bucket = bucket;
                    d_acrs_colidx[d_acrs_rowptr[i] + bucket_offset] = bucket;
                    bucket_offset++;
                }

                d_acrs_values[(d_acrs_rowptr[i] + bucket_offset) * VEC_SIZE + cols[k] % VEC_SIZE] += d_values[k];
            }
        }

        auto acrs         = std::make_shared<ACRS_t>();
        acrs->rowptr      = acrs_rowptr;
        acrs->colidx      = acrs_colidx;
        acrs->values      = acrs_values;
        acrs->execution_space_ = es;

        size_t crs_bytes  = rowptr->nbytes() + colidx->nbytes() + values->nbytes();
        size_t acrs_bytes = acrs->nbytes();

        if (true) {
            printf("CRS KB: %zu\n", crs_bytes / 1024);
            printf("ACRS KB: %zu\n", acrs_bytes / 1024);
            printf("ACRS/CRS: %f\n", static_cast<double>(acrs_bytes) / crs_bytes);
        }

        return acrs;
    }
}  // namespace sfem

#endif  // ACRS_HPP
