#ifndef ACRS_HPP
#define ACRS_HPP

#include "sfem_Buffer.hpp"
#include "sfem_Operator.hpp"
#include "sfem_Tracer.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sfem {

    template <typename R, typename C, typename T, typename TOp = T, int VEC_SIZE_ = 2>
    class ACRS : public Operator<TOp> {
    public:
        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};

        SharedBuffer<R> rowptr;
        SharedBuffer<C> colidx;
        SharedBuffer<T> values;

        static const int VEC_SIZE = VEC_SIZE_;  // 8 * sizeof(double) / sizeof(T);

        size_t nbytes() const {
            size_t ret = 0;
            if (rowptr) ret += rowptr->nbytes();
            if (colidx) ret += colidx->nbytes();
            if (values) ret += values->nbytes();
            return ret;
        }

        void print(std::ostream &os = std::cout) const {
            os << "ACRS (" << rows() << " rows, " << cols() << " cols)\n";
            const ptrdiff_t nrows = rowptr->size() - 1;
            for (ptrdiff_t i = 0; i < nrows; i++) {
                os << i << ") ";
                for (ptrdiff_t j = rowptr->data()[i]; j < rowptr->data()[i + 1]; j++) {
                    os << colidx->data()[j] * VEC_SIZE << " -> (";
                    for (ptrdiff_t k = 0; k < VEC_SIZE; k++) {
                        os << values->data()[j * VEC_SIZE + k] << " ";
                    }
                    os << ") ";
                }
                os << "\n";
            }
            os << "\n";
        }

        int apply(const TOp *const x, TOp *const y) override {
            SFEM_TRACE_SCOPE("ACRS::apply");

            auto d_rowptr = rowptr->data();
            auto d_colidx = colidx->data();
            auto d_values = values->data();

            const ptrdiff_t nrows = this->rows();
            const ptrdiff_t ncols = this->cols();

            // if (ncols % VEC_SIZE == 0) 
            if (false) 
            {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < nrows; i++) {
                    auto            cols   = &d_colidx[d_rowptr[i]];
                    const ptrdiff_t extent = d_rowptr[i + 1] - d_rowptr[i];

                    TOp x_buff[VEC_SIZE * 4] = {0};

                    TOp acc = y[i];
                    for (ptrdiff_t j = 0; j < extent; j += 4) {
                        ptrdiff_t len_segment = MIN(4, extent - j);
                        auto vals = &d_values[(d_rowptr[i] + j) * VEC_SIZE];

                        for (ptrdiff_t k = 0; k < len_segment; k++) {
                            const idx_t     bucket  = cols[j];
                            const ptrdiff_t boffset = bucket * VEC_SIZE;
                            
                            for (ptrdiff_t s = 0; s < VEC_SIZE; s++) {
                                x_buff[k*VEC_SIZE + s] = x[boffset + s];
                            }
                        }

                        TOp lacc[VEC_SIZE] = {0};
                        for(ptrdiff_t k = 0; k < len_segment; k++) {
#pragma unroll(VEC_SIZE)
                            for (ptrdiff_t s = 0; s < VEC_SIZE; s++) {
                                lacc[s] += vals[k] * x_buff[k + s];
                            }
                        }

                        for (ptrdiff_t s = 0; s < VEC_SIZE; s++) {
                            acc += lacc[s];
                        }
                    }

                    y[i] = acc;
                }
            } else {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < nrows; i++) {
                    auto            cols   = &d_colidx[d_rowptr[i]];
                    const ptrdiff_t extent = d_rowptr[i + 1] - d_rowptr[i];

                    TOp acc = y[i];
                    for (ptrdiff_t j = 0; j < extent; j++) {
                        const idx_t     bucket   = cols[j];
                        const ptrdiff_t boffset  = bucket * VEC_SIZE;
                        const auto      vals     = &d_values[(d_rowptr[i] + j) * VEC_SIZE];
                        const int       vec_size = MIN(VEC_SIZE, ncols - boffset);
                        const auto      xb       = &x[boffset];

                        for (ptrdiff_t k = 0; k < vec_size; k++) {
                            acc += vals[k] * xb[k];
                        }
                    }

                    y[i] = acc;
                }
            }

            return SFEM_SUCCESS;
        }

        std::ptrdiff_t rows() const override { return rowptr ? (rowptr->size() - 1) : 0; }
        std::ptrdiff_t cols() const override { return rows(); }
        ExecutionSpace execution_space() const override { return execution_space_; }
    };

    template <typename R, typename C, typename T, typename TOp = T, typename TStorage = T, int VEC_SIZE = 2>
    std::shared_ptr<ACRS<R, C, TStorage, TOp, VEC_SIZE>> acrs_from_crs(const SharedBuffer<R> &rowptr,
                                                             const SharedBuffer<C> &colidx,
                                                             const SharedBuffer<T> &values,
                                                             const ExecutionSpace   es) {
        assert(es == EXECUTION_SPACE_HOST);
        using ACRS_t              = ACRS<R, C, TStorage, TOp, VEC_SIZE>;

        const ptrdiff_t nrows    = rowptr->size() - 1;
        auto            d_rowptr = rowptr->data();
        auto            d_colidx = colidx->data();
        auto            d_values = values->data();

        auto acrs_rowptr   = sfem::create_host_buffer<R>(nrows + 1);
        auto d_acrs_rowptr = acrs_rowptr->data();

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < nrows; i++) {
            auto            cols  = &d_colidx[d_rowptr[i]];
            const ptrdiff_t ncols = d_rowptr[i + 1] - d_rowptr[i];

            count_t nbuckets    = 0;
            idx_t   prev_bucket = 0;
            if (ncols > 0) {
                nbuckets    = 1;
                prev_bucket = cols[0] / VEC_SIZE;
            }

            for (ptrdiff_t j = 0; j < ncols; j++) {
                const idx_t bucket = cols[j] / VEC_SIZE;

                if (bucket != prev_bucket) {
                    prev_bucket = bucket;
                    nbuckets++;
                }
            }

            d_acrs_rowptr[i + 1] = nbuckets;
        }

        for (ptrdiff_t i = 0; i < nrows; i++) {
            d_acrs_rowptr[i + 1] += d_acrs_rowptr[i]; 
        }


        auto acrs_colidx   = sfem::create_host_buffer<C>(d_acrs_rowptr[nrows]);
        auto acrs_values   = sfem::create_host_buffer<TStorage>(d_acrs_rowptr[nrows] * VEC_SIZE);
        auto d_acrs_colidx = acrs_colidx->data();
        auto d_acrs_values = acrs_values->data();

        for (ptrdiff_t i = 0; i < nrows; i++) {
            auto            cols     = &d_colidx[d_rowptr[i]];
            const ptrdiff_t extent   = d_rowptr[i + 1] - d_rowptr[i];
            const ptrdiff_t nbuckets = d_acrs_rowptr[i + 1] - d_acrs_rowptr[i];
            if (!extent) continue;

            ptrdiff_t bucket_offset                         = 0;
            idx_t     prev_bucket                           = cols[0] / VEC_SIZE;
            d_acrs_colidx[d_acrs_rowptr[i] + bucket_offset] = prev_bucket;

            auto crs_values = &d_values[d_rowptr[i]];

            for (ptrdiff_t k = 0; k < extent; k++) {
                const idx_t bucket = cols[k] / VEC_SIZE;
                const idx_t rmnd   = cols[k] - bucket * VEC_SIZE;

                if (bucket != prev_bucket) {
                    assert(prev_bucket < bucket);
                    prev_bucket                                       = bucket;
                    d_acrs_colidx[d_acrs_rowptr[i] + ++bucket_offset] = bucket;
                }

                const ptrdiff_t idx = (d_acrs_rowptr[i] + bucket_offset) * VEC_SIZE + rmnd;
                d_acrs_values[idx] += crs_values[k];
            }
        }

        auto acrs              = std::make_shared<ACRS_t>();
        acrs->rowptr           = acrs_rowptr;
        acrs->colidx           = acrs_colidx;
        acrs->values           = acrs_values;
        acrs->execution_space_ = es;

        size_t crs_bytes  = rowptr->nbytes() + colidx->nbytes() + values->nbytes();
        size_t acrs_bytes = acrs->nbytes();

        if (false) {
            printf("CRS KB: %g\n", crs_bytes / 1024.);
            printf("ACRS KB: %g\n", acrs_bytes / 1024.);
            printf("ACRS/CRS: %f\n", static_cast<double>(acrs_bytes) / crs_bytes);
        }

        return acrs;
    }
}  // namespace sfem

#endif  // ACRS_HPP
