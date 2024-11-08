#ifndef SFEM_CUDA_CRSSPMV_HPP
#define SFEM_CUDA_CRSSPMV_HPP

#include "sfem_defs.h"
#include "sfem_crs_SpMV.hpp"
#include "sfem_bsr_SpMV.hpp"

namespace sfem {
    std::shared_ptr<CRSSpMV<count_t, idx_t, real_t>> d_crs_spmv(
            const ptrdiff_t rows,
            const ptrdiff_t cols,
            const std::shared_ptr<Buffer<count_t>> &rowptr,
            const std::shared_ptr<Buffer<idx_t>> &colidx,
            const std::shared_ptr<Buffer<real_t>> &values,
            const real_t scale_output);

    std::shared_ptr<BSRSpMV<count_t, idx_t, real_t>> d_bsr_spmv(
            const ptrdiff_t rows,
            const ptrdiff_t cols,
            const int block_size,
            const std::shared_ptr<Buffer<count_t>> &rowptr,
            const std::shared_ptr<Buffer<idx_t>> &colidx,
            const std::shared_ptr<Buffer<real_t>> &values,
            const real_t scale_output);
}

#endif  // SFEM_CUDA_CRSSPMV_HPP
