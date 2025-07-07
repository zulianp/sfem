#ifndef SFEM_CUDA_CRSSPMV_HPP
#define SFEM_CUDA_CRSSPMV_HPP

#include "sfem_CooSym.hpp"
#include "sfem_bsr_SpMV.hpp"
#include "sfem_crs_SpMV.hpp"
#include "sfem_crs_sym_SpMV.hpp"
#include "sfem_defs.h"

namespace sfem {
    std::shared_ptr<CRSSpMV<count_t, idx_t, real_t>> d_crs_spmv(const ptrdiff_t              rows,
                                                                const ptrdiff_t              cols,
                                                                const SharedBuffer<count_t>& rowptr,
                                                                const SharedBuffer<idx_t>&   colidx,
                                                                const SharedBuffer<real_t>&  values,
                                                                const real_t                 scale_output);

    std::shared_ptr<CooSymSpMV<idx_t, real_t>> d_sym_coo_spmv(const ptrdiff_t             ndofs,
                                                              const SharedBuffer<idx_t>&  rowidx,
                                                              const SharedBuffer<idx_t>&  colidx,
                                                              const SharedBuffer<real_t>& values,
                                                              const SharedBuffer<real_t>& diag_values,
                                                              const real_t                scale_output);

    std::shared_ptr<BSRSpMV<count_t, idx_t, real_t>> d_bsr_spmv(const ptrdiff_t              block_rows,
                                                                const ptrdiff_t              block_cols,
                                                                const int                    block_size,
                                                                const SharedBuffer<count_t>& rowptr,
                                                                const SharedBuffer<idx_t>&   colidx,
                                                                const SharedBuffer<real_t>&  values,
                                                                const real_t                 scale_output);
}  // namespace sfem

#endif  // SFEM_CUDA_CRSSPMV_HPP
