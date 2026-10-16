#ifndef SFEM_CUDA_BSR_HPP
#define SFEM_CUDA_BSR_HPP

#include "sfem_base.hpp"

namespace sfem {
    /// Device version of `bsr_extract_diagonal` (sfem_BSR.hpp): all pointers on the device.
    template <typename R, typename C, typename TStorage, typename T>
    int d_bsr_extract_diagonal(const ptrdiff_t                     block_rows,
                               const int                           block_size,
                               const R* const SFEM_RESTRICT        rowptr,
                               const C* const SFEM_RESTRICT        colidx,
                               const TStorage* const SFEM_RESTRICT values,
                               T* const SFEM_RESTRICT              out);
}  // namespace sfem

#endif  // SFEM_CUDA_BSR_HPP
