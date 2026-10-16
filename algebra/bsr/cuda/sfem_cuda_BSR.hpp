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

    /// y = scale_output y + A x on the device, A in BSR with square blocks: one thread per scalar
    /// row, no library. scale_output 0 overwrites y.
    template <typename R, typename C, typename TStorage, typename T>
    int d_bsr_spmv_native(const ptrdiff_t                     block_rows,
                          const int                           block_size,
                          const R* const SFEM_RESTRICT        rowptr,
                          const C* const SFEM_RESTRICT        colidx,
                          const TStorage* const SFEM_RESTRICT values,
                          const T                             scale_output,
                          const T* const SFEM_RESTRICT        x,
                          T* const SFEM_RESTRICT              y);

    /// Device version of `bsr_max_abs_row_sum`: max over the scalar rows of sum_j |A_ij|.
    template <typename R, typename C, typename TStorage, typename T>
    T d_bsr_max_abs_row_sum(const ptrdiff_t                     block_rows,
                            const int                           block_size,
                            const R* const SFEM_RESTRICT        rowptr,
                            const C* const SFEM_RESTRICT        colidx,
                            const TStorage* const SFEM_RESTRICT values);

    /// Device version of `bsr_shift_diagonal`: A_dd += factor |reference_d|.
    template <typename R, typename C, typename T>
    int d_bsr_shift_diagonal(const ptrdiff_t              block_rows,
                             const int                    block_size,
                             const R* const SFEM_RESTRICT rowptr,
                             const C* const SFEM_RESTRICT colidx,
                             const T                      factor,
                             const T* const SFEM_RESTRICT reference,
                             T* const SFEM_RESTRICT       values);
}  // namespace sfem

#endif  // SFEM_CUDA_BSR_HPP
