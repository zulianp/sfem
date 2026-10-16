#include "sfem_cuda_BSR.hpp"

#include "sfem_base.hpp"
#include "sfem_cuda_base.hpp"

#include <algorithm>

namespace sfem {
    namespace {
        template <typename R, typename C, typename TStorage, typename T>
        __global__ void bsr_extract_diagonal_kernel(const ptrdiff_t                     block_rows,
                                                    const int                           block_size,
                                                    const R* const SFEM_RESTRICT        rowptr,
                                                    const C* const SFEM_RESTRICT        colidx,
                                                    const TStorage* const SFEM_RESTRICT values,
                                                    T* const SFEM_RESTRICT              out) {
            for (ptrdiff_t row = blockIdx.x * blockDim.x + threadIdx.x; row < block_rows; row += blockDim.x * gridDim.x) {
                const TStorage* B = nullptr;
                for (R k = rowptr[row]; k < rowptr[row + 1]; ++k) {
                    if ((ptrdiff_t)colidx[k] == row) {
                        B = &values[(ptrdiff_t)k * block_size * block_size];
                        break;
                    }
                }
                for (int d = 0; d < block_size; ++d) {
                    out[row * block_size + d] = B ? (T)B[d * block_size + d] : T(0);
                }
            }
        }
    }  // namespace

    template <typename R, typename C, typename TStorage, typename T>
    int d_bsr_extract_diagonal(const ptrdiff_t                     block_rows,
                               const int                           block_size,
                               const R* const SFEM_RESTRICT        rowptr,
                               const C* const SFEM_RESTRICT        colidx,
                               const TStorage* const SFEM_RESTRICT values,
                               T* const SFEM_RESTRICT              out) {
        SFEM_DEBUG_SYNCHRONIZE();

        const int       kernel_block_size = 128;
        const ptrdiff_t n_blocks = std::max(ptrdiff_t(1), (block_rows + kernel_block_size - 1) / kernel_block_size);
        bsr_extract_diagonal_kernel<R, C, TStorage, T>
                <<<n_blocks, kernel_block_size, 0>>>(block_rows, block_size, rowptr, colidx, values, out);

        SFEM_DEBUG_SYNCHRONIZE();
        return SFEM_SUCCESS;
    }

    template int d_bsr_extract_diagonal<count_t, idx_t, real_t, real_t>(const ptrdiff_t,
                                                                        const int,
                                                                        const count_t* const SFEM_RESTRICT,
                                                                        const idx_t* const SFEM_RESTRICT,
                                                                        const real_t* const SFEM_RESTRICT,
                                                                        real_t* const SFEM_RESTRICT);
}  // namespace sfem
