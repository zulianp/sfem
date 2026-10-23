#include "sfem_cuda_BSR.hpp"

#include "sfem_base.hpp"
#include "sfem_cuda_base.hpp"

#include <algorithm>

#include "sfem_cuda_max.hpp"


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

        template <typename R, typename C, typename TStorage, typename T>
        __global__ void bsr_abs_row_sum_kernel(const ptrdiff_t                     block_rows,
                                               const int                           block_size,
                                               const R* const SFEM_RESTRICT        rowptr,
                                               const C* const SFEM_RESTRICT        colidx,
                                               const TStorage* const SFEM_RESTRICT values,
                                               T* const SFEM_RESTRICT              out) {
            const ptrdiff_t rows = block_rows * block_size;
            for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < rows; i += blockDim.x * gridDim.x) {
                const ptrdiff_t row = i / block_size;
                const int       r   = i % block_size;
                T               sum = 0;
                for (R k = rowptr[row]; k < rowptr[row + 1]; ++k) {
                    const TStorage* const B = &values[(ptrdiff_t)k * block_size * block_size];
                    for (int c = 0; c < block_size; ++c) sum += fabs((T)B[r * block_size + c]);
                }
                out[i] = sum;
            }
        }

        template <typename R, typename C, typename T>
        __global__ void bsr_shift_diagonal_kernel(const ptrdiff_t              block_rows,
                                                  const int                    block_size,
                                                  const R* const SFEM_RESTRICT rowptr,
                                                  const C* const SFEM_RESTRICT colidx,
                                                  const T                      factor,
                                                  const T* const SFEM_RESTRICT reference,
                                                  T* const SFEM_RESTRICT       values) {
            for (ptrdiff_t row = blockIdx.x * blockDim.x + threadIdx.x; row < block_rows; row += blockDim.x * gridDim.x) {
                for (R k = rowptr[row]; k < rowptr[row + 1]; ++k) {
                    if ((ptrdiff_t)colidx[k] != row) continue;
                    T* const B = &values[(ptrdiff_t)k * block_size * block_size];
                    for (int d = 0; d < block_size; ++d) B[d * block_size + d] += factor * fabs(reference[row * block_size + d]);
                }
            }
        }

        template <typename R, typename C, typename TStorage, typename T>
        __global__ void bsr_spmv_kernel(const ptrdiff_t                     block_rows,
                                        const int                           block_size,
                                        const R* const SFEM_RESTRICT        rowptr,
                                        const C* const SFEM_RESTRICT        colidx,
                                        const TStorage* const SFEM_RESTRICT values,
                                        const T                             scale_output,
                                        const T* const SFEM_RESTRICT        x,
                                        T* const SFEM_RESTRICT              y) {
            const ptrdiff_t rows = block_rows * block_size;
            for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < rows; i += blockDim.x * gridDim.x) {
                const ptrdiff_t row = i / block_size;
                const int       r   = i % block_size;
                T               acc = 0;
                for (R k = rowptr[row]; k < rowptr[row + 1]; ++k) {
                    const TStorage* const B   = &values[(ptrdiff_t)k * block_size * block_size + r * block_size];
                    const T* const        xj  = &x[(ptrdiff_t)colidx[k] * block_size];
                    for (int c = 0; c < block_size; ++c) acc += (T)B[c] * xj[c];
                }
                y[i] = (scale_output == T(0) ? T(0) : scale_output * y[i]) + acc;
            }
        }
    }  // namespace

    template <typename R, typename C, typename TStorage, typename T>
    int d_bsr_spmv_native(const ptrdiff_t                     block_rows,
                          const int                           block_size,
                          const R* const SFEM_RESTRICT        rowptr,
                          const C* const SFEM_RESTRICT        colidx,
                          const TStorage* const SFEM_RESTRICT values,
                          const T                             scale_output,
                          const T* const SFEM_RESTRICT        x,
                          T* const SFEM_RESTRICT              y) {
        const ptrdiff_t rows              = block_rows * block_size;
        const int       kernel_block_size = 128;
        const ptrdiff_t n_blocks          = std::max(ptrdiff_t(1), (rows + kernel_block_size - 1) / kernel_block_size);
        bsr_spmv_kernel<R, C, TStorage, T><<<n_blocks, kernel_block_size, 0>>>(block_rows, block_size, rowptr, colidx, values, scale_output, x, y);
        SFEM_DEBUG_SYNCHRONIZE();
        return SFEM_SUCCESS;
    }

    template int d_bsr_spmv_native<count_t, idx_t, real_t, real_t>(const ptrdiff_t,
                                                                   const int,
                                                                   const count_t* const SFEM_RESTRICT,
                                                                   const idx_t* const SFEM_RESTRICT,
                                                                   const real_t* const SFEM_RESTRICT,
                                                                   const real_t,
                                                                   const real_t* const SFEM_RESTRICT,
                                                                   real_t* const SFEM_RESTRICT);

    template <typename R, typename C, typename TStorage, typename T>
    T d_bsr_max_abs_row_sum(const ptrdiff_t                     block_rows,
                            const int                           block_size,
                            const R* const SFEM_RESTRICT        rowptr,
                            const C* const SFEM_RESTRICT        colidx,
                            const TStorage* const SFEM_RESTRICT values) {
        const ptrdiff_t rows = block_rows * block_size;
        if (rows <= 0) return T(0);
        T *sums = nullptr;
        SFEM_CUDA_CHECK(cudaMalloc(&sums, rows * sizeof(T)));
        const int       kernel_block_size = 128;
        const ptrdiff_t n_blocks          = std::max(ptrdiff_t(1), (rows + kernel_block_size - 1) / kernel_block_size);
        bsr_abs_row_sum_kernel<R, C, TStorage, T><<<n_blocks, kernel_block_size, 0>>>(block_rows, block_size, rowptr, colidx, values, sums);
        const T ret = d_max(rows, sums, T(0));
        SFEM_CUDA_CHECK(cudaFree(sums));
        return ret;
    }

    template <typename R, typename C, typename T>
    int d_bsr_shift_diagonal(const ptrdiff_t              block_rows,
                             const int                    block_size,
                             const R* const SFEM_RESTRICT rowptr,
                             const C* const SFEM_RESTRICT colidx,
                             const T                      factor,
                             const T* const SFEM_RESTRICT reference,
                             T* const SFEM_RESTRICT       values) {
        const int       kernel_block_size = 128;
        const ptrdiff_t n_blocks          = std::max(ptrdiff_t(1), (block_rows + kernel_block_size - 1) / kernel_block_size);
        bsr_shift_diagonal_kernel<R, C, T><<<n_blocks, kernel_block_size, 0>>>(block_rows, block_size, rowptr, colidx, factor, reference, values);
        SFEM_DEBUG_SYNCHRONIZE();
        return SFEM_SUCCESS;
    }

    template real_t d_bsr_max_abs_row_sum<count_t, idx_t, real_t, real_t>(const ptrdiff_t,
                                                                          const int,
                                                                          const count_t* const SFEM_RESTRICT,
                                                                          const idx_t* const SFEM_RESTRICT,
                                                                          const real_t* const SFEM_RESTRICT);
    template int d_bsr_shift_diagonal<count_t, idx_t, real_t>(const ptrdiff_t,
                                                              const int,
                                                              const count_t* const SFEM_RESTRICT,
                                                              const idx_t* const SFEM_RESTRICT,
                                                              const real_t,
                                                              const real_t* const SFEM_RESTRICT,
                                                              real_t* const SFEM_RESTRICT);

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
