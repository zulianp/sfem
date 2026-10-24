#include "sfem_cuda_BSR.hpp"

#include "sfem_base.hpp"
#include "sfem_cuda_base.hpp"

#include <algorithm>

#include "sfem_cuda_reduction_scratch.hpp"


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
        __global__ void bsr_max_abs_row_sum_kernel(const ptrdiff_t                     block_rows,
                                                   const int                           block_size,
                                                   const R* const SFEM_RESTRICT        rowptr,
                                                   const C* const SFEM_RESTRICT        colidx,
                                                   const TStorage* const SFEM_RESTRICT values,
                                                   const T* const SFEM_RESTRICT        scale,
                                                   T* const SFEM_RESTRICT              partial) {
            // Each thread's largest row sum, then the block's, into partial[blockIdx.x].
            __shared__ T    cache[128];
            T               m    = 0;
            const ptrdiff_t rows = block_rows * block_size;
            for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < rows; i += blockDim.x * gridDim.x) {
                const ptrdiff_t row = i / block_size;
                const int       r   = i % block_size;
                T               sum = 0;
                for (R k = rowptr[row]; k < rowptr[row + 1]; ++k) {
                    const TStorage* const B = &values[(ptrdiff_t)k * block_size * block_size];
                    for (int c = 0; c < block_size; ++c) {
                        const T a = fabs((T)B[r * block_size + c]);
                        sum += scale ? a * scale[(ptrdiff_t)colidx[k] * block_size + c] : a;
                    }
                }
                m = fmax(m, scale ? sum * scale[i] : sum);
            }
            cache[threadIdx.x] = m;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s /= 2) {
                if (threadIdx.x < s) cache[threadIdx.x] = fmax(cache[threadIdx.x], cache[threadIdx.x + s]);
                __syncthreads();
            }
            if (threadIdx.x == 0) partial[blockIdx.x] = cache[0];
        }

        template <typename R, typename C, typename TStorage, typename T>
        __global__ void bsr_shift_diagonal_kernel(const ptrdiff_t               block_rows,
                                                  const int                     block_size,
                                                  const R* const SFEM_RESTRICT  rowptr,
                                                  const C* const SFEM_RESTRICT  colidx,
                                                  const T                       factor,
                                                  const T* const SFEM_RESTRICT  reference,
                                                  TStorage* const SFEM_RESTRICT values) {
            for (ptrdiff_t row = blockIdx.x * blockDim.x + threadIdx.x; row < block_rows; row += blockDim.x * gridDim.x) {
                for (R k = rowptr[row]; k < rowptr[row + 1]; ++k) {
                    if ((ptrdiff_t)colidx[k] != row) continue;
                    TStorage* const B = &values[(ptrdiff_t)k * block_size * block_size];
                    for (int d = 0; d < block_size; ++d) {
                        B[d * block_size + d] = (TStorage)(B[d * block_size + d] + factor * fabs(reference[row * block_size + d]));
                    }
                }
            }
        }

        template <typename R, typename C, typename TStorage, typename T>
        __global__ void bsr_spmv_kernel(const ptrdiff_t                     block_rows,
                                        const int                           block_size,
                                        const R* const SFEM_RESTRICT        rows_begin,
                                        const R* const SFEM_RESTRICT        rows_end,
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
                for (R k = rows_begin[row]; k < rows_end[row]; ++k) {
                    const TStorage* const B   = &values[(ptrdiff_t)k * block_size * block_size + r * block_size];
                    const T* const        xj  = &x[(ptrdiff_t)colidx[k] * block_size];
                    for (int c = 0; c < block_size; ++c) acc += (T)B[c] * xj[c];
                }
                y[i] = (scale_output == T(0) ? T(0) : scale_output * y[i]) + acc;
            }
        }
    }  // namespace

    template <typename R, typename C, typename TStorage, typename T>
    int d_bsr_spmv_ranges(const ptrdiff_t                     block_rows,
                          const int                           block_size,
                          const R* const SFEM_RESTRICT        rows_begin,
                          const R* const SFEM_RESTRICT        rows_end,
                          const C* const SFEM_RESTRICT        colidx,
                          const TStorage* const SFEM_RESTRICT values,
                          const T                             scale_output,
                          const T* const SFEM_RESTRICT        x,
                          T* const SFEM_RESTRICT              y,
                          void* const                         stream) {
        const ptrdiff_t rows = block_rows * block_size;
        if (rows <= 0) return SFEM_SUCCESS;
        const int       kernel_block_size = 128;
        const ptrdiff_t n_blocks          = std::max(ptrdiff_t(1), (rows + kernel_block_size - 1) / kernel_block_size);
        // On the caller's stream, or the thread's launch stream (a capture's) without one.
        bsr_spmv_kernel<R, C, TStorage, T><<<n_blocks, kernel_block_size, 0, stream ? (cudaStream_t)stream : launch_stream()>>>(
                block_rows, block_size, rows_begin, rows_end, colidx, values, scale_output, x, y);
        SFEM_DEBUG_SYNCHRONIZE();
        return SFEM_SUCCESS;
    }

    template <typename R, typename C, typename TStorage, typename T>
    int d_bsr_spmv_native(const ptrdiff_t                     block_rows,
                          const int                           block_size,
                          const R* const SFEM_RESTRICT        rowptr,
                          const C* const SFEM_RESTRICT        colidx,
                          const TStorage* const SFEM_RESTRICT values,
                          const T                             scale_output,
                          const T* const SFEM_RESTRICT        x,
                          T* const SFEM_RESTRICT              y) {
        return d_bsr_spmv_ranges(block_rows, block_size, rowptr, rowptr + 1, colidx, values, scale_output, x, y);
    }

    class DeviceSideStream::Impl {
    public:
        cudaStream_t stream{nullptr};
        cudaEvent_t  on_default{nullptr}, on_side{nullptr};
    };

    DeviceSideStream::DeviceSideStream() : impl_(std::make_unique<Impl>()) {}

    DeviceSideStream::~DeviceSideStream() {
        if (impl_->on_default) cudaEventDestroy(impl_->on_default);
        if (impl_->on_side) cudaEventDestroy(impl_->on_side);
        if (impl_->stream) cudaStreamDestroy(impl_->stream);
    }

    std::shared_ptr<DeviceSideStream> DeviceSideStream::create() {
        std::shared_ptr<DeviceSideStream> ret(new DeviceSideStream());
        SFEM_CUDA_CHECK(cudaStreamCreateWithFlags(&ret->impl_->stream, cudaStreamNonBlocking));
        SFEM_CUDA_CHECK(cudaEventCreateWithFlags(&ret->impl_->on_default, cudaEventDisableTiming));
        SFEM_CUDA_CHECK(cudaEventCreateWithFlags(&ret->impl_->on_side, cudaEventDisableTiming));
        return ret;
    }

    void *DeviceSideStream::stream() const { return impl_->stream; }

    void *current_launch_stream() { return (void *)launch_stream(); }

    void DeviceSideStream::fork(void *const from) {
        SFEM_CUDA_CHECK(cudaEventRecord(impl_->on_default, from ? static_cast<cudaStream_t>(from) : cudaStreamLegacy));
        SFEM_CUDA_CHECK(cudaStreamWaitEvent(impl_->stream, impl_->on_default, 0));
    }

    void DeviceSideStream::join(void *const into) {
        SFEM_CUDA_CHECK(cudaEventRecord(impl_->on_side, impl_->stream));
        SFEM_CUDA_CHECK(cudaStreamWaitEvent(into ? static_cast<cudaStream_t>(into) : cudaStreamLegacy, impl_->on_side, 0));
    }

    template <typename R, typename C, typename TStorage, typename T>
    T d_bsr_max_abs_row_sum(const ptrdiff_t                     block_rows,
                            const int                           block_size,
                            const R* const SFEM_RESTRICT        rowptr,
                            const C* const SFEM_RESTRICT        colidx,
                            const TStorage* const SFEM_RESTRICT values,
                            const T* const SFEM_RESTRICT        scale) {
        const ptrdiff_t rows = block_rows * block_size;
        if (rows <= 0) return T(0);
        // Block maxima of the row sums into the reduction scratch (the kernel's shared cache holds
        // 128 threads), then their maximum on the host: no allocation, no array of row sums.
        const int  kernel_block_size = 128;
        const int  n_blocks          = (int)std::min<ptrdiff_t>(1024, (rows + kernel_block_size - 1) / kernel_block_size);
        const auto partial           = reduction_scratch<T>(n_blocks);
        bsr_max_abs_row_sum_kernel<R, C, TStorage, T>
                <<<n_blocks, kernel_block_size, 0>>>(block_rows, block_size, rowptr, colidx, values, scale, partial.device);
        SFEM_CUDA_CHECK(cudaMemcpyAsync(partial.host, partial.device, n_blocks * sizeof(T), cudaMemcpyDeviceToHost, 0));
        SFEM_CUDA_CHECK(cudaStreamSynchronize(0));
        T ret = 0;
        for (int b = 0; b < n_blocks; ++b) ret = std::max(ret, partial.host[b]);
        return ret;
    }

    template <typename R, typename C, typename TStorage, typename T>
    int d_bsr_shift_diagonal(const ptrdiff_t               block_rows,
                             const int                     block_size,
                             const R* const SFEM_RESTRICT  rowptr,
                             const C* const SFEM_RESTRICT  colidx,
                             const T                       factor,
                             const T* const SFEM_RESTRICT  reference,
                             TStorage* const SFEM_RESTRICT values) {
        const int       kernel_block_size = 128;
        const ptrdiff_t n_blocks          = std::max(ptrdiff_t(1), (block_rows + kernel_block_size - 1) / kernel_block_size);
        bsr_shift_diagonal_kernel<R, C, TStorage, T><<<n_blocks, kernel_block_size, 0>>>(block_rows, block_size, rowptr, colidx, factor, reference, values);
        SFEM_DEBUG_SYNCHRONIZE();
        return SFEM_SUCCESS;
    }

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

    // The tangent's values in real_t or, assembled in single precision, in float; the vectors in
    // real_t.
#define SFEM_D_BSR_INSTANTIATE(TStorage)                                                                                      \
    template int    d_bsr_spmv_ranges<count_t, idx_t, TStorage, real_t>(const ptrdiff_t,                                     \
                                                                     const int,                                           \
                                                                     const count_t* const SFEM_RESTRICT,                  \
                                                                     const count_t* const SFEM_RESTRICT,                  \
                                                                     const idx_t* const SFEM_RESTRICT,                    \
                                                                     const TStorage* const SFEM_RESTRICT,                 \
                                                                     const real_t,                                        \
                                                                     const real_t* const SFEM_RESTRICT,                   \
                                                                     real_t* const SFEM_RESTRICT,                         \
                                                                     void* const);                                        \
    template int    d_bsr_spmv_native<count_t, idx_t, TStorage, real_t>(const ptrdiff_t,                                     \
                                                                     const int,                                           \
                                                                     const count_t* const SFEM_RESTRICT,                  \
                                                                     const idx_t* const SFEM_RESTRICT,                    \
                                                                     const TStorage* const SFEM_RESTRICT,                 \
                                                                     const real_t,                                        \
                                                                     const real_t* const SFEM_RESTRICT,                   \
                                                                     real_t* const SFEM_RESTRICT);                        \
    template real_t d_bsr_max_abs_row_sum<count_t, idx_t, TStorage, real_t>(const ptrdiff_t,                                 \
                                                                         const int,                                       \
                                                                         const count_t* const SFEM_RESTRICT,              \
                                                                         const idx_t* const SFEM_RESTRICT,                \
                                                                         const TStorage* const SFEM_RESTRICT,             \
                                                                         const real_t* const SFEM_RESTRICT);              \
    template int    d_bsr_shift_diagonal<count_t, idx_t, TStorage, real_t>(const ptrdiff_t,                                  \
                                                                        const int,                                        \
                                                                        const count_t* const SFEM_RESTRICT,               \
                                                                        const idx_t* const SFEM_RESTRICT,                 \
                                                                        const real_t,                                     \
                                                                        const real_t* const SFEM_RESTRICT,                \
                                                                        TStorage* const SFEM_RESTRICT);                   \
    template int    d_bsr_extract_diagonal<count_t, idx_t, TStorage, real_t>(const ptrdiff_t,                                \
                                                                          const int,                                      \
                                                                          const count_t* const SFEM_RESTRICT,             \
                                                                          const idx_t* const SFEM_RESTRICT,               \
                                                                          const TStorage* const SFEM_RESTRICT,            \
                                                                          real_t* const SFEM_RESTRICT);

    SFEM_D_BSR_INSTANTIATE(real_t)
#if !SFEM_REAL_T_IS_FLOAT32
    SFEM_D_BSR_INSTANTIATE(float)
#endif
#undef SFEM_D_BSR_INSTANTIATE
}  // namespace sfem
