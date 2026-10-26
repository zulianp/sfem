#include "sfem_ConjugateGradientKernels_cuda.hpp"

#include "sfem_base.hpp"
#include "sfem_cuda_base.hpp"
#include "sfem_reduction_scratch_cuda.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>

#define SFEM_CG_N_WARPS_PER_BLOCK 4

namespace sfem {

    namespace {

        inline __device__ unsigned int cg_lane_id() { return threadIdx.x % SFEM_WARP_SIZE; }

        template <typename T>
        __device__ T cg_warp_reduce_32(const T in) {
            static_assert(SFEM_WARP_SIZE == 32, "Only implemented for CUDA!");
            T out = in;
            out += __shfl_xor_sync(SFEM_WARP_FULL_MASK, out, 16, SFEM_WARP_SIZE);
            out += __shfl_xor_sync(SFEM_WARP_FULL_MASK, out, 8, SFEM_WARP_SIZE);
            out += __shfl_xor_sync(SFEM_WARP_FULL_MASK, out, 4, SFEM_WARP_SIZE);
            out += __shfl_xor_sync(SFEM_WARP_FULL_MASK, out, 2, SFEM_WARP_SIZE);
            out += __shfl_xor_sync(SFEM_WARP_FULL_MASK, out, 1, SFEM_WARP_SIZE);
            return out;
        }

        template <typename T>
        __global__ void update_x_r_and_rtr_kernel(const ptrdiff_t              n,
                                                 const T                      alpha,
                                                 const T* const SFEM_RESTRICT p,
                                                 const T* const SFEM_RESTRICT Ap,
                                                 T* const SFEM_RESTRICT       x,
                                                 T* const SFEM_RESTRICT       r,
                                                 T* SFEM_RESTRICT             result) {
            __shared__ T block_accumulator[SFEM_WARP_SIZE];

            T acc = 0;
            for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                x[i] += alpha * p[i];
                const T ri = r[i] - alpha * Ap[i];
                r[i]       = ri;
                acc += ri * ri;
            }

            acc                        = cg_warp_reduce_32(acc);
            const unsigned int warp_id = threadIdx.x / SFEM_WARP_SIZE;
            const unsigned int lid     = cg_lane_id();
            const unsigned int n_warps = (blockDim.x + SFEM_WARP_SIZE - 1) / SFEM_WARP_SIZE;

            if (!lid) {
                block_accumulator[warp_id] = acc;
            }

            __syncthreads();

            if (!warp_id) {
                acc = lid < n_warps ? block_accumulator[lid] : T(0);
                acc = cg_warp_reduce_32(acc);
                if (!threadIdx.x) {
                    atomicAdd(result, acc);
                }
            }
        }

        template <typename T>
        __global__ void update_x_r_kernel(const ptrdiff_t              n,
                                          const T                      alpha,
                                          const T* const SFEM_RESTRICT p,
                                          const T* const SFEM_RESTRICT Ap,
                                          T* const SFEM_RESTRICT       x,
                                          T* const SFEM_RESTRICT       r) {
            for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                x[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
            }
        }

        template <typename T>
        __global__ void update_p_kernel(const ptrdiff_t              n,
                                        const T                      beta,
                                        const T* const SFEM_RESTRICT z,
                                        T* const SFEM_RESTRICT       p) {
            for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                p[i] = z[i] + beta * p[i];
            }
        }

        static inline void cg_launch_params(const ptrdiff_t n, int& block_size, ptrdiff_t& n_blocks) {
            block_size = SFEM_WARP_SIZE * SFEM_CG_N_WARPS_PER_BLOCK;
            n_blocks   = std::max(ptrdiff_t(1), (n + block_size - 1) / block_size);
            const int max_grid = sfem_cuda_max_grid_dim_x();
            if (n_blocks > max_grid) {
                n_blocks = max_grid;
            }
        }

        template <typename T>
        static T update_x_r_and_rtr(const ptrdiff_t    n,
                                    const T            alpha,
                                    const T* const     p,
                                    const T* const     Ap,
                                    T* const           x,
                                    T* const           r,
                                    const cudaStream_t stream) {
            const auto scratch = reduction_scratch<T>(1);
            int        block_size;
            ptrdiff_t  n_blocks;
            cg_launch_params(n, block_size, n_blocks);

            SFEM_CUDA_CHECK(cudaMemsetAsync(scratch.device, 0, sizeof(T), stream));
            update_x_r_and_rtr_kernel<<<n_blocks, block_size, 0, stream>>>(n, alpha, p, Ap, x, r, scratch.device);
            SFEM_DEBUG_SYNCHRONIZE(stream);

            // One scalar D2H only — required for host-side CG control (beta / convergence).
            SFEM_CUDA_CHECK(cudaMemcpyAsync(scratch.host, scratch.device, sizeof(T), cudaMemcpyDeviceToHost, stream));
            SFEM_CUDA_CHECK(cudaStreamSynchronize(stream));
            return scratch.host[0];
        }

        template <typename T>
        static void update_x_r(const ptrdiff_t    n,
                               const T            alpha,
                               const T* const     p,
                               const T* const     Ap,
                               T* const           x,
                               T* const           r,
                               const cudaStream_t stream) {
            int       block_size;
            ptrdiff_t n_blocks;
            cg_launch_params(n, block_size, n_blocks);
            update_x_r_kernel<<<n_blocks, block_size, 0, stream>>>(n, alpha, p, Ap, x, r);
            SFEM_DEBUG_SYNCHRONIZE(stream);
            // No device↔host scalar traffic.
        }

        template <typename T>
        static void update_p(const ptrdiff_t n, const T beta, const T* const z, T* const p, const cudaStream_t stream) {
            int       block_size;
            ptrdiff_t n_blocks;
            cg_launch_params(n, block_size, n_blocks);
            update_p_kernel<<<n_blocks, block_size, 0, stream>>>(n, beta, z, p);
            SFEM_DEBUG_SYNCHRONIZE(stream);
            // No device↔host scalar traffic.
        }

    }  // namespace

    template <typename T>
    void GPUConjugateGradientKernels<T>::build(struct ConjugateGradientKernels<T>& tpl, const ExecutionContext& context) {
        // Every pass on the context's stream, bound here once.
        const cudaStream_t s   = static_cast<cudaStream_t>(context.stream());
        tpl.update_x_r_and_rtr = [s](auto&&... a) { return update_x_r_and_rtr<T>(a..., s); };
        tpl.update_x_r         = [s](auto&&... a) { update_x_r<T>(a..., s); };
        tpl.update_p           = [s](auto&&... a) { update_p<T>(a..., s); };
    }

    template class GPUConjugateGradientKernels<float>;
    template class GPUConjugateGradientKernels<double>;

}  // namespace sfem
