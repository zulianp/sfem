#include "sfem_cuda_cg_impl.hpp"

#include "sfem_base.hpp"
#include "sfem_cuda_base.hpp"

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

        /// Persistent device + pinned-host scalar to avoid per-call malloc/free and H2D of zeros.
        template <typename T>
        struct CGDeviceScalar {
            T* d_{nullptr};
            T* h_pinned_{nullptr};

            void ensure() {
                if (d_) return;
                SFEM_CUDA_CHECK(cudaMalloc((void**)&d_, sizeof(T)));
                SFEM_CUDA_CHECK(cudaMallocHost((void**)&h_pinned_, sizeof(T)));
                *h_pinned_ = T(0);
            }

            void zero_async() {
                ensure();
                SFEM_CUDA_CHECK(cudaMemsetAsync(d_, 0, sizeof(T)));
            }

            T* device_ptr() {
                ensure();
                return d_;
            }

            /// Single D2H of the scalar (pinned host destination).
            T pull() {
                ensure();
                SFEM_CUDA_CHECK(cudaMemcpy(h_pinned_, d_, sizeof(T), cudaMemcpyDeviceToHost));
                return *h_pinned_;
            }
        };

        template <typename T>
        CGDeviceScalar<T>& cg_scalar_workspace() {
            static CGDeviceScalar<T> ws;
            return ws;
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
        static T update_x_r_and_rtr(const ptrdiff_t n,
                                    const T         alpha,
                                    const T* const  p,
                                    const T* const  Ap,
                                    T* const        x,
                                    T* const        r) {
            auto&     ws = cg_scalar_workspace<T>();
            int       block_size;
            ptrdiff_t n_blocks;
            cg_launch_params(n, block_size, n_blocks);

            // Zero device scalar in-place (no H2D of a host zero).
            ws.zero_async();
            update_x_r_and_rtr_kernel<<<n_blocks, block_size>>>(n, alpha, p, Ap, x, r, ws.device_ptr());
            SFEM_DEBUG_SYNCHRONIZE();

            // One scalar D2H only — required for host-side CG control (beta / convergence).
            return ws.pull();
        }

        template <typename T>
        static void update_x_r(const ptrdiff_t n,
                               const T         alpha,
                               const T* const  p,
                               const T* const  Ap,
                               T* const        x,
                               T* const        r) {
            int       block_size;
            ptrdiff_t n_blocks;
            cg_launch_params(n, block_size, n_blocks);
            update_x_r_kernel<<<n_blocks, block_size>>>(n, alpha, p, Ap, x, r);
            SFEM_DEBUG_SYNCHRONIZE();
            // No device↔host scalar traffic.
        }

        template <typename T>
        static void update_p(const ptrdiff_t n, const T beta, const T* const z, T* const p) {
            int       block_size;
            ptrdiff_t n_blocks;
            cg_launch_params(n, block_size, n_blocks);
            update_p_kernel<<<n_blocks, block_size>>>(n, beta, z, p);
            SFEM_DEBUG_SYNCHRONIZE();
            // No device↔host scalar traffic.
        }

    }  // namespace

    template <typename T>
    void CUDA_CG<T>::build(struct CG_Tpl<T>& tpl) {
        tpl.update_x_r_and_rtr = update_x_r_and_rtr<T>;
        tpl.update_x_r         = update_x_r<T>;
        tpl.update_p           = update_p<T>;
    }

    template class CUDA_CG<float>;
    template class CUDA_CG<double>;

}  // namespace sfem
