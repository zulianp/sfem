#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>

#include "sfem_base.hpp"
#include "sfem_ShiftedPenaltyKernels_cuda.hpp"

#include "sfem_cuda_base.hpp"
#include "sfem_reduction_scratch_cuda.hpp"

#define SFEM_N_WARPS_PER_BLOCK 8

#ifndef MAX
#define MAX(a, b) ((a < b) ? (b) : (a))
#endif

namespace sfem {

    inline __device__ unsigned int lane_id() { return threadIdx.x % SFEM_WARP_SIZE; }

    template <typename T>
    __device__ T warp_reduce_32(const T in) {
        static_assert(SFEM_WARP_SIZE == 32, "Only implemented for CUDA!");
        T out = in;
        out += __shfl_xor_sync(SFEM_WARP_FULL_MASK, out, 16, SFEM_WARP_SIZE);  // 0-16, 1-17, ..., 15-31
        out += __shfl_xor_sync(SFEM_WARP_FULL_MASK, out, 8, SFEM_WARP_SIZE);   // 0-8, ..., 1-7, ..., 23-31
        out += __shfl_xor_sync(SFEM_WARP_FULL_MASK, out, 4, SFEM_WARP_SIZE);
        out += __shfl_xor_sync(SFEM_WARP_FULL_MASK, out, 2, SFEM_WARP_SIZE);
        out += __shfl_xor_sync(SFEM_WARP_FULL_MASK, out, 1, SFEM_WARP_SIZE);
        return out;
    }

    template <typename T>
    inline static __device__ __host__ T tmin(const T a, const T b) {
        return (a < b) ? a : b;
    }

    template <typename T>
    inline static __device__ __host__ T tmax(const T a, const T b) {
        return (a > b) ? a : b;
    }

    template <typename T>
    __device__ T warp_min_32(const T in) {
        static_assert(SFEM_WARP_SIZE == 32, "Only implemented for CUDA!");
        T out = in;
        out   = tmin(out, __shfl_xor_sync(SFEM_WARP_FULL_MASK, out, 16, SFEM_WARP_SIZE));  // 0-16, 1-17, ..., 15-31
        out   = tmin(out, __shfl_xor_sync(SFEM_WARP_FULL_MASK, out, 8, SFEM_WARP_SIZE));   // 0-8, ..., 1-7, ..., 23-31
        out   = tmin(out, __shfl_xor_sync(SFEM_WARP_FULL_MASK, out, 4, SFEM_WARP_SIZE));
        out   = tmin(out, __shfl_xor_sync(SFEM_WARP_FULL_MASK, out, 2, SFEM_WARP_SIZE));
        out   = tmin(out, __shfl_xor_sync(SFEM_WARP_FULL_MASK, out, 1, SFEM_WARP_SIZE));
        return out;
    }

    template <typename T>
    __device__ T warp_max_32(const T in) {
        static_assert(SFEM_WARP_SIZE == 32, "Only implemented for CUDA!");
        T out = in;
        out   = tmax(out, __shfl_xor_sync(SFEM_WARP_FULL_MASK, out, 16, SFEM_WARP_SIZE));  // 0-16, 1-17, ..., 15-31
        out   = tmax(out, __shfl_xor_sync(SFEM_WARP_FULL_MASK, out, 8, SFEM_WARP_SIZE));   // 0-8, ..., 1-7, ..., 23-31
        out   = tmax(out, __shfl_xor_sync(SFEM_WARP_FULL_MASK, out, 4, SFEM_WARP_SIZE));
        out   = tmax(out, __shfl_xor_sync(SFEM_WARP_FULL_MASK, out, 2, SFEM_WARP_SIZE));
        out   = tmax(out, __shfl_xor_sync(SFEM_WARP_FULL_MASK, out, 1, SFEM_WARP_SIZE));
        return out;
    }

#define SFEM_N_WARPS_PER_BLOCK 8

    template <typename T>
    __device__ void t_warp_reduce(const T val, T* block_accumulator, T* result) {
        T                  acc     = warp_reduce_32(val);
        const unsigned int warp_id = threadIdx.x / SFEM_WARP_SIZE;
        const unsigned int lid     = lane_id();
        const unsigned int n_warps = (blockDim.x + SFEM_WARP_SIZE - 1) / SFEM_WARP_SIZE;

        if (!lid) {
            block_accumulator[warp_id] = acc;
        }

        __syncthreads();

        if (!warp_id) {
            assert(warp_id < SFEM_N_WARPS_PER_BLOCK);
            acc = lid < n_warps ? block_accumulator[lid] : 0;
            acc = warp_reduce_32(acc);

            if (!threadIdx.x) {
                assert(acc == acc);
                atomicAdd(result, acc);
            }
        }
    }

    template <typename T>
    __global__ void sq_norm_ramp_p_kernel(const ptrdiff_t              n,
                                          const T* const SFEM_RESTRICT x,
                                          const T* const SFEM_RESTRICT ub,
                                          T* const SFEM_RESTRICT       result) {
        T acc = 0;
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
            const T diff = tmax(T(0), x[i] - ub[i]);
            acc += diff * diff;
        }

        __shared__ T block_accumulator[SFEM_N_WARPS_PER_BLOCK];
        t_warp_reduce(acc, block_accumulator, result);
    }

    template <typename T>
    static T sq_norm_ramp_p_tpl(const ptrdiff_t n, const T* const x, T* const ub, const cudaStream_t stream) {
        SFEM_DEBUG_SYNCHRONIZE(stream);

        int       kernel_block_size = SFEM_WARP_SIZE * SFEM_N_WARPS_PER_BLOCK;
        ptrdiff_t n_blocks          = std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

        const auto scratch      = reduction_scratch<T>(1);
        T* const   device_value = scratch.device;
        SFEM_CUDA_CHECK(cudaMemsetAsync(device_value, 0, sizeof(T), stream));

        sq_norm_ramp_p_kernel<<<n_blocks, kernel_block_size, 0, stream>>>(n, x, ub, device_value);
        SFEM_DEBUG_SYNCHRONIZE(stream);

        SFEM_CUDA_CHECK(cudaMemcpyAsync(scratch.host, device_value, sizeof(T), cudaMemcpyDeviceToHost, stream));
        SFEM_CUDA_CHECK(cudaStreamSynchronize(stream));
        const T host_value = scratch.host[0];

        return host_value;
    }

    /////

    template <typename T>
    __global__ void sq_norm_ramp_m_kernel(const ptrdiff_t              n,
                                          const T* const SFEM_RESTRICT x,
                                          const T* const SFEM_RESTRICT lb,
                                          T* const SFEM_RESTRICT       result) {
        T acc = 0;
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
            const T diff = tmin(T(0), x[i] - lb[i]);
            acc += diff * diff;
        }

        __shared__ T block_accumulator[SFEM_N_WARPS_PER_BLOCK];
        t_warp_reduce(acc, block_accumulator, result);
    }

    template <typename T>
    static T sq_norm_ramp_m_tpl(const ptrdiff_t n, const T* const x, T* const lb, const cudaStream_t stream) {
        SFEM_DEBUG_SYNCHRONIZE(stream);

        int       kernel_block_size = SFEM_WARP_SIZE * SFEM_N_WARPS_PER_BLOCK;
        ptrdiff_t n_blocks          = std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

        const auto scratch      = reduction_scratch<T>(1);
        T* const   device_value = scratch.device;
        SFEM_CUDA_CHECK(cudaMemsetAsync(device_value, 0, sizeof(T), stream));

        sq_norm_ramp_m_kernel<<<n_blocks, kernel_block_size, 0, stream>>>(n, x, lb, device_value);
        SFEM_DEBUG_SYNCHRONIZE(stream);

        SFEM_CUDA_CHECK(cudaMemcpyAsync(scratch.host, device_value, sizeof(T), cudaMemcpyDeviceToHost, stream));
        SFEM_CUDA_CHECK(cudaStreamSynchronize(stream));
        const T host_value = scratch.host[0];

        return host_value;
    }

    /////

    template <typename T>
    __global__ void ramp_p_kernel(const ptrdiff_t n,
                                  const T         penalty_param,
                                  const T* const SFEM_RESTRICT weights,
                                  const T* const SFEM_RESTRICT x,
                                  const T* const SFEM_RESTRICT ub,
                                  const T* const SFEM_RESTRICT lagr_ub,
                                  T* const SFEM_RESTRICT out) {
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
            const T rho = weights ? penalty_param * weights[i] : penalty_param;
            out[i] -= tmax(T(0), rho * (x[i] - ub[i]) + lagr_ub[i]);
        }
    }

    template <typename T>
    static void ramp_p_tpl(const ptrdiff_t    n,
                           const T            penalty_param,
                           const T* const     weights,
                           const T* const     x,
                           const T* const     ub,
                           const T* const     lagr_ub,
                           T* const           out,
                           const cudaStream_t stream) {
        SFEM_DEBUG_SYNCHRONIZE(stream);

        int       kernel_block_size = SFEM_WARP_SIZE * SFEM_N_WARPS_PER_BLOCK;
        ptrdiff_t n_blocks          = std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

        ramp_p_kernel<<<n_blocks, kernel_block_size, 0, stream>>>(n, penalty_param, weights, x, ub, lagr_ub, out);

        SFEM_DEBUG_SYNCHRONIZE(stream);
    }

    /////

    template <typename T>
    __global__ void ramp_m_kernel(const ptrdiff_t n,
                                  const T         penalty_param,
                                  const T* const SFEM_RESTRICT weights,
                                  const T* const SFEM_RESTRICT x,
                                  const T* const SFEM_RESTRICT lb,
                                  const T* const SFEM_RESTRICT lagr_lb,
                                  T* const SFEM_RESTRICT out) {
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
            const T rho = weights ? penalty_param * weights[i] : penalty_param;
            out[i] -= tmin(T(0), rho * (x[i] - lb[i]) + lagr_lb[i]);
        }
    }

    template <typename T>
    static void ramp_m_tpl(const ptrdiff_t    n,
                           const T            penalty_param,
                           const T* const     weights,
                           const T* const     x,
                           const T* const     lb,
                           const T* const     lagr_lb,
                           T* const           out,
                           const cudaStream_t stream) {
        SFEM_DEBUG_SYNCHRONIZE(stream);

        int       kernel_block_size = SFEM_WARP_SIZE * SFEM_N_WARPS_PER_BLOCK;
        ptrdiff_t n_blocks          = std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

        ramp_m_kernel<<<n_blocks, kernel_block_size, 0, stream>>>(n, penalty_param, weights, x, lb, lagr_lb, out);

        SFEM_DEBUG_SYNCHRONIZE(stream);
    }

    /////

    template <typename T>
    __global__ void update_lagr_p_kernel(const ptrdiff_t n,
                                         const T         penalty_param,
                                         const T* const SFEM_RESTRICT weights,
                                         const T* const SFEM_RESTRICT x,
                                         const T* const SFEM_RESTRICT ub,
                                         T* const SFEM_RESTRICT lagr_ub) {
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
            const T rho = weights ? penalty_param * weights[i] : penalty_param;
            lagr_ub[i]  = tmax(T(0), lagr_ub[i] + rho * (x[i] - ub[i]));
        }
    }

    template <typename T>
    static void update_lagr_p_tpl(const ptrdiff_t    n,
                                  const T            penalty_param,
                                  const T* const     weights,
                                  const T* const     x,
                                  const T* const     ub,
                                  T* const           lagr_ub,
                                  const cudaStream_t stream) {
        SFEM_DEBUG_SYNCHRONIZE(stream);

        int       kernel_block_size = SFEM_WARP_SIZE * SFEM_N_WARPS_PER_BLOCK;
        ptrdiff_t n_blocks          = std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

        update_lagr_p_kernel<<<n_blocks, kernel_block_size, 0, stream>>>(n, penalty_param, weights, x, ub, lagr_ub);

        SFEM_DEBUG_SYNCHRONIZE(stream);
    }

    /////

    template <typename T>
    __global__ void update_lagr_m_kernel(const ptrdiff_t n,
                                         const T         penalty_param,
                                         const T* const SFEM_RESTRICT weights,
                                         const T* const SFEM_RESTRICT x,
                                         const T* const SFEM_RESTRICT lb,
                                         T* const SFEM_RESTRICT lagr_lb) {
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
            const T rho = weights ? penalty_param * weights[i] : penalty_param;
            lagr_lb[i]  = tmin(T(0), lagr_lb[i] + rho * (x[i] - lb[i]));
        }
    }

    template <typename T>
    static void update_lagr_m_tpl(const ptrdiff_t    n,
                                  const T            penalty_param,
                                  const T* const     weights,
                                  const T* const     x,
                                  const T* const     lb,
                                  T* const           lagr_lb,
                                  const cudaStream_t stream) {
        SFEM_DEBUG_SYNCHRONIZE(stream);

        int       kernel_block_size = SFEM_WARP_SIZE * SFEM_N_WARPS_PER_BLOCK;
        ptrdiff_t n_blocks          = std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

        update_lagr_m_kernel<<<n_blocks, kernel_block_size, 0, stream>>>(n, penalty_param, weights, x, lb, lagr_lb);

        SFEM_DEBUG_SYNCHRONIZE(stream);
    }

    template <typename T>
    __global__ void calc_J_pen_p_kernel(const ptrdiff_t n,
                                        const T* const SFEM_RESTRICT x,
                                        const T         penalty_param,
                                        const T* const SFEM_RESTRICT weights,
                                        const T* const SFEM_RESTRICT ub,
                                        const T* const SFEM_RESTRICT lagr_ub,
                                        T* const SFEM_RESTRICT result) {
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
            const T rho = weights ? penalty_param * weights[i] : penalty_param;
            result[i] += ((rho * (x[i] - ub[i]) + lagr_ub[i]) >= 0) * rho;
        }
    }

    template <typename T>
    __global__ void calc_J_pen_m_kernel(const ptrdiff_t n,
                                        const T* const SFEM_RESTRICT x,
                                        const T         penalty_param,
                                        const T* const SFEM_RESTRICT weights,
                                        const T* const SFEM_RESTRICT lb,
                                        const T* const SFEM_RESTRICT lagr_lb,
                                        T* const SFEM_RESTRICT result) {
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
            const T rho = weights ? penalty_param * weights[i] : penalty_param;
            result[i] += ((rho * (x[i] - lb[i]) + lagr_lb[i]) <= 0) * rho;
        }
    }

    ///
    template <typename T>
    static void calc_J_pen_tpl(const ptrdiff_t    n,
                               const T* const     x,
                               const T            penalty_param,
                               const T* const     weights,
                               const T* const     lb,
                               const T* const     ub,
                               const T* const     lagr_lb,
                               const T* const     lagr_ub,
                               T* const           result,
                               const cudaStream_t stream) {
        SFEM_DEBUG_SYNCHRONIZE(stream);

        int       kernel_block_size = SFEM_WARP_SIZE * SFEM_N_WARPS_PER_BLOCK;
        ptrdiff_t n_blocks          = std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

        if (lb) {
            calc_J_pen_m_kernel<<<n_blocks, kernel_block_size, 0, stream>>>(n, x, penalty_param, weights, lb, lagr_lb, result);
        }

        if (ub) {
            calc_J_pen_p_kernel<<<n_blocks, kernel_block_size, 0, stream>>>(n, x, penalty_param, weights, ub, lagr_ub, result);
        }

        SFEM_DEBUG_SYNCHRONIZE(stream);
    }

    template <typename T, bool UPPER>
    __global__ void sq_norm_kkt_kernel(const ptrdiff_t              n,
                                       const T                      penalty_param,
                                       const T* const SFEM_RESTRICT weights,
                                       const T* const SFEM_RESTRICT x,
                                       const T* const SFEM_RESTRICT bound,
                                       const T* const SFEM_RESTRICT lagr,
                                       T* const SFEM_RESTRICT       result) {
        T acc = 0;
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
            const T rho = weights ? penalty_param * weights[i] : penalty_param;
            const T v   = UPPER ? tmax(x[i] - bound[i], -lagr[i] / rho) : tmin(x[i] - bound[i], -lagr[i] / rho);
            acc += v * v;
        }

        __shared__ T block_accumulator[SFEM_N_WARPS_PER_BLOCK];
        t_warp_reduce(acc, block_accumulator, result);
    }

    template <typename T, bool UPPER>
    static T sq_norm_kkt_tpl(const ptrdiff_t    n,
                             const T            penalty_param,
                             const T* const     weights,
                             const T* const     x,
                             const T* const     bound,
                             const T* const     lagr,
                             const cudaStream_t stream) {
        SFEM_DEBUG_SYNCHRONIZE(stream);

        int       kernel_block_size = SFEM_WARP_SIZE * SFEM_N_WARPS_PER_BLOCK;
        ptrdiff_t n_blocks          = std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

        const auto scratch      = reduction_scratch<T>(1);
        T* const   device_value = scratch.device;
        SFEM_CUDA_CHECK(cudaMemsetAsync(device_value, 0, sizeof(T), stream));

        sq_norm_kkt_kernel<T, UPPER>
                <<<n_blocks, kernel_block_size, 0, stream>>>(n, penalty_param, weights, x, bound, lagr, device_value);
        SFEM_DEBUG_SYNCHRONIZE(stream);

        SFEM_CUDA_CHECK(cudaMemcpyAsync(scratch.host, device_value, sizeof(T), cudaMemcpyDeviceToHost, stream));
        SFEM_CUDA_CHECK(cudaStreamSynchronize(stream));
        const T host_value = scratch.host[0];

        return host_value;
    }

    template <typename T>
    __global__ void scale_blocks_kernel(const ptrdiff_t n_blocks,
                                        const int       block_size,
                                        const T* const SFEM_RESTRICT scale,
                                        const T* const SFEM_RESTRICT blocks,
                                        T* const SFEM_RESTRICT out) {
        const ptrdiff_t n = n_blocks * block_size;
        for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < n; e += blockDim.x * gridDim.x) {
            out[e] = scale[e / block_size] * blocks[e];
        }
    }

    template <typename T>
    static void scale_blocks_tpl(const ptrdiff_t    n_blocks,
                                 const int          block_size,
                                 const T* const     scale,
                                 const T* const     blocks,
                                 T* const           out,
                                 const cudaStream_t stream) {
        SFEM_DEBUG_SYNCHRONIZE(stream);

        const ptrdiff_t n                 = n_blocks * block_size;
        int             kernel_block_size = SFEM_WARP_SIZE * SFEM_N_WARPS_PER_BLOCK;
        ptrdiff_t       n_cuda_blocks     = std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

        scale_blocks_kernel<<<n_cuda_blocks, kernel_block_size, 0, stream>>>(n_blocks, block_size, scale, blocks, out);

        SFEM_DEBUG_SYNCHRONIZE(stream);
    }

    template <typename T>
    void GPUShiftedPenaltyKernels<T>::build(struct ShiftedPenaltyKernels<T>& tpl, const ExecutionContext& context) {
        // Every pass on the context's stream, bound here once.
        const cudaStream_t s = static_cast<cudaStream_t>(context.stream());
        tpl.sq_norm_ramp_p   = [s](auto&&... a) { return sq_norm_ramp_p_tpl<T>(a..., s); };
        tpl.sq_norm_ramp_m   = [s](auto&&... a) { return sq_norm_ramp_m_tpl<T>(a..., s); };
        tpl.ramp_p           = [s](auto&&... a) { ramp_p_tpl<T>(a..., s); };
        tpl.ramp_m           = [s](auto&&... a) { ramp_m_tpl<T>(a..., s); };
        tpl.update_lagr_p    = [s](auto&&... a) { update_lagr_p_tpl<T>(a..., s); };
        tpl.update_lagr_m    = [s](auto&&... a) { update_lagr_m_tpl<T>(a..., s); };

        auto ramp_m    = tpl.ramp_m;
        auto ramp_p    = tpl.ramp_p;
        tpl.calc_r_pen = [ramp_m, ramp_p](const ptrdiff_t n,
                                          T* const        x,
                                          const T         penalty_param,
                                          const T* const  weights,
                                          const T* const  lb,
                                          const T* const  ub,
                                          const T* const  lagr_lb,
                                          const T* const  lagr_ub,
                                          T*              result) {
            // Ramp negative and positive parts
            if (lb) ramp_m(n, penalty_param, weights, x, lb, lagr_lb, result);
            if (ub) ramp_p(n, penalty_param, weights, x, ub, lagr_ub, result);
        };

        tpl.calc_J_pen    = [s](auto&&... a) { calc_J_pen_tpl<T>(a..., s); };
        tpl.scale_blocks  = [s](auto&&... a) { scale_blocks_tpl<T>(a..., s); };
        tpl.sq_norm_kkt_p = [s](auto&&... a) { return sq_norm_kkt_tpl<T, true>(a..., s); };
        tpl.sq_norm_kkt_m = [s](auto&&... a) { return sq_norm_kkt_tpl<T, false>(a..., s); };

        assert(tpl.good());
    }

    template class GPUShiftedPenaltyKernels<float>;
    template class GPUShiftedPenaltyKernels<double>;

}  // namespace sfem
