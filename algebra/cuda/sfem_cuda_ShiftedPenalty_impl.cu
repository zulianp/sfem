#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>

#include "sfem_base.h"
#include "sfem_cuda_ShiftedPenalty_impl.hpp"

#include "sfem_cuda_base.h"

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
    static T sq_norm_ramp_p_tpl(const ptrdiff_t n, const T* const x, T* const ub) {
        SFEM_DEBUG_SYNCHRONIZE();

        int       kernel_block_size = SFEM_WARP_SIZE * SFEM_N_WARPS_PER_BLOCK;
        ptrdiff_t n_blocks          = std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

        T* device_value = nullptr;

        cudaMalloc((void**)&device_value, sizeof(T));
        cudaMemset((void*)device_value, 0, sizeof(T));

        sq_norm_ramp_p_kernel<<<n_blocks, kernel_block_size>>>(n, x, ub, device_value);
        SFEM_DEBUG_SYNCHRONIZE();

        T host_value = -1;
        cudaMemcpy(&host_value, device_value, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(device_value);

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
    static T sq_norm_ramp_m_tpl(const ptrdiff_t n, const T* const x, T* const lb) {
        SFEM_DEBUG_SYNCHRONIZE();

        int       kernel_block_size = SFEM_WARP_SIZE * SFEM_N_WARPS_PER_BLOCK;
        ptrdiff_t n_blocks          = std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

        T* device_value = nullptr;

        cudaMalloc((void**)&device_value, sizeof(T));
        cudaMemset((void*)device_value, 0, sizeof(T));

        sq_norm_ramp_m_kernel<<<n_blocks, kernel_block_size>>>(n, x, lb, device_value);
        SFEM_DEBUG_SYNCHRONIZE();

        T host_value = -1;
        cudaMemcpy(&host_value, device_value, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(device_value);

        return host_value;
    }

    /////

    template <typename T>
    __global__ void ramp_p_kernel(const ptrdiff_t n,
                                  const T         penalty_param,
                                  const T* const  x,
                                  const T* const  ub,
                                  const T* const  lagr_ub,
                                  T* const        out) {
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
            out[i] -= penalty_param * tmax(T(0), x[i] - ub[i] + lagr_ub[i] / penalty_param);
        }
    }

    template <typename T>
    static void ramp_p_tpl(const ptrdiff_t n,
                           const T         penalty_param,
                           const T* const  x,
                           const T* const  ub,
                           const T* const  lagr_ub,
                           T* const        out) {
        SFEM_DEBUG_SYNCHRONIZE();

        int       kernel_block_size = SFEM_WARP_SIZE * SFEM_N_WARPS_PER_BLOCK;
        ptrdiff_t n_blocks          = std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

        ramp_p_kernel<<<n_blocks, kernel_block_size>>>(n, penalty_param, x, ub, lagr_ub, out);

        SFEM_DEBUG_SYNCHRONIZE();
    }

    /////

    template <typename T>
    __global__ void ramp_m_kernel(const ptrdiff_t n,
                                  const T         penalty_param,
                                  const T* const  x,
                                  const T* const  lb,
                                  const T* const  lagr_lb,
                                  T* const        out) {
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
            out[i] -= penalty_param * tmin(T(0), x[i] - lb[i] + lagr_lb[i] / penalty_param);
        }
    }

    template <typename T>
    static void ramp_m_tpl(const ptrdiff_t n,
                           const T         penalty_param,
                           const T* const  x,
                           const T* const  lb,
                           const T* const  lagr_lb,
                           T* const        out) {
        SFEM_DEBUG_SYNCHRONIZE();

        int       kernel_block_size = SFEM_WARP_SIZE * SFEM_N_WARPS_PER_BLOCK;
        ptrdiff_t n_blocks          = std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

        ramp_m_kernel<<<n_blocks, kernel_block_size>>>(n, penalty_param, x, lb, lagr_lb, out);

        SFEM_DEBUG_SYNCHRONIZE();
    }

    /////

    template <typename T>
    __global__ void update_lagr_p_kernel(const ptrdiff_t n,
                                         const T         penalty_param,
                                         const T* const  x,
                                         const T* const  ub,
                                         T* const        lagr_ub) {
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
            lagr_ub[i] = tmax(T(0), lagr_ub[i] + penalty_param * (x[i] - ub[i]));
        }
    }

    template <typename T>
    static void update_lagr_p_tpl(const ptrdiff_t n,
                                  const T         penalty_param,
                                  const T* const  x,
                                  const T* const  ub,
                                  T* const        lagr_ub) {
        SFEM_DEBUG_SYNCHRONIZE();

        int       kernel_block_size = SFEM_WARP_SIZE * SFEM_N_WARPS_PER_BLOCK;
        ptrdiff_t n_blocks          = std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

        update_lagr_p_kernel<<<n_blocks, kernel_block_size>>>(n, penalty_param, x, ub, lagr_ub);

        SFEM_DEBUG_SYNCHRONIZE();
    }

    /////

    template <typename T>
    __global__ void update_lagr_m_kernel(const ptrdiff_t n,
                                         const T         penalty_param,
                                         const T* const  x,
                                         const T* const  lb,
                                         T* const        lagr_lb) {
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
            lagr_lb[i] = tmin(T(0), lagr_lb[i] + penalty_param * (x[i] - lb[i]));
        }
    }

    template <typename T>
    static void update_lagr_m_tpl(const ptrdiff_t n,
                                  const T         penalty_param,
                                  const T* const  x,
                                  const T* const  lb,
                                  T* const        lagr_lb) {
        SFEM_DEBUG_SYNCHRONIZE();

        int       kernel_block_size = SFEM_WARP_SIZE * SFEM_N_WARPS_PER_BLOCK;
        ptrdiff_t n_blocks          = std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

        update_lagr_m_kernel<<<n_blocks, kernel_block_size>>>(n, penalty_param, x, lb, lagr_lb);

        SFEM_DEBUG_SYNCHRONIZE();
    }

    template <typename T>
    __global__ void calc_J_pen_p_kernel(const ptrdiff_t n,
                                        const T* const  x,
                                        const T         penalty_param,
                                        const T* const  ub,
                                        const T* const  lagr_ub,
                                        T* const        result) {
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
            result[i] += ((x[i] - ub[i] + lagr_ub[i] / penalty_param) >= 0) * penalty_param;
        }
    }

    template <typename T>
    __global__ void calc_J_pen_m_kernel(const ptrdiff_t n,
                                        const T* const  x,
                                        const T         penalty_param,
                                        const T* const  lb,
                                        const T* const  lagr_lb,
                                        T* const        result) {
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
            result[i] += ((x[i] - lb[i] + lagr_lb[i] / penalty_param) <= 0) * penalty_param;
        }
    }

    ///
    template <typename T>
    static void calc_J_pen_tpl(const ptrdiff_t n,
                               const T* const  x,
                               const T         penalty_param,
                               const T* const  lb,
                               const T* const  ub,
                               const T* const  lagr_lb,
                               const T* const  lagr_ub,
                               T* const        result) {
        SFEM_DEBUG_SYNCHRONIZE();

        int       kernel_block_size = SFEM_WARP_SIZE * SFEM_N_WARPS_PER_BLOCK;
        ptrdiff_t n_blocks          = std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

        if (lb) {
            calc_J_pen_m_kernel<<<n_blocks, kernel_block_size>>>(n, x, penalty_param, lb, lagr_lb, result);
        }

        if (ub) {
            calc_J_pen_p_kernel<<<n_blocks, kernel_block_size>>>(n, x, penalty_param, ub, lagr_ub, result);
        }

        SFEM_DEBUG_SYNCHRONIZE();
    }

    template <typename T>
    void CUDA_ShiftedPenalty<T>::build(struct ShiftedPenalty_Tpl<T>& tpl) {
        tpl.sq_norm_ramp_p = &sq_norm_ramp_p_tpl<T>;
        tpl.sq_norm_ramp_m = &sq_norm_ramp_m_tpl<T>;
        tpl.ramp_p         = &ramp_p_tpl<T>;
        tpl.ramp_m         = &ramp_m_tpl<T>;
        tpl.update_lagr_p  = &update_lagr_p_tpl<T>;
        tpl.update_lagr_m  = &update_lagr_m_tpl<T>;

        auto ramp_m    = tpl.ramp_m;
        auto ramp_p    = tpl.ramp_p;
        tpl.calc_r_pen = [ramp_m, ramp_p](const ptrdiff_t n,
                                          T* const        x,
                                          const T         penalty_param,
                                          const T* const  lb,
                                          const T* const  ub,
                                          const T* const  lagr_lb,
                                          const T* const  lagr_ub,
                                          T*              result) {
            // Ramp negative and positive parts
            if (lb) ramp_m(n, penalty_param, x, lb, lagr_lb, result);
            if (ub) ramp_p(n, penalty_param, x, ub, lagr_ub, result);
        };

        tpl.calc_J_pen = &calc_J_pen_tpl<T>;

        assert(tpl.good());
    }

    template class CUDA_ShiftedPenalty<float>;
    template class CUDA_ShiftedPenalty<double>;

}  // namespace sfem
