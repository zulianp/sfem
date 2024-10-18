#include "sfem_base.h"
#include "sfem_cuda_mprgp_impl.hpp"

#include "sfem_cuda_base.h"

#include <cassert>

namespace sfem {
    inline __device__ unsigned int lane_id() { return threadIdx.x % SFEM_WARP_SIZE; }

    template <typename T>
    __device__ T warp_reduce_32(const T in) {
        static_assert(SFEM_WARP_SIZE == 32, "Only implemented for CUDA!");
        T out = in;
        out += __shfl_xor_sync(
                SFEM_WARP_FULL_MASK, out, 16, SFEM_WARP_SIZE);  // 0-16, 1-17, ..., 15-31
        out += __shfl_xor_sync(
                SFEM_WARP_FULL_MASK, out, 8, SFEM_WARP_SIZE);  // 0-8, ..., 1-7, ..., 23-31
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
        out = tmin(
                out,
                __shfl_xor_sync(
                        SFEM_WARP_FULL_MASK, out, 16, SFEM_WARP_SIZE));  // 0-16, 1-17, ..., 15-31
        out = tmin(
                out,
                __shfl_xor_sync(
                        SFEM_WARP_FULL_MASK, out, 8, SFEM_WARP_SIZE));  // 0-8, ..., 1-7, ..., 23-31
        out = tmin(out, __shfl_xor_sync(SFEM_WARP_FULL_MASK, out, 4, SFEM_WARP_SIZE));
        out = tmin(out, __shfl_xor_sync(SFEM_WARP_FULL_MASK, out, 2, SFEM_WARP_SIZE));
        out = tmin(out, __shfl_xor_sync(SFEM_WARP_FULL_MASK, out, 1, SFEM_WARP_SIZE));
        return out;
    }

    template <typename T>
    __device__ T warp_max_32(const T in) {
        static_assert(SFEM_WARP_SIZE == 32, "Only implemented for CUDA!");
        T out = in;
        out = tmax(
                out,
                __shfl_xor_sync(
                        SFEM_WARP_FULL_MASK, out, 16, SFEM_WARP_SIZE));  // 0-16, 1-17, ..., 15-31
        out = tmax(
                out,
                __shfl_xor_sync(
                        SFEM_WARP_FULL_MASK, out, 8, SFEM_WARP_SIZE));  // 0-8, ..., 1-7, ..., 23-31
        out = tmax(out, __shfl_xor_sync(SFEM_WARP_FULL_MASK, out, 4, SFEM_WARP_SIZE));
        out = tmax(out, __shfl_xor_sync(SFEM_WARP_FULL_MASK, out, 2, SFEM_WARP_SIZE));
        out = tmax(out, __shfl_xor_sync(SFEM_WARP_FULL_MASK, out, 1, SFEM_WARP_SIZE));
        return out;
    }

#define SFEM_N_WARPS_PER_BLOCK 8

    template <typename T>
    __device__ void t_warp_reduce(const T val, T* block_accumulator, T* result) {
        T acc = warp_reduce_32(val);

        const unsigned int warp_id = threadIdx.x / SFEM_WARP_SIZE;
        const unsigned int lid = lane_id();

        if (!lid) {
            block_accumulator[warp_id] = acc;
        }

        __syncthreads();

        if (!warp_id) {
            assert(warp_id < SFEM_N_WARPS_PER_BLOCK);
            acc = block_accumulator[lid];
            acc = warp_reduce_32(acc);

            if (!threadIdx.x) {
                atomicAdd(result, acc);
            }
        }
    }

    template <typename T>
    __device__ void t_warp_min(const T val, T* block_accumulator, T* result) {
        T acc = warp_min_32(val);

        const unsigned int warp_id = threadIdx.x / SFEM_WARP_SIZE;
        const unsigned int lid = lane_id();

        if (!lid) {
            block_accumulator[warp_id] = acc;
        }

        __syncthreads();

        if (!warp_id) {
            assert(warp_id < SFEM_N_WARPS_PER_BLOCK);
            acc = block_accumulator[lid];
            acc = warp_min_32(acc);

            if (!threadIdx.x) {
                result[blockIdx.x] = acc;
            }
        }
    }

    // template <typename T>
    // __device__ void t_warp_max(const T val, T* block_accumulator, T* result) {
    //     T acc = warp_max_32(val);

    //     const unsigned int warp_id = threadIdx.x / SFEM_WARP_SIZE;
    //     const unsigned int lid = lane_id();

    //     if (!lid) {
    //         block_accumulator[warp_id] = acc;
    //     }

    //     __syncthreads();

    //     if (!warp_id) {
    //         assert(warp_id < SFEM_N_WARPS_PER_BLOCK);
    //         acc = block_accumulator[lid];
    //         acc = warp_max_32(acc);

    //         if (!threadIdx.x) {
    //             atomicMax(result, acc);
    //         }
    //     }
    // }

    inline static __device__ __host__ double tabs(const double a) { return fabs(a); }
    inline static __device__ __host__ float tabs(const float a) { return fabsf(a); }

    template <typename T>
    inline static __device__ __host__ T gf_lb_ub(const T lbi, const T ubi, const T xi, const T gi) {
#if 1
        return (xi <= lbi || xi >= ubi) ? T(0) : gi;
#else
        return ((tabs(lbi - xi) < eps) || (tabs(ubi - xi) < eps)) ? T(0) : gi;
#endif
    }

    template <typename T>
    inline static __device__ __host__ T gf_lb(const T lbi, const T xi, const T gi) {
#if 1
        return (xi <= lbi) ? T(0) : gi;
#else
        return (tabs(lbi - xi) < eps) ? T(0) : gi;
#endif
    }

    template <typename T>
    inline static __device__ __host__ T gf_ub(const T ubi, const T xi, const T gi) {
#if 1
        return (xi >= ubi) ? T(0) : gi;
#else
        return (tabs(ubi - xi) < eps) ? T(0) : gi;
#endif
    }

    template <typename T>
    inline static __device__ __host__ T
    gc_lb_ub(const T lbi, const T ubi, const T xi, const T gi, const T eps) {
        return ((tabs(lbi - xi) < eps) ? tmin(T(0), gi)
                                       : ((tabs(ubi - xi) < eps) ? tmax(T(0), gi) : T(0)));
    }

    template <typename T>
    inline static __device__ __host__ T gc_lb(const T lbi, const T xi, const T gi, const T eps) {
        return ((tabs(lbi - xi) < eps) ? tmin(T(0), gi) : T(0));
    }

    template <typename T>
    inline static __device__ __host__ T gc_ub(const T ubi, const T xi, const T gi, const T eps) {
        return ((tabs(ubi - xi) < eps) ? tmax(T(0), gi) : T(0));
    }

    template <typename T>
    __global__ void project_lb_ub_kernel(const ptrdiff_t n,
                                         const T* const SFEM_RESTRICT lb,
                                         const T* const SFEM_RESTRICT ub,
                                         T* const SFEM_RESTRICT x) {
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x) {
            x[i] = tmax(tmin(x[i], ub[i]), lb[i]);
        }
    }

    template <typename T>
    __global__ void project_lb_kernel(const ptrdiff_t n,
                                      const T* const SFEM_RESTRICT lb,
                                      T* const SFEM_RESTRICT x) {
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x) {
            x[i] = tmax(x[i], lb[i]);
        }
    }

    template <typename T>
    __global__ void project_ub_kernel(const ptrdiff_t n,
                                      const T* const SFEM_RESTRICT ub,
                                      T* const SFEM_RESTRICT x) {
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x) {
            x[i] = tmin(x[i], ub[i]);
        }
    }

    template <typename T>
    static void project(const ptrdiff_t n,
                        const T* const SFEM_RESTRICT lb,
                        const T* const SFEM_RESTRICT ub,
                        T* const SFEM_RESTRICT x) {
        int kernel_block_size = SFEM_WARP_SIZE * SFEM_N_WARPS_PER_BLOCK;
        ptrdiff_t n_blocks =
                std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

        if (lb && ub) {
            project_lb_ub_kernel<<<n_blocks, kernel_block_size>>>(n, lb, ub, x);
        } else if (ub) {
            project_ub_kernel<<<n_blocks, kernel_block_size>>>(n, ub, x);
        } else if (lb) {
            project_lb_kernel<<<n_blocks, kernel_block_size>>>(n, lb, x);
        }
    }

    template <typename T>
    __global__ void norm_projected_gradient_lb_ub_kernel(const ptrdiff_t n,
                                                         const T* const SFEM_RESTRICT lb,
                                                         const T* const SFEM_RESTRICT ub,
                                                         const T* const SFEM_RESTRICT x,
                                                         const T* const SFEM_RESTRICT g,
                                                         const T eps,
                                                         T* SFEM_RESTRICT result) {
        T acc = 0;
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x) {
            const T d =
                    gf_lb_ub(lb[i], ub[i], x[i], g[i]) + gc_lb_ub(lb[i], ub[i], x[i], g[i], eps);
            acc += d * d;
        }

        __shared__ T block_accumulator[SFEM_N_WARPS_PER_BLOCK];
        t_warp_reduce(acc, block_accumulator, result);
    }

    template <typename T>
    __global__ void norm_projected_gradient_lb_kernel(const ptrdiff_t n,
                                                      const T* const SFEM_RESTRICT lb,
                                                      const T* const SFEM_RESTRICT x,
                                                      const T* const SFEM_RESTRICT g,
                                                      const T eps,
                                                      T* SFEM_RESTRICT result) {
        T acc = 0;
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x) {
            const T d = gf_lb(lb[i], x[i], g[i]) + gc_lb(lb[i], x[i], g[i], eps);
            acc += d * d;
        }

        __shared__ T block_accumulator[SFEM_N_WARPS_PER_BLOCK];
        t_warp_reduce(acc, block_accumulator, result);
    }

    template <typename T>
    __global__ void norm_projected_gradient_ub_kernel(const ptrdiff_t n,
                                                      const T* const SFEM_RESTRICT ub,
                                                      const T* const SFEM_RESTRICT x,
                                                      const T* const SFEM_RESTRICT g,
                                                      const T eps,
                                                      T* result) {
        T acc = 0;
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x) {
            const T d = gf_ub(ub[i], x[i], g[i]) + gc_ub(ub[i], x[i], g[i], eps);
            acc += d * d;
        }

        __shared__ T block_accumulator[SFEM_N_WARPS_PER_BLOCK];
        t_warp_reduce(acc, block_accumulator, result);
    }

    template <typename T>
    static T norm_projected_gradient(const ptrdiff_t n,
                                     const T* const SFEM_RESTRICT lb,
                                     const T* const SFEM_RESTRICT ub,
                                     const T* const SFEM_RESTRICT x,
                                     const T* const SFEM_RESTRICT g,
                                     const T eps) {
        int kernel_block_size = SFEM_WARP_SIZE * SFEM_N_WARPS_PER_BLOCK;
        ptrdiff_t n_blocks =
                std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

        T* device_value = nullptr;

        cudaMalloc((void**)&device_value, sizeof(T));
        cudaMemset((void*)device_value, 0, sizeof(T));

        if (lb && ub) {
            norm_projected_gradient_lb_ub_kernel<<<n_blocks, kernel_block_size>>>(
                    n, lb, ub, x, g, eps, device_value);
            SFEM_DEBUG_SYNCHRONIZE();
        } else if (ub) {
            norm_projected_gradient_ub_kernel<<<n_blocks, kernel_block_size>>>(
                    n, ub, x, g, eps, device_value);
            SFEM_DEBUG_SYNCHRONIZE();

        } else if (lb) {
            norm_projected_gradient_lb_kernel<<<n_blocks, kernel_block_size>>>(
                    n, lb, x, g, eps, device_value);
            SFEM_DEBUG_SYNCHRONIZE();
        } else {
            assert(false);
        }

        T host_value = -1;
        cudaMemcpy(&host_value, device_value, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(device_value);

        return sqrt(host_value);
    }

    template <typename T>
    __global__ void norm_gradients_lb_ub_kernel(const ptrdiff_t n,
                                                const T* const SFEM_RESTRICT lb,
                                                const T* const SFEM_RESTRICT ub,
                                                const T* const SFEM_RESTRICT x,
                                                const T* const SFEM_RESTRICT g,
                                                T* const SFEM_RESTRICT norm_free_gradient,
                                                T* const SFEM_RESTRICT norm_chopped_gradient,
                                                const T eps) {
        T acc_gf = 0;
        T acc_gc = 0;

        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x) {
            const T val_gf = gf_lb_ub(lb[i], ub[i], x[i], g[i]);
            const T val_gc = gc_lb_ub(lb[i], ub[i], x[i], g[i], eps);

            acc_gf += val_gf * val_gf;
            acc_gc += val_gc * val_gc;
        }

        __shared__ T block_accumulator[SFEM_N_WARPS_PER_BLOCK];
        t_warp_reduce(acc_gf, block_accumulator, norm_free_gradient);
        t_warp_reduce(acc_gc, block_accumulator, norm_chopped_gradient);
    }

    template <typename T>
    __global__ void norm_gradients_lb_kernel(const ptrdiff_t n,
                                             const T* const SFEM_RESTRICT lb,
                                             const T* const SFEM_RESTRICT x,
                                             const T* const SFEM_RESTRICT g,
                                             T* const SFEM_RESTRICT norm_free_gradient,
                                             T* const SFEM_RESTRICT norm_chopped_gradient,
                                             const T eps) {
        T acc_gf = 0;
        T acc_gc = 0;

        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x) {
            const T val_gf = gf_lb(lb[i], x[i], g[i]);
            const T val_gc = gc_lb(lb[i], x[i], g[i], eps);

            acc_gf += val_gf * val_gf;
            acc_gc += val_gc * val_gc;
        }

        __shared__ T block_accumulator[SFEM_N_WARPS_PER_BLOCK];
        t_warp_reduce(acc_gf, block_accumulator, norm_free_gradient);
        t_warp_reduce(acc_gc, block_accumulator, norm_chopped_gradient);
    }

    template <typename T>
    __global__ void norm_gradients_ub_kernel(const ptrdiff_t n,
                                             const T* const SFEM_RESTRICT ub,
                                             const T* const SFEM_RESTRICT x,
                                             const T* const SFEM_RESTRICT g,
                                             T* const SFEM_RESTRICT norm_free_gradient,
                                             T* const SFEM_RESTRICT norm_chopped_gradient,
                                             const T eps) {
        T acc_gf = 0;
        T acc_gc = 0;

        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x) {
            const T val_gf = gf_ub(ub[i], x[i], g[i]);
            const T val_gc = gc_ub(ub[i], x[i], g[i], eps);

            acc_gf += val_gf * val_gf;
            acc_gc += val_gc * val_gc;
        }

        __shared__ T block_accumulator[SFEM_N_WARPS_PER_BLOCK];
        t_warp_reduce(acc_gf, block_accumulator, norm_free_gradient);
        t_warp_reduce(acc_gc, block_accumulator, norm_chopped_gradient);
    }

    template <typename T>
    static void norm_gradients(const ptrdiff_t n,
                               const T* const SFEM_RESTRICT lb,
                               const T* const SFEM_RESTRICT ub,
                               const T* const SFEM_RESTRICT x,
                               const T* const SFEM_RESTRICT g,
                               T* const SFEM_RESTRICT norm_free_gradient,
                               T* const SFEM_RESTRICT norm_chopped_gradient,
                               const T eps) {
        int kernel_block_size = SFEM_WARP_SIZE * SFEM_N_WARPS_PER_BLOCK;
        ptrdiff_t n_blocks =
                std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

        T* device_value_gf = nullptr;
        T* device_value_gc = nullptr;

        cudaMalloc((void**)&device_value_gf, sizeof(T));
        cudaMemset((void*)device_value_gf, 0, sizeof(T));

        cudaMalloc((void**)&device_value_gc, sizeof(T));
        cudaMemset((void*)device_value_gc, 0, sizeof(T));

        if (lb && ub) {
            norm_gradients_lb_ub_kernel<<<n_blocks, kernel_block_size>>>(
                    n, lb, ub, x, g, device_value_gf, device_value_gc, eps);
            SFEM_DEBUG_SYNCHRONIZE();
        } else if (ub) {
            norm_gradients_ub_kernel<<<n_blocks, kernel_block_size>>>(
                    n, ub, x, g, device_value_gf, device_value_gc, eps);
            SFEM_DEBUG_SYNCHRONIZE();
        } else if (lb) {
            norm_gradients_lb_kernel<<<n_blocks, kernel_block_size>>>(
                    n, lb, x, g, device_value_gf, device_value_gc, eps);
            SFEM_DEBUG_SYNCHRONIZE();
        } else {
            assert(false);
        }

        T host_value_gf = -1;
        cudaMemcpy(&host_value_gf, device_value_gf, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(device_value_gf);

        T host_value_gc = -1;
        cudaMemcpy(&host_value_gc, device_value_gc, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(device_value_gc);

        *norm_free_gradient = sqrt(host_value_gf);
        *norm_chopped_gradient = sqrt(host_value_gc);
    }

    template <typename T>
    __global__ void chopped_gradient_lb_ub_kernel(const ptrdiff_t n,
                                                  const T* const SFEM_RESTRICT lb,
                                                  const T* const SFEM_RESTRICT ub,
                                                  const T* const SFEM_RESTRICT x,
                                                  const T* const SFEM_RESTRICT g,
                                                  T* SFEM_RESTRICT gc,
                                                  const T eps) {
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x) {
            gc[i] = gc_lb_ub(lb[i], ub[i], x[i], g[i], eps);
        }
    }

    template <typename T>
    __global__ void chopped_gradient_lb_kernel(const ptrdiff_t n,
                                               const T* const SFEM_RESTRICT lb,
                                               const T* const SFEM_RESTRICT x,
                                               const T* const SFEM_RESTRICT g,
                                               T* SFEM_RESTRICT gc,
                                               const T eps) {
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x) {
            gc[i] = gc_lb(lb[i], x[i], g[i], eps);
        }
    }

    template <typename T>
    __global__ void chopped_gradient_ub_kernel(const ptrdiff_t n,
                                               const T* const SFEM_RESTRICT ub,
                                               const T* const SFEM_RESTRICT x,
                                               const T* const SFEM_RESTRICT g,
                                               T* SFEM_RESTRICT gc,
                                               const T eps) {
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x) {
            gc[i] = gc_ub(ub[i], x[i], g[i], eps);
        }
    }

    template <typename T>
    static void chopped_gradient(const ptrdiff_t n,
                                 const T* const SFEM_RESTRICT lb,
                                 const T* const SFEM_RESTRICT ub,
                                 const T* const SFEM_RESTRICT x,
                                 const T* const SFEM_RESTRICT g,
                                 T* SFEM_RESTRICT gc,
                                 const T eps) {
        int kernel_block_size = SFEM_WARP_SIZE * SFEM_N_WARPS_PER_BLOCK;
        ptrdiff_t n_blocks =
                std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

        if (lb && ub) {
            chopped_gradient_lb_ub_kernel<<<n_blocks, kernel_block_size>>>(
                    n, lb, ub, x, g, gc, eps);
            SFEM_DEBUG_SYNCHRONIZE();
        } else if (ub) {
            chopped_gradient_ub_kernel<<<n_blocks, kernel_block_size>>>(n, ub, x, g, gc, eps);
            SFEM_DEBUG_SYNCHRONIZE();
        } else if (lb) {
            chopped_gradient_lb_kernel<<<n_blocks, kernel_block_size>>>(n, lb, x, g, gc, eps);
            SFEM_DEBUG_SYNCHRONIZE();
        }
    }

    template <typename T>
    __global__ void free_gradient_lb_ub_kernel(const ptrdiff_t n,
                                               const T* const SFEM_RESTRICT lb,
                                               const T* const SFEM_RESTRICT ub,
                                               const T* const SFEM_RESTRICT x,
                                               const T* const SFEM_RESTRICT g,
                                               T* SFEM_RESTRICT gf) {
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x) {
            gf[i] = gf_lb_ub(lb[i], ub[i], x[i], g[i]);
        }
    }

    template <typename T>
    __global__ void free_gradient_lb_kernel(const ptrdiff_t n,
                                            const T* const SFEM_RESTRICT lb,
                                            const T* const SFEM_RESTRICT x,
                                            const T* const SFEM_RESTRICT g,
                                            T* SFEM_RESTRICT gf) {
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x) {
            gf[i] = gf_lb(lb[i], x[i], g[i]);
        }
    }

    template <typename T>
    __global__ void free_gradient_ub_kernel(const ptrdiff_t n,
                                            const T* const SFEM_RESTRICT ub,
                                            const T* const SFEM_RESTRICT x,
                                            const T* const SFEM_RESTRICT g,
                                            T* SFEM_RESTRICT gf) {
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x) {
            gf[i] = gf_ub(ub[i], x[i], g[i]);
        }
    }

    template <typename T>
    static void free_gradient(const ptrdiff_t n,
                              const T* const SFEM_RESTRICT lb,
                              const T* const SFEM_RESTRICT ub,
                              const T* const SFEM_RESTRICT x,
                              const T* const SFEM_RESTRICT g,
                              T* SFEM_RESTRICT gf) {
        int kernel_block_size = SFEM_WARP_SIZE * SFEM_N_WARPS_PER_BLOCK;
        ptrdiff_t n_blocks =
                std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

        if (lb && ub) {
            free_gradient_lb_ub_kernel<<<n_blocks, kernel_block_size>>>(n, lb, ub, x, g, gf);
            SFEM_DEBUG_SYNCHRONIZE();
        } else if (ub) {
            free_gradient_ub_kernel<<<n_blocks, kernel_block_size>>>(n, ub, x, g, gf);
            SFEM_DEBUG_SYNCHRONIZE();
        } else if (lb) {
            free_gradient_lb_kernel<<<n_blocks, kernel_block_size>>>(n, lb, x, g, gf);
            SFEM_DEBUG_SYNCHRONIZE();
        }
    }

    template <typename T>
    __global__ void max_alpha_lb_ub_kernel(const ptrdiff_t n,
                                           const T* const SFEM_RESTRICT lb,
                                           const T* const SFEM_RESTRICT ub,
                                           const T* const SFEM_RESTRICT x,
                                           const T* const SFEM_RESTRICT p,
                                           const T infty,
                                           T* result) {
        T acc = infty;
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x) {
            const T alpha_lb = (p[i] > 0) ? ((x[i] - lb[i]) / p[i]) : infty;
            const T alpha_ub = (p[i] < 0) ? ((x[i] - ub[i]) / p[i]) : infty;
            const T alpha = tmin(alpha_lb, alpha_ub);
            acc = tmin(alpha, acc);
        }

        __shared__ T block_accumulator[SFEM_N_WARPS_PER_BLOCK];
        t_warp_min(acc, block_accumulator, result);
    }

    template <typename T>
    __global__ void max_alpha_lb_kernel(const ptrdiff_t n,
                                        const T* const SFEM_RESTRICT lb,
                                        const T* const SFEM_RESTRICT x,
                                        const T* const SFEM_RESTRICT p,
                                        const T infty,
                                        T* result) {
        T acc = infty;
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x) {
            const T alpha = (p[i] > 0) ? ((x[i] - lb[i]) / p[i]) : infty;
            acc = tmin(alpha, acc);
        }

        __shared__ T block_accumulator[SFEM_N_WARPS_PER_BLOCK];
        t_warp_min(acc, block_accumulator, result);
    }

    template <typename T>
    __global__ void max_alpha_ub_kernel(const ptrdiff_t n,
                                        const T* const SFEM_RESTRICT ub,
                                        const T* const SFEM_RESTRICT x,
                                        const T* const SFEM_RESTRICT p,
                                        const T infty,
                                        T* result) {
        T acc = infty;
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x) {
            const T alpha = (p[i] > 0) ? ((x[i] - ub[i]) / p[i]) : infty;
            acc = tmin(alpha, acc);
        }

        __shared__ T block_accumulator[SFEM_N_WARPS_PER_BLOCK];
        t_warp_min(acc, block_accumulator, result);
    }

    template <typename T>
    static T max_alpha(const ptrdiff_t n,
                       const T* const SFEM_RESTRICT lb,
                       const T* const SFEM_RESTRICT ub,
                       const T* const SFEM_RESTRICT x,
                       const T* const SFEM_RESTRICT p,
                       const T infty) {
        int kernel_block_size = SFEM_WARP_SIZE * SFEM_N_WARPS_PER_BLOCK;
        ptrdiff_t n_blocks =
                std::max(ptrdiff_t(1), (n + kernel_block_size - 1) / kernel_block_size);

        T* device_value = nullptr;

        cudaMalloc((void**)&device_value, n_blocks * sizeof(T));
        cudaMemset((void*)device_value, 0, n_blocks * sizeof(T));

        if (lb && ub) {
            max_alpha_lb_ub_kernel<<<n_blocks, kernel_block_size>>>(
                    n, lb, ub, x, p, infty, device_value);
            SFEM_DEBUG_SYNCHRONIZE();
        } else if (ub) {
            max_alpha_ub_kernel<<<n_blocks, kernel_block_size>>>(n, ub, x, p, infty, device_value);
            SFEM_DEBUG_SYNCHRONIZE();
        } else if (lb) {
            max_alpha_lb_kernel<<<n_blocks, kernel_block_size>>>(n, lb, x, p, infty, device_value);
            SFEM_DEBUG_SYNCHRONIZE();
        }

        T* host_value = (T*)malloc(n_blocks * sizeof(T));

        cudaMemcpy(host_value, device_value, n_blocks * sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(device_value);

        T ret = host_value[0];

        for (int i = 1; i < n_blocks; i++) {
            ret = tmin(host_value[i], ret);
        }

        free(host_value);
        return ret;
    }

    template <typename T>
    void CUDA_MPRGP<T>::build_mprgp(struct MPRGP_Tpl<T>& tpl) {
        tpl.project = project<T>;
        tpl.norm_projected_gradient = norm_projected_gradient<T>;
        tpl.norm_gradients = norm_gradients<T>;
        tpl.chopped_gradient = chopped_gradient<T>;
        tpl.free_gradient = free_gradient<T>;
        tpl.max_alpha = max_alpha<T>;
    }

    template class CUDA_MPRGP<float>;
    template class CUDA_MPRGP<double>;

}  // namespace sfem