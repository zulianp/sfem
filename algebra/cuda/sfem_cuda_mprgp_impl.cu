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
    __device__ void t_warp_reduce(const T val, T* block_accumulator, T* result) {
        T acc = warp_reduce_32(val);

        const unsigned int warp_id = threadIdx.x / SFEM_WARP_SIZE;
        const unsigned int lid = lane_id();

        if (!lid) {
            block_accumulator[warp_id] = acc;
        }

        __syncthreads();

        if (!warp_id) {
            acc = block_accumulator[lid];
            acc = warp_reduce_32(acc);

            if (!threadIdx.x) {
                atomicAdd(result, acc);
            }
        }
    }

    inline static __device__ __host__ double tabs(const double a) { return fabs(a); }
    inline static __device__ __host__ float tabs(const float a) { return fabsf(a); }

    template <typename T>
    inline static __device__ __host__ T tmin(const T a, const T b) {
        return (a < b) ? a : b;
    }

    template <typename T>
    inline static __device__ __host__ T tmax(const T a, const T b) {
        return (a > b) ? a : b;
    }

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
                                         const T* const lb,
                                         const T* const ub,
                                         T* const x) {
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x) {
            x[i] = tmax(tmin(x[i], ub[i]), lb[i]);
        }
    }

    template <typename T>
    __global__ void project_lb_kernel(const ptrdiff_t n, const T* const lb, T* const x) {
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x) {
            x[i] = tmax(x[i], lb[i]);
        }
    }

    template <typename T>
    __global__ void project_ub_kernel(const ptrdiff_t n, const T* const ub, T* const x) {
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x) {
            x[i] = tmin(x[i], ub[i]);
        }
    }

    template <typename T>
    static void project(const ptrdiff_t n, const T* const lb, const T* const ub, T* const x) {
        int kernel_block_size = 128;
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
                                                         const T* const lb,
                                                         const T* const ub,
                                                         const T* const x,
                                                         const T* const g,
                                                         const T eps,
                                                         T* result) {
        T acc = 0;
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x) {
            const T d =
                    gf_lb_ub(lb[i], ub[i], x[i], g[i]) + gc_lb_ub(lb[i], ub[i], x[i], g[i], eps);
            acc += d * d;
        }

        __shared__ T block_accumulator[SFEM_WARP_SIZE];
        t_warp_reduce(acc, block_accumulator, result);
    }

    template <typename T>
    __global__ void norm_projected_gradient_lb_kernel(const ptrdiff_t n,
                                                      const T* const lb,
                                                      const T* const x,
                                                      const T* const g,
                                                      const T eps,
                                                      T* result) {
        T acc = 0;
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x) {
            const T d = gf_lb(lb[i], x[i], g[i]) + gc_lb(lb[i], x[i], g[i], eps);
            acc += d * d;
        }

        __shared__ T block_accumulator[SFEM_WARP_SIZE];
        t_warp_reduce(acc, block_accumulator, result);
    }

    template <typename T>
    __global__ void norm_projected_gradient_ub_kernel(const ptrdiff_t n,
                                                      const T* const ub,
                                                      const T* const x,
                                                      const T* const g,
                                                      const T eps,
                                                      T* result) {
        T acc = 0;
        for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x) {
            const T d = gf_ub(ub[i], x[i], g[i]) + gc_ub(ub[i], x[i], g[i], eps);
            acc += d * d;
        }

        __shared__ T block_accumulator[SFEM_WARP_SIZE];
        t_warp_reduce(acc, block_accumulator, result);
    }

    template <typename T>
    static T norm_projected_gradient(const ptrdiff_t n,
                                     const T* const lb,
                                     const T* const ub,
                                     const T* const x,
                                     const T* const g,
                                     const T eps) {
        int kernel_block_size = 128;
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
    static void norm_gradients(const ptrdiff_t n_dofs,
                               const T* const lb,
                               const T* const ub,
                               const T* const x,
                               const T* const g,
                               T* const norm_free_gradient,
                               T* const norm_chopped_gradient,
                               const T eps) {
        T acc_gf = 0;
        T acc_gc = 0;

        if (lb && ub) {
#pragma omp parallel for reduction(+ : acc_gf), reduction(+ : acc_gc)
            for (ptrdiff_t i = 0; i < n_dofs; i++) {
                const T val_gf = gf_lb_ub(lb[i], ub[i], x[i], g[i]);
                const T val_gc = gc_lb_ub(lb[i], ub[i], x[i], g[i], eps);

                acc_gf += val_gf * val_gf;
                acc_gc += val_gc * val_gc;
            }
        } else if (ub) {
#pragma omp parallel for reduction(+ : acc_gf), reduction(+ : acc_gc)
            for (ptrdiff_t i = 0; i < n_dofs; i++) {
                const T val_gf = gf_ub(ub[i], x[i], g[i]);
                const T val_gc = gc_ub(ub[i], x[i], g[i], eps);

                acc_gf += val_gf * val_gf;
                acc_gc += val_gc * val_gc;
            }
        } else if (lb) {
#pragma omp parallel for reduction(+ : acc_gf), reduction(+ : acc_gc)
            for (ptrdiff_t i = 0; i < n_dofs; i++) {
                const T val_gf = gf_lb(lb[i], x[i], g[i]);
                const T val_gc = gc_lb(lb[i], x[i], g[i], eps);

                acc_gf += val_gf * val_gf;
                acc_gc += val_gc * val_gc;
            }
        } else {
            assert(false);
        }

        *norm_free_gradient = sqrt(acc_gf);
        *norm_chopped_gradient = sqrt(acc_gc);
    }

    template <typename T>
    static void chopped_gradient(const ptrdiff_t n_dofs,
                                 const T* const lb,
                                 const T* const ub,
                                 const T* const x,
                                 const T* const g,
                                 T* gc,
                                 const T eps) {
        if (lb && ub) {
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n_dofs; i++) {
                gc[i] = gc_lb_ub(lb[i], ub[i], x[i], g[i], eps);
            }
        } else if (ub) {
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n_dofs; i++) {
                gc[i] = gc_ub(ub[i], x[i], g[i], eps);
            }
        } else if (lb) {
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n_dofs; i++) {
                gc[i] = gc_lb(lb[i], x[i], g[i], eps);
            }
        }
    }

    template <typename T>
    static void free_gradient(const ptrdiff_t n_dofs,
                              const T* const lb,
                              const T* const ub,
                              const T* const x,
                              const T* const g,
                              T* gf) {
        if (lb && ub) {
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n_dofs; i++) {
                gf[i] = gf_lb_ub(lb[i], ub[i], x[i], g[i]);
            }
        } else if (ub) {
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n_dofs; i++) {
                gf[i] = gf_ub(ub[i], x[i], g[i]);
            }
        } else if (lb) {
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n_dofs; i++) {
                gf[i] = gf_lb(lb[i], x[i], g[i]);
            }
        }
    }

    template <typename T>
    static T max_alpha(const ptrdiff_t n_dofs,
                       const T* const lb,
                       const T* const ub,
                       const T* const x,
                       const T* const p,
                       const T infty) {
        T ret = infty;

        if (lb && ub) {
#pragma omp parallel for reduction(min : ret)
            for (ptrdiff_t i = 0; i < n_dofs; i++) {
                const T alpha_lb = (p[i] > 0) ? ((x[i] - lb[i]) / p[i]) : infty;
                const T alpha_ub = (p[i] < 0) ? ((x[i] - ub[i]) / p[i]) : infty;
                const T alpha = tmin(alpha_lb, alpha_ub);
                ret = tmin(alpha, ret);
            }

        } else if (ub) {
#pragma omp parallel for reduction(min : ret)
            for (ptrdiff_t i = 0; i < n_dofs; i++) {
                const T alpha = (p[i] < 0) ? ((x[i] - ub[i]) / p[i]) : infty;
                ret = tmin(alpha, ret);
            }
        } else if (lb) {
#pragma omp parallel for reduction(min : ret)
            for (ptrdiff_t i = 0; i < n_dofs; i++) {
                const T alpha = (p[i] > 0) ? ((x[i] - lb[i]) / p[i]) : infty;
                ret = tmin(alpha, ret);
            }
        }

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