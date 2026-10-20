#include "sfem_base.hpp"
#include "sfem_cuda_mprgp_impl.hpp"

#include "sfem_cuda_base.hpp"
#include "sfem_mprgp_bounds.hpp"

#include <algorithm>
#include <type_traits>

namespace sfem {
    namespace {
        constexpr int N_THREADS  = 256;  // per block
        constexpr int N_WARPS    = N_THREADS / SFEM_WARP_SIZE;
        constexpr int MAX_BLOCKS = 1024;  // the loops stride over dofs past the grid
        constexpr int N_VALUES   = 4;     // reductions of one pass

        // Per-block partials, value-major (k * MAX_BLOCKS + block), then the N_VALUES results.
        constexpr ptrdiff_t WORK_SIZE = N_VALUES * MAX_BLOCKS + N_VALUES;

        int n_blocks(const ptrdiff_t n) {
            return int(std::min<ptrdiff_t>(MAX_BLOCKS, std::max<ptrdiff_t>(1, (n + N_THREADS - 1) / N_THREADS)));
        }

        template <class B>
        using Bounds = typename std::decay<B>::type;

        // The first N_SUM values are sums, the others minima.
        template <typename T, int N_SUM>
        __device__ T combine(const int k, const T a, const T b) {
            return k < N_SUM ? a + b : ((b < a) ? b : a);
        }

        template <typename T, int N_SUM>
        __device__ T identity(const int k, const T infty) {
            return k < N_SUM ? T(0) : infty;
        }

        // Reduces each thread's values over the block, in a fixed order; thread 0 holds the result.
        template <typename T, int N_SUM>
        __device__ void block_reduce(T (&v)[N_VALUES], const T infty) {
            __shared__ T shared[N_VALUES][N_WARPS];
            const int    lane = threadIdx.x % SFEM_WARP_SIZE, warp = threadIdx.x / SFEM_WARP_SIZE;
            for (int k = 0; k < N_VALUES; ++k) {
                for (int offset = SFEM_WARP_SIZE / 2; offset > 0; offset /= 2) {
                    v[k] = combine<T, N_SUM>(k, v[k], __shfl_down_sync(SFEM_WARP_FULL_MASK, v[k], offset));
                }
                if (!lane) shared[k][warp] = v[k];
            }
            __syncthreads();
            if (!warp) {
                for (int k = 0; k < N_VALUES; ++k) {
                    v[k] = lane < N_WARPS ? shared[k][lane] : identity<T, N_SUM>(k, infty);
                    for (int offset = SFEM_WARP_SIZE / 2; offset > 0; offset /= 2) {
                        v[k] = combine<T, N_SUM>(k, v[k], __shfl_down_sync(SFEM_WARP_FULL_MASK, v[k], offset));
                    }
                }
            }
        }

        template <typename T, int N_SUM>
        __device__ void write_partials(T (&v)[N_VALUES], const T infty, T* const SFEM_RESTRICT work) {
            block_reduce<T, N_SUM>(v, infty);
            if (!threadIdx.x) {
                for (int k = 0; k < N_VALUES; ++k) work[k * MAX_BLOCKS + blockIdx.x] = v[k];
            }
        }

        // One block: the partials of n_blocks blocks into the results at work + N_VALUES * MAX_BLOCKS.
        template <typename T, int N_SUM>
        __global__ void finish_kernel(const int n_blocks, const T infty, T* const SFEM_RESTRICT work) {
            T v[N_VALUES];
            for (int k = 0; k < N_VALUES; ++k) {
                v[k] = identity<T, N_SUM>(k, infty);
                for (int b = threadIdx.x; b < n_blocks; b += blockDim.x) {
                    v[k] = combine<T, N_SUM>(k, v[k], work[k * MAX_BLOCKS + b]);
                }
            }
            block_reduce<T, N_SUM>(v, infty);
            if (!threadIdx.x) {
                for (int k = 0; k < N_VALUES; ++k) work[N_VALUES * MAX_BLOCKS + k] = v[k];
            }
        }

        template <typename T, int N_SUM>
        void finish(const int n_blocks, const T infty, T* const work, T* const out) {
            finish_kernel<T, N_SUM><<<1, N_THREADS>>>(n_blocks, infty, work);
            SFEM_DEBUG_SYNCHRONIZE();
            SFEM_CUDA_CHECK(cudaMemcpy(out, work + N_VALUES * MAX_BLOCKS, N_VALUES * sizeof(T), cudaMemcpyDeviceToHost));
        }

        template <typename T, class B>
        __global__ void project_kernel(const ptrdiff_t n, const B b, T* const SFEM_RESTRICT x) {
            for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                x[i] = b.project(i, x[i]);
            }
        }

        template <typename T, class B>
        __global__ void chopped_gradient_kernel(const ptrdiff_t              n,
                                                const B                      b,
                                                const T* const SFEM_RESTRICT x,
                                                const T* const SFEM_RESTRICT g,
                                                T* const SFEM_RESTRICT       gc,
                                                const T                      eps) {
            for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                gc[i] = b.chopped(i, x[i], g[i], eps);
            }
        }

        template <typename T, class B>
        __global__ void curvature_kernel(const ptrdiff_t              n,
                                         const B                      b,
                                         const T* const SFEM_RESTRICT x,
                                         const T* const SFEM_RESTRICT g,
                                         const T* const SFEM_RESTRICT d,
                                         const T* const SFEM_RESTRICT Ad,
                                         const T                      infty,
                                         T* const SFEM_RESTRICT       work) {
            T v[N_VALUES] = {0, 0, 0, infty};
            for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                const T di = d[i];
                v[0] += di * Ad[i];
                v[1] += di * di;
                v[2] += g[i] * di;
                v[3] = B::min(v[3], b.max_alpha(i, x[i], di, infty));
            }
            write_partials<T, 3>(v, infty, work);
        }

        template <typename T, class B, bool ALONG>
        __global__ void step_kernel(const ptrdiff_t              n,
                                    const B                      b,
                                    const T                      alpha,
                                    const T* const SFEM_RESTRICT d,
                                    const T* const SFEM_RESTRICT Ad,
                                    T* const SFEM_RESTRICT       x,
                                    T* const SFEM_RESTRICT       g,
                                    T* const SFEM_RESTRICT       gf,
                                    const T                      eps,
                                    T* const SFEM_RESTRICT       work) {
            T v[N_VALUES] = {0, 0, 0, 0};
            for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                T xi = x[i], gi = g[i];
                if constexpr (ALONG) {
                    xi -= alpha * d[i];
                    gi -= alpha * Ad[i];
                    x[i] = xi;
                    g[i] = gi;
                }
                const T f = b.free(i, xi, gi);
                const T c = b.chopped(i, xi, gi, eps);
                gf[i]     = f;
                if constexpr (ALONG) v[0] += Ad[i] * f;
                v[1] += f * f;
                v[2] += c * c;
                v[3] += (f + c) * (f + c);
            }
            write_partials<T, N_VALUES>(v, T(0), work);
        }

        template <typename T, class B>
        __global__ void expand_kernel(const ptrdiff_t              n,
                                      const B                      b,
                                      const T                      alpha,
                                      const T                      alpha_bar,
                                      const T* const SFEM_RESTRICT d,
                                      const T* const SFEM_RESTRICT Ad,
                                      T* const SFEM_RESTRICT       x,
                                      const T* const SFEM_RESTRICT g) {
            for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                const T xi = x[i] - alpha * d[i];
                const T gi = g[i] - alpha * Ad[i];
                x[i]       = b.project(i, xi - alpha_bar * b.free(i, xi, gi));
            }
        }

        template <typename T>
        void project(const ptrdiff_t n, const T* const lb, const T* const ub, T* const x) {
            with_mprgp_bounds(lb, ub, [&](const auto b) {
                project_kernel<T, Bounds<decltype(b)>><<<n_blocks(n), N_THREADS>>>(n, b, x);
                SFEM_DEBUG_SYNCHRONIZE();
            });
        }

        template <typename T>
        void chopped_gradient(const ptrdiff_t n,
                              const T* const  lb,
                              const T* const  ub,
                              const T* const  x,
                              const T* const  g,
                              T* const        gc,
                              const T         eps) {
            with_mprgp_bounds(lb, ub, [&](const auto b) {
                chopped_gradient_kernel<T, Bounds<decltype(b)>><<<n_blocks(n), N_THREADS>>>(n, b, x, g, gc, eps);
                SFEM_DEBUG_SYNCHRONIZE();
            });
        }

        template <typename T>
        void curvature(const ptrdiff_t n,
                       const T* const  lb,
                       const T* const  ub,
                       const T* const  x,
                       const T* const  g,
                       const T* const  d,
                       const T* const  Ad,
                       const T         infty,
                       T* const        out,
                       T* const        work) {
            const int nb = n_blocks(n);
            with_mprgp_bounds(lb, ub, [&](const auto b) {
                curvature_kernel<T, Bounds<decltype(b)>><<<nb, N_THREADS>>>(n, b, x, g, d, Ad, infty, work);
                SFEM_DEBUG_SYNCHRONIZE();
            });
            finish<T, 3>(nb, infty, work, out);
        }

        template <typename T>
        void step(const ptrdiff_t n,
                  const T* const  lb,
                  const T* const  ub,
                  const T         alpha,
                  const T* const  d,
                  const T* const  Ad,
                  T* const        x,
                  T* const        g,
                  T* const        gf,
                  const T         eps,
                  T* const        out,
                  T* const        work) {
            const int nb = n_blocks(n);
            with_mprgp_bounds(lb, ub, [&](const auto b) {
                if (d) {
                    step_kernel<T, Bounds<decltype(b)>, true><<<nb, N_THREADS>>>(n, b, alpha, d, Ad, x, g, gf, eps, work);
                } else {
                    step_kernel<T, Bounds<decltype(b)>, false><<<nb, N_THREADS>>>(n, b, alpha, d, Ad, x, g, gf, eps, work);
                }
                SFEM_DEBUG_SYNCHRONIZE();
            });
            finish<T, N_VALUES>(nb, T(0), work, out);
        }

        template <typename T>
        void expand(const ptrdiff_t n,
                    const T* const  lb,
                    const T* const  ub,
                    const T         alpha,
                    const T         alpha_bar,
                    const T* const  d,
                    const T* const  Ad,
                    T* const        x,
                    const T* const  g) {
            with_mprgp_bounds(lb, ub, [&](const auto b) {
                expand_kernel<T, Bounds<decltype(b)>><<<n_blocks(n), N_THREADS>>>(n, b, alpha, alpha_bar, d, Ad, x, g);
                SFEM_DEBUG_SYNCHRONIZE();
            });
        }
    }  // namespace

    template <typename T>
    void CUDA_MPRGP<T>::build_mprgp(struct MPRGP_Tpl<T>& tpl) {
        tpl.project          = project<T>;
        tpl.chopped_gradient = chopped_gradient<T>;
        tpl.curvature        = curvature<T>;
        tpl.step             = step<T>;
        tpl.expand           = expand<T>;
        tpl.work_size        = WORK_SIZE;
    }

    template class CUDA_MPRGP<float>;
    template class CUDA_MPRGP<double>;

}  // namespace sfem
