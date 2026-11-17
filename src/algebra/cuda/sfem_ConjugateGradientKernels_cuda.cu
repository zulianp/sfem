#include "sfem_ConjugateGradientKernels_cuda.hpp"

#include "sfem_aliases.hpp"
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

        // The preconditioned loop on the device (ConjugateGradientKernels::run_precond, CGDeviceRun).
        // The host's two waits per iteration dominate the coarse solve (MI300A 646 us an iteration,
        // GH200 337, MI300A's kernels ~130 us). Here the scalars stay in a device array, each
        // iteration's x and r update is skipped once the run has stopped, and RUN_BATCH iterations are
        // one captured graph after which the host reads whether it stopped: x is the loop's, at most
        // RUN_BATCH - 1 iterations' products spent after the stop. The graph is what pays (launched
        // without it the run gains nothing); it needs products that allocate nothing on the stream,
        // which HIP 6.4 does not capture (the generated operators keep their connectivity table
        // under HIP for this).
        enum { RUN_RTZ = 0, RUN_RTZ_NEW, RUN_PTAP, RUN_ITER, RUN_STOP, RUN_RTZ_NEXT, RUN_N };
        static constexpr int RUN_BATCH = 16;

        // An iteration is seven graph nodes and the operators' own: z += M r, r.z, p's update, Ap += A p,
        // p.Ap, x and r's update, the decision. beta and alpha are formed where they are used, each dot
        // adds into a word the kernel before it zeroed, and z and Ap are zeroed by the updates where
        // they are dead (no memset node); a node costs ~1.7 us on MI300A and ~0.5 on GH200 before it
        // moves a byte.

        // s[slot] += a . b. One block of RUN_DOT_BLOCK_SIZE threads per compute unit: the blocks'
        // atomics on the one word serialise (MI300A, 307 k entries: 5.8 us at one block of 512 per
        // unit, 11.8 at four of 128; GH200 5.2 us either way).
        static constexpr int RUN_DOT_BLOCK_SIZE = 512;
        template <typename T>
        __global__ void SFEM_HIP_LAUNCH_BOUNDS(RUN_DOT_BLOCK_SIZE)
                run_dot_kernel(const ptrdiff_t n, T* const SFEM_RESTRICT s, const int slot, const T* const SFEM_RESTRICT a, const T* const SFEM_RESTRICT b) {
            __shared__ T block_accumulator[SFEM_WARP_SIZE];
            if (s[RUN_STOP] != T(0)) return;

            T acc = 0;
            for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                acc += a[i] * b[i];
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
                    atomicAdd(&s[slot], acc);
                }
            }
        }

        // The loop's stop: |r|_M below atol or rtol relative to the first, or max_it iterations. The
        // iteration count is the loop's (the index of the stopping iteration, max_it without a stop).
        // rtz = rtz_new here, after its last reader, and rtz_new zeroed for the next r.z.
        template <typename T>
        __global__ void run_decide_kernel(T* const SFEM_RESTRICT s, const T rtz0, const T atol, const T rtol, const int max_it) {
            if (s[RUN_STOP] != T(0)) return;
            const T rtz     = s[RUN_RTZ_NEW];
            s[RUN_RTZ]      = rtz;
            s[RUN_RTZ_NEW]  = T(0);
            const T anorm = sqrt(rtz);
            const T rnorm = anorm / sqrt(rtz0);
            if (anorm < atol || rnorm < rtol) {
                s[RUN_STOP] = T(1);
                return;
            }
            s[RUN_ITER] += T(1);
            if (s[RUN_ITER] >= T(max_it)) s[RUN_STOP] = T(2);
        }

        // update_p_kernel's p = z + beta p, beta = rtz_new / rtz; p.Ap's word zeroed, and Ap (dead
        // until the product adds into it), `allocation` entries.
        template <typename T>
        __global__ void run_update_p_kernel(const ptrdiff_t              n,
                                            const ptrdiff_t              allocation,
                                            T* const SFEM_RESTRICT       s,
                                            const T* const SFEM_RESTRICT z,
                                            T* const SFEM_RESTRICT       p,
                                            T* const SFEM_RESTRICT       Ap) {
            if (s[RUN_STOP] != T(0)) return;
            const T beta = s[RUN_RTZ_NEW] / s[RUN_RTZ];
            if (!blockIdx.x && !threadIdx.x) s[RUN_PTAP] = T(0);
            for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                p[i]  = z[i] + beta * p[i];
                Ap[i] = T(0);
            }
            for (ptrdiff_t i = n + blockIdx.x * blockDim.x + threadIdx.x; i < allocation; i += blockDim.x * gridDim.x) {
                Ap[i] = T(0);
            }
        }

        // update_x_r_kernel's x += alpha p, r -= alpha Ap, alpha = rtz_new / p.Ap; z (dead until the
        // preconditioner adds into it) zeroed, `allocation` entries.
        template <typename T>
        __global__ void run_update_x_r_kernel(const ptrdiff_t              n,
                                              const ptrdiff_t              allocation,
                                              const T* const SFEM_RESTRICT s,
                                              const T* const SFEM_RESTRICT p,
                                              const T* const SFEM_RESTRICT Ap,
                                              T* const SFEM_RESTRICT       x,
                                              T* const SFEM_RESTRICT       r,
                                              T* const SFEM_RESTRICT       z) {
            if (s[RUN_STOP] != T(0)) return;
            const T alpha = s[RUN_RTZ_NEW] / s[RUN_PTAP];
            for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                x[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
                z[i] = T(0);
            }
            for (ptrdiff_t i = n + blockIdx.x * blockDim.x + threadIdx.x; i < allocation; i += blockDim.x * gridDim.x) {
                z[i] = T(0);
            }
        }

        // Where M is 3 x 3 blocks a node (CGDeviceRun::m_blocks): run_update_x_r_kernel's x and r, then
        // z = M r (as sbv3_apply forms it) and r.z into RUN_RTZ_NEXT, for the next iteration. The
        // preconditioner's node and r.z's go: an iteration is five nodes and the product's. A
        // decision in the last block instead (a fence a block) costs more than its node on MI300A.
        // MaMAL's coarse size (307 k entries, a stand-in product): MI300A 41.5-44.1 us an iteration
        // against 49.1-50.3, H100 21.8 against 27.4.
        static constexpr int RUN_M3_BLOCK_SIZE = 256;
        template <typename T, typename B>
        __global__ void SFEM_HIP_LAUNCH_BOUNDS(RUN_M3_BLOCK_SIZE)
                run_update_x_r_m3_kernel(const ptrdiff_t              n_blocks,
                                         T* const SFEM_RESTRICT       s,
                                         const B* const SFEM_RESTRICT blocks,
                                         const T* const SFEM_RESTRICT p,
                                         const T* const SFEM_RESTRICT Ap,
                                         T* const SFEM_RESTRICT       x,
                                         T* const SFEM_RESTRICT       r,
                                         T* const SFEM_RESTRICT       z) {
            __shared__ T block_accumulator[SFEM_WARP_SIZE];
            if (s[RUN_STOP] != T(0)) return;
            const T alpha = s[RUN_RTZ_NEW] / s[RUN_PTAP];

            T acc = 0;
            for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_blocks; i += blockDim.x * gridDim.x) {
                T ri[3];
                for (int d = 0; d < 3; d++) {
                    x[i * 3 + d] += alpha * p[i * 3 + d];
                    ri[d]        = r[i * 3 + d] - alpha * Ap[i * 3 + d];
                    r[i * 3 + d] = ri[d];
                }
                const B* const di = &blocks[i * 9];
                for (int d1 = 0; d1 < 3; d1++) {
                    T zi = 0;
                    for (int d2 = 0; d2 < 3; d2++) {
                        zi += (T)di[d1 * 3 + d2] * ri[d2];
                    }
                    z[i * 3 + d1] = zi;
                    acc += ri[d1] * zi;
                }
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
                    atomicAdd(&s[RUN_RTZ_NEXT], acc);
                }
            }
        }

        // run_decide_kernel, then the next iteration's r.z (run_update_x_r_m3_kernel's) as rtz_new.
        template <typename T>
        __global__ void run_decide_next_kernel(T* const SFEM_RESTRICT s, const T rtz0, const T atol, const T rtol, const int max_it) {
            if (s[RUN_STOP] != T(0)) return;
            const T rtz     = s[RUN_RTZ_NEW];
            s[RUN_RTZ]      = rtz;
            s[RUN_RTZ_NEW]  = s[RUN_RTZ_NEXT];
            s[RUN_RTZ_NEXT] = T(0);
            const T anorm = sqrt(rtz);
            const T rnorm = anorm / sqrt(rtz0);
            if (anorm < atol || rnorm < rtol) {
                s[RUN_STOP] = T(1);
                return;
            }
            s[RUN_ITER] += T(1);
            if (s[RUN_ITER] >= T(max_it)) s[RUN_STOP] = T(2);
        }

        // The run's scalars on the device and its stop read back, kept for the thread's later runs.
        template <typename T>
        struct RunScalars {
            T* device{nullptr};
            T* host{nullptr};  // pinned: RUN_ITER, RUN_STOP
            RunScalars() {
                SFEM_CUDA_CHECK(cudaMalloc((void**)&device, RUN_N * sizeof(T)));
                SFEM_CUDA_CHECK(cudaMallocHost((void**)&host, 2 * sizeof(T)));
            }
            ~RunScalars() {
                if (device) cudaFree(device);
                if (host) cudaFreeHost(host);
            }
        };

        template <typename T>
        static CGDeviceRunResult run_precond(const CGDeviceRun<T>& run) {
            SFEM_TRACE_SCOPE("ConjugateGradient::device_run");
            static thread_local RunScalars<T> scalars;
            T* const           s      = scalars.device;
            const cudaStream_t stream     = static_cast<cudaStream_t>(run.stream);
            const ptrdiff_t    allocation = std::max(run.n, run.allocation);

            int       block_size;
            ptrdiff_t n_blocks;
            cg_launch_params(run.n, block_size, n_blocks);
            static const int units = [] {
                int device = 0, count = 0;
                cudaGetDevice(&device);
                cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device);
                return std::max(1, count);
            }();
            const int dot_blocks = int(std::min(ptrdiff_t(units), (run.n + RUN_DOT_BLOCK_SIZE - 1) / RUN_DOT_BLOCK_SIZE));

            const ptrdiff_t m3_blocks   = run.n / 3;
            const ptrdiff_t m3_n_blocks = std::max(ptrdiff_t(1), (m3_blocks + RUN_M3_BLOCK_SIZE - 1) / RUN_M3_BLOCK_SIZE);
            auto update_x_r_m3 = [&]() {
                if (run.m_blocks_bytes == int(sizeof(float))) {
                    run_update_x_r_m3_kernel<T, float><<<m3_n_blocks, RUN_M3_BLOCK_SIZE, 0, stream>>>(
                            m3_blocks, s, static_cast<const float*>(run.m_blocks), run.p, run.Ap, run.x, run.r, run.z);
                } else {
                    run_update_x_r_m3_kernel<T, double><<<m3_n_blocks, RUN_M3_BLOCK_SIZE, 0, stream>>>(
                            m3_blocks, s, static_cast<const double*>(run.m_blocks), run.p, run.Ap, run.x, run.r, run.z);
                }
            };
            const bool m3 = run.m_blocks && (run.m_blocks_bytes == int(sizeof(float)) || run.m_blocks_bytes == int(sizeof(double)));

            // One of the loop's iterations, its scalars in s.
            auto iteration = [&]() {
                if (m3) {
                    run_update_p_kernel<T><<<n_blocks, block_size, 0, stream>>>(run.n, allocation, s, run.z, run.p, run.Ap);
                    run.product(run.p, run.Ap);
                    run_dot_kernel<T><<<dot_blocks, RUN_DOT_BLOCK_SIZE, 0, stream>>>(run.n, s, RUN_PTAP, run.p, run.Ap);
                    update_x_r_m3();
                    run_decide_next_kernel<T><<<1, 1, 0, stream>>>(s, run.rtz0, run.atol, run.rtol, run.max_it);
                    return;
                }
                run.precondition(run.r, run.z);
                run_dot_kernel<T><<<dot_blocks, RUN_DOT_BLOCK_SIZE, 0, stream>>>(run.n, s, RUN_RTZ_NEW, run.r, run.z);
                run_update_p_kernel<T><<<n_blocks, block_size, 0, stream>>>(run.n, allocation, s, run.z, run.p, run.Ap);
                run.product(run.p, run.Ap);
                run_dot_kernel<T><<<dot_blocks, RUN_DOT_BLOCK_SIZE, 0, stream>>>(run.n, s, RUN_PTAP, run.p, run.Ap);
                run_update_x_r_kernel<T><<<n_blocks, block_size, 0, stream>>>(run.n, allocation, s, run.p, run.Ap, run.x, run.r, run.z);
                run_decide_kernel<T><<<1, 1, 0, stream>>>(s, run.rtz0, run.atol, run.rtol, run.max_it);
            };
            auto batch = [&]() {
                for (int k = 0; k < RUN_BATCH; ++k) iteration();
                SFEM_CUDA_CHECK(cudaMemcpyAsync(scalars.host, s + RUN_ITER, 2 * sizeof(T), cudaMemcpyDeviceToHost, stream));
            };

            const T start[RUN_N] = {run.rtz, T(0), T(0), T(0), T(0), T(0)};
            SFEM_CUDA_CHECK(cudaMemcpyAsync(s, start, RUN_N * sizeof(T), cudaMemcpyHostToDevice, stream));
            // The first z += M r adds into zeros; each Ap is zeroed by the p update before it.
            SFEM_CUDA_CHECK(cudaMemsetAsync(run.z, 0, allocation * sizeof(T), stream));
            if (m3) {
                // The loop's first z = M r and r.z, which its iterations form for the next.
                run.precondition(run.r, run.z);
                run_dot_kernel<T><<<dot_blocks, RUN_DOT_BLOCK_SIZE, 0, stream>>>(run.n, s, RUN_RTZ_NEW, run.r, run.z);
            }
            SFEM_CUDA_CHECK(cudaStreamSynchronize(stream));

            // The batch as one graph, captured from the run's own launches; launched as they are
            // where it cannot be captured.
            cudaGraph_t     graph = nullptr;
            cudaGraphExec_t exec  = nullptr;
            if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
                batch();
                const cudaError_t ended = cudaStreamEndCapture(stream, &graph);
                // A launch that failed while capturing is missing from the graph: launched as they
                // are, the batch reports it.
                const cudaError_t launched = cudaGetLastError();
                if (ended != cudaSuccess || launched != cudaSuccess || !graph || cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) {
                    if (exec) cudaGraphExecDestroy(exec);
                    exec = nullptr;
                }
            }
            cudaGetLastError();

            // The decision stops the run by max_it at the latest, within max_it / RUN_BATCH + 1 batches.
            const int max_batches = run.max_it / RUN_BATCH + 2;
            for (int b = 0; b < max_batches; ++b) {
                if (exec) {
                    SFEM_CUDA_CHECK(cudaGraphLaunch(exec, stream));
                } else {
                    batch();
                }
                SFEM_CUDA_CHECK(cudaStreamSynchronize(stream));
                if (scalars.host[1] != T(0)) break;
            }
            if (exec) cudaGraphExecDestroy(exec);
            if (graph) cudaGraphDestroy(graph);

            CGDeviceRunResult result;
            result.ran        = true;
            result.iterations = int(scalars.host[0]);
            result.converged  = scalars.host[1] == T(1);
            return result;
        }

        // SFEM_CG_DEVICE_RUN: the preconditioned loop on the device (1, the default) or the host's (0).
        // MaMAL 136 M, alternated three times on one node: MI300A 31.4 s against 34.0; GH200 16.18-16.71
        // s against 16.94-17.56 (0.7 M: 1.04-1.10 against 1.09-1.19 s), the same iterations.
        static bool cg_device_run_enabled() {
            static int enabled = -1;
            if (enabled < 0) {
                int SFEM_CG_DEVICE_RUN = 1;
                SFEM_READ_ENV(SFEM_CG_DEVICE_RUN, atoi);
                enabled = SFEM_CG_DEVICE_RUN;
            }
            return enabled != 0;
        }

    }  // namespace

    template <typename T>
    void GPUConjugateGradientKernels<T>::build(struct ConjugateGradientKernels<T>& tpl, const ExecutionContext& context) {
        // Every pass on the context's stream, bound here once.
        const cudaStream_t s   = static_cast<cudaStream_t>(context.stream());
        tpl.update_x_r_and_rtr = [s](auto&&... a) { return update_x_r_and_rtr<T>(a..., s); };
        tpl.update_x_r         = [s](auto&&... a) { update_x_r<T>(a..., s); };
        tpl.update_p           = [s](auto&&... a) { update_p<T>(a..., s); };
        if (cg_device_run_enabled()) tpl.run_precond = [](const CGDeviceRun<T>& run) { return run_precond<T>(run); };
    }

    template class GPUConjugateGradientKernels<float>;
    template class GPUConjugateGradientKernels<double>;

}  // namespace sfem
