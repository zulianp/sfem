#include "sfem_base.hpp"
#include "sfem_cuda_mprgp_impl.hpp"

#include "sfem_cuda_base.hpp"
#include "sfem_mprgp_bounds.hpp"
#include "sfem_mprgp_control.hpp"

#include <algorithm>
#include <type_traits>

namespace sfem {
    namespace {
        constexpr int N_THREADS  = 256;  // per block
        constexpr int N_WARPS    = N_THREADS / SFEM_WARP_SIZE;
        constexpr int MAX_BLOCKS = 1024;  // the loops stride over dofs past the grid
        constexpr int N_VALUES   = 4;     // reductions of one pass

        // Per-block partials, value-major (k * MAX_BLOCKS + block), then a pass's N_VALUES results
        // while the ranks complete them (REDUCED), then the count of the pass's blocks that have
        // written theirs (COUNTER, a T slot read as unsigned; zero between passes), then a run's
        // state on the device (LOOP: iterations, steps, exit, converged).
        constexpr ptrdiff_t REDUCED   = N_VALUES * MAX_BLOCKS;
        constexpr ptrdiff_t COUNTER   = REDUCED + N_VALUES;
        constexpr ptrdiff_t LOOP      = COUNTER + 1;
        constexpr ptrdiff_t WORK_SIZE = LOOP + 4;

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

        // The block's partials to `work`; the last block of the pass to write them completes the
        // pass's reduction, every partial in index order with the threads striding as one block
        // (the same arithmetic whichever block is last), and hands the results to `result`.
        template <typename T, int N_SUM, class Result>
        __device__ void reduce_pass(T (&v)[N_VALUES], const T infty, T *const work, const Result &result) {
            __shared__ bool last;
            block_reduce<T, N_SUM>(v, infty);
            unsigned *const count = reinterpret_cast<unsigned *>(work + COUNTER);
            if (!threadIdx.x) {
                for (int k = 0; k < N_VALUES; ++k) work[k * MAX_BLOCKS + blockIdx.x] = v[k];
                __threadfence();
                last = atomicAdd(count, 1u) == gridDim.x - 1;
            }
            __syncthreads();
            if (!last) return;

            for (int k = 0; k < N_VALUES; ++k) {
                v[k] = identity<T, N_SUM>(k, infty);
                for (int b = threadIdx.x; b < (int)gridDim.x; b += blockDim.x) {
                    v[k] = combine<T, N_SUM>(k, v[k], __ldcg(work + k * MAX_BLOCKS + b));
                }
            }
            block_reduce<T, N_SUM>(v, infty);
            if (!threadIdx.x) {
                result(v);
                *count = 0;
            }
        }

        template <typename T>
        __device__ MPRGPControl<T>& record(T* const control) {
            return *reinterpret_cast<MPRGPControl<T>*>(control);
        }

        // The curvature pass's results into the record, then the decision.
        template <typename T>
        struct CurvatureResult {
            T* control;
            MPRGPDecision decision;
            MPRGPParams<T> params;
            __device__ void operator()(const T (&v)[N_VALUES]) const {
                MPRGPControl<T>& c = record(control);
                c.dAd              = v[0];
                c.dd               = v[1];
                c.gd               = v[2];
                c.alpha_feas       = v[3];
                mprgp_decide(c, decision, params);
            }
        };

        // The step pass's results into the record, none where the step along d did not run; with
        // `cg` the next direction's beta.
        template <typename T>
        struct StepResult {
            T*   control;
            bool along, cg;
            __device__ void operator()(const T (&v)[N_VALUES]) const {
                MPRGPControl<T>& c = record(control);
                if (along && c.move != T(MPRGP_MOVE_STEP)) return;
                c.Adgf = v[0];
                c.gf2  = v[1];
                c.gc2  = v[2];
                c.gp2  = v[3];
                if (cg) mprgp_after_cg_step(c);
            }
        };

        // A pass's results kept for the ranks to complete.
        template <typename T>
        struct Store {
            T* values;
            __device__ void operator()(const T (&v)[N_VALUES]) const {
                for (int k = 0; k < N_VALUES; ++k) values[k] = v[k];
            }
        };

        // One thread: the completed results to `result`.
        template <typename T, class Result>
        __global__ void apply_kernel(const T* const SFEM_RESTRICT values, const Result result) {
            T v[N_VALUES];
            for (int k = 0; k < N_VALUES; ++k) v[k] = values[k];
            result(v);
        }

        // A pass's results to `result`: on one rank the pass's own (its last block, Result); across
        // ranks the pass keeps them (Store), the ranks complete them on the stream (reduce), and
        // the results of all go to `result`.
        template <typename T, class Launch, class Result>
        void run_pass(T *const work, const MPRGPReduce<T> &reduce, const bool min_last, const Launch &launch, const Result &result) {
            if (!reduce) {
                launch(result);
                SFEM_DEBUG_SYNCHRONIZE();
                return;
            }
            T *const reduced = work + REDUCED;
            launch(Store<T>{reduced});
            reduce(reduced, N_VALUES, min_last, (void *)launch_stream());
            apply_kernel<T, Result><<<1, 1, 0, launch_stream()>>>(reduced, result);
            SFEM_DEBUG_SYNCHRONIZE();
        }

        // Rounded as written, never contracted: the direction update matches the BLAS's scal and
        // axpy, which round the product and the sum one at a time.
        __device__ inline double mul_rn(const double a, const double b) { return __dmul_rn(a, b); }
        __device__ inline float  mul_rn(const float a, const float b) { return __fmul_rn(a, b); }
        __device__ inline double add_rn(const double a, const double b) { return __dadd_rn(a, b); }
        __device__ inline float  add_rn(const float a, const float b) { return __fadd_rn(a, b); }

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

        template <typename T, class B, class Result>
        __global__ void curvature_kernel(const ptrdiff_t              n,
                                         const B                      b,
                                         const T* const SFEM_RESTRICT x,
                                         const T* const SFEM_RESTRICT g,
                                         const T* const SFEM_RESTRICT d,
                                         const T* const SFEM_RESTRICT Ad,
                                         const T                      infty,
                                         T* const                     work,
                                         const Result                 result) {
            T v[N_VALUES] = {0, 0, 0, infty};
            for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                const T di = d[i];
                v[0] += di * Ad[i];
                v[1] += di * di;
                v[2] += g[i] * di;
                v[3] = B::min(v[3], b.max_alpha(i, x[i], di, infty));
            }
            reduce_pass<T, 3>(v, infty, work, result);
        }

        template <typename T, class B, bool ALONG, class Result>
        __global__ void step_kernel(const ptrdiff_t              n,
                                    const B                      b,
                                    const T* const SFEM_RESTRICT control,
                                    const T* const SFEM_RESTRICT d,
                                    const T* const SFEM_RESTRICT Ad,
                                    T* const SFEM_RESTRICT       x,
                                    T* const SFEM_RESTRICT       g,
                                    T* const SFEM_RESTRICT       gf,
                                    const T                      eps,
                                    const T* const SFEM_RESTRICT w,
                                    T* const                     work,
                                    const Result                 result) {
            const MPRGPControl<T>& c = *reinterpret_cast<const MPRGPControl<T>*>(control);
            // The same for every thread: the step along d is the record's, or none (then no block
            // writes partials, and the result, which would not use them, is not taken).
            if (ALONG && c.move != T(MPRGP_MOVE_STEP)) return;
            const T alpha = ALONG ? c.alpha : T(0);
            T       v[N_VALUES] = {0, 0, 0, 0};
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
                const T p = w ? w[i] * (f + c) : f + c;
                v[3] += p * p;
            }
            reduce_pass<T, N_VALUES>(v, T(0), work, result);
        }

        template <typename T>
        __global__ void update_direction_kernel(const ptrdiff_t              n,
                                                const T* const SFEM_RESTRICT gf,
                                                T* const SFEM_RESTRICT       p,
                                                const T* const SFEM_RESTRICT control) {
            const MPRGPControl<T>& c = *reinterpret_cast<const MPRGPControl<T>*>(control);
            if (c.move != T(MPRGP_MOVE_STEP)) return;
            const T beta = -c.beta;
            for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                p[i] = add_rn(gf[i], mul_rn(beta, p[i]));
            }
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
                project_kernel<T, Bounds<decltype(b)>><<<n_blocks(n), N_THREADS, 0, launch_stream()>>>(n, b, x);
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
                chopped_gradient_kernel<T, Bounds<decltype(b)>><<<n_blocks(n), N_THREADS, 0, launch_stream()>>>(n, b, x, g, gc, eps);
                SFEM_DEBUG_SYNCHRONIZE();
            });
        }

        template <typename T>
        void curvature(const ptrdiff_t       n,
                       const T* const        lb,
                       const T* const        ub,
                       const T* const        x,
                       const T* const        g,
                       const T* const        d,
                       const T* const        Ad,
                       const T               infty,
                       const MPRGPDecision   decision,
                       const MPRGPParams<T>& params,
                       T* const              control,
                       T* const              work,
                       const MPRGPReduce<T>& reduce) {
            const int nb = n_blocks(n);
            with_mprgp_bounds(lb, ub, [&](const auto b) {
                run_pass(work, reduce, true, [&](const auto result) {
                    curvature_kernel<<<nb, N_THREADS, 0, launch_stream()>>>(n, b, x, g, d, Ad, infty, work, result);
                }, CurvatureResult<T>{control, decision, params});
            });
        }

        template <typename T>
        void step(const ptrdiff_t n,
                  const T* const  lb,
                  const T* const  ub,
                  const T* const  d,
                  const T* const  Ad,
                  T* const        x,
                  T* const        g,
                  T* const        gf,
                  const T         eps,
                  const T* const  w,
                  const bool            cg,
                  T* const              control,
                  T* const              work,
                  const MPRGPReduce<T>& reduce) {
            const int nb = n_blocks(n);
            with_mprgp_bounds(lb, ub, [&](const auto b) {
                using Bnd = Bounds<decltype(b)>;
                // Across ranks, where the step along d did not run, the stale results are completed
                // and not used.
                run_pass(work, reduce, false, [&](const auto result) {
                    using Res = std::decay_t<decltype(result)>;
                    if (d) {
                        step_kernel<T, Bnd, true, Res><<<nb, N_THREADS, 0, launch_stream()>>>(n, b, control, d, Ad, x, g, gf, eps, w, work, result);
                    } else {
                        step_kernel<T, Bnd, false, Res><<<nb, N_THREADS, 0, launch_stream()>>>(n, b, control, d, Ad, x, g, gf, eps, w, work, result);
                    }
                }, StepResult<T>{control, d != nullptr, cg});
            });
        }

        template <typename T>
        void update_direction(const ptrdiff_t n, const T* const gf, T* const p, const T* const control) {
            update_direction_kernel<T><<<n_blocks(n), N_THREADS, 0, launch_stream()>>>(n, gf, p, control);
            SFEM_DEBUG_SYNCHRONIZE();
        }

        // The record's pinned copy on the host, one per thread: a write to the device is then
        // asynchronous, ordered before the passes that read it. The next write follows a read,
        // which waits for the previous one.
        template <typename T>
        MPRGPControl<T>* staging() {
            static thread_local MPRGPControl<T>* pinned = [] {
                MPRGPControl<T>* p = nullptr;
                SFEM_CUDA_CHECK(cudaMallocHost(&p, sizeof(MPRGPControl<T>)));
                return p;
            }();
            return pinned;
        }

        template <typename T>
        void read_control(const T* const control, MPRGPControl<T>* const host) {
            MPRGPControl<T>* const s = staging<T>();
            SFEM_CUDA_CHECK(cudaMemcpy(s, control, sizeof(MPRGPControl<T>), cudaMemcpyDeviceToHost));
            *host = *s;
        }

        template <typename T>
        void write_control(const MPRGPControl<T>& host, T* const control) {
            MPRGPControl<T>* const s = staging<T>();
            *s                       = host;
            SFEM_CUDA_CHECK(cudaMemcpyAsync(control, s, sizeof(MPRGPControl<T>), cudaMemcpyHostToDevice, 0));
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
                expand_kernel<T, Bounds<decltype(b)>><<<n_blocks(n), N_THREADS, 0, launch_stream()>>>(n, b, alpha, alpha_bar, d, Ad, x, g);
                SFEM_DEBUG_SYNCHRONIZE();
            });
        }
        // A run starts from `iterations` done, no step taken, nothing decided.
        template <typename T>
        __global__ void run_start_kernel(T* const SFEM_RESTRICT loop, const T iterations) {
            loop[0] = iterations;
            loop[1] = 0;
            loop[2] = 0;
            loop[3] = 0;
        }

        // After a CG iteration, as MPRGP::team_cg_steps takes it: a decision that is no step ends
        // the run, not counted; a step is counted, then the run ends converged or at max_it, or
        // before a proportioning step, or goes on (the graph's loop condition).
        template <typename T>
        __global__ void run_decide_kernel(const cudaGraphConditionalHandle handle,
                                          const T* const SFEM_RESTRICT control,
                                          T* const SFEM_RESTRICT       loop,
                                          const T                      gamma,
                                          const T                      atol,
                                          const T                      rtol,
                                          const T                      norm_gp0,
                                          const int                    max_it) {
            const MPRGPControl<T>& c  = *reinterpret_cast<const MPRGPControl<T>*>(control);
            unsigned               go = 0;
            if (c.move != T(MPRGP_MOVE_STEP)) {
                loop[2] = T(MPRGPRunResult::NOT_A_STEP);
            } else {
                loop[1] += 1;
                const T norm_gf = sqrt(c.gf2), norm_gc = sqrt(c.gc2), norm_gp = sqrt(c.gp2);
                const T rel     = norm_gp / norm_gp0;
                const bool done = norm_gp < atol || rel < rtol;
                loop[0] += 1;
                loop[3] = done;
                if (loop[0] >= T(max_it) || done) {
                    loop[2] = T(MPRGPRunResult::DONE);
                } else if (!(norm_gc <= gamma * norm_gf)) {
                    loop[2] = T(MPRGPRunResult::PROPORTIONING);
                } else {
                    go = 1;
                }
            }
            cudaGraphSetConditional(handle, go);
        }

        // A solve's run graph: a WHILE node whose body is one CG iteration -- the operator's
        // product, the curvature pass, the step pass, the next direction -- and the decision that
        // sets its condition, captured from the launches themselves (LaunchStreamScope), so the
        // iteration is the one the host would launch. The stream blocks: it follows the legacy
        // stream's work and that follows it.
        template <typename T>
        struct RunGraph {
            cudaStream_t    stream{nullptr};
            cudaGraph_t     graph{nullptr};
            cudaGraphExec_t exec{nullptr};
            T*              host{nullptr};  // pinned: the run's state read back
            bool            failed{false};

            ~RunGraph() {
                if (exec) cudaGraphExecDestroy(exec);
                if (graph) cudaGraphDestroy(graph);
                if (stream) cudaStreamDestroy(stream);
                if (host) cudaFreeHost(host);
            }

            bool build(const MPRGPRun<T>& r) {
                SFEM_CUDA_CHECK(cudaStreamCreate(&stream));
                SFEM_CUDA_CHECK(cudaMallocHost(&host, 4 * sizeof(T)));
                SFEM_CUDA_CHECK(cudaGraphCreate(&graph, 0));
                cudaGraphConditionalHandle handle;
                SFEM_CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, graph, 1, cudaGraphCondAssignDefault));
                cudaGraphNodeParams params = {};
                params.type                = cudaGraphNodeTypeConditional;
                params.conditional.handle  = handle;
                params.conditional.type    = cudaGraphCondTypeWhile;
                params.conditional.size    = 1;
                cudaGraphNode_t node;
                SFEM_CUDA_CHECK(cudaGraphAddNode(&node, graph, nullptr, 0, &params));
                cudaGraph_t body = params.conditional.phGraph_out[0];

                SFEM_CUDA_CHECK(cudaStreamBeginCaptureToGraph(stream, body, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal));
                {
                    LaunchStreamScope scope(stream);
                    r.product(r.p, r.Ap);
                    curvature<T>(r.n, r.lb, r.ub, r.x, r.g, r.p, r.Ap, r.infty, MPRGP_DECIDE_CG, r.params, r.control, r.work, {});
                    step<T>(r.n, r.lb, r.ub, r.p, r.Ap, r.x, r.g, r.gf, r.eps, r.w, true, r.control, r.work, {});
                    update_direction<T>(r.n, r.gf, r.p, r.control);
                    run_decide_kernel<T><<<1, 1, 0, stream>>>(
                            handle, r.control, r.work + LOOP, r.gamma, r.atol, r.rtol, r.norm_gp0, r.max_it);
                }
                cudaGraph_t captured = nullptr;
                // A launch the capture could not take (on the legacy stream, a synchronisation)
                // leaves it invalid: the run is then the host's, iteration by iteration.
                if (cudaStreamEndCapture(stream, &captured) != cudaSuccess || !captured ||
                    cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) {
                    cudaGetLastError();
                    return false;
                }
                return true;
            }
        };

        template <typename T>
        MPRGPRunResult cg_run(const MPRGPRun<T>& r, const int iterations, std::shared_ptr<void>& state) {
            auto run = std::static_pointer_cast<RunGraph<T>>(state);
            if (!run) {
                run   = std::make_shared<RunGraph<T>>();
                state = run;
                if (!run->build(r)) run->failed = true;
            }
            MPRGPRunResult result;
            if (run->failed) return result;

            T* const loop = r.work + LOOP;
            run_start_kernel<T><<<1, 1, 0, run->stream>>>(loop, T(iterations));
            SFEM_CUDA_CHECK(cudaGraphLaunch(run->exec, run->stream));
            SFEM_CUDA_CHECK(cudaMemcpyAsync(run->host, loop, 4 * sizeof(T), cudaMemcpyDeviceToHost, run->stream));
            SFEM_CUDA_CHECK(cudaStreamSynchronize(run->stream));
            result.ran        = true;
            result.iterations = int(run->host[0]);
            result.steps      = int(run->host[1]);
            result.exit       = MPRGPRunResult::Exit(int(run->host[2]));
            result.converged  = run->host[3] != T(0);
            return result;
        }
    }  // namespace

    template <typename T>
    void CUDA_MPRGP<T>::build_mprgp(struct MPRGP_Tpl<T>& tpl) {
        tpl.project          = project<T>;
        tpl.chopped_gradient = chopped_gradient<T>;
        tpl.curvature        = curvature<T>;
        tpl.step             = step<T>;
        tpl.expand           = expand<T>;
        tpl.update_direction = update_direction<T>;
        tpl.read_control     = read_control<T>;
        tpl.write_control    = write_control<T>;
        tpl.cg_run           = cg_run<T>;
        tpl.work_size        = WORK_SIZE;
    }

    template class CUDA_MPRGP<float>;
    template class CUDA_MPRGP<double>;

}  // namespace sfem
