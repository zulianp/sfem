#pragma once

// The stored-tangent operator on the device (docs/compressed_partial_assembly): the host stores'
// encodings (sfem_PointStore.hpp, sfem_ElementMatrixStore.hpp) in the layouts and kernel shapes the
// GPU measurements chose (spikes/inexact_apply_compare/qp/RESULTS_QP.md):
//
//   TET4, TET10, HEX8  a thread per element.  The codes in blocks of 32 elements, one warp, so a
//                      warp's loads of one code are one transaction.  The element's stages are the
//                      generated device stages; decoding and the contraction are unrolled over the
//                      element's tensors (or node blocks), so the stored numbers stay in registers.
//                      HEX8 folds.  The scatter is atomic or, where it measured faster, block-local:
//                      a block's elements accumulate in shared memory and its nodes are added once.
//   PROTEUS_HEX27      64 points per element are too much for a thread.  The point store runs a
//                      thread per point (codes [element][component][point], so a warp's loads are
//                      consecutive), the output by the transposed stages in three one-dimensional
//                      sweeps over the element's threads; the element matrices a warp per element,
//                      a lane per node.  Not folded: the points' masks differ within a warp, so the
//                      fold saves no multiply-add and its decode costs more than its bytes.
//
// `M` is a generated device unit (`<unit>_<element>_inexact.cuh`); a generated source per unit and
// element instantiates `create_store<M>` (emitters/inexact_codegen.py).

#include "sfem_InexactBlockMap.hpp"
#include "sfem_cuda_InexactSetup.cuh"

namespace sfem {
    namespace inexact {
        namespace cuda {

            // ------------------------------------------------------------------ the shapes

            /// The scatter and the resident blocks per SM a thread-per-element application is
            /// compiled for, as measured per element and store (RESULTS_QP.md, measured_gpu/eshape_*).
            struct Shape {
                bool block;
                int  minb;
            };

            template <class El, bool MATRICES, typename code_t>
            constexpr Shape shape() {
                constexpr bool half = std::is_same_v<code_t, __half>;
                if constexpr (El::N == 4) {
                    return {true, 1};
                } else if constexpr (El::N == 8) {
                    return half ? Shape{true, 4} : Shape{false, 1};
                } else if constexpr (El::N == 10) {
                    return half && !MATRICES ? Shape{true, 4} : Shape{false, 1};
                } else {
                    return {false, 1};
                }
            }

            // --------------------------------------------------------------- the traversal

            /// What an application reads besides its store, by value.  The block-local scatter's map
            /// (`BlockMap`): each block of `THREADS` consecutive elements' distinct nodes, and each
            /// element's nodes, in staging order, as indices into its block's list.
            struct ApplyArgs {
                ptrdiff_t           nelements{0};
                Connectivity        elements;
                ptrdiff_t           stride{3};
                const int32_t      *block_ptr{nullptr};
                const idx_t        *block_nodes{nullptr};
                const uint16_t     *local{nullptr};  // [staged node][element]
                int                 max_block_nodes{0};
            };

            class BlockMap {
            public:
                /// From the device connectivity in the staging order.
                template <class El>
                static std::unique_ptr<BlockMap> build(const ptrdiff_t ne, const Connectivity &elements) {
                    static constexpr int N = El::N;
                    std::vector<std::vector<idx_t>> conn(N, std::vector<idx_t>((size_t)ne));
                    for (int s = 0; s < N; ++s)
                        check(cudaMemcpy(conn[s].data(), elements.node[s], ne * sizeof(idx_t), cudaMemcpyDeviceToHost), "connectivity");
                    BlockLocalMap host = block_local_map(ne, N, conn, THREADS);
                    auto map              = std::make_unique<BlockMap>();
                    map->ptr_             = DeviceArray<int32_t>::from(host.ptr);
                    map->nodes_           = DeviceArray<idx_t>::from(host.nodes);
                    map->local_           = DeviceArray<uint16_t>::from(host.local);
                    map->max_block_nodes_ = host.max_block_nodes;
                    return map;
                }

                void bind(ApplyArgs &a) const {
                    a.block_ptr       = ptr_.data;
                    a.block_nodes     = nodes_.data;
                    a.local           = local_.data;
                    a.max_block_nodes = max_block_nodes_;
                }
                std::size_t shared_bytes() const { return (std::size_t)6 * max_block_nodes_ * sizeof(real_t); }

            private:
                DeviceArray<int32_t>  ptr_;
                DeviceArray<idx_t>    nodes_;
                DeviceArray<uint16_t> local_;
                int                   max_block_nodes_{0};
            };

            /// A thread's element: its gather and its scatter, atomic or block-local.
            template <class El, bool BLOCK>
            struct Traversal {
                static constexpr int N = El::N;
                const ApplyArgs     &a;
                const ptrdiff_t      e;
                const bool           active;
                real_t              *hs{nullptr}, *acc{nullptr};  // block: [local node][component]
                int32_t              nb{0};
                const idx_t         *bn{nullptr};
                idx_t                node[N];

                __device__ Traversal(const ApplyArgs &a_, const real_t *const RSTR h, real_t *const smem)
                    : a(a_), e((ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x), active(e < a_.nelements) {
                    if constexpr (BLOCK) {
                        hs               = smem;
                        acc              = smem + 3 * a.max_block_nodes;
                        const int32_t b0 = a.block_ptr[blockIdx.x];
                        nb               = a.block_ptr[blockIdx.x + 1] - b0;
                        bn               = a.block_nodes + b0;
                        for (int l = threadIdx.x; l < nb; l += blockDim.x) {
                            const ptrdiff_t v = (ptrdiff_t)bn[l] * a.stride;
                            for (int k = 0; k < 3; ++k) hs[3 * l + k] = h[v + k], acc[3 * l + k] = 0;
                        }
                        __syncthreads();
                    }
                }

                // The block's shared arrays are indexed in 32 bits: 64-bit indices cost the TET4 element
                // matrices 14 registers and a resident block per SM.
                __device__ __forceinline__ void gather(const real_t *const RSTR h, real_t hh[3][N]) {
                    if (!active) return;
                    static_for<0, N>([&](auto S) {
                        constexpr int s = decltype(S)::value;
                        if constexpr (BLOCK) {
                            node[s] = a.local[(size_t)s * a.nelements + e];
                            for (int k = 0; k < 3; ++k) hh[k][s] = hs[node[s] * 3 + k];
                        } else {
                            node[s] = a.elements.node[s][e];
                            for (int k = 0; k < 3; ++k) hh[k][s] = h[(ptrdiff_t)node[s] * a.stride + k];
                        }
                    });
                }

                __device__ __forceinline__ void scatter(const real_t o[3][N], real_t *const RSTR out) {
                    if constexpr (BLOCK) {
                        if (active)
                            for (int s = 0; s < N; ++s)
                                for (int k = 0; k < 3; ++k) atomicAdd(&acc[node[s] * 3 + k], o[k][s]);
                        __syncthreads();
                        for (int l = threadIdx.x; l < nb; l += blockDim.x) {
                            const ptrdiff_t v = (ptrdiff_t)bn[l] * a.stride;
                            for (int k = 0; k < 3; ++k) atomicAdd(&out[v + k], acc[3 * l + k]);
                        }
                    } else {
                        if (!active) return;
                        for (int s = 0; s < N; ++s)
                            for (int k = 0; k < 3; ++k) atomicAdd(&out[(ptrdiff_t)node[s] * a.stride + k], o[k][s]);
                    }
                }
            };

            // ------------------------------------------------------------- the point store

            template <bool FOLD, class Fold>
            constexpr int stored_numbers_device(const int unfolded) {
                if constexpr (FOLD) {
                    return Fold::NS;
                } else {
                    return unfolded;
                }
            }

            /// The point store's codes on the device and its encoding (the host's encode_block): the
            /// mean (MD), the fold, the element scale, the codes.  Blocked [block][slot][lane] for a
            /// thread per element; [element][component][point] (`POINT_MAJOR`) for a thread per point.
            template <class M, typename code_t, bool MD, bool FOLD, bool POINT_MAJOR>
            struct PointCodes {
                using El                      = typename M::element;
                using C                       = Code<code_t>;
                using Fold                    = typename FoldOf<M>::type;
                static constexpr int  NV      = El::NV, NC = M::NC, NQ = El::NQ;
                static constexpr int  NSE     = stored_numbers_device<FOLD, Fold>(NV * NC);
                static constexpr bool scaled  = C::scaled;
                static constexpr bool has_mean = MD;
                static constexpr int  extra_doubles = 0, scratch_doubles = 0;
                static_assert(!(POINT_MAJOR && FOLD), "a thread per point does not fold");

                typename C::storage *codes{nullptr};
                float               *scales{nullptr};
                float               *means{nullptr};

                __device__ __forceinline__ std::size_t code_index(const ptrdiff_t e, const int slot) const {
                    return ((size_t)(e / V) * NSE + slot) * V + (size_t)(e % V);
                }
                __device__ __forceinline__ std::size_t point_index(const ptrdiff_t e, const int j, const int c) const {
                    return ((size_t)e * NC + c) * NV + j;
                }
                __device__ __forceinline__ std::size_t mean_index(const ptrdiff_t e, const int c) const {
                    if constexpr (POINT_MAJOR) {
                        return (size_t)e * NC + c;
                    } else {
                        return ((size_t)(e / V) * NC + c) * V + (size_t)(e % V);
                    }
                }
                /// Tensor j's k-th stored number: its component c, or its k-th kept one when folded.
                __device__ __forceinline__ std::size_t index(const TablesView &t, const ptrdiff_t e, const int j, const int k) const {
                    if constexpr (POINT_MAJOR) {
                        return point_index(e, j, k);
                    } else if constexpr (FOLD) {
                        return code_index(e, t.off[j] + k);
                    } else {
                        return code_index(e, j * NC + k);
                    }
                }

                __device__ void encode(const TablesView &t,
                                       const ptrdiff_t   e,
                                       const bool        active,
                                       const int         q,
                                       double (*const S)[NC],
                                       double *const     tops,
                                       double *const,
                                       double *const) const {
                    if constexpr (MD) {
                        if (active)
                            for (int c = q; c < NC; c += NQ) {
                                double acc = 0;
                                for (int j = 0; j < NV; ++j) acc += t.mult[j] * S[j][c];
                                const float mean       = (float)(acc / El::KVOL);
                                means[mean_index(e, c)] = mean;
                                for (int j = 0; j < NV; ++j) S[j][c] -= t.x[j] * (double)mean;
                            }
                        __syncthreads();
                    }
                    double mine = 0;
                    if (active) {
                        if constexpr (FOLD) {
                            // A dropped point's numbers in a direction it drops are never changed, so
                            // the threads fold their own points without waiting for each other.
                            for (int j = q; j < NV; j += NQ) {
                                const int mask = t.mask[j];
                                for (int c = 0; c < NC; ++c) {
                                    const int d = t.dir[c];
                                    if (d >= 0 && !((mask >> d) & 1)) S[j][c] += t.coefficient[j * 3 + d] * S[t.source[j * 3 + d]][c];
                                }
                                for (int k = 0; k < t.nkeep[mask]; ++k) mine = fmax(mine, fabs(S[j][t.keep[mask * NC + k]]));
                            }
                        } else {
                            const double *const flat = &S[0][0];
                            for (int i = q; i < NV * NC; i += NQ) mine = fmax(mine, fabs(flat[i]));
                        }
                    }
                    double inv = 1;
                    if constexpr (scaled) {
                        const double s = C::scale_for(element_max<NQ>(tops, q, mine));
                        if (active && q == 0) scales[e] = (float)s;
                        inv = 1.0 / (double)(float)s;
                    }
                    if (!active) return;
                    if constexpr (FOLD) {
                        for (int j = q; j < NV; j += NQ) {
                            const int mask = t.mask[j];
                            for (int k = 0; k < t.nkeep[mask]; ++k) codes[index(t, e, j, k)] = C::encode(S[j][t.keep[mask * NC + k]], inv);
                        }
                    } else if constexpr (POINT_MAJOR) {
                        // In the codes' own order, [component][point]: a warp's stores are contiguous
                        // (in the tensors' order they are a point apart, NV codes).
                        for (int l = q; l < NV * NC; l += NQ) {
                            const int c = l / NV, j = l % NV;
                            codes[point_index(e, j, c)] = C::encode(S[j][c], inv);
                        }
                    } else {
                        for (int i = q; i < NV * NC; i += NQ) codes[index(t, e, i / NC, i % NC)] = C::encode(S[i / NC][i % NC], inv);
                    }
                }

                /// Tensor j of element e as the contraction reads it, in double (the diagonal).
                __device__ void decode(const TablesView &t, const ptrdiff_t e, const int j, double *const S) const {
                    const double ms = MD ? (double)scales[e] : 1.0;
                    for (int c = 0; c < NC; ++c) S[c] = 0;
                    if constexpr (FOLD) {
                        const int mask = t.mask[j];
                        for (int k = 0; k < t.nkeep[mask]; ++k) {
                            const int    c = t.keep[mask * NC + k];
                            const double v = (double)C::decode(codes[index(t, e, j, k)]);
                            S[c]           = MD ? t.xc[j * NC + c] * (double)means[mean_index(e, c)] + ms * v : v;
                        }
                    } else {
                        for (int c = 0; c < NC; ++c) {
                            const double v = (double)C::decode(codes[index(t, e, j, c)]);
                            S[c]           = MD ? t.x[j] * (double)means[mean_index(e, c)] + ms * v : v;
                        }
                    }
                }
            };

            /// The gradient point tensor j reads (`SIDE` 0) and writes (1): the moment's pair on
            /// TET10, the point itself otherwise.  Read here rather than through ElementTables,
            /// whose array readers device code may not call.
            template <class El, int j, int SIDE>
            __device__ constexpr int gradient_pair() {
                if constexpr (!El::moments) {
                    return j;
                } else if constexpr (SIDE == 0) {
                    constexpr int p = El::PA[j];
                    return p;
                } else {
                    constexpr int p = El::PB[j];
                    return p;
                }
            }

            /// A thread per element (TET4, TET10, HEX8): out += A h.
            template <class M, typename code_t, bool MD, bool FOLD, bool BLOCK, int MINB>
            __global__ void __launch_bounds__(THREADS, MINB)
                    apply_points_kernel(const ApplyArgs a, const PointCodes<M, code_t, MD, FOLD, false> st, const real_t *const RSTR h,
                                        real_t *const RSTR out) {
                using El               = typename M::element;
                using Contraction      = typename M::contraction;
                using Fold             = typename FoldOf<M>::type;
                using T                = ElementTables<El>;
                using C                = Code<code_t>;
                using tau              = typename C::tau;
                static constexpr int N = El::N, R = El::R, NV = El::NV, NC = M::NC;
                extern __shared__ real_t smem_points[];
                Traversal<El, BLOCK> tr(a, h, smem_points);
                const ptrdiff_t      e = tr.e;
                real_t               hh[3][N];
                tr.gather(h, hh);
                real_t o[3][N];
                if (tr.active) {
                    tau P[R][9], Y[R][9];
                    El::template stage_in<real_t, tau>(hh, P);
                    for (int r = 0; r < R; ++r)
                        for (int c = 0; c < 9; ++c) Y[r][c] = 0;
                    [[maybe_unused]] tau A[NC];
                    [[maybe_unused]] tau sc = 1;
                    if constexpr (MD) {
                        for (int c = 0; c < NC; ++c) A[c] = tau(__ldg(&st.means[st.mean_index(e, c)]));
                        sc = tau(__ldg(&st.scales[e]));
                    }
                    static_for<0, NV>([&](auto J) {
                        constexpr int j  = decltype(J)::value;
                        constexpr int pa = gradient_pair<El, j, 0>(), pb = gradient_pair<El, j, 1>();
                        tau           S[NC];
                        if constexpr (FOLD) {
                            // The numbers point j keeps, in their slots; the contraction skips the others.
                            constexpr int mask = Fold::mask(j);
                            static_for<0, Fold::NKEEP[mask]>([&](auto K) {
                                constexpr int k = decltype(K)::value, c = Fold::KEEP[mask][k], slot = Fold::OFF[j] + k;
                                const tau     v = C::decode(__ldg(&st.codes[st.code_index(e, slot)]));
                                if constexpr (MD) {
                                    constexpr double xc = Fold::XC[j][c];
                                    S[c]                = tau(xc) * A[c] + sc * v;
                                } else {
                                    S[c] = v;
                                }
                            });
                            Contraction::template contract_masked<mask, tau>(S, P[pa], Y[pb]);
                        } else {
                            static_for<0, NC>([&](auto CC) {
                                constexpr int c = decltype(CC)::value;
                                const tau     v = C::decode(__ldg(&st.codes[st.code_index(e, j * NC + c)]));
                                if constexpr (MD) {
                                    constexpr double xj = El::X[j];
                                    S[c]                = tau(xj) * A[c] + sc * v;
                                } else {
                                    S[c] = v;
                                }
                            });
                            Contraction::template contract<tau>(S, P[pa], Y[pb]);
                            if constexpr (pa != pb) Contraction::template contract<tau>(S, P[pb], Y[pa]);
                        }
                    });
                    El::template stage_out<tau, real_t>(Y, o);
                    if constexpr (C::scaled && !MD) {
                        const real_t s = (real_t)__ldg(&st.scales[e]);
                        for (int k = 0; k < 3; ++k)
                            for (int p = 0; p < N; ++p) o[k][p] *= s;
                    }
                }
                tr.scatter(o, out);
            }

            /// A thread per point (PROTEUS_HEX27), two elements per block: the point's gradient from
            /// the one-dimensional tables, decoding and the contraction, then the element's outputs by
            /// the transposed stages in three sweeps over its threads: a[qx][qy][sz], then
            /// b[qx][sy][sz], then the nodes.  The increment enters relative to the element's first
            /// node, as in the host's stages.
            template <class M, typename code_t, bool MD>
            __global__ void __launch_bounds__(128)
                    apply_points27_kernel(const ApplyArgs a, const PointCodes<M, code_t, MD, false, true> st, const TablesView t,
                                          const real_t *const RSTR h, real_t *const RSTR out) {
                using El                = typename M::element;
                using Contraction       = typename M::contraction;
                using C                 = Code<code_t>;
                using tau               = typename C::tau;
                static constexpr int N  = El::N, NQ = El::NQ, NC = M::NC, Q1 = El::Q1, S1 = El::S1;
                static constexpr int EPB = 128 / NQ;
                __shared__ real_t  hs[EPB][3][N];
                __shared__ idx_t   nodes[EPB][N];
                __shared__ tau     Ys[EPB][NQ][9];
                __shared__ tau     ax[EPB][3][Q1 * Q1 * S1], ay[EPB][3][Q1 * Q1 * S1], az[EPB][3][Q1 * Q1 * S1];
                __shared__ tau     bx[EPB][3][Q1 * S1 * S1], cc[EPB][3][Q1 * S1 * S1];
                __shared__ tau     B[Q1 * S1], D[Q1 * S1];
                const int          le = threadIdx.x / NQ, j = threadIdx.x % NQ;
                const ptrdiff_t    e  = (ptrdiff_t)blockIdx.x * EPB + le;
                const bool         active = e < a.nelements;
                if (threadIdx.x < Q1 * S1) {
                    B[threadIdx.x] = tau(t.b1[threadIdx.x]);
                    D[threadIdx.x] = tau(t.d1[threadIdx.x]);
                }
                if (active && j < N) {
                    const idx_t node = a.elements.node[j][e];
                    nodes[le][j]     = node;
                    #pragma unroll
                    for (int k = 0; k < 3; ++k) hs[le][k][j] = h[(ptrdiff_t)node * a.stride + k];
                }
                __syncthreads();
                if (active) {
                    const int jx = j % Q1, jy = (j / Q1) % Q1, jz = j / (Q1 * Q1);
                    // The point's line-table rows in registers, read once.  The fixed loops are unrolled
                    // explicitly: left to the compiler, f16 and int8 keep a 24-byte stack frame.
                    tau bx_[S1], dx_[S1], by_[S1], dy_[S1], bz_[S1], dz_[S1];
                    #pragma unroll
                    for (int u = 0; u < S1; ++u) {
                        bx_[u] = B[jx * S1 + u], dx_[u] = D[jx * S1 + u];
                        by_[u] = B[jy * S1 + u], dy_[u] = D[jy * S1 + u];
                        bz_[u] = B[jz * S1 + u], dz_[u] = D[jz * S1 + u];
                    }
                    tau P[9], Y[9];
                    #pragma unroll
                    for (int c = 0; c < 9; ++c) P[c] = 0, Y[c] = 0;
                    #pragma unroll
                    for (int sz = 0; sz < S1; ++sz)
                        #pragma unroll
                        for (int sy = 0; sy < S1; ++sy)
                            #pragma unroll
                            for (int sx = 0; sx < S1; ++sx) {
                                const int sn = sx + S1 * (sy + S1 * sz);
                                const tau w0 = dx_[sx] * by_[sy] * bz_[sz];
                                const tau w1 = bx_[sx] * dy_[sy] * bz_[sz];
                                const tau w2 = bx_[sx] * by_[sy] * dz_[sz];
                                #pragma unroll
                                for (int k = 0; k < 3; ++k) {
                                    const tau v = tau(hs[le][k][sn] - hs[le][k][0]);
                                    P[k * 3 + 0] += w0 * v;
                                    P[k * 3 + 1] += w1 * v;
                                    P[k * 3 + 2] += w2 * v;
                                }
                            }
                    tau S[NC], sc = 1;
                    if constexpr (MD) sc = tau(__ldg(&st.scales[e]));
                    const tau xj = tau(t.x[j]);
                    #pragma unroll
                    for (int c = 0; c < NC; ++c) {
                        const tau v = C::decode(__ldg(&st.codes[st.point_index(e, j, c)]));
                        if constexpr (MD) {
                            S[c] = xj * tau(__ldg(&st.means[st.mean_index(e, c)])) + sc * v;
                        } else {
                            S[c] = v;
                        }
                    }
                    Contraction::template contract<tau>(S, P, Y);
                    #pragma unroll
                    for (int c = 0; c < 9; ++c) Ys[le][j][c] = Y[c];
                }
                __syncthreads();
                if (active)
                    for (int idx = j; idx < 3 * Q1 * Q1 * S1; idx += NQ) {
                        const int i = idx / (Q1 * Q1 * S1), r = idx % (Q1 * Q1 * S1);
                        const int qx = r % Q1, qy = (r / Q1) % Q1, sz = r / (Q1 * Q1);
                        tau       sx_ = 0, sy_ = 0, sz_ = 0;
                        #pragma unroll
                        for (int qz = 0; qz < Q1; ++qz) {
                            const int q = qx + Q1 * (qy + Q1 * qz);
                            sx_ += B[qz * S1 + sz] * Ys[le][q][i * 3 + 0];
                            sy_ += B[qz * S1 + sz] * Ys[le][q][i * 3 + 1];
                            sz_ += D[qz * S1 + sz] * Ys[le][q][i * 3 + 2];
                        }
                        ax[le][i][r] = sx_, ay[le][i][r] = sy_, az[le][i][r] = sz_;
                    }
                __syncthreads();
                if (active)
                    for (int idx = j; idx < 3 * Q1 * S1 * S1; idx += NQ) {
                        const int i = idx / (Q1 * S1 * S1), r = idx % (Q1 * S1 * S1);
                        const int qx = r % Q1, sy = (r / Q1) % S1, sz = r / (Q1 * S1);
                        tau       b_ = 0, c_ = 0;
                        #pragma unroll
                        for (int qy = 0; qy < Q1; ++qy) {
                            const int s = qx + Q1 * (qy + Q1 * sz);
                            b_ += B[qy * S1 + sy] * ax[le][i][s];
                            c_ += D[qy * S1 + sy] * ay[le][i][s] + B[qy * S1 + sy] * az[le][i][s];
                        }
                        bx[le][i][r] = b_, cc[le][i][r] = c_;
                    }
                __syncthreads();
                if (!active) return;
                const real_t s = C::scaled && !MD ? (real_t)__ldg(&st.scales[e]) : real_t(1);
                for (int idx = j; idx < 3 * N; idx += NQ) {
                    const int i = idx / N, node = idx % N;
                    const int sx = node % S1, sy = (node / S1) % S1, sz = node / (S1 * S1);
                    tau       o  = 0;
                    #pragma unroll
                    for (int qx = 0; qx < Q1; ++qx) {
                        const int r = qx + Q1 * (sy + S1 * sz);
                        o += D[qx * S1 + sx] * bx[le][i][r] + B[qx * S1 + sx] * cc[le][i][r];
                    }
                    atomicAdd(&out[(ptrdiff_t)nodes[le][node] * a.stride + i], (real_t)o * s);
                }
            }

            // --------------------------------------------------------- the element matrices

            /// The element matrices on the device and their encoding (the host's encode_block): the
            /// off-diagonal node blocks in difference form, the element scale, the pair factors.
            /// Blocked [block][entry][lane] for a thread per element; [element][entry]
            /// (`ELEMENT_MAJOR`) for a warp per element.
            template <class M, typename code_t, bool PAIR, bool ELEMENT_MAJOR>
            struct MatrixCodes {
                using El                       = typename M::element;
                using C                        = Code<code_t>;
                using T                        = ElementTables<El>;
                using w_t                      = std::conditional_t<(sizeof(typename C::storage) <= 2), float, double>;
                static constexpr int  N        = El::N, NV = El::NV, NC = M::NC, NQ = El::NQ;
                static constexpr bool symmetric = M::symmetric;
                static constexpr bool pair     = PAIR;
                static constexpr int  NP       = N * (N - 1) / 2;
                static constexpr int  NB       = symmetric ? NP : 2 * NP;
                static constexpr int  NE       = NB * 9;
                static constexpr int  extra_doubles = (int)((NE * sizeof(w_t) + sizeof(double) - 1) / sizeof(double));
                // The tangents in the arithmetic type, converted once.
                static constexpr bool converts        = !std::is_same_v<w_t, double>;
                static constexpr int  scratch_doubles = converts ? (int)((NV * NC * sizeof(w_t) + sizeof(double) - 1) / sizeof(double)) : 0;
                static_assert(!PAIR || C::scaled, "pair factors need a scaled code");

                typename C::storage *codes{nullptr};
                float               *scales{nullptr};
                uint8_t             *factors{nullptr};

                __device__ __forceinline__ std::size_t code_index(const ptrdiff_t e, const int entry) const {
                    if constexpr (ELEMENT_MAJOR) {
                        return (size_t)e * NE + entry;
                    } else {
                        return ((size_t)(e / V) * NE + entry) * V + (size_t)(e % V);
                    }
                }
                __device__ __forceinline__ std::size_t factor_index(const ptrdiff_t e, const int k) const {
                    if constexpr (ELEMENT_MAJOR) {
                        return (size_t)e * NB + k;
                    } else {
                        return ((size_t)(e / V) * NB + k) * V + (size_t)(e % V);
                    }
                }

                /// K_pq[i][kk] = sum over the tensors and their gradient pairs (a, b) of
                /// sum_n DT[b][n][p] sum_m S[(i,n),(kk,m)] DT[a][m][q], in the host's order: a block
                /// per thread, the blocks in shared memory until the element's scale is known.
                __device__ void encode(const TablesView &t,
                                       const ptrdiff_t   e,
                                       const bool        active,
                                       const int         q,
                                       double (*const S)[NC],
                                       double *const     tops,
                                       double *const     extra,
                                       double *const     scratch) const {
                    using Contraction = typename M::contraction;
                    w_t *const Kd     = reinterpret_cast<w_t *>(extra);
                    double     mine   = 0;
                    // The tangents in the arithmetic type, as the host rounds them when it reads them.
                    const w_t(*W)[NC] = nullptr;
                    if constexpr (converts) {
                        w_t(*const Wc)[NC] = reinterpret_cast<w_t(*)[NC]>(scratch);
                        if (active)
                            for (int i = q; i < NV * NC; i += NQ) Wc[i / NC][i % NC] = w_t(S[i / NC][i % NC]);
                        __syncthreads();
                        W = Wc;
                    } else {
                        W = S;
                    }
                    if (active)
                        for (int k = q; k < NB; k += NQ) {
                            const int p = T::block_row(k), qq = T::block_col(k);
                            w_t       acc[9];
                            for (int y = 0; y < 9; ++y) acc[y] = 0;
                            for (int j = 0; j < NV; ++j) {
                                const int pa = t.pa[j], pb = t.pb[j];
                                for (int pass = 0; pass < (pa == pb ? 1 : 2); ++pass) {
                                    const int ga = pass == 0 ? pa : pb, gb = pass == 0 ? pb : pa;
                                    const w_t d0 = w_t(t.dt[(ga * 3 + 0) * N + qq]), d1 = w_t(t.dt[(ga * 3 + 1) * N + qq]),
                                              d2 = w_t(t.dt[(ga * 3 + 2) * N + qq]);
                                    const w_t e0 = w_t(t.dt[(gb * 3 + 0) * N + p]), e1 = w_t(t.dt[(gb * 3 + 1) * N + p]),
                                              e2 = w_t(t.dt[(gb * 3 + 2) * N + p]);
                                    w_t T3[27];
                                    static_for<0, 27>([&](auto R) {
                                        constexpr int r = decltype(R)::value, in = r / 3, kk = r % 3;
                                        constexpr int s0 = Contraction::SLOT[in][kk * 3 + 0], s1 = Contraction::SLOT[in][kk * 3 + 1],
                                                      s2 = Contraction::SLOT[in][kk * 3 + 2];
                                        T3[r] = d0 * W[j][s0] + d1 * W[j][s1] + d2 * W[j][s2];
                                    });
                                    for (int i = 0; i < 3; ++i)
                                        for (int kk = 0; kk < 3; ++kk)
                                            acc[i * 3 + kk] += e0 * T3[(i * 3 + 0) * 3 + kk] + e1 * T3[(i * 3 + 1) * 3 + kk] +
                                                               e2 * T3[(i * 3 + 2) * 3 + kk];
                                }
                            }
                            for (int y = 0; y < 9; ++y) {
                                Kd[k * 9 + y] = acc[y];
                                mine          = fmax(mine, fabs((double)acc[y]));
                            }
                        }
                    const double top = element_max<NQ>(tops, q, mine);
                    double       inv = 1;
                    if constexpr (C::scaled) {
                        const double s = C::scale_for(top);
                        if (active && q == 0) scales[e] = (float)s;
                        inv = 1.0 / (double)(float)s;
                    }
                    if (!active) return;
                    for (int k = q; k < NB; k += NQ) {
                        double pinv = inv;
                        if constexpr (PAIR) {
                            double pt = 0;
                            for (int y = 0; y < 9; ++y) pt = fmax(pt, fabs((double)Kd[k * 9 + y]));
                            const uint8_t b         = PairFactor::encode(top > 0 ? pt / top : 0.0);
                            factors[factor_index(e, k)] = b;
                            pinv                    = inv / (double)PairFactor::value(b);
                        }
                        for (int y = 0; y < 9; ++y) codes[code_index(e, k * 9 + y)] = C::encode((double)Kd[k * 9 + y], pinv);
                    }
                }
            };

            /// A thread per element (TET4, TET10, HEX8): y_p += K_pq (h_q - h_p), and for a symmetric
            /// tangent y_q -= K_pq^T (h_q - h_p).
            template <class M, typename code_t, bool PAIR, bool BLOCK, int MINB>
            __global__ void __launch_bounds__(THREADS, MINB)
                    apply_matrices_kernel(const ApplyArgs a, const MatrixCodes<M, code_t, PAIR, false> st, const real_t *const RSTR h,
                                          real_t *const RSTR out) {
                using El               = typename M::element;
                using Codes            = MatrixCodes<M, code_t, PAIR, false>;
                using T                = ElementTables<El>;
                using C                = Code<code_t>;
                using tau              = typename C::tau;
                static constexpr int N = El::N;
                extern __shared__ real_t smem_matrices[];
                Traversal<El, BLOCK> tr(a, h, smem_matrices);
                const ptrdiff_t      e = tr.e;
                real_t               hh[3][N];
                tr.gather(h, hh);
                real_t o[3][N];
                if (tr.active) {
                    tau Y[3][N];
                    for (int k = 0; k < 3; ++k)
                        for (int p = 0; p < N; ++p) Y[k][p] = 0;
                    const typename C::storage *const RSTR c0 = st.codes + st.code_index(e, 0);
                    [[maybe_unused]] const uint8_t *const RSTR f0 = PAIR ? st.factors + st.factor_index(e, 0) : nullptr;
                    static_for<0, Codes::NB>([&](auto KK) {
                        constexpr int k = decltype(KK)::value, p = T::block_row(k), q = T::block_col(k);
                        // Differences in the vector's precision, so a translation cancels exactly.
                        tau d0 = tau(hh[0][q] - hh[0][p]), d1 = tau(hh[1][q] - hh[1][p]), d2 = tau(hh[2][q] - hh[2][p]);
                        if constexpr (PAIR) {
                            const tau r = tau(PairFactor::value(__ldg(&f0[k * V])));
                            d0 *= r, d1 *= r, d2 *= r;
                        }
                        tau K[9];
                        for (int y = 0; y < 9; ++y) K[y] = C::decode(__ldg(&c0[(k * 9 + y) * V]));
                        Y[0][p] += K[0] * d0 + K[1] * d1 + K[2] * d2;
                        Y[1][p] += K[3] * d0 + K[4] * d1 + K[5] * d2;
                        Y[2][p] += K[6] * d0 + K[7] * d1 + K[8] * d2;
                        if constexpr (Codes::symmetric) {
                            Y[0][q] -= K[0] * d0 + K[3] * d1 + K[6] * d2;
                            Y[1][q] -= K[1] * d0 + K[4] * d1 + K[7] * d2;
                            Y[2][q] -= K[2] * d0 + K[5] * d1 + K[8] * d2;
                        }
                    });
                    const real_t s = C::scaled ? (real_t)__ldg(&st.scales[e]) : real_t(1);
                    for (int k = 0; k < 3; ++k)
                        for (int p = 0; p < N; ++p) o[k][p] = (real_t)Y[k][p] * s;
                }
                tr.scatter(o, out);
            }

            /// A warp per element (PROTEUS_HEX27), lane p < 27 owning node p: y_p = sum_{q != p} M (h_q -
            /// h_p), M the stored block of row p, or (symmetric) the transpose of row q's.  No atomics
            /// within the element.
            template <class M, typename code_t, bool PAIR>
            __global__ void __launch_bounds__(128)
                    apply_matrices27_kernel(const ApplyArgs a, const MatrixCodes<M, code_t, PAIR, true> st, const TablesView t,
                                            const real_t *const RSTR h, real_t *const RSTR out) {
                using El                 = typename M::element;
                using Codes              = MatrixCodes<M, code_t, PAIR, true>;
                using C                  = Code<code_t>;
                using tau                = typename C::tau;
                static constexpr int N   = El::N, NP = Codes::NP;
                static constexpr int WARPS = 4;
                __shared__ real_t hs[WARPS][3][N];
                const int         w = threadIdx.x / 32, p = threadIdx.x % 32;
                const ptrdiff_t   e = (ptrdiff_t)blockIdx.x * WARPS + w;
                if (e >= a.nelements) return;
                idx_t node = 0;
                if (p < N) {
                    node = a.elements.node[p][e];
                    for (int k = 0; k < 3; ++k) hs[w][k][p] = h[(ptrdiff_t)node * a.stride + k];
                }
                __syncwarp();
                if (p >= N) return;
                auto pair = [](const int r, const int c) { return r * (N - 1) - r * (r - 1) / 2 + (c - r - 1); };  // r < c
                tau  y0 = 0, y1 = 0, y2 = 0;
                for (int q = 0; q < N; ++q) {
                    if (q == p) continue;
                    // Row p's block: K_pq stored at the pair for p < q; above it, K_pq (unsymmetric)
                    // at NP + pair, or the transpose of K_qp (symmetric).
                    const bool upper = p < q;
                    const int  k     = upper ? pair(p, q) : (Codes::symmetric ? pair(q, p) : NP + pair(q, p));
                    tau        d0 = tau(hs[w][0][q] - hs[w][0][p]), d1 = tau(hs[w][1][q] - hs[w][1][p]), d2 = tau(hs[w][2][q] - hs[w][2][p]);
                    if constexpr (PAIR) {
                        const tau r = tau(PairFactor::value(__ldg(&st.factors[st.factor_index(e, k)])));
                        d0 *= r, d1 *= r, d2 *= r;
                    }
                    tau K[9];
                    for (int y = 0; y < 9; ++y) K[y] = C::decode(__ldg(&st.codes[st.code_index(e, k * 9 + y)]));
                    if (upper || !Codes::symmetric) {
                        y0 += K[0] * d0 + K[1] * d1 + K[2] * d2;
                        y1 += K[3] * d0 + K[4] * d1 + K[5] * d2;
                        y2 += K[6] * d0 + K[7] * d1 + K[8] * d2;
                    } else {
                        y0 += K[0] * d0 + K[3] * d1 + K[6] * d2;
                        y1 += K[1] * d0 + K[4] * d1 + K[7] * d2;
                        y2 += K[2] * d0 + K[5] * d1 + K[8] * d2;
                    }
                }
                const real_t s = C::scaled ? (real_t)__ldg(&st.scales[e]) : real_t(1);
                atomicAdd(&out[(ptrdiff_t)node * a.stride + 0], (real_t)y0 * s);
                atomicAdd(&out[(ptrdiff_t)node * a.stride + 1], (real_t)y1 * s);
                atomicAdd(&out[(ptrdiff_t)node * a.stride + 2], (real_t)y2 * s);
            }

            // ---------------------------------------------------------------- the diagonal

            /// The point store's element diagonal, sum_n sum_m dphi_p/dxi_n S[(i,n),(i,m)] dphi_p/dxi_m
            /// over the tensors and their gradient points, a thread per element, into `diag`.
            template <class M, class Codes>
            __global__ void diagonal_points_kernel(const ApplyArgs a, const Codes st, const TablesView t, real_t *const RSTR diag) {
                using El               = typename M::element;
                static constexpr int N = El::N, NV = El::NV, NC = M::NC;
                const ptrdiff_t      e = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x;
                if (e >= a.nelements) return;
                double d[3][N];
                for (int i = 0; i < 3; ++i)
                    for (int p = 0; p < N; ++p) d[i][p] = 0;
                for (int j = 0; j < NV; ++j) {
                    double S[NC];
                    st.decode(t, e, j, S);
                    const int pa = t.pa[j], pb = t.pb[j];
                    for (int pass = 0; pass < (pa == pb ? 1 : 2); ++pass) {
                        const int ga = pass == 0 ? pa : pb, gb = pass == 0 ? pb : pa;
                        for (int i = 0; i < 3; ++i)
                            for (int p = 0; p < N; ++p) {
                                double acc = 0;
                                for (int n = 0; n < 3; ++n)
                                    for (int m = 0; m < 3; ++m)
                                        acc += t.dt[(gb * 3 + n) * N + p] * S[t.slot[(i * 3 + n) * 9 + i * 3 + m]] * t.dt[(ga * 3 + m) * N + p];
                                d[i][p] += acc;
                            }
                    }
                }
                const double s = Codes::scaled && !Codes::has_mean ? (double)st.scales[e] : 1.0;
                for (int p = 0; p < N; ++p) {
                    const ptrdiff_t node = a.elements.node[p][e];
                    for (int i = 0; i < 3; ++i) atomicAdd(&diag[node * a.stride + i], (real_t)(s * d[i][p]));
                }
            }

            /// The element matrices' diagonal: a diagonal block is minus the stored blocks of its
            /// row (and, symmetric, of its column).
            template <class M, class Codes>
            __global__ void diagonal_matrices_kernel(const ApplyArgs a, const Codes st, const TablesView t, real_t *const RSTR diag) {
                using El               = typename M::element;
                using T                = ElementTables<El>;
                using C                = typename Codes::C;
                static constexpr int N = El::N;
                const ptrdiff_t      e = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x;
                if (e >= a.nelements) return;
                double d[3][N];
                for (int i = 0; i < 3; ++i)
                    for (int p = 0; p < N; ++p) d[i][p] = 0;
                const double s = C::scaled ? (double)st.scales[e] : 1.0;
                for (int k = 0; k < Codes::NB; ++k) {
                    const int    p = T::block_row(k), q = T::block_col(k);
                    const double r = Codes::pair ? (double)PairFactor::value(st.factors[st.factor_index(e, k)]) : 1.0;
                    for (int i = 0; i < 3; ++i) {
                        const double v = s * r * (double)C::decode(st.codes[st.code_index(e, k * 9 + i * 3 + i)]);
                        d[i][p] -= v;
                        if (Codes::symmetric) d[i][q] -= v;
                    }
                }
                for (int p = 0; p < N; ++p) {
                    const ptrdiff_t node = a.elements.node[p][e];
                    for (int i = 0; i < 3; ++i) atomicAdd(&diag[node * a.stride + i], (real_t)d[i][p]);
                }
            }

            // ------------------------------------------------------------------ the stores

            /// The point store on the device.
            template <class M, typename code_t, bool MD, bool FOLD>
            class PointStore final : public InexactStore {
            public:
                using El                          = typename M::element;
                static constexpr bool POINT_MAJOR = El::N == 27;
                using Codes                       = PointCodes<M, code_t, MD, FOLD, POINT_MAJOR>;
                static constexpr int  NV          = El::NV, NC = M::NC;
                static constexpr Shape SHAPE      = shape<El, false, code_t>();

                PointStore(const ptrdiff_t nelements, std::string name, void *const stream)
                    : nelements_(nelements), name_(std::move(name)), stream_((cudaStream_t)stream) {
                    const std::size_t n = POINT_MAJOR ? (std::size_t)nelements : (std::size_t)((nelements + V - 1) / V) * V;
                    codes_              = DeviceArray<typename Codes::C::storage>(n * Codes::NSE);
                    if (Codes::scaled) scales_ = DeviceArray<float>(n);
                    if (MD) means_ = DeviceArray<float>(n * NC);
                    codes_view_.codes  = codes_.data;
                    codes_view_.scales = scales_.data;
                    codes_view_.means  = means_.data;
                }

                void update(const SetupInput &in) override {
                    if (in.nelements != nelements_) {
                        SFEM_ERROR("inexact (device): the store holds %ld elements, the set-up has %ld\n", (long)nelements_, (long)in.nelements);
                    }
                    const Connectivity &elements = connectivity_.bind(in.elements);
                    if (SHAPE.block && !POINT_MAJOR && !map_) map_ = BlockMap::build<El>(nelements_, elements);
                    launch_setup<M>(setup_args<M>(in, elements), tables_.view(), codes_view_, stream_);
                }

                void apply(const ptrdiff_t           nelements,
                           const idx_t *const *const elements,
                           const PackedView *const,
                           const ptrdiff_t           stride,
                           const real_t *const       h,
                           real_t *const             out) const override {
                    ApplyArgs a;
                    a.nelements = nelements;
                    a.elements  = connectivity_(elements);
                    a.stride    = stride;
                    if (nelements == 0) return;
                    if constexpr (POINT_MAJOR) {
                        constexpr int epb = 128 / El::NQ;
                        apply_points27_kernel<M, code_t, MD><<<grid(nelements, epb), 128, 0, stream_>>>(a, codes_view_, tables_.view(), h, out);
                    } else if constexpr (SHAPE.block) {
                        map_->bind(a);
                        apply_points_kernel<M, code_t, MD, FOLD, true, SHAPE.minb>
                                <<<grid(nelements, THREADS), THREADS, map_->shared_bytes(), stream_>>>(a, codes_view_, h, out);
                    } else {
                        apply_points_kernel<M, code_t, MD, FOLD, false, SHAPE.minb>
                                <<<grid(nelements, THREADS), THREADS, 0, stream_>>>(a, codes_view_, h, out);
                    }
                    check(cudaGetLastError(), "point store apply");
                    SFEM_DEBUG_SYNCHRONIZE(stream_);
                }

                void diagonal(const ptrdiff_t nelements, const idx_t *const *const elements, const ptrdiff_t stride, real_t *const diag) const override {
                    ApplyArgs a;
                    a.nelements = nelements;
                    a.elements  = connectivity_(elements);
                    a.stride    = stride;
                    if (nelements == 0) return;
                    diagonal_points_kernel<M, Codes><<<grid(nelements, THREADS), THREADS, 0, stream_>>>(a, codes_view_, tables_.view(), diag);
                    check(cudaGetLastError(), "point store diagonal");
                    SFEM_DEBUG_SYNCHRONIZE(stream_);
                }

                std::size_t bytes() const override { return codes_.bytes() + scales_.bytes() + means_.bytes(); }
                std::string describe() const override { return name_; }

            private:
                ptrdiff_t                                  nelements_;
                std::string                                name_;
                cudaStream_t                               stream_;
                Tables<M, FOLD>                            tables_;
                DeviceArray<typename Codes::C::storage>    codes_;
                DeviceArray<float>                         scales_, means_;
                Codes                                      codes_view_;
                std::unique_ptr<BlockMap>                  map_;
                StagedConnectivity<El>                     connectivity_;
            };

            /// The element matrices on the device.
            template <class M, typename code_t, bool PAIR>
            class MatrixStore final : public InexactStore {
            public:
                using El                            = typename M::element;
                static constexpr bool ELEMENT_MAJOR = El::N == 27;
                using Codes                         = MatrixCodes<M, code_t, PAIR, ELEMENT_MAJOR>;
                static constexpr Shape SHAPE        = shape<El, true, code_t>();

                MatrixStore(const ptrdiff_t nelements, std::string name, void *const stream)
                    : nelements_(nelements), name_(std::move(name)), stream_((cudaStream_t)stream) {
                    const std::size_t n = ELEMENT_MAJOR ? (std::size_t)nelements : (std::size_t)((nelements + V - 1) / V) * V;
                    codes_              = DeviceArray<typename Codes::C::storage>(n * Codes::NE);
                    if (Codes::C::scaled) scales_ = DeviceArray<float>(n);
                    if (PAIR) factors_ = DeviceArray<uint8_t>(n * Codes::NB);
                    view_.codes   = codes_.data;
                    view_.scales  = scales_.data;
                    view_.factors = factors_.data;
                }

                void update(const SetupInput &in) override {
                    if (in.nelements != nelements_) {
                        SFEM_ERROR("inexact (device): the store holds %ld elements, the set-up has %ld\n", (long)nelements_, (long)in.nelements);
                    }
                    const Connectivity &elements = connectivity_.bind(in.elements);
                    if (SHAPE.block && !ELEMENT_MAJOR && !map_) map_ = BlockMap::build<El>(nelements_, elements);
                    launch_setup<M>(setup_args<M>(in, elements), tables_.view(), view_, stream_);
                }

                void apply(const ptrdiff_t           nelements,
                           const idx_t *const *const elements,
                           const PackedView *const,
                           const ptrdiff_t           stride,
                           const real_t *const       h,
                           real_t *const             out) const override {
                    ApplyArgs a;
                    a.nelements = nelements;
                    a.elements  = connectivity_(elements);
                    a.stride    = stride;
                    if (nelements == 0) return;
                    if constexpr (ELEMENT_MAJOR) {
                        apply_matrices27_kernel<M, code_t, PAIR><<<grid(nelements, 4), 128, 0, stream_>>>(a, view_, tables_.view(), h, out);
                    } else if constexpr (SHAPE.block) {
                        map_->bind(a);
                        apply_matrices_kernel<M, code_t, PAIR, true, SHAPE.minb>
                                <<<grid(nelements, THREADS), THREADS, map_->shared_bytes(), stream_>>>(a, view_, h, out);
                    } else {
                        apply_matrices_kernel<M, code_t, PAIR, false, SHAPE.minb>
                                <<<grid(nelements, THREADS), THREADS, 0, stream_>>>(a, view_, h, out);
                    }
                    check(cudaGetLastError(), "element matrices apply");
                    SFEM_DEBUG_SYNCHRONIZE(stream_);
                }

                void diagonal(const ptrdiff_t nelements, const idx_t *const *const elements, const ptrdiff_t stride, real_t *const diag) const override {
                    ApplyArgs a;
                    a.nelements = nelements;
                    a.elements  = connectivity_(elements);
                    a.stride    = stride;
                    if (nelements == 0) return;
                    diagonal_matrices_kernel<M, Codes><<<grid(nelements, THREADS), THREADS, 0, stream_>>>(a, view_, tables_.view(), diag);
                    check(cudaGetLastError(), "element matrices diagonal");
                    SFEM_DEBUG_SYNCHRONIZE(stream_);
                }

                std::size_t bytes() const override { return codes_.bytes() + scales_.bytes() + factors_.bytes(); }
                std::string describe() const override { return name_; }

            private:
                ptrdiff_t                               nelements_;
                std::string                             name_;
                cudaStream_t                            stream_;
                Tables<M, false>                        tables_;
                DeviceArray<typename Codes::C::storage> codes_;
                DeviceArray<float>                      scales_;
                DeviceArray<uint8_t>                    factors_;
                Codes                                   view_;
                std::unique_ptr<BlockMap>               map_;
                StagedConnectivity<El>                  connectivity_;
            };

            template <class M, bool FOLD>
            std::unique_ptr<InexactStore> make_points(const ptrdiff_t n, const InexactOptions &options, void *const stream) {
                using E           = InexactOptions::Encoding;
                const std::string el = M::element::name;
                switch (options.encoding) {
                    case E::F64:
                        return std::make_unique<PointStore<M, double, false, FOLD>>(n, "points f64 " + el + " (device)", stream);
                    case E::F32:
                        return std::make_unique<PointStore<M, float, false, FOLD>>(n, "points f32 " + el + " (device)", stream);
                    case E::F16:
                        return std::make_unique<PointStore<M, __half, false, FOLD>>(n, "points f16 " + el + " (device)", stream);
                    case E::I8_MEAN_DEVIATION:
                        return std::make_unique<PointStore<M, int8_t, true, FOLD>>(n, "points i8md " + el + " (device)", stream);
                    default:
                        break;
                }
                SFEM_ERROR("inexact (device): the point store has no encoding %s\n", options.describe().c_str());
                return nullptr;
            }

            template <class M>
            std::unique_ptr<InexactStore> make_matrices(const ptrdiff_t n, const InexactOptions &options, void *const stream) {
                using E           = InexactOptions::Encoding;
                const std::string el = M::element::name;
                switch (options.encoding) {
                    case E::F64:
                        return std::make_unique<MatrixStore<M, double, false>>(n, "element matrices f64 " + el + " (device)", stream);
                    case E::F32:
                        return std::make_unique<MatrixStore<M, float, false>>(n, "element matrices f32 " + el + " (device)", stream);
                    case E::F16:
                        return std::make_unique<MatrixStore<M, __half, false>>(n, "element matrices f16 " + el + " (device)", stream);
                    case E::I8_PAIR_SCALE:
                        return std::make_unique<MatrixStore<M, int8_t, true>>(n, "element matrices i8pair " + el + " (device)", stream);
                    default:
                        break;
                }
                SFEM_ERROR("inexact (device): the element-matrix store has no encoding %s\n", options.describe().c_str());
                return nullptr;
            }

            /// The device store the options ask for, for a generated device unit `M`.  The device
            /// chooses its traversal (the measured shape); HEX8 folds unless asked not to, HEX27 does
            /// not fold.
            template <class M>
            std::unique_ptr<InexactStore> create_store(const ptrdiff_t nelements, const InexactOptions &options, void *const stream) {
                using El = typename M::element;
                options.validate();
                if (options.store == InexactOptions::Store::BSR) {
                    SFEM_ERROR("inexact: the bsr store is the operator's, not a domain's (OpState::update_bsr)\n");
                }
                if (options.traversal != InexactOptions::Traversal::AUTO) {
                    SFEM_ERROR("inexact: traversal %s asked for on the device, which chooses its own\n", options.describe().c_str());
                }
                if (options.store == InexactOptions::Store::ELEMENT_MATRIX) return make_matrices<M>(nelements, options, stream);
                if constexpr (El::foldable && El::N != 27) {
                    if (options.fold != InexactOptions::Switch::OFF) return make_points<M, true>(nelements, options, stream);
                } else {
                    if (options.fold == InexactOptions::Switch::ON) {
                        SFEM_ERROR("inexact: fold asked for on %s, which the device does not fold\n", El::name);
                    }
                }
                return make_points<M, false>(nelements, options, stream);
            }

        }  // namespace cuda
    }  // namespace inexact
}  // namespace sfem
