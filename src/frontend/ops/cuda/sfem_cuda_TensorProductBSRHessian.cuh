#ifndef SFEM_CUDA_TENSOR_PRODUCT_BSR_HESSIAN_CUH
#define SFEM_CUDA_TENSOR_PRODUCT_BSR_HESSIAN_CUH

// The Hessian of a vector tensor-product element (NS1^3 nodes, NQ1^3 points, 3 components) added to a
// BSR matrix of 3x3 blocks, a block of threads per element. The element matrix is never held: it is
// built a trial node at a time and added as it is made.
//
// With w the point weight, G_s(q) the reference gradient of shape s at point q (a product of the 1D
// tables b, g) and A_q the material's point tangent pulled back to the reference cell (`M::point`,
// generated per material: symmetric, 45 slots, `sfem::cooperative::packed`), the column of trial dof
// (cj, j) is
//
//   K[(ci, i), (cj, j)] = sum_q G_i(q)^T F_q(ci),   F_q(ci)_a = w_q sum_b A_q[(ci, a), (cj, b)] G_j(q)_b,
//
// and the sum over q is factorised along x, y, z (`contract_pair`): 3 NS1 NQ1 (NQ1^2 + 2 NS1 NQ1 + NS1^2)
// multiply-adds per column against NS NQ 9 directly.
//
// Thread q of the block owns point q: it differentiates there, keeps w_q A_q in registers for the
// element's 3 NS columns and writes F_q into shared memory; groups of 32 threads contract. HEX64 in
// double: 46 KB of shared memory and 250 registers a thread, where a thread holding the element
// matrix needs 295 KB of stack, beyond an AMD GPU's 128 KB and more than an H100 reserves for a launch.
// On an H100 (8,000 elements, SVK): 9.3 ms, 0.86 M elements/s.

#include <cstddef>

#include "sfem_base.hpp"

namespace sfem {
    namespace tensor_product {
        namespace cuda {

            /// The slot of entry (r, c) of a symmetric 9x9 tangent packed by rows of its upper
            /// triangle (`inexact_contract_d3<true>::SLOT`).
            __host__ __device__ constexpr int symmetric_slot(const int r, const int c) {
                return r <= c ? r * 9 - r * (r - 1) / 2 + c - r : c * 9 - c * (c - 1) / 2 + r - c;
            }

            template <typename s_t, int N>
            struct Parameters {
                s_t value[N];
            };

            /// w_q A_q at point q: the geometry and the displacement gradient there from the element's
            /// coordinates x and displacement u (shared memory), then the material's tangent.
            template <class M, class Quad, typename s_t, int NQ1, int NS1>
            __device__ __forceinline__ void point_tangent(const s_t *const                         b1,
                                                          const s_t *const                         g1,
                                                          const s_t (*const x)[NS1 * NS1 * NS1],
                                                          const s_t (*const u)[NS1 * NS1 * NS1],
                                                          const int                                q,
                                                          const Parameters<s_t, M::n_parameters> &parameters,
                                                          s_t (&A)[45]) {
                static constexpr int NS = NS1 * NS1 * NS1;
                const s_t(*const b)[NS1] = reinterpret_cast<const s_t(*)[NS1]>(b1);
                const s_t(*const g)[NS1] = reinterpret_cast<const s_t(*)[NS1]>(g1);
                const int qx = q % NQ1, qy = (q / NQ1) % NQ1, qz = q / (NQ1 * NQ1);
                s_t       J[9] = {0}, gu[9] = {0};
                for (int s = 0; s < NS; ++s) {
                    const int sx = s % NS1, sy = (s / NS1) % NS1, sz = s / (NS1 * NS1);
                    const s_t G[3] = {g[qx][sx] * b[qy][sy] * b[qz][sz], b[qx][sx] * g[qy][sy] * b[qz][sz], b[qx][sx] * b[qy][sy] * g[qz][sz]};
#pragma unroll
                    for (int d = 0; d < 3; ++d) {
#pragma unroll
                        for (int r = 0; r < 3; ++r) {
                            J[d * 3 + r] += x[d][s] * G[r];
                            gu[d * 3 + r] += u[d][s] * G[r];
                        }
                    }
                }
                s_t adj[9];
                adj[0]        = J[4] * J[8] - J[5] * J[7];
                adj[1]        = J[2] * J[7] - J[1] * J[8];
                adj[2]        = J[1] * J[5] - J[2] * J[4];
                adj[3]        = J[5] * J[6] - J[3] * J[8];
                adj[4]        = J[0] * J[8] - J[2] * J[6];
                adj[5]        = J[2] * J[3] - J[0] * J[5];
                adj[6]        = J[3] * J[7] - J[4] * J[6];
                adj[7]        = J[1] * J[6] - J[0] * J[7];
                adj[8]        = J[0] * J[4] - J[1] * J[3];
                const s_t det = J[0] * adj[0] + J[1] * adj[3] + J[2] * adj[6];
                const s_t w = Quad::q_weight_1d()[qx] * Quad::q_weight_1d()[qy] * Quad::q_weight_1d()[qz];
                M::template point<s_t>(parameters.value, gu, adj, det, w, A);
            }

            /// The element's NS nodes in ascending order (`sorted`) with the local index of each (`local`):
            /// thread s writes node s at its rank (an element's nodes are distinct).
            template <int NS>
            __device__ __forceinline__ void sort_nodes(const idx_t *const node, idx_t *const sorted, int *const local) {
                for (int s = threadIdx.x; s < NS; s += blockDim.x) {
                    int rank = 0;
                    for (int t = 0; t < NS; ++t) rank += node[t] < node[s];
                    sorted[rank] = node[s];
                    local[rank]  = s;
                }
            }

            /// The blocks of row `row` at the element's NS nodes: block[i] the entry of the column of
            /// node i; whether the row has any. The block's threads read the row once, coalesced, and
            /// look each column up among the element's sorted nodes (shared memory): one trip to global
            /// memory, where a search per node took one per halving.
            template <int NS>
            __device__ __forceinline__ bool row_blocks(const idx_t                        row,
                                                       const count_t *const SFEM_RESTRICT rowptr,
                                                       const idx_t *const SFEM_RESTRICT   colidx,
                                                       const idx_t *const                 sorted,
                                                       const int *const                   local,
                                                       count_t *const                     block) {
                const count_t begin = rowptr[row], end = rowptr[row + 1];
                for (count_t k = begin + threadIdx.x; k < end; k += blockDim.x) {
                    const idx_t col = colidx[k];
                    int         lo = 0, hi = NS;
                    while (lo < hi) {
                        const int mid = (lo + hi) / 2;
                        if (sorted[mid] < col)
                            lo = mid + 1;
                        else
                            hi = mid;
                    }
                    if (lo < NS && sorted[lo] == col) block[local[lo]] = k;
                }
                return begin < end;
            }

            /// Threads that contract one (trial component, test component) pair together.
            static constexpr int GROUP  = 32;
            static constexpr int GROUPS = 4;

            template <typename s_t, int NQ1, int NS1>
            struct BSRHessianShared {
                static constexpr int NQ = NQ1 * NQ1 * NQ1;
                static constexpr int NS = NS1 * NS1 * NS1;

                s_t     b[NQ1 * NS1], g[NQ1 * NS1];
                s_t     x[3][NS], u[3][NS];
                s_t     flux[3][9][NQ];                    // F_q for trial component cj: [cj][ci * 3 + a][q]
                s_t     pqr[GROUPS][3][NS1][NQ1 * NQ1];    // a group's pair after x: [term][ix][qy + NQ1 qz]
                s_t     uv[GROUPS][2][NS1 * NS1][NQ1];     // and after y: [term][ix + NS1 iy][qz]
                count_t block[NS];
                idx_t   node[NS], sorted[NS];
                int     local[NS];
                int     has_row;
            };

            /// out(i) = sum_q G_i(q) . F_q(cj)[ci] by sum factorisation, the 32 threads of one group,
            /// added to the BSR blocks of row j at entry (cj, ci). Each thread reads a line of NQ1 values
            /// once and forms the NS1 results that use it, the 1D tables read at constant indices
            /// (unrolled): shared memory is read once per NS1 multiply-adds, where a result per thread
            /// read two values per multiply-add and bound the kernel to an eighth of the H100's FP64 rate.
            template <class Ref, typename s_t, typename v_t, int NQ1, int NS1>
            __device__ __forceinline__ void contract_pair(BSRHessianShared<s_t, NQ1, NS1> &sh,
                                                          const int                             group,
                                                          const int                             lane,
                                                          const int                             cj,
                                                          const int                             ci,
                                                          v_t *const SFEM_RESTRICT              values) {
                const s_t *const b = Ref::shape_1d();  // b[q * NS1 + s]
                const s_t *const g = Ref::grad_1d();
                s_t(*const pqr)[NS1][NQ1 * NQ1] = sh.pqr[group];
                s_t(*const uv)[NS1 * NS1][NQ1]  = sh.uv[group];

                // Along x, a line (term, qy, qz) of F at a time: P = g^T F0, Q = b^T F1, R = b^T F2.
                for (int t = lane; t < 3 * NQ1 * NQ1; t += GROUP) {
                    const int        yz   = t % (NQ1 * NQ1);
                    const int        term = t / (NQ1 * NQ1);
                    const s_t *const f    = sh.flux[cj][ci * 3 + term] + yz * NQ1;
                    s_t              line[NQ1];
#pragma unroll
                    for (int qx = 0; qx < NQ1; ++qx) line[qx] = f[qx];
                    if (term == 0) {
#pragma unroll
                        for (int ix = 0; ix < NS1; ++ix) {
                            s_t acc = 0;
#pragma unroll
                            for (int qx = 0; qx < NQ1; ++qx) acc += g[qx * NS1 + ix] * line[qx];
                            pqr[0][ix][yz] = acc;
                        }
                    } else {
#pragma unroll
                        for (int ix = 0; ix < NS1; ++ix) {
                            s_t acc = 0;
#pragma unroll
                            for (int qx = 0; qx < NQ1; ++qx) acc += b[qx * NS1 + ix] * line[qx];
                            pqr[term][ix][yz] = acc;
                        }
                    }
                }
                __syncwarp();

                // Along y, a line (term, ix, qz): U = b^T P + g^T Q, V = b^T R.
                for (int t = lane; t < 2 * NS1 * NQ1; t += GROUP) {
                    const int qz   = t % NQ1;
                    const int ix   = (t / NQ1) % NS1;
                    const int term = t / (NQ1 * NS1);
                    if (term == 0) {
                        s_t p[NQ1], r[NQ1];
#pragma unroll
                        for (int qy = 0; qy < NQ1; ++qy) {
                            p[qy] = pqr[0][ix][qy + NQ1 * qz];
                            r[qy] = pqr[1][ix][qy + NQ1 * qz];
                        }
#pragma unroll
                        for (int iy = 0; iy < NS1; ++iy) {
                            s_t acc = 0;
#pragma unroll
                            for (int qy = 0; qy < NQ1; ++qy) acc += b[qy * NS1 + iy] * p[qy] + g[qy * NS1 + iy] * r[qy];
                            uv[0][ix + NS1 * iy][qz] = acc;
                        }
                    } else {
                        s_t p[NQ1];
#pragma unroll
                        for (int qy = 0; qy < NQ1; ++qy) p[qy] = pqr[2][ix][qy + NQ1 * qz];
#pragma unroll
                        for (int iy = 0; iy < NS1; ++iy) {
                            s_t acc = 0;
#pragma unroll
                            for (int qy = 0; qy < NQ1; ++qy) acc += b[qy * NS1 + iy] * p[qy];
                            uv[1][ix + NS1 * iy][qz] = acc;
                        }
                    }
                }
                __syncwarp();

                // Along z, a line (ix, iy), and into the matrix.
                for (int xy = lane; xy < NS1 * NS1; xy += GROUP) {
                    s_t u0[NQ1], u1[NQ1];
#pragma unroll
                    for (int qz = 0; qz < NQ1; ++qz) {
                        u0[qz] = uv[0][xy][qz];
                        u1[qz] = uv[1][xy][qz];
                    }
#pragma unroll
                    for (int iz = 0; iz < NS1; ++iz) {
                        s_t acc = 0;
#pragma unroll
                        for (int qz = 0; qz < NQ1; ++qz) acc += b[qz * NS1 + iz] * u0[qz] + g[qz * NS1 + iz] * u1[qz];
                        atomicAdd(&values[(ptrdiff_t)sh.block[xy + NS1 * NS1 * iz] * 9 + cj * 3 + ci], v_t(acc));
                    }
                }
                __syncwarp();
            }

            /// The node kernel: a pass per trial node j, its three columns' F_q written at once (one
            /// block barrier), then the nine (cj, ci) contractions, each by a group of 32 threads that
            /// synchronise among themselves only: two block barriers per node (a column at a time with
            /// block-wide stages took four per column, 1.55 times as long on an H100).
            template <class M, class Ref, class Quad, typename s_t, typename v_t, typename g_t, int NQ1, int NS1>
            __global__ void __launch_bounds__(GROUP *GROUPS) bsr_hessian_kernel(const ptrdiff_t nelements,
                                                                                      idx_t **const SFEM_RESTRICT elements,
                                                                                      const g_t *const *const SFEM_RESTRICT points,
                                                                                      const Parameters<s_t, M::n_parameters> parameters,
                                                                                      const ptrdiff_t                       u_stride,
                                                                                      const s_t *const SFEM_RESTRICT        ux,
                                                                                      const s_t *const SFEM_RESTRICT        uy,
                                                                                      const s_t *const SFEM_RESTRICT        uz,
                                                                                      const count_t *const SFEM_RESTRICT    rowptr,
                                                                                      const idx_t *const SFEM_RESTRICT      colidx,
                                                                                      v_t *const SFEM_RESTRICT              values) {
                using Shared         = BSRHessianShared<s_t, NQ1, NS1>;
                static constexpr int NQ = Shared::NQ, NS = Shared::NS;
                __shared__ Shared sh;

                const int q     = threadIdx.x;
                const int group = threadIdx.x / GROUP, lane = threadIdx.x % GROUP;
                const int qx = q % NQ1, qy = (q / NQ1) % NQ1, qz = q / (NQ1 * NQ1);

                for (int t = threadIdx.x; t < NQ1 * NS1; t += blockDim.x) {
                    sh.b[t] = Ref::shape_1d()[t];
                    sh.g[t] = Ref::grad_1d()[t];
                }

                for (ptrdiff_t e = blockIdx.x; e < nelements; e += gridDim.x) {
                    __syncthreads();
                    if (threadIdx.x == 0) sh.has_row = 0;
                    __syncthreads();
                    const s_t *const u_components[3] = {ux, uy, uz};
                    for (int s = threadIdx.x; s < NS; s += blockDim.x) {
                        const idx_t node = elements[s][e];
                        sh.node[s]       = node;
                        for (int d = 0; d < 3; ++d) {
                            sh.x[d][s] = s_t(points[d][node]);
                            sh.u[d][s] = u_components[d][node * u_stride];
                        }
                        if (rowptr[node] < rowptr[node + 1]) sh.has_row = 1;
                    }
                    __syncthreads();
                    if (!sh.has_row) continue;
                    sort_nodes<NS>(sh.node, sh.sorted, sh.local);

                    s_t A[45];
                    if (q < NQ) point_tangent<M, Quad, s_t, NQ1, NS1>(sh.b, sh.g, sh.x, sh.u, q, parameters, A);

                    for (int j = 0; j < NS; ++j) {
                        __syncthreads();  // the previous node's contractions are done with flux and block
                        const bool has_row = row_blocks<NS>(sh.node[j], rowptr, colidx, sh.sorted, sh.local, sh.block);
                        if (q < NQ) {
                            const int jx = j % NS1, jy = (j / NS1) % NS1, jz = j / (NS1 * NS1);
                            const s_t Gj[3] = {sh.g[qx * NS1 + jx] * sh.b[qy * NS1 + jy] * sh.b[qz * NS1 + jz],
                                               sh.b[qx * NS1 + jx] * sh.g[qy * NS1 + jy] * sh.b[qz * NS1 + jz],
                                               sh.b[qx * NS1 + jx] * sh.b[qy * NS1 + jy] * sh.g[qz * NS1 + jz]};
#pragma unroll
                            for (int cj = 0; cj < 3; ++cj) {
#pragma unroll
                                for (int r = 0; r < 9; ++r) {
                                    sh.flux[cj][r][q] = A[symmetric_slot(r, cj * 3 + 0)] * Gj[0] + A[symmetric_slot(r, cj * 3 + 1)] * Gj[1] +
                                                        A[symmetric_slot(r, cj * 3 + 2)] * Gj[2];
                                }
                            }
                        }
                        __syncthreads();
                        if (!has_row) continue;
                        for (int pair = group; pair < 9; pair += GROUPS) contract_pair<Ref, s_t, v_t, NQ1, NS1>(sh, group, lane, pair / 3, pair % 3, values);
                    }
                }
            }

            /// Launches the node kernel: a block of 128 threads per element (at most 125 points).
            template <class M, class Ref, class Quad, typename s_t, typename v_t, typename g_t, int NQ1, int NS1>
            int bsr_hessian(const ptrdiff_t                       nelements,
                            idx_t **const                         elements,
                            const g_t *const *const               points,
                            const Parameters<s_t, M::n_parameters> parameters,
                            const ptrdiff_t                       u_stride,
                            const s_t *const                      ux,
                            const s_t *const                      uy,
                            const s_t *const                      uz,
                            const count_t *const                  rowptr,
                            const idx_t *const                    colidx,
                            v_t *const                            values,
                            void *const                           stream) {
                static_assert(NQ1 * NQ1 * NQ1 <= 128, "a thread per point");
                if (nelements <= 0) return SFEM_SUCCESS;
                const int grid = (int)(nelements < 65535 * 16 ? nelements : 65535 * 16);
                bsr_hessian_kernel<M, Ref, Quad, s_t, v_t, g_t, NQ1, NS1><<<grid, GROUP * GROUPS, 0, (cudaStream_t)stream>>>(
                        nelements, elements, points, parameters, u_stride, ux, uy, uz, rowptr, colidx, values);
                return cudaPeekAtLastError() == cudaSuccess ? SFEM_SUCCESS : SFEM_FAILURE;
            }

        }  // namespace cuda
    }  // namespace tensor_product
}  // namespace sfem

#endif  // SFEM_CUDA_TENSOR_PRODUCT_BSR_HESSIAN_CUH
