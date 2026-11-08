#pragma once

// The device assembly of a higher-order simplex's hessian into BSR, a block of threads per
// ELEMENTS elements and a thread per element and quadrature point (the generated `*_assemble_impl`
// kernels of TET10 and TRI6):
//
//   1. the element's nodes, coordinates and state are gathered into shared memory, and its first
//      NS threads find where its node pairs' blocks are in the BSR, a row each (`row_map`);
//   2. each thread forms its point's geometry (`point_geometry`) and its tangent, generated per
//      material, which ends with the tangent pulled back to the reference frame and weighted
//      (`pullback_symmetric`) into shared memory;
//   3. the block's threads form the elements' node-pair blocks from the points' tangents and add
//      each, and its transpose, to the BSR (`pair_blocks_bsr`).
//
// No thread holds an element matrix. A thread per element holding the 30 x 30 matrix of a TET10 in
// local memory (255 registers, 11 KB of stack, 12.5 % occupancy) took 50.7 ms on FF200 TET10
// (494,097 elements, Mooney-Rivlin Prony, H100); this takes 9.2 ms, the same values to rounding.
// Without the row map each pair searched two rows (16.2 ms); with registers unbounded two blocks fit
// per SM (10.9 ms); bounded for three, 9.2 ms; for six, the spills cost more (12.9 ms).

#include <cuda_runtime.h>

#include "sfem_TangentProjection.hpp"
#include "sfem_base.hpp"

namespace sfem {
    namespace cooperative {

        /// Elements per block.
        static constexpr int ELEMENTS = 8;
        /// Resident blocks per SM the kernel's registers are bounded for (`__launch_bounds__`).
        static constexpr int MIN_BLOCKS = 3;

        /// A block's threads: a thread per element and point.
        template <int NQ>
        __host__ __device__ constexpr int threads() {
            return ELEMENTS * NQ;
        }

        /// Entries of the packed upper triangle of a symmetric N x N.
        template <int N>
        __host__ __device__ constexpr int n_packed() {
            return N * (N + 1) / 2;
        }

        /// Row i, column j of the packed upper triangle of a symmetric N x N, by rows.
        template <int N>
        __device__ __forceinline__ int packed(const int i, const int j) {
            const int r = i < j ? i : j, c = i < j ? j : i;
            return r * N - r * (r - 1) / 2 + c - r;
        }

        /// The point's Jacobian J[a][m] = sum_s x[a][s] grad_ref[m][q * NS + s] from the element's
        /// coordinates, and its adjugate (adj[m * ND + a] = det J^-1[m][a], as
        /// `geometry_jacobian_adjugate_and_determinant_*`) and determinant.
        template <typename s_t, int NS, int ND>
        __device__ __forceinline__ void point_geometry(const s_t (*const x)[NS],
                                                       const s_t *const *const grad_ref,
                                                       const int           q,
                                                       s_t *const          adj,
                                                       s_t                &det) {
            static_assert(ND == 2 || ND == 3, "point_geometry is spelled for 2 and 3 dimensions");
            s_t J[ND * ND] = {0};
            for (int s = 0; s < NS; ++s)
                for (int m = 0; m < ND; ++m) {
                    const s_t g = grad_ref[m][q * NS + s];
                    for (int a = 0; a < ND; ++a) J[a * ND + m] += x[a][s] * g;
                }
            if constexpr (ND == 3) {
                adj[0] = J[4] * J[8] - J[5] * J[7];
                adj[1] = J[2] * J[7] - J[1] * J[8];
                adj[2] = J[1] * J[5] - J[2] * J[4];
                adj[3] = J[5] * J[6] - J[3] * J[8];
                adj[4] = J[0] * J[8] - J[2] * J[6];
                adj[5] = J[2] * J[3] - J[0] * J[5];
                adj[6] = J[3] * J[7] - J[4] * J[6];
                adj[7] = J[1] * J[6] - J[0] * J[7];
                adj[8] = J[0] * J[4] - J[1] * J[3];
                det    = J[0] * adj[0] + J[1] * adj[3] + J[2] * adj[6];
            } else {
                adj[0] = J[3];
                adj[1] = -J[1];
                adj[2] = -J[2];
                adj[3] = J[0];
                det    = J[0] * J[3] - J[1] * J[2];
            }
        }

        /// `project_abs_symmetric` as a call: inlined after a generated point tangent in a kernel
        /// this size, nvcc 12.6 at -O3 returned a wrong projection (TET10, 1e18; right as a call),
        /// as in the stored tangent's set-up (`inexact::detail::project_point_tangent`).
        template <typename s_t, int N>
        __device__ __noinline__ void project_abs_symmetric_call(s_t *const D) {
            sfem::project_abs_symmetric<s_t, N>(D);
        }

        /// The physical tangent D[(i, d), (j, e)] ((NC * ND) x (NC * ND), i and j field components)
        /// pulled back to the reference frame and weighted:
        ///   T[(i, m), (j, n)] = w sum_{d, e} adj[m][d] D[(i, d), (j, e)] adj[n][e],
        /// packed (`packed`), so that an element matrix is sum_q g_ref^T T g_ref.
        template <typename s_t, int NC, int ND>
        __device__ __forceinline__ void pullback_symmetric(const s_t *const adj, const s_t w, const s_t *const D, s_t *const T) {
            static constexpr int NA = NC * ND;
            for (int r = 0; r < NA; ++r) {
                const int i = r / ND, m = r % ND;
                s_t       row[NA];  // sum_d adj[m][d] D[(i, d), col]
                for (int col = 0; col < NA; ++col) {
                    s_t acc = 0;
                    for (int d = 0; d < ND; ++d) acc += adj[m * ND + d] * D[(i * ND + d) * NA + col];
                    row[col] = acc;
                }
                for (int c = r; c < NA; ++c) {
                    const int j = c / ND, n = c % ND;
                    s_t       acc = 0;
                    for (int e = 0; e < ND; ++e) acc += row[j * ND + e] * adj[n * ND + e];
                    T[packed<NA>(r, c)] = w * acc;
                }
            }
        }

        /// Row `s` of the element: where the blocks (s, t) of its nodes ev[t] are in the BSR.
        template <int NS>
        __device__ __forceinline__ void row_map(const idx_t *const   ev,
                                                const int            s,
                                                const count_t *const rowptr,
                                                const idx_t *const   colidx,
                                                count_t (*const pos)[NS]) {
            const count_t begin = rowptr[ev[s]], end = rowptr[ev[s] + 1];
            for (count_t k = begin; k < end; ++k) {
                const idx_t column = colidx[k];
                for (int t = 0; t < NS; ++t)
                    if (column == ev[t]) pos[s][t] = k;
            }
        }

        /// The node-pair blocks of the block's `count` elements, B(s, t)[i][j] =
        /// sum_q sum_{m, n} g_ref[m][q * NS + s] T_q[(i, m), (j, n)] g_ref[n][q * NS + t], added to the
        /// BSR at `pos` with their transposes; the block's threads take the pairs (s <= t) in turn.
        template <typename s_t, typename v_t, int NS, int NQ, int NC, int ND>
        __device__ __forceinline__ void pair_blocks_bsr(const int           count,
                                                        const s_t (*const T)[NQ][n_packed<NC * ND>()],
                                                        const s_t *const *const grad_ref,
                                                        const count_t (*const pos)[NS][NS],
                                                        v_t *const          values) {
            static constexpr int NA    = NC * ND;
            static constexpr int PAIRS = NS * (NS + 1) / 2;
            for (int task = threadIdx.x; task < count * PAIRS; task += blockDim.x) {
                const int le = task / PAIRS;
                int       p  = task % PAIRS, s = 0;
                while (p >= NS - s) p -= NS - s++;
                const int t = s + p;
                s_t       B[NC * NC] = {0};
                for (int q = 0; q < NQ; ++q) {
                    s_t gs[ND], gt[ND];
                    for (int m = 0; m < ND; ++m) {
                        gs[m] = grad_ref[m][q * NS + s];
                        gt[m] = grad_ref[m][q * NS + t];
                    }
                    const s_t *const Tq = T[le][q];
                    for (int i = 0; i < NC; ++i)
                        for (int j = 0; j < NC; ++j) {
                            s_t acc = 0;
                            for (int m = 0; m < ND; ++m)
                                for (int n = 0; n < ND; ++n) acc += gs[m] * Tq[packed<NA>(i * ND + m, j * ND + n)] * gt[n];
                            B[i * NC + j] += acc;
                        }
                }
                v_t *block = values + (ptrdiff_t)pos[le][s][t] * (NC * NC);
                for (int k = 0; k < NC * NC; ++k) atomicAdd(&block[k], v_t(B[k]));
                if (s != t) {
                    block = values + (ptrdiff_t)pos[le][t][s] * (NC * NC);
                    for (int i = 0; i < NC; ++i)
                        for (int j = 0; j < NC; ++j) atomicAdd(&block[j * NC + i], v_t(B[i * NC + j]));
                }
            }
        }

    }  // namespace cooperative
}  // namespace sfem
