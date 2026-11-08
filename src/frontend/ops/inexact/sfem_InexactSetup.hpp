#pragma once

#include <algorithm>
#include <cstddef>

#include "packed_thread_scratch.hpp"
#include "sfem_InexactElement.hpp"
#include "sfem_InexactInterface.hpp"
#include "sfem_TangentProjection.hpp"
#include "sfem_base.hpp"

namespace sfem {
    namespace inexact {

        namespace detail {
            /// adj = det * J^{-1} for J[a][m] = dx_a/dxi_m, so that du_c/dx_a = sum_m du_c/dxi_m adj[m][a] / det.
            template <typename s_t>
            static SFEM_INLINE SFEM_HOST_DEVICE void adjugate_and_determinant(const s_t J[9], s_t adj[9], s_t &det) {
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
            }

            /// Row i, column j of a packed symmetric 9 x 9: the upper triangle by rows.
            static SFEM_INLINE SFEM_HOST_DEVICE int packed_slot(const int i, const int j) {
                const int r = i < j ? i : j, c = i < j ? j : i;
                return r * 9 - r * (r - 1) / 2 + c - r;
            }

            /// The symmetric point tangent T (packed_slot, r = component * 3 + reference direction)
            /// projected to positive semidefinite where the assembled tangent is: on dP/dF.
            /// T = (1/det) P^T D P with P = I (x) adj, so D_(c,a)(d,b) = (1/det) sum_mn J_am T_(c,m)(d,n) J_bn,
            /// J = adjugate(adj) / det; D is projected (project_abs_symmetric) and pulled back.
            ///
            /// On the device it is a call: inlined into the set-up kernel after the generated point
            /// tangent, nvcc 12.6 at -O3 returned a wrong projection of a correct tangent (with and
            /// without fast math, at every ptxas level; right at -G and as a call; 2026-11-08).
#if defined(__CUDACC__)
            template <typename s_t>
            static __host__ __device__ __noinline__ void project_point_tangent(s_t *const T, const s_t adj[9], const s_t det) {
#else
            template <typename s_t>
            static SFEM_INLINE void project_point_tangent(s_t *const T, const s_t adj[9], const s_t det) {
#endif
                s_t J[9], det2;
                adjugate_and_determinant(adj, J, det2);
                const s_t inv_det = s_t(1) / det;
                for (int k = 0; k < 9; ++k) J[k] *= inv_det;

                // W_(c,a)(d,n) = sum_m J_am T_(c,m)(d,n), then D_(c,a)(d,b) = (1/det) sum_n W_(c,a)(d,n) J_bn.
                s_t W[81], D[81];
                for (int c = 0; c < 3; ++c)
                    for (int a = 0; a < 3; ++a)
                        for (int col = 0; col < 9; ++col) {
                            s_t acc = 0;
                            for (int m = 0; m < 3; ++m) acc += J[a * 3 + m] * T[packed_slot(c * 3 + m, col)];
                            W[(c * 3 + a) * 9 + col] = acc;
                        }
                for (int row = 0; row < 9; ++row)
                    for (int d = 0; d < 3; ++d)
                        for (int b = 0; b < 3; ++b) {
                            s_t acc = 0;
                            for (int n = 0; n < 3; ++n) acc += W[row * 9 + d * 3 + n] * J[b * 3 + n];
                            D[row * 9 + d * 3 + b] = acc * inv_det;
                        }
                sfem::project_abs_symmetric<s_t, 9>(D);

                // T_(c,m)(d,n) = (1/det) sum_ab adj_ma D_(c,a)(d,b) adj_nb, the upper triangle.
                for (int c = 0; c < 3; ++c)
                    for (int m = 0; m < 3; ++m)
                        for (int col = 0; col < 9; ++col) {
                            s_t acc = 0;
                            for (int a = 0; a < 3; ++a) acc += adj[m * 3 + a] * D[(c * 3 + a) * 9 + col];
                            W[(c * 3 + m) * 9 + col] = acc;
                        }
                for (int row = 0; row < 9; ++row)
                    for (int d = 0; d < 3; ++d)
                        for (int n = 0; n < 3; ++n) {
                            if (d * 3 + n < row) continue;
                            s_t acc = 0;
                            for (int b = 0; b < 3; ++b) acc += W[row * 9 + d * 3 + b] * adj[n * 3 + b];
                            T[packed_slot(row, d * 3 + n)] = acc * inv_det;
                        }
            }
        }  // namespace detail

        /// The set-up: every point's weighted tangent, block by block, handed to the store.
        ///
        /// For a block of `VS` elements: gather the state in the element's staging order, form
        /// its reference gradient at the set-up's points (`setup_in`, the element's own sweeps),
        /// pull it through the geometry and evaluate the material there (`M::point`, generated),
        /// weight it, and accumulate it into the block's stored tensors; then the store encodes
        /// the block.  Scratch comes from the thread's arena: a HEX27 block of tensors is 0.4 MB.
        /// PROJECT: each point's tangent projected first (detail::project_point_tangent).
        template <class Store, bool PROJECT>
        void setup_points(Store &store, const SetupInput &in) {
            using M                = typename Store::Unit;
            using El               = typename M::element;
            static constexpr int VS = Store::VS;
            static constexpr int N  = El::N;
            static constexpr int NQ = El::NQ;
            static constexpr int NV = El::NV;
            static constexpr int NC = M::NC;
            static constexpr int D  = El::DIM;
            static constexpr int D2 = D * D;
            static_assert(D == 3, "the stored tangent is three-dimensional");

            const bool affine = in.adjugate != nullptr;
            if (!affine && !in.points) {
                SFEM_ERROR("inexact::setup: neither the affine geometry nor the coordinates were given\n");
            }
            static constexpr int NF = M::n_quadrature_fields;
            static constexpr int MC = M::max_quadrature_components;
            if (NF > 0 && !in.quadrature) {
                SFEM_ERROR("inexact::setup: the tangent reads integration-point fields that were not given\n");
            }

#pragma omp parallel
            {
                real_t *const scratch = codegen::thread_scratch<real_t>(
                        3, (size_t)(NV * NC + 3 * NQ * D2 + 3 * D * N) * VS);
                real_t(*const bT)[NC][VS] = reinterpret_cast<real_t(*)[NC][VS]>(scratch);
                real_t(*const G)[D2][VS]  = reinterpret_cast<real_t(*)[D2][VS]>(scratch + (size_t)NV * NC * VS);
                real_t(*const GZ)[D2][VS] = G + NQ;
                real_t(*const GX)[D2][VS] = GZ + NQ;
                real_t(*const bu)[N][VS]  = reinterpret_cast<real_t(*)[N][VS]>(scratch + (size_t)(NV * NC + 3 * NQ * D2) * VS);
                real_t(*const bz)[N][VS]  = bu + D;
                real_t(*const bx)[N][VS]  = bz + D;

#pragma omp for schedule(static)
                for (ptrdiff_t evb = 0; evb < in.nelements; evb += VS) {
                    const int ne = (int)std::min<ptrdiff_t>(VS, in.nelements - evb);

                    if constexpr (El::moments) {
                        for (int j = 0; j < NV; ++j)
                            for (int c = 0; c < NC; ++c)
#pragma omp simd
                                for (int lane = 0; lane < VS; ++lane) bT[j][c][lane] = 0;
                    }

                    if constexpr (M::reads_state) {
                        for (int s = 0; s < N; ++s) {
                            const idx_t *const RSTR nodes = in.elements[ElementTables<El>::shape_node(s)];
                            for (int lane = 0; lane < ne; ++lane) {
                                const ptrdiff_t v = (ptrdiff_t)nodes[evb + lane] * in.u_stride;
                                for (int c = 0; c < D; ++c) bu[c][s][lane] = in.u[v + c];
                            }
                        }
                        El::template setup_in<real_t, real_t, VS>(ne, bu, G);
                    }
                    if constexpr (M::reads_previous) {
                        for (int s = 0; s < N; ++s) {
                            const idx_t *const RSTR nodes = in.elements[ElementTables<El>::shape_node(s)];
                            for (int lane = 0; lane < ne; ++lane) {
                                const ptrdiff_t v = (ptrdiff_t)nodes[evb + lane] * in.u_stride;
                                for (int c = 0; c < D; ++c) bz[c][s][lane] = in.z[v + c];
                            }
                        }
                        El::template setup_in<real_t, real_t, VS>(ne, bz, GZ);
                    }
                    if (!affine) {
                        for (int s = 0; s < N; ++s) {
                            const idx_t *const RSTR nodes = in.elements[ElementTables<El>::shape_node(s)];
                            for (int lane = 0; lane < ne; ++lane)
                                for (int c = 0; c < D; ++c) bx[c][s][lane] = (real_t)in.points[c][nodes[evb + lane]];
                        }
                        El::template setup_in<real_t, real_t, VS>(ne, bx, GX);
                    }

                    for (int q = 0; q < NQ; ++q) {
                        const real_t w = real_t(El::QW[q]);
#pragma omp simd
                        for (int lane = 0; lane < ne; ++lane) {
                            real_t gu[D2], gz[D2], adj[D2], det;
                            for (int c = 0; c < D2; ++c) {
                                gu[c] = M::reads_state ? G[q][c][lane] : real_t(0);
                                gz[c] = M::reads_previous ? GZ[q][c][lane] : real_t(0);
                            }
                            if (affine) {
                                for (int c = 0; c < D2; ++c) adj[c] = (real_t)in.adjugate[c][evb + lane];
                                det = (real_t)in.determinant[evb + lane];
                            } else {
                                real_t J[D2];
                                for (int c = 0; c < D2; ++c) J[c] = GX[q][c][lane];
                                detail::adjugate_and_determinant(J, adj, det);
                            }
                            // The integration-point fields at this point, component by component.
                            real_t        qf[NF > 0 ? NF : 1][MC];
                            const real_t *qp[NF > 0 ? NF : 1];
                            if constexpr (NF > 0) {
                                for (int f = 0; f < NF; ++f) {
                                    for (int c = 0; c < M::quadrature_components[f]; ++c)
                                        qf[f][c] = in.quadrature[f][((ptrdiff_t)c * NQ + q) * in.quadrature_stride + evb + lane];
                                    qp[f] = qf[f];
                                }
                            }
                            real_t T[NC];
                            M::template point<real_t>(in.parameters, gu, gz, adj, det, qp, T);
                            if constexpr (PROJECT) detail::project_point_tangent(T, adj, det);
                            for (int c = 0; c < NC; ++c) Store::accumulate(q, bT, c, lane, w * T[c]);
                        }
                    }

                    store.encode_block(evb, ne, bT);
                }
            }
        }

        template <class Store>
        void setup(Store &store, const SetupInput &in) {
            if (!in.project) return setup_points<Store, false>(store, in);
            if constexpr (Store::Unit::symmetric && Store::Unit::NC == 45) {
                setup_points<Store, true>(store, in);
            } else {
                SFEM_ERROR("inexact::setup: only a symmetric tangent is projected\n");
            }
        }

    }  // namespace inexact
}  // namespace sfem
