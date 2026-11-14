#ifndef SFEM_CUDA_TENSOR_PRODUCT_MATRIX_CORE_APPLY_CUH
#define SFEM_CUDA_TENSOR_PRODUCT_MATRIX_CORE_APPLY_CUH

// The matrix-free apply of a tensor-product element (NS1^3 nodes, NQ1^3 points, NC components) on
// AMD's FP64 matrix cores (v_mfma_f64_16x16x4f64, gfx90a and later). HIP builds only.
//
// A wavefront takes 16 elements, one per column of a 16x16 tile. With G the reference gradient
// matrix, G[(k, q)][s] = dphi_s/dxi_k at point q (a product of the 1D tables b, g), the reference
// gradients of every field f at the points are the GEMM
//
//   R_f = G U_f,   U_f[s][e] = u_f(node s of element e),
//
// rows (k, q) taken in tiles of 16 points: lane l then holds, for element l % 16, the three
// directions at the points q = 16 h + 4 i + l / 16 (slot i of the result, D(r, j) in slot r / 4 of
// lane j + 16 (r % 4)). The geometry (isoparametric: the same GEMM on the coordinates), the
// material's flux at the point (`P::point`, generated per material) and the weight are formed in
// registers, and the flux tiles are already the B operand of the test contraction
//
//   Y_c = G^T F_c,
//
// whose output rows are the nodes the lane gathered, so the gather's indices serve the scatter.
// The points are taken a tile row (16 points, all directions) at a time, which bounds the live
// accumulators. G is tabulated in shared memory when it fits (Q2: 44.5 KB isoparametric; its entries
// cost registers when formed on the fly, 138 spilled VGPRs against 54 at Q2), otherwise its entries
// are formed from the 1D tables as the instructions need them.
//
// P provides n_parameters, reads_current (the state u is differentiated too, for the linearisation
// of a nonlinear material) and
//
//   point(parameters, grad_h_ref[NC * 3], gu_ref[NC * 3], adj[9], det, w, loperand[NC * 3])
//
// with index 3 c + k for component c and reference direction k, adj the adjugate of the Jacobian
// dx/dxi and det its determinant, as the generated apply block.

#include "sfem_base.hpp"

#if defined(SFEM_ENABLE_HIP)

#include <cstddef>
#include <cstdlib>

namespace sfem {
    namespace tensor_product {
        namespace hip {

            typedef double matrix_core_double4 __attribute__((ext_vector_type(4)));

            /// What `matrix_core_apply` returns when `SFEM_MATRIX_CORE_APPLY=0` turns it off: the caller
            /// launches its own kernel instead.
            static constexpr int MATRIX_CORE_DECLINED = -1;

            inline bool matrix_core_apply_enabled() {
                static const bool enabled = [] {
                    const char *const value = std::getenv("SFEM_MATRIX_CORE_APPLY");
                    return !value || std::atoi(value) != 0;
                }();
                return enabled;
            }

            template <int N>
            struct MatrixCoreParameters {
                double value[N > 0 ? N : 1];
            };

            /// The affine geometry per element, passed by value: the adjugate's 9 entries, then the
            /// determinant.
            template <typename g_t>
            struct MatrixCoreAffineGeometry {
                const g_t *value[10];
            };

            /// The three entries G[(k, q)][s], k = 0, 1, 2, from the 1D tables (rows: points).
            template <int NS1, int NQ1>
            __device__ __forceinline__ void reference_gradient_entries(const double *const b,
                                                                       const double *const g,
                                                                       const int           q,
                                                                       const int           s,
                                                                       double (&G)[3]) {
                const int qx = q % NQ1, qy = (q / NQ1) % NQ1, qz = q / (NQ1 * NQ1);
                const int sx = s % NS1, sy = (s / NS1) % NS1, sz = s / (NS1 * NS1);
                const double bx = b[qx * NS1 + sx], by = b[qy * NS1 + sy], bz = b[qz * NS1 + sz];
                const double gx = g[qx * NS1 + sx], gy = g[qy * NS1 + sy], gz = g[qz * NS1 + sz];
                G[0] = gx * by * bz;
                G[1] = bx * gy * bz;
                G[2] = bx * by * gz;
            }

            /// The doubles a lane keeps in accumulators: the gradient tiles of a tile row (the field's
            /// components, the coordinates when isoparametric, the state's components when read) and
            /// the output tiles.
            template <int NS1, int NC, bool ISOPARAMETRIC, bool CURRENT>
            constexpr int matrix_core_accumulators() {
                return 4 * (3 * (NC + (ISOPARAMETRIC ? 3 : 0) + (CURRENT ? NC : 0)) + NC * ((NS1 * NS1 * NS1 + 15) / 16));
            }

            /// Wavefronts a block. A block's waves share one compute unit's four SIMDs, so eight
            /// waves give a lane 256 registers and four give it 512. Eight while the accumulators
            /// leave room for the point's arithmetic and the material is linear (linear elasticity
            /// at Q2: no spills), else four: a nonlinear material's linearised stress spills at 256
            /// even at Q2, and isoparametric Q3 reading the state (156 doubles in accumulators)
            /// fails to allocate there.
            template <int NS1, int NC, bool ISOPARAMETRIC, bool CURRENT>
            constexpr int matrix_core_waves() {
                return (!CURRENT && matrix_core_accumulators<NS1, NC, ISOPARAMETRIC, CURRENT>() <= 96) ? 8 : 4;
            }

            template <class Basis, class Quadrature, int NS1, int NQ1, int NC, bool ISOPARAMETRIC, class P, int N_WAVES, typename g_t>
            __global__ void __launch_bounds__(64 * N_WAVES, 1)
                    matrix_core_apply_kernel(const ptrdiff_t                                   nelements,
                                             idx_t **const SFEM_RESTRICT                      elements,
                                             const MatrixCoreAffineGeometry<g_t>              affine,
                                             const g_t *const *const SFEM_RESTRICT            points,
                                             const MatrixCoreParameters<P::n_parameters>      parameters,
                                             const ptrdiff_t                                  u_stride,
                                             const double *const u0,
                                             const double *const u1,
                                             const double *const u2,
                                             const ptrdiff_t                                  h_stride,
                                             const double *const h0,
                                             const double *const h1,
                                             const double *const h2,
                                             const ptrdiff_t                                  out_stride,
                                             double *const out0,
                                             double *const out1,
                                             double *const out2) {
                static constexpr int  NS      = NS1 * NS1 * NS1;
                static constexpr int  NQ      = NQ1 * NQ1 * NQ1;
                static constexpr int  KS      = (NS + 3) / 4;    // k-steps of the gradient GEMM
                static constexpr int  TR      = (NQ + 15) / 16;  // tile rows of points
                static constexpr int  TO      = (NS + 15) / 16;  // output tiles
                static constexpr bool CURRENT = P::reads_current;
                static constexpr int  F_GEO   = NC;                                // first coordinate field
                static constexpr int  F_CUR   = NC + (ISOPARAMETRIC ? 3 : 0);       // first state field
                static constexpr int  NF      = F_CUR + (CURRENT ? NC : 0);
                static constexpr int  NQP     = 16 * TR;  // points, padded to whole tiles
                static constexpr int  NSP     = 4 * KS;   // nodes, padded to whole k-steps
                static constexpr bool TABLE   = 3 * NQP * (NSP + 1) * sizeof(double) <= 46 * 1024;
                static_assert(NC == 1 || NC == 3, "NC must be 1 or 3");

                __shared__ double b[NQ1 * NS1], g[NQ1 * NS1], w[NQ1];
                __shared__ double G_table[TABLE ? 3 * NQP : 1][NSP + 1];
                {
                    const double *const shape_1d = Basis::shape_1d();
                    const double *const grad_1d  = Basis::grad_1d();
                    const double *const q_weight = Quadrature::q_weight_1d();
                    for (int i = threadIdx.x; i < NQ1 * NS1; i += blockDim.x) {
                        b[i] = shape_1d[i];
                        g[i] = grad_1d[i];
                    }
                    for (int i = threadIdx.x; i < NQ1; i += blockDim.x) w[i] = q_weight[i];
                }
                __syncthreads();
                if constexpr (TABLE) {
                    for (int i = threadIdx.x; i < NQP * NSP; i += blockDim.x) {
                        const int q = i / NSP, s = i % NSP;
                        double    G[3];
                        reference_gradient_entries<NS1, NQ1>(b, g, q < NQ ? q : 0, s < NS ? s : 0, G);
#pragma unroll
                        for (int k = 0; k < 3; ++k) G_table[k * NQP + q][s] = (q < NQ && s < NS) ? G[k] : 0.0;
                    }
                    __syncthreads();
                }
                // G[(k, q)][s] for a point q < NQP and a node s < NSP (zero in the padding).
                auto entries = [&](const int q, const int s, double(&G)[3]) {
                    if constexpr (TABLE) {
#pragma unroll
                        for (int k = 0; k < 3; ++k) G[k] = G_table[k * NQP + q][s];
                    } else {
                        reference_gradient_entries<NS1, NQ1>(b, g, q < NQ ? q : 0, s < NS ? s : 0, G);
                        if (q >= NQ || s >= NS) G[0] = G[1] = G[2] = 0;
                    }
                };

                const int lane = threadIdx.x % 64;
                const int col  = lane % 16;
                const int grp  = lane / 16;

                const double *const h_fields[3]   = {h0, h1, h2};
                const double *const u_fields[3]   = {u0, u1, u2};
                double *const       out_fields[3] = {out0, out1, out2};

                for (ptrdiff_t base = ((ptrdiff_t)blockIdx.x * N_WAVES + threadIdx.x / 64) * 16; base < nelements;
                     base += (ptrdiff_t)gridDim.x * N_WAVES * 16) {
                    const ptrdiff_t e     = base + col;
                    const bool      valid = e < nelements;

                    idx_t idx[KS];
#pragma unroll
                    for (int ks = 0; ks < KS; ++ks) {
                        const int s = 4 * ks + grp;
                        idx[ks]     = (valid && s < NS) ? elements[s][e] : idx_t(-1);
                    }

                    double adj_e[9] = {0}, det_e = 1;
                    if (!ISOPARAMETRIC && valid) {
#pragma unroll
                        for (int i = 0; i < 9; ++i) adj_e[i] = (double)affine.value[i][e];
                        det_e = (double)affine.value[9][e];
                    }

                    matrix_core_double4 out[NC][TO];
#pragma unroll
                    for (int c = 0; c < NC; ++c)
#pragma unroll
                        for (int to = 0; to < TO; ++to) out[c][to] = matrix_core_double4{0, 0, 0, 0};

#pragma unroll 1
                    for (int h = 0; h < TR; ++h) {
                        matrix_core_double4 acc[NF][3];
#pragma unroll
                        for (int f = 0; f < NF; ++f)
#pragma unroll
                            for (int k = 0; k < 3; ++k) acc[f][k] = matrix_core_double4{0, 0, 0, 0};

                        // Reference gradients at the tile row's points (A: row 16 h + col, column node 4 ks + grp).
                        const int qa = 16 * h + col;
#pragma unroll
                        for (int ks = 0; ks < KS; ++ks) {
                            const int s = 4 * ks + grp;
                            double    G[3];
                            entries(qa, s, G);

                            double v[NF];
                            const bool has = idx[ks] >= 0;
#pragma unroll
                            for (int c = 0; c < NC; ++c) v[c] = has ? h_fields[c][idx[ks] * h_stride] : 0.0;
                            if constexpr (ISOPARAMETRIC) {
#pragma unroll
                                for (int d = 0; d < 3; ++d) v[F_GEO + d] = has ? (double)points[d][idx[ks]] : 0.0;
                            }
                            if constexpr (CURRENT) {
#pragma unroll
                                for (int c = 0; c < NC; ++c) v[F_CUR + c] = has ? u_fields[c][idx[ks] * u_stride] : 0.0;
                            }
#pragma unroll
                            for (int k = 0; k < 3; ++k)
#pragma unroll
                                for (int f = 0; f < NF; ++f)
                                    acc[f][k] = __builtin_amdgcn_mfma_f64_16x16x4f64(G[k], v[f], acc[f][k], 0, 0, 0);
                        }

                        // The flux at the lane's points, written over the trial gradients.
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const int q  = 16 * h + 4 * i + grp;
                            const int qq = q < NQ ? q : 0;
                            const double qw =
                                    (valid && q < NQ) ? w[qq % NQ1] * w[(qq / NQ1) % NQ1] * w[qq / (NQ1 * NQ1)] : 0.0;

                            double adj[9], det;
                            if constexpr (ISOPARAMETRIC) {
                                const double J00 = acc[F_GEO][0][i], J01 = acc[F_GEO][1][i], J02 = acc[F_GEO][2][i];
                                const double J10 = acc[F_GEO + 1][0][i], J11 = acc[F_GEO + 1][1][i], J12 = acc[F_GEO + 1][2][i];
                                const double J20 = acc[F_GEO + 2][0][i], J21 = acc[F_GEO + 2][1][i], J22 = acc[F_GEO + 2][2][i];
                                adj[0] = J11 * J22 - J12 * J21;
                                adj[1] = J02 * J21 - J01 * J22;
                                adj[2] = J01 * J12 - J02 * J11;
                                adj[3] = J12 * J20 - J10 * J22;
                                adj[4] = J00 * J22 - J02 * J20;
                                adj[5] = J02 * J10 - J00 * J12;
                                adj[6] = J10 * J21 - J11 * J20;
                                adj[7] = J01 * J20 - J00 * J21;
                                adj[8] = J00 * J11 - J01 * J10;
                                det    = J00 * (J11 * J22 - J12 * J21) - J01 * (J10 * J22 - J12 * J20) +
                                      J02 * (J10 * J21 - J11 * J20);
                                if (qw == 0) det = 1;
                            } else {
#pragma unroll
                                for (int j = 0; j < 9; ++j) adj[j] = adj_e[j];
                                det = det_e;
                            }

                            double gh[NC * 3], gu[NC * 3], flux[NC * 3];
#pragma unroll
                            for (int c = 0; c < NC; ++c)
#pragma unroll
                                for (int k = 0; k < 3; ++k) {
                                    gh[3 * c + k] = acc[c][k][i];
                                    gu[3 * c + k] = CURRENT ? acc[F_CUR + c][k][i] : 0.0;
                                }
                            P::template point<double>(parameters.value, gh, gu, adj, det, qw, flux);
#pragma unroll
                            for (int c = 0; c < NC; ++c)
#pragma unroll
                                for (int k = 0; k < 3; ++k) acc[c][k][i] = qw != 0 ? flux[3 * c + k] : 0.0;
                        }

                        // The test contraction (A: row node 16 to + col, column point 16 h + 4 i + grp).
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const int q = 16 * h + 4 * i + grp;
#pragma unroll
                            for (int to = 0; to < TO; ++to) {
                                const int s = 16 * to + col;
                                double    G[3];
                                if (s < NSP) {
                                    entries(q, s, G);
                                } else {
                                    G[0] = G[1] = G[2] = 0;
                                }
#pragma unroll
                                for (int k = 0; k < 3; ++k)
#pragma unroll
                                    for (int c = 0; c < NC; ++c)
                                        out[c][to] = __builtin_amdgcn_mfma_f64_16x16x4f64(G[k], acc[c][k][i], out[c][to], 0, 0, 0);
                            }
                        }
                    }

#pragma unroll
                    for (int to = 0; to < TO; ++to) {
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const int ks = 4 * to + i;
                            if (ks < KS && idx[ks] >= 0) {
#pragma unroll
                                for (int c = 0; c < NC; ++c) atomicAdd(&out_fields[c][idx[ks] * out_stride], out[c][to][i]);
                            }
                        }
                    }
                }
            }

            /// Launches the apply on `stream`. `affine` holds the adjugate (9 entries) and determinant
            /// per element (unused when isoparametric), `points` the device array of the coordinate
            /// arrays (unused when affine), `u` the state when P reads it. Returns SFEM_SUCCESS or
            /// SFEM_FAILURE on a launch error, or MATRIX_CORE_DECLINED when turned off.
            template <class Basis, class Quadrature, int NS1, int NQ1, int NC, bool ISOPARAMETRIC, class P, typename g_t>
            int matrix_core_apply(const ptrdiff_t                              nelements,
                                  idx_t **const                                elements,
                                  const MatrixCoreAffineGeometry<g_t>         &affine,
                                  const g_t *const *const                      points,
                                  const MatrixCoreParameters<P::n_parameters> &parameters,
                                  const ptrdiff_t                              u_stride,
                                  const double *const *const                   u,
                                  const ptrdiff_t                              h_stride,
                                  const double *const *const                   h,
                                  const ptrdiff_t                              out_stride,
                                  double *const *const                         out,
                                  cudaStream_t                                 stream) {
                static constexpr int N_WAVES = matrix_core_waves<NS1, NC, ISOPARAMETRIC, P::reads_current>();
                if (!matrix_core_apply_enabled()) return MATRIX_CORE_DECLINED;
                if (nelements <= 0) return SFEM_SUCCESS;
                const ptrdiff_t per_block = 16 * N_WAVES;
                const int       n_blocks  = (int)((nelements + per_block - 1) / per_block);
                const double   *uu[3]     = {u ? u[0] : nullptr, u && NC > 1 ? u[1] : nullptr, u && NC > 2 ? u[2] : nullptr};
                const double   *hh[3]     = {h[0], NC > 1 ? h[1] : nullptr, NC > 2 ? h[2] : nullptr};
                double         *oo[3]     = {out[0], NC > 1 ? out[1] : nullptr, NC > 2 ? out[2] : nullptr};
                matrix_core_apply_kernel<Basis, Quadrature, NS1, NQ1, NC, ISOPARAMETRIC, P, N_WAVES, g_t>
                        <<<n_blocks, 64 * N_WAVES, 0, stream>>>(nelements,
                                                                elements,
                                                                affine,
                                                                points,
                                                                parameters,
                                                                u_stride,
                                                                uu[0],
                                                                uu[1],
                                                                uu[2],
                                                                h_stride,
                                                                hh[0],
                                                                hh[1],
                                                                hh[2],
                                                                out_stride,
                                                                oo[0],
                                                                oo[1],
                                                                oo[2]);
                return cudaGetLastError() == cudaSuccess ? SFEM_SUCCESS : SFEM_FAILURE;
            }

        }  // namespace hip
    }  // namespace tensor_product
}  // namespace sfem

#endif  // SFEM_ENABLE_HIP
#endif  // SFEM_CUDA_TENSOR_PRODUCT_MATRIX_CORE_APPLY_CUH
