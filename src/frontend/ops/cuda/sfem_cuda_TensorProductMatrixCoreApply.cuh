#ifndef SFEM_CUDA_TENSOR_PRODUCT_MATRIX_CORE_APPLY_CUH
#define SFEM_CUDA_TENSOR_PRODUCT_MATRIX_CORE_APPLY_CUH

// The matrix-free apply of a tensor-product element (NS1^3 nodes, NQ1^3 points, NC components) on
// the GPU's FP64 matrix units: AMD's matrix cores (v_mfma_f64_16x16x4f64, gfx90a and later) in HIP
// builds, NVIDIA's tensor cores (mma.sync m8n8k4 f64, sm_80 and later) in CUDA builds.
//
// With G the reference gradient matrix, G[(k, q)][s] = dphi_s/dxi_k at point q (a product of the 1D
// tables b, g), the reference gradients of every field f at the points are the GEMM
//
//   R_f = G U_f,   U_f[s][e] = u_f(node s of element e),
//
// the geometry (isoparametric: the same GEMM on the coordinates), the material's flux at the point
// (`P::point`, generated per material) and the weight are formed in registers, and the test
// contraction is a second GEMM
//
//   Y_c = G^T F_c,
//
// whose operand is the flux as the first GEMM left it in the registers: no data moves between the
// two. A warp (wavefront) takes a batch of elements, the points a tile at a time, which bounds the
// live accumulators. G is tabulated in shared memory when it fits (Q2), otherwise its entries are
// formed from the 1D tables as the instructions need them (Q2 isoparametric on AMD: 138 spilled
// VGPRs that way against 54 with the table).
//
// P provides n_parameters, reads_current (the state u is differentiated too, for the linearisation
// of a nonlinear material) and
//
//   point(parameters, grad_h_ref[NC * 3], gu_ref[NC * 3], adj[9], det, w, loperand[NC * 3])
//
// with index 3 c + k for component c and reference direction k, adj the adjugate of the Jacobian
// dx/dxi and det its determinant, as the generated apply block.

#include "sfem_base.hpp"

#if defined(SFEM_ENABLE_HIP) || defined(__CUDACC__)

#include <cstddef>
#include <cstdlib>

namespace sfem {
    namespace tensor_product {

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
            const int    qx = q % NQ1, qy = (q / NQ1) % NQ1, qz = q / (NQ1 * NQ1);
            const int    sx = s % NS1, sy = (s / NS1) % NS1, sz = s / (NS1 * NS1);
            const double bx = b[qx * NS1 + sx], by = b[qy * NS1 + sy], bz = b[qz * NS1 + sz];
            const double gx = g[qx * NS1 + sx], gy = g[qy * NS1 + sy], gz = g[qz * NS1 + sz];
            G[0]            = gx * by * bz;
            G[1]            = bx * gy * bz;
            G[2]            = bx * by * gz;
        }

        /// The adjugate and determinant of J (row c: dx_c/dxi).
        __device__ __forceinline__ void matrix_core_adjugate(const double J00,
                                                             const double J01,
                                                             const double J02,
                                                             const double J10,
                                                             const double J11,
                                                             const double J12,
                                                             const double J20,
                                                             const double J21,
                                                             const double J22,
                                                             double (&adj)[9],
                                                             double &det) {
            adj[0] = J11 * J22 - J12 * J21;
            adj[1] = J02 * J21 - J01 * J22;
            adj[2] = J01 * J12 - J02 * J11;
            adj[3] = J12 * J20 - J10 * J22;
            adj[4] = J00 * J22 - J02 * J20;
            adj[5] = J02 * J10 - J00 * J12;
            adj[6] = J10 * J21 - J11 * J20;
            adj[7] = J01 * J20 - J00 * J21;
            adj[8] = J00 * J11 - J01 * J10;
            det    = J00 * (J11 * J22 - J12 * J21) - J01 * (J10 * J22 - J12 * J20) + J02 * (J10 * J21 - J11 * J20);
        }

#if defined(SFEM_ENABLE_HIP)
        namespace hip {
            using sfem::tensor_product::MATRIX_CORE_DECLINED;
            using sfem::tensor_product::matrix_core_apply_enabled;
            using sfem::tensor_product::MatrixCoreAffineGeometry;
            using sfem::tensor_product::MatrixCoreParameters;
            using sfem::tensor_product::reference_gradient_entries;

            typedef double matrix_core_double4 __attribute__((ext_vector_type(4)));

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
                // The gradient GEMM's k-steps are unrolled, the nodes held, up to Q2 (7 k-steps); beyond,
                // the unrolled steps' entries of G and the held nodes overflow the registers.
                static constexpr bool ROLL           = KS > 8;
                static constexpr int  FORWARD_UNROLL = ROLL ? 1 : KS;
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

                    // The nodes the lane gathers (node 4 ks + grp), held when the k-steps are unrolled and
                    // read again when they are rolled.
                    idx_t idx[KS];
                    if constexpr (!ROLL) {
#pragma unroll
                        for (int ks = 0; ks < KS; ++ks) {
                            const int s = 4 * ks + grp;
                            idx[ks]     = (valid && s < NS) ? elements[s][e] : idx_t(-1);
                        }
                    }
                    auto node = [&](const int ks) -> idx_t {
                        if constexpr (ROLL) {
                            const int s = 4 * ks + grp;
                            return (valid && s < NS) ? elements[s][e] : idx_t(-1);
                        } else {
                            return idx[ks];
                        }
                    };

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
#pragma unroll(FORWARD_UNROLL)
                        for (int ks = 0; ks < KS; ++ks) {
                            const int s = 4 * ks + grp;
                            double    G[3];
                            entries(qa, s, G);

                            double      v[NF];
                            const idx_t id  = node(ks);
                            const bool  has = id >= 0;
#pragma unroll
                            for (int c = 0; c < NC; ++c) v[c] = has ? h_fields[c][id * h_stride] : 0.0;
                            if constexpr (ISOPARAMETRIC) {
#pragma unroll
                                for (int d = 0; d < 3; ++d) v[F_GEO + d] = has ? (double)points[d][id] : 0.0;
                            }
                            if constexpr (CURRENT) {
#pragma unroll
                                for (int c = 0; c < NC; ++c) v[F_CUR + c] = has ? u_fields[c][id * u_stride] : 0.0;
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
                            if (ks < KS) {
                                const idx_t id = node(ks);
                                if (id >= 0) {
#pragma unroll
                                    for (int c = 0; c < NC; ++c) atomicAdd(&out_fields[c][id * out_stride], out[c][to][i]);
                                }
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
        using hip::matrix_core_apply;
#else
        namespace cuda {
            // NVIDIA's FP64 tensor cores, mma.sync m8n8k4 (sm_80 and later). Fragments (PTX ISA): A(i, k)
            // in lane 4 i + k, B(k, j) in lane 4 j + k, D(i, j) in slot j % 2 of lane 4 i + j / 2. A warp
            // takes 8 elements and computes the transposes, R^T = U^T G^T and Y^T = F^T G: element
            // e = lane / 4 is a row of every fragment, so lane 4 e + t gathers element e's nodes
            // 4 ks + t, holds its points q = 8 h + 2 t + i (slot i) in all three directions, and its
            // flux slot j is the A operand of the second GEMM's k-step j, whose k index t stands for
            // the point 8 h + 2 t + j: the order of a sum is free, so nothing moves between the GEMMs.
            __device__ __forceinline__ void dmma_m8n8k4(const double a, const double b, double &c0, double &c1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
                asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1},{%2},{%3},{%4,%5};\n"
                             : "=d"(c0), "=d"(c1)
                             : "d"(a), "d"(b), "d"(c0), "d"(c1));
#else
                __trap();  // not launched: matrix_core_apply declines below sm_80
#endif
            }

            /// Whether the current device has FP64 tensor cores (compute capability 8.0 and later).
            inline bool device_has_fp64_tensor_cores() {
                static const bool has = [] {
                    int device = 0, major = 0;
                    if (cudaGetDevice(&device) != cudaSuccess) return false;
                    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess) return false;
                    return major >= 8;
                }();
                return has;
            }

            template <class Basis, class Quadrature, int NS1, int NQ1, int NC, bool ISOPARAMETRIC, class P, int N_WARPS, typename g_t>
            __global__ void __launch_bounds__(32 * N_WARPS)
                    matrix_core_apply_kernel(const ptrdiff_t                              nelements,
                                             idx_t **const SFEM_RESTRICT                 elements,
                                             const MatrixCoreAffineGeometry<g_t>         affine,
                                             const g_t *const *const SFEM_RESTRICT       points,
                                             const MatrixCoreParameters<P::n_parameters> parameters,
                                             const ptrdiff_t                             u_stride,
                                             const double *const                         u0,
                                             const double *const                         u1,
                                             const double *const                         u2,
                                             const ptrdiff_t                             h_stride,
                                             const double *const                         h0,
                                             const double *const                         h1,
                                             const double *const                         h2,
                                             const ptrdiff_t                             out_stride,
                                             double *const                               out0,
                                             double *const                               out1,
                                             double *const                               out2) {
                static constexpr int  NS             = NS1 * NS1 * NS1;
                static constexpr int  NQ             = NQ1 * NQ1 * NQ1;
                static constexpr int  KS             = (NS + 3) / 4;  // k-steps of the gradient GEMM
                static constexpr int  QT             = (NQ + 7) / 8;  // tiles of points
                static constexpr int  NT             = (NS + 7) / 8;  // output tiles of nodes
                static constexpr bool CURRENT        = P::reads_current;
                static constexpr int  F_GEO          = NC;
                static constexpr int  F_CUR          = NC + (ISOPARAMETRIC ? 3 : 0);
                static constexpr int  NF             = F_CUR + (CURRENT ? NC : 0);
                static constexpr int  NQP            = 8 * QT;
                static constexpr int  NSP            = 4 * KS;
                static constexpr bool TABLE          = 3 * NQP * NSP * sizeof(double) <= 46 * 1024;
                static constexpr bool ROLL           = KS > 8;
                static constexpr int  FORWARD_UNROLL = ROLL ? 1 : KS;
                static_assert(NC == 1 || NC == 3, "NC must be 1 or 3");

                __shared__ double b[NQ1 * NS1], g[NQ1 * NS1], w[NQ1];
                __shared__ double G_table[TABLE ? 3 * NQP : 1][TABLE ? NSP : 1];
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
                // G[(k, q)][s] for a point q < NQP and a node s (zero in the padding).
                auto entries = [&](const int q, const int s, double(&G)[3]) {
                    if constexpr (TABLE) {
                        if (s < NSP) {
#pragma unroll
                            for (int k = 0; k < 3; ++k) G[k] = G_table[k * NQP + q][s];
                        } else {
                            G[0] = G[1] = G[2] = 0;
                        }
                    } else {
                        reference_gradient_entries<NS1, NQ1>(b, g, q < NQ ? q : 0, s < NS ? s : 0, G);
                        if (q >= NQ || s >= NS) G[0] = G[1] = G[2] = 0;
                    }
                };

                const int lane = threadIdx.x % 32;
                const int row  = lane >> 2;  // the element (A and D rows), the B column
                const int t    = lane & 3;   // the A column and B row (k)

                const double *const h_fields[3]   = {h0, h1, h2};
                const double *const u_fields[3]   = {u0, u1, u2};
                double *const       out_fields[3] = {out0, out1, out2};

                for (ptrdiff_t base = ((ptrdiff_t)blockIdx.x * N_WARPS + threadIdx.x / 32) * 8; base < nelements;
                     base += (ptrdiff_t)gridDim.x * N_WARPS * 8) {
                    const ptrdiff_t e     = base + row;
                    const bool      valid = e < nelements;

                    idx_t idx[KS];
                    if constexpr (!ROLL) {
#pragma unroll
                        for (int ks = 0; ks < KS; ++ks) {
                            const int s = 4 * ks + t;
                            idx[ks]     = (valid && s < NS) ? elements[s][e] : idx_t(-1);
                        }
                    }
                    auto node = [&](const int ks) -> idx_t {
                        if constexpr (ROLL) {
                            const int s = 4 * ks + t;
                            return (valid && s < NS) ? elements[s][e] : idx_t(-1);
                        } else {
                            return idx[ks];
                        }
                    };

                    double adj_e[9] = {0}, det_e = 1;
                    if (!ISOPARAMETRIC && valid) {
#pragma unroll
                        for (int i = 0; i < 9; ++i) adj_e[i] = (double)affine.value[i][e];
                        det_e = (double)affine.value[9][e];
                    }

                    double out[NC][NT][2];
#pragma unroll
                    for (int c = 0; c < NC; ++c)
#pragma unroll
                        for (int nt = 0; nt < NT; ++nt) out[c][nt][0] = out[c][nt][1] = 0;

#pragma unroll 1
                    for (int h = 0; h < QT; ++h) {
                        double acc[NF][3][2];
#pragma unroll
                        for (int f = 0; f < NF; ++f)
#pragma unroll
                            for (int k = 0; k < 3; ++k) acc[f][k][0] = acc[f][k][1] = 0;

                        // R^T: A(element row, node 4 ks + t) gathered, B(node 4 ks + t, point 8 h + row) = G.
                        const int qb = 8 * h + row;
#pragma unroll(FORWARD_UNROLL)
                        for (int ks = 0; ks < KS; ++ks) {
                            double G[3];
                            entries(qb, 4 * ks + t, G);

                            double      v[NF];
                            const idx_t id  = node(ks);
                            const bool  has = id >= 0;
#pragma unroll
                            for (int c = 0; c < NC; ++c) v[c] = has ? h_fields[c][id * h_stride] : 0.0;
                            if constexpr (ISOPARAMETRIC) {
#pragma unroll
                                for (int d = 0; d < 3; ++d) v[F_GEO + d] = has ? (double)points[d][id] : 0.0;
                            }
                            if constexpr (CURRENT) {
#pragma unroll
                                for (int c = 0; c < NC; ++c) v[F_CUR + c] = has ? u_fields[c][id * u_stride] : 0.0;
                            }
#pragma unroll
                            for (int k = 0; k < 3; ++k)
#pragma unroll
                                for (int f = 0; f < NF; ++f) dmma_m8n8k4(v[f], G[k], acc[f][k][0], acc[f][k][1]);
                        }

                        // The flux at the lane's points q = 8 h + 2 t + i, written over the gradients.
#pragma unroll
                        for (int i = 0; i < 2; ++i) {
                            const int    q  = 8 * h + 2 * t + i;
                            const int    qq = q < NQ ? q : 0;
                            const double qw =
                                    (valid && q < NQ) ? w[qq % NQ1] * w[(qq / NQ1) % NQ1] * w[qq / (NQ1 * NQ1)] : 0.0;

                            double adj[9], det;
                            if constexpr (ISOPARAMETRIC) {
                                matrix_core_adjugate(acc[F_GEO][0][i],
                                                     acc[F_GEO][1][i],
                                                     acc[F_GEO][2][i],
                                                     acc[F_GEO + 1][0][i],
                                                     acc[F_GEO + 1][1][i],
                                                     acc[F_GEO + 1][2][i],
                                                     acc[F_GEO + 2][0][i],
                                                     acc[F_GEO + 2][1][i],
                                                     acc[F_GEO + 2][2][i],
                                                     adj,
                                                     det);
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

                        // Y^T: A(element row, k t) = flux slot j, B(k t, node 8 nt + row) = G at the point 8 h + 2 t + j.
#pragma unroll
                        for (int j = 0; j < 2; ++j) {
                            const int q = 8 * h + 2 * t + j;
#pragma unroll
                            for (int nt = 0; nt < NT; ++nt) {
                                double G[3];
                                entries(q, 8 * nt + row, G);
#pragma unroll
                                for (int k = 0; k < 3; ++k)
#pragma unroll
                                    for (int c = 0; c < NC; ++c) dmma_m8n8k4(acc[c][k][j], G[k], out[c][nt][0], out[c][nt][1]);
                            }
                        }
                    }

                    // D(element row, node 8 nt + 2 t + i).
                    if (valid) {
#pragma unroll
                        for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
                            for (int i = 0; i < 2; ++i) {
                                const int s = 8 * nt + 2 * t + i;
                                if (s < NS) {
                                    const idx_t id = elements[s][e];
#pragma unroll
                                    for (int c = 0; c < NC; ++c) atomicAdd(&out_fields[c][id * out_stride], out[c][nt][i]);
                                }
                            }
                        }
                    }
                }
            }

            /// As the HIP one: launches the apply on `stream`; MATRIX_CORE_DECLINED when turned off or
            /// on a device without FP64 tensor cores.
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
                static constexpr int N_WARPS = 8;
                if (!matrix_core_apply_enabled() || !device_has_fp64_tensor_cores()) return MATRIX_CORE_DECLINED;
                if (nelements <= 0) return SFEM_SUCCESS;
                const ptrdiff_t per_block = 8 * N_WARPS;
                const int       n_blocks  = (int)((nelements + per_block - 1) / per_block);
                const double   *uu[3]     = {u ? u[0] : nullptr, u && NC > 1 ? u[1] : nullptr, u && NC > 2 ? u[2] : nullptr};
                const double   *hh[3]     = {h[0], NC > 1 ? h[1] : nullptr, NC > 2 ? h[2] : nullptr};
                double         *oo[3]     = {out[0], NC > 1 ? out[1] : nullptr, NC > 2 ? out[2] : nullptr};
                matrix_core_apply_kernel<Basis, Quadrature, NS1, NQ1, NC, ISOPARAMETRIC, P, N_WARPS, g_t>
                        <<<n_blocks, 32 * N_WARPS, 0, stream>>>(nelements,
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
        }  // namespace cuda
        using cuda::matrix_core_apply;
#endif
    }  // namespace tensor_product
}  // namespace sfem

#endif  // SFEM_ENABLE_HIP || __CUDACC__
#endif  // SFEM_CUDA_TENSOR_PRODUCT_MATRIX_CORE_APPLY_CUH
