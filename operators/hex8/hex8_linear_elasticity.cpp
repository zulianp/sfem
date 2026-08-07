#include "hex8_linear_elasticity.hpp"

#include "hex8_inline_cpu.hpp"
#include "hex8_linear_elasticity_inline_cpu.hpp"
// #include "hex8_quadrature.hpp"
#include "hex8_laplacian_inline_cpu.hpp"
#include "line_quadrature.hpp"

#include "geometry_kernels.hpp"
#include "linear_elasticity/d3/linear_elasticity_d3_tensor_product_local.hpp"

#include <stdio.h>
#include <type_traits>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

namespace {

template <typename scalar_t, typename SrcT, int VECTOR_SIZE>
SFEM_INLINE const scalar_t *hex8_affine_geometry_stream(const int                       nelems,
                                                       const SrcT *const SFEM_RESTRICT source,
                                                       scalar_t *const SFEM_RESTRICT   converted,
                                                       std::true_type) {
    SFEM_UNUSED(nelems);
    SFEM_UNUSED(converted);
    return source;
}

template <typename scalar_t, typename SrcT, int VECTOR_SIZE>
SFEM_INLINE const scalar_t *hex8_affine_geometry_stream(const int                       nelems,
                                                       const SrcT *const SFEM_RESTRICT source,
                                                       scalar_t *const SFEM_RESTRICT   converted,
                                                       std::false_type) {
#pragma omp simd
    for (int lane = 0; lane < nelems; ++lane) {
        converted[lane] = (scalar_t)source[lane];
    }
    return converted;
}

/// VTK HEX8 → proteus/tensor lexicographic node order for sum-factorization.
static SFEM_INLINE void hex8_proteus_element_streams(idx_t **const SFEM_RESTRICT elements,
                                                     idx_t                      *proteus_elements[8]) {
    proteus_elements[0] = elements[0];
    proteus_elements[1] = elements[1];
    proteus_elements[2] = elements[3];
    proteus_elements[3] = elements[2];
    proteus_elements[4] = elements[4];
    proteus_elements[5] = elements[5];
    proteus_elements[6] = elements[7];
    proteus_elements[7] = elements[6];
}

static const scalar_t *hex8_q2_shape_1d() {
    static const scalar_t data[4] = {(scalar_t)0.78867513459481287,
                                     (scalar_t)0.21132486540518708,
                                     (scalar_t)0.21132486540518713,
                                     (scalar_t)0.78867513459481287};
    return data;
}

static const scalar_t *hex8_q2_grad_1d() {
    static const scalar_t data[4] = {(scalar_t)-1, (scalar_t)1, (scalar_t)-1, (scalar_t)1};
    return data;
}

static const scalar_t *hex8_q2_weight_1d() {
    static const scalar_t data[2] = {(scalar_t)0.5, (scalar_t)0.5};
    return data;
}

}  // namespace

static void print_matrix(int r, int c, const accumulator_t *const m) {
    printf("-------------------\n");
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            printf("%g\t", m[i * c + j]);
        }
        printf("\n");
    }
    printf("-------------------\n");
}

int hex8_linear_elasticity_apply(const ptrdiff_t              nelements,
                                 const ptrdiff_t              nnodes,
                                 idx_t **const SFEM_RESTRICT  elements,
                                 geom_t **const SFEM_RESTRICT points,
                                 const real_t                 mu,
                                 const real_t                 lambda,
                                 const ptrdiff_t              u_stride,
                                 const real_t *const          ux,
                                 const real_t *const          uy,
                                 const real_t *const          uz,
                                 const ptrdiff_t              out_stride,
                                 real_t *const                outx,
                                 real_t *const                outy,
                                 real_t *const                outz) {
    SFEM_UNUSED(nnodes);

    int SFEM_HEX8_QUADRATURE_ORDER = 2;
    SFEM_READ_ENV(SFEM_HEX8_QUADRATURE_ORDER, atoi);

    // Higher-order quadrature fallback (legacy triple loop). Q=2 uses sum-factorization below.
    if (SFEM_HEX8_QUADRATURE_ORDER != 1 && SFEM_HEX8_QUADRATURE_ORDER != 2) {
        const geom_t *const x = points[0];
        const geom_t *const y = points[1];
        const geom_t *const z = points[2];

        int             n_qp = line_q3_n;
        const scalar_t *qx   = line_q3_x;
        const scalar_t *qw   = line_q3_w;
        if (SFEM_HEX8_QUADRATURE_ORDER == 5) {
            n_qp = line_q6_n;
            qx   = line_q6_x;
            qw   = line_q6_w;
        }

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t         ev[8];
            scalar_t      element_ux[8];
            scalar_t      element_uy[8];
            scalar_t      element_uz[8];
            scalar_t      lx[8];
            scalar_t      ly[8];
            scalar_t      lz[8];
            accumulator_t element_outx[8];
            accumulator_t element_outy[8];
            accumulator_t element_outz[8];
            scalar_t      jacobian_adjugate[9];
            scalar_t      jacobian_determinant = 0;

            for (int v = 0; v < 8; ++v) {
                ev[v] = elements[v][i];
            }
            for (int v = 0; v < 8; ++v) {
                const ptrdiff_t idx = (ptrdiff_t)ev[v] * u_stride;
                element_ux[v]       = ux[idx];
                element_uy[v]       = uy[idx];
                element_uz[v]       = uz[idx];
            }
            for (int d = 0; d < 8; d++) {
                lx[d]           = x[ev[d]];
                ly[d]           = y[ev[d]];
                lz[d]           = z[ev[d]];
                element_outx[d] = 0;
                element_outy[d] = 0;
                element_outz[d] = 0;
            }

            for (int kz = 0; kz < n_qp; kz++) {
                for (int ky = 0; ky < n_qp; ky++) {
                    for (int kx = 0; kx < n_qp; kx++) {
                        hex8_adjugate_and_det(lx, ly, lz, qx[kx], qx[ky], qx[kz], jacobian_adjugate, &jacobian_determinant);
                        hex8_linear_elasticity_apply_adj(mu,
                                                         lambda,
                                                         jacobian_adjugate,
                                                         jacobian_determinant,
                                                         qx[kx],
                                                         qx[ky],
                                                         qx[kz],
                                                         qw[kx] * qw[ky] * qw[kz],
                                                         element_ux,
                                                         element_uy,
                                                         element_uz,
                                                         element_outx,
                                                         element_outy,
                                                         element_outz);
                    }
                }
            }

            for (int edof_i = 0; edof_i < 8; edof_i++) {
                const ptrdiff_t idx = (ptrdiff_t)ev[edof_i] * out_stride;
#pragma omp atomic update
                outx[idx] += element_outx[edof_i];
#pragma omp atomic update
                outy[idx] += element_outy[edof_i];
#pragma omp atomic update
                outz[idx] += element_outz[edof_i];
            }
        }
        return SFEM_SUCCESS;
    }

    // Isoparametric Q=2/S=2 sum-factorization (matches GeneratedLinearElasticity).
    idx_t *proteus_elements[8];
    hex8_proteus_element_streams(elements, proteus_elements);

    static constexpr int N_QP        = 8;
    static constexpr int N_SHAPE     = 8;
    static constexpr int VECTOR_SIZE = 16;
    static constexpr int DIM         = 3;

    const scalar_t *const shape_1d    = hex8_q2_shape_1d();
    const scalar_t *const grad_1d     = hex8_q2_grad_1d();
    const scalar_t *const q_weight_1d = hex8_q2_weight_1d();
    const scalar_t        mu_s        = (scalar_t)mu;
    const scalar_t        lambda_s    = (scalar_t)lambda;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);

        idx_t    ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = proteus_elements[element_node];
#pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }

        const geom_t *const coordinate_components[DIM] = {x, y, z};
        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_coordinate_data[shape * DIM + d][lane] =
                            (scalar_t)coordinate_components[d][ev[shape * VECTOR_SIZE + lane]];
                }
            }
        }

        const real_t *const h_components[DIM] = {ux, uy, uz};
        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_h_data[shape * DIM + d][lane] = (scalar_t)h_components[d][(ptrdiff_t)node * u_stride];
                }
            }
        }

        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
#pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_data[stream][lane] = (scalar_t)0;
            }
        }

        const scalar_t *block_h_streams[N_SHAPE * DIM];
        scalar_t       *block_out_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_h_streams[stream]   = block_h_data[stream];
            block_out_streams[stream] = block_out_data[stream];
        }

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        sfem::codegen::tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, shape_1d, grad_1d, block_coordinate_data, 0, coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        sfem::codegen::tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, shape_1d, grad_1d, block_coordinate_data, 1, coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        sfem::codegen::tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, shape_1d, grad_1d, block_coordinate_data, 2, coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0,
                                                 block_jacobian_adjugate1,
                                                 block_jacobian_adjugate2,
                                                 block_jacobian_adjugate3,
                                                 block_jacobian_adjugate4,
                                                 block_jacobian_adjugate5,
                                                 block_jacobian_adjugate6,
                                                 block_jacobian_adjugate7,
                                                 block_jacobian_adjugate8};
        sfem::codegen::geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, adjugate_streams, block_jacobian_determinant0);

        sfem::codegen::linear_elasticity_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(
                nelems,
                VECTOR_SIZE,
                block_jacobian_adjugate0,
                block_jacobian_adjugate1,
                block_jacobian_adjugate2,
                block_jacobian_adjugate3,
                block_jacobian_adjugate4,
                block_jacobian_adjugate5,
                block_jacobian_adjugate6,
                block_jacobian_adjugate7,
                block_jacobian_adjugate8,
                block_jacobian_determinant0,
                shape_1d,
                grad_1d,
                q_weight_1d,
                lambda_s,
                mu_s,
                block_h_streams,
                block_out_streams);

        real_t *const out_components[DIM] = {outx, outy, outz};
        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                for (int scatter = 0; scatter < nelems; ++scatter) {
#pragma omp atomic update
                    out_components[d][(ptrdiff_t)ev[shape * VECTOR_SIZE + scatter] * out_stride] +=
                            block_out_data[shape * DIM + d][scatter];
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

int affine_hex8_linear_elasticity_apply(const ptrdiff_t              nelements,
                                        const ptrdiff_t              nnodes,
                                        idx_t **const SFEM_RESTRICT  elements,
                                        geom_t **const SFEM_RESTRICT points,
                                        const real_t                 mu,
                                        const real_t                 lambda,
                                        const ptrdiff_t              u_stride,
                                        const real_t *const          ux,
                                        const real_t *const          uy,
                                        const real_t *const          uz,
                                        const ptrdiff_t              out_stride,
                                        real_t *const                outx,
                                        real_t *const                outy,
                                        real_t *const                outz) {
    SFEM_UNUSED(nnodes);

    // Affine: Jac at element center, then same Q=2 sum-factorized apply as the SoA adjugate path.
    idx_t *proteus_elements[8];
    hex8_proteus_element_streams(elements, proteus_elements);

    static constexpr int N_QP        = 8;
    static constexpr int N_SHAPE     = 8;
    static constexpr int VECTOR_SIZE = 16;
    static constexpr int DIM         = 3;

    const scalar_t *const shape_1d    = hex8_q2_shape_1d();
    const scalar_t *const grad_1d     = hex8_q2_grad_1d();
    const scalar_t *const q_weight_1d = hex8_q2_weight_1d();
    const scalar_t        mu_s        = (scalar_t)mu;
    const scalar_t        lambda_s    = (scalar_t)lambda;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);

        idx_t    ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_adj[9][VECTOR_SIZE];
        scalar_t block_det[VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = proteus_elements[element_node];
#pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }

        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t e = evbegin + lane;
            scalar_t        lx[8], ly[8], lz[8];
            scalar_t        jacobian_adjugate[9];
            scalar_t        jacobian_determinant = 0;
            for (int v = 0; v < 8; ++v) {
                const idx_t node = elements[v][e];
                lx[v]            = (scalar_t)x[node];
                ly[v]            = (scalar_t)y[node];
                lz[v]            = (scalar_t)z[node];
            }
            hex8_adjugate_and_det(lx, ly, lz, 0.5, 0.5, 0.5, jacobian_adjugate, &jacobian_determinant);
            block_det[lane] = jacobian_determinant;
            for (int d = 0; d < 9; ++d) {
                block_adj[d][lane] = jacobian_adjugate[d];
            }
        }

        const real_t *const h_components[DIM] = {ux, uy, uz};
        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_h_data[shape * DIM + d][lane] = (scalar_t)h_components[d][(ptrdiff_t)node * u_stride];
                }
            }
        }

        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
#pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_data[stream][lane] = (scalar_t)0;
            }
        }

        const scalar_t *block_h_streams[N_SHAPE * DIM];
        scalar_t       *block_out_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_h_streams[stream]   = block_h_data[stream];
            block_out_streams[stream] = block_out_data[stream];
        }

        sfem::codegen::linear_elasticity_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(
                nelems,
                0,
                block_adj[0],
                block_adj[1],
                block_adj[2],
                block_adj[3],
                block_adj[4],
                block_adj[5],
                block_adj[6],
                block_adj[7],
                block_adj[8],
                block_det,
                shape_1d,
                grad_1d,
                q_weight_1d,
                lambda_s,
                mu_s,
                block_h_streams,
                block_out_streams);

        real_t *const out_components[DIM] = {outx, outy, outz};
        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                for (int scatter = 0; scatter < nelems; ++scatter) {
#pragma omp atomic update
                    out_components[d][(ptrdiff_t)ev[shape * VECTOR_SIZE + scatter] * out_stride] +=
                            block_out_data[shape * DIM + d][scatter];
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

int affine_hex8_linear_elasticity_apply_adjugate(const ptrdiff_t                       nelements,
                                                 const ptrdiff_t                       nnodes,
                                                 idx_t **const SFEM_RESTRICT           elements,
                                                 const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate,
                                                 const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant,
                                                 const real_t                          mu,
                                                 const real_t                          lambda,
                                                 const ptrdiff_t                       u_stride,
                                                 const real_t *const                   ux,
                                                 const real_t *const                   uy,
                                                 const real_t *const                   uz,
                                                 const ptrdiff_t                       out_stride,
                                                 real_t *const                         outx,
                                                 real_t *const                         outy,
                                                 real_t *const                         outz) {
    SFEM_UNUSED(nnodes);

    const Hex8AffineQ2GradTable *const table = hex8_affine_q2_grad_table();
    constexpr int                      VS    = 8;
    const scalar_t                     mu_s  = (scalar_t)mu;
    const scalar_t                     lambda_s = (scalar_t)lambda;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e0 = 0; e0 < nelements; e0 += VS) {
        const int ne = (int)MIN((ptrdiff_t)VS, nelements - e0);

        idx_t         ev[8][VS];
        scalar_t      element_ux[8][VS];
        scalar_t      element_uy[8][VS];
        scalar_t      element_uz[8][VS];
        scalar_t      adj[9][VS];
        scalar_t      det[VS];
        accumulator_t element_outx[8][VS];
        accumulator_t element_outy[8][VS];
        accumulator_t element_outz[8][VS];

        for (int L = 0; L < ne; L++) {
            const ptrdiff_t e = e0 + L;
            det[L]            = (scalar_t)g_jacobian_determinant[e];
#pragma unroll
            for (int d = 0; d < 9; d++) {
                adj[d][L] = (scalar_t)g_jacobian_adjugate[e * 9 + d];
            }
#pragma unroll
            for (int v = 0; v < 8; v++) {
                const idx_t     node = elements[v][e];
                const ptrdiff_t idx  = (ptrdiff_t)node * u_stride;
                ev[v][L]             = node;
                element_ux[v][L]     = ux[idx];
                element_uy[v][L]     = uy[idx];
                element_uz[v][L]     = uz[idx];
                element_outx[v][L]   = 0;
                element_outy[v][L]   = 0;
                element_outz[v][L]   = 0;
            }
        }

        for (int qp = 0; qp < HEX8_AFFINE_Q2_NQP; qp++) {
            const scalar_t *const SFEM_RESTRICT gx = table->gx[qp];
            const scalar_t *const SFEM_RESTRICT gy = table->gy[qp];
            const scalar_t *const SFEM_RESTRICT gz = table->gz[qp];
            const scalar_t                      qw = table->qw[qp];

            scalar_t temp[9][VS];
#pragma omp simd
            for (int L = 0; L < ne; L++) {
                temp[0][L] = 0;
                temp[1][L] = 0;
                temp[2][L] = 0;
                temp[3][L] = 0;
                temp[4][L] = 0;
                temp[5][L] = 0;
                temp[6][L] = 0;
                temp[7][L] = 0;
                temp[8][L] = 0;
            }

#pragma unroll
            for (int i = 0; i < 8; i++) {
                const scalar_t gxi = gx[i];
                const scalar_t gyi = gy[i];
                const scalar_t gzi = gz[i];
#pragma omp simd
                for (int L = 0; L < ne; L++) {
                    const scalar_t uxi = element_ux[i][L];
                    const scalar_t uyi = element_uy[i][L];
                    const scalar_t uzi = element_uz[i][L];
                    temp[0][L] += uxi * gxi;
                    temp[1][L] += uxi * gyi;
                    temp[2][L] += uxi * gzi;
                    temp[3][L] += uyi * gxi;
                    temp[4][L] += uyi * gyi;
                    temp[5][L] += uyi * gzi;
                    temp[6][L] += uzi * gxi;
                    temp[7][L] += uzi * gyi;
                    temp[8][L] += uzi * gzi;
                }
            }

            scalar_t disp[9][VS];
#pragma unroll
            for (int i = 0; i < 3; i++) {
#pragma unroll
                for (int j = 0; j < 3; j++) {
#pragma omp simd
                    for (int L = 0; L < ne; L++) {
                        disp[i * 3 + j][L] = temp[i * 3 + 0][L] * adj[0 * 3 + j][L] +
                                             temp[i * 3 + 1][L] * adj[1 * 3 + j][L] +
                                             temp[i * 3 + 2][L] * adj[2 * 3 + j][L];
                    }
                }
            }

            scalar_t P[9][VS];
#pragma omp simd
            for (int L = 0; L < ne; L++) {
                const scalar_t x0    = mu_s * (disp[1][L] + disp[3][L]);
                const scalar_t x1    = mu_s * (disp[2][L] + disp[6][L]);
                const scalar_t x2    = 2 * mu_s;
                const scalar_t x3    = lambda_s * (disp[0][L] + disp[4][L] + disp[8][L]);
                const scalar_t x4    = disp[0][L] * x2 + x3;
                const scalar_t x5    = mu_s * (disp[5][L] + disp[7][L]);
                const scalar_t x6    = disp[4][L] * x2 + x3;
                const scalar_t x7    = disp[8][L] * x2 + x3;
                const scalar_t scale = qw / det[L];
                P[0][L]              = (adj[0][L] * x4 + adj[1][L] * x0 + adj[2][L] * x1) * scale;
                P[1][L]              = (adj[3][L] * x4 + adj[4][L] * x0 + adj[5][L] * x1) * scale;
                P[2][L]              = (adj[6][L] * x4 + adj[7][L] * x0 + adj[8][L] * x1) * scale;
                P[3][L]              = (adj[0][L] * x0 + adj[1][L] * x6 + adj[2][L] * x5) * scale;
                P[4][L]              = (adj[3][L] * x0 + adj[4][L] * x6 + adj[5][L] * x5) * scale;
                P[5][L]              = (adj[6][L] * x0 + adj[7][L] * x6 + adj[8][L] * x5) * scale;
                P[6][L]              = (adj[0][L] * x1 + adj[1][L] * x5 + adj[2][L] * x7) * scale;
                P[7][L]              = (adj[3][L] * x1 + adj[4][L] * x5 + adj[5][L] * x7) * scale;
                P[8][L]              = (adj[6][L] * x1 + adj[7][L] * x5 + adj[8][L] * x7) * scale;
            }

#pragma unroll
            for (int i = 0; i < 8; i++) {
                const scalar_t gxi = gx[i];
                const scalar_t gyi = gy[i];
                const scalar_t gzi = gz[i];
#pragma omp simd
                for (int L = 0; L < ne; L++) {
                    element_outx[i][L] += P[0][L] * gxi + P[1][L] * gyi + P[2][L] * gzi;
                    element_outy[i][L] += P[3][L] * gxi + P[4][L] * gyi + P[5][L] * gzi;
                    element_outz[i][L] += P[6][L] * gxi + P[7][L] * gyi + P[8][L] * gzi;
                }
            }
        }

        for (int L = 0; L < ne; L++) {
#pragma unroll
            for (int v = 0; v < 8; v++) {
                const ptrdiff_t idx = (ptrdiff_t)ev[v][L] * out_stride;
#pragma omp atomic update
                outx[idx] += element_outx[v][L];
#pragma omp atomic update
                outy[idx] += element_outy[v][L];
#pragma omp atomic update
                outz[idx] += element_outz[v][L];
            }
        }
    }

    return SFEM_SUCCESS;
}

int affine_hex8_linear_elasticity_apply_adjugate_soa(const ptrdiff_t                              nelements,
                                                     const ptrdiff_t                              nnodes,
                                                     idx_t **const SFEM_RESTRICT                  elements,
                                                     const jacobian_t *const SFEM_RESTRICT *const adjugate,
                                                     const geom_t *const SFEM_RESTRICT            jacobian_determinant,
                                                     const real_t                                 mu,
                                                     const real_t                                 lambda,
                                                     const ptrdiff_t                              u_stride,
                                                     const real_t *const                          ux,
                                                     const real_t *const                          uy,
                                                     const real_t *const                          uz,
                                                     const ptrdiff_t                              out_stride,
                                                     real_t *const                                outx,
                                                     real_t *const                                outy,
                                                     real_t *const                                outz) {
    SFEM_UNUSED(nnodes);

    // VTK HEX8 → proteus/tensor lexicographic node order used by sum-factorization.
    idx_t *proteus_elements[8];
    hex8_proteus_element_streams(elements, proteus_elements);

    static constexpr int N_QP        = 8;
    static constexpr int N_SHAPE     = 8;
    static constexpr int VECTOR_SIZE = 16;
    static constexpr int DIM         = 3;

    const scalar_t *const shape_1d    = hex8_q2_shape_1d();
    const scalar_t *const grad_1d     = hex8_q2_grad_1d();
    const scalar_t *const q_weight_1d = hex8_q2_weight_1d();
    const scalar_t        mu_s        = (scalar_t)mu;
    const scalar_t        lambda_s    = (scalar_t)lambda;

    const jacobian_t *const SFEM_RESTRICT adj0 = adjugate[0];
    const jacobian_t *const SFEM_RESTRICT adj1 = adjugate[1];
    const jacobian_t *const SFEM_RESTRICT adj2 = adjugate[2];
    const jacobian_t *const SFEM_RESTRICT adj3 = adjugate[3];
    const jacobian_t *const SFEM_RESTRICT adj4 = adjugate[4];
    const jacobian_t *const SFEM_RESTRICT adj5 = adjugate[5];
    const jacobian_t *const SFEM_RESTRICT adj6 = adjugate[6];
    const jacobian_t *const SFEM_RESTRICT adj7 = adjugate[7];
    const jacobian_t *const SFEM_RESTRICT adj8 = adjugate[8];

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);

        idx_t    ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = proteus_elements[element_node];
#pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }

        const real_t *const h_components[DIM] = {ux, uy, uz};
        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_h_data[shape * DIM + d][lane] = (scalar_t)h_components[d][(ptrdiff_t)node * u_stride];
                }
            }
        }

        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
#pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_data[stream][lane] = (scalar_t)0;
            }
        }

        const scalar_t *block_h_streams[N_SHAPE * DIM];
        scalar_t       *block_out_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_h_streams[stream]   = block_h_data[stream];
            block_out_streams[stream] = block_out_data[stream];
        }

        scalar_t block_adj0_data[VECTOR_SIZE];
        scalar_t block_adj1_data[VECTOR_SIZE];
        scalar_t block_adj2_data[VECTOR_SIZE];
        scalar_t block_adj3_data[VECTOR_SIZE];
        scalar_t block_adj4_data[VECTOR_SIZE];
        scalar_t block_adj5_data[VECTOR_SIZE];
        scalar_t block_adj6_data[VECTOR_SIZE];
        scalar_t block_adj7_data[VECTOR_SIZE];
        scalar_t block_adj8_data[VECTOR_SIZE];
        scalar_t block_det_data[VECTOR_SIZE];

        using same_jac = std::is_same<jacobian_t, scalar_t>;
        using same_det = std::is_same<geom_t, scalar_t>;

        const scalar_t *const block_adj0 =
                hex8_affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, adj0 + evbegin, block_adj0_data, same_jac{});
        const scalar_t *const block_adj1 =
                hex8_affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, adj1 + evbegin, block_adj1_data, same_jac{});
        const scalar_t *const block_adj2 =
                hex8_affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, adj2 + evbegin, block_adj2_data, same_jac{});
        const scalar_t *const block_adj3 =
                hex8_affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, adj3 + evbegin, block_adj3_data, same_jac{});
        const scalar_t *const block_adj4 =
                hex8_affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, adj4 + evbegin, block_adj4_data, same_jac{});
        const scalar_t *const block_adj5 =
                hex8_affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, adj5 + evbegin, block_adj5_data, same_jac{});
        const scalar_t *const block_adj6 =
                hex8_affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, adj6 + evbegin, block_adj6_data, same_jac{});
        const scalar_t *const block_adj7 =
                hex8_affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, adj7 + evbegin, block_adj7_data, same_jac{});
        const scalar_t *const block_adj8 =
                hex8_affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, adj8 + evbegin, block_adj8_data, same_jac{});
        const scalar_t *const block_det =
                hex8_affine_geometry_stream<scalar_t, geom_t, VECTOR_SIZE>(
                        nelems, jacobian_determinant + evbegin, block_det_data, same_det{});

        sfem::codegen::linear_elasticity_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(
                nelems,
                0,
                block_adj0,
                block_adj1,
                block_adj2,
                block_adj3,
                block_adj4,
                block_adj5,
                block_adj6,
                block_adj7,
                block_adj8,
                block_det,
                shape_1d,
                grad_1d,
                q_weight_1d,
                lambda_s,
                mu_s,
                block_h_streams,
                block_out_streams);

        real_t *const out_components[DIM] = {outx, outy, outz};
        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                for (int scatter = 0; scatter < nelems; ++scatter) {
#pragma omp atomic update
                    out_components[d][(ptrdiff_t)ev[shape * VECTOR_SIZE + scatter] * out_stride] +=
                            block_out_data[shape * DIM + d][scatter];
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

int affine_hex8_linear_elasticity_bsr(const ptrdiff_t                    nelements,
                                      const ptrdiff_t                    nnodes,
                                      idx_t **const SFEM_RESTRICT        elements,
                                      geom_t **const SFEM_RESTRICT       points,
                                      const real_t                       mu,
                                      const real_t                       lambda,
                                      const count_t *const SFEM_RESTRICT rowptr,
                                      const idx_t *const SFEM_RESTRICT   colidx,
                                      real_t *const SFEM_RESTRICT        values) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    // int SFEM_HEX8_QUADRATURE_ORDER = 2;
    // SFEM_READ_ENV(SFEM_HEX8_QUADRATURE_ORDER, atoi);
    // // printf("SFEM_HEX8_QUADRATURE_ORDER = %d\n", SFEM_HEX8_QUADRATURE_ORDER);

    // int n_qp = line_q3_n;
    // const scalar_t *qx = line_q3_x;
    // const scalar_t *qw = line_q3_w;
    // if (SFEM_HEX8_QUADRATURE_ORDER == 1) {
    //     n_qp = line_q2_n;
    //     qx = line_q2_x;
    //     qw = line_q2_w;
    // } else if (SFEM_HEX8_QUADRATURE_ORDER == 5) {
    //     n_qp = line_q6_n;
    //     qx = line_q6_x;
    //     qw = line_q6_w;
    // }

#pragma omp parallel
    {
        scalar_t element_matrix[(3 * 8) * (3 * 8)];
#pragma omp for
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[8];

            scalar_t lx[8];
            scalar_t ly[8];
            scalar_t lz[8];

            for (int v = 0; v < 8; ++v) {
                ev[v] = elements[v][i];
            }

            for (int d = 0; d < 8; d++) {
                lx[d] = x[ev[d]];
                ly[d] = y[ev[d]];
                lz[d] = z[ev[d]];
            }

            scalar_t jacobian_adjugate[9];
            scalar_t jacobian_determinant;
            hex8_adjugate_and_det(lx, ly, lz, 0.5, 0.5, 0.5, jacobian_adjugate, &jacobian_determinant);

            hex8_linear_elasticity_matrix(mu, lambda, jacobian_adjugate, jacobian_determinant, element_matrix);

            hex8_local_to_global_bsr3(ev, element_matrix, rowptr, colidx, values);
        }
    }

    return SFEM_SUCCESS;
}

// The input CRS is created only for the upper-triangular part of the matrix
// And the diagonal is stored with for each node with a stride of 6
int affine_hex8_linear_elasticity_crs_sym(const ptrdiff_t                    nelements,
                                          const ptrdiff_t                    nnodes,
                                          idx_t **const SFEM_RESTRICT        elements,
                                          geom_t **const SFEM_RESTRICT       points,
                                          const real_t                       mu,
                                          const real_t                       lambda,
                                          const count_t *const SFEM_RESTRICT rowptr,
                                          const idx_t *const SFEM_RESTRICT   colidx,
                                          const ptrdiff_t block_stride,  // stride of the block matrix to interchange SoA and AoS.
                                          real_t **const SFEM_RESTRICT block_diag,
                                          real_t **const SFEM_RESTRICT block_offdiag) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    int SFEM_HEX8_QUADRATURE_ORDER = 2;
    SFEM_READ_ENV(SFEM_HEX8_QUADRATURE_ORDER, atoi);
    // printf("SFEM_HEX8_QUADRATURE_ORDER = %d\n", SFEM_HEX8_QUADRATURE_ORDER);

    int             n_qp = line_q3_n;
    const scalar_t *qx   = line_q3_x;
    const scalar_t *qw   = line_q3_w;
    if (SFEM_HEX8_QUADRATURE_ORDER == 1) {
        n_qp = line_q2_n;
        qx   = line_q2_x;
        qw   = line_q2_w;
    } else if (SFEM_HEX8_QUADRATURE_ORDER == 5) {
        n_qp = line_q6_n;
        qx   = line_q6_x;
        qw   = line_q6_w;
    }

#pragma omp parallel
    {
#pragma omp for
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[8];

            scalar_t lx[8];
            scalar_t ly[8];
            scalar_t lz[8];

            for (int v = 0; v < 8; ++v) {
                ev[v] = elements[v][i];
            }

            for (int v = 0; v < 8; v++) {
                lx[v] = x[ev[v]];
                ly[v] = y[ev[v]];
                lz[v] = z[ev[v]];
            }

            scalar_t jacobian_adjugate[9];
            scalar_t jacobian_determinant;
            hex8_adjugate_and_det(lx, ly, lz, 0.5, 0.5, 0.5, jacobian_adjugate, &jacobian_determinant);

            // Assemble the diagonal part of the matrix
            for (int edof_i = 0; edof_i < 8; edof_i++) {
                accumulator_t element_matrix[6] = {0, 0, 0, 0, 0, 0};
                // Using Taylor expansion technique for symbolic integration for i,j pair
                // hex8_linear_elasticity_matrix_coord_taylor_sym(mu,
                //                                                lambda,
                //                                                jacobian_adjugate,
                //                                                jacobian_determinant,
                //                                                hex8_g_0[edof_i],
                //                                                hex8_g_0[edof_i],
                //                                                hex8_H_0[edof_i],
                //                                                hex8_H_0[edof_i],
                //                                                hex8_diff3_0,
                //                                                hex8_diff3_0,
                //                                                element_matrix);

                for (int zi = 0; zi < n_qp; zi++) {
                    for (int yi = 0; yi < n_qp; yi++) {
                        for (int xi = 0; xi < n_qp; xi++) {
                            scalar_t test_grad[3];
                            hex8_ref_shape_grad(edof_i, qx[xi], qx[yi], qx[zi], test_grad);
                            linear_elasticity_matrix_sym(mu,
                                                         lambda,
                                                         jacobian_adjugate,
                                                         jacobian_determinant,
                                                         test_grad,
                                                         test_grad,
                                                         qw[xi] * qw[yi] * qw[zi],
                                                         element_matrix);
                        }
                    }
                }

                // printf("(%d) -> (%d):\n", edof_i, ev[edof_i]);
                // print_matrix(1, 6, element_matrix);

                // local to global
                int d_idx = 0;
                for (int d1 = 0; d1 < 3; d1++) {
                    for (int d2 = d1; d2 < 3; d2++, d_idx++) {
                        real_t *values = &block_diag[d_idx][ev[edof_i] * block_stride];
                        assert(element_matrix[d_idx] == element_matrix[d_idx]);
#pragma omp atomic update
                        *values += element_matrix[d_idx];
                    }
                }
            }

            // Assemble the upper-triangular part of the matrix
            for (int edof_i = 0; edof_i < 8; edof_i++) {
                // For each row we find the corresponding entries in the off-diag
                // We select the entries associated with ev[row] < ev[col]
                const int    lenrow = rowptr[ev[edof_i] + 1] - rowptr[ev[edof_i]];
                const idx_t *cols   = &colidx[rowptr[ev[edof_i]]];
                // Find the columns associated with the current row and mask what is not found with
                // -1
                int ks[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
                for (int i = 0; i < lenrow; i++) {
                    for (int k = 0; k < 8; k++) {
                        if (cols[i] == ev[k]) {
                            ks[k] = i;
                            break;
                        }
                    }
                }

                for (int edof_j = 0; edof_j < 8; edof_j++) {
                    if (ev[edof_j] > ev[edof_i]) {
                        assert(ks[edof_j] != -1);

                        accumulator_t element_matrix[6] = {0, 0, 0, 0, 0, 0};
                        // Using Taylor expansion technique for symbolic integration for i,j pair
                        // (2667 * 6)/3953 = 4 X Ops than assemblying the whole element matrix
                        // 6/576 = 1/96 of the buffer memory used to store the local results or 1/50
                        // for symmetric storage for whole element matrix.
                        // Overall smaller code size of the computational kernel.
                        // hex8_linear_elasticity_matrix_coord_taylor_sym(mu,
                        //                                                lambda,
                        //                                                jacobian_adjugate,
                        //                                                jacobian_determinant,
                        //                                                hex8_g_0[edof_i],
                        //                                                hex8_g_0[edof_j],
                        //                                                hex8_H_0[edof_i],
                        //                                                hex8_H_0[edof_j],
                        //                                                hex8_diff3_0,
                        //                                                hex8_diff3_0,
                        //                                                element_matrix);

                        for (int zi = 0; zi < n_qp; zi++) {
                            for (int yi = 0; yi < n_qp; yi++) {
                                for (int xi = 0; xi < n_qp; xi++) {
                                    scalar_t trial_grad[3];
                                    scalar_t test_grad[3];
                                    hex8_ref_shape_grad(edof_i, qx[xi], qx[yi], qx[zi], trial_grad);
                                    hex8_ref_shape_grad(edof_j, qx[xi], qx[yi], qx[zi], test_grad);
                                    linear_elasticity_matrix_sym(mu,
                                                                 lambda,
                                                                 jacobian_adjugate,
                                                                 jacobian_determinant,
                                                                 trial_grad,
                                                                 test_grad,
                                                                 qw[xi] * qw[yi] * qw[zi],
                                                                 element_matrix);
                                }
                            }
                        }

                        // printf("(%d, %d) -> (%d, %d):\n", edof_i, edof_j, ev[edof_i],
                        // ev[edof_j]); print_matrix(1, 6, element_matrix);

                        // local to global
                        int d_idx = 0;
                        for (int d1 = 0; d1 < 3; d1++) {
                            for (int d2 = d1; d2 < 3; d2++, d_idx++) {
                                real_t *values = &block_offdiag[d_idx][(rowptr[ev[edof_i]] + ks[edof_j]) * block_stride];
#pragma omp atomic update
                                *values += element_matrix[d_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

int affine_hex8_linear_elasticity_diag(const ptrdiff_t              nelements,
                                       const ptrdiff_t              nnodes,
                                       idx_t **const SFEM_RESTRICT  elements,
                                       geom_t **const SFEM_RESTRICT points,
                                       const real_t                 mu,
                                       const real_t                 lambda,
                                       const ptrdiff_t              out_stride,
                                       real_t *const                outx,
                                       real_t *const                outy,
                                       real_t *const                outz) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[8];

        scalar_t lx[8];
        scalar_t ly[8];
        scalar_t lz[8];

        accumulator_t element_diag[3 * 8];

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i];
        }

        for (int d = 0; d < 8; d++) {
            lx[d] = x[ev[d]];
            ly[d] = y[ev[d]];
            lz[d] = z[ev[d]];
        }

        hex8_adjugate_and_det(lx, ly, lz, 0.5, 0.5, 0.5, jacobian_adjugate, &jacobian_determinant);

        hex8_linear_elasticity_diag(mu, lambda, jacobian_adjugate, jacobian_determinant, element_diag);

        for (int edof_i = 0; edof_i < 8; edof_i++) {
            const ptrdiff_t idx = ev[edof_i] * out_stride;

#pragma omp atomic update
            outx[idx] += element_diag[0 * 8 + edof_i];

#pragma omp atomic update
            outy[idx] += element_diag[1 * 8 + edof_i];

#pragma omp atomic update
            outz[idx] += element_diag[2 * 8 + edof_i];
        }
    }

    return SFEM_SUCCESS;
}

int affine_hex8_linear_elasticity_block_diag_sym(const ptrdiff_t              nelements,
                                                 const ptrdiff_t              nnodes,
                                                 idx_t **const SFEM_RESTRICT  elements,
                                                 geom_t **const SFEM_RESTRICT points,
                                                 const real_t                 mu,
                                                 const real_t                 lambda,
                                                 const ptrdiff_t              out_stride,
                                                 real_t *const                out0,
                                                 real_t *const                out1,
                                                 real_t *const                out2,
                                                 real_t *const                out3,
                                                 real_t *const                out4,
                                                 real_t *const                out5) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    int SFEM_HEX8_QUADRATURE_ORDER = 2;
    SFEM_READ_ENV(SFEM_HEX8_QUADRATURE_ORDER, atoi);
    // printf("SFEM_HEX8_QUADRATURE_ORDER = %d\n", SFEM_HEX8_QUADRATURE_ORDER);

    int             n_qp = line_q3_n;
    const scalar_t *qx   = line_q3_x;
    const scalar_t *qw   = line_q3_w;

    if (SFEM_HEX8_QUADRATURE_ORDER == 1) {
        n_qp = line_q2_n;
        qx   = line_q2_x;
        qw   = line_q2_w;
    } else if (SFEM_HEX8_QUADRATURE_ORDER == 5) {
        n_qp = line_q6_n;
        qx   = line_q6_x;
        qw   = line_q6_w;
    }

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[8];

        scalar_t lx[8];
        scalar_t ly[8];
        scalar_t lz[8];

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 8; v++) {
            lx[v] = x[ev[v]];
            ly[v] = y[ev[v]];
            lz[v] = z[ev[v]];
        }

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant;
        hex8_adjugate_and_det(lx, ly, lz, 0.5, 0.5, 0.5, jacobian_adjugate, &jacobian_determinant);

        // Assemble the diagonal part of the matrix
        for (int edof_i = 0; edof_i < 8; edof_i++) {
            accumulator_t element_matrix[6] = {0, 0, 0, 0, 0, 0};
            for (int zi = 0; zi < n_qp; zi++) {
                for (int yi = 0; yi < n_qp; yi++) {
                    for (int xi = 0; xi < n_qp; xi++) {
                        scalar_t test_grad[3];
                        hex8_ref_shape_grad(edof_i, qx[xi], qx[yi], qx[zi], test_grad);
                        linear_elasticity_matrix_sym(mu,
                                                     lambda,
                                                     jacobian_adjugate,
                                                     jacobian_determinant,
                                                     test_grad,
                                                     test_grad,
                                                     qw[xi] * qw[yi] * qw[zi],
                                                     element_matrix);
                    }
                }
            }

            const ptrdiff_t v = ev[edof_i];

            // local to global
#pragma omp atomic update
            out0[v * out_stride] += element_matrix[0];
#pragma omp atomic update
            out1[v * out_stride] += element_matrix[1];
#pragma omp atomic update
            out2[v * out_stride] += element_matrix[2];
#pragma omp atomic update
            out3[v * out_stride] += element_matrix[3];
#pragma omp atomic update
            out4[v * out_stride] += element_matrix[4];
#pragma omp atomic update
            out5[v * out_stride] += element_matrix[5];
        }
    }

    return SFEM_SUCCESS;
}

int hex8_linear_elasticity_l2_project_cauchy_stress(const ptrdiff_t              nelements,
                                                    const ptrdiff_t              nnodes,
                                                    idx_t **const SFEM_RESTRICT  elements,
                                                    geom_t **const SFEM_RESTRICT points,
                                                    const real_t                 mu,
                                                    const real_t                 lambda,
                                                    const ptrdiff_t              u_stride,
                                                    const real_t *const          ux,
                                                    const real_t *const          uy,
                                                    const real_t *const          uz,
                                                    const ptrdiff_t              out_stride,
                                                    real_t *const                s00,
                                                    real_t *const                s01,
                                                    real_t *const                s02,
                                                    real_t *const                s11,
                                                    real_t *const                s12,
                                                    real_t *const                s22) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    int SFEM_HEX8_QUADRATURE_ORDER = 2;
    SFEM_READ_ENV(SFEM_HEX8_QUADRATURE_ORDER, atoi);
    // printf("SFEM_HEX8_QUADRATURE_ORDER = %d\n", SFEM_HEX8_QUADRATURE_ORDER);

    int             n_qp = line_q3_n;
    const scalar_t *qx   = line_q3_x;
    const scalar_t *qw   = line_q3_w;

    if (SFEM_HEX8_QUADRATURE_ORDER == 1) {
        n_qp = line_q2_n;
        qx   = line_q2_x;
        qw   = line_q2_w;
    } else if (SFEM_HEX8_QUADRATURE_ORDER == 5) {
        n_qp = line_q6_n;
        qx   = line_q6_x;
        qw   = line_q6_w;
    }

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[8];

        scalar_t lx[8];
        scalar_t ly[8];
        scalar_t lz[8];

        scalar_t lux[8];
        scalar_t luy[8];
        scalar_t luz[8];

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 8; v++) {
            lx[v] = x[ev[v]];
            ly[v] = y[ev[v]];
            lz[v] = z[ev[v]];
        }

        for (int v = 0; v < 8; v++) {
            lux[v] = ux[ev[v] * u_stride];
            luy[v] = uy[ev[v] * u_stride];
            luz[v] = uz[ev[v] * u_stride];
        }

        // scalar_t jacobian_adjugate[9];
        // scalar_t jacobian_determinant;
        // hex8_adjugate_and_det(lx, ly, lz, 0.5, 0.5, 0.5, jacobian_adjugate, &jacobian_determinant);

        accumulator_t projected_stress[6][8];
        for (int k = 0; k < 6; k++) {
            for (int edof_i = 0; edof_i < 8; edof_i++) {
                projected_stress[k][edof_i] = 0;
            }
        }

        for (int zi = 0; zi < n_qp; zi++) {
            for (int yi = 0; yi < n_qp; yi++) {
                for (int xi = 0; xi < n_qp; xi++) {
                    scalar_t jacobian_adjugate[9];
                    scalar_t jacobian_determinant;
                    hex8_adjugate_and_det(lx, ly, lz, qx[xi], qx[yi], qx[zi], jacobian_adjugate, &jacobian_determinant);

                    scalar_t disp_grad[9];
                    hex8_displacement_gradient(
                            jacobian_adjugate, jacobian_determinant, qx[xi], qx[yi], qx[zi], lux, luy, luz, disp_grad);

                    scalar_t cauchy_stress[6];
                    hex8_cauchy_stress(mu, lambda, disp_grad, cauchy_stress);
                    // hex8_strain(jacobian_adjugate, jacobian_determinant, qx[xi], qx[yi], qx[zi], lux, luy, luz, cauchy_stress);

                    for (int k = 0; k < 6; k++) {
                        assert(cauchy_stress[k] == cauchy_stress[k]);
                        hex8_l2_project(jacobian_determinant,
                                        qx[xi],
                                        qx[yi],
                                        qx[zi],
                                        qw[xi] * qw[yi] * qw[zi],
                                        cauchy_stress[k],
                                        projected_stress[k]);
                    }
                }
            }
        }

        // local to global
        for (int edof_i = 0; edof_i < 8; edof_i++) {
            const ptrdiff_t idx = ev[edof_i] * out_stride;

#pragma omp atomic update
            s00[idx] += projected_stress[0][edof_i];
        }

        for (int edof_i = 0; edof_i < 8; edof_i++) {
            const ptrdiff_t idx = ev[edof_i] * out_stride;
#pragma omp atomic update
            s01[idx] += projected_stress[1][edof_i];
        }

        for (int edof_i = 0; edof_i < 8; edof_i++) {
            const ptrdiff_t idx = ev[edof_i] * out_stride;
#pragma omp atomic update
            s02[idx] += projected_stress[2][edof_i];
        }

        for (int edof_i = 0; edof_i < 8; edof_i++) {
            const ptrdiff_t idx = ev[edof_i] * out_stride;
#pragma omp atomic update
            s11[idx] += projected_stress[3][edof_i];
        }

        for (int edof_i = 0; edof_i < 8; edof_i++) {
            const ptrdiff_t idx = ev[edof_i] * out_stride;
#pragma omp atomic update
            s12[idx] += projected_stress[4][edof_i];
        }

        for (int edof_i = 0; edof_i < 8; edof_i++) {
            const ptrdiff_t idx = ev[edof_i] * out_stride;
#pragma omp atomic update
            s22[idx] += projected_stress[5][edof_i];
        }
    }

    return SFEM_SUCCESS;
}

int hex8_linear_elasticity_objective_steps(const ptrdiff_t                   nelements,
                                           const ptrdiff_t                   stride,
                                           const ptrdiff_t                   nnodes,
                                           idx_t **const SFEM_RESTRICT       elements,
                                           geom_t **const SFEM_RESTRICT      points,
                                           const real_t                      mu,
                                           const real_t                      lambda,
                                           const ptrdiff_t                   u_stride,
                                           const real_t *const SFEM_RESTRICT ux,
                                           const real_t *const SFEM_RESTRICT uy,
                                           const real_t *const SFEM_RESTRICT uz,
                                           const ptrdiff_t                   inc_stride,
                                           const real_t *const SFEM_RESTRICT incx,
                                           const real_t *const SFEM_RESTRICT incy,
                                           const real_t *const SFEM_RESTRICT incz,
                                           const int                         nsteps,
                                           const real_t *const               steps,
                                           real_t *const SFEM_RESTRICT       out) {
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    static const int       n_qp = line_q2_n;
    static const scalar_t *qx   = line_q2_x;
    static const scalar_t *qw   = line_q2_w;

#pragma omp parallel
    {
        scalar_t *out_local = (scalar_t *)calloc(nsteps, sizeof(scalar_t));

#pragma omp for
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[8];

            scalar_t lx[8];
            scalar_t ly[8];
            scalar_t lz[8];

            scalar_t edispx[8];
            scalar_t edispy[8];
            scalar_t edispz[8];

            scalar_t eincx[8];
            scalar_t eincy[8];
            scalar_t eincz[8];

            for (int v = 0; v < 8; ++v) {
                ev[v] = elements[v][i * stride];
            }

            for (int d = 0; d < 8; d++) {
                lx[d] = x[ev[d]];
                ly[d] = y[ev[d]];
                lz[d] = z[ev[d]];
            }

            for (int v = 0; v < 8; ++v) {
                const ptrdiff_t idx = ev[v] * u_stride;
                edispx[v]           = ux[idx];
                edispy[v]           = uy[idx];
                edispz[v]           = uz[idx];
            }

            for (int v = 0; v < 8; ++v) {
                const ptrdiff_t idx = ev[v] * inc_stride;
                eincx[v]            = incx[idx];
                eincy[v]            = incy[idx];
                eincz[v]            = incz[idx];
            }

            hex8_linear_elasticity_objective_steps_integral(
                    lx, ly, lz, n_qp, qx, qw, mu, lambda, edispx, edispy, edispz, eincx, eincy, eincz, nsteps, steps, out_local);
        }

        for (int s = 0; s < nsteps; s++) {
#pragma omp atomic update
            out[s] += out_local[s];
        }

        free(out_local);
    }

    for (int s = 0; s < nsteps; s++) {
        if (out[s] != out[s]) {
            out[s] = 1e10;
        }
    }

    return SFEM_SUCCESS;
}
