#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>

#include "cu_macro_tet4_linear_elasticity.h"
#include "sfem_base.h"
#include "sfem_defs.h"
#include "sfem_vec.h"
#include "sortreduce.h"

#include "sfem_cuda_base.h"

#include "cu_tet4_inline.hpp"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define POW2(a) ((a) * (a))

static inline __device__ __host__ void adjugate_and_det_micro_kernel(const geom_t    px0,
                                                                     const geom_t    px1,
                                                                     const geom_t    px2,
                                                                     const geom_t    px3,
                                                                     const geom_t    py0,
                                                                     const geom_t    py1,
                                                                     const geom_t    py2,
                                                                     const geom_t    py3,
                                                                     const geom_t    pz0,
                                                                     const geom_t    pz1,
                                                                     const geom_t    pz2,
                                                                     const geom_t    pz3,
                                                                     const ptrdiff_t stride,
                                                                     cu_jacobian_t  *adjugate,
                                                                     cu_jacobian_t  *jacobian_determinant) {
    // Compute jacobian in high precision
    real_t jacobian[9];
    jacobian[0] = -px0 + px1;
    jacobian[1] = -px0 + px2;
    jacobian[2] = -px0 + px3;
    jacobian[3] = -py0 + py1;
    jacobian[4] = -py0 + py2;
    jacobian[5] = -py0 + py3;
    jacobian[6] = -pz0 + pz1;
    jacobian[7] = -pz0 + pz2;
    jacobian[8] = -pz0 + pz3;

    const real_t x0 = jacobian[4] * jacobian[8];
    const real_t x1 = jacobian[5] * jacobian[7];
    const real_t x2 = jacobian[1] * jacobian[8];
    const real_t x3 = jacobian[1] * jacobian[5];
    const real_t x4 = jacobian[2] * jacobian[4];

    // Store adjugate in lower precision
    adjugate[0 * stride] = x0 - x1;
    adjugate[1 * stride] = jacobian[2] * jacobian[7] - x2;
    adjugate[2 * stride] = x3 - x4;
    adjugate[3 * stride] = -jacobian[3] * jacobian[8] + jacobian[5] * jacobian[6];
    adjugate[4 * stride] = jacobian[0] * jacobian[8] - jacobian[2] * jacobian[6];
    adjugate[5 * stride] = -jacobian[0] * jacobian[5] + jacobian[2] * jacobian[3];
    adjugate[6 * stride] = jacobian[3] * jacobian[7] - jacobian[4] * jacobian[6];
    adjugate[7 * stride] = -jacobian[0] * jacobian[7] + jacobian[1] * jacobian[6];
    adjugate[8 * stride] = jacobian[0] * jacobian[4] - jacobian[1] * jacobian[3];

    // Store determinant in lower precision
    jacobian_determinant[0] = jacobian[0] * x0 - jacobian[0] * x1 + jacobian[2] * jacobian[3] * jacobian[7] - jacobian[3] * x2 +
                              jacobian[6] * x3 - jacobian[6] * x4;
}

static inline __device__ __host__ void apply_micro_kernel(const scalar_t                      mu,
                                                          const scalar_t                      lambda,
                                                          const scalar_t *const SFEM_RESTRICT adjugate,
                                                          const scalar_t                      jacobian_determinant,
                                                          const scalar_t *const SFEM_RESTRICT ux,
                                                          const scalar_t *const SFEM_RESTRICT uy,
                                                          const scalar_t *const SFEM_RESTRICT uz,
                                                          accumulator_t *const SFEM_RESTRICT  outx,
                                                          accumulator_t *const SFEM_RESTRICT  outy,
                                                          accumulator_t *const SFEM_RESTRICT  outz) {
    scalar_t disp_grad[9];
    {
        const scalar_t x0  = (scalar_t)1.0 / jacobian_determinant;
        const scalar_t x1  = adjugate[0] * x0;
        const scalar_t x2  = adjugate[3] * x0;
        const scalar_t x3  = adjugate[6] * x0;
        const scalar_t x4  = -x1 - x2 - x3;
        const scalar_t x5  = adjugate[1] * x0;
        const scalar_t x6  = adjugate[4] * x0;
        const scalar_t x7  = adjugate[7] * x0;
        const scalar_t x8  = -x5 - x6 - x7;
        const scalar_t x9  = adjugate[2] * x0;
        const scalar_t x10 = adjugate[5] * x0;
        const scalar_t x11 = adjugate[8] * x0;
        const scalar_t x12 = -x10 - x11 - x9;
        // X
        disp_grad[0] = ux[0] * x4 + ux[1] * x1 + ux[2] * x2 + ux[3] * x3;
        disp_grad[1] = ux[0] * x8 + ux[1] * x5 + ux[2] * x6 + ux[3] * x7;
        disp_grad[2] = ux[0] * x12 + ux[1] * x9 + ux[2] * x10 + ux[3] * x11;

        // Y
        disp_grad[3] = uy[0] * x4 + uy[1] * x1 + uy[2] * x2 + uy[3] * x3;
        disp_grad[4] = uy[0] * x8 + uy[1] * x5 + uy[2] * x6 + uy[3] * x7;
        disp_grad[5] = uy[0] * x12 + uy[1] * x9 + uy[2] * x10 + uy[3] * x11;

        // Z
        disp_grad[6] = uz[2] * x2 + uz[3] * x3 + uz[0] * x4 + uz[1] * x1;
        disp_grad[7] = uz[2] * x6 + uz[3] * x7 + uz[0] * x8 + uz[1] * x5;
        disp_grad[8] = uz[2] * x10 + uz[3] * x11 + uz[0] * x12 + uz[1] * x9;
    }

    // We can reuse the buffer to avoid additional register usage
    scalar_t *P = disp_grad;
    {
        const scalar_t x0 = (scalar_t)(1.0 / 3.0) * mu;
        const scalar_t x1 = (scalar_t)(1.0 / 12.0) * lambda * (2 * disp_grad[0] + 2 * disp_grad[4] + 2 * disp_grad[8]);
        const scalar_t x2 = (scalar_t)(1.0 / 6.0) * mu;
        const scalar_t x3 = x2 * (disp_grad[1] + disp_grad[3]);
        const scalar_t x4 = x2 * (disp_grad[2] + disp_grad[6]);
        const scalar_t x5 = x2 * (disp_grad[5] + disp_grad[7]);
        P[0]              = disp_grad[0] * x0 + x1;
        P[1]              = x3;
        P[2]              = x4;
        P[3]              = x3;
        P[4]              = disp_grad[4] * x0 + x1;
        P[5]              = x5;
        P[6]              = x4;
        P[7]              = x5;
        P[8]              = disp_grad[8] * x0 + x1;
    }

    // Bilinear form
    {
        const scalar_t x0 = adjugate[0] + adjugate[3] + adjugate[6];
        const scalar_t x1 = adjugate[1] + adjugate[4] + adjugate[7];
        const scalar_t x2 = adjugate[2] + adjugate[5] + adjugate[8];
        // X
        outx[0] = -P[0] * x0 - P[1] * x1 - P[2] * x2;
        outx[1] = P[0] * adjugate[0] + P[1] * adjugate[1] + P[2] * adjugate[2];
        outx[2] = P[0] * adjugate[3] + P[1] * adjugate[4] + P[2] * adjugate[5];
        outx[3] = P[0] * adjugate[6] + P[1] * adjugate[7] + P[2] * adjugate[8];
        // Y
        outy[0] = -P[3] * x0 - P[4] * x1 - P[5] * x2;
        outy[1] = P[3] * adjugate[0] + P[4] * adjugate[1] + P[5] * adjugate[2];
        outy[2] = P[3] * adjugate[3] + P[4] * adjugate[4] + P[5] * adjugate[5];
        outy[3] = P[3] * adjugate[6] + P[4] * adjugate[7] + P[5] * adjugate[8];
        // Z
        outz[0] = -P[6] * x0 - P[7] * x1 - P[8] * x2;
        outz[1] = P[6] * adjugate[0] + P[7] * adjugate[1] + P[8] * adjugate[2];
        outz[2] = P[6] * adjugate[3] + P[7] * adjugate[4] + P[8] * adjugate[5];
        outz[3] = P[6] * adjugate[6] + P[7] * adjugate[7] + P[8] * adjugate[8];
    }
}

static inline __device__ __host__ void diag_micro_kernel(const scalar_t                      mu,
                                                         const scalar_t                      lambda,
                                                         const scalar_t *const SFEM_RESTRICT adjugate,
                                                         const scalar_t                      jacobian_determinant,
                                                         accumulator_t *const SFEM_RESTRICT  diag) {
    // TODO
}

static inline __device__ void sub_adj_0(const scalar_t *const SFEM_RESTRICT adjugate,
                                        const ptrdiff_t                     stride,
                                        scalar_t *const SFEM_RESTRICT       sub_adjugate) {
    sub_adjugate[0] = 2 * adjugate[0 * stride];
    sub_adjugate[1] = 2 * adjugate[1 * stride];
    sub_adjugate[2] = 2 * adjugate[2 * stride];
    sub_adjugate[3] = 2 * adjugate[3 * stride];
    sub_adjugate[4] = 2 * adjugate[4 * stride];
    sub_adjugate[5] = 2 * adjugate[5 * stride];
    sub_adjugate[6] = 2 * adjugate[6 * stride];
    sub_adjugate[7] = 2 * adjugate[7 * stride];
    sub_adjugate[8] = 2 * adjugate[8 * stride];
}

static inline __device__ void sub_adj_4(const scalar_t *const SFEM_RESTRICT adjugate,
                                        const ptrdiff_t                     stride,
                                        scalar_t *const SFEM_RESTRICT       sub_adjugate) {
    const scalar_t x0 = 2 * adjugate[0 * stride];
    const scalar_t x1 = 2 * adjugate[1 * stride];
    const scalar_t x2 = 2 * adjugate[2 * stride];
    sub_adjugate[0]   = 2 * adjugate[3 * stride] + x0;
    sub_adjugate[1]   = 2 * adjugate[4 * stride] + x1;
    sub_adjugate[2]   = 2 * adjugate[5 * stride] + x2;
    sub_adjugate[3]   = -x0;
    sub_adjugate[4]   = -x1;
    sub_adjugate[5]   = -x2;
    sub_adjugate[6]   = 2 * adjugate[6 * stride];
    sub_adjugate[7]   = 2 * adjugate[7 * stride];
    sub_adjugate[8]   = 2 * adjugate[8 * stride];
}

static inline __device__ void sub_adj_5(const scalar_t *const SFEM_RESTRICT adjugate,
                                        const ptrdiff_t                     stride,
                                        scalar_t *const SFEM_RESTRICT       sub_adjugate) {
    const scalar_t x0 = 2 * adjugate[3 * stride];
    const scalar_t x1 = 2 * adjugate[6 * stride] + x0;
    const scalar_t x2 = 2 * adjugate[4 * stride];
    const scalar_t x3 = 2 * adjugate[7 * stride] + x2;
    const scalar_t x4 = 2 * adjugate[5 * stride];
    const scalar_t x5 = 2 * adjugate[8 * stride] + x4;
    sub_adjugate[0]   = -x1;
    sub_adjugate[1]   = -x3;
    sub_adjugate[2]   = -x5;
    sub_adjugate[3]   = x0;
    sub_adjugate[4]   = x2;
    sub_adjugate[5]   = x4;
    sub_adjugate[6]   = 2 * adjugate[0 * stride] + x1;
    sub_adjugate[7]   = 2 * adjugate[1 * stride] + x3;
    sub_adjugate[8]   = 2 * adjugate[2 * stride] + x5;
}

static inline __device__ void sub_adj_6(const scalar_t *const SFEM_RESTRICT adjugate,
                                        const ptrdiff_t                     stride,
                                        scalar_t *const SFEM_RESTRICT       sub_adjugate) {
    const scalar_t x0 = 2 * adjugate[3 * stride];
    const scalar_t x1 = 2 * adjugate[4 * stride];
    const scalar_t x2 = 2 * adjugate[5 * stride];
    sub_adjugate[0]   = 2 * adjugate[0 * stride] + x0;
    sub_adjugate[1]   = 2 * adjugate[1 * stride] + x1;
    sub_adjugate[2]   = 2 * adjugate[2 * stride] + x2;
    sub_adjugate[3]   = 2 * adjugate[6 * stride] + x0;
    sub_adjugate[4]   = 2 * adjugate[7 * stride] + x1;
    sub_adjugate[5]   = 2 * adjugate[8 * stride] + x2;
    sub_adjugate[6]   = -x0;
    sub_adjugate[7]   = -x1;
    sub_adjugate[8]   = -x2;
}

static inline __device__ void sub_adj_7(const scalar_t *const SFEM_RESTRICT adjugate,
                                        const ptrdiff_t                     stride,
                                        scalar_t *const SFEM_RESTRICT       sub_adjugate) {
    const scalar_t x0 = 2 * adjugate[6 * stride];
    const scalar_t x1 = 2 * adjugate[7 * stride];
    const scalar_t x2 = 2 * adjugate[8 * stride];
    sub_adjugate[0]   = -x0;
    sub_adjugate[1]   = -x1;
    sub_adjugate[2]   = -x2;
    sub_adjugate[3]   = 2 * adjugate[3 * stride] + x0;
    sub_adjugate[4]   = 2 * adjugate[4 * stride] + x1;
    sub_adjugate[5]   = 2 * adjugate[5 * stride] + x2;
    sub_adjugate[6]   = 2 * adjugate[0 * stride];
    sub_adjugate[7]   = 2 * adjugate[1 * stride];
    sub_adjugate[8]   = 2 * adjugate[2 * stride];
}

static inline __device__ void subtet_gather(const int                           i0,
                                            const int                           i1,
                                            const int                           i2,
                                            const int                           i3,
                                            const scalar_t *const SFEM_RESTRICT in,
                                            scalar_t *const SFEM_RESTRICT       out) {
    out[0] = in[i0];
    out[1] = in[i1];
    out[2] = in[i2];
    out[3] = in[i3];
}

static inline __device__ void subtet_scatter_add(const int                                i0,
                                                 const int                                i1,
                                                 const int                                i2,
                                                 const int                                i3,
                                                 const accumulator_t *const SFEM_RESTRICT in,
                                                 accumulator_t *const SFEM_RESTRICT       out) {
    out[i0] += in[0];
    out[i1] += in[1];
    out[i2] += in[2];
    out[i3] += in[3];
}

template <typename T>
__global__ void cu_macro_tet4_linear_elasticity_apply_kernel(const ptrdiff_t                          nelements,
                                                             idx_t **const SFEM_RESTRICT         elements,
                                                             const ptrdiff_t                          jacobian_stride,
                                                             const cu_jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate,
                                                             const cu_jacobian_t *const SFEM_RESTRICT g_jacobian_determinant,
                                                             const real_t                             mu,
                                                             const real_t                             lambda,
                                                             const ptrdiff_t                          u_stride,
                                                             const T *const SFEM_RESTRICT             g_ux,
                                                             const T *const SFEM_RESTRICT             g_uy,
                                                             const T *const SFEM_RESTRICT             g_uz,
                                                             const ptrdiff_t                          out_stride,
                                                             T *const SFEM_RESTRICT                   g_outx,
                                                             T *const SFEM_RESTRICT                   g_outy,
                                                             T *const SFEM_RESTRICT                   g_outz) {
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        idx_t ev[10];

        // Sub-geometry
        scalar_t adjugate[9];
        scalar_t sub_adjugate[9];

        scalar_t ux[10];
        scalar_t uy[10];
        scalar_t uz[10];

        scalar_t sub_ux[4];
        scalar_t sub_uy[4];
        scalar_t sub_uz[4];

        accumulator_t outx[10] = {0};
        accumulator_t outy[10] = {0};
        accumulator_t outz[10] = {0};

        accumulator_t sub_outx[4];
        accumulator_t sub_outy[4];
        accumulator_t sub_outz[4];

        // Copy over jacobian adjugate
        {
            const cu_jacobian_t *const jacobian_adjugate = &g_jacobian_adjugate[e];
            for (int i = 0; i < 9; i++) {
                adjugate[i] = jacobian_adjugate[i * jacobian_stride];
            }
        }

#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v][e];
        }

        for (int v = 0; v < 10; ++v) {
            ux[v] = g_ux[ev[v] * u_stride];
            uy[v] = g_uy[ev[v] * u_stride];
            uz[v] = g_uz[ev[v] * u_stride];
        }

        {  // Corner tests
            sub_adj_0(adjugate, 1, sub_adjugate);

            // 0)
            subtet_gather(0, 4, 6, 7, ux, sub_ux);
            subtet_gather(0, 4, 6, 7, uy, sub_uy);
            subtet_gather(0, 4, 6, 7, uz, sub_uz);

            apply_micro_kernel(mu, lambda, sub_adjugate, (scalar_t)1.0, sub_ux, sub_uy, sub_uz, sub_outx, sub_outy, sub_outz);

            subtet_scatter_add(0, 4, 6, 7, sub_outx, outx);
            subtet_scatter_add(0, 4, 6, 7, sub_outy, outy);
            subtet_scatter_add(0, 4, 6, 7, sub_outz, outz);

            // 1)
            subtet_gather(4, 1, 5, 8, ux, sub_ux);
            subtet_gather(4, 1, 5, 8, uy, sub_uy);
            subtet_gather(4, 1, 5, 8, uz, sub_uz);

            apply_micro_kernel(mu, lambda, sub_adjugate, (scalar_t)1.0, sub_ux, sub_uy, sub_uz, sub_outx, sub_outy, sub_outz);

            subtet_scatter_add(4, 1, 5, 8, sub_outx, outx);
            subtet_scatter_add(4, 1, 5, 8, sub_outy, outy);
            subtet_scatter_add(4, 1, 5, 8, sub_outz, outz);

            // 2)
            subtet_gather(6, 5, 2, 9, ux, sub_ux);
            subtet_gather(6, 5, 2, 9, uy, sub_uy);
            subtet_gather(6, 5, 2, 9, uz, sub_uz);

            apply_micro_kernel(mu, lambda, sub_adjugate, (scalar_t)1.0, sub_ux, sub_uy, sub_uz, sub_outx, sub_outy, sub_outz);

            subtet_scatter_add(6, 5, 2, 9, sub_outx, outx);
            subtet_scatter_add(6, 5, 2, 9, sub_outy, outy);
            subtet_scatter_add(6, 5, 2, 9, sub_outz, outz);

            // 3)
            subtet_gather(7, 8, 9, 3, ux, sub_ux);
            subtet_gather(7, 8, 9, 3, uy, sub_uy);
            subtet_gather(7, 8, 9, 3, uz, sub_uz);

            apply_micro_kernel(mu, lambda, sub_adjugate, (scalar_t)1.0, sub_ux, sub_uy, sub_uz, sub_outx, sub_outy, sub_outz);
            subtet_scatter_add(7, 8, 9, 3, sub_outx, outx);
            subtet_scatter_add(7, 8, 9, 3, sub_outy, outy);
            subtet_scatter_add(7, 8, 9, 3, sub_outz, outz);
        }

        {  // Octahedron tets
            // 4)
            sub_adj_4(adjugate, 1, sub_adjugate);

            subtet_gather(4, 5, 6, 8, ux, sub_ux);
            subtet_gather(4, 5, 6, 8, uy, sub_uy);
            subtet_gather(4, 5, 6, 8, uz, sub_uz);

            apply_micro_kernel(mu, lambda, sub_adjugate, (scalar_t)1.0, sub_ux, sub_uy, sub_uz, sub_outx, sub_outy, sub_outz);

            subtet_scatter_add(4, 5, 6, 8, sub_outx, outx);
            subtet_scatter_add(4, 5, 6, 8, sub_outy, outy);
            subtet_scatter_add(4, 5, 6, 8, sub_outz, outz);

            // 5)
            sub_adj_5(adjugate, 1, sub_adjugate);

            subtet_gather(7, 4, 6, 8, ux, sub_ux);
            subtet_gather(7, 4, 6, 8, uy, sub_uy);
            subtet_gather(7, 4, 6, 8, uz, sub_uz);

            apply_micro_kernel(mu, lambda, sub_adjugate, (scalar_t)1.0, sub_ux, sub_uy, sub_uz, sub_outx, sub_outy, sub_outz);
            subtet_scatter_add(7, 4, 6, 8, sub_outx, outx);
            subtet_scatter_add(7, 4, 6, 8, sub_outy, outy);
            subtet_scatter_add(7, 4, 6, 8, sub_outz, outz);

            // 6)
            sub_adj_6(adjugate, 1, sub_adjugate);

            subtet_gather(6, 5, 9, 8, ux, sub_ux);
            subtet_gather(6, 5, 9, 8, uy, sub_uy);
            subtet_gather(6, 5, 9, 8, uz, sub_uz);

            apply_micro_kernel(mu, lambda, sub_adjugate, (scalar_t)1.0, sub_ux, sub_uy, sub_uz, sub_outx, sub_outy, sub_outz);

            subtet_scatter_add(6, 5, 9, 8, sub_outx, outx);
            subtet_scatter_add(6, 5, 9, 8, sub_outy, outy);
            subtet_scatter_add(6, 5, 9, 8, sub_outz, outz);

            // 7)
            sub_adj_7(adjugate, 1, sub_adjugate);

            subtet_gather(7, 6, 9, 8, ux, sub_ux);
            subtet_gather(7, 6, 9, 8, uy, sub_uy);
            subtet_gather(7, 6, 9, 8, uz, sub_uz);

            apply_micro_kernel(mu, lambda, sub_adjugate, (scalar_t)1.0, sub_ux, sub_uy, sub_uz, sub_outx, sub_outy, sub_outz);

            subtet_scatter_add(7, 6, 9, 8, sub_outx, outx);
            subtet_scatter_add(7, 6, 9, 8, sub_outy, outy);
            subtet_scatter_add(7, 6, 9, 8, sub_outz, outz);
        }

        {
            // real_t use here instead of scalar_t to have division in full precision
            const scalar_t jacobian_determinant = (scalar_t)g_jacobian_determinant[e] * 8;

            for (int v = 0; v < 10; v++) {
                atomicAdd(&g_outx[ev[v] * out_stride], outx[v] / jacobian_determinant);
            }

            for (int v = 0; v < 10; v++) {
                atomicAdd(&g_outy[ev[v] * out_stride], outy[v] / jacobian_determinant);
            }

            for (int v = 0; v < 10; v++) {
                atomicAdd(&g_outz[ev[v] * out_stride], outz[v] / jacobian_determinant);
            }
        }
    }
}

template <typename T>
int cu_macro_tet4_linear_elasticity_apply_tpl(const ptrdiff_t                          nelements,
                                               idx_t **const SFEM_RESTRICT         elements,
                                              const ptrdiff_t                          jacobian_stride,
                                              const cu_jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
                                              const cu_jacobian_t *const SFEM_RESTRICT jacobian_determinant,
                                              const real_t                             mu,
                                              const real_t                             lambda,
                                              const ptrdiff_t                          u_stride,
                                              const T *const SFEM_RESTRICT             ux,
                                              const T *const SFEM_RESTRICT             uy,
                                              const T *const SFEM_RESTRICT             uz,
                                              const ptrdiff_t                          out_stride,
                                              T *const SFEM_RESTRICT                   outx,
                                              T *const SFEM_RESTRICT                   outy,
                                              T *const SFEM_RESTRICT                   outz,
                                              void                                    *stream) {
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, cu_macro_tet4_linear_elasticity_apply_kernel<T>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_macro_tet4_linear_elasticity_apply_kernel<<<n_blocks, block_size, 0, s>>>(nelements,
                                                                                     elements,
                                                                                     jacobian_stride,
                                                                                     jacobian_adjugate,
                                                                                     jacobian_determinant,
                                                                                     mu,
                                                                                     lambda,
                                                                                     u_stride,
                                                                                     ux,
                                                                                     uy,
                                                                                     uz,
                                                                                     out_stride,
                                                                                     outx,
                                                                                     outy,
                                                                                     outz);
    } else {
        cu_macro_tet4_linear_elasticity_apply_kernel<<<n_blocks, block_size, 0>>>(nelements,
                                                                                  elements,
                                                                                  jacobian_stride,
                                                                                  jacobian_adjugate,
                                                                                  jacobian_determinant,
                                                                                  mu,
                                                                                  lambda,
                                                                                  u_stride,
                                                                                  ux,
                                                                                  uy,
                                                                                  uz,
                                                                                  out_stride,
                                                                                  outx,
                                                                                  outy,
                                                                                  outz);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

extern int cu_macro_tet4_linear_elasticity_apply(const ptrdiff_t                 nelements,
                                                 idx_t **const SFEM_RESTRICT     elements,
                                                 const ptrdiff_t                 jacobian_stride,
                                                 const void *const SFEM_RESTRICT jacobian_adjugate,
                                                 const void *const SFEM_RESTRICT jacobian_determinant,
                                                 const real_t                    mu,
                                                 const real_t                    lambda,
                                                 const enum RealType             real_type,
                                                 const ptrdiff_t                 u_stride,
                                                 const void *const SFEM_RESTRICT ux,
                                                 const void *const SFEM_RESTRICT uy,
                                                 const void *const SFEM_RESTRICT uz,
                                                 const ptrdiff_t                 out_stride,
                                                 void *const SFEM_RESTRICT       outx,
                                                 void *const SFEM_RESTRICT       outy,
                                                 void *const SFEM_RESTRICT       outz,
                                                 void                           *stream) {
    switch (real_type) {
        case SFEM_REAL_DEFAULT: {
            return cu_macro_tet4_linear_elasticity_apply_tpl(nelements,
                                                             elements,
                                                             jacobian_stride,
                                                             (cu_jacobian_t *)jacobian_adjugate,
                                                             (cu_jacobian_t *)jacobian_determinant,
                                                             mu,
                                                             lambda,
                                                             u_stride,
                                                             (real_t *)ux,
                                                             (real_t *)uy,
                                                             (real_t *)uz,
                                                             out_stride,
                                                             (real_t *)outx,
                                                             (real_t *)outy,
                                                             (real_t *)outz,
                                                             stream);
        }
        case SFEM_FLOAT32: {
            return cu_macro_tet4_linear_elasticity_apply_tpl(nelements,
                                                             elements,
                                                             jacobian_stride,
                                                             (cu_jacobian_t *)jacobian_adjugate,
                                                             (cu_jacobian_t *)jacobian_determinant,
                                                             mu,
                                                             lambda,
                                                             u_stride,
                                                             (float *)ux,
                                                             (float *)uy,
                                                             (float *)uz,
                                                             out_stride,
                                                             (float *)outx,
                                                             (float *)outy,
                                                             (float *)outz,
                                                             stream);
        }
        case SFEM_FLOAT64: {
            return cu_macro_tet4_linear_elasticity_apply_tpl(nelements,
                                                             elements,
                                                             jacobian_stride,
                                                             (cu_jacobian_t *)jacobian_adjugate,
                                                             (cu_jacobian_t *)jacobian_determinant,
                                                             mu,
                                                             lambda,
                                                             u_stride,
                                                             (double *)ux,
                                                             (double *)uy,
                                                             (double *)uz,
                                                             out_stride,
                                                             (double *)outx,
                                                             (double *)outy,
                                                             (double *)outz,
                                                             stream);
        }
        default: {
            SFEM_ERROR(
                    "[Error] cu_macro_tet4_linear_elasticity_apply: not implemented for type %s "
                    "(code %d)\n",
                    real_type_to_string(real_type),
                    real_type);
            return SFEM_FAILURE;
        }
    }
}

// ----- DIAG

__global__ void cu_macro_tet4_linear_elasticity_diag_kernel(const ptrdiff_t                          nelements,
                                                            idx_t **const SFEM_RESTRICT              elements,
                                                            const ptrdiff_t                          jacobian_stride,
                                                            const cu_jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate,
                                                            const cu_jacobian_t *const SFEM_RESTRICT g_jacobian_determinant,
                                                            const scalar_t                           mu,
                                                            const scalar_t                           lambda,
                                                            real_t *const SFEM_RESTRICT              values) {
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        idx_t ev[10];

        // Sub-geometry
        scalar_t      adjugate[9];
        accumulator_t element_vector[30] = {0};

        // Copy over jacobian adjugate
        {
            const cu_jacobian_t *const jacobian_adjugate = &g_jacobian_adjugate[e];
            for (int i = 0; i < 9; i++) {
                adjugate[i] = jacobian_adjugate[i * jacobian_stride];
            }
        }

#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v][e];
        }

        {  // TODO
            diag_micro_kernel(mu, lambda, adjugate, (scalar_t)1.0, element_vector);
        }

        //
        {
            // real_t use here instead of scalar_t to have division in full precision
            const scalar_t jacobian_determinant = (scalar_t)g_jacobian_determinant[e] * 8;

            for (int v = 0; v < 10; v++) {
                atomicAdd(&values[ev[v] * 3], element_vector[v] / jacobian_determinant);
            }

            for (int v = 0; v < 10; v++) {
                atomicAdd(&values[ev[v] * 3 + 1], element_vector[10 + v] / jacobian_determinant);
            }

            for (int v = 0; v < 10; v++) {
                atomicAdd(&values[ev[v] * 3 + 2], element_vector[20 + v] / jacobian_determinant);
            }
        }
    }
}

extern int cu_macro_tet4_linear_elasticity_diag(const ptrdiff_t                 nelements,
                                                idx_t **const SFEM_RESTRICT     elements,
                                                const ptrdiff_t                 jacobian_stride,
                                                const void *const SFEM_RESTRICT jacobian_adjugate,
                                                const void *const SFEM_RESTRICT jacobian_determinant,
                                                const real_t                    mu,
                                                const real_t                    lambda,
                                                const enum RealType             real_type,
                                                const ptrdiff_t                 diag_stride,
                                                void *const SFEM_RESTRICT       diagx,
                                                void *const SFEM_RESTRICT       diagy,
                                                void *const SFEM_RESTRICT       diagz,
                                                void                           *stream) {
    switch (real_type) {
        default: {
            SFEM_ERROR(
                    "[Error] cu_macro_tet4_linear_elasticity_diag: not implemented for type %s "
                    "(code %d)\n",
                    real_type_to_string(real_type),
                    real_type);
            return SFEM_FAILURE;
        }
    }
}
