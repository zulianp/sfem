#include "tet4_linear_elasticity_incore_cuda.h"

#include "sfem_cuda_base.h"
#include "sfem_defs.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define POW2(a) ((a) * (a))

#ifdef SFEM_ENABLE_FP32_KERNELS
typedef float scalar_t;
#else
typedef real_t scalar_t;
#endif

#ifdef SFEM_ENABLE_FP32_KERNELS
typedef float scalar_t;
#else
typedef real_t scalar_t;
#endif

#ifdef SFEM_ENABLE_FP16_JACOBIANS
#include <cuda_fp16.h>
typedef half cu_jacobian_t;
#else
typedef geom_t cu_jacobian_t;
#endif

typedef scalar_t accumulator_t;

static inline __device__ __host__ void adjugate_and_det_micro_kernel(
    const geom_t px0,
    const geom_t px1,
    const geom_t px2,
    const geom_t px3,
    const geom_t py0,
    const geom_t py1,
    const geom_t py2,
    const geom_t py3,
    const geom_t pz0,
    const geom_t pz1,
    const geom_t pz2,
    const geom_t pz3,
    const ptrdiff_t stride,
    cu_jacobian_t *adjugate,
    cu_jacobian_t *jacobian_determinant) {
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
    jacobian_determinant[0 * stride] = jacobian[0] * x0 - jacobian[0] * x1 +
                                       jacobian[2] * jacobian[3] * jacobian[7] - jacobian[3] * x2 +
                                       jacobian[6] * x3 - jacobian[6] * x4;
}

static inline __device__ __host__ void tet4_linear_elasticity_micro_kernel(
    const scalar_t mu,
    const scalar_t lambda,
    const scalar_t *const SFEM_RESTRICT adjugate,
    const scalar_t jacobian_determinant,
    const scalar_t *const SFEM_RESTRICT ux,
    const scalar_t *const SFEM_RESTRICT uy,
    const scalar_t *const SFEM_RESTRICT uz,
    accumulator_t *const SFEM_RESTRICT outx,
    accumulator_t *const SFEM_RESTRICT outy,
    accumulator_t *const SFEM_RESTRICT outz) {
    // Evaluation of displacement gradient
    scalar_t disp_grad[9];
    {
        const scalar_t x0 = (scalar_t)1.0 / jacobian_determinant;
        const scalar_t x1 = adjugate[0] * x0;
        const scalar_t x2 = adjugate[3] * x0;
        const scalar_t x3 = adjugate[6] * x0;
        const scalar_t x4 = -x1 - x2 - x3;
        const scalar_t x5 = adjugate[1] * x0;
        const scalar_t x6 = adjugate[4] * x0;
        const scalar_t x7 = adjugate[7] * x0;
        const scalar_t x8 = -x5 - x6 - x7;
        const scalar_t x9 = adjugate[2] * x0;
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
        const scalar_t x1 = (scalar_t)(1.0 / 12.0) * lambda *
                            (2 * disp_grad[0] + 2 * disp_grad[4] + 2 * disp_grad[8]);
        const scalar_t x2 = (scalar_t)(1.0 / 6.0) * mu;
        const scalar_t x3 = x2 * (disp_grad[1] + disp_grad[3]);
        const scalar_t x4 = x2 * (disp_grad[2] + disp_grad[6]);
        const scalar_t x5 = x2 * (disp_grad[5] + disp_grad[7]);
        P[0] = disp_grad[0] * x0 + x1;
        P[1] = x3;
        P[2] = x4;
        P[3] = x3;
        P[4] = disp_grad[4] * x0 + x1;
        P[5] = x5;
        P[6] = x4;
        P[7] = x5;
        P[8] = disp_grad[8] * x0 + x1;
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

int tet4_cuda_incore_linear_elasticity_init(cuda_incore_linear_elasticity_t *const ctx,
                                            const real_t mu,
                                            const real_t lambda,
                                            const ptrdiff_t nelements,
                                            idx_t **const SFEM_RESTRICT elements,
                                            geom_t **const SFEM_RESTRICT points) {
    {
        cu_jacobian_t *jacobian_adjugate =
            (cu_jacobian_t *)calloc(9 * nelements, sizeof(cu_jacobian_t));
        cu_jacobian_t *jacobian_determinant =
            (cu_jacobian_t *)calloc(nelements, sizeof(cu_jacobian_t));

#pragma omp parallel
        {
#pragma omp for
            for (ptrdiff_t e = 0; e < nelements; e++) {
                adjugate_and_det_micro_kernel(points[0][elements[0][e]],
                                              points[0][elements[1][e]],
                                              points[0][elements[2][e]],
                                              points[0][elements[3][e]],
                                              points[1][elements[0][e]],
                                              points[1][elements[1][e]],
                                              points[1][elements[2][e]],
                                              points[1][elements[3][e]],
                                              points[2][elements[0][e]],
                                              points[2][elements[1][e]],
                                              points[2][elements[2][e]],
                                              points[2][elements[3][e]],
                                              nelements,
                                              &jacobian_adjugate[e],
                                              &jacobian_determinant[e]);
            }
        }

        SFEM_CUDA_CHECK(cudaMalloc(&ctx->jacobian_adjugate, 9 * nelements * sizeof(cu_jacobian_t)));
        SFEM_CUDA_CHECK(cudaMemcpy(ctx->jacobian_adjugate,
                                   jacobian_adjugate,
                                   9 * nelements * sizeof(cu_jacobian_t),
                                   cudaMemcpyHostToDevice));
        free(jacobian_adjugate);

        SFEM_CUDA_CHECK(cudaMalloc(&ctx->jacobian_determinant, nelements * sizeof(cu_jacobian_t)));
        SFEM_CUDA_CHECK(cudaMemcpy(ctx->jacobian_determinant,
                                   jacobian_determinant,
                                   nelements * sizeof(cu_jacobian_t),
                                   cudaMemcpyHostToDevice));
        free(jacobian_determinant);
    }

    {
        // Store elem indices on device
        SFEM_CUDA_CHECK(cudaMalloc(&ctx->elements, 4 * nelements * sizeof(idx_t)));

        for (int d = 0; d < 4; d++) {
            SFEM_CUDA_CHECK(cudaMemcpy(ctx->elements + d * nelements,
                                       elements[d],
                                       nelements * sizeof(idx_t),
                                       cudaMemcpyHostToDevice));
        }
    }

    ctx->mu = mu;
    ctx->lambda = lambda;
    ctx->nelements = nelements;
    ctx->element_type = TET4;

    return 0;
}

int tet4_cuda_incore_linear_elasticity_destroy(cuda_incore_linear_elasticity_t *const ctx) {
    cudaFree(ctx->jacobian_adjugate);
    cudaFree(ctx->jacobian_determinant);

    ctx->jacobian_adjugate = 0;
    ctx->jacobian_determinant = 0;

    ctx->elements = 0;
    ctx->nelements = 0;
    ctx->element_type = INVALID;
    return 0;
}

__global__ void tet4_cuda_incore_linear_elasticity_apply_opt_kernel(
    const ptrdiff_t nelements,
    idx_t *const elements,
    const cu_jacobian_t *const g_jacobian_adjugate,
    const cu_jacobian_t *const g_jacobian_determinant,
    const real_t mu,
    const real_t lambda,
    const real_t *const u,
    real_t *const values) {
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {
        idx_t ev[4];

        // Sub-geometry
        scalar_t adjugate[9];

        scalar_t ux[4];
        scalar_t uy[4];
        scalar_t uz[4];

        accumulator_t outx[4] = {0};
        accumulator_t outy[4] = {0};
        accumulator_t outz[4] = {0};

        // Copy over jacobian adjugate
        {
            const cu_jacobian_t *const jacobian_adjugate = &g_jacobian_adjugate[e];
            for (int i = 0; i < 9; i++) {
                adjugate[i] = jacobian_adjugate[i * nelements];
            }
        }

        const scalar_t jacobian_determinant = g_jacobian_determinant[e];

#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elements[v * nelements + e];
        }

        for (int v = 0; v < 4; ++v) {
            ux[v] = u[ev[v] * 3];
            uy[v] = u[ev[v] * 3 + 1];
            uz[v] = u[ev[v] * 3 + 2];
        }

        tet4_linear_elasticity_micro_kernel(
            mu, lambda, adjugate, jacobian_determinant, ux, uy, uz, outx, outy, outz);

        for (int v = 0; v < 4; v++) {
            atomicAdd(&values[ev[v] * 3], outx[v]);
            atomicAdd(&values[ev[v] * 3 + 1], outy[v]);
            atomicAdd(&values[ev[v] * 3 + 2], outz[v]);
        }
    }
}

#define SFEM_USE_OCCUPANCY_MAX_POTENTIAL

extern int tet4_cuda_incore_linear_elasticity_apply(
    const cuda_incore_linear_elasticity_t *const ctx,
    const real_t *const SFEM_RESTRICT u,
    real_t *const SFEM_RESTRICT values) {
    const real_t mu = ctx->mu;
    const real_t lambda = ctx->lambda;

    const cu_jacobian_t *const jacobian_adjugate = (cu_jacobian_t *)ctx->jacobian_adjugate;
    const cu_jacobian_t *const jacobian_determinant = (cu_jacobian_t *)ctx->jacobian_determinant;

    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size, &block_size, tet4_cuda_incore_linear_elasticity_apply_opt_kernel, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = std::max(ptrdiff_t(1), (ctx->nelements + block_size - 1) / block_size);
    tet4_cuda_incore_linear_elasticity_apply_opt_kernel<<<n_blocks, block_size, 0>>>(
        ctx->nelements,
        ctx->elements,
        jacobian_adjugate,
        jacobian_determinant,
        mu,
        lambda,
        u,
        values);

    return 0;
}

extern int tet4_cuda_incore_linear_elasticity_diag(const cuda_incore_linear_elasticity_t *const ctx,
                                                   real_t *const SFEM_RESTRICT diag) {
    //
    assert(0);
    return 1;
}

// extern int tet4_cuda_incore_linear_elasticity_apply_aos(const ptrdiff_t nelements,
//                                                         const ptrdiff_t nnodes,
//                                                         idx_t **const SFEM_RESTRICT elements,
//                                                         geom_t **const SFEM_RESTRICT points,
//                                                         const real_t mu,
//                                                         const real_t lambda,
//                                                         const real_t *const SFEM_RESTRICT u,
//                                                         real_t *const SFEM_RESTRICT values) {
//     cuda_incore_linear_elasticity_t ctx;
//     tet4_cuda_incore_linear_elasticity_init(&ctx, mu, lambda, nelements, elements, points);
//     tet4_cuda_incore_linear_elasticity_apply(&ctx, u, values);
//     tet4_cuda_incore_linear_elasticity_destroy(&ctx);
//     return 0;
// }
