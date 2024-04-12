#include <cassert>
#include <cmath>
// #include <cstdio>
#include <algorithm>
#include <cstddef>

extern "C" {
#include "sfem_base.h"
#include "sfem_vec.h"
#include "sortreduce.h"
}
#include "sfem_cuda_base.h"
#include "sfem_defs.h"
#include "tet10_linear_elasticity_incore_cuda.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define POW2(a) ((a) * (a))

// #define SFEM_ENABLE_FP32_KERNELS

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
    jacobian_t *adjugate,
    jacobian_t *jacobian_determinant) {
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
    jacobian_determinant[0] = jacobian[0] * x0 - jacobian[0] * x1 +
                              jacobian[2] * jacobian[3] * jacobian[7] - jacobian[3] * x2 +
                              jacobian[6] * x3 - jacobian[6] * x4;
}

static inline __device__ __host__ void ref_shape_grad_x(const scalar_t qx,
                                                        const scalar_t qy,
                                                        const scalar_t qz,
                                                        scalar_t *const out) {
    const scalar_t x0 = 4 * qx;
    const scalar_t x1 = 4 * qy;
    const scalar_t x2 = 4 * qz;
    const scalar_t x3 = x1 + x2;
    out[0] = x0 + x3 - 3;
    out[1] = x0 - 1;
    out[2] = 0;
    out[3] = 0;
    out[4] = -8 * qx - x3 + 4;
    out[5] = x1;
    out[6] = -x1;
    out[7] = -x2;
    out[8] = x2;
    out[9] = 0;
}

static inline __device__ __host__ void ref_shape_grad_y(const scalar_t qx,
                                                        const scalar_t qy,
                                                        const scalar_t qz,
                                                        scalar_t *const out) {
    const scalar_t x0 = 4 * qy;
    const scalar_t x1 = 4 * qx;
    const scalar_t x2 = 4 * qz;
    const scalar_t x3 = x1 + x2;
    out[0] = x0 + x3 - 3;
    out[1] = 0;
    out[2] = x0 - 1;
    out[3] = 0;
    out[4] = -x1;
    out[5] = x1;
    out[6] = -8 * qy - x3 + 4;
    out[7] = -x2;
    out[8] = 0;
    out[9] = x2;
}

static inline __device__ __host__ void ref_shape_grad_z(const scalar_t qx,
                                                        const scalar_t qy,
                                                        const scalar_t qz,
                                                        scalar_t *const out) {
    const scalar_t x0 = 4 * qz;
    const scalar_t x1 = 4 * qx;
    const scalar_t x2 = 4 * qy;
    const scalar_t x3 = x1 + x2;
    out[0] = x0 + x3 - 3;
    out[1] = 0;
    out[2] = 0;
    out[3] = x0 - 1;
    out[4] = -x1;
    out[5] = 0;
    out[6] = -x2;
    out[7] = -8 * qz - x3 + 4;
    out[8] = x1;
    out[9] = x2;
}

static inline __device__ __host__ void apply_micro_kernel(
    const scalar_t mu,
    const scalar_t lambda,
    const scalar_t *const SFEM_RESTRICT adjugate,
    const scalar_t jacobian_determinant,
    const scalar_t qx,
    const scalar_t qy,
    const scalar_t qz,
    const scalar_t qw,
    const scalar_t *const SFEM_RESTRICT u,
    scalar_t *const SFEM_RESTRICT element_vector) {
    // This can be reduced with 1D products (ref_shape_grad_{x,y,z})
    scalar_t disp_grad[9] = {0};

#define MICRO_KERNEL_USE_CODEGEN 1

#if MICRO_KERNEL_USE_CODEGEN
    // Code-gen way

    const scalar_t denom = 1;
    {
        const scalar_t x0 = 1.0 / jacobian_determinant;
        const scalar_t x1 = 4 * qx;
        const scalar_t x2 = x1 - 1;
        const scalar_t x3 = 4 * qy;
        const scalar_t x4 = -u[6] * x3;
        const scalar_t x5 = qz - 1;
        const scalar_t x6 = 8 * qx + 4 * qy + 4 * x5;
        const scalar_t x7 = 4 * qz;
        const scalar_t x8 = x1 + x3 + x7 - 3;
        const scalar_t x9 = u[0] * x8;
        const scalar_t x10 = -u[7] * x7 + x9;
        const scalar_t x11 = u[1] * x2 - u[4] * x6 + u[5] * x3 + u[8] * x7 + x10 + x4;
        const scalar_t x12 = x3 - 1;
        const scalar_t x13 = -u[4] * x1;
        const scalar_t x14 = 4 * qx + 8 * qy + 4 * x5;
        const scalar_t x15 = u[2] * x12 + u[5] * x1 - u[6] * x14 + u[9] * x7 + x10 + x13;
        const scalar_t x16 = x7 - 1;
        const scalar_t x17 = 4 * qx + 4 * qy + 8 * qz - 4;
        const scalar_t x18 = u[3] * x16 - u[7] * x17 + u[8] * x1 + u[9] * x3 + x13 + x4 + x9;
        const scalar_t x19 = -u[16] * x3;
        const scalar_t x20 = u[10] * x8;
        const scalar_t x21 = -u[17] * x7 + x20;
        const scalar_t x22 = u[11] * x2 - u[14] * x6 + u[15] * x3 + u[18] * x7 + x19 + x21;
        const scalar_t x23 = -u[14] * x1;
        const scalar_t x24 = u[12] * x12 + u[15] * x1 - u[16] * x14 + u[19] * x7 + x21 + x23;
        const scalar_t x25 = u[13] * x16 - u[17] * x17 + u[18] * x1 + u[19] * x3 + x19 + x20 + x23;
        const scalar_t x26 = -u[26] * x3;
        const scalar_t x27 = u[20] * x8;
        const scalar_t x28 = -u[27] * x7 + x27;
        const scalar_t x29 = u[21] * x2 - u[24] * x6 + u[25] * x3 + u[28] * x7 + x26 + x28;
        const scalar_t x30 = -u[24] * x1;
        const scalar_t x31 = u[22] * x12 + u[25] * x1 - u[26] * x14 + u[29] * x7 + x28 + x30;
        const scalar_t x32 = u[23] * x16 - u[27] * x17 + u[28] * x1 + u[29] * x3 + x26 + x27 + x30;
        disp_grad[0] = x0 * (adjugate[0] * x11 + adjugate[3] * x15 + adjugate[6] * x18);
        disp_grad[1] = x0 * (adjugate[1] * x11 + adjugate[4] * x15 + adjugate[7] * x18);
        disp_grad[2] = x0 * (adjugate[2] * x11 + adjugate[5] * x15 + adjugate[8] * x18);
        disp_grad[3] = x0 * (adjugate[0] * x22 + adjugate[3] * x24 + adjugate[6] * x25);
        disp_grad[4] = x0 * (adjugate[1] * x22 + adjugate[4] * x24 + adjugate[7] * x25);
        disp_grad[5] = x0 * (adjugate[2] * x22 + adjugate[5] * x24 + adjugate[8] * x25);
        disp_grad[6] = x0 * (adjugate[0] * x29 + adjugate[3] * x31 + adjugate[6] * x32);
        disp_grad[7] = x0 * (adjugate[1] * x29 + adjugate[4] * x31 + adjugate[7] * x32);
        disp_grad[8] = x0 * (adjugate[2] * x29 + adjugate[5] * x31 + adjugate[8] * x32);
    }
#else
    // Programmatic way

    const scalar_t denom = jacobian_determinant;
    {
        scalar_t temp[9] = {0};
        scalar_t grad[10];

        ref_shape_grad_x(qx, qy, qz, grad);
#pragma unroll
        for (int i = 0; i < 10; i++) {
            const scalar_t g = grad[i];
            temp[0] += u[i] * g;
            temp[3] += u[10 + i] * g;
            temp[6] += u[20 + i] * g;
        }

        ref_shape_grad_y(qx, qy, qz, grad);
#pragma unroll
        for (int i = 0; i < 10; i++) {
            const scalar_t g = grad[i];
            temp[1] += u[i] * g;
            temp[4] += u[10 + i] * g;
            temp[7] += u[20 + i] * g;
        }

        ref_shape_grad_z(qx, qy, qz, grad);
#pragma unroll
        for (int i = 0; i < 10; i++) {
            const scalar_t g = grad[i];
            temp[2] += u[i] * g;
            temp[5] += u[10 + i] * g;
            temp[8] += u[20 + i] * g;
        }

        for (int i = 0; i < 3; i++) {
#pragma unroll
            for (int j = 0; j < 3; j++) {
#pragma unroll
                for (int k = 0; k < 3; k++) {
                    disp_grad[i * 3 + j] += temp[i * 3 + k] * adjugate[k * 3 + j];
                }
            }
        }
    }

#endif
    // Includes first Piola-Kirchoff stress: P^T * J^-T * det(J)

    scalar_t *P_tXJinv_t = disp_grad;
    {
        const scalar_t x0 = (1.0 / 6.0) * mu;
        const scalar_t x1 = x0 * (disp_grad[1] + disp_grad[3]);
        const scalar_t x2 = x0 * (disp_grad[2] + disp_grad[6]);
        const scalar_t x3 = 2 * mu;
        const scalar_t x4 = lambda * (disp_grad[0] + disp_grad[4] + disp_grad[8]);
        const scalar_t x5 = (1.0 / 6.0) * disp_grad[0] * x3 + (1.0 / 6.0) * x4;
        const scalar_t x6 = x0 * (disp_grad[5] + disp_grad[7]);
        const scalar_t x7 = (1.0 / 6.0) * disp_grad[4] * x3 + (1.0 / 6.0) * x4;
        const scalar_t x8 = (1.0 / 6.0) * disp_grad[8] * x3 + (1.0 / 6.0) * x4;
        P_tXJinv_t[0] = adjugate[0] * x5 + adjugate[1] * x1 + adjugate[2] * x2;
        P_tXJinv_t[1] = adjugate[3] * x5 + adjugate[4] * x1 + adjugate[5] * x2;
        P_tXJinv_t[2] = adjugate[6] * x5 + adjugate[7] * x1 + adjugate[8] * x2;
        P_tXJinv_t[3] = adjugate[0] * x1 + adjugate[1] * x7 + adjugate[2] * x6;
        P_tXJinv_t[4] = adjugate[3] * x1 + adjugate[4] * x7 + adjugate[5] * x6;
        P_tXJinv_t[5] = adjugate[6] * x1 + adjugate[7] * x7 + adjugate[8] * x6;
        P_tXJinv_t[6] = adjugate[0] * x2 + adjugate[1] * x6 + adjugate[2] * x8;
        P_tXJinv_t[7] = adjugate[3] * x2 + adjugate[4] * x6 + adjugate[5] * x8;
        P_tXJinv_t[8] = adjugate[6] * x2 + adjugate[7] * x6 + adjugate[8] * x8;
    }

    // Scale by quadrature weight
    for (int i = 0; i < 9; i++) {
        P_tXJinv_t[i] *= qw / denom;
    }

// On CPU both versions are equivalent
#if MICRO_KERNEL_USE_CODEGEN
    {
        const scalar_t x0 = 4 * qx;
        const scalar_t x1 = 4 * qy;
        const scalar_t x2 = 4 * qz;
        const scalar_t x3 = x0 + x1 + x2 - 3;
        const scalar_t x4 = x0 - 1;
        const scalar_t x5 = x1 - 1;
        const scalar_t x6 = x2 - 1;
        const scalar_t x7 = P_tXJinv_t[1] * x0;
        const scalar_t x8 = P_tXJinv_t[2] * x0;
        const scalar_t x9 = qz - 1;
        const scalar_t x10 = 8 * qx + 4 * qy + 4 * x9;
        const scalar_t x11 = P_tXJinv_t[0] * x1;
        const scalar_t x12 = P_tXJinv_t[2] * x1;
        const scalar_t x13 = 4 * qx + 8 * qy + 4 * x9;
        const scalar_t x14 = P_tXJinv_t[0] * x2;
        const scalar_t x15 = P_tXJinv_t[1] * x2;
        const scalar_t x16 = 4 * qx + 4 * qy + 8 * qz - 4;
        const scalar_t x17 = P_tXJinv_t[4] * x0;
        const scalar_t x18 = P_tXJinv_t[5] * x0;
        const scalar_t x19 = P_tXJinv_t[3] * x1;
        const scalar_t x20 = P_tXJinv_t[5] * x1;
        const scalar_t x21 = P_tXJinv_t[3] * x2;
        const scalar_t x22 = P_tXJinv_t[4] * x2;
        const scalar_t x23 = P_tXJinv_t[7] * x0;
        const scalar_t x24 = P_tXJinv_t[8] * x0;
        const scalar_t x25 = P_tXJinv_t[6] * x1;
        const scalar_t x26 = P_tXJinv_t[8] * x1;
        const scalar_t x27 = P_tXJinv_t[6] * x2;
        const scalar_t x28 = P_tXJinv_t[7] * x2;
        element_vector[0] += x3 * (P_tXJinv_t[0] + P_tXJinv_t[1] + P_tXJinv_t[2]);
        element_vector[1] += P_tXJinv_t[0] * x4;
        element_vector[2] += P_tXJinv_t[1] * x5;
        element_vector[3] += P_tXJinv_t[2] * x6;
        element_vector[4] += -P_tXJinv_t[0] * x10 - x7 - x8;
        element_vector[5] += x11 + x7;
        element_vector[6] += -P_tXJinv_t[1] * x13 - x11 - x12;
        element_vector[7] += -P_tXJinv_t[2] * x16 - x14 - x15;
        element_vector[8] += x14 + x8;
        element_vector[9] += x12 + x15;
        element_vector[10] += x3 * (P_tXJinv_t[3] + P_tXJinv_t[4] + P_tXJinv_t[5]);
        element_vector[11] += P_tXJinv_t[3] * x4;
        element_vector[12] += P_tXJinv_t[4] * x5;
        element_vector[13] += P_tXJinv_t[5] * x6;
        element_vector[14] += -P_tXJinv_t[3] * x10 - x17 - x18;
        element_vector[15] += x17 + x19;
        element_vector[16] += -P_tXJinv_t[4] * x13 - x19 - x20;
        element_vector[17] += -P_tXJinv_t[5] * x16 - x21 - x22;
        element_vector[18] += x18 + x21;
        element_vector[19] += x20 + x22;
        element_vector[20] += x3 * (P_tXJinv_t[6] + P_tXJinv_t[7] + P_tXJinv_t[8]);
        element_vector[21] += P_tXJinv_t[6] * x4;
        element_vector[22] += P_tXJinv_t[7] * x5;
        element_vector[23] += P_tXJinv_t[8] * x6;
        element_vector[24] += -P_tXJinv_t[6] * x10 - x23 - x24;
        element_vector[25] += x23 + x25;
        element_vector[26] += -P_tXJinv_t[7] * x13 - x25 - x26;
        element_vector[27] += -P_tXJinv_t[8] * x16 - x27 - x28;
        element_vector[28] += x24 + x27;
        element_vector[29] += x26 + x28;
    }

#else

    {
        scalar_t grad[10];
        ref_shape_grad_x(qx, qy, qz, grad);

#pragma unroll
        for (int i = 0; i < 10; i++) {
            scalar_t g = grad[i];
            element_vector[i] += P_tXJinv_t[0] * g;
            element_vector[10 + i] += P_tXJinv_t[3] * g;
            element_vector[20 + i] += P_tXJinv_t[6] * g;
        }

        ref_shape_grad_y(qx, qy, qz, grad);

#pragma unroll
        for (int i = 0; i < 10; i++) {
            scalar_t g = grad[i];
            element_vector[i] += P_tXJinv_t[1] * g;
            element_vector[10 + i] += P_tXJinv_t[4] * g;
            element_vector[20 + i] += P_tXJinv_t[7] * g;
        }

        ref_shape_grad_z(qx, qy, qz, grad);

#pragma unroll
        for (int i = 0; i < 10; i++) {
            scalar_t g = grad[i];
            element_vector[i] += P_tXJinv_t[2] * g;
            element_vector[10 + i] += P_tXJinv_t[5] * g;
            element_vector[20 + i] += P_tXJinv_t[8] * g;
        }
    }

#endif

#undef MICRO_KERNEL_USE_CODEGEN
}

static const int n_qp = 8;
static const scalar_t h_qx[8] =
    {0.0, 1.0, 0.0, 0.0, 0.333333333333, 0.333333333333, 0.0, 0.333333333333};

static const scalar_t h_qy[8] =
    {0.0, 0.0, 1.0, 0.0, 0.333333333333, 0.0, 0.333333333333, 0.333333333333};

static const scalar_t h_qz[8] =
    {0.0, 0.0, 0.0, 1.0, 0.0, 0.333333333333, 0.333333333333, 0.333333333333};

static const scalar_t h_qw[8] = {0.025, 0.025, 0.025, 0.025, 0.225, 0.225, 0.225, 0.225};

__constant__ scalar_t qx[8];
__constant__ scalar_t qy[8];
__constant__ scalar_t qz[8];
__constant__ scalar_t qw[8];

static void init_quadrature() {
    static bool initialized = false;
    if (initialized) {
        SFEM_CUDA_CHECK(cudaMemcpyToSymbol(qx, h_qx, 8 * sizeof(scalar_t)));
        SFEM_CUDA_CHECK(cudaMemcpyToSymbol(qy, h_qy, 8 * sizeof(scalar_t)));
        SFEM_CUDA_CHECK(cudaMemcpyToSymbol(qz, h_qz, 8 * sizeof(scalar_t)));
        SFEM_CUDA_CHECK(cudaMemcpyToSymbol(qw, h_qw, 8 * sizeof(scalar_t)));
    }
}

int tet10_cuda_incore_linear_elasticity_init(cuda_incore_linear_elasticity_t *const ctx,
                                             const real_t mu,
                                             const real_t lambda,
                                             const ptrdiff_t nelements,
                                             idx_t **const SFEM_RESTRICT elements,
                                             geom_t **const SFEM_RESTRICT points) {
    {
        init_quadrature();
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
        SFEM_CUDA_CHECK(cudaMalloc(&ctx->elements, 10 * nelements * sizeof(idx_t)));

        for (int d = 0; d < 10; d++) {
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

int tet10_cuda_incore_linear_elasticity_destroy(cuda_incore_linear_elasticity_t *const ctx) {
    cudaFree(ctx->jacobian_adjugate);
    cudaFree(ctx->jacobian_determinant);

    ctx->jacobian_adjugate = 0;
    ctx->jacobian_determinant = 0;

    ctx->elements = 0;
    ctx->nelements = 0;
    ctx->element_type = INVALID;
    return 0;
}

__global__ void tet10_cuda_incore_linear_elasticity_apply_opt_kernel(
    const ptrdiff_t nelements,
    idx_t *const elements,
    const cu_jacobian_t *const g_jacobian_adjugate,
    const cu_jacobian_t *const g_jacobian_determinant,
    const scalar_t mu,
    const scalar_t lambda,
    const real_t *const u,
    real_t *const values) {
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {
        idx_t ev[10];

        // Sub-geometry
        scalar_t adjugate[9];
        scalar_t element_u[30];
        scalar_t element_vector[30] = {0};
        ;

        // Copy over jacobian adjugate
        {
            const cu_jacobian_t *const jacobian_adjugate = &g_jacobian_adjugate[e];
            for (int i = 0; i < 9; i++) {
                adjugate[i] = jacobian_adjugate[i * nelements];
            }
        }

#ifdef SFEM_ENABLE_FP32_KERNELS
        const scalar_t jacobian_determinant = 1;
#else
        const scalar_t jacobian_determinant = g_jacobian_determinant[e];
#endif

#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v * nelements + e];
        }

        for (int v = 0; v < 10; ++v) {
            element_u[v] = u[ev[v] * 3];
            element_u[10 + v] = u[ev[v] * 3 + 1];
            element_u[20 + v] = u[ev[v] * 3 + 2];
        }

        for (int k = 0; k < n_qp; k++) {
            apply_micro_kernel(mu,
                               lambda,
                               adjugate,
                               jacobian_determinant,
                               qx[k],
                               qy[k],
                               qz[k],
                               qw[k],
                               element_u,
                               element_vector);
        }

#ifdef SFEM_ENABLE_FP32_KERNELS
        //
        {
            // real_t use here instead of scalar_t to have division in full precision
            const real_t jacobian_determinant = g_jacobian_determinant[e];

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
#else

        for (int v = 0; v < 10; v++) {
            atomicAdd(&values[ev[v] * 3], element_vector[v]);
        }

        for (int v = 0; v < 10; v++) {
            atomicAdd(&values[ev[v] * 3 + 1], element_vector[10 + v]);
        }

        for (int v = 0; v < 10; v++) {
            atomicAdd(&values[ev[v] * 3 + 2], element_vector[20 + v]);
        }
#endif
    }
}

#define SFEM_USE_OCCUPANCY_MAX_POTENTIAL

int tet10_cuda_incore_linear_elasticity_apply_opt(const cuda_incore_linear_elasticity_t *const ctx,
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
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size,
                                           &block_size,
                                           tet10_cuda_incore_linear_elasticity_apply_opt_kernel,
                                           0,
                                           0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = std::max(ptrdiff_t(1), (ctx->nelements + block_size - 1) / block_size);
    tet10_cuda_incore_linear_elasticity_apply_opt_kernel<<<n_blocks, block_size, 0>>>(
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

int tet10_cuda_incore_linear_elasticity_diag(const cuda_incore_linear_elasticity_t *const ctx,
                                             real_t *const SFEM_RESTRICT diag) {
    //
    assert(0);
    return 1;
}

int tet10_cuda_incore_linear_elasticity_apply_aos(const ptrdiff_t nelements,
                                                  const ptrdiff_t nnodes,
                                                  idx_t **const SFEM_RESTRICT elements,
                                                  geom_t **const SFEM_RESTRICT points,
                                                  const real_t mu,
                                                  const real_t lambda,
                                                  const real_t *const SFEM_RESTRICT u,
                                                  real_t *const SFEM_RESTRICT values) {
    cuda_incore_linear_elasticity_t ctx;
    tet10_cuda_incore_linear_elasticity_init(&ctx, mu, lambda, nelements, elements, points);
    tet10_cuda_incore_linear_elasticity_apply_opt(&ctx, u, values);
    tet10_cuda_incore_linear_elasticity_destroy(&ctx);
    return 0;
}
