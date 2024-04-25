#include <cassert>
#include <cmath>
// #include <cstdio>
#include <algorithm>
#include <cstddef>

#include "sfem_base.h"
#include "sfem_vec.h"
#include "sortreduce.h"

#include "sfem_cuda_base.h"
#include "sfem_defs.h"
#include "tet10_linear_elasticity_incore_cuda.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define POW2(a) ((a) * (a))

// #define SFEM_ENABLE_FP32_KERNELS
// #define SFEM_ENABLE_FP16_JACOBIANS

#ifdef SFEM_ENABLE_FP32_KERNELS
typedef float scalar_t;
#else
typedef real_t scalar_t;
#endif

typedef scalar_t accumulator_t;

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
    accumulator_t *const SFEM_RESTRICT element_vector) {
    // This can be reduced with 1D products (ref_shape_grad_{x,y,z})
    scalar_t disp_grad[9] = {0};

#define MICRO_KERNEL_USE_CODEGEN 0

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

static inline __device__ __host__ void diag_micro_kernel(
    const scalar_t mu,
    const scalar_t lambda,
    const scalar_t *const SFEM_RESTRICT adjugate,
    const scalar_t jacobian_determinant,
    const scalar_t qx,
    const scalar_t qy,
    const scalar_t qz,
    const scalar_t qw,
    accumulator_t *const SFEM_RESTRICT diag) {
    const real_t x0 = lambda + 2 * mu;
    const real_t x1 = adjugate[0] + adjugate[3] + adjugate[6];
    const real_t x2 = x0 * x1;
    const real_t x3 = adjugate[2] + adjugate[5] + adjugate[8];
    const real_t x4 = mu * x3;
    const real_t x5 = adjugate[2] * x4 + adjugate[5] * x4 + adjugate[8] * x4;
    const real_t x6 = adjugate[1] + adjugate[4] + adjugate[7];
    const real_t x7 = mu * x6;
    const real_t x8 = adjugate[1] * x7 + adjugate[4] * x7 + adjugate[7] * x7;
    const real_t x9 = 4 * qx;
    const real_t x10 = 4 * qy;
    const real_t x11 = 4 * qz;
    const real_t x12 = 1.0 / jacobian_determinant;
    const real_t x13 = (1.0 / 6.0) * x12;
    const real_t x14 = x13 * POW2(x10 + x11 + x9 - 3);
    const real_t x15 = POW2(adjugate[1]);
    const real_t x16 = mu * x15;
    const real_t x17 = POW2(adjugate[2]);
    const real_t x18 = mu * x17;
    const real_t x19 = POW2(adjugate[0]);
    const real_t x20 = x13 * POW2(x9 - 1);
    const real_t x21 = POW2(adjugate[4]);
    const real_t x22 = mu * x21;
    const real_t x23 = POW2(adjugate[5]);
    const real_t x24 = mu * x23;
    const real_t x25 = POW2(adjugate[3]);
    const real_t x26 = x13 * POW2(x10 - 1);
    const real_t x27 = POW2(adjugate[7]);
    const real_t x28 = mu * x27;
    const real_t x29 = POW2(adjugate[8]);
    const real_t x30 = mu * x29;
    const real_t x31 = POW2(adjugate[6]);
    const real_t x32 = x13 * POW2(x11 - 1);
    const real_t x33 = adjugate[4] * qx;
    const real_t x34 = adjugate[7] * qx;
    const real_t x35 = qz - 1;
    const real_t x36 = 2 * qx + qy + x35;
    const real_t x37 = adjugate[1] * x36 + x33 + x34;
    const real_t x38 = mu * x37;
    const real_t x39 = adjugate[4] * x38;
    const real_t x40 = adjugate[5] * qx;
    const real_t x41 = adjugate[8] * qx;
    const real_t x42 = adjugate[2] * x36 + x40 + x41;
    const real_t x43 = mu * x42;
    const real_t x44 = adjugate[5] * x43;
    const real_t x45 = adjugate[3] * qx;
    const real_t x46 = adjugate[6] * qx;
    const real_t x47 = adjugate[0] * x36 + x45 + x46;
    const real_t x48 = x0 * x47;
    const real_t x49 = adjugate[7] * x38;
    const real_t x50 = adjugate[8] * x43;
    const real_t x51 = adjugate[1] * x38;
    const real_t x52 = adjugate[2] * x43;
    const real_t x53 = (8.0 / 3.0) * x12;
    const real_t x54 = adjugate[1] * qy;
    const real_t x55 = x33 + x54;
    const real_t x56 = mu * x55;
    const real_t x57 = adjugate[4] * x56;
    const real_t x58 = adjugate[2] * qy;
    const real_t x59 = x40 + x58;
    const real_t x60 = mu * x59;
    const real_t x61 = adjugate[5] * x60;
    const real_t x62 = adjugate[0] * qy;
    const real_t x63 = x45 + x62;
    const real_t x64 = x0 * x63;
    const real_t x65 = adjugate[1] * x56;
    const real_t x66 = adjugate[2] * x60;
    const real_t x67 = adjugate[7] * qy;
    const real_t x68 = qx + 2 * qy + x35;
    const real_t x69 = adjugate[4] * x68 + x54 + x67;
    const real_t x70 = mu * x69;
    const real_t x71 = adjugate[1] * x70;
    const real_t x72 = adjugate[8] * qy;
    const real_t x73 = adjugate[5] * x68 + x58 + x72;
    const real_t x74 = mu * x73;
    const real_t x75 = adjugate[2] * x74;
    const real_t x76 = adjugate[6] * qy;
    const real_t x77 = adjugate[3] * x68 + x62 + x76;
    const real_t x78 = x0 * x77;
    const real_t x79 = adjugate[7] * x70;
    const real_t x80 = adjugate[8] * x74;
    const real_t x81 = adjugate[4] * x70;
    const real_t x82 = adjugate[5] * x74;
    const real_t x83 = adjugate[1] * qz;
    const real_t x84 = adjugate[4] * qz;
    const real_t x85 = qx + qy + 2 * qz - 1;
    const real_t x86 = adjugate[7] * x85 + x83 + x84;
    const real_t x87 = mu * x86;
    const real_t x88 = adjugate[1] * x87;
    const real_t x89 = adjugate[2] * qz;
    const real_t x90 = adjugate[5] * qz;
    const real_t x91 = adjugate[8] * x85 + x89 + x90;
    const real_t x92 = mu * x91;
    const real_t x93 = adjugate[2] * x92;
    const real_t x94 = adjugate[0] * qz;
    const real_t x95 = adjugate[3] * qz;
    const real_t x96 = adjugate[6] * x85 + x94 + x95;
    const real_t x97 = x0 * x96;
    const real_t x98 = adjugate[4] * x87;
    const real_t x99 = adjugate[5] * x92;
    const real_t x100 = adjugate[7] * x87;
    const real_t x101 = adjugate[8] * x92;
    const real_t x102 = x34 + x83;
    const real_t x103 = mu * x102;
    const real_t x104 = adjugate[7] * x103;
    const real_t x105 = x41 + x89;
    const real_t x106 = mu * x105;
    const real_t x107 = adjugate[8] * x106;
    const real_t x108 = x46 + x94;
    const real_t x109 = x0 * x108;
    const real_t x110 = adjugate[1] * x103;
    const real_t x111 = adjugate[2] * x106;
    const real_t x112 = x67 + x84;
    const real_t x113 = mu * x112;
    const real_t x114 = adjugate[7] * x113;
    const real_t x115 = x72 + x90;
    const real_t x116 = mu * x115;
    const real_t x117 = adjugate[8] * x116;
    const real_t x118 = x76 + x95;
    const real_t x119 = x0 * x118;
    const real_t x120 = adjugate[4] * x113;
    const real_t x121 = adjugate[5] * x116;
    const real_t x122 = x0 * x6;
    const real_t x123 = mu * x1;
    const real_t x124 = adjugate[0] * x123 + adjugate[3] * x123 + adjugate[6] * x123;
    const real_t x125 = mu * x19;
    const real_t x126 = mu * x25;
    const real_t x127 = mu * x31;
    const real_t x128 = mu * x47;
    const real_t x129 = adjugate[3] * x128;
    const real_t x130 = x0 * x37;
    const real_t x131 = adjugate[6] * x128;
    const real_t x132 = adjugate[0] * x128;
    const real_t x133 = mu * x63;
    const real_t x134 = adjugate[3] * x133;
    const real_t x135 = x0 * x55;
    const real_t x136 = adjugate[0] * x133;
    const real_t x137 = mu * x77;
    const real_t x138 = adjugate[0] * x137;
    const real_t x139 = x0 * x69;
    const real_t x140 = adjugate[6] * x137;
    const real_t x141 = adjugate[3] * x137;
    const real_t x142 = mu * x96;
    const real_t x143 = adjugate[0] * x142;
    const real_t x144 = x0 * x86;
    const real_t x145 = adjugate[3] * x142;
    const real_t x146 = adjugate[6] * x142;
    const real_t x147 = mu * x108;
    const real_t x148 = adjugate[6] * x147;
    const real_t x149 = x0 * x102;
    const real_t x150 = adjugate[0] * x147;
    const real_t x151 = mu * x118;
    const real_t x152 = adjugate[6] * x151;
    const real_t x153 = x0 * x112;
    const real_t x154 = adjugate[3] * x151;
    const real_t x155 = x0 * x3;
    const real_t x156 = x0 * x42;
    const real_t x157 = x0 * x59;
    const real_t x158 = x0 * x73;
    const real_t x159 = x0 * x91;
    const real_t x160 = x0 * x105;
    const real_t x161 = x0 * x115;
    diag[0] += x14 * (adjugate[0] * x2 + adjugate[3] * x2 + adjugate[6] * x2 + x5 + x8);
    diag[1] += x20 * (x0 * x19 + x16 + x18);
    diag[2] += x26 * (x0 * x25 + x22 + x24);
    diag[3] += x32 * (x0 * x31 + x28 + x30);
    diag[4] +=
        x53 * (qx * (adjugate[3] * x48 + x39 + x44) + qx * (adjugate[6] * x48 + x49 + x50) +
               x36 * (adjugate[0] * x48 + x51 + x52));
    diag[5] +=
        x53 * (qx * (adjugate[3] * x64 + x57 + x61) + qy * (adjugate[0] * x64 + x65 + x66));
    diag[6] +=
        x53 * (qy * (adjugate[0] * x78 + x71 + x75) + qy * (adjugate[6] * x78 + x79 + x80) +
               x68 * (adjugate[3] * x78 + x81 + x82));
    diag[7] +=
        x53 * (qz * (adjugate[0] * x97 + x88 + x93) + qz * (adjugate[3] * x97 + x98 + x99) +
               x85 * (adjugate[6] * x97 + x100 + x101));
    diag[8] +=
        x53 * (qx * (adjugate[6] * x109 + x104 + x107) + qz * (adjugate[0] * x109 + x110 + x111));
    diag[9] +=
        x53 * (qy * (adjugate[6] * x119 + x114 + x117) + qz * (adjugate[3] * x119 + x120 + x121));
    diag[10] +=
        x14 * (adjugate[1] * x122 + adjugate[4] * x122 + adjugate[7] * x122 + x124 + x5);
    diag[11] += x20 * (x0 * x15 + x125 + x18);
    diag[12] += x26 * (x0 * x21 + x126 + x24);
    diag[13] += x32 * (x0 * x27 + x127 + x30);
    diag[14] +=
        x53 * (qx * (adjugate[4] * x130 + x129 + x44) + qx * (adjugate[7] * x130 + x131 + x50) +
               x36 * (adjugate[1] * x130 + x132 + x52));
    diag[15] +=
        x53 * (qx * (adjugate[4] * x135 + x134 + x61) + qy * (adjugate[1] * x135 + x136 + x66));
    diag[16] +=
        x53 * (qy * (adjugate[1] * x139 + x138 + x75) + qy * (adjugate[7] * x139 + x140 + x80) +
               x68 * (adjugate[4] * x139 + x141 + x82));
    diag[17] +=
        x53 * (qz * (adjugate[1] * x144 + x143 + x93) + qz * (adjugate[4] * x144 + x145 + x99) +
               x85 * (adjugate[7] * x144 + x101 + x146));
    diag[18] +=
        x53 * (qx * (adjugate[7] * x149 + x107 + x148) + qz * (adjugate[1] * x149 + x111 + x150));
    diag[19] +=
        x53 * (qy * (adjugate[7] * x153 + x117 + x152) + qz * (adjugate[4] * x153 + x121 + x154));
    diag[20] +=
        x14 * (adjugate[2] * x155 + adjugate[5] * x155 + adjugate[8] * x155 + x124 + x8);
    diag[21] += x20 * (x0 * x17 + x125 + x16);
    diag[22] += x26 * (x0 * x23 + x126 + x22);
    diag[23] += x32 * (x0 * x29 + x127 + x28);
    diag[24] +=
        x53 * (qx * (adjugate[5] * x156 + x129 + x39) + qx * (adjugate[8] * x156 + x131 + x49) +
               x36 * (adjugate[2] * x156 + x132 + x51));
    diag[25] +=
        x53 * (qx * (adjugate[5] * x157 + x134 + x57) + qy * (adjugate[2] * x157 + x136 + x65));
    diag[26] +=
        x53 * (qy * (adjugate[2] * x158 + x138 + x71) + qy * (adjugate[8] * x158 + x140 + x79) +
               x68 * (adjugate[5] * x158 + x141 + x81));
    diag[27] +=
        x53 * (qz * (adjugate[2] * x159 + x143 + x88) + qz * (adjugate[5] * x159 + x145 + x98) +
               x85 * (adjugate[8] * x159 + x100 + x146));
    diag[28] +=
        x53 * (qx * (adjugate[8] * x160 + x104 + x148) + qz * (adjugate[2] * x160 + x110 + x150));
    diag[29] +=
        x53 * (qy * (adjugate[8] * x161 + x114 + x152) + qz * (adjugate[5] * x161 + x120 + x154));
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
    if (!initialized) {
        SFEM_CUDA_CHECK(cudaMemcpyToSymbol(qx, h_qx, 8 * sizeof(scalar_t)));
        SFEM_CUDA_CHECK(cudaMemcpyToSymbol(qy, h_qy, 8 * sizeof(scalar_t)));
        SFEM_CUDA_CHECK(cudaMemcpyToSymbol(qz, h_qz, 8 * sizeof(scalar_t)));
        SFEM_CUDA_CHECK(cudaMemcpyToSymbol(qw, h_qw, 8 * sizeof(scalar_t)));
        initialized = true;
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
    ctx->element_type = TET10;
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

__global__ void tet10_cuda_incore_linear_elasticity_apply_kernel(
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
        accumulator_t element_vector[30] = {0};

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

__global__ void tet10_cuda_incore_linear_elasticity_diag_kernel(
    const ptrdiff_t nelements,
    idx_t *const elements,
    const cu_jacobian_t *const g_jacobian_adjugate,
    const cu_jacobian_t *const g_jacobian_determinant,
    const scalar_t mu,
    const scalar_t lambda,
    real_t *const values) {
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {
        idx_t ev[10];

        // Sub-geometry
        scalar_t adjugate[9];
        accumulator_t element_vector[30] = {0};

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

        for (int k = 0; k < n_qp; k++) {
            diag_micro_kernel(mu,
                              lambda,
                              adjugate,
                              jacobian_determinant,
                              qx[k],
                              qy[k],
                              qz[k],
                              qw[k],
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

extern int tet10_cuda_incore_linear_elasticity_apply(
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
            &min_grid_size, &block_size, tet10_cuda_incore_linear_elasticity_apply_kernel, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = std::max(ptrdiff_t(1), (ctx->nelements + block_size - 1) / block_size);
    tet10_cuda_incore_linear_elasticity_apply_kernel<<<n_blocks, block_size, 0>>>(
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

extern int tet10_cuda_incore_linear_elasticity_diag(
    const cuda_incore_linear_elasticity_t *const ctx,
    real_t *const SFEM_RESTRICT diag) {
    const real_t mu = ctx->mu;
    const real_t lambda = ctx->lambda;

    const cu_jacobian_t *const jacobian_adjugate = (cu_jacobian_t *)ctx->jacobian_adjugate;
    const cu_jacobian_t *const jacobian_determinant = (cu_jacobian_t *)ctx->jacobian_determinant;

    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size, &block_size, tet10_cuda_incore_linear_elasticity_apply_kernel, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = std::max(ptrdiff_t(1), (ctx->nelements + block_size - 1) / block_size);
    tet10_cuda_incore_linear_elasticity_diag_kernel<<<n_blocks, block_size, 0>>>(
        ctx->nelements, ctx->elements, jacobian_adjugate, jacobian_determinant, mu, lambda, diag);

    return 0;
}

extern int tet10_cuda_incore_linear_elasticity_apply_aos(const ptrdiff_t nelements,
                                                         const ptrdiff_t nnodes,
                                                         idx_t **const SFEM_RESTRICT elements,
                                                         geom_t **const SFEM_RESTRICT points,
                                                         const real_t mu,
                                                         const real_t lambda,
                                                         const real_t *const SFEM_RESTRICT u,
                                                         real_t *const SFEM_RESTRICT values) {
    cuda_incore_linear_elasticity_t ctx;
    tet10_cuda_incore_linear_elasticity_init(&ctx, mu, lambda, nelements, elements, points);
    tet10_cuda_incore_linear_elasticity_apply(&ctx, u, values);
    tet10_cuda_incore_linear_elasticity_destroy(&ctx);
    return 0;
}

extern int tet10_cuda_incore_linear_elasticity_diag_aos(const ptrdiff_t nelements,
                                                        const ptrdiff_t nnodes,
                                                        idx_t **const SFEM_RESTRICT elements,
                                                        geom_t **const SFEM_RESTRICT points,
                                                        const real_t mu,
                                                        const real_t lambda,
                                                        real_t *const SFEM_RESTRICT values) {
    cuda_incore_linear_elasticity_t ctx;
    tet10_cuda_incore_linear_elasticity_init(&ctx, mu, lambda, nelements, elements, points);
    tet10_cuda_incore_linear_elasticity_diag(&ctx, values);
    tet10_cuda_incore_linear_elasticity_destroy(&ctx);
    return 0;
}
