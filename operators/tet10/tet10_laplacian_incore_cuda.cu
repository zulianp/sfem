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
#include "tet10_laplacian_incore_cuda.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define POW2(a) ((a) * (a))

// #define SFEM_ENABLE_FP32_KERNELS

#ifdef SFEM_ENABLE_FP32_KERNELS
typedef float scalar_t;
#else
typedef real_t scalar_t;
#endif


#include <cuda_fp16.h>
typedef geom_t cu_jacobian_t;


static inline __device__ __host__ void fff_micro_kernel(const geom_t px0,
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
                                                        const count_t stride,
                                                        cu_jacobian_t *fff) {
    const geom_t x0 = -px0 + px1;
    const geom_t x1 = -py0 + py2;
    const geom_t x2 = -pz0 + pz3;
    const geom_t x3 = x1 * x2;
    const geom_t x4 = x0 * x3;
    const geom_t x5 = -py0 + py3;
    const geom_t x6 = -pz0 + pz2;
    const geom_t x7 = x5 * x6;
    const geom_t x8 = x0 * x7;
    const geom_t x9 = -py0 + py1;
    const geom_t x10 = -px0 + px2;
    const geom_t x11 = x10 * x2;
    const geom_t x12 = x11 * x9;
    const geom_t x13 = -pz0 + pz1;
    const geom_t x14 = x10 * x5;
    const geom_t x15 = x13 * x14;
    const geom_t x16 = -px0 + px3;
    const geom_t x17 = x16 * x6 * x9;
    const geom_t x18 = x1 * x16;
    const geom_t x19 = x13 * x18;
    const geom_t x20 = -1.0 / 6.0 * x12 + (1.0 / 6.0) * x15 + (1.0 / 6.0) * x17 - 1.0 / 6.0 * x19 +
                       (1.0 / 6.0) * x4 - 1.0 / 6.0 * x8;
    const geom_t x21 = x14 - x18;
    const geom_t x22 = 1. / POW2(-x12 + x15 + x17 - x19 + x4 - x8);
    const geom_t x23 = -x11 + x16 * x6;
    const geom_t x24 = x3 - x7;
    const geom_t x25 = -x0 * x5 + x16 * x9;
    const geom_t x26 = x21 * x22;
    const geom_t x27 = x0 * x2 - x13 * x16;
    const geom_t x28 = x22 * x23;
    const geom_t x29 = x13 * x5 - x2 * x9;
    const geom_t x30 = x22 * x24;
    const geom_t x31 = x0 * x1 - x10 * x9;
    const geom_t x32 = -x0 * x6 + x10 * x13;
    const geom_t x33 = -x1 * x13 + x6 * x9;
    fff[0 * stride] = x20 * (POW2(x21) * x22 + x22 * POW2(x23) + x22 * POW2(x24));
    fff[1 * stride] = x20 * (x25 * x26 + x27 * x28 + x29 * x30);
    fff[2 * stride] = x20 * (x26 * x31 + x28 * x32 + x30 * x33);
    fff[3 * stride] = x20 * (x22 * POW2(x25) + x22 * POW2(x27) + x22 * POW2(x29));
    fff[4 * stride] = x20 * (x22 * x25 * x31 + x22 * x27 * x32 + x22 * x29 * x33);
    fff[5 * stride] = x20 * (x22 * POW2(x31) + x22 * POW2(x32) + x22 * POW2(x33));
}

static inline __device__ __host__ void trial_operand(const scalar_t qx,
                                                     const scalar_t qy,
                                                     const scalar_t qz,
                                                     const scalar_t qw,
                                                     const ptrdiff_t stride,
                                                     const geom_t *const SFEM_RESTRICT fff,
                                                     const scalar_t *const SFEM_RESTRICT u,
                                                     scalar_t *const SFEM_RESTRICT out) {
    const scalar_t x0 = 4 * qx;
    const scalar_t x1 = 4 * qy;
    const scalar_t x2 = 4 * qz;
    const scalar_t x3 = x1 + x2;
    const scalar_t x4 = -u[6] * x1;
    const scalar_t x5 = u[0] * (x0 + x3 - 3);
    const scalar_t x6 = -u[7] * x2 + x5;
    const scalar_t x7 = u[1] * (x0 - 1) + u[4] * (-8 * qx - x3 + 4) + u[5] * x1 + u[8] * x2 + x4 + x6;
    const scalar_t x8 = x0 - 4;
    const scalar_t x9 = -u[4] * x0;
    const scalar_t x10 =
        u[2] * (x1 - 1) + u[5] * x0 + u[6] * (-8 * qy - x2 - x8) + u[9] * x2 + x6 + x9;
    const scalar_t x11 =
        u[3] * (x2 - 1) + u[7] * (-8 * qz - x1 - x8) + u[8] * x0 + u[9] * x1 + x4 + x5 + x9;
    out[0] = qw * (fff[0 * stride] * x7 + fff[1 * stride] * x10 + fff[2 * stride] * x11);
    out[1] = qw * (fff[1 * stride] * x7 + fff[3 * stride] * x10 + fff[4 * stride] * x11);
    out[2] = qw * (fff[2 * stride] * x7 + fff[4 * stride] * x10 + fff[5 * stride] * x11);
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

static inline __device__ __host__ void lapl_apply_micro_kernel(
    const scalar_t qx,
    const scalar_t qy,
    const scalar_t qz,
    const scalar_t qw,
    const ptrdiff_t stride,
    const geom_t *const SFEM_RESTRICT fff,
    const scalar_t *const SFEM_RESTRICT u,
    scalar_t *const SFEM_RESTRICT element_vector) {
    // Registers
    scalar_t ref_grad[10];
    scalar_t grad_u[3];

    // Evaluate gradient fe function transformed with fff and scaling factors
    trial_operand(qx, qy, qz, qw, stride, fff, u, grad_u);

    {  // X-components
        ref_shape_grad_x(qx, qy, qz, ref_grad);
#pragma unroll(10)
        for (int i = 0; i < 10; i++) {
            element_vector[i] += ref_grad[i] * grad_u[0];
        }
    }
    {  // Y-components
        ref_shape_grad_y(qx, qy, qz, ref_grad);
#pragma unroll(10)
        for (int i = 0; i < 10; i++) {
            element_vector[i] += ref_grad[i] * grad_u[1];
        }
    }

    {  // Z-components
        ref_shape_grad_z(qx, qy, qz, ref_grad);
#pragma unroll(10)
        for (int i = 0; i < 10; i++) {
            element_vector[i] += ref_grad[i] * grad_u[2];
        }
    }
}

__global__ void tet10_cuda_incore_laplacian_apply_kernel(const ptrdiff_t nelements,
                                                         idx_t *const SFEM_RESTRICT elems,
                                                         const cu_jacobian_t *const SFEM_RESTRICT fff,
                                                         const real_t *const SFEM_RESTRICT x,
                                                         real_t *const SFEM_RESTRICT y) {
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {
        scalar_t ex[10];
        scalar_t ey[10];
        idx_t vidx[10];

        // collect coeffs
#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            vidx[v] = elems[v * nelements + e];
            ex[v] = x[vidx[v]];
        }

        geom_t fffe[6];
#pragma unroll(6)
        for(int d = 0; d < 6; d++) {
            fffe[d] = fff[d*nelements];
        }

        {  // Numerical quadrature
            lapl_apply_micro_kernel(0, 0, 0, 0.025, 1, fffe, ex, ey);
            lapl_apply_micro_kernel(1, 0, 0, 0.025, 1, fffe, ex, ey);
            lapl_apply_micro_kernel(0, 1, 0, 0.025, 1, fffe, ex, ey);
            lapl_apply_micro_kernel(0, 0, 1, 0.025, 1, fffe, ex, ey);

            static const scalar_t athird = 1. / 3;
            lapl_apply_micro_kernel(athird, athird, 0., 0.225, 1, fffe, ex, ey);
            lapl_apply_micro_kernel(athird, 0., athird, 0.225, 1, fffe, ex, ey);
            lapl_apply_micro_kernel(0., athird, athird, 0.225, 1, fffe, ex, ey);
            lapl_apply_micro_kernel(athird, athird, athird, 0.225, 1, fffe, ex, ey);
        }

        // redistribute coeffs
#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            atomicAdd(&y[vidx[v]], ey[v]);
        }
    }
}

extern int tet10_cuda_incore_laplacian_apply(cuda_incore_laplacian_t *ctx,
                                             const real_t *const d_x,
                                             real_t *const d_y) {
    static int block_size = 128;
    ptrdiff_t n_blocks = std::max(ptrdiff_t(1), (ctx->nelements + block_size - 1) / block_size);
    tet10_cuda_incore_laplacian_apply_kernel<<<n_blocks, block_size, 0>>>(
        ctx->nelements, ctx->d_elems, (cu_jacobian_t*)ctx->d_fff, d_x, d_y);
    return 0;
}

extern int tet10_cuda_incore_laplacian_init(cuda_incore_laplacian_t *ctx,
                                            const ptrdiff_t nelements,
                                            idx_t **const SFEM_RESTRICT elements,
                                            geom_t **const SFEM_RESTRICT points) {
    {  // Create FFF and store it on device
        cu_jacobian_t *h_fff = (cu_jacobian_t *)calloc(6 * nelements, sizeof(cu_jacobian_t));

#pragma omp parallel
        {
#pragma omp for
            for (ptrdiff_t e = 0; e < nelements; e++) {
                fff_micro_kernel(points[0][elements[0][e]],
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
                                 &h_fff[e]);
            }
        }

        SFEM_CUDA_CHECK(cudaMalloc(&ctx->d_fff, 6 * nelements * sizeof(cu_jacobian_t)));
        SFEM_CUDA_CHECK(
            cudaMemcpy(ctx->d_fff, h_fff, 6 * nelements * sizeof(cu_jacobian_t), cudaMemcpyHostToDevice));
        free(h_fff);
    }

    {
        // Store elem indices on device
        SFEM_CUDA_CHECK(cudaMalloc(&ctx->d_elems, 10 * nelements * sizeof(idx_t)));

        for (int d = 0; d < 10; d++) {
            SFEM_CUDA_CHECK(cudaMemcpy(ctx->d_elems + d * nelements,
                                       elements[d],
                                       nelements * sizeof(idx_t),
                                       cudaMemcpyHostToDevice));
        }
    }

    ctx->nelements = nelements;
    return 0;
}

extern int tet10_cuda_incore_laplacian_destroy(cuda_incore_laplacian_t *ctx) {
    cudaFree(ctx->d_elems);
    cudaFree(ctx->d_fff);

    ctx->nelements = 0;
    ctx->d_elems = nullptr;
    ctx->d_fff = nullptr;
    return 0;
}
