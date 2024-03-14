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
#include "macro_tet4_laplacian_incore_cuda.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define POW2(a) ((a) * (a))


#define block_size 128

// #define SFEM_ENABLE_FP32_KERNELS
// #define SFEM_ENABLE_FP16_JACOBIANS

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

static /*inline*/ __device__ __host__ void sub_fff_0(const geom_t *const SFEM_RESTRICT fff,
                                                     const ptrdiff_t stride,
                                                     geom_t *const SFEM_RESTRICT sub_fff) {
    sub_fff[0] = (geom_t)(1.0 / 2.0) * fff[0 * stride];
    sub_fff[1] = (geom_t)(1.0 / 2.0) * fff[1 * stride];
    sub_fff[2] = (geom_t)(1.0 / 2.0) * fff[2 * stride];
    sub_fff[3] = (geom_t)(1.0 / 2.0) * fff[3 * stride];
    sub_fff[4] = (geom_t)(1.0 / 2.0) * fff[4 * stride];
    sub_fff[5] = (geom_t)(1.0 / 2.0) * fff[5 * stride];
}

static /*inline*/ __device__ __host__ void sub_fff_4(const geom_t *const SFEM_RESTRICT fff,
                                                     const ptrdiff_t stride,
                                                     geom_t *const SFEM_RESTRICT sub_fff) {
    const geom_t x0 = (geom_t)(1.0 / 2.0) * fff[0 * stride];
    const geom_t x1 = (geom_t)(1.0 / 2.0) * fff[2 * stride];
    sub_fff[0] = fff[1 * stride] + (1.0 / 2.0) * fff[3 * stride] + x0;
    sub_fff[1] = (geom_t)(-1.0 / 2.0) * fff[1 * stride] - x0;
    sub_fff[2] = (geom_t)(1.0 / 2.0) * fff[4 * stride] + x1;
    sub_fff[3] = x0;
    sub_fff[4] = -x1;
    sub_fff[5] = (geom_t)(1.0 / 2.0) * fff[5 * stride];
}

static /*inline*/ __device__ __host__ void sub_fff_5(const geom_t *const SFEM_RESTRICT fff,
                                                     const ptrdiff_t stride,
                                                     geom_t *const SFEM_RESTRICT sub_fff) {
    const geom_t x0 = (geom_t)(1.0 / 2.0) * fff[3 * stride];
    const geom_t x1 = fff[4 * stride] + (geom_t)(1.0 / 2.0) * fff[5 * stride] + x0;
    const geom_t x2 = (geom_t)(1.0 / 2.0) * fff[4 * stride] + x0;
    const geom_t x3 = (geom_t)(1.0 / 2.0) * fff[1 * stride];
    sub_fff[0] = x1;
    sub_fff[1] = -x2;
    sub_fff[2] = (geom_t)(-1.0 / 2.0) * fff[2 * stride] - x1 - x3;
    sub_fff[3] = x0;
    sub_fff[4] = x2 + x3;
    sub_fff[5] = (geom_t)(1.0 / 2.0) * fff[0 * stride] + fff[1 * stride] + fff[2 * stride] + x1;
}

static /*inline*/ __device__ __host__ void sub_fff_6(const geom_t *const SFEM_RESTRICT fff,
                                                     const ptrdiff_t stride,
                                                     geom_t *const SFEM_RESTRICT sub_fff) {
    const geom_t x0 = (geom_t)(1.0 / 2.0) * fff[3 * stride];
    const geom_t x1 = (geom_t)(1.0 / 2.0) * fff[4 * stride];
    const geom_t x2 = (geom_t)(1.0 / 2.0) * fff[1 * stride] + x0;
    sub_fff[0] = (geom_t)(1.0 / 2.0) * fff[0 * stride] + fff[1 * stride] + x0;
    sub_fff[1] = (geom_t)(1.0 / 2.0) * fff[2 * stride] + x1 + x2;
    sub_fff[2] = -x2;
    sub_fff[3] = fff[4 * stride] + (geom_t)(1.0 / 2.0) * fff[5 * stride] + x0;
    sub_fff[4] = -x0 - x1;
    sub_fff[5] = x0;
}

static /*inline*/ __device__ __host__ void sub_fff_7(const geom_t *const SFEM_RESTRICT fff,
                                                     const ptrdiff_t stride,
                                                     geom_t *const SFEM_RESTRICT sub_fff) {
    const geom_t x0 = (geom_t)(1.0 / 2.0) * fff[5 * stride];
    const geom_t x1 = (geom_t)(1.0 / 2.0) * fff[2 * stride];
    sub_fff[0] = x0;
    sub_fff[1] = (geom_t)(-1.0 / 2.0) * fff[4 * stride] - x0;
    sub_fff[2] = -x1;
    sub_fff[3] = (geom_t)(1.0 / 2.0) * fff[3 * stride] + fff[4 * stride] + x0;
    sub_fff[4] = (geom_t)(1.0 / 2.0) * fff[1 * stride] + x1;
    sub_fff[5] = (geom_t)(1.0 / 2.0) * fff[0 * stride];
}

static /*inline*/ __device__ __host__ void lapl_apply_micro_kernel(
    const geom_t *const SFEM_RESTRICT fff,
    const scalar_t u0,
    const scalar_t u1,
    const scalar_t u2,
    const scalar_t u3,
    scalar_t *const SFEM_RESTRICT e0,
    scalar_t *const SFEM_RESTRICT e1,
    scalar_t *const SFEM_RESTRICT e2,
    scalar_t *const SFEM_RESTRICT e3) {
    const scalar_t x0 = fff[0] + fff[1] + fff[2];
    const scalar_t x1 = fff[1] + fff[3] + fff[4];
    const scalar_t x2 = fff[2] + fff[4] + fff[5];
    const scalar_t x3 = fff[1] * u0;
    const scalar_t x4 = fff[2] * u0;
    const scalar_t x5 = fff[4] * u0;
    *e0 += u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
    *e1 += -fff[0] * u0 + fff[0] * u1 + fff[1] * u2 + fff[2] * u3 - x3 - x4;
    *e2 += fff[1] * u1 - fff[3] * u0 + fff[3] * u2 + fff[4] * u3 - x3 - x5;
    *e3 += fff[2] * u1 + fff[4] * u2 - fff[5] * u0 + fff[5] * u3 - x4 - x5;
}

// Worse
// #define MACRO_TET4_USE_SHARED

__global__ void macro_tet4_cuda_incore_laplacian_apply_kernel(const ptrdiff_t nelements,
                                                              idx_t *const SFEM_RESTRICT elems,
                                                              const cu_jacobian_t *const SFEM_RESTRICT fff,
                                                              const real_t *const SFEM_RESTRICT x,
                                                              real_t *const SFEM_RESTRICT y) {
    scalar_t ex[10];
    scalar_t ey[10];
    geom_t sub_fff[6];

#ifdef MACRO_TET4_USE_SHARED
    __shared__ geom_t sfff[6 * block_size];
#endif

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {
#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ey[v] = 0;
        }
        // collect coeffs
        // #pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ex[v] = x[elems[v * nelements + e]];
        }

#ifdef MACRO_TET4_USE_SHARED
        for (int d = 0; d < 6; d++) {
            sfff[d * block_size + threadIdx.x] = fff[d * nelements + e];
        }

        const ptrdiff_t stride = block_size;
        const geom_t *const offf = &sfff[threadIdx.x];

#else
        // const ptrdiff_t stride = nelements;
        // const cu_jacobian_t *const offf = &fff[e];

         const ptrdiff_t stride = 1;
        geom_t offf[6];
        #pragma unroll(6)
        for(int d = 0; d < 6; d++) {
            offf[d] = fff[d*nelements + e];
        }
#endif

        // apply operator

        {  // Corner tests
            sub_fff_0(offf, stride, sub_fff);

            // [0, 4, 6, 7],
            lapl_apply_micro_kernel(sub_fff,
                                    ex[0],
                                    ex[4],
                                    ex[6],
                                    ex[7],  //
                                    &ey[0],
                                    &ey[4],
                                    &ey[6],
                                    &ey[7]);

            // [4, 1, 5, 8],
            lapl_apply_micro_kernel(sub_fff,
                                    ex[4],
                                    ex[1],
                                    ex[5],
                                    ex[8],  //
                                    &ey[4],
                                    &ey[1],
                                    &ey[5],
                                    &ey[8]);

            // [6, 5, 2, 9],
            lapl_apply_micro_kernel(sub_fff,
                                    ex[6],
                                    ex[5],
                                    ex[2],
                                    ex[9],  //
                                    &ey[6],
                                    &ey[5],
                                    &ey[2],
                                    &ey[9]);

            // [7, 8, 9, 3],
            lapl_apply_micro_kernel(sub_fff,
                                    ex[7],
                                    ex[8],
                                    ex[9],
                                    ex[3],  //
                                    &ey[7],
                                    &ey[8],
                                    &ey[9],
                                    &ey[3]);
        }

        {  // Octahedron tets

            // [4, 5, 6, 8],
            sub_fff_4(offf, stride, sub_fff);
            lapl_apply_micro_kernel(sub_fff,
                                    ex[4],
                                    ex[5],
                                    ex[6],
                                    ex[8],  //
                                    &ey[4],
                                    &ey[5],
                                    &ey[6],
                                    &ey[8]);

            // [7, 4, 6, 8],
            sub_fff_5(offf, stride, sub_fff);
            lapl_apply_micro_kernel(sub_fff,
                                    ex[7],
                                    ex[4],
                                    ex[6],
                                    ex[8],  //
                                    &ey[7],
                                    &ey[4],
                                    &ey[6],
                                    &ey[8]);

            // [6, 5, 9, 8],
            sub_fff_6(offf, stride, sub_fff);
            lapl_apply_micro_kernel(sub_fff,
                                    ex[6],
                                    ex[5],
                                    ex[9],
                                    ex[8],  //
                                    &ey[6],
                                    &ey[5],
                                    &ey[9],
                                    &ey[8]);

            // [7, 6, 9, 8]]
            sub_fff_7(offf, stride, sub_fff);
            lapl_apply_micro_kernel(sub_fff,
                                    ex[7],
                                    ex[6],
                                    ex[9],
                                    ex[8],  //
                                    &ey[7],
                                    &ey[6],
                                    &ey[9],
                                    &ey[8]);
        }

        // redistribute coeffs
        // #pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            atomicAdd(&y[elems[v * nelements + e]], ey[v]);
        }
    }
}

extern int macro_tet4_cuda_incore_laplacian_apply(cuda_incore_laplacian_t *ctx,
                                                  const real_t *const d_x,
                                                  real_t *const d_y) {
    ptrdiff_t n_blocks = std::max(ptrdiff_t(1), (ctx->nelements + block_size - 1) / block_size);
    macro_tet4_cuda_incore_laplacian_apply_kernel<<<n_blocks, block_size, 0>>>(
        ctx->nelements, ctx->d_elems, (cu_jacobian_t*)ctx->d_fff, d_x, d_y);
    return 0;
}

extern int macro_tet4_cuda_incore_laplacian_init(cuda_incore_laplacian_t *ctx,
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

extern int macro_tet4_cuda_incore_laplacian_destroy(cuda_incore_laplacian_t *ctx) {
    cudaFree(ctx->d_elems);
    cudaFree(ctx->d_fff);

    ctx->nelements = 0;
    ctx->d_elems = nullptr;
    ctx->d_fff = nullptr;
    return 0;
}
