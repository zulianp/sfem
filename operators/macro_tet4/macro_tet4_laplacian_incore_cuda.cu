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
#include "sfem_mesh.h"

#include "macro_tet4_cuda_incore_laplacian.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define POW2(a) ((a) * (a))

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
                                                        geom_t *fff) {
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

static inline __device__ __host__ void sub_fff_0(const geom_t *const SFEM_RESTRICT fff,
                                                 geom_t *const SFEM_RESTRICT sub_fff) {
    sub_fff[0] = (1.0 / 2.0) * fff[0];
    sub_fff[1] = (1.0 / 2.0) * fff[1];
    sub_fff[2] = (1.0 / 2.0) * fff[2];
    sub_fff[3] = (1.0 / 2.0) * fff[3];
    sub_fff[4] = (1.0 / 2.0) * fff[4];
    sub_fff[5] = (1.0 / 2.0) * fff[5];
}

static inline __device__ __host__ void sub_fff_4(const geom_t *const SFEM_RESTRICT fff,
                                                 geom_t *const SFEM_RESTRICT sub_fff) {
    const geom_t x0 = (1.0 / 2.0) * fff[0];
    const geom_t x1 = (1.0 / 2.0) * fff[2];
    sub_fff[0] = fff[1] + (1.0 / 2.0) * fff[3] + x0;
    sub_fff[1] = -1.0 / 2.0 * fff[1] - x0;
    sub_fff[2] = (1.0 / 2.0) * fff[4] + x1;
    sub_fff[3] = x0;
    sub_fff[4] = -x1;
    sub_fff[5] = (1.0 / 2.0) * fff[5];
}

static inline __device__ __host__ void sub_fff_5(const geom_t *const SFEM_RESTRICT fff,
                                                 geom_t *const SFEM_RESTRICT sub_fff) {
    const geom_t x0 = (1.0 / 2.0) * fff[3];
    const geom_t x1 = fff[4] + (1.0 / 2.0) * fff[5] + x0;
    const geom_t x2 = (1.0 / 2.0) * fff[4] + x0;
    const geom_t x3 = (1.0 / 2.0) * fff[1];
    sub_fff[0] = x1;
    sub_fff[1] = -x2;
    sub_fff[2] = -1.0 / 2.0 * fff[2] - x1 - x3;
    sub_fff[3] = x0;
    sub_fff[4] = x2 + x3;
    sub_fff[5] = (1.0 / 2.0) * fff[0] + fff[1] + fff[2] + x1;
}

static inline __device__ __host__ void sub_fff_6(const geom_t *const SFEM_RESTRICT fff,
                                                 geom_t *const SFEM_RESTRICT sub_fff) {
    const geom_t x0 = (1.0 / 2.0) * fff[3];
    const geom_t x1 = (1.0 / 2.0) * fff[4];
    const geom_t x2 = (1.0 / 2.0) * fff[1] + x0;
    sub_fff[0] = (1.0 / 2.0) * fff[0] + fff[1] + x0;
    sub_fff[1] = (1.0 / 2.0) * fff[2] + x1 + x2;
    sub_fff[2] = -x2;
    sub_fff[3] = fff[4] + (1.0 / 2.0) * fff[5] + x0;
    sub_fff[4] = -x0 - x1;
    sub_fff[5] = x0;
}

static inline __device__ __host__ void sub_fff_7(const geom_t *const SFEM_RESTRICT fff,
                                                 geom_t *const SFEM_RESTRICT sub_fff) {
    const geom_t x0 = (1.0 / 2.0) * fff[5];
    const geom_t x1 = (1.0 / 2.0) * fff[2];
    sub_fff[0] = x0;
    sub_fff[1] = -1.0 / 2.0 * fff[4] - x0;
    sub_fff[2] = -x1;
    sub_fff[3] = (1.0 / 2.0) * fff[3] + fff[4] + x0;
    sub_fff[4] = (1.0 / 2.0) * fff[1] + x1;
    sub_fff[5] = (1.0 / 2.0) * fff[0];
}

static inline __device__ __host__ void lapl_apply_micro_kernel(const geom_t *const SFEM_RESTRICT
                                                                   fff,
                                                               const real_t u0,
                                                               const real_t u1,
                                                               const real_t u2,
                                                               const real_t u3,
                                                               real_t *const SFEM_RESTRICT e0,
                                                               real_t *const SFEM_RESTRICT e1,
                                                               real_t *const SFEM_RESTRICT e2,
                                                               real_t *const SFEM_RESTRICT e3) {
    const real_t x0 = (1.0 / 6.0) * u0;
    const real_t x1 = fff[0] * x0;
    const real_t x2 = (1.0 / 6.0) * u1;
    const real_t x3 = fff[0] * x2;
    const real_t x4 = fff[1] * x2;
    const real_t x5 = (1.0 / 6.0) * u2;
    const real_t x6 = fff[1] * x5;
    const real_t x7 = fff[2] * x2;
    const real_t x8 = (1.0 / 6.0) * u3;
    const real_t x9 = fff[2] * x8;
    const real_t x10 = fff[3] * x0;
    const real_t x11 = fff[3] * x5;
    const real_t x12 = fff[4] * x5;
    const real_t x13 = fff[4] * x8;
    const real_t x14 = fff[5] * x0;
    const real_t x15 = fff[5] * x8;
    const real_t x16 = fff[1] * x0;
    const real_t x17 = fff[2] * x0;
    const real_t x18 = fff[4] * x0;
    *e0 += (1.0 / 3.0) * fff[1] * u0 + (1.0 / 3.0) * fff[2] * u0 + (1.0 / 3.0) * fff[4] * u0 + x1 +
           x10 - x11 - x12 - x13 + x14 - x15 - x3 - x4 - x6 - x7 - x9;
    *e1 += -x1 - x16 - x17 + x3 + x6 + x9;
    *e2 += -x10 + x11 + x13 - x16 - x18 + x4;
    *e3 += x12 - x14 + x15 - x17 - x18 + x7;
}

__global__ void macro_tet4_cuda_incore_laplacian_apply_kernel(const ptrdiff_t nelements,
                                                        idx_t *const SFEM_RESTRICT elems,
                                                        const geom_t *const SFEM_RESTRICT fff,
                                                        const real_t *const SFEM_RESTRICT x,
                                                        real_t *const SFEM_RESTRICT y) {
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {
        real_t ex[10];
        real_t ey[10];
        idx_t vidx[10];

        // collect coeffs
#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            vidx[v] = elems[v * nelements + e];
            ex[v] = x[vidx[v]];
        }

        // apply operator

        {  // Corner tests
            sub_fff_0(fff, sub_fff);

            // [0, 4, 6, 7],
            lapl_apply_micro_kernel(
                sub_fff, ex[0], ex[4], ex[6], ex[7], &ey[0], &ey[4], &ey[6], &ey[7]);

            // [4, 1, 5, 8],
            lapl_apply_micro_kernel(
                sub_fff, ex[4], ex[1], ex[5], ex[8], &ey[4], &ey[1], &ey[5], &ey[8]);

            // [6, 5, 2, 9],
            lapl_apply_micro_kernel(
                sub_fff, ex[6], ex[5], ex[2], ex[9], &ey[6], &ey[5], &ey[2], &ey[9]);

            // [7, 8, 9, 3],
            lapl_apply_micro_kernel(
                sub_fff, ex[7], ex[8], ex[9], ex[3], &ey[7], &ey[8], &ey[9], &ey[3]);
        }

        {  // Octahedron tets

            // [4, 5, 6, 8],
            sub_fff_4(fff, sub_fff);
            lapl_apply_micro_kernel(
                sub_fff, ex[4], ex[5], ex[6], ex[8], &ey[4], &ey[5], &ey[6], &ey[8]);

            // [7, 4, 6, 8],
            sub_fff_5(fff, sub_fff);
            lapl_apply_micro_kernel(
                sub_fff, ex[7], ex[4], ex[6], ex[8], &ey[7], &ey[4], &ey[6], &ey[8]);

            // [6, 5, 9, 8],
            sub_fff_6(fff, sub_fff);
            lapl_apply_micro_kernel(
                sub_fff, ex[6], ex[5], ex[9], ex[8], &ey[6], &ey[5], &ey[9], &ey[8]);

            // [7, 6, 9, 8]]
            sub_fff_7(fff, sub_fff);
            lapl_apply_micro_kernel(
                sub_fff, ex[7], ex[6], ex[9], ex[8], &ey[7], &ey[6], &ey[9], &ey[8]);
        }

        // redistribute coeffs
#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            atomicAdd(&y[vidx[v]], ey[v]);
        }
    }
}

extern int macro_tet4_cuda_incore_laplacian_apply(cuda_incore_laplacian_t *ctx,
                                            const real_t *const d_x,
                                            real_t *const d_y) {
    static int block_size = 128;
    ptrdiff_t n_blocks = std::max(ptrdiff_t(1), (ctx->nelements + block_size - 1) / block_size);
    macro_tet4_cuda_incore_laplacian_apply_kernel<<<n_blocks, block_size, 0>>>(
        ctx->nelements, ctx->d_elems, ctx->d_fff, d_x, d_y);
    return 0;
}

extern int macro_tet4_cuda_incore_laplacian_init(cuda_incore_laplacian_t *ctx, mesh_t mesh) {
    {  // Create FFF and store it on device
        geom_t *h_fff = (geom_t *)calloc(6 * mesh.nelements, sizeof(geom_t));

#pragma omp parallel
        {
#pragma omp for
            for (ptrdiff_t e = 0; e < mesh.nelements; e++) {
                fff_micro_kernel(mesh.points[0][mesh.elements[0][e]],
                                 mesh.points[0][mesh.elements[1][e]],
                                 mesh.points[0][mesh.elements[2][e]],
                                 mesh.points[0][mesh.elements[3][e]],
                                 mesh.points[1][mesh.elements[0][e]],
                                 mesh.points[1][mesh.elements[1][e]],
                                 mesh.points[1][mesh.elements[2][e]],
                                 mesh.points[1][mesh.elements[3][e]],
                                 mesh.points[2][mesh.elements[0][e]],
                                 mesh.points[2][mesh.elements[1][e]],
                                 mesh.points[2][mesh.elements[2][e]],
                                 mesh.points[2][mesh.elements[3][e]],
                                 mesh.nelements,
                                 &h_fff[e]);
            }
        }

        SFEM_CUDA_CHECK(cudaMalloc(&ctx->d_fff, 6 * mesh.nelements * sizeof(geom_t)));
        SFEM_CUDA_CHECK(cudaMemcpy(
            ctx->d_fff, h_fff, 6 * mesh.nelements * sizeof(geom_t), cudaMemcpyHostToDevice));
        free(h_fff);
    }

    {
        // Store elem indices on device
        SFEM_CUDA_CHECK(cudaMalloc(&ctx->d_elems, 10 * mesh.nelements * sizeof(geom_t)));

        for (int d = 0; d < 10; d++) {
            SFEM_CUDA_CHECK(cudaMemcpy(ctx->d_elems + d * mesh.nelements,
                                       mesh.elements[d],
                                       mesh.nelements * sizeof(idx_t),
                                       cudaMemcpyHostToDevice));
        }
    }

    ctx->nelements = mesh.nelements;
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
