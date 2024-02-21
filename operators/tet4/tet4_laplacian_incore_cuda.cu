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

#include "tet4_laplacian_incore_cuda.h"

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
    //      - Result: 6*ADD + 6*ASSIGNMENT + 24*MUL + 9*POW
    //      - Subexpressions: 4*ADD + 6*DIV + 28*MUL + NEG + POW + 24*SUB
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

static inline __device__ __host__ void lapl_apply_micro_kernel(const geom_t *const SFEM_RESTRICT fff,
                                                     const ptrdiff_t stride,
                                                     const real_t *const SFEM_RESTRICT x,
                                                     real_t *const SFEM_RESTRICT y) {
    const real_t x0 = (1.0 / 6.0) * x[0];
    const real_t x1 = fff[0 * stride] * x0;
    const real_t x2 = (1.0 / 6.0) * x[1];
    const real_t x3 = fff[0 * stride] * x2;
    const real_t x4 = fff[1 * stride] * x2;
    const real_t x5 = (1.0 / 6.0) * x[2];
    const real_t x6 = fff[1 * stride] * x5;
    const real_t x7 = fff[2 * stride] * x2;
    const real_t x8 = (1.0 / 6.0) * x[3];
    const real_t x9 = fff[2 * stride] * x8;
    const real_t x10 = fff[3 * stride] * x0;
    const real_t x11 = fff[3 * stride] * x5;
    const real_t x12 = fff[4 * stride] * x5;
    const real_t x13 = fff[4 * stride] * x8;
    const real_t x14 = fff[5 * stride] * x0;
    const real_t x15 = fff[5 * stride] * x8;
    const real_t x16 = fff[1 * stride] * x0;
    const real_t x17 = fff[2 * stride] * x0;
    const real_t x18 = fff[4 * stride] * x0;
    y[0] = (1.0 / 3.0) * fff[1 * stride] * x[0] + (1.0 / 3.0) * fff[2 * stride] * x[0] +
           (1.0 / 3.0) * fff[4 * stride] * x[0] + x1 + x10 - x11 - x12 - x13 + x14 - x15 - x3 - x4 -
           x6 - x7 - x9;
    y[1] = -x1 - x16 - x17 + x3 + x6 + x9;
    y[2] = -x10 + x11 + x13 - x16 - x18 + x4;
    y[3] = x12 - x14 + x15 - x17 - x18 + x7;
}

__global__ void tet4_cuda_incore_laplacian_apply_kernel(const ptrdiff_t nelements,
                                                        idx_t *const SFEM_RESTRICT elems,
                                                        const geom_t *const SFEM_RESTRICT fff,
                                                        const real_t *const SFEM_RESTRICT x,
                                                        real_t *const SFEM_RESTRICT y) {
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {
        real_t ex[4];
        real_t ey[4];
        idx_t vidx[4];

        // collect coeffs
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
        	vidx[v] = elems[v * nelements + e];
            ex[v] = x[vidx[v]];
        }

        // apply operator
        lapl_apply_micro_kernel(&fff[e], nelements, ex, ey);

        // redistribute coeffs
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            atomicAdd(&y[vidx[v]], ey[v]);
        }
    }
}

extern int tet4_cuda_incore_laplacian_apply(cuda_incore_laplacian_t *ctx,
                                            const real_t *const d_x,
                                            real_t *const d_y) {
    static int block_size = 128;
    ptrdiff_t n_blocks = std::max(ptrdiff_t(1), (ctx->nelements + block_size - 1) / block_size);
    tet4_cuda_incore_laplacian_apply_kernel<<<n_blocks, block_size, 0>>>(
        ctx->nelements, ctx->d_elems, ctx->d_fff, d_x, d_y);
    return 0;
}

extern int tet4_cuda_incore_laplacian_init(cuda_incore_laplacian_t *ctx, mesh_t mesh) {
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
        SFEM_CUDA_CHECK(cudaMalloc(&ctx->d_elems, 4 * mesh.nelements * sizeof(geom_t)));

        for (int d = 0; d < 4; d++) {
            SFEM_CUDA_CHECK(cudaMemcpy(ctx->d_elems + d * mesh.nelements,
                                       mesh.elements[d],
                                       mesh.nelements * sizeof(idx_t),
                                       cudaMemcpyHostToDevice));
        }
    }

    ctx->nelements = mesh.nelements;
    return 0;
}

extern int tet4_cuda_incore_laplacian_destroy(cuda_incore_laplacian_t *ctx) {
    cudaFree(ctx->d_elems);
    cudaFree(ctx->d_fff);

    ctx->nelements = 0;
    ctx->d_elems = nullptr;
    ctx->d_fff = nullptr;
    return 0;
}
