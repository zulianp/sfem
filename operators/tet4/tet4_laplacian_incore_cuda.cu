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


#include "sfem_defs.h"
#include "sfem_cuda_base.h"
#include "tet4_laplacian_incore_cuda.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define POW2(a) ((a) * (a))

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
    const geom_t x20 = (geom_t)(-1.0 / 6.0) * x12 + (geom_t)(1.0 / 6.0) * x15 +
                       (geom_t)(1.0 / 6.0) * x17 - (geom_t)(1.0 / 6.0) * x19 +
                       (geom_t)(1.0 / 6.0) * x4 - (geom_t)(1.0 / 6.0) * x8;
    const geom_t x21 = x14 - x18;
    const geom_t x22 = 1 / POW2(-x12 + x15 + x17 - x19 + x4 - x8);
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

static inline __device__ __host__ void lapl_apply_micro_kernel(
    const geom_t *const SFEM_RESTRICT fff,
    const ptrdiff_t stride,
    const scalar_t *const SFEM_RESTRICT u,
    scalar_t *const SFEM_RESTRICT element_vector) {
    const scalar_t x0 = fff[0 * stride] + fff[1 * stride] + fff[2 * stride];
    const scalar_t x1 = fff[1 * stride] + fff[3 * stride] + fff[4 * stride];
    const scalar_t x2 = fff[2 * stride] + fff[4 * stride] + fff[5 * stride];
    const scalar_t x3 = fff[1 * stride] * u[0];
    const scalar_t x4 = fff[2 * stride] * u[0];
    const scalar_t x5 = fff[4 * stride] * u[0];
    element_vector[0] = u[0] * x0 + u[0] * x1 + u[0] * x2 - u[1] * x0 - u[2] * x1 - u[3] * x2;
    element_vector[1] = -fff[0 * stride] * u[0] + fff[0 * stride] * u[1] + fff[1 * stride] * u[2] +
                        fff[2 * stride] * u[3] - x3 - x4;
    element_vector[2] = fff[1 * stride] * u[1] - fff[3 * stride] * u[0] + fff[3 * stride] * u[2] +
                        fff[4 * stride] * u[3] - x3 - x5;
    element_vector[3] = fff[2 * stride] * u[1] + fff[4 * stride] * u[2] - fff[5 * stride] * u[0] +
                        fff[5 * stride] * u[3] - x4 - x5;
}

static inline __device__ __host__ void lapl_diag_micro_kernel(const geom_t *const SFEM_RESTRICT fff,
                                                              const ptrdiff_t stride,
                                                              scalar_t *const SFEM_RESTRICT
                                                                  element_vector) {
    element_vector[0] = fff[0 * stride] + 2 * fff[1 * stride] + 2 * fff[2 * stride] +
                        fff[3 * stride] + 2 * fff[4 * stride] + fff[5 * stride];
    element_vector[1] = fff[0 * stride];
    element_vector[2] = fff[3 * stride];
    element_vector[3] = fff[5 * stride];
}

__global__ void tet4_cuda_incore_laplacian_apply_kernel(const ptrdiff_t nelements,
                                                        idx_t *const SFEM_RESTRICT elems,
                                                        const cu_jacobian_t *const SFEM_RESTRICT
                                                            fff,
                                                        const real_t *const SFEM_RESTRICT x,
                                                        real_t *const SFEM_RESTRICT y) {
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {
        scalar_t ex[4];
        scalar_t ey[4];
        idx_t vidx[4];

        // collect coeffs
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            vidx[v] = elems[v * nelements + e];
            ex[v] = x[vidx[v]];
        }

        geom_t fffe[6];
#pragma unroll(6)
        for (int d = 0; d < 6; d++) {
            fffe[d] = fff[d * nelements + e];
        }

        // Apply operator
        // lapl_apply_micro_kernel(&fff[e], nelements, ex, ey);

        lapl_apply_micro_kernel(fffe, 1, ex, ey);

        // redistribute coeffs
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            atomicAdd(&y[vidx[v]], ey[v]);
        }
    }
}

__global__ void tet4_cuda_incore_laplacian_diag_kernel(
    const ptrdiff_t nelements,
    idx_t *const SFEM_RESTRICT elems,
    const cu_jacobian_t *const SFEM_RESTRICT fff,
    real_t *const SFEM_RESTRICT diag) {
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {
        scalar_t ed[4];
        idx_t vidx[4];

        // collect coeffs
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            vidx[v] = elems[v * nelements + e];
        }

        geom_t fffe[6];
#pragma unroll(6)
        for (int d = 0; d < 6; d++) {
            fffe[d] = fff[d * nelements + e];
        }

        // Assembler operator diagonal
        lapl_diag_micro_kernel(fffe, 1, ed);

        // redistribute coeffs
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            atomicAdd(&diag[vidx[v]], ed[v]);
        }
    }
}

extern int tet4_cuda_incore_laplacian_init(cuda_incore_laplacian_t *ctx,
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
        SFEM_CUDA_CHECK(cudaMemcpy(
            ctx->d_fff, h_fff, 6 * nelements * sizeof(cu_jacobian_t), cudaMemcpyHostToDevice));
        free(h_fff);
    }

    {
        // Store elem indices on device
        SFEM_CUDA_CHECK(cudaMalloc(&ctx->d_elems, 4 * nelements * sizeof(idx_t)));

        for (int d = 0; d < 4; d++) {
            SFEM_CUDA_CHECK(cudaMemcpy(ctx->d_elems + d * nelements,
                                       elements[d],
                                       nelements * sizeof(idx_t),
                                       cudaMemcpyHostToDevice));
        }
    }

    ctx->element_type = TET4;
    ctx->nelements = nelements;
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

// Version 2

__global__ void tet4_cuda_incore_laplacian_apply_kernel_V2(const ptrdiff_t nelements,
                                                           idx_t *const SFEM_RESTRICT elems,
                                                           const cu_jacobian_t *const SFEM_RESTRICT
                                                               fff,
                                                           const real_t *const SFEM_RESTRICT x,
                                                           real_t *const SFEM_RESTRICT y) {
    int v = threadIdx.y;
    scalar_t ref_grad[3] = {0., 0., 0.};

    switch (v) {
        case 0: {
            ref_grad[0] = -1;
            ref_grad[1] = -1;
            ref_grad[2] = -1;
            break;
        }
        case 1: {
            ref_grad[0] = 1;
            break;
        }
        case 2: {
            ref_grad[1] = 1;
            break;
        }
        case 3: {
            ref_grad[2] = 1;
            break;
        }
        default: {
            assert(false);
        }
    }

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {
        const idx_t vidx = elems[v * nelements + e];
        const scalar_t ex = x[vidx];

        // apply operator
        scalar_t gradu[3] = {0., 0., 0};

// Compute gradient of quantity
#pragma unroll
        for (int i = (4) / 2; i >= 1; i /= 2) {
            gradu[0] += __shfl_xor_sync(0xffffffff, ex * ref_grad[0], i, 32);
            gradu[1] += __shfl_xor_sync(0xffffffff, ex * ref_grad[1], i, 32);
            gradu[2] += __shfl_xor_sync(0xffffffff, ex * ref_grad[2], i, 32);
        }

        geom_t fffe[6];
#pragma unroll(6)
        for (int d = 0; d < 6; d++) {
            fffe[d] = fff[d * nelements + e];
        }

        scalar_t JinvTgradu[3] = {fffe[0] * gradu[0] + fffe[1] * gradu[1] + fffe[2] * gradu[2],
                                  fffe[1] * gradu[0] + fffe[3] * gradu[1] + fffe[4] * gradu[2],
                                  fffe[2] * gradu[0] + fffe[4] * gradu[1] + fffe[5] * gradu[2]};

        //  dot product
        scalar_t ey =
            ref_grad[0] * JinvTgradu[0] + ref_grad[1] * JinvTgradu[1] + ref_grad[2] * JinvTgradu[2];

        // redistribute coeffs
        atomicAdd(&y[vidx], ey);
    }
}

extern int tet4_cuda_incore_laplacian_apply_V2(cuda_incore_laplacian_t *ctx,
                                               const real_t *const d_x,
                                               real_t *const d_y) {
    static int block_size = 128;
    ptrdiff_t n_blocks = std::max(ptrdiff_t(1), (ctx->nelements + block_size - 1) / block_size);

    dim3 n_blocks_2(n_blocks, 1);
    dim3 block_size_2(block_size, 4);

    tet4_cuda_incore_laplacian_apply_kernel_V2<<<n_blocks_2, block_size_2, 0>>>(
        ctx->nelements, ctx->d_elems, (cu_jacobian_t *)ctx->d_fff, d_x, d_y);
    printf("tet4_cuda_incore_laplacian_apply_V2\n");
    return 0;
}

extern int tet4_cuda_incore_laplacian_apply(cuda_incore_laplacian_t *ctx,
                                            const real_t *const d_x,
                                            real_t *const d_y) {
    if (0) {
        // This implementation is slower on the NVIDIA GeForce RTX 3060
        return tet4_cuda_incore_laplacian_apply_V2(ctx, d_x, d_y);
    }

    // Hand tuned
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size, &block_size, tet4_cuda_incore_laplacian_apply_kernel, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = std::max(ptrdiff_t(1), (ctx->nelements + block_size - 1) / block_size);
    tet4_cuda_incore_laplacian_apply_kernel<<<n_blocks, block_size, 0>>>(
        ctx->nelements, ctx->d_elems, (cu_jacobian_t *)ctx->d_fff, d_x, d_y);
    return 0;
}

extern int tet4_cuda_incore_laplacian_diag(cuda_incore_laplacian_t *ctx, real_t *const d_d) {
        // Hand tuned
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size, &block_size, tet4_cuda_incore_laplacian_apply_kernel, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = std::max(ptrdiff_t(1), (ctx->nelements + block_size - 1) / block_size);
    tet4_cuda_incore_laplacian_diag_kernel<<<n_blocks, block_size, 0>>>(
        ctx->nelements, ctx->d_elems, (cu_jacobian_t *)ctx->d_fff, d_d);
    return 0;
}
