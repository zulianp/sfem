#include <cassert>
#include <cmath>
#include <algorithm>
#include <cstddef>

#include "sfem_base.h"
#include "sfem_vec.h"
#include "sortreduce.h"

#include "sfem_cuda_base.h"
#include "sfem_defs.h"
#include "tet4_laplacian_incore_cuda.h"

#include "tet4_inline_gpu.hpp"
#include "tet4_laplacian_inline_gpu.hpp"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define POW2(a) ((a) * (a))


template <typename cu_jacobian_t, typename real_t, typename scalar_t = real_t>
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

        tet4_laplacian_apply_fff(fffe, 1, ex, ey);

        // redistribute coeffs
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            atomicAdd(&y[vidx[v]], ey[v]);
        }
    }
}

__global__ void tet4_cuda_incore_laplacian_diag_kernel(const ptrdiff_t nelements,
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
        tet4_laplacian_diag_fff(fffe, 1, ed);

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
                tet4_fff(points[0][elements[0][e]],
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
        scalar_t ey = ref_grad[0] * JinvTgradu[0] + ref_grad[1] * JinvTgradu[1] +
                      ref_grad[2] * JinvTgradu[2];

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
            &min_grid_size, &block_size, tet4_cuda_incore_laplacian_apply_kernel<cu_jacobian_t, real_t, real_t>, 0, 0);
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
            &min_grid_size, &block_size, tet4_cuda_incore_laplacian_diag_kernel, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = std::max(ptrdiff_t(1), (ctx->nelements + block_size - 1) / block_size);
    tet4_cuda_incore_laplacian_diag_kernel<<<n_blocks, block_size, 0>>>(
        ctx->nelements, ctx->d_elems, (cu_jacobian_t *)ctx->d_fff, d_d);
    return 0;
}
