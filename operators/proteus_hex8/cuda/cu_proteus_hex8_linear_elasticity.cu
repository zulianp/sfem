#include "cu_hex8_linear_elasticity_inline.hpp"
#include "cu_proteus_hex8_linear_elasticity.h"
#include "sfem_cuda_base.h"

#include "cu_proteus_hex8_inline.hpp"

#ifndef B_
#define B_(x, y, z) SFEM_HEX8_BLOCK_IDX(x, y, z)
#endif

#ifndef MAX
#define MAX(a, b) ((a) < (b) ? (b) : (a))
#endif

#ifndef MIN
#define MIN(a, b) ((a) > (b) ? (b) : (a))
#endif

template <typename T, int LEVEL>
__global__ void cu_proteus_affine_hex8_linear_elasticity_apply_local_mem_kernel(
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const ptrdiff_t interior_start,
        const idx_t *const SFEM_RESTRICT elements,
        const cu_jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate,
        const cu_jacobian_t *const SFEM_RESTRICT g_jacobian_determinant,
        const T mu,
        const T lambda,
        const ptrdiff_t u_stride,
        const T *const SFEM_RESTRICT g_ux,
        const T *const SFEM_RESTRICT g_uy,
        const T *const SFEM_RESTRICT g_uz,
        const ptrdiff_t out_stride,
        T *const SFEM_RESTRICT g_outx,
        T *const SFEM_RESTRICT g_outy,
        T *const SFEM_RESTRICT g_outz) {
    // "local" memory
    T u_block[3][BLOCK_SIZE_3];
    T out_block[3][BLOCK_SIZE_3];
    T sub_adjugate[9];
    T sub_determinant;

    const T *g_u[3] = {g_ux, g_uy, g_uz};
    T *g_out[3] = {g_outx, g_outy, g_outz};

    static const int n_qp = 6;
    const T qw[6] = {0.16666666666666666666666666666667,
                     0.16666666666666666666666666666667,
                     0.16666666666666666666666666666667,
                     0.16666666666666666666666666666667,
                     0.16666666666666666666666666666667,
                     0.16666666666666666666666666666667};
    const T qx[6] = {0.0, 0.5, 0.5, 0.5, 0.5, 1.0};
    const T qy[6] = {0.5, 0.0, 0.5, 0.5, 1.0, 0.5};
    const T qz[6] = {0.5, 0.5, 0.0, 1.0, 0.5, 0.5};

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {
        // Gather from global to "local"
        for (int d = 0; d < 3; d++) {
            cu_proteus_hex8_gather<real_t, LEVEL>(
                    nelements, stride, interior_start, e, elements, g_u[d], u_block[d]);
        }

        // Get geometry
        sub_adjugate[0] = g_jacobian_adjugate[0 * stride + e];
        sub_adjugate[1] = g_jacobian_adjugate[1 * stride + e];
        sub_adjugate[2] = g_jacobian_adjugate[2 * stride + e];
        sub_adjugate[3] = g_jacobian_adjugate[3 * stride + e];
        sub_adjugate[4] = g_jacobian_adjugate[4 * stride + e];
        sub_adjugate[5] = g_jacobian_adjugate[5 * stride + e];
        sub_adjugate[6] = g_jacobian_adjugate[6 * stride + e];
        sub_adjugate[7] = g_jacobian_adjugate[7 * stride + e];
        sub_adjugate[8] = g_jacobian_adjugate[8 * stride + e];
        sub_determinant = g_jacobian_determinant[e];

        // Reset block accumulator
        for (int d = 0; d < 3; d++) {
            for (int i = 0; i < BLOCK_SIZE_3; i++) {
                out_block[d][i] = 0;
            }
        }

        {
            const T h = 1. / LEVEL;
            cu_hex8_sub_adj_0_in_place(h, sub_adjugate, &sub_determinant);
        }

        // Micro-loop
        for (int zi = 0; zi < LEVEL; zi++) {
            for (int yi = 0; yi < LEVEL; yi++) {
                for (int xi = 0; xi < LEVEL; xi++) {
                    scalar_t u[3][8];
                    scalar_t out[3][8];

                    // "local" to registers
                    for (int d = 0; d < 3; d++) {
                        u[d][0] = u_block[d][B_(threadIdx.x, threadIdx.y, threadIdx.z)];
                        u[d][1] = u_block[d][B_(threadIdx.x + 1, threadIdx.y, threadIdx.z)];
                        u[d][2] = u_block[d][B_(threadIdx.x + 1, threadIdx.y + 1, threadIdx.z)];
                        u[d][3] = u_block[d][B_(threadIdx.x, threadIdx.y + 1, threadIdx.z)];
                        u[d][4] = u_block[d][B_(threadIdx.x, threadIdx.y, threadIdx.z + 1)];
                        u[d][5] = u_block[d][B_(threadIdx.x + 1, threadIdx.y, threadIdx.z + 1)];
                        u[d][6] = u_block[d][B_(threadIdx.x + 1, threadIdx.y + 1, threadIdx.z + 1)];
                        u[d][7] = u_block[d][B_(threadIdx.x, threadIdx.y + 1, threadIdx.z + 1)];
                    }

                    // Reset micro-accumulator
                    for (int d = 0; d < 3; d++) {
                        for (int i = 0; i < 8; i++) {
                            out[d][i] = 0;
                        }
                    }

                    // Compute
                    for (int k = 0; k < n_qp; k++) {
                        cu_hex8_linear_elasticity_apply_adj(mu,
                                                            lambda,
                                                            sub_adjugate,
                                                            sub_determinant,
                                                            qx[k],
                                                            qy[k],
                                                            qz[k],
                                                            qw[k],
                                                            u[0],
                                                            u[1],
                                                            u[2],
                                                            out[0],
                                                            out[1],
                                                            out[2]);
                    }

                    // registers to "local"
                    for (int d = 0; d < 3; d++) {
                        out_block[d][B_(threadIdx.x, threadIdx.y, threadIdx.z)] += out[d][0];
                        out_block[d][B_(threadIdx.x + 1, threadIdx.y, threadIdx.z)] += out[d][1];
                        out_block[d][B_(threadIdx.x + 1, threadIdx.y + 1, threadIdx.z)] +=
                                out[d][2];
                        out_block[d][B_(threadIdx.x, threadIdx.y + 1, threadIdx.z)] += out[d][3];
                        out_block[d][B_(threadIdx.x, threadIdx.y, threadIdx.z + 1)] += out[d][4];
                        out_block[d][B_(threadIdx.x + 1, threadIdx.y, threadIdx.z + 1)] +=
                                out[d][5];
                        out_block[d][B_(threadIdx.x + 1, threadIdx.y + 1, threadIdx.z + 1)] +=
                                out[d][6];
                        out_block[d][B_(threadIdx.x, threadIdx.y + 1, threadIdx.z + 1)] +=
                                out[d][7];
                    }
                }
            }

            // Scatter from "local" to global
            for (int d = 0; d < 3; d++) {
                cu_proteus_hex8_scatter_add<real_t, LEVEL>(
                        nelements, stride, interior_start, e, elements, out_block[d], g_out[d]);
            }
        }
    }
}

template <typename T, int LEVEL>
int cu_proteus_affine_hex8_linear_elasticity_apply_local_mem_tpl(
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const ptrdiff_t interior_start,
        const idx_t *const SFEM_RESTRICT elements,
        const cu_jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
        const cu_jacobian_t *const SFEM_RESTRICT jacobian_determinant,
        const T mu,
        const T lambda,
        const ptrdiff_t u_stride,
        const T *const SFEM_RESTRICT ux,
        const T *const SFEM_RESTRICT uy,
        const T *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        T *const SFEM_RESTRICT outx,
        T *const SFEM_RESTRICT outy,
        T *const SFEM_RESTRICT outz,
        void *stream) {
    SFEM_DEBUG_SYNCHRONIZE();

    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(
                &min_grid_size,
                &block_size,
                cu_proteus_affine_hex8_laplacian_apply_kernel_volgen<T, LEVEL>,
                0,
                0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    const ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_proteus_affine_hex8_linear_elasticity_apply_local_mem_kernel<T, LEVEL>
                <<<n_blocks, block_size, 0, s>>>(nelements,
                                                 stride,
                                                 interior_start,
                                                 elements,
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
        cu_proteus_affine_hex8_linear_elasticity_apply_local_mem_kernel<T, LEVEL>
                <<<n_blocks, block_size, 0>>>(nelements,
                                              stride,
                                              interior_start,
                                              elements,
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

template <typename T, int LEVEL>
__global__ void cu_proteus_affine_hex8_linear_elasticity_apply_warp_kernel(
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const ptrdiff_t interior_start,
        const idx_t *const SFEM_RESTRICT elements,
        const cu_jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate,
        const cu_jacobian_t *const SFEM_RESTRICT g_jacobian_determinant,
        const T mu,
        const T lambda,
        const ptrdiff_t u_stride,
        const T *const SFEM_RESTRICT g_ux,
        const T *const SFEM_RESTRICT g_uy,
        const T *const SFEM_RESTRICT g_uz,
        const ptrdiff_t out_stride,
        T *const SFEM_RESTRICT g_outx,
        T *const SFEM_RESTRICT g_outy,
        T *const SFEM_RESTRICT g_outz) {
    static const int BLOCK_SIZE = LEVEL + 1;
    static const int BLOCK_SIZE_2 = BLOCK_SIZE * BLOCK_SIZE;
    static const int BLOCK_SIZE_3 = BLOCK_SIZE_2 * BLOCK_SIZE;

    assert(blockDim.x == BLOCK_SIZE);
    assert(blockDim.y == BLOCK_SIZE);
    assert(blockDim.z == BLOCK_SIZE);

    // Shared mem
    __shared__ T u_block[BLOCK_SIZE_3];
    __shared__ T out_block[BLOCK_SIZE_3];

    static const int n_qp = 6;
    const T qw[6] = {0.16666666666666666666666666666667,
                     0.16666666666666666666666666666667,
                     0.16666666666666666666666666666667,
                     0.16666666666666666666666666666667,
                     0.16666666666666666666666666666667,
                     0.16666666666666666666666666666667};
    const T qx[6] = {0.0, 0.5, 0.5, 0.5, 0.5, 1.0};
    const T qy[6] = {0.5, 0.0, 0.5, 0.5, 1.0, 0.5};
    const T qz[6] = {0.5, 0.5, 0.0, 1.0, 0.5, 0.5};

    const T *g_u[3] = {g_ux, g_uy, g_uz};
    T *g_out[3] = {g_outx, g_outy, g_outz};

    T out[3][8];
    T u[3][8];
    T sub_adjugate[9];
    T sub_determinant;

    for (ptrdiff_t e = blockIdx.x; e < nelements; e += gridDim.x) {
        const int lidx = threadIdx.z * BLOCK_SIZE_2 + threadIdx.y * BLOCK_SIZE + threadIdx.x;
        assert(lidx < BLOCK_SIZE_3);
        assert(lidx >= 0);

        const ptrdiff_t idx = elements[lidx * stride + e];
        const bool is_element = threadIdx.x < LEVEL && threadIdx.y < LEVEL && threadIdx.z < LEVEL;

        if (is_element) {
            sub_adjugate[0] = g_jacobian_adjugate[0 * stride + e];
            sub_adjugate[1] = g_jacobian_adjugate[1 * stride + e];
            sub_adjugate[2] = g_jacobian_adjugate[2 * stride + e];
            sub_adjugate[3] = g_jacobian_adjugate[3 * stride + e];
            sub_adjugate[4] = g_jacobian_adjugate[4 * stride + e];
            sub_adjugate[5] = g_jacobian_adjugate[5 * stride + e];
            sub_adjugate[6] = g_jacobian_adjugate[6 * stride + e];
            sub_adjugate[7] = g_jacobian_adjugate[7 * stride + e];
            sub_adjugate[8] = g_jacobian_adjugate[8 * stride + e];
            sub_determinant = g_jacobian_determinant[e];
        }

        // if (lidx < BLOCK_SIZE_3)  // DO WE NEED THIS?
        { out_block[lidx] = 0; }

        for (int d = 0; d < 3; d++) {
            // if (lidx < BLOCK_SIZE_3)  // DO WE NEED THIS?
            {
                u_block[lidx] = g_u[d][idx * u_stride];
                assert(u_block[lidx] == u_block[lidx]);
            }

            __syncthreads();

            if (is_element) {
                u[d][0] = u_block[B_(threadIdx.x, threadIdx.y, threadIdx.z)];
                u[d][1] = u_block[B_(threadIdx.x + 1, threadIdx.y, threadIdx.z)];
                u[d][2] = u_block[B_(threadIdx.x + 1, threadIdx.y + 1, threadIdx.z)];
                u[d][3] = u_block[B_(threadIdx.x, threadIdx.y + 1, threadIdx.z)];
                u[d][4] = u_block[B_(threadIdx.x, threadIdx.y, threadIdx.z + 1)];
                u[d][5] = u_block[B_(threadIdx.x + 1, threadIdx.y, threadIdx.z + 1)];
                u[d][6] = u_block[B_(threadIdx.x + 1, threadIdx.y + 1, threadIdx.z + 1)];
                u[d][7] = u_block[B_(threadIdx.x, threadIdx.y + 1, threadIdx.z + 1)];
            }
        }

        if (is_element) {
            const T h = 1. / LEVEL;
            cu_hex8_sub_adj_0_in_place(h, sub_adjugate, &sub_determinant);

            for (int d = 0; d < 3; d++) {
                for (int v = 0; v < 8; v++) {
                    out[d][v] = 0;
                }
            }

            for (int k = 0; k < n_qp; k++) {
                cu_hex8_linear_elasticity_apply_adj(mu,
                                                    lambda,
                                                    sub_adjugate,
                                                    sub_determinant,
                                                    qx[k],
                                                    qy[k],
                                                    qz[k],
                                                    qw[k],
                                                    u[0],
                                                    u[1],
                                                    u[2],
                                                    out[0],
                                                    out[1],
                                                    out[2]);
            }
        }

        const int interior = threadIdx.x > 0 && threadIdx.y > 0 && threadIdx.z > 0 &&
                             threadIdx.x < LEVEL && threadIdx.y < LEVEL && threadIdx.z < LEVEL;

        for (int d = 0; d < 3; d++) {
            if (is_element) {
                atomicAdd(&out_block[B_(threadIdx.x, threadIdx.y, threadIdx.z)], out[d][0]);
                atomicAdd(&out_block[B_(threadIdx.x + 1, threadIdx.y, threadIdx.z)], out[d][1]);
                atomicAdd(&out_block[B_(threadIdx.x + 1, threadIdx.y + 1, threadIdx.z)], out[d][2]);
                atomicAdd(&out_block[B_(threadIdx.x, threadIdx.y + 1, threadIdx.z)], out[d][3]);
                atomicAdd(&out_block[B_(threadIdx.x, threadIdx.y, threadIdx.z + 1)], out[d][4]);
                atomicAdd(&out_block[B_(threadIdx.x + 1, threadIdx.y, threadIdx.z + 1)], out[d][5]);
                atomicAdd(&out_block[B_(threadIdx.x + 1, threadIdx.y + 1, threadIdx.z + 1)],
                          out[d][6]);
                atomicAdd(&out_block[B_(threadIdx.x, threadIdx.y + 1, threadIdx.z + 1)], out[d][7]);
            }

            __syncthreads();

            // if (lidx < BLOCK_SIZE_3)  // DO WE NEED THIS?
            {
                assert(out_block[lidx] == out_block[lidx]);

                if (interior) {
                    g_out[d][idx * out_stride] += out_block[lidx];
                } else {
                    atomicAdd(&(g_out[d][idx * out_stride]), out_block[lidx]);
                }

                out_block[lidx] = 0;
            }

            if (d < 2) {
                __syncthreads();
            }
        }
    }
}

template <typename T, int LEVEL>
int cu_proteus_affine_hex8_linear_elasticity_apply_warp_tpl(
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const ptrdiff_t interior_start,
        const idx_t *const SFEM_RESTRICT elements,
        const cu_jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
        const cu_jacobian_t *const SFEM_RESTRICT jacobian_determinant,
        const T mu,
        const T lambda,
        const ptrdiff_t u_stride,
        const T *const SFEM_RESTRICT ux,
        const T *const SFEM_RESTRICT uy,
        const T *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        T *const SFEM_RESTRICT outx,
        T *const SFEM_RESTRICT outy,
        T *const SFEM_RESTRICT outz,
        void *stream) {
    SFEM_DEBUG_SYNCHRONIZE();

    static const int BLOCK_SIZE = LEVEL + 1;

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 n_blocks(MIN(nelements, 65535), 1, 1);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_proteus_affine_hex8_linear_elasticity_apply_warp_kernel<T, LEVEL>
                <<<n_blocks, block_size, 0, s>>>(nelements,
                                                 stride,
                                                 interior_start,
                                                 elements,
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
        cu_proteus_affine_hex8_linear_elasticity_apply_warp_kernel<T, LEVEL>
                <<<n_blocks, block_size, 0>>>(nelements,
                                              stride,
                                              interior_start,
                                              elements,
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

template <typename real_t>
int cu_proteus_affine_hex8_linear_elasticity_apply_tpl(
        const int level,
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const ptrdiff_t interior_start,
        const idx_t *const SFEM_RESTRICT elements,
        const cu_jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
        const cu_jacobian_t *const SFEM_RESTRICT jacobian_determinant,
        const real_t mu,
        const real_t lambda,
        const ptrdiff_t u_stride,
        const real_t *const SFEM_RESTRICT ux,
        const real_t *const SFEM_RESTRICT uy,
        const real_t *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        real_t *const SFEM_RESTRICT outx,
        real_t *const SFEM_RESTRICT outy,
        real_t *const SFEM_RESTRICT outz,
        void *stream) {
    switch (level) {
        case 2: {
            return cu_proteus_affine_hex8_linear_elasticity_apply_local_mem_tpl<real_t, 3>(
                    nelements,
                    stride,
                    interior_start,
                    elements,
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
        case 3: {
            return cu_proteus_affine_hex8_linear_elasticity_apply_warp_tpl<real_t, 3>(
                    nelements,
                    stride,
                    interior_start,
                    elements,
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
        case 4: {
            return cu_proteus_affine_hex8_linear_elasticity_apply_warp_tpl<real_t, 4>(
                    nelements,
                    stride,
                    interior_start,
                    elements,
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
        case 5: {
            return cu_proteus_affine_hex8_linear_elasticity_apply_warp_tpl<real_t, 5>(
                    nelements,
                    stride,
                    interior_start,
                    elements,
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
        case 6: {
            return cu_proteus_affine_hex8_linear_elasticity_apply_warp_tpl<real_t, 6>(
                    nelements,
                    stride,
                    interior_start,
                    elements,
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
        case 7: {
            return cu_proteus_affine_hex8_linear_elasticity_apply_warp_tpl<real_t, 7>(
                    nelements,
                    stride,
                    interior_start,
                    elements,
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
        case 8: {
            return cu_proteus_affine_hex8_linear_elasticity_apply_warp_tpl<real_t, 8>(
                    nelements,
                    stride,
                    interior_start,
                    elements,
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
        case 9: {
            return cu_proteus_affine_hex8_linear_elasticity_apply_local_mem_tpl<real_t, 9>(
                    nelements,
                    stride,
                    interior_start,
                    elements,
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
        default: {
            fprintf(stderr,
                    "cu_proteus_affine_hex8_linear_elasticity_apply_tpl: level %d not "
                    "supported!\n",
                    level);
            assert(false);
            return SFEM_FAILURE;
        }
    }
}

extern int cu_proteus_affine_hex8_linear_elasticity_apply(
        const int level,
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const ptrdiff_t interior_start,
        const idx_t *const SFEM_RESTRICT elements,
        const void *const SFEM_RESTRICT jacobian_adjugate,
        const void *const SFEM_RESTRICT jacobian_determinant,
        const real_t mu,
        const real_t lambda,
        const enum RealType real_type,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const void *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy,
        void *const SFEM_RESTRICT outz,
        void *stream) {
    switch (real_type) {
        case SFEM_REAL_DEFAULT: {
            return cu_proteus_affine_hex8_linear_elasticity_apply_tpl<real_t>(
                    level,
                    nelements,
                    stride,
                    interior_start,
                    elements,
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
            return cu_proteus_affine_hex8_linear_elasticity_apply_tpl<float>(
                    level,
                    nelements,
                    stride,
                    interior_start,
                    elements,
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
            return cu_proteus_affine_hex8_linear_elasticity_apply_tpl<double>(
                    level,
                    nelements,
                    stride,
                    interior_start,
                    elements,
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
            fprintf(stderr,
                    "[Error] cu_proteus_affine_hex8_linear_elasticity_apply: not implemented "
                    "for "
                    "type %s "
                    "(code %d)\n",
                    real_type_to_string(real_type),
                    real_type);
            assert(0);
            return SFEM_FAILURE;
        }
    }
}
