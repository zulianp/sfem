#include "cu_hex8_linear_elasticity_inline.hpp"
#include "cu_proteus_hex8_linear_elasticity.h"
#include "sfem_cuda_base.h"

#include "cu_hex8_linear_elasticity_inline.hpp"
#include "cu_proteus_hex8_inline.hpp"

#ifndef MAX
#define MAX(a, b) ((a) < (b) ? (b) : (a))
#endif

#ifndef MIN
#define MIN(a, b) ((a) > (b) ? (b) : (a))
#endif

static const int n_qp = 27;

static const scalar_t h_qw[27] = {
        0.021433470507545, 0.034293552812071, 0.021433470507545, 0.034293552812071,
        0.054869684499314, 0.034293552812071, 0.021433470507545, 0.034293552812071,
        0.021433470507545, 0.034293552812071, 0.054869684499314, 0.034293552812071,
        0.054869684499314, 0.087791495198903, 0.054869684499314, 0.034293552812071,
        0.054869684499314, 0.034293552812071, 0.021433470507545, 0.034293552812071,
        0.021433470507545, 0.034293552812071, 0.054869684499314, 0.034293552812071,
        0.021433470507545, 0.034293552812071, 0.021433470507545};

static const scalar_t h_qx[27] = {
        0.112701665379258, 0.500000000000000, 0.887298334620742, 0.112701665379258,
        0.500000000000000, 0.887298334620742, 0.112701665379258, 0.500000000000000,
        0.887298334620742, 0.112701665379258, 0.500000000000000, 0.887298334620742,
        0.112701665379258, 0.500000000000000, 0.887298334620742, 0.112701665379258,
        0.500000000000000, 0.887298334620742, 0.112701665379258, 0.500000000000000,
        0.887298334620742, 0.112701665379258, 0.500000000000000, 0.887298334620742,
        0.112701665379258, 0.500000000000000, 0.887298334620742};

static const scalar_t h_qy[27] = {
        0.112701665379258, 0.112701665379258, 0.112701665379258, 0.500000000000000,
        0.500000000000000, 0.500000000000000, 0.887298334620742, 0.887298334620742,
        0.887298334620742, 0.112701665379258, 0.112701665379258, 0.112701665379258,
        0.500000000000000, 0.500000000000000, 0.500000000000000, 0.887298334620742,
        0.887298334620742, 0.887298334620742, 0.112701665379258, 0.112701665379258,
        0.112701665379258, 0.500000000000000, 0.500000000000000, 0.500000000000000,
        0.887298334620742, 0.887298334620742, 0.887298334620742};

static const scalar_t h_qz[27] = {
        0.112701665379258, 0.112701665379258, 0.112701665379258, 0.112701665379258,
        0.112701665379258, 0.112701665379258, 0.112701665379258, 0.112701665379258,
        0.112701665379258, 0.500000000000000, 0.500000000000000, 0.500000000000000,
        0.500000000000000, 0.500000000000000, 0.500000000000000, 0.500000000000000,
        0.500000000000000, 0.500000000000000, 0.887298334620742, 0.887298334620742,
        0.887298334620742, 0.887298334620742, 0.887298334620742, 0.887298334620742,
        0.887298334620742, 0.887298334620742, 0.887298334620742};

__constant__ scalar_t qx[27];
__constant__ scalar_t qy[27];
__constant__ scalar_t qz[27];
__constant__ scalar_t qw[27];

static void init_quadrature() {
    static bool initialized = false;
    if (!initialized) {
        SFEM_CUDA_CHECK(cudaMemcpyToSymbol(qx, h_qx, n_qp * sizeof(scalar_t)));
        SFEM_CUDA_CHECK(cudaMemcpyToSymbol(qy, h_qy, n_qp * sizeof(scalar_t)));
        SFEM_CUDA_CHECK(cudaMemcpyToSymbol(qz, h_qz, n_qp * sizeof(scalar_t)));
        SFEM_CUDA_CHECK(cudaMemcpyToSymbol(qw, h_qw, n_qp * sizeof(scalar_t)));
        initialized = true;
    }
}

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
    static const int BLOCK_SIZE = LEVEL + 1;
    static const int BLOCK_SIZE_2 = BLOCK_SIZE * BLOCK_SIZE;
    static const int BLOCK_SIZE_3 = BLOCK_SIZE_2 * BLOCK_SIZE;

    // "local" memory
    T u_block[3][BLOCK_SIZE_3];
    T out_block[3][BLOCK_SIZE_3];
    T sub_adjugate[9];
    T sub_determinant;

    const T *g_u[3] = {g_ux, g_uy, g_uz};
    T *g_out[3] = {g_outx, g_outy, g_outz};

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {
        // Gather from global to "local"
        for (int d = 0; d < 3; d++) {
            cu_proteus_hex8_gather<T, LEVEL, T>(
                    nelements, stride, interior_start, e, elements, u_stride, g_u[d], u_block[d]);
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
                    T u[3][8];
                    T out[3][8];
                    int lev[8] = {cu_proteus_hex8_lidx(LEVEL, xi, yi, zi),
                                  cu_proteus_hex8_lidx(LEVEL, xi + 1, yi, zi),
                                  cu_proteus_hex8_lidx(LEVEL, xi + 1, yi + 1, zi),
                                  cu_proteus_hex8_lidx(LEVEL, xi, yi + 1, zi),
                                  cu_proteus_hex8_lidx(LEVEL, xi, yi, zi + 1),
                                  cu_proteus_hex8_lidx(LEVEL, xi + 1, yi, zi + 1),
                                  cu_proteus_hex8_lidx(LEVEL, xi + 1, yi + 1, zi + 1),
                                  cu_proteus_hex8_lidx(LEVEL, xi, yi + 1, zi + 1)};

                    // "local" to micro-buffer
                    for (int d = 0; d < 3; d++) {
                        for (int v = 0; v < 8; v++) {
                            u[d][v] = u_block[d][lev[v]];
                        }
                    }

                    // Reset micro-accumulator
                    for (int d = 0; d < 3; d++) {
                        for (int i = 0; i < 8; i++) {
                            out[d][i] = 0;
                        }
                    }

                    // Compute
                    for (int k = 0; k < n_qp; k++) {
                        cu_hex8_linear_elasticity_apply_adj<T, T>(mu,
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

                    // micro-buffer to "local"
                    for (int d = 0; d < 3; d++) {
                        for (int v = 0; v < 8; v++) {
                            out_block[d][lev[v]] += out[d][v];
                        }
                    }
                }
            }
        }

        // Scatter from "local" to global
        for (int d = 0; d < 3; d++) {
            cu_proteus_hex8_scatter_add<T, LEVEL, T>(nelements,
                                                     stride,
                                                     interior_start,
                                                     e,
                                                     elements,
                                                     out_block[d],
                                                     out_stride,
                                                     g_out[d]);
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
                cu_proteus_affine_hex8_linear_elasticity_apply_local_mem_kernel<T, LEVEL>,
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

    // Global mem
    const T *g_u[3] = {g_ux, g_uy, g_uz};
    T *g_out[3] = {g_outx, g_outy, g_outz};

    // Shared mem
    __shared__ T u_block[BLOCK_SIZE_3];
    __shared__ T out_block[BLOCK_SIZE_3];

    const T h = 1. / LEVEL;

    const auto xi = threadIdx.x;
    const auto yi = threadIdx.y;
    const auto zi = threadIdx.z;
    const int interior = xi > 0 && yi > 0 && zi > 0 && xi < LEVEL && yi < LEVEL && zi < LEVEL;
    const bool is_element = xi < LEVEL && yi < LEVEL && zi < LEVEL;

    assert(xi < BLOCK_SIZE);
    assert(yi < BLOCK_SIZE);
    assert(zi < BLOCK_SIZE);

    const int lidx = cu_proteus_hex8_lidx(LEVEL, xi, yi, zi);

    assert(lidx < BLOCK_SIZE_3);
    assert(lidx >= 0);

    int lev[8];
    if (is_element) {
        lev[0] = cu_proteus_hex8_lidx(LEVEL, xi, yi, zi);
        lev[1] = cu_proteus_hex8_lidx(LEVEL, xi + 1, yi, zi);
        lev[2] = cu_proteus_hex8_lidx(LEVEL, xi + 1, yi + 1, zi);
        lev[3] = cu_proteus_hex8_lidx(LEVEL, xi, yi + 1, zi);
        lev[4] = cu_proteus_hex8_lidx(LEVEL, xi, yi, zi + 1);
        lev[5] = cu_proteus_hex8_lidx(LEVEL, xi + 1, yi, zi + 1);
        lev[6] = cu_proteus_hex8_lidx(LEVEL, xi + 1, yi + 1, zi + 1);
        lev[7] = cu_proteus_hex8_lidx(LEVEL, xi, yi + 1, zi + 1);
    }

    T out[3][8];
    T u[3][8];
    T sub_adjugate[9];
    T sub_determinant;

    for (ptrdiff_t e = blockIdx.x; e < nelements; e += gridDim.x) {
        const ptrdiff_t idx = elements[lidx * stride + e];

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

        out_block[lidx] = 0;

        // Gather
        for (int d = 0; d < 3; d++) {
            u_block[lidx] = g_u[d][idx * u_stride];
            assert(u_block[lidx] == u_block[lidx]);

            __syncthreads();

            if (is_element) {
                for (int v = 0; v < 8; v++) {
                    u[d][v] = u_block[lev[v]];
                }
            }
        }

        // Compute
        if (is_element) {
            cu_hex8_sub_adj_0_in_place(h, sub_adjugate, &sub_determinant);

            for (int d = 0; d < 3; d++) {
                for (int v = 0; v < 8; v++) {
                    out[d][v] = 0;
                }
            }

            for (int k = 0; k < n_qp; k++) {
                cu_hex8_linear_elasticity_apply_adj<T, T>(mu,
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

        // Scatter
        for (int d = 0; d < 3; d++) {
            if (is_element) {
                for (int v = 0; v < 8; v++) {
                    atomicAdd(&out_block[lev[v]], out[d][v]);
                }
            }

            __syncthreads();

            assert(out_block[lidx] == out_block[lidx]);

            if (interior) {
                g_out[d][idx * out_stride] += out_block[lidx];
            } else {
                atomicAdd(&(g_out[d][idx * out_stride]), out_block[lidx]);
            }

            out_block[lidx] = 0;

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

#define my_kernel cu_proteus_affine_hex8_linear_elasticity_apply_warp_tpl
// #define my_kernel cu_proteus_affine_hex8_linear_elasticity_apply_local_mem_tpl
#define my_kernel_large cu_proteus_affine_hex8_linear_elasticity_apply_local_mem_tpl

// Dispatch based on the level
template <typename real_t>
static int cu_proteus_affine_hex8_linear_elasticity_apply_tpl(
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
            return my_kernel<real_t, 2>(nelements,
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
        // case 3: {
        //     return my_kernel<real_t, 3>(nelements,
        //                                 stride,
        //                                 interior_start,
        //                                 elements,
        //                                 (cu_jacobian_t *)jacobian_adjugate,
        //                                 (cu_jacobian_t *)jacobian_determinant,
        //                                 mu,
        //                                 lambda,
        //                                 u_stride,
        //                                 (real_t *)ux,
        //                                 (real_t *)uy,
        //                                 (real_t *)uz,
        //                                 out_stride,
        //                                 (real_t *)outx,
        //                                 (real_t *)outy,
        //                                 (real_t *)outz,
        //                                 stream);
        // }
        case 4: {
            return my_kernel<real_t, 4>(nelements,
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
        // case 5: {
        //     return my_kernel_large<real_t, 5>(nelements,
        //                                       stride,
        //                                       interior_start,
        //                                       elements,
        //                                       (cu_jacobian_t *)jacobian_adjugate,
        //                                       (cu_jacobian_t *)jacobian_determinant,
        //                                       mu,
        //                                       lambda,
        //                                       u_stride,
        //                                       (real_t *)ux,
        //                                       (real_t *)uy,
        //                                       (real_t *)uz,
        //                                       out_stride,
        //                                       (real_t *)outx,
        //                                       (real_t *)outy,
        //                                       (real_t *)outz,
        //                                       stream);
        // }
        // case 6: {
        //     return my_kernel_large<real_t, 6>(nelements,
        //                                       stride,
        //                                       interior_start,
        //                                       elements,
        //                                       (cu_jacobian_t *)jacobian_adjugate,
        //                                       (cu_jacobian_t *)jacobian_determinant,
        //                                       mu,
        //                                       lambda,
        //                                       u_stride,
        //                                       (real_t *)ux,
        //                                       (real_t *)uy,
        //                                       (real_t *)uz,
        //                                       out_stride,
        //                                       (real_t *)outx,
        //                                       (real_t *)outy,
        //                                       (real_t *)outz,
        //                                       stream);
        // }
        // case 7: {
        //     return my_kernel_large<real_t, 7>(nelements,
        //                                       stride,
        //                                       interior_start,
        //                                       elements,
        //                                       (cu_jacobian_t *)jacobian_adjugate,
        //                                       (cu_jacobian_t *)jacobian_determinant,
        //                                       mu,
        //                                       lambda,
        //                                       u_stride,
        //                                       (real_t *)ux,
        //                                       (real_t *)uy,
        //                                       (real_t *)uz,
        //                                       out_stride,
        //                                       (real_t *)outx,
        //                                       (real_t *)outy,
        //                                       (real_t *)outz,
        //                                       stream);
        // }
        // case 8: {
        //     return my_kernel_large<real_t, 8>(nelements,
        //                                       stride,
        //                                       interior_start,
        //                                       elements,
        //                                       (cu_jacobian_t *)jacobian_adjugate,
        //                                       (cu_jacobian_t *)jacobian_determinant,
        //                                       mu,
        //                                       lambda,
        //                                       u_stride,
        //                                       (real_t *)ux,
        //                                       (real_t *)uy,
        //                                       (real_t *)uz,
        //                                       out_stride,
        //                                       (real_t *)outx,
        //                                       (real_t *)outy,
        //                                       (real_t *)outz,
        //                                       stream);
        // }
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
    init_quadrature();

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
