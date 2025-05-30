#include "cu_hex8_linear_elasticity_inline.hpp"
#include "cu_sshex8_linear_elasticity.h"
#include "sfem_cuda_base.h"

#include "sfem_macros.h"

#include "cu_hex8_linear_elasticity_inline.hpp"
#include "cu_hex8_linear_elasticity_integral_inline.hpp"
#include "cu_hex8_linear_elasticity_matrix_inline.hpp"
#include "cu_sshex8_inline.hpp"

template <typename T, int LEVEL>
__global__ void cu_affine_sshex8_linear_elasticity_apply_local_mem_kernel(
        const ptrdiff_t                          nelements,
        idx_t **const SFEM_RESTRICT              elements,
        const ptrdiff_t                          jacobian_stride,
        const cu_jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate,
        const cu_jacobian_t *const SFEM_RESTRICT g_jacobian_determinant,
        const T                                  mu,
        const T                                  lambda,
        const ptrdiff_t                          u_stride,
        const T *const SFEM_RESTRICT             g_ux,
        const T *const SFEM_RESTRICT             g_uy,
        const T *const SFEM_RESTRICT             g_uz,
        const ptrdiff_t                          out_stride,
        T *const SFEM_RESTRICT                   g_outx,
        T *const SFEM_RESTRICT                   g_outy,
        T *const SFEM_RESTRICT                   g_outz) {
    static const int BLOCK_SIZE   = LEVEL + 1;
    static const int BLOCK_SIZE_2 = BLOCK_SIZE * BLOCK_SIZE;
    static const int BLOCK_SIZE_3 = BLOCK_SIZE_2 * BLOCK_SIZE;

    static const int n_qp  = 2;
    static const T   qx[2] = {0.2113248654, 0.7886751346};
    static const T   qw[2] = {1. / 2, 1. / 2};

    // "local" memory
    T u_block[3][BLOCK_SIZE_3];
    T out_block[3][BLOCK_SIZE_3];
    T sub_adjugate[9];
    T sub_determinant;

    const T *g_u[3]   = {g_ux, g_uy, g_uz};
    T       *g_out[3] = {g_outx, g_outy, g_outz};

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        // Gather from global to "local"
        for (int d = 0; d < 3; d++) {
            cu_sshex8_gather<T, LEVEL, T>(nelements, e, elements, u_stride, g_u[d], u_block[d]);
        }

        // Get geometry
        sub_adjugate[0] = g_jacobian_adjugate[0 * jacobian_stride + e];
        sub_adjugate[1] = g_jacobian_adjugate[1 * jacobian_stride + e];
        sub_adjugate[2] = g_jacobian_adjugate[2 * jacobian_stride + e];
        sub_adjugate[3] = g_jacobian_adjugate[3 * jacobian_stride + e];
        sub_adjugate[4] = g_jacobian_adjugate[4 * jacobian_stride + e];
        sub_adjugate[5] = g_jacobian_adjugate[5 * jacobian_stride + e];
        sub_adjugate[6] = g_jacobian_adjugate[6 * jacobian_stride + e];
        sub_adjugate[7] = g_jacobian_adjugate[7 * jacobian_stride + e];
        sub_adjugate[8] = g_jacobian_adjugate[8 * jacobian_stride + e];
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
                    T   u[3][8];
                    T   out[3][8];
                    int lev[8] = {cu_sshex8_lidx(LEVEL, xi, yi, zi),
                                  cu_sshex8_lidx(LEVEL, xi + 1, yi, zi),
                                  cu_sshex8_lidx(LEVEL, xi + 1, yi + 1, zi),
                                  cu_sshex8_lidx(LEVEL, xi, yi + 1, zi),
                                  cu_sshex8_lidx(LEVEL, xi, yi, zi + 1),
                                  cu_sshex8_lidx(LEVEL, xi + 1, yi, zi + 1),
                                  cu_sshex8_lidx(LEVEL, xi + 1, yi + 1, zi + 1),
                                  cu_sshex8_lidx(LEVEL, xi, yi + 1, zi + 1)};

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
                    // for (int k = 0; k < n_qp; k++) {
                    //     cu_hex8_linear_elasticity_apply_adj<T, T>(mu,
                    //                                               lambda,
                    //                                               sub_adjugate,
                    //                                               sub_determinant,
                    //                                               qx[k],
                    //                                               qy[k],
                    //                                               qz[k],
                    //                                               qw[k],
                    //                                               u[0],
                    //                                               u[1],
                    //                                               u[2],
                    //                                               out[0],
                    //                                               out[1],
                    //                                               out[2]);
                    // }
                    for (int kz = 0; kz < n_qp; kz++) {
                        for (int ky = 0; ky < n_qp; ky++) {
                            for (int kx = 0; kx < n_qp; kx++) {
                                cu_hex8_linear_elasticity_apply_adj<T, T>(mu,
                                                                          lambda,
                                                                          sub_adjugate,
                                                                          sub_determinant,
                                                                          qx[kx],
                                                                          qx[ky],
                                                                          qx[kz],
                                                                          qw[kx] * qw[ky] * qw[kz],
                                                                          u[0],
                                                                          u[1],
                                                                          u[2],
                                                                          out[0],
                                                                          out[1],
                                                                          out[2]);
                            }
                        }
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
            cu_sshex8_scatter_add<T, LEVEL, T>(nelements, e, elements, out_block[d], out_stride, g_out[d]);
        }
    }
}

template <typename T, int LEVEL>
static __host__ __device__ void apply_micro_loop(const T *const elemental_matrix, const T *const u_block, T *const out_block) {
    // Micro-loop
    for (int zi = 0; zi < LEVEL; zi++) {
        for (int yi = 0; yi < LEVEL; yi++) {
            for (int xi = 0; xi < LEVEL; xi++) {
                T u[8];
                T out[8];

                int lev[8] = {cu_sshex8_lidx(LEVEL, xi, yi, zi),
                              cu_sshex8_lidx(LEVEL, xi + 1, yi, zi),
                              cu_sshex8_lidx(LEVEL, xi + 1, yi + 1, zi),
                              cu_sshex8_lidx(LEVEL, xi, yi + 1, zi),
                              cu_sshex8_lidx(LEVEL, xi, yi, zi + 1),
                              cu_sshex8_lidx(LEVEL, xi + 1, yi, zi + 1),
                              cu_sshex8_lidx(LEVEL, xi + 1, yi + 1, zi + 1),
                              cu_sshex8_lidx(LEVEL, xi, yi + 1, zi + 1)};

                // "local" to micro-buffer
                for (int v = 0; v < 8; v++) {
                    u[v] = u_block[lev[v]];
                }

                // Reset micro-accumulator
                for (int i = 0; i < 8; i++) {
                    out[i] = 0;
                }

                // Compute
                for (int i = 0; i < 8; i++) {
                    const T *const row = &elemental_matrix[i * 8];
                    const T        ui  = u[i];

                    for (int j = 0; j < 8; j++) {
                        assert(row[j] == row[j]);
                        out[j] += ui * row[j];
                    }
                }

                // micro-buffer to "local"
                for (int v = 0; v < 8; v++) {
                    out_block[lev[v]] += out[v];
                }
            }
        }
    }
}

// #define HEX8_SEGMENTED_SYMBOLIC

#define HEX8_SEGMENTED_TENSOR_LOOP
#ifndef HEX8_SEGMENTED_TENSOR_LOOP
#define SEGEMENTED_QUADRATURE_LOOP(fun)                                                                         \
    do                                                                                                          \
        for (int k = 0; k < n_qp; k++) {                                                                        \
            fun<T, T>(mu, lambda, sub_adjugate, sub_determinant, qx[k], qy[k], qz[k], qw[k], elemental_matrix); \
        }                                                                                                       \
    while (0)

#else
#define SEGEMENTED_QUADRATURE_LOOP(fun)                 \
    do                                                  \
        for (int kz = 0; kz < n_qp; kz++) {             \
            for (int ky = 0; ky < n_qp; ky++) {         \
                for (int kx = 0; kx < n_qp; kx++) {     \
                    fun<T, T>(mu,                       \
                              lambda,                   \
                              sub_adjugate,             \
                              sub_determinant,          \
                              qx[kx],                   \
                              qx[ky],                   \
                              qx[kz],                   \
                              qw[kx] * qw[ky] * qw[kz], \
                              elemental_matrix);        \
                }                                       \
            }                                           \
        }                                               \
    while (0)
#endif

template <typename T, int LEVEL>
__global__ void cu_affine_sshex8_linear_elasticity_apply_local_mem_segmented_kernel(
        const ptrdiff_t                          nelements,
        idx_t **const SFEM_RESTRICT              elements,
        const ptrdiff_t                          jacobian_stride,
        const cu_jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate,
        const cu_jacobian_t *const SFEM_RESTRICT g_jacobian_determinant,
        const T                                  mu,
        const T                                  lambda,
        const ptrdiff_t                          u_stride,
        const T *const SFEM_RESTRICT             g_ux,
        const T *const SFEM_RESTRICT             g_uy,
        const T *const SFEM_RESTRICT             g_uz,
        const ptrdiff_t                          out_stride,
        T *const SFEM_RESTRICT                   g_outx,
        T *const SFEM_RESTRICT                   g_outy,
        T *const SFEM_RESTRICT                   g_outz) {
    static const int BLOCK_SIZE   = LEVEL + 1;
    static const int BLOCK_SIZE_2 = BLOCK_SIZE * BLOCK_SIZE;
    static const int BLOCK_SIZE_3 = BLOCK_SIZE_2 * BLOCK_SIZE;

#ifdef HEX8_SEGMENTED_TENSOR_LOOP
    static const int n_qp  = 2;
    static const T   qx[2] = {0.2113248654, 0.7886751346};
    static const T   qw[2] = {1. / 2, 1. / 2};
#endif

    // "local" memory
    T u_block[BLOCK_SIZE_3];
    T out_block[3][BLOCK_SIZE_3];

    T sub_adjugate[9];
    T sub_determinant;
    T elemental_matrix[8 * 8];

    const T *g_u[3]   = {g_ux, g_uy, g_uz};
    T       *g_out[3] = {g_outx, g_outy, g_outz};

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        // Reset block accumulator
        for (int d = 0; d < 3; d++) {
            for (int i = 0; i < BLOCK_SIZE_3; i++) {
                out_block[d][i] = 0;
            }
        }

        // Get geometry
        sub_adjugate[0] = g_jacobian_adjugate[0 * jacobian_stride + e];
        sub_adjugate[1] = g_jacobian_adjugate[1 * jacobian_stride + e];
        sub_adjugate[2] = g_jacobian_adjugate[2 * jacobian_stride + e];
        sub_adjugate[3] = g_jacobian_adjugate[3 * jacobian_stride + e];
        sub_adjugate[4] = g_jacobian_adjugate[4 * jacobian_stride + e];
        sub_adjugate[5] = g_jacobian_adjugate[5 * jacobian_stride + e];
        sub_adjugate[6] = g_jacobian_adjugate[6 * jacobian_stride + e];
        sub_adjugate[7] = g_jacobian_adjugate[7 * jacobian_stride + e];
        sub_adjugate[8] = g_jacobian_adjugate[8 * jacobian_stride + e];
        sub_determinant = g_jacobian_determinant[e];

        {
            const T h = 1. / LEVEL;
            cu_hex8_sub_adj_0_in_place<T>(h, sub_adjugate, &sub_determinant);
        }

        // X
        {
            // Gather from global to "local"
            cu_sshex8_gather<T, LEVEL, T>(nelements, e, elements, u_stride, g_u[0], u_block);

#ifdef HEX8_SEGMENTED_SYMBOLIC
            cu_hex8_linear_elasticity_integral_matrix_block_0_0<T, T>(
                    mu, lambda, sub_adjugate, sub_determinant, elemental_matrix);
#else
            for (int i = 0; i < 64; i++) {
                elemental_matrix[i] = 0;
            }
            SEGEMENTED_QUADRATURE_LOOP(cu_hex8_linear_elasticity_matrix_block_0_0);
#endif

            apply_micro_loop<T, LEVEL>(elemental_matrix, u_block, out_block[0]);

#ifdef HEX8_SEGMENTED_SYMBOLIC
            cu_hex8_linear_elasticity_integral_matrix_block_1_0<T, T>(
                    mu, lambda, sub_adjugate, sub_determinant, elemental_matrix);
#else
            for (int i = 0; i < 64; i++) {
                elemental_matrix[i] = 0;
            }
            SEGEMENTED_QUADRATURE_LOOP(cu_hex8_linear_elasticity_matrix_block_1_0);
#endif

            apply_micro_loop<T, LEVEL>(elemental_matrix, u_block, out_block[1]);

#ifdef HEX8_SEGMENTED_SYMBOLIC
            cu_hex8_linear_elasticity_integral_matrix_block_2_0<T, T>(
                    mu, lambda, sub_adjugate, sub_determinant, elemental_matrix);
#else
            for (int i = 0; i < 64; i++) {
                elemental_matrix[i] = 0;
            }
            SEGEMENTED_QUADRATURE_LOOP(cu_hex8_linear_elasticity_matrix_block_2_0);
#endif

            apply_micro_loop<T, LEVEL>(elemental_matrix, u_block, out_block[2]);
        }

        // Y
        {
            // Gather from global to "local"
            cu_sshex8_gather<T, LEVEL, T>(nelements, e, elements, u_stride, g_u[1], u_block);

#ifdef HEX8_SEGMENTED_SYMBOLIC
            cu_hex8_linear_elasticity_integral_matrix_block_0_1<T, T>(
                    mu, lambda, sub_adjugate, sub_determinant, elemental_matrix);
#else
            for (int i = 0; i < 64; i++) {
                elemental_matrix[i] = 0;
            }
            SEGEMENTED_QUADRATURE_LOOP(cu_hex8_linear_elasticity_matrix_block_0_1);
#endif

            apply_micro_loop<T, LEVEL>(elemental_matrix, u_block, out_block[0]);

#ifdef HEX8_SEGMENTED_SYMBOLIC
            cu_hex8_linear_elasticity_integral_matrix_block_1_1<T, T>(
                    mu, lambda, sub_adjugate, sub_determinant, elemental_matrix);
#else
            for (int i = 0; i < 64; i++) {
                elemental_matrix[i] = 0;
            }
            SEGEMENTED_QUADRATURE_LOOP(cu_hex8_linear_elasticity_matrix_block_1_1);
#endif

            apply_micro_loop<T, LEVEL>(elemental_matrix, u_block, out_block[1]);

#ifdef HEX8_SEGMENTED_SYMBOLIC
            cu_hex8_linear_elasticity_integral_matrix_block_2_1<T, T>(
                    mu, lambda, sub_adjugate, sub_determinant, elemental_matrix);
#else
            for (int i = 0; i < 64; i++) {
                elemental_matrix[i] = 0;
            }
            SEGEMENTED_QUADRATURE_LOOP(cu_hex8_linear_elasticity_matrix_block_2_1);
#endif

            apply_micro_loop<T, LEVEL>(elemental_matrix, u_block, out_block[2]);
        }

        // Z
        {
            // Gather from global to "local"
            cu_sshex8_gather<T, LEVEL, T>(nelements, e, elements, u_stride, g_u[2], u_block);

#ifdef HEX8_SEGMENTED_SYMBOLIC
            cu_hex8_linear_elasticity_integral_matrix_block_0_2<T, T>(
                    mu, lambda, sub_adjugate, sub_determinant, elemental_matrix);
#else
            for (int i = 0; i < 64; i++) {
                elemental_matrix[i] = 0;
            }
            SEGEMENTED_QUADRATURE_LOOP(cu_hex8_linear_elasticity_matrix_block_0_2);
#endif

            apply_micro_loop<T, LEVEL>(elemental_matrix, u_block, out_block[0]);

#ifdef HEX8_SEGMENTED_SYMBOLIC
            cu_hex8_linear_elasticity_integral_matrix_block_1_2<T, T>(
                    mu, lambda, sub_adjugate, sub_determinant, elemental_matrix);
#else
            for (int i = 0; i < 64; i++) {
                elemental_matrix[i] = 0;
            }
            SEGEMENTED_QUADRATURE_LOOP(cu_hex8_linear_elasticity_matrix_block_1_2);
#endif

            apply_micro_loop<T, LEVEL>(elemental_matrix, u_block, out_block[1]);

#ifdef HEX8_SEGMENTED_SYMBOLIC
            cu_hex8_linear_elasticity_integral_matrix_block_2_2<T, T>(
                    mu, lambda, sub_adjugate, sub_determinant, elemental_matrix);
#else
            for (int i = 0; i < 64; i++) {
                elemental_matrix[i] = 0;
            }
            SEGEMENTED_QUADRATURE_LOOP(cu_hex8_linear_elasticity_matrix_block_2_2);
#endif

            apply_micro_loop<T, LEVEL>(elemental_matrix, u_block, out_block[2]);
        }

        // // Scatter from "local" to global
        for (int d = 0; d < 3; d++) {
            cu_sshex8_scatter_add<T, LEVEL, T>(nelements, e, elements, out_block[d], out_stride, g_out[d]);
        }
    }
}

#define local_mem_kernel cu_affine_sshex8_linear_elasticity_apply_local_mem_segmented_kernel
// #define local_mem_kernel cu_affine_sshex8_linear_elasticity_apply_local_mem_kernel

template <typename T, int LEVEL>
int cu_affine_sshex8_linear_elasticity_apply_local_mem_tpl(const ptrdiff_t                          nelements,
                                                           idx_t **const SFEM_RESTRICT              elements,
                                                           const ptrdiff_t                          jacobian_stride,
                                                           const cu_jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
                                                           const cu_jacobian_t *const SFEM_RESTRICT jacobian_determinant,
                                                           const T                                  mu,
                                                           const T                                  lambda,
                                                           const ptrdiff_t                          u_stride,
                                                           const T *const SFEM_RESTRICT             ux,
                                                           const T *const SFEM_RESTRICT             uy,
                                                           const T *const SFEM_RESTRICT             uz,
                                                           const ptrdiff_t                          out_stride,
                                                           T *const SFEM_RESTRICT                   outx,
                                                           T *const SFEM_RESTRICT                   outy,
                                                           T *const SFEM_RESTRICT                   outz,
                                                           void                                    *stream) {
    SFEM_DEBUG_SYNCHRONIZE();

    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, local_mem_kernel<T, LEVEL>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    const ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        local_mem_kernel<T, LEVEL><<<n_blocks, block_size, 0, s>>>(nelements,
                                                                   elements,
                                                                   jacobian_stride,
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
        local_mem_kernel<T, LEVEL><<<n_blocks, block_size, 0>>>(nelements,
                                                                elements,
                                                                jacobian_stride,
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

static __device__ inline bool cu_sshex8_is_interior(const int level, const int xi, const int yi, const int zi) {
    return xi > 0 && yi > 0 && zi > 0 && xi < level && yi < level && zi < level;
}

template <typename T, int LEVEL>
__global__ void cu_affine_sshex8_linear_elasticity_apply_warp_kernel(const ptrdiff_t                          nelements,
                                                                     idx_t **const SFEM_RESTRICT              elements,
                                                                     const ptrdiff_t                          jacobian_stride,
                                                                     const cu_jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate,
                                                                     const cu_jacobian_t *const SFEM_RESTRICT
                                                                                                  g_jacobian_determinant,
                                                                     const T                      mu,
                                                                     const T                      lambda,
                                                                     const ptrdiff_t              u_stride,
                                                                     const T *const SFEM_RESTRICT g_ux,
                                                                     const T *const SFEM_RESTRICT g_uy,
                                                                     const T *const SFEM_RESTRICT g_uz,
                                                                     const ptrdiff_t              out_stride,
                                                                     T *const SFEM_RESTRICT       g_outx,
                                                                     T *const SFEM_RESTRICT       g_outy,
                                                                     T *const SFEM_RESTRICT       g_outz) {
    const auto xi         = threadIdx.x;
    const auto yi         = threadIdx.y;
    const auto zi         = threadIdx.z;
    const bool is_element = xi < LEVEL && yi < LEVEL && zi < LEVEL;

    static const int n_qp  = 2;
    static const T   qx[2] = {0.2113248654, 0.7886751346};
    static const T   qw[2] = {1. / 2, 1. / 2};

    assert(is_element);

    if (is_element) {
        T out[3][8];
        T u[3][8];
        T sub_adjugate[9];
        T sub_determinant;

        for (ptrdiff_t e = blockIdx.x; e < nelements; e += gridDim.x) {
            idx_t ev[8];
            ev[0] = cu_sshex8_lidx(LEVEL, xi, yi, zi);
            ev[1] = cu_sshex8_lidx(LEVEL, xi + 1, yi, zi);
            ev[2] = cu_sshex8_lidx(LEVEL, xi + 1, yi + 1, zi);
            ev[3] = cu_sshex8_lidx(LEVEL, xi, yi + 1, zi);
            ev[4] = cu_sshex8_lidx(LEVEL, xi, yi, zi + 1);
            ev[5] = cu_sshex8_lidx(LEVEL, xi + 1, yi, zi + 1);
            ev[6] = cu_sshex8_lidx(LEVEL, xi + 1, yi + 1, zi + 1);
            ev[7] = cu_sshex8_lidx(LEVEL, xi, yi + 1, zi + 1);

            for (int v = 0; v < 8; v++) {
                ev[v] = elements[ev[v]][e];
            }

            for (int v = 0; v < 8; v++) {
                ptrdiff_t idx = ev[v] * u_stride;
                u[0][v]       = g_ux[idx];
                u[1][v]       = g_uy[idx];
                u[2][v]       = g_uz[idx];
            }

            sub_adjugate[0] = g_jacobian_adjugate[0 * jacobian_stride + e];
            sub_adjugate[1] = g_jacobian_adjugate[1 * jacobian_stride + e];
            sub_adjugate[2] = g_jacobian_adjugate[2 * jacobian_stride + e];
            sub_adjugate[3] = g_jacobian_adjugate[3 * jacobian_stride + e];
            sub_adjugate[4] = g_jacobian_adjugate[4 * jacobian_stride + e];
            sub_adjugate[5] = g_jacobian_adjugate[5 * jacobian_stride + e];
            sub_adjugate[6] = g_jacobian_adjugate[6 * jacobian_stride + e];
            sub_adjugate[7] = g_jacobian_adjugate[7 * jacobian_stride + e];
            sub_adjugate[8] = g_jacobian_adjugate[8 * jacobian_stride + e];
            sub_determinant = g_jacobian_determinant[e];

            cu_hex8_sub_adj_0_in_place((T)(1. / LEVEL), sub_adjugate, &sub_determinant);

            for (int d = 0; d < 3; d++) {
                for (int v = 0; v < 8; v++) {
                    out[d][v] = 0;
                }
            }

            for (int kz = 0; kz < n_qp; kz++) {
                for (int ky = 0; ky < n_qp; ky++) {
                    for (int kx = 0; kx < n_qp; kx++) {
                        cu_hex8_linear_elasticity_apply_adj<T, T>(mu,
                                                                  lambda,
                                                                  sub_adjugate,
                                                                  sub_determinant,
                                                                  qx[kx],
                                                                  qx[ky],
                                                                  qx[kz],
                                                                  qw[kx] * qw[ky] * qw[kz],
                                                                  u[0],
                                                                  u[1],
                                                                  u[2],
                                                                  out[0],
                                                                  out[1],
                                                                  out[2]);
                    }
                }
            }

            for (int v = 0; v < 8; v++) {
                const ptrdiff_t idx = ev[v] * out_stride;
                atomicAdd(&g_outx[idx], out[0][v]);
                atomicAdd(&g_outy[idx], out[1][v]);
                atomicAdd(&g_outz[idx], out[2][v]);
            }
        }
    }
}

template <typename T, int LEVEL>
int cu_affine_sshex8_linear_elasticity_apply_warp_tpl(const ptrdiff_t                          nelements,
                                                      idx_t **const SFEM_RESTRICT              elements,
                                                      const ptrdiff_t                          jacobian_stride,
                                                      const cu_jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
                                                      const cu_jacobian_t *const SFEM_RESTRICT jacobian_determinant,
                                                      const T                                  mu,
                                                      const T                                  lambda,
                                                      const ptrdiff_t                          u_stride,
                                                      const T *const SFEM_RESTRICT             ux,
                                                      const T *const SFEM_RESTRICT             uy,
                                                      const T *const SFEM_RESTRICT             uz,
                                                      const ptrdiff_t                          out_stride,
                                                      T *const SFEM_RESTRICT                   outx,
                                                      T *const SFEM_RESTRICT                   outy,
                                                      T *const SFEM_RESTRICT                   outz,
                                                      void                                    *stream) {
    SFEM_DEBUG_SYNCHRONIZE();

    int device_id;
    cudaGetDevice(&device_id);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    dim3 block_size(LEVEL, LEVEL, LEVEL);
    dim3 n_blocks(MIN(nelements, prop.maxGridSize[0]), 1, 1);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_affine_sshex8_linear_elasticity_apply_warp_kernel<T, LEVEL><<<n_blocks, block_size, 0, s>>>(nelements,
                                                                                                       elements,
                                                                                                       jacobian_stride,
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
        cu_affine_sshex8_linear_elasticity_apply_warp_kernel<T, LEVEL><<<n_blocks, block_size, 0>>>(nelements,
                                                                                                    elements,
                                                                                                    jacobian_stride,
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

#define my_kernel cu_affine_sshex8_linear_elasticity_apply_warp_tpl
// #define my_kernel_large cu_affine_sshex8_linear_elasticity_apply_warp_tpl
// #define my_kernel cu_affine_sshex8_linear_elasticity_apply_local_mem_tpl
#define my_kernel_large cu_affine_sshex8_linear_elasticity_apply_local_mem_tpl

// Dispatch based on the level
template <typename real_t>
static int cu_affine_sshex8_linear_elasticity_apply_tpl(const int                                level,
                                                        const ptrdiff_t                          nelements,
                                                        idx_t **const SFEM_RESTRICT              elements,
                                                        const ptrdiff_t                          jacobian_stride,
                                                        const cu_jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
                                                        const cu_jacobian_t *const SFEM_RESTRICT jacobian_determinant,
                                                        const real_t                             mu,
                                                        const real_t                             lambda,
                                                        const ptrdiff_t                          u_stride,
                                                        const real_t *const SFEM_RESTRICT        ux,
                                                        const real_t *const SFEM_RESTRICT        uy,
                                                        const real_t *const SFEM_RESTRICT        uz,
                                                        const ptrdiff_t                          out_stride,
                                                        real_t *const SFEM_RESTRICT              outx,
                                                        real_t *const SFEM_RESTRICT              outy,
                                                        real_t *const SFEM_RESTRICT              outz,
                                                        void                                    *stream) {
    switch (level) {
        case 2: {
            return my_kernel<real_t, 2>(nelements,
                                        elements,
                                        jacobian_stride,
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
            return my_kernel<real_t, 4>(nelements,
                                        elements,
                                        jacobian_stride,
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
            return my_kernel_large<real_t, 8>(nelements,
                                              elements,
                                              jacobian_stride,
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
        case 16: {
            return my_kernel_large<real_t, 16>(nelements,
                                               elements,
                                               jacobian_stride,
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
            SFEM_ERROR(
                    "cu_affine_sshex8_linear_elasticity_apply_tpl: level %d not "
                    "supported!\n",
                    level);
            return SFEM_FAILURE;
        }
    }
}

extern int cu_affine_sshex8_linear_elasticity_apply(const int                       level,
                                                    const ptrdiff_t                 nelements,
                                                    idx_t **const SFEM_RESTRICT     elements,
                                                    const ptrdiff_t                 jacobian_stride,
                                                    const void *const SFEM_RESTRICT jacobian_adjugate,
                                                    const void *const SFEM_RESTRICT jacobian_determinant,
                                                    const real_t                    mu,
                                                    const real_t                    lambda,
                                                    const enum RealType             real_type,
                                                    const ptrdiff_t                 u_stride,
                                                    const void *const SFEM_RESTRICT ux,
                                                    const void *const SFEM_RESTRICT uy,
                                                    const void *const SFEM_RESTRICT uz,
                                                    const ptrdiff_t                 out_stride,
                                                    void *const SFEM_RESTRICT       outx,
                                                    void *const SFEM_RESTRICT       outy,
                                                    void *const SFEM_RESTRICT       outz,
                                                    void                           *stream) {
    // init_quadrature();

    switch (real_type) {
        case SFEM_REAL_DEFAULT: {
            return cu_affine_sshex8_linear_elasticity_apply_tpl<real_t>(level,
                                                                        nelements,
                                                                        elements,
                                                                        jacobian_stride,
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
            return cu_affine_sshex8_linear_elasticity_apply_tpl<float>(level,
                                                                       nelements,
                                                                       elements,
                                                                       jacobian_stride,
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
            return cu_affine_sshex8_linear_elasticity_apply_tpl<double>(level,
                                                                        nelements,
                                                                        elements,
                                                                        jacobian_stride,
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
            SFEM_ERROR(
                    "[Error] cu_affine_sshex8_linear_elasticity_apply: not implemented "
                    "for "
                    "type %s "
                    "(code %d)\n",
                    real_type_to_string(real_type),
                    real_type);
            return SFEM_FAILURE;
        }
    }
}

template <typename T>
__global__ void cu_affine_sshex8_linear_elasticity_diag_kernel(const int                                level,
                                                               const ptrdiff_t                          nelements,
                                                               idx_t **const SFEM_RESTRICT              elements,
                                                               const ptrdiff_t                          jacobian_stride,
                                                               const cu_jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
                                                               const cu_jacobian_t *const SFEM_RESTRICT jacobian_determinant,
                                                               const T                                  mu,
                                                               const T                                  lambda,
                                                               const ptrdiff_t                          out_stride,
                                                               T *const SFEM_RESTRICT                   outx,
                                                               T *const SFEM_RESTRICT                   outy,
                                                               T *const SFEM_RESTRICT                   outz) {
    T *const out[3] = {outx, outy, outz};

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        T linear_elasticity_diag[3 * 8];
        // Build operator
        {
            T sub_adjugate[9];
            T sub_determinant = jacobian_determinant[e];

            for (int d = 0; d < 9; d++) {
                sub_adjugate[d] = jacobian_adjugate[d * jacobian_stride + e];
            }

            const T h = 1. / level;
            cu_hex8_sub_adj_0_in_place(h, sub_adjugate, &sub_determinant);

            cu_hex8_linear_elasticity_diag<T>(mu, lambda, sub_adjugate, sub_determinant, linear_elasticity_diag);
        }

        // Iterate over sub-elements
        for (int zi = 0; zi < level; zi++) {
            for (int yi = 0; yi < level; yi++) {
                for (int xi = 0; xi < level; xi++) {
                    int ev[8] = {// Bottom
                                 elements[cu_sshex8_lidx(level, xi, yi, zi)][e],
                                 elements[cu_sshex8_lidx(level, xi + 1, yi, zi)][e],
                                 elements[cu_sshex8_lidx(level, xi + 1, yi + 1, zi)][e],
                                 elements[cu_sshex8_lidx(level, xi, yi + 1, zi)][e],
                                 // Top
                                 elements[cu_sshex8_lidx(level, xi, yi, zi + 1)][e],
                                 elements[cu_sshex8_lidx(level, xi + 1, yi, zi + 1)][e],
                                 elements[cu_sshex8_lidx(level, xi + 1, yi + 1, zi + 1)][e],
                                 elements[cu_sshex8_lidx(level, xi, yi + 1, zi + 1)][e]};

                    for (int d = 0; d < 3; d++) {
                        for (int v = 0; v < 8; v++) {
                            assert(linear_elasticity_diag[d * 8 + v] == linear_elasticity_diag[d * 8 + v]);

                            atomicAdd(&out[d][ev[v] * out_stride], linear_elasticity_diag[d * 8 + v]);
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
static int cu_affine_sshex8_linear_elasticity_diag_tpl(const int                                level,
                                                       const ptrdiff_t                          nelements,
                                                       idx_t **const SFEM_RESTRICT              elements,
                                                       const ptrdiff_t                          jacobian_stride,
                                                       const cu_jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
                                                       const cu_jacobian_t *const SFEM_RESTRICT jacobian_determinant,
                                                       const T                                  mu,
                                                       const T                                  lambda,
                                                       const ptrdiff_t                          out_stride,
                                                       T *const SFEM_RESTRICT                   outx,
                                                       T *const SFEM_RESTRICT                   outy,
                                                       T *const SFEM_RESTRICT                   outz,
                                                       void                                    *stream) {
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, cu_affine_sshex8_linear_elasticity_diag_kernel<T>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    const ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_affine_sshex8_linear_elasticity_diag_kernel<T><<<n_blocks, block_size, 0, s>>>(level,
                                                                                          nelements,
                                                                                          elements,
                                                                                          jacobian_stride,
                                                                                          jacobian_adjugate,
                                                                                          jacobian_determinant,
                                                                                          mu,
                                                                                          lambda,
                                                                                          out_stride,
                                                                                          outx,
                                                                                          outy,
                                                                                          outz);
    } else {
        cu_affine_sshex8_linear_elasticity_diag_kernel<T><<<n_blocks, block_size, 0>>>(level,
                                                                                       nelements,
                                                                                       elements,
                                                                                       jacobian_stride,
                                                                                       jacobian_adjugate,
                                                                                       jacobian_determinant,
                                                                                       mu,
                                                                                       lambda,
                                                                                       out_stride,
                                                                                       outx,
                                                                                       outy,
                                                                                       outz);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

extern int cu_affine_sshex8_linear_elasticity_diag(const int                       level,
                                                   const ptrdiff_t                 nelements,
                                                   idx_t **const SFEM_RESTRICT     elements,
                                                   const ptrdiff_t                 jacobian_stride,
                                                   const void *const SFEM_RESTRICT jacobian_adjugate,
                                                   const void *const SFEM_RESTRICT jacobian_determinant,
                                                   const real_t                    mu,
                                                   const real_t                    lambda,
                                                   const enum RealType             real_type,
                                                   const ptrdiff_t                 out_stride,
                                                   void *const SFEM_RESTRICT       outx,
                                                   void *const SFEM_RESTRICT       outy,
                                                   void *const SFEM_RESTRICT       outz,
                                                   void                           *stream) {
    switch (real_type) {
        case SFEM_REAL_DEFAULT: {
            return cu_affine_sshex8_linear_elasticity_diag_tpl<real_t>(level,
                                                                       nelements,
                                                                       elements,
                                                                       jacobian_stride,
                                                                       (cu_jacobian_t *)jacobian_adjugate,
                                                                       (cu_jacobian_t *)jacobian_determinant,
                                                                       mu,
                                                                       lambda,
                                                                       out_stride,
                                                                       (real_t *)outx,
                                                                       (real_t *)outy,
                                                                       (real_t *)outz,
                                                                       stream);
        }
        case SFEM_FLOAT32: {
            return cu_affine_sshex8_linear_elasticity_diag_tpl<float>(level,
                                                                      nelements,
                                                                      elements,
                                                                      jacobian_stride,
                                                                      (cu_jacobian_t *)jacobian_adjugate,
                                                                      (cu_jacobian_t *)jacobian_determinant,
                                                                      mu,
                                                                      lambda,
                                                                      out_stride,
                                                                      (float *)outx,
                                                                      (float *)outy,
                                                                      (float *)outz,
                                                                      stream);
        }
        case SFEM_FLOAT64: {
            return cu_affine_sshex8_linear_elasticity_diag_tpl<double>(level,
                                                                       nelements,
                                                                       elements,
                                                                       jacobian_stride,
                                                                       (cu_jacobian_t *)jacobian_adjugate,
                                                                       (cu_jacobian_t *)jacobian_determinant,
                                                                       mu,
                                                                       lambda,
                                                                       out_stride,
                                                                       (double *)outx,
                                                                       (double *)outy,
                                                                       (double *)outz,
                                                                       stream);
        }
        default: {
            SFEM_ERROR(
                    "[Error] cu_affine_sshex8_linear_elasticity_diag: not implemented "
                    "for "
                    "type %s "
                    "(code %d)\n",
                    real_type_to_string(real_type),
                    real_type);
            return SFEM_FAILURE;
        }
    }
}

template <typename T>
__global__ void cu_affine_sshex8_linear_elasticity_block_diag_sym_kernel(
        const int                                level,
        const ptrdiff_t                          nelements,
        idx_t **const SFEM_RESTRICT              elements,
        const ptrdiff_t                          jacobian_stride,
        const cu_jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
        const cu_jacobian_t *const SFEM_RESTRICT jacobian_determinant,
        const T                                  mu,
        const T                                  lambda,
        const ptrdiff_t                          out_stride,
        T *const                                 out0,
        T *const                                 out1,
        T *const                                 out2,
        T *const                                 out3,
        T *const                                 out4,
        T *const                                 out5) {
    static const int n_qp  = 2;
    static const T   qx[2] = {0.2113248654, 0.7886751346};
    static const T   qw[2] = {1. / 2, 1. / 2};

    const int hex8_to_grid_map[8] = {// Bottom
                                     0,
                                     1,
                                     3,
                                     2,
                                     // Top
                                     4,
                                     5,
                                     7,
                                     6};

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        T adjugate[9];

        // Copy over jacobian adjugate
        for (int i = 0; i < 9; i++) {
            adjugate[i] = jacobian_adjugate[i * jacobian_stride + e];
        }

        T determinant = jacobian_determinant[e];

        cu_hex8_sub_adj_0_in_place<T>(1. / level, adjugate, &determinant);

        // Assemble the diagonal part of the matrix
        for (int edof_i = 0; edof_i < 8; edof_i++) {
            T element_matrix[6] = {0, 0, 0, 0, 0, 0};

            // Quadrature
            for (int qzi = 0; qzi < n_qp; qzi++) {
                for (int qyi = 0; qyi < n_qp; qyi++) {
                    for (int qxi = 0; qxi < n_qp; qxi++) {
                        T test_grad[3] = {0, 0, 0};
                        cu_hex8_ref_shape_grad(edof_i, qx[qxi], qx[qyi], qx[qzi], test_grad);
                        cu_linear_elasticity_matrix_sym<T>(mu,
                                                           lambda,
                                                           adjugate,
                                                           determinant,
                                                           test_grad,
                                                           test_grad,
                                                           qw[qxi] * qw[qyi] * qw[qzi],
                                                           element_matrix);
                    }
                }
            }

            const int x_map = hex8_to_grid_map[edof_i] & 1;
            const int y_map = (hex8_to_grid_map[edof_i] >> 1) & 1;
            const int z_map = hex8_to_grid_map[edof_i] >> 2;

            // Iterate over sub-elements
            for (int zi = 0; zi < level; zi++) {
                for (int yi = 0; yi < level; yi++) {
                    for (int xi = 0; xi < level; xi++) {
                        const int lidx = cu_sshex8_lidx(level, xi + x_map, yi + y_map, zi + z_map);
                        // local to global
                        const ptrdiff_t idx = elements[lidx][e] * out_stride;

                        atomicAdd(&out0[idx], element_matrix[0]);
                        atomicAdd(&out1[idx], element_matrix[1]);
                        atomicAdd(&out2[idx], element_matrix[2]);
                        atomicAdd(&out3[idx], element_matrix[3]);
                        atomicAdd(&out4[idx], element_matrix[4]);
                        atomicAdd(&out5[idx], element_matrix[5]);
                    }
                }
            }
        }
    }
}

template <typename T>
int cu_affine_sshex8_linear_elasticity_block_diag_sym_tpl(const int                                level,
                                                          const ptrdiff_t                          nelements,
                                                          idx_t **const SFEM_RESTRICT              elements,
                                                          const ptrdiff_t                          jacobian_stride,
                                                          const cu_jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
                                                          const cu_jacobian_t *const SFEM_RESTRICT jacobian_determinant,
                                                          const real_t                             mu,
                                                          const real_t                             lambda,
                                                          const ptrdiff_t                          out_stride,
                                                          T *const                                 out0,
                                                          T *const                                 out1,
                                                          T *const                                 out2,
                                                          T *const                                 out3,
                                                          T *const                                 out4,
                                                          T *const                                 out5,
                                                          void                                    *stream) {
    SFEM_DEBUG_SYNCHRONIZE();

    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(
                &min_grid_size, &block_size, cu_affine_sshex8_linear_elasticity_block_diag_sym_kernel<T>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_affine_sshex8_linear_elasticity_block_diag_sym_kernel<T><<<n_blocks, block_size, 0, s>>>(level,
                                                                                                    nelements,
                                                                                                    elements,
                                                                                                    jacobian_stride,
                                                                                                    jacobian_adjugate,
                                                                                                    jacobian_determinant,
                                                                                                    mu,
                                                                                                    lambda,
                                                                                                    out_stride,
                                                                                                    out0,
                                                                                                    out1,
                                                                                                    out2,
                                                                                                    out3,
                                                                                                    out4,
                                                                                                    out5);
    } else {
        cu_affine_sshex8_linear_elasticity_block_diag_sym_kernel<T><<<n_blocks, block_size, 0>>>(level,
                                                                                                 nelements,
                                                                                                 elements,
                                                                                                 jacobian_stride,
                                                                                                 jacobian_adjugate,
                                                                                                 jacobian_determinant,
                                                                                                 mu,
                                                                                                 lambda,
                                                                                                 out_stride,
                                                                                                 out0,
                                                                                                 out1,
                                                                                                 out2,
                                                                                                 out3,
                                                                                                 out4,
                                                                                                 out5);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

extern int cu_affine_sshex8_linear_elasticity_block_diag_sym(const int                       level,
                                                             const ptrdiff_t                 nelements,
                                                             idx_t **const SFEM_RESTRICT     elements,
                                                             const ptrdiff_t                 jacobian_stride,
                                                             const void *const SFEM_RESTRICT jacobian_adjugate,
                                                             const void *const SFEM_RESTRICT jacobian_determinant,
                                                             const real_t                    mu,
                                                             const real_t                    lambda,
                                                             const ptrdiff_t                 out_stride,
                                                             const enum RealType             real_type,
                                                             void *const                     out0,
                                                             void *const                     out1,
                                                             void *const                     out2,
                                                             void *const                     out3,
                                                             void *const                     out4,
                                                             void *const                     out5,
                                                             void                           *stream) {
    switch (real_type) {
        case SFEM_REAL_DEFAULT: {
            return cu_affine_sshex8_linear_elasticity_block_diag_sym_tpl<real_t>(level,
                                                                                 nelements,
                                                                                 elements,
                                                                                 jacobian_stride,
                                                                                 (cu_jacobian_t *)jacobian_adjugate,
                                                                                 (cu_jacobian_t *)jacobian_determinant,
                                                                                 mu,
                                                                                 lambda,
                                                                                 out_stride,
                                                                                 (real_t *)out0,
                                                                                 (real_t *)out1,
                                                                                 (real_t *)out2,
                                                                                 (real_t *)out3,
                                                                                 (real_t *)out4,
                                                                                 (real_t *)out5,
                                                                                 stream);
        }
        case SFEM_FLOAT32: {
            return cu_affine_sshex8_linear_elasticity_block_diag_sym_tpl<float>(level,
                                                                                nelements,
                                                                                elements,
                                                                                jacobian_stride,
                                                                                (cu_jacobian_t *)jacobian_adjugate,
                                                                                (cu_jacobian_t *)jacobian_determinant,
                                                                                mu,
                                                                                lambda,
                                                                                out_stride,
                                                                                (float *)out0,
                                                                                (float *)out1,
                                                                                (float *)out2,
                                                                                (float *)out3,
                                                                                (float *)out4,
                                                                                (float *)out5,
                                                                                stream);
        }
        case SFEM_FLOAT64: {
            return cu_affine_sshex8_linear_elasticity_block_diag_sym_tpl<double>(level,
                                                                                 nelements,
                                                                                 elements,
                                                                                 jacobian_stride,
                                                                                 (cu_jacobian_t *)jacobian_adjugate,
                                                                                 (cu_jacobian_t *)jacobian_determinant,
                                                                                 mu,
                                                                                 lambda,
                                                                                 out_stride,
                                                                                 (double *)out0,
                                                                                 (double *)out1,
                                                                                 (double *)out2,
                                                                                 (double *)out3,
                                                                                 (double *)out4,
                                                                                 (double *)out5,
                                                                                 stream);
        }
        default: {
            SFEM_ERROR(
                    "[Error] cu_affine_sshex8_linear_elasticity_block_diag_sym_tpl: not implemented for "
                    "type %s "
                    "(code %d)\n",
                    real_type_to_string(real_type),
                    real_type);
            return SFEM_FAILURE;
        }
    }
}
