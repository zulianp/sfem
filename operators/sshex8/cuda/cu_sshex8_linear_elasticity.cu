#include "cu_hex8_linear_elasticity_inline.hpp"
#include "cu_sshex8_linear_elasticity.hpp"
#include "sfem_cuda_base.hpp"

#include "sfem_macros.hpp"

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

template <typename T, int LEVEL>
__global__ void cu_affine_sshex8_linear_elasticity_apply_shared_mem_segmented_kernel(
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
    static const int N_MICRO      = LEVEL * LEVEL * LEVEL;

    static const int n_qp  = 2;
    static const T   qx[2] = {0.2113248654, 0.7886751346};
    static const T   qw[2] = {1. / 2, 1. / 2};

    __shared__ T u_block[3][BLOCK_SIZE_3];
    __shared__ T out_block[3][BLOCK_SIZE_3];
    __shared__ T elemental_matrices[9][8 * 8];

    const int thread_id = threadIdx.x;

    for (ptrdiff_t e = blockIdx.x; e < nelements; e += gridDim.x) {
        for (int i = thread_id; i < BLOCK_SIZE_3; i += blockDim.x) {
            const idx_t     node = elements[i][e];
            const ptrdiff_t in   = node * u_stride;

            u_block[0][i]   = g_ux[in];
            u_block[1][i]   = g_uy[in];
            u_block[2][i]   = g_uz[in];
            out_block[0][i] = 0;
            out_block[1][i] = 0;
            out_block[2][i] = 0;
        }

        if (thread_id == 0) {
            T sub_adjugate[9];
            T sub_determinant;

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

            cu_hex8_sub_adj_0_in_place<T>((T)(1. / LEVEL), sub_adjugate, &sub_determinant);

#define BUILD_SEGMENTED_MATRIX_BLOCK(block_idx, fun)               \
    do {                                                           \
        T *const elemental_matrix = elemental_matrices[block_idx]; \
        for (int i = 0; i < 64; i++) {                             \
            elemental_matrix[i] = 0;                               \
        }                                                          \
        SEGEMENTED_QUADRATURE_LOOP(fun);                           \
    } while (0)

            BUILD_SEGMENTED_MATRIX_BLOCK(0, cu_hex8_linear_elasticity_matrix_block_0_0);
            BUILD_SEGMENTED_MATRIX_BLOCK(1, cu_hex8_linear_elasticity_matrix_block_1_0);
            BUILD_SEGMENTED_MATRIX_BLOCK(2, cu_hex8_linear_elasticity_matrix_block_2_0);
            BUILD_SEGMENTED_MATRIX_BLOCK(3, cu_hex8_linear_elasticity_matrix_block_0_1);
            BUILD_SEGMENTED_MATRIX_BLOCK(4, cu_hex8_linear_elasticity_matrix_block_1_1);
            BUILD_SEGMENTED_MATRIX_BLOCK(5, cu_hex8_linear_elasticity_matrix_block_2_1);
            BUILD_SEGMENTED_MATRIX_BLOCK(6, cu_hex8_linear_elasticity_matrix_block_0_2);
            BUILD_SEGMENTED_MATRIX_BLOCK(7, cu_hex8_linear_elasticity_matrix_block_1_2);
            BUILD_SEGMENTED_MATRIX_BLOCK(8, cu_hex8_linear_elasticity_matrix_block_2_2);

#undef BUILD_SEGMENTED_MATRIX_BLOCK
        }

        __syncthreads();

        for (int micro_id = thread_id; micro_id < N_MICRO; micro_id += blockDim.x) {
            const int xi = micro_id % LEVEL;
            const int yi = (micro_id / LEVEL) % LEVEL;
            const int zi = micro_id / (LEVEL * LEVEL);

            const int lev[8] = {cu_sshex8_lidx(LEVEL, xi, yi, zi),
                                cu_sshex8_lidx(LEVEL, xi + 1, yi, zi),
                                cu_sshex8_lidx(LEVEL, xi + 1, yi + 1, zi),
                                cu_sshex8_lidx(LEVEL, xi, yi + 1, zi),
                                cu_sshex8_lidx(LEVEL, xi, yi, zi + 1),
                                cu_sshex8_lidx(LEVEL, xi + 1, yi, zi + 1),
                                cu_sshex8_lidx(LEVEL, xi + 1, yi + 1, zi + 1),
                                cu_sshex8_lidx(LEVEL, xi, yi + 1, zi + 1)};

            T out[3][8];

            for (int d = 0; d < 3; d++) {
                for (int v = 0; v < 8; v++) {
                    out[d][v] = 0;
                }
            }

            for (int c = 0; c < 3; c++) {
                T u[8];

                for (int v = 0; v < 8; v++) {
                    u[v] = u_block[c][lev[v]];
                }

                for (int d = 0; d < 3; d++) {
                    const T *const elemental_matrix = elemental_matrices[c * 3 + d];

                    for (int i = 0; i < 8; i++) {
                        const T *const row = &elemental_matrix[i * 8];
                        const T        ui  = u[i];

                        for (int j = 0; j < 8; j++) {
                            out[d][j] += ui * row[j];
                        }
                    }
                }
            }

            for (int d = 0; d < 3; d++) {
                for (int v = 0; v < 8; v++) {
                    atomicAdd(&out_block[d][lev[v]], out[d][v]);
                }
            }
        }

        __syncthreads();

        for (int i = thread_id; i < BLOCK_SIZE_3; i += blockDim.x) {
            const idx_t     node    = elements[i][e];
            const ptrdiff_t out_idx = node * out_stride;

            atomicAdd(&g_outx[out_idx], out_block[0][i]);
            atomicAdd(&g_outy[out_idx], out_block[1][i]);
            atomicAdd(&g_outz[out_idx], out_block[2][i]);
        }

        __syncthreads();
    }
}

template <typename T>
static __device__ __forceinline__ T cu_sshex8_tp_shape(const int q, const int s) {
    return q == s ? (T)0.78867513459481288225 : (T)0.21132486540518711775;
}

static __device__ __forceinline__ int cu_sshex8_hex_vertex_x(const int v) { return (v + (v >> 1)) & 1; }

static __device__ __forceinline__ int cu_sshex8_hex_vertex_y(const int v) { return (v >> 1) & 1; }

static __device__ __forceinline__ int cu_sshex8_hex_vertex_z(const int v) { return v >> 2; }

static __device__ __forceinline__ int cu_sshex8_hex_vertex_from_offsets(const int x, const int y, const int z) {
    return x + (3 - 2 * x) * y + (z << 2);
}

static __device__ __forceinline__ void cu_sshex8_mma_m8n8k4_f64(const double A, const double B, double &C0, double &C1) {
    asm volatile(
            "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
            "{%0,%1},{%2},{%3},{%4,%5};\n"
            : "=d"(C0), "=d"(C1)
            : "d"(A), "d"(B), "d"(C0), "d"(C1));
}

template <typename T>
static __device__ __forceinline__ void cu_sshex8_tp_ref_grad(const int q, const int node, T *const SFEM_RESTRICT g) {
    const int qx = q & 1;
    const int qy = (q >> 1) & 1;
    const int qz = (q >> 2) & 1;

    const int sx = cu_sshex8_hex_vertex_x(node);
    const int sy = cu_sshex8_hex_vertex_y(node);
    const int sz = cu_sshex8_hex_vertex_z(node);

    const T Sx = cu_sshex8_tp_shape<T>(qx, sx);
    const T Sy = cu_sshex8_tp_shape<T>(qy, sy);
    const T Sz = cu_sshex8_tp_shape<T>(qz, sz);
    const T Gx = sx ? (T)1 : (T)-1;
    const T Gy = sy ? (T)1 : (T)-1;
    const T Gz = sz ? (T)1 : (T)-1;

    const T gx_ref = Gx * Sy * Sz;
    const T gy_ref = Sx * Gy * Sz;
    const T gz_ref = Sx * Sy * Gz;

    g[0] = gx_ref;
    g[1] = gy_ref;
    g[2] = gz_ref;
}

static __device__ __forceinline__ int cu_sshex8_level8_padded_lidx(const int x, const int y, const int z) {
    return z * 90 + y * 10 + x;
}

template <typename T>
__global__ void cu_affine_sshex8_linear_elasticity_apply_level8_gather_kernel(
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
    static const int LEVEL              = 8;
    static const int BLOCK_SIZE         = LEVEL + 1;
    static const int BLOCK_SIZE_2       = BLOCK_SIZE * BLOCK_SIZE;
    static const int BLOCK_SIZE_3       = BLOCK_SIZE_2 * BLOCK_SIZE;
    static const int PADDED_BLOCK_SIZE  = 10;
    static const int PADDED_BLOCK_SIZE2 = PADDED_BLOCK_SIZE * BLOCK_SIZE;
    static const int PADDED_BLOCK_SIZE3 = PADDED_BLOCK_SIZE2 * BLOCK_SIZE;
    static const int N_NODE_PAIRS       = 8 * 8;
    static const int TILE_STRIDE        = 9;
    static const int COMP_PAIR_STRIDE   = 8 * TILE_STRIDE;
    static const int N_TILED_MATRIX     = 3 * 3 * COMP_PAIR_STRIDE;

    __shared__ T u_block[3][PADDED_BLOCK_SIZE3];
    __shared__ T shared_adjugate[9];
    __shared__ T shared_determinant;
    __shared__ T emat[N_TILED_MATRIX];

    const int thread_id = threadIdx.x;
    const int nthreads  = blockDim.x;

    for (ptrdiff_t e = blockIdx.x; e < nelements; e += gridDim.x) {
        for (int i = thread_id; i < BLOCK_SIZE_3; i += nthreads) {
            const int z = i / BLOCK_SIZE_2;
            const int r = i - z * BLOCK_SIZE_2;
            const int y = r / BLOCK_SIZE;
            const int x = r - y * BLOCK_SIZE;

            const idx_t     node = elements[i][e];
            const ptrdiff_t in   = node * u_stride;
            const int       pi   = cu_sshex8_level8_padded_lidx(x, y, z);

            u_block[0][pi] = g_ux[in];
            u_block[1][pi] = g_uy[in];
            u_block[2][pi] = g_uz[in];
        }

        if (thread_id == 0) {
            shared_adjugate[0] = g_jacobian_adjugate[0 * jacobian_stride + e];
            shared_adjugate[1] = g_jacobian_adjugate[1 * jacobian_stride + e];
            shared_adjugate[2] = g_jacobian_adjugate[2 * jacobian_stride + e];
            shared_adjugate[3] = g_jacobian_adjugate[3 * jacobian_stride + e];
            shared_adjugate[4] = g_jacobian_adjugate[4 * jacobian_stride + e];
            shared_adjugate[5] = g_jacobian_adjugate[5 * jacobian_stride + e];
            shared_adjugate[6] = g_jacobian_adjugate[6 * jacobian_stride + e];
            shared_adjugate[7] = g_jacobian_adjugate[7 * jacobian_stride + e];
            shared_adjugate[8] = g_jacobian_adjugate[8 * jacobian_stride + e];
            shared_determinant = g_jacobian_determinant[e];
            cu_hex8_sub_adj_0_in_place<T>((T)(1. / LEVEL), shared_adjugate, &shared_determinant);
        }

        __syncthreads();

        const T determinant = shared_determinant;

        for (int pair = thread_id; pair < N_NODE_PAIRS; pair += nthreads) {
            const int out_vertex = pair >> 3;
            const int in_vertex  = pair & 7;

            T block[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
            for (int q = 0; q < 8; q++) {
                T test_grad[3];
                T trial_grad[3];

                cu_sshex8_tp_ref_grad<T>(q, out_vertex, test_grad);
                cu_sshex8_tp_ref_grad<T>(q, in_vertex, trial_grad);
                cu_linear_elasticity_matrix_block<T>(
                        mu, lambda, shared_adjugate, determinant, (T)0.125, trial_grad, test_grad, block);
            }

#pragma unroll
            for (int out_comp = 0; out_comp < 3; out_comp++) {
#pragma unroll
                for (int in_comp = 0; in_comp < 3; in_comp++) {
                    emat[(out_comp * 3 + in_comp) * COMP_PAIR_STRIDE + out_vertex * TILE_STRIDE + in_vertex] =
                            block[out_comp * 3 + in_comp];
                }
            }
        }

        __syncthreads();

        for (int i = thread_id; i < BLOCK_SIZE_3; i += nthreads) {
            const int z = i / BLOCK_SIZE_2;
            const int r = i - z * BLOCK_SIZE_2;
            const int y = r / BLOCK_SIZE;
            const int x = r - y * BLOCK_SIZE;

            const int xe_begin = x - (x > 0);
            const int xe_end   = x - (x == LEVEL);
            const int ye_begin = y - (y > 0);
            const int ye_end   = y - (y == LEVEL);
            const int ze_begin = z - (z > 0);
            const int ze_end   = z - (z == LEVEL);

            T acc0 = 0;
            T acc1 = 0;
            T acc2 = 0;

            for (int ze = ze_begin; ze <= ze_end; ze++) {
                const int out_z_offset = z - ze;

                for (int ye = ye_begin; ye <= ye_end; ye++) {
                    const int out_y_offset = y - ye;

                    for (int xe = xe_begin; xe <= xe_end; xe++) {
                        const int out_x_offset = x - xe;
                        const int out_vertex   = cu_sshex8_hex_vertex_from_offsets(out_x_offset, out_y_offset, out_z_offset);

#pragma unroll
                        for (int in_vertex = 0; in_vertex < 8; in_vertex++) {
                            const int ix = xe + cu_sshex8_hex_vertex_x(in_vertex);
                            const int iy = ye + cu_sshex8_hex_vertex_y(in_vertex);
                            const int iz = ze + cu_sshex8_hex_vertex_z(in_vertex);
                            const int ip = cu_sshex8_level8_padded_lidx(ix, iy, iz);
                            const T   ux = u_block[0][ip];
                            const T   uy = u_block[1][ip];
                            const T   uz = u_block[2][ip];
                            const int k  = out_vertex * TILE_STRIDE + in_vertex;

                            acc0 += emat[k + 0 * COMP_PAIR_STRIDE] * ux;
                            acc0 += emat[k + 1 * COMP_PAIR_STRIDE] * uy;
                            acc0 += emat[k + 2 * COMP_PAIR_STRIDE] * uz;

                            acc1 += emat[k + 3 * COMP_PAIR_STRIDE] * ux;
                            acc1 += emat[k + 4 * COMP_PAIR_STRIDE] * uy;
                            acc1 += emat[k + 5 * COMP_PAIR_STRIDE] * uz;

                            acc2 += emat[k + 6 * COMP_PAIR_STRIDE] * ux;
                            acc2 += emat[k + 7 * COMP_PAIR_STRIDE] * uy;
                            acc2 += emat[k + 8 * COMP_PAIR_STRIDE] * uz;
                        }
                    }
                }
            }

            const idx_t     node    = elements[i][e];
            const ptrdiff_t out_idx = node * out_stride;

            atomicAdd(&g_outx[out_idx], acc0);
            atomicAdd(&g_outy[out_idx], acc1);
            atomicAdd(&g_outz[out_idx], acc2);
        }

        __syncthreads();
    }
}

template <typename T, int LEVEL>
__global__ void cu_affine_sshex8_linear_elasticity_apply_tensor_product_matrix_kernel(
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
    static const int N_DOF        = 24;
    static const int N_MICRO      = LEVEL * LEVEL * LEVEL;
    static const int N_MATRIX     = N_DOF * N_DOF;
    static const int N_NODE_PAIRS = 8 * 8;

    __shared__ T u_block[3][BLOCK_SIZE_3];
    __shared__ T out_block[3][BLOCK_SIZE_3];
    __shared__ T shared_adjugate[9];
    __shared__ T shared_determinant;
    __shared__ T emat[N_MATRIX];

    const int lidx     = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const int nthreads = blockDim.x * blockDim.y * blockDim.z;

    for (ptrdiff_t e = blockIdx.x; e < nelements; e += gridDim.x) {
        for (int i = lidx; i < BLOCK_SIZE_3; i += nthreads) {
            const idx_t     node = elements[i][e];
            const ptrdiff_t in   = node * u_stride;

            u_block[0][i]   = g_ux[in];
            u_block[1][i]   = g_uy[in];
            u_block[2][i]   = g_uz[in];
            out_block[0][i] = 0;
            out_block[1][i] = 0;
            out_block[2][i] = 0;
        }

        if (lidx == 0) {
            shared_adjugate[0] = g_jacobian_adjugate[0 * jacobian_stride + e];
            shared_adjugate[1] = g_jacobian_adjugate[1 * jacobian_stride + e];
            shared_adjugate[2] = g_jacobian_adjugate[2 * jacobian_stride + e];
            shared_adjugate[3] = g_jacobian_adjugate[3 * jacobian_stride + e];
            shared_adjugate[4] = g_jacobian_adjugate[4 * jacobian_stride + e];
            shared_adjugate[5] = g_jacobian_adjugate[5 * jacobian_stride + e];
            shared_adjugate[6] = g_jacobian_adjugate[6 * jacobian_stride + e];
            shared_adjugate[7] = g_jacobian_adjugate[7 * jacobian_stride + e];
            shared_adjugate[8] = g_jacobian_adjugate[8 * jacobian_stride + e];
            shared_determinant = g_jacobian_determinant[e];
            cu_hex8_sub_adj_0_in_place<T>((T)(1. / LEVEL), shared_adjugate, &shared_determinant);
        }

        __syncthreads();

        const T determinant = shared_determinant;

        for (int pair = lidx; pair < N_NODE_PAIRS; pair += nthreads) {
            const int row_node = pair >> 3;
            const int col_node = pair & 7;

            T block[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
            for (int q = 0; q < 8; q++) {
                T test_grad[3];
                T trial_grad[3];

                cu_sshex8_tp_ref_grad<T>(q, row_node, test_grad);
                cu_sshex8_tp_ref_grad<T>(q, col_node, trial_grad);
                cu_linear_elasticity_matrix_block<T>(
                        mu, lambda, shared_adjugate, determinant, (T)0.125, trial_grad, test_grad, block);
            }

            T *const row0 = &emat[(row_node * 3 + 0) * N_DOF + col_node * 3];
            T *const row1 = &emat[(row_node * 3 + 1) * N_DOF + col_node * 3];
            T *const row2 = &emat[(row_node * 3 + 2) * N_DOF + col_node * 3];

            row0[0] = block[0];
            row0[1] = block[1];
            row0[2] = block[2];
            row1[0] = block[3];
            row1[1] = block[4];
            row1[2] = block[5];
            row2[0] = block[6];
            row2[1] = block[7];
            row2[2] = block[8];
        }

        __syncthreads();

        assert(blockDim.x == 4);
        assert(blockDim.y == 8);

        const int batch_size  = blockDim.y * blockDim.z;
        const int batch_start = threadIdx.z * blockDim.y;
        const int n_rounds    = (N_MICRO + batch_size - 1) / batch_size;

        const int in_vertex_0  = threadIdx.x;
        const int in_vertex_1  = threadIdx.x + 4;
        const int out_vertex   = threadIdx.y;
        const int in_x_offset0 = cu_sshex8_hex_vertex_x(in_vertex_0);
        const int in_y_offset0 = cu_sshex8_hex_vertex_y(in_vertex_0);
        const int in_z_offset0 = cu_sshex8_hex_vertex_z(in_vertex_0);
        const int in_x_offset1 = cu_sshex8_hex_vertex_x(in_vertex_1);
        const int in_y_offset1 = cu_sshex8_hex_vertex_y(in_vertex_1);
        const int in_z_offset1 = cu_sshex8_hex_vertex_z(in_vertex_1);
        const int out_x_offset = cu_sshex8_hex_vertex_x(out_vertex);
        const int out_y_offset = cu_sshex8_hex_vertex_y(out_vertex);
        const int out_z_offset = cu_sshex8_hex_vertex_z(out_vertex);

        for (int r = 0; r < n_rounds; r++) {
            const int in_micro_e  = threadIdx.y + batch_start + r * batch_size;
            const int out_micro_e = 2 * threadIdx.x + batch_start + r * batch_size;

            double u00 = 0;
            double u01 = 0;
            double u02 = 0;
            double u10 = 0;
            double u11 = 0;
            double u12 = 0;

            if (in_micro_e < N_MICRO) {
                const int in_xe = in_micro_e % LEVEL;
                const int in_ye = (in_micro_e / LEVEL) % LEVEL;
                const int in_ze = in_micro_e / (LEVEL * LEVEL);

                const int idx0 =
                        (in_ze + in_z_offset0) * BLOCK_SIZE_2 + (in_ye + in_y_offset0) * BLOCK_SIZE + (in_xe + in_x_offset0);
                const int idx1 =
                        (in_ze + in_z_offset1) * BLOCK_SIZE_2 + (in_ye + in_y_offset1) * BLOCK_SIZE + (in_xe + in_x_offset1);

                u00 = (double)u_block[0][idx0];
                u01 = (double)u_block[1][idx0];
                u02 = (double)u_block[2][idx0];
                u10 = (double)u_block[0][idx1];
                u11 = (double)u_block[1][idx1];
                u12 = (double)u_block[2][idx1];
            }

#pragma unroll
            for (int out_comp = 0; out_comp < 3; out_comp++) {
                double C0 = 0;
                double C1 = 0;

                const int row  = out_vertex * 3 + out_comp;
                const int col0 = in_vertex_0 * 3;
                const int col1 = in_vertex_1 * 3;

                cu_sshex8_mma_m8n8k4_f64((double)emat[row * N_DOF + col0 + 0], u00, C0, C1);
                cu_sshex8_mma_m8n8k4_f64((double)emat[row * N_DOF + col1 + 0], u10, C0, C1);
                cu_sshex8_mma_m8n8k4_f64((double)emat[row * N_DOF + col0 + 1], u01, C0, C1);
                cu_sshex8_mma_m8n8k4_f64((double)emat[row * N_DOF + col1 + 1], u11, C0, C1);
                cu_sshex8_mma_m8n8k4_f64((double)emat[row * N_DOF + col0 + 2], u02, C0, C1);
                cu_sshex8_mma_m8n8k4_f64((double)emat[row * N_DOF + col1 + 2], u12, C0, C1);

                if (out_micro_e < N_MICRO) {
                    const int out_xe = out_micro_e % LEVEL;
                    const int out_ye = (out_micro_e / LEVEL) % LEVEL;
                    const int out_ze = out_micro_e / (LEVEL * LEVEL);
                    const int idx    = (out_ze + out_z_offset) * BLOCK_SIZE_2 + (out_ye + out_y_offset) * BLOCK_SIZE +
                                    (out_xe + out_x_offset);

                    atomicAdd(&out_block[out_comp][idx], (T)C0);
                }

                if (out_micro_e + 1 < N_MICRO) {
                    const int out_xe = (out_micro_e + 1) % LEVEL;
                    const int out_ye = ((out_micro_e + 1) / LEVEL) % LEVEL;
                    const int out_ze = (out_micro_e + 1) / (LEVEL * LEVEL);
                    const int idx    = (out_ze + out_z_offset) * BLOCK_SIZE_2 + (out_ye + out_y_offset) * BLOCK_SIZE +
                                    (out_xe + out_x_offset);

                    atomicAdd(&out_block[out_comp][idx], (T)C1);
                }
            }
        }

        __syncthreads();

        for (int i = lidx; i < BLOCK_SIZE_3; i += nthreads) {
            const idx_t     node    = elements[i][e];
            const ptrdiff_t out_idx = node * out_stride;

            atomicAdd(&g_outx[out_idx], out_block[0][i]);
            atomicAdd(&g_outy[out_idx], out_block[1][i]);
            atomicAdd(&g_outz[out_idx], out_block[2][i]);
        }

        __syncthreads();
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

template <typename T, int LEVEL>
int cu_affine_sshex8_linear_elasticity_apply_local_mem_original_tpl(const ptrdiff_t                          nelements,
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
        cudaOccupancyMaxPotentialBlockSize(
                &min_grid_size, &block_size, cu_affine_sshex8_linear_elasticity_apply_local_mem_kernel<T, LEVEL>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    const ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_affine_sshex8_linear_elasticity_apply_local_mem_kernel<T, LEVEL><<<n_blocks, block_size, 0, s>>>(nelements,
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
        cu_affine_sshex8_linear_elasticity_apply_local_mem_kernel<T, LEVEL><<<n_blocks, block_size, 0>>>(nelements,
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

template <typename T, int LEVEL>
int cu_affine_sshex8_linear_elasticity_apply_shared_mem_segmented_tpl(const ptrdiff_t                          nelements,
                                                                      idx_t **const SFEM_RESTRICT              elements,
                                                                      const ptrdiff_t                          jacobian_stride,
                                                                      const cu_jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
                                                                      const cu_jacobian_t *const SFEM_RESTRICT
                                                                                                   jacobian_determinant,
                                                                      const T                      mu,
                                                                      const T                      lambda,
                                                                      const ptrdiff_t              u_stride,
                                                                      const T *const SFEM_RESTRICT ux,
                                                                      const T *const SFEM_RESTRICT uy,
                                                                      const T *const SFEM_RESTRICT uz,
                                                                      const ptrdiff_t              out_stride,
                                                                      T *const SFEM_RESTRICT       outx,
                                                                      T *const SFEM_RESTRICT       outy,
                                                                      T *const SFEM_RESTRICT       outz,
                                                                      void                        *stream) {
    SFEM_DEBUG_SYNCHRONIZE();

    dim3 block_size(128, 1, 1);
    dim3 n_blocks(MIN(nelements, sfem_cuda_max_grid_dim_x()), 1, 1);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_affine_sshex8_linear_elasticity_apply_shared_mem_segmented_kernel<T, LEVEL>
                <<<n_blocks, block_size, 0, s>>>(nelements,
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
        cu_affine_sshex8_linear_elasticity_apply_shared_mem_segmented_kernel<T, LEVEL>
                <<<n_blocks, block_size, 0>>>(nelements,
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

template <typename T, int LEVEL>
int cu_affine_sshex8_linear_elasticity_apply_tensor_product_tpl(const ptrdiff_t                          nelements,
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

    dim3 block_size(4, 8, 4);
    dim3 n_blocks(MIN(nelements, sfem_cuda_max_grid_dim_x()), 1, 1);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_affine_sshex8_linear_elasticity_apply_tensor_product_matrix_kernel<T, LEVEL>
                <<<n_blocks, block_size, 0, s>>>(nelements,
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
        cu_affine_sshex8_linear_elasticity_apply_tensor_product_matrix_kernel<T, LEVEL>
                <<<n_blocks, block_size, 0>>>(nelements,
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

    SFEM_CUDA_CHECK(cudaPeekAtLastError());
    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

template <typename T, int LEVEL>
int cu_affine_sshex8_linear_elasticity_apply_level8_gather_tpl(const ptrdiff_t                          nelements,
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

    if (LEVEL != 8) {
        SFEM_ERROR("cu_affine_sshex8_linear_elasticity_apply_level8_gather_tpl: level %d not supported!\n", LEVEL);
        return SFEM_FAILURE;
    }

    dim3 block_size(128, 1, 1);
    dim3 n_blocks(MIN(nelements, sfem_cuda_max_grid_dim_x()), 1, 1);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_affine_sshex8_linear_elasticity_apply_level8_gather_kernel<T><<<n_blocks, block_size, 0, s>>>(nelements,
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
        cu_affine_sshex8_linear_elasticity_apply_level8_gather_kernel<T><<<n_blocks, block_size, 0>>>(nelements,
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

    SFEM_CUDA_CHECK(cudaPeekAtLastError());
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

    dim3 block_size(LEVEL, LEVEL, LEVEL);
    dim3 n_blocks(MIN(nelements, sfem_cuda_max_grid_dim_x()), 1, 1);

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
#define my_kernel_segmented cu_affine_sshex8_linear_elasticity_apply_shared_mem_segmented_tpl
#define my_kernel_local_mem_original cu_affine_sshex8_linear_elasticity_apply_local_mem_original_tpl
#define my_kernel_tensor_product cu_affine_sshex8_linear_elasticity_apply_tensor_product_tpl
#define my_kernel_level8_gather cu_affine_sshex8_linear_elasticity_apply_level8_gather_tpl

static int cu_sshex8_linear_elasticity_level8_kernel() {
    static int kernel = -1;

    if (kernel < 0) {
        int SFEM_HEX8_LINEAR_ELASTICITY_LEVEL8_KERNEL = 4;
        SFEM_READ_ENV(SFEM_HEX8_LINEAR_ELASTICITY_LEVEL8_KERNEL, atoi);
        kernel = SFEM_HEX8_LINEAR_ELASTICITY_LEVEL8_KERNEL;
    }

    return kernel;
}

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
            switch (cu_sshex8_linear_elasticity_level8_kernel()) {
                case 0: {
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
                case 2: {
                    return my_kernel<real_t, 8>(nelements,
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
                case 3: {
                    return my_kernel_local_mem_original<real_t, 8>(nelements,
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
                    return my_kernel_tensor_product<real_t, 8>(nelements,
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
                case 5: {
                    return my_kernel_level8_gather<real_t, 8>(nelements,
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
                    return my_kernel_segmented<real_t, 8>(nelements,
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
            }
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
                                                    const enum smesh::PrimitiveType real_type,
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
        case smesh::SMESH_DEFAULT: {
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
        case smesh::SMESH_FLOAT32: {
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
        case smesh::SMESH_FLOAT64: {
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
                    smesh::to_string(real_type),
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
                                                   const enum smesh::PrimitiveType real_type,
                                                   const ptrdiff_t                 out_stride,
                                                   void *const SFEM_RESTRICT       outx,
                                                   void *const SFEM_RESTRICT       outy,
                                                   void *const SFEM_RESTRICT       outz,
                                                   void                           *stream) {
    switch (real_type) {
        case smesh::SMESH_DEFAULT: {
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
        case smesh::SMESH_FLOAT32: {
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
        case smesh::SMESH_FLOAT64: {
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
                    smesh::to_string(real_type),
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
                                                             const enum smesh::PrimitiveType real_type,
                                                             void *const                     out0,
                                                             void *const                     out1,
                                                             void *const                     out2,
                                                             void *const                     out3,
                                                             void *const                     out4,
                                                             void *const                     out5,
                                                             void                           *stream) {
    switch (real_type) {
        case smesh::SMESH_DEFAULT: {
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
        case smesh::SMESH_FLOAT32: {
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
        case smesh::SMESH_FLOAT64: {
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
                    smesh::to_string(real_type),
                    real_type);
            return SFEM_FAILURE;
        }
    }
}
