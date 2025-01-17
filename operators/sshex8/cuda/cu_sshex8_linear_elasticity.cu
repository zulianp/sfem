#include "cu_hex8_linear_elasticity_inline.hpp"
#include "cu_sshex8_linear_elasticity.h"
#include "sfem_cuda_base.h"

#include "cu_hex8_linear_elasticity_inline.hpp"
#include "cu_hex8_linear_elasticity_integral_inline.hpp"
#include "cu_sshex8_inline.hpp"

#ifndef MAX
#define MAX(a, b) ((a) < (b) ? (b) : (a))
#endif

#ifndef MIN
#define MIN(a, b) ((a) > (b) ? (b) : (a))
#endif

#if 0

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

#else

// static const int n_qp = 6;
// static const scalar_t h_qw[6] = {0.16666666666666666666666666666667,
//                                  0.16666666666666666666666666666667,
//                                  0.16666666666666666666666666666667,
//                                  0.16666666666666666666666666666667,
//                                  0.16666666666666666666666666666667,
//                                  0.16666666666666666666666666666667};

// static const scalar_t h_qx[6] = {0.0, 0.5, 0.5, 0.5, 0.5, 1.0};
// static const scalar_t h_qy[6] = {0.5, 0.0, 0.5, 0.5, 1.0, 0.5};
// static const scalar_t h_qz[6] = {0.5, 0.5, 0.0, 1.0, 0.5, 0.5};
// __constant__ scalar_t qx[6];
// __constant__ scalar_t qy[6];
// __constant__ scalar_t qz[6];
// __constant__ scalar_t qw[6];

static const int n_qp = 8;

static const scalar_t h_qw[8] = {1. / 8, 1. / 8, 1. / 8, 1. / 8, 1. / 8, 1. / 8};

static const scalar_t h_qx[8] = {0.2113248654,
                                 0.7886751346,
                                 0.2113248654,
                                 0.7886751346,
                                 0.2113248654,
                                 0.7886751346,
                                 0.2113248654,
                                 0.7886751346};

static const scalar_t h_qy[8] = {0.2113248654,
                                 0.2113248654,
                                 0.7886751346,
                                 0.7886751346,
                                 0.2113248654,
                                 0.2113248654,
                                 0.7886751346,
                                 0.7886751346};

static const scalar_t h_qz[8] = {0.2113248654,
                                 0.2113248654,
                                 0.2113248654,
                                 0.2113248654,
                                 0.7886751346,
                                 0.7886751346,
                                 0.7886751346,
                                 0.7886751346};

__constant__ scalar_t qx[8];
__constant__ scalar_t qy[8];
__constant__ scalar_t qz[8];
__constant__ scalar_t qw[8];

#endif

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
__global__ void cu_affine_sshex8_linear_elasticity_apply_local_mem_kernel(
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const ptrdiff_t interior_start, const idx_t *const SFEM_RESTRICT elements,
        const cu_jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate,
        const cu_jacobian_t *const SFEM_RESTRICT g_jacobian_determinant, const T mu, const T lambda,
        const ptrdiff_t u_stride, const T *const SFEM_RESTRICT g_ux,
        const T *const SFEM_RESTRICT g_uy, const T *const SFEM_RESTRICT g_uz,
        const ptrdiff_t out_stride, T *const SFEM_RESTRICT g_outx, T *const SFEM_RESTRICT g_outy,
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
            cu_sshex8_gather<T, LEVEL, T>(
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
            cu_sshex8_scatter_add<T, LEVEL, T>(nelements,
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
static __host__ __device__ void apply_micro_loop(const T *const elemental_matrix,
                                                 const T *const u_block, T *const out_block) {
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
                    const T ui = u[i];

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
#define SEGEMENTED_QUADRATURE_LOOP(fun)  \
    do                                   \
        for (int k = 0; k < n_qp; k++) { \
            fun<T, T>(mu,                \
                      lambda,            \
                      sub_adjugate,      \
                      sub_determinant,   \
                      qx[k],             \
                      qy[k],             \
                      qz[k],             \
                      qw[k],             \
                      elemental_matrix); \
        }                                \
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
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const ptrdiff_t interior_start, const idx_t *const SFEM_RESTRICT elements,
        const cu_jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate,
        const cu_jacobian_t *const SFEM_RESTRICT g_jacobian_determinant, const T mu, const T lambda,
        const ptrdiff_t u_stride, const T *const SFEM_RESTRICT g_ux,
        const T *const SFEM_RESTRICT g_uy, const T *const SFEM_RESTRICT g_uz,
        const ptrdiff_t out_stride, T *const SFEM_RESTRICT g_outx, T *const SFEM_RESTRICT g_outy,
        T *const SFEM_RESTRICT g_outz) {
    static const int BLOCK_SIZE = LEVEL + 1;
    static const int BLOCK_SIZE_2 = BLOCK_SIZE * BLOCK_SIZE;
    static const int BLOCK_SIZE_3 = BLOCK_SIZE_2 * BLOCK_SIZE;

#ifdef HEX8_SEGMENTED_TENSOR_LOOP
    static const int n_qp = 2;
    static const T qx[2] = {0.2113248654, 0.7886751346};
    static const T qw[2] = {1. / 2, 1. / 2};
#endif

    // "local" memory
    T u_block[BLOCK_SIZE_3];
    T out_block[3][BLOCK_SIZE_3];

    T sub_adjugate[9];
    T sub_determinant;
    T elemental_matrix[8 * 8];

    const T *g_u[3] = {g_ux, g_uy, g_uz};
    T *g_out[3] = {g_outx, g_outy, g_outz};

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {
        // Reset block accumulator
        for (int d = 0; d < 3; d++) {
            for (int i = 0; i < BLOCK_SIZE_3; i++) {
                out_block[d][i] = 0;
            }
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

        {
            const T h = 1. / LEVEL;
            cu_hex8_sub_adj_0_in_place<T>(h, sub_adjugate, &sub_determinant);
        }

        // X
        {
            // Gather from global to "local"
            cu_sshex8_gather<T, LEVEL, T>(
                    nelements, stride, interior_start, e, elements, u_stride, g_u[0], u_block);

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
            cu_sshex8_gather<T, LEVEL, T>(
                    nelements, stride, interior_start, e, elements, u_stride, g_u[1], u_block);

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
            cu_sshex8_gather<T, LEVEL, T>(
                    nelements, stride, interior_start, e, elements, u_stride, g_u[2], u_block);

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
            cu_sshex8_scatter_add<T, LEVEL, T>(nelements,
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

#define local_mem_kernel cu_affine_sshex8_linear_elasticity_apply_local_mem_segmented_kernel
// #define local_mem_kernel cu_affine_sshex8_linear_elasticity_apply_local_mem_kernel

template <typename T, int LEVEL>
int cu_affine_sshex8_linear_elasticity_apply_local_mem_tpl(
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const ptrdiff_t interior_start, const idx_t *const SFEM_RESTRICT elements,
        const cu_jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
        const cu_jacobian_t *const SFEM_RESTRICT jacobian_determinant, const T mu, const T lambda,
        const ptrdiff_t u_stride, const T *const SFEM_RESTRICT ux, const T *const SFEM_RESTRICT uy,
        const T *const SFEM_RESTRICT uz, const ptrdiff_t out_stride, T *const SFEM_RESTRICT outx,
        T *const SFEM_RESTRICT outy, T *const SFEM_RESTRICT outz, void *stream) {
    SFEM_DEBUG_SYNCHRONIZE();

    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(
                &min_grid_size, &block_size, local_mem_kernel<T, LEVEL>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    const ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        local_mem_kernel<T, LEVEL><<<n_blocks, block_size, 0, s>>>(nelements,
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
        local_mem_kernel<T, LEVEL><<<n_blocks, block_size, 0>>>(nelements,
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

#if 0

// #define B_(x, y, z) ((z)*BLOCK_SIZE_2 + (y)*BLOCK_SIZE + (x))

// // Warp with limited shared mem buffer
// template <typename T, int LEVEL>
// __global__ void cu_affine_sshex8_linear_elasticity_apply_warp_kernel(
//         const ptrdiff_t nelements,
//         const ptrdiff_t stride,  // Stride for elements and fff
//         const ptrdiff_t interior_start,
//         const idx_t *const SFEM_RESTRICT elements,
//         const cu_jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate,
//         const cu_jacobian_t *const SFEM_RESTRICT g_jacobian_determinant,
//         const T mu,
//         const T lambda,
//         const ptrdiff_t u_stride,
//         const T *const SFEM_RESTRICT g_ux,
//         const T *const SFEM_RESTRICT g_uy,
//         const T *const SFEM_RESTRICT g_uz,
//         const ptrdiff_t out_stride,
//         T *const SFEM_RESTRICT g_outx,
//         T *const SFEM_RESTRICT g_outy,
//         T *const SFEM_RESTRICT g_outz) {
//     static const int BLOCK_SIZE = LEVEL + 1;
//     static const int BLOCK_SIZE_2 = BLOCK_SIZE * BLOCK_SIZE;
//     static const int BLOCK_SIZE_3 = BLOCK_SIZE_2 * BLOCK_SIZE;

//     assert(blockDim.x == BLOCK_SIZE);
//     assert(blockDim.y == BLOCK_SIZE);
//     assert(blockDim.z == BLOCK_SIZE);

//     // Global mem
//     const T *g_u[3] = {g_ux, g_uy, g_uz};
//     T *g_out[3] = {g_outx, g_outy, g_outz};

//     // Shared mem
//     __shared__ T u_block[BLOCK_SIZE_3];
//     __shared__ T out_block[BLOCK_SIZE_3];

//     const T h = 1. / LEVEL;

//     const auto xi = threadIdx.x;
//     const auto yi = threadIdx.y;
//     const auto zi = threadIdx.z;
//     const int interior = xi > 0 && yi > 0 && zi > 0 && xi < LEVEL && yi < LEVEL && zi < LEVEL;
//     const bool is_element = xi < LEVEL && yi < LEVEL && zi < LEVEL;

//     assert(xi < BLOCK_SIZE);
//     assert(yi < BLOCK_SIZE);
//     assert(zi < BLOCK_SIZE);

//     const int lidx = cu_sshex8_lidx(LEVEL, xi, yi, zi);

//     assert(lidx < BLOCK_SIZE_3);
//     assert(lidx >= 0);

//     int lev[8];
//     if (is_element) {
//         lev[0] = cu_sshex8_lidx(LEVEL, xi, yi, zi);
//         lev[1] = cu_sshex8_lidx(LEVEL, xi + 1, yi, zi);
//         lev[2] = cu_sshex8_lidx(LEVEL, xi + 1, yi + 1, zi);
//         lev[3] = cu_sshex8_lidx(LEVEL, xi, yi + 1, zi);
//         lev[4] = cu_sshex8_lidx(LEVEL, xi, yi, zi + 1);
//         lev[5] = cu_sshex8_lidx(LEVEL, xi + 1, yi, zi + 1);
//         lev[6] = cu_sshex8_lidx(LEVEL, xi + 1, yi + 1, zi + 1);
//         lev[7] = cu_sshex8_lidx(LEVEL, xi, yi + 1, zi + 1);
//     }

//     T out[3][8];
//     T u[3][8];
//     T sub_adjugate[9];
//     T sub_determinant;

//     for (ptrdiff_t e = blockIdx.x; e < nelements; e += gridDim.x) {
//         const ptrdiff_t idx = elements[lidx * stride + e];

//         if (is_element) {
//             sub_adjugate[0] = g_jacobian_adjugate[0 * stride + e];
//             sub_adjugate[1] = g_jacobian_adjugate[1 * stride + e];
//             sub_adjugate[2] = g_jacobian_adjugate[2 * stride + e];
//             sub_adjugate[3] = g_jacobian_adjugate[3 * stride + e];
//             sub_adjugate[4] = g_jacobian_adjugate[4 * stride + e];
//             sub_adjugate[5] = g_jacobian_adjugate[5 * stride + e];
//             sub_adjugate[6] = g_jacobian_adjugate[6 * stride + e];
//             sub_adjugate[7] = g_jacobian_adjugate[7 * stride + e];
//             sub_adjugate[8] = g_jacobian_adjugate[8 * stride + e];
//             sub_determinant = g_jacobian_determinant[e];
//         }

//         out_block[lidx] = 0;

//         // Gather
//         for (int d = 0; d < 3; d++) {
//             u_block[lidx] = g_u[d][idx * u_stride];
//             assert(u_block[lidx] == u_block[lidx]);

//             __syncthreads();

//             if (is_element) {
//                 for (int v = 0; v < 8; v++) {
//                     u[d][v] = u_block[lev[v]];
//                 }
//             }

//             __syncthreads();
//         }

//         // Compute
//         if (is_element) {
//             cu_hex8_sub_adj_0_in_place(h, sub_adjugate, &sub_determinant);

//             for (int d = 0; d < 3; d++) {
//                 for (int v = 0; v < 8; v++) {
//                     out[d][v] = 0;
//                 }
//             }

//             for (int k = 0; k < n_qp; k++) {
//                 cu_hex8_linear_elasticity_apply_adj<T, T>(mu,
//                                                           lambda,
//                                                           sub_adjugate,
//                                                           sub_determinant,
//                                                           qx[k],
//                                                           qy[k],
//                                                           qz[k],
//                                                           qw[k],
//                                                           u[0],
//                                                           u[1],
//                                                           u[2],
//                                                           out[0],
//                                                           out[1],
//                                                           out[2]);
//             }
//         }

//         // Scatter
//         for (int d = 0; d < 3; d++) {
//             if (is_element) {
//                 for (int v = 0; v < 8; v++) {
//                     atomicAdd(&out_block[lev[v]], out[d][v]);
//                 }
//             }

//             __syncthreads();

//             assert(out_block[lidx] == out_block[lidx]);

//             if (interior) {
//                 g_out[d][idx * out_stride] += out_block[lidx];
//             } else {
//                 atomicAdd(&(g_out[d][idx * out_stride]), out_block[lidx]);
//             }

//             out_block[lidx] = 0;

//             // if (d < 2) {
//                 __syncthreads();
//             // }
//         }
//     }
// }


// // Read from global mem
// template <typename T, int LEVEL>
// __global__ void cu_affine_sshex8_linear_elasticity_apply_warp_kernel(
//         const ptrdiff_t nelements,
//         const ptrdiff_t stride,  // Stride for elements and fff
//         const ptrdiff_t interior_start,
//         const idx_t *const SFEM_RESTRICT elements,
//         const cu_jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate,
//         const cu_jacobian_t *const SFEM_RESTRICT g_jacobian_determinant,
//         const T mu,
//         const T lambda,
//         const ptrdiff_t u_stride,
//         const T *const SFEM_RESTRICT g_ux,
//         const T *const SFEM_RESTRICT g_uy,
//         const T *const SFEM_RESTRICT g_uz,
//         const ptrdiff_t out_stride,
//         T *const SFEM_RESTRICT g_outx,
//         T *const SFEM_RESTRICT g_outy,
//         T *const SFEM_RESTRICT g_outz) {
//     static const int BLOCK_SIZE = LEVEL + 1;
//     static const int BLOCK_SIZE_2 = BLOCK_SIZE * BLOCK_SIZE;
//     static const int BLOCK_SIZE_3 = BLOCK_SIZE_2 * BLOCK_SIZE;

//     assert(blockDim.x == BLOCK_SIZE);
//     assert(blockDim.y == BLOCK_SIZE);
//     assert(blockDim.z == BLOCK_SIZE);

//     // Global mem
//     const T *g_u[3] = {g_ux, g_uy, g_uz};
//     T *g_out[3] = {g_outx, g_outy, g_outz};

//     // Shared mem
//     __shared__ T out_block[BLOCK_SIZE_3];

//     const T h = 1. / LEVEL;

//     const auto xi = threadIdx.x;
//     const auto yi = threadIdx.y;
//     const auto zi = threadIdx.z;
//     const int interior = xi > 0 && yi > 0 && zi > 0 && xi < LEVEL && yi < LEVEL && zi < LEVEL;
//     const bool is_element = xi < LEVEL && yi < LEVEL && zi < LEVEL;

//     assert(xi < BLOCK_SIZE);
//     assert(yi < BLOCK_SIZE);
//     assert(zi < BLOCK_SIZE);

//     const int lidx = cu_sshex8_lidx(LEVEL, xi, yi, zi);

//     assert(lidx < BLOCK_SIZE_3);
//     assert(lidx >= 0);

//     int lev[8];
//     if (is_element) {
//         lev[0] = cu_sshex8_lidx(LEVEL, xi, yi, zi);
//         lev[1] = cu_sshex8_lidx(LEVEL, xi + 1, yi, zi);
//         lev[2] = cu_sshex8_lidx(LEVEL, xi + 1, yi + 1, zi);
//         lev[3] = cu_sshex8_lidx(LEVEL, xi, yi + 1, zi);
//         lev[4] = cu_sshex8_lidx(LEVEL, xi, yi, zi + 1);
//         lev[5] = cu_sshex8_lidx(LEVEL, xi + 1, yi, zi + 1);
//         lev[6] = cu_sshex8_lidx(LEVEL, xi + 1, yi + 1, zi + 1);
//         lev[7] = cu_sshex8_lidx(LEVEL, xi, yi + 1, zi + 1);
//     }

//     T out[3][8];
//     T u[3][8];
//     T sub_adjugate[9];
//     T sub_determinant;

//     for (ptrdiff_t e = blockIdx.x; e < nelements; e += gridDim.x) {
//         const ptrdiff_t idx = elements[lidx * stride + e];

//         // Gather
//         if (is_element) {
//             sub_adjugate[0] = g_jacobian_adjugate[0 * stride + e];
//             sub_adjugate[1] = g_jacobian_adjugate[1 * stride + e];
//             sub_adjugate[2] = g_jacobian_adjugate[2 * stride + e];
//             sub_adjugate[3] = g_jacobian_adjugate[3 * stride + e];
//             sub_adjugate[4] = g_jacobian_adjugate[4 * stride + e];
//             sub_adjugate[5] = g_jacobian_adjugate[5 * stride + e];
//             sub_adjugate[6] = g_jacobian_adjugate[6 * stride + e];
//             sub_adjugate[7] = g_jacobian_adjugate[7 * stride + e];
//             sub_adjugate[8] = g_jacobian_adjugate[8 * stride + e];
//             sub_determinant = g_jacobian_determinant[e];

//             for (int d = 0; d < 3; d++) {
//                 for (int v = 0; v < 8; v++) {
//                     u[d][v] = g_u[d][elements[lev[v] * stride + e] * u_stride];
//                 }
//             }
//         }

//         out_block[lidx] = 0;

//         // Compute
//         if (is_element) {
//             cu_hex8_sub_adj_0_in_place(h, sub_adjugate, &sub_determinant);

//             //

//             for (int d = 0; d < 3; d++) {
//                 for (int v = 0; v < 8; v++) {
//                     out[d][v] = 0;
//                 }
//             }

//             for (int k = 0; k < n_qp; k++) {
//                 cu_hex8_linear_elasticity_apply_adj<T, T>(mu,
//                                                           lambda,
//                                                           sub_adjugate,
//                                                           sub_determinant,
//                                                           qx[k],
//                                                           qy[k],
//                                                           qz[k],
//                                                           qw[k],
//                                                           u[0],
//                                                           u[1],
//                                                           u[2],
//                                                           out[0],
//                                                           out[1],
//                                                           out[2]);
//             }
//         }

//         // Scatter
//         for (int d = 0; d < 3; d++) {
//             if (is_element) {
//                 for (int v = 0; v < 8; v++) {
//                     atomicAdd(&out_block[lev[v]], out[d][v]);
//                 }
//             }

//             __syncthreads();

//             assert(out_block[lidx] == out_block[lidx]);

//             if (interior) {
//                 g_out[d][idx * out_stride] += out_block[lidx];
//             } else {
//                 atomicAdd(&(g_out[d][idx * out_stride]), out_block[lidx]);
//             }

//             out_block[lidx] = 0;
//             __syncthreads();
//         }
//     }
// }



template <typename T, int LEVEL>
__global__ void cu_affine_sshex8_linear_elasticity_apply_warp_kernel(
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
    __shared__ T u_block[3][BLOCK_SIZE_3];
    __shared__ T out_block[3][BLOCK_SIZE_3];

    const T h = 1. / LEVEL;

    const auto xi = threadIdx.x;
    const auto yi = threadIdx.y;
    const auto zi = threadIdx.z;
    const int interior = xi > 0 && yi > 0 && zi > 0 && xi < LEVEL && yi < LEVEL && zi < LEVEL;
    const bool is_element = xi < LEVEL && yi < LEVEL && zi < LEVEL;

    assert(xi < BLOCK_SIZE);
    assert(yi < BLOCK_SIZE);
    assert(zi < BLOCK_SIZE);

    const int lidx = cu_sshex8_lidx(LEVEL, xi, yi, zi);

    assert(lidx < BLOCK_SIZE_3);
    assert(lidx >= 0);

    int lev[8];
    if (is_element) {
        lev[0] = cu_sshex8_lidx(LEVEL, xi, yi, zi);
        lev[1] = cu_sshex8_lidx(LEVEL, xi + 1, yi, zi);
        lev[2] = cu_sshex8_lidx(LEVEL, xi + 1, yi + 1, zi);
        lev[3] = cu_sshex8_lidx(LEVEL, xi, yi + 1, zi);
        lev[4] = cu_sshex8_lidx(LEVEL, xi, yi, zi + 1);
        lev[5] = cu_sshex8_lidx(LEVEL, xi + 1, yi, zi + 1);
        lev[6] = cu_sshex8_lidx(LEVEL, xi + 1, yi + 1, zi + 1);
        lev[7] = cu_sshex8_lidx(LEVEL, xi, yi + 1, zi + 1);
    }

    T out[3][8];
    T u[3][8];
    T sub_adjugate[9];
    T sub_determinant;

    for (int d = 0; d < 3; d++) {
        out_block[d][lidx] = 0;
    }

    for (ptrdiff_t e = blockIdx.x; e < nelements; e += gridDim.x) {
        const ptrdiff_t idx = elements[lidx * stride + e];

        // Gather
        for (int d = 0; d < 3; d++) {
            u_block[d][lidx] = g_u[d][idx * u_stride];
            assert(u_block[d][lidx] == u_block[d][lidx]);
        }

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

            cu_hex8_sub_adj_0_in_place(h, sub_adjugate, &sub_determinant);
            for (int d = 0; d < 3; d++) {
                for (int v = 0; v < 8; v++) {
                    out[d][v] = 0;
                }
            }
        }

        __syncthreads();

        // Compute
        if (is_element) {
            for (int d = 0; d < 3; d++) {
                for (int v = 0; v < 8; v++) {
                    u[d][v] = u_block[d][lev[v]];
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

        // Scatter to local mem
        for (int d = 0; d < 3; d++) {
            if (is_element) {
                for (int v = 0; v < 8; v++) {
                    atomicAdd(&out_block[d][lev[v]], out[d][v]);
                }
            }
        }

        __syncthreads();

        // Scatter to global mem
        for (int d = 0; d < 3; d++) {
            assert(out_block[d][lidx] == out_block[d][lidx]);

            if (interior) {
                g_out[d][idx * out_stride] += out_block[d][lidx];
            } else {
                atomicAdd(&(g_out[d][idx * out_stride]), out_block[d][lidx]);
            }

            out_block[d][lidx] = 0;
        }
    }
}

#else

static __device__ inline bool cu_sshex8_is_interior(const int level, const int xi,
                                                          const int yi, const int zi) {
    return xi > 0 && yi > 0 && zi > 0 && xi < level && yi < level && zi < level;
}

template <typename T, int LEVEL>
__global__ void cu_affine_sshex8_linear_elasticity_apply_warp_kernel(
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const ptrdiff_t interior_start, const idx_t *const SFEM_RESTRICT elements,
        const cu_jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate,
        const cu_jacobian_t *const SFEM_RESTRICT g_jacobian_determinant, const T mu, const T lambda,
        const ptrdiff_t u_stride, const T *const SFEM_RESTRICT g_ux,
        const T *const SFEM_RESTRICT g_uy, const T *const SFEM_RESTRICT g_uz,
        const ptrdiff_t out_stride, T *const SFEM_RESTRICT g_outx, T *const SFEM_RESTRICT g_outy,
        T *const SFEM_RESTRICT g_outz) {
    const auto xi = threadIdx.x;
    const auto yi = threadIdx.y;
    const auto zi = threadIdx.z;
    const bool is_element = xi < LEVEL && yi < LEVEL && zi < LEVEL;

#define HEX8_USE_TENSOR_PRODUCT_QUADRATURE
#ifdef HEX8_USE_TENSOR_PRODUCT_QUADRATURE

    static const int n_qp = 2;
    static const T qx[2] = {0.2113248654, 0.7886751346};
    static const T qw[2] = {1. / 2, 1. / 2};

    // static const int n_qp = 3;
    // static const T qx[3] = {0.1127016654, 1. / 2, 0.8872983346};
    // static const T qw[3] = {0.2777777778, 0.4444444444, 0.2777777778};
#endif

    assert(is_element);

    if (is_element) {
        T out[3][8];
        T u[3][8];
        T sub_adjugate[9];
        T sub_determinant;

        for (ptrdiff_t e = blockIdx.x; e < nelements; e += gridDim.x) {
#if 1
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
                ev[v] = elements[ev[v] * stride + e];
            }
#else
            // Something like this can be used when economizing on indices
            ptrdiff_t ev[8];
            int v = 0;
            for (int zz = zi; zz <= zi + 1; zz++) {
                for (int yy = yi; yy <= yi + 1; yy++) {
                    for (int xx = xi; xx <= xi + 1; xx++, v++) {
                        const bool interior = cu_sshex8_is_interior(LEVEL, xx, yy, zz);
                        const int lidx = cu_sshex8_lidx(LEVEL, xx, yy, zz);

                        if (interior) {
                            static const int Lm1 = LEVEL - 1;
                            const int en = (zz - 1) * Lm1 * Lm1 + (yy - 1) * Lm1 + xx - 1;
                            const ptrdiff_t idx = interior_start + e * (Lm1 * Lm1 * Lm1) + en;
                            ev[v] = idx;
                        } else {
                            ev[v] = elements[lidx * stride + e];
                        }
                    }
                }
            }

            assert(v == 8);

            // Proteus to standard!
            {
                ptrdiff_t temp = ev[2];
                ev[2] = ev[3];
                ev[3] = temp;

                temp = ev[6];
                ev[6] = ev[7];
                ev[7] = temp;
            }
#endif

            for (int v = 0; v < 8; v++) {
                ptrdiff_t idx = ev[v] * u_stride;
                u[0][v] = g_ux[idx];
                u[1][v] = g_uy[idx];
                u[2][v] = g_uz[idx];
            }

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

            cu_hex8_sub_adj_0_in_place((T)(1. / LEVEL), sub_adjugate, &sub_determinant);

            for (int d = 0; d < 3; d++) {
                for (int v = 0; v < 8; v++) {
                    out[d][v] = 0;
                }
            }

#ifdef HEX8_USE_TENSOR_PRODUCT_QUADRATURE
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
#else
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

#endif

            for (int v = 0; v < 8; v++) {
                const ptrdiff_t idx = ev[v] * out_stride;
                atomicAdd(&g_outx[idx], out[0][v]);
                atomicAdd(&g_outy[idx], out[1][v]);
                atomicAdd(&g_outz[idx], out[2][v]);
            }
        }
    }
}

#endif

template <typename T, int LEVEL>
int cu_affine_sshex8_linear_elasticity_apply_warp_tpl(
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const ptrdiff_t interior_start, const idx_t *const SFEM_RESTRICT elements,
        const cu_jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
        const cu_jacobian_t *const SFEM_RESTRICT jacobian_determinant, const T mu, const T lambda,
        const ptrdiff_t u_stride, const T *const SFEM_RESTRICT ux, const T *const SFEM_RESTRICT uy,
        const T *const SFEM_RESTRICT uz, const ptrdiff_t out_stride, T *const SFEM_RESTRICT outx,
        T *const SFEM_RESTRICT outy, T *const SFEM_RESTRICT outz, void *stream) {
    SFEM_DEBUG_SYNCHRONIZE();

    // static const int BLOCK_SIZE = LEVEL + 1;

    // dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    // dim3 n_blocks(MIN(nelements, 65535), 1, 1);

    dim3 block_size(LEVEL, LEVEL, LEVEL);
    dim3 n_blocks(MIN(nelements, 65535), 1, 1);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_affine_sshex8_linear_elasticity_apply_warp_kernel<T, LEVEL>
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
        cu_affine_sshex8_linear_elasticity_apply_warp_kernel<T, LEVEL>
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

#define my_kernel cu_affine_sshex8_linear_elasticity_apply_warp_tpl
#define my_kernel_large cu_affine_sshex8_linear_elasticity_apply_warp_tpl
// #define my_kernel cu_affine_sshex8_linear_elasticity_apply_local_mem_tpl
// #define my_kernel_large cu_affine_sshex8_linear_elasticity_apply_local_mem_tpl

// Dispatch based on the level
template <typename real_t>
static int cu_affine_sshex8_linear_elasticity_apply_tpl(
        const int level, const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const ptrdiff_t interior_start, const idx_t *const SFEM_RESTRICT elements,
        const cu_jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
        const cu_jacobian_t *const SFEM_RESTRICT jacobian_determinant, const real_t mu,
        const real_t lambda, const ptrdiff_t u_stride, const real_t *const SFEM_RESTRICT ux,
        const real_t *const SFEM_RESTRICT uy, const real_t *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride, real_t *const SFEM_RESTRICT outx,
        real_t *const SFEM_RESTRICT outy, real_t *const SFEM_RESTRICT outz, void *stream) {
    switch (level) {
        // case 2: {
        //     return my_kernel<real_t, 2>(nelements,
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
        case 3: {
            return my_kernel<real_t, 3>(nelements,
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
        case 5: {
            return my_kernel_large<real_t, 5>(nelements,
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
                    "cu_affine_sshex8_linear_elasticity_apply_tpl: level %d not "
                    "supported!\n",
                    level);
            assert(false);
            return SFEM_FAILURE;
        }
    }
}

extern int cu_affine_sshex8_linear_elasticity_apply(
        const int level, const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const ptrdiff_t interior_start, const idx_t *const SFEM_RESTRICT elements,
        const void *const SFEM_RESTRICT jacobian_adjugate,
        const void *const SFEM_RESTRICT jacobian_determinant, const real_t mu, const real_t lambda,
        const enum RealType real_type, const ptrdiff_t u_stride, const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy, const void *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride, void *const SFEM_RESTRICT outx, void *const SFEM_RESTRICT outy,
        void *const SFEM_RESTRICT outz, void *stream) {
    init_quadrature();

    switch (real_type) {
        case SFEM_REAL_DEFAULT: {
            return cu_affine_sshex8_linear_elasticity_apply_tpl<real_t>(
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
            return cu_affine_sshex8_linear_elasticity_apply_tpl<float>(
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
            return cu_affine_sshex8_linear_elasticity_apply_tpl<double>(
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
                    "[Error] cu_affine_sshex8_linear_elasticity_apply: not implemented "
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

template <typename T>
__global__ void cu_affine_sshex8_linear_elasticity_diag_kernel(
        const int level, const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const ptrdiff_t interior_start, const idx_t *const SFEM_RESTRICT elements,
        const cu_jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
        const cu_jacobian_t *const SFEM_RESTRICT jacobian_determinant, const T mu,
        const T lambda, const ptrdiff_t out_stride, T *const SFEM_RESTRICT outx,
        T *const SFEM_RESTRICT outy, T *const SFEM_RESTRICT outz) {
    T *const out[3] = {outx, outy, outz};

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {
        T linear_elasticity_diag[3 * 8];
        // Build operator
        {
            T sub_adjugate[9];
            T sub_determinant = jacobian_determinant[e];

            for (int d = 0; d < 9; d++) {
                sub_adjugate[d] = jacobian_adjugate[d * stride + e];
            }

            const T h = 1. / level;
            cu_hex8_sub_adj_0_in_place(h, sub_adjugate, &sub_determinant);

            cu_hex8_linear_elasticity_diag<T>(
                    mu, lambda, sub_adjugate, sub_determinant, linear_elasticity_diag);
        }

        // Iterate over sub-elements
        for (int zi = 0; zi < level; zi++) {
            for (int yi = 0; yi < level; yi++) {
                for (int xi = 0; xi < level; xi++) {

                    int ev[8] = {
                            // Bottom
                            elements[cu_sshex8_lidx(level, xi, yi, zi) * stride + e],
                            elements[cu_sshex8_lidx(level, xi + 1, yi, zi) * stride + e],
                            elements[cu_sshex8_lidx(level, xi + 1, yi + 1, zi) * stride + e],
                            elements[cu_sshex8_lidx(level, xi, yi + 1, zi) * stride + e],
                            // Top
                            elements[cu_sshex8_lidx(level, xi, yi, zi + 1) * stride + e],
                            elements[cu_sshex8_lidx(level, xi + 1, yi, zi + 1) * stride + e],
                            elements[cu_sshex8_lidx(level, xi + 1, yi + 1, zi + 1) * stride +
                                     e],
                            elements[cu_sshex8_lidx(level, xi, yi + 1, zi + 1) * stride + e]};

                    for (int d = 0; d < 3; d++) {
                        for (int v = 0; v < 8; v++) {
                            assert(linear_elasticity_diag[d * 8 + v] ==
                                   linear_elasticity_diag[d * 8 + v]);
                            
                            atomicAdd(&out[d][ev[v] * out_stride],
                                      linear_elasticity_diag[d * 8 + v]);
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
static int cu_affine_sshex8_linear_elasticity_diag_tpl(
        const int level, const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const ptrdiff_t interior_start, const idx_t *const SFEM_RESTRICT elements,
        const cu_jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
        const cu_jacobian_t *const SFEM_RESTRICT jacobian_determinant, const T mu, const T lambda,
        const ptrdiff_t out_stride, T *const SFEM_RESTRICT outx, T *const SFEM_RESTRICT outy,
        T *const SFEM_RESTRICT outz, void *stream) {
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(
                &min_grid_size, &block_size, local_mem_kernel<T, LEVEL>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    const ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_affine_sshex8_linear_elasticity_diag_kernel<T>
                <<<n_blocks, block_size, 0, s>>>(level,
                                                 nelements,
                                                 stride,
                                                 interior_start,
                                                 elements,
                                                 jacobian_adjugate,
                                                 jacobian_determinant,
                                                 mu,
                                                 lambda,
                                                 out_stride,
                                                 outx,
                                                 outy,
                                                 outz);
    } else {
        cu_affine_sshex8_linear_elasticity_diag_kernel<T>
                <<<n_blocks, block_size, 0>>>(level,
                                              nelements,
                                              stride,
                                              interior_start,
                                              elements,
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

extern int cu_affine_sshex8_linear_elasticity_diag(
        const int level, const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const ptrdiff_t interior_start, const idx_t *const SFEM_RESTRICT elements,
        const void *const SFEM_RESTRICT jacobian_adjugate,
        const void *const SFEM_RESTRICT jacobian_determinant, const real_t mu, const real_t lambda,
        const enum RealType real_type, const ptrdiff_t out_stride, void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy, void *const SFEM_RESTRICT outz, void *stream) {
    switch (real_type) {
        case SFEM_REAL_DEFAULT: {
            return cu_affine_sshex8_linear_elasticity_diag_tpl<real_t>(
                    level,
                    nelements,
                    stride,
                    interior_start,
                    elements,
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
            return cu_affine_sshex8_linear_elasticity_diag_tpl<float>(
                    level,
                    nelements,
                    stride,
                    interior_start,
                    elements,
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
            return cu_affine_sshex8_linear_elasticity_diag_tpl<double>(
                    level,
                    nelements,
                    stride,
                    interior_start,
                    elements,
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
            fprintf(stderr,
                    "[Error] cu_affine_sshex8_linear_elasticity_diag: not implemented "
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