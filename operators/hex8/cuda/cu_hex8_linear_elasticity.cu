#include "cu_hex8_linear_elasticity.h"

#include "sfem_cuda_base.h"

#include "cu_hex8_linear_elasticity_inline.hpp"

#include <stdio.h>

#if 1

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

static const int n_qp = 6;
static const scalar_t h_qw[6] = {0.16666666666666666666666666666667,
                                 0.16666666666666666666666666666667,
                                 0.16666666666666666666666666666667,
                                 0.16666666666666666666666666666667,
                                 0.16666666666666666666666666666667,
                                 0.16666666666666666666666666666667};

static const scalar_t h_qx[6] = {0.0, 0.5, 0.5, 0.5, 0.5, 1.0};
static const scalar_t h_qy[6] = {0.5, 0.0, 0.5, 0.5, 1.0, 0.5};
static const scalar_t h_qz[6] = {0.5, 0.5, 0.0, 1.0, 0.5, 0.5};
__constant__ scalar_t qx[6];
__constant__ scalar_t qy[6];
__constant__ scalar_t qz[6];
__constant__ scalar_t qw[6];

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

template <typename T>
__global__ void cu_affine_hex8_linear_elasticity_apply_kernel(
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
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
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {
        idx_t ev[8];

        // Sub-geometry
        T adjugate[9];

        T ux[8];
        T uy[8];
        T uz[8];

        T outx[8];
        T outy[8];
        T outz[8];

        for (int d = 0; d < 8; d++) {
            outx[d] = 0;
            outy[d] = 0;
            outz[d] = 0;
        }

        // Copy over jacobian adjugate
        {
            const cu_jacobian_t *const jacobian_adjugate = &g_jacobian_adjugate[e];
            for (int i = 0; i < 9; i++) {
                adjugate[i] = jacobian_adjugate[i * nelements];
            }
        }

        const T jacobian_determinant = g_jacobian_determinant[e];

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v * nelements + e];
        }

        for (int v = 0; v < 8; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            ux[v] = g_ux[idx];
            uy[v] = g_uy[idx];
            uz[v] = g_uz[idx];

            assert(ux[v] == ux[v]);
            assert(uy[v] == uy[v]);
            assert(uz[v] == uz[v]);
        }

        for (int k = 0; k < n_qp; k++) {
            cu_hex8_linear_elasticity_apply_adj<T, T>(mu,
                                                      lambda,
                                                      adjugate,
                                                      jacobian_determinant,
                                                      qx[k],
                                                      qy[k],
                                                      qz[k],
                                                      qw[k],
                                                      ux,
                                                      uy,
                                                      uz,
                                                      outx,
                                                      outy,
                                                      outz);
        }

        for (int v = 0; v < 8; v++) {
            const ptrdiff_t idx = ev[v] * out_stride;
            assert(outx[v] == outx[v]);
            assert(outy[v] == outy[v]);
            assert(outz[v] == outz[v]);

            atomicAdd(&g_outx[idx], outx[v]);
            atomicAdd(&g_outy[idx], outy[v]);
            atomicAdd(&g_outz[idx], outz[v]);
        }
    }
}

template <typename T>
int cu_affine_hex8_linear_elasticity_apply_tpl(
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const idx_t *const SFEM_RESTRICT elements,
        const cu_jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
        const cu_jacobian_t *const SFEM_RESTRICT jacobian_determinant,
        const real_t mu,
        const real_t lambda,
        const ptrdiff_t u_stride,
        const T *const SFEM_RESTRICT ux,
        const T *const SFEM_RESTRICT uy,
        const T *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        T *const SFEM_RESTRICT outx,
        T *const SFEM_RESTRICT outy,
        T *const SFEM_RESTRICT outz,
        void *stream) {
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size,
                                           &block_size,
                                           cu_affine_hex8_linear_elasticity_apply_kernel<T>,
                                           0,
                                           0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_affine_hex8_linear_elasticity_apply_kernel<T>
                <<<n_blocks, block_size, 0, s>>>(nelements,
                                                 stride,
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
        cu_affine_hex8_linear_elasticity_apply_kernel<T>
                <<<n_blocks, block_size, 0>>>(nelements,
                                              stride,
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

extern int cu_affine_hex8_linear_elasticity_apply(
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
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
            return cu_affine_hex8_linear_elasticity_apply_tpl(nelements,
                                                              stride,
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
            return cu_affine_hex8_linear_elasticity_apply_tpl(nelements,
                                                              stride,
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
            return cu_affine_hex8_linear_elasticity_apply_tpl(nelements,
                                                              stride,
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
                    "[Error] cu_affine_hex8_linear_elasticity_apply: not implemented for type %s "
                    "(code %d)\n",
                    real_type_to_string(real_type),
                    real_type);
            assert(0);
            return SFEM_FAILURE;
        }
    }
}
