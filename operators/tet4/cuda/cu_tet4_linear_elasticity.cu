#include "cu_tet4_linear_elasticity.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>

#include "sfem_cuda_base.h"
#include "sfem_defs.h"

#include "cu_tet4_inline.hpp"

// #define MIN(a, b) ((a) < (b) ? (a) : (b))
// #define POW2(a) ((a) * (a))

static inline __device__ __host__ void cu_tet4_linear_elasticity_apply_adj(const scalar_t                      mu,
                                                                           const scalar_t                      lambda,
                                                                           const scalar_t *const SFEM_RESTRICT adjugate,
                                                                           const scalar_t jacobian_determinant,
                                                                           const scalar_t *const SFEM_RESTRICT ux,
                                                                           const scalar_t *const SFEM_RESTRICT uy,
                                                                           const scalar_t *const SFEM_RESTRICT uz,
                                                                           accumulator_t *const SFEM_RESTRICT  outx,
                                                                           accumulator_t *const SFEM_RESTRICT  outy,
                                                                           accumulator_t *const SFEM_RESTRICT  outz) {
    // Evaluation of displacement gradient
    scalar_t disp_grad[9];
    {
        const scalar_t x0  = (scalar_t)1.0 / jacobian_determinant;
        const scalar_t x1  = adjugate[0] * x0;
        const scalar_t x2  = adjugate[3] * x0;
        const scalar_t x3  = adjugate[6] * x0;
        const scalar_t x4  = -x1 - x2 - x3;
        const scalar_t x5  = adjugate[1] * x0;
        const scalar_t x6  = adjugate[4] * x0;
        const scalar_t x7  = adjugate[7] * x0;
        const scalar_t x8  = -x5 - x6 - x7;
        const scalar_t x9  = adjugate[2] * x0;
        const scalar_t x10 = adjugate[5] * x0;
        const scalar_t x11 = adjugate[8] * x0;
        const scalar_t x12 = -x10 - x11 - x9;
        // X
        disp_grad[0] = ux[0] * x4 + ux[1] * x1 + ux[2] * x2 + ux[3] * x3;
        disp_grad[1] = ux[0] * x8 + ux[1] * x5 + ux[2] * x6 + ux[3] * x7;
        disp_grad[2] = ux[0] * x12 + ux[1] * x9 + ux[2] * x10 + ux[3] * x11;

        // Y
        disp_grad[3] = uy[0] * x4 + uy[1] * x1 + uy[2] * x2 + uy[3] * x3;
        disp_grad[4] = uy[0] * x8 + uy[1] * x5 + uy[2] * x6 + uy[3] * x7;
        disp_grad[5] = uy[0] * x12 + uy[1] * x9 + uy[2] * x10 + uy[3] * x11;

        // Z
        disp_grad[6] = uz[2] * x2 + uz[3] * x3 + uz[0] * x4 + uz[1] * x1;
        disp_grad[7] = uz[2] * x6 + uz[3] * x7 + uz[0] * x8 + uz[1] * x5;
        disp_grad[8] = uz[2] * x10 + uz[3] * x11 + uz[0] * x12 + uz[1] * x9;
    }

    // We can reuse the buffer to avoid additional register usage
    scalar_t *P = disp_grad;
    {
        const scalar_t x0 = (scalar_t)(1.0 / 3.0) * mu;
        const scalar_t x1 = (scalar_t)(1.0 / 12.0) * lambda * (2 * disp_grad[0] + 2 * disp_grad[4] + 2 * disp_grad[8]);
        const scalar_t x2 = (scalar_t)(1.0 / 6.0) * mu;
        const scalar_t x3 = x2 * (disp_grad[1] + disp_grad[3]);
        const scalar_t x4 = x2 * (disp_grad[2] + disp_grad[6]);
        const scalar_t x5 = x2 * (disp_grad[5] + disp_grad[7]);
        P[0]              = disp_grad[0] * x0 + x1;
        P[1]              = x3;
        P[2]              = x4;
        P[3]              = x3;
        P[4]              = disp_grad[4] * x0 + x1;
        P[5]              = x5;
        P[6]              = x4;
        P[7]              = x5;
        P[8]              = disp_grad[8] * x0 + x1;
    }

    // Bilinear form
    {
        const scalar_t x0 = adjugate[0] + adjugate[3] + adjugate[6];
        const scalar_t x1 = adjugate[1] + adjugate[4] + adjugate[7];
        const scalar_t x2 = adjugate[2] + adjugate[5] + adjugate[8];
        // X
        outx[0] = -P[0] * x0 - P[1] * x1 - P[2] * x2;
        outx[1] = P[0] * adjugate[0] + P[1] * adjugate[1] + P[2] * adjugate[2];
        outx[2] = P[0] * adjugate[3] + P[1] * adjugate[4] + P[2] * adjugate[5];
        outx[3] = P[0] * adjugate[6] + P[1] * adjugate[7] + P[2] * adjugate[8];
        // Y
        outy[0] = -P[3] * x0 - P[4] * x1 - P[5] * x2;
        outy[1] = P[3] * adjugate[0] + P[4] * adjugate[1] + P[5] * adjugate[2];
        outy[2] = P[3] * adjugate[3] + P[4] * adjugate[4] + P[5] * adjugate[5];
        outy[3] = P[3] * adjugate[6] + P[4] * adjugate[7] + P[5] * adjugate[8];
        // Z
        outz[0] = -P[6] * x0 - P[7] * x1 - P[8] * x2;
        outz[1] = P[6] * adjugate[0] + P[7] * adjugate[1] + P[8] * adjugate[2];
        outz[2] = P[6] * adjugate[3] + P[7] * adjugate[4] + P[8] * adjugate[5];
        outz[3] = P[6] * adjugate[6] + P[7] * adjugate[7] + P[8] * adjugate[8];
    }
}

template <typename T>
__global__ void cu_tet4_linear_elasticity_apply_kernel(const ptrdiff_t                          nelements,
                                                       idx_t **const SFEM_RESTRICT              elements,
                                                       const ptrdiff_t                          jacobian_stride,
                                                       const cu_jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate,
                                                       const cu_jacobian_t *const SFEM_RESTRICT g_jacobian_determinant,
                                                       const real_t                             mu,
                                                       const real_t                             lambda,
                                                       const ptrdiff_t                          u_stride,
                                                       const T *const SFEM_RESTRICT             g_ux,
                                                       const T *const SFEM_RESTRICT             g_uy,
                                                       const T *const SFEM_RESTRICT             g_uz,
                                                       const ptrdiff_t                          out_stride,
                                                       T *const SFEM_RESTRICT                   g_outx,
                                                       T *const SFEM_RESTRICT                   g_outy,
                                                       T *const SFEM_RESTRICT                   g_outz) {
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        idx_t ev[4];

        // Sub-geometry
        scalar_t adjugate[9];

        scalar_t ux[4];
        scalar_t uy[4];
        scalar_t uz[4];

        accumulator_t outx[4] = {0};
        accumulator_t outy[4] = {0};
        accumulator_t outz[4] = {0};

        // Copy over jacobian adjugate
        {
            const cu_jacobian_t *const jacobian_adjugate = &g_jacobian_adjugate[e];
            for (int i = 0; i < 9; i++) {
                adjugate[i] = jacobian_adjugate[i * jacobian_stride];
            }
        }

        const scalar_t jacobian_determinant = g_jacobian_determinant[e];

#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elements[v][e];
        }

        for (int v = 0; v < 4; ++v) {
            ux[v] = g_ux[ev[v] * u_stride];
            uy[v] = g_uy[ev[v] * u_stride];
            uz[v] = g_uz[ev[v] * u_stride];
        }

        cu_tet4_linear_elasticity_apply_adj(mu, lambda, adjugate, jacobian_determinant, ux, uy, uz, outx, outy, outz);

        for (int v = 0; v < 4; v++) {
            atomicAdd(&g_outx[ev[v] * out_stride], outx[v]);
            atomicAdd(&g_outy[ev[v] * out_stride], outy[v]);
            atomicAdd(&g_outz[ev[v] * out_stride], outz[v]);
        }
    }
}

template <typename T>
int cu_tet4_linear_elasticity_apply_tpl(const ptrdiff_t                          nelements,
                                        idx_t **const SFEM_RESTRICT              elements,
                                        const ptrdiff_t                          jacobian_stride,
                                        const cu_jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
                                        const cu_jacobian_t *const SFEM_RESTRICT jacobian_determinant,
                                        const real_t                             mu,
                                        const real_t                             lambda,
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
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, cu_tet4_linear_elasticity_apply_kernel<T>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_tet4_linear_elasticity_apply_kernel<<<n_blocks, block_size, 0, s>>>(nelements,
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
        cu_tet4_linear_elasticity_apply_kernel<<<n_blocks, block_size, 0>>>(nelements,
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

extern int cu_tet4_linear_elasticity_apply(const ptrdiff_t                 nelements,
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
    switch (real_type) {
        case SFEM_REAL_DEFAULT: {
            return cu_tet4_linear_elasticity_apply_tpl(nelements,
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
            return cu_tet4_linear_elasticity_apply_tpl(nelements,
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
            return cu_tet4_linear_elasticity_apply_tpl(nelements,
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
                    "[Error] cu_tet4_linear_elasticity_apply: not implemented for type %s "
                    "(code %d)\n",
                    real_type_to_string(real_type),
                    real_type);
            return SFEM_FAILURE;
        }
    }
}

// --- DIAG (TODO)

extern int cu_tet4_linear_elasticity_diag(const ptrdiff_t                 nelements,
                                          idx_t **const SFEM_RESTRICT     elements,
                                          const ptrdiff_t                 jacobian_stride,
                                          const void *const SFEM_RESTRICT jacobian_adjugate,
                                          const void *const SFEM_RESTRICT jacobian_determinant,
                                          const real_t                    mu,
                                          const real_t                    lambda,
                                          const enum RealType             real_type,
                                          const ptrdiff_t                 diag_stride,
                                          void *const SFEM_RESTRICT       diagx,
                                          void *const SFEM_RESTRICT       diagy,
                                          void *const SFEM_RESTRICT       diagz,
                                          void                           *stream) {
    switch (real_type) {
        // case SFEM_REAL_DEFAULT: {
        //     return cu_tet4_linear_elasticity_diag_tpl(nelements,
        //                                                     stride,
        //                                                     elements,
        // jacobian_stride,
        //                                                     (cu_jacobian_t *)jacobian_adjugate,
        //                                                     (cu_jacobian_t
        //                                                     *)jacobian_determinant, mu, lambda,
        //                                                     diag_stride,
        //                                                     (real_t *)diagx,
        //                                                     (real_t *)diagy,
        //                                                     (real_t *)diagz,
        //                                                     stream);
        // }
        // case SFEM_FLOAT32: {
        //     return cu_tet4_linear_elasticity_diag_tpl(nelements,
        //                                                     stride,
        //                                                     elements,
        // jacobian_stride,
        //                                                     (cu_jacobian_t *)jacobian_adjugate,
        //                                                     (cu_jacobian_t
        //                                                     *)jacobian_determinant, mu, lambda,
        //                                                     diag_stride,
        //                                                     (float *)diagx,
        //                                                     (float *)diagy,
        //                                                     (float *)diagz,
        //                                                     stream);
        // }
        // case SFEM_FLOAT64: {
        //     return cu_tet4_linear_elasticity_diag_tpl(nelements,
        //                                                     stride,
        //                                                     elements,
        // jacobian_stride,
        //                                                     (cu_jacobian_t *)jacobian_adjugate,
        //                                                     (cu_jacobian_t
        //                                                     *)jacobian_determinant, mu, lambda,
        //                                                     diag_stride,
        //                                                     (double *)diagx,
        //                                                     (double *)diagy,
        //                                                     (double *)diagz,
        //                                                     stream);
        // }
        default: {
            SFEM_ERROR(
                    "[Error] cu_tet4_linear_elasticity_diag: not implemented for type %s "
                    "(code %d)\n",
                    real_type_to_string(real_type),
                    real_type);
            return SFEM_FAILURE;
        }
    }
}