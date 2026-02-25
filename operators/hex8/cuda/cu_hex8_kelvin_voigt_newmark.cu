#include "cu_hex8_kelvin_voigt_newmark.h"

#include "sfem_cuda_base.h"

#include "cu_hex8_kelvin_voigt_newmark_inline.hpp"

#include <stdio.h>

template <typename T>
__global__ void cu_affine_hex8_kelvin_voigt_newmark_apply_kernel(const ptrdiff_t                          nelements,
                                                                 idx_t **const SFEM_RESTRICT              elements,
                                                                 const ptrdiff_t                          jacobian_stride,
                                                                 const cu_jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate,
                                                                 const cu_jacobian_t *const SFEM_RESTRICT g_jacobian_determinant,
                                                                 const T                                  k,
                                                                 const T                                  K,
                                                                 const T                                  eta,
                                                                 const T                                  rho,
                                                                 const ptrdiff_t                          u_stride,
                                                                 const T *const SFEM_RESTRICT             g_ux,
                                                                 const T *const SFEM_RESTRICT             g_uy,
                                                                 const T *const SFEM_RESTRICT             g_uz,
                                                                 const T *const SFEM_RESTRICT             g_vx,
                                                                 const T *const SFEM_RESTRICT             g_vy,
                                                                 const T *const SFEM_RESTRICT             g_vz,
                                                                 const T *const SFEM_RESTRICT             g_ax,
                                                                 const T *const SFEM_RESTRICT             g_ay,
                                                                 const T *const SFEM_RESTRICT             g_az,
                                                                 const ptrdiff_t                          out_stride,
                                                                 T *const SFEM_RESTRICT                   g_outx,
                                                                 T *const SFEM_RESTRICT                   g_outy,
                                                                 T *const SFEM_RESTRICT                   g_outz) {
    static const int n_qp  = 2;
    static const T   qx[2] = {0.2113248654, 0.7886751346};
    static const T   qw[2] = {1. / 2, 1. / 2};

    // static const int n_qp = 3;
    // static const T qx[3] = {0.1127016654, 1. / 2, 0.8872983346};
    // static const T qw[3] = {0.2777777778, 0.4444444444, 0.2777777778};

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        idx_t ev[8];

        // Sub-geometry
        T adjugate[9];

        T ux[8];
        T uy[8];
        T uz[8];

        T vx[8];
        T vy[8];
        T vz[8];

        T ax[8];
        T ay[8];
        T az[8];

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
                adjugate[i] = jacobian_adjugate[i * jacobian_stride];
            }
        }

        const T jacobian_determinant = g_jacobian_determinant[e];

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][e];
        }

        for (int v = 0; v < 8; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            ux[v]               = g_ux[idx];
            uy[v]               = g_uy[idx];
            uz[v]               = g_uz[idx];
            vx[v]               = g_vx[idx];
            vy[v]               = g_vy[idx];
            vz[v]               = g_vz[idx];
            ax[v]               = g_ax[idx];
            ay[v]               = g_ay[idx];
            az[v]               = g_az[idx];

            assert(ux[v] == ux[v]);
            assert(uy[v] == uy[v]);
            assert(uz[v] == uz[v]);
            assert(vx[v] == vx[v]);
            assert(vy[v] == vy[v]);
            assert(vz[v] == vz[v]);
            assert(ax[v] == ax[v]);
            assert(ay[v] == ay[v]);
            assert(az[v] == az[v]);
        }

        for (int kz = 0; kz < n_qp; kz++) {
            for (int ky = 0; ky < n_qp; ky++) {
                for (int kx = 0; kx < n_qp; kx++) {
                    cu_hex8_kelvin_voigt_newmark_apply_adj<T, T>(k,
                                                                 K,
                                                                 eta,
                                                                 rho,
                                                                 adjugate,
                                                                 jacobian_determinant,
                                                                 qx[kx],
                                                                 qx[ky],
                                                                 qx[kz],
                                                                 qw[kx] * qw[ky] * qw[kz],
                                                                 ux,
                                                                 uy,
                                                                 uz,
                                                                 vx,
                                                                 vy,
                                                                 vz,
                                                                 ax,
                                                                 ay,
                                                                 az,
                                                                 outx,
                                                                 outy,
                                                                 outz);
                }
            }
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
int cu_affine_hex8_kelvin_voigt_newmark_apply_tpl(const ptrdiff_t                          nelements,
                                                  idx_t **const SFEM_RESTRICT              elements,
                                                  const ptrdiff_t                          jacobian_stride,
                                                  const cu_jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
                                                  const cu_jacobian_t *const SFEM_RESTRICT jacobian_determinant,
                                                  const real_t                             k,
                                                  const real_t                             K,
                                                  const real_t                             eta,
                                                  const real_t                             rho,
                                                  const ptrdiff_t                          u_stride,
                                                  const T *const SFEM_RESTRICT             ux,
                                                  const T *const SFEM_RESTRICT             uy,
                                                  const T *const SFEM_RESTRICT             uz,
                                                  const T *const SFEM_RESTRICT             vx,
                                                  const T *const SFEM_RESTRICT             vy,
                                                  const T *const SFEM_RESTRICT             vz,
                                                  const T *const SFEM_RESTRICT             ax,
                                                  const T *const SFEM_RESTRICT             ay,
                                                  const T *const SFEM_RESTRICT             az,
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
                &min_grid_size, &block_size, cu_affine_hex8_kelvin_voigt_newmark_apply_kernel<T>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_affine_hex8_kelvin_voigt_newmark_apply_kernel<T><<<n_blocks, block_size, 0, s>>>(nelements,
                                                                                            elements,
                                                                                            jacobian_stride,
                                                                                            jacobian_adjugate,
                                                                                            jacobian_determinant,
                                                                                            k,
                                                                                            K,
                                                                                            eta,
                                                                                            rho,
                                                                                            u_stride,
                                                                                            ux,
                                                                                            uy,
                                                                                            uz,
                                                                                            vx,
                                                                                            vy,
                                                                                            vz,
                                                                                            ax,
                                                                                            ay,
                                                                                            az,
                                                                                            out_stride,
                                                                                            outx,
                                                                                            outy,
                                                                                            outz);
    } else {
        cu_affine_hex8_kelvin_voigt_newmark_apply_kernel<T><<<n_blocks, block_size, 0>>>(nelements,
                                                                                         elements,
                                                                                         jacobian_stride,
                                                                                         jacobian_adjugate,
                                                                                         jacobian_determinant,
                                                                                         k,
                                                                                         K,
                                                                                         eta,
                                                                                         rho,
                                                                                         u_stride,
                                                                                         ux,
                                                                                         uy,
                                                                                         uz,
                                                                                         vx,
                                                                                         vy,
                                                                                         vz,
                                                                                         ax,
                                                                                         ay,
                                                                                         az,
                                                                                         out_stride,
                                                                                         outx,
                                                                                         outy,
                                                                                         outz);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

extern int cu_affine_hex8_kelvin_voigt_newmark_apply(const ptrdiff_t                 nelements,
                                                     idx_t **const SFEM_RESTRICT     elements,
                                                     const ptrdiff_t                 jacobian_stride,
                                                     const void *const SFEM_RESTRICT jacobian_adjugate,
                                                     const void *const SFEM_RESTRICT jacobian_determinant,
                                                     const real_t                    k,
                                                     const real_t                    K,
                                                     const real_t                    eta,
                                                     const real_t                    rho,
                                                     const enum RealType             real_type,
                                                     const ptrdiff_t                 u_stride,
                                                     const void *const SFEM_RESTRICT ux,
                                                     const void *const SFEM_RESTRICT uy,
                                                     const void *const SFEM_RESTRICT uz,
                                                     const void *const SFEM_RESTRICT vx,
                                                     const void *const SFEM_RESTRICT vy,
                                                     const void *const SFEM_RESTRICT vz,
                                                     const void *const SFEM_RESTRICT ax,
                                                     const void *const SFEM_RESTRICT ay,
                                                     const void *const SFEM_RESTRICT az,
                                                     const ptrdiff_t                 out_stride,
                                                     void *const SFEM_RESTRICT       outx,
                                                     void *const SFEM_RESTRICT       outy,
                                                     void *const SFEM_RESTRICT       outz,
                                                     void                           *stream) {
    switch (real_type) {
        case SFEM_REAL_DEFAULT: {
            return cu_affine_hex8_kelvin_voigt_newmark_apply_tpl(nelements,
                                                                 elements,
                                                                 jacobian_stride,
                                                                 (cu_jacobian_t *)jacobian_adjugate,
                                                                 (cu_jacobian_t *)jacobian_determinant,
                                                                 k,
                                                                 K,
                                                                 eta,
                                                                 rho,
                                                                 u_stride,
                                                                 (real_t *)ux,
                                                                 (real_t *)uy,
                                                                 (real_t *)uz,
                                                                 (real_t *)vx,
                                                                 (real_t *)vy,
                                                                 (real_t *)vz,
                                                                 (real_t *)ax,
                                                                 (real_t *)ay,
                                                                 (real_t *)az,
                                                                 out_stride,
                                                                 (real_t *)outx,
                                                                 (real_t *)outy,
                                                                 (real_t *)outz,
                                                                 stream);
        }
        case SFEM_FLOAT32: {
            return cu_affine_hex8_kelvin_voigt_newmark_apply_tpl(nelements,
                                                                 elements,
                                                                 jacobian_stride,
                                                                 (cu_jacobian_t *)jacobian_adjugate,
                                                                 (cu_jacobian_t *)jacobian_determinant,
                                                                 k,
                                                                 K,
                                                                 eta,
                                                                 rho,
                                                                 u_stride,
                                                                 (float *)ux,
                                                                 (float *)uy,
                                                                 (float *)uz,
                                                                 (float *)vx,
                                                                 (float *)vy,
                                                                 (float *)vz,
                                                                 (float *)ax,
                                                                 (float *)ay,
                                                                 (float *)az,
                                                                 out_stride,
                                                                 (float *)outx,
                                                                 (float *)outy,
                                                                 (float *)outz,
                                                                 stream);
        }
        case SFEM_FLOAT64: {
            return cu_affine_hex8_kelvin_voigt_newmark_apply_tpl(nelements,
                                                                 elements,
                                                                 jacobian_stride,
                                                                 (cu_jacobian_t *)jacobian_adjugate,
                                                                 (cu_jacobian_t *)jacobian_determinant,
                                                                 k,
                                                                 K,
                                                                 eta,
                                                                 rho,
                                                                 u_stride,
                                                                 (double *)ux,
                                                                 (double *)uy,
                                                                 (double *)uz,
                                                                 (double *)vx,
                                                                 (double *)vy,
                                                                 (double *)vz,
                                                                 (double *)ax,
                                                                 (double *)ay,
                                                                 (double *)az,
                                                                 out_stride,
                                                                 (double *)outx,
                                                                 (double *)outy,
                                                                 (double *)outz,
                                                                 stream);
        }
        default: {
            SFEM_ERROR(
                    "[Error] cu_affine_hex8_kelvin_voigt_newmark_apply: not implemented for type %s "
                    "(code %d)\n",
                    real_type_to_string(real_type),
                    real_type);
            return SFEM_FAILURE;
        }
    }
}

/* Removed duplicate overload that incorrectly referenced dt/gamma/beta */