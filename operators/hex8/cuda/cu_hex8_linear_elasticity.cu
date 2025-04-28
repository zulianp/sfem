#include "cu_hex8_linear_elasticity.h"

#include "sfem_cuda_base.h"

#include "cu_hex8_linear_elasticity_inline.hpp"
#include "cu_hex8_linear_elasticity_matrix_inline.hpp"

#include <stdio.h>

template <typename T>
__global__ void cu_affine_hex8_linear_elasticity_apply_kernel(const ptrdiff_t nelements,
                                                              const ptrdiff_t stride,  // Stride for elements and fff
                                                              const idx_t *const SFEM_RESTRICT         elements,
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
            ux[v]               = g_ux[idx];
            uy[v]               = g_uy[idx];
            uz[v]               = g_uz[idx];

            assert(ux[v] == ux[v]);
            assert(uy[v] == uy[v]);
            assert(uz[v] == uz[v]);
        }

        for (int kz = 0; kz < n_qp; kz++) {
            for (int ky = 0; ky < n_qp; ky++) {
                for (int kx = 0; kx < n_qp; kx++) {
                    cu_hex8_linear_elasticity_apply_adj<T, T>(mu,
                                                              lambda,
                                                              adjugate,
                                                              jacobian_determinant,
                                                              qx[kx],
                                                              qx[ky],
                                                              qx[kz],
                                                              qw[kx] * qw[ky] * qw[kz],
                                                              ux,
                                                              uy,
                                                              uz,
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
int cu_affine_hex8_linear_elasticity_apply_tpl(const ptrdiff_t                          nelements,
                                               const ptrdiff_t                          stride,  // Stride for elements and fff
                                               const idx_t *const SFEM_RESTRICT         elements,
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
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, cu_affine_hex8_linear_elasticity_apply_kernel<T>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_affine_hex8_linear_elasticity_apply_kernel<T><<<n_blocks, block_size, 0, s>>>(nelements,
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
        cu_affine_hex8_linear_elasticity_apply_kernel<T><<<n_blocks, block_size, 0>>>(nelements,
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

extern int cu_affine_hex8_linear_elasticity_apply(const ptrdiff_t                  nelements,
                                                  const ptrdiff_t                  stride,  // Stride for elements and fff
                                                  const idx_t *const SFEM_RESTRICT elements,
                                                  const void *const SFEM_RESTRICT  jacobian_adjugate,
                                                  const void *const SFEM_RESTRICT  jacobian_determinant,
                                                  const real_t                     mu,
                                                  const real_t                     lambda,
                                                  const enum RealType              real_type,
                                                  const ptrdiff_t                  u_stride,
                                                  const void *const SFEM_RESTRICT  ux,
                                                  const void *const SFEM_RESTRICT  uy,
                                                  const void *const SFEM_RESTRICT  uz,
                                                  const ptrdiff_t                  out_stride,
                                                  void *const SFEM_RESTRICT        outx,
                                                  void *const SFEM_RESTRICT        outy,
                                                  void *const SFEM_RESTRICT        outz,
                                                  void                            *stream) {
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
            SFEM_ERROR(
                    "[Error] cu_affine_hex8_linear_elasticity_apply: not implemented for type %s "
                    "(code %d)\n",
                    real_type_to_string(real_type),
                    real_type);
            return SFEM_FAILURE;
        }
    }
}

template <typename T>
__global__ void cu_affine_hex8_linear_elasticity_bsr_kernel(const ptrdiff_t nelements,
                                                            const ptrdiff_t stride,  // Stride for elements and fff
                                                            const idx_t *const SFEM_RESTRICT         elements,
                                                            const cu_jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate,
                                                            const cu_jacobian_t *const SFEM_RESTRICT g_jacobian_determinant,
                                                            const T                                  mu,
                                                            const T                                  lambda,
                                                            const count_t *const SFEM_RESTRICT       rowptr,
                                                            const idx_t *const SFEM_RESTRICT         colidx,
                                                            T *const SFEM_RESTRICT                   values) {
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

        T block[9];
        for (int d = 0; d < 9; d++) {
            block[d] = 0;
        }

        for (int i = 0; i < 8; i++) {
            const int          lenrow   = rowptr[ev[i] + 1] - rowptr[ev[i]];
            const idx_t *const row      = &colidx[rowptr[ev[i]]];
            T *const           g_blocks = &values[rowptr[ev[i]] * 9];

            int ks[8];
            cu_hex8_find_cols(ev, row, lenrow, ks);

            for (int j = 0; j < 8; j++) {
                T *const g_block = &g_blocks[ks[j] * 9];

                for (int kz = 0; kz < n_qp; kz++) {
                    for (int ky = 0; ky < n_qp; ky++) {
                        for (int kx = 0; kx < n_qp; kx++) {
                            T trial_grad[3], test_grad[3];
                            cu_hex8_ref_shape_grad(i, qx[kx], qx[ky], qx[kz], test_grad);
                            cu_hex8_ref_shape_grad(j, qx[kx], qx[ky], qx[kz], trial_grad);
                            cu_linear_elasticity_matrix_block(mu,
                                                              lambda,
                                                              adjugate,
                                                              jacobian_determinant,
                                                              qw[kx] * qw[ky] * qw[kz],
                                                              trial_grad,
                                                              test_grad,
                                                              block);
                        }
                    }
                }

                for (int d = 0; d < 9; d++) {
                    atomicAdd(&g_block[d], block[d]);
                }
            }
        }
    }
}

template <typename T>
int cu_affine_hex8_linear_elasticity_bsr_tpl(const ptrdiff_t                          nelements,
                                             const ptrdiff_t                          stride,
                                             const idx_t *const SFEM_RESTRICT         elements,
                                             const cu_jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
                                             const cu_jacobian_t *const SFEM_RESTRICT jacobian_determinant,
                                             const real_t                             mu,
                                             const real_t                             lambda,
                                             const count_t *const SFEM_RESTRICT       rowptr,
                                             const idx_t *const SFEM_RESTRICT         colidx,
                                             T *const SFEM_RESTRICT                   values,
                                             void                                    *stream) {
    SFEM_DEBUG_SYNCHRONIZE();

    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, cu_affine_hex8_linear_elasticity_bsr_kernel<T>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_affine_hex8_linear_elasticity_bsr_kernel<T><<<n_blocks, block_size, 0, s>>>(
                nelements, stride, elements, jacobian_adjugate, jacobian_determinant, mu, lambda, rowptr, colidx, values);
    } else {
        cu_affine_hex8_linear_elasticity_bsr_kernel<T><<<n_blocks, block_size, 0>>>(
                nelements, stride, elements, jacobian_adjugate, jacobian_determinant, mu, lambda, rowptr, colidx, values);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

extern int cu_affine_hex8_linear_elasticity_bsr(const ptrdiff_t                    nelements,
                                                const ptrdiff_t                    stride,
                                                const idx_t *const SFEM_RESTRICT   elements,
                                                const void *const SFEM_RESTRICT    jacobian_adjugate,
                                                const void *const SFEM_RESTRICT    jacobian_determinant,
                                                const real_t                       mu,
                                                const real_t                       lambda,
                                                const enum RealType                real_type,
                                                const count_t *const SFEM_RESTRICT rowptr,
                                                const idx_t *const SFEM_RESTRICT   colidx,
                                                void *const SFEM_RESTRICT          values,
                                                void                              *stream) {
    switch (real_type) {
        case SFEM_REAL_DEFAULT: {
            return cu_affine_hex8_linear_elasticity_bsr_tpl(nelements,
                                                            stride,
                                                            elements,
                                                            (cu_jacobian_t *)jacobian_adjugate,
                                                            (cu_jacobian_t *)jacobian_determinant,
                                                            mu,
                                                            lambda,
                                                            rowptr,
                                                            colidx,
                                                            (real_t *)values,
                                                            stream);
        }
        case SFEM_FLOAT32: {
            return cu_affine_hex8_linear_elasticity_bsr_tpl(nelements,
                                                            stride,
                                                            elements,
                                                            (cu_jacobian_t *)jacobian_adjugate,
                                                            (cu_jacobian_t *)jacobian_determinant,
                                                            mu,
                                                            lambda,
                                                            rowptr,
                                                            colidx,
                                                            (float *)values,
                                                            stream);
        }
        case SFEM_FLOAT64: {
            return cu_affine_hex8_linear_elasticity_bsr_tpl(nelements,
                                                            stride,
                                                            elements,
                                                            (cu_jacobian_t *)jacobian_adjugate,
                                                            (cu_jacobian_t *)jacobian_determinant,
                                                            mu,
                                                            lambda,
                                                            rowptr,
                                                            colidx,
                                                            (double *)values,
                                                            stream);
        }
        default: {
            SFEM_ERROR(
                    "[Error] cu_affine_hex8_linear_elasticity_bsr_tpl: not implemented for "
                    "type %s "
                    "(code %d)\n",
                    real_type_to_string(real_type),
                    real_type);
            return SFEM_FAILURE;
        }
    }
}

template <typename T>
__global__ void cu_affine_hex8_linear_elasticity_block_diag_sym_kernel(const ptrdiff_t                 nelements,
                                                                       const ptrdiff_t                 stride,
                                                                       idx_t *const SFEM_RESTRICT      elements,
                                                                       const cu_jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
                                                                       const cu_jacobian_t *const SFEM_RESTRICT jacobian_determinant,
                                                                       const T                         mu,
                                                                       const T                         lambda,
                                                                       const ptrdiff_t                 out_stride,
                                                                       T *const                        out0,
                                                                       T *const                        out1,
                                                                       T *const                        out2,
                                                                       T *const                        out3,
                                                                       T *const                        out4,
                                                                       T *const                        out5) {
    static const int n_qp  = 2;
    static const T   qx[2] = {0.2113248654, 0.7886751346};
    static const T   qw[2] = {1. / 2, 1. / 2};

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        idx_t ev[8];
        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v * stride + e];
        }

        T adjugate[9];

        // Copy over jacobian adjugate
        {
            for (int i = 0; i < 9; i++) {
                adjugate[i] = jacobian_adjugate[i * stride + e];
            }
        }

        const T determinant = jacobian_determinant[e];

        // Assemble the diagonal part of the matrix
        for (int edof_i = 0; edof_i < 8; edof_i++) {
            T element_matrix[6] = {0, 0, 0, 0, 0, 0};
            for (int zi = 0; zi < n_qp; zi++) {
                for (int yi = 0; yi < n_qp; yi++) {
                    for (int xi = 0; xi < n_qp; xi++) {
                        T test_grad[3];
                        cu_hex8_ref_shape_grad(edof_i, qx[xi], qx[yi], qx[zi], test_grad);
                        cu_linear_elasticity_matrix_sym<T>(mu,
                                                           lambda,
                                                           adjugate,
                                                           determinant,
                                                           test_grad,
                                                           test_grad,
                                                           qw[xi] * qw[yi] * qw[zi],
                                                           element_matrix);
                    }
                }
            }

            // local to global
            const ptrdiff_t idx = ev[edof_i] * out_stride;
            atomicAdd(&out0[idx], element_matrix[0]);
            atomicAdd(&out1[idx], element_matrix[1]);
            atomicAdd(&out2[idx], element_matrix[2]);
            atomicAdd(&out3[idx], element_matrix[3]);
            atomicAdd(&out4[idx], element_matrix[4]);
            atomicAdd(&out5[idx], element_matrix[5]);
        }
    }
}

template <typename T>
int cu_affine_hex8_linear_elasticity_block_diag_sym_tpl(const ptrdiff_t                 nelements,
                                                        const ptrdiff_t                 stride,
                                                        idx_t *const SFEM_RESTRICT      elements,
                                                        const cu_jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
                                                        const cu_jacobian_t *const SFEM_RESTRICT jacobian_determinant,
                                                        const real_t                    mu,
                                                        const real_t                    lambda,
                                                        const ptrdiff_t                 out_stride,
                                                        T *const                        out0,
                                                        T *const                        out1,
                                                        T *const                        out2,
                                                        T *const                        out3,
                                                        T *const                        out4,
                                                        T *const                        out5,
                                                        void                           *stream) {
    SFEM_DEBUG_SYNCHRONIZE();

    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(
                &min_grid_size, &block_size, cu_affine_hex8_linear_elasticity_block_diag_sym_kernel<T>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_affine_hex8_linear_elasticity_block_diag_sym_kernel<T><<<n_blocks, block_size, 0, s>>>(nelements,
                                                                                                  stride,
                                                                                                  elements,
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
        cu_affine_hex8_linear_elasticity_block_diag_sym_kernel<T><<<n_blocks, block_size, 0>>>(nelements,
                                                                                               stride,
                                                                                               elements,
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

extern int cu_affine_hex8_linear_elasticity_block_diag_sym(const ptrdiff_t                 nelements,
                                                           const ptrdiff_t                 stride,
                                                           idx_t *const SFEM_RESTRICT      elements,
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
            return cu_affine_hex8_linear_elasticity_block_diag_sym_tpl(nelements,
                                                                       stride,
                                                                       elements,
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
            return cu_affine_hex8_linear_elasticity_block_diag_sym_tpl(nelements,
                                                                       stride,
                                                                       elements,
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
            return cu_affine_hex8_linear_elasticity_block_diag_sym_tpl(nelements,
                                                                       stride,
                                                                       elements,
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
                    "[Error] cu_affine_hex8_linear_elasticity_block_diag_sym: not implemented for "
                    "type %s "
                    "(code %d)\n",
                    real_type_to_string(real_type),
                    real_type);
            return SFEM_FAILURE;
        }
    }
}
