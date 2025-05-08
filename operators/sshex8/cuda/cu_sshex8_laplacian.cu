#include "cu_sshex8_laplacian.h"

#include "cu_hex8_laplacian_inline.hpp"
#include "cu_sshex8_inline.hpp"
#include "sfem_cuda_base.h"

// #define CU_AFFINE_SSHEX8_LAPLACIAN_APPLY_USE_ELEMENTAL_MATRIX

template <typename real_t>
__global__ void cu_affine_sshex8_laplacian_apply_kernel(const int                                level,
                                                        const ptrdiff_t                          nelements,
                                                        idx_t **const SFEM_RESTRICT              elements,
                                                        const ptrdiff_t                          fff_stride,
                                                        const cu_jacobian_t *const SFEM_RESTRICT fff,
                                                        const real_t *const SFEM_RESTRICT        x,
                                                        real_t *const SFEM_RESTRICT              y) {
#ifndef NDEBUG
    const int nxe = cu_sshex8_nxe(level);
#endif

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
#ifdef CU_AFFINE_SSHEX8_LAPLACIAN_APPLY_USE_ELEMENTAL_MATRIX
        scalar_t laplacian_matrix[8 * 8];
        // Build operator
        {
            scalar_t       sub_fff[6];
            const scalar_t h = 1. / level;
            cu_hex8_sub_fff_0(fff_stride, &fff[e], h, sub_fff);
            cu_hex8_laplacian_matrix_fff_integral(sub_fff, laplacian_matrix);
        }
#else
        scalar_t sub_fff[6];
        {
            const scalar_t h = 1. / level;
            cu_hex8_sub_fff_0(fff_stride, &fff[e], h, sub_fff);
        }
#endif

        // Iterate over sub-elements
        for (int zi = 0; zi < level; zi++) {
            for (int yi = 0; yi < level; yi++) {
                for (int xi = 0; xi < level; xi++) {
                    assert(cu_sshex8_lidx(level, xi + 1, yi + 1, zi + 1) < nxe);

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

                    scalar_t element_u[8];
                    for (int d = 0; d < 8; d++) {
                        element_u[d] = x[ev[d]];
                    }

                    scalar_t element_vector[8];

#ifdef CU_AFFINE_SSHEX8_LAPLACIAN_APPLY_USE_ELEMENTAL_MATRIX
                    for (int i = 0; i < 8; i++) {
                        element_vector[i] = 0;
                    }

                    for (int i = 0; i < 8; i++) {
                        const scalar_t *const row = &laplacian_matrix[i * 8];
                        const scalar_t        ui  = element_u[i];
                        assert(ui == ui);
                        for (int j = 0; j < 8; j++) {
                            assert(row[j] == row[j]);
                            element_vector[j] += ui * row[j];
                        }
                    }
#else
                    cu_hex8_laplacian_apply_fff_integral(sub_fff, element_u, element_vector);
#endif

                    for (int d = 0; d < 8; d++) {
                        assert(element_vector[d] == element_vector[d]);
                        atomicAdd(&y[ev[d]], element_vector[d]);
                    }
                }
            }
        }
    }
}

#include "cu_sshex8_laplacian_variants.hpp"
#include "cu_sshex8_laplacian_warp.hpp"

template <typename T>
static int cu_affine_sshex8_laplacian_apply_tpl(const int                                level,
                                                const ptrdiff_t                          nelements,
                                                idx_t **const SFEM_RESTRICT              elements,
                                                const ptrdiff_t                          fff_stride,
                                                const cu_jacobian_t *const SFEM_RESTRICT fff,
                                                const T *const                           x,
                                                T *const                                 y,
                                                void                                    *stream) {
    SFEM_DEBUG_SYNCHRONIZE();

    int SFEM_HEX8_SHARED_MEM_KERNEL = 1;
    SFEM_READ_ENV(SFEM_HEX8_SHARED_MEM_KERNEL, atoi);

    if (SFEM_HEX8_SHARED_MEM_KERNEL) {
        switch (level) {
            case 4: {
                return cu_affine_sshex8_laplacian_apply_warp_tpl<T, 4>(nelements, elements, fff_stride, fff, x, y, stream);
            }
            case 6: {
                return cu_affine_sshex8_laplacian_apply_warp_tpl<T, 6>(nelements, elements, fff_stride, fff, x, y, stream);
            }
            case 8: {
                return cu_affine_sshex8_laplacian_apply_warp_tpl<T, 8>(nelements, elements, fff_stride, fff, x, y, stream);
            }
            default:
                break;
        }
    }

    int SFEM_HEX8_LOCAL_MEM_KERNEL = 1;
    SFEM_READ_ENV(SFEM_HEX8_LOCAL_MEM_KERNEL, atoi);
    if (SFEM_HEX8_LOCAL_MEM_KERNEL) {
        switch (level) {
            case 2: {
                return acu_affine_sshex8_laplacian_apply_local_mem_tpl<T, 2>(nelements, elements, fff_stride, fff, x, y, stream);
            }
            default:
                break;
        }
    }

    int SFEM_HEX8_LOCAL_MEM_BATCHED_KERNEL = 1;
    SFEM_READ_ENV(SFEM_HEX8_LOCAL_MEM_BATCHED_KERNEL, atoi);
    if (SFEM_HEX8_LOCAL_MEM_BATCHED_KERNEL) {
        switch (level) {
            case 16: {
                return acu_affine_sshex8_laplacian_apply_local_mem_batched_tpl<T, 16, 4>(
                        nelements, elements, fff_stride, fff, x, y, stream);
            }
            default:
                break;
        }
    }

    // Hand tuned
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, cu_affine_sshex8_laplacian_apply_kernel<T>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    const ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_affine_sshex8_laplacian_apply_kernel<<<n_blocks, block_size, 0, s>>>(
                level, nelements, elements, fff_stride, fff, x, y);
    } else {
        cu_affine_sshex8_laplacian_apply_kernel<<<n_blocks, block_size, 0>>>(level, nelements, elements, fff_stride, fff, x, y);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

extern int cu_affine_sshex8_laplacian_apply(const int                       level,
                                            const ptrdiff_t                 nelements,
                                            idx_t **const SFEM_RESTRICT     elements,
                                            const ptrdiff_t                 fff_stride,
                                            const void *const SFEM_RESTRICT fff,
                                            const enum RealType             real_type_xy,
                                            const void *const               x,
                                            void *const                     y,
                                            void                           *stream) {
    switch (real_type_xy) {
        case SFEM_REAL_DEFAULT: {
            return cu_affine_sshex8_laplacian_apply_tpl(
                    level, nelements, elements, fff_stride, (cu_jacobian_t *)fff, (real_t *)x, (real_t *)y, stream);
        }
        case SFEM_FLOAT32: {
            return cu_affine_sshex8_laplacian_apply_tpl(
                    level, nelements, elements, fff_stride, (cu_jacobian_t *)fff, (float *)x, (float *)y, stream);
        }
        case SFEM_FLOAT64: {
            return cu_affine_sshex8_laplacian_apply_tpl(
                    level, nelements, elements, fff_stride, (cu_jacobian_t *)fff, (double *)x, (double *)y, stream);
        }
        default: {
            SFEM_ERROR("[Error] cu_tet4_laplacian_apply: not implemented for type %s (code %d)\n",
                       real_type_to_string(real_type_xy),
                       real_type_xy);
            return SFEM_FAILURE;
        }
    }
}
