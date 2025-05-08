#include "cu_sshex8_laplacian.h"

#include "cu_hex8_laplacian_inline.hpp"
#include "cu_sshex8_inline.hpp"
#include "sfem_cuda_base.h"

#include <cassert>
#include <cstdio>

#ifndef MAX
#define MAX(a, b) ((a) < (b) ? (b) : (a))
#endif

#ifndef MIN
#define MIN(a, b) ((a) > (b) ? (b) : (a))
#endif

template <typename real_t>
__global__ void cu_affine_sshex8_laplacian_diag_kernel(const int                                level,
                                                       const ptrdiff_t                          nelements,
                                                       idx_t **const SFEM_RESTRICT              elements,
                                                       const ptrdiff_t                          fff_stride,
                                                       const cu_jacobian_t *const SFEM_RESTRICT fff,
                                                       real_t *const SFEM_RESTRICT              out) {
#ifndef NDEBUG
    const int nxe = cu_sshex8_nxe(level);
#endif

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        real_t laplacian_diag[8];
        // Build operator
        {
            scalar_t       sub_fff[6];
            const scalar_t h = 1. / level;
            cu_hex8_sub_fff_0(fff_stride, &fff[e], h, sub_fff);
            cu_hex8_laplacian_diag_fff_integral(sub_fff, laplacian_diag);
        }

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

                    for (int d = 0; d < 8; d++) {
                        assert(laplacian_diag[d] == laplacian_diag[d]);
                        atomicAdd(&out[ev[d]], laplacian_diag[d]);
                    }
                }
            }
        }
    }
}

template <typename T>
static int cu_affine_sshex8_laplacian_diag_tpl(const int                                level,
                                               const ptrdiff_t                          nelements,
                                               idx_t **const SFEM_RESTRICT              elements,
                                               const ptrdiff_t                          fff_stride,
                                               const cu_jacobian_t *const SFEM_RESTRICT fff,
                                               T *const                                 out,
                                               void                                    *stream) {
    SFEM_DEBUG_SYNCHRONIZE();

    // Hand tuned
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, cu_affine_sshex8_laplacian_diag_kernel<T>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    const ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_affine_sshex8_laplacian_diag_kernel<<<n_blocks, block_size, 0, s>>>(level, nelements, elements, fff_stride, fff, out);
    } else {
        cu_affine_sshex8_laplacian_diag_kernel<<<n_blocks, block_size, 0>>>(level, nelements, elements, fff_stride, fff, out);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

extern int cu_affine_sshex8_laplacian_diag(const int                       level,
                                           const ptrdiff_t                 nelements,
                                           idx_t **const SFEM_RESTRICT     elements,
                                           const ptrdiff_t                 fff_stride,
                                           const void *const SFEM_RESTRICT fff,
                                           const enum RealType             real_type_out,
                                           void *const                     out,
                                           void                           *stream) {
    switch (real_type_out) {
        case SFEM_REAL_DEFAULT: {
            return cu_affine_sshex8_laplacian_diag_tpl(
                    level, nelements, elements, fff_stride, (cu_jacobian_t *)fff, (real_t *)out, stream);
        }
        case SFEM_FLOAT32: {
            return cu_affine_sshex8_laplacian_diag_tpl(
                    level, nelements, elements, fff_stride, (cu_jacobian_t *)fff, (float *)out, stream);
        }
        case SFEM_FLOAT64: {
            return cu_affine_sshex8_laplacian_diag_tpl(
                    level, nelements, elements, fff_stride, (cu_jacobian_t *)fff, (double *)out, stream);
        }
        default: {
            SFEM_ERROR(
                    "[Error] cu_tet4_laplacian_diag: not implemented for type %s (code "
                    "%d)\n",
                    real_type_to_string(real_type_out),
                    real_type_out);

            return SFEM_FAILURE;
        }
    }
}