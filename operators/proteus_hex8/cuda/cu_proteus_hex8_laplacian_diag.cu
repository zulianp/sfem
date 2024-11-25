#include "cu_proteus_hex8_laplacian.h"

#include "sfem_cuda_base.h"
#include "cu_proteus_hex8_inline.hpp"
#include "cu_hex8_laplacian_inline.hpp"

#include <cassert>
#include <cstdio>

#ifndef MAX
#define MAX(a, b) ((a) < (b) ? (b) : (a))
#endif

#ifndef MIN
#define MIN(a, b) ((a) > (b) ? (b) : (a))
#endif

template <typename real_t>
__global__ void cu_proteus_affine_hex8_laplacian_diag_kernel(
        const int level, const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const idx_t *const SFEM_RESTRICT elements, const cu_jacobian_t *const SFEM_RESTRICT fff,
        real_t *const SFEM_RESTRICT out) {
#ifndef NDEBUG
    const int nxe = cu_proteus_hex8_nxe(level);
#endif


    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {

        real_t laplacian_diag[8];
        // Build operator
        {
            scalar_t sub_fff[6];
            const scalar_t h = 1. / level;
            cu_hex8_sub_fff_0(stride, &fff[e], h, sub_fff);
            cu_hex8_laplacian_diag_fff_integral(sub_fff, laplacian_diag);
        }

        // Iterate over sub-elements
        for (int zi = 0; zi < level; zi++) {
            for (int yi = 0; yi < level; yi++) {
                for (int xi = 0; xi < level; xi++) {
                    assert(cu_proteus_hex8_lidx(level, xi + 1, yi + 1, zi + 1) < nxe);

                    int ev[8] = {
                            // Bottom
                            elements[cu_proteus_hex8_lidx(level, xi, yi, zi) * stride + e],
                            elements[cu_proteus_hex8_lidx(level, xi + 1, yi, zi) * stride + e],
                            elements[cu_proteus_hex8_lidx(level, xi + 1, yi + 1, zi) * stride + e],
                            elements[cu_proteus_hex8_lidx(level, xi, yi + 1, zi) * stride + e],
                            // Top
                            elements[cu_proteus_hex8_lidx(level, xi, yi, zi + 1) * stride + e],
                            elements[cu_proteus_hex8_lidx(level, xi + 1, yi, zi + 1) * stride + e],
                            elements[cu_proteus_hex8_lidx(level, xi + 1, yi + 1, zi + 1) * stride +
                                     e],
                            elements[cu_proteus_hex8_lidx(level, xi, yi + 1, zi + 1) * stride + e]};

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
static int cu_proteus_affine_hex8_laplacian_diag_tpl(
        const int level, const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const ptrdiff_t interior_start, const idx_t *const SFEM_RESTRICT elements,
        const cu_jacobian_t *const SFEM_RESTRICT fff, T *const out, void *stream) {
    SFEM_DEBUG_SYNCHRONIZE();

    // Hand tuned
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(
                &min_grid_size, &block_size, cu_proteus_affine_hex8_laplacian_diag_kernel<T>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    const ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_proteus_affine_hex8_laplacian_diag_kernel<<<n_blocks, block_size, 0, s>>>(
                level, nelements, stride, elements, fff, out);
    } else {
        cu_proteus_affine_hex8_laplacian_diag_kernel<<<n_blocks, block_size, 0>>>(
                level, nelements, stride, elements, fff, out);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

extern int cu_proteus_affine_hex8_laplacian_diag(
        const int level, const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const ptrdiff_t interior_start, const idx_t *const SFEM_RESTRICT elements,
        const void *const SFEM_RESTRICT fff, const enum RealType real_type_out, void *const out,
        void *stream) {
    switch (real_type_out) {
        case SFEM_REAL_DEFAULT: {
            return cu_proteus_affine_hex8_laplacian_diag_tpl(level,
                                                             nelements,
                                                             stride,
                                                             interior_start,
                                                             elements,
                                                             (cu_jacobian_t *)fff,
                                                             (real_t *)out,
                                                             stream);
        }
        case SFEM_FLOAT32: {
            return cu_proteus_affine_hex8_laplacian_diag_tpl(level,
                                                             nelements,
                                                             stride,
                                                             interior_start,
                                                             elements,
                                                             (cu_jacobian_t *)fff,
                                                             (float *)out,
                                                             stream);
        }
        case SFEM_FLOAT64: {
            return cu_proteus_affine_hex8_laplacian_diag_tpl(level,
                                                             nelements,
                                                             stride,
                                                             interior_start,
                                                             elements,
                                                             (cu_jacobian_t *)fff,
                                                             (double *)out,
                                                             stream);
        }
        default: {
            fprintf(stderr,
                    "[Error] cu_tet4_laplacian_diag: not implemented for type %s (code "
                    "%d)\n",
                    real_type_to_string(real_type_out),
                    real_type_out);
            assert(0);
            return SFEM_FAILURE;
        }
    }
}