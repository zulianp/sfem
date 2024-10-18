#include "cu_hex8_laplacian.h"

#include "cu_hex8_laplacian_inline.hpp"
#include "sfem_cuda_base.h"

#ifndef MAX
#define MAX(a, b) ((a) >= (b) ? (a) : (b))
#endif

template <typename real_t>
__global__ void cu_affine_hex8_laplacian_apply_kernel(
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const idx_t *const SFEM_RESTRICT elements,
        const cu_jacobian_t *const SFEM_RESTRICT g_fff,
        const real_t *const SFEM_RESTRICT u,
        real_t *const SFEM_RESTRICT values) {
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {
        idx_t ev[8];
        accumulator_t element_vector[8];
        scalar_t element_u[8];
        scalar_t fff[6];

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v * stride + e];
        }

        for (int d = 0; d < 6; d++) {
            fff[d] = g_fff[d * stride];
        }

        for (int v = 0; v < 8; ++v) {
            element_u[v] = u[ev[v]];
        }

        cu_hex8_laplacian_apply_fff_integral(fff, element_u, element_vector);

        for (int edof_i = 0; edof_i < 8; ++edof_i) {
            const idx_t dof_i = ev[edof_i];
            atomicAdd(&values[dof_i], element_vector[edof_i]);
        }
    }
}

template <typename T>
static int cu_affine_hex8_laplacian_apply_tpl(
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const idx_t *const SFEM_RESTRICT elements,
        const cu_jacobian_t *const SFEM_RESTRICT fff,
        const T *const x,
        T *const y,
        void *stream) {
    SFEM_DEBUG_SYNCHRONIZE();

    // Hand tuned
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(
                &min_grid_size, &block_size, cu_affine_hex8_laplacian_apply_kernel<T>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    const ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_affine_hex8_laplacian_apply_kernel<<<n_blocks, block_size, 0, s>>>(
                nelements, stride, elements, fff, x, y);
    } else {
        cu_affine_hex8_laplacian_apply_kernel<<<n_blocks, block_size, 0>>>(
                nelements, stride, elements, fff, x, y);
    }

    SFEM_DEBUG_SYNCHRONIZE();

    return SFEM_SUCCESS;
}

extern int cu_affine_hex8_laplacian_apply(const ptrdiff_t nelements,
                                          const ptrdiff_t stride,  // Stride for elements and fff
                                          const idx_t *const SFEM_RESTRICT elements,
                                          const void *const SFEM_RESTRICT fff,
                                          const enum RealType real_type_xy,
                                          const void *const x,
                                          void *const y,
                                          void *stream) {
    switch (real_type_xy) {
        case SFEM_REAL_DEFAULT: {
            return cu_affine_hex8_laplacian_apply_tpl(nelements,
                                                      stride,
                                                      elements,
                                                      (cu_jacobian_t *)fff,
                                                      (real_t *)x,
                                                      (real_t *)y,
                                                      stream);
        }
        case SFEM_FLOAT32: {
            return cu_affine_hex8_laplacian_apply_tpl(nelements,
                                                      stride,
                                                      elements,
                                                      (cu_jacobian_t *)fff,
                                                      (float *)x,
                                                      (float *)y,
                                                      stream);
        }
        case SFEM_FLOAT64: {
            return cu_affine_hex8_laplacian_apply_tpl(nelements,
                                                      stride,
                                                      elements,
                                                      (cu_jacobian_t *)fff,
                                                      (double *)x,
                                                      (double *)y,
                                                      stream);
        }
        default: {
            fprintf(stderr,
                    "[Error] cu_hex8_laplacian_apply: not implemented for type %s (code %d)\n",
                    real_type_to_string(real_type_xy),
                    real_type_xy);
            assert(0);
            return SFEM_FAILURE;
        }
    }
}
