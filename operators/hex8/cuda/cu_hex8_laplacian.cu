#include "cu_hex8_laplacian.h"

#include "sfem_cuda_base.h"

#ifndef MAX
#define MAX(a, b) ((a) >= (b) ? (a) : (b))
#endif

template <typename scalar_t, typename accumulator_t>
static SFEM_INLINE __host__ __device__ void cu_hex8_laplacian_apply_fff_integral(
        const scalar_t *const SFEM_RESTRICT fff,
        const scalar_t *SFEM_RESTRICT u,
        accumulator_t *SFEM_RESTRICT element_vector) {
    const scalar_t x0 = (1.0 / 6.0) * fff[4];
    const scalar_t x1 = u[7] * x0;
    const scalar_t x2 = (1.0 / 9.0) * fff[3];
    const scalar_t x3 = u[3] * x2;
    const scalar_t x4 = (1.0 / 9.0) * fff[5];
    const scalar_t x5 = u[4] * x4;
    const scalar_t x6 = (1.0 / 12.0) * u[6];
    const scalar_t x7 = fff[4] * x6;
    const scalar_t x8 = (1.0 / 36.0) * u[6];
    const scalar_t x9 = fff[3] * x8;
    const scalar_t x10 = fff[5] * x8;
    const scalar_t x11 = u[0] * x0;
    const scalar_t x12 = u[0] * x2;
    const scalar_t x13 = u[0] * x4;
    const scalar_t x14 = (1.0 / 12.0) * fff[4];
    const scalar_t x15 = u[1] * x14;
    const scalar_t x16 = (1.0 / 36.0) * fff[3];
    const scalar_t x17 = u[5] * x16;
    const scalar_t x18 = (1.0 / 36.0) * fff[5];
    const scalar_t x19 = u[2] * x18;
    const scalar_t x20 = (1.0 / 6.0) * fff[1];
    const scalar_t x21 = (1.0 / 12.0) * fff[1];
    const scalar_t x22 = -fff[1] * x6 + u[0] * x20 - u[2] * x20 + u[4] * x21;
    const scalar_t x23 = (1.0 / 6.0) * fff[2];
    const scalar_t x24 = (1.0 / 12.0) * fff[2];
    const scalar_t x25 = -fff[2] * x6 + u[0] * x23 + u[3] * x24 - u[5] * x23;
    const scalar_t x26 = (1.0 / 9.0) * fff[0];
    const scalar_t x27 = (1.0 / 36.0) * u[7];
    const scalar_t x28 = (1.0 / 18.0) * fff[0];
    const scalar_t x29 = -u[2] * x28 + u[3] * x28 + u[4] * x28 - u[5] * x28;
    const scalar_t x30 = fff[0] * x27 - fff[0] * x8 + u[0] * x26 - u[1] * x26 + x29;
    const scalar_t x31 = (1.0 / 18.0) * fff[3];
    const scalar_t x32 = u[2] * x31;
    const scalar_t x33 = u[7] * x31;
    const scalar_t x34 = (1.0 / 18.0) * fff[5];
    const scalar_t x35 = u[5] * x34;
    const scalar_t x36 = u[7] * x34;
    const scalar_t x37 = u[1] * x31;
    const scalar_t x38 = u[4] * x31;
    const scalar_t x39 = u[1] * x34;
    const scalar_t x40 = u[3] * x34;
    const scalar_t x41 = -x32 - x33 - x35 - x36 + x37 + x38 + x39 + x40;
    const scalar_t x42 = u[1] * x0;
    const scalar_t x43 = u[1] * x2;
    const scalar_t x44 = u[1] * x4;
    const scalar_t x45 = u[0] * x14;
    const scalar_t x46 = u[4] * x16;
    const scalar_t x47 = u[3] * x18;
    const scalar_t x48 = u[6] * x0;
    const scalar_t x49 = u[2] * x2;
    const scalar_t x50 = u[5] * x4;
    const scalar_t x51 = u[7] * x14;
    const scalar_t x52 = fff[3] * x27;
    const scalar_t x53 = fff[5] * x27;
    const scalar_t x54 = u[1] * x20 - u[3] * x20 + u[5] * x21 - u[7] * x21;
    const scalar_t x55 = u[1] * x23 + u[2] * x24 - u[4] * x23 - u[7] * x24;
    const scalar_t x56 = u[0] * x31;
    const scalar_t x57 = u[5] * x31;
    const scalar_t x58 = u[0] * x34;
    const scalar_t x59 = u[2] * x34;
    const scalar_t x60 = u[3] * x31;
    const scalar_t x61 = u[6] * x31;
    const scalar_t x62 = u[4] * x34;
    const scalar_t x63 = u[6] * x34;
    const scalar_t x64 = -x56 - x57 - x58 - x59 + x60 + x61 + x62 + x63;
    const scalar_t x65 = u[5] * x0;
    const scalar_t x66 = u[2] * x4;
    const scalar_t x67 = u[4] * x14;
    const scalar_t x68 = u[0] * x18;
    const scalar_t x69 = u[2] * x0;
    const scalar_t x70 = u[6] * x4;
    const scalar_t x71 = u[3] * x14;
    const scalar_t x72 = u[4] * x18;
    const scalar_t x73 = u[1] * x24 + u[2] * x23 - u[4] * x24 - u[7] * x23;
    const scalar_t x74 = (1.0 / 36.0) * fff[0];
    const scalar_t x75 = u[0] * x28 - u[1] * x28 - u[6] * x28 + u[7] * x28;
    const scalar_t x76 = -u[2] * x26 + u[3] * x26 + u[4] * x74 - u[5] * x74 + x75;
    const scalar_t x77 = x35 + x36 - x39 - x40 + x56 + x57 - x60 - x61;
    const scalar_t x78 = u[3] * x0;
    const scalar_t x79 = u[7] * x4;
    const scalar_t x80 = u[2] * x14;
    const scalar_t x81 = u[5] * x18;
    const scalar_t x82 = u[4] * x0;
    const scalar_t x83 = u[3] * x4;
    const scalar_t x84 = u[5] * x14;
    const scalar_t x85 = u[1] * x18;
    const scalar_t x86 = u[0] * x24 + u[3] * x23 - u[5] * x24 - u[6] * x23;
    const scalar_t x87 = x32 + x33 - x37 - x38 + x58 + x59 - x62 - x63;
    const scalar_t x88 = u[7] * x2;
    const scalar_t x89 = u[2] * x16;
    const scalar_t x90 = u[4] * x2;
    const scalar_t x91 = u[1] * x16;
    const scalar_t x92 = u[0] * x21 - u[2] * x21 + u[4] * x20 - u[6] * x20;
    const scalar_t x93 = -u[2] * x74 + u[3] * x74 + u[4] * x26 - u[5] * x26 + x75;
    const scalar_t x94 = u[5] * x2;
    const scalar_t x95 = u[0] * x16;
    const scalar_t x96 = u[6] * x2;
    const scalar_t x97 = u[3] * x16;
    const scalar_t x98 = u[1] * x21 - u[3] * x21 + u[5] * x20 - u[7] * x20;
    const scalar_t x99 = u[0] * x74 - u[1] * x74 - u[6] * x26 + u[7] * x26 + x29;
    element_vector[0] = -x1 - x10 + x11 + x12 + x13 + x15 + x17 + x19 + x22 + x25 - x3 + x30 + x41 -
                        x5 - x7 - x9;
    element_vector[1] = -x30 + x42 + x43 + x44 + x45 + x46 + x47 - x48 - x49 - x50 - x51 - x52 -
                        x53 - x54 - x55 - x64;
    element_vector[2] = -x22 - x43 - x46 + x49 + x52 + x65 + x66 + x67 + x68 - x69 - x70 - x71 -
                        x72 - x73 - x76 - x77;
    element_vector[3] = -x12 - x17 + x3 + x54 + x76 - x78 - x79 - x80 - x81 + x82 + x83 + x84 +
                        x85 + x86 + x87 + x9;
    element_vector[4] = x10 - x13 - x19 + x5 + x55 + x77 + x78 + x80 - x82 - x84 - x88 - x89 + x90 +
                        x91 + x92 + x93;
    element_vector[5] = -x25 - x44 - x47 + x50 + x53 - x65 - x67 + x69 + x71 - x87 - x93 + x94 +
                        x95 - x96 - x97 - x98;
    element_vector[6] = -x41 - x42 - x45 + x48 + x51 - x66 - x68 + x70 + x72 - x86 - x92 - x94 -
                        x95 + x96 + x97 - x99;
    element_vector[7] = x1 - x11 - x15 + x64 + x7 + x73 + x79 + x81 - x83 - x85 + x88 + x89 - x90 -
                        x91 + x98 + x99;
}

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
