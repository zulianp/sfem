#include "cu_proteus_hex8_laplacian.h"

#include "sfem_cuda_base.h"

#ifndef MAX
#define MAX(a, b)((a) >= (b)? (a) : (b))
#endif

static inline __device__ __host__ int cu_proteus_hex8_nxe(int level) {
    const int corners = 8;
    const int edge_nodes = 12 * (level - 1);
    const int face_nodes = 6 * (level - 1) * (level - 1);
    const int vol_nodes = (level - 1) * (level - 1) * (level - 1);
    return corners + edge_nodes + face_nodes + vol_nodes;
}

static inline __device__ __host__ int cu_proteus_hex8_txe(int level) {
    return level * level * level;
}

static inline __device__ __host__ int cu_proteus_hex8_lidx(const int L,
                                                           const int x,
                                                           const int y,
                                                           const int z) {
    int Lp1 = L + 1;
    int ret = z * (Lp1 * Lp1) + y * Lp1 + x;
    assert(ret < cu_proteus_hex8_nxe(L));
    assert(ret >= 0);
    return ret;
}

static inline __device__ __host__ void cu_hex8_sub_fff_0(const ptrdiff_t stride,
                                                         const cu_jacobian_t *const SFEM_RESTRICT
                                                                 fff,
                                                         const scalar_t h,
                                                         scalar_t *const SFEM_RESTRICT sub_fff) {
    sub_fff[0] = (scalar_t)fff[0 * stride] * h;
    sub_fff[1] = (scalar_t)fff[1 * stride] * h;
    sub_fff[2] = (scalar_t)fff[2 * stride] * h;
    sub_fff[3] = (scalar_t)fff[3 * stride] * h;
    sub_fff[4] = (scalar_t)fff[4 * stride] * h;
    sub_fff[5] = (scalar_t)fff[5 * stride] * h;
}

static inline __device__ __host__ void cu_hex8_laplacian_matrix_fff_integral(
        const scalar_t *const SFEM_RESTRICT fff,
        accumulator_t *const SFEM_RESTRICT element_matrix) {
    const scalar_t x0 = (1.0 / 6.0) * fff[1];
    const scalar_t x1 = (1.0 / 6.0) * fff[2];
    const scalar_t x2 = (1.0 / 6.0) * fff[4];
    const scalar_t x3 = (1.0 / 9.0) * fff[0];
    const scalar_t x4 = (1.0 / 9.0) * fff[3];
    const scalar_t x5 = (1.0 / 9.0) * fff[5];
    const scalar_t x6 = x2 + x3 + x4 + x5;
    const scalar_t x7 = x0 + x1 + x6;
    const scalar_t x8 = (1.0 / 12.0) * fff[4];
    const scalar_t x9 = (1.0 / 18.0) * fff[3];
    const scalar_t x10 = (1.0 / 18.0) * fff[5];
    const scalar_t x11 = x10 + x9;
    const scalar_t x12 = x11 - x3 + x8;
    const scalar_t x13 = (1.0 / 36.0) * fff[5];
    const scalar_t x14 = (1.0 / 18.0) * fff[0];
    const scalar_t x15 = x14 + x9;
    const scalar_t x16 = -x13 + x15;
    const scalar_t x17 = -x0 - x16;
    const scalar_t x18 = (1.0 / 12.0) * fff[2];
    const scalar_t x19 = x10 + x14;
    const scalar_t x20 = x19 - x4;
    const scalar_t x21 = x18 + x20;
    const scalar_t x22 = (1.0 / 12.0) * fff[1];
    const scalar_t x23 = x15 - x5;
    const scalar_t x24 = x22 + x23;
    const scalar_t x25 = (1.0 / 36.0) * fff[3];
    const scalar_t x26 = x19 - x25;
    const scalar_t x27 = -x1 - x26;
    const scalar_t x28 = (1.0 / 36.0) * fff[0];
    const scalar_t x29 = x13 + x25 + x28 + x8;
    const scalar_t x30 = -x18 - x22 - x29;
    const scalar_t x31 = -x11 - x2 + x28;
    const scalar_t x32 = -x0;
    const scalar_t x33 = -x1;
    const scalar_t x34 = x32 + x33 + x6;
    const scalar_t x35 = -x18;
    const scalar_t x36 = x20 + x35;
    const scalar_t x37 = -x16 - x32;
    const scalar_t x38 = -x26 - x33;
    const scalar_t x39 = -x22;
    const scalar_t x40 = x23 + x39;
    const scalar_t x41 = -x29 - x35 - x39;
    const scalar_t x42 = -x2 + x3 + x4 + x5;
    const scalar_t x43 = x0 + x33 + x42;
    const scalar_t x44 = -x10 - x9;
    const scalar_t x45 = -x3 - x44 - x8;
    const scalar_t x46 = x13 + x25 + x28 - x8;
    const scalar_t x47 = -x22 - x35 - x46;
    const scalar_t x48 = x2 + x28 + x44;
    const scalar_t x49 = x1 + x32 + x42;
    const scalar_t x50 = -x18 - x39 - x46;
    element_matrix[0] = x7;
    element_matrix[1] = x12;
    element_matrix[2] = x17;
    element_matrix[3] = x21;
    element_matrix[4] = x24;
    element_matrix[5] = x27;
    element_matrix[6] = x30;
    element_matrix[7] = x31;
    element_matrix[8] = x12;
    element_matrix[9] = x34;
    element_matrix[10] = x36;
    element_matrix[11] = x37;
    element_matrix[12] = x38;
    element_matrix[13] = x40;
    element_matrix[14] = x31;
    element_matrix[15] = x41;
    element_matrix[16] = x17;
    element_matrix[17] = x36;
    element_matrix[18] = x43;
    element_matrix[19] = x45;
    element_matrix[20] = x47;
    element_matrix[21] = x48;
    element_matrix[22] = x24;
    element_matrix[23] = x38;
    element_matrix[24] = x21;
    element_matrix[25] = x37;
    element_matrix[26] = x45;
    element_matrix[27] = x49;
    element_matrix[28] = x48;
    element_matrix[29] = x50;
    element_matrix[30] = x27;
    element_matrix[31] = x40;
    element_matrix[32] = x24;
    element_matrix[33] = x38;
    element_matrix[34] = x47;
    element_matrix[35] = x48;
    element_matrix[36] = x43;
    element_matrix[37] = x45;
    element_matrix[38] = x17;
    element_matrix[39] = x36;
    element_matrix[40] = x27;
    element_matrix[41] = x40;
    element_matrix[42] = x48;
    element_matrix[43] = x50;
    element_matrix[44] = x45;
    element_matrix[45] = x49;
    element_matrix[46] = x21;
    element_matrix[47] = x37;
    element_matrix[48] = x30;
    element_matrix[49] = x31;
    element_matrix[50] = x24;
    element_matrix[51] = x27;
    element_matrix[52] = x17;
    element_matrix[53] = x21;
    element_matrix[54] = x7;
    element_matrix[55] = x12;
    element_matrix[56] = x31;
    element_matrix[57] = x41;
    element_matrix[58] = x38;
    element_matrix[59] = x40;
    element_matrix[60] = x36;
    element_matrix[61] = x37;
    element_matrix[62] = x12;
    element_matrix[63] = x34;
}

template <typename real_t>
__global__ void cu_proteus_affine_hex8_laplacian_apply_kernel(
        const int level,
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const idx_t *const SFEM_RESTRICT elements,
        const cu_jacobian_t *const SFEM_RESTRICT fff,
        const real_t *const SFEM_RESTRICT x,
        real_t *const SFEM_RESTRICT y) {
    scalar_t laplacian_matrix[8 * 8];

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {
        // Build operator
        {
            scalar_t sub_fff[6];
            const scalar_t h = 1. / level;
            cu_hex8_sub_fff_0(stride, &fff[e], h, sub_fff);
            cu_hex8_laplacian_matrix_fff_integral(sub_fff, laplacian_matrix);
        }

        // Iterate over sub-elements
        for (int zi = 0; zi < level; zi++) {
            for (int yi = 0; yi < level; yi++) {
                for (int xi = 0; xi < level; xi++) {
                    int ev[8] = {// Bottom
                                 elements[cu_proteus_hex8_lidx(level, xi, yi, zi)*stride + e],
                                 elements[cu_proteus_hex8_lidx(level, xi + 1, yi, zi)*stride + e],
                                 elements[cu_proteus_hex8_lidx(level, xi + 1, yi + 1, zi)*stride + e],
                                 elements[cu_proteus_hex8_lidx(level, xi, yi + 1, zi)*stride + e],
                                 // Top
                                 elements[cu_proteus_hex8_lidx(level, xi, yi, zi + 1)*stride + e],
                                 elements[cu_proteus_hex8_lidx(level, xi + 1, yi, zi + 1)*stride + e],
                                 elements[cu_proteus_hex8_lidx(level, xi + 1, yi + 1, zi + 1)*stride + e],
                                 elements[cu_proteus_hex8_lidx(level, xi, yi + 1, zi + 1)*stride + e]};

                    scalar_t element_u[8];

                    for (int d = 0; d < 8; d++) {
                        element_u[d] = x[ev[d]];
                    }

                    scalar_t element_vector[8];
                    for (int i = 0; i < 8; i++) {
                        element_vector[i] = 0;
                    }

                    for (int i = 0; i < 8; i++) {
                        const scalar_t *const row = &laplacian_matrix[i * 8];
                        const scalar_t ui = element_u[i];
                        assert(ui == ui);
                        for (int j = 0; j < 8; j++) {
                            assert(row[j] == row[j]);
                            element_vector[j] += ui * row[j];
                        }
                    }

                    for (int d = 0; d < 8; d++) {
                        assert(element_vector[d] == element_vector[d]);
                        atomicAdd(&y[ev[d]], element_vector[d]);
                    }
                }
            }
        }
    }
}

template <typename T>
static int cu_proteus_affine_hex8_laplacian_apply_tpl(
        const int level,
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const idx_t *const SFEM_RESTRICT elements,
        const cu_jacobian_t *const SFEM_RESTRICT fff,
        const T *const x,
        T *const y,
        void *stream) {
    // Hand tuned
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(
                &min_grid_size, &block_size, cu_proteus_affine_hex8_laplacian_apply_kernel<T>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    const ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_proteus_affine_hex8_laplacian_apply_kernel<<<n_blocks, block_size, 0, s>>>(
            level, nelements, stride, elements, fff, x, y);
    } else {
        cu_proteus_affine_hex8_laplacian_apply_kernel<<<n_blocks, block_size, 0>>>(
                level, nelements, stride, elements, fff, x, y);
    }

    return SFEM_SUCCESS;
}

extern int cu_proteus_affine_hex8_laplacian_apply(
        const int level,
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const idx_t *const SFEM_RESTRICT elements,
        const void *const SFEM_RESTRICT fff,
        const enum RealType real_type_xy,
        const void *const x,
        void *const y,
        void *stream) {
    switch (real_type_xy) {
        case SFEM_REAL_DEFAULT: {
            return cu_proteus_affine_hex8_laplacian_apply_tpl(level,
                                                              nelements,
                                                              stride,
                                                              elements,
                                                              (cu_jacobian_t *)fff,
                                                              (real_t *)x,
                                                              (real_t *)y,
                                                              stream);
        }
        case SFEM_FLOAT32: {
            return cu_proteus_affine_hex8_laplacian_apply_tpl(level,
                                                              nelements,
                                                              stride,
                                                              elements,
                                                              (cu_jacobian_t *)fff,
                                                              (float *)x,
                                                              (float *)y,
                                                              stream);
        }
        case SFEM_FLOAT64: {
            return cu_proteus_affine_hex8_laplacian_apply_tpl(level,
                                                              nelements,
                                                              stride,
                                                              elements,
                                                              (cu_jacobian_t *)fff,
                                                              (double *)x,
                                                              (double *)y,
                                                              stream);
        }
        default: {
            fprintf(stderr,
                    "[Error] cu_tet4_laplacian_apply: not implemented for type %s (code %d)\n",
                    real_type_to_string(real_type_xy),
                    real_type_xy);
            assert(0);
            return SFEM_FAILURE;
        }
    }
}
