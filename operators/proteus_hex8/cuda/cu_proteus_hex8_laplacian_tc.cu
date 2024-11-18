#include "cu_proteus_hex8_laplacian_tc.h"

#include <assert.h>
#include <stdio.h>

// REFERENCES:
// https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#distributed-shared-memory
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies-using-the-tensor-memory-accelerator-tma
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma


// NOTES
// 1) memcpy_async
// cooperative_groups::memcpy_async(group, shared, global_in + batch_idx, sizeof(int) *
// block.size()); The cooperative_groups::memcpy_async API copies sizeof(int) * block.size() bytes
// from global memory starting at global_in + batch_idx to the shared data. This operation happens
// as-if performed by another thread, which synchronizes with the current thread’s call to
// cooperative_groups::wait after the copy has completed. Until the copy operation completes,
// modifying the global data or reading or writing the shared data introduces a data race.
// 2) Tensor cores MMA (M-N-K) C=A*B+C  A (MxK), B (KxN), C (MxN)
// double precision 8 x 8 x 4

template <typename scalar_t, typename accumulator_t>
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

extern int cu_proteus_affine_hex8_laplacian_tc_allocate_macro_ops(const ptrdiff_t nelements,
                                                                  const enum RealType real_type,
                                                                  void **mem) {
    int real_size = real_type_size(real_type);
    CHECK_CUDA(cudaMalloc(mem, real_size * nelements * 8 * 8));
    return (*mem) ? SFEM_FAILURE : SFEM_SUCCESS;
}

template <typename T>
__global__ void cu_proteus_affine_hex8_laplacian_tc_fill_ops_kernel(
        const ptrdiff_t nelements,
        const ptrdiff_t stride,
        const idx_t *const SFEM_RESTRICT elements,
        const cu_jacobian_t *const SFEM_RESTRICT fff,
        T *const SFEM_RESTRICT macro_element_ops) {
    // Your kernel implementation here
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {
        // Your kernel implementation here
        T fff_e[6];
        T element_matrix[8 * 8];

        for (int i = 0; i < 6; i++) {
            fff_e[i] = fff[e * stride + i];
        }

        cu_hex8_laplacian_matrix_fff_integral(fff_e, element_matrix);

        // AoS 1 macro element per thread-block?
        for (int i = 0; i < 8 * 8; i++) {
            macro_element_ops[e * 8 * 8 + i] = element_matrix[i];
        }
    }
}

template <typename T>
static int cu_proteus_affine_hex8_laplacian_tc_fill_ops_tpl(
        const ptrdiff_t nelements,
        const ptrdiff_t stride,
        const idx_t *const SFEM_RESTRICT elements,
        const cu_jacobian_t *const SFEM_RESTRICT fff,
        T *const SFEM_RESTRICT macro_element_ops) {
    // Hand tuned
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size,
                                           &block_size,
                                           cu_proteus_affine_hex8_laplacian_tc_fill_ops_kernel<T>,
                                           0,
                                           0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    const ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_proteus_affine_hex8_laplacian_tc_fill_ops_kernel<<<n_blocks, block_size, 0, s>>>(
                level, nelements, stride, elements, fff, x, y);
    } else {
        cu_proteus_affine_hex8_laplacian_tc_fill_ops_kernel<<<n_blocks, block_size, 0>>>(
                level, nelements, stride, elements, fff, x, y);
    }
}

extern int cu_proteus_affine_hex8_laplacian_tc_fill_ops(const ptrdiff_t nelements,
                                                        const ptrdiff_t stride,
                                                        const idx_t *const SFEM_RESTRICT elements,
                                                        const void *const SFEM_RESTRICT fff,
                                                        const enum RealType real_type,
                                                        void *const SFEM_RESTRICT
                                                                macro_element_ops) {
    switch (real_type) {
        case SFEM_FLOAT64:
            return cu_proteus_affine_hex8_laplacian_tc_fill_ops_tpl<double>(
                    nelements,
                    stride,
                    elements,
                    (const cu_jacobian_t *)fff,
                    (double *)macro_element_ops);
        case SFEM_FLOAT32:
            return cu_proteus_affine_hex8_laplacian_tc_fill_ops_tpl<float>(
                    nelements,
                    stride,
                    elements,
                    (const cu_jacobian_t *)fff,
                    (float *)macro_element_ops);
        case SFEM_REAL_DEFAULT:
            return cu_proteus_affine_hex8_laplacian_tc_fill_ops_tpl<real_t>(
                    nelements,
                    stride,
                    elements,
                    (const cu_jacobian_t *)fff,
                    (real_t *)macro_element_ops);
        default: {
            fprintf(stderr,
                    "[Error] cu_proteus_affine_hex8_laplacian_tc_fill_ops: not implemented for "
                    "type %s (code %d)\n",
                    real_type_to_string(real_type),
                    real_type);
            assert(0);
            return SFEM_FAILURE;
        }
    }
    return SFEM_FAILURE;
}

template <typename T>
__global__ int cu_proteus_affine_hex8_laplacian_tc_apply_kernel(
        const int level,
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const ptrdiff_t interior_start,
        const idx_t *const SFEM_RESTRICT elements,
        const T *const SFEM_RESTRICT macro_element_ops,
        const T *const SFEM_RESTRICT x,
        T *const SFEM_RESTRICT y) {
    // Your kernel implementation here
    // TODO
}

template <typename T>
static int cu_proteus_affine_hex8_laplacian_tc_apply_tpl(
        const int level,
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const ptrdiff_t interior_start,
        const idx_t *const SFEM_RESTRICT elements,
        const T *const SFEM_RESTRICT macro_element_ops,
        const T *const SFEM_RESTRICT x,
        T *const SFEM_RESTRICT y,
        void *stream) {

              // Hand tuned
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size,
                                           &block_size,
                                           cu_proteus_affine_hex8_laplacian_tc_apply_kernel<T>,
                                           0,
                                           0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    const ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_proteus_affine_hex8_laplacian_tc_apply_kernel<<<n_blocks, block_size, 0, s>>>(
                level, nelements, stride, interior_start, elements, macro_element_ops, x, y);
    } else {
        cu_proteus_affine_hex8_laplacian_tc_apply_kernel<<<n_blocks, block_size, 0>>>(
                level, nelements, stride, interior_start, elements, macro_element_ops, x, y);
    }
}

extern int cu_proteus_affine_hex8_laplacian_tc_apply(
        const int level,
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const ptrdiff_t interior_start,
        const idx_t *const SFEM_RESTRICT elements,
        const enum RealType real_type,
        const void *const SFEM_RESTRICT macro_element_ops,
        const void *const SFEM_RESTRICT x,
        void *const SFEM_RESTRICT y,
        void *stream) {

    switch (real_type) {
        case SFEM_FLOAT64:
            return cu_proteus_affine_hex8_laplacian_tc_apply_tpl<double>(
                    level, nelements, stride, interior_start, elements, (const double *)macro_element_ops, (const double *)x, (double *)y, stream);
        case SFEM_FLOAT32:
            return cu_proteus_affine_hex8_laplacian_tc_apply_tpl<float>(
                    level, nelements, stride, interior_start, elements, (const float *)macro_element_ops, (const float *)x, (float *)y, stream);
        case SFEM_REAL_DEFAULT:
            return cu_proteus_affine_hex8_laplacian_tc_apply_tpl<real_t>(
                    level, nelements, stride, interior_start, elements, (const real_t *)macro_element_ops, (const real_t *)x, (real_t *)y, stream);
        default: {
            fprintf(stderr,
                    "[Error] cu_proteus_affine_hex8_laplacian_tc_apply: not implemented for "
                    "type %s (code %d)\n",
                    real_type_to_string(real_type),
                    real_type);
            assert(0);
            return SFEM_FAILURE;
        }
    }
}
