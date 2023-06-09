#include "sfem_base.h"

#include "sfem_cuda_base.h"

#include "FE3D_phase_field_for_fracture_kernels.h"

static inline __device__ __host__ void jacobian_inverse_micro_kernel(const real_t px0,
                                                                     const real_t px1,
                                                                     const real_t px2,
                                                                     const real_t px3,
                                                                     const real_t py0,
                                                                     const real_t py1,
                                                                     const real_t py2,
                                                                     const real_t py3,
                                                                     const real_t pz0,
                                                                     const real_t pz1,
                                                                     const real_t pz2,
                                                                     const real_t pz3,
                                                                     const count_t stride,
                                                                     real_t *jacobian_inverse) {
    const real_t x0 = -py0 + py2;
    const real_t x1 = -pz0 + pz3;
    const real_t x2 = x0 * x1;
    const real_t x3 = -py0 + py3;
    const real_t x4 = -pz0 + pz2;
    const real_t x5 = x3 * x4;
    const real_t x6 = -px0 + px1;
    const real_t x7 = x2 * x6;
    const real_t x8 = -pz0 + pz1;
    const real_t x9 = -px0 + px2;
    const real_t x10 = x3 * x9;
    const real_t x11 = x10 * x8;
    const real_t x12 = -py0 + py1;
    const real_t x13 = -px0 + px3;
    const real_t x14 = x12 * x13 * x4;
    const real_t x15 = x5 * x6;
    const real_t x16 = x1 * x9;
    const real_t x17 = x12 * x16;
    const real_t x18 = x0 * x13;
    const real_t x19 = x18 * x8;
    const real_t x20 = 1.0 / (x11 + x14 - x15 - x17 - x19 + x7);
    jacobian_inverse[0 * stride] = x20 * (x2 - x5);
    jacobian_inverse[1 * stride] = x20 * (x13 * x4 - x16);
    jacobian_inverse[2 * stride] = x20 * (x10 - x18);
    jacobian_inverse[3 * stride] = x20 * (-x1 * x12 + x3 * x8);
    jacobian_inverse[4 * stride] = x20 * (x1 * x6 - x13 * x8);
    jacobian_inverse[5 * stride] = x20 * (x12 * x13 - x3 * x6);
    jacobian_inverse[6 * stride] = x20 * (-x0 * x8 + x12 * x4);
    jacobian_inverse[7 * stride] = x20 * (-x4 * x6 + x8 * x9);
    jacobian_inverse[8 * stride] = x20 * (x0 * x6 - x12 * x9);
}

__global__ void jacobian_inverse_kernel(const ptrdiff_t nelements,
                                        const geom_t *const SFEM_RESTRICT xyz,
                                        real_t *const SFEM_RESTRICT jacobian_inverse) {
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {
        // Thy element coordinates and jacobian
        const geom_t *const this_xyz = &xyz[e];
        real_t *const this_jacobian_inverse = &jacobian_inverse[e];

        const ptrdiff_t xi = 0 * 4;
        const ptrdiff_t yi = 1 * 4;
        const ptrdiff_t zi = 2 * 4;

        jacobian_inverse_micro_kernel(
            // X-coordinates
            this_xyz[(xi + 0) * nelements],
            this_xyz[(xi + 1) * nelements],
            this_xyz[(xi + 2) * nelements],
            this_xyz[(xi + 3) * nelements],
            // Y-coordinates
            this_xyz[(yi + 0) * nelements],
            this_xyz[(yi + 1) * nelements],
            this_xyz[(yi + 2) * nelements],
            this_xyz[(yi + 3) * nelements],
            // Z-coordinates
            this_xyz[(zi + 0) * nelements],
            this_xyz[(zi + 1) * nelements],
            this_xyz[(zi + 2) * nelements],
            this_xyz[(zi + 3) * nelements],
            nelements,
            this_jacobian_inverse);
    }
}


static inline __device__ __host__ int linear_search(const idx_t target, const idx_t *const arr, const int size) {
    int i;
    for (i = 0; i < size - SFEM_VECTOR_SIZE; i += SFEM_VECTOR_SIZE) {
        if (arr[i] == target) return i;
        if (arr[i + 1] == target) return i + 1;
        if (arr[i + 2] == target) return i + 2;
        if (arr[i + 3] == target) return i + 3;
    }
    for (; i < size; i++) {
        if (arr[i] == target) return i;
    }
    return -1;
}

static inline __device__ __host__ int find_col(const idx_t key, const idx_t *const row, const int lenrow) {
    // if (lenrow <= 32)
    // {
    return linear_search(key, row, lenrow);

    // Using sentinel (potentially dangerous if matrix is buggy and column does not exist)
    // while (key > row[++k]) {
    //     // Hi
    // }
    // assert(k < lenrow);
    // assert(key == row[k]);
    // } else {
    //     // Use this for larger number of dofs per row
    //     return find_idx_binary_search(key, row, lenrow);
    // }
}

static inline __device__ __host__ void find_cols4(const idx_t *targets,
                                                  const idx_t *const row,
                                                  const int lenrow,
                                                  int *ks) {
    if (lenrow > 32) {
        for (int d = 0; d < 4; ++d) {
            ks[d] = find_col(targets[d], row, lenrow);
        }
    } else {
#pragma unroll(4)
        for (int d = 0; d < 4; ++d) {
            ks[d] = 0;
        }

        for (int i = 0; i < lenrow; ++i) {
#pragma unroll(4)
            for (int d = 0; d < 4; ++d) {
                ks[d] += row[i] < targets[d];
            }
        }
    }
}
