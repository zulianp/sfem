#ifndef POW2
#define POW2(a) ((a) * (a))
#endif

template <typename geom_t, typename jacobian_t, typename jacobian_determinant_t>
static inline __device__ __host__ void tet4_adjugate_and_det(
        const geom_t px0,
        const geom_t px1,
        const geom_t px2,
        const geom_t px3,
        const geom_t py0,
        const geom_t py1,
        const geom_t py2,
        const geom_t py3,
        const geom_t pz0,
        const geom_t pz1,
        const geom_t pz2,
        const geom_t pz3,
        const ptrdiff_t stride,
        jacobian_t *const SFEM_RESTRICT adjugate,
        jacobian_determinant_t *const SFEM_RESTRICT jacobian_determinant) {
    // Compute jacobian in high precision
    geom_t jacobian[9];
    jacobian[0] = -px0 + px1;
    jacobian[1] = -px0 + px2;
    jacobian[2] = -px0 + px3;
    jacobian[3] = -py0 + py1;
    jacobian[4] = -py0 + py2;
    jacobian[5] = -py0 + py3;
    jacobian[6] = -pz0 + pz1;
    jacobian[7] = -pz0 + pz2;
    jacobian[8] = -pz0 + pz3;

    const geom_t x0 = jacobian[4] * jacobian[8];
    const geom_t x1 = jacobian[5] * jacobian[7];
    const geom_t x2 = jacobian[1] * jacobian[8];
    const geom_t x3 = jacobian[1] * jacobian[5];
    const geom_t x4 = jacobian[2] * jacobian[4];

    // Store adjugate in lower precision
    adjugate[0 * stride] = x0 - x1;
    adjugate[1 * stride] = jacobian[2] * jacobian[7] - x2;
    adjugate[2 * stride] = x3 - x4;
    adjugate[3 * stride] = -jacobian[3] * jacobian[8] + jacobian[5] * jacobian[6];
    adjugate[4 * stride] = jacobian[0] * jacobian[8] - jacobian[2] * jacobian[6];
    adjugate[5 * stride] = -jacobian[0] * jacobian[5] + jacobian[2] * jacobian[3];
    adjugate[6 * stride] = jacobian[3] * jacobian[7] - jacobian[4] * jacobian[6];
    adjugate[7 * stride] = -jacobian[0] * jacobian[7] + jacobian[1] * jacobian[6];
    adjugate[8 * stride] = jacobian[0] * jacobian[4] - jacobian[1] * jacobian[3];

    // Store determinant in lower precision
    jacobian_determinant[0] = jacobian[0] * x0 - jacobian[0] * x1 +
                              jacobian[2] * jacobian[3] * jacobian[7] - jacobian[3] * x2 +
                              jacobian[6] * x3 - jacobian[6] * x4;
}

template <typename geom_t, typename fff_t>
static inline __device__ __host__ void tet4_fff(const geom_t px0,
                                                const geom_t px1,
                                                const geom_t px2,
                                                const geom_t px3,
                                                const geom_t py0,
                                                const geom_t py1,
                                                const geom_t py2,
                                                const geom_t py3,
                                                const geom_t pz0,
                                                const geom_t pz1,
                                                const geom_t pz2,
                                                const geom_t pz3,
                                                const ptrdiff_t stride,
                                                fff_t *const fff) {
    const geom_t x0 = -px0 + px1;
    const geom_t x1 = -py0 + py2;
    const geom_t x2 = -pz0 + pz3;
    const geom_t x3 = x1 * x2;
    const geom_t x4 = x0 * x3;
    const geom_t x5 = -py0 + py3;
    const geom_t x6 = -pz0 + pz2;
    const geom_t x7 = x5 * x6;
    const geom_t x8 = x0 * x7;
    const geom_t x9 = -py0 + py1;
    const geom_t x10 = -px0 + px2;
    const geom_t x11 = x10 * x2;
    const geom_t x12 = x11 * x9;
    const geom_t x13 = -pz0 + pz1;
    const geom_t x14 = x10 * x5;
    const geom_t x15 = x13 * x14;
    const geom_t x16 = -px0 + px3;
    const geom_t x17 = x16 * x6 * x9;
    const geom_t x18 = x1 * x16;
    const geom_t x19 = x13 * x18;
    const geom_t x20 = -(geom_t)(1.0 / 6.0) * x12 + (geom_t)(1.0 / 6.0) * x15 +
                       (geom_t)(1.0 / 6.0) * x17 - (geom_t)(1.0 / 6.0) * x19 +
                       (geom_t)(1.0 / 6.0) * x4 - (geom_t)(1.0 / 6.0) * x8;
    const geom_t x21 = x14 - x18;
    const geom_t x22 = 1. / POW2(-x12 + x15 + x17 - x19 + x4 - x8);
    const geom_t x23 = -x11 + x16 * x6;
    const geom_t x24 = x3 - x7;
    const geom_t x25 = -x0 * x5 + x16 * x9;
    const geom_t x26 = x21 * x22;
    const geom_t x27 = x0 * x2 - x13 * x16;
    const geom_t x28 = x22 * x23;
    const geom_t x29 = x13 * x5 - x2 * x9;
    const geom_t x30 = x22 * x24;
    const geom_t x31 = x0 * x1 - x10 * x9;
    const geom_t x32 = -x0 * x6 + x10 * x13;
    const geom_t x33 = -x1 * x13 + x6 * x9;
    fff[0 * stride] = x20 * (POW2(x21) * x22 + x22 * POW2(x23) + x22 * POW2(x24));
    fff[1 * stride] = x20 * (x25 * x26 + x27 * x28 + x29 * x30);
    fff[2 * stride] = x20 * (x26 * x31 + x28 * x32 + x30 * x33);
    fff[3 * stride] = x20 * (x22 * POW2(x25) + x22 * POW2(x27) + x22 * POW2(x29));
    fff[4 * stride] = x20 * (x22 * x25 * x31 + x22 * x27 * x32 + x22 * x29 * x33);
    fff[5 * stride] = x20 * (x22 * POW2(x31) + x22 * POW2(x32) + x22 * POW2(x33));
}

template <typename idx_t>
static inline __device__ __host__ int tet4_linear_search(const idx_t target,
                                                         const idx_t *const arr,
                                                         const int size) {
    int i;
    for (i = 0; i < size - 4; i += 4) {
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

template <typename idx_t>
static inline __device__ __host__ int tet4_find_col(const idx_t key,
                                                    const idx_t *const row,
                                                    const int lenrow) {
    return tet4_linear_search(key, row, lenrow);
}

template <typename idx_t>
static inline __device__ __host__ void tet4_find_cols(const idx_t *SFEM_RESTRICT targets,
                                                      const idx_t *const SFEM_RESTRICT row,
                                                      const int lenrow,
                                                      int *SFEM_RESTRICT ks) {
    if (lenrow > 32) {
        for (int d = 0; d < 4; ++d) {
            ks[d] = tet4_find_col(targets[d], row, lenrow);
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
