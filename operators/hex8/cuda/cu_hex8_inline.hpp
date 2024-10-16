#ifndef CU_HEX_INLINE_HPP
#define CU_HEX_INLINE_HPP

#include "sfem_defs.h"

#ifndef POW2
#define POW2(a) ((a) * (a))
#endif

#ifndef POW3
#define POW3(a) ((a) * (a) * (a))
#endif

#ifndef MAX
#define MAX(a, b) ((a) < (b) ? (b) : (a))
#endif

#ifndef MIN
#define MIN(a, b) ((a) > (b) ? (b) : (a))
#endif

template <typename geom_t, typename scalar_t, typename jacobian_t>
static inline __host__ __device__ void cu_hex8_adjugate_and_det(
        const geom_t *const SFEM_RESTRICT x,
        const geom_t *const SFEM_RESTRICT y,
        const geom_t *const SFEM_RESTRICT z,
        const scalar_t qx,
        const scalar_t qy,
        const scalar_t qz,
        const ptrdiff_t stride,
        jacobian_t *const SFEM_RESTRICT adjugate,
        jacobian_t *const SFEM_RESTRICT jacobian_determinant) {
    scalar_t jacobian[9];
    {
        const scalar_t x0 = qy * qz;
        const scalar_t x1 = 1 - qz;
        const scalar_t x2 = qy * x1;
        const scalar_t x3 = 1 - qy;
        const scalar_t x4 = qz * x3;
        const scalar_t x5 = x1 * x3;
        const scalar_t x6 = qx * qz;
        const scalar_t x7 = qx * x1;
        const scalar_t x8 = 1 - qx;
        const scalar_t x9 = qz * x8;
        const scalar_t x10 = x1 * x8;
        const scalar_t x11 = qx * qy;
        const scalar_t x12 = qx * x3;
        const scalar_t x13 = qy * x8;
        const scalar_t x14 = x3 * x8;

        jacobian[0] = x0 * x[6] - x0 * x[7] + x2 * x[2] - x2 * x[3] - x4 * x[4] + x4 * x[5] -
                      x5 * x[0] + x5 * x[1];
        jacobian[1] = qx * qz * x[6] + qx * x1 * x[2] + qz * x8 * x[7] + x1 * x8 * x[3] -
                      x10 * x[0] - x6 * x[5] - x7 * x[1] - x9 * x[4];
        jacobian[2] = qx * qy * x[6] + qx * x3 * x[5] + qy * x8 * x[7] - x11 * x[2] - x12 * x[1] -
                      x13 * x[3] - x14 * x[0] + x3 * x8 * x[4];
        jacobian[3] = x0 * y[6] - x0 * y[7] + x2 * y[2] - x2 * y[3] - x4 * y[4] + x4 * y[5] -
                      x5 * y[0] + x5 * y[1];
        jacobian[4] = qx * qz * y[6] + qx * x1 * y[2] + qz * x8 * y[7] + x1 * x8 * y[3] -
                      x10 * y[0] - x6 * y[5] - x7 * y[1] - x9 * y[4];
        jacobian[5] = qx * qy * y[6] + qx * x3 * y[5] + qy * x8 * y[7] - x11 * y[2] - x12 * y[1] -
                      x13 * y[3] - x14 * y[0] + x3 * x8 * y[4];
        jacobian[6] = x0 * z[6] - x0 * z[7] + x2 * z[2] - x2 * z[3] - x4 * z[4] + x4 * z[5] -
                      x5 * z[0] + x5 * z[1];
        jacobian[7] = qx * qz * z[6] + qx * x1 * z[2] + qz * x8 * z[7] + x1 * x8 * z[3] -
                      x10 * z[0] - x6 * z[5] - x7 * z[1] - x9 * z[4];
        jacobian[8] = qx * qy * z[6] + qx * x3 * z[5] + qy * x8 * z[7] - x11 * z[2] - x12 * z[1] -
                      x13 * z[3] - x14 * z[0] + x3 * x8 * z[4];
    }

    const scalar_t x0 = jacobian[4] * jacobian[8];
    const scalar_t x1 = jacobian[5] * jacobian[7];
    const scalar_t x2 = jacobian[1] * jacobian[8];
    const scalar_t x3 = jacobian[1] * jacobian[5];
    const scalar_t x4 = jacobian[2] * jacobian[4];

    adjugate[0 * stride] = x0 - x1;
    adjugate[1 * stride] = jacobian[2] * jacobian[7] - x2;
    adjugate[2 * stride] = x3 - x4;
    adjugate[3 * stride] = -jacobian[3] * jacobian[8] + jacobian[5] * jacobian[6];
    adjugate[4 * stride] = jacobian[0] * jacobian[8] - jacobian[2] * jacobian[6];
    adjugate[5 * stride] = -jacobian[0] * jacobian[5] + jacobian[2] * jacobian[3];
    adjugate[6 * stride] = jacobian[3] * jacobian[7] - jacobian[4] * jacobian[6];
    adjugate[7 * stride] = -jacobian[0] * jacobian[7] + jacobian[1] * jacobian[6];
    adjugate[8 * stride] = jacobian[0] * jacobian[4] - jacobian[1] * jacobian[3];
    *jacobian_determinant = jacobian[0] * x0 - jacobian[0] * x1 +
                            jacobian[2] * jacobian[3] * jacobian[7] - jacobian[3] * x2 +
                            jacobian[6] * x3 - jacobian[6] * x4;
}

template <typename adjugate_t, typename determinat_t, typename scalar_t>
static __host__ __device__ void cu_hex8_sub_adj_0(const ptrdiff_t stride,
                                                  const adjugate_t *const SFEM_RESTRICT adjugate,
                                                  const determinat_t determinant,
                                                  const scalar_t h,
                                                  scalar_t *const SFEM_RESTRICT sub_adjugate,
                                                  scalar_t *const SFEM_RESTRICT sub_determinant) {
    const scalar_t x0 = POW2(h);
    sub_adjugate[0] = (scalar_t)adjugate[0 * stride] * x0;
    sub_adjugate[1] = (scalar_t)adjugate[1 * stride] * x0;
    sub_adjugate[2] = (scalar_t)adjugate[2 * stride] * x0;
    sub_adjugate[3] = (scalar_t)adjugate[3 * stride] * x0;
    sub_adjugate[4] = (scalar_t)adjugate[4 * stride] * x0;
    sub_adjugate[5] = (scalar_t)adjugate[5 * stride] * x0;
    sub_adjugate[6] = (scalar_t)adjugate[6 * stride] * x0;
    sub_adjugate[7] = (scalar_t)adjugate[7 * stride] * x0;
    sub_adjugate[8] = (scalar_t)adjugate[8 * stride] * x0;
    sub_determinant[0] = (scalar_t)determinant * (POW3(h));
}

template <typename scalar_t>
static __host__ __device__ void cu_hex8_sub_adj_0_in_place(
                                                  const scalar_t h,
                                                  scalar_t *const SFEM_RESTRICT adjugate,
                                                  scalar_t *const SFEM_RESTRICT determinant) {
    const scalar_t x0 = POW2(h);
    adjugate[0] = (scalar_t)adjugate[0] * x0;
    adjugate[1] = (scalar_t)adjugate[1] * x0;
    adjugate[2] = (scalar_t)adjugate[2] * x0;
    adjugate[3] = (scalar_t)adjugate[3] * x0;
    adjugate[4] = (scalar_t)adjugate[4] * x0;
    adjugate[5] = (scalar_t)adjugate[5] * x0;
    adjugate[6] = (scalar_t)adjugate[6] * x0;
    adjugate[7] = (scalar_t)adjugate[7] * x0;
    adjugate[8] = (scalar_t)adjugate[8] * x0;
    determinant[0] *= POW3(h);
}

#endif  // CU_HEX_INLINE_HPP
