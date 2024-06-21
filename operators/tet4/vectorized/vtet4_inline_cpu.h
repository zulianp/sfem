#ifndef VTET4_INLINE_CPU_H
#define VTET4_INLINE_CPU_H

#include "operator_inline_cpu.h"
#include "tet4_inline_cpu.h"

static SFEM_INLINE void vtet4_fff(const vscalar_t px0,
                                  const vscalar_t px1,
                                  const vscalar_t px2,
                                  const vscalar_t px3,
                                  const vscalar_t py0,
                                  const vscalar_t py1,
                                  const vscalar_t py2,
                                  const vscalar_t py3,
                                  const vscalar_t pz0,
                                  const vscalar_t pz1,
                                  const vscalar_t pz2,
                                  const vscalar_t pz3,
                                  vscalar_t *const fff) {
    const vscalar_t x0 = -px0 + px1;
    const vscalar_t x1 = -py0 + py2;
    const vscalar_t x2 = -pz0 + pz3;
    const vscalar_t x3 = x1 * x2;
    const vscalar_t x4 = x0 * x3;
    const vscalar_t x5 = -py0 + py3;
    const vscalar_t x6 = -pz0 + pz2;
    const vscalar_t x7 = x5 * x6;
    const vscalar_t x8 = x0 * x7;
    const vscalar_t x9 = -py0 + py1;
    const vscalar_t x10 = -px0 + px2;
    const vscalar_t x11 = x10 * x2;
    const vscalar_t x12 = x11 * x9;
    const vscalar_t x13 = -pz0 + pz1;
    const vscalar_t x14 = x10 * x5;
    const vscalar_t x15 = x13 * x14;
    const vscalar_t x16 = -px0 + px3;
    const vscalar_t x17 = x16 * x6 * x9;
    const vscalar_t x18 = x1 * x16;
    const vscalar_t x19 = x13 * x18;
    const vscalar_t x20 = -(scalar_t)(1.0 / 6.0) * x12 + (scalar_t)(1.0 / 6.0) * x15 +
                          (scalar_t)(1.0 / 6.0) * x17 - (scalar_t)(1.0 / 6.0) * x19 +
                          (scalar_t)(1.0 / 6.0) * x4 - (scalar_t)(1.0 / 6.0) * x8;
    const vscalar_t x21 = x14 - x18;
    const vscalar_t x22 = (scalar_t)1. / POW2(-x12 + x15 + x17 - x19 + x4 - x8);
    const vscalar_t x23 = -x11 + x16 * x6;
    const vscalar_t x24 = x3 - x7;
    const vscalar_t x25 = -x0 * x5 + x16 * x9;
    const vscalar_t x26 = x21 * x22;
    const vscalar_t x27 = x0 * x2 - x13 * x16;
    const vscalar_t x28 = x22 * x23;
    const vscalar_t x29 = x13 * x5 - x2 * x9;
    const vscalar_t x30 = x22 * x24;
    const vscalar_t x31 = x0 * x1 - x10 * x9;
    const vscalar_t x32 = -x0 * x6 + x10 * x13;
    const vscalar_t x33 = -x1 * x13 + x6 * x9;
    fff[0] = x20 * (POW2(x21) * x22 + x22 * POW2(x23) + x22 * POW2(x24));
    fff[1] = x20 * (x25 * x26 + x27 * x28 + x29 * x30);
    fff[2] = x20 * (x26 * x31 + x28 * x32 + x30 * x33);
    fff[3] = x20 * (x22 * POW2(x25) + x22 * POW2(x27) + x22 * POW2(x29));
    fff[4] = x20 * (x22 * x25 * x31 + x22 * x27 * x32 + x22 * x29 * x33);
    fff[5] = x20 * (x22 * POW2(x31) + x22 * POW2(x32) + x22 * POW2(x33));
}

static SFEM_INLINE vscalar_t vtet4_det_fff(const vscalar_t *const fff) {
    return fff[0] * fff[3] * fff[5] - fff[0] * POW2(fff[4]) - POW2(fff[1]) * fff[5] +
           2 * fff[1] * fff[2] * fff[4] - POW2(fff[2]) * fff[3];
}

static SFEM_INLINE void vtet4_adjugate_and_det(const vscalar_t px0,
                                               const vscalar_t px1,
                                               const vscalar_t px2,
                                               const vscalar_t px3,
                                               const vscalar_t py0,
                                               const vscalar_t py1,
                                               const vscalar_t py2,
                                               const vscalar_t py3,
                                               const vscalar_t pz0,
                                               const vscalar_t pz1,
                                               const vscalar_t pz2,
                                               const vscalar_t pz3,
                                               vscalar_t *const SFEM_RESTRICT adjugate,
                                               vscalar_t *const SFEM_RESTRICT
                                                       jacobian_determinant) {
    // Compute jacobian in high precision
    vscalar_t jacobian[9];
    jacobian[0] = -px0 + px1;
    jacobian[1] = -px0 + px2;
    jacobian[2] = -px0 + px3;
    jacobian[3] = -py0 + py1;
    jacobian[4] = -py0 + py2;
    jacobian[5] = -py0 + py3;
    jacobian[6] = -pz0 + pz1;
    jacobian[7] = -pz0 + pz2;
    jacobian[8] = -pz0 + pz3;

    const vscalar_t x0 = jacobian[4] * jacobian[8];
    const vscalar_t x1 = jacobian[5] * jacobian[7];
    const vscalar_t x2 = jacobian[1] * jacobian[8];
    const vscalar_t x3 = jacobian[1] * jacobian[5];
    const vscalar_t x4 = jacobian[2] * jacobian[4];

    // Store adjugate in lower precision
    adjugate[0] = x0 - x1;
    adjugate[1] = jacobian[2] * jacobian[7] - x2;
    adjugate[2] = x3 - x4;
    adjugate[3] = -jacobian[3] * jacobian[8] + jacobian[5] * jacobian[6];
    adjugate[4] = jacobian[0] * jacobian[8] - jacobian[2] * jacobian[6];
    adjugate[5] = -jacobian[0] * jacobian[5] + jacobian[2] * jacobian[3];
    adjugate[6] = jacobian[3] * jacobian[7] - jacobian[4] * jacobian[6];
    adjugate[7] = -jacobian[0] * jacobian[7] + jacobian[1] * jacobian[6];
    adjugate[8] = jacobian[0] * jacobian[4] - jacobian[1] * jacobian[3];

    // Store determinant in lower precision
    jacobian_determinant[0] = jacobian[0] * x0 - jacobian[0] * x1 +
                              jacobian[2] * jacobian[3] * jacobian[7] - jacobian[3] * x2 +
                              jacobian[6] * x3 - jacobian[6] * x4;
}

#endif  // VTET4_INLINE_CPU_H
