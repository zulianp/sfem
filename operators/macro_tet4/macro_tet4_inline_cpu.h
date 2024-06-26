#ifndef MACRO_TET4_INLINE_CPU_H
#define MACRO_TET4_INLINE_CPU_H

#include "tet4_inline_cpu.h"

// FFF

static SFEM_INLINE void tet4_sub_fff_0(const scalar_t *const SFEM_RESTRICT fff,
                                       scalar_t *const SFEM_RESTRICT sub_fff) {
    sub_fff[0] = (1.0 / 2.0) * fff[0];
    sub_fff[1] = (1.0 / 2.0) * fff[1];
    sub_fff[2] = (1.0 / 2.0) * fff[2];
    sub_fff[3] = (1.0 / 2.0) * fff[3];
    sub_fff[4] = (1.0 / 2.0) * fff[4];
    sub_fff[5] = (1.0 / 2.0) * fff[5];

    assert(tet4_det_fff(sub_fff) > 0);
}

static SFEM_INLINE void tet4_sub_fff_4(const scalar_t *const SFEM_RESTRICT fff,
                                       scalar_t *const SFEM_RESTRICT sub_fff) {
    const scalar_t x0 = (1.0 / 2.0) * fff[0];
    const scalar_t x1 = (1.0 / 2.0) * fff[2];
    sub_fff[0] = fff[1] + (1.0 / 2.0) * fff[3] + x0;
    sub_fff[1] = -1.0 / 2.0 * fff[1] - x0;
    sub_fff[2] = (1.0 / 2.0) * fff[4] + x1;
    sub_fff[3] = x0;
    sub_fff[4] = -x1;
    sub_fff[5] = (1.0 / 2.0) * fff[5];

    assert(tet4_det_fff(sub_fff) > 0);
}

static SFEM_INLINE void tet4_sub_fff_5(const scalar_t *const SFEM_RESTRICT fff,
                                       scalar_t *const SFEM_RESTRICT sub_fff) {
    const scalar_t x0 = (1.0 / 2.0) * fff[3];
    const scalar_t x1 = fff[4] + (1.0 / 2.0) * fff[5] + x0;
    const scalar_t x2 = (1.0 / 2.0) * fff[4] + x0;
    const scalar_t x3 = (1.0 / 2.0) * fff[1];
    sub_fff[0] = x1;
    sub_fff[1] = -x2;
    sub_fff[2] = -1.0 / 2.0 * fff[2] - x1 - x3;
    sub_fff[3] = x0;
    sub_fff[4] = x2 + x3;
    sub_fff[5] = (1.0 / 2.0) * fff[0] + fff[1] + fff[2] + x1;

    assert(tet4_det_fff(sub_fff) > 0);
}

static SFEM_INLINE void tet4_sub_fff_6(const scalar_t *const SFEM_RESTRICT fff,
                                       scalar_t *const SFEM_RESTRICT sub_fff) {
    const scalar_t x0 = (1.0 / 2.0) * fff[3];
    const scalar_t x1 = (1.0 / 2.0) * fff[4];
    const scalar_t x2 = (1.0 / 2.0) * fff[1] + x0;
    sub_fff[0] = (1.0 / 2.0) * fff[0] + fff[1] + x0;
    sub_fff[1] = (1.0 / 2.0) * fff[2] + x1 + x2;
    sub_fff[2] = -x2;
    sub_fff[3] = fff[4] + (1.0 / 2.0) * fff[5] + x0;
    sub_fff[4] = -x0 - x1;
    sub_fff[5] = x0;

    assert(tet4_det_fff(sub_fff) > 0);
}

static SFEM_INLINE void tet4_sub_fff_7(const scalar_t *const SFEM_RESTRICT fff,
                                       scalar_t *const SFEM_RESTRICT sub_fff) {
    const scalar_t x0 = (1.0 / 2.0) * fff[5];
    const scalar_t x1 = (1.0 / 2.0) * fff[2];
    sub_fff[0] = x0;
    sub_fff[1] = -1.0 / 2.0 * fff[4] - x0;
    sub_fff[2] = -x1;
    sub_fff[3] = (1.0 / 2.0) * fff[3] + fff[4] + x0;
    sub_fff[4] = (1.0 / 2.0) * fff[1] + x1;
    sub_fff[5] = (1.0 / 2.0) * fff[0];

    assert(tet4_det_fff(sub_fff) > 0);
}

// Adjugate

static SFEM_INLINE void tet4_sub_adj_0(const scalar_t *const SFEM_RESTRICT adjugate,
                                       const ptrdiff_t stride,
                                       scalar_t *const SFEM_RESTRICT sub_adjugate) {
    sub_adjugate[0] = 2 * adjugate[0 * stride];
    sub_adjugate[1] = 2 * adjugate[1 * stride];
    sub_adjugate[2] = 2 * adjugate[2 * stride];
    sub_adjugate[3] = 2 * adjugate[3 * stride];
    sub_adjugate[4] = 2 * adjugate[4 * stride];
    sub_adjugate[5] = 2 * adjugate[5 * stride];
    sub_adjugate[6] = 2 * adjugate[6 * stride];
    sub_adjugate[7] = 2 * adjugate[7 * stride];
    sub_adjugate[8] = 2 * adjugate[8 * stride];
}
static SFEM_INLINE void tet4_sub_adj_4(const scalar_t *const SFEM_RESTRICT adjugate,
                                       const ptrdiff_t stride,
                                       scalar_t *const SFEM_RESTRICT sub_adjugate) {
    const scalar_t x0 = 2 * adjugate[0 * stride];
    const scalar_t x1 = 2 * adjugate[1 * stride];
    const scalar_t x2 = 2 * adjugate[2 * stride];
    sub_adjugate[0] = 2 * adjugate[3 * stride] + x0;
    sub_adjugate[1] = 2 * adjugate[4 * stride] + x1;
    sub_adjugate[2] = 2 * adjugate[5 * stride] + x2;
    sub_adjugate[3] = -x0;
    sub_adjugate[4] = -x1;
    sub_adjugate[5] = -x2;
    sub_adjugate[6] = 2 * adjugate[6 * stride];
    sub_adjugate[7] = 2 * adjugate[7 * stride];
    sub_adjugate[8] = 2 * adjugate[8 * stride];
}

static SFEM_INLINE void tet4_sub_adj_5(const scalar_t *const SFEM_RESTRICT adjugate,
                                       const ptrdiff_t stride,
                                       scalar_t *const SFEM_RESTRICT sub_adjugate) {
    const scalar_t x0 = 2 * adjugate[3 * stride];
    const scalar_t x1 = 2 * adjugate[6 * stride] + x0;
    const scalar_t x2 = 2 * adjugate[4 * stride];
    const scalar_t x3 = 2 * adjugate[7 * stride] + x2;
    const scalar_t x4 = 2 * adjugate[5 * stride];
    const scalar_t x5 = 2 * adjugate[8 * stride] + x4;
    sub_adjugate[0] = -x1;
    sub_adjugate[1] = -x3;
    sub_adjugate[2] = -x5;
    sub_adjugate[3] = x0;
    sub_adjugate[4] = x2;
    sub_adjugate[5] = x4;
    sub_adjugate[6] = 2 * adjugate[0 * stride] + x1;
    sub_adjugate[7] = 2 * adjugate[1 * stride] + x3;
    sub_adjugate[8] = 2 * adjugate[2 * stride] + x5;
}

static SFEM_INLINE void tet4_sub_adj_6(const scalar_t *const SFEM_RESTRICT adjugate,
                                       const ptrdiff_t stride,
                                       scalar_t *const SFEM_RESTRICT sub_adjugate) {
    const scalar_t x0 = 2 * adjugate[3 * stride];
    const scalar_t x1 = 2 * adjugate[4 * stride];
    const scalar_t x2 = 2 * adjugate[5 * stride];
    sub_adjugate[0] = 2 * adjugate[0 * stride] + x0;
    sub_adjugate[1] = 2 * adjugate[1 * stride] + x1;
    sub_adjugate[2] = 2 * adjugate[2 * stride] + x2;
    sub_adjugate[3] = 2 * adjugate[6 * stride] + x0;
    sub_adjugate[4] = 2 * adjugate[7 * stride] + x1;
    sub_adjugate[5] = 2 * adjugate[8 * stride] + x2;
    sub_adjugate[6] = -x0;
    sub_adjugate[7] = -x1;
    sub_adjugate[8] = -x2;
}

static SFEM_INLINE void tet4_sub_adj_7(const scalar_t *const SFEM_RESTRICT adjugate,
                                       const ptrdiff_t stride,
                                       scalar_t *const SFEM_RESTRICT sub_adjugate) {
    const scalar_t x0 = 2 * adjugate[6 * stride];
    const scalar_t x1 = 2 * adjugate[7 * stride];
    const scalar_t x2 = 2 * adjugate[8 * stride];
    sub_adjugate[0] = -x0;
    sub_adjugate[1] = -x1;
    sub_adjugate[2] = -x2;
    sub_adjugate[3] = 2 * adjugate[3 * stride] + x0;
    sub_adjugate[4] = 2 * adjugate[4 * stride] + x1;
    sub_adjugate[5] = 2 * adjugate[5 * stride] + x2;
    sub_adjugate[6] = 2 * adjugate[0 * stride];
    sub_adjugate[7] = 2 * adjugate[1 * stride];
    sub_adjugate[8] = 2 * adjugate[2 * stride];
}

#define tet4_gather_idx(from, i0, i1, i2, i3, to) \
    do {                                     \
        to[0] = from[i0];                    \
        to[1] = from[i1];                    \
        to[2] = from[i2];                    \
        to[3] = from[i3];                    \
    } while (0);

#endif  // MACRO_TET4_INLINE_CPU_H
