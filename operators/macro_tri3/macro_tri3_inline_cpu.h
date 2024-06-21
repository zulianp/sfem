#ifndef MACRO_TET4_INLINE_CPU_H
#define MACRO_TET4_INLINE_CPU_H

#include "tri3_inline_cpu.h"

static SFEM_INLINE void tri3_sub_fff_1(const jacobian_t *const SFEM_RESTRICT fff,
                                       jacobian_t *const SFEM_RESTRICT sub_fff) {
    sub_fff[0] = fff[0] + 2 * fff[1] + fff[2];
    sub_fff[1] = -fff[0] - fff[1];
    sub_fff[2] = fff[0];
}

#define tri3_gather_idx(from, i0, i1, i2, to) \
    do {                                      \
        to[0] = from[i0];                     \
        to[1] = from[i1];                     \
        to[2] = from[i2];                     \
    } while (0);

#endif  // MACRO_TET4_INLINE_CPU_H
