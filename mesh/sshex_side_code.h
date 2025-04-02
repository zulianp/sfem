#ifndef SSHEX_SIDE_CODE_H
#define SSHEX_SIDE_CODE_H

#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

static SFEM_INLINE void sshex_coords_from_side_code(const sshex_side_code_t code,
                                                    const uint8_t           side,
                                                    uint8_t                *begin,
                                                    uint8_t                *end) {
    assert(side < 8);

    // 6 bits per side
    uint8_t begin_code = (code & (7 << (2 * side)) >> side);
    uint8_t end_code   = (code & (7 << (2 * side + 1)) >> side);

    begin[0] = begin_code & 1;
    begin[1] = begin_code & 2;
    begin[2] = begin_code & 4;

    end[0] = end_code & 1;
    end[1] = end_code & 2;
    end[2] = end_code & 4;

    assert(begin[0] <= 1);
    assert(begin[1] <= 1);
    assert(begin[2] <= 1);

    assert(end[0] <= 1);
    assert(end[1] <= 1);
    assert(end[2] <= 1);
}

static SFEM_INLINE sshex_side_code_t sshex_side_code_from_coords(const uint8_t side, const uint8_t *begin, const uint8_t *end) {
    assert(side < 8);
    assert(begin[0] <= 1);
    assert(begin[1] <= 1);
    assert(begin[2] <= 1);

    assert(end[0] <= 1);
    assert(end[1] <= 1);
    assert(end[2] <= 1);

    sshex_side_code_t ret        = 0;
    uint8_t           begin_code = (begin[0] | begin[1] * 2 | begin[2] * 4) << (2 * side);
    uint8_t           end_code   = (end[0] | end[1] * 2 | end[2] * 4) & (7 << (2 * side + 1));
    return begin_code | end_code;
}

#ifdef __cplusplus
}
#endif

#endif  // SSHEX_SIDE_CODE_H
