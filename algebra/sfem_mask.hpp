#ifndef SFEM_MASK_HPP
#define SFEM_MASK_HPP

#include <assert.h>
#include <stddef.h>

#include "sfem_base.hpp"

namespace sfem {

static SFEM_INLINE ptrdiff_t mask_count(ptrdiff_t n) { return (n + sizeof(mask_t) - 1) / sizeof(mask_t); }

static SFEM_INLINE int mask_get(ptrdiff_t i, const mask_t *const mask) {
    const static size_t nbits = (sizeof(mask_t) * 8);
    const ptrdiff_t     idx   = i / nbits;
    mask_t              shift = i - idx * nbits;

    assert(shift < nbits);
    assert(shift >= 0);

    mask_t m;
#pragma omp atomic read
    m = mask[idx];
    mask_t q = 1 << shift;
    return !!(m & q);
}

static SFEM_INLINE void mask_set(ptrdiff_t i, mask_t *const mask) {
    const static size_t nbits = (sizeof(mask_t) * 8);
    const ptrdiff_t     idx   = i / nbits;
    mask_t              shift = i - idx * nbits;
    mask_t              q     = 1 << shift;

#pragma omp atomic update
    mask[idx] |= q;
}

static SFEM_INLINE void mask_unset(ptrdiff_t i, mask_t *const mask) {
    const static size_t nbits = (sizeof(mask_t) * 8);
    const ptrdiff_t     idx   = i / nbits;
    mask_t              shift = i - idx * nbits;
    mask_t              q     = 1 << shift;

    {
        mask_t m;
#pragma omp atomic read
        m = mask[idx];

        if (!(q & m)) {
            return;
        }
    }

#pragma omp atomic update
    mask[idx] = q ^ mask[idx];
}

}  // namespace sfem

using sfem::mask_count;
using sfem::mask_get;
using sfem::mask_set;
using sfem::mask_unset;

#endif  // SFEM_MASK_HPP
