#ifndef CU_MASK_CUH
#define CU_MASK_CUH

#include "sfem_mask.h"

static inline __device__ int cu_mask_get(ptrdiff_t i, const mask_t* const mask) {
    const static size_t nbits = (sizeof(mask_t) * 8);
    const ptrdiff_t     idx   = i / nbits;
    mask_t              shift = i - idx * nbits;

    assert(shift < nbits);
    assert(shift >= 0);

    mask_t m;
    m        = mask[idx];
    mask_t q = 1 << shift;
    return !!(m & q);
}

static inline __device__ void cu_mask_set(ptrdiff_t i, mask_t* const mask) {
    const static size_t nbits = (sizeof(mask_t) * 8);
    const ptrdiff_t     idx   = i / nbits;
    mask_t              shift = i - idx * nbits;
    mask_t              q     = 1 << shift;
    mask[idx] |= q;
}

static inline __device__ void cu_mask_unset(ptrdiff_t i, mask_t* const mask) {
    const static size_t nbits = (sizeof(mask_t) * 8);
    const ptrdiff_t     idx   = i / nbits;
    mask_t              shift = i - idx * nbits;
    mask_t              q     = 1 << shift;
    mask[idx]                 = q ^ mask[idx];
}

static inline __device__ void cu_mask_atomic_set(ptrdiff_t i, mask_t* const mask) {
    const static size_t nbits = (sizeof(mask_t) * 8);
    const ptrdiff_t     idx   = i / nbits;
    mask_t              shift = i - idx * nbits;
    mask_t              q     = 1 << shift;
    atomicOr(&mask[idx], q);

}

#endif // CU_MASK_CUH