#ifndef SFEM_MASK_H
#define SFEM_MASK_H

#include "sfem_base.h"
#include <stddef.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

mask_t * mask_create(ptrdiff_t n);
void mask_destroy(mask_t *ptr);

static SFEM_INLINE ptrdiff_t mask_count(ptrdiff_t n) {
	return (n + sizeof(mask_t) - 1)/sizeof(mask_t);
}

static SFEM_INLINE int mask_get(ptrdiff_t i, const mask_t *const mask) 
{	

	const static size_t nbits = (sizeof(mask_t) * 8);
	const ptrdiff_t idx = i/nbits;
	mask_t shift = i - idx * nbits;
	
	assert(shift < nbits);
	assert(shift >= 0);

	mask_t m = mask[idx];
	mask_t q = 1 << shift;
	return !!(m & q);
}

static SFEM_INLINE void mask_set(ptrdiff_t i, mask_t *const mask) 
{
	const static size_t nbits = (sizeof(mask_t) * 8);
	const ptrdiff_t idx = i/nbits;
	mask_t shift = i - idx * nbits;
	mask_t q = 1 << shift;

#pragma omp atomic update
	mask[idx] |= q;
}

#ifdef __cplusplus
}
#endif


#endif //SFEM_MASK_H
