#include "sfem_mask.h"


#include <assert.h>

mask_t * mask_create(ptrdiff_t n)
{
	ptrdiff_t nm = mask_count(n);
	mask_t * mem = calloc(nm, sizeof(mask_t));
	assert(mem);
	return mem;
}

void mask_destroy(mask_t *ptr)
{
	free(ptr);
}
