#include "sfem_mask.h"

#include <assert.h>
#include <stdio.h>

mask_t *mask_create(ptrdiff_t n) {
    ptrdiff_t nm = mask_count(n);
    mask_t *mem = calloc(nm, sizeof(mask_t));
    assert(mem);
    return mem;
}

void mask_destroy(mask_t *ptr) { free(ptr); }

void mask_print(const ptrdiff_t n, mask_t *const mask) {
    for (ptrdiff_t i = 0; i < n; i++) {
        int val = mask_get(i, mask);
        if (val) {
            printf("%ld ", i);
        }
    }
    printf("\n");
}