#ifndef SFEM_SORT_REDUCE_H
#define SFEM_SORT_REDUCE_H

#include "sfem_base.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void sort_idx(idx_t *arr, idx_t size);
idx_t sortreduce(idx_t *arr, idx_t size);
idx_t find_idx_binary_search(const idx_t key, const idx_t *arr, idx_t size);
idx_t safe_find_idx_binary_search(const idx_t key, const idx_t *arr, idx_t size);

ptrdiff_t lower_bound(const ptrdiff_t key, const ptrdiff_t *arr, ptrdiff_t size);

#ifdef __cplusplus
}
#endif

#endif //SFEM_SORT_REDUCE_H
