#ifndef SFEM_SORT_REDUCE_H
#define SFEM_SORT_REDUCE_H

#include "sfem_base.h"

void sort_idx(idx_t *arr, idx_t size);
idx_t sortreduce(idx_t *arr, idx_t size);
idx_t find_idx_binary_search(const idx_t key, const idx_t *arr, idx_t size);

#endif //SFEM_SORT_REDUCE_H
