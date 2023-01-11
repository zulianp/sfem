#include "sfem_base.h"

#include <algorithm>
#include <cassert>

extern "C" idx_t sortreduce(idx_t *arr, idx_t size) {
    idx_t *end = arr + size;
    std::sort(arr, end);
    auto it = std::unique(arr, arr + size);
    return std::distance(arr, it);
}

extern "C" idx_t find_idx_binary_search(const idx_t key, const idx_t *arr, idx_t size) {
    auto low = std::lower_bound (arr, arr+size, key); 
    assert(key == *low);
    return std::distance(arr, low);
}