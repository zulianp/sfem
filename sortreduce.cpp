#include "sfem_base.h"

#include <algorithm>

extern "C" idx_t sortreduce(idx_t *arr, idx_t size) {
    idx_t *end = arr + size;
    std::sort(arr, end);
    // std::stable_sort (arr, end);
    auto it = std::unique(arr, arr + size);
    return std::distance(arr, it);
}
