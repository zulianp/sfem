#include "sfem_base.h"

#include <algorithm>
#include <cassert>

#ifdef SFEM_ENABLE_AVX2_SORT
#include "avx2sort.h"
#endif

extern "C" idx_t sortreduce(idx_t *arr, idx_t size) {

#ifdef SFEM_ENABLE_AVX2_SORT
    avx2::quicksort(arr, size);
#else
    std::sort(arr,  arr + size);
#endif
    auto it = std::unique(arr, arr + size);
    return std::distance(arr, it);
}

extern "C" idx_t find_idx_binary_search(const idx_t key, const idx_t *arr, idx_t size) {
    auto low = std::lower_bound(arr, arr+size, key); 
    assert(key == *low);
    return std::distance(arr, low);
}