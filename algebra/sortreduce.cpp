#include "sfem_base.h"

#include <algorithm>
#include <cassert>
#include <cstddef>

#ifdef SFEM_ENABLE_AVX512_SORT
#include "avx512-16bit-qsort.hpp"
#include "avx512-32bit-qsort.hpp"
#include "avx512-64bit-qsort.hpp"
#else
#ifdef SFEM_ENABLE_AVX2_SORT
#include "avx2sort.h"
#endif
#endif

extern "C" void sort_idx(idx_t *arr, idx_t size) {
    std::sort(arr,  arr + size);
}

extern "C" idx_t sortreduce(idx_t *arr, idx_t size) {

#ifdef SFEM_ENABLE_AVX512_SORT
    avx512_qsort<idx_t>(arr, size);
#else
#ifdef SFEM_ENABLE_AVX2_SORT
    avx2::quicksort(arr, size);
#else
    std::sort(arr,  arr + size);
#endif
    #endif
    auto it = std::unique(arr, arr + size);
    return std::distance(arr, it);
}

extern "C" idx_t find_idx_binary_search(const idx_t key, const idx_t *arr, idx_t size) {
    auto low = std::lower_bound(arr, arr+size, key); 
    assert(key == *low);
    return std::distance(arr, low);
}

extern "C" ptrdiff_t lower_bound(const ptrdiff_t key, const ptrdiff_t *arr, ptrdiff_t size)
{
    auto low = std::lower_bound(arr, arr+size, key); 
    return std::distance(arr, low);
}