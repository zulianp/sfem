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

#ifdef SFEM_ENABLE_HXTSORT
#include "HXTSort.h"
#endif

// Consider these:
// HXTSort: https://git.immc.ucl.ac.be/hextreme/HXTSort/-/tree/master?ref_type=heads

extern "C" void sort_idx(idx_t *arr, idx_t size) { std::sort(arr, arr + size); }

extern "C" idx_t sortreduce(idx_t *arr, idx_t size) {
#ifdef SFEM_ENABLE_AVX512_SORT
    avx512_qsort<idx_t>(arr, size);
#else
#ifdef SFEM_ENABLE_AVX2_SORT
    avx2::quicksort(arr, size);
#else
    std::sort(arr, arr + size);
#endif
#endif
    auto it = std::unique(arr, arr + size);
    return std::distance(arr, it);
}

template <typename T>
struct psortreduce_impl_ {
    static idx_t apply(idx_t *arr, idx_t size) { return sortreduce(arr, size); }
};

#ifdef SFEM_ENABLE_HXTSORT

template <>
struct psortreduce_impl_<uint32_t> {
    static uint32_t apply(uint32_t *arr, uint32_t size) {
        uint32_t max_val = arr[0];
#pragma omp parallel for reduction(max : max_val)
        for (idx_t i = 1; i < size; i++) {
            max_val = arr[i] > max_val ? arr[i] : max_val;
        }

        auto get_key = [](uint32_t *elem, const void *USER_DATA) -> uint32_t { return *elem; };
        // HXTSORT32_UNIFORM
        HXTSORT32(uint32_t, arr, size, max_val, get_key, NULL);
        auto it = std::unique(arr, arr + size);
        return std::distance(arr, it);
    }
};

template <>
struct psortreduce_impl_<uint64_t> {
    static uint64_t apply(uint64_t *arr, uint64_t size) {
        uint64_t max_val = arr[0];

#pragma omp parallel for reduction(max : max_val)
        for (idx_t i = 1; i < size; i++) {
            max_val = arr[i] > max_val ? arr[i] : max_val;
        }

        auto get_key = [](uint64_t *elem, const void *USER_DATA) -> uint64_t { return *elem; };
        // HXTSORT64_UNIFORM
        HXTSORT64(uint64_t, arr, size, max_val, get_key, NULL);
        auto it = std::unique(arr, arr + size);
        return std::distance(arr, it);
    }
};

#endif  // SFEM_ENABLE_HXTSORT

extern "C" idx_t psortreduce(idx_t *arr, idx_t size) { return psortreduce_impl_<idx_t>::apply(arr, size); }

extern "C" idx_t find_idx_binary_search(const idx_t key, const idx_t *arr, idx_t size) {
    auto low = std::lower_bound(arr, arr + size, key);
    assert(key == *low);
    return std::distance(arr, low);
}

extern "C" idx_t safe_find_idx_binary_search(const idx_t key, const idx_t *arr, idx_t size) {
    auto low = std::lower_bound(arr, arr + size, key);
    if (low == arr + size || key != *low) {
        return SFEM_IDX_INVALID;
    }
    return std::distance(arr, low);
}

extern "C" ptrdiff_t lower_bound(const ptrdiff_t key, const ptrdiff_t *arr, ptrdiff_t size) {
    auto low = std::lower_bound(arr, arr + size, key);
    return std::distance(arr, low);
}