
#include <algorithm>

#include "sfem_base.h"

#include <cstddef>

extern "C" void argsort_f(const ptrdiff_t n, const geom_t *key, idx_t *idx) {
    for (ptrdiff_t i = 0; i < n; ++i) {
        idx[i] = i;
    }

    std::sort(idx, idx + n, [key](const idx_t l, const idx_t r) { return key[l] < key[r]; });
}
