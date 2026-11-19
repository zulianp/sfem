#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef SFEM_RESTRICT
#define SFEM_RESTRICT __restrict__
#endif
#ifndef RSTR
#define RSTR SFEM_RESTRICT
#endif

namespace sfem {
namespace codegen {

// body(element, owned): owned has bit a set where node a of the element is a row of the calling thread.
template <int NS, typename C, typename I, typename E, typename Body>
inline void owner_computes(const ptrdiff_t nnodes, I **const RSTR elements, const C *const RSTR rowptr,
                           const C *const RSTR n2e_ptr, const E *const RSTR n2e_idx, Body &&body) {
  static_assert(NS <= 64, "an element's nodes as the bits of one word");
#pragma omp parallel
  {
    int nt = 1, t = 0;
#ifdef _OPENMP
    nt = omp_get_num_threads();
    t = omp_get_thread_num();
#endif
    const double per = (double)(rowptr[nnodes] - rowptr[0]) / nt;
    auto cut = [&](const int k) -> ptrdiff_t {
      if (k <= 0) return 0;
      if (k >= nt) return nnodes;
      return std::lower_bound(rowptr, rowptr + nnodes, (C)(rowptr[0] + per * k)) - rowptr;
    };
    const ptrdiff_t a0 = cut(t), a1 = cut(t + 1);
    for (ptrdiff_t i = a0; i < a1; ++i) {
      if (rowptr[i] == rowptr[i + 1]) continue;
      for (C p = n2e_ptr[i]; p < n2e_ptr[i + 1]; ++p) {
        const ptrdiff_t e = (ptrdiff_t)n2e_idx[p];
        uint64_t owned = 0;
        bool first = true;
        for (int a = 0; a < NS; ++a) {
          const ptrdiff_t n = (ptrdiff_t)elements[a][e];
          if (n < a0 || n >= a1 || rowptr[n] == rowptr[n + 1]) continue;
          owned |= uint64_t(1) << a;
          first = first && n >= i;
        }
        if (first) body(e, owned);
      }
    }
  }
}

}  // namespace codegen
}  // namespace sfem
