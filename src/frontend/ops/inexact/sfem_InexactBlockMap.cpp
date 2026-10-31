#include "sfem_InexactBlockMap.hpp"

#include <algorithm>

namespace sfem {
    namespace inexact {

        BlockLocalMap block_local_map(const ptrdiff_t                         ne,
                                      const int                               N,
                                      const std::vector<std::vector<idx_t>> &conn,
                                      const int                               block) {
            BlockLocalMap   map;
            const ptrdiff_t nblocks = (ne + block - 1) / block;
            std::vector<int32_t> count((size_t)nblocks, 0);
            std::vector<idx_t>   scratch((size_t)nblocks * block * N);
            map.local.resize((size_t)N * ne);
#pragma omp parallel for schedule(static)
            for (ptrdiff_t b = 0; b < nblocks; ++b) {
                idx_t *const    v  = &scratch[(size_t)b * block * N];
                const ptrdiff_t e0 = b * block, e1 = std::min(ne, e0 + block);
                int             n  = 0;
                for (int s = 0; s < N; ++s)
                    for (ptrdiff_t e = e0; e < e1; ++e) v[n++] = conn[s][e];
                std::sort(v, v + n);
                n = (int)(std::unique(v, v + n) - v);
                for (int s = 0; s < N; ++s)
                    for (ptrdiff_t e = e0; e < e1; ++e)
                        map.local[(size_t)s * ne + e] = (uint16_t)(std::lower_bound(v, v + n, conn[s][e]) - v);
                count[(size_t)b] = n;
            }
            map.ptr.assign((size_t)nblocks + 1, 0);
            for (ptrdiff_t b = 0; b < nblocks; ++b) {
                map.ptr[(size_t)b + 1] = map.ptr[(size_t)b] + count[(size_t)b];
                map.max_block_nodes    = std::max(map.max_block_nodes, (int)count[(size_t)b]);
            }
            map.nodes.resize((size_t)map.ptr[(size_t)nblocks]);
#pragma omp parallel for schedule(static)
            for (ptrdiff_t b = 0; b < nblocks; ++b)
                std::copy(&scratch[(size_t)b * block * N], &scratch[(size_t)b * block * N] + count[(size_t)b],
                          map.nodes.begin() + map.ptr[(size_t)b]);
            return map;
        }

    }  // namespace inexact
}  // namespace sfem
