#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "sfem_aliases.hpp"

namespace sfem {
    namespace inexact {

        /// The block-local scatter's map of the device stores (cuda/sfem_cuda_InexactStore.cuh):
        /// for each block of `block` consecutive elements its distinct nodes, sorted, and each
        /// element's nodes as indices into its block's list.  Built on the host, in parallel; a
        /// .cu source's host code is compiled without OpenMP.
        struct BlockLocalMap {
            std::vector<int32_t>  ptr;    // [block + 1]
            std::vector<idx_t>    nodes;  // [ptr[b], ptr[b + 1])
            std::vector<uint16_t> local;  // [staged node][element]
            int                   max_block_nodes{0};
        };

        /// `connectivity[s]` is staged node s of every element (host arrays, `n_nodes` of them).
        BlockLocalMap block_local_map(const ptrdiff_t                    nelements,
                                      const int                          n_nodes,
                                      const std::vector<std::vector<idx_t>> &connectivity,
                                      const int                          block);

    }  // namespace inexact
}  // namespace sfem
