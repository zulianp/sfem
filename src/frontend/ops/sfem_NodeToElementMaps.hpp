#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "sfem_MultiDomainOp.hpp"
#include "smesh_alloc.hpp"
#include "smesh_graph.hpp"

namespace sfem {

    /// Each domain's node-to-element map, for the host's owner-computes BSR assembly of the generated
    /// operators (owner_computes.hpp): built from the connectivity the kernels read at the domain's
    /// first BSR assembly and kept -- a count per node and an element per element node (4 bytes each),
    /// each node's elements in ascending order.
    class NodeToElementMaps {
    public:
        struct Map {
            count_t       *ptr{nullptr};
            element_idx_t *idx{nullptr};
            ~Map() {
                SMESH_FREE(ptr);
                SMESH_FREE(idx);
            }
        };

        const Map &of(const OpDomain &domain, idx_t **const elements, const ptrdiff_t nnodes) {
            for (const auto &m : maps_)
                if (m.first == domain.block.get()) return *m.second;
            auto m = std::make_unique<Map>();
            smesh::create_n2e<idx_t, count_t, element_idx_t>(
                    domain.block->n_elements(), nnodes, domain.block->n_nodes_per_element(), elements, &m->ptr, &m->idx);
            smesh::sort_n2e(nnodes, m->ptr, m->idx);
            maps_.emplace_back(domain.block.get(), std::move(m));
            return *maps_.back().second;
        }

    private:
        std::vector<std::pair<const void *, std::unique_ptr<Map>>> maps_;
    };

}  // namespace sfem
