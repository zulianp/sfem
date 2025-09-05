#ifndef SFEM_DUAL_GRAPH_HPP
#define SFEM_DUAL_GRAPH_HPP

#include "sfem_base.h"

#include "sfem_Buffer.hpp"
#include "sfem_ForwardDeclarations.hpp"

namespace sfem {
    class DualGraph {
    public:
        DualGraph();
        ~DualGraph();

        SharedBuffer<count_t> adj_ptr();
        SharedBuffer<element_idx_t> adj_idx();

        static std::shared_ptr<DualGraph> create(const std::shared_ptr<Mesh> &mesh);
    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
}

#endif