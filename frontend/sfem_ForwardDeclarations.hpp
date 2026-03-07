#ifndef SFEM_FORWARD_DECLARATIONS_HPP
#define SFEM_FORWARD_DECLARATIONS_HPP

#include "sfem_base.hpp"

namespace smesh {
    class Communicator;
    class Mesh;
    template <typename T>
    class Buffer;
    template <typename count_t, typename idx_t>
    class CRSGraph;
}

namespace sfem {
    class Function;
    class SemiStructuredMesh;
    class FunctionSpace;
    class Op;
    class Sideset;

    class DirichletConditions;
    class NeumannConditions;

    template<typename T>
    using Buffer = smesh::Buffer<T>;

    using Communicator = smesh::Communicator;
    using Mesh = smesh::Mesh;
    using CRSGraph = smesh::CRSGraph<count_t, idx_t>;

    template <typename pack_idx_t>
    class Packed;
}

#endif //SFEM_FORWARD_DECLARATIONS_HPP
