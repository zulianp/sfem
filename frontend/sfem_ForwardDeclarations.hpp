#ifndef SFEM_FORWARD_DECLARATIONS_HPP
#define SFEM_FORWARD_DECLARATIONS_HPP

#include "sfem_base.hpp"
#include "smesh_crs_graph.hpp"
#include "smesh_forward_declarations.hpp"

namespace sfem {
    class Function;

    class FunctionSpace;
    class Op;

    // class SemiStructuredMesh;
    // class Sideset;

    class DirichletConditions;
    class NeumannConditions;

    template <typename T>
    using Buffer = smesh::Buffer<T>;

    using Communicator = smesh::Communicator;
    using Mesh         = smesh::Mesh;
    using CRSGraph     = smesh::CRSGraph<count_t, idx_t>;
    using Sideset      = smesh::Sideset;

    template <typename pack_idx_t>
    using PackedMesh = smesh::PackedMesh<pack_idx_t>;

    template <typename T>
    class Operator;

    template <typename T>
    class MatrixFreeLinearSolver;

    // template <typename pack_idx_t>
    // class Packed;
}  // namespace sfem

#endif  // SFEM_FORWARD_DECLARATIONS_HPP
