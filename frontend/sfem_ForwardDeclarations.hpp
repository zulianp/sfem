#ifndef SFEM_FORWARD_DECLARATIONS_HPP
#define SFEM_FORWARD_DECLARATIONS_HPP

namespace sfem {
    class Function;
    class Mesh;
    class SemiStructuredMesh;
    class FunctionSpace;
    class Op;
    class CRSGraph;
    class Sideset;

    class DirichletConditions;
    class NeumannConditions;

    template<typename T>
    class Buffer;

    template <typename pack_idx_t>
    class Packed;
}

#endif //SFEM_FORWARD_DECLARATIONS_HPP
