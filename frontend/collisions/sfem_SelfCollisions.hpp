#ifndef SFEM_SELF_COLLISIONS_HPP
#define SFEM_SELF_COLLISIONS_HPP

#include "sfem_base.hpp"

#include "sfem_ForwardDeclarations.hpp"

#include <memory>

namespace sfem {

    struct CollisionPairs {
        smesh::SharedBuffer<smesh::idx_t> first;
        smesh::SharedBuffer<smesh::idx_t> second;
    };

    class SelfCollisions {
    public:
        SelfCollisions();
        ~SelfCollisions();

        static std::shared_ptr<SelfCollisions> create(const std::shared_ptr<smesh::Mesh>& surface);

        void find(const ptrdiff_t                          stride_displacement,  // 2 or 3 for AoS, 1 for SoA
                  std::vector<smesh::SharedBuffer<real_t>> displacement0,
                  std::vector<smesh::SharedBuffer<real_t>> displacement1);

        const CollisionPairs& vertex_to_face() const;
        const CollisionPairs& edge_to_edge() const;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
}  // namespace sfem

#endif  // SFEM_SELF_COLLISIONS_HPP
