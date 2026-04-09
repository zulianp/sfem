#ifndef SFEM_SELF_COLLISIONS_HPP
#define SFEM_SELF_COLLISIONS_HPP

#include "sfem_base.hpp"

#include "sfem_ForwardDeclarations.hpp"
#include "sfem_Op.hpp"

#include <memory>

namespace sfem {

    struct Edges {
        smesh::SharedBuffer<smesh::idx_t> v0;
        smesh::SharedBuffer<smesh::idx_t> v1;
    };

    struct CollisionPairs {
        smesh::SharedBuffer<smesh::idx_t> first;
        smesh::SharedBuffer<smesh::idx_t> second;
    };

    class SelfCollisions {
    public:
        SelfCollisions();
        ~SelfCollisions();

        static std::shared_ptr<SelfCollisions> create(const std::shared_ptr<smesh::Mesh>& surface);

        void find(const ptrdiff_t stride_displacement,  // 3 for AoS layout, 1 for SoA layout
                  const real_t* const SFEM_RESTRICT* const SFEM_RESTRICT displacement0,
                  const real_t* const SFEM_RESTRICT* const SFEM_RESTRICT displacement1);

        real_t time_of_impact();

        void discrete_detection_with_side_effects(const real_t toi);

        void distance_and_normal(const real_t                                     toi,
                                 real_t* const SFEM_RESTRICT                      d,
                                 const ptrdiff_t                                  stride_normal,
                                 real_t* const SFEM_RESTRICT* const SFEM_RESTRICT n);

        const CollisionPairs& vertex_to_face() const;
        const CollisionPairs& edge_to_edge() const;
        const Edges&          edges() const;

        std::shared_ptr<smesh::Mesh> surface() const;
        SharedBuffer<real_t*>        points0() const;
        SharedBuffer<real_t*>        points1() const;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class SelfContactPenalty final : public Op {
    public:
        SelfContactPenalty();
        ~SelfContactPenalty();

        static std::shared_ptr<SelfContactPenalty> create(const std::shared_ptr<FunctionSpace>& space);

        static std::shared_ptr<SelfContactPenalty> create(const std::shared_ptr<FunctionSpace>& space,
                                                          const std::shared_ptr<smesh::Mesh>&   surface);

        real_t max_step_size();

        const char* name() const override;
        bool        is_linear() const override;
        int         hessian_crs(const real_t* const  x,
                                const count_t* const rowptr,
                                const idx_t* const   colidx,
                                real_t* const        values) override;
        int         gradient(const real_t* const x, real_t* const out) override;
        int         apply(const real_t* const x, const real_t* const h, real_t* const out) override;
        int         value(const real_t* x, real_t* const out) override;
        ptrdiff_t   n_dofs_domain() const override;
        ptrdiff_t   n_dofs_image() const override;

        int update(const real_t* const SFEM_RESTRICT x_prev, const real_t* const SFEM_RESTRICT x_curr) override;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
}  // namespace sfem

#endif  // SFEM_SELF_COLLISIONS_HPP
