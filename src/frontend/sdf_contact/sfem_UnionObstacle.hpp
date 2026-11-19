#ifndef SFEM_UNION_OBSTACLE_HPP
#define SFEM_UNION_OBSTACLE_HPP

#include <memory>
#include <vector>

#include "sfem_Obstacle.hpp"
#include "sfem_aliases.hpp"
#include "sfem_execution_context.hpp"

namespace sfem {

    // The union of several obstacles (each an SDFObstacle on its own grid, e.g. a fine grid around
    // small features and a coarse one around large ones), sampled at nodes: the union's SDF is the
    // largest of the parts', so each node is constrained by the obstacle with the smallest gap
    // there, with that obstacle's normal and curvature. Where a node is outside every grid, all
    // parts report obstacle_unreachable and so does the union.
    //
    // Node sampling only (NodeObstacleContact, the node_sdf back-end): the surface-sampling methods
    // fail. The points, idx, disp and outputs are in the context's memory space, as the parts'.
    class UnionObstacle final : public Obstacle {
    public:
        UnionObstacle(const std::vector<std::shared_ptr<Obstacle>> &parts, const ExecutionContext &context);
        ~UnionObstacle() override;

        int sample_nodes(const ptrdiff_t                   nnodes,
                         const idx_t *const SFEM_RESTRICT  idx,
                         geom_t **const SFEM_RESTRICT      points,
                         const real_t *const SFEM_RESTRICT disp,
                         real_t *const SFEM_RESTRICT       gap,
                         real_t **const SFEM_RESTRICT      normals) override;

        // The curvature of the part each node is constrained by: the selection is made again at
        // these points.
        int sample_nodes_curvature(const ptrdiff_t                   nnodes,
                                   const idx_t *const SFEM_RESTRICT  idx,
                                   geom_t **const SFEM_RESTRICT      points,
                                   const real_t *const SFEM_RESTRICT disp,
                                   real_t **const SFEM_RESTRICT      curvature) override;

        int sample(smesh::ElemType, const ptrdiff_t, const ptrdiff_t, idx_t **const, geom_t **const, real_t **const, real_t *const) override;
        int sample_normals(smesh::ElemType, const ptrdiff_t, const ptrdiff_t, idx_t **const, geom_t **const, real_t **const) override;
        int sample_value(smesh::ElemType, const ptrdiff_t, const ptrdiff_t, idx_t **const, geom_t **const, real_t *const) override;
        int sample_normals_accumulate(smesh::ElemType, const ptrdiff_t, const ptrdiff_t, idx_t **const, geom_t **const, real_t **const) override;
        int sample_value_accumulate(smesh::ElemType, const ptrdiff_t, const ptrdiff_t, idx_t **const, geom_t **const, real_t *const) override;
        int sample_weights_accumulate(smesh::ElemType, const ptrdiff_t, const ptrdiff_t, idx_t **const, geom_t **const, real_t *const) override;

    private:
        std::vector<std::shared_ptr<Obstacle>> parts_;
        ExecutionContext                       context_;
        ptrdiff_t                              n_{-1};
        SharedBuffer<real_t>                   gap_;          // the kept gap, when the caller's is null
        SharedBuffer<real_t>                   normals_;      // 3 rows, the same
        SharedBuffer<real_t>                   part_gap_;     // part k's samples
        SharedBuffer<real_t>                   part_normals_; // 3 rows
        SharedBuffer<real_t>                   part_curvature_;  // 6 rows
        SharedBuffer<idx_t>                    which_;        // the part each node is constrained by

        void ensure(const ptrdiff_t n);
    };

}  // namespace sfem

#endif  // SFEM_UNION_OBSTACLE_HPP
