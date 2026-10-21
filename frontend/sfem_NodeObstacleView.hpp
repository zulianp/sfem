#ifndef SFEM_NODE_OBSTACLE_VIEW_HPP
#define SFEM_NODE_OBSTACLE_VIEW_HPP

#include "sfem_aliases.hpp"
#include "sfem_base.hpp"
#include "smesh_types.hpp"

namespace sfem {

    // The contact nodes of a nodal obstacle, node by node (sfem_node_obstacle_kernels.hpp), with
    // every pointer in the execution space `es`: normals[3] and curvature[6] are SoA rows, frames 9
    // per node. A member an operation does not read may be null. The loops below run where `es`
    // says, on the host or the device.
    struct NodeObstacleView {
        ExecutionSpace es{EXECUTION_SPACE_HOST};
        ptrdiff_t      size{0};
        const idx_t   *idx{nullptr};
        const real_t  *normals[3]{nullptr, nullptr, nullptr};
        const real_t  *curvature[6]{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
        const real_t  *gap{nullptr};
        real_t        *frames{nullptr};
        real_t        *mu{nullptr};
    };

    void node_obstacle_frames(const NodeObstacleView &c, const int dir);
    void node_obstacle_to_frame(const NodeObstacleView &c, real_t *const v);
    void node_obstacle_from_frame(const NodeObstacleView &c, real_t *const v);
    void node_obstacle_upper_bound(const NodeObstacleView &c, const ptrdiff_t ndofs, const real_t unbounded, real_t *const ub);

    // projected (a copy of the residual) less the reaction, mu kept; complementarity into c and
    // -gap into penetration, per contact node.
    void node_obstacle_kkt(const NodeObstacleView &c,
                           const real_t *const     k,
                           real_t *const           projected,
                           real_t *const           complementarity,
                           real_t *const           penetration);

    // The curvature term (node_obstacle_curvature) added to each contact node's diagonal block of
    // a BSR matrix, its values of `values_type` (sfem_TangentValues.hpp).
    void node_obstacle_curvature_bsr(const NodeObstacleView &c,
                                     const real_t *const     r,
                                     const real_t *const     k,
                                     const bool              capped,
                                     const real_t            sign,
                                     const count_t *const    rowptr,
                                     const idx_t *const      colidx,
                                     const smesh::PrimitiveType values_type,
                                     void *const             values);

    // The curvature term per contact node, a packed symmetric block [xx, xy, xz, yy, yz, zz]
    // (6 per node, overwritten; zero where the term is).
    void node_obstacle_curvature_blocks(const NodeObstacleView &c,
                                        const real_t *const     r,
                                        const real_t *const     k,
                                        const bool              capped,
                                        real_t *const           blocks);

    // out (zeroed by the caller) at the contact nodes: -mu n.
    void node_obstacle_reaction(const NodeObstacleView &c, real_t *const out);
    // out (zeroed by the caller) at the contact nodes: gap n.
    void node_obstacle_gap_field(const NodeObstacleView &c, real_t *const out);
    // v += gap n / dt where gap < 0, with the gap and normals of `c` the samples ahead.
    void node_obstacle_project_velocity(const NodeObstacleView &c, real_t *const v, const real_t dt);

    // The sampled gap and normals of `size` contact nodes (in es) to what the constraints see
    // (node_obstacle_offset_gap): gap - shift + release * p0_i (p0 may be null: zero), and past
    // `cutoff` no obstacle.
    void node_obstacle_offset_gap(const ExecutionSpace es,
                                  const ptrdiff_t      size,
                                  const real_t         shift,
                                  const real_t         cutoff,
                                  const real_t         release,
                                  const real_t *const  p0,
                                  real_t *const        gap,
                                  real_t *const *const normals);

    // max(0, max_i x_i) over n values in es.
    real_t max_or_zero(const ExecutionSpace es, const ptrdiff_t n, const real_t *const x);

}  // namespace sfem

#endif  // SFEM_NODE_OBSTACLE_VIEW_HPP
