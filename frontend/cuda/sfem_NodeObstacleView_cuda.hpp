#ifndef SFEM_NODE_OBSTACLE_VIEW_CUDA_HPP
#define SFEM_NODE_OBSTACLE_VIEW_CUDA_HPP

#include "sfem_NodeObstacleView.hpp"

namespace sfem {

    // The device side of the NodeObstacleView loops (sfem_NodeObstacleView.hpp): every pointer in
    // the view and every array is device memory.
    void d_node_obstacle_frames(const NodeObstacleView &c, const int dir);
    void d_node_obstacle_to_frame(const NodeObstacleView &c, const real_t *const s, const real_t *const in, real_t *const out);
    void d_node_obstacle_from_frame(const NodeObstacleView &c, const real_t *const s, const real_t *const in, real_t *const out);
    void d_node_obstacle_upper_bound(const NodeObstacleView &c, const ptrdiff_t ndofs, const real_t unbounded, real_t *const ub);

    void d_node_obstacle_offset_gap(const ptrdiff_t      size,
                                    const real_t         shift,
                                    const real_t         cutoff,
                                    const real_t         release,
                                    const real_t *const  p0,
                                    real_t *const        gap,
                                    real_t *const *const normals);

    // projected (a copy of the residual) less the reaction, mu kept; complementarity into c and
    // -gap into penetration, per contact node.
    void d_node_obstacle_kkt(const NodeObstacleView &c,
                             const real_t *const       k,
                             real_t *const             projected,
                             real_t *const             complementarity,
                             real_t *const             penetration);

    void d_node_obstacle_curvature_bsr(const NodeObstacleView &c,
                                   const real_t *const       r,
                                   const real_t *const       k,
                                   const bool                capped,
                                   const real_t              sign,
                                   const count_t *const      rowptr,
                                   const idx_t *const        colidx,
                                   const smesh::PrimitiveType values_type,
                                   void *const               values);

    void d_node_obstacle_curvature_blocks(const NodeObstacleView &c,
                                          const real_t *const     r,
                                          const real_t *const     k,
                                          const bool              capped,
                                          real_t *const           blocks);

    // out (zeroed by the caller) at the contact nodes: -mu n.
    void d_node_obstacle_reaction(const NodeObstacleView &c, real_t *const out);
    // out (zeroed by the caller) at the contact nodes: gap n.
    void d_node_obstacle_gap_field(const NodeObstacleView &c, real_t *const out);
    // v += gap n / dt where gap < 0, with the gap and normals of `c` the samples ahead.
    void d_node_obstacle_project_velocity(const NodeObstacleView &c, real_t *const v, const real_t dt);

    // max(0, max_i x_i) over n device values.
    real_t d_max_or_zero(const ptrdiff_t n, const real_t *const x);

}  // namespace sfem

#endif  // SFEM_NODE_OBSTACLE_VIEW_CUDA_HPP
