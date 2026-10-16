#ifndef SFEM_NODE_OBSTACLE_CONTACT_CUDA_HPP
#define SFEM_NODE_OBSTACLE_CONTACT_CUDA_HPP

#include "sfem_base.hpp"

namespace sfem {

    // The device side of NodeObstacleContact (sfem_node_obstacle_kernels.hpp, node by node): every
    // pointer is device memory; normals[3] and curvature[6] are SoA rows, frames 9 per node.
    struct NodeObstacleDevice {
        ptrdiff_t      size;
        const idx_t   *idx;
        const real_t  *normals[3];
        const real_t  *curvature[6];
        const real_t  *gap;
        real_t        *frames;
        real_t        *mu;
    };

    void d_node_obstacle_frames(const NodeObstacleDevice &c, const int dir);
    void d_node_obstacle_to_frame(const NodeObstacleDevice &c, real_t *const v);
    void d_node_obstacle_from_frame(const NodeObstacleDevice &c, real_t *const v);
    void d_node_obstacle_upper_bound(const NodeObstacleDevice &c, const ptrdiff_t ndofs, const real_t unbounded, real_t *const ub);

    // projected (a copy of the residual) less the reaction, mu kept; complementarity into c and
    // -gap into penetration, per contact node.
    void d_node_obstacle_kkt(const NodeObstacleDevice &c,
                             const real_t *const       k,
                             real_t *const             projected,
                             real_t *const             complementarity,
                             real_t *const             penetration);

    void d_node_obstacle_curvature(const NodeObstacleDevice &c,
                                   const real_t *const       r,
                                   const real_t *const       k,
                                   const bool                capped,
                                   const real_t              sign,
                                   const count_t *const      rowptr,
                                   const idx_t *const        colidx,
                                   real_t *const             values);

    // out (zeroed by the caller) at the contact nodes: -mu n.
    void d_node_obstacle_reaction(const NodeObstacleDevice &c, real_t *const out);
    // out (zeroed by the caller) at the contact nodes: gap n.
    void d_node_obstacle_gap_field(const NodeObstacleDevice &c, real_t *const out);
    // v += gap n / dt where gap < 0, with the gap and normals of `c` the samples ahead.
    void d_node_obstacle_project_velocity(const NodeObstacleDevice &c, real_t *const v, const real_t dt);

    // max(0, max_i x_i) over n device values.
    real_t d_max_or_zero(const ptrdiff_t n, const real_t *const x);

}  // namespace sfem

#endif  // SFEM_NODE_OBSTACLE_CONTACT_CUDA_HPP
