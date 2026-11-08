#ifndef SFEM_UNION_OBSTACLE_CUDA_HPP
#define SFEM_UNION_OBSTACLE_CUDA_HPP

#include <stddef.h>

#include "sfem_base.hpp"

namespace sfem {

    // UnionObstacle's node loops on the device (sdf_contact/sfem_UnionObstacle.cpp); every array is
    // in device memory, the row arrays themselves on the host.

    // which[i] <- 0.
    void d_union_obstacle_reset(const ptrdiff_t n, idx_t *const which, void *const stream);

    // union_obstacle_select at every node: obstacle k's gap gk and normals nk (3 rows) against the
    // kept gap g and normals n.
    void d_union_obstacle_select(const ptrdiff_t      n,
                                 const int            k,
                                 const real_t *const  gk,
                                 real_t *const *const nk,
                                 real_t *const        g,
                                 real_t *const *const normals,
                                 idx_t *const         which,
                                 void *const          stream);

    // curvature <- obstacle k's ck (6 rows) at the nodes where which[i] == k.
    void d_union_obstacle_select_curvature(const ptrdiff_t      n,
                                           const int            k,
                                           const idx_t *const   which,
                                           real_t *const *const ck,
                                           real_t *const *const curvature,
                                           void *const          stream);

}  // namespace sfem

#endif  // SFEM_UNION_OBSTACLE_CUDA_HPP
