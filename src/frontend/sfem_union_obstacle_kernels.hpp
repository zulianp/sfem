#ifndef SFEM_UNION_OBSTACLE_KERNELS_HPP
#define SFEM_UNION_OBSTACLE_KERNELS_HPP

#include "sfem_base.hpp"

// One node of a UnionObstacle (sdf_contact/sfem_UnionObstacle.hpp), for its host loops and its
// device kernels (cuda/sfem_UnionObstacle_cuda.cu) alike.

// Obstacle k's sample at the node replaces the one kept so far where its gap is smaller: the union
// of the obstacles has the largest of their SDFs, so the binding constraint is the nearest
// obstacle's. Ties keep the earlier obstacle.
SFEM_HOST_DEVICE SFEM_INLINE void union_obstacle_select(const int     k,
                                                        const real_t  gk,
                                                        const real_t  nk0,
                                                        const real_t  nk1,
                                                        const real_t  nk2,
                                                        real_t *const g,
                                                        real_t *const n0,
                                                        real_t *const n1,
                                                        real_t *const n2,
                                                        idx_t *const  which) {
    if (gk < *g) {
        *g     = gk;
        *n0    = nk0;
        *n1    = nk1;
        *n2    = nk2;
        *which = (idx_t)k;
    }
}

#endif  // SFEM_UNION_OBSTACLE_KERNELS_HPP
