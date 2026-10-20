#ifndef SFEM_NODE_OBSTACLE_KERNELS_HPP
#define SFEM_NODE_OBSTACLE_KERNELS_HPP

#include <math.h>

#include "sfem_base.hpp"
#include "sfem_sdf_sampling.hpp"

// One contact node of NodeObstacleContact, for its host loops and its device kernels
// (cuda/sfem_NodeObstacleView_cuda.cu) alike. A node's frame q is [n, t1, t2] by rows.

// t1 is n crossed with the axis least aligned with it, t2 = n x t1. A node outside the grid has
// no normal and an unreachable gap; its frame is that of the contact direction.
SFEM_HOST_DEVICE SFEM_INLINE void node_obstacle_frame(const real_t n0, const real_t n1, const real_t n2, const int dir, real_t *const q) {
    real_t n[3] = {n0, n1, n2};
    if (n[0] == 0 && n[1] == 0 && n[2] == 0) n[dir] = 1;
    int axis = 0;
    for (int d = 1; d < 3; ++d) {
        if (fabs(n[d]) < fabs(n[axis])) axis = d;
    }
    real_t e[3]        = {0, 0, 0};
    e[axis]            = 1;
    real_t       t1[3] = {n[1] * e[2] - n[2] * e[1], n[2] * e[0] - n[0] * e[2], n[0] * e[1] - n[1] * e[0]};
    const real_t len   = sqrt(t1[0] * t1[0] + t1[1] * t1[1] + t1[2] * t1[2]);
    for (int d = 0; d < 3; ++d) t1[d] /= len;
    const real_t t2[3] = {n[1] * t1[2] - n[2] * t1[1], n[2] * t1[0] - n[0] * t1[2], n[0] * t1[1] - n[1] * t1[0]};
    for (int d = 0; d < 3; ++d) {
        q[d]     = n[d];
        q[3 + d] = t1[d];
        q[6 + d] = t2[d];
    }
}

// vi <- q vi: into the node's frame.
SFEM_HOST_DEVICE SFEM_INLINE void node_obstacle_to_frame(const real_t *const q, real_t *const vi) {
    const real_t w[3] = {vi[0], vi[1], vi[2]};
    for (int c = 0; c < 3; ++c) {
        vi[c] = q[3 * c] * w[0] + q[3 * c + 1] * w[1] + q[3 * c + 2] * w[2];
    }
}

// vi <- q^T vi: back from the node's frame.
SFEM_HOST_DEVICE SFEM_INLINE void node_obstacle_from_frame(const real_t *const q, real_t *const vi) {
    const real_t w[3] = {vi[0], vi[1], vi[2]};
    for (int d = 0; d < 3; ++d) {
        vi[d] = q[d] * w[0] + q[3 + d] * w[1] + q[6 + d] * w[2];
    }
}

// The admissible reaction mu = max(0, -n . ri), removed from ri; *c is the complementarity
// min(mu, k max(gap, 0)).
SFEM_HOST_DEVICE SFEM_INLINE real_t node_obstacle_kkt(const real_t *const n,
                                                      real_t *const       ri,
                                                      const real_t        gap,
                                                      const real_t        k,
                                                      real_t *const       c) {
    const real_t mu = fmax(real_t(0), -(n[0] * ri[0] + n[1] * ri[1] + n[2] * ri[2]));
    for (int d = 0; d < 3; ++d) {
        ri[d] += mu * n[d];
    }
    *c = fmin(mu, k * fmax(gap, real_t(0)));
    return mu;
}

// B += sign scale mu H, the constraint's curvature C (6 packed) at a node in contact (mu > 0, gap
// within mu / k); with `capped` scaled so that its most negative part stays within k / 2.
SFEM_HOST_DEVICE SFEM_INLINE void node_obstacle_curvature(const real_t *const n,
                                                          const real_t *const C,
                                                          const real_t *const ri,
                                                          const real_t        gap,
                                                          const real_t        k,
                                                          const bool          capped,
                                                          const real_t        sign,
                                                          real_t *const       B) {
    const real_t mu = fmax(real_t(0), -(n[0] * ri[0] + n[1] * ri[1] + n[2] * ri[2]));
    if (mu == 0 || gap > mu / k) return;

    const real_t H[9] = {C[0], C[1], C[2], C[1], C[3], C[4], C[2], C[4], C[5]};

    real_t lowest = 0;
    for (int a = 0; a < 3; ++a) {
        real_t off = 0;
        for (int b = 0; b < 3; ++b) {
            if (b != a) off += fabs(H[a * 3 + b]);
        }
        lowest = fmin(lowest, mu * (H[a * 3 + a] - off));
    }
    const real_t scale = capped && lowest < -k / 2 ? k / (2 * -lowest) : 1;

    for (int c = 0; c < 9; ++c) {
        B[c] += sign * scale * mu * H[c];
    }
}

// The gap the constraints see, from the sampled one: less `shift`, plus `admitted` (the share of
// the starting penetration still allowed). Past `cutoff`, as outside the grid, there is no
// obstacle: the gap is unreachable and the normal zero.
SFEM_HOST_DEVICE SFEM_INLINE void node_obstacle_offset_gap(const real_t  shift,
                                                           const real_t  cutoff,
                                                           const real_t  admitted,
                                                           real_t *const gap,
                                                           real_t *const n0,
                                                           real_t *const n1,
                                                           real_t *const n2) {
    if (*gap >= SFEM_SDF_UNREACHABLE) return;
    const real_t g = *gap - shift + admitted;
    if (g > cutoff) {
        *gap = SFEM_SDF_UNREACHABLE;
        *n0  = 0;
        *n1  = 0;
        *n2  = 0;
        return;
    }
    *gap = g;
}

// The upper triangle of a symmetric 3x3 block B, packed [xx, xy, xz, yy, yz, zz].
SFEM_HOST_DEVICE SFEM_INLINE void node_obstacle_pack_sym(const real_t *const B, real_t *const packed) {
    packed[0] = B[0];
    packed[1] = B[1];
    packed[2] = B[2];
    packed[3] = B[4];
    packed[4] = B[5];
    packed[5] = B[8];
}

#endif  // SFEM_NODE_OBSTACLE_KERNELS_HPP
