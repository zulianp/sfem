#ifndef SFEM_NODE_OBSTACLE_KERNELS_HPP
#define SFEM_NODE_OBSTACLE_KERNELS_HPP

#include <math.h>

#include "sfem_base.hpp"
#include "sfem_Obstacle.hpp"

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

// out <- si .* (q ini), q null for the identity, si null for no scaling: into the node's frame,
// then scaled, as the two passes would round it. ini may be out.
SFEM_HOST_DEVICE SFEM_INLINE void node_obstacle_scaled_to_frame(const real_t *const q,
                                                                const real_t *const si,
                                                                const real_t *const ini,
                                                                real_t *const       out) {
    real_t v[3] = {ini[0], ini[1], ini[2]};
    if (q) node_obstacle_to_frame(q, v);
    for (int d = 0; d < 3; ++d) out[d] = si ? si[d] * v[d] : v[d];
}

// out <- q^T (si .* ini): scaled, then back from the node's frame. ini may be out.
SFEM_HOST_DEVICE SFEM_INLINE void node_obstacle_scaled_from_frame(const real_t *const q,
                                                                  const real_t *const si,
                                                                  const real_t *const ini,
                                                                  real_t *const       out) {
    real_t v[3];
    for (int d = 0; d < 3; ++d) v[d] = si ? si[d] * ini[d] : ini[d];
    if (q) node_obstacle_from_frame(q, v);
    for (int d = 0; d < 3; ++d) out[d] = v[d];
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

// How much of the constraints' curvature a tangent takes (node_obstacle_curvature).
enum NodeObstacleCurvature : int {
    // The whole term.
    NODE_OBSTACLE_CURVATURE_FULL = 0,
    // Scaled so that each node's most negative part stays within k / 2.
    NODE_OBSTACLE_CURVATURE_CAPPED = 1,
    // Each node's block without its negative eigenvalues: positive semidefinite, as the material's
    // tangent projected per quadrature point (Newton's tangent_projection).
    NODE_OBSTACLE_CURVATURE_PSD = 2,
};

// The symmetric 3x3 A (row major) with its negative eigenvalues set to zero: cyclic Jacobi
// rotations, A = V diag(lambda) V^T, then V diag(max(lambda, 0)) V^T.
SFEM_HOST_DEVICE SFEM_INLINE void node_obstacle_psd3(real_t *const A) {
    real_t V[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    for (int sweep = 0; sweep < 12; ++sweep) {
        const real_t off   = A[1] * A[1] + A[2] * A[2] + A[5] * A[5];
        const real_t scale = A[0] * A[0] + A[4] * A[4] + A[8] * A[8];
        if (off <= real_t(1e-30) * scale || off == 0) break;
        for (int pq = 0; pq < 3; ++pq) {
            const int p = pq == 2 ? 1 : 0, q = pq == 0 ? 1 : 2;
            const real_t apq = A[p * 3 + q];
            if (apq == 0) continue;
            // The rotation J (c at pp and qq, s at pq, -s at qp) that zeroes A'_pq in J^T A J.
            const real_t theta = (A[q * 3 + q] - A[p * 3 + p]) / (2 * apq);
            const real_t t     = (theta >= 0 ? real_t(1) : real_t(-1)) / (fabs(theta) + sqrt(theta * theta + 1));
            const real_t c     = 1 / sqrt(t * t + 1);
            const real_t s     = t * c;
            real_t       J[9]  = {1, 0, 0, 0, 1, 0, 0, 0, 1};
            J[p * 3 + p] = c;
            J[q * 3 + q] = c;
            J[p * 3 + q] = s;
            J[q * 3 + p] = -s;
            real_t AJ[9], VJ[9];
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    AJ[i * 3 + j] = A[i * 3] * J[j] + A[i * 3 + 1] * J[3 + j] + A[i * 3 + 2] * J[6 + j];
                    VJ[i * 3 + j] = V[i * 3] * J[j] + V[i * 3 + 1] * J[3 + j] + V[i * 3 + 2] * J[6 + j];
                }
            }
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    A[i * 3 + j] = J[i] * AJ[j] + J[3 + i] * AJ[3 + j] + J[6 + i] * AJ[6 + j];
                }
            }
            for (int c9 = 0; c9 < 9; ++c9) V[c9] = VJ[c9];
            A[p * 3 + q] = A[q * 3 + p] = 0;
        }
    }
    const real_t lambda[3] = {fmax(A[0], real_t(0)), fmax(A[4], real_t(0)), fmax(A[8], real_t(0))};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            A[i * 3 + j] = V[i * 3] * lambda[0] * V[j * 3] + V[i * 3 + 1] * lambda[1] * V[j * 3 + 1] +
                           V[i * 3 + 2] * lambda[2] * V[j * 3 + 2];
        }
    }
}

// B += sign mu H, the constraint's curvature C (6 packed) at a node in contact (mu > 0, gap within
// mu / k), as much of it as `mode` takes (NodeObstacleCurvature). B in the tangent's value type, each
// entry rounded once.
template <typename B_t>
SFEM_HOST_DEVICE SFEM_INLINE void node_obstacle_curvature(const real_t *const n,
                                                          const real_t *const C,
                                                          const real_t *const ri,
                                                          const real_t        gap,
                                                          const real_t        k,
                                                          const int           mode,
                                                          const real_t        sign,
                                                          B_t *const          B) {
    const real_t mu = fmax(real_t(0), -(n[0] * ri[0] + n[1] * ri[1] + n[2] * ri[2]));
    if (mu == 0 || gap > mu / k) return;

    real_t H[9] = {C[0], C[1], C[2], C[1], C[3], C[4], C[2], C[4], C[5]};
    for (int c = 0; c < 9; ++c) H[c] *= mu;

    if (mode == NODE_OBSTACLE_CURVATURE_PSD) {
        node_obstacle_psd3(H);
    } else if (mode == NODE_OBSTACLE_CURVATURE_CAPPED) {
        real_t lowest = 0;
        for (int a = 0; a < 3; ++a) {
            real_t off = 0;
            for (int b = 0; b < 3; ++b) {
                if (b != a) off += fabs(H[a * 3 + b]);
            }
            lowest = fmin(lowest, H[a * 3 + a] - off);
        }
        const real_t scale = lowest < -k / 2 ? k / (2 * -lowest) : 1;
        for (int c = 0; c < 9; ++c) H[c] *= scale;
    }

    for (int c = 0; c < 9; ++c) {
        B[c] = B_t(B[c] + sign * H[c]);
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
    if (*gap >= sfem::obstacle_unreachable) return;
    const real_t g = *gap - shift + admitted;
    if (g > cutoff) {
        *gap = sfem::obstacle_unreachable;
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

// out += B in, B a symmetric 3x3 block packed as node_obstacle_pack_sym leaves it.
SFEM_HOST_DEVICE SFEM_INLINE void node_obstacle_packed_sym_apply(const real_t *const B, const real_t *const in, real_t *const out) {
    out[0] += B[0] * in[0] + B[1] * in[1] + B[2] * in[2];
    out[1] += B[1] * in[0] + B[3] * in[1] + B[4] * in[2];
    out[2] += B[2] * in[0] + B[4] * in[1] + B[5] * in[2];
}

#endif  // SFEM_NODE_OBSTACLE_KERNELS_HPP
