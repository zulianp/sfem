#ifndef NODE_INTERPOLATE_KERNEL_HPP
#define NODE_INTERPOLATE_KERNEL_HPP

#include <math.h>

#include "sfem_sdf_sampling.hpp"

// One node of the nodal sampling, for the host loop (node_interpolate.cpp) and the device kernel
// (cuda/node_interpolate_cuda.cu) alike: everything it reads or writes is passed by value.

// The SDF's B-spline as sdf_sample reads it: n nodes per axis from `origin`, spaced `delta`, and
// the coefficients `data` with strides `stride` (the coefficient of the grid's node (0, 0, 0)).
struct NodeSamplerGrid {
    ptrdiff_t     n[3];
    ptrdiff_t     stride[3];
    geom_t        origin[3];
    geom_t        delta[3];
    const geom_t *data;
};

// The node coordinates, by axis, and where the samples go; a null output is not written.
struct NodeSamplerPoints {
    const geom_t *xyz[3];
};

struct NodeSamplerOutput {
    real_t *gap;
    real_t *normals[3];
    real_t *curvature[6];
};

// The ORDER-th derivatives of the B-spline s at the points X[node] + disp[3 node + d], for
// node = idx[i] (i when idx is null; disp may be null), summed in real_t. Writes entry i of
// whichever outputs are given:
//   gap        -s, with ORDER >= 1 -s / |grad s|
//   normals    grad s / |grad s|                              (ORDER >= 1, 3 rows)
//   curvature  (H - n m^T - m n^T) / |grad s|, m = H n        (ORDER >= 2, 6 rows)
// Outside the grid the gap is SFEM_SDF_UNREACHABLE and the normal and curvature are zero; so
// are they where the gradient vanishes.
template <int ORDER>
SFEM_HOST_DEVICE SFEM_INLINE void sample_node(const ptrdiff_t                   i,
                                              const idx_t *const SFEM_RESTRICT  idx,
                                              const NodeSamplerPoints          &points,
                                              const real_t *const SFEM_RESTRICT disp,
                                              const NodeSamplerGrid            &grid,
                                              const NodeSamplerOutput          &out) {
    const ptrdiff_t node = idx ? (ptrdiff_t)idx[i] : i;
    real_t          p[3];
    for (int d = 0; d < 3; ++d) {
        p[d] = (real_t)points.xyz[d][node] + (disp ? disp[node * 3 + d] : real_t(0));
    }

    real_t s, g[3] = {0, 0, 0}, H[6] = {0, 0, 0, 0, 0, 0};
    if (!sdf_sample<ORDER>(grid.n, grid.stride, grid.origin, grid.delta, grid.data, p[0], p[1], p[2], &s, g, H)) {
        if (out.gap) out.gap[i] = SFEM_SDF_UNREACHABLE;
        if (out.normals[0]) {
            for (int d = 0; d < 3; ++d) out.normals[d][i] = 0;
        }
        if (out.curvature[0]) {
            for (int c = 0; c < 6; ++c) out.curvature[c][i] = 0;
        }
        return;
    }

    const real_t q = sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
    if (ORDER == 0 || q == 0) {
        if (out.gap) out.gap[i] = -s;
        if (out.normals[0]) {
            for (int d = 0; d < 3; ++d) out.normals[d][i] = 0;
        }
        if (out.curvature[0]) {
            for (int c = 0; c < 6; ++c) out.curvature[c][i] = 0;
        }
        return;
    }

    const real_t nn[3] = {g[0] / q, g[1] / q, g[2] / q};
    if (out.gap) out.gap[i] = -s / q;
    if (out.normals[0]) {
        for (int d = 0; d < 3; ++d) out.normals[d][i] = nn[d];
    }

    if (ORDER >= 2 && out.curvature[0]) {
        // m = H n = grad |grad s|; the Hessian of the gap -s / |grad s| at s = 0 is
        // -(H - n m^T - m n^T) / |grad s|.
        const real_t m[3] = {H[0] * nn[0] + H[1] * nn[1] + H[2] * nn[2],
                             H[1] * nn[0] + H[3] * nn[1] + H[4] * nn[2],
                             H[2] * nn[0] + H[4] * nn[1] + H[5] * nn[2]};
        int          c    = 0;
        for (int a = 0; a < 3; ++a) {
            for (int b = a; b < 3; ++b) {
                out.curvature[c][i] = (H[c] - nn[a] * m[b] - m[a] * nn[b]) / q;
                c++;
            }
        }
    }
}

#endif  // NODE_INTERPOLATE_KERNEL_HPP
