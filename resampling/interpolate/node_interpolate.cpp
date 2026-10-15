#include "node_interpolate.hpp"

#include <math.h>
#include <stdio.h>

#include "sfem_sdf_sampling.hpp"

// Nodal sampling of an SDF grid's interpolating cubic B-spline: `data` and `stride` are its
// coefficients (sfem_sdf_sampling.hpp). The gap is -sdf / |grad sdf| (-sdf for the value alone),
// the normal the field's unit gradient.

namespace {
    // The ORDER-th derivatives of the B-spline s at the points X[node] + disp[3 node + d], for
    // node = idx[i] (i when idx is null; disp may be null), summed in real_t. Writes entry i of
    // whichever outputs are given:
    //   gap        -s, with ORDER >= 1 -s / |grad s|
    //   normals    grad s / |grad s|                              (ORDER >= 1, 3 rows)
    //   curvature  (H - n m^T - m n^T) / |grad s|, m = H n        (ORDER >= 2, 6 rows)
    // Outside the grid the gap is SFEM_SDF_UNREACHABLE and the normal and curvature are zero; so
    // are they where the gradient vanishes.
    template <int ORDER>
    void sample_nodes(const ptrdiff_t                      nnodes,
                      const idx_t* const SFEM_RESTRICT     idx,
                      geom_t** const SFEM_RESTRICT         xyz,
                      const real_t* const SFEM_RESTRICT    disp,
                      const ptrdiff_t* const SFEM_RESTRICT n,
                      const ptrdiff_t* const SFEM_RESTRICT stride,
                      const geom_t* const SFEM_RESTRICT    origin,
                      const geom_t* const SFEM_RESTRICT    delta,
                      const geom_t* const SFEM_RESTRICT    data,
                      real_t* const SFEM_RESTRICT          gap,
                      real_t** const SFEM_RESTRICT         normals,
                      real_t** const SFEM_RESTRICT         curvature) {
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < nnodes; ++i) {
            const ptrdiff_t node = idx ? (ptrdiff_t)idx[i] : i;
            real_t          p[3];
            for (int d = 0; d < 3; ++d) {
                p[d] = (real_t)xyz[d][node] + (disp ? disp[node * 3 + d] : real_t(0));
            }

            real_t s, g[3] = {0, 0, 0}, H[6] = {0, 0, 0, 0, 0, 0};
            if (!sdf_sample<ORDER>(n, stride, origin, delta, data, p[0], p[1], p[2], &s, g, H)) {
                if (gap) gap[i] = SFEM_SDF_UNREACHABLE;
                if (normals) {
                    for (int d = 0; d < 3; ++d) normals[d][i] = 0;
                }
                if (curvature) {
                    for (int c = 0; c < 6; ++c) curvature[c][i] = 0;
                }
                continue;
            }

            const real_t q = sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
            if (ORDER == 0 || q == 0) {
                if (gap) gap[i] = -s;
                if (normals) {
                    for (int d = 0; d < 3; ++d) normals[d][i] = 0;
                }
                if (curvature) {
                    for (int c = 0; c < 6; ++c) curvature[c][i] = 0;
                }
                continue;
            }

            const real_t nn[3] = {g[0] / q, g[1] / q, g[2] / q};
            if (gap) gap[i] = -s / q;
            if (normals) {
                for (int d = 0; d < 3; ++d) normals[d][i] = nn[d];
            }

            if (ORDER >= 2 && curvature) {
                // m = H n = grad |grad s|; the Hessian of the gap -s / |grad s| at s = 0 is
                // -(H - n m^T - m n^T) / |grad s|.
                const real_t m[3] = {H[0] * nn[0] + H[1] * nn[1] + H[2] * nn[2],
                                     H[1] * nn[0] + H[3] * nn[1] + H[4] * nn[2],
                                     H[2] * nn[0] + H[4] * nn[1] + H[5] * nn[2]};
                int c = 0;
                for (int a = 0; a < 3; ++a) {
                    for (int b = a; b < 3; ++b) {
                        curvature[c][i] = (H[c] - nn[a] * m[b] - m[a] * nn[b]) / q;
                        c++;
                    }
                }
            }
        }
    }
}  // namespace

int interpolate_gap(const ptrdiff_t              nnodes,
                    geom_t** const SFEM_RESTRICT xyz,
                    // SDF
                    const ptrdiff_t* const SFEM_RESTRICT n,
                    const ptrdiff_t* const SFEM_RESTRICT stride,
                    const geom_t* const SFEM_RESTRICT    origin,
                    const geom_t* const SFEM_RESTRICT    delta,
                    const geom_t* const SFEM_RESTRICT    data,
                    // Output
                    real_t* const SFEM_RESTRICT g,
                    real_t* const SFEM_RESTRICT xnormal,
                    real_t* const SFEM_RESTRICT ynormal,
                    real_t* const SFEM_RESTRICT znormal) {
    real_t* normals[3] = {xnormal, ynormal, znormal};
    sample_nodes<1>(nnodes, nullptr, xyz, nullptr, n, stride, origin, delta, data, g, normals, nullptr);
    return SFEM_SUCCESS;
}

int interpolate_gap_normals(const ptrdiff_t              nnodes,
                            geom_t** const SFEM_RESTRICT xyz,
                            // SDF
                            const ptrdiff_t* const SFEM_RESTRICT n,
                            const ptrdiff_t* const SFEM_RESTRICT stride,
                            const geom_t* const SFEM_RESTRICT    origin,
                            const geom_t* const SFEM_RESTRICT    delta,
                            const geom_t* const SFEM_RESTRICT    data,
                            // Output
                            real_t* const SFEM_RESTRICT xnormal,
                            real_t* const SFEM_RESTRICT ynormal,
                            real_t* const SFEM_RESTRICT znormal) {
    real_t* normals[3] = {xnormal, ynormal, znormal};
    sample_nodes<1>(nnodes, nullptr, xyz, nullptr, n, stride, origin, delta, data, nullptr, normals, nullptr);
    return SFEM_SUCCESS;
}

int interpolate_gap_value(const ptrdiff_t              nnodes,
                          geom_t** const SFEM_RESTRICT xyz,
                          // SDF
                          const ptrdiff_t* const SFEM_RESTRICT n,
                          const ptrdiff_t* const SFEM_RESTRICT stride,
                          const geom_t* const SFEM_RESTRICT    origin,
                          const geom_t* const SFEM_RESTRICT    delta,
                          const geom_t* const SFEM_RESTRICT    data,
                          // Output
                          real_t* const SFEM_RESTRICT g) {
    sample_nodes<0>(nnodes, nullptr, xyz, nullptr, n, stride, origin, delta, data, g, nullptr, nullptr);
    return SFEM_SUCCESS;
}

int interpolate_displaced_gap(const ptrdiff_t                   nnodes,
                              const idx_t* const SFEM_RESTRICT  idx,
                              geom_t** const SFEM_RESTRICT      xyz,
                              const real_t* const SFEM_RESTRICT disp,
                              // SDF
                              const ptrdiff_t* const SFEM_RESTRICT n,
                              const ptrdiff_t* const SFEM_RESTRICT stride,
                              const geom_t* const SFEM_RESTRICT    origin,
                              const geom_t* const SFEM_RESTRICT    delta,
                              const geom_t* const SFEM_RESTRICT    data,
                              // Output
                              real_t* const SFEM_RESTRICT  g,
                              real_t** const SFEM_RESTRICT normals) {
    sample_nodes<1>(nnodes, idx, xyz, disp, n, stride, origin, delta, data, g, normals, nullptr);
    return SFEM_SUCCESS;
}

int interpolate_displaced_curvature(const ptrdiff_t                   nnodes,
                                    const idx_t* const SFEM_RESTRICT  idx,
                                    geom_t** const SFEM_RESTRICT      xyz,
                                    const real_t* const SFEM_RESTRICT disp,
                                    // SDF
                                    const ptrdiff_t* const SFEM_RESTRICT n,
                                    const ptrdiff_t* const SFEM_RESTRICT stride,
                                    const geom_t* const SFEM_RESTRICT    origin,
                                    const geom_t* const SFEM_RESTRICT    delta,
                                    const geom_t* const SFEM_RESTRICT    data,
                                    // Output
                                    real_t** const SFEM_RESTRICT curvature) {
    sample_nodes<2>(nnodes, idx, xyz, disp, n, stride, origin, delta, data, nullptr, nullptr, curvature);
    return SFEM_SUCCESS;
}
