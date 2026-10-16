#include "node_interpolate.hpp"

#include <math.h>
#include <stdio.h>

#include "node_interpolate_kernel.hpp"

// Nodal sampling of an SDF grid's interpolating cubic B-spline: `data` and `stride` are its
// coefficients (sfem_sdf_sampling.hpp). The gap is -sdf / |grad sdf| (-sdf for the value alone),
// the normal the field's unit gradient.

namespace {
    // sample_node (node_interpolate_kernel.hpp) at every node.
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
        NodeSamplerGrid grid;
        for (int d = 0; d < 3; ++d) {
            grid.n[d]      = n[d];
            grid.stride[d] = stride[d];
            grid.origin[d] = origin[d];
            grid.delta[d]  = delta[d];
        }
        grid.data = data;

        const NodeSamplerPoints points = {{xyz[0], xyz[1], xyz[2]}};
        NodeSamplerOutput       out    = {gap, {nullptr, nullptr, nullptr}, {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr}};
        if (normals) {
            for (int d = 0; d < 3; ++d) out.normals[d] = normals[d];
        }
        if (curvature) {
            for (int c = 0; c < 6; ++c) out.curvature[c] = curvature[c];
        }

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < nnodes; ++i) {
            sample_node<ORDER>(i, idx, points, disp, grid, out);
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
