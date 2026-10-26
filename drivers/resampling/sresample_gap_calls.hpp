#ifndef SFEM_DRIVERS_SRESAMPLE_GAP_CALLS_HPP
#define SFEM_DRIVERS_SRESAMPLE_GAP_CALLS_HPP

// The raw-array SDF sampling calls the gap_from_sdf drivers were written against, forwarded to
// sresample. Temporary: the drivers become sresample demos when the migration is finished
// (MIGRATION_SRESAMPLE.md, stage S8), and this goes with them.

#include <mpi.h>

#include "sfem_base.hpp"
#include "smesh_elem_type.hpp"

#include "sresample_node_sample.hpp"
#include "sresample_surface_sample.hpp"
#include "sresample_zslab.hpp"

#include <vector>

namespace sresample_gap_calls {

    inline sresample::ElemType shell(const smesh::ElemType element_type) {
        switch (smesh::shell_type(element_type)) {
            case smesh::TRISHELL3:
                return sresample::ElemType::TRISHELL3;
            case smesh::QUADSHELL4:
                return sresample::ElemType::QUADSHELL4;
            case smesh::BEAM2:
            case smesh::EDGESHELL2:
                return sresample::ElemType::BEAM2;
            default:
                SFEM_ERROR("Invalid shell element type from element type %d\n", element_type);
                return sresample::ElemType::TRISHELL3;
        }
    }

    inline sresample::BSplineGrid<geom_t, geom_t> grid(const ptrdiff_t *const n,
                                                       const ptrdiff_t *const stride,
                                                       const geom_t *const    origin,
                                                       const geom_t *const    delta,
                                                       const geom_t *const    data) {
        sresample::BSplineGrid<geom_t, geom_t> g;
        for (int d = 0; d < 3; ++d) {
            g.n[d]      = n[d];
            g.stride[d] = stride[d];
            g.origin[d] = origin[d];
            g.delta[d]  = delta[d];
        }
        g.coeffs = data;
        return g;
    }

    inline sresample::MeshView<geom_t, idx_t> mesh(const smesh::ElemType element_type,
                                                   const ptrdiff_t       nelements,
                                                   const ptrdiff_t       nnodes,
                                                   idx_t **const         elems,
                                                   geom_t **const        xyz) {
        sresample::MeshView<geom_t, idx_t> m;
        m.type      = shell(element_type);
        m.nelements = nelements;
        m.nnodes    = nnodes;
        m.elements  = elems;
        m.points    = xyz;
        return m;
    }

    inline sresample::sdf::SurfaceOutput<real_t> out(real_t *const g, real_t *const nx, real_t *const ny, real_t *const nz) {
        sresample::sdf::SurfaceOutput<real_t> o;
        o.gap        = g;
        o.normals[0] = nx;
        o.normals[1] = ny;
        o.normals[2] = nz;
        return o;
    }

}  // namespace sresample_gap_calls

inline int resample_gap_local(const smesh::ElemType                element_type,
                              const ptrdiff_t                      nelements,
                              const ptrdiff_t                      nnodes,
                              idx_t **const SFEM_RESTRICT          elems,
                              geom_t **const SFEM_RESTRICT         xyz,
                              const ptrdiff_t *const SFEM_RESTRICT n,
                              const ptrdiff_t *const SFEM_RESTRICT stride,
                              const geom_t *const SFEM_RESTRICT    origin,
                              const geom_t *const SFEM_RESTRICT    delta,
                              const geom_t *const SFEM_RESTRICT    data,
                              real_t *const SFEM_RESTRICT          g,
                              real_t *const SFEM_RESTRICT          xnormal,
                              real_t *const SFEM_RESTRICT          ynormal,
                              real_t *const SFEM_RESTRICT          znormal) {
    using namespace sresample_gap_calls;
    return sresample::sdf::surface_sample_local(mesh(element_type, nelements, nnodes, elems, xyz),
                                                grid(n, stride, origin, delta, data),
                                                out(g, xnormal, ynormal, znormal),
                                                sresample::Exec::host());
}

inline int resample_gap(const smesh::ElemType                element_type,
                        const ptrdiff_t                      nelements,
                        const ptrdiff_t                      nnodes,
                        idx_t **const SFEM_RESTRICT          elems,
                        geom_t **const SFEM_RESTRICT         xyz,
                        const ptrdiff_t *const SFEM_RESTRICT n,
                        const ptrdiff_t *const SFEM_RESTRICT stride,
                        const geom_t *const SFEM_RESTRICT    origin,
                        const geom_t *const SFEM_RESTRICT    delta,
                        const geom_t *const SFEM_RESTRICT    data,
                        real_t *const SFEM_RESTRICT          g,
                        real_t *const SFEM_RESTRICT          xnormal,
                        real_t *const SFEM_RESTRICT          ynormal,
                        real_t *const SFEM_RESTRICT          znormal) {
    using namespace sresample_gap_calls;
    std::vector<real_t> w(nnodes);
    return sresample::sdf::surface_sample(mesh(element_type, nelements, nnodes, elems, xyz),
                                          grid(n, stride, origin, delta, data),
                                          out(g, xnormal, ynormal, znormal),
                                          w.data(),
                                          sresample::Exec::host());
}

inline int interpolate_gap(const ptrdiff_t                      nnodes,
                           geom_t **const SFEM_RESTRICT         xyz,
                           const ptrdiff_t *const SFEM_RESTRICT n,
                           const ptrdiff_t *const SFEM_RESTRICT stride,
                           const geom_t *const SFEM_RESTRICT    origin,
                           const geom_t *const SFEM_RESTRICT    delta,
                           const geom_t *const SFEM_RESTRICT    data,
                           real_t *const SFEM_RESTRICT          g,
                           real_t *const SFEM_RESTRICT          xnormal,
                           real_t *const SFEM_RESTRICT          ynormal,
                           real_t *const SFEM_RESTRICT          znormal) {
    using namespace sresample_gap_calls;
    sresample::sdf::NodePoints<geom_t> points;
    for (int d = 0; d < 3; ++d) points.xyz[d] = xyz[d];
    sresample::sdf::NodeOutput<real_t> o;
    o.gap        = g;
    o.normals[0] = xnormal;
    o.normals[1] = ynormal;
    o.normals[2] = znormal;
    return sresample::sdf::sample_nodes(
            1, nnodes, (const idx_t *)nullptr, points, (const real_t *)nullptr, grid(n, stride, origin, delta, data), o, sresample::Exec::host());
}

inline int sdf_view_ensure_margin(MPI_Comm                             comm,
                                  const ptrdiff_t                      nnodes,
                                  const geom_t *SFEM_RESTRICT          z_coordinate,
                                  const ptrdiff_t *const               nlocal,
                                  const ptrdiff_t *const SFEM_RESTRICT nglobal,
                                  const ptrdiff_t *const SFEM_RESTRICT stride,
                                  const geom_t *const                  origin,
                                  const geom_t *const SFEM_RESTRICT    delta,
                                  const geom_t *const                  sdf,
                                  const ptrdiff_t                      z_margin,
                                  geom_t                             **sdf_out,
                                  ptrdiff_t                           *z_nlocal_out,
                                  geom_t *const SFEM_RESTRICT          z_origin_out) {
    return sresample::mpi::zslab_view(
            comm, nnodes, z_coordinate, nlocal, nglobal, stride, origin, delta, sdf, z_margin, sdf_out, z_nlocal_out, z_origin_out);
}

#endif  // SFEM_DRIVERS_SRESAMPLE_GAP_CALLS_HPP
