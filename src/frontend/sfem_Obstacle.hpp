#ifndef SFEM_OBSTACLE_HPP
#define SFEM_OBSTACLE_HPP

#include "sfem_base.hpp"
#include "sfem_defs.hpp"

#include "smesh_elem_type.hpp"

namespace sfem {

    // The gap an obstacle reports where it cannot be sampled (outside its grid): large, so the
    // contact constraint there stays inactive. sresample's sdf::unreachable.
    static constexpr int obstacle_unreachable = 10000;

    // An obstacle sampled on a contact surface (dual-weighted over its elements) or at nodes.
    // SDFObstacle (sdf_contact/) is the one implementation, built with SFEM_ENABLE_SRESAMPLE.
    class Obstacle {
    public:
        virtual ~Obstacle() = default;

        virtual int sample(smesh::ElemType              element_type,
                           const ptrdiff_t              nelements,
                           const ptrdiff_t              nnodes,
                           idx_t **const SFEM_RESTRICT  elements,
                           geom_t **const SFEM_RESTRICT points,
                           real_t **const SFEM_RESTRICT normals,
                           real_t *const SFEM_RESTRICT  gap) = 0;

        virtual int sample_normals(smesh::ElemType              element_type,
                                   const ptrdiff_t              nelements,
                                   const ptrdiff_t              nnodes,
                                   idx_t **const SFEM_RESTRICT  elements,
                                   geom_t **const SFEM_RESTRICT points,
                                   real_t **const SFEM_RESTRICT normals) = 0;

        virtual int sample_value(smesh::ElemType              element_type,
                                 const ptrdiff_t              nelements,
                                 const ptrdiff_t              nnodes,
                                 idx_t **const SFEM_RESTRICT  elements,
                                 geom_t **const SFEM_RESTRICT points,
                                 real_t *const SFEM_RESTRICT  gap) = 0;

        // Local assembly only (no lumped-mass divide / unit normalize). MPI callers
        // scatter-add then finish the average themselves.
        // Host only.
        virtual int sample_normals_accumulate(smesh::ElemType              element_type,
                                              const ptrdiff_t              nelements,
                                              const ptrdiff_t              nnodes,
                                              idx_t **const SFEM_RESTRICT  elements,
                                              geom_t **const SFEM_RESTRICT points,
                                              real_t **const SFEM_RESTRICT normals) = 0;
        virtual int sample_value_accumulate(smesh::ElemType              element_type,
                                            const ptrdiff_t              nelements,
                                            const ptrdiff_t              nnodes,
                                            idx_t **const SFEM_RESTRICT  elements,
                                            geom_t **const SFEM_RESTRICT points,
                                            real_t *const SFEM_RESTRICT  gap) = 0;

        // Adds the lumped mass the sampled gap is divided by (the integral of each node's basis
        // function over the surface) into w. Host only.
        virtual int sample_weights_accumulate(smesh::ElemType              element_type,
                                              const ptrdiff_t              nelements,
                                              const ptrdiff_t              nnodes,
                                              idx_t **const SFEM_RESTRICT  elements,
                                              geom_t **const SFEM_RESTRICT points,
                                              real_t *const SFEM_RESTRICT  w) = 0;

        // At nodes: the gap and unit normals (3 rows; either output may be null) at the points
        // X[idx[i]] + disp[3 idx[i] + d] (idx and disp may be null), in the obstacle's memory space.
        virtual int sample_nodes(const ptrdiff_t                   nnodes,
                                 const idx_t *const SFEM_RESTRICT  idx,
                                 geom_t **const SFEM_RESTRICT      points,
                                 const real_t *const SFEM_RESTRICT disp,
                                 real_t *const SFEM_RESTRICT       gap,
                                 real_t **const SFEM_RESTRICT      normals) = 0;

        // At the same points: the curvature of the gap's constraint (6 packed rows [xx, xy, xz, yy,
        // yz, zz]), minus the Hessian of -sdf / |grad sdf|.
        virtual int sample_nodes_curvature(const ptrdiff_t                   nnodes,
                                           const idx_t *const SFEM_RESTRICT  idx,
                                           geom_t **const SFEM_RESTRICT      points,
                                           const real_t *const SFEM_RESTRICT disp,
                                           real_t **const SFEM_RESTRICT      curvature) = 0;
    };

}  // namespace sfem

#endif  // SFEM_OBSTACLE_HPP
