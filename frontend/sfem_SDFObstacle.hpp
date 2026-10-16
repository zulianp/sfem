#ifndef SFEM_SDFOBSTACLE_HPP
#define SFEM_SDFOBSTACLE_HPP

#include "sfem_defs.hpp"

#include "sfem_aliases.hpp"
#include "smesh_grid.hpp"

#include <mpi.h>
#include <memory>

namespace sfem {
    class Obstacle {
    public:
        virtual ~Obstacle()                                 = default;
        virtual int sample(smesh::ElemType                element_type,
                           const ptrdiff_t              nelements,
                           const ptrdiff_t              nnodes,
                           idx_t **const SFEM_RESTRICT  elements,
                           geom_t **const SFEM_RESTRICT points,
                           real_t **const SFEM_RESTRICT normals,
                           real_t *const SFEM_RESTRICT  gap) = 0;

        virtual int sample_normals(smesh::ElemType                element_type,
                                   const ptrdiff_t              nelements,
                                   const ptrdiff_t              nnodes,
                                   idx_t **const SFEM_RESTRICT  elements,
                                   geom_t **const SFEM_RESTRICT points,
                                   real_t **const SFEM_RESTRICT normals) = 0;

        virtual int sample_value(smesh::ElemType                element_type,
                                 const ptrdiff_t              nelements,
                                 const ptrdiff_t              nnodes,
                                 idx_t **const SFEM_RESTRICT  elements,
                                 geom_t **const SFEM_RESTRICT points,
                                 real_t *const SFEM_RESTRICT  gap) = 0;
    };

    // The interpolating cubic B-spline coefficients of an SDF grid (sfem_sdf_sampling.hpp), padded
    // by one node on every side, so the sampler's four-node stencil stays in the array for every
    // point inside the grid. A sampler is given the SDF grid's n, origin and delta, `stride` and
    // origin_coefficient(), the coefficient of the SDF grid's node (0, 0, 0).
    class SDFBSpline {
    public:
        SharedBuffer<geom_t>    coefficients;  // (n[0] + 2) (n[1] + 2) (n[2] + 2), x fastest
        SharedBuffer<ptrdiff_t> stride;        // of the padded array, in the coefficients' memory space

        geom_t *origin_coefficient() const { return coefficients->data() + offset_; }

        // From host SDF samples, n per axis with strides `stride`; computed on the host.
        static std::shared_ptr<SDFBSpline> create(const ptrdiff_t *const n,
                                                  const ptrdiff_t *const stride,
                                                  const geom_t *const    samples);
        static std::shared_ptr<SDFBSpline> create(const std::shared_ptr<smesh::Grid<geom_t>> &sdf);
#ifdef SFEM_ENABLE_CUDA
        std::shared_ptr<SDFBSpline> to_device() const;
#endif

    private:
        ptrdiff_t offset_{0};
    };

    class SDFObstacle final : public Obstacle {
    public:
        SDFObstacle();
        ~SDFObstacle();

        void set_variational(const bool enabled);
        int  sample(smesh::ElemType                element_type,
                    const ptrdiff_t              nelements,
                    const ptrdiff_t              nnodes,
                    idx_t **const SFEM_RESTRICT  elements,
                    geom_t **const SFEM_RESTRICT points,
                    real_t **const SFEM_RESTRICT normals,
                    real_t *const SFEM_RESTRICT  gap) override;

        int sample_normals(smesh::ElemType                element_type,
                           const ptrdiff_t              nelements,
                           const ptrdiff_t              nnodes,
                           idx_t **const SFEM_RESTRICT  elements,
                           geom_t **const SFEM_RESTRICT points,
                           real_t **const SFEM_RESTRICT normals) override;

        int sample_value(smesh::ElemType                element_type,
                         const ptrdiff_t              nelements,
                         const ptrdiff_t              nnodes,
                         idx_t **const SFEM_RESTRICT  elements,
                         geom_t **const SFEM_RESTRICT points,
                         real_t *const SFEM_RESTRICT  gap) override;

        // Local assembly only (no lumped-mass divide / unit normalize). MPI callers
        // scatter-add then finish the average themselves.
        int sample_normals_accumulate(smesh::ElemType              element_type,
                                      const ptrdiff_t              nelements,
                                      const ptrdiff_t              nnodes,
                                      idx_t **const SFEM_RESTRICT  elements,
                                      geom_t **const SFEM_RESTRICT points,
                                      real_t **const SFEM_RESTRICT normals);
        int sample_value_accumulate(smesh::ElemType              element_type,
                                    const ptrdiff_t              nelements,
                                    const ptrdiff_t              nnodes,
                                    idx_t **const SFEM_RESTRICT  elements,
                                    geom_t **const SFEM_RESTRICT points,
                                    real_t *const SFEM_RESTRICT  gap);

        // At nodes: the gap, unit normals (3 rows) and curvature (6 rows) of node_interpolate.hpp
        // at the points X[idx[i]] + disp[3 idx[i] + d] (idx and disp may be null). Host only.
        // (Also on the device, with the points, idx, disp and outputs in device memory.)
        int sample_nodes(const ptrdiff_t                   nnodes,
                         const idx_t *const SFEM_RESTRICT  idx,
                         geom_t **const SFEM_RESTRICT      points,
                         const real_t *const SFEM_RESTRICT disp,
                         real_t *const SFEM_RESTRICT       gap,
                         real_t **const SFEM_RESTRICT      normals);
        int sample_nodes_curvature(const ptrdiff_t                   nnodes,
                                   const idx_t *const SFEM_RESTRICT  idx,
                                   geom_t **const SFEM_RESTRICT      points,
                                   const real_t *const SFEM_RESTRICT disp,
                                   real_t **const SFEM_RESTRICT      curvature);

        // `nnodes` points with z-coordinates `z` are where this rank will sample: a grid split
        // into z-slabs across the ranks (Grid::create_from_file under MPI) is gathered to the
        // slab they need. A grid every rank holds whole needs no points.
        static std::shared_ptr<SDFObstacle> create_from_file(const std::shared_ptr<Communicator>& comm,
                                                             const std::string        &path,
                                                             const ptrdiff_t           nnodes,
                                                             const geom_t *const       z,
                                                             const enum ExecutionSpace es);
        static std::shared_ptr<SDFObstacle> create(const std::shared_ptr<smesh::Grid<geom_t>> &sdf,
                                                   const std::shared_ptr<Communicator>        &comm,
                                                   const ptrdiff_t                             nnodes,
                                                   const geom_t *const                         z,
                                                   const enum ExecutionSpace                   es);

    public:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
}  // namespace sfem

#endif  // SFEM_SDFOBSTACLE_HPP
