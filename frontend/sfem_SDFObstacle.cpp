#include "sfem_SDFObstacle.hpp"

#include "node_interpolate.hpp"
#include "node_interpolate_kernel.hpp"
#include "sfem_resample_gap.hpp"
#include "sfem_sdf_sampling.hpp"

#include <stdlib.h>

#ifdef SFEM_ENABLE_CUDA
#include "cu_node_interpolate.hpp"
#include "cu_resample_gap.hpp"
#endif

namespace sfem {

    std::shared_ptr<SDFBSpline> SDFBSpline::create(const std::shared_ptr<smesh::Grid<geom_t>> &sdf) {
        return create(sdf->nlocal(), sdf->stride(), sdf->data());
    }

    std::shared_ptr<SDFBSpline> SDFBSpline::create(const ptrdiff_t *const n,
                                                   const ptrdiff_t *const sample_stride,
                                                   const geom_t *const    samples) {
        SFEM_TRACE_SCOPE("SDFBSpline::create");

        auto ret    = std::make_shared<SDFBSpline>();
        ret->stride = create_host_buffer<ptrdiff_t>(3);
        ptrdiff_t *const stride = ret->stride->data();
        stride[0]               = 1;
        stride[1]               = n[0] + 2;
        stride[2]               = (n[0] + 2) * (n[1] + 2);
        ret->coefficients       = create_host_buffer<geom_t>((n[0] + 2) * (n[1] + 2) * (n[2] + 2));
        ret->offset_            = stride[0] + stride[1] + stride[2];

        if (sdf_bspline_coefficients(n, sample_stride, samples, stride, ret->origin_coefficient()) != SFEM_SUCCESS) {
            SFEM_ERROR("SDFBSpline: the SDF grid needs at least 2 nodes per axis\n");
        }
        return ret;
    }

#ifdef SFEM_ENABLE_CUDA
    std::shared_ptr<SDFBSpline> SDFBSpline::to_device() const {
        auto ret          = std::make_shared<SDFBSpline>();
        ret->coefficients = smesh::to_device(coefficients);
        ret->stride       = smesh::to_device(stride);
        ret->offset_      = offset_;
        return ret;
    }
#endif

    class SDFObstacle::Impl {
    public:
        // What every sampling kernel reads: the SDF's cubic B-spline over the part of the grid
        // this rank holds, n nodes per axis from `origin`, spaced `delta`.
        std::shared_ptr<SDFBSpline> spline;
        ptrdiff_t                   n[3];
        geom_t                      origin[3];
        geom_t                      delta[3];
        // The spline's strides on the host, for kernels that take the grid by value.
        ptrdiff_t                   stride[3];
        bool                        variational{true};
        ExecutionContext            context{EXECUTION_SPACE_HOST};

        NodeSamplerGrid grid() const {
            NodeSamplerGrid g;
            for (int d = 0; d < 3; ++d) {
                g.n[d]      = n[d];
                g.stride[d] = stride[d];
                g.origin[d] = origin[d];
                g.delta[d]  = delta[d];
            }
            g.data = spline->origin_coefficient();
            return g;
        }
    };

    std::shared_ptr<SDFObstacle> SDFObstacle::create_from_file(const std::shared_ptr<Communicator> &comm,
                                                               const std::string                   &path,
                                                               const ptrdiff_t                      nnodes,
                                                               const geom_t *const                  z,
                                                               const ExecutionContext              &context) {
        return create(smesh::Grid<geom_t>::create_from_file(comm, path.c_str()), comm, nnodes, z, context);
    }

    std::shared_ptr<SDFObstacle> SDFObstacle::create(const std::shared_ptr<smesh::Grid<geom_t>> &sdf,
                                                     const std::shared_ptr<Communicator>        &comm,
                                                     const ptrdiff_t                             nnodes,
                                                     const geom_t *const                         z,
                                                     const ExecutionContext                     &context) {
        auto obs = std::make_shared<SDFObstacle>();
        for (int d = 0; d < 3; ++d) {
            obs->impl_->n[d]      = sdf->nlocal()[d];
            obs->impl_->origin[d] = sdf->origin(d);
            obs->impl_->delta[d]  = sdf->delta(d);
        }

        // A grid read from file under MPI is split into z-slabs, each with the whole grid's
        // origin, while the rank's points can lie anywhere in z. The spline is then built on the
        // z-range the points cover, widened by 12 layers: the prefilter's reach to float
        // precision, so there it equals the whole grid's spline.
        const bool slabs = comm->size() > 1 && comm->max<ptrdiff_t>(sdf->nglobal(2) - sdf->nlocal()[2]) > 0;

        std::shared_ptr<SDFBSpline> spline;
        if (slabs) {
            geom_t *view = nullptr;
            if (sdf_view_ensure_margin(comm->get(),
                                       nnodes,
                                       z,
                                       sdf->nlocal(),
                                       sdf->nglobal(),
                                       sdf->stride(),
                                       sdf->origin(),
                                       sdf->delta(),
                                       sdf->data(),
                                       12,
                                       &view,
                                       &obs->impl_->n[2],
                                       &obs->impl_->origin[2]) != SFEM_SUCCESS) {
                SFEM_ERROR("SDFObstacle: unable to gather the SDF slab of this rank's points\n");
            }
            spline = SDFBSpline::create(obs->impl_->n, sdf->stride(), view);
            free(view);
        } else {
            spline = SDFBSpline::create(sdf);
        }

        for (int d = 0; d < 3; ++d) obs->impl_->stride[d] = spline->stride->data()[d];
#ifdef SFEM_ENABLE_CUDA
        if (context.space() == EXECUTION_SPACE_DEVICE) {
            spline = spline->to_device();
        }
#endif
        obs->impl_->spline          = spline;
        obs->impl_->context         = context;
        return obs;
    }

    SDFObstacle::SDFObstacle() : impl_(std::make_unique<Impl>()) {}
    SDFObstacle::~SDFObstacle() {}

    void SDFObstacle::set_variational(const bool enabled) { impl_->variational = enabled; }

    int SDFObstacle::sample(smesh::ElemType              element_type,
                            const ptrdiff_t              nelements,
                            const ptrdiff_t              nnodes,
                            idx_t **const SFEM_RESTRICT  elements,
                            geom_t **const SFEM_RESTRICT points,
                            real_t **const SFEM_RESTRICT normals,
                            real_t *const SFEM_RESTRICT  gap) {
        SFEM_TRACE_SCOPE("SDFObstacle::sample");

#ifdef SFEM_ENABLE_CUDA
        if (impl_->context.space() == EXECUTION_SPACE_DEVICE) {
            return cu_resample_gap(
                    // Mesh
                    element_type,
                    nelements,
                    nnodes,
                    elements,
                    points,
                    // SDF
                    impl_->n,
                    impl_->spline->stride->data(),
                    impl_->origin,
                    impl_->delta,
                    impl_->spline->origin_coefficient(),
                    // Output
                    gap,
                    normals,
                    impl_->context.stream());
        }
#endif
        return resample_gap(
                // Mesh
                element_type,
                nelements,
                nnodes,
                elements,
                points,
                // SDF
                impl_->n,
                impl_->spline->stride->data(),
                impl_->origin,
                impl_->delta,
                impl_->spline->origin_coefficient(),
                // Output
                gap,
                normals[0],
                normals[1],
                normals[2]);
    }

    int SDFObstacle::sample_normals(smesh::ElemType              element_type,
                                    const ptrdiff_t              nelements,
                                    const ptrdiff_t              nnodes,
                                    idx_t **const SFEM_RESTRICT  elements,
                                    geom_t **const SFEM_RESTRICT points,
                                    real_t **const SFEM_RESTRICT normals) {
        SFEM_TRACE_SCOPE("SDFObstacle::sample_normals");

#ifdef SFEM_ENABLE_CUDA
        if (impl_->context.space() == EXECUTION_SPACE_DEVICE) {
            return cu_resample_gap_normals(
                    // Mesh
                    element_type,
                    nelements,
                    nnodes,
                    elements,
                    points,
                    // SDF
                    impl_->n,
                    impl_->spline->stride->data(),
                    impl_->origin,
                    impl_->delta,
                    impl_->spline->origin_coefficient(),
                    // Output
                    normals,
                    impl_->context.stream());
        }
#endif
        return resample_gap_normals(
                // Mesh
                element_type,
                nelements,
                nnodes,
                elements,
                points,
                // SDF
                impl_->n,
                impl_->spline->stride->data(),
                impl_->origin,
                impl_->delta,
                impl_->spline->origin_coefficient(),
                // Output
                normals[0],
                normals[1],
                normals[2]);
    }

    int SDFObstacle::sample_normals_accumulate(smesh::ElemType              element_type,
                                               const ptrdiff_t              nelements,
                                               const ptrdiff_t              nnodes,
                                               idx_t **const SFEM_RESTRICT  elements,
                                               geom_t **const SFEM_RESTRICT points,
                                               real_t **const SFEM_RESTRICT normals) {
        SFEM_TRACE_SCOPE("SDFObstacle::sample_normals_accumulate");

        return resample_gap_normals_local(element_type,
                                          nelements,
                                          nnodes,
                                          elements,
                                          points,
                                          impl_->n,
                                          impl_->spline->stride->data(),
                                          impl_->origin,
                                          impl_->delta,
                                          impl_->spline->origin_coefficient(),
                                          normals[0],
                                          normals[1],
                                          normals[2]);
    }

    int SDFObstacle::sample_value_accumulate(smesh::ElemType              element_type,
                                             const ptrdiff_t              nelements,
                                             const ptrdiff_t              nnodes,
                                             idx_t **const SFEM_RESTRICT  elements,
                                             geom_t **const SFEM_RESTRICT points,
                                             real_t *const SFEM_RESTRICT  gap) {
        SFEM_TRACE_SCOPE("SDFObstacle::sample_value_accumulate");

        return resample_gap_value_local(element_type,
                                        nelements,
                                        nnodes,
                                        elements,
                                        points,
                                        impl_->n,
                                        impl_->spline->stride->data(),
                                        impl_->origin,
                                        impl_->delta,
                                        impl_->spline->origin_coefficient(),
                                        gap);
    }

    int SDFObstacle::sample_value(smesh::ElemType              element_type,
                                  const ptrdiff_t              nelements,
                                  const ptrdiff_t              nnodes,
                                  idx_t **const SFEM_RESTRICT  elements,
                                  geom_t **const SFEM_RESTRICT points,
                                  real_t *const SFEM_RESTRICT  gap) {
        SFEM_TRACE_SCOPE("SDFObstacle::sample_value");

#ifdef SFEM_ENABLE_CUDA
        if (impl_->context.space() == EXECUTION_SPACE_DEVICE) {
            return cu_resample_gap_value(
                    // Mesh
                    element_type,
                    nelements,
                    nnodes,
                    elements,
                    points,
                    // SDF
                    impl_->n,
                    impl_->spline->stride->data(),
                    impl_->origin,
                    impl_->delta,
                    impl_->spline->origin_coefficient(),
                    // Output
                    gap,
                    impl_->context.stream());
        }
#endif
        return resample_gap_value(
                // Mesh
                element_type,
                nelements,
                nnodes,
                elements,
                points,
                // SDF
                impl_->n,
                impl_->spline->stride->data(),
                impl_->origin,
                impl_->delta,
                impl_->spline->origin_coefficient(),
                // Output
                gap);
    }

    int SDFObstacle::sample_nodes(const ptrdiff_t                   nnodes,
                                  const idx_t *const SFEM_RESTRICT  idx,
                                  geom_t **const SFEM_RESTRICT      points,
                                  const real_t *const SFEM_RESTRICT disp,
                                  real_t *const SFEM_RESTRICT       gap,
                                  real_t **const SFEM_RESTRICT      normals) {
        SFEM_TRACE_SCOPE("SDFObstacle::sample_nodes");

#ifdef SFEM_ENABLE_CUDA
        if (impl_->context.space() == EXECUTION_SPACE_DEVICE) {
            NodeSamplerOutput out = {gap,
                                     {normals ? normals[0] : nullptr, normals ? normals[1] : nullptr, normals ? normals[2] : nullptr},
                                     {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr}};
            return cu_sample_nodes(
                    1, nnodes, idx, {{points[0], points[1], points[2]}}, disp, impl_->grid(), out, impl_->context.stream());
        }
#endif
        return interpolate_displaced_gap(nnodes,
                                         idx,
                                         points,
                                         disp,
                                         impl_->n,
                                         impl_->spline->stride->data(),
                                         impl_->origin,
                                         impl_->delta,
                                         impl_->spline->origin_coefficient(),
                                         gap,
                                         normals);
    }

    int SDFObstacle::sample_nodes_curvature(const ptrdiff_t                   nnodes,
                                            const idx_t *const SFEM_RESTRICT  idx,
                                            geom_t **const SFEM_RESTRICT      points,
                                            const real_t *const SFEM_RESTRICT disp,
                                            real_t **const SFEM_RESTRICT      curvature) {
        SFEM_TRACE_SCOPE("SDFObstacle::sample_nodes_curvature");

#ifdef SFEM_ENABLE_CUDA
        if (impl_->context.space() == EXECUTION_SPACE_DEVICE) {
            NodeSamplerOutput out = {nullptr, {nullptr, nullptr, nullptr}, {curvature[0], curvature[1], curvature[2], curvature[3], curvature[4], curvature[5]}};
            return cu_sample_nodes(
                    2, nnodes, idx, {{points[0], points[1], points[2]}}, disp, impl_->grid(), out, impl_->context.stream());
        }
#endif
        return interpolate_displaced_curvature(nnodes,
                                               idx,
                                               points,
                                               disp,
                                               impl_->n,
                                               impl_->spline->stride->data(),
                                               impl_->origin,
                                               impl_->delta,
                                               impl_->spline->origin_coefficient(),
                                               curvature);
    }
}  // namespace sfem
