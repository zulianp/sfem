#include "sfem_SDFObstacle.hpp"

#include "sresample_bspline.hpp"
#include "sresample_node_sample.hpp"
#include "sresample_surface_sample.hpp"

#ifdef SMESH_ENABLE_MPI
#include "sresample_zslab.hpp"
#endif

#include <stdlib.h>

namespace sfem {

    static_assert(obstacle_unreachable == sresample::sdf::unreachable, "sfem and sresample agree on the unreachable gap");

    namespace {
        // The sresample element type of a contact surface's elements.
        bool to_sresample(const smesh::ElemType element_type, sresample::ElemType *const type) {
            switch (smesh::shell_type(element_type)) {
                case smesh::TRISHELL3:
                    *type = sresample::ElemType::TRISHELL3;
                    return true;
                case smesh::QUADSHELL4:
                    *type = sresample::ElemType::QUADSHELL4;
                    return true;
                case smesh::BEAM2:
                case smesh::EDGESHELL2:
                    *type = sresample::ElemType::BEAM2;
                    return true;
                default:
                    SFEM_ERROR("SDFObstacle: invalid shell element type %d from element type %d\n",
                               smesh::shell_type(element_type),
                               element_type);
                    return false;
            }
        }

        sresample::MeshView<geom_t, idx_t> mesh_view(const smesh::ElemType element_type,
                                                     const ptrdiff_t       nelements,
                                                     const ptrdiff_t       nnodes,
                                                     idx_t **const         elements,
                                                     geom_t **const        points) {
            sresample::MeshView<geom_t, idx_t> m;
            to_sresample(element_type, &m.type);
            m.nelements = nelements;
            m.nnodes    = nnodes;
            m.elements  = elements;
            m.points    = points;
            return m;
        }

        sresample::Exec exec_of(const ExecutionContext &context) {
            return context.space() == EXECUTION_SPACE_DEVICE ? sresample::Exec::device(context.stream()) : sresample::Exec::host();
        }

        int check(const int err) { return err == sresample::SRESAMPLE_SUCCESS ? SFEM_SUCCESS : SFEM_FAILURE; }
    }  // namespace

    std::shared_ptr<SDFBSpline> SDFBSpline::create(const std::shared_ptr<smesh::Grid<geom_t>> &sdf) {
        return create(sdf->nlocal(), sdf->stride(), sdf->data());
    }

    std::shared_ptr<SDFBSpline> SDFBSpline::create(const ptrdiff_t *const n,
                                                   const ptrdiff_t *const sample_stride,
                                                   const geom_t *const    samples) {
        SFEM_TRACE_SCOPE("SDFBSpline::create");

        auto ret    = std::make_shared<SDFBSpline>();
        ret->stride = create_host_buffer<ptrdiff_t>(3);
        ptrdiff_t size;
        sresample::sdf::bspline_layout(n, ret->stride->data(), &size, &ret->offset_);
        ret->coefficients = create_host_buffer<geom_t>(size);

        if (sresample::sdf::bspline_coefficients(n, sample_stride, samples, ret->stride->data(), ret->origin_coefficient()) !=
            sresample::SRESAMPLE_SUCCESS) {
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
        // The lumped mass of the last surface sampled, in the obstacle's memory space.
        SharedBuffer<real_t>        workspace;

        sresample::BSplineGrid<geom_t, geom_t> grid() const {
            sresample::BSplineGrid<geom_t, geom_t> g;
            for (int d = 0; d < 3; ++d) {
                g.n[d]      = n[d];
                g.stride[d] = stride[d];
                g.origin[d] = origin[d];
                g.delta[d]  = delta[d];
            }
            g.coeffs = spline->origin_coefficient();
            return g;
        }

        real_t *workspace_of(const ptrdiff_t nnodes) {
            if (!workspace || (ptrdiff_t)workspace->size() < nnodes) {
                workspace = create_buffer<real_t>(nnodes, context.space());
            }
            return workspace->data();
        }

        int surface_sample(const smesh::ElemType element_type,
                           const ptrdiff_t       nelements,
                           const ptrdiff_t       nnodes,
                           idx_t **const         elements,
                           geom_t **const        points,
                           real_t *const         gap,
                           real_t **const        normals,
                           const bool            local) {
            sresample::sdf::SurfaceOutput<real_t> out;
            out.gap = gap;
            if (normals) {
                for (int d = 0; d < 3; ++d) out.normals[d] = normals[d];
            }
            const auto mesh = mesh_view(element_type, nelements, nnodes, elements, points);
            if (local) {
                return check(sresample::sdf::surface_sample_local(mesh, grid(), out, sresample::Exec::host()));
            }
            return check(sresample::sdf::surface_sample(
                    mesh, grid(), out, gap ? workspace_of(nnodes) : (real_t *)nullptr, exec_of(context)));
        }

        int node_sample(const int                         order,
                        const ptrdiff_t                   nnodes,
                        const idx_t *const SFEM_RESTRICT  idx,
                        geom_t **const SFEM_RESTRICT      points,
                        const real_t *const SFEM_RESTRICT disp,
                        const sresample::sdf::NodeOutput<real_t> &out) {
            sresample::sdf::NodePoints<geom_t> p;
            for (int d = 0; d < 3; ++d) p.xyz[d] = points[d];
            return check(sresample::sdf::sample_nodes(order, nnodes, idx, p, disp, grid(), out, exec_of(context)));
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
#ifdef SMESH_ENABLE_MPI
        if (slabs) {
            geom_t *view = nullptr;
            if (sresample::mpi::zslab_view(comm->get(),
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
                                           &obs->impl_->origin[2]) != sresample::SRESAMPLE_SUCCESS) {
                SFEM_ERROR("SDFObstacle: unable to gather the SDF slab of this rank's points\n");
            }
            spline = SDFBSpline::create(obs->impl_->n, sdf->stride(), view);
            free(view);
        } else
#else
        SFEM_UNUSED(slabs);  // one rank: the grid is whole
        SFEM_UNUSED(nnodes);
        SFEM_UNUSED(z);
#endif
        {
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
        return impl_->surface_sample(element_type, nelements, nnodes, elements, points, gap, normals, false);
    }

    int SDFObstacle::sample_normals(smesh::ElemType              element_type,
                                    const ptrdiff_t              nelements,
                                    const ptrdiff_t              nnodes,
                                    idx_t **const SFEM_RESTRICT  elements,
                                    geom_t **const SFEM_RESTRICT points,
                                    real_t **const SFEM_RESTRICT normals) {
        SFEM_TRACE_SCOPE("SDFObstacle::sample_normals");
        return impl_->surface_sample(element_type, nelements, nnodes, elements, points, nullptr, normals, false);
    }

    int SDFObstacle::sample_normals_accumulate(smesh::ElemType              element_type,
                                               const ptrdiff_t              nelements,
                                               const ptrdiff_t              nnodes,
                                               idx_t **const SFEM_RESTRICT  elements,
                                               geom_t **const SFEM_RESTRICT points,
                                               real_t **const SFEM_RESTRICT normals) {
        SFEM_TRACE_SCOPE("SDFObstacle::sample_normals_accumulate");
        return impl_->surface_sample(element_type, nelements, nnodes, elements, points, nullptr, normals, true);
    }

    int SDFObstacle::sample_value_accumulate(smesh::ElemType              element_type,
                                             const ptrdiff_t              nelements,
                                             const ptrdiff_t              nnodes,
                                             idx_t **const SFEM_RESTRICT  elements,
                                             geom_t **const SFEM_RESTRICT points,
                                             real_t *const SFEM_RESTRICT  gap) {
        SFEM_TRACE_SCOPE("SDFObstacle::sample_value_accumulate");
        return impl_->surface_sample(element_type, nelements, nnodes, elements, points, gap, nullptr, true);
    }

    int SDFObstacle::sample_weights_accumulate(smesh::ElemType              element_type,
                                               const ptrdiff_t              nelements,
                                               const ptrdiff_t              nnodes,
                                               idx_t **const SFEM_RESTRICT  elements,
                                               geom_t **const SFEM_RESTRICT points,
                                               real_t *const SFEM_RESTRICT  w) {
        SFEM_TRACE_SCOPE("SDFObstacle::sample_weights_accumulate");
        return check(sresample::sdf::surface_weights_local(
                mesh_view(element_type, nelements, nnodes, elements, points), w, sresample::Exec::host()));
    }

    int SDFObstacle::sample_value(smesh::ElemType              element_type,
                                  const ptrdiff_t              nelements,
                                  const ptrdiff_t              nnodes,
                                  idx_t **const SFEM_RESTRICT  elements,
                                  geom_t **const SFEM_RESTRICT points,
                                  real_t *const SFEM_RESTRICT  gap) {
        SFEM_TRACE_SCOPE("SDFObstacle::sample_value");
        return impl_->surface_sample(element_type, nelements, nnodes, elements, points, gap, nullptr, false);
    }

    int SDFObstacle::sample_nodes(const ptrdiff_t                   nnodes,
                                  const idx_t *const SFEM_RESTRICT  idx,
                                  geom_t **const SFEM_RESTRICT      points,
                                  const real_t *const SFEM_RESTRICT disp,
                                  real_t *const SFEM_RESTRICT       gap,
                                  real_t **const SFEM_RESTRICT      normals) {
        SFEM_TRACE_SCOPE("SDFObstacle::sample_nodes");
        sresample::sdf::NodeOutput<real_t> out;
        out.gap = gap;
        if (normals) {
            for (int d = 0; d < 3; ++d) out.normals[d] = normals[d];
        }
        return impl_->node_sample(1, nnodes, idx, points, disp, out);
    }

    int SDFObstacle::sample_nodes_curvature(const ptrdiff_t                   nnodes,
                                            const idx_t *const SFEM_RESTRICT  idx,
                                            geom_t **const SFEM_RESTRICT      points,
                                            const real_t *const SFEM_RESTRICT disp,
                                            real_t **const SFEM_RESTRICT      curvature) {
        SFEM_TRACE_SCOPE("SDFObstacle::sample_nodes_curvature");
        sresample::sdf::NodeOutput<real_t> out;
        for (int c = 0; c < 6; ++c) out.curvature[c] = curvature[c];
        return impl_->node_sample(2, nnodes, idx, points, disp, out);
    }
}  // namespace sfem
