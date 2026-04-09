#include "sfem_SelfCollisions.hpp"

#include "broadphase.hpp"
#include "narrowphase.hpp"

#include "smesh_graph.hpp"
#include "smesh_mesh.hpp"
#include "smesh_tracer.hpp"

#include "sfem_API.hpp"
#include "sfem_OpFactory.hpp"

#include "ssdf.hpp"

namespace sfem {

    struct AABB {
        smesh::SharedBuffer<smesh::geom_t*> min;
        smesh::SharedBuffer<smesh::geom_t*> max;
    };

    struct SweepAndPruneData {
        smesh::SharedBuffer<smesh::geom_t> scratch;
        smesh::SharedBuffer<ptrdiff_t>     ccdptr;

        smesh::SharedBuffer<smesh::idx_t> vidx;
        smesh::SharedBuffer<smesh::idx_t> fidx;
        smesh::SharedBuffer<smesh::idx_t> eidx;

        void init(const std::shared_ptr<smesh::Mesh>& surface) {
            const int       dim     = surface->spatial_dimension();
            const ptrdiff_t n_nodes = surface->n_nodes();
            const ptrdiff_t n_faces = surface->n_elements();
            const ptrdiff_t n_edges = surface->edge_graph()->nnz();

            scratch = smesh::create_host_buffer<smesh::geom_t>(std::max(n_nodes, std::max(n_faces, n_edges)));
            ccdptr  = smesh::create_host_buffer<ptrdiff_t>(std::max(n_faces, n_edges) + 1);

            vidx = smesh::create_host_buffer<smesh::idx_t>(n_nodes);
            fidx = smesh::create_host_buffer<smesh::idx_t>(n_faces);
            eidx = smesh::create_host_buffer<smesh::idx_t>(n_edges);
        }
    };

    class SelfCollisions::Impl {
    public:
        Impl()  = default;
        ~Impl() = default;

        std::shared_ptr<smesh::Mesh>        surface;
        smesh::SharedBuffer<smesh::real_t*> p0, p1;

        // Search data-structures
        AABB aabb_nodes;
        AABB aabb_faces;
        AABB aabb_edges;

        Edges             edges;
        SweepAndPruneData sweep_and_prune_data;

        // Collisions
        CollisionPairs vertex_to_face;
        CollisionPairs edge_to_edge;

        // To be called at the beginning of the simulation
        void init(const std::shared_ptr<smesh::Mesh>& surface) {
            SMESH_TRACE_SCOPE("SelfCollisions::Impl::init");
            this->surface           = surface;
            const int       dim     = surface->spatial_dimension();
            const ptrdiff_t n_nodes = surface->n_nodes();

            aabb_nodes.min = smesh::create_host_buffer<smesh::geom_t>(dim, n_nodes);
            aabb_nodes.max = smesh::create_host_buffer<smesh::geom_t>(dim, n_nodes);

            aabb_faces.min = smesh::create_host_buffer<smesh::geom_t>(dim, surface->n_elements());
            aabb_faces.max = smesh::create_host_buffer<smesh::geom_t>(dim, surface->n_elements());

            auto n2n_crs = surface->edge_graph();
            auto row_idx = smesh::create_host_buffer<smesh::idx_t>(n2n_crs->nnz());
            smesh::crs_to_coo(surface->n_nodes(), n2n_crs->rowptr()->data(), row_idx->data());

            edges.v0 = row_idx;
            edges.v1 = n2n_crs->colidx();

            aabb_edges.min = smesh::create_host_buffer<smesh::geom_t>(dim, n2n_crs->nnz());
            aabb_edges.max = smesh::create_host_buffer<smesh::geom_t>(dim, n2n_crs->nnz());

            sweep_and_prune_data.init(surface);

            p0 = smesh::create_host_buffer<smesh::real_t>(dim, n_nodes);
            p1 = smesh::create_host_buffer<smesh::real_t>(dim, n_nodes);
        }

        void displace_points(const ptrdiff_t stride_displacement,  // 2 or 3 for AoS, 1 for SoA
                             const real_t* const SFEM_RESTRICT* const SFEM_RESTRICT displacement,
                             real_t* const SFEM_RESTRICT* const SFEM_RESTRICT       points) {
            const int       dim     = surface->spatial_dimension();
            const ptrdiff_t n_nodes = surface->n_nodes();

            auto node_mapping = surface->node_mapping()->data();

            for (int d = 0; d < dim; d++) {
                const auto* const    x   = surface->points()->data()[d];
                smesh::real_t* const x_s = points[d];

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_nodes; ++i) {
                    const ptrdiff_t idx = node_mapping[i];
                    x_s[i]              = x[i] + displacement[d][idx * stride_displacement];
                }
            }
        }

        void compute_displaced_points(const ptrdiff_t stride_displacement,  // 2 or 3 for AoS, 1 for SoA
                                      const real_t* const SFEM_RESTRICT* const SFEM_RESTRICT displacement0,
                                      const real_t* const SFEM_RESTRICT* const SFEM_RESTRICT displacement1) {
            SMESH_TRACE_SCOPE("SelfCollisions::Impl::compute_displaced_points");

            displace_points(stride_displacement, displacement0, p0->data());
            displace_points(stride_displacement, displacement1, p1->data());
        }

        // To be called before finding collisions
        void compute_aabbs(const ptrdiff_t stride_displacement,  // 2 or 3 for AoS, 1 for SoA
                           const real_t* const SFEM_RESTRICT* const SFEM_RESTRICT displacement0,
                           const real_t* const SFEM_RESTRICT* const SFEM_RESTRICT displacement1) {
            SMESH_TRACE_SCOPE("SelfCollisions::Impl::compute_aabbs");

            const int       dim     = surface->spatial_dimension();
            const ptrdiff_t n_nodes = surface->n_nodes();

            compute_displaced_points(stride_displacement, displacement0, displacement1);

            sccd::compute_aabbs(dim, n_nodes, p0->data(), p1->data(), aabb_nodes.min->data(), aabb_nodes.max->data());

            ptrdiff_t element_offset = 0;
            for (auto b : surface->blocks()) {
                int             nxe        = b->n_nodes_per_element();
                const ptrdiff_t n_elements = b->n_elements();

                auto amin = smesh::view(aabb_faces.min, 0, dim, element_offset, element_offset + n_elements);
                auto amax = smesh::view(aabb_faces.max, 0, dim, element_offset, element_offset + n_elements);

                sccd::compute_aabbs(
                        nxe, n_elements, b->elements()->data(), dim, p0->data(), p1->data(), amin->data(), amax->data());

                element_offset += n_elements;
            }

            smesh::idx_t* edges_raw[2] = {edges.v0->data(), edges.v1->data()};
            sccd::compute_aabbs(2,
                                edges.v0->size(),
                                edges_raw,
                                dim,
                                surface->points()->data(),
                                1,
                                p0->data(),
                                p1->data(),
                                aabb_edges.min->data(),
                                aabb_edges.max->data());
        }

        void find_collisions() {
            SMESH_TRACE_SCOPE("SelfCollisions::Impl::find_collisions");

            if (surface->n_blocks() > 1) {
                SFEM_ERROR("Not implemented");
                return;
            }

            smesh::geom_t* vaabb[6] = {aabb_nodes.min->data()[0],
                                       aabb_nodes.min->data()[1],
                                       aabb_nodes.min->data()[2],
                                       aabb_nodes.max->data()[0],
                                       aabb_nodes.max->data()[1],
                                       aabb_nodes.max->data()[2]};

            smesh::geom_t* faabb[6] = {aabb_faces.min->data()[0],
                                       aabb_faces.min->data()[1],
                                       aabb_faces.min->data()[2],
                                       aabb_faces.max->data()[0],
                                       aabb_faces.max->data()[1],
                                       aabb_faces.max->data()[2]};

            smesh::geom_t* eaabb[6] = {aabb_edges.min->data()[0],
                                       aabb_edges.min->data()[1],
                                       aabb_edges.min->data()[2],
                                       aabb_edges.max->data()[0],
                                       aabb_edges.max->data()[1],
                                       aabb_edges.max->data()[2]};

            const ptrdiff_t n_nodes = surface->n_nodes();
            const ptrdiff_t n_faces = surface->n_elements();
            const ptrdiff_t n_edges = edges.v0->size();

            int sort_axis = sccd::choose_axis(n_nodes, vaabb);

            auto vidx    = sweep_and_prune_data.vidx;
            auto fidx    = sweep_and_prune_data.fidx;
            auto eidx    = sweep_and_prune_data.eidx;
            auto scratch = sweep_and_prune_data.scratch;
            auto ccdptr  = sweep_and_prune_data.ccdptr;

            for (int i = 0; i < n_nodes; i++) {
                vidx->data()[i] = i;
            }

            for (int i = 0; i < n_faces; i++) {
                fidx->data()[i] = i;
            }

            for (int i = 0; i < n_edges; i++) {
                eidx->data()[i] = i;
            }

            sccd::sort_along_axis(n_nodes, sort_axis, vaabb, vidx->data(), scratch->data());
            sccd::sort_along_axis(n_faces, sort_axis, faabb, fidx->data(), scratch->data());
            sccd::sort_along_axis(n_edges, sort_axis, eaabb, eidx->data(), scratch->data());

            {
                SMESH_TRACE_SCOPE("Broadphase: E2E");

                smesh::idx_t* edges_raw[2] = {edges.v0->data(), edges.v1->data()};
                sccd::count_self_overlaps<2>(sort_axis, n_edges, eaabb, eidx->data(), 1, edges_raw, ccdptr->data());

                // FIXME: Mandatory allocation, see if buffer could be reused
                edge_to_edge.first  = smesh::create_host_buffer<smesh::idx_t>(ccdptr->data()[n_edges]);
                edge_to_edge.second = smesh::create_host_buffer<smesh::idx_t>(ccdptr->data()[n_edges]);

                sccd::collect_self_overlaps<2>(sort_axis,
                                               n_edges,
                                               eaabb,
                                               eidx->data(),
                                               1,
                                               edges_raw,
                                               ccdptr->data(),
                                               edge_to_edge.first->data(),
                                               edge_to_edge.second->data());
            }

            {
                SMESH_TRACE_SCOPE("Broadphase: F2V");

                sccd::count_overlaps<3, 1, smesh::geom_t, smesh::idx_t>(sort_axis,
                                                                        n_faces,
                                                                        faabb,
                                                                        fidx->data(),
                                                                        1,
                                                                        surface->elements(0)->data(),
                                                                        n_nodes,
                                                                        vaabb,
                                                                        vidx->data(),
                                                                        0,
                                                                        nullptr,
                                                                        ccdptr->data());

                vertex_to_face.first  = smesh::create_host_buffer<smesh::idx_t>(ccdptr->data()[n_faces]);
                vertex_to_face.second = smesh::create_host_buffer<smesh::idx_t>(ccdptr->data()[n_faces]);

                sccd::collect_overlaps<3, 1, smesh::geom_t, smesh::idx_t>(sort_axis,
                                                                          n_faces,
                                                                          faabb,
                                                                          fidx->data(),
                                                                          1,
                                                                          surface->elements(0)->data(),
                                                                          n_nodes,
                                                                          vaabb,
                                                                          vidx->data(),
                                                                          0,
                                                                          nullptr,
                                                                          ccdptr->data(),
                                                                          vertex_to_face.second->data(),
                                                                          vertex_to_face.first->data());
            }

            printf("V2F %zu %zu\n", vertex_to_face.first->size(), vertex_to_face.second->size());
            printf("E2E %zu %zu\n", edge_to_edge.first->size(), edge_to_edge.second->size());
        }

        real_t time_of_impact() {
            SMESH_TRACE_SCOPE("SelfCollisions::Impl::time_of_impact");

            // Narrow phase
            smesh::real_t toi = std::numeric_limits<smesh::real_t>::max();
            smesh::real_t toi_vf, toi_ee;

            {
                SMESH_TRACE_SCOPE("Narrow phase: F2V");
                toi_vf = sccd::narrow_phase_vf<3, smesh::real_t>(vertex_to_face.first->size(),
                                                                 vertex_to_face.first->data(),
                                                                 vertex_to_face.second->data(),
                                                                 p0->data(),
                                                                 p1->data(),
                                                                 1,
                                                                 surface->elements(0)->data(),
                                                                 toi);
                toi    = toi_vf;
            }

            {
                SMESH_TRACE_SCOPE("Narrow phase: E2E");

                smesh::idx_t* edges_raw[2] = {edges.v0->data(), edges.v1->data()};
                toi_ee                     = sccd::narrow_phase_ee<smesh::real_t>(edge_to_edge.first->size(),
                                                              edge_to_edge.first->data(),
                                                              edge_to_edge.second->data(),
                                                              p0->data(),
                                                              p1->data(),
                                                              1,
                                                              edges_raw,
                                                              toi);
                toi                        = toi_ee;
            }

            return toi;
        }

        void discrete_detection_with_side_effects(const real_t toi) {
            int             dim     = surface->spatial_dimension();
            const ptrdiff_t n_nodes = surface->n_nodes();

            auto points0 = p0->data();
            auto points1 = p1->data();

            for (int d = 0; d < dim; d++) {
                for (ptrdiff_t i = 0; i < n_nodes; i++) {
                    const geom_t p0 = points0[d][i];
                    const geom_t p1 = points1[d][i];

                    points0[d][i] = (1 - toi) * p0 + toi * p1;

                    const real_t offset = toi + 1e-1;
                    points1[d][i]       = (1 - offset) * p0 + offset * p1;
                }
            }

            sccd::compute_aabbs(dim, n_nodes, points0, points1, aabb_nodes.min->data(), aabb_nodes.max->data());

            ptrdiff_t element_offset = 0;
            for (auto b : surface->blocks()) {
                int             nxe        = b->n_nodes_per_element();
                const ptrdiff_t n_elements = b->n_elements();

                auto amin = smesh::view(aabb_faces.min, 0, dim, element_offset, element_offset + n_elements);
                auto amax = smesh::view(aabb_faces.max, 0, dim, element_offset, element_offset + n_elements);

                sccd::compute_aabbs(nxe, n_elements, b->elements()->data(), dim, points0, points1, amin->data(), amax->data());

                element_offset += n_elements;
            }

            smesh::idx_t* edges_raw[2] = {edges.v0->data(), edges.v1->data()};
            sccd::compute_aabbs(
                    2, edges.v0->size(), edges_raw, dim, points0, points1, aabb_edges.min->data(), aabb_edges.max->data());

            find_collisions();
        }

        void distance_and_normal(const real_t                                     toi,
                                 real_t* const SFEM_RESTRICT                      d,
                                 const ptrdiff_t                                  stride_normal,
                                 real_t* const SFEM_RESTRICT* const SFEM_RESTRICT n) {
            SMESH_TRACE_SCOPE("SelfCollisions::Impl::distance_and_normal");

            int             dim     = surface->spatial_dimension();
            const ptrdiff_t n_nodes = surface->n_nodes();

            discrete_detection_with_side_effects(toi);

            real_t infty = 10000;

#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n_nodes; i++) {
                d[i] = infty;
            }

            // VF

            auto points0 = p0->data();

            auto x = points0[0];
            auto y = points0[1];
            auto z = points0[2];

            if (surface->n_blocks() > 1) {
                SFEM_ERROR("Not implemented");
                return;
            }

            const ptrdiff_t n_vf_collisions = vertex_to_face.first->size();
            for (ptrdiff_t i = 0; i < n_vf_collisions; i++) {
                const ptrdiff_t vidx = vertex_to_face.first->data()[i];
                const ptrdiff_t fidx = vertex_to_face.second->data()[i];

                const auto i0 = surface->elements(0)->data()[0][fidx];
                const auto i1 = surface->elements(0)->data()[1][fidx];
                const auto i2 = surface->elements(0)->data()[2][fidx];

                real_t nx, ny, nz;
                // Compute normal of the face
                ssdf::triangle_area_weighted_normal<smesh::real_t>(
                        x[i0], y[i0], z[i0], x[i1], y[i1], z[i1], x[i2], y[i2], z[i2], &nx, &ny, &nz);

                real_t cx, cy, cz;
                ssdf::point_triangle_closest_point<smesh::real_t>(
                        x[vidx], y[vidx], z[vidx], x[i0], y[i0], z[i0], x[i1], y[i1], z[i1], x[i2], y[i2], z[i2], &cx, &cy, &cz);

                const real_t dx = x[vidx] - cx;
                const real_t dy = y[vidx] - cy;
                const real_t dz = z[vidx] - cz;

                const real_t d2  = dx * dx + dy * dy + dz * dz;
                const real_t dot = dx * nx + dy * ny + dz * nz;

                // Remove points past the face, unless they are on the face (this works only if the simulation is penetration
                // free)
                if (dot < 1e-8 && d2 > 1e-6 || d[vidx] < d2) {
                    continue;
                }

                // printf("%d -> %d %d %d (%g, %g)\n", vidx, i0, i1, i2, dot, d2);

                d[vidx]    = std::min(d[vidx], d2);
                n[0][vidx] = -dx;
                n[1][vidx] = -dy;
                n[2][vidx] = -dz;
            }

#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n_nodes; i++) {
                d[i]    = std::sqrt(d[i]);
                n[0][i] = n[0][i] / d[i];
                n[1][i] = n[1][i] / d[i];
                n[2][i] = n[2][i] / d[i];
            }
        }
    };

    SelfCollisions::SelfCollisions() : impl_(std::make_unique<Impl>()) {}

    SelfCollisions::~SelfCollisions() = default;

    std::shared_ptr<SelfCollisions> SelfCollisions::create(const std::shared_ptr<smesh::Mesh>& surface) {
        auto self_collisions = std::make_shared<SelfCollisions>();
        self_collisions->impl_->init(surface);
        return self_collisions;
    }

    void SelfCollisions::find(const ptrdiff_t stride_displacement,  // 2 or 3 for AoS, 1 for SoA
                              const real_t* const SFEM_RESTRICT* const SFEM_RESTRICT displacement0,
                              const real_t* const SFEM_RESTRICT* const SFEM_RESTRICT displacement1) {
        impl_->compute_aabbs(stride_displacement, displacement0, displacement1);
        impl_->find_collisions();
    }

    const CollisionPairs& SelfCollisions::vertex_to_face() const { return impl_->vertex_to_face; }
    const CollisionPairs& SelfCollisions::edge_to_edge() const { return impl_->edge_to_edge; }
    const Edges&          SelfCollisions::edges() const { return impl_->edges; }

    std::shared_ptr<smesh::Mesh> SelfCollisions::surface() const { return impl_->surface; }

    real_t SelfCollisions::time_of_impact() { return impl_->time_of_impact(); }

    void SelfCollisions::discrete_detection_with_side_effects(const real_t toi) {
        impl_->discrete_detection_with_side_effects(toi);
    }

    void SelfCollisions::distance_and_normal(const real_t                                     toi,
                                             real_t* const SFEM_RESTRICT                      d,
                                             const ptrdiff_t                                  stride_normal,
                                             real_t* const SFEM_RESTRICT* const SFEM_RESTRICT n) {
        impl_->distance_and_normal(toi, d, stride_normal, n);
    }

    SharedBuffer<real_t*> SelfCollisions::points0() const { return impl_->p0; }
    SharedBuffer<real_t*> SelfCollisions::points1() const { return impl_->p1; }

    //
    //-----------------
    //

    class SelfContactPenalty::Impl {
    public:
        Impl()  = default;
        ~Impl() = default;

        std::shared_ptr<FunctionSpace>  space;
        std::shared_ptr<smesh::Mesh>    surface;
        std::shared_ptr<SelfCollisions> collisions;

        real_t               penalty_offset{1e-3};
        real_t               penalty_weight{1000};
        SharedBuffer<real_t> vf_penalty_gradient;
        SharedBuffer<real_t> ee_penalty_gradient;
        real_t               toi{1};

        SharedBuffer<real_t> mass_vector;
    };

    SelfContactPenalty::SelfContactPenalty() : impl_(std::make_unique<Impl>()) {}

    SelfContactPenalty::~SelfContactPenalty() = default;

    std::shared_ptr<SelfContactPenalty> SelfContactPenalty::create(const std::shared_ptr<FunctionSpace>& space) {
        auto surface = smesh::skin(space->mesh_ptr());
        return create(space, surface);
    }

    std::shared_ptr<SelfContactPenalty> SelfContactPenalty::create(const std::shared_ptr<FunctionSpace>& space,
                                                                   const std::shared_ptr<smesh::Mesh>&   surface) {
        auto ret               = std::make_shared<SelfContactPenalty>();
        ret->impl_->space      = space;
        ret->impl_->surface    = surface;
        ret->impl_->collisions = SelfCollisions::create(surface);

        auto trace_space = FunctionSpace::create(surface, 1);
        auto mass        = Factory::create_op(trace_space, "Mass");
        mass->initialize();

        auto m    = create_host_buffer<real_t>(trace_space->n_dofs());
        auto ones = create_host_buffer<real_t>(trace_space->n_dofs());
        sfem::blas<real_t>(EXECUTION_SPACE_HOST)->values(trace_space->n_dofs(), 1, ones->data());
        mass->apply(nullptr, ones->data(), m->data());
        ret->impl_->mass_vector = m;

        return ret;
    }

    const char* SelfContactPenalty::name() const { return "SelfContactPenalty"; }

    bool SelfContactPenalty::is_linear() const { return false; }

    int SelfContactPenalty::hessian_crs(const real_t* const, const count_t* const, const idx_t* const, real_t* const) {
        SFEM_ERROR("SelfContactPenalty::hessian_crs not implemented\n");
        return SFEM_FAILURE;
    }

    int SelfContactPenalty::gradient(const real_t* const /*x*/, real_t* const g) {
        auto vf_penalty_gradient = impl_->vf_penalty_gradient->data();
        auto mass_vector         = impl_->mass_vector->data();
        auto mass_vector_size    = impl_->mass_vector->size();
        auto s2v                 = impl_->surface->node_mapping()->data();

        const ptrdiff_t n_nodes = impl_->surface->n_nodes();
        const int       dim     = impl_->surface->spatial_dimension();

        for (ptrdiff_t i = 0; i < n_nodes; i++) {
            const ptrdiff_t idx = s2v[i] * dim;
            const auto      m   = mass_vector[i];
            for (int d = 0; d < dim; d++) {
                // g[idx + d] += vf_penalty_gradient[i * dim + d] * m;
                g[idx + d] += vf_penalty_gradient[i * dim + d];  // FIXME
            }
        }

        // auto& edges = impl_->collisions->edges();
        // auto  ev0   = edges.v0->data();
        // auto  ev1   = edges.v1->data();

        // const ptrdiff_t n_edges = edges.v0->size();
        // for (ptrdiff_t i = 0; i < n_edges; i++) {
        //     const ptrdiff_t e0 = edges.v0->data()[i];
        //     const ptrdiff_t e1 = edges.v1->data()[i];

        //     const auto v0 = ev0[e0];
        //     const auto v1 = ev1[e0];
        // }

        return SFEM_SUCCESS;
    }

    int SelfContactPenalty::apply(const real_t* const, const real_t* const, real_t* const) {
        SFEM_ERROR("SelfContactPenalty::apply not implemented\n");
        return SFEM_FAILURE;
    }

    int SelfContactPenalty::value(const real_t*, real_t* const) {
        SFEM_ERROR("SelfContactPenalty::value not implemented\n");
        return SFEM_FAILURE;
    }

    ptrdiff_t SelfContactPenalty::n_dofs_domain() const { return impl_->space->n_dofs(); }
    ptrdiff_t SelfContactPenalty::n_dofs_image() const { return impl_->space->n_dofs(); }

    real_t SelfContactPenalty::max_step_size() { return impl_->toi; }

    int SelfContactPenalty::update(const real_t* const SFEM_RESTRICT x_prev, const real_t* const SFEM_RESTRICT x_curr) {
        const real_t* x_prev3[3] = {&x_prev[0], &x_prev[1], &x_prev[2]};
        const real_t* x_curr3[3] = {&x_curr[0], &x_curr[1], &x_curr[2]};

        impl_->collisions->find(impl_->space->mesh_ptr()->spatial_dimension(), x_prev3, x_curr3);
        impl_->toi = impl_->collisions->time_of_impact();

        impl_->collisions->discrete_detection_with_side_effects(impl_->toi);

        auto& vertex_to_face = impl_->collisions->vertex_to_face();
        auto& edge_to_edge   = impl_->collisions->edge_to_edge();

        // This is the displaced points after the discrete detection with side effects
        auto points = impl_->collisions->points0()->data();
        auto x      = points[0];
        auto y      = points[1];
        auto z      = points[2];

        auto         elements               = impl_->surface->elements(0)->data();
        const real_t penalty_weight         = impl_->penalty_weight;
        const real_t penalty_offset         = impl_->penalty_offset;
        const real_t penalty_offset_squared = penalty_offset * penalty_offset;

        if (!impl_->vf_penalty_gradient) {
            impl_->vf_penalty_gradient = create_host_buffer<real_t>(impl_->surface->n_nodes() * 3);
        }

        auto vf_penalty_gradient = impl_->vf_penalty_gradient->data();
        std::memset(vf_penalty_gradient, 0, impl_->vf_penalty_gradient->size() * sizeof(real_t));

        const ptrdiff_t n_vf_collisions = vertex_to_face.first->size();
        for (ptrdiff_t i = 0; i < n_vf_collisions; i++) {
            const ptrdiff_t vidx = vertex_to_face.first->data()[i];
            const ptrdiff_t fidx = vertex_to_face.second->data()[i];

            const auto i0 = elements[0][fidx];
            const auto i1 = elements[1][fidx];
            const auto i2 = elements[2][fidx];

            real_t nx, ny, nz;
            // Compute normal of the face
            ssdf::triangle_area_weighted_normal<smesh::real_t>(
                    x[i0], y[i0], z[i0], x[i1], y[i1], z[i1], x[i2], y[i2], z[i2], &nx, &ny, &nz);

            // real_t cx, cy, cz;
            // ssdf::point_triangle_closest_point<smesh::real_t>(
            //         x[vidx], y[vidx], z[vidx], x[i0], y[i0], z[i0], x[i1], y[i1], z[i1], x[i2], y[i2], z[i2], &cx, &cy, &cz);

            real_t w1, w2;
            ssdf::point_triangle_closest_point_param<smesh::real_t>(
                    x[vidx], y[vidx], z[vidx], x[i0], y[i0], z[i0], x[i1], y[i1], z[i1], x[i2], y[i2], z[i2], &w1, &w2);

            const real_t w0 = 1 - w1 - w2;

            const real_t cx = w0 * x[i0] + w1 * x[i1] + w2 * x[i2];
            const real_t cy = w0 * y[i0] + w1 * y[i1] + w2 * y[i2];
            const real_t cz = w0 * z[i0] + w1 * z[i1] + w2 * z[i2];

            const real_t dx = x[vidx] - cx;
            const real_t dy = y[vidx] - cy;
            const real_t dz = z[vidx] - cz;

            const real_t d2  = dx * dx + dy * dy + dz * dz;
            const real_t dot = dx * nx + dy * ny + dz * nz;

            if (std::abs(dot) < 1e-8 && d2 > 1e-6 || penalty_offset_squared <= d2 || w1 < 0 || w1 < 0 || w0 + w1 > 1) {
                continue;
            }

            const real_t d = std::sqrt(d2);
            const real_t w = penalty_weight * (d - penalty_offset) / penalty_offset;

            vf_penalty_gradient[vidx * 3 + 0] -= w * nx;
            vf_penalty_gradient[vidx * 3 + 1] -= w * ny;
            vf_penalty_gradient[vidx * 3 + 2] -= w * nz;

            vf_penalty_gradient[i0 * 3 + 0] += w0 * w * nx;
            vf_penalty_gradient[i0 * 3 + 1] += w0 * w * ny;
            vf_penalty_gradient[i0 * 3 + 2] += w0 * w * nz;

            vf_penalty_gradient[i1 * 3 + 0] += w1 * w * nx;
            vf_penalty_gradient[i1 * 3 + 1] += w1 * w * ny;
            vf_penalty_gradient[i1 * 3 + 2] += w1 * w * nz;

            vf_penalty_gradient[i2 * 3 + 0] += w2 * w * nx;
            vf_penalty_gradient[i2 * 3 + 1] += w2 * w * ny;
            vf_penalty_gradient[i2 * 3 + 2] += w2 * w * nz;
        }

        auto& edges = impl_->collisions->edges();
        auto  ev0   = edges.v0->data();
        auto  ev1   = edges.v1->data();

        if (!impl_->ee_penalty_gradient) {
            impl_->ee_penalty_gradient = create_host_buffer<real_t>(impl_->surface->edge_graph()->nnz() * 3);
        }

        // auto ee_penalty_gradient = impl_->ee_penalty_gradient->data();
        // std::memset(ee_penalty_gradient, 0, impl_->ee_penalty_gradient->size() * sizeof(real_t));

        const ptrdiff_t n_ee_collisions = edge_to_edge.first->size();
        for (ptrdiff_t i = 0; i < n_ee_collisions; i++) {
            const ptrdiff_t e0 = edge_to_edge.first->data()[i];
            const ptrdiff_t e1 = edge_to_edge.second->data()[i];

            const auto v0 = ev0[e0];
            const auto v1 = ev1[e0];
            const auto v2 = ev0[e1];
            const auto v3 = ev1[e1];

            real_t s0, s1;
            ssdf::edge_to_edge_closest_points<smesh::real_t>(
                    x[v0], y[v0], z[v0], x[v1], y[v1], z[v1], x[v2], y[v2], z[v2], x[v3], y[v3], z[v3], &s0, &s1);

            real_t c0x, c0y, c0z;
            real_t c1x, c1y, c1z;
            c0x = (1 - s0) * x[v0] + s0 * x[v1];
            c0y = (1 - s0) * y[v0] + s0 * y[v1];
            c0z = (1 - s0) * z[v0] + s0 * z[v1];

            c1x = (1 - s1) * x[v2] + s1 * x[v3];
            c1y = (1 - s1) * y[v2] + s1 * y[v3];
            c1z = (1 - s1) * z[v2] + s1 * z[v3];

            const real_t dx = c0x - c1x;
            const real_t dy = c0y - c1y;
            const real_t dz = c0z - c1z;

            const real_t d2 = dx * dx + dy * dy + dz * dz;

            if (penalty_offset_squared <= d2 || s0 < 0 || s0 > 1 || s1 < 0 || s1 > 1) {
                continue;
            }

            const real_t d = std::sqrt(d2);
            const real_t w = penalty_weight * (d - penalty_offset) / penalty_offset;

            real_t nx = dx / d, ny = dy / d, nz = dz / d;

            // FIXME?
            //     vf_penalty_gradient[v0 * 3 + 0] -= w * nx;
            //     vf_penalty_gradient[v0 * 3 + 1] -= w * ny;
            //     vf_penalty_gradient[v0 * 3 + 2] -= w * nz;

            //     vf_penalty_gradient[v1 * 3 + 0] -= w * nx;
            //     vf_penalty_gradient[v1 * 3 + 1] -= w * ny;
            //     vf_penalty_gradient[v1 * 3 + 2] -= w * nz;

            //     vf_penalty_gradient[v2 * 3 + 0] += w * nx;
            //     vf_penalty_gradient[v2 * 3 + 1] += w * ny;
            //     vf_penalty_gradient[v2 * 3 + 2] += w * nz;

            //     vf_penalty_gradient[v3 * 3 + 0] += w * nx;
            //     vf_penalty_gradient[v3 * 3 + 1] += w * ny;
            //     vf_penalty_gradient[v3 * 3 + 2] += w * nz;
        }

        return SFEM_SUCCESS;
    }

}  // namespace sfem

// TODOs:
// Create Hessian op (Local contact Hessian and (4x3 x 4x3) and 4 indices)
// On top of the IPC idea I would like to evaluate additional linearizations with fixed geometric linearization.
// - Distance gradients and hessians are precoputed and stored
// - Penalty function is evaluated multiple times
