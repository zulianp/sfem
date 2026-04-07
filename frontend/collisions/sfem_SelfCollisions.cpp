#include "sfem_SelfCollisions.hpp"

#include "broadphase.hpp"
#include "narrowphase.hpp"

#include "smesh_graph.hpp"
#include "smesh_mesh.hpp"
#include "smesh_tracer.hpp"

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
            const ptrdiff_t n_edges = surface->node_to_node_graph()->nnz();

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

            auto n2n_crs = surface->node_to_node_graph();
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

        void compute_displaced_points(const ptrdiff_t stride_displacement,  // 2 or 3 for AoS, 1 for SoA
                                      const real_t* const SFEM_RESTRICT* const SFEM_RESTRICT displacement0,
                                      const real_t* const SFEM_RESTRICT* const SFEM_RESTRICT displacement1) {
            SMESH_TRACE_SCOPE("SelfCollisions::Impl::compute_displaced_points");

            const int       dim     = surface->spatial_dimension();
            const ptrdiff_t n_nodes = surface->n_nodes();

            auto node_mapping = surface->node_mapping()->data();

            for (int d = 0; d < dim; d++) {
                const auto* const    x   = surface->points()->data()[d];
                smesh::real_t* const x_s = p0->data()[d];

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_nodes; ++i) {
                    const ptrdiff_t idx = node_mapping[i];
                    x_s[i]              = x[i] + displacement0[d][idx * stride_displacement];
                }
            }

            for (int d = 0; d < dim; d++) {
                const auto* const    x   = surface->points()->data()[d];
                smesh::real_t* const x_s = p1->data()[d];

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_nodes; ++i) {
                    const ptrdiff_t idx = node_mapping[i];
                    x_s[i]              = x[i] + displacement1[d][idx * stride_displacement];
                }
            }
        }

        // To be called before finding collisions
        void compute_aabbs(const ptrdiff_t stride_displacement,  // 2 or 3 for AoS, 1 for SoA
                           const real_t* const SFEM_RESTRICT* const SFEM_RESTRICT displacement0,
                           const real_t* const SFEM_RESTRICT* const SFEM_RESTRICT displacement1) {
            SMESH_TRACE_SCOPE("SelfCollisions::Impl::compute_aabbs");

            const int       dim     = surface->spatial_dimension();
            const ptrdiff_t n_nodes = surface->n_nodes();

            compute_displaced_points(stride_displacement, displacement0, displacement1);

            sccd::compute_aabbs(dim,
                                n_nodes,
                                surface->points()->data(),
                                1,
                                p0->data(),
                                p1->data(),
                                aabb_nodes.min->data(),
                                aabb_nodes.max->data());

            ptrdiff_t element_offset = 0;
            for (auto b : surface->blocks()) {
                int             nxe        = b->n_nodes_per_element();
                const ptrdiff_t n_elements = b->n_elements();

                auto amin = smesh::view(aabb_faces.min, 0, dim, element_offset, element_offset + n_elements);
                auto amax = smesh::view(aabb_faces.max, 0, dim, element_offset, element_offset + n_elements);

                sccd::compute_aabbs(nxe,
                                    n_elements,
                                    b->elements()->data(),
                                    dim,
                                    surface->points()->data(),
                                    1,
                                    p0->data(),
                                    p1->data(),
                                    amin->data(),
                                    amax->data());

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
        }

        real_t time_of_impact() {
            SMESH_TRACE_SCOPE("SelfCollisions::Impl::time_of_impact");

            // Narrow phase
            smesh::real_t toi = std::numeric_limits<smesh::real_t>::max();
            smesh::real_t toi_vf, toi_ee;

            printf("V2F %zu %zu\n", vertex_to_face.first->size(), vertex_to_face.second->size());
            printf("E2E %zu %zu\n", edge_to_edge.first->size(), edge_to_edge.second->size());

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

}  // namespace sfem
