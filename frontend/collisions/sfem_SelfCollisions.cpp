#include "sfem_SelfCollisions.hpp"

#include "broadphase.hpp"

#include "smesh_graph.hpp"
#include "smesh_mesh.hpp"

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
            for (int i = 0; i < n_nodes; i++) {
                vidx->data()[i] = i;
            }

            fidx = smesh::create_host_buffer<smesh::idx_t>(n_faces);
            for (int i = 0; i < n_faces; i++) {
                fidx->data()[i] = i;
            }

            eidx = smesh::create_host_buffer<smesh::idx_t>(n_edges);
            for (int i = 0; i < n_edges; i++) {
                eidx->data()[i] = i;
            }
        }
    };

    struct Edges {
        smesh::SharedBuffer<smesh::idx_t> v0;
        smesh::SharedBuffer<smesh::idx_t> v1;
    };

    struct CollisionPairs {
        smesh::SharedBuffer<smesh::idx_t> i0;
        smesh::SharedBuffer<smesh::idx_t> i1;
    };

    class SelfCollisions::Impl {
    public:
        Impl()  = default;
        ~Impl() = default;

        std::shared_ptr<smesh::Mesh> surface;

        AABB aabb_nodes;
        AABB aabb_faces;
        AABB aabb_edges;

        Edges             edges;
        SweepAndPruneData sweep_and_prune_data;

        CollisionPairs vf;
        CollisionPairs ee;

        void init(const std::shared_ptr<smesh::Mesh>& surface) {
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
        }

        void compute_aabbs(const ptrdiff_t                          stride_displacement,  // 2 or 3 for AoS, 1 for SoA
                           std::vector<smesh::SharedBuffer<real_t>> displacement0,
                           std::vector<smesh::SharedBuffer<real_t>> displacement1) {
            const int       dim     = surface->spatial_dimension();
            const ptrdiff_t n_nodes = surface->n_nodes();

            real_t* _disp0[3] = {
                    displacement0[0]->data(), displacement0[1]->data(), (dim > 2) ? displacement0[2]->data() : nullptr};

            real_t* _disp1[3] = {
                    displacement1[0]->data(), displacement1[1]->data(), (dim > 2) ? displacement1[2]->data() : nullptr};

            sccd::compute_aabbs(dim,
                                n_nodes,
                                surface->points()->data(),
                                stride_displacement,
                                _disp0,
                                _disp1,
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
                                    stride_displacement,
                                    _disp0,
                                    _disp1,
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
                                stride_displacement,
                                _disp0,
                                _disp1,
                                aabb_edges.min->data(),
                                aabb_edges.max->data());
        }

        void find_collisions() {}
    };

    SelfCollisions::SelfCollisions() : impl_(std::make_unique<Impl>()) {}

    SelfCollisions::~SelfCollisions() = default;

}  // namespace sfem
