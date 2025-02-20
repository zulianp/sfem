#include "sfem_Mesh.hpp"

// C includes
#include "adj_table.h"
#include "crs_graph.h"
#include "read_mesh.h"
#include "sfem_mesh.h"
#include "sfem_mesh_write.h"

// C++ includes
#include "sfem_CRSGraph.hpp"
#include "sfem_Tracer.hpp"

// FIXME
#include "sfem_prolongation_restriction.h"

namespace sfem {

    class Mesh::Impl {
    public:
        MPI_Comm comm;
        mesh_t   mesh;

        std::shared_ptr<CRSGraph>   crs_graph;
        std::shared_ptr<CRSGraph>   crs_graph_upper_triangular;
        std::function<void(void *)> destroy;

        ~Impl() { destroy(&mesh); }
    };

    MPI_Comm Mesh::comm() const { return impl_->comm; }

    Mesh::Mesh(int                         spatial_dim,
               enum ElemType               element_type,
               ptrdiff_t                   nelements,
               idx_t                     **elements,
               ptrdiff_t                   nnodes,
               geom_t                    **points,
               std::function<void(void *)> destroy)
        : impl_(std::make_unique<Impl>()) {
        if (!destroy) {
            impl_->destroy = [](void *m) { mesh_destroy((mesh_t *)m); };
        } else {
            impl_->destroy = destroy;
        }

        mesh_create_serial(&impl_->mesh, spatial_dim, element_type, nelements, elements, nnodes, points);
    }

    std::shared_ptr<Mesh> Mesh::create_hex8_cube(MPI_Comm     comm,
                                                 const int    nx,
                                                 const int    ny,
                                                 const int    nz,
                                                 const geom_t xmin,
                                                 const geom_t ymin,
                                                 const geom_t zmin,
                                                 const geom_t xmax,
                                                 const geom_t ymax,
                                                 const geom_t zmax) {
        auto ret = std::make_shared<Mesh>(comm);
        mesh_create_hex8_cube(&ret->impl_->mesh, nx, ny, nz, xmin, ymin, zmin, xmax, ymax, zmax);
        return ret;
    }

    std::shared_ptr<Mesh> Mesh::create_tri3_square(MPI_Comm     comm,
                                                   const int    nx,
                                                   const int    ny,
                                                   const geom_t xmin,
                                                   const geom_t ymin,
                                                   const geom_t xmax,
                                                   const geom_t ymax) {
        auto ret = std::make_shared<Mesh>(comm);
        mesh_create_tri3_square(&ret->impl_->mesh, nx, ny, xmin, ymin, xmax, ymax);
        return ret;
    }

    int Mesh::spatial_dimension() const { return impl_->mesh.spatial_dim; }
    int Mesh::n_nodes_per_elem() const { return elem_num_nodes((enum ElemType)impl_->mesh.element_type); }

    ptrdiff_t Mesh::n_nodes() const { return impl_->mesh.nnodes; }
    ptrdiff_t Mesh::n_elements() const { return impl_->mesh.nelements; }

    enum ElemType Mesh::element_type() const { return impl_->mesh.element_type; }

    std::shared_ptr<Buffer<geom_t *>> Mesh::points() {
        return Buffer<geom_t *>::wrap(spatial_dimension(), n_nodes(), impl_->mesh.points);
    }

    std::shared_ptr<Buffer<idx_t *>> Mesh::elements() {
        return Buffer<idx_t *>::wrap(n_nodes_per_elem(), n_elements(), impl_->mesh.elements);
    }

    Mesh::Mesh() : impl_(std::make_unique<Impl>()) {
        impl_->comm = MPI_COMM_WORLD;
        mesh_init(&impl_->mesh);
        impl_->destroy = [](void *m) { mesh_destroy((mesh_t *)m); };
    }

    Mesh::Mesh(MPI_Comm comm) : impl_(std::make_unique<Impl>()) {
        impl_->comm = comm;
        mesh_init(&impl_->mesh);
        impl_->destroy = [](void *m) { mesh_destroy((mesh_t *)m); };
    }

    Mesh::~Mesh() = default;

    int Mesh::read(const char *path) {
        SFEM_TRACE_SCOPE("Mesh::read");

        if (mesh_read(impl_->comm, path, &impl_->mesh)) {
            return SFEM_FAILURE;
        }

        int SFEM_USE_MACRO = 0;
        SFEM_READ_ENV(SFEM_USE_MACRO, atoi);

        if (SFEM_USE_MACRO) {
            impl_->mesh.element_type = macro_type_variant((enum ElemType)impl_->mesh.element_type);
        }

        return SFEM_SUCCESS;
    }

    int Mesh::write(const char *path) const {
        SFEM_TRACE_SCOPE("Mesh::write");

        if (mesh_write(path, &impl_->mesh) != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }

        return SFEM_SUCCESS;
    }

    const geom_t *const Mesh::points(const int coord) const {
        assert(coord < spatial_dimension());
        assert(coord >= 0);
        return impl_->mesh.points[coord];
    }

    const idx_t *const Mesh::idx(const int node_num) const {
        assert(node_num < n_nodes_per_elem());
        assert(node_num >= 0);
        return impl_->mesh.elements[node_num];
    }

    std::shared_ptr<CRSGraph> Mesh::node_to_node_graph() {
        initialize_node_to_node_graph();
        return impl_->crs_graph;
    }

    std::shared_ptr<Buffer<element_idx_t>> Mesh::half_face_table() {
        // FIXME it should be allocated outisde
        element_idx_t *table{nullptr};
        create_element_adj_table(n_elements(), n_nodes(), element_type(), elements()->data(), &table);

        int nsxe = elem_num_sides(element_type());
        return manage_host_buffer<element_idx_t>(n_elements() * nsxe, table);
    }

    std::shared_ptr<CRSGraph> Mesh::create_node_to_node_graph(const enum ElemType element_type) {
        auto mesh = &impl_->mesh;
        if (mesh->element_type == element_type) {
            return node_to_node_graph();
        }

        const ptrdiff_t n_nodes = max_node_id(element_type, mesh->nelements, mesh->elements) + 1;

        count_t *rowptr{nullptr};
        idx_t   *colidx{nullptr};
        build_crs_graph_for_elem_type(element_type, mesh->nelements, n_nodes, mesh->elements, &rowptr, &colidx);

        auto crs_graph = std::make_shared<CRSGraph>(Buffer<count_t>::own(n_nodes + 1, rowptr, free, MEMORY_SPACE_HOST),
                                                    Buffer<idx_t>::own(rowptr[n_nodes], colidx, free, MEMORY_SPACE_HOST));

        return crs_graph;
    }

    int Mesh::initialize_node_to_node_graph() {
        if (impl_->crs_graph) {
            return SFEM_SUCCESS;
        }

        SFEM_TRACE_SCOPE("Mesh::initialize_node_to_node_graph");

        impl_->crs_graph = std::make_shared<CRSGraph>();

        auto mesh = &impl_->mesh;

        count_t *rowptr{nullptr};
        idx_t   *colidx{nullptr};
        build_crs_graph_for_elem_type(mesh->element_type, mesh->nelements, mesh->nnodes, mesh->elements, &rowptr, &colidx);

        impl_->crs_graph = std::make_shared<CRSGraph>(Buffer<count_t>::own(mesh->nnodes + 1, rowptr, free, MEMORY_SPACE_HOST),
                                                      Buffer<idx_t>::own(rowptr[mesh->nnodes], colidx, free, MEMORY_SPACE_HOST));

        return SFEM_SUCCESS;
    }

    std::shared_ptr<CRSGraph> Mesh::node_to_node_graph_upper_triangular() {
        if (impl_->crs_graph_upper_triangular) return impl_->crs_graph_upper_triangular;
        SFEM_TRACE_SCOPE("Mesh::node_to_node_graph_upper_triangular");

        auto mesh = &impl_->mesh;

        count_t *rowptr{nullptr};
        idx_t   *colidx{nullptr};
        build_crs_graph_upper_triangular_from_element(mesh->nelements,
                                                      mesh->nnodes,
                                                      elem_num_nodes((enum ElemType)mesh->element_type),
                                                      mesh->elements,
                                                      &rowptr,
                                                      &colidx);

        impl_->crs_graph_upper_triangular =
                std::make_shared<CRSGraph>(Buffer<count_t>::own(mesh->nnodes + 1, rowptr, free, MEMORY_SPACE_HOST),
                                           Buffer<idx_t>::own(rowptr[mesh->nnodes], colidx, free, MEMORY_SPACE_HOST));

        // impl_->crs_graph_upper_triangular->print(std::cout);

        return impl_->crs_graph_upper_triangular;
    }

    int Mesh::convert_to_macro_element_mesh() {
        impl_->mesh.element_type = macro_type_variant((enum ElemType)impl_->mesh.element_type);
        return SFEM_SUCCESS;
    }

    void *Mesh::impl_mesh() { return (void *)&impl_->mesh; }

    std::shared_ptr<Buffer<count_t>> Mesh::node_to_node_rowptr() const { return impl_->crs_graph->rowptr(); }
    std::shared_ptr<Buffer<idx_t>>   Mesh::node_to_node_colidx() const { return impl_->crs_graph->colidx(); }

}  // namespace sfem
