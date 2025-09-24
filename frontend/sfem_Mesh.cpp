#include "sfem_Mesh.hpp"

// C includes
#include "adj_table.h"
#include "crs_graph.h"
#include "multiblock_crs_graph.h"
#include "read_mesh.h"
#include "sfem_macros.h"
#include "sfem_mask.h"
#include "sfem_mesh.h"
#include "sfem_mesh_write.h"

// C++ includes
#include "sfem_CRSGraph.hpp"
#include "sfem_Sideset.hpp"
#include "sfem_Tracer.hpp"
#include "sfem_glob.hpp"

// FIXME
#include "sfem_prolongation_restriction.h"

#include <list>

namespace sfem {

    class Mesh::Block::Impl {
    public:
        std::string           name;
        enum ElemType         element_type;
        SharedBuffer<idx_t *> elements;
    };

    Mesh::Block::Block() : impl_(std::make_unique<Impl>()) {}

    Mesh::Block::~Block() = default;

    const std::string           &Mesh::Block::name() const { return impl_->name; }
    enum ElemType                Mesh::Block::element_type() const { return impl_->element_type; }
    int                          Mesh::Block::n_nodes_per_element() const { return elem_num_nodes(impl_->element_type); }
    const SharedBuffer<idx_t *> &Mesh::Block::elements() const { return impl_->elements; }

    void Mesh::Block::set_name(const std::string &name) { impl_->name = name; }
    void Mesh::Block::set_element_type(enum ElemType element_type) { impl_->element_type = element_type; }
    void Mesh::Block::set_elements(SharedBuffer<idx_t *> elements) { impl_->elements = elements; }

    ptrdiff_t Mesh::Block::n_elements() const { return impl_->elements->extent(1); }

    class Mesh::Impl {
    public:
        std::shared_ptr<Communicator>       comm;
        std::vector<std::shared_ptr<Block>> blocks;

        SharedBuffer<geom_t *> points;  // Node coordinates

        // MPI-related data using Buffers
        SharedBuffer<idx_t>         node_mapping;
        SharedBuffer<int>           node_owner;
        SharedBuffer<element_idx_t> element_mapping;
        SharedBuffer<idx_t>         node_offsets;
        SharedBuffer<idx_t>         ghosts;

        // Metadata
        int       spatial_dim;
        ptrdiff_t nnodes;

        // MPI ownership info
        ptrdiff_t n_owned_nodes;
        ptrdiff_t n_owned_nodes_with_ghosts;
        ptrdiff_t n_owned_elements;
        ptrdiff_t n_owned_elements_with_ghosts;
        ptrdiff_t n_shared_elements;

        std::shared_ptr<CRSGraph> crs_graph;
        std::shared_ptr<CRSGraph> crs_graph_upper_triangular;

        ~Impl() {}

        void clear() {
            comm = nullptr;
            blocks.clear();
            spatial_dim                  = 0;
            nnodes                       = 0;
            points                       = nullptr;
            node_mapping                 = nullptr;
            node_owner                   = nullptr;
            element_mapping              = nullptr;
            node_offsets                 = nullptr;
            ghosts                       = nullptr;
            n_owned_nodes                = 0;
            n_owned_nodes_with_ghosts    = 0;
            n_owned_elements             = 0;
            n_owned_elements_with_ghosts = 0;
            n_shared_elements            = 0;
            crs_graph                    = nullptr;
            crs_graph_upper_triangular   = nullptr;
        }

        // Helper methods for backward compatibility
        ptrdiff_t total_elements() const {
            ptrdiff_t total = 0;
            for (const auto &block : blocks) {
                if (block && block->elements()) {
                    // For Buffer<T*>, extent(1) gives the number of elements
                    total += block->elements()->extent(1);
                }
            }
            return total;
        }

        enum ElemType default_element_type() const {
            if (blocks.empty() || !blocks[0]) {
                return INVALID;
            }
            return blocks[0]->element_type();
        }

        SharedBuffer<idx_t *> default_elements() const {
            if (blocks.empty() || !blocks[0]) {
                return nullptr;
            }
            return blocks[0]->elements();
        }
    };

    std::shared_ptr<Communicator> Mesh::comm() const { return impl_->comm; }

    Mesh::Mesh(const std::shared_ptr<Communicator> &comm,
               int                                  spatial_dim,
               enum ElemType                        element_type,
               ptrdiff_t                            nelements,
               SharedBuffer<idx_t *>                elements,
               ptrdiff_t                            nnodes,
               SharedBuffer<geom_t *>               points)
        : impl_(std::make_unique<Impl>()) {
        impl_->comm        = comm;
        impl_->spatial_dim = spatial_dim;
        impl_->nnodes      = nnodes;
        impl_->points      = points;

        // Create default block
        auto default_block = std::make_shared<Block>();
        default_block->set_name("default");
        default_block->set_element_type(element_type);
        default_block->set_elements(elements);
        impl_->blocks.push_back(default_block);
    }

    Mesh::Mesh() : impl_(std::make_unique<Impl>()) {
        impl_->clear();
        impl_->comm = Communicator::world();
    }

    Mesh::Mesh(const std::shared_ptr<Communicator> &comm) : impl_(std::make_unique<Impl>()) {
        impl_->clear();
        impl_->comm = comm;
    }

    Mesh::~Mesh() = default;

    // Block-related methods
    size_t Mesh::n_blocks() const { return impl_->blocks.size(); }

    std::shared_ptr<const Mesh::Block> Mesh::block(size_t index) const {
        if (index >= impl_->blocks.size() || !impl_->blocks[index]) {
            SFEM_ERROR("Block index out of range");
        }
        return impl_->blocks[index];
    }

    std::shared_ptr<Mesh::Block> Mesh::block(size_t index) {
        if (index >= impl_->blocks.size() || !impl_->blocks[index]) {
            SFEM_ERROR("Block index out of range");
        }
        return impl_->blocks[index];
    }

    void Mesh::add_block(const std::string &name, enum ElemType element_type, SharedBuffer<idx_t *> elements) {
        auto new_block = std::make_shared<Block>();
        new_block->set_name(name);
        new_block->set_element_type(element_type);
        new_block->set_elements(elements);
        impl_->blocks.push_back(new_block);
    }

    void Mesh::remove_block(size_t index) {
        if (index >= impl_->blocks.size()) {
            SFEM_ERROR("Block index out of range");
        }

        impl_->blocks.erase(impl_->blocks.begin() + index);
    }

    int Mesh::read(const char *path) {
        SFEM_TRACE_SCOPE("Mesh::read");

#ifdef SFEM_ENABLE_MPI
        int comm_size;
        MPI_Comm_size(impl_->comm->get(), &comm_size);
        if (comm_size == 1)
#endif
        {
            idx_t   **elements = nullptr;
            geom_t  **points   = nullptr;
            int       nnodesxelem;
            int       spatial_dim;
            ptrdiff_t nelements;

            if (mesh_read_serial(path, &nnodesxelem, &nelements, &elements, &spatial_dim, &impl_->nnodes, &points) !=
                SFEM_SUCCESS) {
                return SFEM_FAILURE;
            }

            auto elements_buffer                = manage_host_buffer<idx_t>(nnodesxelem, nelements, elements);
            impl_->points                       = manage_host_buffer<geom_t>(spatial_dim, impl_->nnodes, points);
            impl_->spatial_dim                  = spatial_dim;
            impl_->nnodes                       = impl_->nnodes;
            impl_->n_owned_nodes                = impl_->nnodes;
            impl_->n_owned_elements             = nelements;
            impl_->n_owned_elements_with_ghosts = 0;
            impl_->n_shared_elements            = 0;
            impl_->n_owned_nodes_with_ghosts    = 0;

            // Create default block
            auto default_block = std::make_shared<Block>();
            default_block->set_name("default");
            default_block->set_element_type((enum ElemType)nnodesxelem);
            default_block->set_elements(elements_buffer);
            impl_->blocks.push_back(default_block);
        }
#ifdef SFEM_ENABLE_MPI
        else {
            int            nnodesxelem;
            ptrdiff_t      nelements;
            idx_t        **elements;
            int            spatial_dim;
            ptrdiff_t      nnodes;
            geom_t       **points;
            ptrdiff_t      n_owned_nodes;
            ptrdiff_t      n_owned_elements;
            element_idx_t *element_mapping;
            idx_t         *node_mapping;
            int           *node_owner;
            idx_t         *node_offsets;
            idx_t         *ghosts;
            ptrdiff_t      n_owned_nodes_with_ghosts;
            ptrdiff_t      n_shared_elements;
            ptrdiff_t      n_owned_elements_with_ghosts;

            if (mesh_read_mpi(impl_->comm->get(),
                              path,
                              &nnodesxelem,
                              &nelements,
                              &elements,
                              &spatial_dim,
                              &nnodes,
                              &points,
                              &n_owned_nodes,
                              &n_owned_elements,
                              &element_mapping,
                              &node_mapping,
                              &node_owner,
                              &node_offsets,
                              &ghosts,
                              &n_owned_nodes_with_ghosts,
                              &n_shared_elements,
                              &n_owned_elements_with_ghosts) != SFEM_SUCCESS) {
                return SFEM_FAILURE;
            }

            auto elements_buffer   = manage_host_buffer<idx_t>(nnodesxelem, nelements, elements);
            impl_->points          = manage_host_buffer<geom_t>(spatial_dim, nnodes, points);
            impl_->node_mapping    = manage_host_buffer<idx_t>(nnodes, node_mapping);
            impl_->node_owner      = manage_host_buffer<int>(nnodes, node_owner);
            impl_->element_mapping = manage_host_buffer<element_idx_t>(nelements, element_mapping);

            int comm_size;
            MPI_Comm_size(impl_->comm->get(), &comm_size);
            impl_->node_offsets = manage_host_buffer<idx_t>(comm_size + 1, node_offsets);

            ptrdiff_t n_ghost_nodes = nnodes - n_owned_nodes;
            impl_->ghosts           = manage_host_buffer<idx_t>(n_ghost_nodes, ghosts);

            impl_->n_owned_nodes                = n_owned_nodes;
            impl_->n_owned_nodes_with_ghosts    = n_owned_nodes_with_ghosts;
            impl_->n_owned_elements             = n_owned_elements;
            impl_->n_owned_elements_with_ghosts = n_owned_elements_with_ghosts;
            impl_->n_shared_elements            = n_shared_elements;

            // Create default block
            auto default_block = std::make_shared<Block>();
            default_block->set_name("default");
            default_block->set_element_type((enum ElemType)nnodesxelem);
            default_block->set_elements(elements_buffer);
            impl_->blocks.push_back(default_block);
        }
#endif  // SFEM_ENABLE_MPI

        int SFEM_USE_MACRO = 0;
        SFEM_READ_ENV(SFEM_USE_MACRO, atoi);

        if (SFEM_USE_MACRO) {
            for (auto &block : blocks()) {
                if (block) {
                    block->set_element_type(macro_type_variant(block->element_type()));
                }
            }
        }

        return SFEM_SUCCESS;
    }

    const std::vector<std::shared_ptr<Mesh::Block>> &Mesh::blocks() const { return impl_->blocks; }

    int Mesh::write(const char *path) const {
        SFEM_TRACE_SCOPE("Mesh::write");

        create_directory(path);

        if (impl_->node_mapping) {
            std::string path_node_mapping = std::string(path) + "/node_mapping.raw";
            impl_->node_mapping->to_file(path_node_mapping.c_str());
        }

        // Write the default block (block 0)
        if (impl_->blocks.empty() || !impl_->blocks[0]) {
            return SFEM_FAILURE;
        }

        if (impl_->blocks.size() == 1) {
            return mesh_write_serial(path,
                                     impl_->blocks[0]->element_type(),
                                     impl_->blocks[0]->elements()->extent(1),
                                     impl_->blocks[0]->elements()->data(),
                                     impl_->spatial_dim,
                                     impl_->nnodes,
                                     impl_->points->data());
        } else {
            std::vector<ptrdiff_t>     n_elements;
            std::vector<enum ElemType> element_types;
            std::vector<idx_t **>      elements;
            std::vector<const char *>  block_names;

            for (auto &block : impl_->blocks) {
                n_elements.push_back(block->elements()->extent(1));
                element_types.push_back(block->element_type());
                elements.push_back(block->elements()->data());
                block_names.push_back(block->name().c_str());
            }
            return mesh_multiblock_write_serial(path,
                                                impl_->blocks.size(),
                                                block_names.data(),
                                                element_types.data(),
                                                n_elements.data(),
                                                elements.data(),
                                                impl_->spatial_dim,
                                                impl_->nnodes,
                                                impl_->points->data());
        }
    }

    const geom_t *const Mesh::points(const int coord) const {
        assert(coord < spatial_dimension());
        assert(coord >= 0);
        return impl_->points->data()[coord];
    }

    const idx_t *const Mesh::idx(const int node_num) const {
        assert(node_num < n_nodes_per_element());
        assert(node_num >= 0);
        return impl_->default_elements()->data()[node_num];
    }

    std::shared_ptr<CRSGraph> Mesh::node_to_node_graph() {
        initialize_node_to_node_graph();
        return impl_->crs_graph;
    }

    SharedBuffer<element_idx_t> Mesh::half_face_table() {
        // FIXME it should be allocated outisde
        element_idx_t *table{nullptr};
        create_element_adj_table(n_elements(), n_nodes(), element_type(), default_elements()->data(), &table);

        int nsxe = elem_num_sides(element_type());
        return manage_host_buffer<element_idx_t>(n_elements() * nsxe, table);
    }

    std::shared_ptr<CRSGraph> Mesh::create_node_to_node_graph(const enum ElemType element_type) {
        if (impl_->default_element_type() == element_type) {
            return node_to_node_graph();
        }

        const ptrdiff_t n_nodes = max_node_id(element_type, impl_->total_elements(), impl_->default_elements()->data()) + 1;

        count_t *rowptr{nullptr};
        idx_t   *colidx{nullptr};
        build_crs_graph_for_elem_type(
                element_type, impl_->total_elements(), n_nodes, impl_->default_elements()->data(), &rowptr, &colidx);

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

        count_t *rowptr{nullptr};
        idx_t   *colidx{nullptr};

        if (impl_->blocks.size() == 1) {
            build_crs_graph_for_elem_type(impl_->default_element_type(),
                                          impl_->total_elements(),
                                          impl_->nnodes,
                                          impl_->default_elements()->data(),
                                          &rowptr,
                                          &colidx);
        } else {
            // AoS to SoA
            std::vector<enum ElemType> element_types;
            std::vector<ptrdiff_t>     n_elements;
            std::vector<idx_t **>      elements;

            for (auto &block : impl_->blocks) {
                element_types.push_back(block->element_type());
                n_elements.push_back(block->elements()->extent(1));
                elements.push_back(block->elements()->data());
            }

            build_multiblock_crs_graph(impl_->blocks.size(),
                                       element_types.data(),
                                       n_elements.data(),
                                       elements.data(),
                                       impl_->nnodes,
                                       &rowptr,
                                       &colidx);
        }

        impl_->crs_graph = std::make_shared<CRSGraph>(Buffer<count_t>::own(impl_->nnodes + 1, rowptr, free, MEMORY_SPACE_HOST),
                                                      Buffer<idx_t>::own(rowptr[impl_->nnodes], colidx, free, MEMORY_SPACE_HOST));

        return SFEM_SUCCESS;
    }

    std::shared_ptr<CRSGraph> Mesh::node_to_node_graph_upper_triangular() {
        if (impl_->crs_graph_upper_triangular) return impl_->crs_graph_upper_triangular;
        SFEM_TRACE_SCOPE("Mesh::node_to_node_graph_upper_triangular");

        count_t *rowptr{nullptr};
        idx_t   *colidx{nullptr};

        if (impl_->blocks.size() == 1) {
            build_crs_graph_upper_triangular_from_element(impl_->total_elements(),
                                                          impl_->nnodes,
                                                          elem_num_nodes(impl_->default_element_type()),
                                                          impl_->default_elements()->data(),
                                                          &rowptr,
                                                          &colidx);
        } else {
            // AoS to SoA
            std::vector<enum ElemType> element_types;
            std::vector<ptrdiff_t>     n_elements;
            std::vector<idx_t **>      elements;

            for (auto &block : impl_->blocks) {
                element_types.push_back(block->element_type());
                n_elements.push_back(block->elements()->extent(1));
                elements.push_back(block->elements()->data());
            }

            build_multiblock_crs_graph_upper_triangular(impl_->blocks.size(),
                                                        element_types.data(),
                                                        n_elements.data(),
                                                        elements.data(),
                                                        impl_->nnodes,
                                                        &rowptr,
                                                        &colidx);
        }

        impl_->crs_graph_upper_triangular =
                std::make_shared<CRSGraph>(Buffer<count_t>::own(impl_->nnodes + 1, rowptr, free, MEMORY_SPACE_HOST),
                                           Buffer<idx_t>::own(rowptr[impl_->nnodes], colidx, free, MEMORY_SPACE_HOST));

        return impl_->crs_graph_upper_triangular;
    }

    int Mesh::convert_to_macro_element_mesh() {
        if (!impl_->blocks.empty() && impl_->blocks[0]) {
            impl_->blocks[0]->set_element_type(macro_type_variant(impl_->blocks[0]->element_type()));
        }
        return SFEM_SUCCESS;
    }

    SharedBuffer<count_t> Mesh::node_to_node_rowptr() const { return impl_->crs_graph->rowptr(); }
    SharedBuffer<idx_t>   Mesh::node_to_node_colidx() const { return impl_->crs_graph->colidx(); }

    std::shared_ptr<Mesh> Mesh::create_hex8_cube(const std::shared_ptr<Communicator> &comm,
                                                 const int                            nx,
                                                 const int                            ny,
                                                 const int                            nz,
                                                 const geom_t                         xmin,
                                                 const geom_t                         ymin,
                                                 const geom_t                         zmin,
                                                 const geom_t                         xmax,
                                                 const geom_t                         ymax,
                                                 const geom_t                         zmax) {
        auto            ret       = std::make_shared<Mesh>(comm);
        const ptrdiff_t nelements = nx * ny * nz;
        const ptrdiff_t nnodes    = (nx + 1) * (ny + 1) * (nz + 1);

        ret->impl_->spatial_dim = 3;
        ret->impl_->nnodes      = nnodes;
        ret->impl_->points      = create_host_buffer<geom_t>(3, nnodes);
        auto elements_buffer    = create_host_buffer<idx_t>(8, nelements);

        auto points   = ret->impl_->points->data();
        auto elements = elements_buffer->data();

        const ptrdiff_t ldz = (ny + 1) * (nx + 1);
        const ptrdiff_t ldy = nx + 1;
        const ptrdiff_t ldx = 1;

        const double hx = (xmax - xmin) * 1. / nx;
        const double hy = (ymax - ymin) * 1. / ny;
        const double hz = (zmax - zmin) * 1. / nz;

        assert(hx > 0);
        assert(hy > 0);
        assert(hz > 0);

        for (ptrdiff_t zi = 0; zi < nz; zi++) {
            for (ptrdiff_t yi = 0; yi < ny; yi++) {
                for (ptrdiff_t xi = 0; xi < nx; xi++) {
                    const ptrdiff_t e = zi * ny * nx + yi * nx + xi;

                    const idx_t i0 = (xi + 0) * ldx + (yi + 0) * ldy + (zi + 0) * ldz;
                    const idx_t i1 = (xi + 1) * ldx + (yi + 0) * ldy + (zi + 0) * ldz;
                    const idx_t i2 = (xi + 1) * ldx + (yi + 1) * ldy + (zi + 0) * ldz;
                    const idx_t i3 = (xi + 0) * ldx + (yi + 1) * ldy + (zi + 0) * ldz;

                    const idx_t i4 = (xi + 0) * ldx + (yi + 0) * ldy + (zi + 1) * ldz;
                    const idx_t i5 = (xi + 1) * ldx + (yi + 0) * ldy + (zi + 1) * ldz;
                    const idx_t i6 = (xi + 1) * ldx + (yi + 1) * ldy + (zi + 1) * ldz;
                    const idx_t i7 = (xi + 0) * ldx + (yi + 1) * ldy + (zi + 1) * ldz;

                    elements[0][e] = i0;
                    elements[1][e] = i1;
                    elements[2][e] = i2;
                    elements[3][e] = i3;

                    elements[4][e] = i4;
                    elements[5][e] = i5;
                    elements[6][e] = i6;
                    elements[7][e] = i7;
                }
            }
        }

        for (ptrdiff_t zi = 0; zi <= nz; zi++) {
            for (ptrdiff_t yi = 0; yi <= ny; yi++) {
                for (ptrdiff_t xi = 0; xi <= nx; xi++) {
                    ptrdiff_t node  = xi * ldx + yi * ldy + zi * ldz;
                    points[0][node] = (double)xmin + xi * hx;
                    points[1][node] = (double)ymin + yi * hy;
                    points[2][node] = (double)zmin + zi * hz;
                }
            }
        }

        // Create default block
        auto default_block = std::make_shared<Block>();
        default_block->set_name("default");
        default_block->set_element_type(HEX8);
        default_block->set_elements(elements_buffer);
        ret->impl_->blocks.push_back(default_block);

        return ret;
    }

    std::shared_ptr<Mesh> Mesh::create_tri3_square(const std::shared_ptr<Communicator> &comm,
                                                   const int                            nx,
                                                   const int                            ny,
                                                   const geom_t                         xmin,
                                                   const geom_t                         ymin,
                                                   const geom_t                         xmax,
                                                   const geom_t                         ymax) {
        auto            ret       = std::make_shared<Mesh>(comm);
        const ptrdiff_t nelements = 2 * nx * ny;
        const ptrdiff_t nnodes    = (nx + 1) * (ny + 1);

        ret->impl_->spatial_dim = 2;
        ret->impl_->nnodes      = nnodes;
        ret->impl_->points      = create_host_buffer<geom_t>(2, nnodes);
        auto elements_buffer    = create_host_buffer<idx_t>(3, nelements);

        auto points   = ret->impl_->points->data();
        auto elements = elements_buffer->data();

        const ptrdiff_t ldy = nx + 1;
        const ptrdiff_t ldx = 1;

        const double hx = (xmax - xmin) * 1. / nx;
        const double hy = (ymax - ymin) * 1. / ny;

        assert(hx > 0);
        assert(hy > 0);

        for (ptrdiff_t yi = 0; yi < ny; yi++) {
            for (ptrdiff_t xi = 0; xi < nx; xi++) {
                const ptrdiff_t e = 2 * (yi * nx + xi);

                const idx_t i0 = (xi + 0) * ldx + (yi + 0) * ldy;
                const idx_t i1 = (xi + 1) * ldx + (yi + 0) * ldy;
                const idx_t i2 = (xi + 1) * ldx + (yi + 1) * ldy;
                const idx_t i3 = (xi + 0) * ldx + (yi + 1) * ldy;

                elements[0][e] = i0;
                elements[1][e] = i1;
                elements[2][e] = i3;

                elements[0][e + 1] = i1;
                elements[1][e + 1] = i2;
                elements[2][e + 1] = i3;
            }
        }

        for (ptrdiff_t yi = 0; yi <= ny; yi++) {
            for (ptrdiff_t xi = 0; xi <= nx; xi++) {
                ptrdiff_t node  = xi * ldx + yi * ldy;
                points[0][node] = (double)xmin + xi * hx;
                points[1][node] = (double)ymin + yi * hy;
            }
        }

        // Create default block
        auto default_block = std::make_shared<Block>();
        default_block->set_name("default");
        default_block->set_element_type(TRI3);
        default_block->set_elements(elements_buffer);
        ret->impl_->blocks.push_back(default_block);

        return ret;
    }

    std::shared_ptr<Mesh> Mesh::create_quad4_square(const std::shared_ptr<Communicator> &comm,
                                                    const int                            nx,
                                                    const int                            ny,
                                                    const geom_t                         xmin,
                                                    const geom_t                         ymin,
                                                    const geom_t                         xmax,
                                                    const geom_t                         ymax) {
        auto            ret       = std::make_shared<Mesh>(comm);
        const ptrdiff_t nelements = nx * ny;
        const ptrdiff_t nnodes    = (nx + 1) * (ny + 1);

        ret->impl_->spatial_dim = 2;
        ret->impl_->nnodes      = nnodes;
        ret->impl_->points      = create_host_buffer<geom_t>(2, nnodes);
        auto elements_buffer    = create_host_buffer<idx_t>(4, nelements);

        auto points   = ret->impl_->points->data();
        auto elements = elements_buffer->data();

        const ptrdiff_t ldy = nx + 1;
        const ptrdiff_t ldx = 1;

        const double hx = (xmax - xmin) * 1. / nx;
        const double hy = (ymax - ymin) * 1. / ny;

        assert(hx > 0);
        assert(hy > 0);

        for (ptrdiff_t yi = 0; yi < ny; yi++) {
            for (ptrdiff_t xi = 0; xi < nx; xi++) {
                const ptrdiff_t e = yi * nx + xi;

                const idx_t i0 = (xi + 0) * ldx + (yi + 0) * ldy;
                const idx_t i1 = (xi + 1) * ldx + (yi + 0) * ldy;
                const idx_t i2 = (xi + 1) * ldx + (yi + 1) * ldy;
                const idx_t i3 = (xi + 0) * ldx + (yi + 1) * ldy;

                elements[0][e] = i0;
                elements[1][e] = i1;
                elements[2][e] = i2;
                elements[3][e] = i3;
            }
        }

        for (ptrdiff_t yi = 0; yi <= ny; yi++) {
            for (ptrdiff_t xi = 0; xi <= nx; xi++) {
                ptrdiff_t node  = xi * ldx + yi * ldy;
                points[0][node] = (double)xmin + xi * hx;
                points[1][node] = (double)ymin + yi * hy;
            }
        }

        // Create default block
        auto default_block = std::make_shared<Block>();
        default_block->set_name("default");
        default_block->set_element_type(QUAD4);
        default_block->set_elements(elements_buffer);
        ret->impl_->blocks.push_back(default_block);

        return ret;
    }

    std::shared_ptr<Mesh> Mesh::create_hex8_checkerboard_cube(const std::shared_ptr<Communicator> &comm,
                                                              const int                            nx,
                                                              const int                            ny,
                                                              const int                            nz,
                                                              const geom_t                         xmin,
                                                              const geom_t                         ymin,
                                                              const geom_t                         zmin,
                                                              const geom_t                         xmax,
                                                              const geom_t                         ymax,
                                                              const geom_t                         zmax) {
        auto            ret       = std::make_shared<Mesh>(comm);
        const ptrdiff_t nelements = nx * ny * nz;
        const ptrdiff_t nnodes    = (nx + 1) * (ny + 1) * (nz + 1);

        if (nx % 2 != 0 || ny % 2 != 0 || nz % 2 != 0) {
            SFEM_ERROR("nx, ny, and nz must be even");
        }

        ret->impl_->spatial_dim    = 3;
        ret->impl_->nnodes         = nnodes;
        ret->impl_->points         = create_host_buffer<geom_t>(3, nnodes);
        auto white_elements_buffer = create_host_buffer<idx_t>(8, nelements / 2);
        auto black_elements_buffer = create_host_buffer<idx_t>(8, nelements / 2);

        auto points         = ret->impl_->points->data();
        auto white_elements = white_elements_buffer->data();
        auto black_elements = black_elements_buffer->data();

        const ptrdiff_t ldz = (ny + 1) * (nx + 1);
        const ptrdiff_t ldy = nx + 1;
        const ptrdiff_t ldx = 1;

        const double hx = (xmax - xmin) * 1. / nx;
        const double hy = (ymax - ymin) * 1. / ny;
        const double hz = (zmax - zmin) * 1. / nz;

        assert(hx > 0);
        assert(hy > 0);
        assert(hz > 0);

        ptrdiff_t white_elements_count = 0;
        ptrdiff_t black_elements_count = 0;

        for (ptrdiff_t zi = 0; zi < nz; zi++) {
            for (ptrdiff_t yi = 0; yi < ny; yi++) {
                for (ptrdiff_t xi = 0; xi < nx; xi++) {
                    const ptrdiff_t e = zi * ny * nx + yi * nx + xi;

                    const idx_t i0 = (xi + 0) * ldx + (yi + 0) * ldy + (zi + 0) * ldz;
                    const idx_t i1 = (xi + 1) * ldx + (yi + 0) * ldy + (zi + 0) * ldz;
                    const idx_t i2 = (xi + 1) * ldx + (yi + 1) * ldy + (zi + 0) * ldz;
                    const idx_t i3 = (xi + 0) * ldx + (yi + 1) * ldy + (zi + 0) * ldz;

                    const idx_t i4 = (xi + 0) * ldx + (yi + 0) * ldy + (zi + 1) * ldz;
                    const idx_t i5 = (xi + 1) * ldx + (yi + 0) * ldy + (zi + 1) * ldz;
                    const idx_t i6 = (xi + 1) * ldx + (yi + 1) * ldy + (zi + 1) * ldz;
                    const idx_t i7 = (xi + 0) * ldx + (yi + 1) * ldy + (zi + 1) * ldz;

                    if ((xi + yi + zi) % 2 == 0) {
                        white_elements[0][white_elements_count] = i0;
                        white_elements[1][white_elements_count] = i1;
                        white_elements[2][white_elements_count] = i2;
                        white_elements[3][white_elements_count] = i3;

                        white_elements[4][white_elements_count] = i4;
                        white_elements[5][white_elements_count] = i5;
                        white_elements[6][white_elements_count] = i6;
                        white_elements[7][white_elements_count] = i7;

                        white_elements_count++;
                    } else {
                        black_elements[0][black_elements_count] = i0;
                        black_elements[1][black_elements_count] = i1;
                        black_elements[2][black_elements_count] = i2;
                        black_elements[3][black_elements_count] = i3;

                        black_elements[4][black_elements_count] = i4;
                        black_elements[5][black_elements_count] = i5;
                        black_elements[6][black_elements_count] = i6;
                        black_elements[7][black_elements_count] = i7;

                        black_elements_count++;
                    }
                }
            }
        }

        assert(white_elements_count == nelements / 2);
        assert(black_elements_count == nelements / 2);

        for (ptrdiff_t zi = 0; zi <= nz; zi++) {
            for (ptrdiff_t yi = 0; yi <= ny; yi++) {
                for (ptrdiff_t xi = 0; xi <= nx; xi++) {
                    ptrdiff_t node  = xi * ldx + yi * ldy + zi * ldz;
                    points[0][node] = (double)xmin + xi * hx;
                    points[1][node] = (double)ymin + yi * hy;
                    points[2][node] = (double)zmin + zi * hz;
                }
            }
        }

        // Create white and black blocks
        auto white_block = std::make_shared<Block>();
        white_block->set_name("white");
        white_block->set_element_type(HEX8);
        white_block->set_elements(white_elements_buffer);
        ret->impl_->blocks.push_back(white_block);

        auto black_block = std::make_shared<Block>();
        black_block->set_name("black");
        black_block->set_element_type(HEX8);
        black_block->set_elements(black_elements_buffer);
        ret->impl_->blocks.push_back(black_block);
        return ret;
    }

    std::shared_ptr<Mesh> Mesh::create_hex8_bidomain_cube(const std::shared_ptr<Communicator> &comm,
                                                          const int                            nx,
                                                          const int                            ny,
                                                          const int                            nz,
                                                          const geom_t                         xmin,
                                                          const geom_t                         ymin,
                                                          const geom_t                         zmin,
                                                          const geom_t                         xmax,
                                                          const geom_t                         ymax,
                                                          const geom_t                         zmax) {
        auto            ret       = std::make_shared<Mesh>(comm);
        const ptrdiff_t nelements = nx * ny * nz;
        const ptrdiff_t nnodes    = (nx + 1) * (ny + 1) * (nz + 1);

        ret->impl_->spatial_dim    = 3;
        ret->impl_->nnodes         = nnodes;
        ret->impl_->points         = create_host_buffer<geom_t>(3, nnodes);
        auto left_elements_buffer  = create_host_buffer<idx_t>(8, nelements / 2);
        auto right_elements_buffer = create_host_buffer<idx_t>(8, nelements / 2);

        auto points         = ret->impl_->points->data();
        auto left_elements  = left_elements_buffer->data();
        auto right_elements = right_elements_buffer->data();

        const ptrdiff_t ldz = (ny + 1) * (nx + 1);
        const ptrdiff_t ldy = nx + 1;
        const ptrdiff_t ldx = 1;

        const double hx = (xmax - xmin) * 1. / nx;
        const double hy = (ymax - ymin) * 1. / ny;
        const double hz = (zmax - zmin) * 1. / nz;

        assert(hx > 0);
        assert(hy > 0);
        assert(hz > 0);

        ptrdiff_t left_elements_count  = 0;
        ptrdiff_t right_elements_count = 0;

        for (ptrdiff_t zi = 0; zi < nz; zi++) {
            for (ptrdiff_t yi = 0; yi < ny; yi++) {
                for (ptrdiff_t xi = 0; xi < nx; xi++) {
                    const ptrdiff_t e = zi * ny * nx + yi * nx + xi;

                    const idx_t i0 = (xi + 0) * ldx + (yi + 0) * ldy + (zi + 0) * ldz;
                    const idx_t i1 = (xi + 1) * ldx + (yi + 0) * ldy + (zi + 0) * ldz;
                    const idx_t i2 = (xi + 1) * ldx + (yi + 1) * ldy + (zi + 0) * ldz;
                    const idx_t i3 = (xi + 0) * ldx + (yi + 1) * ldy + (zi + 0) * ldz;

                    const idx_t i4 = (xi + 0) * ldx + (yi + 0) * ldy + (zi + 1) * ldz;
                    const idx_t i5 = (xi + 1) * ldx + (yi + 0) * ldy + (zi + 1) * ldz;
                    const idx_t i6 = (xi + 1) * ldx + (yi + 1) * ldy + (zi + 1) * ldz;
                    const idx_t i7 = (xi + 0) * ldx + (yi + 1) * ldy + (zi + 1) * ldz;

                    if (xi < nx / 2) {
                        left_elements[0][left_elements_count] = i0;
                        left_elements[1][left_elements_count] = i1;
                        left_elements[2][left_elements_count] = i2;
                        left_elements[3][left_elements_count] = i3;

                        left_elements[4][left_elements_count] = i4;
                        left_elements[5][left_elements_count] = i5;
                        left_elements[6][left_elements_count] = i6;
                        left_elements[7][left_elements_count] = i7;

                        left_elements_count++;
                    } else {
                        right_elements[0][right_elements_count] = i0;
                        right_elements[1][right_elements_count] = i1;
                        right_elements[2][right_elements_count] = i2;
                        right_elements[3][right_elements_count] = i3;

                        right_elements[4][right_elements_count] = i4;
                        right_elements[5][right_elements_count] = i5;
                        right_elements[6][right_elements_count] = i6;
                        right_elements[7][right_elements_count] = i7;

                        right_elements_count++;
                    }
                }
            }
        }

        assert(left_elements_count == nelements / 2);
        assert(right_elements_count == nelements / 2);

        for (ptrdiff_t zi = 0; zi <= nz; zi++) {
            for (ptrdiff_t yi = 0; yi <= ny; yi++) {
                for (ptrdiff_t xi = 0; xi <= nx; xi++) {
                    ptrdiff_t node  = xi * ldx + yi * ldy + zi * ldz;
                    points[0][node] = (double)xmin + xi * hx;
                    points[1][node] = (double)ymin + yi * hy;
                    points[2][node] = (double)zmin + zi * hz;
                }
            }
        }

        // Create left and right blocks
        auto left_block = std::make_shared<Block>();
        left_block->set_name("left");
        left_block->set_element_type(HEX8);
        left_block->set_elements(left_elements_buffer);
        ret->impl_->blocks.push_back(left_block);

        auto right_block = std::make_shared<Block>();
        right_block->set_name("right");
        right_block->set_element_type(HEX8);
        right_block->set_elements(right_elements_buffer);
        ret->impl_->blocks.push_back(right_block);

        return ret;
    }

    int Mesh::spatial_dimension() const { return impl_->spatial_dim; }
    int Mesh::n_nodes_per_element() const { return elem_num_nodes(impl_->default_element_type()); }

    ptrdiff_t Mesh::n_nodes() const { return impl_->nnodes; }
    ptrdiff_t Mesh::n_elements() const { return impl_->total_elements(); }

    enum ElemType Mesh::element_type() const { return impl_->default_element_type(); }

    ptrdiff_t Mesh::n_owned_nodes() const { return impl_->n_owned_nodes; }
    ptrdiff_t Mesh::n_owned_nodes_with_ghosts() const { return impl_->n_owned_nodes_with_ghosts; }
    ptrdiff_t Mesh::n_owned_elements() const { return impl_->n_owned_elements; }
    ptrdiff_t Mesh::n_owned_elements_with_ghosts() const { return impl_->n_owned_elements_with_ghosts; }
    ptrdiff_t Mesh::n_shared_elements() const { return impl_->n_shared_elements; }

    SharedBuffer<idx_t> Mesh::node_mapping() const { return impl_->node_mapping; }
    SharedBuffer<idx_t> Mesh::element_mapping() const { return impl_->element_mapping; }

    SharedBuffer<idx_t> Mesh::node_offsets() const { return impl_->node_offsets; }
    SharedBuffer<idx_t> Mesh::ghosts() const { return impl_->ghosts; }
    SharedBuffer<int>   Mesh::node_owner() const { return impl_->node_owner; }

    SharedBuffer<geom_t *> Mesh::points() { return impl_->points; }

    SharedBuffer<idx_t *> Mesh::elements() { return impl_->default_elements(); }
    SharedBuffer<idx_t *> Mesh::default_elements() { return impl_->default_elements(); }

    void Mesh::set_node_mapping(const SharedBuffer<idx_t> &node_mapping) { impl_->node_mapping = node_mapping; }

    void Mesh::set_comm(const std::shared_ptr<Communicator> &comm) { impl_->comm = comm; }

    void Mesh::set_element_type(const enum ElemType element_type) {
        if (!impl_->blocks.empty() && impl_->blocks[0]) {
            impl_->blocks[0]->set_element_type(element_type);
        }
    }

    std::vector<std::shared_ptr<Mesh::Block>> Mesh::blocks(const std::vector<std::string> &block_names) const {

        if(block_names.empty()) {
            return impl_->blocks;
        }

        std::vector<std::shared_ptr<Mesh::Block>> ret;
        for(auto &block : impl_->blocks) {
            if(std::find(block_names.begin(), block_names.end(), block->name()) != block_names.end()) {
                ret.push_back(block);
            }
        }

        return ret;
    }

    void Mesh::extract_deprecated(mesh_t *mesh) {
#ifdef SFEM_ENABLE_MPI
        mesh->comm = impl_->comm->get();
#endif
        mesh->spatial_dim  = impl_->spatial_dim;
        mesh->element_type = impl_->default_element_type();

        mesh->nelements = impl_->total_elements();
        mesh->nnodes    = impl_->nnodes;

        mesh->elements = impl_->default_elements()->data();
        mesh->points   = impl_->points->data();

        mesh->n_owned_nodes             = impl_->n_owned_nodes;
        mesh->n_owned_nodes_with_ghosts = impl_->n_owned_nodes_with_ghosts;

        mesh->n_owned_elements             = impl_->n_owned_elements;
        mesh->n_owned_elements_with_ghosts = impl_->n_owned_elements_with_ghosts;
        mesh->n_shared_elements            = impl_->n_shared_elements;

        mesh->node_mapping    = (impl_->node_mapping) ? impl_->node_mapping->data() : nullptr;
        mesh->node_owner      = (impl_->node_owner) ? impl_->node_owner->data() : nullptr;
        mesh->element_mapping = (impl_->element_mapping) ? impl_->element_mapping->data() : nullptr;

        mesh->node_offsets = (impl_->node_offsets) ? impl_->node_offsets->data() : nullptr;
        mesh->ghosts       = (impl_->ghosts) ? impl_->ghosts->data() : nullptr;
    }

    std::shared_ptr<Mesh> Mesh::create_hex8_reference_cube() {
        auto ret                = std::make_shared<Mesh>(Communicator::null());
        ret->impl_->spatial_dim = 3;
        ret->impl_->nnodes      = 8;
        ret->impl_->points      = create_host_buffer<geom_t>(3, 8);
        auto elements_buffer    = create_host_buffer<idx_t>(8, 1);

        auto points   = ret->impl_->points->data();
        auto elements = elements_buffer->data();

        for (int i = 0; i < 8; i++) {
            elements[i][0] = i;
        }

        points[0][0] = 0;
        points[1][0] = 0;
        points[2][0] = 0;

        points[0][1] = 1;
        points[1][1] = 0;
        points[2][1] = 0;

        points[0][2] = 1;
        points[1][2] = 1;
        points[2][2] = 0;

        points[0][3] = 0;
        points[1][3] = 1;
        points[2][3] = 0;

        points[0][4] = 0;
        points[1][4] = 0;
        points[2][4] = 1;

        points[0][5] = 1;
        points[1][5] = 0;
        points[2][5] = 1;

        points[0][6] = 1;
        points[1][6] = 1;
        points[2][6] = 1;

        points[0][7] = 0;
        points[1][7] = 1;
        points[2][7] = 1;

        // Create default block
        auto default_block = std::make_shared<Block>();
        default_block->set_name("default");
        default_block->set_element_type(HEX8);
        default_block->set_elements(elements_buffer);
        ret->impl_->blocks.push_back(default_block);

        return ret;
    }

    std::pair<SharedBuffer<geom_t>, SharedBuffer<geom_t>> Mesh::compute_bounding_box() {
        auto points = impl_->points->data();

        int  dim = spatial_dimension();
        auto min = create_host_buffer<geom_t>(dim);
        auto max = create_host_buffer<geom_t>(dim);

        auto d_min = min->data();
        auto d_max = max->data();

        for (int d = 0; d < dim; d++) {
            d_min[d] = points[d][0];
            d_max[d] = points[d][0];
        }

        ptrdiff_t n_nodes = this->n_nodes();

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n_nodes; i++) {
            for (int d = 0; d < dim; d++) {
                d_min[d] = std::min(d_min[d], points[d][i]);
                d_max[d] = std::max(d_max[d], points[d][i]);
            }
        }

        return {min, max};
    }

    std::shared_ptr<Mesh::Block> Mesh::find_block(const std::string &name) const {
        for (auto &block : impl_->blocks) {
            if (block->name() == name) {
                return block;
            }
        }
        return nullptr;
    }

    int Mesh::split_block(const SharedBuffer<element_idx_t> &elements, const std::string &name) {
        if (n_blocks() != 1) {
            SFEM_ERROR("Mesh must have exactly one block to split boundary layer!\n");
            return SFEM_FAILURE;
        }

        {
            const int       nxe        = n_nodes_per_element();
            const ptrdiff_t n_elements = this->n_elements();

            auto bdry_mask = create_host_buffer<mask_t>(mask_count(n_elements));

            auto            d_parent     = elements->data();
            auto            d_bdry_mask  = bdry_mask->data();
            const ptrdiff_t size_sideset = elements->size();

            auto default_block = impl_->blocks[0];

            auto d_elements = default_block->elements()->data();

            ptrdiff_t n_bdry_elements = 0;
            for (ptrdiff_t i = 0; i < size_sideset; i++) {
                if (mask_get(d_parent[i], d_bdry_mask) == 0) {
                    n_bdry_elements++;
                    mask_set(d_parent[i], d_bdry_mask);
                }
            }

            memset(d_bdry_mask, 0, mask_count(n_elements) * sizeof(mask_t));

            auto      bdry_elements         = create_host_buffer<idx_t>(nxe, n_bdry_elements);
            ptrdiff_t n_bdry_elements_count = 0;

            auto d_bdry_elements = bdry_elements->data();

            for (int e = 0; e < size_sideset; e++) {
                if (mask_get(d_parent[e], d_bdry_mask) == 0) {
                    for (int v = 0; v < nxe; v++) {
                        d_bdry_elements[v][n_bdry_elements_count] = d_elements[v][d_parent[e]];
                    }
                    mask_set(d_parent[e], d_bdry_mask);
                    n_bdry_elements_count++;
                }
            }

            auto      interior_elements         = create_host_buffer<idx_t>(nxe, n_elements - n_bdry_elements);
            ptrdiff_t n_interior_elements_count = 0;
            auto      d_interior_elements       = interior_elements->data();

            for (ptrdiff_t i = 0; i < n_elements; i++) {
                if (mask_get(i, d_bdry_mask) == 0) {
                    for (int v = 0; v < nxe; v++) {
                        assert(n_interior_elements_count < interior_elements->extent(1));
                        d_interior_elements[v][n_interior_elements_count] = d_elements[v][i];
                    }
                    n_interior_elements_count++;
                }
            }

            // !!!!
            remove_block(0);

            {  // Boundary block
                auto block = std::make_shared<Block>();
                block->set_name(name);
                block->set_element_type(default_block->element_type());
                block->set_elements(bdry_elements);
                impl_->blocks.push_back(block);
            }

            {  // Interior block
                auto block = std::make_shared<Block>();
                block->set_name(default_block->name());
                block->set_element_type(default_block->element_type());
                block->set_elements(interior_elements);
                impl_->blocks.push_back(block);
            }
        }

        return SFEM_SUCCESS;
    }

    int Mesh::split_boundary_layer() {
        if (n_blocks() != 1) {
            SFEM_ERROR("Mesh must have exactly one block to split boundary layer!\n");
            return SFEM_FAILURE;
        }

        std::shared_ptr<Sideset> sideset;

        {
            ptrdiff_t      n_surf_elements = 0;
            element_idx_t *parent          = 0;
            int16_t       *side_idx        = 0;

            if (extract_skin_sideset(this->n_elements(),
                                     this->n_nodes(),
                                     this->element_type(),
                                     this->elements()->data(),
                                     &n_surf_elements,
                                     &parent,
                                     &side_idx) != SFEM_SUCCESS) {
                SFEM_ERROR("Failed to extract skin!\n");
            }

            sideset = std::make_shared<sfem::Sideset>(this->comm(),
                                                      sfem::manage_host_buffer(n_surf_elements, parent),
                                                      sfem::manage_host_buffer(n_surf_elements, side_idx));
        }

        return split_block(sideset->parent(), "boundary_layer");
    }

    int Mesh::renumber_nodes() {
        const int nxe = n_nodes_per_element();
        auto n_nodes    = this->n_nodes();
        auto n_elements = this->n_elements();

        auto new_idx_buff = create_host_buffer<idx_t>(n_nodes);

        auto new_idx = new_idx_buff->data();
        for (ptrdiff_t i = 0; i < n_nodes; i++) {
            new_idx[i] = -1;
        }

        idx_t next_node_id = 0;
        for (auto &b : impl_->blocks) {
            auto elements   = b->elements()->data();
            auto n_elements = b->n_elements();

            for (ptrdiff_t e = 0; e < n_elements; e++) {
                for (int v = 0; v < nxe; v++) {
                    auto node = elements[v][e];
                    if (new_idx[node] == -1) {
                        new_idx[node] = next_node_id++;
                    }
                }
            }
        }

        return renumber_nodes(new_idx_buff);

        // const int dim = spatial_dimension();

        // auto new_points_buff = create_host_buffer<geom_t>(dim, n_nodes);
        // auto new_points      = new_points_buff->data();

        // for (int d = 0; d < dim; d++) {
        //     for (ptrdiff_t i = 0; i < n_nodes; i++) {
        //         new_points[d][new_idx[i]] = points[d][i];
        //     }
        // }

        // impl_->points = new_points_buff;

        // for (auto &b : impl_->blocks) {
        //     auto elements   = b->elements()->data();
        //     auto n_elements = b->n_elements();

        //     for (ptrdiff_t e = 0; e < n_elements; e++) {
        //         for (int v = 0; v < nxe; v++) {
        //             elements[v][e] = new_idx[elements[v][e]];
        //         }
        //     }
        // }

        // return SFEM_SUCCESS;
    }

    int Mesh::renumber_nodes(const SharedBuffer<idx_t> &node_mapping) {
        const int dim = spatial_dimension();
        const int nxe = n_nodes_per_element();
        const ptrdiff_t n_nodes = this->n_nodes();
        const ptrdiff_t n_elements = this->n_elements();

        auto points     = this->points()->data();
        auto elements   = this->elements()->data();
        auto new_points_buff = create_host_buffer<geom_t>(dim, n_nodes);
        auto new_points      = new_points_buff->data();

        auto d_node_mapping = node_mapping->data();

        for (int d = 0; d < dim; d++) {
            for (ptrdiff_t i = 0; i < n_nodes; i++) {
                new_points[d][d_node_mapping[i]] = points[d][i];
            }
        }

        impl_->points = new_points_buff;

        for (auto &b : impl_->blocks) {
            auto elements   = b->elements()->data();
            auto n_elements = b->n_elements();

            for (ptrdiff_t e = 0; e < n_elements; e++) {
                for (int v = 0; v < nxe; v++) {
                    elements[v][e] = d_node_mapping[elements[v][e]];
                }
            }
        }

        return SFEM_SUCCESS;
    }

    std::vector<std::pair<block_idx_t, SharedBuffer<element_idx_t>>> Mesh::select_elements(
            const std::function<bool(const geom_t, const geom_t, const geom_t)> &selector,
            const std::vector<std::string>                                      &block_names) {
        SFEM_TRACE_SCOPE("Sideset::create_from_selector");

        // const ptrdiff_t nelements = mesh->n_elements();
        const ptrdiff_t nnodes = n_nodes();
        const int       dim    = spatial_dimension();

        auto points = this->points()->data();
        int  nxe    = n_nodes_per_element();

        size_t                                                           n_blocks = this->n_blocks();
        std::vector<std::pair<block_idx_t, SharedBuffer<element_idx_t>>> selected_elements;

        for (size_t b = 0; b < n_blocks; b++) {
            auto block = this->block(b);
            if (!block_names.empty() &&  //
                std::find(block_names.begin(), block_names.end(), block->name()) == block_names.end()) {
                continue;
            }

            const ptrdiff_t nelements = block->n_elements();
            auto            elements  = block->elements()->data();

            std::list<element_idx_t> selected_element_list;
            for (ptrdiff_t e = 0; e < nelements; e++) {
                // Barycenter of element
                double p[3] = {0, 0, 0};

                for (int v = 0; v < nxe; v++) {
                    const idx_t node = elements[v][e];

                    for (int d = 0; d < dim; d++) {
                        p[d] += points[d][node];
                    }
                }

                for (int d = 0; d < dim; d++) {
                    p[d] /= nxe;
                }

                if (selector(p[0], p[1], p[2])) {
                    selected_element_list.push_back(e);
                }
            }

            const ptrdiff_t nselected_elements = selected_element_list.size();
            auto            selected_element   = create_host_buffer<element_idx_t>(nselected_elements);

            {
                ptrdiff_t idx = 0;
                for (auto p : selected_element_list) {
                    selected_element->data()[idx++] = p;
                }
            }

            selected_elements.push_back(std::make_pair(b, selected_element));
        }

        return selected_elements;
    }

    void Mesh::reorder_elements_from_tags(const SharedBuffer<idx_t> &tags) {
        const ptrdiff_t nelems = n_elements();
        auto            temp   = create_host_buffer<idx_t>(nelems);
        auto            d_temp = temp->data();
        auto            d_tags = tags->data();

        auto d_elements = elements()->data();

        idx_t ntags = 0;
        for (ptrdiff_t i = 0; i < nelems; i++) {
            ntags = MAX(ntags, d_tags[i]);
        }

        if (ntags) return;

        ntags += 1;

        auto bookkeeping = create_host_buffer<ptrdiff_t>(ntags);
        auto d_bk        = bookkeeping->data();

        int nxe = n_nodes_per_element();
        for (int d = 0; d < nxe; d++) {
            memcpy(d_temp, d_elements[d], nelems * sizeof(idx_t));

            for (ptrdiff_t i = 0; i < nelems; i++) {
                auto t                   = d_tags[i];
                d_elements[d][d_bk[t]++] = d_temp[i];
            }
        }
    }

    std::shared_ptr<Mesh> Mesh::clone() const {
        auto ret = std::make_shared<Mesh>();
        SFEM_IMPLEMENT_ME();
        return ret;
    }
}  // namespace sfem
