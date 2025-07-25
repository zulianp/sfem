#include "sfem_SemiStructuredMesh.hpp"

// C Includes
#include "line_quadrature_gauss_lobatto.h"
#include "sfem_macros.h"

// Mesh
#include "adj_table.h"
#include "sfem_hex8_mesh_graph.h"
#include "sshex8.h"
#include "sshex8_mesh.h"

// C++ Includes
#include "sfem_CRSGraph.hpp"
#include "sfem_Mesh.hpp"
#include "sfem_Tracer.hpp"
#include "sfem_glob.hpp"

// STL
#include <fstream>
#include <sstream>

namespace sfem {

    class SemiStructuredMesh::Impl {
    public:
        std::shared_ptr<Mesh> macro_mesh;
        int                   level;

        ptrdiff_t                 n_unique_nodes{-1}, interior_start{-1};
        std::shared_ptr<CRSGraph> node_to_node_graph;

        std::shared_ptr<Buffer<geom_t *>>   points;
        std::vector<std::shared_ptr<Block>> blocks;

        void init(const std::shared_ptr<Mesh> macro_mesh, const int level) {
            SFEM_TRACE_SCOPE("SemiStructuredMesh::init");

            this->macro_mesh = macro_mesh;
            this->level      = level;
            blocks.clear();
            points.reset();

            if (macro_mesh->n_blocks() == 1) {
                auto block = macro_mesh->block(0);

                auto default_block = std::make_shared<Block>();
                default_block->set_name(block->name());

                // FIXME hardcoded for sshex8
                default_block->set_element_type(SSHEX8);

                const int nxe            = sshex8_nxe(level);
                auto      micro_elements = create_host_buffer<idx_t>(nxe, macro_mesh->n_elements());
                default_block->set_elements(micro_elements);

                auto elements = micro_elements->data();
                sshex8_generate_elements(level,
                                         macro_mesh->n_elements(),
                                         macro_mesh->n_nodes(),
                                         macro_mesh->elements()->data(),
                                         elements,
                                         &this->n_unique_nodes,
                                         &this->interior_start);

                blocks.push_back(default_block);

            } else {
                SFEM_ERROR("SemiStructuredMesh is not supported for multi-block meshes");
            }
        }

        const SharedBuffer<idx_t *> &default_elements() const { return blocks[0]->elements(); }
    };

    std::shared_ptr<Mesh> SemiStructuredMesh::macro_mesh() { return impl_->macro_mesh; }

    size_t SemiStructuredMesh::n_blocks() const { return impl_->blocks.size(); }
    std::vector<std::shared_ptr<SemiStructuredMesh ::Block>> &SemiStructuredMesh::blocks() { return impl_->blocks; }
    std::shared_ptr<SemiStructuredMesh::Block>                SemiStructuredMesh::block(const block_idx_t block_id) {
        return impl_->blocks[block_id];
    }

    std::vector<int> SemiStructuredMesh::derefinement_levels() {
        const int        L       = level();
        const int        nlevels = sshex8_hierarchical_n_levels(L);
        std::vector<int> levels(nlevels);
        sshex8_hierarchical_mesh_levels(L, nlevels, levels.data());
        return levels;
    }

    std::shared_ptr<SemiStructuredMesh> SemiStructuredMesh::derefine(const int to_level) {
        const int from_level  = this->level();
        const int step_factor = from_level / to_level;
        const int nxe         = (to_level + 1) * (to_level + 1) * (to_level + 1);

        auto ret               = std::make_shared<SemiStructuredMesh>();
        ret->impl_->macro_mesh = this->impl_->macro_mesh;
        ret->impl_->level      = to_level;

        ptrdiff_t n_unique_nodes{-1};
        for (auto &block : this->blocks()) {
            auto elements = block->elements();
            auto view     = std::make_shared<Buffer<idx_t *>>(
                    nxe,
                    n_elements(),
                    (idx_t **)malloc(nxe * sizeof(idx_t *)),
                    [keep_alive = elements](int, void **v) {
                        (void)keep_alive;
                        free(v);
                    },
                    elements->mem_space());

            for (int zi = 0; zi <= to_level; zi++) {
                for (int yi = 0; yi <= to_level; yi++) {
                    for (int xi = 0; xi <= to_level; xi++) {
                        const int from_lidx   = sshex8_lidx(from_level, xi * step_factor, yi * step_factor, zi * step_factor);
                        const int to_lidx     = sshex8_lidx(to_level, xi, yi, zi);
                        view->data()[to_lidx] = elements->data()[from_lidx];
                    }
                }
            }

            auto derefined_block = std::make_shared<Block>();
            derefined_block->set_name(block->name());
            derefined_block->set_element_type(block->element_type());
            derefined_block->set_elements(view);
            ret->impl_->blocks.push_back(derefined_block);

            // Find max node id
            {
                auto            vv        = view->data();
                const ptrdiff_t nelements = block->n_elements();
                for (int v = 0; v < view->extent(0); v++) {
                    for (ptrdiff_t e = 0; e < nelements; e++) {
                        n_unique_nodes = MAX(vv[v][e], n_unique_nodes);
                    }
                }
            }
        }

        n_unique_nodes += 1;

        ret->impl_->n_unique_nodes = n_unique_nodes;
        ret->impl_->interior_start = this->impl_->interior_start;

        if (this->impl_->points) {
            int sdim           = this->impl_->macro_mesh->spatial_dimension();
            ret->impl_->points = sfem::view(this->impl_->points, 0, sdim, 0, n_unique_nodes);
        }

        return ret;
    }

    int SemiStructuredMesh::apply_hierarchical_renumbering() {
        if (n_blocks() > 1) {
            SFEM_ERROR("Applying hierarchical renumbering is not supported for multi-block meshes");
        }

        const int L = level();

        const int nlevels = sshex8_hierarchical_n_levels(L);

        std::vector<int> levels(nlevels);

        // FiXME harcoded for sshex8
        sshex8_hierarchical_mesh_levels(L, nlevels, levels.data());

        return sshex8_hierarchical_renumbering(L,
                                               nlevels,
                                               levels.data(),
                                               this->n_elements(),
                                               this->impl_->n_unique_nodes,
                                               this->impl_->default_elements()->data());
    }

    std::shared_ptr<Buffer<geom_t *>> SemiStructuredMesh::points() {
        if (!impl_->points) {
            auto p       = sfem::create_host_buffer<geom_t>(impl_->macro_mesh->spatial_dimension(), impl_->n_unique_nodes);
            auto macro_p = impl_->macro_mesh->points()->data();

            SFEM_TRACE_SCOPE("sshex8_fill_points");

            int SFEM_USE_GLL = 0;
            SFEM_READ_ENV(SFEM_USE_GLL, atoi);

            if (SFEM_USE_GLL) {
                const scalar_t *qx{nullptr};
                switch (level()) {
                    case 1: {
                        qx = line_GL_q2_x;
                        break;
                    }
                    case 2: {
                        qx = line_GL_q3_x;
                        break;
                    }
                    case 4: {
                        qx = line_GL_q5_x;
                        break;
                    }
                    case 8: {
                        qx = line_GL_q9_x;
                        break;
                    }
                    case 16: {
                        qx = line_GL_q17_x;
                        break;
                    }
                    default: {
                        SFEM_ERROR("Unsupported order %d!", level());
                    }
                }

                sshex8_fill_points_1D_map(level(), n_elements(), element_data(), macro_p, qx, p->data());
            } else {
                sshex8_fill_points(level(), n_elements(), element_data(), macro_p, p->data());
            }

            impl_->points = p;
        }

        return impl_->points;
    }

    std::shared_ptr<Buffer<idx_t *>> SemiStructuredMesh::elements() { return impl_->default_elements(); }

    std::shared_ptr<CRSGraph> SemiStructuredMesh::node_to_node_graph() {
        // printf("SemiStructuredMesh::node_to_node_graph\n");
        if (impl_->node_to_node_graph) {
            return impl_->node_to_node_graph;
        }

        if (n_blocks() > 1) {
            SFEM_ERROR("Node-to-node graph is not supported for multi-block meshes");
        }

        SFEM_TRACE_SCOPE("SemiStructuredMesh::node_to_node_graph");

        count_t *rowptr{nullptr};
        idx_t   *colidx{nullptr};

        sshex8_crs_graph(impl_->level, this->n_elements(), this->n_nodes(), this->element_data(), &rowptr, &colidx);

        impl_->node_to_node_graph =
                std::make_shared<CRSGraph>(Buffer<count_t>::own(this->n_nodes() + 1, rowptr, free, MEMORY_SPACE_HOST),
                                           Buffer<idx_t>::own(rowptr[this->n_nodes()], colidx, free, MEMORY_SPACE_HOST));

        return impl_->node_to_node_graph;
    }

    int SemiStructuredMesh::n_nodes_per_element() const { return sshex8_nxe(impl_->level); }

    idx_t   **SemiStructuredMesh::element_data() { return impl_->default_elements()->data(); }
    geom_t  **SemiStructuredMesh::point_data() { return impl_->macro_mesh->points()->data(); }
    ptrdiff_t SemiStructuredMesh::interior_start() const { return impl_->interior_start; }

    SemiStructuredMesh::SemiStructuredMesh(const std::shared_ptr<Mesh> macro_mesh, const int level)
        : impl_(std::make_unique<Impl>()) {
        impl_->init(macro_mesh, level);
    }

    SemiStructuredMesh::SemiStructuredMesh() : impl_(std::make_unique<Impl>()) {}

    SemiStructuredMesh::~SemiStructuredMesh() {}

    ptrdiff_t SemiStructuredMesh::n_nodes() const { return impl_->n_unique_nodes; }
    int       SemiStructuredMesh::level() const { return impl_->level; }
    ptrdiff_t SemiStructuredMesh::n_elements() const { return impl_->macro_mesh->n_elements(); }

    int SemiStructuredMesh::export_as_standard(const char *path) {
        SFEM_TRACE_SCOPE("SemiStructuredMesh::export_as_standard");

        if (n_blocks() > 1) {
            SFEM_ERROR("Exporting multi-block meshes is not supported");
        }

        sfem::create_directory(path);

        std::string folder   = path;
        auto        elements = impl_->default_elements();
        auto        points   = this->points();

        const int txe              = sshex8_txe(this->level());
        ptrdiff_t n_micro_elements = this->n_elements() * txe;
        auto      hex8_elements    = create_host_buffer<idx_t>(8, n_micro_elements);

        sshex8_to_standard_hex8_mesh(level(), n_elements(), elements->data(), hex8_elements->data());

        // hex8_elements->print(std::cout);
        // points->print(std::cout);

        std::string points_path = folder + "/x%d.raw";
        points->to_files(points_path.c_str());

        std::string elements_path = folder + "/i%d.raw";
        hex8_elements->to_files(elements_path.c_str());

        std::stringstream ss;
        ss << "# SFEM mesh meta file (generated by SemiStructuredMesh::export_as_standard)\n";
        ss << "n_elements: " << hex8_elements->extent(1) << "\n";
        ss << "n_nodes: " << points->extent(1) << "\n";
        ss << "spatial_dimension: 3\n";
        ss << "elem_num_nodes: 8\n";
        ss << "element_type: HEX8\n";
        ss << "points:\n";
        ss << "- x: x0.raw\n";
        ss << "- y: x1.raw\n";
        ss << "- z: x2.raw\n";
        ss << "rpath: true\n";

        const double mem_hex8_mesh   = hex8_elements->extent(0) * hex8_elements->extent(1) * sizeof(idx_t) * 1e-9;
        const double mem_sshex8_mesh = elements->extent(0) * elements->extent(1) * sizeof(idx_t) * 1e-9;
        const double mem_points      = points->extent(0) * points->extent(1) * sizeof(geom_t) * 1e-9;
        const double mem_macro_points =
                impl_->macro_mesh->points()->extent(0) * impl_->macro_mesh->points()->extent(1) * sizeof(geom_t) * 1e-9;

        ss << "mem_hex8_mesh:    " << mem_hex8_mesh << " [GB]\n";
        ss << "mem_sshex8_mesh:  " << mem_sshex8_mesh << " [GB]\n";
        ss << "mem_points:       " << mem_points << " [GB]\n";
        ss << "mem_macro_points: " << mem_macro_points << " [GB]\n";
        ss << "mem_disc_ss:      " << (mem_points + mem_macro_points) << " [GB]\n";
        ss << "mem_disc_std:     " << (mem_hex8_mesh + mem_sshex8_mesh) << " [GB]\n";
        ss << "n_macro_elements: " << elements->extent(1) << "\n";

        std::string   meta_path = folder + "/meta.yaml";
        std::ofstream os(meta_path.c_str());

        if (!os.good()) {
            return SFEM_FAILURE;
        }

        os << ss.str();
        os.close();

        return SFEM_SUCCESS;
    }

    int SemiStructuredMesh::write(const char *path) {
        SFEM_TRACE_SCOPE("SemiStructuredMesh::write");

        if (n_blocks() > 1) {
            SFEM_ERROR("Writing multi-block meshes is not supported");
        }

        sfem::create_directory(path);

        std::string folder   = path;
        auto        elements = impl_->default_elements();
        auto        points   = this->points();

        std::string points_path = folder + "/x%d.raw";
        points->to_files(points_path.c_str());

        std::string elements_path = folder + "/i%d.raw";
        elements->to_files(elements_path.c_str());

        std::stringstream ss;
        ss << "# SFEM mesh meta file (generated by SemiStructuredMesh::write)\n";
        ss << "level: " << this->level() << "\n";
        ss << "n_elements: " << elements->extent(1) << "\n";
        ss << "n_nodes: " << points->extent(1) << "\n";
        ss << "spatial_dimension: 3\n";
        ss << "elem_num_nodes: " << ((level() - 1) * (level() - 1) * (level() - 1)) << "\n";
        ss << "element_type: SSHEX8\n";
        ss << "points:\n";
        ss << "- x: x0.raw\n";
        ss << "- y: x1.raw\n";
        ss << "- z: x2.raw\n";
        ss << "rpath: true\n";

        std::string   meta_path = folder + "/meta.yaml";
        std::ofstream os(meta_path.c_str());

        if (!os.good()) {
            return SFEM_FAILURE;
        }

        os << ss.str();
        os.close();
        return SFEM_SUCCESS;
    }

}  // namespace sfem
