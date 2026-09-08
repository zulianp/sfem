#include "sfem_FunctionSpace.hpp"

#include <stddef.h>
#include <cstddef>
#include <memory>

#include "sfem_aliases.hpp"
#include "smesh_mesh.hpp"
#include "smesh_packed_mesh.hpp"

namespace sfem {

    class FunctionSpace::Impl {
    public:
        std::shared_ptr<Mesh> mesh;
        int                   block_size{1};
        // Multi-block support: dedicated element type for each block
        std::vector<smesh::ElemType> element_types;

        ptrdiff_t nlocal{0};
        ptrdiff_t nowned{0};
        ptrdiff_t nglobal{0};

        // CRS graph
        std::shared_ptr<CRSGraph>              node_to_node_graph;
        std::shared_ptr<CRSGraph>              dof_to_dof_graph;
        std::shared_ptr<sfem::Buffer<idx_t *>> device_elements;
        std::shared_ptr<FunctionSpace::PackedMesh> packed_mesh;

        ~Impl() {}

        // Helper method to get element type for a specific block
        smesh::ElemType get_element_type_for_block(int block) const {
            if (block < 0 || block >= static_cast<int>(element_types.size()) || element_types.empty()) {
                // Fallback to default element type
                return smesh::INVALID;
            }
            return element_types[block];
        }

        // Helper method to initialize element types from mesh blocks
        void initialize_element_types() {
            if (!mesh) return;

            size_t n_blocks = mesh->n_blocks();
            if (n_blocks > 0) {
                element_types.clear();
                element_types.reserve(n_blocks);

                for (size_t i = 0; i < n_blocks; ++i) {
                    auto block = mesh->block(i);
                    if (block) {
                        element_types.push_back(block->element_type());
                    } else {
                        // Fallback to default element type
                        element_types.push_back(smesh::INVALID);
                    }
                }
            }
        }

        void override_element_types(const smesh::ElemType element_type) {
            size_t n_blocks = mesh->n_blocks();
            if (n_blocks > 0) {
                element_types.clear();
                element_types.reserve(n_blocks);

                for (size_t i = 0; i < n_blocks; ++i) {
                    element_types.push_back(element_type);
                }
            }
        }

        void initialize_dof_counts() {
            if (!mesh) {
                nlocal  = 0;
                nowned  = 0;
                nglobal = 0;
                return;
            }

            const ptrdiff_t bs = block_size;
            if (mesh->is_distributed()) {
                auto dist = mesh->distributed();
                nlocal    = dist->n_nodes_local() * bs;
                nowned    = dist->n_nodes_owned() * bs;
                nglobal   = dist->n_nodes_global() * bs;
            } else {
                nlocal = nowned = nglobal = mesh->n_nodes() * bs;
            }
        }

        static bool mesh_all_semistructured(const Mesh &m) {
            if (m.n_blocks() == 0) {
                return false;
            }
            for (size_t b = 0; b < m.n_blocks(); ++b) {
                if (!smesh::is_semistructured_type(m.element_type(static_cast<smesh::block_idx_t>(b)))) {
                    return false;
                }
            }
            return true;
        }

        static bool mesh_all_proteus_hex8(const Mesh &m) {
            if (m.n_blocks() == 0) {
                return false;
            }
            for (size_t b = 0; b < m.n_blocks(); ++b) {
                if (m.element_type(static_cast<smesh::block_idx_t>(b)) != smesh::PROTEUS_HEX8) {
                    return false;
                }
            }
            return true;
        }

        int initialize_dof_to_dof_graph(const int block_size) {
            if (mesh && mesh_all_semistructured(*mesh)) {
                if (!node_to_node_graph) {
                    node_to_node_graph = mesh->node_to_node_graph();
                }
                if (block_size == 1) {
                    dof_to_dof_graph = node_to_node_graph;
                } else if (!dof_to_dof_graph) {
                    dof_to_dof_graph = node_to_node_graph->block_to_scalar(block_size);
                }
                return SFEM_SUCCESS;
            }

            // This is for nodal discretizations (CG)
            if (!node_to_node_graph) {
                bool types_match_mesh = true;
                for (size_t b = 0; b < element_types.size(); ++b) {
                    if (element_types[b] != mesh->element_type(static_cast<smesh::block_idx_t>(b))) {
                        types_match_mesh = false;
                        break;
                    }
                }
                if (types_match_mesh) {
                    node_to_node_graph = mesh->node_to_node_graph();
                } else {
                    node_to_node_graph = mesh->create_node_to_node_graph(
                            static_cast<smesh::ElemType>(get_element_type_for_block(0)));
                }
            }

            if (block_size == 1) {
                dof_to_dof_graph = node_to_node_graph;
            } else {
                if (!dof_to_dof_graph) {
                    dof_to_dof_graph = node_to_node_graph->block_to_scalar(block_size);
                }
            }

            return SFEM_SUCCESS;
        }
    };

    void FunctionSpace::set_device_elements(const std::shared_ptr<sfem::Buffer<idx_t *>> &elems) {
        impl_->device_elements = elems;
    }

    std::shared_ptr<sfem::Buffer<idx_t *>> FunctionSpace::device_elements() { return impl_->device_elements; }

    std::shared_ptr<CRSGraph> FunctionSpace::dof_to_dof_graph() {
        impl_->initialize_dof_to_dof_graph(this->block_size());
        return impl_->dof_to_dof_graph;
    }

    std::shared_ptr<CRSGraph> FunctionSpace::node_to_node_graph() {
        impl_->initialize_dof_to_dof_graph(this->block_size());

        return impl_->node_to_node_graph;
    }

    smesh::ElemType FunctionSpace::element_type(const int block) const { return impl_->get_element_type_for_block(block); }

    std::shared_ptr<FunctionSpace> FunctionSpace::derefine(const int to_level) {
        if (!has_semi_structured_mesh()) {
            SMESH_ERROR("Cannot derefine mesh!\n");
            return nullptr;
        }

        auto derefined_mesh = smesh::derefine(impl_->mesh, to_level);
        if (!derefined_mesh) {
            SMESH_ERROR("FunctionSpace::derefine: smesh::derefine failed\n");
            return nullptr;
        }

        // Homogeneous HEX SS at level 1 becomes HEX8. Mixed HEX+TET stays SS (B5.6).
        if (Impl::mesh_all_proteus_hex8(*derefined_mesh)) {
            derefined_mesh = smesh::sshex_to_hex8(derefined_mesh);
            if (!derefined_mesh) {
                SMESH_ERROR("FunctionSpace::derefine: sshex_to_hex8 failed\n");
                return nullptr;
            }
        }

        return std::make_shared<FunctionSpace>(derefined_mesh, impl_->block_size);
    }

    FunctionSpace::FunctionSpace() : impl_(std::make_unique<Impl>()) {}

    std::shared_ptr<FunctionSpace> FunctionSpace::create(const std::shared_ptr<FunctionSpace::PackedMesh> &mesh, const int block_size) {
        auto ret               = std::make_shared<FunctionSpace>();
        ret->impl_->mesh       = mesh->mesh();
        ret->impl_->block_size = block_size;
        ret->impl_->packed_mesh = mesh;
        ret->impl_->initialize_element_types();
        ret->impl_->initialize_dof_counts();
        return ret;
    }

    FunctionSpace::FunctionSpace(const std::shared_ptr<Mesh> &mesh, const int block_size, const smesh::ElemType element_type)
        : impl_(std::make_unique<Impl>()) {
        impl_->mesh       = mesh;
        impl_->block_size = block_size;
        assert(block_size > 0);

        if (element_type == smesh::INVALID) {
            impl_->initialize_element_types();
        } else {
            impl_->override_element_types(element_type);
        }

        impl_->initialize_dof_counts();
    }
    FunctionSpace::~FunctionSpace() = default;

    bool FunctionSpace::has_semi_structured_mesh() const {
        return impl_->mesh && Impl::mesh_all_semistructured(*impl_->mesh);
    }

    Mesh &FunctionSpace::mesh() { return *impl_->mesh; }

    std::shared_ptr<Mesh> FunctionSpace::mesh_ptr() const { return impl_->mesh; }

    int FunctionSpace::block_size() const { return impl_->block_size; }

    ptrdiff_t FunctionSpace::n_dofs() const { return impl_->nlocal; }

    ptrdiff_t FunctionSpace::n_owned_dofs() const { return impl_->nowned; }

    ptrdiff_t FunctionSpace::n_dofs_global() const { return impl_->nglobal; }

    SharedBuffer<geom_t *> FunctionSpace::points() { return impl_->mesh->points(); }

    std::shared_ptr<FunctionSpace> FunctionSpace::lor() const {
        auto ret = std::make_shared<FunctionSpace>(impl_->mesh, impl_->block_size);
        for (size_t i = 0; i < ret->impl_->element_types.size(); ++i) {
            const auto t = ret->impl_->element_types[i];
            if (t == smesh::TET10 || t == smesh::TRI6) {
                ret->impl_->element_types[i] = macro_type_variant(t);
            }
        }
        return ret;
    }

    int FunctionSpace::create_vector(ptrdiff_t *nlocal, ptrdiff_t *nglobal, real_t **values) {
        *nlocal  = impl_->nlocal;
        *nglobal = impl_->nglobal;
        *values  = (real_t *)malloc(sizeof(real_t) * impl_->nlocal);
        return SFEM_SUCCESS;
    }

    int FunctionSpace::destroy_vector(real_t *values) {
        free(values);
        return SFEM_SUCCESS;
    }

    // Helper method to get number of blocks (for internal use)
    size_t FunctionSpace::n_blocks() const {
        assert(mesh_ptr()->n_blocks() == impl_->element_types.size());
        return impl_->element_types.size();
    }

    // Helper method to check if this is a multi-block function space
    bool FunctionSpace::is_multi_block() const { return impl_->mesh && impl_->mesh->n_blocks() > 1; }

    std::vector<smesh::ElemType> FunctionSpace::element_types() const { return impl_->element_types; }

    int FunctionSpace::initialize_packed_mesh() {
        impl_->packed_mesh = FunctionSpace::PackedMesh::create(impl_->mesh, {}, true);
        return SFEM_SUCCESS;
    }

    bool FunctionSpace::has_packed_mesh() const { return static_cast<bool>(impl_->packed_mesh); }

    std::shared_ptr<FunctionSpace::PackedMesh> FunctionSpace::packed_mesh() { return impl_->packed_mesh; }
}  // namespace sfem

