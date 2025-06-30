#include "sfem_FunctionSpace.hpp"

#include <stddef.h>
#include <cstddef>
#include <memory>

#include "sfem_prolongation_restriction.h"

#include "sfem_Buffer.hpp"
#include "sfem_CRSGraph.hpp"
#include "sfem_Mesh.hpp"
#include "sfem_SemiStructuredMesh.hpp"

namespace sfem {

    class FunctionSpace::Impl {
    public:
        std::shared_ptr<Mesh> mesh;
        int                   block_size{1};
        // Multi-block support: dedicated element type for each block
        std::vector<enum ElemType> element_types;

        // Number of nodes of function-space (TODO)
        ptrdiff_t nlocal{0};
        ptrdiff_t nglobal{0};

        // CRS graph
        std::shared_ptr<CRSGraph>              node_to_node_graph;
        std::shared_ptr<CRSGraph>              dof_to_dof_graph;
        std::shared_ptr<sfem::Buffer<idx_t *>> device_elements;

        // Data-structures for semistructured mesh
        std::shared_ptr<SemiStructuredMesh> semi_structured_mesh;

        ~Impl() {}

        // Helper method to get element type for a specific block
        enum ElemType get_element_type_for_block(int block) const {
            if (block < 0 || block >= static_cast<int>(element_types.size()) || element_types.empty()) {
                // Fallback to default element type
                return INVALID;
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
                        element_types.push_back(INVALID);
                    }
                }
            }
        }

        void override_element_types(const enum ElemType element_type) {
            size_t n_blocks = mesh->n_blocks();
            if (n_blocks > 0) {
                element_types.clear();
                element_types.reserve(n_blocks);
                
                for (size_t i = 0; i < n_blocks; ++i) {
                    element_types.push_back(element_type);
                }
            }
        }

        int initialize_dof_to_dof_graph(const int block_size) {
            if (semi_structured_mesh) {
                // printf("SemiStructuredMesh::node_to_node_graph (in FunctionSpace)\n");
                if (!node_to_node_graph) {
                    node_to_node_graph = semi_structured_mesh->node_to_node_graph();
                }
                // FIXME
                dof_to_dof_graph = node_to_node_graph;
                return SFEM_SUCCESS;
            }

            // This is for nodal discretizations (CG)
            if (!node_to_node_graph) {
                // Use the default element type for graph creation
                node_to_node_graph = mesh->create_node_to_node_graph(get_element_type_for_block(0));
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

    enum ElemType FunctionSpace::element_type(const int block) const {
        return impl_->get_element_type_for_block(block);
    }

    std::shared_ptr<FunctionSpace> FunctionSpace::derefine(const int to_level) {
        if (to_level == 1) {
            // FIXME the number of nodes in mesh does not change, will lead to bugs
            return std::make_shared<FunctionSpace>(impl_->mesh, impl_->block_size, macro_base_elem(impl_->get_element_type_for_block(0)));
        }

        assert(has_semi_structured_mesh());
        return create(semi_structured_mesh().derefine(to_level), block_size());
    }

    FunctionSpace::FunctionSpace() : impl_(std::make_unique<Impl>()) {}

    std::shared_ptr<FunctionSpace> FunctionSpace::create(const std::shared_ptr<SemiStructuredMesh> &mesh, const int block_size) {
        auto ret                         = std::make_shared<FunctionSpace>();
        ret->impl_->mesh                 = mesh->macro_mesh();
        ret->impl_->block_size           = block_size;
        ret->impl_->semi_structured_mesh = mesh;
        ret->impl_->nlocal               = mesh->n_nodes() * block_size;
        ret->impl_->nglobal              = ret->impl_->nlocal;
        
        ret->impl_->element_types.push_back(SSHEX8);

        // TODO: Once multi-block support is available in the semistructured mesh,
        // Initialize element types for multi-block support
        // ret->impl_->initialize_element_types();
        
        return ret;
    }

    FunctionSpace::FunctionSpace(const std::shared_ptr<Mesh> &mesh, const int block_size, const enum ElemType element_type)
        : impl_(std::make_unique<Impl>()) {
        impl_->mesh       = mesh;
        impl_->block_size = block_size;
        assert(block_size > 0);

        if (element_type == INVALID) {
            impl_->initialize_element_types();

        } else {
            impl_->override_element_types(element_type);
        }

        if (element_type == INVALID) {
            impl_->nlocal  = mesh->n_nodes() * block_size;
            impl_->nglobal = mesh->n_nodes() * block_size;
        } else {
            assert(mesh->n_blocks() == 1);
            // FIXME in parallel it will not work
            impl_->nlocal  = (max_node_id(impl_->get_element_type_for_block(0), mesh->n_elements(), mesh->elements()->data()) + 1) * block_size;
            impl_->nglobal = impl_->nlocal;
        }
        
        // Initialize element types for multi-block support
        
    }

    int FunctionSpace::promote_to_semi_structured(const int level) {
        // Check if we have a multi-block mesh
        if (impl_->mesh->n_blocks() > 1) {
            // For multi-block meshes, we need to check if all blocks are compatible
            // For now, we'll only promote if the default element type is HEX8
            if (impl_->get_element_type_for_block(0) == HEX8) {
                impl_->semi_structured_mesh = std::make_shared<SemiStructuredMesh>(impl_->mesh, level);
                impl_->override_element_types(SSHEX8);
                impl_->nlocal               = impl_->semi_structured_mesh->n_nodes() * impl_->block_size;
                impl_->nglobal              = impl_->nlocal;

                return SFEM_SUCCESS;
            }
            return SFEM_FAILURE;
        } else {
            // Single block mesh - original behavior
            if (impl_->get_element_type_for_block(0) == HEX8) {
                impl_->semi_structured_mesh = std::make_shared<SemiStructuredMesh>(impl_->mesh, level);
                impl_->override_element_types(SSHEX8);
                impl_->nlocal               = impl_->semi_structured_mesh->n_nodes() * impl_->block_size;
                impl_->nglobal              = impl_->nlocal;
                                
                return SFEM_SUCCESS;
            }
        }

        return SFEM_FAILURE;
    }

    FunctionSpace::~FunctionSpace() = default;

    bool FunctionSpace::has_semi_structured_mesh() const { return static_cast<bool>(impl_->semi_structured_mesh); }

    Mesh &FunctionSpace::mesh() { return *impl_->mesh; }

    std::shared_ptr<Mesh> FunctionSpace::mesh_ptr() const { return impl_->mesh; }

    SemiStructuredMesh &FunctionSpace::semi_structured_mesh() { return *impl_->semi_structured_mesh; }

    int FunctionSpace::block_size() const { return impl_->block_size; }

    ptrdiff_t FunctionSpace::n_dofs() const { return impl_->nlocal; }

    std::shared_ptr<FunctionSpace> FunctionSpace::lor() const {
        return std::make_shared<FunctionSpace>(impl_->mesh, impl_->block_size, macro_type_variant(impl_->get_element_type_for_block(0)));
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
    bool FunctionSpace::is_multi_block() const {
        return impl_->mesh && impl_->mesh->n_blocks() > 1;
    }

} // namespace sfem 
