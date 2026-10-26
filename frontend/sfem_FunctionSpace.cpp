#include "sfem_FunctionSpace.hpp"

#include <stddef.h>
#include <cstddef>
#include <memory>

#include "sfem_aliases.hpp"
#include "smesh_exchange.hpp"
#include "smesh_mesh.hpp"
#include "smesh_packed_mesh.hpp"
#include "smesh_multiblock_graph.hpp"

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
        std::shared_ptr<CRSGraph>              element_node_graph;
        SemiStructuredBasis                    basis{SemiStructuredBasis::MICRO};
        std::shared_ptr<FunctionSpace::PackedMesh> packed_mesh;
        //: old node id -> new node id, for the renumbering `packed_mesh` applied
        //: to `mesh` in place.  Kept because nothing else records it: anything
        //: still speaking the numbering the mesh had -- a nodeset read from a
        //: file, most of all -- is unreconstructable without it.
        SharedBuffer<idx_t> packed_node_map;
        //: synchronize's exchanges, host and device, built on first use: packing renumbers the
        //: nodes they name.
        std::shared_ptr<smesh::Exchange> exchange[2];

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

    std::shared_ptr<CRSGraph> FunctionSpace::dof_to_dof_graph() {
        impl_->initialize_dof_to_dof_graph(this->block_size());
        return impl_->dof_to_dof_graph;
    }

    std::shared_ptr<CRSGraph> FunctionSpace::node_to_node_graph() {
        impl_->initialize_dof_to_dof_graph(this->block_size());

        return impl_->node_to_node_graph;
    }

    void FunctionSpace::set_semi_structured_basis(const SemiStructuredBasis basis) {
        impl_->basis              = basis;
        impl_->element_node_graph = nullptr;
    }

    FunctionSpace::SemiStructuredBasis FunctionSpace::semi_structured_basis() const { return impl_->basis; }

    bool FunctionSpace::micro_elements() const {
        return impl_->basis == SemiStructuredBasis::MICRO && has_semi_structured_mesh();
    }

    std::shared_ptr<CRSGraph> FunctionSpace::element_node_graph() {
        auto &mesh = *impl_->mesh;
        if (!Impl::mesh_all_semistructured(mesh) || impl_->basis == SemiStructuredBasis::MICRO) return node_to_node_graph();
        if (impl_->element_node_graph) return impl_->element_node_graph;

        const auto                        n_blocks = static_cast<smesh::block_idx_t>(mesh.n_blocks());
        std::vector<smesh::ElemType>      types(n_blocks);
        std::vector<ptrdiff_t>            n_elements(n_blocks);
        std::vector<const idx_t *const *> elements(n_blocks);
        for (smesh::block_idx_t b = 0; b < n_blocks; ++b) {
            types[b]      = mesh.element_type(b);
            n_elements[b] = mesh.n_elements(b);
            elements[b]   = mesh.elements(b)->data();
        }
        count_t *rowptr{nullptr};
        idx_t   *colidx{nullptr};
        if (smesh::create_multiblock_crs_graph<idx_t, count_t>(
                    n_blocks, types.data(), n_elements.data(), elements.data(), mesh.n_nodes(), &rowptr, &colidx) !=
            SFEM_SUCCESS) {
            return nullptr;
        }
        const ptrdiff_t n = mesh.n_nodes();
        impl_->element_node_graph =
                std::make_shared<CRSGraph>(smesh::Buffer<count_t>::own(n + 1, rowptr, free, smesh::MEMORY_SPACE_HOST),
                                           smesh::Buffer<idx_t>::own(rowptr[n], colidx, free, smesh::MEMORY_SPACE_HOST));
        return impl_->element_node_graph;
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

        auto ret = std::make_shared<FunctionSpace>(derefined_mesh, impl_->block_size);
        ret->set_semi_structured_basis(impl_->basis);
        return ret;
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

    const std::shared_ptr<Mesh> &FunctionSpace::mesh() const { return impl_->mesh; }

    int FunctionSpace::block_size() const { return impl_->block_size; }

    ptrdiff_t FunctionSpace::n_dofs() const { return impl_->nlocal; }

    ptrdiff_t FunctionSpace::n_dofs_owned() const { return impl_->nowned; }

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
        assert(mesh()->n_blocks() == impl_->element_types.size());
        return impl_->element_types.size();
    }

    // Helper method to check if this is a multi-block function space
    bool FunctionSpace::is_multi_block() const { return impl_->mesh && impl_->mesh->n_blocks() > 1; }

    std::vector<smesh::ElemType> FunctionSpace::element_types() const { return impl_->element_types; }

    int FunctionSpace::initialize_packed_mesh() {
        // Built twice, on purpose.  `create(..., modify_mesh=true)` renumbers the
        // mesh in place and then drops the permutation it used, so a caller
        // holding node ids in the old numbering has no way back.  A
        // non-modifying build keeps that map, is a pure function of the same
        // mesh and pack size, and therefore produces the map the modifying one
        // is about to apply.  The cost is one extra partition pass at setup,
        // against a class of bug whose symptom is a solve that converges to the
        // wrong problem in silence.
        auto probe = FunctionSpace::PackedMesh::create(impl_->mesh, {}, false);
        if (!probe) return SFEM_FAILURE;
        impl_->packed_node_map = probe->node_map();

        // The renumbering moves the mesh's node_mapping with the nodes. A mesh read whole has none;
        // give it the identity, so that a written mesh still says which node of its file each of
        // its nodes is (a block-selective read already set one).
        if (impl_->mesh->comm()->size() == 1 && !impl_->mesh->node_mapping()) {
            const ptrdiff_t n_nodes = impl_->mesh->n_nodes();
            auto            ids     = create_host_buffer<idx_t>(n_nodes);
            for (ptrdiff_t i = 0; i < n_nodes; ++i) ids->data()[i] = static_cast<idx_t>(i);
            impl_->mesh->set_node_mapping(ids);
        }

        impl_->packed_mesh = FunctionSpace::PackedMesh::create(impl_->mesh, {}, true);
        // Everything keyed by node number describes the numbering before.
        for (auto &e : impl_->exchange) e.reset();
        impl_->node_to_node_graph.reset();
        impl_->dof_to_dof_graph.reset();
        impl_->element_node_graph.reset();
        return SFEM_SUCCESS;
    }

    bool FunctionSpace::has_packed_mesh() const { return static_cast<bool>(impl_->packed_mesh); }

    smesh::Exchange *FunctionSpace::exchange_for(const ExecutionSpace es, int *const status) {
        auto &impl = *impl_;
        *status    = SFEM_SUCCESS;
        if (!impl.mesh->is_distributed() || impl.mesh->comm()->size() == 1) return nullptr;
        auto &exchange = impl.exchange[es == EXECUTION_SPACE_DEVICE ? 1 : 0];
        if (!exchange) {
            exchange = smesh::Exchange::create_nodal(impl.mesh, smesh::Exchange::ExchangeScope::GhostsAndAura, es);
            if (!exchange) *status = SFEM_FAILURE;
        }
        return exchange.get();
    }

    int FunctionSpace::synchronize(real_t *const x, const ExecutionContext &context) {
        if (synchronize_begin(x, context) != SFEM_SUCCESS) return SFEM_FAILURE;
        return synchronize_end(context);
    }

    int FunctionSpace::add_to_owners(real_t *const x, const ExecutionSpace es) {
        int  status   = SFEM_SUCCESS;
        auto exchange = exchange_for(es, &status);
        if (!exchange) return status;
        return exchange->scatter_add(x, impl_->block_size) == SMESH_SUCCESS ? SFEM_SUCCESS : SFEM_FAILURE;
    }

    int FunctionSpace::synchronize_begin(real_t *const x, const ExecutionContext &context) {
        int  status   = SFEM_SUCCESS;
        auto exchange = exchange_for(context.space(), &status);
        return exchange ? exchange->gather_begin(x, impl_->block_size, context.stream()) : status;
    }

    int FunctionSpace::synchronize_end(const ExecutionContext &context) {
        int  status   = SFEM_SUCCESS;
        auto exchange = exchange_for(context.space(), &status);
        return exchange ? exchange->gather_wait() : status;
    }

    std::shared_ptr<FunctionSpace::PackedMesh> FunctionSpace::packed_mesh() { return impl_->packed_mesh; }

    SharedBuffer<idx_t> FunctionSpace::packed_node_map() const { return impl_->packed_node_map; }
}  // namespace sfem

