#ifndef SFEM_FUNCTIONSPACE_HPP
#define SFEM_FUNCTIONSPACE_HPP

#include "sfem_ForwardDeclarations.hpp"
#include "sfem_defs.hpp"
#include "sfem_aliases.hpp"
#include "sfem_execution_context.hpp"
#include "smesh_semistructured.hpp"

#include "smesh_packed_mesh.hpp"

#include <memory>
#include <string>
#include <vector>

namespace smesh {
    class Exchange;
}

namespace sfem {

    class FunctionSpace final {
    public:
        using PackedIdxType = uint16_t;
        // using PackedIdxType = uint8_t;
        using PackedMesh = smesh::PackedMesh<PackedIdxType>;

        FunctionSpace(const std::shared_ptr<Mesh> &mesh, const int block_size = 1, const smesh::ElemType element_type = smesh::INVALID);
        ~FunctionSpace();

        static std::shared_ptr<FunctionSpace> create(const std::shared_ptr<Mesh> &mesh,
                                                     const int                    block_size   = 1,
                                                     const smesh::ElemType          element_type = smesh::INVALID) {
            return std::make_shared<FunctionSpace>(mesh, block_size, element_type);
        }

        // static std::shared_ptr<FunctionSpace> create(const std::shared_ptr<SemiStructuredMesh> &mesh, const int block_size = 1);
        static std::shared_ptr<FunctionSpace> create(const std::shared_ptr<PackedMesh> &mesh, const int block_size = 1);

        int create_vector(ptrdiff_t *nlocal, ptrdiff_t *nglobal, real_t **values);
        int destroy_vector(real_t *values);

        // Multi-block support methods (for internal use)
        size_t n_blocks() const;
        bool is_multi_block() const;

        std::shared_ptr<Mesh> mesh_ptr() const;

        bool                has_semi_structured_mesh() const;

        
        // using PackedIdxType = uint8_t;
        int initialize_packed_mesh();
        bool has_packed_mesh() const;

        /**
         * @brief old node id -> new node id for the packed renumbering, or null
         *
         * Building the packed mesh renumbers this space's mesh in place. Anything
         * holding node ids from before -- a nodeset read from a file speaks the
         * on-disk numbering -- has to come through here, or it names different
         * nodes and says nothing about it.
         */
        SharedBuffer<idx_t> packed_node_map() const;
        std::shared_ptr<PackedMesh> packed_mesh();

        SharedBuffer<geom_t *> points();

        int       block_size() const;
        ptrdiff_t n_dofs() const;
        ptrdiff_t n_dofs_owned() const;
        ptrdiff_t n_dofs_global() const;

        /// The owned prefix of x (n_dofs() long) copied into its ghost and aura slots, from the
        /// ranks that own them: what every evaluation over all local elements reads. Nothing on a
        /// space that is not distributed. x lives in the context's space (on the device: GPU-aware
        /// MPI, or NCCL within a node), and on the device the copies are ordered on its stream.
        int synchronize(real_t *x, const ExecutionContext &context = ExecutionContext::host());
        /// synchronize in two halves: begin sends x's owned values, end receives the ghost and
        /// aura ones into x. Between them x's owned values must stay as they are; computation that
        /// reads only those can run while the messages travel.
        int synchronize_begin(real_t *x, const ExecutionContext &context = ExecutionContext::host());
        int synchronize_end(const ExecutionContext &context = ExecutionContext::host());
        /// The reverse of synchronize: x's ghost and aura slots added to the owned entries they
        /// copy, on the ranks that own them, for a sum over elements that only one rank holds.
        /// Nothing on a space that is not distributed.
        int add_to_owners(real_t *x, ExecutionSpace es = EXECUTION_SPACE_HOST);

        smesh::ElemType element_type(const int block = 0) const;
        std::vector<smesh::ElemType> element_types() const;

        std::shared_ptr<FunctionSpace> derefine(const int to_level = 1);
        std::shared_ptr<FunctionSpace> lor() const;

        std::shared_ptr<CRSGraph> dof_to_dof_graph();
        std::shared_ptr<CRSGraph> node_to_node_graph();
        /// How a semi-structured mesh is discretised. MICRO: each macro element as its micro elements,
        /// trilinear -- what SFEM's semi-structured kernels and solvers (SSGMG, SSMGC) mean, and the
        /// default. HIGH_ORDER: one Lagrange element of the mesh's level per macro element, which the
        /// generated operators also offer (PROTEUS_HEX(L+1)^3). Hand-written semi-structured kernels
        /// are micro only.
        enum class SemiStructuredBasis { MICRO, HIGH_ORDER };
        void                set_semi_structured_basis(SemiStructuredBasis basis);
        SemiStructuredBasis semi_structured_basis() const;
        /// Whether operators see this space's semi-structured domains as their micro elements.
        bool micro_elements() const;

        /// Every two nodes of an element of the basis coupled: the graph a block matrix on this space
        /// fits. node_to_node_graph, except on a HIGH_ORDER semi-structured space, where the element
        /// is the macro element and node_to_node_graph couples only the nodes of a micro element.
        std::shared_ptr<CRSGraph> element_node_graph();

        friend class Op;

        // private
        FunctionSpace();

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;

        // synchronize's exchange for buffers in `es`; null where there is nothing to exchange.
        smesh::Exchange *exchange_for(ExecutionSpace es, int *status);
    };

}  // namespace sfem

#endif  // SFEM_FUNCTIONSPACE_HPP
