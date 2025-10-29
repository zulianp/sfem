#ifndef SFEM_FUNCTIONSPACE_HPP
#define SFEM_FUNCTIONSPACE_HPP

#include "sfem_ForwardDeclarations.hpp"
#include "sfem_defs.h"
<<<<<<< HEAD
=======
#include "sfem_Buffer.hpp"
>>>>>>> origin/main

#include <memory>
#include <string>
#include <vector>

namespace sfem {

    class FunctionSpace final {
    public:
<<<<<<< HEAD
=======
        using PackedIdxType = uint16_t;
        // using PackedIdxType = uint8_t;
        using PackedMesh = Packed<PackedIdxType>;

>>>>>>> origin/main
        FunctionSpace(const std::shared_ptr<Mesh> &mesh, const int block_size = 1, const enum ElemType element_type = INVALID);
        ~FunctionSpace();

        int promote_to_semi_structured(const int level);

        static std::shared_ptr<FunctionSpace> create(const std::shared_ptr<Mesh> &mesh,
                                                     const int                    block_size   = 1,
                                                     const enum ElemType          element_type = INVALID) {
            return std::make_shared<FunctionSpace>(mesh, block_size, element_type);
        }

        static std::shared_ptr<FunctionSpace> create(const std::shared_ptr<SemiStructuredMesh> &mesh, const int block_size = 1);
<<<<<<< HEAD
=======
        static std::shared_ptr<FunctionSpace> create(const std::shared_ptr<PackedMesh> &mesh, const int block_size = 1);
>>>>>>> origin/main

        int create_vector(ptrdiff_t *nlocal, ptrdiff_t *nglobal, real_t **values);
        int destroy_vector(real_t *values);

        // Multi-block support methods (for internal use)
        size_t n_blocks() const;
        bool is_multi_block() const;

        void                                   set_device_elements(const std::shared_ptr<sfem::Buffer<idx_t *>> &elems);
        std::shared_ptr<sfem::Buffer<idx_t *>> device_elements();

        Mesh                 &mesh();
        std::shared_ptr<Mesh> mesh_ptr() const;

        bool                has_semi_structured_mesh() const;
        SemiStructuredMesh &semi_structured_mesh();

<<<<<<< HEAD
=======
        
        // using PackedIdxType = uint8_t;
        int initialize_packed_mesh();
        bool has_packed_mesh() const;
        std::shared_ptr<PackedMesh> packed_mesh();

        SharedBuffer<geom_t *> points();

>>>>>>> origin/main
        int       block_size() const;
        ptrdiff_t n_dofs() const;

        enum ElemType element_type(const int block = 0) const;
        std::vector<enum ElemType> element_types() const;

        std::shared_ptr<FunctionSpace> derefine(const int to_level = 1);
        std::shared_ptr<FunctionSpace> lor() const;

        std::shared_ptr<CRSGraph> dof_to_dof_graph();
        std::shared_ptr<CRSGraph> node_to_node_graph();

        friend class Op;

        // private
        FunctionSpace();

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

}  // namespace sfem

#endif  // SFEM_FUNCTIONSPACE_HPP
