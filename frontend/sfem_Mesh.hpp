#ifndef SFEM_MESH_HPP
#define SFEM_MESH_HPP

// C includes
#include "sfem_base.h"
#include "sfem_defs.h"

// Deprecated
#include "sfem_mesh.h"

// C++ includes
#include "sfem_Buffer.hpp"
#include "sfem_Communicator.hpp"
#include "sfem_ForwardDeclarations.hpp"

// External
#include <mpi.h>

// STL
#include <functional>

namespace sfem {
    class Mesh final {
    public:
        class Block {
        public:
            Block();
            ~Block();

            const std::string           &name() const;
            enum ElemType                element_type() const;
            const SharedBuffer<idx_t *> &elements() const;

            // Setters for internal use
            void set_name(const std::string &name);
            void set_element_type(enum ElemType element_type);
            void set_elements(SharedBuffer<idx_t *> elements);
            ptrdiff_t n_elements() const;

        private:
            class Impl;
            std::unique_ptr<Impl> impl_;
        };

        Mesh();
        Mesh(const std::shared_ptr<Communicator> &comm);
        ~Mesh();

        Mesh(const std::shared_ptr<Communicator> &comm,
             int                                  spatial_dim,
             enum ElemType                        element_type,
             ptrdiff_t                            nelements,
             SharedBuffer<idx_t *>                elements,
             ptrdiff_t                            nnodes,
             SharedBuffer<geom_t *>               points);

        friend class FunctionSpace;
        friend class Op;
        // friend class NeumannConditions;

        int                                        read(const char *path);
        int                                        write(const char *path) const;
        int                                        initialize_node_to_node_graph();
        int                                        convert_to_macro_element_mesh();
        const std::vector<std::shared_ptr<Block>> &blocks() const;

        // Block-related methods
        size_t                       n_blocks() const;
        std::shared_ptr<const Block> block(size_t index) const;
        std::shared_ptr<Block>       block(size_t index);
        void add_block(const std::string &name, enum ElemType element_type, SharedBuffer<idx_t *> elements);
        void remove_block(size_t index);

        int           spatial_dimension() const;
        int           n_nodes_per_element() const;
        ptrdiff_t     n_nodes() const;
        ptrdiff_t     n_elements() const;
        enum ElemType element_type() const;
        ptrdiff_t     n_owned_nodes() const;
        ptrdiff_t     n_owned_nodes_with_ghosts() const;
        ptrdiff_t     n_owned_elements() const;
        ptrdiff_t     n_owned_elements_with_ghosts() const;
        ptrdiff_t     n_shared_elements() const;

        std::shared_ptr<CRSGraph>   node_to_node_graph();
        std::shared_ptr<CRSGraph>   node_to_node_graph_upper_triangular();
        SharedBuffer<element_idx_t> half_face_table();
        std::shared_ptr<CRSGraph>   create_node_to_node_graph(const enum ElemType element_type);

        SharedBuffer<count_t> node_to_node_rowptr() const;
        SharedBuffer<idx_t>   node_to_node_colidx() const;
        SharedBuffer<idx_t>   node_offsets() const;
        SharedBuffer<idx_t>   ghosts() const;
        SharedBuffer<int>     node_owner() const;
        SharedBuffer<idx_t>   node_mapping() const;
        SharedBuffer<idx_t>   element_mapping() const;

        const geom_t *const points(const int coord) const;
        const idx_t *const  idx(const int node_num) const;

        SharedBuffer<geom_t *> points();
        SharedBuffer<idx_t *>  elements();
        SharedBuffer<idx_t *>  default_elements();  // For backward compatibility

        std::shared_ptr<Communicator> comm() const;

        inline static std::shared_ptr<Mesh> create_from_file(const std::shared_ptr<Communicator> &comm, const char *path) {
            auto ret = std::make_shared<Mesh>(comm);
            ret->read(path);
            return ret;
        }

        static std::shared_ptr<Mesh> create_hex8_reference_cube();

        static std::shared_ptr<Mesh> create_hex8_cube(const std::shared_ptr<Communicator> &comm,
                                                      const int                            nx   = 1,
                                                      const int                            ny   = 1,
                                                      const int                            nz   = 1,
                                                      const geom_t                         xmin = 0,
                                                      const geom_t                         ymin = 0,
                                                      const geom_t                         zmin = 0,
                                                      const geom_t                         xmax = 1,
                                                      const geom_t                         ymax = 1,
                                                      const geom_t                         zmax = 1);

        static std::shared_ptr<Mesh> create_tri3_square(const std::shared_ptr<Communicator> &comm,
                                                        const int                            nx   = 1,
                                                        const int                            ny   = 1,
                                                        const geom_t                         xmin = 0,
                                                        const geom_t                         ymin = 0,
                                                        const geom_t                         xmax = 1,
                                                        const geom_t                         ymax = 1);

        static std::shared_ptr<Mesh> create_quad4_square(const std::shared_ptr<Communicator> &comm,
                                                         const int                            nx   = 1,
                                                         const int                            ny   = 1,
                                                         const geom_t                         xmin = 0,
                                                         const geom_t                         ymin = 0,
                                                         const geom_t                         xmax = 1,
                                                         const geom_t                         ymax = 1);

        static std::shared_ptr<Mesh> create_hex8_checkerboard_cube(const std::shared_ptr<Communicator> &comm,
                                                                   const int                            nx   = 2,
                                                                   const int                            ny   = 2,
                                                                   const int                            nz   = 2,
                                                                   const geom_t                         xmin = 0,
                                                                   const geom_t                         ymin = 0,
                                                                   const geom_t                         zmin = 0,
                                                                   const geom_t                         xmax = 1,
                                                                   const geom_t                         ymax = 1,
                                                                   const geom_t                         zmax = 1);

        void set_node_mapping(const SharedBuffer<idx_t> &node_mapping);
        void set_comm(const std::shared_ptr<Communicator> &comm);
        void set_element_type(const enum ElemType element_type);

        void extract_deprecated(mesh_t *mesh);

        std::pair<SharedBuffer<geom_t>, SharedBuffer<geom_t>> compute_bounding_box();

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
}  // namespace sfem

#endif  // SFEM_MESH_HPP
