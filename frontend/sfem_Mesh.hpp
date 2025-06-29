#ifndef SFEM_MESH_HPP
#define SFEM_MESH_HPP

// C includes
#include "sfem_base.h"
#include "sfem_defs.h"

// C++ includes
#include "sfem_Buffer.hpp"
#include "sfem_ForwardDeclarations.hpp"

// External
#include <mpi.h>

// STL
#include <functional>

namespace sfem {
    class Mesh final {
    public:
        Mesh();
        Mesh(MPI_Comm comm);
        ~Mesh();

        Mesh(MPI_Comm                    comm,
             int                         spatial_dim,
             enum ElemType               element_type,
             ptrdiff_t                   nelements,
             SharedBuffer<idx_t *>       elements,
             ptrdiff_t                   nnodes,
             SharedBuffer<geom_t *>      points);

        friend class FunctionSpace;
        friend class Op;
        // friend class NeumannConditions;

        int read(const char *path);
        int write(const char *path) const;
        int initialize_node_to_node_graph();
        int convert_to_macro_element_mesh();

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

        std::shared_ptr<CRSGraph>              node_to_node_graph();
        std::shared_ptr<CRSGraph>              node_to_node_graph_upper_triangular();
        std::shared_ptr<Buffer<element_idx_t>> half_face_table();
        std::shared_ptr<CRSGraph>              create_node_to_node_graph(const enum ElemType element_type);

        std::shared_ptr<Buffer<count_t>> node_to_node_rowptr() const;
        std::shared_ptr<Buffer<idx_t>>   node_to_node_colidx() const;

        const geom_t *const points(const int coord) const;
        const idx_t *const  idx(const int node_num) const;

        std::shared_ptr<Buffer<geom_t *>> points();
        std::shared_ptr<Buffer<idx_t *>>  elements();

        // void *impl_mesh();

        MPI_Comm comm() const;

        inline static std::shared_ptr<Mesh> create_from_file(MPI_Comm comm, const char *path) {
            auto ret = std::make_shared<Mesh>(comm);
            ret->read(path);
            return ret;
        }

        static std::shared_ptr<Mesh> create_hex8_reference_cube();

        static std::shared_ptr<Mesh> create_hex8_cube(MPI_Comm     comm,
                                                      const int    nx   = 1,
                                                      const int    ny   = 1,
                                                      const int    nz   = 1,
                                                      const geom_t xmin = 0,
                                                      const geom_t ymin = 0,
                                                      const geom_t zmin = 0,
                                                      const geom_t xmax = 1,
                                                      const geom_t ymax = 1,
                                                      const geom_t zmax = 1);

        static std::shared_ptr<Mesh> create_tri3_square(MPI_Comm     comm,
                                                        const int    nx   = 1,
                                                        const int    ny   = 1,
                                                        const geom_t xmin = 0,
                                                        const geom_t ymin = 0,
                                                        const geom_t xmax = 1,
                                                        const geom_t ymax = 1);

        static std::shared_ptr<Mesh> create_quad4_square(MPI_Comm     comm,
                                                         const int    nx   = 1,
                                                         const int    ny   = 1,
                                                         const geom_t xmin = 0,
                                                         const geom_t ymin = 0,
                                                         const geom_t xmax = 1,
                                                         const geom_t ymax = 1);

        void set_node_mapping(const std::shared_ptr<Buffer<idx_t>> &node_mapping);
    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
}  // namespace sfem

#endif  // SFEM_MESH_HPP
