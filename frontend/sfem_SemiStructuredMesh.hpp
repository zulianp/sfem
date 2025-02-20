#ifndef SFEM_SEMISTRUCTUREDMESH_HPP
#define SFEM_SEMISTRUCTUREDMESH_HPP

// C Includes
#include "sfem_base.h"

// C++ Includes
#include "sfem_ForwardDeclarations.hpp"
#include "sfem_Buffer.hpp"

#include <memory>

namespace sfem {

    class SemiStructuredMesh {
    public:
        idx_t   **element_data();
        geom_t  **point_data();
        ptrdiff_t interior_start() const;

        SemiStructuredMesh();
        SemiStructuredMesh(const std::shared_ptr<Mesh> macro_mesh, const int level);
        ~SemiStructuredMesh();

        std::shared_ptr<CRSGraph> node_to_node_graph();

        static std::shared_ptr<SemiStructuredMesh> create(const std::shared_ptr<Mesh> macro_mesh, const int level) {
            return std::make_shared<SemiStructuredMesh>(macro_mesh, level);
        }

        std::vector<int> derefinement_levels();
        int              apply_hierarchical_renumbering();

        int       n_nodes_per_element() const;
        ptrdiff_t n_nodes() const;
        int       level() const;
        ptrdiff_t n_elements() const;

        std::shared_ptr<SemiStructuredMesh> derefine(const int to_level);

        std::shared_ptr<Buffer<geom_t *>> points();
        std::shared_ptr<Buffer<idx_t *>>  elements();

        int export_as_standard(const char *path);
        int write(const char *path);

        std::shared_ptr<Mesh> macro_mesh();

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

}  // namespace sfem

#endif  // SFEM_SEMISTRUCTUREDMESH_HPP
