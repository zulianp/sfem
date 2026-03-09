#ifndef SFEM_SEMISTRUCTUREDMESH_HPP
#define SFEM_SEMISTRUCTUREDMESH_HPP

#include "sfem_ForwardDeclarations.hpp"
#include "sfem_defs.hpp"

#include "smesh_mesh.hpp"
#include "smesh_path.hpp"
#include "smesh_semistructured.hpp"
#include "smesh_sshex8_graph.hpp"

#include <vector>

namespace sfem {

    // inline std::shared_ptr<Mesh> to_semi_structured(const std::shared_ptr<Mesh> &macro_mesh,
    //                                                 const int                    level,
    //                                                 const bool                   hierarchical_ordering = false,
    //                                                 const bool                   use_gll               = false) {
    //     return smesh::to_semistructured(level, macro_mesh, hierarchical_ordering, use_gll);
    // }

    inline int semi_structured_level(const Mesh &mesh) { return smesh::proteus_hex_micro_elements_per_dim(mesh.element_type(0)); }

    inline idx_t **semi_structured_element_data(Mesh &mesh) { return mesh.elements(0)->data(); }

    inline geom_t **semi_structured_point_data(Mesh &mesh) { return mesh.points()->data(); }

    inline ptrdiff_t semi_structured_interior_start(const Mesh &mesh) {
        const int       level                  = semi_structured_level(mesh);
        const ptrdiff_t n_interior_per_element = level > 1 ? static_cast<ptrdiff_t>(level - 1) * (level - 1) * (level - 1) : 0;
        return mesh.n_nodes() - mesh.n_elements() * n_interior_per_element;
    }

    inline std::vector<int> semi_structured_derefinement_levels(const Mesh &mesh) {
        const int        level   = semi_structured_level(mesh);
        const int        nlevels = smesh::sshex8_hierarchical_n_levels(level);
        std::vector<int> levels(nlevels);
        smesh::sshex8_hierarchical_mesh_levels(level, nlevels, levels.data());
        return levels;
    }

    // inline int semi_structured_apply_hierarchical_renumbering(Mesh &mesh) {
    //     return smesh::semistructured_hierarchical_renumbering(
    //             mesh.element_type(0), semi_structured_level(mesh), mesh.n_nodes(), mesh.elements(0), mesh.points());
    // }

    inline std::shared_ptr<Mesh> semi_structured_derefine(const std::shared_ptr<Mesh> &mesh, const int to_level) {
        return smesh::derefine(mesh, to_level);
    }

    inline int semi_structured_export_as_standard(const std::shared_ptr<Mesh> &mesh, const char *path) {
        auto standard_mesh = smesh::sshex_to_hex8(mesh);
        if (!standard_mesh) {
            return SFEM_FAILURE;
        }

        return standard_mesh->write(smesh::Path(path));
    }

}  // namespace sfem

#endif  // SFEM_SEMISTRUCTUREDMESH_HPP
