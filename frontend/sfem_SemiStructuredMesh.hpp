#ifndef SFEM_SEMISTRUCTUREDMESH_HPP
#define SFEM_SEMISTRUCTUREDMESH_HPP

#include "sfem_ForwardDeclarations.hpp"
#include "sfem_defs.hpp"

#include "smesh_mesh.hpp"
#include "smesh_path.hpp"
#include "smesh_semistructured.hpp"
#include "smesh_sshex8_graph.hpp"

// #include <vector>

// namespace sfem {

//     inline std::vector<int> semi_structured_derefinement_levels(const Mesh &mesh) {
//         const int        level   = semi_structured_level(mesh);
//         const int        nlevels = smesh::sshex8_hierarchical_n_levels(level);
//         std::vector<int> levels(nlevels);
//         smesh::sshex8_hierarchical_mesh_levels(level, nlevels, levels.data());
//         return levels;
//     }

//     // inline int semi_structured_apply_hierarchical_renumbering(Mesh &mesh) {
//     //     return smesh::semistructured_hierarchical_renumbering(
//     //             mesh.element_type(0), semi_structured_level(mesh), mesh.n_nodes(), mesh.elements(0), mesh.points());
//     // }

//     inline std::shared_ptr<Mesh> semi_structured_derefine(const std::shared_ptr<Mesh> &mesh, const int to_level) {
//         return smesh::derefine(mesh, to_level);
//     }

//     inline int semi_structured_export_as_standard(const std::shared_ptr<Mesh> &mesh, const char *path) {
//         auto standard_mesh = smesh::sshex_to_hex8(mesh);
//         if (!standard_mesh) {
//             return SFEM_FAILURE;
//         }

//         return standard_mesh->write(smesh::Path(path));
//     }

// }  // namespace sfem

#endif  // SFEM_SEMISTRUCTUREDMESH_HPP
