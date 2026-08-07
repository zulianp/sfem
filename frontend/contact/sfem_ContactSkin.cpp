#include "sfem_ContactSkin.hpp"

#include "smesh_mask.hpp"
#include "smesh_semistructured.hpp"

#include <cassert>

namespace sfem {
    void remove_contraints_connected_elements(const std::shared_ptr<smesh::Mesh>&       mesh,
                                              const smesh::SharedBuffer<smesh::mask_t>& constraints_mask,
                                              const int                                 block_size,
                                              const bool                                compact_nodes) {
        assert(mesh->node_mapping());
        assert(constraints_mask);

        const ptrdiff_t            n_nodes           = mesh->n_nodes();
        const smesh::idx_t* const  node_mapping_data = mesh->node_mapping()->data();
        const smesh::mask_t* const mask_data         = constraints_mask->data();
        std::vector<unsigned char> constrained_node(n_nodes);

        if (mesh->node_mapping()) {
            auto node_mapping = mesh->node_mapping();
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n_nodes; ++i) {
                const ptrdiff_t dof = node_mapping_data[i] * block_size;
                bool            constrained{false};
                for (int d = 0; d < block_size; ++d) {
                    constrained |= smesh::mask_get(dof + d, mask_data);
                }

                constrained_node[i] = constrained;
            }
        } else {
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n_nodes; ++i) {
                const ptrdiff_t dof = i * block_size;
                bool            constrained{false};
                for (int d = 0; d < block_size; ++d) {
                    constrained |= smesh::mask_get(dof + d, mask_data);
                }

                constrained_node[i] = constrained;
            }
        }

        for (size_t b = 0; b < mesh->n_blocks(); ++b) {
            auto                       block         = mesh->block(b);
            const int                  nxe           = block->n_nodes_per_element();
            const ptrdiff_t            n_elements    = block->n_elements();
            auto                       elements      = block->elements();
            auto                       elements_data = elements->data();
            ptrdiff_t                  n_kept        = 0;
            std::vector<unsigned char> keep(n_elements);

#pragma omp parallel for reduction(+ : n_kept)
            for (ptrdiff_t e = 0; e < n_elements; ++e) {
                bool remove{false};
                for (int v = 0; v < nxe; ++v) {
                    remove |= constrained_node[elements_data[v][e]];
                }

                keep[e] = !remove;
                n_kept += keep[e];
            }

            if (n_kept == n_elements) {
                continue;
            }

            auto filtered_elements      = smesh::create_host_buffer<smesh::idx_t>(nxe, n_kept);
            auto filtered_elements_data = filtered_elements->data();

            ptrdiff_t out = 0;
            for (ptrdiff_t e = 0; e < n_elements; ++e) {
                if (!keep[e]) {
                    continue;
                }

                for (int v = 0; v < nxe; ++v) {
                    filtered_elements_data[v][out] = elements_data[v][e];
                }

                ++out;
            }

            block->set_elements(filtered_elements);
        }

        if (!compact_nodes) {
            return;
        }

        std::vector<unsigned char> used_node(n_nodes);
        std::vector<smesh::idx_t>  old_to_new(n_nodes);
        ptrdiff_t                  n_used_nodes = 0;

        for (size_t b = 0; b < mesh->n_blocks(); ++b) {
            auto            block         = mesh->block(b);
            const int       nxe           = block->n_nodes_per_element();
            const ptrdiff_t n_elements    = block->n_elements();
            auto            elements_data = block->elements()->data();

            for (ptrdiff_t e = 0; e < n_elements; ++e) {
                for (int v = 0; v < nxe; ++v) {
                    const smesh::idx_t node = elements_data[v][e];
                    if (!used_node[node]) {
                        used_node[node]  = true;
                        old_to_new[node] = n_used_nodes++;
                    }
                }
            }
        }

        if (n_used_nodes == n_nodes) {
            return;
        }

        const int dim              = mesh->spatial_dimension();
        auto      points           = mesh->points();
        auto      points_data      = points->data();
        auto      compact_points   = smesh::create_host_buffer<smesh::geom_t>(dim, n_used_nodes);
        auto      compact_mapping  = smesh::create_host_buffer<smesh::idx_t>(n_used_nodes);
        auto      compact_p_data   = compact_points->data();
        auto      compact_map_data = compact_mapping->data();

        if (mesh->node_mapping()) {
            auto node_mapping = mesh->node_mapping();
            for (ptrdiff_t i = 0; i < n_nodes; ++i) {
                if (!used_node[i]) {
                    continue;
                }

                const smesh::idx_t new_node = old_to_new[i];
                compact_map_data[new_node]  = node_mapping_data[i];
                for (int d = 0; d < dim; ++d) {
                    compact_p_data[d][new_node] = points_data[d][i];
                }
            }
        } else {
            for (ptrdiff_t i = 0; i < n_nodes; ++i) {
                if (!used_node[i]) {
                    continue;
                }

                const smesh::idx_t new_node = old_to_new[i];
                for (int d = 0; d < dim; ++d) {
                    compact_p_data[d][new_node] = points_data[d][i];
                }
            }
        }

        for (size_t b = 0; b < mesh->n_blocks(); ++b) {
            auto            block         = mesh->block(b);
            const int       nxe           = block->n_nodes_per_element();
            const ptrdiff_t n_elements    = block->n_elements();
            auto            elements_data = block->elements()->data();

#pragma omp parallel for
            for (ptrdiff_t e = 0; e < n_elements; ++e) {
                for (int v = 0; v < nxe; ++v) {
                    elements_data[v][e] = old_to_new[elements_data[v][e]];
                }
            }
        }

        mesh->set_points(compact_points);

        if (mesh->node_mapping()) {
            mesh->set_node_mapping(compact_mapping);
        }
    }

    std::shared_ptr<smesh::Mesh> create_contact_skin(const std::shared_ptr<smesh::Mesh>&       mesh,
                                                     const smesh::SharedBuffer<smesh::mask_t>& constraints_mask,
                                                     const bool                                compact_nodes) {
        auto surface = smesh::skin(mesh);

        if (smesh::is_semistructured_type(mesh->element_type(0))) {
            surface = smesh::ssquad_to_quad4(surface);
            surface->block(0)->set_element_type(smesh::QUADSHELL4);
        }

        remove_contraints_connected_elements(surface, constraints_mask, mesh->spatial_dimension(), compact_nodes);

        return surface;
    }

}  // namespace sfem
