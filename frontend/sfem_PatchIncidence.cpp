#include "sfem_PatchIncidence.hpp"

#include "sfem_API.hpp"
#include "smesh_mesh.hpp"

#include <limits>

namespace sfem {

    ptrdiff_t PatchIncidence::n_nodes() const {
        return node_ptr ? (ptrdiff_t)node_ptr->size() - 1 : 0;
    }

    ptrdiff_t PatchIncidence::n_incidences() const {
        return element ? (ptrdiff_t)element->size() : 0;
    }

    std::shared_ptr<PatchIncidence> build_patch_incidence(const std::shared_ptr<Mesh> &mesh,
                                                          const int block) {
        if (!mesh) {
            SFEM_ERROR("build_patch_incidence: no mesh\n");
        }
        if (mesh->n_blocks() != 1) {
            SFEM_ERROR(
                    "build_patch_incidence: the patch kernels this serves refuse a "
                    "multi-block space, because a node on a block boundary has incident "
                    "elements outside its block; got %zu blocks\n",
                    mesh->n_blocks());
        }

        auto graph = mesh->node_to_element_graph();
        if (!graph) {
            SFEM_ERROR("build_patch_incidence: the mesh has no node-to-element graph\n");
        }

        const ptrdiff_t n_nodes = graph->n_nodes();
        const ptrdiff_t nnz     = graph->nnz();
        const count_t *const rowptr = graph->rowptr()->data();
        const idx_t *const colidx = graph->colidx()->data();

        auto out           = std::make_shared<PatchIncidence>();
        out->node_ptr      = create_host_buffer<count_t>(n_nodes + 1);
        out->element       = create_host_buffer<element_idx_t>(nnz);
        out->element_local = create_host_buffer<uint8_t>(nnz);

        const auto  elements            = mesh->elements(block)->data();
        const int   n_nodes_per_element = mesh->block(block)->n_nodes_per_element();
        if (n_nodes_per_element > (int)std::numeric_limits<uint8_t>::max()) {
            SFEM_ERROR("build_patch_incidence: %d nodes per element does not fit a slot byte\n",
                       n_nodes_per_element);
        }

        for (ptrdiff_t node = 0; node <= n_nodes; ++node) {
            out->node_ptr->data()[node] = rowptr[node];
        }

#pragma omp parallel for
        for (ptrdiff_t node = 0; node < n_nodes; ++node) {
            for (count_t k = rowptr[node]; k < rowptr[node + 1]; ++k) {
                const element_idx_t e = (element_idx_t)colidx[k];
                out->element->data()[k] = e;

                // Which slot of `e` is this node.  Searched once, here, rather
                // than inside the kernel: the scan is serial and would sit in
                // the middle of a vectorised lane loop.
                int slot = -1;
                for (int j = 0; j < n_nodes_per_element; ++j) {
                    if ((ptrdiff_t)elements[j][e] == node) {
                        slot = j;
                        break;
                    }
                }
                if (slot < 0) {
                    // The graph said this element touches the node and its
                    // connectivity disagrees, so one of the two is wrong and
                    // the kernel would contract against the wrong basis
                    // function without ever saying so.
                    SFEM_ERROR(
                            "build_patch_incidence: node %td is not in element %td, which the "
                            "node-to-element graph says it touches\n",
                            node,
                            (ptrdiff_t)e);
                }
                out->element_local->data()[k] = (uint8_t)slot;
            }
        }

        return out;
    }

}  // namespace sfem
