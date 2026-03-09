#include "sfem_DualGraph.hpp"

// C includes


// C++ includes
#include "smesh_mesh.hpp"

#include "sfem_defs.hpp"


namespace sfem {
    class DualGraph::Impl {
    public:
        SharedBuffer<count_t>       adj_ptr;
        SharedBuffer<element_idx_t> adj_idx;
    };

    DualGraph::DualGraph() : impl_(std::make_unique<Impl>()) {}
    DualGraph::~DualGraph() {}

    SharedBuffer<count_t> DualGraph::adj_ptr() { return impl_->adj_ptr; }

    SharedBuffer<element_idx_t> DualGraph::adj_idx() { return impl_->adj_idx; }

    std::shared_ptr<DualGraph> DualGraph::create(const std::shared_ptr<Mesh> &mesh) {
        auto ret = std::make_shared<DualGraph>();

        const ptrdiff_t     n_elements            = mesh->n_elements();
        const ptrdiff_t     n_nodes               = mesh->n_nodes();
        const smesh::ElemType element_type          = mesh->element_type(0);
        smesh::ElemType element_type_for_algo = element_type;

        auto elems = mesh->elements(0)->data();
        if (element_type == smesh::TET10) {
            element_type_for_algo = smesh::TET4;
        } else if (element_type == smesh::TRI6) {
            element_type_for_algo = smesh::TRI3;
        }

        count_t       *adj_ptr = 0;
        element_idx_t *adj_idx = 0;
        smesh::create_dual_graph(n_elements, n_nodes, element_type_for_algo, elems, &adj_ptr, &adj_idx);

        ret->impl_->adj_ptr = manage_host_buffer<count_t>(n_elements + 1, adj_ptr);
        ret->impl_->adj_idx = manage_host_buffer<element_idx_t>(adj_ptr[n_elements], adj_idx);

        return ret;
    }
}  // namespace sfem
