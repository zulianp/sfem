#ifndef SFEM_CONTACT_SKIN_HPP
#define SFEM_CONTACT_SKIN_HPP

#include "smesh_mesh.hpp"

namespace sfem {
    void remove_contraints_connected_elements(const std::shared_ptr<smesh::Mesh>&       mesh,
                                              const smesh::SharedBuffer<smesh::mask_t>& constraints_mask,
                                              const int                                 block_size,
                                              const bool                                compact_nodes = false);

    std::shared_ptr<smesh::Mesh> create_contact_skin(const std::shared_ptr<smesh::Mesh>&       mesh,
                                                     const smesh::SharedBuffer<smesh::mask_t>& constraints_mask,
                                                     const bool                                compact_nodes = false);
}  // namespace sfem

#endif  // SFEM_CONTACT_SKIN_HPP
