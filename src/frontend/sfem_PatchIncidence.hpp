#pragma once

#include <cstdint>
#include <memory>

#include "sfem_ForwardDeclarations.hpp"
#include "sfem_aliases.hpp"
#include "sfem_base.hpp"

namespace sfem {

    /// The elements incident on each node, with the node's slot in each.
    ///
    /// A node-centric patch kernel walks one node's incident elements and
    /// contracts against the basis function belonging to that node.  The
    /// adjacency alone is not enough for that: the node sits at a different
    /// local slot in each element, and the kernel has to know which, because
    /// it presents the element with that slot brought to the front so the
    /// basis function is the same one in every SIMD lane.
    ///
    /// `element_local` is the slot, and it is the only thing here that
    /// `Mesh::node_to_element_graph` does not already carry.  It is a separate
    /// array rather than a search inside the kernel because the search is over
    /// the element's connectivity and would be a serial scan in the middle of
    /// a vectorised loop.
    struct PatchIncidence {
        //! Offsets into the two arrays below, one per node plus a terminator.
        SharedBuffer<count_t> node_ptr;
        //! The incident elements, grouped by node.
        SharedBuffer<element_idx_t> element;
        //! Which local slot of that element holds the node.
        SharedBuffer<uint8_t> element_local;

        ptrdiff_t n_nodes() const;
        ptrdiff_t n_incidences() const;
    };

    /// Build the incidence for one block of a mesh.
    ///
    /// One block, because the patch kernels it serves refuse a multi-block
    /// space: a node on a block boundary has incident elements outside its
    /// block, so the sum over `element` would be incomplete and the square a
    /// patch kernel takes would be of a partial residual.
    std::shared_ptr<PatchIncidence> build_patch_incidence(const std::shared_ptr<Mesh> &mesh,
                                                          const int block = 0);

}  // namespace sfem
