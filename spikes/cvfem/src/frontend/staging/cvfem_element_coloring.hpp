#ifndef CVFEM_ELEMENT_COLORING_HPP
#define CVFEM_ELEMENT_COLORING_HPP

// Element-level colouring: two elements get different colours if they share any node.
//
// The colouring itself, the balanced greedy that builds it and the renumbering that groups the
// elements by colour all live in smesh (smesh_element_coloring.hpp). It started here and moved
// there once the layout measured well, so production codes can use it; what remains in this file
// is the plain-array view the sweeps below want and the one piece of plumbing that is specific
// to this spike -- which pass claims the nodal-gradient scatter.
//
// This is NOT the same thing as the pack colouring in cvfem_pack_coloring.hpp. Pack colouring
// removes races *between* packs, which is sufficient on a CPU where a pack is one thread. On a
// GPU a pack is a block of many threads, so the race between two elements of the same pack
// survives. Colouring the elements themselves removes both.

#include <memory>
#include <vector>

#include "smesh_element_coloring.hpp"
#include "smesh_mesh.hpp"

// A view of one block's colouring as plain pointers, so a sweep reads the colour ranges without
// an accessor call inside its loop. It holds the smesh object alive; the pointers belong to it.
struct ElementColoring {
    std::shared_ptr<smesh::ElementColoring> coloring;
    const ptrdiff_t                        *color_ptr{nullptr};
    const smesh::element_idx_t             *element_order{nullptr};
    int                                     n_colors{0};
    ptrdiff_t                               min_per_color{0}, max_per_color{0};
};

// Colours the mesh's block 0 and RENUMBERS its elements into colour order, so the elements of
// colour c occupy the contiguous range [color_ptr[c], color_ptr[c+1]) and nothing has to carry
// the order array at all.
//
// Renumbering rather than driving the sweep off `element_order` is what lets an element-coloured
// CPU sweep reuse the atomic sweep unchanged. Driven off the order array the sweep would read
// scattered element ids, and the geometry gather (gather_hex8_adj_soa) -- ten memcpys over a
// contiguous element range -- would have to become ten per-lane scalar reads. That cost belongs
// to the indirection, not to element colouring, and a comparison meant to isolate the scatter
// strategy should not be paying it.
//
// CALL THIS BEFORE ANYTHING ELEMENT-INDEXED IS BUILT. The adjugate and determinant arrays, the
// Rhie-Chow surface tables and the partially assembled tangent are all indexed by element, and
// permuting the connectivity underneath them is the producer/consumer mistake this spike has
// already made three times with the node numbering. Renumber first, derive afterwards -- the
// same rule the packed format states for nodes.
static inline ElementColoring cvfem_build_element_coloring(const std::shared_ptr<smesh::Mesh> &mesh,
                                                           const bool renumber = true) {
    ElementColoring out;
    out.coloring = smesh::ElementColoring::create(mesh, {}, renumber);
    if (!out.coloring || out.coloring->n_blocks() == 0) return out;
    out.n_colors      = out.coloring->n_colors(0);
    out.color_ptr     = out.coloring->color_ptr(0)->data();
    out.element_order = out.coloring->element_order(0)->data();
    out.min_per_color = out.coloring->min_elements_per_color(0);
    out.max_per_color = out.coloring->max_elements_per_color(0);
    return out;
}

// The colouring the nodal-gradient sweep should follow, or null for the atomic scatter.
//
// Set by whoever selects the layout, read by cvfem_hex8_nodal_grads_atomic_nc. It exists because
// the reconstruction is a scatter over elements like any other pass, and leaving it atomic under
// an element-coloured operator left the only nondeterminism in an otherwise reproducible sweep:
// the element kernel was bitwise identical across runs and the Rhie-Chow arm was not, because the
// gradient it reads had been accumulated in thread-arrival order.
//
// Translation-unit local, like the other sweep-selection flags in the benchmark driver: the one
// unit that sets it is the one that instantiates the sweep, and every other sees the null default
// and the atomic path.
static inline const ElementColoring *&cvfem_hex8_qgrad_ecolors_ref() {
    static const ElementColoring *p = nullptr;
    return p;
}
static inline const ElementColoring *cvfem_hex8_qgrad_ecolors() { return cvfem_hex8_qgrad_ecolors_ref(); }
static inline void cvfem_hex8_set_qgrad_ecolors(const ElementColoring *p) { cvfem_hex8_qgrad_ecolors_ref() = p; }

#endif  // CVFEM_ELEMENT_COLORING_HPP
