#ifndef CVFEM_ELEMENT_COLORING_HPP
#define CVFEM_ELEMENT_COLORING_HPP

// Element-level colouring: two elements get different colours if they share any node.
//
// This is the colouring the GPU assembly actually needs, and it is NOT the same thing as
// the pack colouring in cvfem_pack_coloring.hpp. Pack colouring removes races *between*
// packs, which is sufficient on a CPU where a pack is one thread. On a GPU a pack is a
// block of many threads, so the race between two elements of the same pack survives.
// Colouring the elements themselves removes both.
//
// With this, an assembly kernel writes the matrix with a plain += instead of atomicAdd:
// within one colour no two elements touch a common node, so no two threads can target
// the same block of the matrix.

#include <algorithm>
#include <vector>

#include "smesh_mesh.hpp"

struct ElementColoring {
    std::vector<int32_t>   element_order;   // elements grouped by colour
    std::vector<ptrdiff_t> color_ptr;       // colour c = element_order[color_ptr[c] .. c+1)
    int                    n_colors{0};
    ptrdiff_t              min_per_color{0}, max_per_color{0};
};

// `elems[a][e]` is the global node id of local node a of element e.
template <typename IdxT>
static ElementColoring cvfem_build_element_coloring(const ptrdiff_t nelements,
                                                    const ptrdiff_t nnodes,
                                                    IdxT **const elems,
                                                    const int nodes_per_element = 8) {
    // node -> elements, in CSR form.
    std::vector<ptrdiff_t> n2e_ptr((size_t)nnodes + 1, 0);
    for (ptrdiff_t e = 0; e < nelements; ++e)
        for (int a = 0; a < nodes_per_element; ++a) n2e_ptr[(size_t)elems[a][e] + 1]++;
    for (ptrdiff_t i = 0; i < nnodes; ++i) n2e_ptr[(size_t)i + 1] += n2e_ptr[(size_t)i];
    std::vector<int32_t>   n2e((size_t)n2e_ptr[nnodes]);
    std::vector<ptrdiff_t> fill = n2e_ptr;
    for (ptrdiff_t e = 0; e < nelements; ++e)
        for (int a = 0; a < nodes_per_element; ++a) n2e[(size_t)fill[(size_t)elems[a][e]]++] = (int32_t)e;

    // BALANCED greedy colouring, visiting elements in element order.
    //
    // Two choices, and they pull in different directions.
    //
    // The visit order stays ascending. Element order is already SFC-ordered by the time this is
    // called, so neighbours are visited close together, the colour count stays near the lower
    // bound, and -- the reason it matters for a CPU sweep -- the elements that end up sharing a
    // colour remain strided through the curve rather than scattered. Sorting by degree first, as
    // the pack colouring does (cvfem_pack_coloring.hpp), would give up exactly that locality;
    // packs are few and already large, so it costs them nothing and costs elements a great deal.
    //
    // The colour CHOICE is the pack colouring's: the least loaded feasible colour, not the lowest
    // feasible index. Every colour is a parallel region ending in a barrier, so the classic
    // lowest-index rule -- which leaves the late colours nearly empty -- pays a full barrier for
    // a handful of elements. Balancing costs no extra colours and is what makes a measurement
    // against the balanced pack colouring a comparison of strategies rather than of imbalance.
    std::vector<int>       color((size_t)nelements, -1);
    std::vector<char>      used;
    std::vector<ptrdiff_t> count;
    int                    n_colors = 0;
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        used.assign((size_t)n_colors, 0);
        for (int a = 0; a < nodes_per_element; ++a) {
            const ptrdiff_t nd = elems[a][e];
            for (ptrdiff_t j = n2e_ptr[(size_t)nd]; j < n2e_ptr[(size_t)nd + 1]; ++j) {
                const int c = color[(size_t)n2e[(size_t)j]];
                if (c >= 0 && c < (int)used.size()) used[(size_t)c] = 1;
            }
        }
        int chosen = -1;
        for (int c = 0; c < n_colors; ++c) {
            if (used[(size_t)c]) continue;
            if (chosen < 0 || count[(size_t)c] < count[(size_t)chosen]) chosen = c;
        }
        if (chosen < 0) {
            chosen = n_colors++;
            count.push_back(0);
        }
        color[(size_t)e] = chosen;
        count[(size_t)chosen]++;
    }

    ElementColoring out;
    out.n_colors = n_colors;
    out.color_ptr.assign((size_t)n_colors + 1, 0);
    for (ptrdiff_t e = 0; e < nelements; ++e) out.color_ptr[(size_t)color[(size_t)e] + 1]++;
    out.min_per_color = out.color_ptr[1];
    out.max_per_color = out.color_ptr[1];
    for (int c = 0; c < n_colors; ++c) {
        out.min_per_color = std::min(out.min_per_color, out.color_ptr[(size_t)c + 1]);
        out.max_per_color = std::max(out.max_per_color, out.color_ptr[(size_t)c + 1]);
    }
    for (int c = 0; c < n_colors; ++c) out.color_ptr[(size_t)c + 1] += out.color_ptr[(size_t)c];
    out.element_order.resize((size_t)nelements);
    std::vector<ptrdiff_t> pos(out.color_ptr.begin(), out.color_ptr.end());
    for (ptrdiff_t e = 0; e < nelements; ++e)
        out.element_order[(size_t)pos[(size_t)color[(size_t)e]]++] = (int32_t)e;
    return out;
}

// Apply the colouring as an element RENUMBERING: new element i is old element
// `element_order[i]`. After this the elements of colour c occupy the contiguous range
// [color_ptr[c], color_ptr[c+1]) and nothing has to carry the order array at all.
//
// This is what lets an element-coloured CPU sweep reuse the atomic sweep unchanged. Driven off
// `element_order` the sweep would read scattered element ids, and the geometry gather
// (gather_hex8_adj_soa) -- ten memcpys over a contiguous element range -- would have to become
// ten per-lane scalar reads. That cost belongs to the indirection, not to element colouring, and
// a comparison meant to isolate the scatter strategy should not be paying it.
//
// CALL THIS BEFORE ANYTHING ELEMENT-INDEXED IS BUILT. The adjugate and determinant arrays, the
// Rhie-Chow surface tables and the partially assembled tangent are all indexed by element, and
// permuting the connectivity underneath them is the producer/consumer mistake this spike has
// already made three times with the node numbering. Renumber first, derive afterwards -- the
// same rule the packed format states for nodes.
template <typename IdxT>
static void cvfem_apply_element_coloring(const ElementColoring &ec, const ptrdiff_t nelements,
                                         IdxT **const elems, const int nodes_per_element = 8) {
    if ((ptrdiff_t)ec.element_order.size() != nelements) return;
    std::vector<IdxT> scratch((size_t)nelements);
    for (int a = 0; a < nodes_per_element; ++a) {
        for (ptrdiff_t i = 0; i < nelements; ++i)
            scratch[(size_t)i] = elems[a][ec.element_order[(size_t)i]];
        for (ptrdiff_t i = 0; i < nelements; ++i) elems[a][i] = scratch[(size_t)i];
    }
}

#endif  // CVFEM_ELEMENT_COLORING_HPP
