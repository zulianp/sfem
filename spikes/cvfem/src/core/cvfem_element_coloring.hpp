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

    // Greedy colouring in element order. Element order is already SFC-ordered by the
    // time this is called, so neighbours are visited close together and the colour count
    // stays near the lower bound without a more elaborate ordering heuristic.
    std::vector<int> color((size_t)nelements, -1);
    std::vector<int> used;
    int              n_colors = 0;
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        used.assign((size_t)n_colors + 1, 0);
        for (int a = 0; a < nodes_per_element; ++a) {
            const ptrdiff_t nd = elems[a][e];
            for (ptrdiff_t j = n2e_ptr[(size_t)nd]; j < n2e_ptr[(size_t)nd + 1]; ++j) {
                const int c = color[(size_t)n2e[(size_t)j]];
                if (c >= 0 && c < (int)used.size()) used[(size_t)c] = 1;
            }
        }
        int c = 0;
        while (c < (int)used.size() && used[(size_t)c]) ++c;
        color[(size_t)e] = c;
        if (c >= n_colors) n_colors = c + 1;
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

#endif  // CVFEM_ELEMENT_COLORING_HPP
