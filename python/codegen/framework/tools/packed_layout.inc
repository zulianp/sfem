#pragma once
//
// The packed mesh layout, built rather than borrowed.
//
// Two consumers share this file and neither owns the other: `tools/reproducibility.py`
// compiles it into its driver, and `spikes/inexact_apply_compare` includes it to
// benchmark the packed kernels.  Both need a packed layout without linking anything --
// `smesh::PackedMesh` is the real one, but it lives behind `SFEM::sfem`, it renumbers
// the mesh it is given in place, and its `max_nodes_per_pack()` is the index type's
// ceiling (65536 for uint16_t) rather than the actual maximum, which would make every
// thread scratch slot a megabyte.  What is built here is the same contract, sized to
// the mesh it is given.
//
// It expects the including translation unit to have defined `idx_t` already; the
// mesh type is not its business.
//
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef SFEM_PACKED_LAYOUT_MIN
#define SFEM_PACKED_LAYOUT_MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
// The packed layout, built here rather than borrowed.
//
// A packed kernel takes a mesh partitioned into packs of elements, each pack
// owning a contiguous range of node ids so its gather and scatter run through
// thread-local scratch instead of through atomics.  `smesh::PackedMesh` builds
// exactly this, and calling it would be the same move as calling `tet4_fff` --
// but the driver deliberately links no library: it compiles the generated
// operators and nothing else, which is what keeps this gate cheap to run.  So
// the layout is built here, to the contract the generated kernel states:
//
//   * elements are partitioned into contiguous ranges of `n_elements_per_pack`
//   * `owned_nodes_ptr[p] .. owned_nodes_ptr[p+1]` are the global ids the pack
//     owns, and they are contiguous because the nodes are renumbered to make
//     them so
//   * the last `n_shared_nodes[p]` of those are touched by another pack too,
//     so the kernel scatters them atomically and the rest plainly
//   * `ghost_idx[ghost_ptr[p] .. ghost_ptr[p+1])` are the global ids the pack
//     touches but does not own
//   * `elements[a][e]` is a *pack-local* index: below `n_contiguous` it is an
//     owned slot, at or above it a ghost slot
//
// The renumbering means a packed kernel and an unpacked one are driven over
// differently numbered meshes.  `to_old` records the permutation so the input
// can be seeded through it, and then the two are the same problem relabelled:
// the l1 and l2 digests are permutation-invariant, so a packed kernel must
// reproduce the unpacked kernel's answer exactly.  That is the check, and it
// is stronger than running the packed kernel on its own and trusting it.
struct PackedLayout {
    //: The connectivity under the packed numbering, as global ids.  The caller
    //: rebuilds its own mesh from this and `to_old`; only it knows what else it
    //: carries per node.
    std::vector<std::vector<idx_t>> renumbered;
    std::vector<idx_t> to_new;
    ptrdiff_t n_packs = 0;
    ptrdiff_t n_elements_per_pack = 0;
    ptrdiff_t max_nodes_per_pack = 0;
    std::vector<std::vector<uint16_t>> elements;
    std::vector<uint16_t *> element_ptrs;
    std::vector<ptrdiff_t> owned_nodes_ptr;
    std::vector<ptrdiff_t> n_shared_nodes;
    std::vector<ptrdiff_t> ghost_ptr;
    std::vector<idx_t> ghost_idx;
    std::vector<idx_t> to_old;
    // The gather graph a two-pass apply reduces through: row r sums the ghost
    // buffer slots ghost_reduce_idx[ghost_reduce_ptr[r] .. r+1) into the global
    // dof ghost_reduce_dest[r].  Derived from the ghost lists above, because it
    // is the same information grouped by destination instead of by pack.
    ptrdiff_t n_ghost_entries = 0;
    ptrdiff_t n_ghost_reduce_rows = 0;
    std::vector<ptrdiff_t> ghost_reduce_ptr;
    std::vector<ptrdiff_t> ghost_reduce_idx;
    std::vector<idx_t> ghost_reduce_dest;
};

// The layout, from the connectivity and nothing else.
//
// `elements[a][e]` is the global node of shape `a` of element `e`; `nnodes` is
// one past the largest of them.  What comes back is the pack partition plus the
// permutation that produced it -- the caller applies `to_new` to its own
// coordinates and whatever else it stores per node.
static PackedLayout build_packed_layout(const std::vector<std::vector<idx_t>> &elements,
                                        const ptrdiff_t nelements,
                                        const ptrdiff_t nnodes,
                                        ptrdiff_t elements_per_pack) {
    PackedLayout p;
    const int n_shape = (int)elements.size();
    p.n_elements_per_pack = elements_per_pack;
    p.n_packs = (nelements + elements_per_pack - 1) / elements_per_pack;

    // Which pack owns each node, and whether more than one touches it.  The
    // owner is the first pack that reaches it, which is deterministic.
    std::vector<ptrdiff_t> owner((size_t)nnodes, -1);
    std::vector<char> shared((size_t)nnodes, 0);
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        const ptrdiff_t pack = e / elements_per_pack;
        for (int a = 0; a < n_shape; ++a) {
            const idx_t node = elements[a][e];
            if (owner[node] < 0) {
                owner[node] = pack;
            } else if (owner[node] != pack) {
                shared[node] = 1;
            }
        }
    }

    // Renumber: pack by pack, owned-and-private first, owned-and-shared last,
    // which is the order the kernel's two scatter loops assume.
    p.to_new.assign((size_t)nnodes, 0);
    p.to_old.assign((size_t)nnodes, 0);
    p.owned_nodes_ptr.assign((size_t)p.n_packs + 1, 0);
    p.n_shared_nodes.assign((size_t)p.n_packs, 0);
    ptrdiff_t next = 0;
    for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
        p.owned_nodes_ptr[pack] = next;
        for (int pass = 0; pass < 2; ++pass) {
            for (ptrdiff_t node = 0; node < nnodes; ++node) {
                if (owner[node] != pack) continue;
                if ((int)shared[node] != pass) continue;
                p.to_new[node] = (idx_t)next;
                p.to_old[next] = (idx_t)node;
                ++next;
                if (pass == 1) ++p.n_shared_nodes[pack];
            }
        }
    }
    p.owned_nodes_ptr[p.n_packs] = next;

    // The connectivity under the new numbering.  Geometry is per element and so
    // does not move at all; the caller places its own per-node data through
    // `to_new`.
    p.renumbered.assign((size_t)n_shape, std::vector<idx_t>((size_t)nelements));
    for (int a = 0; a < n_shape; ++a)
        for (ptrdiff_t e = 0; e < nelements; ++e)
            p.renumbered[a][e] = p.to_new[elements[a][e]];

    // Ghosts, and the pack-local connectivity that indexes them.
    p.elements.assign((size_t)n_shape, std::vector<uint16_t>((size_t)nelements, 0));
    p.ghost_ptr.assign((size_t)p.n_packs + 1, 0);
    for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
        const ptrdiff_t e_start = pack * elements_per_pack;
        const ptrdiff_t e_end = SFEM_PACKED_LAYOUT_MIN(nelements, (pack + 1) * elements_per_pack);
        const ptrdiff_t owned_begin = p.owned_nodes_ptr[pack];
        const ptrdiff_t n_contiguous = p.owned_nodes_ptr[pack + 1] - owned_begin;
        std::vector<idx_t> ghosts;
        for (ptrdiff_t e = e_start; e < e_end; ++e)
            for (int a = 0; a < n_shape; ++a) {
                const idx_t node = p.renumbered[a][e];
                if (node < owned_begin || node >= owned_begin + n_contiguous)
                    ghosts.push_back(node);
            }
        std::sort(ghosts.begin(), ghosts.end());
        ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());
        p.ghost_ptr[pack + 1] = p.ghost_ptr[pack] + (ptrdiff_t)ghosts.size();
        for (const idx_t node : ghosts) p.ghost_idx.push_back(node);
        const ptrdiff_t slots = n_contiguous + (ptrdiff_t)ghosts.size();
        if (slots > p.max_nodes_per_pack) p.max_nodes_per_pack = slots;
        if (slots > 65535) {
            std::fprintf(stderr, "pack %ld needs %ld slots; the ABI indexes with uint16_t\n",
                         (long)pack, (long)slots);
            std::exit(1);
        }
        for (ptrdiff_t e = e_start; e < e_end; ++e)
            for (int a = 0; a < n_shape; ++a) {
                const idx_t node = p.renumbered[a][e];
                if (node >= owned_begin && node < owned_begin + n_contiguous) {
                    p.elements[a][e] = (uint16_t)(node - owned_begin);
                } else {
                    const ptrdiff_t slot =
                            std::lower_bound(ghosts.begin(), ghosts.end(), node) - ghosts.begin();
                    p.elements[a][e] = (uint16_t)(n_contiguous + slot);
                }
            }
    }
    for (auto &row : p.elements) p.element_ptrs.push_back(row.data());

    p.n_ghost_entries = p.ghost_ptr[p.n_packs];
    std::vector<std::pair<idx_t, ptrdiff_t>> by_destination;
    by_destination.reserve((size_t)p.n_ghost_entries);
    for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack)
        for (ptrdiff_t slot = p.ghost_ptr[pack]; slot < p.ghost_ptr[pack + 1]; ++slot)
            by_destination.push_back(std::make_pair(p.ghost_idx[slot], slot));
    std::sort(by_destination.begin(), by_destination.end());
    p.ghost_reduce_ptr.push_back(0);
    for (size_t i = 0; i < by_destination.size();) {
        const idx_t dest = by_destination[i].first;
        size_t j = i;
        while (j < by_destination.size() && by_destination[j].first == dest) {
            p.ghost_reduce_idx.push_back(by_destination[j].second);
            ++j;
        }
        p.ghost_reduce_dest.push_back(dest);
        p.ghost_reduce_ptr.push_back((ptrdiff_t)p.ghost_reduce_idx.size());
        i = j;
    }
    p.n_ghost_reduce_rows = (ptrdiff_t)p.ghost_reduce_dest.size();
    return p;
}
