#ifndef CVFEM_PACK_COLORING_HPP
#define CVFEM_PACK_COLORING_HPP

// Coloring of a smesh::PackedMesh element partition.
//
// A pack owns a contiguous range of node ids, [owned_nodes_ptr[pack],
// owned_nodes_ptr[pack + 1]), and lists the nodes it touches but does not own
// in ghost_idx[ghost_ptr[pack] .. ghost_ptr[pack + 1]). Two packs therefore
// share a node exactly when one ghosts a node the other owns, or when both
// ghost the same node -- the ghost lists carry all of the conflict information.
//
// Packs that share no node can assemble concurrently straight into a global
// matrix without atomics: no per-pack local matrix, no local-to-global copy and
// no ghost reduction. Assembling one color at a time makes that safe.

#include <algorithm>
#include <cstddef>
#include <vector>

struct PackColoring {
    std::vector<ptrdiff_t> pack_order;  // packs grouped by color
    std::vector<ptrdiff_t> color_ptr;   // color c owns pack_order[color_ptr[c] .. color_ptr[c + 1])
    int                    n_colors{0};
    ptrdiff_t              max_packs_per_color{0};
    ptrdiff_t              min_packs_per_color{0};
};

template <typename IdxT>
static PackColoring cvfem_build_pack_coloring(const ptrdiff_t  n_packs,
                                              const ptrdiff_t *owned_nodes_ptr,
                                              const ptrdiff_t *ghost_ptr,
                                              const IdxT      *ghost_idx) {
    PackColoring c;
    if (n_packs <= 0) return c;

    auto owner_of = [&](const IdxT node) -> ptrdiff_t {
        const ptrdiff_t *const it = std::upper_bound(owned_nodes_ptr, owned_nodes_ptr + n_packs + 1, (ptrdiff_t)node);
        return (it - owned_nodes_ptr) - 1;
    };

    std::vector<std::vector<ptrdiff_t>> adj((size_t)n_packs);
    {
        std::vector<IdxT> shared;
        shared.reserve((size_t)(ghost_ptr[n_packs] - ghost_ptr[0]));
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            for (ptrdiff_t k = ghost_ptr[pack]; k < ghost_ptr[pack + 1]; ++k) shared.push_back(ghost_idx[k]);
        }
        std::sort(shared.begin(), shared.end());
        shared.erase(std::unique(shared.begin(), shared.end()), shared.end());

        std::vector<std::vector<ptrdiff_t>> node_packs(shared.size());
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            for (ptrdiff_t k = ghost_ptr[pack]; k < ghost_ptr[pack + 1]; ++k) {
                const size_t pos =
                        (size_t)(std::lower_bound(shared.begin(), shared.end(), ghost_idx[k]) - shared.begin());
                node_packs[pos].push_back(pack);
            }
        }
        for (size_t i = 0; i < shared.size(); ++i) {
            auto &pk = node_packs[i];
            pk.push_back(owner_of(shared[i]));
            std::sort(pk.begin(), pk.end());
            pk.erase(std::unique(pk.begin(), pk.end()), pk.end());
            for (size_t a = 0; a < pk.size(); ++a) {
                for (size_t b = a + 1; b < pk.size(); ++b) {
                    adj[(size_t)pk[a]].push_back(pk[b]);
                    adj[(size_t)pk[b]].push_back(pk[a]);
                }
            }
        }
    }
    for (auto &row : adj) {
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
    }

    // Balanced greedy coloring, highest degree first.
    //
    // Every color is a parallel region ending in a barrier, so a color holding
    // only a couple of packs leaves most threads idle -- with the classic
    // "lowest feasible color index" rule the later colors end up nearly empty and
    // the barriers cost more than the coloring saves. Picking the *least loaded*
    // feasible color instead keeps the color sizes even at no cost in color count.
    std::vector<ptrdiff_t> order((size_t)n_packs);
    for (ptrdiff_t i = 0; i < n_packs; ++i) order[(size_t)i] = i;
    std::stable_sort(order.begin(), order.end(), [&](const ptrdiff_t a, const ptrdiff_t b) {
        return adj[(size_t)a].size() > adj[(size_t)b].size();
    });

    std::vector<int>       color((size_t)n_packs, -1);
    std::vector<char>      used;
    std::vector<ptrdiff_t> count;
    int                    n_colors = 0;
    for (const ptrdiff_t pack : order) {
        used.assign((size_t)n_colors, 0);
        for (const ptrdiff_t q : adj[(size_t)pack]) {
            if (color[(size_t)q] >= 0) used[(size_t)color[(size_t)q]] = 1;
        }
        int chosen = -1;
        for (int k = 0; k < n_colors; ++k) {
            if (used[(size_t)k]) continue;
            if (chosen < 0 || count[(size_t)k] < count[(size_t)chosen]) chosen = k;
        }
        if (chosen < 0) {
            chosen = n_colors++;
            count.push_back(0);
        }
        color[(size_t)pack] = chosen;
        count[(size_t)chosen]++;
    }

    c.n_colors = n_colors;
    c.color_ptr.assign((size_t)n_colors + 1, 0);
    for (ptrdiff_t pack = 0; pack < n_packs; ++pack) c.color_ptr[(size_t)color[(size_t)pack] + 1]++;
    c.max_packs_per_color = 0;
    c.min_packs_per_color = n_packs;
    for (int i = 0; i < n_colors; ++i) {
        c.max_packs_per_color = std::max(c.max_packs_per_color, c.color_ptr[(size_t)i + 1]);
        c.min_packs_per_color = std::min(c.min_packs_per_color, c.color_ptr[(size_t)i + 1]);
        c.color_ptr[(size_t)i + 1] += c.color_ptr[(size_t)i];
    }
    c.pack_order.resize((size_t)n_packs);
    {
        std::vector<ptrdiff_t> cursor(c.color_ptr.begin(), c.color_ptr.end() - 1);
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            c.pack_order[(size_t)cursor[(size_t)color[(size_t)pack]]++] = pack;
        }
    }
    return c;
}

#endif  // CVFEM_PACK_COLORING_HPP
