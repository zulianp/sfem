#ifndef CVFEM_HEX8_PACK_COMMON_HPP
#define CVFEM_HEX8_PACK_COMMON_HPP

// The packed-mesh machinery, shared by the benchmark and the steady solver.
//
// This existed twice, incompatibly: cvfem_hex8_layout_common.hpp had the full
// definition and cvfem_hex8_ns_packed.hpp a divergent trimmed copy, so there was no
// single thing a device path could consume. The full definition wins; the trimmed
// copy is gone.
//
// Scope is deliberately narrow -- the node partition and its scratch helpers, nothing
// that knows about MeshData or BSR4. The benchmark and the solver have genuinely
// different versions of those two (the solver carries Lx/Ly/Lz, the nodal pressure
// gradient, rhie_chow_scale and diag_slots), and unifying them is not justified by the
// CUDA port. See PACKED_FORMAT.md section 10.
//
// The format contract itself -- the owned/ghost id split, the non-shared-before-shared
// ordering, the ghost reduction graph -- is written up in PACKED_FORMAT.md.
//
// Not self-contained, matching the convention of the other CVFEM headers: the includer
// must define `scalar_t` and `N_FIELDS` before including this.

#include <cstdlib>
#include <memory>
#include <vector>

#include "smesh_mesh.hpp"
#include "smesh_packed_mesh.hpp"

#include "cvfem_portability.hpp"

using pack_idx_t = uint16_t;

struct PackedData {
    std::shared_ptr<smesh::PackedMesh<pack_idx_t>> packed;
    ptrdiff_t                                      n_packs{0};
    ptrdiff_t                                      n_elements_per_pack{0};
    ptrdiff_t                                      max_nodes_per_pack{0};
    pack_idx_t                                   **elems{nullptr};
    const ptrdiff_t                               *owned_nodes_ptr{nullptr};
    const ptrdiff_t                               *n_shared{nullptr};
    const ptrdiff_t                               *ghost_ptr{nullptr};
    const smesh::idx_t                            *ghost_idx{nullptr};
    ptrdiff_t                                      n_ghost_entries{0};
    ptrdiff_t                                      n_ghost_reduce_rows{0};
    const ptrdiff_t                               *ghost_reduce_ptr{nullptr};
    const ptrdiff_t                               *ghost_reduce_idx{nullptr};
    const smesh::idx_t                            *ghost_reduce_dest{nullptr};
    std::vector<scalar_t>                          ghost_buf;
    ptrdiff_t                                      mean_nodes_per_pack{0};
    ptrdiff_t                                      max_actual_nodes_per_pack{0};
    std::vector<std::vector<int>>                  local_rowptr;
    std::vector<std::vector<pack_idx_t>>           local_colidx;
    std::vector<std::vector<smesh::count_t>>       local_global_slot;
    std::vector<int>                               local_element_slot;
    ptrdiff_t                                      max_local_nnz{0};
    std::vector<ptrdiff_t>                         ghost_mat_ptr;
    std::vector<smesh::count_t>                    ghost_mat_slot;
    std::vector<scalar_t>                          ghost_mat_val;

    // --- "store" layout -------------------------------------------------
    // Owned rows of a pack use the *global* row pattern, so the pack's owned
    // block is a contiguous slice of the global BSR values and can be written
    // with a plain streaming store: no pre-zeroing and no read-modify-write.
    // Ghost rows keep the compact pack-local pattern and are reduced after.
    std::vector<int>                     st_owned_nnz;      // per pack, = global nnz of its owned rows
    std::vector<int>                     st_local_nnz;      // per pack, owned + ghost
    std::vector<std::vector<int>>        st_rowptr;         // per pack, n_pack_nodes + 1
    std::vector<int>                     st_element_slot;   // per element, 64 local block ids
    std::vector<ptrdiff_t>               st_ghost_ptr;      // per ghost entry + 1, into st_ghost_slot
    std::vector<smesh::count_t>          st_ghost_slot;     // global block id per ghost nnz
    std::vector<scalar_t>                st_ghost_val;
    ptrdiff_t                            st_max_local_nnz{0};
};

// Per-thread scratch arena, four slots, grown on demand and never shrunk.
template <typename T>
static T *thread_scratch(const int slot, const size_t n) {
    static thread_local T     *ptr[4] = {nullptr, nullptr, nullptr, nullptr};
    static thread_local size_t cap[4] = {0, 0, 0, 0};
    if (cap[slot] < n) {
        std::free(ptr[slot]);
        ptr[slot] = static_cast<T *>(std::calloc(n, sizeof(T)));
        cap[slot] = ptr[slot] ? n : 0;
    }
    return ptr[slot];
}

static PackedData make_packed(const std::shared_ptr<smesh::Mesh> &mesh, const int pack_size) {
    PackedData p;
    p.packed              = smesh::PackedMesh<pack_idx_t>::create(mesh, {}, true, pack_size);
    p.n_packs             = p.packed->n_packs(0);
    p.n_elements_per_pack = p.packed->n_elements_per_pack(0);
    p.max_nodes_per_pack  = p.packed->max_nodes_per_pack();
    p.elems               = p.packed->elements(0)->data();
    p.owned_nodes_ptr     = p.packed->owned_nodes_ptr(0)->data();
    p.n_shared            = p.packed->n_shared(0)->data();
    p.ghost_ptr           = p.packed->ghost_ptr(0)->data();
    p.ghost_idx           = p.packed->ghost_idx(0)->data();
    p.n_ghost_entries     = p.packed->n_ghost_entries(0);
    p.n_ghost_reduce_rows = p.packed->n_ghost_reduce_rows(0);
    p.ghost_reduce_ptr    = p.packed->ghost_reduce_ptr(0)->data();
    p.ghost_reduce_idx    = p.packed->ghost_reduce_idx(0)->data();
    p.ghost_reduce_dest   = p.packed->ghost_reduce_dest(0)->data();
    p.ghost_buf.assign((size_t)N_FIELDS * (size_t)p.n_ghost_entries, 0.0);

    ptrdiff_t sum_nodes = 0;
    ptrdiff_t max_nodes = 0;
    for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
        const ptrdiff_t n_pack_nodes =
                (p.owned_nodes_ptr[pack + 1] - p.owned_nodes_ptr[pack]) + (p.ghost_ptr[pack + 1] - p.ghost_ptr[pack]);
        sum_nodes += n_pack_nodes;
        max_nodes = std::max(max_nodes, n_pack_nodes);
    }
    p.mean_nodes_per_pack       = p.n_packs ? sum_nodes / p.n_packs : 0;
    p.max_actual_nodes_per_pack = max_nodes;
    if (getenv("CVFEM_PACK_STATS")) {
        std::printf("[pack-stats] n_packs=%td owned_ptr[0]=%td owned_ptr[n]=%td n_ghost_entries=%td n_ghost_reduce_rows=%td sum_pack_nodes=%td\n",
                    p.n_packs, p.owned_nodes_ptr[0], p.owned_nodes_ptr[p.n_packs], p.n_ghost_entries, p.n_ghost_reduce_rows, sum_nodes);
    }
    return p;
}

static SFEM_INLINE size_t packed_scratch_n(const PackedData &p) {
    const ptrdiff_t n = p.max_actual_nodes_per_pack > 0 ? p.max_actual_nodes_per_pack : 1;
    return (size_t)N_FIELDS * (size_t)n;
}

static SFEM_INLINE size_t packed_xyz_n(const PackedData &p) {
    const ptrdiff_t n = p.max_actual_nodes_per_pack > 0 ? p.max_actual_nodes_per_pack : 1;
    return 3 * (size_t)n;
}

static SFEM_INLINE size_t packed_rc_n(const PackedData &p) {
    const ptrdiff_t n = p.max_actual_nodes_per_pack > 0 ? p.max_actual_nodes_per_pack : 1;
    return 6 * (size_t)n;
}

static SFEM_INLINE smesh::idx_t pack_local_to_global(const PackedData &p,
                                                     const ptrdiff_t   pack,
                                                     const ptrdiff_t   n_contiguous,
                                                     const pack_idx_t  local) {
    if ((ptrdiff_t)local < n_contiguous) return smesh::idx_t(p.owned_nodes_ptr[pack] + (ptrdiff_t)local);
    return p.ghost_idx[p.ghost_ptr[pack] + ((ptrdiff_t)local - n_contiguous)];
}

static SFEM_INLINE int find_pack_col(const pack_idx_t target, const pack_idx_t *const SFEM_RESTRICT row, const int n) {
    for (int i = 0; i < n; ++i) {
        if (row[i] == target) return i;
    }
    return 0;
}

#endif  // CVFEM_HEX8_PACK_COMMON_HPP
