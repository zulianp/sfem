#ifndef CVFEM_HEX8_PACK_COMMON_HPP
#define CVFEM_HEX8_PACK_COMMON_HPP

// The packed-mesh machinery, shared by the benchmark and the steady solver.
//
// This existed twice, incompatibly: cvfem_hex8_best_common.hpp had the full
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

#include <cassert>
#include <cstdlib>
#include <limits>
#include <memory>
#include <vector>

#include "smesh_mesh.hpp"
#include "smesh_packed_mesh.hpp"

#include "core/cvfem_portability.hpp"

#include "kernels/packed/cvfem_pack_scratch.hpp"   // pack_idx_t, thread_scratch, sizings

struct PackedData {
    std::shared_ptr<smesh::PackedMesh<pack_idx_t>> packed;
    ptrdiff_t                                      n_packs{0};
    ptrdiff_t                                      n_elements_per_pack{0};
    // Elements the packs cover. Not n_packs * n_elements_per_pack -- the last pack is short,
    // and on a distributed mesh the packs span only the owned-not-shared prefix of the block,
    // so the packed element array is shorter than the block. Anything walking p.elems has to
    // stop here or it reads past the allocation.
    ptrdiff_t                                      n_packed_elements{0};
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
    // THE SAME TWO, AS ARRAYS OF POINTERS, so that a kernel can be handed them without being
    // handed this object. They are vectors of vectors, so there is no flat pointer into them and
    // `local_rowptr[pack]` in a kernel would otherwise require naming PackedData. Published by
    // build_pack_local_crs, which is the only thing that sizes the vectors behind them, exactly
    // as MeshData publishes adj_ptr beside jacobian_adjugate.
    std::vector<const int *>            local_rowptr_ptr;
    std::vector<const smesh::count_t *> local_global_slot_ptr;
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


// THE DEFAULT PACK SIZE, DERIVED FROM THE MACHINE RATHER THAN FROM THE INDEX TYPE.
//
// A pack is one OpenMP iteration, so the number of packs IS the available parallelism, and the
// pack-size sweep locates the optimum at a fixed number of packs per THREAD rather than at a
// fixed element count: plotted on that axis the problem sizes measured fail together, while
// failing at different pack sizes. Sizing against the index-width ceiling instead -- the
// pessimistic bound n_elements * nodes_per_element against what pack_idx_t can address, which
// credits no sharing at all -- knows nothing about how many cores the machine has, and at 72
// threads it gave up 1.2x.
//
// So the ceiling stays as the correctness bound it is and the default targets the measured
// ratio, rounded to a power of two because that is the grid the sweep measured on.
static constexpr int CVFEM_PACKS_PER_THREAD = 28;

static int cvfem_default_pack_size(const ptrdiff_t nelements, const int threads) {
    // What the index type admits, with the same pessimistic bound the packer uses.
    const double ceiling = double(size_t(std::numeric_limits<pack_idx_t>::max()) + 1) / 8.0;
    const double want    = double(nelements) / (double(threads > 0 ? threads : 1)
                                                * double(CVFEM_PACKS_PER_THREAD));
    // Nearest power of two in log space, not the next one down: the target ratio is a minimum
    // of a curve rather than an upper bound, so overshooting it and undershooting it cost the
    // same and the nearer grid point is the better estimate of the minimum.
    double p = 64.0;
    while (p * 2.0 <= ceiling && (p * 2.0) / want < want / p) p *= 2.0;
    if (p > ceiling) p = ceiling;
    return int(p);
}

static PackedData make_packed(const std::shared_ptr<smesh::Mesh> &mesh, const int pack_size) {
    PackedData p;
    p.packed              = smesh::PackedMesh<pack_idx_t>::create(mesh, {}, true, pack_size);
    p.n_packs             = p.packed->n_packs(0);
    p.n_elements_per_pack = p.packed->n_elements_per_pack(0);
    p.n_packed_elements   = p.packed->n_packed_elements(0);
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
