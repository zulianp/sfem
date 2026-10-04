#ifndef CVFEM_PACK_SCRATCH_HPP
#define CVFEM_PACK_SCRATCH_HPP

// WHAT THE PACKED KERNELS NEED THAT IS NEITHER A KERNEL NOR STAGING.
//
// These were in cvfem_hex8_pack_common.hpp beside PackedData and the pack builders, and the
// packed sweeps name all of them: the pack's index type, the per-thread scratch and the four
// scratch sizings. None of them touches a mesh, a pack or a library -- the scratch is malloc and
// the sizings are arithmetic -- so they belong on this side of DESIGN.md's boundary, with the
// kernels that use them, rather than in the header that owns the staging objects.
//
// What stayed behind is everything that does touch those objects: PackedData itself,
// make_packed, the default pack size and find_pack_col.
//
// pack_local_to_global came over, as the four values it actually reads. It was described here as
// used only by the pack builders, and that stopped being true: the semi-structured packed
// gradient resolves a global node per element node inside its sweep. Rather than let that sweep
// inline the mapping -- a second spelling of the pack layout's addressing, which is how the
// layout drifts -- the mapping is here and the staging form in cvfem_hex8_pack_common.hpp is a
// one-line adapter over it.
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

// CVFEM_HEX8_N_FIELDS, which the scratch sizings multiply by. It replaced a bare N_FIELDS that
// each operator family declared for itself -- the microkernels already own the constant.
#include "kernels/microkernels/hex8/cvfem_hex8_ns_upwind_kernels.hpp"

using pack_idx_t = uint16_t;

// A pack-local node index to its global one. The owned nodes of a pack are contiguous from
// owned_begin; everything past n_contiguous is a ghost and is named by the pack's ghost list.
//
// It takes the pack's own two values rather than the pack table and an index into it, which
// removes a double indirection the caller has already done: every sweep that needs this has
// `owned` and `ghosts` in hand before the element loop.
static SFEM_INLINE idx_t cvfem_pack_local_to_global(const ptrdiff_t                  owned_begin,
                                                    const idx_t *const SFEM_RESTRICT ghosts,
                                                    const ptrdiff_t                  n_contiguous,
                                                    const pack_idx_t                 local) {
    if ((ptrdiff_t)local < n_contiguous) return idx_t(owned_begin + (ptrdiff_t)local);
    return ghosts[(ptrdiff_t)local - n_contiguous];
}

// Per-thread scratch arena, CVFEM_PACK_SCRATCH_SLOTS slots, grown on demand and never shrunk.
// Ten, not eight: the semi-structured packed gradient needs two of its own.
//
// It cannot share the flat gradient's slots 5 and 6. A semi-structured multigrid hierarchy
// has semi-structured fine levels and a FLAT coarse level in the same process, and the two
// want very different sizes from the same slot -- the scratch grows on demand and never
// shrinks, so sharing would reallocate on every alternation rather than once.
static constexpr int CVFEM_PACK_SCRATCH_SLOTS = 10;

// Per-thread scratch, indexed by slot. An out-of-range slot used to walk straight off the
// end of these arrays and corrupt whatever thread_local storage followed -- the symptom was
// a malloc abort ("pointer being freed was not allocated") far from the cause, so the bound
// is checked rather than assumed. The check is once per call, not per element.
template <typename T>
static T *thread_scratch(const int slot, const size_t n) {
    static thread_local T     *ptr[CVFEM_PACK_SCRATCH_SLOTS] = {};
    static thread_local size_t cap[CVFEM_PACK_SCRATCH_SLOTS] = {};
    assert(slot >= 0 && slot < CVFEM_PACK_SCRATCH_SLOTS);
    if (cap[slot] < n) {
        std::free(ptr[slot]);
        ptr[slot] = static_cast<T *>(std::calloc(n, sizeof(T)));
        cap[slot] = ptr[slot] ? n : 0;
    }
    return ptr[slot];
}


static SFEM_INLINE size_t packed_scratch_n(const ptrdiff_t max_actual_nodes_per_pack) {
    const ptrdiff_t n = max_actual_nodes_per_pack > 0 ? max_actual_nodes_per_pack : 1;
    return (size_t)CVFEM_HEX8_N_FIELDS * (size_t)n;
}

// THEY TAKE THE COUNT, NOT THE STAGING OBJECT. DESIGN.md requires src/kernels/ to name no
// library, and these are called from inside the pack sweeps -- so a PackedData parameter here is
// a PackedData dependency there. The count is the only thing any of them reads.
static SFEM_INLINE size_t packed_xyz_n(const ptrdiff_t max_actual_nodes_per_pack) {
    const ptrdiff_t n = max_actual_nodes_per_pack > 0 ? max_actual_nodes_per_pack : 1;
    return 3 * (size_t)n;
}

static SFEM_INLINE size_t packed_rc_n(const ptrdiff_t max_actual_nodes_per_pack) {
    const ptrdiff_t n = max_actual_nodes_per_pack > 0 ? max_actual_nodes_per_pack : 1;
    return 6 * (size_t)n;
}

// The direction's reconstructed pressure gradient, staged only by the Jacobian action.
// Kept out of packed_rc_n so the residual, which never reads it, allocates exactly what it
// did before.
static SFEM_INLINE size_t packed_qg_n(const ptrdiff_t max_actual_nodes_per_pack) {
    const ptrdiff_t n = max_actual_nodes_per_pack > 0 ? max_actual_nodes_per_pack : 1;
    return 3 * (size_t)n;
}

#endif  // CVFEM_PACK_SCRATCH_HPP
