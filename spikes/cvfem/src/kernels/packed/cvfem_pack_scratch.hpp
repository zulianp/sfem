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
// make_packed, the default pack size, pack_local_to_global and find_pack_col -- the last two
// used only by the pack builders.
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

using pack_idx_t = uint16_t;

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
    return (size_t)N_FIELDS * (size_t)n;
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
