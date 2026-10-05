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

// For MIN, which cvfem_hex8_pack_extent uses to clamp the last pack's element range. It is
// defined once, there, behind an include guard.
#include "kernels/cvfem_scatter.hpp"

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

// THE SLOT-3 CARVE-UP, ONCE. Every pack sweep that stages coordinates computed these six
// pointers itself, and the five copies were identical but for which of them the sweep went on
// to read -- fifteen of sixteen lines the same between the residual and the assembly. They have
// to agree exactly: the arrays are one allocation and a sweep that offset `pgx` differently
// from the sweep that filled it would read another array's coordinates.
//
// It is also what makes DESIGN.md's affine/isoparametric split free of duplication. Splitting a
// `template <bool ISO>` sweep into two files copies everything the two geometries share, and
// this staging is most of it; owning it here means the two sweeps each carry one line.
struct Hex8PackCoords {
    scalar_t *SFEM_RESTRICT x;
    scalar_t *SFEM_RESTRICT y;
    scalar_t *SFEM_RESTRICT z;
    scalar_t *SFEM_RESTRICT pgx;
    scalar_t *SFEM_RESTRICT pgy;
    scalar_t *SFEM_RESTRICT pgz;
};

// `want_xyz` is the geometry's own need for node coordinates -- the isoparametric kernels derive
// the Jacobian from them -- and `with_rc` adds the pressure gradient. Either one claims the
// slot; both together claim it at the larger size, which is why the size and the carve-up are
// one decision and not two.
static SFEM_INLINE Hex8PackCoords cvfem_hex8_pack_coords(const bool      want_xyz,
                                                         const int       with_rc,
                                                         const ptrdiff_t max_actual_nodes_per_pack) {
    scalar_t *const SFEM_RESTRICT base =
            (want_xyz || with_rc)
                    ? thread_scratch<scalar_t>(3, with_rc ? packed_rc_n(max_actual_nodes_per_pack)
                                                          : packed_xyz_n(max_actual_nodes_per_pack))
                    : nullptr;
    const ptrdiff_t n = max_actual_nodes_per_pack > 0 ? max_actual_nodes_per_pack : 1;
    Hex8PackCoords  c{};
    c.x   = base;
    c.y   = base ? base + n : nullptr;
    c.z   = base ? base + 2 * n : nullptr;
    c.pgx = with_rc ? base + 3 * n : nullptr;
    c.pgy = with_rc ? base + 4 * n : nullptr;
    c.pgz = with_rc ? base + 5 * n : nullptr;
    return c;
}

// SLOT 4, THE DIRECTION'S RECONSTRUCTED PRESSURE GRADIENT. Staged only by the Jacobian action,
// and carved up exactly as slot 3 is -- it shared slot 3's `xyz_n` local, which is the kind of
// coupling that makes a sizing change in one array silently re-offset another.
struct Hex8PackQGrad {
    scalar_t *SFEM_RESTRICT x;
    scalar_t *SFEM_RESTRICT y;
    scalar_t *SFEM_RESTRICT z;
};

static SFEM_INLINE Hex8PackQGrad cvfem_hex8_pack_qgrad(const int       with_qg,
                                                       const ptrdiff_t max_actual_nodes_per_pack) {
    scalar_t *const SFEM_RESTRICT base =
            with_qg ? thread_scratch<scalar_t>(4, packed_qg_n(max_actual_nodes_per_pack)) : nullptr;
    const ptrdiff_t n = max_actual_nodes_per_pack > 0 ? max_actual_nodes_per_pack : 1;
    Hex8PackQGrad   q{};
    q.x = base;
    q.y = base ? base + n : nullptr;
    q.z = base ? base + 2 * n : nullptr;
    return q;
}

// WHICH ELEMENTS AND NODES ONE PACK COVERS. The same six lines opened the pack loop in all five
// sweeps. Reading them out of one place is what keeps "owned" and the ghost slice consistent
// between the sweep that fills a pack and the sweep that drains it.
struct Hex8PackExtent {
    ptrdiff_t                  e_start;
    ptrdiff_t                  e_end;
    ptrdiff_t                  owned;
    ptrdiff_t                  n_contiguous;
    ptrdiff_t                  n_ghost;
    // The pack-local node count, which is what sizes its private accumulation buffer, and the
    // offset of its ghost slice in the global ghost array. Both were separate locals in every
    // sweep; they are derived from the five above and belong with them.
    ptrdiff_t                  n_pack_nodes;
    ptrdiff_t                  ghost_off;
    const idx_t *SFEM_RESTRICT ghosts;
};

static SFEM_INLINE Hex8PackExtent cvfem_hex8_pack_extent(
        const ptrdiff_t                      pack,
        const ptrdiff_t                      nelements,
        const ptrdiff_t                      n_elements_per_pack,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const idx_t *const SFEM_RESTRICT     ghost_idx,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr) {
    Hex8PackExtent x{};
    x.e_start      = pack * n_elements_per_pack;
    x.e_end        = MIN(nelements, (pack + 1) * n_elements_per_pack);
    x.owned        = owned_nodes_ptr[pack];
    x.n_contiguous = owned_nodes_ptr[pack + 1] - x.owned;
    x.n_ghost      = ghost_ptr[pack + 1] - ghost_ptr[pack];
    x.n_pack_nodes = x.n_contiguous + x.n_ghost;
    x.ghost_off    = ghost_ptr[pack];
    x.ghosts       = &ghost_idx[ghost_ptr[pack]];
    return x;
}

#endif  // CVFEM_PACK_SCRATCH_HPP
