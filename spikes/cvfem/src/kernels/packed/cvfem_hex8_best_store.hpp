#ifndef CVFEM_HEX8_BEST_STORE_HPP
#define CVFEM_HEX8_BEST_STORE_HPP

// Store layout: a packed assembly whose pack-local matrix has its *owned* rows
// laid out in the global sparsity pattern. A pack's owned block is then a
// contiguous slice of the global BSR values and is flushed with one streaming
// memcpy, so every global block is written exactly once: no zero_bsr4 pass and no
// read-modify-write. Only the ghost rows still need a reduction.

#include "kernels/microkernels/hex8/affine/cvfem_hex8_ns_upwind_affine.hpp"
#include "kernels/microkernels/hex8/isoparametric/cvfem_hex8_ns_upwind_isoparam.hpp"
#include "kernels/cvfem_scatter.hpp"
#include "kernels/cvfem_phases.hpp"
#include "kernels/cvfem_range.hpp"

// Build the "store" layout. Owned rows of a pack map 1:1 onto the contiguous
// global slice [rowptr_g[owned], rowptr_g[owned + n_contiguous]), so assembling
// a pack ends in one memcpy that writes every one of those blocks exactly once.

// Write-once assembly. Each pack accumulates into a cache-resident local matrix
// whose owned rows already carry the global sparsity pattern, then streams that
// block straight into the global BSR with a single memcpy. Every global block is
// written exactly once, so there is no zero_bsr4 pass and no read-modify-write.
// Only the ghost rows, which are shared between packs, need a reduction.
// ISO IS A TEMPLATE PARAMETER, NOT A RUNTIME ENUM. DESIGN.md asks for the affine and the
// isoparametric kernels to be "logically separated (now they are mixed in with enum and
// booleans)", and this was the enum: GeomKind arrived as an argument and was tested per pack,
// inside the sweep. It also left the geometry undecided at the point where it matters -- the
// lane loop -- which is the shape of guard this file's own notes record costing 1.83x, and which
// the vectorisation gate now refuses outright. The caller picks the instantiation.
// THE STORE SWEEP KEEPS ITS DYNAMIC SCHEDULE, which is why its range is one pack and its
// scratch arrives as arguments. The other pack sweeps take an equal static slice, because
// cvfem_range_split reproduces `schedule(static)` exactly; this one was written with
// `schedule(dynamic, 1)` -- it writes each matrix entry once rather than accumulating, so packs
// differ in how much work they carry and the balancing matters. A static split would have been a
// silent change to how the work is shared, so the launcher keeps the dynamic `omp for` and hands
// this kernel one pack at a time.
//
// The consequence is that the scratch cannot move in here: acquired per call it would be
// re-acquired once per pack rather than once per thread. It stays in the launcher's parallel
// region and is passed, which is what DESIGN.md's "only arguments that are actually used" asks
// for and what leaves no hidden per-thread state in the kernel.
template <bool ISO>
static SFEM_NOINLINE void assemble_jacobian_store_range(
        const cvfem_range packs,
        // The mesh and the pack are staging objects -- they own vectors and a shared_ptr to a
        // mesh -- so what this kernel reads out of them is what it takes. DESIGN.md: only
        // arguments that are actually used are passed.
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        const ptrdiff_t nelements,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT pgx,
        const scalar_t *const SFEM_RESTRICT pgy,
        const scalar_t *const SFEM_RESTRICT pgz,
        geom_t **const SFEM_RESTRICT points,
        const scalar_t rhie_chow_scale,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        pack_idx_t **const SFEM_RESTRICT pack_elems,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const int *const SFEM_RESTRICT st_element_slot,
        const ptrdiff_t *const SFEM_RESTRICT st_ghost_ptr,
        scalar_t *const SFEM_RESTRICT st_ghost_val,
        const int *const SFEM_RESTRICT st_local_nnz,
        const int *const SFEM_RESTRICT st_owned_nnz,
        // BSR4 is a staging type (it owns a SharedBuffer and a graph); the kernel reads one
        // array out of it, so that is what it takes.
        const count_t *const SFEM_RESTRICT rowptr,
        const scalar_t   rho,
        const scalar_t   mu,
        const KernelKind kernel_kind,
        scalar_t *const SFEM_RESTRICT gvalues,
        const int                     with_rc,
        CVFEM_PHASE_ACC_PARAM
        scalar_t *const SFEM_RESTRICT pack_u,
        scalar_t *const SFEM_RESTRICT local_vals,
        scalar_t *const SFEM_RESTRICT pack_x,
        scalar_t *const SFEM_RESTRICT pack_y,
        scalar_t *const SFEM_RESTRICT pack_z,
        scalar_t *const SFEM_RESTRICT pack_pgx,
        scalar_t *const SFEM_RESTRICT pack_pgy,
        scalar_t *const SFEM_RESTRICT pack_pgz,
        // Resolved once per solve, in the launcher, not per element here. This parameter replaced
        // the cvfem_hex8_rc_config_for(d) call that used to sit in this body: that function takes
        // the mesh, which a kernel is not meant to name.
        const Hex8RcConfig &rc_cfg) {
    for (ptrdiff_t pack = packs.begin; pack < packs.end; ++pack) {
            const ptrdiff_t                         e_start      = pack * n_elements_per_pack;
            const ptrdiff_t                         e_end        = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t                         owned        = owned_nodes_ptr[pack];
            const ptrdiff_t                         n_contiguous = owned_nodes_ptr[pack + 1] - owned;
            const ptrdiff_t                         n_ghost      = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const idx_t *const SFEM_RESTRICT ghosts       = &ghost_idx[ghost_ptr[pack]];
            const int                               owned_nnz    = st_owned_nnz[(size_t)pack];
            const int                               local_nnz    = st_local_nnz[(size_t)pack];

            CVFEM_PHASE_CLOCK(_t);
            std::memset(local_vals, 0, (size_t)local_nnz * 16 * sizeof(scalar_t));
            CVFEM_PHASE_MARK(acc, _t, PH_LOCAL_MEMSET);

            fill_pack_fields(owned_nodes_ptr, ux, uy, uz, pres, pack, n_contiguous, n_ghost, ghosts, pack_u);
            if (with_rc)
                cvfem_hex8_fill_pack_xyz_pgrad(owned_nodes_ptr, points, pgx, pgy, pgz, with_rc, pack, n_contiguous, n_ghost, ghosts, pack_x, pack_y, pack_z,
                                               pack_pgx, pack_pgy, pack_pgz);
            if constexpr (ISO)
                fill_pack_xyz(owned_nodes_ptr, points, pack, n_contiguous, n_ghost, ghosts, pack_x, pack_y, pack_z);
            CVFEM_PHASE_MARK(acc, _t, PH_GATHER);

            for (ptrdiff_t e = e_start; e < e_end; ++e) {
                scalar_t ux_e[8], uy_e[8], uz_e[8], p_e[8];
                for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                    const scalar_t *const SFEM_RESTRICT u = pack_u + (ptrdiff_t)pack_elems[a][e] * CVFEM_HEX8_N_FIELDS;
                    ux_e[a]                               = u[0];
                    uy_e[a]                               = u[1];
                    uz_e[a]                               = u[2];
                    p_e[a]                                = u[3];
                }
                const int *const SFEM_RESTRICT slots = st_element_slot + (size_t)e * 64;

                scalar_t     rc_x[8], rc_y[8], rc_z[8], rc_pgx[8], rc_pgy[8], rc_pgz[8];
                const Hex8RcConfig rcfg = rc_cfg;
                Hex8RhieChow rc{};
                if (with_rc) {
                    gather_hex8_coords_from_pack(pack_elems, pack_x, pack_y, pack_z, e, rc_x, rc_y, rc_z);
                    gather_hex8_coords_from_pack(pack_elems, pack_pgx, pack_pgy, pack_pgz, e, rc_pgx, rc_pgy, rc_pgz);
                    rc = Hex8RhieChow{rc_x,    rc_y,    rc_z,    rc_pgx, rc_pgy, rc_pgz, rcfg.scale,
                                      nullptr, nullptr, nullptr, ux_e,   uy_e,   uz_e,   rcfg.tau};
                }
                const scalar_t *const rc_p = with_rc ? p_e : nullptr;

                if constexpr (ISO) {
                    scalar_t x[8], y[8], z[8];
                    gather_hex8_coords_from_pack(pack_elems, pack_x, pack_y, pack_z, e, x, y, z);
                    cvfem_hex8_ns_upwind_jacobian_add_slots_isoparam<false>(
                            rho, mu, x, y, z, ux_e, uy_e, uz_e, slots, local_vals, rc, rc_p);
                } else {
                    scalar_t adj[9], det;
                    load_hex8_adj(adj_ptr, det_ptr, e, adj, &det);
                    switch (kernel_kind) {
                        case KernelKind::Sympy:
                            cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots(
                                    rho, mu, adj, det, ux_e, uy_e, uz_e, slots, local_vals);
                            break;
                        case KernelKind::SympyBlock:
                            cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots_blockwise(
                                    rho, mu, adj, det, ux_e, uy_e, uz_e, slots, local_vals);
                            break;
                        case KernelKind::SympyRow:
                            cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots_rowwise(
                                    rho, mu, adj, det, ux_e, uy_e, uz_e, slots, local_vals);
                            break;
                        case KernelKind::SympyFace:
                            cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots_facewise(
                                    rho, mu, adj, det, ux_e, uy_e, uz_e, slots, local_vals);
                            break;
                        case KernelKind::Sumfact:
                            if (g_dense_flush) {
                                alignas(ALIGN_BYTES) scalar_t ke[64 * 16] = {};
                                cvfem_hex8_ns_upwind_jacobian_add_slots<false>(
                                        rho, mu, adj, det, ux_e, uy_e, uz_e, g_identity_slots, ke, rc, rc_p);
                                hex8_blocks_to_slots(slots, ke, local_vals);
                            } else {
                                cvfem_hex8_ns_upwind_jacobian_add_slots<false>(
                                        rho, mu, adj, det, ux_e, uy_e, uz_e, slots, local_vals, rc, rc_p);
                            }
                            break;
                        default: {
                            scalar_t ke[CVFEM_HEX8_N_DOF * CVFEM_HEX8_N_DOF];
                            cvfem_hex8_ns_upwind_jacobian_fd(rho, mu, adj, det, ux_e, uy_e, uz_e, p_e, ke);
                            hex8_local_slots_to_bsr4(slots, ke, local_vals);
                            break;
                        }
                    }
                }
            }
            CVFEM_PHASE_MARK(acc, _t, PH_KERNEL);

            // owned rows: one streaming store over a contiguous global slice
            std::memcpy(gvalues + (ptrdiff_t)rowptr[owned] * 16, local_vals, (size_t)owned_nnz * 16 * sizeof(scalar_t));

            // ghost rows: park for the reduction below
            const ptrdiff_t ghost_off = ghost_ptr[pack];
            if (n_ghost > 0) {
                const ptrdiff_t dest = st_ghost_ptr[(size_t)ghost_off];
                const ptrdiff_t n    = st_ghost_ptr[(size_t)ghost_off + (size_t)n_ghost] - dest;
                std::memcpy(st_ghost_val + dest * 16,
                            local_vals + (ptrdiff_t)owned_nnz * 16,
                            (size_t)n * 16 * sizeof(scalar_t));
            }
            CVFEM_PHASE_MARK_LAST(acc, _t, PH_LOCAL_TO_GLOBAL);
    }
}



#endif  // CVFEM_HEX8_BEST_STORE_HPP
