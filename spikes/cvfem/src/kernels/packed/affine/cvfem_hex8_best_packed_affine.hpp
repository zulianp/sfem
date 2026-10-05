#ifndef CVFEM_HEX8_BEST_PACKED_AFFINE_HPP
#define CVFEM_HEX8_BEST_PACKED_AFFINE_HPP

// The packed layout's AFFINE sweeps.
//
// DESIGN.md's correction: "I indicated separate folders for geometry affine vs isoparametric.
// This implies that the kernels should be separated. Quite obvious isn't it? GeomKind is used at
// the front end level to dispatch based on the type of elements in the block (now smesh also
// provides such enums) and it can be overriden at runtime."
//
// So these are whole sweeps, one per geometry, and the front end chooses between them. What the
// two share -- the slot carve-up, the pack extent, the staging and the drain -- is in
// cvfem_pack_scratch.hpp and cvfem_hex8_best_packed.hpp, where both can call it; that is what
// keeps the split from duplicating anything.
//
// The affine sweep reads ONE adjugate and determinant per element from a precomputed table.
// Every geometric quantity it needs comes from that constant Jacobian, which is what makes the
// variant affine; true per-element geometry belongs to the isoparametric sweep beside it.

#include "kernels/packed/cvfem_hex8_best_packed.hpp"
#include "kernels/packed/cvfem_hex8_best_store.hpp"
#include "kernels/standard/cvfem_hex8_best_atomic.hpp"

// A SEPARATE SWEEP, NOT A TEMPLATE PARAMETER. This note used to argue that `template <bool ISO>`
// already satisfied DESIGN.md's "logically separated (now they are mixed in with enum and
// booleans)", on the grounds that the parenthetical named the enum and the enum was gone. The
// correction settles it: "I indicated separate folders for geometry affine vs isoparametric.
// This implies that the kernels should be separated. Quite obvious isn't it?"
//
// What the template DID fix is still true and still matters: GeomKind used to arrive as an
// argument and be tested per pack, which left the geometry undecided at the point where it
// matters -- the lane loop -- and that is the shape of guard this layout's notes record costing
// 1.83x. Two sweeps fix it the same way and go further: each takes only what it reads.
// The pack sweep, driven by a range. The `#pragma omp parallel` is in the launcher below; see
// kernels/cvfem_range.hpp for why DESIGN.md wants it there. The packs of a range touch only
// nodes this part owns -- that is what the packed layout is for -- so the parts need no
// synchronisation between them, and the ghost rows they do share are reduced afterwards in the
// launcher, which is the second and independent parallel loop.
template <typename scalar_t, typename idx_t, typename pack_idx_t, typename geom_t>
static SFEM_NOINLINE void apply_residual_packed_affine_range(
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
        scalar_t *const SFEM_RESTRICT ghost_buf,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const ptrdiff_t max_actual_nodes_per_pack,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const scalar_t rho,
        const scalar_t mu,
        scalar_t *const SFEM_RESTRICT rx,
        scalar_t *const SFEM_RESTRICT ry,
        scalar_t *const SFEM_RESTRICT rz,
        scalar_t *const SFEM_RESTRICT rc,
        const size_t scratch_n,
        const int with_rc) {
        scalar_t *const SFEM_RESTRICT pack_u   = thread_scratch<scalar_t>(0, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_out = thread_scratch<scalar_t>(1, scratch_n);
        const Hex8PackCoordsT<scalar_t> pk =
                cvfem_hex8_pack_coords<scalar_t>(with_rc != 0, with_rc, max_actual_nodes_per_pack);



    for (ptrdiff_t pack = packs.begin; pack < packs.end; ++pack) {
            const Hex8PackExtentT<idx_t> x = cvfem_hex8_pack_extent<idx_t>(
                    pack, nelements, n_elements_per_pack, owned_nodes_ptr, ghost_idx, ghost_ptr);

            std::memset(pack_out, 0, (size_t)x.n_pack_nodes * (size_t)CVFEM_HEX8_N_FIELDS * sizeof(scalar_t));

            fill_pack_fields(owned_nodes_ptr, ux, uy, uz, pres, pack, x.n_contiguous, x.n_ghost, x.ghosts, pack_u);
            if (with_rc)
                cvfem_hex8_fill_pack_xyz_pgrad(owned_nodes_ptr, points, pgx, pgy, pgz, with_rc, pack, x.n_contiguous, x.n_ghost, x.ghosts, pk.x, pk.y,
                                               pk.z, pk.pgx, pk.pgy, pk.pgz);

            // THE LANE LOOP IS WRITTEN OUT HERE, AND IN THE PACK-COLOURED SWEEP BELOW, AND
            // THE REASON IS NOT THE ONE THIS COMMENT USED TO GIVE.
            //
            // It was shared between them for exactly the reason the one-path rule asks -- the
            // two sweeps differ only in their drain. Grace then reported the bare packed
            // residual at -9.4% and the pack-coloured one at -17.3%, reproduced in two
            // allocations (jobs/ab_refactor.sbatch 4983280, 4983377), so the sharing was
            // reverted and those numbers were written here as its cost.
            //
            // THAT ATTRIBUTION WAS WRONG. The revert did not clear the row: it stayed at -6.7%
            // and then -9.4% in a fresh allocation on another node. The loss was in the
            // LAUNCHER, in the same commit -- its colour loop had replaced
            // `#pragma omp for schedule(dynamic, 1)` with cvfem_range_split, an equal static
            // slice per thread, and a colour's packs are not equal work. Restoring dynamic
            // scheduling took the row to +2.0% and the 21-row gate to PASSED (4983762).
            //
            // So whether sharing this loop costs anything is UNTESTED. One commit changed the
            // kernel's code shape and the work distribution together, and a throughput A/B
            // attributes a loss to a commit, never to a line. The copies stay because that is
            // what is measured clean today; anyone re-sharing them should re-measure rather
            // than trust a number this comment no longer claims.

                alignas(ALIGN_BYTES) scalar_t cof0[cvfem_hex8_vec_size<scalar_t>], cof1[cvfem_hex8_vec_size<scalar_t>], cof2[cvfem_hex8_vec_size<scalar_t>];
                alignas(ALIGN_BYTES) scalar_t cof3[cvfem_hex8_vec_size<scalar_t>], cof4[cvfem_hex8_vec_size<scalar_t>], cof5[cvfem_hex8_vec_size<scalar_t>];
                alignas(ALIGN_BYTES) scalar_t cof6[cvfem_hex8_vec_size<scalar_t>], cof7[cvfem_hex8_vec_size<scalar_t>], cof8[cvfem_hex8_vec_size<scalar_t>];
                alignas(ALIGN_BYTES) scalar_t det[cvfem_hex8_vec_size<scalar_t>];
                Hex8InputPackT<scalar_t>    in;
                Hex8ResidualPackT<scalar_t> outp;
                Hex8RhieChowPackT<scalar_t> rcp;
            for (ptrdiff_t begin = x.e_start; begin < x.e_end; begin += cvfem_hex8_vec_size<scalar_t>) {
                const int nlanes = int(MIN((ptrdiff_t)cvfem_hex8_vec_size<scalar_t>, x.e_end - begin));
                gather_hex8_simd_from_pack(pack_elems, pack_u, adj_ptr, det_ptr, begin, nlanes, in,
                                           cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, det);
                if (with_rc) {
                    cvfem_hex8_gather_rc_from_pack(pack_elems, pk.pgx, pk.pgy, pk.pgz, begin, nlanes, rcp);
                }
                cvfem_hex8_ns_upwind_residual_sumfact_simd(
                        rho, mu, cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, det, in, outp,
                        with_rc ? &rcp : nullptr, rhie_chow_scale);
                scatter_hex8_simd_to_pack(pack_elems, pack_out, begin, nlanes, outp);
            }

            cvfem_hex8_drain_pack_soa(x, pack_out, n_ghost_entries, ghost_buf, rx, ry, rz, rc);
    }
}


// THE EXACT HIGHER-ORDER ACTION is this same sweep with two extra fields staged, so it is this
// same function with two extra arguments rather than a second copy of two hundred lines.
//
//   ugrad  the state's nodal velocity gradient, the field the residual's correction reads;
//   vgrad  the DIRECTION's, reconstructed by its own pass before every matvec.
//
// Both null -- the default -- is the lagged action: the correction is a constant within the
// Newton step, contributes nothing to J, and this function computes exactly what it computed
// before the arguments existed. Passing them carries the correction's derivative, which is what
// makes the action exact for a higher-order residual and is what costs the extra pass.
// The pack sweep, driven by a range. The `#pragma omp parallel` is in the launcher below; see
// kernels/cvfem_range.hpp for why DESIGN.md wants it there. The packs of a range touch only
// nodes this part owns -- that is what the packed layout is for -- so the parts need no
// synchronisation between them, and the ghost rows they do share are reduced afterwards in the
// launcher, which is the second and independent parallel loop.
template <typename scalar_t, typename idx_t, typename pack_idx_t, typename geom_t>
static SFEM_NOINLINE void apply_jacobian_action_packed_affine_range(
        const cvfem_range packs,
        // The mesh and the pack are staging objects -- they own vectors and a shared_ptr to a
        // mesh -- so what this kernel reads out of them is what it takes. DESIGN.md: only
        // arguments that are actually used are passed.
        idx_t **const SFEM_RESTRICT mesh_elems,
        const ptrdiff_t nelements,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT pgx,
        const scalar_t *const SFEM_RESTRICT pgy,
        const scalar_t *const SFEM_RESTRICT pgz,
        geom_t **const SFEM_RESTRICT points,
        const scalar_t *const SFEM_RESTRICT qgx,
        const scalar_t *const SFEM_RESTRICT qgy,
        const scalar_t *const SFEM_RESTRICT qgz,
        const scalar_t *const SFEM_RESTRICT rc_coeff,
        const scalar_t *const SFEM_RESTRICT rc_w,
        const scalar_t rhie_chow_scale,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        pack_idx_t **const SFEM_RESTRICT pack_elems,
        scalar_t *const SFEM_RESTRICT ghost_buf,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const ptrdiff_t max_actual_nodes_per_pack,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const scalar_t rho,
        const scalar_t mu,
        const scalar_t *const dir,
        scalar_t *const jv,
        const scalar_t *const SFEM_RESTRICT ugrad,
        const scalar_t *const SFEM_RESTRICT vgrad,
        const int limiter,
        const scalar_t venkat_c,
        const bool with_ho,
        const size_t scratch_n,
        const int with_rc,
        const bool with_qg,
        const size_t slot3_n,
        // Resolved once per solve, in the launcher, not per element here. This parameter replaced
        // the cvfem_hex8_rc_config_for(d) call that used to sit in this body: that function takes
        // the mesh, which a kernel is not meant to name.
        const Hex8RcConfigT<scalar_t> &rc_cfg,
        // The affine geometry, which this kernel forwards to the pack gather. It used to hand
        // that gather the mesh instead, so the promotion did not see adj_ptr in the body and
        // did not add it here.
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT        det_ptr) {
        // The breakdown covered packed assembly and the colored matvec but not this one --
        // the operator the solver's Krylov loop actually applies. Without it nothing here
        // could be attributed to a phase.
        CVFEM_PHASE_ACC(acc);
        scalar_t *const SFEM_RESTRICT pack_u   = thread_scratch<scalar_t>(0, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_dir = thread_scratch<scalar_t>(1, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_out = thread_scratch<scalar_t>(2, scratch_n);
        const Hex8PackCoordsT<scalar_t> pk =
                cvfem_hex8_pack_coords<scalar_t>(with_rc != 0, with_rc, max_actual_nodes_per_pack);
        const Hex8PackQGradT<scalar_t> qg = cvfem_hex8_pack_qgrad<scalar_t>(with_qg, max_actual_nodes_per_pack);



    for (ptrdiff_t pack = packs.begin; pack < packs.end; ++pack) {
            const Hex8PackExtentT<idx_t> x = cvfem_hex8_pack_extent<idx_t>(
                    pack, nelements, n_elements_per_pack, owned_nodes_ptr, ghost_idx, ghost_ptr);

            CVFEM_PHASE_CLOCK(_t);
            std::memset(pack_out, 0, (size_t)x.n_pack_nodes * (size_t)CVFEM_HEX8_N_FIELDS * sizeof(scalar_t));
            CVFEM_PHASE_MARK(acc, _t, PH_LOCAL_MEMSET);

            fill_pack_fields(owned_nodes_ptr, ux, uy, uz, pres, pack, x.n_contiguous, x.n_ghost, x.ghosts, pack_u);
            fill_pack_interleaved(owned_nodes_ptr, pack, x.n_contiguous, x.n_ghost, x.ghosts, dir, pack_dir);


            // The two gradient packs, staged exactly as apply_residual_packed_defcor stages its
            // one. The limiter and eps^2 live on the state pack because that is where the
            // correction's own kernel reads them; the direction pack carries only the field.
            if (with_rc)
                cvfem_hex8_fill_pack_xyz_pgrad(owned_nodes_ptr, points, pgx, pgy, pgz, with_rc, pack, x.n_contiguous, x.n_ghost, x.ghosts, pk.x, pk.y, pk.z,
                                               pk.pgx, pk.pgy, pk.pgz);
            if (with_qg)
                cvfem_hex8_fill_pack_qgrad(owned_nodes_ptr, qgx, qgy, qgz, pack, x.n_contiguous, x.n_ghost, x.ghosts, qg.x, qg.y, qg.z);
            // No coordinate staging for the geometry: the affine sweep reads one adjugate and
            // determinant per element from the precomputed table. The pack stages coordinates
            // only when Rhie-Chow or the higher-order reconstruction needs them, which the two
            // branches above cover.
            CVFEM_PHASE_MARK(acc, _t, PH_GATHER);

            cvfem_hex8_action_lanes_affine(x, pk, qg, adj_ptr, det_ptr, mesh_elems, points,
                                           pack_elems, pack_u, pack_dir, pack_out, rc_coeff, rc_w,
                                           ugrad, vgrad, rho, mu, rhie_chow_scale, with_rc,
                                           with_qg, with_ho, rc_cfg, limiter, venkat_c);
            CVFEM_PHASE_MARK(acc, _t, PH_KERNEL);

            cvfem_hex8_drain_pack_aos(x, pack_out, n_ghost_entries, ghost_buf, jv);
            CVFEM_PHASE_MARK_LAST(acc, _t, PH_LOCAL_TO_GLOBAL);
    }
        CVFEM_PHASE_FLUSH(acc);
}

// The pack sweep, driven by a range. The `#pragma omp parallel` is in the launcher below; see
// kernels/cvfem_range.hpp for why DESIGN.md wants it there. The packs of a range touch only
// nodes this part owns -- that is what the packed layout is for -- so the parts need no
// synchronisation between them, and the ghost rows they do share are reduced afterwards in the
// launcher, which is the second and independent parallel loop.
template <typename scalar_t, typename idx_t, typename pack_idx_t, typename geom_t, typename count_t>
static SFEM_NOINLINE void assemble_jacobian_packed_affine_range(
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
        const ptrdiff_t *const SFEM_RESTRICT ghost_mat_ptr,
        scalar_t *const SFEM_RESTRICT ghost_mat_val,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const int *const SFEM_RESTRICT local_element_slot,
        const count_t *const *const SFEM_RESTRICT local_global_slot,
        const int *const *const SFEM_RESTRICT local_rowptr,
        const ptrdiff_t max_actual_nodes_per_pack,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        // BSR4 is a staging type (it owns a SharedBuffer and a graph); the kernel reads one
        // array out of it, so that is what it takes.
        scalar_t *const SFEM_RESTRICT gvalues,
        const scalar_t rho,
        const scalar_t mu,
        const size_t u_n,
        const size_t bsr_n,
        const int with_rc,
        // Resolved once per solve, in the launcher, not per element here. This parameter replaced
        // the cvfem_hex8_rc_config_for(d) call that used to sit in this body: that function takes
        // the mesh, which a kernel is not meant to name.
        const Hex8RcConfigT<scalar_t> &rc_cfg,
        // --kernel-only / --dense-flush, resolved by the front end: the identity slot array when
        // the caller wants the element kernel to write a dense stack buffer instead of scattering
        // into pack-local storage, and null otherwise. It replaced three globals the sweep used
        // to read -- DESIGN.md: "No user level option flags are propgated down here ... they are
        // handled outside in the front-end".
        const int *const SFEM_RESTRICT identity_slots) {
        CVFEM_PHASE_ACC(acc);
        alignas(ALIGN_BYTES) scalar_t dense_ke[64 * 16];
        std::memset(dense_ke, 0, sizeof(dense_ke));
        scalar_t *const SFEM_RESTRICT pack_u          = thread_scratch<scalar_t>(0, u_n);
        scalar_t *const SFEM_RESTRICT local_vals_pack = thread_scratch<scalar_t>(2, bsr_n);
        const Hex8PackCoordsT<scalar_t> pk =
                cvfem_hex8_pack_coords<scalar_t>(with_rc != 0, with_rc, max_actual_nodes_per_pack);



    for (ptrdiff_t pack = packs.begin; pack < packs.end; ++pack) {
            const Hex8PackExtentT<idx_t> x = cvfem_hex8_pack_extent<idx_t>(
                    pack, nelements, n_elements_per_pack, owned_nodes_ptr, ghost_idx, ghost_ptr);
            const auto                             &lrowptr      = local_rowptr[(size_t)pack];
            const auto                             &lslots       = local_global_slot[(size_t)pack];
            // lrowptr has exactly x.n_contiguous + x.n_ghost + 1 entries -- build_pack_local_crs
            // assigns it that length -- so back() is the last of them and the empty() guard
            // could only fire on a pack the builder never saw. Indexed rather than called,
            // because this becomes a raw pointer when the kernel stops taking PackedData.
            const int                               local_nnz    = lrowptr[(size_t)(x.n_contiguous + x.n_ghost)];

            CVFEM_PHASE_CLOCK(_t);
            std::memset(local_vals_pack, 0, (size_t)local_nnz * 16 * sizeof(scalar_t));
            CVFEM_PHASE_MARK(acc, _t, PH_LOCAL_MEMSET);

            fill_pack_fields(owned_nodes_ptr, ux, uy, uz, pres, pack, x.n_contiguous, x.n_ghost, x.ghosts, pack_u);
            if (with_rc)
                cvfem_hex8_fill_pack_xyz_pgrad(owned_nodes_ptr, points, pgx, pgy, pgz, with_rc, pack, x.n_contiguous, x.n_ghost, x.ghosts, pk.x, pk.y, pk.z,
                                               pk.pgx, pk.pgy, pk.pgz);
            CVFEM_PHASE_MARK(acc, _t, PH_GATHER);

            for (ptrdiff_t e = x.e_start; e < x.e_end; ++e) {
                Hex8PackElementT<scalar_t> el;
                cvfem_hex8_stage_pack_element(pack_elems, pack_u, pk, e, with_rc, rc_cfg, el);
                const int *const SFEM_RESTRICT slots =
                        identity_slots ? identity_slots : local_element_slot + (size_t)e * 64;
                scalar_t *const SFEM_RESTRICT local_vals = identity_slots ? dense_ke : local_vals_pack;
                // THE GEOMETRY: one adjugate and determinant per element, read from the
                // precomputed table. This is the whole of what distinguishes this sweep from
                // its isoparametric twin, which derives them per sub-control surface from the
                // staged node coordinates instead.
                scalar_t adj[9], det;
                load_hex8_adj(adj_ptr, det_ptr, e, adj, &det);
                cvfem_hex8_ns_upwind_jacobian_add_slots<false>(
                        rho, mu, adj, det, el.ux, el.uy, el.uz, slots, local_vals, el.rc, el.rc_p);
            }

            CVFEM_PHASE_MARK(acc, _t, PH_KERNEL);
            const int                     owned_nnz = x.n_contiguous > 0 ? lrowptr[(size_t)x.n_contiguous] : 0;
            if (!identity_slots)
                for (int t = 0; t < owned_nnz; ++t)
                    bsr4_add16(&gvalues[(ptrdiff_t)lslots[(size_t)t] * 16], local_vals_pack + (ptrdiff_t)t * 16);

            for (ptrdiff_t k = 0; k < x.n_ghost; ++k) {
                const ptrdiff_t local_i = x.n_contiguous + k;
                const int       begin   = lrowptr[(size_t)local_i];
                const int       end     = lrowptr[(size_t)local_i + 1];
                const ptrdiff_t dest    = ghost_mat_ptr[(size_t)x.ghost_off + (size_t)k];
                std::memcpy(ghost_mat_val + dest * 16,
                            local_vals_pack + (ptrdiff_t)begin * 16,
                            (size_t)(end - begin) * 16 * sizeof(scalar_t));
            }
            CVFEM_PHASE_MARK_LAST(acc, _t, PH_LOCAL_TO_GLOBAL);
    }
        CVFEM_PHASE_FLUSH(acc);
}

// Write-once assembly. Each pack accumulates into a cache-resident local matrix
// whose owned rows already carry the global sparsity pattern, then streams that
// block straight into the global BSR with a single memcpy. Every global block is
// written exactly once, so there is no zero_bsr4 pass and no read-modify-write.
// Only the ghost rows, which are shared between packs, need a reduction.
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
template <typename scalar_t, typename idx_t, typename pack_idx_t, typename geom_t, typename count_t>
static SFEM_NOINLINE void assemble_jacobian_store_affine_range(
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
        scalar_t *const SFEM_RESTRICT gvalues,
        const int                     with_rc,
        CVFEM_PHASE_ACC_PARAM
        scalar_t *const SFEM_RESTRICT pack_u,
        scalar_t *const SFEM_RESTRICT local_vals,
        // The pack's staged coordinates and pressure gradient, carved out of slot 3 by the
        // launcher. One object rather than six pointers that have to be offset consistently.
        const Hex8PackCoordsT<scalar_t> &pk,
        // Resolved once per solve, in the launcher, not per element here. This parameter replaced
        // the cvfem_hex8_rc_config_for(d) call that used to sit in this body: that function takes
        // the mesh, which a kernel is not meant to name.
        const Hex8RcConfigT<scalar_t> &rc_cfg,
        // --kernel-only / --dense-flush, resolved by the front end: the identity slot array when
        // the caller wants the element kernel to write a dense stack buffer instead of scattering
        // into pack-local storage, and null otherwise. It replaced three globals the sweep used
        // to read -- DESIGN.md: "No user level option flags are propgated down here ... they are
        // handled outside in the front-end".
        const int *const SFEM_RESTRICT identity_slots) {

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
                cvfem_hex8_fill_pack_xyz_pgrad(owned_nodes_ptr, points, pgx, pgy, pgz, with_rc, pack, n_contiguous, n_ghost, ghosts, pk.x, pk.y, pk.z,
                                               pk.pgx, pk.pgy, pk.pgz);
            CVFEM_PHASE_MARK(acc, _t, PH_GATHER);

            for (ptrdiff_t e = e_start; e < e_end; ++e) {
                Hex8PackElementT<scalar_t> el;
                cvfem_hex8_stage_pack_element(pack_elems, pack_u, pk, e, with_rc, rc_cfg, el);
                const int *const SFEM_RESTRICT slots = st_element_slot + (size_t)e * 64;
                scalar_t adj[9], det;
                load_hex8_adj(adj_ptr, det_ptr, e, adj, &det);
                if (identity_slots) {
                    alignas(ALIGN_BYTES) scalar_t ke[64 * 16] = {};
                    cvfem_hex8_ns_upwind_jacobian_add_slots<false>(
                            rho, mu, adj, det, el.ux, el.uy, el.uz, identity_slots, ke, el.rc, el.rc_p);
                    hex8_blocks_to_slots(slots, ke, local_vals);
                } else {
                    cvfem_hex8_ns_upwind_jacobian_add_slots<false>(
                            rho, mu, adj, det, el.ux, el.uy, el.uz, slots, local_vals, el.rc, el.rc_p);
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

// ---------------------------------------------------------------------------------------------
// THE TWO AFFINE-ONLY SWEEPS. They were never templated on the geometry: both read the adjugate
// table, so there is no isoparametric form of either and nothing to split. They are here because
// the folder is where the affine kernels live, not because they were separated from a twin.
//
//   apply_residual_packed_defcor_range     the deferred-correction higher-order residual
//   apply_jacobian_action_packed_pa_range  the partially assembled Jacobian action (quarantined
//                                          at the driver: 17-19% slower than direct evaluation)

// The pack sweep, driven by a range. The `#pragma omp parallel` is in the launcher below; see
// kernels/cvfem_range.hpp for why DESIGN.md wants it there. The packs of a range touch only
// nodes this part owns -- that is what the packed layout is for -- so the parts need no
// synchronisation between them, and the ghost rows they do share are reduced afterwards in the
// launcher, which is the second and independent parallel loop.
template <typename scalar_t, typename idx_t, typename pack_idx_t, typename geom_t>
static SFEM_NOINLINE void apply_residual_packed_defcor_range(
        const cvfem_range packs,
        // The mesh and the pack are staging objects -- they own vectors and a shared_ptr to a
        // mesh -- so what this kernel reads out of them is what it takes. DESIGN.md: only
        // arguments that are actually used are passed.
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        idx_t **const SFEM_RESTRICT mesh_elems,
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
        scalar_t *const SFEM_RESTRICT ghost_buf,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const ptrdiff_t max_actual_nodes_per_pack,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const scalar_t rho,
        const scalar_t mu,
        const scalar_t *const SFEM_RESTRICT ugrad,
        const int limiter,
        const scalar_t venkat_c,
        scalar_t *const SFEM_RESTRICT rx,
        scalar_t *const SFEM_RESTRICT ry,
        scalar_t *const SFEM_RESTRICT rz,
        scalar_t *const SFEM_RESTRICT rc,
        const size_t scratch_n,
        const int with_rc) {
        scalar_t *const SFEM_RESTRICT pack_u   = thread_scratch<scalar_t>(0, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_out = thread_scratch<scalar_t>(1, scratch_n);
        // Coordinates always, and the pressure gradient when Rhie-Chow is on: the same
        // six-array slot the first-order SIMD path uses, so no new scratch shape appears.
        const Hex8PackCoordsT<scalar_t> pk =
                cvfem_hex8_pack_coords<scalar_t>(true, with_rc, max_actual_nodes_per_pack);


    for (ptrdiff_t pack = packs.begin; pack < packs.end; ++pack) {
            const Hex8PackExtentT<idx_t> x = cvfem_hex8_pack_extent<idx_t>(
                    pack, nelements, n_elements_per_pack, owned_nodes_ptr, ghost_idx, ghost_ptr);

            std::memset(pack_out, 0, (size_t)x.n_pack_nodes * (size_t)CVFEM_HEX8_N_FIELDS * sizeof(scalar_t));
            fill_pack_fields(owned_nodes_ptr, ux, uy, uz, pres, pack, x.n_contiguous, x.n_ghost, x.ghosts, pack_u);
            if (with_rc)
                cvfem_hex8_fill_pack_xyz_pgrad(owned_nodes_ptr, points, pgx, pgy, pgz, with_rc, pack, x.n_contiguous, x.n_ghost, x.ghosts, pk.x,
                                               pk.y, pk.z, pk.pgx, pk.pgy, pk.pgz);
            else
                fill_pack_xyz(owned_nodes_ptr, points, pack, x.n_contiguous, x.n_ghost, x.ghosts, pk.x, pk.y, pk.z);

            alignas(ALIGN_BYTES) scalar_t cof0[CVFEM_HEX8_VEC_SIZE], cof1[CVFEM_HEX8_VEC_SIZE],
                    cof2[CVFEM_HEX8_VEC_SIZE], cof3[CVFEM_HEX8_VEC_SIZE], cof4[CVFEM_HEX8_VEC_SIZE],
                    cof5[CVFEM_HEX8_VEC_SIZE], cof6[CVFEM_HEX8_VEC_SIZE], cof7[CVFEM_HEX8_VEC_SIZE],
                    cof8[CVFEM_HEX8_VEC_SIZE], detv[CVFEM_HEX8_VEC_SIZE];
            Hex8InputPackT<scalar_t>    in;
            Hex8ResidualPackT<scalar_t> outp;
            Hex8RhieChowPackT<scalar_t> rcp;
            Hex8UGradPackT<scalar_t>    hop;
            hop.limiter  = limiter;
            hop.venkat_c = venkat_c;
            for (ptrdiff_t begin = x.e_start; begin < x.e_end; begin += CVFEM_HEX8_VEC_SIZE) {
                const int nlanes = int(MIN((ptrdiff_t)CVFEM_HEX8_VEC_SIZE, x.e_end - begin));
                gather_hex8_simd_from_pack(pack_elems, pack_u, adj_ptr, det_ptr, begin, nlanes, in,
                                           cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, detv);
                if (with_rc) {
                    cvfem_hex8_gather_rc_from_pack(pack_elems, pk.pgx,
                                                   pk.pgy, pk.pgz, begin, nlanes, rcp);
                }
                for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
                    const ptrdiff_t e = begin + lane;
                    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                        if (lane >= nlanes) {
                            hop.x[a][lane] = hop.y[a][lane] = hop.z[a][lane] = scalar_t(0);
                            for (int c = 0; c < 9; ++c) hop.g[a][c][lane] = scalar_t(0);
                            continue;
                        }
                        const idx_t g = mesh_elems[a][e];
                        hop.x[a][lane] = scalar_t(points[0][g]);
                        hop.y[a][lane] = scalar_t(points[1][g]);
                        hop.z[a][lane] = scalar_t(points[2][g]);
                        for (int c = 0; c < 9; ++c) hop.g[a][c][lane] = ugrad[(ptrdiff_t)g * 9 + c];
                    }
                }
                cvfem_hex8_ns_upwind_residual_sumfact_simd(
                        rho, mu, cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, detv, in,
                        outp, with_rc ? &rcp : nullptr, rhie_chow_scale, scalar_t(0), &hop);
                scatter_hex8_simd_to_pack(pack_elems, pack_out, begin, nlanes, outp);
            }
            cvfem_hex8_drain_pack_soa(x, pack_out, n_ghost_entries, ghost_buf, rx, ry, rz, rc);
    }
}

// ------------------------------------------------- the partially assembled Jacobian action
//
// The same operator as apply_jacobian_action_packed, reading the state out of a store built
// once per Newton step instead of gathering and re-deriving it on every matvec. What
// disappears from the per-matvec work, in order of size:
//
//   * fill_pack_fields -- the state u,p is never staged at all;
//   * the pgx/pgy/pgz half of cvfem_hex8_fill_pack_xyz_pgrad, and with it half of the 768
//     doubles cvfem_hex8_gather_rc_from_pack moves per SIMD group. The coordinates stay,
//     because the direction's reconstructed gradient is contracted against the edge vectors;
//   * per element, twelve mass fluxes, twelve Rhie-Chow corrections and twelve upwind
//     switches.
//
// and what appears is one contiguous SoA read of sixty scalars per element.
//
// Affine only. The store holds a per-element tangent built from one adjugate at the element
// centre, which is not what the isoparametric kernels evaluate -- they build an area vector
// per sub-control surface from a trilinear Jacobian. The driver refuses the combination
// rather than measuring a store that describes a different operator.
// The pack sweep, driven by a range. The `#pragma omp parallel` is in the launcher below; see
// kernels/cvfem_range.hpp for why DESIGN.md wants it there. The packs of a range touch only
// nodes this part owns -- that is what the packed layout is for -- so the parts need no
// synchronisation between them, and the ghost rows they do share are reduced afterwards in the
// launcher, which is the second and independent parallel loop.
template <typename scalar_t, typename idx_t, typename pack_idx_t, typename geom_t>
static SFEM_NOINLINE void apply_jacobian_action_packed_pa_range(
        const cvfem_range packs,
        // The mesh and the pack are staging objects -- they own vectors and a shared_ptr to a
        // mesh -- so what this kernel reads out of them is what it takes. DESIGN.md: only
        // arguments that are actually used are passed.
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        const ptrdiff_t nelements,
        scalar_t *const SFEM_RESTRICT pa_tangent,
        geom_t **const SFEM_RESTRICT points,
        const scalar_t *const SFEM_RESTRICT qgx,
        const scalar_t *const SFEM_RESTRICT qgy,
        const scalar_t *const SFEM_RESTRICT qgz,
        const scalar_t *const SFEM_RESTRICT rc_coeff,
        const scalar_t *const SFEM_RESTRICT rc_w,
        const scalar_t rhie_chow_scale,
        pack_idx_t **const SFEM_RESTRICT pack_elems,
        scalar_t *const SFEM_RESTRICT ghost_buf,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const ptrdiff_t max_actual_nodes_per_pack,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const scalar_t rho,
        const scalar_t mu,
        const scalar_t *const dir,
        scalar_t *const jv,
        const size_t scratch_n,
        const int with_rc,
        const bool with_qg,
        // Resolved once per solve, in the launcher, not per element here. This parameter replaced
        // the cvfem_hex8_rc_config_for(d) call that used to sit in this body: that function takes
        // the mesh, which a kernel is not meant to name.
        const Hex8RcConfigT<scalar_t> &rc_cfg) {
        CVFEM_PHASE_ACC(acc);
        scalar_t *const SFEM_RESTRICT pack_dir = thread_scratch<scalar_t>(1, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_out = thread_scratch<scalar_t>(2, scratch_n);
        // Three arrays in slot 3, not six: the nodal pressure gradient is inside the store.
        const Hex8PackCoordsT<scalar_t> pk =
                cvfem_hex8_pack_coords<scalar_t>(with_qg, /*with_rc=*/0, max_actual_nodes_per_pack);
        const Hex8PackQGradT<scalar_t> qg = cvfem_hex8_pack_qgrad<scalar_t>(with_qg, max_actual_nodes_per_pack);



    for (ptrdiff_t pack = packs.begin; pack < packs.end; ++pack) {
            const Hex8PackExtentT<idx_t> x = cvfem_hex8_pack_extent<idx_t>(
                    pack, nelements, n_elements_per_pack, owned_nodes_ptr, ghost_idx, ghost_ptr);

            CVFEM_PHASE_CLOCK(_t);
            std::memset(pack_out, 0, (size_t)x.n_pack_nodes * (size_t)CVFEM_HEX8_N_FIELDS * sizeof(scalar_t));
            CVFEM_PHASE_MARK(acc, _t, PH_LOCAL_MEMSET);

            fill_pack_interleaved(owned_nodes_ptr, pack, x.n_contiguous, x.n_ghost, x.ghosts, dir, pack_dir);
            if (with_qg) {
                fill_pack_xyz(owned_nodes_ptr, points, pack, x.n_contiguous, x.n_ghost, x.ghosts, pk.x, pk.y, pk.z);
                cvfem_hex8_fill_pack_qgrad(owned_nodes_ptr, qgx, qgy, qgz, pack, x.n_contiguous, x.n_ghost, x.ghosts, qg.x, qg.y, qg.z);
            }
            CVFEM_PHASE_MARK(acc, _t, PH_GATHER);

            Hex8InputPackT<scalar_t>    du_pack;
            Hex8ResidualPackT<scalar_t> outp;
            Hex8RhieChowPackT<scalar_t> rcp;
            for (ptrdiff_t begin = x.e_start; begin < x.e_end; begin += CVFEM_HEX8_VEC_SIZE) {
                const int nlanes = int(MIN((ptrdiff_t)CVFEM_HEX8_VEC_SIZE, x.e_end - begin));
                alignas(ALIGN_BYTES) scalar_t cof0[CVFEM_HEX8_VEC_SIZE], cof1[CVFEM_HEX8_VEC_SIZE],
                        cof2[CVFEM_HEX8_VEC_SIZE];
                alignas(ALIGN_BYTES) scalar_t cof3[CVFEM_HEX8_VEC_SIZE], cof4[CVFEM_HEX8_VEC_SIZE],
                        cof5[CVFEM_HEX8_VEC_SIZE];
                alignas(ALIGN_BYTES) scalar_t cof6[CVFEM_HEX8_VEC_SIZE], cof7[CVFEM_HEX8_VEC_SIZE],
                        cof8[CVFEM_HEX8_VEC_SIZE];
                alignas(ALIGN_BYTES) scalar_t det[CVFEM_HEX8_VEC_SIZE];
                gather_hex8_simd_from_pack(pack_elems, pack_dir, adj_ptr, det_ptr, begin, nlanes, du_pack, cof0, cof1, cof2, cof3,
                                           cof4, cof5, cof6, cof7, cof8, det);
                if (with_rc) cvfem_hex8_gather_rc_coeff(rc_coeff, rc_w, rc_cfg, begin, nlanes, rcp);
                if (with_qg) {
                    cvfem_hex8_gather_qg_from_pack(pack_elems, qg.x, qg.y, qg.z, begin, nlanes, rcp);
                }
                cvfem_hex8_ns_upwind_jacobian_action_pa_simd(rho, mu, cof0, cof1, cof2, cof3, cof4, cof5, cof6,
                                                             cof7, cof8, det, du_pack,
                                                             pa_tangent + begin, nelements, outp,
                                                             with_rc ? &rcp : nullptr, rhie_chow_scale, with_qg);
                scatter_hex8_simd_to_pack(pack_elems, pack_out, begin, nlanes, outp);
            }
            CVFEM_PHASE_MARK(acc, _t, PH_KERNEL);

            cvfem_hex8_drain_pack_aos(x, pack_out, n_ghost_entries, ghost_buf, jv);
            CVFEM_PHASE_MARK_LAST(acc, _t, PH_LOCAL_TO_GLOBAL);
    }
        CVFEM_PHASE_FLUSH(acc);
}

// ---------------------------------------------------------------------------------------------
// PACK COLOURING: the same affine lane loop, drained differently.
//
// DESIGN.md's correction asks for the geometries to be separate kernels, and pack colouring is a
// packed layout -- it keeps the packed staging and colours the PACKS -- so its sweeps belong
// here, beside the contiguous ones, not in colored/, which is the ELEMENT colouring.
//
// What differs from apply_residual_packed_affine_range is one thing: no two packs of a colour
// share a node, so this accumulates straight into the global arrays instead of staging the
// shared rows for a second reduction pass. That pass disappearing is the whole method. The lane
// loop itself is the same function both call.
//
// Two more consequences of the colouring, both the caller's business: the global arrays must be
// zeroed before the first colour (the drain accumulates), and the colours must be separated by a
// barrier. The launcher owns the colour loop and does both.
template <typename scalar_t, typename idx_t, typename pack_idx_t, typename geom_t>
static SFEM_NOINLINE void apply_residual_packcolored_affine_range(
        // The packs of ONE COLOUR this call is to cover, as indices into pack_order.
        const cvfem_range packs,
        const ptrdiff_t *const SFEM_RESTRICT pack_order,
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
        const ptrdiff_t max_actual_nodes_per_pack,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const scalar_t rho,
        const scalar_t mu,
        scalar_t *const SFEM_RESTRICT rx,
        scalar_t *const SFEM_RESTRICT ry,
        scalar_t *const SFEM_RESTRICT rz,
        scalar_t *const SFEM_RESTRICT rc,
        const size_t scratch_n,
        const int with_rc) {
    scalar_t *const SFEM_RESTRICT pack_u   = thread_scratch<scalar_t>(0, scratch_n);
    scalar_t *const SFEM_RESTRICT pack_out = thread_scratch<scalar_t>(1, scratch_n);
    const Hex8PackCoordsT<scalar_t> pk =
            cvfem_hex8_pack_coords<scalar_t>(with_rc != 0, with_rc, max_actual_nodes_per_pack);


    for (ptrdiff_t i = packs.begin; i < packs.end; ++i) {
        const ptrdiff_t pack = pack_order[i];
        const Hex8PackExtentT<idx_t> x = cvfem_hex8_pack_extent<idx_t>(
                pack, nelements, n_elements_per_pack, owned_nodes_ptr, ghost_idx, ghost_ptr);

        std::memset(pack_out, 0, (size_t)x.n_pack_nodes * (size_t)CVFEM_HEX8_N_FIELDS * sizeof(scalar_t));
        fill_pack_fields(owned_nodes_ptr, ux, uy, uz, pres, pack, x.n_contiguous, x.n_ghost, x.ghosts, pack_u);
        if (with_rc)
            cvfem_hex8_fill_pack_xyz_pgrad(owned_nodes_ptr, points, pgx, pgy, pgz, with_rc, pack,
                                           x.n_contiguous, x.n_ghost, x.ghosts, pk.x, pk.y, pk.z,
                                           pk.pgx, pk.pgy, pk.pgz);

            // THE LANE LOOP IS WRITTEN OUT HERE, AND IN THE PACK-COLOURED SWEEP BELOW, AND
            // THE REASON IS NOT THE ONE THIS COMMENT USED TO GIVE.
            //
            // It was shared between them for exactly the reason the one-path rule asks -- the
            // two sweeps differ only in their drain. Grace then reported the bare packed
            // residual at -9.4% and the pack-coloured one at -17.3%, reproduced in two
            // allocations (jobs/ab_refactor.sbatch 4983280, 4983377), so the sharing was
            // reverted and those numbers were written here as its cost.
            //
            // THAT ATTRIBUTION WAS WRONG. The revert did not clear the row: it stayed at -6.7%
            // and then -9.4% in a fresh allocation on another node. The loss was in the
            // LAUNCHER, in the same commit -- its colour loop had replaced
            // `#pragma omp for schedule(dynamic, 1)` with cvfem_range_split, an equal static
            // slice per thread, and a colour's packs are not equal work. Restoring dynamic
            // scheduling took the row to +2.0% and the 21-row gate to PASSED (4983762).
            //
            // So whether sharing this loop costs anything is UNTESTED. One commit changed the
            // kernel's code shape and the work distribution together, and a throughput A/B
            // attributes a loss to a commit, never to a line. The copies stay because that is
            // what is measured clean today; anyone re-sharing them should re-measure rather
            // than trust a number this comment no longer claims.

            alignas(ALIGN_BYTES) scalar_t cof0[cvfem_hex8_vec_size<scalar_t>], cof1[cvfem_hex8_vec_size<scalar_t>], cof2[cvfem_hex8_vec_size<scalar_t>];
            alignas(ALIGN_BYTES) scalar_t cof3[cvfem_hex8_vec_size<scalar_t>], cof4[cvfem_hex8_vec_size<scalar_t>], cof5[cvfem_hex8_vec_size<scalar_t>];
            alignas(ALIGN_BYTES) scalar_t cof6[cvfem_hex8_vec_size<scalar_t>], cof7[cvfem_hex8_vec_size<scalar_t>], cof8[cvfem_hex8_vec_size<scalar_t>];
            alignas(ALIGN_BYTES) scalar_t det[cvfem_hex8_vec_size<scalar_t>];
            Hex8InputPackT<scalar_t>    in;
            Hex8ResidualPackT<scalar_t> outp;
            Hex8RhieChowPackT<scalar_t> rcp;
            for (ptrdiff_t begin = x.e_start; begin < x.e_end; begin += cvfem_hex8_vec_size<scalar_t>) {
                const int nlanes = int(MIN((ptrdiff_t)cvfem_hex8_vec_size<scalar_t>, x.e_end - begin));
                gather_hex8_simd_from_pack(pack_elems, pack_u, adj_ptr, det_ptr, begin, nlanes, in,
                                           cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, det);
                if (with_rc) {
                    cvfem_hex8_gather_rc_from_pack(pack_elems, pk.pgx, pk.pgy, pk.pgz, begin, nlanes, rcp);
                }
                cvfem_hex8_ns_upwind_residual_sumfact_simd(
                        rho, mu, cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, det, in, outp,
                        with_rc ? &rcp : nullptr, rhie_chow_scale);
                scatter_hex8_simd_to_pack(pack_elems, pack_out, begin, nlanes, outp);
            }

        cvfem_hex8_flush_pack_to_global_soa(x, pack_out, rx, ry, rz, rc);
    }
}

// PACK COLOURING, the Jacobian action. Same method as the residual's coloured twin: it
// accumulates straight into jv instead of staging shared rows for a reduction pass, which is
// what the colouring buys. The lane loop is the same function the contiguous sweep calls.
//
// No higher-order correction: ugrad and vgrad are null, so that branch folds away in the loop.
template <typename scalar_t, typename idx_t, typename pack_idx_t, typename geom_t>
static SFEM_NOINLINE void apply_jacobian_action_packcolored_affine_range(
        const cvfem_range packs,
        const ptrdiff_t *const SFEM_RESTRICT pack_order,
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        idx_t **const SFEM_RESTRICT mesh_elems,
        const ptrdiff_t nelements,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT pgx,
        const scalar_t *const SFEM_RESTRICT pgy,
        const scalar_t *const SFEM_RESTRICT pgz,
        geom_t **const SFEM_RESTRICT points,
        const scalar_t *const SFEM_RESTRICT qgx,
        const scalar_t *const SFEM_RESTRICT qgy,
        const scalar_t *const SFEM_RESTRICT qgz,
        const scalar_t *const SFEM_RESTRICT rc_coeff,
        const scalar_t *const SFEM_RESTRICT rc_w,
        const scalar_t rhie_chow_scale,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        pack_idx_t **const SFEM_RESTRICT pack_elems,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const ptrdiff_t max_actual_nodes_per_pack,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const scalar_t rho,
        const scalar_t mu,
        const scalar_t *const dir,
        scalar_t *const SFEM_RESTRICT jv,
        const size_t scratch_n,
        const int with_rc,
        const bool with_qg,
        const Hex8RcConfigT<scalar_t> &rc_cfg) {
    scalar_t *const SFEM_RESTRICT pack_u   = thread_scratch<scalar_t>(0, scratch_n);
    scalar_t *const SFEM_RESTRICT pack_dir = thread_scratch<scalar_t>(1, scratch_n);
    scalar_t *const SFEM_RESTRICT pack_out = thread_scratch<scalar_t>(2, scratch_n);
    const Hex8PackCoordsT<scalar_t> pk =
            cvfem_hex8_pack_coords<scalar_t>(with_rc != 0, with_rc, max_actual_nodes_per_pack);
    const Hex8PackQGradT<scalar_t> qg = cvfem_hex8_pack_qgrad<scalar_t>(with_qg, max_actual_nodes_per_pack);


    for (ptrdiff_t i = packs.begin; i < packs.end; ++i) {
        const ptrdiff_t pack = pack_order[i];
        const Hex8PackExtentT<idx_t> x = cvfem_hex8_pack_extent<idx_t>(
                pack, nelements, n_elements_per_pack, owned_nodes_ptr, ghost_idx, ghost_ptr);

        std::memset(pack_out, 0, (size_t)x.n_pack_nodes * (size_t)CVFEM_HEX8_N_FIELDS * sizeof(scalar_t));
        fill_pack_fields(owned_nodes_ptr, ux, uy, uz, pres, pack, x.n_contiguous, x.n_ghost, x.ghosts, pack_u);
        fill_pack_interleaved(owned_nodes_ptr, pack, x.n_contiguous, x.n_ghost, x.ghosts, dir, pack_dir);
        if (with_rc)
            cvfem_hex8_fill_pack_xyz_pgrad(owned_nodes_ptr, points, pgx, pgy, pgz, with_rc, pack,
                                           x.n_contiguous, x.n_ghost, x.ghosts, pk.x, pk.y, pk.z,
                                           pk.pgx, pk.pgy, pk.pgz);
        if (with_qg)
            cvfem_hex8_fill_pack_qgrad(owned_nodes_ptr, qgx, qgy, qgz, pack, x.n_contiguous,
                                       x.n_ghost, x.ghosts, qg.x, qg.y, qg.z);

        cvfem_hex8_action_lanes_affine(x, pk, qg, adj_ptr, det_ptr,
                                       mesh_elems, points, pack_elems, pack_u, pack_dir, pack_out,
                                       rc_coeff, rc_w,
                                       // No higher-order correction on this path, and a
                                       // nullptr literal deduces nothing.
                                       static_cast<const scalar_t *>(nullptr),
                                       static_cast<const scalar_t *>(nullptr),
                                       rho, mu, rhie_chow_scale, with_rc, with_qg,
                                       /*with_ho=*/false, rc_cfg, 0, scalar_t(0));

        cvfem_hex8_flush_pack_to_global_aos(x, pack_out, jv);
    }
}

// PACK COLOURING, the assembled Jacobian. Unlike the coloured residual and action, this sweep
// is ELEMENT-indexed with global gathers -- it is the atomic assembly's element body with the
// colouring standing in for the atomics, not a packed sweep with a different drain. The pack
// supplies only the element RANGE and the colour ordering.
//
// So what it shares is cvfem_hex8_assemble_element_affine, with ATOMIC false: no two elements
// of a colour share a node, so the accumulation into the global matrix is a plain `+=`. The
// atomic sweep calls the same body with ATOMIC true and pays per entry.
template <typename scalar_t, typename idx_t, typename geom_t>
static SFEM_NOINLINE void assemble_jacobian_packcolored_affine_range(
        // The packs of ONE COLOUR, as indices into pack_order.
        const cvfem_range packs,
        const ptrdiff_t *const SFEM_RESTRICT pack_order,
        idx_t **const SFEM_RESTRICT mesh_elems,
        geom_t **const SFEM_RESTRICT points,
        const uint8_t *const SFEM_RESTRICT face_mask,
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        const ptrdiff_t nelements,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT pgx,
        const scalar_t *const SFEM_RESTRICT pgy,
        const scalar_t *const SFEM_RESTRICT pgz,
        const scalar_t *const SFEM_RESTRICT qgx,
        const scalar_t *const SFEM_RESTRICT qgy,
        const scalar_t *const SFEM_RESTRICT qgz,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src,
        const Hex8Extras &opt,
        const int *const SFEM_RESTRICT slots,
        const ptrdiff_t n_elements_per_pack,
        const scalar_t rho,
        const scalar_t mu,
        scalar_t *const SFEM_RESTRICT values) {

    for (ptrdiff_t i = packs.begin; i < packs.end; ++i) {
        const ptrdiff_t pack    = pack_order[i];
        const ptrdiff_t e_start = pack * n_elements_per_pack;
        const ptrdiff_t e_end   = MIN(nelements, (pack + 1) * n_elements_per_pack);
        for (ptrdiff_t e = e_start; e < e_end; ++e)
            cvfem_hex8_assemble_element_affine<false>(
                    mesh_elems, points, face_mask, adj_ptr, det_ptr, pres, pgx, pgy, pgz, qgx, qgy,
                    qgz, ux_src, uy_src, uz_src, opt, slots, e, rho, mu, values);
    }
}

#endif  // CVFEM_HEX8_BEST_PACKED_AFFINE_HPP
