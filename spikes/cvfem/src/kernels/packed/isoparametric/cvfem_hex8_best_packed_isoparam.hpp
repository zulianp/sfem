#ifndef CVFEM_HEX8_BEST_PACKED_ISOPARAM_HPP
#define CVFEM_HEX8_BEST_PACKED_ISOPARAM_HPP

// The packed layout's ISOPARAMETRIC sweeps.
//
// The other half of the split DESIGN.md's correction asks for; see
// packed/affine/cvfem_hex8_best_packed_affine.hpp for the reasoning and for where the shared
// pack machinery lives.
//
// These sweeps derive the Jacobian per sub-control surface from the element's node
// coordinates, which the pack stages for them -- so they take `points` and no adjugate table.
//
// SIX PARAMETERS FEWER THAN THE TEMPLATED SWEEP THEY CAME FROM, which is the clause the
// template could not satisfy: "the signatures of the functions are lean-and-mean only arguments
// that are acually used are passed". The isoparametric SIMD kernels carry no Rhie-Chow term --
// it was never put into them -- so the driver refuses --rhie-chow on this geometry for any
// pack-based layout, and with_rc was dead here. Gone with it: adj_ptr, det_ptr, pgx, pgy, pgz,
// rhie_chow_scale and the staging call that filled a pressure gradient nothing read.

#include "kernels/packed/cvfem_hex8_best_packed.hpp"
#include "kernels/packed/cvfem_hex8_best_store.hpp"

template <typename scalar_t, typename idx_t, typename pack_idx_t, typename geom_t>
static SFEM_NOINLINE void apply_residual_packed_isoparam_range(
        const cvfem_range packs,
        // The mesh and the pack are staging objects -- they own vectors and a shared_ptr to a
        // mesh -- so what this kernel reads out of them is what it takes. DESIGN.md: only
        // arguments that are actually used are passed.
        const ptrdiff_t nelements,
        const scalar_t *const SFEM_RESTRICT pres,
        geom_t **const SFEM_RESTRICT points,
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
        const size_t scratch_n) {
        scalar_t *const SFEM_RESTRICT pack_u   = thread_scratch<scalar_t>(0, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_out = thread_scratch<scalar_t>(1, scratch_n);
        // Coordinates always: this geometry derives its Jacobian from them.
        const Hex8PackCoordsT<scalar_t> pk = cvfem_hex8_pack_coords<scalar_t>(true, 0, max_actual_nodes_per_pack);


    for (ptrdiff_t pack = packs.begin; pack < packs.end; ++pack) {
            const Hex8PackExtentT<idx_t> x = cvfem_hex8_pack_extent<idx_t>(
                    pack, nelements, n_elements_per_pack, owned_nodes_ptr, ghost_idx, ghost_ptr);

            std::memset(pack_out, 0, (size_t)x.n_pack_nodes * (size_t)CVFEM_HEX8_N_FIELDS * sizeof(scalar_t));

            fill_pack_fields(owned_nodes_ptr, ux, uy, uz, pres, pack, x.n_contiguous, x.n_ghost, x.ghosts, pack_u);

            fill_pack_xyz(owned_nodes_ptr, points, pack, x.n_contiguous, x.n_ghost, x.ghosts, pk.x, pk.y, pk.z);
            cvfem_hex8_residual_lanes_isoparam(x, pk, pack_elems, pack_u, pack_out, rho, mu);

            cvfem_hex8_drain_pack_soa(x, pack_out, n_ghost_entries, ghost_buf, rx, ry, rz, rc);
    }
}


// The ISOPARAMETRIC Jacobian action. It derives the Jacobian per sub-control surface from the
// element's node coordinates, which the pack stages for it.
//
// TWELVE PARAMETERS FEWER than the templated sweep it came from. Its SIMD kernel takes the
// coordinates and the two lane packs and nothing else: no Rhie-Chow term (never put into the
// isoparametric kernels, which is why the driver refuses --rhie-chow on this geometry for a
// pack-based layout), no staged direction gradient, no adjugate table, no boundary extras. All
// of that was dead in this half of the `if constexpr` and had to be in the signature anyway,
// because a sweep templated on the geometry takes the union of both halves' needs.
template <typename scalar_t, typename idx_t, typename pack_idx_t, typename geom_t>
static SFEM_NOINLINE void apply_jacobian_action_packed_isoparam_range(
        const cvfem_range packs,
        // The mesh and the pack are staging objects -- they own vectors and a shared_ptr to a
        // mesh -- so what this kernel reads out of them is what it takes. DESIGN.md: only
        // arguments that are actually used are passed.
        idx_t **const SFEM_RESTRICT mesh_elems,
        const ptrdiff_t nelements,
        const scalar_t *const SFEM_RESTRICT pres,
        geom_t **const SFEM_RESTRICT points,
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
        const size_t slot3_n) {
        // The breakdown covered packed assembly and the colored matvec but not this one --
        // the operator the solver's Krylov loop actually applies. Without it nothing here
        // could be attributed to a phase.
        CVFEM_PHASE_ACC(acc);
        scalar_t *const SFEM_RESTRICT pack_u   = thread_scratch<scalar_t>(0, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_dir = thread_scratch<scalar_t>(1, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_out = thread_scratch<scalar_t>(2, scratch_n);
        const Hex8PackCoordsT<scalar_t> pk =
                cvfem_hex8_pack_coords<scalar_t>(/*want_xyz=*/true, /*with_rc=*/0, max_actual_nodes_per_pack);


    for (ptrdiff_t pack = packs.begin; pack < packs.end; ++pack) {
            const Hex8PackExtentT<idx_t> x = cvfem_hex8_pack_extent<idx_t>(
                    pack, nelements, n_elements_per_pack, owned_nodes_ptr, ghost_idx, ghost_ptr);

            CVFEM_PHASE_CLOCK(_t);
            std::memset(pack_out, 0, (size_t)x.n_pack_nodes * (size_t)CVFEM_HEX8_N_FIELDS * sizeof(scalar_t));
            CVFEM_PHASE_MARK(acc, _t, PH_LOCAL_MEMSET);

            fill_pack_fields(owned_nodes_ptr, ux, uy, uz, pres, pack, x.n_contiguous, x.n_ghost, x.ghosts, pack_u);
            fill_pack_interleaved(owned_nodes_ptr, pack, x.n_contiguous, x.n_ghost, x.ghosts, dir, pack_dir);

            Hex8InputPackT<scalar_t>    u_pack;
            Hex8InputPackT<scalar_t>    du_pack;
            Hex8ResidualPackT<scalar_t> outp;
            Hex8CoordPackT<scalar_t>    xyz;
            Hex8RhieChowPackT<scalar_t> rcp;
            // The two gradient packs, staged exactly as apply_residual_packed_defcor stages its
            // one. The limiter and eps^2 live on the state pack because that is where the
            // correction's own kernel reads them; the direction pack carries only the field.
            Hex8UGradPackT<scalar_t>    hop, hovp;
            hop.limiter  = limiter;
            hop.venkat_c = venkat_c;
            // Coordinates always: this geometry derives its Jacobian from them.
                fill_pack_xyz(owned_nodes_ptr, points, pack, x.n_contiguous, x.n_ghost, x.ghosts, pk.x, pk.y, pk.z);
            CVFEM_PHASE_MARK(acc, _t, PH_GATHER);

            for (ptrdiff_t begin = x.e_start; begin < x.e_end; begin += CVFEM_HEX8_VEC_SIZE) {
                const int nlanes = int(MIN((ptrdiff_t)CVFEM_HEX8_VEC_SIZE, x.e_end - begin));
                gather_hex8_isoparam_action_simd_from_pack(pack_elems,
                                                           pack_u,
                                                           pack_dir,
                                                           pk.x,
                                                           pk.y,
                                                           pk.z,
                                                           begin,
                                                           nlanes,
                                                           u_pack,
                                                           du_pack,
                                                           xyz);
                cvfem_hex8_ns_upwind_jacobian_action_isoparam_simd(rho, mu, xyz, u_pack, du_pack, outp);
                scatter_hex8_simd_to_pack(pack_elems, pack_out, begin, nlanes, outp);
            }
            CVFEM_PHASE_MARK(acc, _t, PH_KERNEL);

            cvfem_hex8_drain_pack_aos(x, pack_out, n_ghost_entries, ghost_buf, jv);
            CVFEM_PHASE_MARK_LAST(acc, _t, PH_LOCAL_TO_GLOBAL);
    }
        CVFEM_PHASE_FLUSH(acc);
}

// The ISOPARAMETRIC assembly. Scalar per element rather than lane-blocked -- the local matrix
// is large enough that the round trip through a lane pack costs more than it saves -- so this
// and the affine sweep beside it share the element staging (cvfem_hex8_stage_pack_element) and
// differ only in where the geometry comes from and which kernel reads it.
//
// It takes no adjugate table: the Jacobian comes per sub-control surface from the node
// coordinates the pack stages.
template <typename scalar_t, typename idx_t, typename pack_idx_t, typename geom_t, typename count_t>
static SFEM_NOINLINE void assemble_jacobian_packed_isoparam_range(
        const cvfem_range packs,
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
                cvfem_hex8_pack_coords<scalar_t>(/*want_xyz=*/true, with_rc, max_actual_nodes_per_pack);


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
            // Coordinates always: this geometry derives its Jacobian from them.
            fill_pack_xyz(owned_nodes_ptr, points, pack, x.n_contiguous, x.n_ghost, x.ghosts, pk.x, pk.y, pk.z);
            CVFEM_PHASE_MARK(acc, _t, PH_GATHER);

            for (ptrdiff_t e = x.e_start; e < x.e_end; ++e) {
                Hex8PackElementT<scalar_t> el;
                cvfem_hex8_stage_pack_element(pack_elems, pack_u, pk, e, with_rc, rc_cfg, el);
                const int *const SFEM_RESTRICT slots =
                        identity_slots ? identity_slots : local_element_slot + (size_t)e * 64;
                scalar_t *const SFEM_RESTRICT local_vals = identity_slots ? dense_ke : local_vals_pack;
                scalar_t x[8], y[8], z[8];
                gather_hex8_coords_from_pack(pack_elems, pk.x, pk.y, pk.z, e, x, y, z);
                cvfem_hex8_ns_upwind_jacobian_add_slots_isoparam<false>(
                        rho, mu, x, y, z, el.ux, el.uy, el.uz, slots, local_vals, el.rc, el.rc_p);
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

// The STORE layout's isoparametric assembly. Same two-geometry split as the packed one;
// the store's difference is its drain, not its element kernel.
template <typename scalar_t, typename idx_t, typename pack_idx_t, typename geom_t, typename count_t>
static SFEM_NOINLINE void assemble_jacobian_store_isoparam_range(
        const cvfem_range packs,
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
        const Hex8RcConfigT<scalar_t> &rc_cfg) {
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
            // Coordinates always: this geometry derives its Jacobian from them.
            fill_pack_xyz(owned_nodes_ptr, points, pack, n_contiguous, n_ghost, ghosts, pk.x, pk.y, pk.z);
            CVFEM_PHASE_MARK(acc, _t, PH_GATHER);

            for (ptrdiff_t e = e_start; e < e_end; ++e) {
                Hex8PackElementT<scalar_t> el;
                cvfem_hex8_stage_pack_element(pack_elems, pack_u, pk, e, with_rc, rc_cfg, el);
                const int *const SFEM_RESTRICT slots = st_element_slot + (size_t)e * 64;
                scalar_t x[8], y[8], z[8];
                gather_hex8_coords_from_pack(pack_elems, pk.x, pk.y, pk.z, e, x, y, z);
                cvfem_hex8_ns_upwind_jacobian_add_slots_isoparam<false>(
                        rho, mu, x, y, z, el.ux, el.uy, el.uz, slots, local_vals, el.rc, el.rc_p);
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
// PACK COLOURING, isoparametric. See the affine twin in ../affine/ for the method; this half
// differs from apply_residual_packed_isoparam_range in the drain and nothing else.
template <typename scalar_t, typename idx_t, typename pack_idx_t, typename geom_t>
static SFEM_NOINLINE void apply_residual_packcolored_isoparam_range(
        const cvfem_range packs,
        const ptrdiff_t *const SFEM_RESTRICT pack_order,
        const ptrdiff_t nelements,
        const scalar_t *const SFEM_RESTRICT pres,
        geom_t **const SFEM_RESTRICT points,
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
        const size_t scratch_n) {
    scalar_t *const SFEM_RESTRICT pack_u   = thread_scratch<scalar_t>(0, scratch_n);
    scalar_t *const SFEM_RESTRICT pack_out = thread_scratch<scalar_t>(1, scratch_n);
    // Coordinates always: this geometry derives its Jacobian from them.
    const Hex8PackCoordsT<scalar_t> pk =
            cvfem_hex8_pack_coords<scalar_t>(/*want_xyz=*/true, /*with_rc=*/0, max_actual_nodes_per_pack);

    for (ptrdiff_t i = packs.begin; i < packs.end; ++i) {
        const ptrdiff_t pack = pack_order[i];
        const Hex8PackExtentT<idx_t> x = cvfem_hex8_pack_extent<idx_t>(
                pack, nelements, n_elements_per_pack, owned_nodes_ptr, ghost_idx, ghost_ptr);

        std::memset(pack_out, 0, (size_t)x.n_pack_nodes * (size_t)CVFEM_HEX8_N_FIELDS * sizeof(scalar_t));
        fill_pack_fields(owned_nodes_ptr, ux, uy, uz, pres, pack, x.n_contiguous, x.n_ghost, x.ghosts, pack_u);
        fill_pack_xyz(owned_nodes_ptr, points, pack, x.n_contiguous, x.n_ghost, x.ghosts, pk.x, pk.y, pk.z);

        cvfem_hex8_residual_lanes_isoparam(x, pk, pack_elems, pack_u, pack_out, rho, mu);

        cvfem_hex8_flush_pack_to_global_soa(x, pack_out, rx, ry, rz, rc);
    }
}

#endif  // CVFEM_HEX8_BEST_PACKED_ISOPARAM_HPP
