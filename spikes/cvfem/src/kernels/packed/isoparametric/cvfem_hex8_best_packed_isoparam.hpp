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
        const Hex8PackCoords pk = cvfem_hex8_pack_coords(true, 0, max_actual_nodes_per_pack);


    for (ptrdiff_t pack = packs.begin; pack < packs.end; ++pack) {
            const Hex8PackExtent x = cvfem_hex8_pack_extent(
                    pack, nelements, n_elements_per_pack, owned_nodes_ptr, ghost_idx, ghost_ptr);

            std::memset(pack_out, 0, (size_t)x.n_pack_nodes * (size_t)CVFEM_HEX8_N_FIELDS * sizeof(scalar_t));

            fill_pack_fields(owned_nodes_ptr, ux, uy, uz, pres, pack, x.n_contiguous, x.n_ghost, x.ghosts, pack_u);

            fill_pack_xyz(owned_nodes_ptr, points, pack, x.n_contiguous, x.n_ghost, x.ghosts, pk.x, pk.y, pk.z);
            Hex8InputPack    in;
            Hex8CoordPack    xyz;
            Hex8ResidualPack outp;
            for (ptrdiff_t begin = x.e_start; begin < x.e_end; begin += CVFEM_HEX8_VEC_SIZE) {
                const int nlanes = int(MIN((ptrdiff_t)CVFEM_HEX8_VEC_SIZE, x.e_end - begin));
                gather_hex8_isoparam_simd_from_pack(
                        pack_elems, pack_u, pk.x, pk.y, pk.z, begin, nlanes, in, xyz);
                cvfem_hex8_ns_upwind_residual_isoparam_simd(rho, mu, xyz, in, outp);
                scatter_hex8_simd_to_pack(pack_elems, pack_out, begin, nlanes, outp);
            }

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
        const Hex8PackCoords pk =
                cvfem_hex8_pack_coords(/*want_xyz=*/true, /*with_rc=*/0, max_actual_nodes_per_pack);


    for (ptrdiff_t pack = packs.begin; pack < packs.end; ++pack) {
            const Hex8PackExtent x = cvfem_hex8_pack_extent(
                    pack, nelements, n_elements_per_pack, owned_nodes_ptr, ghost_idx, ghost_ptr);

            CVFEM_PHASE_CLOCK(_t);
            std::memset(pack_out, 0, (size_t)x.n_pack_nodes * (size_t)CVFEM_HEX8_N_FIELDS * sizeof(scalar_t));
            CVFEM_PHASE_MARK(acc, _t, PH_LOCAL_MEMSET);

            fill_pack_fields(owned_nodes_ptr, ux, uy, uz, pres, pack, x.n_contiguous, x.n_ghost, x.ghosts, pack_u);
            fill_pack_interleaved(owned_nodes_ptr, pack, x.n_contiguous, x.n_ghost, x.ghosts, dir, pack_dir);

            Hex8InputPack    u_pack;
            Hex8InputPack    du_pack;
            Hex8ResidualPack outp;
            Hex8CoordPack    xyz;
            Hex8RhieChowPack rcp;
            // The two gradient packs, staged exactly as apply_residual_packed_defcor stages its
            // one. The limiter and eps^2 live on the state pack because that is where the
            // correction's own kernel reads them; the direction pack carries only the field.
            Hex8UGradPack    hop, hovp;
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

#endif  // CVFEM_HEX8_BEST_PACKED_ISOPARAM_HPP
