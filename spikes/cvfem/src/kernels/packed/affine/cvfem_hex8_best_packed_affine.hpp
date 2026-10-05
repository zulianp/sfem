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

// ISO IS A TEMPLATE PARAMETER, NOT A RUNTIME ENUM. DESIGN.md asks for the affine and the
// isoparametric kernels to be "logically separated (now they are mixed in with enum and
// booleans)", and this was the enum: GeomKind arrived as an argument and was tested per pack,
// inside the sweep. It also left the geometry undecided at the point where it matters -- the
// lane loop -- which is the shape of guard this file's own notes record costing 1.83x, and which
// the vectorisation gate now refuses outright. The caller picks the instantiation.
// The pack sweep, driven by a range. The `#pragma omp parallel` is in the launcher below; see
// kernels/cvfem_range.hpp for why DESIGN.md wants it there. The packs of a range touch only
// nodes this part owns -- that is what the packed layout is for -- so the parts need no
// synchronisation between them, and the ghost rows they do share are reduced afterwards in the
// launcher, which is the second and independent parallel loop.
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
        const Hex8PackCoords pk =
                cvfem_hex8_pack_coords(with_rc != 0, with_rc, max_actual_nodes_per_pack);


    for (ptrdiff_t pack = packs.begin; pack < packs.end; ++pack) {
            const Hex8PackExtent x = cvfem_hex8_pack_extent(
                    pack, nelements, n_elements_per_pack, owned_nodes_ptr, ghost_idx, ghost_ptr);

            std::memset(pack_out, 0, (size_t)x.n_pack_nodes * (size_t)CVFEM_HEX8_N_FIELDS * sizeof(scalar_t));

            fill_pack_fields(owned_nodes_ptr, ux, uy, uz, pres, pack, x.n_contiguous, x.n_ghost, x.ghosts, pack_u);
            if (with_rc)
                cvfem_hex8_fill_pack_xyz_pgrad(owned_nodes_ptr, points, pgx, pgy, pgz, with_rc, pack, x.n_contiguous, x.n_ghost, x.ghosts, pk.x, pk.y,
                                               pk.z, pk.pgx, pk.pgy, pk.pgz);

            alignas(ALIGN_BYTES) scalar_t cof0[CVFEM_HEX8_VEC_SIZE], cof1[CVFEM_HEX8_VEC_SIZE], cof2[CVFEM_HEX8_VEC_SIZE];
            alignas(ALIGN_BYTES) scalar_t cof3[CVFEM_HEX8_VEC_SIZE], cof4[CVFEM_HEX8_VEC_SIZE], cof5[CVFEM_HEX8_VEC_SIZE];
            alignas(ALIGN_BYTES) scalar_t cof6[CVFEM_HEX8_VEC_SIZE], cof7[CVFEM_HEX8_VEC_SIZE], cof8[CVFEM_HEX8_VEC_SIZE];
            alignas(ALIGN_BYTES) scalar_t det[CVFEM_HEX8_VEC_SIZE];
            Hex8InputPack    in;
            Hex8ResidualPack outp;
            Hex8RhieChowPack rcp;
            for (ptrdiff_t begin = x.e_start; begin < x.e_end; begin += CVFEM_HEX8_VEC_SIZE) {
                const int nlanes = int(MIN((ptrdiff_t)CVFEM_HEX8_VEC_SIZE, x.e_end - begin));
                gather_hex8_simd_from_pack(pack_elems,
                                           pack_u,
                                           adj_ptr, det_ptr,
                                           begin,
                                           nlanes,
                                           in,
                                           cof0,
                                           cof1,
                                           cof2,
                                           cof3,
                                           cof4,
                                           cof5,
                                           cof6,
                                           cof7,
                                           cof8,
                                           det);
                if (with_rc) {
                    cvfem_hex8_gather_rc_from_pack(pack_elems, pk.pgx, pk.pgy,
                                                   pk.pgz, begin, nlanes, rcp);
                }
                cvfem_hex8_ns_upwind_residual_sumfact_simd(
                        rho, mu, cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, det, in, outp,
                        with_rc ? &rcp : nullptr, rhie_chow_scale);
                scatter_hex8_simd_to_pack(pack_elems, pack_out, begin, nlanes, outp);
            }

            cvfem_hex8_drain_pack_soa(x, pack_out, n_ghost_entries, ghost_buf, rx, ry, rz, rc);
    }
}

#endif  // CVFEM_HEX8_BEST_PACKED_AFFINE_HPP
