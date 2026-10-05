#ifndef CVFEM_HEX8_BEST_PACKED_HPP
#define CVFEM_HEX8_BEST_PACKED_HPP

// Packed layout: elements are grouped into packs, each pack accumulates into a
// thread-private buffer indexed by pack-local node ids, and the buffer is folded
// back into the global structure afterwards. Ghost rows -- nodes a pack touches
// but does not own -- are reduced in a second pass.
//
// The pack-local indexing lets the residual and Jacobian-action run 16-wide SIMD
// over elements. For assembly the local matrix is large enough that the round
// trip through it costs more than it saves; see cvfem_hex8_best_colored.hpp
// and cvfem_hex8_best_store.hpp.

#include "kernels/microkernels/hex8/affine/cvfem_hex8_ns_upwind_affine.hpp"
#include "kernels/microkernels/hex8/isoparametric/cvfem_hex8_ns_upwind_isoparam.hpp"
#include "kernels/cvfem_scatter.hpp"
#include "kernels/cvfem_phases.hpp"
#include "kernels/cvfem_range.hpp"





// The pack sweep, driven by a range. The `#pragma omp parallel` is in the launcher below; see
// kernels/cvfem_range.hpp for why DESIGN.md wants it there. The packs of a range touch only
// nodes this part owns -- that is what the packed layout is for -- so the parts need no
// synchronisation between them, and the ghost rows they do share are reduced afterwards in the
// launcher, which is the second and independent parallel loop.
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
        const Hex8PackCoords pk =
                cvfem_hex8_pack_coords(true, with_rc, max_actual_nodes_per_pack);

    for (ptrdiff_t pack = packs.begin; pack < packs.end; ++pack) {
            const Hex8PackExtent x = cvfem_hex8_pack_extent(
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
            Hex8InputPack    in;
            Hex8ResidualPack outp;
            Hex8RhieChowPack rcp;
            Hex8UGradPack    hop;
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
            for (ptrdiff_t k = 0; k < x.n_contiguous; ++k) {
                const scalar_t *const SFEM_RESTRICT out = pack_out + k * CVFEM_HEX8_N_FIELDS;
                const ptrdiff_t                     g   = x.owned + k;
                rx[g] = out[0]; ry[g] = out[1]; rz[g] = out[2]; rc[g] = out[3];
            }
            scalar_t *const SFEM_RESTRICT gx = ghost_buf + 0 * n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gy = ghost_buf + 1 * n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gz = ghost_buf + 2 * n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gc = ghost_buf + 3 * n_ghost_entries;
            for (ptrdiff_t k = 0; k < x.n_ghost; ++k) {
                const scalar_t *const SFEM_RESTRICT out = pack_out + (x.n_contiguous + k) * CVFEM_HEX8_N_FIELDS;
                gx[x.ghost_off + k] = out[0]; gy[x.ghost_off + k] = out[1];
                gz[x.ghost_off + k] = out[2]; gc[x.ghost_off + k] = out[3];
            }
    }
}



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
template <bool ISO>
static SFEM_NOINLINE void apply_residual_packed_range(
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
                cvfem_hex8_pack_coords(ISO || with_rc, with_rc, max_actual_nodes_per_pack);


    for (ptrdiff_t pack = packs.begin; pack < packs.end; ++pack) {
            const Hex8PackExtent x = cvfem_hex8_pack_extent(
                    pack, nelements, n_elements_per_pack, owned_nodes_ptr, ghost_idx, ghost_ptr);

            std::memset(pack_out, 0, (size_t)x.n_pack_nodes * (size_t)CVFEM_HEX8_N_FIELDS * sizeof(scalar_t));

            fill_pack_fields(owned_nodes_ptr, ux, uy, uz, pres, pack, x.n_contiguous, x.n_ghost, x.ghosts, pack_u);
            if (with_rc)
                cvfem_hex8_fill_pack_xyz_pgrad(owned_nodes_ptr, points, pgx, pgy, pgz, with_rc, pack, x.n_contiguous, x.n_ghost, x.ghosts, pk.x, pk.y,
                                               pk.z, pk.pgx, pk.pgy, pk.pgz);

            if constexpr (ISO) {
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
            } else {
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
            }

            for (ptrdiff_t k = 0; k < x.n_contiguous; ++k) {
                const scalar_t *const SFEM_RESTRICT out = pack_out + k * CVFEM_HEX8_N_FIELDS;
                const ptrdiff_t                     g   = x.owned + k;
                rx[g]                                   = out[0];
                ry[g]                                   = out[1];
                rz[g]                                   = out[2];
                rc[g]                                   = out[3];
            }

            scalar_t *const SFEM_RESTRICT gx = ghost_buf + 0 * n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gy = ghost_buf + 1 * n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gz = ghost_buf + 2 * n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gc = ghost_buf + 3 * n_ghost_entries;
            for (ptrdiff_t k = 0; k < x.n_ghost; ++k) {
                const scalar_t *const SFEM_RESTRICT out = pack_out + (x.n_contiguous + k) * CVFEM_HEX8_N_FIELDS;
                gx[x.ghost_off + k]                       = out[0];
                gy[x.ghost_off + k]                       = out[1];
                gz[x.ghost_off + k]                       = out[2];
                gc[x.ghost_off + k]                       = out[3];
            }
    }
}




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
template <bool ISO>
static SFEM_NOINLINE void assemble_jacobian_packed_range(
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
        const Hex8RcConfig &rc_cfg) {
        CVFEM_PHASE_ACC(acc);
        alignas(ALIGN_BYTES) scalar_t dense_ke[64 * 16];
        std::memset(dense_ke, 0, sizeof(dense_ke));
        scalar_t *const SFEM_RESTRICT pack_u          = thread_scratch<scalar_t>(0, u_n);
        scalar_t *const SFEM_RESTRICT local_vals_pack = thread_scratch<scalar_t>(2, bsr_n);
        const Hex8PackCoords pk =
                cvfem_hex8_pack_coords(ISO || with_rc, with_rc, max_actual_nodes_per_pack);


    for (ptrdiff_t pack = packs.begin; pack < packs.end; ++pack) {
            const Hex8PackExtent x = cvfem_hex8_pack_extent(
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
            if constexpr (ISO)
                fill_pack_xyz(owned_nodes_ptr, points, pack, x.n_contiguous, x.n_ghost, x.ghosts, pk.x, pk.y, pk.z);
            CVFEM_PHASE_MARK(acc, _t, PH_GATHER);

            for (ptrdiff_t e = x.e_start; e < x.e_end; ++e) {
                scalar_t ux_e[8], uy_e[8], uz_e[8], p_e[8];
                for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                    const scalar_t *const SFEM_RESTRICT u = pack_u + (ptrdiff_t)pack_elems[a][e] * CVFEM_HEX8_N_FIELDS;
                    ux_e[a]                              = u[0];
                    uy_e[a]                              = u[1];
                    uz_e[a]                              = u[2];
                    p_e[a]                               = u[3];
                }

                const int *const SFEM_RESTRICT slots =
                        g_kernel_only ? g_identity_slots : local_element_slot + (size_t)e * 64;
                scalar_t *const SFEM_RESTRICT local_vals = g_kernel_only ? dense_ke : local_vals_pack;
                scalar_t adj[9], det;
                if constexpr (!ISO) load_hex8_adj(adj_ptr, det_ptr, e, adj, &det);
                // The coordinates and the nodal gradient come out of the pack; the
                // Hex8RhieChow points at these locals, so they must outlive the call, which
                // they do.
                scalar_t     rc_x[8], rc_y[8], rc_z[8], rc_pgx[8], rc_pgy[8], rc_pgz[8];
                const Hex8RcConfig rcfg = rc_cfg;
                Hex8RhieChow rc{};
                if (with_rc) {
                    gather_hex8_coords_from_pack(pack_elems, pk.x, pk.y, pk.z, e, rc_x, rc_y, rc_z);
                    gather_hex8_coords_from_pack(pack_elems, pk.pgx, pk.pgy, pk.pgz, e, rc_pgx, rc_pgy, rc_pgz);
                    rc = Hex8RhieChow{rc_x,    rc_y,  rc_z,  rc_pgx, rc_pgy, rc_pgz, rcfg.scale,
                                      nullptr, nullptr, nullptr, ux_e, uy_e, uz_e, rcfg.tau};
                }
                const scalar_t *const rc_p = with_rc ? p_e : nullptr;
                if constexpr (ISO) {
                    scalar_t x[8], y[8], z[8];
                    gather_hex8_coords_from_pack(pack_elems, pk.x, pk.y, pk.z, e, x, y, z);
                    cvfem_hex8_ns_upwind_jacobian_add_slots_isoparam<false>(
                            rho, mu, x, y, z, ux_e, uy_e, uz_e, slots, local_vals, rc, rc_p);
                } else {
                    cvfem_hex8_ns_upwind_jacobian_add_slots<false>(
                            rho, mu, adj, det, ux_e, uy_e, uz_e, slots, local_vals, rc, rc_p);
                }
            }

            CVFEM_PHASE_MARK(acc, _t, PH_KERNEL);
            const int                     owned_nnz = x.n_contiguous > 0 ? lrowptr[(size_t)x.n_contiguous] : 0;
            if (!g_kernel_only)
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
template <bool ISO>
static SFEM_NOINLINE void apply_jacobian_action_packed_range(
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
        const Hex8RcConfig &rc_cfg,
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
        const Hex8PackCoords pk =
                cvfem_hex8_pack_coords(ISO || with_rc, with_rc, max_actual_nodes_per_pack);
        const Hex8PackQGrad qg = cvfem_hex8_pack_qgrad(with_qg, max_actual_nodes_per_pack);


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
            if (with_rc)
                cvfem_hex8_fill_pack_xyz_pgrad(owned_nodes_ptr, points, pgx, pgy, pgz, with_rc, pack, x.n_contiguous, x.n_ghost, x.ghosts, pk.x, pk.y, pk.z,
                                               pk.pgx, pk.pgy, pk.pgz);
            if (with_qg)
                cvfem_hex8_fill_pack_qgrad(owned_nodes_ptr, qgx, qgy, qgz, pack, x.n_contiguous, x.n_ghost, x.ghosts, qg.x, qg.y, qg.z);
            if constexpr (ISO)
                fill_pack_xyz(owned_nodes_ptr, points, pack, x.n_contiguous, x.n_ghost, x.ghosts, pk.x, pk.y, pk.z);
            CVFEM_PHASE_MARK(acc, _t, PH_GATHER);

            for (ptrdiff_t begin = x.e_start; begin < x.e_end; begin += CVFEM_HEX8_VEC_SIZE) {
                const int nlanes = int(MIN((ptrdiff_t)CVFEM_HEX8_VEC_SIZE, x.e_end - begin));
                if constexpr (ISO) {
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
                } else {
                    alignas(ALIGN_BYTES) scalar_t cof0[CVFEM_HEX8_VEC_SIZE], cof1[CVFEM_HEX8_VEC_SIZE],
                            cof2[CVFEM_HEX8_VEC_SIZE];
                    alignas(ALIGN_BYTES) scalar_t cof3[CVFEM_HEX8_VEC_SIZE], cof4[CVFEM_HEX8_VEC_SIZE],
                            cof5[CVFEM_HEX8_VEC_SIZE];
                    alignas(ALIGN_BYTES) scalar_t cof6[CVFEM_HEX8_VEC_SIZE], cof7[CVFEM_HEX8_VEC_SIZE],
                            cof8[CVFEM_HEX8_VEC_SIZE];
                    alignas(ALIGN_BYTES) scalar_t det[CVFEM_HEX8_VEC_SIZE];
                    gather_hex8_action_simd_from_pack(pack_elems,
                                                      pack_u,
                                                      pack_dir,
                                                      adj_ptr,
                                                      det_ptr,
                                                      begin,
                                                      nlanes,
                                                      u_pack,
                                                      du_pack,
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
                        cvfem_hex8_gather_rc_from_pack(pack_elems, pk.pgx, pk.pgy, pk.pgz,
                                                       begin, nlanes, rcp);
                        cvfem_hex8_gather_rc_coeff(rc_coeff, rc_w, rc_cfg, begin, nlanes, rcp);
                    }
                    if (with_qg)
                        cvfem_hex8_gather_qg_from_pack(pack_elems, qg.x, qg.y, qg.z, begin, nlanes, rcp);
                    if (with_ho) {
                        for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
                            const ptrdiff_t e = begin + lane;
                            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                                if (lane >= nlanes) {
                                    hop.x[a][lane] = hop.y[a][lane] = hop.z[a][lane] = scalar_t(0);
                                    for (int c = 0; c < 9; ++c) {
                                        hop.g[a][c][lane]  = scalar_t(0);
                                        hovp.g[a][c][lane] = scalar_t(0);
                                    }
                                    continue;
                                }
                                const idx_t gn = mesh_elems[a][e];
                                hop.x[a][lane] = scalar_t(points[0][gn]);
                                hop.y[a][lane] = scalar_t(points[1][gn]);
                                hop.z[a][lane] = scalar_t(points[2][gn]);
                                for (int c = 0; c < 9; ++c) {
                                    hop.g[a][c][lane]  = ugrad[(ptrdiff_t)gn * 9 + c];
                                    hovp.g[a][c][lane] = vgrad[(ptrdiff_t)gn * 9 + c];
                                }
                            }
                        }
                    }
                    cvfem_hex8_ns_upwind_jacobian_action_simd(rho,
                                                              mu,
                                                              cof0,
                                                              cof1,
                                                              cof2,
                                                              cof3,
                                                              cof4,
                                                              cof5,
                                                              cof6,
                                                              cof7,
                                                              cof8,
                                                              det,
                                                              u_pack,
                                                              du_pack,
                                                              outp,
                                                              with_rc ? &rcp : nullptr,
                                                              rhie_chow_scale,
                                                              with_qg,
                                                              scalar_t(0),
                                                              with_ho ? &hop : nullptr,
                                                              with_ho ? &hovp : nullptr);
                }
                scatter_hex8_simd_to_pack(pack_elems, pack_out, begin, nlanes, outp);
            }
            CVFEM_PHASE_MARK(acc, _t, PH_KERNEL);

            std::memcpy(jv + x.owned * CVFEM_HEX8_N_FIELDS, pack_out, (size_t)x.n_contiguous * (size_t)CVFEM_HEX8_N_FIELDS * sizeof(scalar_t));

            scalar_t *const SFEM_RESTRICT gx = ghost_buf + 0 * n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gy = ghost_buf + 1 * n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gz = ghost_buf + 2 * n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gc = ghost_buf + 3 * n_ghost_entries;
            for (ptrdiff_t k = 0; k < x.n_ghost; ++k) {
                const scalar_t *const SFEM_RESTRICT out = pack_out + (x.n_contiguous + k) * CVFEM_HEX8_N_FIELDS;
                gx[x.ghost_off + k]                       = out[0];
                gy[x.ghost_off + k]                       = out[1];
                gz[x.ghost_off + k]                       = out[2];
                gc[x.ghost_off + k]                       = out[3];
            }
            CVFEM_PHASE_MARK_LAST(acc, _t, PH_LOCAL_TO_GLOBAL);
    }
        CVFEM_PHASE_FLUSH(acc);
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
        const Hex8RcConfig &rc_cfg) {
        CVFEM_PHASE_ACC(acc);
        scalar_t *const SFEM_RESTRICT pack_dir = thread_scratch<scalar_t>(1, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_out = thread_scratch<scalar_t>(2, scratch_n);
        // Three arrays in slot 3, not six: the nodal pressure gradient is inside the store.
        const Hex8PackCoords pk =
                cvfem_hex8_pack_coords(with_qg, /*with_rc=*/0, max_actual_nodes_per_pack);
        const Hex8PackQGrad qg = cvfem_hex8_pack_qgrad(with_qg, max_actual_nodes_per_pack);


    for (ptrdiff_t pack = packs.begin; pack < packs.end; ++pack) {
            const Hex8PackExtent x = cvfem_hex8_pack_extent(
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

            Hex8InputPack    du_pack;
            Hex8ResidualPack outp;
            Hex8RhieChowPack rcp;
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

            std::memcpy(jv + x.owned * CVFEM_HEX8_N_FIELDS, pack_out, (size_t)x.n_contiguous * (size_t)CVFEM_HEX8_N_FIELDS * sizeof(scalar_t));
            scalar_t *const SFEM_RESTRICT gx = ghost_buf + 0 * n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gy = ghost_buf + 1 * n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gz = ghost_buf + 2 * n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gc = ghost_buf + 3 * n_ghost_entries;
            for (ptrdiff_t k = 0; k < x.n_ghost; ++k) {
                const scalar_t *const SFEM_RESTRICT out = pack_out + (x.n_contiguous + k) * CVFEM_HEX8_N_FIELDS;
                gx[x.ghost_off + k]                       = out[0];
                gy[x.ghost_off + k]                       = out[1];
                gz[x.ghost_off + k]                       = out[2];
                gc[x.ghost_off + k]                       = out[3];
            }
            CVFEM_PHASE_MARK_LAST(acc, _t, PH_LOCAL_TO_GLOBAL);
    }
        CVFEM_PHASE_FLUSH(acc);
}



#endif  // CVFEM_HEX8_BEST_PACKED_HPP
