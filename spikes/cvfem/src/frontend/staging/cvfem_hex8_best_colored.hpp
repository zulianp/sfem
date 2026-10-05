#ifndef CVFEM_HEX8_BEST_COLORED_HPP
#define CVFEM_HEX8_BEST_COLORED_HPP

// Colored layout: the pack decomposition is colored so that two packs sharing a
// color never touch a common node. Within a color the element kernels can write
// straight into the global residual / matrix with plain non-atomic updates, so
// there is no pack-local buffer, no local->global fold and no ghost reduction --
// only a barrier between colors.
//
// Like the packed layout it stages a pack's nodal fields into a compact buffer,
// so the element kernels still run 16-wide SIMD; what it drops is the *reduction*
// on the way out.
//
// When to use which (measured on Apple M1 Max, 8 threads, n=48, interleaved A/B):
//
//                      vs packed   vs atomic
//   residual              0.61x       1.28x
//   jacobian action       0.60x       1.07x
//   jacobian assemble     1.46x       2.06x
//
// The payoff scales with how much reduction work coloring removes, while its cost
// is a fixed number of barriers -- one per color, 12-16 for an SFC pack
// decomposition. Assembly reduces a 372 MiB matrix, so coloring wins by a lot.
// The residual only reduces a 3.6 MiB vector, so the barriers cost more than the
// ghost reduce they replace: single-threaded the colored residual is in fact 1.18x
// *faster* than packed, and the whole advantage is given back to barrier waits by
// 8 threads. Coloring is still the better choice than atomics for every operation,
// and it is the useful shape where a ghost-reduction pass is awkward to express.
//
// Colors are balanced by construction (see cvfem_pack_coloring.hpp); an unbalanced
// coloring costs the residual another ~20% in barrier waits.

#include "frontend/staging/cvfem_hex8_best_common.hpp"
#include "frontend/staging/cvfem_pack_coloring.hpp"
#include "kernels/packed/cvfem_hex8_pack_staging.hpp"
#include "kernels/packed/affine/cvfem_hex8_best_packed_affine.hpp"
#include "kernels/packed/isoparametric/cvfem_hex8_best_packed_isoparam.hpp"
#include "kernels/cvfem_range.hpp"

// ---------------------------------------------------------------------------
// Global-index gather / scatter
// ---------------------------------------------------------------------------


static SFEM_INLINE void flush_pack_to_global_interleaved(const PackedData                       &p,
                                                         const ptrdiff_t                         pack,
                                                         const ptrdiff_t                         n_contiguous,
                                                         const ptrdiff_t                         n_ghost,
                                                         const smesh::idx_t *const SFEM_RESTRICT ghosts,
                                                         const scalar_t *const SFEM_RESTRICT     pack_out,
                                                         scalar_t *const SFEM_RESTRICT           jv) {
    const ptrdiff_t owned = p.owned_nodes_ptr[pack];
    for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
        const scalar_t *const SFEM_RESTRICT src = pack_out + k * N_FIELDS;
        scalar_t *const SFEM_RESTRICT       dst = jv + (owned + k) * N_FIELDS;
        for (int f = 0; f < N_FIELDS; ++f) dst[f] += src[f];
    }
    for (ptrdiff_t k = 0; k < n_ghost; ++k) {
        const scalar_t *const SFEM_RESTRICT src = pack_out + (n_contiguous + k) * N_FIELDS;
        scalar_t *const SFEM_RESTRICT       dst = jv + (ptrdiff_t)ghosts[k] * N_FIELDS;
        for (int f = 0; f < N_FIELDS; ++f) dst[f] += src[f];
    }
}

// ---------------------------------------------------------------------------
// Residual
// ---------------------------------------------------------------------------

// THE PACK-COLOURED RESIDUAL. A launcher now: it pulls the arrays out, owns the one parallel
// region, hands each thread a slice of one colour, and barriers between colours -- the same
// shape frontend/staging/cvfem_hex8_ecolored_launch.hpp uses for the element colouring, and for
// the same reason: the barrier that used to be the implicit one at the end of `#pragma omp for`.
//
// The sweeps are in kernels/packed/affine/ and kernels/packed/isoparametric/, one per geometry,
// because DESIGN.md's correction asks for the geometries to be separate kernels and this is a
// packed layout -- it keeps the packed staging and colours the PACKS. colored/ is the ELEMENT
// colouring, which is a different method.
//
// The global arrays are zeroed here before the first colour, because the sweeps' drain
// accumulates: a node a pack owns also receives contributions from packs that ghost it, and
// those may run in an earlier colour.
static SFEM_NOINLINE void apply_residual_colored(MeshData           &d,
                                                 PackedData         &p,
                                                 const PackColoring &c,
                                                 const scalar_t      rho,
                                                 const scalar_t      mu,
                                                 const GeomKind      geom) {
    reset_residual(d.nnodes, d.rx.data(), d.ry.data(), d.rz.data(), d.rc.data());

    scalar_t *const SFEM_RESTRICT rx        = d.rx.data();
    scalar_t *const SFEM_RESTRICT ry        = d.ry.data();
    scalar_t *const SFEM_RESTRICT rz        = d.rz.data();
    scalar_t *const SFEM_RESTRICT rc        = d.rc.data();
    const size_t                  scratch_n = packed_scratch_n(p.max_actual_nodes_per_pack);
    const int                     with_rc   = !d.pgx.empty() && d.rhie_chow_scale != scalar_t(0);

#pragma omp parallel
    {
        const int n_parts = cvfem_n_threads();
        const int part    = cvfem_thread_index();
        for (int color = 0; color < c.n_colors; ++color) {
            const cvfem_range r = cvfem_range_split(c.color_ptr[(size_t)color],
                                                    c.color_ptr[(size_t)color + 1], 1,
                                                    part, n_parts);
            if (geom == GeomKind::Isoparam)
                apply_residual_packcolored_isoparam_range(
                        r, c.pack_order.data(), d.nelements, d.p.data(), d.points, d.ux.data(),
                        d.uy.data(), d.uz.data(), p.elems, p.ghost_idx, p.ghost_ptr,
                        p.max_actual_nodes_per_pack, p.n_elements_per_pack, p.owned_nodes_ptr,
                        rho, mu, rx, ry, rz, rc, scratch_n);
            else
                apply_residual_packcolored_affine_range(
                        r, c.pack_order.data(), d.adj_ptr, d.det_ptr, d.nelements, d.p.data(),
                        d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.rhie_chow_scale,
                        d.ux.data(), d.uy.data(), d.uz.data(), p.elems, p.ghost_idx, p.ghost_ptr,
                        p.max_actual_nodes_per_pack, p.n_elements_per_pack, p.owned_nodes_ptr,
                        rho, mu, rx, ry, rz, rc, scratch_n, with_rc);
            // No two packs of a colour share a node, so the slices above need no
            // synchronisation between them. The next colour does.
            cvfem_thread_barrier();
        }
    }
}

// ---------------------------------------------------------------------------
// Jacobian action
// ---------------------------------------------------------------------------

// The same colored sweep applied to the matrix-free Jacobian action, so --layout
// colored covers all three operations instead of silently falling back.
static SFEM_NOINLINE void apply_jacobian_action_colored(MeshData                           &d,
                                                        PackedData                         &p,
                                                        const PackColoring                 &c,
                                                        const scalar_t                      rho,
                                                        const scalar_t                      mu,
                                                        const scalar_t *const SFEM_RESTRICT dir,
                                                        scalar_t *const SFEM_RESTRICT       jv,
                                                        const GeomKind                      geom_kind) {
    cvfem_zero_scalars(jv, d.nnodes * N_FIELDS);

    // The hoisted Rhie-Chow coefficient, twelve per element. Its own cache key means this
    // is a no-op unless rho, mu, the scale or the mesh moved -- see Hex8RhieChowPack::coeff.
    cvfem_hex8_build_rc_coeff(d, rho, mu);

    const size_t scratch_n = packed_scratch_n(p.max_actual_nodes_per_pack);
    const int    with_rc   = !d.pgx.empty() && d.rhie_chow_scale != scalar_t(0);
    // The exact form differentiates through the nodal gradient reconstruction, so it needs
    // that reconstruction applied to the DIRECTION's pressure too. Present only when the
    // caller filled qgx/qgy/qgz; otherwise the kernel takes the frozen-gradient form.
    const bool   with_qg   = with_rc && !d.qgx.empty();

#pragma omp parallel
    {
        PhaseAcc                          acc;
        scalar_t *const SFEM_RESTRICT pack_u   = thread_scratch<scalar_t>(0, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_dir = thread_scratch<scalar_t>(1, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_out = thread_scratch<scalar_t>(2, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_xyz =
                (geom_kind == GeomKind::Isoparam || with_rc)
                        ? thread_scratch<scalar_t>(3, with_rc ? packed_rc_n(p.max_actual_nodes_per_pack) : packed_xyz_n(p.max_actual_nodes_per_pack))
                        : nullptr;
        const ptrdiff_t               xyz_n  = p.max_actual_nodes_per_pack > 0 ? p.max_actual_nodes_per_pack : 1;
        scalar_t *const SFEM_RESTRICT pack_x = pack_xyz;
        scalar_t *const SFEM_RESTRICT pack_y = pack_xyz ? pack_xyz + xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_z = pack_xyz ? pack_xyz + 2 * xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_pgx = with_rc ? pack_xyz + 3 * xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_pgy = with_rc ? pack_xyz + 4 * xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_pgz = with_rc ? pack_xyz + 5 * xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_qg  = with_qg ? thread_scratch<scalar_t>(4, packed_qg_n(p.max_actual_nodes_per_pack)) : nullptr;
        scalar_t *const SFEM_RESTRICT pack_qgx = pack_qg;
        scalar_t *const SFEM_RESTRICT pack_qgy = with_qg ? pack_qg + xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_qgz = with_qg ? pack_qg + 2 * xyz_n : nullptr;

        for (int color = 0; color < c.n_colors; ++color) {
            const ptrdiff_t cbegin = c.color_ptr[(size_t)color];
            const ptrdiff_t cend   = c.color_ptr[(size_t)color + 1];
#pragma omp for schedule(dynamic, 1)
            for (ptrdiff_t i = cbegin; i < cend; ++i) {
                const ptrdiff_t                         pack         = c.pack_order[(size_t)i];
                const ptrdiff_t                         e_start      = pack * p.n_elements_per_pack;
                const ptrdiff_t                         e_end        = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
                const ptrdiff_t                         owned        = p.owned_nodes_ptr[pack];
                const ptrdiff_t                         n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
                const ptrdiff_t                         n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
                const smesh::idx_t *const SFEM_RESTRICT ghosts       = &p.ghost_idx[p.ghost_ptr[pack]];

                double _t = phase_now();
                std::memset(pack_out, 0, (size_t)(n_contiguous + n_ghost) * (size_t)N_FIELDS * sizeof(scalar_t));
                fill_pack_fields(p.owned_nodes_ptr, d.ux.data(), d.uy.data(), d.uz.data(), d.p.data(), pack, n_contiguous, n_ghost, ghosts, pack_u);
                fill_pack_interleaved(p.owned_nodes_ptr, pack, n_contiguous, n_ghost, ghosts, dir, pack_dir);
                if (with_rc)
                    cvfem_hex8_fill_pack_xyz_pgrad(p.owned_nodes_ptr, d.points, d.pgx.data(), d.pgy.data(), d.pgz.data(), !d.pgx.empty() && d.rhie_chow_scale != scalar_t(0), pack, n_contiguous, n_ghost, ghosts, pack_x, pack_y,
                                                   pack_z, pack_pgx, pack_pgy, pack_pgz);
                if (with_qg)
                    cvfem_hex8_fill_pack_qgrad(p.owned_nodes_ptr, d.qgx.data(), d.qgy.data(), d.qgz.data(), pack, n_contiguous, n_ghost, ghosts, pack_qgx, pack_qgy,
                                               pack_qgz);
                if (geom_kind == GeomKind::Isoparam)
                    fill_pack_xyz(p.owned_nodes_ptr, d.points, pack, n_contiguous, n_ghost, ghosts, pack_x, pack_y, pack_z);
                if (g_breakdown) { const double _n = wall_time(); acc.t[PH_GATHER] += _n - _t; _t = _n; }

                Hex8InputPack    u_pack, du_pack;
                Hex8ResidualPack outp;
                Hex8CoordPack    xyz;
                Hex8RhieChowPack rcp;
                for (ptrdiff_t begin = e_start; begin < e_end; begin += CVFEM_HEX8_VEC_SIZE) {
                    const int nlanes = int(MIN((ptrdiff_t)CVFEM_HEX8_VEC_SIZE, e_end - begin));
                    if (geom_kind == GeomKind::Isoparam) {
                        gather_hex8_isoparam_action_simd_from_pack(p.elems,
                                                                   pack_u,
                                                                   pack_dir,
                                                                   pack_x,
                                                                   pack_y,
                                                                   pack_z,
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
                        gather_hex8_action_simd_from_pack(p.elems,
                                                          pack_u,
                                                          pack_dir,
                                                          d.adj_ptr, d.det_ptr,
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
                            cvfem_hex8_gather_rc_from_pack(p.elems, pack_pgx, pack_pgy,
                                                           pack_pgz, begin, nlanes, rcp);
                            cvfem_hex8_gather_rc_coeff(d.rc_coeff.data(), d.rc_w.data(), cvfem_hex8_rc_config_for(d), begin, nlanes, rcp);
                        }
                        if (with_qg)
                            cvfem_hex8_gather_qg_from_pack(p.elems, pack_qgx, pack_qgy, pack_qgz, begin, nlanes, rcp);
                        cvfem_hex8_ns_upwind_jacobian_action_simd(
                                rho, mu, cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, det, u_pack, du_pack,
                                outp, with_rc ? &rcp : nullptr, d.rhie_chow_scale, with_qg);
                    }
                    scatter_hex8_simd_to_pack(p.elems, pack_out, begin, nlanes, outp);
                }
                if (g_breakdown) { const double _n = wall_time(); acc.t[PH_KERNEL] += _n - _t; _t = _n; }

                flush_pack_to_global_interleaved(p, pack, n_contiguous, n_ghost, ghosts, pack_out, jv);
                if (g_breakdown) acc.t[PH_LOCAL_TO_GLOBAL] += wall_time() - _t;
            }
        }
        acc.flush();
    }
}

// ---------------------------------------------------------------------------
// Jacobian assembly
// ---------------------------------------------------------------------------

// Colored assembly: elements are visited pack by pack, one color at a time, and
// the element kernel accumulates straight into the global BSR values. No local
// pack matrix, no local->global copy, no ghost reduction, no atomics.
static SFEM_NOINLINE void assemble_jacobian_colored(MeshData        &d,
                                                    PackedData      &p,
                                                    const PackColoring &c,
                                                    BSR4            &b,
                                                    const scalar_t   rho,
                                                    const scalar_t   mu,
                                                    const GeomKind   geom_kind) {
    zero_bsr4(b);

    scalar_t *const SFEM_RESTRICT       values = b.values->data();
    const int *const SFEM_RESTRICT      gslots = reinterpret_cast<const int *>(b.element_slots.data());
    // This assembly reads the mesh directly rather than a staged pack, so Rhie-Chow enters
    // exactly as it does on the atomic layout -- through Hex8ExtraScratch. Only the two
    // hand-written kernels take the term.
    const Hex8Extras                    opt = cvfem_hex8_extras_of(d);

#pragma omp parallel
    {
        PhaseAcc acc;
        for (int color = 0; color < c.n_colors; ++color) {
            const ptrdiff_t cbegin = c.color_ptr[(size_t)color];
            const ptrdiff_t cend   = c.color_ptr[(size_t)color + 1];
#pragma omp for schedule(dynamic, 1)
            for (ptrdiff_t i = cbegin; i < cend; ++i) {
                const ptrdiff_t pack    = c.pack_order[(size_t)i];
                const ptrdiff_t e_start = pack * p.n_elements_per_pack;
                const ptrdiff_t e_end   = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
                const double    _t      = phase_now();

                for (ptrdiff_t e = e_start; e < e_end; ++e) {
                    scalar_t ux_e[8], uy_e[8], uz_e[8], p_e[8];
                    gather_element_fields(d.elems, d.ux.data(), d.uy.data(), d.uz.data(), d.p.data(), e, ux_e, uy_e, uz_e, p_e);
                    Hex8ExtraScratch ex;
                    ex.load(d.elems, d.points, d.face_mask.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.qgx.data(), d.qgy.data(), d.qgz.data(), d.ux.data(), d.uy.data(), d.uz.data(), d.adj_ptr, d.det_ptr, opt, e);
                    const scalar_t *const          rc_p  = opt.with_rc ? p_e : nullptr;
                    const int *const SFEM_RESTRICT slots = gslots + (size_t)e * 64;

                    if (geom_kind == GeomKind::Isoparam) {
                        scalar_t x[8], y[8], z[8];
                        gather_element_coords(d.elems, d.points, e, x, y, z);
                        cvfem_hex8_ns_upwind_jacobian_add_slots_isoparam<false>(
                                rho, mu, x, y, z, ux_e, uy_e, uz_e, slots, values, ex.rc, rc_p);
                    } else {
                        scalar_t adj[9], det;
                        load_hex8_adj(d.adj_ptr, d.det_ptr, e, adj, &det);
                        if (g_dense_flush) {
                            alignas(ALIGN_BYTES) scalar_t ke[64 * 16] = {};
                            cvfem_hex8_ns_upwind_jacobian_add_slots<false>(
                                    rho, mu, adj, det, ux_e, uy_e, uz_e, g_identity_slots, ke, ex.rc, rc_p);
                            hex8_blocks_to_slots(slots, ke, values);
                        } else {
                            cvfem_hex8_ns_upwind_jacobian_add_slots<false>(
                                    rho, mu, adj, det, ux_e, uy_e, uz_e, slots, values, ex.rc, rc_p);
                        }
                    }
                }
                if (g_breakdown) acc.t[PH_KERNEL] += wall_time() - _t;
            }
        }
        acc.flush();
    }
}

#endif  // CVFEM_HEX8_BEST_COLORED_HPP
