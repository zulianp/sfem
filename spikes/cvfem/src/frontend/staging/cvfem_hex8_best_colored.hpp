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

// THE PACK-COLOURED JACOBIAN ACTION. A launcher, like the residual above: one parallel region,
// a slice of one colour per thread, a barrier between colours. The sweeps are in
// kernels/packed/affine/ and kernels/packed/isoparametric/.
//
// jv is zeroed here because the sweeps' drain accumulates, and the hoisted Rhie-Chow coefficient
// is built here because it is a per-solve quantity with its own cache key -- neither belongs
// inside a sweep that runs once per thread per colour.
static SFEM_NOINLINE void apply_jacobian_action_colored(MeshData                           &d,
                                                        PackedData                         &p,
                                                        const PackColoring                 &c,
                                                        const scalar_t                      rho,
                                                        const scalar_t                      mu,
                                                        const scalar_t *const SFEM_RESTRICT dir,
                                                        scalar_t *const SFEM_RESTRICT       jv,
                                                        const GeomKind                      geom) {
    cvfem_zero_scalars(jv, d.nnodes * N_FIELDS);
    cvfem_hex8_build_rc_coeff(d, rho, mu);

    const size_t scratch_n = packed_scratch_n(p.max_actual_nodes_per_pack);
    const int    with_rc   = !d.pgx.empty() && d.rhie_chow_scale != scalar_t(0);
    const bool   with_qg   = with_rc && !d.qgx.empty();

#pragma omp parallel
    {
        const int n_parts = cvfem_n_threads();
        const int part    = cvfem_thread_index();
        for (int color = 0; color < c.n_colors; ++color) {
            const cvfem_range r = cvfem_range_split(c.color_ptr[(size_t)color],
                                                    c.color_ptr[(size_t)color + 1], 1,
                                                    part, n_parts);
            if (geom == GeomKind::Isoparam)
                apply_jacobian_action_packcolored_isoparam_range(
                        r, c.pack_order.data(), d.nelements, d.p.data(), d.points, d.ux.data(),
                        d.uy.data(), d.uz.data(), p.elems, p.ghost_idx, p.ghost_ptr,
                        p.max_actual_nodes_per_pack, p.n_elements_per_pack, p.owned_nodes_ptr,
                        rho, mu, dir, jv, scratch_n);
            else
                apply_jacobian_action_packcolored_affine_range(
                        r, c.pack_order.data(), d.adj_ptr, d.det_ptr, d.elems, d.nelements, d.p.data(), d.pgx.data(),
                        d.pgy.data(), d.pgz.data(), d.points, d.qgx.data(), d.qgy.data(),
                        d.qgz.data(), d.rc_coeff.data(), d.rc_w.data(), d.rhie_chow_scale,
                        d.ux.data(), d.uy.data(), d.uz.data(), p.elems, p.ghost_idx, p.ghost_ptr,
                        p.max_actual_nodes_per_pack, p.n_elements_per_pack, p.owned_nodes_ptr,
                        rho, mu, dir, jv, scratch_n, with_rc, with_qg,
                        cvfem_hex8_rc_config_for(d));
            cvfem_thread_barrier();
        }
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
