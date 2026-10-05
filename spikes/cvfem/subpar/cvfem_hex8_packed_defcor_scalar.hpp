#pragma once

// THE PACKED HIGHER-ORDER RESIDUAL, SCALAR -- quarantined.
//
// Two clauses of DESIGN.md put it here: "for the matrix-free kernels only the SIMD version is
// kept, the rest is moved to subpar", and the correction's one micro-kernel per kernel -- the
// packed layout had two higher-order residual kernels and this is the slower one. Grace job
// 4981920, 8,586,756 dof, 72 threads, best of three:
//
//   unlimited          hand-written lane-blocked 1059.3   this sweep  940.6 MDOF/s
//   unlimited + rc     hand-written lane-blocked  944.9   this sweep  571.6 MDOF/s
//
// The comment this sweep used to carry said the opposite -- "the scalar kernel is the FASTER of
// the two for this operator, 659 against 500 MDOF/s (job 4812910)". That measurement predates
// the lane-blocked higher-order kernel's own optimisation; the ranking it recorded has since
// inverted, and the number above is the current one.
//
// The same job retired the GENERATED lane-blocked family, which was the packed default and
// which this sweep was the reference for, so both of its oracles went with them:
// verify_packed_ho_simd_vs_packed_ho_scalar_abs and
// verify_packed_ho_sympy_vs_packed_ho_scalar_abs. What checks the surviving kernel is
// verify_packed_ho_residual_vs_atomic_abs, against the ATOMIC higher-order sweep -- a different
// layout AND a different arrangement, so a stronger check than either of those two.
//
// Kept rather than deleted because the measurement above is what justifies preferring the
// lane-blocked kernel, and a number nobody can reproduce is not a justification. Build with
// -DCVFEM_ENABLE_SUBPAR to reach it; the driver's `--ho-scalar` went with it.

#include "frontend/staging/cvfem_hex8_best_common.hpp"
#include "kernels/packed/cvfem_hex8_best_packed.hpp"

// The DEFERRED-CORRECTION higher-order convective flux, on the packed layout.
//
// Same two-pass shape as apply_residual_packed: stage the pack's fields, accumulate into a
// pack-private buffer with plain `+=`, write the owned rows straight out, stage the ghosts and
// close them with the reduction graph. No atomic anywhere, and the same fixed summation order,
// so the higher-order operator inherits the format's reproducibility rather than giving it up.
//
// It runs the SCALAR sum-factored kernel, not the 16-wide SIMD one, because the scalar kernel
// is the FASTER of the two for this operator -- 659 against 500 MDOF/s on Grace at 8,586,756
// dof (job 4812910). Both accept the correction; the SIMD one re-gathers each lane's 96 inputs
// inside every one of the twelve face loops, and that costs more than vectorising the flux
// around it returns. That is also the honest basis for the comparison this enables: the atomic
// higher-order sweep runs the same scalar kernel, so packed against atomic here isolates the
// LAYOUT with the kernel held fixed. It also means a packed higher-order number is not
// comparable with a packed first-order one, which is SIMD -- the paper says so rather than
// letting the two sit in one column.
//
// `ugrad` is nine interleaved components per node and is hoisted, as the solver lags the
// correction one Newton step. The reconstruction also needs the element's node coordinates, so
// the pack stages its coordinates whenever the correction is on, exactly as Rhie-Chow does.
// The pack sweep, driven by a range. The `#pragma omp parallel` is in the launcher below; see
// kernels/cvfem_range.hpp for why DESIGN.md wants it there. The packs of a range touch only
// nodes this part owns -- that is what the packed layout is for -- so the parts need no
// synchronisation between them, and the ghost rows they do share are reduced afterwards in the
// launcher, which is the second and independent parallel loop.
static SFEM_NOINLINE void apply_residual_packed_defcor_scalar_range(
        const cvfem_range packs,
        // The mesh and the pack are staging objects -- they own vectors and a shared_ptr to a
        // mesh -- so what this kernel reads out of them is what it takes. DESIGN.md: only
        // arguments that are actually used are passed.
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        idx_t **const SFEM_RESTRICT mesh_elems,
        const uint8_t *const SFEM_RESTRICT face_mask,
        const ptrdiff_t nelements,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT pgx,
        const scalar_t *const SFEM_RESTRICT pgy,
        const scalar_t *const SFEM_RESTRICT pgz,
        geom_t **const SFEM_RESTRICT points,
        const scalar_t *const SFEM_RESTRICT qgx,
        const scalar_t *const SFEM_RESTRICT qgy,
        const scalar_t *const SFEM_RESTRICT qgz,
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
        const Hex8Extras & opt,
        const int with_rc) {
        scalar_t *const SFEM_RESTRICT pack_u   = thread_scratch<scalar_t>(0, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_out = thread_scratch<scalar_t>(1, scratch_n);
        // Coordinates always, and the pressure gradient when Rhie-Chow is on: the same
        // six-array slot the first-order SIMD path uses, so no new scratch shape appears.
        scalar_t *const SFEM_RESTRICT pack_xyz =
                thread_scratch<scalar_t>(3, with_rc ? packed_rc_n(max_actual_nodes_per_pack) : packed_xyz_n(max_actual_nodes_per_pack));
        const ptrdiff_t xyz_n = max_actual_nodes_per_pack > 0 ? max_actual_nodes_per_pack : 1;
        scalar_t *const SFEM_RESTRICT pack_x   = pack_xyz;
        scalar_t *const SFEM_RESTRICT pack_y   = pack_xyz + xyz_n;
        scalar_t *const SFEM_RESTRICT pack_z   = pack_xyz + 2 * xyz_n;
        scalar_t *const SFEM_RESTRICT pack_pgx = with_rc ? pack_xyz + 3 * xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_pgy = with_rc ? pack_xyz + 4 * xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_pgz = with_rc ? pack_xyz + 5 * xyz_n : nullptr;

    for (ptrdiff_t pack = packs.begin; pack < packs.end; ++pack) {
            const ptrdiff_t                         e_start      = pack * n_elements_per_pack;
            const ptrdiff_t                         e_end        = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t                         owned        = owned_nodes_ptr[pack];
            const ptrdiff_t                         n_contiguous = owned_nodes_ptr[pack + 1] - owned;
            const ptrdiff_t                         n_ghost      = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t                         n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts       = &ghost_idx[ghost_ptr[pack]];
            const ptrdiff_t                         ghost_off    = ghost_ptr[pack];

            std::memset(pack_out, 0, (size_t)n_pack_nodes * (size_t)CVFEM_HEX8_N_FIELDS * sizeof(scalar_t));
            fill_pack_fields(owned_nodes_ptr, ux, uy, uz, pres, pack, n_contiguous, n_ghost, ghosts, pack_u);
            if (with_rc)
                cvfem_hex8_fill_pack_xyz_pgrad(owned_nodes_ptr, points, pgx, pgy, pgz, with_rc, pack, n_contiguous, n_ghost, ghosts, pack_x,
                                               pack_y, pack_z, pack_pgx, pack_pgy, pack_pgz);
            else
                fill_pack_xyz(owned_nodes_ptr, points, pack, n_contiguous, n_ghost, ghosts, pack_x, pack_y, pack_z);

            // THE SIMD HIGHER-ORDER PATH IS CORRECT AND SLOWER, which is why this sweep runs
            // the scalar kernel. The 2.5e-05 discrepancy this comment used to record was real
            // but was not the reconstruction's geometry: the two EPS=false branches of
            // cvfem_hex8_ns_upwind_residual_sumfact_simd did not forward `ho` to
            // cvfem_hex8_conv_all_simd, so the correction was silently never applied on the
            // path the benchmark took. Found by observing that disabling the correction left
            // the discrepancy bit-identical. With `ho` forwarded the two layouts agree to
            // 1.3e-18, and the remaining reason to prefer the scalar kernel is throughput --
            // see the note on this function above.
            for (ptrdiff_t e = e_start; e < e_end; ++e) {
                scalar_t ux_e[8], uy_e[8], uz_e[8], p_e[8], r[CVFEM_HEX8_N_DOF], g8[72];
                // Fields come from the PACK -- read once per pack node, contiguously for the
                // owned majority, which is the layout's advantage.
                for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                    const scalar_t *const SFEM_RESTRICT u = pack_u + (ptrdiff_t)pack_elems[a][e] * CVFEM_HEX8_N_FIELDS;
                    ux_e[a] = u[0]; uy_e[a] = u[1]; uz_e[a] = u[2]; p_e[a] = u[3];
                }
                // The Rhie-Chow inputs come through the shared scratch, exactly as the atomic
                // sweep takes them, so the kernel sees identical inputs in both layouts.
                Hex8ExtraScratch ex;
                ex.load(mesh_elems, points, face_mask, pgx, pgy, pgz, qgx, qgy, qgz, ux, uy, uz, adj_ptr, det_ptr, opt, e);
                // Coordinates are gathered here rather than taken from `ex`, and that is not
                // redundant: Hex8ExtraScratch::load returns EARLY when neither Rhie-Chow nor
                // the boundary closure is on, leaving its x/y/z untouched. The reconstruction
                // works in physical space and needs them whether or not those terms are on, so
                // reading ex.x there gave an uninitialised buffer -- caught by the packed-vs-
                // atomic check at 2.9e-03, which is what that check is for.
                scalar_t xe[8], ye[8], ze[8];
                for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                    const idx_t g = mesh_elems[a][e];
                    xe[a] = scalar_t(points[0][g]);
                    ye[a] = scalar_t(points[1][g]);
                    ze[a] = scalar_t(points[2][g]);
                    for (int c = 0; c < 9; ++c) g8[a * 9 + c] = ugrad[(ptrdiff_t)g * 9 + c];
                }
                scalar_t adj[9], det;
                load_hex8_adj(adj_ptr, det_ptr, e, adj, &det);
                cvfem_hex8_ns_upwind_residual_sumfact(rho, mu, adj, det, ux_e, uy_e, uz_e, p_e, r,
                                                      ex.rc, /*ueps=*/scalar_t(0),
                                                      g8, xe, ye, ze,
                                                      limiter, venkat_c, nullptr);

                for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                    scalar_t *const SFEM_RESTRICT out = pack_out + (ptrdiff_t)pack_elems[a][e] * CVFEM_HEX8_N_FIELDS;
                    out[0] += r[a * 4 + 0];
                    out[1] += r[a * 4 + 1];
                    out[2] += r[a * 4 + 2];
                    out[3] += r[a * 4 + 3];
                }
            }

            for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                const scalar_t *const SFEM_RESTRICT out = pack_out + k * CVFEM_HEX8_N_FIELDS;
                const ptrdiff_t                     g   = owned + k;
                rx[g] = out[0]; ry[g] = out[1]; rz[g] = out[2]; rc[g] = out[3];
            }
            scalar_t *const SFEM_RESTRICT gx = ghost_buf + 0 * n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gy = ghost_buf + 1 * n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gz = ghost_buf + 2 * n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gc = ghost_buf + 3 * n_ghost_entries;
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                const scalar_t *const SFEM_RESTRICT out = pack_out + (n_contiguous + k) * CVFEM_HEX8_N_FIELDS;
                gx[ghost_off + k] = out[0]; gy[ghost_off + k] = out[1];
                gz[ghost_off + k] = out[2]; gc[ghost_off + k] = out[3];
            }
    }
}


static SFEM_NOINLINE void apply_residual_packed_defcor_scalar(MeshData       &d,
                                                       PackedData     &p,
                                                       const scalar_t  rho,
                                                       const scalar_t  mu,
                                                       const scalar_t *const SFEM_RESTRICT ugrad,
                                                       const int       limiter,
                                                       const scalar_t  venkat_c) {
    scalar_t *const SFEM_RESTRICT rx = d.rx.data();
    scalar_t *const SFEM_RESTRICT ry = d.ry.data();
    scalar_t *const SFEM_RESTRICT rz = d.rz.data();
    scalar_t *const SFEM_RESTRICT rc = d.rc.data();
    const size_t                  scratch_n = packed_scratch_n(p.max_actual_nodes_per_pack);
    const Hex8Extras              opt = cvfem_hex8_extras_of(d);
    const int                     with_rc = !d.pgx.empty() && d.rhie_chow_scale != scalar_t(0);


#pragma omp parallel
    apply_residual_packed_defcor_scalar_range(cvfem_range_split(0, p.n_packs, 1, cvfem_thread_index(), cvfem_n_threads()),
            d.adj_ptr, d.det_ptr, d.elems, d.face_mask.data(), d.nelements, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.data(), d.qgy.data(), d.qgz.data(), d.rhie_chow_scale, d.ux.data(), d.uy.data(), d.uz.data(), p.elems, p.ghost_buf.data(), p.ghost_idx, p.ghost_ptr, p.max_actual_nodes_per_pack, p.n_elements_per_pack, p.n_ghost_entries, p.owned_nodes_ptr, rho, mu, ugrad, limiter, venkat_c, rx, ry, rz, rc, scratch_n, opt, with_rc);

    scalar_t *const fields[CVFEM_HEX8_N_FIELDS] = {d.rx.data(), d.ry.data(), d.rz.data(), d.rc.data()};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < p.n_ghost_reduce_rows; ++row) {
        const idx_t dest  = p.ghost_reduce_dest[row];
        const ptrdiff_t    begin = p.ghost_reduce_ptr[row];
        const ptrdiff_t    end   = p.ghost_reduce_ptr[row + 1];
        for (int f = 0; f < CVFEM_HEX8_N_FIELDS; ++f) {
            const scalar_t *const SFEM_RESTRICT ghost = p.ghost_buf.data() + (ptrdiff_t)f * p.n_ghost_entries;
            scalar_t                            sum   = 0;
            for (ptrdiff_t j = begin; j < end; ++j) sum += ghost[p.ghost_reduce_idx[j]];
            fields[f][dest] += sum;
        }
    }
}
