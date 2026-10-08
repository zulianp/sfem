#pragma once

// The colored layout's affine sweeps.
//
// Separated by geometry as DESIGN.md asks, by MOVING whole functions: these two sets were always
// distinct sweeps, differing in where the Jacobian comes from -- one adjugate per element read
// from a table, against one derived per sub-control volume from the node coordinates. Nothing is
// duplicated to achieve the split.
//
// Where a format's two geometries are ONE sweep templated on `bool ISO` -- the packed and store
// layouts -- they stay that way. DESIGN.md's clause is "logically separated (now they are mixed
// in with enum and booleans)"; the enum and the booleans are gone and the choice is made at
// compile time, which is the separation it asks for. Splitting those physically would mean two
// copies of the pack staging, the drain and the ghost reduction.

#include "kernels/colored/cvfem_hex8_best_ecolored.hpp"


// THE THREADING IS OUTSIDE THE KERNEL, AND THE KERNEL TAKES A RANGE.
//
// DESIGN.md: the threading model is abstract outside the function and what arrives is a range,
// so that a thread library other than OpenMP can drive these sweeps. Each kernel below is now a
// plain loop over one cvfem_range of elements, and the `#pragma omp parallel` lives in the
// launcher beneath it.
//
// THE COLOUR BARRIER IS THE REASON THIS IS NOT A SUBSTITUTION. The synchronisation between
// colours used to BE the implicit barrier at the end of `#pragma omp for`; with the loop gone
// that barrier has to be written. The launcher therefore keeps the colour loop, hands each thread
// its own slice of the colour through cvfem_range_split, and barriers once per colour. The split
// is group-aligned and reproduces schedule(static) over the same strided loop, so the work per
// thread is what it was.
//
// The scratch comes in as arguments rather than being declared inside the kernel, which is what
// DESIGN.md's "only arguments that are actually used" asks for and what removes the hidden
// per-thread state: the launcher owns one set per thread, in its parallel region, and passes it.
template <typename scalar_t, typename geom_t, typename idx_t, typename grad_t>
static SFEM_NOINLINE void apply_residual_ecolored_range(
        const cvfem_range r,
        // The mesh and the pack are staging objects -- they own vectors and a shared_ptr to a
        // mesh -- so what this kernel reads out of them is what it takes. DESIGN.md: only
        // arguments that are actually used are passed.
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        idx_t **const SFEM_RESTRICT mesh_elems,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT pgx,
        const scalar_t *const SFEM_RESTRICT pgy,
        const scalar_t *const SFEM_RESTRICT pgz,
        geom_t **const SFEM_RESTRICT points,
        const scalar_t rhie_chow_scale,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        const scalar_t    rho,
        const scalar_t    mu,
        const grad_t *const SFEM_RESTRICT ugrad,
        const int         limiter,
        const scalar_t    venkat_c,
        const Hex8Extras &opt,
        const bool        with_ho,
        scalar_t *const SFEM_RESTRICT rx,
        scalar_t *const SFEM_RESTRICT ry,
        scalar_t *const SFEM_RESTRICT rz,
        scalar_t *const SFEM_RESTRICT rc_out,
        scalar_t *const SFEM_RESTRICT cof0,
        scalar_t *const SFEM_RESTRICT cof1,
        scalar_t *const SFEM_RESTRICT cof2,
        scalar_t *const SFEM_RESTRICT cof3,
        scalar_t *const SFEM_RESTRICT cof4,
        scalar_t *const SFEM_RESTRICT cof5,
        scalar_t *const SFEM_RESTRICT cof6,
        scalar_t *const SFEM_RESTRICT cof7,
        scalar_t *const SFEM_RESTRICT cof8,
        scalar_t *const SFEM_RESTRICT detv,
        Hex8InputPackT<scalar_t>    &in,
        Hex8ResidualPackT<scalar_t> &outp,
        Hex8RhieChowPackT<scalar_t> &rcp,
        Hex8UGradPackT<scalar_t>    &hop) {
    for (ptrdiff_t e0 = r.begin; e0 < r.end; e0 += CVFEM_HEX8_VEC_SIZE) {
            const int nlanes = (int)MIN((ptrdiff_t)CVFEM_HEX8_VEC_SIZE, r.end - e0);
            gather_hex8_adj_soa(adj_ptr, det_ptr, e0, nlanes, cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, detv);

            for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
                if (lane < nlanes) {
                    const ptrdiff_t e = e0 + lane;
                    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                        const idx_t g = mesh_elems[a][e];
                        in.ux[a][lane]       = ux[g];
                        in.uy[a][lane]       = uy[g];
                        in.uz[a][lane]       = uz[g];
                        in.p[a][lane]        = pres[g];
                    }
                } else {
                    // A padding lane must carry a STATE, not zeros that the flux would treat as a
                    // real element: the kernel has no lane mask, so its output is discarded by the
                    // scatter below rather than by the kernel. Zeros are safe because the geometry
                    // gather already zeroed the adjugate and set det to one for these lanes.
                    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a)
                        in.ux[a][lane] = in.uy[a][lane] = in.uz[a][lane] = in.p[a][lane] = scalar_t(0);
                }
            }

            if (opt.with_rc) {
                const auto *const px = points[0];
                const auto *const py = points[1];
                const auto *const pz = points[2];
                for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
                    if (lane < nlanes) {
                        const ptrdiff_t e = e0 + lane;
                        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                            const idx_t g = mesh_elems[a][e];
                            rcp.pgx[a][lane]     = pgx[g];
                            rcp.pgy[a][lane]     = pgy[g];
                            rcp.pgz[a][lane]     = pgz[g];
                        }
                    } else {
                        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                            rcp.pgx[a][lane] = rcp.pgy[a][lane] = rcp.pgz[a][lane] = scalar_t(0);
                        }
                    }
                }
            }

            if (with_ho) {
                for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
                    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                        if (lane >= nlanes) {
                            for (int c = 0; c < 9; ++c) hop.g[a][c][lane] = scalar_t(0);
                            continue;
                        }
                        const idx_t gn = mesh_elems[a][e0 + lane];
                        for (int c = 0; c < 9; ++c) hop.g[a][c][lane] = scalar_t(ugrad[(ptrdiff_t)gn * 9 + c]);
                    }
                }
                hop.limiter  = limiter;
                hop.venkat_c = venkat_c;
            }

            // ONE HIGHER-ORDER MICRO-KERNEL. The generated per-limiter variants used to run here
            // whenever the Venkatakrishnan eps squared was zero, on the grounds that they specialise
            // the limiter where the hand-written kernel selects inside the vector body. Grace job
            // 4981920 measured the two against each other for the first time and the hand-written one
            // wins all seven pairs (0.775x to 0.976x), so the generated family is in subpar/ and this
            // is the only arm left.
            cvfem_hex8_ns_upwind_residual_sumfact_simd(rho, mu, cof0, cof1, cof2, cof3, cof4, cof5,
                                                       cof6, cof7, cof8, detv, in, outp,
                                                       opt.with_rc ? &rcp : nullptr, rhie_chow_scale,
                                                       scalar_t(0), with_ho ? &hop : nullptr);

            for (int lane = 0; lane < nlanes; ++lane) {
                const ptrdiff_t e = e0 + lane;
                for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                    const idx_t g = mesh_elems[a][e];
                    rx[g] += outp.rx[a][lane];
                    ry[g] += outp.ry[a][lane];
                    rz[g] += outp.rz[a][lane];
                    rc_out[g] += outp.rc[a][lane];
                }
            }
    }
}


// The Jacobian action, split the same way and for the same reasons. See
// apply_residual_ecolored_range above.
template <typename scalar_t, typename geom_t, typename idx_t, typename grad_t>
static SFEM_NOINLINE void apply_jacobian_action_ecolored_range(
        const cvfem_range r,
        // The mesh and the pack are staging objects -- they own vectors and a shared_ptr to a
        // mesh -- so what this kernel reads out of them is what it takes. DESIGN.md: only
        // arguments that are actually used are passed.
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        idx_t **const SFEM_RESTRICT mesh_elems,
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
        const scalar_t    rho,
        const scalar_t    mu,
        const scalar_t *const dir,
        scalar_t *const       jv,
        const grad_t *const SFEM_RESTRICT ugrad,
        const grad_t *const SFEM_RESTRICT vgrad,
        const int         limiter,
        const scalar_t    venkat_c,
        const Hex8Extras &opt,
        const bool        has_qg,
        const bool        with_ho,
        scalar_t *const SFEM_RESTRICT cof0,
        scalar_t *const SFEM_RESTRICT cof1,
        scalar_t *const SFEM_RESTRICT cof2,
        scalar_t *const SFEM_RESTRICT cof3,
        scalar_t *const SFEM_RESTRICT cof4,
        scalar_t *const SFEM_RESTRICT cof5,
        scalar_t *const SFEM_RESTRICT cof6,
        scalar_t *const SFEM_RESTRICT cof7,
        scalar_t *const SFEM_RESTRICT cof8,
        scalar_t *const SFEM_RESTRICT detv,
        Hex8InputPackT<scalar_t>    &u_pack,
        Hex8InputPackT<scalar_t>    &du_pack,
        Hex8ResidualPackT<scalar_t> &outp,
        Hex8RhieChowPackT<scalar_t> &rcp,
        Hex8UGradPackT<scalar_t>    &hop,
        Hex8UGradPackT<scalar_t>    &hovp,
        // Resolved once per solve, in the launcher, not per element here. This parameter replaced
        // the cvfem_hex8_rc_config_for(d) call that used to sit in this body: that function takes
        // the mesh, which a kernel is not meant to name.
        const Hex8RcConfigT<scalar_t> &rc_cfg) {
    for (ptrdiff_t e0 = r.begin; e0 < r.end; e0 += CVFEM_HEX8_VEC_SIZE) {
            const int nlanes = (int)MIN((ptrdiff_t)CVFEM_HEX8_VEC_SIZE, r.end - e0);
            gather_hex8_adj_soa(adj_ptr, det_ptr, e0, nlanes, cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, detv);

            for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
                if (lane < nlanes) {
                    const ptrdiff_t e = e0 + lane;
                    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                        const idx_t                  g  = mesh_elems[a][e];
                        const scalar_t *const SFEM_RESTRICT dv = dir + (ptrdiff_t)g * CVFEM_HEX8_N_FIELDS;
                        u_pack.ux[a][lane]                     = ux[g];
                        u_pack.uy[a][lane]                     = uy[g];
                        u_pack.uz[a][lane]                     = uz[g];
                        u_pack.p[a][lane]                      = pres[g];
                        du_pack.ux[a][lane]                    = dv[0];
                        du_pack.uy[a][lane]                    = dv[1];
                        du_pack.uz[a][lane]                    = dv[2];
                        du_pack.p[a][lane]                     = dv[3];
                    }
                } else {
                    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                        u_pack.ux[a][lane] = u_pack.uy[a][lane] = u_pack.uz[a][lane] = u_pack.p[a][lane] = scalar_t(0);
                        du_pack.ux[a][lane] = du_pack.uy[a][lane] = du_pack.uz[a][lane] = du_pack.p[a][lane] =
                                scalar_t(0);
                    }
                }
            }

            if (opt.with_rc) {
                const auto *const px = points[0];
                const auto *const py = points[1];
                const auto *const pz = points[2];
                for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
                    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                        if (lane < nlanes) {
                            const idx_t g = mesh_elems[a][e0 + lane];
                            rcp.pgx[a][lane]     = pgx[g];
                            rcp.pgy[a][lane]     = pgy[g];
                            rcp.pgz[a][lane]     = pgz[g];
                            rcp.qgx[a][lane]     = has_qg ? qgx[g] : scalar_t(0);
                            rcp.qgy[a][lane]     = has_qg ? qgy[g] : scalar_t(0);
                            rcp.qgz[a][lane]     = has_qg ? qgz[g] : scalar_t(0);
                        } else {
                            rcp.pgx[a][lane] = rcp.pgy[a][lane] = rcp.pgz[a][lane] = scalar_t(0);
                            rcp.qgx[a][lane] = rcp.qgy[a][lane] = rcp.qgz[a][lane] = scalar_t(0);
                        }
                    }
                }
            }

            if (opt.with_rc) cvfem_hex8_gather_rc_coeff(rc_coeff, rc_w, rc_cfg, e0, nlanes, rcp);

            // The state's nodal velocity gradient and the direction's, staged exactly as the
            // packed Jacobian stages them. Both are needed by the EXACT higher-order action; the
            // lagged one passes neither and gets the first-order kernel bit for bit.
            if (with_ho) {
                for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
                    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                        if (lane >= nlanes) {
                            for (int c = 0; c < 9; ++c) {
                                hop.g[a][c][lane]  = scalar_t(0);
                                hovp.g[a][c][lane] = scalar_t(0);
                            }
                            continue;
                        }
                        const idx_t gn = mesh_elems[a][e0 + lane];
                        for (int c = 0; c < 9; ++c) {
                            hop.g[a][c][lane]  = scalar_t(ugrad[(ptrdiff_t)gn * 9 + c]);
                            hovp.g[a][c][lane] = scalar_t(vgrad[(ptrdiff_t)gn * 9 + c]);
                        }
                    }
                }
                hop.limiter  = limiter;
                hop.venkat_c = venkat_c;
            }

            cvfem_hex8_ns_upwind_jacobian_action_simd(rho, mu, cof0, cof1, cof2, cof3, cof4, cof5, cof6,
                                                      cof7, cof8, detv, u_pack, du_pack, outp,
                                                      opt.with_rc ? &rcp : nullptr, rhie_chow_scale, has_qg,
                                                      scalar_t(0), with_ho ? &hop : nullptr,
                                                      with_ho ? &hovp : nullptr);

            for (int lane = 0; lane < nlanes; ++lane) {
                const ptrdiff_t e = e0 + lane;
                for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                    const idx_t g = mesh_elems[a][e];
                    jv[g * CVFEM_HEX8_N_FIELDS + 0] += outp.rx[a][lane];
                    jv[g * CVFEM_HEX8_N_FIELDS + 1] += outp.ry[a][lane];
                    jv[g * CVFEM_HEX8_N_FIELDS + 2] += outp.rz[a][lane];
                    jv[g * CVFEM_HEX8_N_FIELDS + 3] += outp.rc[a][lane];
                }
            }
    }
}
