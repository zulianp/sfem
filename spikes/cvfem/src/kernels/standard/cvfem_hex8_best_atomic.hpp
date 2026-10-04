#ifndef CVFEM_HEX8_BEST_ATOMIC_HPP
#define CVFEM_HEX8_BEST_ATOMIC_HPP

// Atomic layout: a flat parallel sweep over elements that writes into the global
// residual / matrix with #pragma omp atomic on every entry. No mesh partitioning
// and no scratch, which makes it the simplest and the reference for correctness,
// but assembly pays ~1024 atomic read-modify-writes per element.

#include "kernels/cvfem_phases.hpp"
#include "best/cvfem_hex8_best_common.hpp"

// The two gradient arguments carry the exact higher-order action, as on the packed sweep; both
// null is the lagged one. The atomic path exists here so the layouts can be compared on the same
// operator -- a packed row carrying the correction against an atomic row that silently dropped it
// would not be a layout comparison.
static SFEM_NOINLINE void apply_jacobian_action_atomic(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        idx_t **const SFEM_RESTRICT mesh_elems,
        const uint8_t *const SFEM_RESTRICT face_mask,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT pgx,
        const scalar_t *const SFEM_RESTRICT pgy,
        const scalar_t *const SFEM_RESTRICT pgz,
        geom_t **const SFEM_RESTRICT points,
        const scalar_t *const SFEM_RESTRICT qgx,
        const scalar_t *const SFEM_RESTRICT qgy,
        const scalar_t *const SFEM_RESTRICT qgz,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src,
                                              // Which optional terms are on, resolved once per
                                              // solve by the caller rather than per sweep here:
                                              // cvfem_hex8_extras_of reads the mesh, which this
                                              // kernel is not meant to name.
                                              const Hex8Extras &opt,
                                                       const scalar_t        rho,
                                                       const scalar_t        mu,
                                                       const scalar_t *const dir,
                                                       scalar_t *const       jv,
                                                       const KernelKind      kernel = KernelKind::Sumfact,
                                                       const scalar_t *const SFEM_RESTRICT ugrad = nullptr,
                                                       const scalar_t *const SFEM_RESTRICT vgrad = nullptr,
                                                       const int             limiter = 0,
                                                       const scalar_t        venkat_c = scalar_t(0)) {
    const bool with_ho = ugrad != nullptr && vgrad != nullptr;
    cvfem_zero_scalars(jv, nnodes * N_FIELDS);


#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8], vx[8], vy[8], vz[8], q[8], r[CVFEM_HEX8_N_DOF];
        scalar_t xe[8], ye[8], ze[8], g8[72], gv8[72];
        gather_element_fields(mesh_elems, ux_src, uy_src, uz_src, pres, e, ux, uy, uz, p);
        if (with_ho) {
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const idx_t gn = mesh_elems[a][e];
                xe[a] = scalar_t(points[0][gn]);
                ye[a] = scalar_t(points[1][gn]);
                ze[a] = scalar_t(points[2][gn]);
                for (int c = 0; c < 9; ++c) {
                    g8[a * 9 + c]  = ugrad[(ptrdiff_t)gn * 9 + c];
                    gv8[a * 9 + c] = vgrad[(ptrdiff_t)gn * 9 + c];
                }
            }
        }
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const idx_t g         = mesh_elems[a][e];
            const scalar_t *const SFEM_RESTRICT dv = dir + (ptrdiff_t)g * N_FIELDS;
            vx[a]                            = dv[0];
            vy[a]                            = dv[1];
            vz[a]                            = dv[2];
            q[a]                             = dv[3];
        }
        scalar_t adj[9], det;
        load_hex8_adj(adj_ptr, det_ptr, e, adj, &det);
        // The generated Jacobian-action arrangements. They carry no Rhie-Chow term -- the
        // generator builds them from the bare flux algebra, as it does the residual and the
        // assembly -- so the driver refuses --rhie-chow with them rather than letting a row
        // claim a term the kernel does not compute.
        if (kernel == KernelKind::SympyAction) {
            cvfem_hex8_ns_upwind_sympy_jacobian_action(rho, mu, adj, det, ux, uy, uz, vx, vy, vz, q, r);
        } else if (kernel == KernelKind::SympyActionNode) {
            cvfem_hex8_ns_upwind_sympy_jacobian_action_nodewise(rho, mu, adj, det, ux, uy, uz, vx, vy, vz, q, r);
        } else if (kernel == KernelKind::SympyActionComp) {
            cvfem_hex8_ns_upwind_sympy_jacobian_action_componentwise(rho, mu, adj, det, ux, uy, uz, vx, vy, vz, q, r);
        } else if (kernel == KernelKind::SympyActionFace) {
            cvfem_hex8_ns_upwind_sympy_jacobian_action_facewise(rho, mu, adj, det, ux, uy, uz, vx, vy, vz, q, r);
        } else if (kernel == KernelKind::SympyActionGeom) {
            cvfem_hex8_ns_upwind_sympy_jacobian_action_geom(rho, mu, adj, det, ux, uy, uz, vx, vy, vz, q, r);
        } else if (kernel == KernelKind::SympyActionGeomFace) {
            cvfem_hex8_ns_upwind_sympy_jacobian_action_geomface(rho, mu, adj, det, ux, uy, uz, vx, vy, vz, q, r);
        }
        // Branch rather than pass `ex.rc` and `p` unconditionally: with --rhie-chow off the
        // literal call below hands the kernel a default-constructed rc and a null pressure,
        // both of which fold away at inline time, so the default path emits exactly the code
        // it emitted before this option existed. A runtime-valued rc would leave the
        // Rhie-Chow branch in the hot loop for every run that does not ask for it.
        else if (opt.with_rc) {
            Hex8ExtraScratch ex;
            ex.load(mesh_elems, points, face_mask, pgx, pgy, pgz, qgx, qgy, qgz, ux_src, uy_src, uz_src, adj_ptr, det_ptr, opt, e);
            // The limiter is selected at compile time, as it is on the packed sweep and for the
            // same reason; the switch sits here, outside the element loop's face loop.
#define CVFEM_HEX8_JV_RC(LIM_)                                                                  \
    cvfem_hex8_ns_upwind_jacobian_action<LIM_>(rho, mu, adj, det, ux, uy, uz, vx, vy, vz, q, r,  \
                                               ex.rc, p, scalar_t(0),                           \
                                               with_ho ? xe : nullptr, with_ho ? ye : nullptr,  \
                                               with_ho ? ze : nullptr, with_ho ? g8 : nullptr,  \
                                               with_ho ? gv8 : nullptr, venkat_c)
            switch (limiter) {
                case 1: CVFEM_HEX8_JV_RC(1); break;
                case 2: CVFEM_HEX8_JV_RC(2); break;
                case 3: CVFEM_HEX8_JV_RC(3); break;
                default: CVFEM_HEX8_JV_RC(0); break;
            }
#undef CVFEM_HEX8_JV_RC
        } else {
#define CVFEM_HEX8_JV_BARE(LIM_)                                                                \
    cvfem_hex8_ns_upwind_jacobian_action<LIM_>(rho, mu, adj, det, ux, uy, uz, vx, vy, vz, q, r,  \
                                               Hex8RhieChowT<scalar_t>{},                       \
                                               (const scalar_t *)nullptr, scalar_t(0),          \
                                               with_ho ? xe : nullptr, with_ho ? ye : nullptr,  \
                                               with_ho ? ze : nullptr, with_ho ? g8 : nullptr,  \
                                               with_ho ? gv8 : nullptr, venkat_c)
            switch (limiter) {
                case 1: CVFEM_HEX8_JV_BARE(1); break;
                case 2: CVFEM_HEX8_JV_BARE(2); break;
                case 3: CVFEM_HEX8_JV_BARE(3); break;
                default: CVFEM_HEX8_JV_BARE(0); break;
            }
#undef CVFEM_HEX8_JV_BARE
        }
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const idx_t g = mesh_elems[a][e];
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 0, 0, r[a * 4 + 0]);
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 1, 0, r[a * 4 + 1]);
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 2, 0, r[a * 4 + 2]);
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 3, 0, r[a * 4 + 3]);
        }
    }
}

// THE JACOBIAN ACTION, LANE-BLOCKED, with the atomic scatter. Same reasoning as the residual
// twin above: cvfem_hex8_ns_upwind_jacobian_action_simd is what packed, coloured and the solver's
// packed path all call, and the atomic sweep was the only family still walking one element at a
// time through the scalar kernel. Vectorising it is the difference between measuring the format
// and measuring the vectorisation.
//
// Global gather through d.elems, wide index, untouched element order, per-lane atomic scatter --
// everything that makes this the standard layout is kept.
static SFEM_NOINLINE void apply_jacobian_action_atomic_simd(MeshData             &d,
                                                            const scalar_t        rho,
                                                            const scalar_t        mu,
                                                            const scalar_t *const dir,
                                                            scalar_t *const       jv,
                                                            const scalar_t *const SFEM_RESTRICT ugrad = nullptr,
                                                            const scalar_t *const SFEM_RESTRICT vgrad = nullptr,
                                                            const int             limiter  = 0,
                                                            const scalar_t        venkat_c = scalar_t(0)) {
    cvfem_zero_scalars(jv, d.nnodes * N_FIELDS);
    const Hex8Extras opt = cvfem_hex8_extras_of(d);
    const bool       has_qg  = opt.with_qg;
    const bool       with_ho = ugrad != nullptr && vgrad != nullptr;
    // The per-surface Rhie-Chow coefficient is hoisted out of the face loops, so it has to be
    // built before the sweep and staged per lane group -- exactly as the packed Jacobian does.
    // Omitting the staging leaves rcp.coeff, rcp.scale and rcp.tau untouched and the kernel
    // returns nan, which is how this was found.
    if (opt.with_rc) cvfem_hex8_build_rc_coeff(d, rho, mu);

#pragma omp parallel
    {
        alignas(ALIGN_BYTES) scalar_t cof0[CVFEM_HEX8_VEC_SIZE], cof1[CVFEM_HEX8_VEC_SIZE];
        alignas(ALIGN_BYTES) scalar_t cof2[CVFEM_HEX8_VEC_SIZE], cof3[CVFEM_HEX8_VEC_SIZE];
        alignas(ALIGN_BYTES) scalar_t cof4[CVFEM_HEX8_VEC_SIZE], cof5[CVFEM_HEX8_VEC_SIZE];
        alignas(ALIGN_BYTES) scalar_t cof6[CVFEM_HEX8_VEC_SIZE], cof7[CVFEM_HEX8_VEC_SIZE];
        alignas(ALIGN_BYTES) scalar_t cof8[CVFEM_HEX8_VEC_SIZE], detv[CVFEM_HEX8_VEC_SIZE];
        Hex8InputPack    u_pack, du_pack;
        Hex8ResidualPack outp;
        Hex8RhieChowPack rcp;
        Hex8UGradPack    hop, hovp;

#pragma omp for schedule(static)
        for (ptrdiff_t e0 = 0; e0 < d.nelements; e0 += CVFEM_HEX8_VEC_SIZE) {
            const int nlanes = (int)MIN((ptrdiff_t)CVFEM_HEX8_VEC_SIZE, d.nelements - e0);
            gather_hex8_adj_soa(d.adj_ptr, d.det_ptr, e0, nlanes, cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, detv);

            for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
                if (lane < nlanes) {
                    const ptrdiff_t e = e0 + lane;
                    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                        const idx_t                  g  = d.elems[a][e];
                        const scalar_t *const SFEM_RESTRICT dv = dir + (ptrdiff_t)g * N_FIELDS;
                        u_pack.ux[a][lane]                     = d.ux[g];
                        u_pack.uy[a][lane]                     = d.uy[g];
                        u_pack.uz[a][lane]                     = d.uz[g];
                        u_pack.p[a][lane]                      = d.p[g];
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
                const auto *const px = d.points[0];
                const auto *const py = d.points[1];
                const auto *const pz = d.points[2];
                for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
                    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                        if (lane < nlanes) {
                            const idx_t g = d.elems[a][e0 + lane];
                            rcp.pgx[a][lane]     = d.pgx[g];
                            rcp.pgy[a][lane]     = d.pgy[g];
                            rcp.pgz[a][lane]     = d.pgz[g];
                            rcp.qgx[a][lane]     = has_qg ? d.qgx[g] : scalar_t(0);
                            rcp.qgy[a][lane]     = has_qg ? d.qgy[g] : scalar_t(0);
                            rcp.qgz[a][lane]     = has_qg ? d.qgz[g] : scalar_t(0);
                        } else {
                            rcp.pgx[a][lane] = rcp.pgy[a][lane] = rcp.pgz[a][lane] = scalar_t(0);
                            rcp.qgx[a][lane] = rcp.qgy[a][lane] = rcp.qgz[a][lane] = scalar_t(0);
                        }
                    }
                }
            }

            if (opt.with_rc) cvfem_hex8_gather_rc_coeff(d.rc_coeff.data(), d.rc_w.data(), cvfem_hex8_rc_config_for(d), e0, nlanes, rcp);

            // The state's nodal velocity gradient and the direction's, staged exactly as the
            // packed Jacobian stages them. Both are needed by the EXACT higher-order action; the
            // lagged one passes neither and gets the first-order kernel bit for bit.
            if (with_ho) {
                for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
                    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                        if (lane >= nlanes) {
                            hop.x[a][lane] = hop.y[a][lane] = hop.z[a][lane] = scalar_t(0);
                            for (int c = 0; c < 9; ++c) {
                                hop.g[a][c][lane]  = scalar_t(0);
                                hovp.g[a][c][lane] = scalar_t(0);
                            }
                            continue;
                        }
                        const idx_t gn = d.elems[a][e0 + lane];
                        hop.x[a][lane]        = scalar_t(d.points[0][gn]);
                        hop.y[a][lane]        = scalar_t(d.points[1][gn]);
                        hop.z[a][lane]        = scalar_t(d.points[2][gn]);
                        for (int c = 0; c < 9; ++c) {
                            hop.g[a][c][lane]  = ugrad[(ptrdiff_t)gn * 9 + c];
                            hovp.g[a][c][lane] = vgrad[(ptrdiff_t)gn * 9 + c];
                        }
                    }
                }
                hop.limiter  = limiter;
                hop.venkat_c = venkat_c;
            }

            cvfem_hex8_ns_upwind_jacobian_action_simd(rho, mu, cof0, cof1, cof2, cof3, cof4, cof5, cof6,
                                                      cof7, cof8, detv, u_pack, du_pack, outp,
                                                      opt.with_rc ? &rcp : nullptr, d.rhie_chow_scale, has_qg,
                                                      scalar_t(0), with_ho ? &hop : nullptr,
                                                      with_ho ? &hovp : nullptr);

            for (int lane = 0; lane < nlanes; ++lane) {
                const ptrdiff_t e = e0 + lane;
                for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                    const idx_t g = d.elems[a][e];
                    atomic_add(jv, (idx_t)(g * N_FIELDS + 0), outp.rx[a][lane]);
                    atomic_add(jv, (idx_t)(g * N_FIELDS + 1), outp.ry[a][lane]);
                    atomic_add(jv, (idx_t)(g * N_FIELDS + 2), outp.rz[a][lane]);
                    atomic_add(jv, (idx_t)(g * N_FIELDS + 3), outp.rc[a][lane]);
                }
            }
        }
    }
}

static SFEM_NOINLINE void apply_jacobian_action_atomic_isoparam(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        idx_t **const SFEM_RESTRICT mesh_elems,
        const uint8_t *const SFEM_RESTRICT face_mask,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT pgx,
        const scalar_t *const SFEM_RESTRICT pgy,
        const scalar_t *const SFEM_RESTRICT pgz,
        geom_t **const SFEM_RESTRICT points,
        const scalar_t *const SFEM_RESTRICT qgx,
        const scalar_t *const SFEM_RESTRICT qgy,
        const scalar_t *const SFEM_RESTRICT qgz,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src,
                                              // Which optional terms are on, resolved once per
                                              // solve by the caller rather than per sweep here:
                                              // cvfem_hex8_extras_of reads the mesh, which this
                                              // kernel is not meant to name.
                                              const Hex8Extras &opt,
                                                                const scalar_t        rho,
                                                                const scalar_t        mu,
                                                                const scalar_t *const dir,
                                                                scalar_t *const       jv) {
    cvfem_zero_scalars(jv, nnodes * N_FIELDS);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        scalar_t         ux[8], uy[8], uz[8], p[8], vx[8], vy[8], vz[8], q[8], r[CVFEM_HEX8_N_DOF];
        Hex8ExtraScratch ex;
        ex.load(mesh_elems, points, face_mask, pgx, pgy, pgz, qgx, qgy, qgz, ux_src, uy_src, uz_src, adj_ptr, det_ptr, opt, e);
        if (!opt.with_rc && !opt.with_bnd) gather_element_coords(mesh_elems, points, e, ex.x, ex.y, ex.z);
        gather_element_fields(mesh_elems, ux_src, uy_src, uz_src, pres, e, ux, uy, uz, p);
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const idx_t                  g  = mesh_elems[a][e];
            const scalar_t *const SFEM_RESTRICT dv = dir + (ptrdiff_t)g * N_FIELDS;
            vx[a]                                  = dv[0];
            vy[a]                                  = dv[1];
            vz[a]                                  = dv[2];
            q[a]                                   = dv[3];
        }
        cvfem_hex8_ns_upwind_jacobian_action_isoparam(rho, mu, ex.x, ex.y, ex.z, ux, uy, uz, vx, vy, vz, q, r,
                                                      ex.rc, p);
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const idx_t g = mesh_elems[a][e];
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 0, 0, r[a * 4 + 0]);
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 1, 0, r[a * 4 + 1]);
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 2, 0, r[a * 4 + 2]);
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 3, 0, r[a * 4 + 3]);
        }
    }
}

static SFEM_NOINLINE void apply_residual_atomic(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        idx_t **const SFEM_RESTRICT mesh_elems,
        const uint8_t *const SFEM_RESTRICT face_mask,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT pgx,
        const scalar_t *const SFEM_RESTRICT pgy,
        const scalar_t *const SFEM_RESTRICT pgz,
        geom_t **const SFEM_RESTRICT points,
        const scalar_t *const SFEM_RESTRICT qgx,
        const scalar_t *const SFEM_RESTRICT qgy,
        const scalar_t *const SFEM_RESTRICT qgz,
        scalar_t *const SFEM_RESTRICT rc_out,
        scalar_t *const SFEM_RESTRICT rx,
        scalar_t *const SFEM_RESTRICT ry,
        scalar_t *const SFEM_RESTRICT rz,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src,
                                              // Which optional terms are on, resolved once per
                                              // solve by the caller rather than per sweep here:
                                              // cvfem_hex8_extras_of reads the mesh, which this
                                              // kernel is not meant to name.
                                              const Hex8Extras &opt, const scalar_t rho, const scalar_t mu) {
    reset_residual(nnodes, rx, ry, rz, rc_out);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8], r[CVFEM_HEX8_N_DOF];
        gather_element_fields(mesh_elems, ux_src, uy_src, uz_src, pres, e, ux, uy, uz, p);
        Hex8ExtraScratch ex;
        ex.load(mesh_elems, points, face_mask, pgx, pgy, pgz, qgx, qgy, qgz, ux_src, uy_src, uz_src, adj_ptr, det_ptr, opt, e);
        scalar_t adj[9], det;
        load_hex8_adj(adj_ptr, det_ptr, e, adj, &det);
        // No rc: the hand-written `current` kernel carries no Rhie-Chow term. --rhie-chow
        // is rejected for this kernel at the CLI, so reaching here with it on is a bug.
        cvfem_hex8_ns_upwind_residual(rho, mu, adj, det, ux, uy, uz, p, r);

        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const idx_t g = mesh_elems[a][e];
            atomic_add(rx, g, r[a * 4 + 0]);
            atomic_add(ry, g, r[a * 4 + 1]);
            atomic_add(rz, g, r[a * 4 + 2]);
            atomic_add(rc_out, g, r[a * 4 + 3]);
        }
    }
}

static SFEM_NOINLINE void apply_residual_atomic_sumfact(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        idx_t **const SFEM_RESTRICT mesh_elems,
        const uint8_t *const SFEM_RESTRICT face_mask,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT pgx,
        const scalar_t *const SFEM_RESTRICT pgy,
        const scalar_t *const SFEM_RESTRICT pgz,
        geom_t **const SFEM_RESTRICT points,
        const scalar_t *const SFEM_RESTRICT qgx,
        const scalar_t *const SFEM_RESTRICT qgy,
        const scalar_t *const SFEM_RESTRICT qgz,
        scalar_t *const SFEM_RESTRICT rc_out,
        scalar_t *const SFEM_RESTRICT rx,
        scalar_t *const SFEM_RESTRICT ry,
        scalar_t *const SFEM_RESTRICT rz,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src,
                                              // Which optional terms are on, resolved once per
                                              // solve by the caller rather than per sweep here:
                                              // cvfem_hex8_extras_of reads the mesh, which this
                                              // kernel is not meant to name.
                                              const Hex8Extras &opt, const scalar_t rho, const scalar_t mu) {
    reset_residual(nnodes, rx, ry, rz, rc_out);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8], r[CVFEM_HEX8_N_DOF];
        gather_element_fields(mesh_elems, ux_src, uy_src, uz_src, pres, e, ux, uy, uz, p);
        Hex8ExtraScratch ex;
        ex.load(mesh_elems, points, face_mask, pgx, pgy, pgz, qgx, qgy, qgz, ux_src, uy_src, uz_src, adj_ptr, det_ptr, opt, e);
        scalar_t adj[9], det;
        load_hex8_adj(adj_ptr, det_ptr, e, adj, &det);
        cvfem_hex8_ns_upwind_residual_sumfact(rho, mu, adj, det, ux, uy, uz, p, r, ex.rc);

        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const idx_t g = mesh_elems[a][e];
            atomic_add(rx, g, r[a * 4 + 0]);
            atomic_add(ry, g, r[a * 4 + 1]);
            atomic_add(rz, g, r[a * 4 + 2]);
            atomic_add(rc_out, g, r[a * 4 + 3]);
        }
    }
}

// THE SAME SWEEP, LANE-BLOCKED. Identical arithmetic, identical scatter, vectorised.
//
// The atomic family had no SIMD path at all while the packed and coloured families both call
// cvfem_hex8_ns_upwind_residual_sumfact_simd, so "standard against packed" was a scalar-against-
// vector comparison as much as a format one. Counters at n=128 on Grace: the scalar atomic sweep
// issues 0.1% vector instructions against the packed sweep's 26.6%, and 19e9 scalar FP ops, for
// 1.95x the instructions in 2.66x the cycles.
//
// Nothing about atomics forces that. The element arithmetic vectorises here exactly as it does
// for the packed sweep, and only the scatter stays a scalar loop over lanes -- which it is in the
// packed sweep too, and for the same reason: two elements in one group may share a node, so the
// eight updates per element cannot be a vector store whatever they land in.
//
// The gather reads the GLOBAL arrays through d.elems, so this keeps every property that makes
// this the standard layout: the wide index, no pack-local staging, no reordering. What it gains
// is the vector body. That is the point -- it isolates the format from the vectorisation.
//
// This IS the atomic residual now; the scalar sweep below survives only as a verification
// reference, where it is compared against the non-sum-factorised kernel rather than against
// another layout. Measured at n=128 on Grace, bare and with Rhie-Chow: lane-blocking the atomic
// sweep alone is worth 1.29x and 1.37x, and what remains between the layouts -- 2.45x and 1.76x
// -- is the format.
static SFEM_NOINLINE void apply_residual_atomic_sumfact_simd(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        idx_t **const SFEM_RESTRICT mesh_elems,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT pgx,
        const scalar_t *const SFEM_RESTRICT pgy,
        const scalar_t *const SFEM_RESTRICT pgz,
        geom_t **const SFEM_RESTRICT points,
        scalar_t *const SFEM_RESTRICT rc_out,
        const scalar_t rhie_chow_scale,
        scalar_t *const SFEM_RESTRICT rx,
        scalar_t *const SFEM_RESTRICT ry,
        scalar_t *const SFEM_RESTRICT rz,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
                                              // Which optional terms are on, resolved once per
                                              // solve by the caller rather than per sweep here:
                                              // cvfem_hex8_extras_of reads the mesh, which this
                                              // kernel is not meant to name.
                                              const Hex8Extras &opt,
                                                             const scalar_t  rho,
                                                             const scalar_t  mu,
                                                             const scalar_t *const SFEM_RESTRICT ugrad = nullptr,
                                                             const int       limiter  = 0,
                                                             const scalar_t  venkat_c = scalar_t(0)) {
    reset_residual(nnodes, rx, ry, rz, rc_out);
    // The deferred correction, on the same sweep. The hand-written SIMD kernel takes the
    // higher-order pack, so the standard layout's higher-order arms vectorise too; the generated
    // sympy variants are packed-only, which is why this is the hand-written kernel rather than
    // the one the packed layout runs by default. --ho-scalar still reaches the scalar sweep,
    // which is required for a non-zero Venkatakrishnan eps squared.
    const bool with_ho = ugrad != nullptr;


#pragma omp parallel
    {
        alignas(ALIGN_BYTES) scalar_t cof0[CVFEM_HEX8_VEC_SIZE], cof1[CVFEM_HEX8_VEC_SIZE];
        alignas(ALIGN_BYTES) scalar_t cof2[CVFEM_HEX8_VEC_SIZE], cof3[CVFEM_HEX8_VEC_SIZE];
        alignas(ALIGN_BYTES) scalar_t cof4[CVFEM_HEX8_VEC_SIZE], cof5[CVFEM_HEX8_VEC_SIZE];
        alignas(ALIGN_BYTES) scalar_t cof6[CVFEM_HEX8_VEC_SIZE], cof7[CVFEM_HEX8_VEC_SIZE];
        alignas(ALIGN_BYTES) scalar_t cof8[CVFEM_HEX8_VEC_SIZE], detv[CVFEM_HEX8_VEC_SIZE];
        Hex8InputPack    in;
        Hex8ResidualPack outp;
        Hex8RhieChowPack rcp;
        Hex8UGradPack    hop;

#pragma omp for schedule(static)
        for (ptrdiff_t e0 = 0; e0 < nelements; e0 += CVFEM_HEX8_VEC_SIZE) {
            const int nlanes = (int)MIN((ptrdiff_t)CVFEM_HEX8_VEC_SIZE, nelements - e0);
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
                            hop.x[a][lane] = hop.y[a][lane] = hop.z[a][lane] = scalar_t(0);
                            for (int c = 0; c < 9; ++c) hop.g[a][c][lane] = scalar_t(0);
                            continue;
                        }
                        const idx_t gn = mesh_elems[a][e0 + lane];
                        hop.x[a][lane]        = scalar_t(points[0][gn]);
                        hop.y[a][lane]        = scalar_t(points[1][gn]);
                        hop.z[a][lane]        = scalar_t(points[2][gn]);
                        for (int c = 0; c < 9; ++c) hop.g[a][c][lane] = ugrad[(ptrdiff_t)gn * 9 + c];
                    }
                }
                hop.limiter  = limiter;
                hop.venkat_c = venkat_c;
            }

            if (with_ho && venkat_c == scalar_t(0)) {
                // THE SAME MICRO-KERNEL THE PACKED SWEEP RUNS. The generated variants take
                // Hex8InputPack and Hex8UGradPack -- lane packs, not pack-local storage -- so
                // nothing about them is tied to the packed layout; it was simply their only
                // caller. Using them here means the two layouts are compared on one kernel, and
                // on the faster one: they are specialised per limiter, where the hand-written
                // kernel selects inside the vector body and costs the same for every limiter.
                //
                // They assign rather than accumulate, so there is no zero-fill for them to add
                // onto, and they do not carry a non-zero Venkatakrishnan eps squared -- which is
                // why that case falls through to the hand-written kernel below rather than
                // silently dropping the term.
#define CVFEM_HEX8_SYMPY_HO_ARGS \
    rho, mu, cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, detv, in, hop
                if (opt.with_rc) {
                    switch (limiter) {
                        case 1: cvfem_hex8_ns_upwind_sympy_residual_defcor_rc_lim1_simd(
                                        CVFEM_HEX8_SYMPY_HO_ARGS, rcp, outp); break;
                        case 2: cvfem_hex8_ns_upwind_sympy_residual_defcor_rc_lim2_simd(
                                        CVFEM_HEX8_SYMPY_HO_ARGS, rcp, outp); break;
                        case 3: cvfem_hex8_ns_upwind_sympy_residual_defcor_rc_lim3_simd(
                                        CVFEM_HEX8_SYMPY_HO_ARGS, rcp, outp); break;
                        default: cvfem_hex8_ns_upwind_sympy_residual_defcor_rc_lim0_simd(
                                         CVFEM_HEX8_SYMPY_HO_ARGS, rcp, outp); break;
                    }
                } else {
                    switch (limiter) {
                        case 1: cvfem_hex8_ns_upwind_sympy_residual_defcor_lim1_simd(
                                        CVFEM_HEX8_SYMPY_HO_ARGS, outp); break;
                        case 2: cvfem_hex8_ns_upwind_sympy_residual_defcor_lim2_simd(
                                        CVFEM_HEX8_SYMPY_HO_ARGS, outp); break;
                        case 3: cvfem_hex8_ns_upwind_sympy_residual_defcor_lim3_simd(
                                        CVFEM_HEX8_SYMPY_HO_ARGS, outp); break;
                        default: cvfem_hex8_ns_upwind_sympy_residual_defcor_lim0_simd(
                                         CVFEM_HEX8_SYMPY_HO_ARGS, outp); break;
                    }
                }
#undef CVFEM_HEX8_SYMPY_HO_ARGS
            } else {
                cvfem_hex8_ns_upwind_residual_sumfact_simd(rho, mu, cof0, cof1, cof2, cof3, cof4, cof5,
                                                           cof6, cof7, cof8, detv, in, outp,
                                                           opt.with_rc ? &rcp : nullptr, rhie_chow_scale,
                                                           scalar_t(0), with_ho ? &hop : nullptr);
            }

            for (int lane = 0; lane < nlanes; ++lane) {
                const ptrdiff_t e = e0 + lane;
                for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                    const idx_t g = mesh_elems[a][e];
                    atomic_add(rx, g, outp.rx[a][lane]);
                    atomic_add(ry, g, outp.ry[a][lane]);
                    atomic_add(rz, g, outp.rz[a][lane]);
                    atomic_add(rc_out, g, outp.rc[a][lane]);
                }
            }
        }
    }
}

// The DEFERRED-CORRECTION higher-order convective flux, on the atomic sum-factored sweep.
//
// The face value is reconstructed from the donor node and its nodal velocity gradient,
// u_face = u_donor + grad u_donor . (x_scs - x_donor), and only the DIFFERENCE from the
// first-order upwind value is added. That is what makes it a deferred correction: the
// correction is residual-only and the Jacobian stays first-order, which keeps the assembled
// stencil at one ring. A reconstructed face value reaches outside the element and would
// otherwise regrow it to two.
//
// Only this sweep carries it. The packed and SIMD residual kernels take no ugrad8 argument, so
// there is no higher-order packed kernel to compare against; that is an implementation gap and
// the paper reports it as one rather than as a property of the format.
//
// `ugrad` is nine components per node, interleaved, and is NOT recomputed here: the solver lags
// the correction one Newton step, so the gradient is a hoisted input to the apply exactly as
// the Rhie-Chow state gradient is.
static SFEM_NOINLINE void apply_residual_atomic_sumfact_defcor(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        idx_t **const SFEM_RESTRICT mesh_elems,
        const uint8_t *const SFEM_RESTRICT face_mask,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT pgx,
        const scalar_t *const SFEM_RESTRICT pgy,
        const scalar_t *const SFEM_RESTRICT pgz,
        geom_t **const SFEM_RESTRICT points,
        const scalar_t *const SFEM_RESTRICT qgx,
        const scalar_t *const SFEM_RESTRICT qgy,
        const scalar_t *const SFEM_RESTRICT qgz,
        scalar_t *const SFEM_RESTRICT rc_out,
        scalar_t *const SFEM_RESTRICT rx,
        scalar_t *const SFEM_RESTRICT ry,
        scalar_t *const SFEM_RESTRICT rz,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src,
                                              // Which optional terms are on, resolved once per
                                              // solve by the caller rather than per sweep here:
                                              // cvfem_hex8_extras_of reads the mesh, which this
                                              // kernel is not meant to name.
                                              const Hex8Extras &opt,
                                                               const scalar_t  rho,
                                                               const scalar_t  mu,
                                                               const scalar_t *const SFEM_RESTRICT ugrad,
                                                               const int       limiter,
                                                               const scalar_t  venkat_c) {
    reset_residual(nnodes, rx, ry, rz, rc_out);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8], r[CVFEM_HEX8_N_DOF];
        gather_element_fields(mesh_elems, ux_src, uy_src, uz_src, pres, e, ux, uy, uz, p);
        Hex8ExtraScratch ex;
        ex.load(mesh_elems, points, face_mask, pgx, pgy, pgz, qgx, qgy, qgz, ux_src, uy_src, uz_src, adj_ptr, det_ptr, opt, e);
        scalar_t adj[9], det;
        load_hex8_adj(adj_ptr, det_ptr, e, adj, &det);

        // The reconstruction needs the element's node coordinates and the eight nodes' nodal
        // velocity gradients; both are gathered per element, like the fields above.
        scalar_t xe[8], ye[8], ze[8], g8[72];
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const idx_t g = mesh_elems[a][e];
            xe[a] = scalar_t(points[0][g]);
            ye[a] = scalar_t(points[1][g]);
            ze[a] = scalar_t(points[2][g]);
            for (int c = 0; c < 9; ++c) g8[a * 9 + c] = ugrad[(ptrdiff_t)g * 9 + c];
        }

        cvfem_hex8_ns_upwind_residual_sumfact(rho, mu, adj, det, ux, uy, uz, p, r, ex.rc,
                                              /*ueps=*/scalar_t(0), g8, xe, ye, ze,
                                              limiter, venkat_c, nullptr);

        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const idx_t g = mesh_elems[a][e];
            atomic_add(rx, g, r[a * 4 + 0]);
            atomic_add(ry, g, r[a * 4 + 1]);
            atomic_add(rz, g, r[a * 4 + 2]);
            atomic_add(rc_out, g, r[a * 4 + 3]);
        }
    }
}

static SFEM_NOINLINE void apply_residual_atomic_isoparam(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        idx_t **const SFEM_RESTRICT mesh_elems,
        const uint8_t *const SFEM_RESTRICT face_mask,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT pgx,
        const scalar_t *const SFEM_RESTRICT pgy,
        const scalar_t *const SFEM_RESTRICT pgz,
        geom_t **const SFEM_RESTRICT points,
        const scalar_t *const SFEM_RESTRICT qgx,
        const scalar_t *const SFEM_RESTRICT qgy,
        const scalar_t *const SFEM_RESTRICT qgz,
        scalar_t *const SFEM_RESTRICT rc_out,
        scalar_t *const SFEM_RESTRICT rx,
        scalar_t *const SFEM_RESTRICT ry,
        scalar_t *const SFEM_RESTRICT rz,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src,
                                              // Which optional terms are on, resolved once per
                                              // solve by the caller rather than per sweep here:
                                              // cvfem_hex8_extras_of reads the mesh, which this
                                              // kernel is not meant to name.
                                              const Hex8Extras &opt, const scalar_t rho, const scalar_t mu) {
    reset_residual(nnodes, rx, ry, rz, rc_out);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        scalar_t         ux[8], uy[8], uz[8], p[8], r[CVFEM_HEX8_N_DOF];
        Hex8ExtraScratch ex;
        ex.load(mesh_elems, points, face_mask, pgx, pgy, pgz, qgx, qgy, qgz, ux_src, uy_src, uz_src, adj_ptr, det_ptr, opt, e);
        if (!opt.with_rc && !opt.with_bnd) gather_element_coords(mesh_elems, points, e, ex.x, ex.y, ex.z);
        gather_element_fields(mesh_elems, ux_src, uy_src, uz_src, pres, e, ux, uy, uz, p);
        cvfem_hex8_ns_upwind_residual_isoparam(rho, mu, ex.x, ex.y, ex.z, ux, uy, uz, p, r, ex.rc);

        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const idx_t g = mesh_elems[a][e];
            atomic_add(rx, g, r[a * 4 + 0]);
            atomic_add(ry, g, r[a * 4 + 1]);
            atomic_add(rz, g, r[a * 4 + 2]);
            atomic_add(rc_out, g, r[a * 4 + 3]);
        }
    }
}

static SFEM_NOINLINE void apply_residual_atomic_sympy(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        idx_t **const SFEM_RESTRICT mesh_elems,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const scalar_t *const SFEM_RESTRICT pres,
        scalar_t *const SFEM_RESTRICT rc_out,
        scalar_t *const SFEM_RESTRICT rx,
        scalar_t *const SFEM_RESTRICT ry,
        scalar_t *const SFEM_RESTRICT rz,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src, const scalar_t rho, const scalar_t mu) {
    reset_residual(nnodes, rx, ry, rz, rc_out);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8], r[CVFEM_HEX8_N_DOF];
        gather_element_fields(mesh_elems, ux_src, uy_src, uz_src, pres, e, ux, uy, uz, p);
        scalar_t adj[9], det;
        load_hex8_adj(adj_ptr, det_ptr, e, adj, &det);
        cvfem_hex8_ns_upwind_sympy_residual(rho, mu, adj, det, ux, uy, uz, p, r);

        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const idx_t g = mesh_elems[a][e];
            atomic_add(rx, g, r[a * 4 + 0]);
            atomic_add(ry, g, r[a * 4 + 1]);
            atomic_add(rz, g, r[a * 4 + 2]);
            atomic_add(rc_out, g, r[a * 4 + 3]);
        }
    }
}

static SFEM_NOINLINE void assemble_jacobian_atomic_fd(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        idx_t **const SFEM_RESTRICT mesh_elems,
        const ptrdiff_t nelements,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src, 
        const idx_t *const SFEM_RESTRICT bsr_colidx,
        const count_t *const SFEM_RESTRICT slots,
        const ptrdiff_t bsr_nnz,
        const count_t *const SFEM_RESTRICT bsr_rowptr,
        scalar_t *const SFEM_RESTRICT values, const scalar_t rho, const scalar_t mu) {
    // zero_bsr4 inlined, so that this sweep names the matrix arrays it already writes rather
    // than the BSR4 object that owns them. Identical work: that function is this zeroing plus
    // the phase probe, which the macros carry here.
    CVFEM_PHASE_CLOCK(_tz);
    cvfem_zero_scalars(values, bsr_nnz * 16);
    CVFEM_PHASE_GLOBAL(_tz, PH_ZERO);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8], ke[CVFEM_HEX8_N_DOF * CVFEM_HEX8_N_DOF];
        gather_element_fields(mesh_elems, ux_src, uy_src, uz_src, pres, e, ux, uy, uz, p);
        scalar_t adj[9], det;
        load_hex8_adj(adj_ptr, det_ptr, e, adj, &det);
        cvfem_hex8_ns_upwind_jacobian_fd(rho, mu, adj, det, ux, uy, uz, p, ke);

        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const idx_t row = mesh_elems[a][e];
            for (int bnode = 0; bnode < CVFEM_HEX8_N_NODES; ++bnode) {
                const count_t slot =
                        slots ? slots[(size_t)e * 64 + a * 8 + bnode] : find_bsr_slot(bsr_rowptr, bsr_colidx, row, mesh_elems[bnode][e]);
                scalar_t *const      blk  = values + (ptrdiff_t)slot * 16;
                for (int rf = 0; rf < 4; ++rf) {
                    for (int cf = 0; cf < 4; ++cf) {
                        const scalar_t v = ke[(a * 4 + rf) * CVFEM_HEX8_N_DOF + (bnode * 4 + cf)];
                        CVFEM_ATOMIC_ADD(blk[rf * 4 + cf], v);
                    }
                }
            }
        }
    }
}

// Finite-difference Jacobian on isoparametric geometry. The affine layout has had this
// since the beginning; the isoparametric atomic path did not, so `--kernel fd --geom
// isoparam --layout atomic` silently ran the hand-written kernel and reported its speed
// under the name `fd`. The kernel it needs already existed and was already used by the
// packed layout (cvfem_hex8_best_packed.hpp), which is why that layout reported the
// honest -- and much slower, as a finite-difference Jacobian should be -- figure.
static SFEM_NOINLINE void assemble_jacobian_atomic_fd_isoparam(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        idx_t **const SFEM_RESTRICT mesh_elems,
        const ptrdiff_t nelements,
        const scalar_t *const SFEM_RESTRICT pres,
        geom_t **const SFEM_RESTRICT points,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src,
                                                               
        const idx_t *const SFEM_RESTRICT bsr_colidx,
        const count_t *const SFEM_RESTRICT slots,
        const ptrdiff_t bsr_nnz,
        const count_t *const SFEM_RESTRICT bsr_rowptr,
        scalar_t *const SFEM_RESTRICT values,
                                                               const scalar_t rho,
                                                               const scalar_t mu) {
    // zero_bsr4 inlined, so that this sweep names the matrix arrays it already writes rather
    // than the BSR4 object that owns them. Identical work: that function is this zeroing plus
    // the phase probe, which the macros carry here.
    CVFEM_PHASE_CLOCK(_tz);
    cvfem_zero_scalars(values, bsr_nnz * 16);
    CVFEM_PHASE_GLOBAL(_tz, PH_ZERO);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8];
        scalar_t ke[CVFEM_HEX8_N_DOF * CVFEM_HEX8_N_DOF];
        gather_element_coords(mesh_elems, points, e, x, y, z);
        gather_element_fields(mesh_elems, ux_src, uy_src, uz_src, pres, e, ux, uy, uz, p);
        cvfem_hex8_ns_upwind_jacobian_fd_isoparam(rho, mu, x, y, z, ux, uy, uz, p, ke);

        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const idx_t row = mesh_elems[a][e];
            for (int bnode = 0; bnode < CVFEM_HEX8_N_NODES; ++bnode) {
                const count_t slot =
                        slots ? slots[(size_t)e * 64 + a * 8 + bnode]
                              : find_bsr_slot(bsr_rowptr, bsr_colidx, row, mesh_elems[bnode][e]);
                scalar_t *const blk = values + (ptrdiff_t)slot * 16;
                for (int rf = 0; rf < 4; ++rf)
                    for (int cf = 0; cf < 4; ++cf)
                        CVFEM_ATOMIC_ADD(blk[rf * 4 + cf],
                                         ke[(a * 4 + rf) * CVFEM_HEX8_N_DOF + (bnode * 4 + cf)]);
            }
        }
    }
}

static SFEM_NOINLINE void assemble_jacobian_atomic_sympy(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        idx_t **const SFEM_RESTRICT mesh_elems,
        const ptrdiff_t nelements,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src, 
        const count_t *const SFEM_RESTRICT slots,
        const ptrdiff_t bsr_nnz,
        scalar_t *const SFEM_RESTRICT values, const scalar_t rho, const scalar_t mu) {
    // zero_bsr4 inlined, so that this sweep names the matrix arrays it already writes rather
    // than the BSR4 object that owns them. Identical work: that function is this zeroing plus
    // the phase probe, which the macros carry here.
    CVFEM_PHASE_CLOCK(_tz);
    cvfem_zero_scalars(values, bsr_nnz * 16);
    CVFEM_PHASE_GLOBAL(_tz, PH_ZERO);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8];
        gather_element_fields(mesh_elems, ux_src, uy_src, uz_src, pres, e, ux, uy, uz, p);
        scalar_t adj[9], det;
        load_hex8_adj(adj_ptr, det_ptr, e, adj, &det);
        cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots(rho, mu, adj, det, ux, uy, uz, slots + (size_t)e * 64, values);
    }
}

static SFEM_NOINLINE void assemble_jacobian_atomic_sympy_block(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        idx_t **const SFEM_RESTRICT mesh_elems,
        const ptrdiff_t nelements,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src, 
        const count_t *const SFEM_RESTRICT slots,
        const ptrdiff_t bsr_nnz,
        scalar_t *const SFEM_RESTRICT values, const scalar_t rho, const scalar_t mu) {
    // zero_bsr4 inlined, so that this sweep names the matrix arrays it already writes rather
    // than the BSR4 object that owns them. Identical work: that function is this zeroing plus
    // the phase probe, which the macros carry here.
    CVFEM_PHASE_CLOCK(_tz);
    cvfem_zero_scalars(values, bsr_nnz * 16);
    CVFEM_PHASE_GLOBAL(_tz, PH_ZERO);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8];
        gather_element_fields(mesh_elems, ux_src, uy_src, uz_src, pres, e, ux, uy, uz, p);
        scalar_t adj[9], det;
        load_hex8_adj(adj_ptr, det_ptr, e, adj, &det);
        cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots_blockwise(
                rho, mu, adj, det, ux, uy, uz, slots + (size_t)e * 64, values);
    }
}

static SFEM_NOINLINE void assemble_jacobian_atomic_sympy_row(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        idx_t **const SFEM_RESTRICT mesh_elems,
        const ptrdiff_t nelements,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src, 
        const count_t *const SFEM_RESTRICT slots,
        const ptrdiff_t bsr_nnz,
        scalar_t *const SFEM_RESTRICT values, const scalar_t rho, const scalar_t mu) {
    // zero_bsr4 inlined, so that this sweep names the matrix arrays it already writes rather
    // than the BSR4 object that owns them. Identical work: that function is this zeroing plus
    // the phase probe, which the macros carry here.
    CVFEM_PHASE_CLOCK(_tz);
    cvfem_zero_scalars(values, bsr_nnz * 16);
    CVFEM_PHASE_GLOBAL(_tz, PH_ZERO);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8];
        gather_element_fields(mesh_elems, ux_src, uy_src, uz_src, pres, e, ux, uy, uz, p);
        scalar_t adj[9], det;
        load_hex8_adj(adj_ptr, det_ptr, e, adj, &det);
        cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots_rowwise(
                rho, mu, adj, det, ux, uy, uz, slots + (size_t)e * 64, values);
    }
}

static SFEM_NOINLINE void assemble_jacobian_atomic_sympy_face(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        idx_t **const SFEM_RESTRICT mesh_elems,
        const ptrdiff_t nelements,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src, 
        const count_t *const SFEM_RESTRICT slots,
        const ptrdiff_t bsr_nnz,
        scalar_t *const SFEM_RESTRICT values, const scalar_t rho, const scalar_t mu) {
    // zero_bsr4 inlined, so that this sweep names the matrix arrays it already writes rather
    // than the BSR4 object that owns them. Identical work: that function is this zeroing plus
    // the phase probe, which the macros carry here.
    CVFEM_PHASE_CLOCK(_tz);
    cvfem_zero_scalars(values, bsr_nnz * 16);
    CVFEM_PHASE_GLOBAL(_tz, PH_ZERO);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8];
        gather_element_fields(mesh_elems, ux_src, uy_src, uz_src, pres, e, ux, uy, uz, p);
        scalar_t adj[9], det;
        load_hex8_adj(adj_ptr, det_ptr, e, adj, &det);
        cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots_facewise(
                rho, mu, adj, det, ux, uy, uz, slots + (size_t)e * 64, values);
    }
}

// Split assembly: the viscous part of the Jacobian depends only on the mesh and mu, so
// in a Newton loop it is the same matrix every iteration. Build it once, then each
// iteration restore it and add only the velocity-dependent terms.
//
// `linear` is a buffer of the same shape as b.values. The pair is exact, not an
// approximation: linear + nonlinear reproduces assemble_jacobian_atomic_sumfact
// bit-for-bit, because they are the two halves of the same kernel.
static SFEM_NOINLINE void assemble_jacobian_atomic_linear(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        const ptrdiff_t nelements,
                                                          
        const count_t *const SFEM_RESTRICT slots,
                                                          const scalar_t        mu,
                                                          scalar_t *const SFEM_RESTRICT linear) {
    // The buffer arrives sized. It used to be a std::vector& that this sweep called .assign() on,
    // which is an allocation inside a kernel -- and a kernel that allocates cannot be handed a
    // device buffer or a sub-range. The caller sizes and zeroes it.
    scalar_t *const SFEM_RESTRICT             values = linear;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        scalar_t adj[9], det;
        load_hex8_adj(adj_ptr, det_ptr, e, adj, &det);
        cvfem_hex8_ns_upwind_jacobian_add_slots_linear<true>(mu, adj, det,
                                                             slots + (size_t)e * 64, values);
    }
}

static SFEM_NOINLINE void assemble_jacobian_atomic_nonlinear(MeshData                    &d,
                                              // Which optional terms are on, resolved once per
                                              // solve by the caller rather than per sweep here:
                                              // cvfem_hex8_extras_of reads the mesh, which this
                                              // kernel is not meant to name.
                                              const Hex8Extras &opt,
                                                             BSR4                        &b,
                                                             const scalar_t               rho,
                                                             const scalar_t               mu,
                                                             const scalar_t *const SFEM_RESTRICT linear) {
    scalar_t *const SFEM_RESTRICT             values = b.values->data();
    const count_t *const SFEM_RESTRICT slots  = b.element_slots.data();

    // Restore the constant part. A streaming copy, in place of the scattered
    // accumulation it replaces.
    std::memcpy(values, linear, (size_t)b.nnz * 16 * sizeof(scalar_t));

    // Rhie-Chow belongs entirely to this half: the linear half is the viscous block, which
    // depends on the geometry and mu alone. So linear + nonlinear still reproduces the full
    // assembly with the term on, and verify_split_isoparam_vs_full_rel still proves it.

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t         ux[8], uy[8], uz[8], p[8];
        Hex8ExtraScratch ex;
        ex.load(d.elems, d.points, d.face_mask.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.qgx.data(), d.qgy.data(), d.qgz.data(), d.ux.data(), d.uy.data(), d.uz.data(), d.adj_ptr, d.det_ptr, opt, e);
        gather_element_fields(d.elems, d.ux.data(), d.uy.data(), d.uz.data(), d.p.data(), e, ux, uy, uz, p);
        scalar_t adj[9], det;
        load_hex8_adj(d.adj_ptr, d.det_ptr, e, adj, &det);
        cvfem_hex8_ns_upwind_jacobian_add_slots_nonlinear<true>(
                rho, mu, adj, det, ux, uy, uz, slots + (size_t)e * 64, values, ex.rc, p);
    }
}

static SFEM_NOINLINE void assemble_jacobian_atomic_sumfact(MeshData &d,
                                              // Which optional terms are on, resolved once per
                                              // solve by the caller rather than per sweep here:
                                              // cvfem_hex8_extras_of reads the mesh, which this
                                              // kernel is not meant to name.
                                              const Hex8Extras &opt, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    // zero_bsr4 inlined, so that this sweep names the matrix arrays it already writes rather
    // than the BSR4 object that owns them. Identical work: that function is this zeroing plus
    // the phase probe, which the macros carry here.
    CVFEM_PHASE_CLOCK(_tz);
    cvfem_zero_scalars(b.values->data(), b.nnz * 16);
    CVFEM_PHASE_GLOBAL(_tz, PH_ZERO);
    scalar_t *const SFEM_RESTRICT                 values = b.values->data();
    const count_t *const SFEM_RESTRICT     slots  = b.element_slots.data();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8];
        gather_element_fields(d.elems, d.ux.data(), d.uy.data(), d.uz.data(), d.p.data(), e, ux, uy, uz, p);
        Hex8ExtraScratch ex;
        ex.load(d.elems, d.points, d.face_mask.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.qgx.data(), d.qgy.data(), d.qgz.data(), d.ux.data(), d.uy.data(), d.uz.data(), d.adj_ptr, d.det_ptr, opt, e);
        scalar_t adj[9], det;
        load_hex8_adj(d.adj_ptr, d.det_ptr, e, adj, &det);
        // rc and p go through the same upwind switch the residual uses, so this matches
        // the matrix-free action. Without --rhie-chow the pressure-pressure block of this
        // matrix is structurally zero, which is the saddle-point structure the solver's
        // block-Jacobi cannot invert -- see cvfem_hex8_ns_core.hpp on why the benchmark's
        // assembly is a different operator from the solver's.
        cvfem_hex8_ns_upwind_jacobian_add_slots<true>(
                rho, mu, adj, det, ux, uy, uz, slots + (size_t)e * 64, values, ex.rc, p);
    }
    // The boundary closure used to be an `if (ex.fmask)` inside this loop, which is why it
    // reached this kernel and no other. It is now assemble_boundary_scs_jacobian_pass, one
    // sweep over the compacted boundary shell that every assembly entry point shares --
    // the same arrangement the residual and the Jacobian action already used.
}

static SFEM_NOINLINE void assemble_jacobian_atomic_isoparam(MeshData &d,
                                              // Which optional terms are on, resolved once per
                                              // solve by the caller rather than per sweep here:
                                              // cvfem_hex8_extras_of reads the mesh, which this
                                              // kernel is not meant to name.
                                              const Hex8Extras &opt, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    // zero_bsr4 inlined, so that this sweep names the matrix arrays it already writes rather
    // than the BSR4 object that owns them. Identical work: that function is this zeroing plus
    // the phase probe, which the macros carry here.
    CVFEM_PHASE_CLOCK(_tz);
    cvfem_zero_scalars(b.values->data(), b.nnz * 16);
    CVFEM_PHASE_GLOBAL(_tz, PH_ZERO);
    scalar_t *const SFEM_RESTRICT             values = b.values->data();
    const count_t *const SFEM_RESTRICT slots  = b.element_slots.data();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t         ux[8], uy[8], uz[8], p[8];
        Hex8ExtraScratch ex;
        ex.load(d.elems, d.points, d.face_mask.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.qgx.data(), d.qgy.data(), d.qgz.data(), d.ux.data(), d.uy.data(), d.uz.data(), d.adj_ptr, d.det_ptr, opt, e);
        // load() gathers the coordinates only when it has a reason to. This kernel always
        // needs them, so gather into the same buffers when it did not.
        if (!opt.with_rc && !opt.with_bnd) gather_element_coords(d.elems, d.points, e, ex.x, ex.y, ex.z);
        gather_element_fields(d.elems, d.ux.data(), d.uy.data(), d.uz.data(), d.p.data(), e, ux, uy, uz, p);
        cvfem_hex8_ns_upwind_jacobian_add_slots_isoparam<true>(
                rho, mu, ex.x, ex.y, ex.z, ux, uy, uz, slots + (size_t)e * 64, values, ex.rc, p);
    }
}

// Generated (CSE) kernels on isoparametric geometry. The affine SymPy kernels beat the
// hand-written ones because all twelve faces share one adjugate, so CSE has a great deal
// to factor out. Isoparametrically each face carries its own geometry and there is much
// less to share -- these exist to measure how much of the advantage survives.
static SFEM_NOINLINE void apply_residual_atomic_isoparam_sympy(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        idx_t **const SFEM_RESTRICT mesh_elems,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const scalar_t *const SFEM_RESTRICT pres,
        geom_t **const SFEM_RESTRICT points,
        scalar_t *const SFEM_RESTRICT rc_out,
        scalar_t *const SFEM_RESTRICT rx,
        scalar_t *const SFEM_RESTRICT ry,
        scalar_t *const SFEM_RESTRICT rz,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src,
                                                               const scalar_t rho,
                                                               const scalar_t mu) {
    reset_residual(nnodes, rx, ry, rz, rc_out);
#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8], r[CVFEM_HEX8_N_DOF];
        gather_element_coords(mesh_elems, points, e, x, y, z);
        gather_element_fields(mesh_elems, ux_src, uy_src, uz_src, pres, e, ux, uy, uz, p);
        cvfem_hex8_ns_upwind_sympy_residual_isoparam(rho, mu, x, y, z, ux, uy, uz, p, r);
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const idx_t g = mesh_elems[a][e];
            atomic_add(rx, g, r[a * 4 + 0]);
            atomic_add(ry, g, r[a * 4 + 1]);
            atomic_add(rz, g, r[a * 4 + 2]);
            atomic_add(rc_out, g, r[a * 4 + 3]);
        }
    }
}

static SFEM_NOINLINE void assemble_jacobian_atomic_isoparam_sympy(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        idx_t **const SFEM_RESTRICT mesh_elems,
        const ptrdiff_t nelements,
        const scalar_t *const SFEM_RESTRICT pres,
        geom_t **const SFEM_RESTRICT points,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src,
                                                                  
        const count_t *const SFEM_RESTRICT slots,
        const ptrdiff_t bsr_nnz,
        scalar_t *const SFEM_RESTRICT values,
                                                                  const scalar_t rho,
                                                                  const scalar_t mu) {
    // zero_bsr4 inlined, so that this sweep names the matrix arrays it already writes rather
    // than the BSR4 object that owns them. Identical work: that function is this zeroing plus
    // the phase probe, which the macros carry here.
    CVFEM_PHASE_CLOCK(_tz);
    cvfem_zero_scalars(values, bsr_nnz * 16);
    CVFEM_PHASE_GLOBAL(_tz, PH_ZERO);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8];
        gather_element_coords(mesh_elems, points, e, x, y, z);
        gather_element_fields(mesh_elems, ux_src, uy_src, uz_src, pres, e, ux, uy, uz, p);
        cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots_isoparam(
                rho, mu, x, y, z, ux, uy, uz, slots + (size_t)e * 64, values);
        (void)p;
    }
}

// ---------------------------------------------------------------------------
// Block diagonal only, for the block-Jacobi preconditioner.
//
// When the preconditioner is all that is being rebuilt, there is no reason to touch
// the off-diagonal blocks. Passing a slot array that is -1 everywhere except the
// diagonal makes the existing element kernel drop those writes (cvfem_hex8_bsr_acc
// returns on a negative slot), so this reuses the full element kernel rather than
// duplicating it -- the same trick the CUDA path uses.
//
// The destination is indexed by node, 16 doubles per node, not by BSR block.

// The masked slot array for one element: -1 everywhere but the diagonal, where it is the
// global node index into the node-indexed destination.
static SFEM_INLINE void diag_node_slots(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        idx_t **const SFEM_RESTRICT mesh_elems, const ptrdiff_t e, ptrdiff_t sl[64]) {
    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
        for (int b = 0; b < CVFEM_HEX8_N_NODES; ++b) sl[a * 8 + b] = -1;
        sl[a * 8 + a] = (ptrdiff_t)mesh_elems[a][e];
    }
}

// The boundary closure's contribution to the block diagonal, as a sweep over the compacted
// boundary shell -- the same arrangement assemble_boundary_scs_jacobian_pass uses for the
// full matrix, and for the same reason: the closure is a per-face term that neither
// geometry nor kernel choice changes, so it does not belong inside the element loops.
static SFEM_NOINLINE void assemble_diag_boundary_scs_pass(MeshData             &d,
                                                          const scalar_t        rho,
                                                          const scalar_t        mu,
                                                          const int             isoparam,
                                                          scalar_t *const SFEM_RESTRICT diag) {
    if (d.face_mask.empty()) return;
    cvfem_hex8_build_face_mask_eff(d);
    scalar_t *const SFEM_RESTRICT values = diag;
    const ptrdiff_t               n_bnd  = (ptrdiff_t)d.bnd_elems.size();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < n_bnd; ++i) {
        const ptrdiff_t e     = d.bnd_elems[(size_t)i];
        const int       fmask = (int)d.face_mask_eff[(size_t)e];
        ptrdiff_t       sl[64];
        scalar_t        x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8];
        diag_node_slots(d.elems, e, sl);
        gather_element_coords(d.elems, d.points, e, x, y, z);
        gather_element_fields(d.elems, d.ux.data(), d.uy.data(), d.uz.data(), d.p.data(), e, ux, uy, uz, p);
        scalar_t adj[9], det = scalar_t(0);
        if (!isoparam) load_hex8_adj(d.adj_ptr, d.det_ptr, e, adj, &det);
        if (isoparam)
            boundary_scs_add_jacobian<true, true>(rho, mu, (const scalar_t *)nullptr, det, d.Lx, d.Ly, d.Lz,
                                        x, y, z, ux, uy, uz, sl, values, fmask, 0);
        else
            boundary_scs_add_jacobian<true, false>(rho, mu, adj, det, d.Lx, d.Ly, d.Lz,
                                        x, y, z, ux, uy, uz, sl, values, fmask, 0);
        (void)p;
    }
}
// The -1 masking stays valid with Rhie-Chow and the boundary closure on, because both go
// through the same guarded accessor the interior kernel uses. It is a `ptrdiff_t` array
// deliberately: -1 has to be negative, and count_t's signedness is a build option
// (SMESH_COUNT_TYPE), so an unsigned build would turn every dropped write into an
// out-of-bounds one. boundary_scs_add_jacobian is templated on the slot type for exactly
// this reason. The solver's assemble_block_diag cannot use the trick at all -- it runs the
// generated kernel, which writes values[slot * 16 + f] without the guard.
//
// Rhie-Chow is not optional here in any meaningful sense: without it the pressure-pressure
// entry of every block is structurally zero, which is the degenerate saddle point that
// block-Jacobi cannot invert. A diagonal measured without it is a preconditioner that
// could never be used.
static SFEM_NOINLINE void assemble_diag_atomic(MeshData             &d,
                                               const scalar_t        rho,
                                               const scalar_t        mu,
                                               scalar_t *const SFEM_RESTRICT diag) {
    // The buffer arrives sized. It used to be a std::vector& that this sweep called .assign() on,
    // which is an allocation inside a kernel -- and a kernel that allocates cannot be handed a
    // device buffer or a sub-range. The caller sizes and zeroes it.
    scalar_t *const SFEM_RESTRICT values = diag;
    const Hex8Extras              opt = cvfem_hex8_extras_of(d);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        ptrdiff_t        sl[64];
        scalar_t         ux[8], uy[8], uz[8], p[8];
        Hex8ExtraScratch ex;
        ex.load(d.elems, d.points, d.face_mask.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.qgx.data(), d.qgy.data(), d.qgz.data(), d.ux.data(), d.uy.data(), d.uz.data(), d.adj_ptr, d.det_ptr, opt, e);
        diag_node_slots(d.elems, e, sl);
        gather_element_fields(d.elems, d.ux.data(), d.uy.data(), d.uz.data(), d.p.data(), e, ux, uy, uz, p);
        scalar_t adj[9], det;
        load_hex8_adj(d.adj_ptr, d.det_ptr, e, adj, &det);
        cvfem_hex8_ns_upwind_jacobian_add_slots<true>(rho, mu, adj, det, ux, uy, uz, sl, values, ex.rc, p);
    }
    assemble_diag_boundary_scs_pass(d, rho, mu, 0, diag);
}

static SFEM_NOINLINE void assemble_diag_atomic_isoparam(MeshData             &d,
                                                        const scalar_t        rho,
                                                        const scalar_t        mu,
                                                        scalar_t *const SFEM_RESTRICT diag) {
    // The buffer arrives sized. It used to be a std::vector& that this sweep called .assign() on,
    // which is an allocation inside a kernel -- and a kernel that allocates cannot be handed a
    // device buffer or a sub-range. The caller sizes and zeroes it.
    scalar_t *const SFEM_RESTRICT values = diag;
    const Hex8Extras              opt = cvfem_hex8_extras_of(d);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        ptrdiff_t        sl[64];
        scalar_t         ux[8], uy[8], uz[8], p[8];
        Hex8ExtraScratch ex;
        ex.load(d.elems, d.points, d.face_mask.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.qgx.data(), d.qgy.data(), d.qgz.data(), d.ux.data(), d.uy.data(), d.uz.data(), d.adj_ptr, d.det_ptr, opt, e);
        if (!opt.with_rc && !opt.with_bnd) gather_element_coords(d.elems, d.points, e, ex.x, ex.y, ex.z);
        diag_node_slots(d.elems, e, sl);
        gather_element_fields(d.elems, d.ux.data(), d.uy.data(), d.uz.data(), d.p.data(), e, ux, uy, uz, p);
        cvfem_hex8_ns_upwind_jacobian_add_slots_isoparam<true>(
                rho, mu, ex.x, ex.y, ex.z, ux, uy, uz, sl, values, ex.rc, p);
    }
    assemble_diag_boundary_scs_pass(d, rho, mu, 1, diag);
}

// ---------------------------------------------------------------------------
// Split assembly on isoparametric geometry.
//
// The viscous block depends on geometry and mu only, so it is constant across Newton
// iterations even though the geometry is rebuilt at each sub-control surface. The two
// halves are selected out of one kernel body by the Part parameter, so linear +
// nonlinear reproduces the full assembly by construction.
static SFEM_NOINLINE void assemble_jacobian_atomic_linear_isoparam(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        idx_t **const SFEM_RESTRICT mesh_elems,
        const ptrdiff_t nelements,
        const scalar_t *const SFEM_RESTRICT pres,
        geom_t **const SFEM_RESTRICT points,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src,
                                                                   
        const count_t *const SFEM_RESTRICT slots,
                                                                   const scalar_t        mu,
                                                                   scalar_t *const SFEM_RESTRICT linear) {
    // The buffer arrives sized. It used to be a std::vector& that this sweep called .assign() on,
    // which is an allocation inside a kernel -- and a kernel that allocates cannot be handed a
    // device buffer or a sub-range. The caller sizes and zeroes it.
    scalar_t *const SFEM_RESTRICT             values = linear;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8];
        gather_element_coords(mesh_elems, points, e, x, y, z);
        gather_element_fields(mesh_elems, ux_src, uy_src, uz_src, pres, e, ux, uy, uz, p);
        cvfem_hex8_ns_upwind_jacobian_add_slots_isoparam<true, CVFEM_HEX8_PART_LINEAR>(
                scalar_t(0), mu, x, y, z, ux, uy, uz, slots + (size_t)e * 64, values);
        (void)p;
    }
}

static SFEM_NOINLINE void assemble_jacobian_atomic_nonlinear_isoparam(
        MeshData &d,
                                              // Which optional terms are on, resolved once per
                                              // solve by the caller rather than per sweep here:
                                              // cvfem_hex8_extras_of reads the mesh, which this
                                              // kernel is not meant to name.
                                              const Hex8Extras &opt, BSR4 &b, const scalar_t rho, const scalar_t mu,
        const scalar_t *const SFEM_RESTRICT linear) {
    scalar_t *const SFEM_RESTRICT             values = b.values->data();
    const count_t *const SFEM_RESTRICT slots  = b.element_slots.data();

    // Restore the constant part, then add only what the velocity changes.
    std::memcpy(values, linear, (size_t)b.nnz * 16 * sizeof(scalar_t));


#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t         ux[8], uy[8], uz[8], p[8];
        Hex8ExtraScratch ex;
        ex.load(d.elems, d.points, d.face_mask.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.qgx.data(), d.qgy.data(), d.qgz.data(), d.ux.data(), d.uy.data(), d.uz.data(), d.adj_ptr, d.det_ptr, opt, e);
        if (!opt.with_rc && !opt.with_bnd) gather_element_coords(d.elems, d.points, e, ex.x, ex.y, ex.z);
        gather_element_fields(d.elems, d.ux.data(), d.uy.data(), d.uz.data(), d.p.data(), e, ux, uy, uz, p);
        cvfem_hex8_ns_upwind_jacobian_add_slots_isoparam<true, CVFEM_HEX8_PART_NONLINEAR>(
                rho, mu, ex.x, ex.y, ex.z, ux, uy, uz, slots + (size_t)e * 64, values, ex.rc, p);
    }
}

#endif  // CVFEM_HEX8_BEST_ATOMIC_HPP
