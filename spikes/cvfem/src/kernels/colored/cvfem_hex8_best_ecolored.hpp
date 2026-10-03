#ifndef CVFEM_HEX8_BEST_ECOLORED_HPP
#define CVFEM_HEX8_BEST_ECOLORED_HPP

// Element-coloured layout: the flat sweep of cvfem_hex8_best_atomic.hpp with the atomics
// removed. Elements are coloured so that no two of a colour share a node, the sweep runs one
// colour at a time, and every write into the global residual / action is a plain `+=`.
//
// This is the colouring the literature means -- Reguly and Giles, deal.II's matrix-free path,
// the GPU assembly work -- and it is NOT the pack colouring of cvfem_hex8_best_colored.hpp,
// which colours packs and keeps the packed layout's staging. Pack colouring is enough on a CPU
// where a pack is one thread; element colouring is what a comparison against the published
// baseline needs, and it is the arm that says how much of the packed format's margin is the
// layout rather than the absence of atomics.
//
// The element numbering is permuted into colour order at setup, by smesh::ElementColoring with
// `modify_mesh`, so a colour is a CONTIGUOUS element range and this file is
// the atomic sweep verbatim apart from the loop bounds and the write-back. That is deliberate:
// the 16-wide lane blocking, the memcpy geometry gather and the generated micro-kernels are
// identical, so what the measurement separates is the scatter strategy and not the kernel.
//
// Two consequences worth knowing. The sweep carries the higher-order deferred correction,
// which the pack-coloured sweep does not -- it takes no ugrad argument, which is why
// jobs/dram_traffic.sbatch skips it for the -ho arms. And each node takes exactly one
// contribution per colour, in colour order, so the summation order is fixed by the colouring:
// reproducible as long as the colouring is, which it is for a fixed mesh and element order.

#include "core/cvfem_element_coloring.hpp"
#include "kernels/standard/cvfem_hex8_best_atomic.hpp"

static SFEM_NOINLINE void apply_residual_ecolored(MeshData              &d,
                                                  const ElementColoring &ec,
                                                  const scalar_t  rho,
                                                             const scalar_t  mu,
                                                             const scalar_t *const SFEM_RESTRICT ugrad = nullptr,
                                                             const int       limiter  = 0,
                                                             const scalar_t  venkat_c = scalar_t(0)) {
    reset_residual(d);
    const Hex8Extras opt(d);
    // The deferred correction, on the same sweep. The hand-written SIMD kernel takes the
    // higher-order pack, so the standard layout's higher-order arms vectorise too; the generated
    // sympy variants are packed-only, which is why this is the hand-written kernel rather than
    // the one the packed layout runs by default. --ho-scalar still reaches the scalar sweep,
    // which is required for a non-zero Venkatakrishnan eps squared.
    const bool with_ho = ugrad != nullptr;

    scalar_t *const SFEM_RESTRICT rx = d.rx.data();
    scalar_t *const SFEM_RESTRICT ry = d.ry.data();
    scalar_t *const SFEM_RESTRICT rz = d.rz.data();
    scalar_t *const SFEM_RESTRICT rc_out = d.rc.data();

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

        for (int color = 0; color < ec.n_colors; ++color) {
            const ptrdiff_t c_begin = ec.color_ptr[(size_t)color];
            const ptrdiff_t c_end   = ec.color_ptr[(size_t)color + 1];
            // The implicit barrier at the end of this `for` is the whole synchronisation: no two
            // elements of a colour share a node, so every write below is conflict-free.
#pragma omp for schedule(static)
        for (ptrdiff_t e0 = c_begin; e0 < c_end; e0 += CVFEM_HEX8_VEC_SIZE) {
            const int nlanes = (int)MIN((ptrdiff_t)CVFEM_HEX8_VEC_SIZE, c_end - e0);
            gather_hex8_adj_soa(d, e0, nlanes, cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, detv);

            for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
                if (lane < nlanes) {
                    const ptrdiff_t e = e0 + lane;
                    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                        const smesh::idx_t g = d.elems[a][e];
                        in.ux[a][lane]       = d.ux[g];
                        in.uy[a][lane]       = d.uy[g];
                        in.uz[a][lane]       = d.uz[g];
                        in.p[a][lane]        = d.p[g];
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
                const auto *const px = d.points[0];
                const auto *const py = d.points[1];
                const auto *const pz = d.points[2];
                for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
                    if (lane < nlanes) {
                        const ptrdiff_t e = e0 + lane;
                        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                            const smesh::idx_t g = d.elems[a][e];
                            rcp.pgx[a][lane]     = d.pgx[g];
                            rcp.pgy[a][lane]     = d.pgy[g];
                            rcp.pgz[a][lane]     = d.pgz[g];
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
                        const smesh::idx_t gn = d.elems[a][e0 + lane];
                        hop.x[a][lane]        = scalar_t(d.points[0][gn]);
                        hop.y[a][lane]        = scalar_t(d.points[1][gn]);
                        hop.z[a][lane]        = scalar_t(d.points[2][gn]);
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
                                                           opt.with_rc ? &rcp : nullptr, d.rhie_chow_scale,
                                                           scalar_t(0), with_ho ? &hop : nullptr);
            }

            for (int lane = 0; lane < nlanes; ++lane) {
                const ptrdiff_t e = e0 + lane;
                for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                    const smesh::idx_t g = d.elems[a][e];
                    rx[g] += outp.rx[a][lane];
                    ry[g] += outp.ry[a][lane];
                    rz[g] += outp.rz[a][lane];
                    rc_out[g] += outp.rc[a][lane];
                }
            }
        }
        }
    }
}

static SFEM_NOINLINE void apply_jacobian_action_ecolored(MeshData              &d,
                                                         const ElementColoring &ec,
                                                         const scalar_t        rho,
                                                            const scalar_t        mu,
                                                            const scalar_t *const dir,
                                                            scalar_t *const       jv,
                                                            const scalar_t *const SFEM_RESTRICT ugrad = nullptr,
                                                            const scalar_t *const SFEM_RESTRICT vgrad = nullptr,
                                                            const int             limiter  = 0,
                                                            const scalar_t        venkat_c = scalar_t(0)) {
    cvfem_zero_scalars(jv, d.nnodes * N_FIELDS);
    const Hex8Extras opt(d);
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

        for (int color = 0; color < ec.n_colors; ++color) {
            const ptrdiff_t c_begin = ec.color_ptr[(size_t)color];
            const ptrdiff_t c_end   = ec.color_ptr[(size_t)color + 1];
            // The implicit barrier at the end of this `for` is the whole synchronisation: no two
            // elements of a colour share a node, so every write below is conflict-free.
#pragma omp for schedule(static)
        for (ptrdiff_t e0 = c_begin; e0 < c_end; e0 += CVFEM_HEX8_VEC_SIZE) {
            const int nlanes = (int)MIN((ptrdiff_t)CVFEM_HEX8_VEC_SIZE, c_end - e0);
            gather_hex8_adj_soa(d, e0, nlanes, cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, detv);

            for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
                if (lane < nlanes) {
                    const ptrdiff_t e = e0 + lane;
                    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                        const smesh::idx_t                  g  = d.elems[a][e];
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
                            const smesh::idx_t g = d.elems[a][e0 + lane];
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

            if (opt.with_rc) cvfem_hex8_gather_rc_coeff(d, e0, nlanes, rcp);

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
                        const smesh::idx_t gn = d.elems[a][e0 + lane];
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
                    const smesh::idx_t g = d.elems[a][e];
                    jv[g * N_FIELDS + 0] += outp.rx[a][lane];
                    jv[g * N_FIELDS + 1] += outp.ry[a][lane];
                    jv[g * N_FIELDS + 2] += outp.rz[a][lane];
                    jv[g * N_FIELDS + 3] += outp.rc[a][lane];
                }
            }
        }
        }
    }
}

#endif  // CVFEM_HEX8_BEST_ECOLORED_HPP
