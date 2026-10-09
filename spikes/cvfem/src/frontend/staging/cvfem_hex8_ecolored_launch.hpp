#ifndef CVFEM_HEX8_ECOLORED_LAUNCH_HPP
#define CVFEM_HEX8_ECOLORED_LAUNCH_HPP

// THE LAUNCHERS FOR THE ELEMENT-COLOURED SWEEPS.
//
// DESIGN.md splits this directory from src/kernels/: the kernels compute and the front end
// resolves. These two functions are the resolving half -- they take the mesh and the colouring,
// pull the arrays out, own the `#pragma omp parallel`, hand each thread a cvfem_range of one
// colour and barrier between colours. The kernels they call take nothing but arrays and a range.
//
// Moving them here is what lets kernels/colored/ stop including frontend/staging/cvfem_element_coloring.hpp:
// the colouring object is read here, and what crosses the boundary is the element range it
// implies.
#include "frontend/staging/cvfem_hex8_best_common.hpp"
#include "frontend/staging/cvfem_element_coloring.hpp"
#include "kernels/colored/cvfem_hex8_best_ecolored.hpp"

template <typename grad_t = scalar_t>
static SFEM_NOINLINE void apply_residual_ecolored(MeshData              &d,
                                                  const ElementColoring &ec,
                                                  const scalar_t  rho,
                                                  const scalar_t  mu,
                                                  const grad_t *const SFEM_RESTRICT ugrad = nullptr,
                                                  const int       limiter  = 0,
                                                  const scalar_t  venkat_c = scalar_t(0)) {
    reset_residual(d.nnodes, d.rx.data(), d.ry.data(), d.rz.data(), d.rc.data());
    const Hex8Extras opt = cvfem_hex8_extras_of(d);
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
        const int n_parts = cvfem_n_threads();
        const int part    = cvfem_thread_index();

        for (int color = 0; color < ec.n_colors; ++color) {
            apply_residual_ecolored_range(
                    cvfem_range_split(ec.color_ptr[(size_t)color],
                                      ec.color_ptr[(size_t)color + 1],
                                      CVFEM_HEX8_VEC_SIZE, part, n_parts),
                    d.adj_ptr, d.det_ptr, d.elems, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.rhie_chow_scale, d.ux.data(), d.uy.data(), d.uz.data(), rho, mu, ugrad, limiter, venkat_c, opt, with_ho, rx, ry, rz, rc_out,
                    cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, detv, in, outp, rcp, hop);
            // No two elements of a colour share a node, so the slices above need no
            // synchronisation between them. The next colour does: this is the barrier that used
            // to be the implicit one at the end of `#pragma omp for`.
            cvfem_thread_barrier();
        }
    }
}

template <typename grad_t = scalar_t>
static SFEM_NOINLINE void apply_jacobian_action_ecolored(MeshData              &d,
                                                         const ElementColoring &ec,
                                                         const scalar_t        rho,
                                                         const scalar_t        mu,
                                                         const scalar_t *const dir,
                                                         scalar_t *const       jv,
                                                         const grad_t *const SFEM_RESTRICT ugrad = nullptr,
                                                         const grad_t *const SFEM_RESTRICT vgrad = nullptr,
                                                         const int             limiter  = 0,
                                                         const scalar_t        venkat_c = scalar_t(0)) {
    cvfem_zero_scalars(jv, d.nnodes * CVFEM_HEX8_N_FIELDS);
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
        const int n_parts = cvfem_n_threads();
        const int part    = cvfem_thread_index();

        for (int color = 0; color < ec.n_colors; ++color) {
            apply_jacobian_action_ecolored_range(
                    cvfem_range_split(ec.color_ptr[(size_t)color],
                                      ec.color_ptr[(size_t)color + 1],
                                      CVFEM_HEX8_VEC_SIZE, part, n_parts),
                    d.adj_ptr, d.det_ptr, d.elems, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.data(), d.qgy.data(), d.qgz.data(), d.rc_coeff.data(), d.rc_w.data(), d.rhie_chow_scale, d.ux.data(), d.uy.data(), d.uz.data(), rho, mu, dir, jv, ugrad, vgrad, limiter, venkat_c, opt, has_qg, with_ho,
                    cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, detv,
                    u_pack, du_pack, outp, rcp, hop, hovp,
            cvfem_hex8_rc_config_for(d));
            cvfem_thread_barrier();
        }
    }
}

#endif  // CVFEM_HEX8_ECOLORED_LAUNCH_HPP
