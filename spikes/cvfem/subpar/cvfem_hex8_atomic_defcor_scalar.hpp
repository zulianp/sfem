#pragma once

// THE ATOMIC HIGHER-ORDER RESIDUAL, SCALAR -- quarantined.
//
// DESIGN.md: "for the matrix-free kernels only the SIMD version is kept, the rest is moved to
// subpar". The standard layout's higher-order residual has a lane-blocked kernel --
// apply_residual_atomic_sumfact_simd with a non-null ugrad -- and this sweep computed the same
// operator scalar. The driver's `--ho-scalar` chose between them and is gone.
//
// Its comment claimed the scalar kernel was the faster of the two "659 against 500 MDOF/s on
// Grace (job 4812910)". That ranking has since inverted: Grace job 4981920 measures the
// lane-blocked higher-order kernel at 1059.3 MDOF/s against the scalar packed sweep's 940.6 on
// the packed layout, and the two layouts run the same micro-kernel.
//
// It was also the reference for verify_packed_ho_residual_vs_atomic_abs. That oracle now
// compares the two LANE-BLOCKED sweeps -- packed against atomic -- which is the stronger
// comparison: the staging, accumulation and scatter differ (atomics against a pack-private
// buffer with a ghost reduction) while the micro-kernel is held fixed, so an indexing or
// reduction error has nowhere to hide. Measured agreement after the rewiring: 4.8e-18 bare and
// 4.8e-18 with Rhie-Chow at limiter 0, 4.3e-18 and 5.2e-18 at limiter 2.
//
// Build with -DCVFEM_ENABLE_SUBPAR to reach it.

#include "kernels/standard/affine/cvfem_hex8_best_atomic_affine.hpp"


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
