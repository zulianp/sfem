#pragma once

// THE AFFINE HEX8 ENTRY POINTS.
//
// Every geometric quantity here comes from ONE adjugate and ONE determinant for the whole
// element -- the caller derives them once with cvfem_hex8_affine_adj and the sweep reads them out
// of a precomputed table. That is the assumption the sum-factorised forms rest on, and it is what
// separates these from isoparametric/, where each sub-control volume has its own.
//
// The geometry-independent leaf layer they are built from -- the flux, the limiters, the upwind
// switch, the sub-control-surface faces, cvfem_hex8_area_dir, cvfem_hex8_grad_at,
// cvfem_hex8_pushforward -- stays in cvfem_hex8_ns_upwind_kernels.hpp, because the isoparametric
// kernels call the same functions with a per-cell adjugate. Separating by geometry means
// separating where the Jacobian comes from, not every function that takes one.
#include "kernels/microkernels/hex8/cvfem_hex8_ns_upwind_kernels.hpp"

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_residual(const scalar_t                        rho,
                                                      const scalar_t                        mu,
                                                      const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
                                                      const scalar_t *const SFEM_RESTRICT   ux,
                                                      const scalar_t *const SFEM_RESTRICT   uy,
                                                      const scalar_t *const SFEM_RESTRICT   uz,
                                                      const scalar_t *const SFEM_RESTRICT   p,
                                                      scalar_t *const SFEM_RESTRICT         r,
                                               const scalar_t ueps = scalar_t(0)) {
    for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) r[i] = scalar_t(0);

    scalar_t grad[9];
    cvfem_hex8_grad(adj, det, ux, uy, uz, grad);

    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        const int i = CVFEM_HEX8_SCS[s].i;
        const int j = CVFEM_HEX8_SCS[s].j;

        scalar_t ax, ay, az;
        cvfem_hex8_area_dir(adj, s >> 2, ax, ay, az);

        const scalar_t adv_x = scalar_t(0.5) * (ux[i] + ux[j]);
        const scalar_t adv_y = scalar_t(0.5) * (uy[i] + uy[j]);
        const scalar_t adv_z = scalar_t(0.5) * (uz[i] + uz[j]);
        const scalar_t mdot  = rho * (adv_x * ax + adv_y * ay + adv_z * az);
        scalar_t amdot, sgn;
        cvfem_upwind_abs(mdot, ueps, amdot, sgn);
        const scalar_t mpos  = scalar_t(0.5) * (mdot + amdot);
        const scalar_t mneg  = scalar_t(0.5) * (mdot - amdot);
        const scalar_t pmid  = scalar_t(0.5) * (p[i] + p[j]);

        const scalar_t tau_x = mu * ((2 * grad[0]) * ax + (grad[1] + grad[3]) * ay + (grad[2] + grad[6]) * az);
        const scalar_t tau_y = mu * ((grad[3] + grad[1]) * ax + (2 * grad[4]) * ay + (grad[5] + grad[7]) * az);
        const scalar_t tau_z = mu * ((grad[6] + grad[2]) * ax + (grad[7] + grad[5]) * ay + (2 * grad[8]) * az);

        const scalar_t fx = mpos * ux[i] + mneg * ux[j] + pmid * ax - tau_x;
        const scalar_t fy = mpos * uy[i] + mneg * uy[j] + pmid * ay - tau_y;
        const scalar_t fz = mpos * uz[i] + mneg * uz[j] + pmid * az - tau_z;

        r[i * 4 + 0] += fx;
        r[i * 4 + 1] += fy;
        r[i * 4 + 2] += fz;
        r[i * 4 + 3] += mdot;
        r[j * 4 + 0] -= fx;
        r[j * 4 + 1] -= fy;
        r[j * 4 + 2] -= fz;
        r[j * 4 + 3] -= mdot;
    }
}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_residual_sumfact(const scalar_t                        rho,
                                                              const scalar_t                        mu,
                                                              const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
                                                              const scalar_t *const SFEM_RESTRICT   ux,
                                                              const scalar_t *const SFEM_RESTRICT   uy,
                                                              const scalar_t *const SFEM_RESTRICT   uz,
                                                              const scalar_t *const SFEM_RESTRICT   p,
                                                              scalar_t *const SFEM_RESTRICT         r,
                                                              const Hex8RhieChowT<scalar_t>        &rc = {},
                                                              const scalar_t ueps = scalar_t(0),
                                                              // Deferred-correction inputs: the element's eight nodal velocity
                                                              // gradients and its node coordinates. All null -- the default, and
                                                              // what every caller that has not asked for the correction passes --
                                                              // leaves this kernel bit-for-bit what it was.
                                                              const scalar_t *const SFEM_RESTRICT ugrad8 = nullptr,
                                                              const scalar_t *const SFEM_RESTRICT xe = nullptr,
                                                              const scalar_t *const SFEM_RESTRICT ye = nullptr,
                                                              const scalar_t *const SFEM_RESTRICT ze = nullptr,
                                                              const int limiter = 0,
                                                              const scalar_t venkat_c = scalar_t(0),
                                                              Hex8LimiterStats *const stats = nullptr,
                                                              // Cell-Peclet blending, passed as DATA rather than read
                                                              // from the environment here: these kernels are
                                                              // SFEM_HOST_DEVICE and a function-local static with a
                                                              // lambda initialiser is not available on the device.
                                                              // The default has form 0, so eta is 1 and the kernel is
                                                              // bit-for-bit what it was.
                                                              const Hex8PecletConfig<scalar_t> &pcfg = {}) {
    for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) r[i] = scalar_t(0);

    // The twelve centroids, once for the element, only when the correction is on. Twelve triples
    // of doubles on the stack against reloading all eight node coordinates twelve times.
    scalar_t cenx[CVFEM_HEX8_N_SCS], ceny[CVFEM_HEX8_N_SCS], cenz[CVFEM_HEX8_N_SCS];
    if (ugrad8) cvfem_hex8_scs_centroids<1>(xe, ye, ze, /*off=*/0, cenx, ceny, cenz);

    scalar_t grad[9];
    cvfem_hex8_grad_sumfact(adj, det, ux, uy, uz, grad);

    scalar_t A[3][3];
    cvfem_hex8_dir_areas(adj, A);

    for (int d = 0; d < 3; ++d) {
        scalar_t tx, ty, tz;
        cvfem_hex8_traction(mu,
                            grad[0],
                            grad[1],
                            grad[2],
                            grad[3],
                            grad[4],
                            grad[5],
                            grad[6],
                            grad[7],
                            grad[8],
                            A[d][0],
                            A[d][1],
                            A[d][2],
                            tx,
                            ty,
                            tz);
        for (int e = 0; e < 4; ++e) {
            const int i = CVFEM_HEX8_DIR_EDGES[d][e][0];
            const int j = CVFEM_HEX8_DIR_EDGES[d][e][1];
            r[i * 4 + 0] -= tx;
            r[i * 4 + 1] -= ty;
            r[i * 4 + 2] -= tz;
            r[j * 4 + 0] += tx;
            r[j * 4 + 1] += ty;
            r[j * 4 + 2] += tz;
        }
    }

    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        const int i = CVFEM_HEX8_SCS[s].i;
        const int j = CVFEM_HEX8_SCS[s].j;
        const int d = s >> 2;
        scalar_t  fx, fy, fz, mdot;
        const scalar_t mdot_rc =
                cvfem_hex8_rhie_chow_mdotc(rho, mu, rc, i, j, A[d][0], A[d][1], A[d][2], p[i], p[j]);
        cvfem_hex8_scs_convection(rho,
                                  ux[i],
                                  ux[j],
                                  uy[i],
                                  uy[j],
                                  uz[i],
                                  uz[j],
                                  p[i],
                                  p[j],
                                  A[d][0],
                                  A[d][1],
                                  A[d][2],
                                  fx,
                                  fy,
                                  fz,
                                  mdot,
                                  mdot_rc,
                                  ueps,
                                  // The node separation, from the Rhie-Chow block's copy of the
                                  // element coordinates. Zero when it is absent, which makes the
                                  // Peclet number zero and the tanh blend hand back pure central
                                  // -- silently switching off all upwinding. The driver refuses
                                  // the blend unless this is available rather than relying on
                                  // that not happening.
                                  // Same edge vector the rest of this path uses: the affine column
                                  // where it is available, the coordinate difference otherwise.
                                  rc.has_ecol ? rc.ecol[0 * 3 + cvfem_hex8_edge_dir(i, j)]
                                              : (rc.x ? rc.x[j] - rc.x[i] : scalar_t(0)),
                                  rc.has_ecol ? rc.ecol[1 * 3 + cvfem_hex8_edge_dir(i, j)]
                                              : (rc.y ? rc.y[j] - rc.y[i] : scalar_t(0)),
                                  rc.has_ecol ? rc.ecol[2 * 3 + cvfem_hex8_edge_dir(i, j)]
                                              : (rc.z ? rc.z[j] - rc.z[i] : scalar_t(0)),
                                  rho > scalar_t(0) ? mu / rho : mu,
                                  rc.x ? pcfg : Hex8PecletConfig<scalar_t>{});
        // The deferred correction, added to the first-order flux and to nothing else. The
        // Jacobian below is untouched by design; see cvfem_hex8_scs_defcor.
        if (ugrad8) {
            scalar_t dfx, dfy, dfz;
            // (STRIDE=1, off=0): the flat per-element arrays this scalar sweep holds. The arm
            // is a template argument on the leaf now, so the switch is here -- which is where a
            // sweep-uniform choice belongs. This is the scalar sweep, so the switch costs a
            // predictable branch per surface rather than one inside a lane loop.
#define CVFEM_HEX8_SCS_DEFCOR_ARM(LIM_)                                                         \
    cvfem_hex8_scs_defcor<1, LIM_, true>(ugrad8, xe, ye, ze, ux, uy, uz, s, i, j, mdot, ueps,   \
                                   venkat_c, stats, /*off=*/0, cenx[s], ceny[s], cenz[s],       \
                                   dfx, dfy, dfz)
            switch (limiter) {
                case 1: CVFEM_HEX8_SCS_DEFCOR_ARM(1); break;
                case 2: CVFEM_HEX8_SCS_DEFCOR_ARM(2); break;
                case 3: CVFEM_HEX8_SCS_DEFCOR_ARM(3); break;
                default: CVFEM_HEX8_SCS_DEFCOR_ARM(0); break;
            }
#undef CVFEM_HEX8_SCS_DEFCOR_ARM
            fx += dfx;
            fy += dfy;
            fz += dfz;
        }
        r[i * 4 + 0] += fx;
        r[i * 4 + 1] += fy;
        r[i * 4 + 2] += fz;
        r[i * 4 + 3] += mdot;
        r[j * 4 + 0] -= fx;
        r[j * 4 + 1] -= fy;
        r[j * 4 + 2] -= fz;
        r[j * 4 + 3] -= mdot;
    }
}

// LIM is compile-time here too, so the atomic sweep selects the limiter once rather than at every
// surface, and so that this and the packed sweep call one instantiation of one function. It leads
// the list because it is given explicitly while scalar_t is deduced.
template <int LIM, typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_jacobian_action(const scalar_t                        rho,
                                                             const scalar_t                        mu,
                                                             const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
                                                             const scalar_t *const SFEM_RESTRICT   ux,
                                                             const scalar_t *const SFEM_RESTRICT   uy,
                                                             const scalar_t *const SFEM_RESTRICT   uz,
                                                             const scalar_t *const SFEM_RESTRICT   vx,
                                                             const scalar_t *const SFEM_RESTRICT   vy,
                                                             const scalar_t *const SFEM_RESTRICT   vz,
                                                             const scalar_t *const SFEM_RESTRICT   q,
                                                             scalar_t *const SFEM_RESTRICT         r,
                                                             const Hex8RhieChowT<scalar_t>        &rc = {},
                                                             const scalar_t *const SFEM_RESTRICT   p  = nullptr,
                                                             const scalar_t ueps = scalar_t(0),
                                                             // The exact higher-order action, as on
                                                             // the packed sweep: the element's node
                                                             // coordinates, the state's nodal velocity
                                                             // gradient and the direction's. All null
                                                             // is the lagged action, which is what
                                                             // this kernel computed before they existed.
                                                             const scalar_t *const SFEM_RESTRICT xe = nullptr,
                                                             const scalar_t *const SFEM_RESTRICT ye = nullptr,
                                                             const scalar_t *const SFEM_RESTRICT ze = nullptr,
                                                             const scalar_t *const SFEM_RESTRICT ugrad8 = nullptr,
                                                             const scalar_t *const SFEM_RESTRICT vgrad8 = nullptr,
                                                             const scalar_t venkat_c = scalar_t(0)) {
    for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) r[i] = scalar_t(0);

    // The twelve centroids, once, exactly as the scalar residual builds them when it carries the
    // correction.
    scalar_t cenx[CVFEM_HEX8_N_SCS], ceny[CVFEM_HEX8_N_SCS], cenz[CVFEM_HEX8_N_SCS];
    if (ugrad8 && vgrad8) cvfem_hex8_scs_centroids<1>(xe, ye, ze, /*off=*/0, cenx, ceny, cenz);

    scalar_t dgrad[9];
    cvfem_hex8_grad_sumfact(adj, det, vx, vy, vz, dgrad);

    scalar_t A[3][3];
    cvfem_hex8_dir_areas(adj, A);

    for (int d = 0; d < 3; ++d) {
        scalar_t tx, ty, tz;
        cvfem_hex8_traction(mu,
                            dgrad[0],
                            dgrad[1],
                            dgrad[2],
                            dgrad[3],
                            dgrad[4],
                            dgrad[5],
                            dgrad[6],
                            dgrad[7],
                            dgrad[8],
                            A[d][0],
                            A[d][1],
                            A[d][2],
                            tx,
                            ty,
                            tz);
        for (int e = 0; e < 4; ++e) {
            const int i = CVFEM_HEX8_DIR_EDGES[d][e][0];
            const int j = CVFEM_HEX8_DIR_EDGES[d][e][1];
            r[i * 4 + 0] -= tx;
            r[i * 4 + 1] -= ty;
            r[i * 4 + 2] -= tz;
            r[j * 4 + 0] += tx;
            r[j * 4 + 1] += ty;
            r[j * 4 + 2] += tz;
        }
    }

    const scalar_t half = scalar_t(0.5);
    const scalar_t one  = scalar_t(1);
    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        const int      i     = CVFEM_HEX8_SCS[s].i;
        const int      j     = CVFEM_HEX8_SCS[s].j;
        const int      d     = s >> 2;
        const scalar_t ax    = A[d][0];
        const scalar_t ay    = A[d][1];
        const scalar_t az    = A[d][2];
        const scalar_t adv_x = half * (ux[i] + ux[j]);
        const scalar_t adv_y = half * (uy[i] + uy[j]);
        const scalar_t adv_z = half * (uz[i] + uz[j]);
        scalar_t       rc_coeff;
        const scalar_t rc_corr = cvfem_hex8_rhie_chow_coeff_corr(rho, mu, rc, i, j, ax, ay, az, p, rc_coeff);
        const scalar_t mdot_rc = -rc_coeff * rc_corr;
        const scalar_t mdot    = rho * (adv_x * ax + adv_y * ay + adv_z * az) + mdot_rc;
        scalar_t amdot, sgn;
        cvfem_upwind_abs(mdot, ueps, amdot, sgn);
        const scalar_t mpos  = half * (mdot + amdot);
        const scalar_t mneg  = half * (mdot - amdot);
        const scalar_t d_pos = half * (one + sgn);
        const scalar_t d_neg = half * (one - sgn);
        const scalar_t dmdot =
                rho * half * ((vx[i] + vx[j]) * ax + (vy[i] + vy[j]) * ay + (vz[i] + vz[j]) * az) +
                cvfem_hex8_rhie_chow_dmdotc(rc_coeff, rc_corr, rc, i, j, ax, ay, az, q[i], q[j], vx, vy, vz);
        const scalar_t dpos  = d_pos * dmdot;
        const scalar_t dneg  = d_neg * dmdot;
        const scalar_t qmid  = half * (q[i] + q[j]);
        scalar_t fx    = dpos * ux[i] + mpos * vx[i] + dneg * ux[j] + mneg * vx[j] + qmid * ax;
        scalar_t fy    = dpos * uy[i] + mpos * vy[i] + dneg * uy[j] + mneg * vy[j] + qmid * ay;
        scalar_t fz    = dpos * uz[i] + mpos * vz[i] + dneg * uz[j] + mneg * vz[j] + qmid * az;
        // The correction's derivative, from the same function the packed sweep calls, so the two
        // layouts cannot carry different higher-order terms.
        if (ugrad8 && vgrad8) {
            scalar_t hx, hy, hz;
            cvfem_hex8_scs_defcor_jv<1, LIM>(ugrad8, vgrad8, xe, ye, ze, ux, uy, uz, vx, vy, vz,
                                             s, i, j, mdot, dmdot, ueps, venkat_c, /*off=*/0,
                                             cenx[s], ceny[s], cenz[s], hx, hy, hz);
            fx += hx; fy += hy; fz += hz;
        }
        r[i * 4 + 0] += fx;
        r[i * 4 + 1] += fy;
        r[i * 4 + 2] += fz;
        r[i * 4 + 3] += dmdot;
        r[j * 4 + 0] -= fx;
        r[j * 4 + 1] -= fy;
        r[j * 4 + 2] -= fz;
        r[j * 4 + 3] -= dmdot;
    }
}

// Viscous half restricted to one side of that split. RECOMPUTED == true emits only the
// pairs that get rebuilt every iteration; false emits only the pairs written once.
// Viscous half, split by entry rather than by block.
//
// RECOMPUTED == false emits every momentum entry the convection will NOT overwrite --
// written once at setup and never touched again. RECOMPUTED == true emits exactly the
// entries it will, which are the ones that get zeroed and rebuilt each iteration.
// Together they reproduce the full viscous contribution exactly.
// The viscous entries the mask does NOT cover: written once at setup, never rebuilt.
template <bool Atomic, typename Slot, typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_jacobian_add_slots_static(
        const scalar_t mu,
        const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
        const Slot *const SFEM_RESTRICT slots,
        const uint16_t *const SFEM_RESTRICT block_mask,
        scalar_t *const SFEM_RESTRICT values) {
    scalar_t A[3][3];
    cvfem_hex8_dir_areas(adj, A);
    scalar_t w[CVFEM_HEX8_N_NODES][3];
    const scalar_t inv_det = scalar_t(1) / det;
    for (int k = 0; k < CVFEM_HEX8_N_NODES; ++k)
        cvfem_hex8_pushforward<scalar_t>(adj, inv_det, CVFEM_HEX8_DN_REF[k][0],
                                         CVFEM_HEX8_DN_REF[k][1], CVFEM_HEX8_DN_REF[k][2],
                                         w[k][0], w[k][1], w[k][2]);
    scalar_t Anet[CVFEM_HEX8_N_NODES][3];
    for (int i = 0; i < CVFEM_HEX8_N_NODES; ++i)
        for (int c = 0; c < 3; ++c)
            Anet[i][c] = CVFEM_HEX8_SNET[i][0] * A[0][c] + CVFEM_HEX8_SNET[i][1] * A[1][c] +
                         CVFEM_HEX8_SNET[i][2] * A[2][c];

    for (int i = 0; i < CVFEM_HEX8_N_NODES; ++i) {
        const scalar_t Ax = Anet[i][0], Ay = Anet[i][1], Az = Anet[i][2];
        for (int k = 0; k < CVFEM_HEX8_N_NODES; ++k) {
            const Slot slot = slots[i * 8 + k];
            const uint16_t m = block_mask[(ptrdiff_t)slot];
            if (m == 0xFFFFu) continue;                 // nothing survives here
            const scalar_t wx = w[k][0], wy = w[k][1], wz = w[k][2];
            const scalar_t dm[3][3] = {
                {-(scalar_t(2) * wx * Ax + wy * Ay + wz * Az) * mu, -(wx * Ay) * mu, -(wx * Az) * mu},
                {-(wy * Ax) * mu, -(wx * Ax + scalar_t(2) * wy * Ay + wz * Az) * mu, -(wy * Az) * mu},
                {-(wz * Ax) * mu, -(wz * Ay) * mu,
                 -(wx * Ax + wy * Ay + scalar_t(2) * wz * Az) * mu}};
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    if (!(m & (uint16_t)(1u << (r * 4 + c))))
                        cvfem_hex8_bsr_acc<Atomic>(values, slot, r, c, dm[r][c]);
        }
    }
}

// Everything that is rebuilt each Newton iteration, in one pass.
//
// Fuses the recomputed viscous pairs with the convection so the element geometry -- the
// area tensor A, the pushed-forward gradients w, and Anet -- is built once instead of
// once per half. Iterates the 32 recomputed pairs directly rather than testing all 64.
template <bool Atomic, typename Slot, typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_jacobian_add_slots_dynamic(
        const scalar_t rho, const scalar_t mu,
        const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        const Slot *const SFEM_RESTRICT slots,
        const uint16_t *const SFEM_RESTRICT block_mask,
        scalar_t *const SFEM_RESTRICT values,
        const Hex8RhieChowT<scalar_t> &rc = {},
        const scalar_t *const SFEM_RESTRICT p = nullptr) {
    scalar_t A[3][3];
    cvfem_hex8_dir_areas(adj, A);

    scalar_t w[CVFEM_HEX8_N_NODES][3];
    const scalar_t inv_det = scalar_t(1) / det;
    for (int k = 0; k < CVFEM_HEX8_N_NODES; ++k)
        cvfem_hex8_pushforward<scalar_t>(adj, inv_det, CVFEM_HEX8_DN_REF[k][0],
                                         CVFEM_HEX8_DN_REF[k][1], CVFEM_HEX8_DN_REF[k][2],
                                         w[k][0], w[k][1], w[k][2]);
    scalar_t Anet[CVFEM_HEX8_N_NODES][3];
    for (int i = 0; i < CVFEM_HEX8_N_NODES; ++i)
        for (int c = 0; c < 3; ++c)
            Anet[i][c] = CVFEM_HEX8_SNET[i][0] * A[0][c] + CVFEM_HEX8_SNET[i][1] * A[1][c] +
                         CVFEM_HEX8_SNET[i][2] * A[2][c];

    // Rebuild exactly the entries the zeroing cleared -- read from the same mask, so the
    // two can never disagree. Entries outside it keep the viscous value written at setup.
    for (int t = 0; t < CVFEM_HEX8_N_RECOMPUTED_PAIRS; ++t) {
        const int i = CVFEM_HEX8_RECOMPUTED_PAIRS[t][0];
        const int k = CVFEM_HEX8_RECOMPUTED_PAIRS[t][1];
        const Slot slot = slots[i * 8 + k];
        const uint16_t m = block_mask[(ptrdiff_t)slot];
        if (!m) continue;
        const scalar_t Ax = Anet[i][0], Ay = Anet[i][1], Az = Anet[i][2];
        const scalar_t wx = w[k][0], wy = w[k][1], wz = w[k][2];
        const scalar_t dm[3][3] = {
            {-(scalar_t(2) * wx * Ax + wy * Ay + wz * Az) * mu, -(wx * Ay) * mu, -(wx * Az) * mu},
            {-(wy * Ax) * mu, -(wx * Ax + scalar_t(2) * wy * Ay + wz * Az) * mu, -(wy * Az) * mu},
            {-(wz * Ax) * mu, -(wz * Ay) * mu,
             -(wx * Ax + wy * Ay + scalar_t(2) * wz * Az) * mu}};
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                if (m & (uint16_t)(1u << (r * 4 + c)))
                    cvfem_hex8_bsr_acc<Atomic>(values, slot, r, c, dm[r][c]);
    }

    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        const int d = s >> 2;
        const int i = CVFEM_HEX8_SCS[s].i;
        const int j = CVFEM_HEX8_SCS[s].j;
        scalar_t       rc_coeff, rkx, rky, rkz;
        const scalar_t rc_corr = cvfem_hex8_rhie_chow_coeff_corr(rho, mu, rc, i, j, A[d][0], A[d][1], A[d][2], p, rc_coeff);
        const scalar_t mdot_rc = -rc_coeff * rc_corr;
        cvfem_hex8_rhie_chow_kvec(rc_coeff, rc, i, j, A[d][0], A[d][1], A[d][2], rc_corr, rkx, rky, rkz);
        cvfem_hex8_jac_conv_face<Atomic>(rho, A[d][0], A[d][1], A[d][2], i, j, ux, uy, uz,
                                         slots, values, mdot_rc, scalar_t(0), rkx, rky, rkz);
        cvfem_hex8_jac_rhie_chow_p<Atomic>(rho, mu, rc, A[d][0], A[d][1], A[d][2], i, j,
                                           ux, uy, uz, p, slots, values);
    }
}

// Geometry-only half. No velocity argument at all -- that is the point.
template <bool Atomic, typename Slot, typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_jacobian_add_slots_linear(
        const scalar_t mu,
        const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
        const Slot *const SFEM_RESTRICT slots,
        scalar_t *const SFEM_RESTRICT values) {
    scalar_t A[3][3];
    cvfem_hex8_dir_areas(adj, A);

    scalar_t w[CVFEM_HEX8_N_NODES][3];
    const scalar_t inv_det = scalar_t(1) / det;
    for (int k = 0; k < CVFEM_HEX8_N_NODES; ++k) {
        cvfem_hex8_pushforward<scalar_t>(adj,
                               inv_det,
                               CVFEM_HEX8_DN_REF[k][0],
                               CVFEM_HEX8_DN_REF[k][1],
                               CVFEM_HEX8_DN_REF[k][2],
                               w[k][0],
                               w[k][1],
                               w[k][2]);
    }

    scalar_t Anet[CVFEM_HEX8_N_NODES][3];
    for (int i = 0; i < CVFEM_HEX8_N_NODES; ++i) {
        Anet[i][0] = CVFEM_HEX8_SNET[i][0] * A[0][0] + CVFEM_HEX8_SNET[i][1] * A[1][0] +
                     CVFEM_HEX8_SNET[i][2] * A[2][0];
        Anet[i][1] = CVFEM_HEX8_SNET[i][0] * A[0][1] + CVFEM_HEX8_SNET[i][1] * A[1][1] +
                     CVFEM_HEX8_SNET[i][2] * A[2][1];
        Anet[i][2] = CVFEM_HEX8_SNET[i][0] * A[0][2] + CVFEM_HEX8_SNET[i][1] * A[1][2] +
                     CVFEM_HEX8_SNET[i][2] * A[2][2];
    }

    for (int i = 0; i < CVFEM_HEX8_N_NODES; ++i) {
        const scalar_t Ax = Anet[i][0];
        const scalar_t Ay = Anet[i][1];
        const scalar_t Az = Anet[i][2];
        for (int k = 0; k < CVFEM_HEX8_N_NODES; ++k) {
            const scalar_t wx   = w[k][0];
            const scalar_t wy   = w[k][1];
            const scalar_t wz   = w[k][2];
            const scalar_t d00  = -(scalar_t(2) * wx * Ax + wy * Ay + wz * Az) * mu;
            const scalar_t d01  = -(wx * Ay) * mu;
            const scalar_t d02  = -(wx * Az) * mu;
            const scalar_t d10  = -(wy * Ax) * mu;
            const scalar_t d11  = -(wx * Ax + scalar_t(2) * wy * Ay + wz * Az) * mu;
            const scalar_t d12  = -(wy * Az) * mu;
            const scalar_t d20  = -(wz * Ax) * mu;
            const scalar_t d21  = -(wz * Ay) * mu;
            const scalar_t d22  = -(wx * Ax + wy * Ay + scalar_t(2) * wz * Az) * mu;
            const Slot     slot = slots[i * 8 + k];
            cvfem_hex8_bsr_acc_mom<Atomic>(values, slot, d00, d01, d02, d10, d11, d12, d20, d21, d22);
        }
    }

}

// Velocity-dependent half: convection across the 12 sub-control surfaces, plus the
// Rhie-Chow pressure coupling when it is active.
template <bool Atomic, typename Slot, typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_jacobian_add_slots_nonlinear(
        const scalar_t rho, const scalar_t mu,
        const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        const Slot *const SFEM_RESTRICT slots,
        scalar_t *const SFEM_RESTRICT values,
        const Hex8RhieChowT<scalar_t> &rc = {},
        const scalar_t *const SFEM_RESTRICT p = nullptr) {
    scalar_t A[3][3];
    cvfem_hex8_dir_areas(adj, A);
    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        const int      d       = s >> 2;
        const int      i       = CVFEM_HEX8_SCS[s].i;
        const int      j       = CVFEM_HEX8_SCS[s].j;
        scalar_t       rc_coeff, rkx, rky, rkz;
        const scalar_t rc_corr = cvfem_hex8_rhie_chow_coeff_corr(rho, mu, rc, i, j, A[d][0], A[d][1], A[d][2], p, rc_coeff);
        const scalar_t mdot_rc = -rc_coeff * rc_corr;
        cvfem_hex8_rhie_chow_kvec(rc_coeff, rc, i, j, A[d][0], A[d][1], A[d][2], rc_corr, rkx, rky, rkz);
        cvfem_hex8_jac_conv_face<Atomic>(rho, A[d][0], A[d][1], A[d][2], i, j, ux, uy, uz, slots, values, mdot_rc,
                                         scalar_t(0), rkx, rky, rkz);
        cvfem_hex8_jac_rhie_chow_p<Atomic>(rho, mu, rc, A[d][0], A[d][1], A[d][2], i, j, ux, uy, uz, p, slots, values);
    }
}

template <bool Atomic, typename Slot, typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_jacobian_add_slots(const scalar_t                        rho,
                                                                const scalar_t                        mu,
                                                                const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
                                                                const scalar_t *const SFEM_RESTRICT   ux,
                                                                const scalar_t *const SFEM_RESTRICT   uy,
                                                                const scalar_t *const SFEM_RESTRICT   uz,
                                                                const Slot *const SFEM_RESTRICT       slots,
                                                                scalar_t *const SFEM_RESTRICT         values,
                                                                const Hex8RhieChowT<scalar_t>        &rc = {},
                                                                const scalar_t *const SFEM_RESTRICT   p  = nullptr) {
    scalar_t A[3][3];
    cvfem_hex8_dir_areas(adj, A);

    scalar_t w[CVFEM_HEX8_N_NODES][3];
    const scalar_t inv_det = scalar_t(1) / det;
    for (int k = 0; k < CVFEM_HEX8_N_NODES; ++k) {
        cvfem_hex8_pushforward<scalar_t>(adj,
                               inv_det,
                               CVFEM_HEX8_DN_REF[k][0],
                               CVFEM_HEX8_DN_REF[k][1],
                               CVFEM_HEX8_DN_REF[k][2],
                               w[k][0],
                               w[k][1],
                               w[k][2]);
    }

    scalar_t Anet[CVFEM_HEX8_N_NODES][3];
    for (int i = 0; i < CVFEM_HEX8_N_NODES; ++i) {
        Anet[i][0] = CVFEM_HEX8_SNET[i][0] * A[0][0] + CVFEM_HEX8_SNET[i][1] * A[1][0] +
                     CVFEM_HEX8_SNET[i][2] * A[2][0];
        Anet[i][1] = CVFEM_HEX8_SNET[i][0] * A[0][1] + CVFEM_HEX8_SNET[i][1] * A[1][1] +
                     CVFEM_HEX8_SNET[i][2] * A[2][1];
        Anet[i][2] = CVFEM_HEX8_SNET[i][0] * A[0][2] + CVFEM_HEX8_SNET[i][1] * A[1][2] +
                     CVFEM_HEX8_SNET[i][2] * A[2][2];
    }

    for (int i = 0; i < CVFEM_HEX8_N_NODES; ++i) {
        const scalar_t Ax = Anet[i][0];
        const scalar_t Ay = Anet[i][1];
        const scalar_t Az = Anet[i][2];
        for (int k = 0; k < CVFEM_HEX8_N_NODES; ++k) {
            const scalar_t wx   = w[k][0];
            const scalar_t wy   = w[k][1];
            const scalar_t wz   = w[k][2];
            const scalar_t d00  = -(scalar_t(2) * wx * Ax + wy * Ay + wz * Az) * mu;
            const scalar_t d01  = -(wx * Ay) * mu;
            const scalar_t d02  = -(wx * Az) * mu;
            const scalar_t d10  = -(wy * Ax) * mu;
            const scalar_t d11  = -(wx * Ax + scalar_t(2) * wy * Ay + wz * Az) * mu;
            const scalar_t d12  = -(wy * Az) * mu;
            const scalar_t d20  = -(wz * Ax) * mu;
            const scalar_t d21  = -(wz * Ay) * mu;
            const scalar_t d22  = -(wx * Ax + wy * Ay + scalar_t(2) * wz * Az) * mu;
            const Slot     slot = slots[i * 8 + k];
            cvfem_hex8_bsr_acc_mom<Atomic>(values, slot, d00, d01, d02, d10, d11, d12, d20, d21, d22);
        }
    }

    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        const int      d       = s >> 2;
        const int      i       = CVFEM_HEX8_SCS[s].i;
        const int      j       = CVFEM_HEX8_SCS[s].j;
        scalar_t       rc_coeff, rkx, rky, rkz;
        const scalar_t rc_corr = cvfem_hex8_rhie_chow_coeff_corr(rho, mu, rc, i, j, A[d][0], A[d][1], A[d][2], p, rc_coeff);
        const scalar_t mdot_rc = -rc_coeff * rc_corr;
        cvfem_hex8_rhie_chow_kvec(rc_coeff, rc, i, j, A[d][0], A[d][1], A[d][2], rc_corr, rkx, rky, rkz);
        cvfem_hex8_jac_conv_face<Atomic>(rho, A[d][0], A[d][1], A[d][2], i, j, ux, uy, uz, slots, values, mdot_rc,
                                         scalar_t(0), rkx, rky, rkz);
        cvfem_hex8_jac_rhie_chow_p<Atomic>(rho, mu, rc, A[d][0], A[d][1], A[d][2], i, j, ux, uy, uz, p, slots, values);
    }
}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_jacobian_fd(const scalar_t                        rho,
                                                         const scalar_t                        mu,
                                                         const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
                                                         const scalar_t *const SFEM_RESTRICT   ux,
                                                         const scalar_t *const SFEM_RESTRICT   uy,
                                                         const scalar_t *const SFEM_RESTRICT   uz,
                                                         const scalar_t *const SFEM_RESTRICT   p,
                                                         scalar_t *const SFEM_RESTRICT         ke) {
    scalar_t q[CVFEM_HEX8_N_DOF];
    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
        q[a * 4 + 0] = ux[a];
        q[a * 4 + 1] = uy[a];
        q[a * 4 + 2] = uz[a];
        q[a * 4 + 3] = p[a];
    }

    scalar_t       up[8], vp[8], wp[8], pp[8];
    scalar_t       rm[CVFEM_HEX8_N_DOF], rp[CVFEM_HEX8_N_DOF];
    const scalar_t eps = scalar_t(1.0e-6);

    for (int col = 0; col < CVFEM_HEX8_N_DOF; ++col) {
        for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) {
            const scalar_t delta = i == col ? eps : scalar_t(0);
            const int      a     = i / 4;
            const int      f     = i & 3;
            if (f == 0) up[a] = q[i] - delta;
            if (f == 1) vp[a] = q[i] - delta;
            if (f == 2) wp[a] = q[i] - delta;
            if (f == 3) pp[a] = q[i] - delta;
        }
        cvfem_hex8_ns_upwind_residual(rho, mu, adj, det, up, vp, wp, pp, rm);

        for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) {
            const scalar_t delta = i == col ? eps : scalar_t(0);
            const int      a     = i / 4;
            const int      f     = i & 3;
            if (f == 0) up[a] = q[i] + delta;
            if (f == 1) vp[a] = q[i] + delta;
            if (f == 2) wp[a] = q[i] + delta;
            if (f == 3) pp[a] = q[i] + delta;
        }
        cvfem_hex8_ns_upwind_residual(rho, mu, adj, det, up, vp, wp, pp, rp);

        for (int row = 0; row < CVFEM_HEX8_N_DOF; ++row) {
            ke[row * CVFEM_HEX8_N_DOF + col] = (rp[row] - rm[row]) / (2 * eps);
        }
    }
}
