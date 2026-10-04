#pragma once

// THE ISOPARAMETRIC HEX8 ENTRY POINTS.
//
// Each sub-control volume derives its own geometry from the element's node coordinates, so these
// take x/y/z where the affine forms take an adjugate and a determinant. That is the whole of the
// difference: they call the same leaf layer -- cvfem_hex8_area_dir, cvfem_hex8_grad_at,
// cvfem_hex8_pushforward -- with the geometry of the cell in hand rather than the element's.
//
// Two of the six are lane-blocked: _residual_isoparam_simd and _jacobian_action_isoparam_simd
// are what the packed and element-coloured sweeps run, and the scalar forms are the reference
// the tests compare them against.
#include "kernels/microkernels/hex8/cvfem_hex8_ns_upwind_kernels.hpp"


template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_residual_isoparam(const scalar_t                        rho,
                                                              const scalar_t                        mu,
                                                              const scalar_t *const SFEM_RESTRICT   x,
                                                              const scalar_t *const SFEM_RESTRICT   y,
                                                              const scalar_t *const SFEM_RESTRICT   z,
                                                              const scalar_t *const SFEM_RESTRICT   ux,
                                                              const scalar_t *const SFEM_RESTRICT   uy,
                                                              const scalar_t *const SFEM_RESTRICT   uz,
                                                              const scalar_t *const SFEM_RESTRICT   p,
                                                              scalar_t *const SFEM_RESTRICT         r,
                                                              const Hex8RhieChowT<scalar_t>        &rc = {},
                                                              const scalar_t ueps = scalar_t(0)) {
    for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) r[i] = scalar_t(0);

    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        scalar_t dN[CVFEM_HEX8_N_NODES][3];
        cvfem_hex8_dn_ref<scalar_t>(CVFEM_HEX8_SCS_XI[s][0], CVFEM_HEX8_SCS_XI[s][1], CVFEM_HEX8_SCS_XI[s][2], dN);

        scalar_t adj[9], det;
        cvfem_hex8_geom_at<scalar_t>(x, y, z, CVFEM_HEX8_SCS_XI[s][0], CVFEM_HEX8_SCS_XI[s][1], CVFEM_HEX8_SCS_XI[s][2], adj, &det);

        scalar_t ax, ay, az;
        cvfem_hex8_area_dir(adj, s >> 2, ax, ay, az);

        scalar_t grad[9];
        cvfem_hex8_grad_at(adj, det, dN, ux, uy, uz, grad);

        const int i = CVFEM_HEX8_SCS[s].i;
        const int j = CVFEM_HEX8_SCS[s].j;

        scalar_t tau_x, tau_y, tau_z;
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
                            ax,
                            ay,
                            az,
                            tau_x,
                            tau_y,
                            tau_z);

        scalar_t fx, fy, fz, mdot;
        const scalar_t mdot_rc = cvfem_hex8_rhie_chow_mdotc(rho, mu, rc, i, j, ax, ay, az, p[i], p[j]);
        cvfem_hex8_scs_convection(rho, ux[i], ux[j], uy[i], uy[j], uz[i], uz[j], p[i], p[j], ax, ay, az, fx, fy, fz, mdot,
                                  mdot_rc, ueps);
        fx -= tau_x;
        fy -= tau_y;
        fz -= tau_z;

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
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_jacobian_action_isoparam(const scalar_t                        rho,
                                                                     const scalar_t                        mu,
                                                                     const scalar_t *const SFEM_RESTRICT   x,
                                                                     const scalar_t *const SFEM_RESTRICT   y,
                                                                     const scalar_t *const SFEM_RESTRICT   z,
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
                                                                     const scalar_t ueps = scalar_t(0)) {
    for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) r[i] = scalar_t(0);

    const scalar_t half = scalar_t(0.5);
    const scalar_t one  = scalar_t(1);
    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        scalar_t dN[CVFEM_HEX8_N_NODES][3];
        cvfem_hex8_dn_ref<scalar_t>(CVFEM_HEX8_SCS_XI[s][0], CVFEM_HEX8_SCS_XI[s][1], CVFEM_HEX8_SCS_XI[s][2], dN);

        scalar_t adj[9], det;
        cvfem_hex8_geom_at<scalar_t>(x, y, z, CVFEM_HEX8_SCS_XI[s][0], CVFEM_HEX8_SCS_XI[s][1], CVFEM_HEX8_SCS_XI[s][2], adj, &det);

        scalar_t ax, ay, az;
        cvfem_hex8_area_dir(adj, s >> 2, ax, ay, az);

        scalar_t dgrad[9];
        cvfem_hex8_grad_at(adj, det, dN, vx, vy, vz, dgrad);

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
                            ax,
                            ay,
                            az,
                            tx,
                            ty,
                            tz);

        const int i = CVFEM_HEX8_SCS[s].i;
        const int j = CVFEM_HEX8_SCS[s].j;
        r[i * 4 + 0] -= tx;
        r[i * 4 + 1] -= ty;
        r[i * 4 + 2] -= tz;
        r[j * 4 + 0] += tx;
        r[j * 4 + 1] += ty;
        r[j * 4 + 2] += tz;

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
        const scalar_t dmdot = rho * half * ((vx[i] + vx[j]) * ax + (vy[i] + vy[j]) * ay + (vz[i] + vz[j]) * az) +
                               cvfem_hex8_rhie_chow_dmdotc(rc_coeff, rc_corr, rc, i, j, ax, ay, az, q[i], q[j], vx, vy, vz);
        const scalar_t dpos  = d_pos * dmdot;
        const scalar_t dneg  = d_neg * dmdot;
        const scalar_t qmid  = half * (q[i] + q[j]);
        const scalar_t fx    = dpos * ux[i] + mpos * vx[i] + dneg * ux[j] + mneg * vx[j] + qmid * ax;
        const scalar_t fy    = dpos * uy[i] + mpos * vy[i] + dneg * uy[j] + mneg * vy[j] + qmid * ay;
        const scalar_t fz    = dpos * uz[i] + mpos * vz[i] + dneg * uz[j] + mneg * vz[j] + qmid * az;
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

// Part selects which half of the element matrix is written:
//   CVFEM_HEX8_PART_ALL       everything, the original behaviour
//   CVFEM_HEX8_PART_LINEAR    the viscous block only -- geometry and mu, no velocity, so
//                             it is constant across Newton iterations
//   CVFEM_HEX8_PART_NONLINEAR the convection and Rhie-Chow faces, which are what change
//
// The two halves are selected out of one body rather than derived separately, so
// LINEAR + NONLINEAR reproduces ALL by construction. Deriving them independently is
// The per-node accumulation of BOTH the Jacobian of the map and the reference-space
// field gradients, in one pass over the eight nodes. A macro and not a function because
// it accumulates into eighteen named locals the compiler must keep in registers; it is
// defined here and undefined below, which is the "localized macros" DESIGN.md allows
// src/kernels/. It moved with the kernels that use it -- the definition was left behind
// in the leaf header beside an #undef that now had nothing between them.

#define CVFEM_HEX8_ISO_NODE(a, fld)                                                              \
    do {                                                                                         \
        const scalar_t d0 = dN[a][0];                                                            \
        const scalar_t d1 = dN[a][1];                                                            \
        const scalar_t d2 = dN[a][2];                                                            \
        const scalar_t xa = xyz.x[a][lane];                                                      \
        const scalar_t ya = xyz.y[a][lane];                                                      \
        const scalar_t za = xyz.z[a][lane];                                                      \
        jx0 += xa * d0;                                                                          \
        jy0 += xa * d1;                                                                          \
        jz0 += xa * d2;                                                                          \
        jx1 += ya * d0;                                                                          \
        jy1 += ya * d1;                                                                          \
        jz1 += ya * d2;                                                                          \
        jx2 += za * d0;                                                                          \
        jy2 += za * d1;                                                                          \
        jz2 += za * d2;                                                                          \
        ur0 += fld.ux[a][lane] * d0;                                                             \
        ur1 += fld.ux[a][lane] * d1;                                                             \
        ur2 += fld.ux[a][lane] * d2;                                                             \
        vr0 += fld.uy[a][lane] * d0;                                                             \
        vr1 += fld.uy[a][lane] * d1;                                                             \
        vr2 += fld.uy[a][lane] * d2;                                                             \
        wr0 += fld.uz[a][lane] * d0;                                                             \
        wr1 += fld.uz[a][lane] * d1;                                                             \
        wr2 += fld.uz[a][lane] * d2;                                                             \
    } while (0)

// how the affine split first went wrong.
template <bool Atomic, int Part = CVFEM_HEX8_PART_ALL, typename Slot, typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_jacobian_add_slots_isoparam(const scalar_t                        rho,
                                                                        const scalar_t                        mu,
                                                                        const scalar_t *const SFEM_RESTRICT   x,
                                                                        const scalar_t *const SFEM_RESTRICT   y,
                                                                        const scalar_t *const SFEM_RESTRICT   z,
                                                                        const scalar_t *const SFEM_RESTRICT   ux,
                                                                        const scalar_t *const SFEM_RESTRICT   uy,
                                                                        const scalar_t *const SFEM_RESTRICT   uz,
                                                                        const Slot *const SFEM_RESTRICT       slots,
                                                                        scalar_t *const SFEM_RESTRICT         values,
                                                                        const Hex8RhieChowT<scalar_t>        &rc = {},
                                                                        const scalar_t *const SFEM_RESTRICT   p  = nullptr) {
    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        scalar_t dN[CVFEM_HEX8_N_NODES][3];
        cvfem_hex8_dn_ref<scalar_t>(CVFEM_HEX8_SCS_XI[s][0], CVFEM_HEX8_SCS_XI[s][1], CVFEM_HEX8_SCS_XI[s][2], dN);

        scalar_t adj[9], det;
        cvfem_hex8_geom_at<scalar_t>(x, y, z, CVFEM_HEX8_SCS_XI[s][0], CVFEM_HEX8_SCS_XI[s][1], CVFEM_HEX8_SCS_XI[s][2], adj, &det);

        scalar_t ax, ay, az;
        cvfem_hex8_area_dir(adj, s >> 2, ax, ay, az);

        const scalar_t inv_det = scalar_t(1) / det;
        scalar_t       w[CVFEM_HEX8_N_NODES][3];
        for (int k = 0; k < CVFEM_HEX8_N_NODES; ++k) {
            cvfem_hex8_pushforward(adj, inv_det, dN[k][0], dN[k][1], dN[k][2], w[k][0], w[k][1], w[k][2]);
        }

        const int i = CVFEM_HEX8_SCS[s].i;
        const int j = CVFEM_HEX8_SCS[s].j;
        if (Part != CVFEM_HEX8_PART_NONLINEAR)
        for (int k = 0; k < CVFEM_HEX8_N_NODES; ++k) {
            const scalar_t wx  = w[k][0];
            const scalar_t wy  = w[k][1];
            const scalar_t wz  = w[k][2];
            const scalar_t d00 = -(scalar_t(2) * wx * ax + wy * ay + wz * az) * mu;
            const scalar_t d01 = -(wx * ay) * mu;
            const scalar_t d02 = -(wx * az) * mu;
            const scalar_t d10 = -(wy * ax) * mu;
            const scalar_t d11 = -(wx * ax + scalar_t(2) * wy * ay + wz * az) * mu;
            const scalar_t d12 = -(wy * az) * mu;
            const scalar_t d20 = -(wz * ax) * mu;
            const scalar_t d21 = -(wz * ay) * mu;
            const scalar_t d22 = -(wx * ax + wy * ay + scalar_t(2) * wz * az) * mu;
            cvfem_hex8_bsr_acc_mom<Atomic>(values, slots[i * 8 + k], d00, d01, d02, d10, d11, d12, d20, d21, d22);
            cvfem_hex8_bsr_acc_mom<Atomic>(values,
                                           slots[j * 8 + k],
                                           -d00,
                                           -d01,
                                           -d02,
                                           -d10,
                                           -d11,
                                           -d12,
                                           -d20,
                                           -d21,
                                           -d22);
        }

        if (Part != CVFEM_HEX8_PART_LINEAR) {
            scalar_t       rc_coeff, rkx, rky, rkz;
            const scalar_t rc_corr = cvfem_hex8_rhie_chow_coeff_corr(rho, mu, rc, i, j, ax, ay, az, p, rc_coeff);
            const scalar_t mdot_rc = -rc_coeff * rc_corr;
            cvfem_hex8_rhie_chow_kvec(rc_coeff, rc, i, j, ax, ay, az, rc_corr, rkx, rky, rkz);
            cvfem_hex8_jac_conv_face<Atomic>(rho, ax, ay, az, i, j, ux, uy, uz, slots, values, mdot_rc, scalar_t(0), rkx,
                                             rky, rkz);
            cvfem_hex8_jac_rhie_chow_p<Atomic>(rho, mu, rc, ax, ay, az, i, j, ux, uy, uz, p, slots, values);
        }
    }
}

static SFEM_INLINE void cvfem_hex8_ns_upwind_residual_isoparam_simd(const scalar_t      rho_s,
                                                                   const scalar_t      mu_s,
                                                                   const Hex8CoordPack &xyz,
                                                                   const Hex8InputPack &in,
                                                                   Hex8ResidualPack    &out,
                                               const scalar_t ueps = scalar_t(0)) {
    const scalar_t rho  = rho_s;
    const scalar_t mu   = mu_s;
    const scalar_t half = scalar_t(0.5);
    const scalar_t qtr  = scalar_t(0.25);

    cvfem_hex8_zero_residual_pack(out);

    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        scalar_t dN[CVFEM_HEX8_N_NODES][3];
        cvfem_hex8_dn_ref<scalar_t>(CVFEM_HEX8_SCS_XI[s][0], CVFEM_HEX8_SCS_XI[s][1], CVFEM_HEX8_SCS_XI[s][2], dN);
        const int i = CVFEM_HEX8_SCS[s].i;
        const int j = CVFEM_HEX8_SCS[s].j;
        const int d = s >> 2;

#pragma omp simd
        for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
            scalar_t jx0 = 0, jx1 = 0, jx2 = 0;
            scalar_t jy0 = 0, jy1 = 0, jy2 = 0;
            scalar_t jz0 = 0, jz1 = 0, jz2 = 0;
            scalar_t ur0 = 0, ur1 = 0, ur2 = 0;
            scalar_t vr0 = 0, vr1 = 0, vr2 = 0;
            scalar_t wr0 = 0, wr1 = 0, wr2 = 0;
            CVFEM_HEX8_ISO_NODE(0, in);
            CVFEM_HEX8_ISO_NODE(1, in);
            CVFEM_HEX8_ISO_NODE(2, in);
            CVFEM_HEX8_ISO_NODE(3, in);
            CVFEM_HEX8_ISO_NODE(4, in);
            CVFEM_HEX8_ISO_NODE(5, in);
            CVFEM_HEX8_ISO_NODE(6, in);
            CVFEM_HEX8_ISO_NODE(7, in);

            const scalar_t c0  = jy1 * jz2 - jy2 * jz1;
            const scalar_t c1  = jy2 * jz0 - jy0 * jz2;
            const scalar_t c2  = jy0 * jz1 - jy1 * jz0;
            const scalar_t c3  = jz1 * jx2 - jz2 * jx1;
            const scalar_t c4  = jz2 * jx0 - jz0 * jx2;
            const scalar_t c5  = jz0 * jx1 - jz1 * jx0;
            const scalar_t c6  = jx1 * jy2 - jx2 * jy1;
            const scalar_t c7  = jx2 * jy0 - jx0 * jy2;
            const scalar_t c8  = jx0 * jy1 - jx1 * jy0;
            const scalar_t det = jx0 * c0 + jx1 * c1 + jx2 * c2;
            const scalar_t inv = scalar_t(1) / det;
            scalar_t       ax, ay, az;
            if (d == 0) {
                ax = qtr * c0;
                ay = qtr * c1;
                az = qtr * c2;
            } else if (d == 1) {
                ax = qtr * c3;
                ay = qtr * c4;
                az = qtr * c5;
            } else {
                ax = qtr * c6;
                ay = qtr * c7;
                az = qtr * c8;
            }

            const scalar_t g00 = (c0 * ur0 + c3 * ur1 + c6 * ur2) * inv;
            const scalar_t g01 = (c1 * ur0 + c4 * ur1 + c7 * ur2) * inv;
            const scalar_t g02 = (c2 * ur0 + c5 * ur1 + c8 * ur2) * inv;
            const scalar_t g10 = (c0 * vr0 + c3 * vr1 + c6 * vr2) * inv;
            const scalar_t g11 = (c1 * vr0 + c4 * vr1 + c7 * vr2) * inv;
            const scalar_t g12 = (c2 * vr0 + c5 * vr1 + c8 * vr2) * inv;
            const scalar_t g20 = (c0 * wr0 + c3 * wr1 + c6 * wr2) * inv;
            const scalar_t g21 = (c1 * wr0 + c4 * wr1 + c7 * wr2) * inv;
            const scalar_t g22 = (c2 * wr0 + c5 * wr1 + c8 * wr2) * inv;

            const scalar_t tau_x =
                    mu * ((scalar_t(2) * g00) * ax + (g01 + g10) * ay + (g02 + g20) * az);
            const scalar_t tau_y =
                    mu * ((g10 + g01) * ax + (scalar_t(2) * g11) * ay + (g12 + g21) * az);
            const scalar_t tau_z =
                    mu * ((g20 + g02) * ax + (g21 + g12) * ay + (scalar_t(2) * g22) * az);

            const scalar_t adv_x = half * (in.ux[i][lane] + in.ux[j][lane]);
            const scalar_t adv_y = half * (in.uy[i][lane] + in.uy[j][lane]);
            const scalar_t adv_z = half * (in.uz[i][lane] + in.uz[j][lane]);
            const scalar_t mdot  = rho * (adv_x * ax + adv_y * ay + adv_z * az);
            scalar_t amdot, sgn;
            cvfem_upwind_abs(mdot, ueps, amdot, sgn);
            const scalar_t mpos  = half * (mdot + amdot);
            const scalar_t mneg  = half * (mdot - amdot);
            const scalar_t pmid  = half * (in.p[i][lane] + in.p[j][lane]);
            const scalar_t fx    = mpos * in.ux[i][lane] + mneg * in.ux[j][lane] + pmid * ax - tau_x;
            const scalar_t fy    = mpos * in.uy[i][lane] + mneg * in.uy[j][lane] + pmid * ay - tau_y;
            const scalar_t fz    = mpos * in.uz[i][lane] + mneg * in.uz[j][lane] + pmid * az - tau_z;
            out.rx[i][lane] += fx;
            out.ry[i][lane] += fy;
            out.rz[i][lane] += fz;
            out.rc[i][lane] += mdot;
            out.rx[j][lane] -= fx;
            out.ry[j][lane] -= fy;
            out.rz[j][lane] -= fz;
            out.rc[j][lane] -= mdot;
        }
    }
}

static SFEM_INLINE void cvfem_hex8_ns_upwind_jacobian_action_isoparam_simd(const scalar_t      rho_s,
                                                                          const scalar_t      mu_s,
                                                                          const Hex8CoordPack &xyz,
                                                                          const Hex8InputPack &u,
                                                                          const Hex8InputPack &du,
                                                                          Hex8ResidualPack    &out,
                                               const scalar_t ueps = scalar_t(0)) {
    const scalar_t rho  = rho_s;
    const scalar_t mu   = mu_s;
    const scalar_t half = scalar_t(0.5);
    const scalar_t one  = scalar_t(1);
    const scalar_t qtr  = scalar_t(0.25);

    cvfem_hex8_zero_residual_pack(out);

    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        scalar_t dN[CVFEM_HEX8_N_NODES][3];
        cvfem_hex8_dn_ref<scalar_t>(CVFEM_HEX8_SCS_XI[s][0], CVFEM_HEX8_SCS_XI[s][1], CVFEM_HEX8_SCS_XI[s][2], dN);
        const int i = CVFEM_HEX8_SCS[s].i;
        const int j = CVFEM_HEX8_SCS[s].j;
        const int d = s >> 2;

#pragma omp simd
        for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
            scalar_t jx0 = 0, jx1 = 0, jx2 = 0;
            scalar_t jy0 = 0, jy1 = 0, jy2 = 0;
            scalar_t jz0 = 0, jz1 = 0, jz2 = 0;
            scalar_t ur0 = 0, ur1 = 0, ur2 = 0;
            scalar_t vr0 = 0, vr1 = 0, vr2 = 0;
            scalar_t wr0 = 0, wr1 = 0, wr2 = 0;
            CVFEM_HEX8_ISO_NODE(0, du);
            CVFEM_HEX8_ISO_NODE(1, du);
            CVFEM_HEX8_ISO_NODE(2, du);
            CVFEM_HEX8_ISO_NODE(3, du);
            CVFEM_HEX8_ISO_NODE(4, du);
            CVFEM_HEX8_ISO_NODE(5, du);
            CVFEM_HEX8_ISO_NODE(6, du);
            CVFEM_HEX8_ISO_NODE(7, du);

            const scalar_t c0  = jy1 * jz2 - jy2 * jz1;
            const scalar_t c1  = jy2 * jz0 - jy0 * jz2;
            const scalar_t c2  = jy0 * jz1 - jy1 * jz0;
            const scalar_t c3  = jz1 * jx2 - jz2 * jx1;
            const scalar_t c4  = jz2 * jx0 - jz0 * jx2;
            const scalar_t c5  = jz0 * jx1 - jz1 * jx0;
            const scalar_t c6  = jx1 * jy2 - jx2 * jy1;
            const scalar_t c7  = jx2 * jy0 - jx0 * jy2;
            const scalar_t c8  = jx0 * jy1 - jx1 * jy0;
            const scalar_t det = jx0 * c0 + jx1 * c1 + jx2 * c2;
            const scalar_t inv = scalar_t(1) / det;
            scalar_t       ax, ay, az;
            if (d == 0) {
                ax = qtr * c0;
                ay = qtr * c1;
                az = qtr * c2;
            } else if (d == 1) {
                ax = qtr * c3;
                ay = qtr * c4;
                az = qtr * c5;
            } else {
                ax = qtr * c6;
                ay = qtr * c7;
                az = qtr * c8;
            }

            const scalar_t g00 = (c0 * ur0 + c3 * ur1 + c6 * ur2) * inv;
            const scalar_t g01 = (c1 * ur0 + c4 * ur1 + c7 * ur2) * inv;
            const scalar_t g02 = (c2 * ur0 + c5 * ur1 + c8 * ur2) * inv;
            const scalar_t g10 = (c0 * vr0 + c3 * vr1 + c6 * vr2) * inv;
            const scalar_t g11 = (c1 * vr0 + c4 * vr1 + c7 * vr2) * inv;
            const scalar_t g12 = (c2 * vr0 + c5 * vr1 + c8 * vr2) * inv;
            const scalar_t g20 = (c0 * wr0 + c3 * wr1 + c6 * wr2) * inv;
            const scalar_t g21 = (c1 * wr0 + c4 * wr1 + c7 * wr2) * inv;
            const scalar_t g22 = (c2 * wr0 + c5 * wr1 + c8 * wr2) * inv;

            const scalar_t tx = mu * ((scalar_t(2) * g00) * ax + (g01 + g10) * ay + (g02 + g20) * az);
            const scalar_t ty = mu * ((g10 + g01) * ax + (scalar_t(2) * g11) * ay + (g12 + g21) * az);
            const scalar_t tz = mu * ((g20 + g02) * ax + (g21 + g12) * ay + (scalar_t(2) * g22) * az);

            const scalar_t adv_x = half * (u.ux[i][lane] + u.ux[j][lane]);
            const scalar_t adv_y = half * (u.uy[i][lane] + u.uy[j][lane]);
            const scalar_t adv_z = half * (u.uz[i][lane] + u.uz[j][lane]);
            const scalar_t mdot  = rho * (adv_x * ax + adv_y * ay + adv_z * az);
            scalar_t amdot, sgn;
            cvfem_upwind_abs(mdot, ueps, amdot, sgn);
            const scalar_t mpos  = half * (mdot + amdot);
            const scalar_t mneg  = half * (mdot - amdot);
            const scalar_t d_pos = half * (one + sgn);
            const scalar_t d_neg = half * (one - sgn);
            const scalar_t dmdot = rho * half *
                                   ((du.ux[i][lane] + du.ux[j][lane]) * ax + (du.uy[i][lane] + du.uy[j][lane]) * ay +
                                    (du.uz[i][lane] + du.uz[j][lane]) * az);
            const scalar_t dpos = d_pos * dmdot;
            const scalar_t dneg = d_neg * dmdot;
            const scalar_t qmid = half * (du.p[i][lane] + du.p[j][lane]);
            const scalar_t fx =
                    dpos * u.ux[i][lane] + mpos * du.ux[i][lane] + dneg * u.ux[j][lane] + mneg * du.ux[j][lane] + qmid * ax -
                    tx;
            const scalar_t fy =
                    dpos * u.uy[i][lane] + mpos * du.uy[i][lane] + dneg * u.uy[j][lane] + mneg * du.uy[j][lane] + qmid * ay -
                    ty;
            const scalar_t fz =
                    dpos * u.uz[i][lane] + mpos * du.uz[i][lane] + dneg * u.uz[j][lane] + mneg * du.uz[j][lane] + qmid * az -
                    tz;
            out.rx[i][lane] += fx;
            out.ry[i][lane] += fy;
            out.rz[i][lane] += fz;
            out.rc[i][lane] += dmdot;
            out.rx[j][lane] -= fx;
            out.ry[j][lane] -= fy;
            out.rz[j][lane] -= fz;
            out.rc[j][lane] -= dmdot;
        }
    }
}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_jacobian_fd_isoparam(const scalar_t                        rho,
                                                                  const scalar_t                        mu,
                                                                  const scalar_t *const SFEM_RESTRICT   x,
                                                                  const scalar_t *const SFEM_RESTRICT   y,
                                                                  const scalar_t *const SFEM_RESTRICT   z,
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
        cvfem_hex8_ns_upwind_residual_isoparam(rho, mu, x, y, z, up, vp, wp, pp, rm);

        for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) {
            const scalar_t delta = i == col ? eps : scalar_t(0);
            const int      a     = i / 4;
            const int      f     = i & 3;
            if (f == 0) up[a] = q[i] + delta;
            if (f == 1) vp[a] = q[i] + delta;
            if (f == 2) wp[a] = q[i] + delta;
            if (f == 3) pp[a] = q[i] + delta;
        }
        cvfem_hex8_ns_upwind_residual_isoparam(rho, mu, x, y, z, up, vp, wp, pp, rp);

        for (int row = 0; row < CVFEM_HEX8_N_DOF; ++row) {
            ke[row * CVFEM_HEX8_N_DOF + col] = (rp[row] - rm[row]) / (2 * eps);
        }
    }
}

#undef CVFEM_HEX8_ISO_NODE
