#ifndef CVFEM_HEX8_BOUNDARY_SCS_HPP
#define CVFEM_HEX8_BOUNDARY_SCS_HPP

// Boundary sub-control-surface terms for the CVFEM HEX8 Navier-Stokes operators.
//
// Lifted verbatim out of cvfem_hex8_ns_steady.cpp so the CUDA kernels can call the same
// code the solver does, rather than a second copy of it. Templated on the scalar type
// and marked device-callable, exactly as the volume kernels were.
//
// Not self-contained: the includer must already provide scalar_t, SFEM_RESTRICT and the
// CVFEM HEX8 volume kernels (cvfem_hex8_grad_sumfact, cvfem_hex8_dir_areas, ...).

#include "cvfem_portability.hpp"

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE bool on_plane(const scalar_t c, const scalar_t value, const scalar_t L) {
    const scalar_t tol = scalar_t(1e-8) * std::max(L, scalar_t(1));
    return std::fabs(c - value) <= tol;
}

#define CVFEM_HEX8_BFACE_NODES_INIT {{0, 3, 7, 4}, \
                                                     {1, 2, 6, 5}, \
                                                     {0, 1, 5, 4}, \
                                                     {3, 2, 6, 7}, \
                                                     {0, 1, 2, 3}, \
                                                     {4, 5, 6, 7}}
#if defined(__CUDACC__)
static SFEM_INLINE SFEM_HOST_DEVICE const int (&cvfem_hex8_bface_nodes_tbl())[6][4] {
    static constexpr int t[6][4] = CVFEM_HEX8_BFACE_NODES_INIT;
    return t;
}
#define CVFEM_HEX8_BFACE_NODES cvfem_hex8_bface_nodes_tbl()
#else
static constexpr int CVFEM_HEX8_BFACE_NODES[6][4] = CVFEM_HEX8_BFACE_NODES_INIT;
#endif
#define CVFEM_HEX8_BFACE_AXIS_INIT {0, 0, 1, 1, 2, 2}
#if defined(__CUDACC__)
static SFEM_INLINE SFEM_HOST_DEVICE const int (&cvfem_hex8_bface_axis_tbl())[6] {
    static constexpr int t[6] = CVFEM_HEX8_BFACE_AXIS_INIT;
    return t;
}
#define CVFEM_HEX8_BFACE_AXIS cvfem_hex8_bface_axis_tbl()
#else
static constexpr int CVFEM_HEX8_BFACE_AXIS[6] = CVFEM_HEX8_BFACE_AXIS_INIT;
#endif
#define CVFEM_HEX8_BFACE_OUT_INIT {-1, 1, -1, 1, -1, 1}
#if defined(__CUDACC__)
static SFEM_INLINE SFEM_HOST_DEVICE const double (&cvfem_hex8_bface_out_tbl())[6] {
    static constexpr double t[6] = CVFEM_HEX8_BFACE_OUT_INIT;
    return t;
}
#define CVFEM_HEX8_BFACE_OUT cvfem_hex8_bface_out_tbl()
#else
static constexpr double CVFEM_HEX8_BFACE_OUT[6] = CVFEM_HEX8_BFACE_OUT_INIT;
#endif
#define CVFEM_HEX8_BFACE_XI_INIT { \
        {{0, double(0.25), double(0.25)}, \
         {0, double(0.75), double(0.25)}, \
         {0, double(0.75), double(0.75)}, \
         {0, double(0.25), double(0.75)}}, \
        {{1, double(0.25), double(0.25)}, \
         {1, double(0.75), double(0.25)}, \
         {1, double(0.75), double(0.75)}, \
         {1, double(0.25), double(0.75)}}, \
        {{double(0.25), 0, double(0.25)}, \
         {double(0.75), 0, double(0.25)}, \
         {double(0.75), 0, double(0.75)}, \
         {double(0.25), 0, double(0.75)}}, \
        {{double(0.25), 1, double(0.25)}, \
         {double(0.75), 1, double(0.25)}, \
         {double(0.75), 1, double(0.75)}, \
         {double(0.25), 1, double(0.75)}}, \
        {{double(0.25), double(0.25), 0}, \
         {double(0.75), double(0.25), 0}, \
         {double(0.75), double(0.75), 0}, \
         {double(0.25), double(0.75), 0}}, \
        {{double(0.25), double(0.25), 1}, \
         {double(0.75), double(0.25), 1}, \
         {double(0.75), double(0.75), 1}, \
         {double(0.25), double(0.75), 1}}}
#if defined(__CUDACC__)
static SFEM_INLINE SFEM_HOST_DEVICE const double (&cvfem_hex8_bface_xi_tbl())[6][4][3] {
    static constexpr double t[6][4][3] = CVFEM_HEX8_BFACE_XI_INIT;
    return t;
}
#define CVFEM_HEX8_BFACE_XI cvfem_hex8_bface_xi_tbl()
#else
static constexpr double CVFEM_HEX8_BFACE_XI[6][4][3] = CVFEM_HEX8_BFACE_XI_INIT;
#endif

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE int hex8_face_on_domain(const int f, const scalar_t *const SFEM_RESTRICT x,
                                           const scalar_t *const SFEM_RESTRICT y, const scalar_t *const SFEM_RESTRICT z,
                                           const scalar_t Lx, const scalar_t Ly, const scalar_t Lz) {
    const int      axis  = CVFEM_HEX8_BFACE_AXIS[f];
    const scalar_t L     = axis == 0 ? Lx : (axis == 1 ? Ly : Lz);
    const scalar_t plane = CVFEM_HEX8_BFACE_OUT[f] < 0 ? scalar_t(0) : L;
    for (int k = 0; k < 4; ++k) {
        const int      a = CVFEM_HEX8_BFACE_NODES[f][k];
        const scalar_t c = axis == 0 ? x[a] : (axis == 1 ? y[a] : z[a]);
        if (!on_plane(c, plane, L)) return 0;
    }
    return 1;
}

template <bool Atomic, typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void hex8_visc_jac_row(const scalar_t mu, const scalar_t ax, const scalar_t ay, const scalar_t az,
                                          const scalar_t w[][3], const int row, const smesh::count_t *const SFEM_RESTRICT slots,
                                          scalar_t *const SFEM_RESTRICT values) {
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
        cvfem_hex8_bsr_acc_mom<Atomic>(values, slots[row * 8 + k], d00, d01, d02, d10, d11, d12, d20, d21, d22);
    }
}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void boundary_scs_add_residual(const scalar_t rho, const scalar_t mu, const int isoparam, const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
                                                  const scalar_t Lx, const scalar_t Ly, const scalar_t Lz,
                                                  const scalar_t *const SFEM_RESTRICT x, const scalar_t *const SFEM_RESTRICT y,
                                                  const scalar_t *const SFEM_RESTRICT z, const scalar_t *const SFEM_RESTRICT ux,
                                                  const scalar_t *const SFEM_RESTRICT uy, const scalar_t *const SFEM_RESTRICT uz,
                                                  const scalar_t *const SFEM_RESTRICT p, scalar_t *const SFEM_RESTRICT r) {
    scalar_t grad_el[9];
    scalar_t A[3][3];
    if (!isoparam) {
        if (std::fabs(det) < scalar_t(1e-30)) return;
        cvfem_hex8_grad_sumfact(adj, det, ux, uy, uz, grad_el);
        cvfem_hex8_dir_areas(adj, A);
    }

    for (int f = 0; f < 6; ++f) {
        if (!hex8_face_on_domain(f, x, y, z, Lx, Ly, Lz)) continue;
        const int      axis = CVFEM_HEX8_BFACE_AXIS[f];
        const scalar_t out  = CVFEM_HEX8_BFACE_OUT[f];
        for (int k = 0; k < 4; ++k) {
            const int i = CVFEM_HEX8_BFACE_NODES[f][k];
            scalar_t  ax, ay, az, grad[9];
            if (isoparam) {
                scalar_t dN[CVFEM_HEX8_N_NODES][3];
                cvfem_hex8_dn_ref(CVFEM_HEX8_BFACE_XI[f][k][0], CVFEM_HEX8_BFACE_XI[f][k][1], CVFEM_HEX8_BFACE_XI[f][k][2], dN);
                scalar_t adj[9], det;
                cvfem_hex8_geom_at(x, y, z, CVFEM_HEX8_BFACE_XI[f][k][0], CVFEM_HEX8_BFACE_XI[f][k][1],
                                   CVFEM_HEX8_BFACE_XI[f][k][2], adj, &det);
                if (std::fabs(det) < scalar_t(1e-30)) continue;
                cvfem_hex8_area_dir(adj, axis, ax, ay, az);
                ax *= out;
                ay *= out;
                az *= out;
                cvfem_hex8_grad_at(adj, det, dN, ux, uy, uz, grad);
            } else {
                ax = out * A[axis][0];
                ay = out * A[axis][1];
                az = out * A[axis][2];
                for (int c = 0; c < 9; ++c) grad[c] = grad_el[c];
            }
            scalar_t tau_x, tau_y, tau_z;
            cvfem_hex8_traction(mu, grad[0], grad[1], grad[2], grad[3], grad[4], grad[5], grad[6], grad[7], grad[8], ax, ay, az,
                                tau_x, tau_y, tau_z);
            const scalar_t mdot = rho * (ux[i] * ax + uy[i] * ay + uz[i] * az);
            r[i * 4 + 0] += mdot * ux[i] + p[i] * ax - tau_x;
            r[i * 4 + 1] += mdot * uy[i] + p[i] * ay - tau_y;
            r[i * 4 + 2] += mdot * uz[i] + p[i] * az - tau_z;
            r[i * 4 + 3] += mdot;
        }
    }
}

template <bool Atomic, typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void boundary_scs_add_jacobian(const scalar_t rho, const scalar_t mu, const int isoparam, const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
                                                 const scalar_t Lx, const scalar_t Ly, const scalar_t Lz,
                                                 const scalar_t *const SFEM_RESTRICT x, const scalar_t *const SFEM_RESTRICT y,
                                                 const scalar_t *const SFEM_RESTRICT z, const scalar_t *const SFEM_RESTRICT ux,
                                                 const scalar_t *const SFEM_RESTRICT uy, const scalar_t *const SFEM_RESTRICT uz,
                                                 const smesh::count_t *const SFEM_RESTRICT slots, scalar_t *const SFEM_RESTRICT values) {
    scalar_t A[3][3];
    scalar_t w_el[CVFEM_HEX8_N_NODES][3];
    if (!isoparam) {
        if (std::fabs(det) < scalar_t(1e-30)) return;
        cvfem_hex8_dir_areas(adj, A);
        const scalar_t inv_det = scalar_t(1) / det;
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            cvfem_hex8_pushforward(adj, inv_det, CVFEM_HEX8_DN_REF[a][0], CVFEM_HEX8_DN_REF[a][1], CVFEM_HEX8_DN_REF[a][2],
                                   w_el[a][0], w_el[a][1], w_el[a][2]);
        }
    }

    for (int f = 0; f < 6; ++f) {
        if (!hex8_face_on_domain(f, x, y, z, Lx, Ly, Lz)) continue;
        const int      axis = CVFEM_HEX8_BFACE_AXIS[f];
        const scalar_t out  = CVFEM_HEX8_BFACE_OUT[f];
        for (int k = 0; k < 4; ++k) {
            const int i = CVFEM_HEX8_BFACE_NODES[f][k];
            scalar_t  ax, ay, az;
            scalar_t  w[CVFEM_HEX8_N_NODES][3];
            if (isoparam) {
                scalar_t dN[CVFEM_HEX8_N_NODES][3];
                cvfem_hex8_dn_ref(CVFEM_HEX8_BFACE_XI[f][k][0], CVFEM_HEX8_BFACE_XI[f][k][1], CVFEM_HEX8_BFACE_XI[f][k][2], dN);
                scalar_t adj[9], det;
                cvfem_hex8_geom_at(x, y, z, CVFEM_HEX8_BFACE_XI[f][k][0], CVFEM_HEX8_BFACE_XI[f][k][1],
                                   CVFEM_HEX8_BFACE_XI[f][k][2], adj, &det);
                if (std::fabs(det) < scalar_t(1e-30)) continue;
                cvfem_hex8_area_dir(adj, axis, ax, ay, az);
                ax *= out;
                ay *= out;
                az *= out;
                const scalar_t inv_det = scalar_t(1) / det;
                for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                    cvfem_hex8_pushforward(adj, inv_det, dN[a][0], dN[a][1], dN[a][2], w[a][0], w[a][1], w[a][2]);
                }
            } else {
                ax = out * A[axis][0];
                ay = out * A[axis][1];
                az = out * A[axis][2];
                for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                    w[a][0] = w_el[a][0];
                    w[a][1] = w_el[a][1];
                    w[a][2] = w_el[a][2];
                }
            }

            hex8_visc_jac_row<Atomic>(mu, ax, ay, az, w, i, slots, values);

            const scalar_t un   = ux[i] * ax + uy[i] * ay + uz[i] * az;
            const scalar_t mdot = rho * un;
            const smesh::count_t sii = slots[i * 8 + i];
            cvfem_hex8_bsr_acc<Atomic>(values, sii, 0, 0, rho * ax * ux[i] + mdot);
            cvfem_hex8_bsr_acc<Atomic>(values, sii, 0, 1, rho * ay * ux[i]);
            cvfem_hex8_bsr_acc<Atomic>(values, sii, 0, 2, rho * az * ux[i]);
            cvfem_hex8_bsr_acc<Atomic>(values, sii, 0, 3, ax);
            cvfem_hex8_bsr_acc<Atomic>(values, sii, 1, 0, rho * ax * uy[i]);
            cvfem_hex8_bsr_acc<Atomic>(values, sii, 1, 1, rho * ay * uy[i] + mdot);
            cvfem_hex8_bsr_acc<Atomic>(values, sii, 1, 2, rho * az * uy[i]);
            cvfem_hex8_bsr_acc<Atomic>(values, sii, 1, 3, ay);
            cvfem_hex8_bsr_acc<Atomic>(values, sii, 2, 0, rho * ax * uz[i]);
            cvfem_hex8_bsr_acc<Atomic>(values, sii, 2, 1, rho * ay * uz[i]);
            cvfem_hex8_bsr_acc<Atomic>(values, sii, 2, 2, rho * az * uz[i] + mdot);
            cvfem_hex8_bsr_acc<Atomic>(values, sii, 2, 3, az);
            cvfem_hex8_bsr_acc<Atomic>(values, sii, 3, 0, rho * ax);
            cvfem_hex8_bsr_acc<Atomic>(values, sii, 3, 1, rho * ay);
            cvfem_hex8_bsr_acc<Atomic>(values, sii, 3, 2, rho * az);
        }
    }
}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void boundary_scs_add_jacobian_action(const scalar_t rho, const scalar_t mu, const int isoparam,
                                                         const scalar_t *const SFEM_RESTRICT adj, const scalar_t det, const scalar_t Lx, const scalar_t Ly,
                                                         const scalar_t Lz, const scalar_t *const SFEM_RESTRICT x,
                                                         const scalar_t *const SFEM_RESTRICT y, const scalar_t *const SFEM_RESTRICT z,
                                                         const scalar_t *const SFEM_RESTRICT ux, const scalar_t *const SFEM_RESTRICT uy,
                                                         const scalar_t *const SFEM_RESTRICT uz, const scalar_t *const SFEM_RESTRICT vx,
                                                         const scalar_t *const SFEM_RESTRICT vy, const scalar_t *const SFEM_RESTRICT vz,
                                                         const scalar_t *const SFEM_RESTRICT q, scalar_t *const SFEM_RESTRICT r) {
    scalar_t dgrad_el[9];
    scalar_t A[3][3];
    if (!isoparam) {
        if (std::fabs(det) < scalar_t(1e-30)) return;
        cvfem_hex8_grad_sumfact(adj, det, vx, vy, vz, dgrad_el);
        cvfem_hex8_dir_areas(adj, A);
    }

    for (int f = 0; f < 6; ++f) {
        if (!hex8_face_on_domain(f, x, y, z, Lx, Ly, Lz)) continue;
        const int      axis = CVFEM_HEX8_BFACE_AXIS[f];
        const scalar_t out  = CVFEM_HEX8_BFACE_OUT[f];
        for (int k = 0; k < 4; ++k) {
            const int i = CVFEM_HEX8_BFACE_NODES[f][k];
            scalar_t  ax, ay, az, dgrad[9];
            if (isoparam) {
                scalar_t dN[CVFEM_HEX8_N_NODES][3];
                cvfem_hex8_dn_ref(CVFEM_HEX8_BFACE_XI[f][k][0], CVFEM_HEX8_BFACE_XI[f][k][1], CVFEM_HEX8_BFACE_XI[f][k][2], dN);
                scalar_t adj[9], det;
                cvfem_hex8_geom_at(x, y, z, CVFEM_HEX8_BFACE_XI[f][k][0], CVFEM_HEX8_BFACE_XI[f][k][1],
                                   CVFEM_HEX8_BFACE_XI[f][k][2], adj, &det);
                if (std::fabs(det) < scalar_t(1e-30)) continue;
                cvfem_hex8_area_dir(adj, axis, ax, ay, az);
                ax *= out;
                ay *= out;
                az *= out;
                cvfem_hex8_grad_at(adj, det, dN, vx, vy, vz, dgrad);
            } else {
                ax = out * A[axis][0];
                ay = out * A[axis][1];
                az = out * A[axis][2];
                for (int c = 0; c < 9; ++c) dgrad[c] = dgrad_el[c];
            }
            scalar_t dtx, dty, dtz;
            cvfem_hex8_traction(mu, dgrad[0], dgrad[1], dgrad[2], dgrad[3], dgrad[4], dgrad[5], dgrad[6], dgrad[7], dgrad[8], ax,
                                ay, az, dtx, dty, dtz);
            const scalar_t mdot  = rho * (ux[i] * ax + uy[i] * ay + uz[i] * az);
            const scalar_t dmdot = rho * (vx[i] * ax + vy[i] * ay + vz[i] * az);
            r[i * 4 + 0] += dmdot * ux[i] + mdot * vx[i] + q[i] * ax - dtx;
            r[i * 4 + 1] += dmdot * uy[i] + mdot * vy[i] + q[i] * ay - dty;
            r[i * 4 + 2] += dmdot * uz[i] + mdot * vz[i] + q[i] * az - dtz;
            r[i * 4 + 3] += dmdot;
        }
    }
}

// Nodal pressure gradient, used by the Rhie-Chow mass-flux interpolation. Lives here
// rather than in the solver so the device kernels call the same code.
template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_grad_scalar(const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
                                               const scalar_t *const SFEM_RESTRICT p, scalar_t &gx, scalar_t &gy,
                                               scalar_t &gz) {
    scalar_t dr, ds, dt;
    cvfem_hex8_face_diff(p, dr, ds, dt);
    cvfem_hex8_pushforward(adj, scalar_t(1) / det, dr, ds, dt, gx, gy, gz);
}

// ---------------------------------------------------------------------------
// Block-Jacobi preconditioner block.
//
// The 4x4 diagonal block is singular for incompressible flow -- the pressure-pressure
// entry is zero, which is the saddle-point structure -- so a plain 4x4 inverse is the
// wrong operation. This mirrors build_block_jacobi in cvfem_hex8_ns_steady.cpp: invert
// the 3x3 velocity sub-block, take the reciprocal of the pressure diagonal, and leave
// the velocity-pressure coupling out.
template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE bool cvfem_hex8_invert3_vel(
        const scalar_t *const SFEM_RESTRICT a, scalar_t *const SFEM_RESTRICT inv) {
    const scalar_t a00 = a[0], a01 = a[1], a02 = a[2];
    const scalar_t a10 = a[4], a11 = a[5], a12 = a[6];
    const scalar_t a20 = a[8], a21 = a[9], a22 = a[10];
    const scalar_t x0 = a11 * a22, x1 = a12 * a21, x2 = a01 * a12;
    const scalar_t x3 = a01 * a22, x4 = a02 * a11;
    const scalar_t det = a00 * (x0 - x1) + a02 * a10 * a21 - a10 * x3 + a20 * x2 - a20 * x4;
    // Magnitude bounds rather than isfinite(): a classification call can be folded away
    // by fast-math, and this kernel is compiled with -use_fast_math.
    const scalar_t ad = det < scalar_t(0) ? -det : det;
    if (!(ad > scalar_t(1e-30)) || !(ad < scalar_t(1e300))) return false;
    const scalar_t s = scalar_t(1) / det;
    inv[0]  = s * (x0 - x1);
    inv[1]  = s * (a02 * a21 - x3);
    inv[2]  = s * (x2 - x4);
    inv[4]  = s * (-a10 * a22 + a12 * a20);
    inv[5]  = s * (a00 * a22 - a02 * a20);
    inv[6]  = s * (-a00 * a12 + a02 * a10);
    inv[8]  = s * (a10 * a21 - a11 * a20);
    inv[9]  = s * (-a00 * a21 + a01 * a20);
    inv[10] = s * (a00 * a11 - a01 * a10);
    return true;
}

// One node's preconditioner block. `constrained` is the 4 per-field Dirichlet flags.
template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_block_jacobi_block(
        const scalar_t *const SFEM_RESTRICT blk,
        const unsigned char *const SFEM_RESTRICT constrained,
        scalar_t *const SFEM_RESTRICT inv) {
    for (int i = 0; i < 16; ++i) inv[i] = scalar_t(0);
    const int c0 = constrained ? constrained[0] : 0;
    const int c1 = constrained ? constrained[1] : 0;
    const int c2 = constrained ? constrained[2] : 0;
    const int c3 = constrained ? constrained[3] : 0;

    if (!(c0 | c1 | c2) && cvfem_hex8_invert3_vel(blk, inv)) {
        // velocity 3x3 inverse written above
    } else {
        for (int f = 0; f < 3; ++f) {
            if (constrained && constrained[f]) {
                inv[f * 4 + f] = scalar_t(1);
            } else {
                const scalar_t d  = blk[f * 4 + f];
                const scalar_t ad = d < scalar_t(0) ? -d : d;
                inv[f * 4 + f] = (ad > scalar_t(1e-30)) ? scalar_t(1) / d : scalar_t(1);
            }
        }
    }
    if (c3) {
        inv[15] = scalar_t(1);
    } else {
        const scalar_t d  = blk[15];
        const scalar_t ad = d < scalar_t(0) ? -d : d;
        inv[15] = (ad > scalar_t(1e-30)) ? scalar_t(1) / d : scalar_t(1);
    }
}

#endif  // CVFEM_HEX8_BOUNDARY_SCS_HPP
