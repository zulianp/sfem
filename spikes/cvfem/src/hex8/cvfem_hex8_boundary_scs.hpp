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

// Prescribed boundary data, carried alongside the face masks.
//
// A default-constructed instance means "nothing prescribed", which reproduces the existing
// behaviour on every face exactly -- so this is a trailing default argument on the routines
// below and no call site had to change. That is the shape Hex8RhieChowT already uses to
// make the Rhie-Chow term optional, and it is preferred here over widening the face masks
// to two bits per face: the values a boundary condition needs are per-sideset constants,
// not per-face bits, and re-encoding the masks would have touched all three signatures and
// about a dozen call sites for no gain.
//
// Two conditions, each reducing to something already supported:
//
//   traction   (pI - tau).n = t on the faces tmask selects, which must be a subset of
//              nmask -- a traction condition IS the natural condition, with a value. t = 0,
//              or a face in nmask but not tmask, is the do-nothing outflow bit for bit:
//              the term added vanishes and its square root is not even evaluated.
//
//              tmask exists rather than the value simply applying to all of nmask because a
//              single run routinely has both -- an outlet that is genuinely traction-free
//              and a surface that is pushed -- and one scalar triple covering every natural
//              face cannot express that. It would instead apply the pushed surface's
//              traction to the outlet as well, silently. This mirrors the pmask/p_bar pair
//              below: one per-face selector, one per-sideset constant.
//   pressure   p = p_bar on the faces pmask selects, with the viscous traction still taken
//              from the interior state. This is the closed-face flux with the nodal
//              pressure replaced by a prescribed one, which is what a port held at a
//              pressure is.
//
// nmask wins where both select the same face: a face cannot be both traction-free and
// pressure-prescribed, and silently applying both would be worse than picking one and
// saying so.
template <typename scalar_t>
struct Hex8BoundaryDataT {
    scalar_t tx{0}, ty{0}, tz{0};  // prescribed traction, on the faces tmask selects
    int      tmask{0};             // faces carrying it; 0 means every natural face is free
    int      pmask{0};             // faces carrying a prescribed pressure
    scalar_t p_bar{0};             // and its value
};

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
                                                  const scalar_t *const SFEM_RESTRICT p, scalar_t *const SFEM_RESTRICT r,
                                                  const int fmask = -1,
                                                  const int nmask = 0,
                                                  const Hex8BoundaryDataT<scalar_t> &bd = {}) {
    scalar_t grad_el[9];
    // Hoisted: the area magnitude a prescribed traction needs costs a square root per node
    // per face, and t = 0 is the overwhelmingly common case.
    const int have_traction =
            bd.tmask != 0 && (bd.tx != scalar_t(0) || bd.ty != scalar_t(0) || bd.tz != scalar_t(0));
    scalar_t A[3][3];
    if (!isoparam) {
        if (std::fabs(det) < scalar_t(1e-30)) return;
        cvfem_hex8_grad_sumfact(adj, det, ux, uy, uz, grad_el);
        cvfem_hex8_dir_areas(adj, A);
    }

    for (int f = 0; f < 6; ++f) {
        // fmask < 0 keeps the historical behaviour: decide from the bounding box. A
        // non-negative mask is an explicit per-element bitfield, one bit per local face,
        // which is the only way to get this right on a domain that is not a box -- the
        // coordinate test cannot see a re-entrant face such as the step of a
        // backward-facing step, and silently leaves those control volumes unclosed.
        if (fmask < 0 ? !hex8_face_on_domain(f, x, y, z, Lx, Ly, Lz)
                      : !((fmask >> f) & 1))
            continue;
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
            if (((bd.pmask >> f) & 1) && !((nmask >> f) & 1)) {
                // Prescribed pressure: the closed-face flux with p_bar in place of the
                // nodal pressure. The viscous traction still comes from the interior state,
                // so this prescribes the pressure and not the whole normal traction.
                r[i * 4 + 0] += mdot * ux[i] + bd.p_bar * ax - tau_x;
                r[i * 4 + 1] += mdot * uy[i] + bd.p_bar * ay - tau_y;
                r[i * 4 + 2] += mdot * uz[i] + bd.p_bar * az - tau_z;
                r[i * 4 + 3] += mdot;
            } else if ((nmask >> f) & 1) {
                // Do-nothing (natural) outflow: (p I - tau) . n = 0, so the pressure and
                // viscous traction are prescribed rather than evaluated. Dropping them is
                // what makes this a genuine outflow condition and what removes the constant-
                // pressure nullspace -- with p_i * a retained, a uniform pressure shift
                // integrates to zero over every closed control volume and the gauge stays
                // undetermined, so the solve needs a pin and behaves badly with one.
                //
                // Backflow guard: the convective term uses max(mdot, 0). The interior kernel
                // has an upwind switch (mpos/mneg) and this one does not, so on a face where
                // mdot < 0 the unguarded form would convect the *downwind* value into the
                // domain -- the classic finite-volume backflow instability, and a
                // recirculating outlet is where it bites.
                const scalar_t mup = mdot > scalar_t(0) ? mdot : scalar_t(0);
                scalar_t dS = scalar_t(0);
                if (have_traction && ((bd.tmask >> f) & 1)) dS = std::sqrt(ax * ax + ay * ay + az * az);
                r[i * 4 + 0] += mup * ux[i] + bd.tx * dS;
                r[i * 4 + 1] += mup * uy[i] + bd.ty * dS;
                r[i * 4 + 2] += mup * uz[i] + bd.tz * dS;
                // Continuity carries the true flux, not the guarded one: clipping it would
                // destroy global mass conservation, which is the property being verified.
                r[i * 4 + 3] += mdot;
            } else {
                r[i * 4 + 0] += mdot * ux[i] + p[i] * ax - tau_x;
                r[i * 4 + 1] += mdot * uy[i] + p[i] * ay - tau_y;
                r[i * 4 + 2] += mdot * uz[i] + p[i] * az - tau_z;
                r[i * 4 + 3] += mdot;
            }
        }
    }
}

template <bool Atomic, typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void boundary_scs_add_jacobian(const scalar_t rho, const scalar_t mu, const int isoparam, const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
                                                 const scalar_t Lx, const scalar_t Ly, const scalar_t Lz,
                                                 const scalar_t *const SFEM_RESTRICT x, const scalar_t *const SFEM_RESTRICT y,
                                                 const scalar_t *const SFEM_RESTRICT z, const scalar_t *const SFEM_RESTRICT ux,
                                                 const scalar_t *const SFEM_RESTRICT uy, const scalar_t *const SFEM_RESTRICT uz,
                                                 const smesh::count_t *const SFEM_RESTRICT slots, scalar_t *const SFEM_RESTRICT values,
                                                  const int fmask = -1,
                                                  const int nmask = 0,
                                                  const Hex8BoundaryDataT<scalar_t> &bd = {}) {
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
        // fmask < 0 keeps the historical behaviour: decide from the bounding box. A
        // non-negative mask is an explicit per-element bitfield, one bit per local face,
        // which is the only way to get this right on a domain that is not a box -- the
        // coordinate test cannot see a re-entrant face such as the step of a
        // backward-facing step, and silently leaves those control volumes unclosed.
        if (fmask < 0 ? !hex8_face_on_domain(f, x, y, z, Lx, Ly, Lz)
                      : !((fmask >> f) & 1))
            continue;
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

            const scalar_t un   = ux[i] * ax + uy[i] * ay + uz[i] * az;
            const scalar_t mdot = rho * un;
            const smesh::count_t sii = slots[i * 8 + i];

            // The natural-outflow branch, matching boundary_scs_add_residual and
            // boundary_scs_add_jacobian_action. This function used to take nmask and
            // ignore it, so on a do-nothing face the assembled matrix kept the pressure
            // column and the viscous row that the residual drops -- it was not the
            // derivative of anything the solver evaluates. The matrix-free path was
            // unaffected (the action above has always branched), but the assembled
            // matrix feeds the coarse-grid operators, the Vanka patch solves and the
            // block-diagonal preconditioner, so the inconsistency reached the multigrid
            // for every case with an open outlet.
            if (((bd.pmask >> f) & 1) && !((nmask >> f) & 1)) {
                // Prescribed pressure: the closed-face block without its pressure column.
                // p_bar is data, not an unknown, so d(residual)/dp is zero on this face;
                // the viscous row stays because tau still depends on the velocity.
                hex8_visc_jac_row<Atomic>(mu, ax, ay, az, w, i, slots, values);
                cvfem_hex8_bsr_acc<Atomic>(values, sii, 0, 0, rho * ax * ux[i] + mdot);
                cvfem_hex8_bsr_acc<Atomic>(values, sii, 0, 1, rho * ay * ux[i]);
                cvfem_hex8_bsr_acc<Atomic>(values, sii, 0, 2, rho * az * ux[i]);
                cvfem_hex8_bsr_acc<Atomic>(values, sii, 1, 0, rho * ax * uy[i]);
                cvfem_hex8_bsr_acc<Atomic>(values, sii, 1, 1, rho * ay * uy[i] + mdot);
                cvfem_hex8_bsr_acc<Atomic>(values, sii, 1, 2, rho * az * uy[i]);
                cvfem_hex8_bsr_acc<Atomic>(values, sii, 2, 0, rho * ax * uz[i]);
                cvfem_hex8_bsr_acc<Atomic>(values, sii, 2, 1, rho * ay * uz[i]);
                cvfem_hex8_bsr_acc<Atomic>(values, sii, 2, 2, rho * az * uz[i] + mdot);
                cvfem_hex8_bsr_acc<Atomic>(values, sii, 3, 0, rho * ax);
                cvfem_hex8_bsr_acc<Atomic>(values, sii, 3, 1, rho * ay);
                cvfem_hex8_bsr_acc<Atomic>(values, sii, 3, 2, rho * az);
                continue;
            }
            if ((nmask >> f) & 1) {
                // d/du of max(mdot, 0) * u_i: the same velocity block as the closed face
                // where mdot > 0, and nothing where it is not. No pressure column, because
                // the residual has no p_i * a term here; no viscous row, for the same
                // reason. Continuity still carries the true flux and so keeps its row.
                if (mdot > scalar_t(0)) {
                    cvfem_hex8_bsr_acc<Atomic>(values, sii, 0, 0, rho * ax * ux[i] + mdot);
                    cvfem_hex8_bsr_acc<Atomic>(values, sii, 0, 1, rho * ay * ux[i]);
                    cvfem_hex8_bsr_acc<Atomic>(values, sii, 0, 2, rho * az * ux[i]);
                    cvfem_hex8_bsr_acc<Atomic>(values, sii, 1, 0, rho * ax * uy[i]);
                    cvfem_hex8_bsr_acc<Atomic>(values, sii, 1, 1, rho * ay * uy[i] + mdot);
                    cvfem_hex8_bsr_acc<Atomic>(values, sii, 1, 2, rho * az * uy[i]);
                    cvfem_hex8_bsr_acc<Atomic>(values, sii, 2, 0, rho * ax * uz[i]);
                    cvfem_hex8_bsr_acc<Atomic>(values, sii, 2, 1, rho * ay * uz[i]);
                    cvfem_hex8_bsr_acc<Atomic>(values, sii, 2, 2, rho * az * uz[i] + mdot);
                }
                cvfem_hex8_bsr_acc<Atomic>(values, sii, 3, 0, rho * ax);
                cvfem_hex8_bsr_acc<Atomic>(values, sii, 3, 1, rho * ay);
                cvfem_hex8_bsr_acc<Atomic>(values, sii, 3, 2, rho * az);
                continue;
            }

            hex8_visc_jac_row<Atomic>(mu, ax, ay, az, w, i, slots, values);

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
                                                         const scalar_t *const SFEM_RESTRICT q, scalar_t *const SFEM_RESTRICT r,
                                                  const int fmask = -1,
                                                  const int nmask = 0,
                                                  const Hex8BoundaryDataT<scalar_t> &bd = {}) {
    scalar_t dgrad_el[9];
    scalar_t A[3][3];
    if (!isoparam) {
        if (std::fabs(det) < scalar_t(1e-30)) return;
        cvfem_hex8_grad_sumfact(adj, det, vx, vy, vz, dgrad_el);
        cvfem_hex8_dir_areas(adj, A);
    }

    for (int f = 0; f < 6; ++f) {
        // fmask < 0 keeps the historical behaviour: decide from the bounding box. A
        // non-negative mask is an explicit per-element bitfield, one bit per local face,
        // which is the only way to get this right on a domain that is not a box -- the
        // coordinate test cannot see a re-entrant face such as the step of a
        // backward-facing step, and silently leaves those control volumes unclosed.
        if (fmask < 0 ? !hex8_face_on_domain(f, x, y, z, Lx, Ly, Lz)
                      : !((fmask >> f) & 1))
            continue;
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
            if (((bd.pmask >> f) & 1) && !((nmask >> f) & 1)) {
                // Prescribed pressure: as the closed face but without the q[i] * a term,
                // since the pressure on this face is data and the direction cannot move it.
                r[i * 4 + 0] += dmdot * ux[i] + mdot * vx[i] - dtx;
                r[i * 4 + 1] += dmdot * uy[i] + mdot * vy[i] - dty;
                r[i * 4 + 2] += dmdot * uz[i] + mdot * vz[i] - dtz;
                r[i * 4 + 3] += dmdot;
            } else if ((nmask >> f) & 1) {
                // Exact derivative of the natural-outflow residual above. The max(mdot, 0)
                // guard is piecewise linear, so its derivative is dmdot*u_i + mdot*v_i where
                // mdot > 0 and zero where it is not. The kink at mdot == 0 is the same class
                // of non-differentiability the interior upwind switch already has.
                if (mdot > scalar_t(0)) {
                    r[i * 4 + 0] += dmdot * ux[i] + mdot * vx[i];
                    r[i * 4 + 1] += dmdot * uy[i] + mdot * vy[i];
                    r[i * 4 + 2] += dmdot * uz[i] + mdot * vz[i];
                }
                r[i * 4 + 3] += dmdot;
            } else {
                r[i * 4 + 0] += dmdot * ux[i] + mdot * vx[i] + q[i] * ax - dtx;
                r[i * 4 + 1] += dmdot * uy[i] + mdot * vy[i] + q[i] * ay - dty;
                r[i * 4 + 2] += dmdot * uz[i] + mdot * vz[i] + q[i] * az - dtz;
                r[i * 4 + 3] += dmdot;
            }
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

    // Scale of the block, for relative floors below. An absolute threshold cannot express
    // "small compared with this block": a diagonal of 1e-20 passes |d| > 1e-30 and yields an
    // inverse of 1e20, which the smoother then applies to the residual every sweep.
    scalar_t blk_scale = scalar_t(0);
    for (int k = 0; k < 16; ++k) {
        const scalar_t a = blk[k] < scalar_t(0) ? -blk[k] : blk[k];
        if (a > blk_scale) blk_scale = a;
    }
    const scalar_t blk_floor = blk_scale * scalar_t(1e-14);

    if (!(c0 | c1 | c2) && cvfem_hex8_invert3_vel(blk, inv)) {
        // velocity 3x3 inverse written above
    } else {
        for (int f = 0; f < 3; ++f) {
            if (constrained && constrained[f]) {
                inv[f * 4 + f] = scalar_t(1);
            } else {
                const scalar_t d  = blk[f * 4 + f];
                const scalar_t ad = d < scalar_t(0) ? -d : d;
                // Falls back to 1, not to a huge inverse: a degenerate diagonal means this
                // dof gets an unscaled (weak) update rather than an explosive one.
                inv[f * 4 + f] =
                        (ad > blk_floor && ad > scalar_t(1e-30)) ? scalar_t(1) / d : scalar_t(1);
            }
        }
    }
    if (c3) {
        inv[15] = scalar_t(1);
    } else {
        const scalar_t d  = blk[15];
        const scalar_t ad = d < scalar_t(0) ? -d : d;
        inv[15] = (ad > blk_floor && ad > scalar_t(1e-30)) ? scalar_t(1) / d : scalar_t(1);
    }
}

#endif  // CVFEM_HEX8_BOUNDARY_SCS_HPP
