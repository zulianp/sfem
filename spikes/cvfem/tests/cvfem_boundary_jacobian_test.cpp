// The boundary Jacobian must be the derivative of the boundary residual.
//
// Three routines have to agree about what a boundary face contributes:
// boundary_scs_add_residual, boundary_scs_add_jacobian_action (its directional derivative)
// and boundary_scs_add_jacobian (the assembled matrix). Nothing checked that they did.
//
// The comparison is against a central finite difference of the residual, not against the
// formula any of them was derived from, so an error shared between a derivative and its
// own analytic check cannot hide. It runs on a single element with identity BSR slots --
// the trick the benchmark's --kernel-only path uses -- so the assembled 8x8 block matrix
// is just a dense 1024-entry buffer and no mesh or sparsity pattern is needed.
//
// Both boundary treatments are checked. The do-nothing outflow is the one that matters:
// it drops p*a and the viscous traction from the residual, so a Jacobian that keeps their
// derivatives is not the derivative of anything the solver evaluates.

#include "smesh_types.hpp"

#include "cvfem_portability.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using scalar_t = double;
static constexpr int N_FIELDS = 4;

#include "cvfem_hex8_ns_upwind_kernels.hpp"
#include "cvfem_hex8_boundary_scs.hpp"

static int g_failures = 0;

static constexpr scalar_t RHO = 1.0, MU = 0.01, L = 1.0;

struct State {
    scalar_t x[8], y[8], z[8], u[CVFEM_HEX8_N_DOF];
};

static void init(State &s) {
    static const scalar_t cx[8] = {0, 1, 1, 0, 0, 1, 1, 0};
    static const scalar_t cy[8] = {0, 0, 1, 1, 0, 0, 1, 1};
    static const scalar_t cz[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    for (int a = 0; a < 8; ++a) {
        s.x[a] = cx[a];
        s.y[a] = cy[a];
        s.z[a] = cz[a];
        // Deliberately not symmetric, and with a genuine backflow face: ux is negative at
        // x = 0, so the do-nothing branch's max(mdot, 0) guard is actually exercised.
        s.u[a * 4 + 0] = -0.6 + 1.4 * cx[a] - 0.2 * cy[a];
        s.u[a * 4 + 1] = -0.4 + 0.2 * cx[a] + 0.5 * cy[a];
        s.u[a * 4 + 2] = 0.2 - 0.1 * cx[a] + 0.25 * cz[a];
        s.u[a * 4 + 3] = 1.0 + 0.1 * cx[a] + 0.2 * cy[a];
    }
}

static Hex8BoundaryDataT<scalar_t> g_bd{};

static void residual_of(const State &s, const int fmask, const int nmask, scalar_t *r) {
    scalar_t ux[8], uy[8], uz[8], p[8], adj[9], det;
    for (int a = 0; a < 8; ++a) {
        ux[a] = s.u[a * 4 + 0];
        uy[a] = s.u[a * 4 + 1];
        uz[a] = s.u[a * 4 + 2];
        p[a]  = s.u[a * 4 + 3];
    }
    cvfem_hex8_affine_adj(s.x, s.y, s.z, adj, &det);
    std::memset(r, 0, sizeof(scalar_t) * CVFEM_HEX8_N_DOF);
    boundary_scs_add_residual(RHO, MU, 0, adj, det, L, L, L, s.x, s.y, s.z, ux, uy, uz, p, r, fmask, nmask, g_bd);
}

// The assembled boundary Jacobian as a dense 8x8 block matrix, via identity slots.
static void assembled_of(const State &s, const int fmask, const int nmask, std::vector<scalar_t> &values) {
    scalar_t ux[8], uy[8], uz[8], adj[9], det;
    for (int a = 0; a < 8; ++a) {
        ux[a] = s.u[a * 4 + 0];
        uy[a] = s.u[a * 4 + 1];
        uz[a] = s.u[a * 4 + 2];
    }
    cvfem_hex8_affine_adj(s.x, s.y, s.z, adj, &det);
    static smesh::count_t slots[64];
    for (int k = 0; k < 64; ++k) slots[k] = (smesh::count_t)k;
    values.assign(64 * 16, scalar_t(0));
    boundary_scs_add_jacobian<false>(RHO, MU, 0, adj, det, L, L, L, s.x, s.y, s.z, ux, uy, uz, slots,
                                     values.data(), fmask, nmask, g_bd);
}

// d r[row] / d u[col], central difference.
static scalar_t fd_entry(State s, const int fmask, const int nmask, const int row, const int col) {
    const scalar_t h    = 1e-6;
    const scalar_t save = s.u[col];
    scalar_t       rp[CVFEM_HEX8_N_DOF], rm[CVFEM_HEX8_N_DOF];
    s.u[col] = save + h;
    residual_of(s, fmask, nmask, rp);
    s.u[col] = save - h;
    residual_of(s, fmask, nmask, rm);
    s.u[col] = save;
    return (rp[row] - rm[row]) / (2 * h);
}

// Worst absolute disagreement between the assembled matrix and the finite difference.
static scalar_t compare(const State &s, const int fmask, const int nmask, int *worst_row, int *worst_col) {
    std::vector<scalar_t> values;
    assembled_of(s, fmask, nmask, values);
    scalar_t worst = 0;
    for (int ai = 0; ai < 8; ++ai) {
        for (int aj = 0; aj < 8; ++aj) {
            for (int ri = 0; ri < 4; ++ri) {
                for (int ci = 0; ci < 4; ++ci) {
                    const scalar_t got  = values[(size_t)(ai * 8 + aj) * 16 + (size_t)ri * 4 + (size_t)ci];
                    const scalar_t want = fd_entry(s, fmask, nmask, ai * 4 + ri, aj * 4 + ci);
                    const scalar_t d    = std::fabs(got - want);
                    if (d > worst) {
                        worst = d;
                        if (worst_row) *worst_row = ai * 4 + ri;
                        if (worst_col) *worst_col = aj * 4 + ci;
                    }
                }
            }
        }
    }
    return worst;
}

static void check(const bool ok, const char *what) {
    std::printf("%-64s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) ++g_failures;
}

int main() {
    State s;
    init(s);

    // The finite difference is second-order in h and the residual is smooth away from the
    // upwind switch, so 1e-6 perturbations give roughly 1e-9 accuracy. The tolerance is set
    // well inside that and well outside it, so neither a true match nor a real disagreement
    // is ambiguous.
    const scalar_t tol = 1e-7;

    // 1. Closed faces: the assembled Jacobian is the derivative of the residual.
    {
        int      row = -1, col = -1;
        const scalar_t worst = compare(s, 0x3F, 0, &row, &col);
        std::printf("closed faces:      worst |assembled - FD| = %.3e  (row %d, col %d)\n", (double)worst, row, col);
        check(worst < tol, "the assembled boundary Jacobian matches FD on closed faces");
    }

    // 2. The Jacobian action agrees with the residual's derivative too. It is the path the
    //    matrix-free solve actually uses, so it is checked separately from the matrix.
    {
        scalar_t ux[8], uy[8], uz[8], vx[8], vy[8], vz[8], q[8], adj[9], det;
        for (int a = 0; a < 8; ++a) {
            ux[a] = s.u[a * 4 + 0];
            uy[a] = s.u[a * 4 + 1];
            uz[a] = s.u[a * 4 + 2];
            // An arbitrary direction, all four components active.
            vx[a] = 0.7 - 0.3 * s.x[a];
            vy[a] = -0.2 + 0.4 * s.y[a];
            vz[a] = 0.15 * s.z[a];
            q[a]  = 0.5 - 0.25 * s.x[a];
        }
        cvfem_hex8_affine_adj(s.x, s.y, s.z, adj, &det);

        for (const int nmask : {0, 0x3F}) {
            scalar_t jv[CVFEM_HEX8_N_DOF];
            std::memset(jv, 0, sizeof(jv));
            boundary_scs_add_jacobian_action(RHO, MU, 0, adj, det, L, L, L, s.x, s.y, s.z, ux, uy, uz, vx, vy, vz,
                                             q, jv, 0x3F, nmask, g_bd);
            // The same directional derivative by finite difference on the residual.
            State sp = s, sm = s;
            const scalar_t h = 1e-6;
            for (int a = 0; a < 8; ++a) {
                sp.u[a * 4 + 0] += h * vx[a]; sm.u[a * 4 + 0] -= h * vx[a];
                sp.u[a * 4 + 1] += h * vy[a]; sm.u[a * 4 + 1] -= h * vy[a];
                sp.u[a * 4 + 2] += h * vz[a]; sm.u[a * 4 + 2] -= h * vz[a];
                sp.u[a * 4 + 3] += h * q[a];  sm.u[a * 4 + 3] -= h * q[a];
            }
            scalar_t rp[CVFEM_HEX8_N_DOF], rm[CVFEM_HEX8_N_DOF];
            residual_of(sp, 0x3F, nmask, rp);
            residual_of(sm, 0x3F, nmask, rm);
            scalar_t worst = 0;
            for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i)
                worst = std::max(worst, std::fabs(jv[i] - (rp[i] - rm[i]) / (2 * h)));
            std::printf("action, nmask %#04x: worst |J v - FD| = %.3e\n", nmask, (double)worst);
            check(worst < tol, nmask == 0 ? "the Jacobian action matches FD on closed faces"
                                          : "the Jacobian action matches FD on do-nothing faces");
        }
    }

    // 3. The same for the assembled matrix on do-nothing faces. This is the one that was
    //    not honouring nmask at all.
    {
        int            row = -1, col = -1;
        const scalar_t worst = compare(s, 0x3F, 0x3F, &row, &col);
        std::printf("do-nothing faces:  worst |assembled - FD| = %.3e  (row %d, col %d)\n", (double)worst, row, col);
        check(worst < tol, "the assembled boundary Jacobian matches FD on do-nothing faces");
    }

    // 4. Prescribed traction. t = 0 must reproduce the do-nothing outflow exactly -- that
    //    is what makes this a generalisation and not a change -- and a non-zero t must
    //    move the residual without touching the Jacobian, since a constant traction has no
    //    derivative.
    {
        scalar_t r_zero[CVFEM_HEX8_N_DOF], r_t[CVFEM_HEX8_N_DOF];
        g_bd = {};
        residual_of(s, 0x3F, 0x3F, r_zero);

        Hex8BoundaryDataT<scalar_t> bd{};
        bd.tx = 0.0; bd.ty = 0.0; bd.tz = 0.0;
        g_bd  = bd;
        residual_of(s, 0x3F, 0x3F, r_t);
        scalar_t d = 0;
        for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) d = std::max(d, std::fabs(r_t[i] - r_zero[i]));
        check(d == 0.0, "traction t = 0 reproduces the do-nothing outflow exactly");

        // A value with no face selected is still the do-nothing outflow. tmask is what
        // makes a traction condition addressable, so that a run can hold one surface under
        // traction while another stays genuinely free; without this check the two could
        // drift apart and every natural face would quietly inherit the value.
        bd.tx = 0.35; bd.ty = -0.2; bd.tz = 0.1;
        bd.tmask = 0;
        g_bd  = bd;
        residual_of(s, 0x3F, 0x3F, r_t);
        d = 0;
        for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) d = std::max(d, std::fabs(r_t[i] - r_zero[i]));
        check(d == 0.0, "a traction value with tmask = 0 is still the do-nothing outflow");

        bd.tmask = 0x3F;
        g_bd  = bd;
        residual_of(s, 0x3F, 0x3F, r_t);
        d = 0;
        for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) d = std::max(d, std::fabs(r_t[i] - r_zero[i]));
        check(d > 1e-3, "a non-zero traction moves the residual");

        // Only the selected faces. One face under traction must move strictly less than
        // all six, which is what says the mask is read per face rather than as a flag.
        scalar_t r_one[CVFEM_HEX8_N_DOF];
        bd.tmask = 0x02;  // x-max alone
        g_bd     = bd;
        residual_of(s, 0x3F, 0x3F, r_one);
        scalar_t d_one = 0, d_all = 0;
        for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) {
            d_one = std::max(d_one, std::fabs(r_one[i] - r_zero[i]));
            d_all = std::max(d_all, std::fabs(r_t[i] - r_zero[i]));
        }
        check(d_one > 1e-3 && d_one < d_all, "tmask selects which faces carry the traction");
        bd.tmask = 0x3F;
        g_bd     = bd;

        int            row = -1, col = -1;
        const scalar_t worst = compare(s, 0x3F, 0x3F, &row, &col);
        std::printf("traction faces:    worst |assembled - FD| = %.3e  (row %d, col %d)\n", (double)worst, row, col);
        check(worst < tol, "the Jacobian still matches FD with a prescribed traction");
        g_bd = {};
    }

    // 5. Prescribed pressure. The face behaves like a closed one with p_bar substituted, so
    //    the residual must move with p_bar and the Jacobian must lose its pressure column
    //    -- p_bar is data, not an unknown. Both are checked against FD, which is what would
    //    catch the column being kept.
    {
        Hex8BoundaryDataT<scalar_t> bd{};
        bd.pmask = 0x02;  // the x-max face only, so the others stay closed
        bd.p_bar = 2.5;
        g_bd     = bd;

        scalar_t r_p[CVFEM_HEX8_N_DOF], r_closed[CVFEM_HEX8_N_DOF];
        residual_of(s, 0x3F, 0, r_p);
        g_bd = {};
        residual_of(s, 0x3F, 0, r_closed);
        scalar_t d = 0;
        for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) d = std::max(d, std::fabs(r_p[i] - r_closed[i]));
        check(d > 1e-3, "a prescribed pressure moves the residual");

        g_bd = bd;
        int            row = -1, col = -1;
        const scalar_t worst = compare(s, 0x3F, 0, &row, &col);
        std::printf("pressure faces:    worst |assembled - FD| = %.3e  (row %d, col %d)\n", (double)worst, row, col);
        check(worst < tol, "the assembled Jacobian matches FD with a prescribed pressure");

        // And the action, on the same configuration.
        scalar_t ux[8], uy[8], uz[8], vx[8], vy[8], vz[8], q[8], adj[9], det, jv[CVFEM_HEX8_N_DOF];
        for (int a = 0; a < 8; ++a) {
            ux[a] = s.u[a * 4 + 0]; uy[a] = s.u[a * 4 + 1]; uz[a] = s.u[a * 4 + 2];
            vx[a] = 0.7 - 0.3 * s.x[a]; vy[a] = -0.2 + 0.4 * s.y[a];
            vz[a] = 0.15 * s.z[a];      q[a]  = 0.5 - 0.25 * s.x[a];
        }
        cvfem_hex8_affine_adj(s.x, s.y, s.z, adj, &det);
        std::memset(jv, 0, sizeof(jv));
        boundary_scs_add_jacobian_action(RHO, MU, 0, adj, det, L, L, L, s.x, s.y, s.z, ux, uy, uz, vx, vy, vz, q,
                                         jv, 0x3F, 0, g_bd);
        State sp = s, sm = s;
        const scalar_t h = 1e-6;
        for (int a = 0; a < 8; ++a) {
            sp.u[a * 4 + 0] += h * vx[a]; sm.u[a * 4 + 0] -= h * vx[a];
            sp.u[a * 4 + 1] += h * vy[a]; sm.u[a * 4 + 1] -= h * vy[a];
            sp.u[a * 4 + 2] += h * vz[a]; sm.u[a * 4 + 2] -= h * vz[a];
            sp.u[a * 4 + 3] += h * q[a];  sm.u[a * 4 + 3] -= h * q[a];
        }
        scalar_t rp[CVFEM_HEX8_N_DOF], rm[CVFEM_HEX8_N_DOF];
        residual_of(sp, 0x3F, 0, rp);
        residual_of(sm, 0x3F, 0, rm);
        scalar_t worst_a = 0;
        for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i)
            worst_a = std::max(worst_a, std::fabs(jv[i] - (rp[i] - rm[i]) / (2 * h)));
        std::printf("pressure faces:    worst |J v - FD|       = %.3e\n", (double)worst_a);
        check(worst_a < tol, "the Jacobian action matches FD with a prescribed pressure");

        // nmask must win where both select a face, or the two treatments would compound.
        g_bd.pmask = 0x3F;
        scalar_t r_both[CVFEM_HEX8_N_DOF], r_nat[CVFEM_HEX8_N_DOF];
        residual_of(s, 0x3F, 0x3F, r_both);
        g_bd = {};
        residual_of(s, 0x3F, 0x3F, r_nat);
        d = 0;
        for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) d = std::max(d, std::fabs(r_both[i] - r_nat[i]));
        check(d == 0.0, "nmask wins where nmask and pmask select the same face");
    }

    if (g_failures) {
        std::fprintf(stderr, "cvfem_boundary_jacobian_test: %d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all boundary Jacobian checks passed\n");
    return 0;
}
