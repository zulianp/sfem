// The generated Jacobian action, against the hand-written one.
//
// HEX8 had no generated Jacobian action until now: the four `sympy*` kernel names covered
// the residual and the assembly, and no `apply_jacobian_action_*` took a kernel selector,
// so no CSE arrangement had ever been measured for the operation a Krylov solve spends
// its time in. Four now exist, differing only in the scope each `sp.cse` call was given:
// the whole kernel, one node, one component, or one sub-control surface.
//
// They are only worth measuring if they are the same operator, and "the same operator" is
// a stronger claim than it looks. The generated kernels treat the upwind sign as a frozen
// input -- `sgn0..sgn11`, evaluated by the emitted prologue -- because that is what makes
// the flux algebra differentiable and the kernel generatable at all. The hand-written
// action freezes it too, via `cvfem_upwind_abs`, so the two agree; the generated assembly
// rests on the same assumption and is checked the same way. If that ever stops being true
// this test is what says so.
//
// Three separate things are checked, because passing the first two is easy by accident:
//
//   * each arrangement agrees with the hand-written action on a general state;
//   * the arrangements agree with EACH OTHER to round-off -- they are one
//     expression tree cut into different CSE scopes, so any disagreement is a generator
//     bug rather than a slow variant;
//   * the action really is linear in the direction, which no comparison against another
//     hand-written kernel can establish.

#include "smesh_types.hpp"

#include "cvfem_portability.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>

using scalar_t = double;
static constexpr int N_FIELDS = 4;

#include "cvfem_hex8_ns_upwind_kernels.hpp"
#include "cvfem_hex8_ns_upwind_sympy_kernels.hpp"

static int g_failures = 0;

static void check(const bool ok, const char *what, const double got = 0.0) {
    std::printf("%-58s %-12.3e %s\n", what, got, ok ? "OK" : "FAIL");
    if (!ok) ++g_failures;
}

static scalar_t max_abs_diff(const scalar_t *a, const scalar_t *b, const int n) {
    scalar_t m = 0;
    for (int i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

static scalar_t max_abs(const scalar_t *a, const int n) {
    scalar_t m = 0;
    for (int i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i]));
    return m;
}

// A general state on a warped cube. Warped rather than a unit cube so the adjugate has no
// zero entries: on an axis-aligned element most cofactors vanish and a term dropped by a
// wrong CSE scope would never show up.
static void state(scalar_t x[8], scalar_t y[8], scalar_t z[8], scalar_t ux[8], scalar_t uy[8],
                  scalar_t uz[8], scalar_t p[8], scalar_t vx[8], scalar_t vy[8], scalar_t vz[8],
                  scalar_t q[8]) {
    static const scalar_t cx[8] = {0, 1, 1, 0, 0, 1, 1, 0};
    static const scalar_t cy[8] = {0, 0, 1, 1, 0, 0, 1, 1};
    static const scalar_t cz[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    for (int a = 0; a < 8; ++a) {
        x[a]  = cx[a] + scalar_t(0.11) * cy[a] - scalar_t(0.07) * cz[a];
        y[a]  = cy[a] + scalar_t(0.05) * cz[a] + scalar_t(0.13) * cx[a];
        z[a]  = cz[a] - scalar_t(0.09) * cx[a] + scalar_t(0.06) * cy[a];
        ux[a] = scalar_t(0.7) + scalar_t(0.31) * x[a] - scalar_t(0.22) * y[a] + scalar_t(0.4) * z[a];
        uy[a] = scalar_t(-0.4) + scalar_t(0.2) * x[a] + scalar_t(0.53) * y[a] - scalar_t(0.17) * z[a];
        uz[a] = scalar_t(0.2) - scalar_t(0.11) * x[a] + scalar_t(0.25) * z[a] + scalar_t(0.3) * y[a];
        p[a]  = scalar_t(1.0) + scalar_t(0.1) * x[a] + scalar_t(0.2) * y[a] - scalar_t(0.15) * z[a];
        // A direction unrelated to the state, so a kernel that confused the two fails.
        vx[a] = scalar_t(0.13) - scalar_t(0.4) * y[a] + scalar_t(0.29) * z[a];
        vy[a] = scalar_t(-0.21) + scalar_t(0.33) * x[a] - scalar_t(0.12) * z[a];
        vz[a] = scalar_t(0.37) + scalar_t(0.18) * y[a] - scalar_t(0.26) * x[a];
        q[a]  = scalar_t(0.44) - scalar_t(0.19) * x[a] + scalar_t(0.23) * y[a];
    }
}

int main() {
    scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8], vx[8], vy[8], vz[8], q[8];
    state(x, y, z, ux, uy, uz, p, vx, vy, vz, q);

    scalar_t adj[9], det;
    cvfem_hex8_affine_adj(x, y, z, adj, &det);
    const scalar_t rho = 1.0, mu = 0.037;

    scalar_t ref[CVFEM_HEX8_N_DOF], flat[CVFEM_HEX8_N_DOF], nodew[CVFEM_HEX8_N_DOF],
            compw[CVFEM_HEX8_N_DOF], facew[CVFEM_HEX8_N_DOF], geomw[CVFEM_HEX8_N_DOF],
            geomf[CVFEM_HEX8_N_DOF];

    cvfem_hex8_ns_upwind_jacobian_action(rho, mu, adj, det, ux, uy, uz, vx, vy, vz, q, ref);
    cvfem_hex8_ns_upwind_sympy_jacobian_action(rho, mu, adj, det, ux, uy, uz, vx, vy, vz, q, flat);
    cvfem_hex8_ns_upwind_sympy_jacobian_action_nodewise(rho, mu, adj, det, ux, uy, uz, vx, vy, vz, q, nodew);
    cvfem_hex8_ns_upwind_sympy_jacobian_action_componentwise(rho, mu, adj, det, ux, uy, uz, vx, vy, vz, q, compw);
    cvfem_hex8_ns_upwind_sympy_jacobian_action_facewise(rho, mu, adj, det, ux, uy, uz, vx, vy, vz, q, facew);
    cvfem_hex8_ns_upwind_sympy_jacobian_action_geom(rho, mu, adj, det, ux, uy, uz, vx, vy, vz, q, geomw);
    cvfem_hex8_ns_upwind_sympy_jacobian_action_geomface(rho, mu, adj, det, ux, uy, uz, vx, vy, vz, q, geomf);

    const scalar_t scale = max_abs(ref, CVFEM_HEX8_N_DOF);
    check(scale > scalar_t(1e-6), "the reference action is not trivially zero", (double)scale);

    // 1e-12 relative rather than round-off: the two arrangements sum the same terms in a
    // different order, so they are equal in exact arithmetic and close, not identical, in
    // floating point.
    const scalar_t tol = scalar_t(1e-12) * scale;
    check(max_abs_diff(ref, flat, CVFEM_HEX8_N_DOF) <= tol,
          "flat CSE agrees with the hand-written action",
          (double)max_abs_diff(ref, flat, CVFEM_HEX8_N_DOF));
    check(max_abs_diff(ref, nodew, CVFEM_HEX8_N_DOF) <= tol,
          "node-wise CSE agrees with the hand-written action",
          (double)max_abs_diff(ref, nodew, CVFEM_HEX8_N_DOF));
    check(max_abs_diff(ref, compw, CVFEM_HEX8_N_DOF) <= tol,
          "component-wise CSE agrees with the hand-written action",
          (double)max_abs_diff(ref, compw, CVFEM_HEX8_N_DOF));

    check(max_abs_diff(ref, facew, CVFEM_HEX8_N_DOF) <= tol,
          "face-wise CSE agrees with the hand-written action",
          (double)max_abs_diff(ref, facew, CVFEM_HEX8_N_DOF));

    check(max_abs_diff(ref, geomw, CVFEM_HEX8_N_DOF) <= tol,
          "geometry-hoisted CSE agrees with the hand-written action",
          (double)max_abs_diff(ref, geomw, CVFEM_HEX8_N_DOF));
    check(max_abs_diff(ref, geomf, CVFEM_HEX8_N_DOF) <= tol,
          "hoisted face-wise CSE agrees with the hand-written action",
          (double)max_abs_diff(ref, geomf, CVFEM_HEX8_N_DOF));

    // The arrangements against each other. One expression tree, six cuts.
    check(max_abs_diff(flat, nodew, CVFEM_HEX8_N_DOF) <= tol,
          "flat and node-wise agree with each other",
          (double)max_abs_diff(flat, nodew, CVFEM_HEX8_N_DOF));
    check(max_abs_diff(flat, compw, CVFEM_HEX8_N_DOF) <= tol,
          "flat and component-wise agree with each other",
          (double)max_abs_diff(flat, compw, CVFEM_HEX8_N_DOF));
    check(max_abs_diff(flat, facew, CVFEM_HEX8_N_DOF) <= tol,
          "flat and face-wise agree with each other",
          (double)max_abs_diff(flat, facew, CVFEM_HEX8_N_DOF));

    // Linearity in the direction. The action is J(u) v with u fixed, so doubling v must
    // double the result exactly -- and a kernel that accidentally read the state where it
    // meant the direction would fail this while passing everything above.
    {
        scalar_t v2x[8], v2y[8], v2z[8], q2[8], twice[CVFEM_HEX8_N_DOF];
        for (int a = 0; a < 8; ++a) {
            v2x[a] = 2 * vx[a];
            v2y[a] = 2 * vy[a];
            v2z[a] = 2 * vz[a];
            q2[a]  = 2 * q[a];
        }
        cvfem_hex8_ns_upwind_sympy_jacobian_action(rho, mu, adj, det, ux, uy, uz, v2x, v2y, v2z, q2, twice);
        scalar_t worst = 0;
        for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i)
            worst = std::max(worst, std::fabs(twice[i] - 2 * flat[i]));
        check(worst <= tol, "the action is linear in the direction", (double)worst);
    }

    // A zero direction must give exactly zero, which catches a state-dependent term that
    // leaked into the action.
    {
        scalar_t zero[8] = {0}, out[CVFEM_HEX8_N_DOF];
        cvfem_hex8_ns_upwind_sympy_jacobian_action(rho, mu, adj, det, ux, uy, uz, zero, zero, zero, zero, out);
        check(max_abs(out, CVFEM_HEX8_N_DOF) == scalar_t(0),
              "a zero direction gives exactly zero", (double)max_abs(out, CVFEM_HEX8_N_DOF));
    }

    if (g_failures) {
        std::fprintf(stderr, "cvfem_sympy_action_test: %d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all generated Jacobian-action checks passed\n");
    return 0;
}
