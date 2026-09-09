// Global mass conservation, against a closed form rather than against itself.
//
// The step case already checks this at the solver level -- it sums the continuity residual
// over every node and compares against the exact inflow flux 1/9, currently agreeing to
// 7e-14 -- but that check lives inside one driver, runs only for one case, and needs a
// converged solve to reach. This asserts the same property where it originates, on one
// element with a velocity field whose divergence is known exactly, so it runs in
// microseconds and a failure names the term at fault.
//
// The two halves are separate properties and both matter:
//
//   * The interior sub-control-surface fluxes must telescope. Every face adds its flux at
//     one node and subtracts it at the other, so summed over all nodes the volume kernel
//     contributes exactly zero to continuity. A sign error or a mis-paired node shows up
//     here and nowhere else -- it leaves the residual plausible and the mass wrong.
//
//   * The boundary closure must carry the true flux. Summed over the closed surface of a
//     single element, the boundary term's continuity contribution is the net outward mass
//     flux, which by the divergence theorem is div(u) times the volume. That is a real
//     number here, not zero, so the test cannot pass by everything being empty.

#include "smesh_types.hpp"

#include "cvfem_portability.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>

using scalar_t = double;
static constexpr int N_FIELDS = 4;

#include "cvfem_hex8_ns_upwind_kernels.hpp"
#include "cvfem_hex8_boundary_scs.hpp"

static int g_failures = 0;

static void check_close(const scalar_t got, const scalar_t want, const scalar_t tol, const char *what) {
    const scalar_t err = std::fabs(got - want);
    std::printf("%-56s got %+.15e  want %+.15e  %s\n", what, (double)got, (double)want,
                err <= tol ? "OK" : "FAIL");
    if (!(err <= tol)) ++g_failures;
}

// A linear velocity field on the unit cube. Linear so its divergence is a constant and the
// flux through the surface is exactly div(u) * V; the CVFEM face quadrature is exact for it,
// which is what makes the comparison a closed form rather than an approximation.
//
//   ux = 1.00 + 0.30 x - 0.20 y          d/dx = 0.30
//   uy = -0.40 + 0.20 x + 0.50 y         d/dy = 0.50
//   uz = 0.20 - 0.10 x + 0.25 z          d/dz = 0.25
//
// so div(u) = 1.05 and, on the unit cube, the net outward mass flux is rho * 1.05.
static constexpr scalar_t DIV_U = 0.30 + 0.50 + 0.25;
static constexpr scalar_t RHO   = 1.0;
static constexpr scalar_t MU    = 0.01;

static void unit_cube_state(scalar_t x[8], scalar_t y[8], scalar_t z[8], scalar_t ux[8], scalar_t uy[8],
                            scalar_t uz[8], scalar_t p[8]) {
    static const scalar_t cx[8] = {0, 1, 1, 0, 0, 1, 1, 0};
    static const scalar_t cy[8] = {0, 0, 1, 1, 0, 0, 1, 1};
    static const scalar_t cz[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    for (int a = 0; a < 8; ++a) {
        x[a]  = cx[a];
        y[a]  = cy[a];
        z[a]  = cz[a];
        ux[a] = 1.00 + 0.30 * x[a] - 0.20 * y[a];
        uy[a] = -0.40 + 0.20 * x[a] + 0.50 * y[a];
        uz[a] = 0.20 - 0.10 * x[a] + 0.25 * z[a];
        p[a]  = 1.00 + 0.10 * x[a] + 0.20 * y[a];
    }
}

static scalar_t sum_continuity(const scalar_t *r) {
    scalar_t s = 0;
    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) s += r[a * 4 + 3];
    return s;
}

int main() {
    scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8], adj[9], det;
    unit_cube_state(x, y, z, ux, uy, uz, p);
    cvfem_hex8_affine_adj(x, y, z, adj, &det);

    // 1. The interior fluxes telescope to zero.
    {
        scalar_t r[CVFEM_HEX8_N_DOF];
        cvfem_hex8_ns_upwind_residual_sumfact(RHO, MU, adj, det, ux, uy, uz, p, r);
        check_close(sum_continuity(r), 0.0, 1e-14, "interior fluxes telescope to zero");
    }

    // 2. The closed boundary carries div(u) * V.
    {
        scalar_t r[CVFEM_HEX8_N_DOF];
        std::memset(r, 0, sizeof(r));
        boundary_scs_add_residual(RHO, MU, 0, adj, det, 1.0, 1.0, 1.0, x, y, z, ux, uy, uz, p, r, 0x3F, 0);
        check_close(sum_continuity(r), RHO * DIV_U, 1e-13, "closed boundary carries div(u) * V");
    }

    // 3. Interior plus boundary is the whole operator's continuity balance, and it must
    //    still be div(u) * V -- the property the step case checks after a solve.
    {
        scalar_t r[CVFEM_HEX8_N_DOF];
        cvfem_hex8_ns_upwind_residual_sumfact(RHO, MU, adj, det, ux, uy, uz, p, r);
        boundary_scs_add_residual(RHO, MU, 0, adj, det, 1.0, 1.0, 1.0, x, y, z, ux, uy, uz, p, r, 0x3F, 0);
        check_close(sum_continuity(r), RHO * DIV_U, 1e-13, "interior + boundary balances");
    }

    // 4. A divergence-free field closes to zero. This is the case the solver is actually
    //    in once continuity is satisfied, and it is the one where a sign error hides:
    //    with div(u) = 0 the answer is zero whether or not the fluxes cancel correctly,
    //    so it is checked *alongside* the non-zero case above, never instead of it.
    {
        scalar_t dx[8], dy[8], dz[8], dp[8];
        for (int a = 0; a < 8; ++a) {
            // div = 0.4 - 0.4 + 0 = 0, and not constant in any one direction.
            dx[a] = 0.4 * x[a] + 0.7 * y[a];
            dy[a] = 0.3 * x[a] - 0.4 * y[a];
            dz[a] = 0.9 * x[a] - 0.2 * y[a];
            dp[a] = 0.0;
        }
        scalar_t r[CVFEM_HEX8_N_DOF];
        cvfem_hex8_ns_upwind_residual_sumfact(RHO, MU, adj, det, dx, dy, dz, dp, r);
        boundary_scs_add_residual(RHO, MU, 0, adj, det, 1.0, 1.0, 1.0, x, y, z, dx, dy, dz, dp, r, 0x3F, 0);
        check_close(sum_continuity(r), 0.0, 1e-14, "a divergence-free field closes to zero");
    }

    // 5. The do-nothing outflow keeps the true flux in the continuity row. The momentum
    //    rows clamp the convective term with max(mdot, 0) to guard against backflow, and
    //    clipping continuity too would destroy exactly the conservation checked above --
    //    so the balance must be unchanged by which faces are declared outflow.
    {
        scalar_t r[CVFEM_HEX8_N_DOF];
        cvfem_hex8_ns_upwind_residual_sumfact(RHO, MU, adj, det, ux, uy, uz, p, r);
        boundary_scs_add_residual(RHO, MU, 0, adj, det, 1.0, 1.0, 1.0, x, y, z, ux, uy, uz, p, r, 0x3F, 0x02);
        check_close(sum_continuity(r), RHO * DIV_U, 1e-13, "a do-nothing face keeps its true flux");

        scalar_t r_all[CVFEM_HEX8_N_DOF];
        cvfem_hex8_ns_upwind_residual_sumfact(RHO, MU, adj, det, ux, uy, uz, p, r_all);
        boundary_scs_add_residual(RHO, MU, 0, adj, det, 1.0, 1.0, 1.0, x, y, z, ux, uy, uz, p, r_all, 0x3F, 0x3F);
        check_close(sum_continuity(r_all), RHO * DIV_U, 1e-13, "all faces do-nothing keeps the balance");
    }

    if (g_failures) {
        std::fprintf(stderr, "cvfem_conservation_test: %d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all conservation checks passed\n");
    return 0;
}
