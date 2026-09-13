// The flow diagnostics must be right before anything is judged on them.
//
// The kinetic-energy budget is the gate for the turbulent step case, and its headline number
// -- the numerical dissipation -- is defined as what the other terms leave over. A residual
// definition inherits every error in the terms it is subtracting from, so each of those has
// to be checked against a closed form first, on fields where the closed form exists.
//
// Three things are checked, in increasing order of what they can hide:
//
//   1. the nodal velocity gradient, on a globally linear field, where the reconstruction is
//      exact and any disagreement is a wiring error rather than a discretisation one;
//   2. the contraction -- dissipation, enstrophy, divergence -- against values computed by
//      hand from a gradient this test constructs, so the contraction is tested apart from
//      the reconstruction that feeds it;
//   3. both together on a divergence-free trigonometric field, where the dissipation has a
//      closed form and the divergence must go to zero under refinement rather than being
//      zero outright.
//
// The energy budget itself is NOT tested here. It needs a converged solve, so its control --
// laminar Poiseuille, where dE/dt is zero and the flow is resolved -- belongs in the
// verification matrix beside the other cases that need a driver run.

#include "cvfem_flow_diagnostics.hpp"
#include "cvfem_hex8_ns_op.hpp"

#include "sfem_context.hpp"
#include "smesh_mesh.hpp"
#include "smesh_sideset.hpp"
#include "smesh_semistructured.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

static int g_failures = 0;

static void check(const bool ok, const char *what) {
    std::printf("%-62s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) ++g_failures;
}

static void check_close(const double got, const double want, const double tol, const char *what) {
    const double scale = std::fmax(std::fabs(want), 1e-300);
    const double rel   = std::fabs(got - want) / scale;
    std::printf("%-62s %s  (got %.9e want %.9e rel %.2e)\n", what, rel <= tol ? "OK" : "FAIL", got, want, rel);
    if (!(rel <= tol)) ++g_failures;
}

// Names the assertion with the arm it came from, so a failure says which mesh it was on.
static const char *msg(const char *tag, const char *what) {
    static char buf[128];
    std::snprintf(buf, sizeof(buf), "%s: %s", tag, what);
    return buf;
}

static constexpr int    N_FIELDS = 4;
static constexpr double LX = 1.0, LY = 1.0, LZ = 1.0;
static constexpr double RHO = 1.3, MU = 0.07;

struct Setup {
    std::shared_ptr<smesh::Mesh>              mesh;
    std::shared_ptr<sfem::CVFEMNavierStokes>  op;
    ptrdiff_t                                 nnodes{0};
    std::vector<real_t>                       x, g, vol;
};

static void build(Setup &s, sfem::Context &ctx, const int n, const int ss_level = 1) {
    s.mesh = smesh::Mesh::create_hex8_cube(ctx.communicator(), n, n, n, 0, 0, 0, LX, LY, LZ);
    if (ss_level > 1) {
        s.mesh = smesh::to_semistructured(ss_level, s.mesh, true, false);
        if (!s.mesh) {
            check(false, "to_semistructured built a mesh");
            return;
        }
    }
    auto fs = sfem::FunctionSpace::create(s.mesh, N_FIELDS);
    s.op    = std::make_shared<sfem::CVFEMNavierStokes>(fs);
    s.op->rho  = RHO;
    s.op->mu   = MU;
    s.op->geom = sfem::CVFEMGeometry::Affine;
    s.op->initialize();  // renumbers nodes; every coordinate read below must come after it
    s.nnodes = s.mesh->n_nodes();
    s.x.assign((size_t)s.nnodes * N_FIELDS, 0);
    s.g.assign((size_t)s.nnodes * 9, 0);
    s.vol.assign((size_t)s.nnodes, 0);
    s.op->node_volume(s.vol.data());
}

// ---------------------------------------------------------------- 1. linear field
//
// u = (a0 + a1 x + a2 y + a3 z, ...) has a constant gradient the reconstruction reproduces
// exactly -- tests/cvfem_nodal_grad_test asserts that property for a scalar, and this is the
// statement that the three-component wrapper did not transpose or misorder anything.
static void test_linear(sfem::Context &ctx, const int ss_level = 1) {
    // The whole body is level-agnostic on purpose. A globally linear field has the same
    // constant gradient whatever the mesh, so the semi-structured arm is the identical
    // assertion against the identical closed form -- which is what makes it a wiring check
    // for sscvfem_nodal_grad_strided rather than a second, weaker test.
    char tag[64];
    std::snprintf(tag, sizeof(tag), ss_level > 1 ? "linear/ss%d" : "linear", ss_level);

    Setup s;
    // Fewer macro elements at level 4 so the node count, and the run time, stay comparable
    // with the flat arm: 6^3 flat against 2^3 macros of 4^3 micro-cells.
    build(s, ctx, ss_level > 1 ? 2 : 6, ss_level);
    if (!s.mesh) return;
    const auto *const px = s.mesh->points()->data()[0];
    const auto *const py = s.mesh->points()->data()[1];
    const auto *const pz = s.mesh->points()->data()[2];

    // Deliberately asymmetric, and with every one of the nine entries distinct, so a
    // transposed or rotated index cannot pass.
    const double A[9] = {0.31, -0.17, 0.44, 0.52, 0.13, -0.28, -0.09, 0.36, 0.21};
    for (ptrdiff_t i = 0; i < s.nnodes; ++i) {
        const double X = px[i], Y = py[i], Z = pz[i];
        for (int r = 0; r < 3; ++r)
            s.x[(size_t)i * N_FIELDS + r] = (real_t)(A[r * 3 + 0] * X + A[r * 3 + 1] * Y + A[r * 3 + 2] * Z);
        s.x[(size_t)i * N_FIELDS + 3] = (real_t)0;
    }
    check(s.op->nodal_velocity_gradient(s.x.data(), s.g.data()) == SFEM_SUCCESS,
          msg(tag, "nodal_velocity_gradient succeeds"));

    double worst = 0;
    for (ptrdiff_t i = 0; i < s.nnodes; ++i)
        for (int k = 0; k < 9; ++k) worst = std::fmax(worst, std::fabs((double)s.g[(size_t)i * 9 + k] - A[k]));
    std::printf("%-62s %s  (worst %.3e)\n", msg(tag, "gradient exact at every node and entry"),
                worst < 1e-11 ? "OK" : "FAIL", worst);
    if (!(worst < 1e-11)) ++g_failures;

    // And the contraction of a known constant gradient, against arithmetic done by hand.
    const double s00 = A[0], s11 = A[4], s22 = A[8];
    const double s01 = 0.5 * (A[1] + A[3]), s02 = 0.5 * (A[2] + A[6]), s12 = 0.5 * (A[5] + A[7]);
    const double SS  = s00 * s00 + s11 * s11 + s22 * s22 + 2 * (s01 * s01 + s02 * s02 + s12 * s12);
    const double V   = LX * LY * LZ;
    const auto   st  = cvfem_diag::contract(s.nnodes, s.x.data(), s.g.data(), s.vol.data(), RHO, MU, 0.0);
    check_close(st.eps_visc, 2 * MU * SS * V, 1e-10, msg(tag, "dissipation is 2 mu S:S times the volume"));
    check_close(st.div_inf, std::fabs(A[0] + A[4] + A[8]), 1e-10, msg(tag, "divergence is the trace"));

    const double wx = A[7] - A[5], wy = A[2] - A[6], wz = A[3] - A[1];
    check_close(st.omega_max, std::sqrt(wx * wx + wy * wy + wz * wz), 1e-10, msg(tag, "vorticity is curl u"));
}

// ---------------------------------------------------------------- 2. CFL
static void test_cfl(sfem::Context &ctx) {
    Setup s;
    build(s, ctx, 4);
    for (ptrdiff_t i = 0; i < s.nnodes; ++i) {
        s.x[(size_t)i * N_FIELDS + 0] = (real_t)3.0;  // |u| = 5 with the next line
        s.x[(size_t)i * N_FIELDS + 1] = (real_t)4.0;
    }
    const double dt = 0.01;
    const auto   st = cvfem_diag::contract(s.nnodes, s.x.data(), s.g.data(), s.vol.data(), RHO, MU, dt);
    check_close(st.u_max, 5.0, 1e-12, "cfl: |u| is the magnitude, not a component");
    // The interior control volume is a full cell; corners and edges are fractions of one, so
    // the largest CFL comes from the smallest volume. Assert against that rather than h.
    double vmin = 1e300;
    for (ptrdiff_t i = 0; i < s.nnodes; ++i) vmin = std::fmin(vmin, (double)s.vol[(size_t)i]);
    check_close(st.cfl_max, 5.0 * dt / std::cbrt(vmin), 1e-12, "cfl: uses the smallest control volume");

    const auto st0 = cvfem_diag::contract(s.nnodes, s.x.data(), s.g.data(), s.vol.data(), RHO, MU, 0.0);
    check(st0.cfl_max == 0.0, "cfl: a steady run reports zero rather than dividing by dt");
}

// ---------------------------------------------------------------- 3. a solenoidal field
//
// u = (sin kx cos ky, -cos kx sin ky, 0) on the unit box with k = 2 pi.
//
//   S_00 = -S_11 = k cos kx cos ky,  S_01 = 0 because the two shear terms cancel exactly,
//   so  S:S = 2 k^2 cos^2 kx cos^2 ky  and  integral of 2 mu S:S over the box = mu k^2.
//
// Two separate statements, because they fail for different reasons. The dissipation is a
// discretisation error and must converge at second order. The divergence is analytically
// zero AND reconstructs to round-off on this field -- the element-wise divergences cancel --
// so the only honest assertion is that it stays at round-off; a convergence ratio on a
// quantity already at 1e-16 measures nothing. Divergence on a field where it is genuinely
// nonzero is covered by the linear test above, which asserts it equals the trace.
static void test_trig(sfem::Context &ctx) {
    const double k    = 2.0 * M_PI;
    const double want = MU * k * k;
    double       prev_err = 0;
    for (int pass = 0; pass < 2; ++pass) {
        const int n = pass == 0 ? 16 : 32;
        Setup     s;
        build(s, ctx, n);
        const auto *const px = s.mesh->points()->data()[0];
        const auto *const py = s.mesh->points()->data()[1];
        for (ptrdiff_t i = 0; i < s.nnodes; ++i) {
            const double X = px[i], Y = py[i];
            s.x[(size_t)i * N_FIELDS + 0] = (real_t)(std::sin(k * X) * std::cos(k * Y));
            s.x[(size_t)i * N_FIELDS + 1] = (real_t)(-std::cos(k * X) * std::sin(k * Y));
        }
        s.op->nodal_velocity_gradient(s.x.data(), s.g.data());
        const auto st = cvfem_diag::contract(s.nnodes, s.x.data(), s.g.data(), s.vol.data(), RHO, MU, 0.0);

        char msg[160];
        std::snprintf(msg, sizeof(msg), "trig n=%d: div_l2 at round-off (%.2e)", n, st.div_l2);
        check(st.div_l2 < 1e-12, msg);

        const double err = std::fabs(st.eps_visc - want) / want;
        std::snprintf(msg, sizeof(msg), "trig n=%d: dissipation err %.3f (eps %.6f, want %.6f)", n,
                      err, st.eps_visc, want);
        std::printf("%-62s\n", msg);
        if (pass == 1) {
            check(err < 0.05, "trig: dissipation within 5% of the closed form at n=32");
            // Second order is 4x. Accept 3x so a mesh-dependent constant cannot fail a
            // correct implementation, while a first-order or broken one -- which would sit
            // near 2x or 1x -- still shows up.
            const double ratio = prev_err / std::fmax(err, 1e-300);
            std::printf("%-62s %s  (ratio %.2f)\n", "trig: dissipation error converges at second order",
                        ratio > 3.0 ? "OK" : "FAIL", ratio);
            if (!(ratio > 3.0)) ++g_failures;
        }
        prev_err = err;
    }
}

// ------------------------------------------------- 4. the weighted boundary flux, both paths
//
// The energy budget's P_in and P_out are sideset_flux_weighted with w = |u|^2/2 + p/rho, and
// that call used to be refused on a semi-structured mesh. It is the one term of the budget
// that is an integral over a surface rather than a sum over nodes, so it is the one that can
// disagree between the two paths without anything else noticing.
//
// Checked two ways on the same cube. First the UNWEIGHTED flux against a closed form: with a
// uniform u = (U,0,0) the flux through the x = LX face is exactly rho U Ly Lz, whatever the
// mesh. Then the weighted flux against w times that, using a CONSTANT weight -- which makes
// the expected value exact, so a disagreement is the weighting and not the field.
static void test_sideset_flux(sfem::Context &ctx, const int ss_level) {
    char tag[64];
    std::snprintf(tag, sizeof(tag), ss_level > 1 ? "flux/ss%d" : "flux/flat", ss_level);

    Setup s;
    build(s, ctx, ss_level > 1 ? 2 : 6, ss_level);
    if (!s.mesh) return;

    // The outlet plane has to exist as a named sideset before the operator is initialized,
    // the same order the driver uses.
    const double U = 1.7, P = 0.3, W = 2.9;
    for (ptrdiff_t i = 0; i < s.nnodes; ++i) {
        s.x[(size_t)i * N_FIELDS + 0] = (real_t)U;
        s.x[(size_t)i * N_FIELDS + 1] = 0;
        s.x[(size_t)i * N_FIELDS + 2] = 0;
        s.x[(size_t)i * N_FIELDS + 3] = (real_t)P;
    }

    auto outs = smesh::Sideset::create_from_plane(s.mesh, 1, 0, 0, (smesh::geom_t)LX, 1e-6);
    if (outs.empty() || !outs.front()) {
        check(false, msg(tag, "outlet sideset built"));
        return;
    }
    s.mesh->add_sideset("outlet", outs.front());

    real_t q = 0;
    const int rc = s.op->sideset_mass_flux(s.x.data(), "outlet", q);
    check(rc == SFEM_SUCCESS, msg(tag, "unweighted flux succeeds"));
    if (rc == SFEM_SUCCESS) check_close((double)q, RHO * U * LY * LZ, 1e-12, msg(tag, "flux is rho U Ly Lz"));

    std::vector<real_t> w((size_t)s.nnodes, (real_t)W);
    real_t              qw = 0;
    const int           rw = s.op->sideset_flux_weighted(s.x.data(), "outlet", w.data(), qw);
    check(rw == SFEM_SUCCESS, msg(tag, "weighted flux succeeds"));
    if (rw == SFEM_SUCCESS)
        check_close((double)qw, W * RHO * U * LY * LZ, 1e-12, msg(tag, "constant weight scales the flux"));
}

int main(int argc, char **argv) {
    sfem::Context ctx(argc, argv);
    test_linear(ctx);
    // The semi-structured arm. The budget it feeds has to work where the solver is run, and
    // FGMRES preconditioned by multigrid needs a micro-element lattice, so a diagnostic that
    // only worked flat would report on a configuration nobody uses.
    test_linear(ctx, 4);
    test_cfl(ctx);
    test_trig(ctx);
    test_sideset_flux(ctx, 1);
    test_sideset_flux(ctx, 4);
    std::printf("\n%s\n", g_failures ? "FAILED" : "PASSED");
    return g_failures ? 1 : 0;
}
