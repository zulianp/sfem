// The value-carrying boundary conditions, from the sideset name down to the kernel.
//
// The kernel side of traction and prescribed pressure is checked by
// cvfem_boundary_jacobian_test, which drives Hex8BoundaryDataT directly. Nothing checked
// that a *named sideset* ever reaches it, and until this test there was no way it could:
// CVFEMNavierStokes had the kernels and the MeshData fields and no path between them, so
// setting a boundary condition on the operator did nothing at all and looked exactly like
// a run that had none.
//
// That is the failure this file exists to make impossible, so the assertions are about the
// wiring rather than about the physics: does naming a sideset change the residual, does it
// change it only on the faces named, and does a mistake in the naming stop the run instead
// of being absorbed.

#include "cvfem_hex8_ns_core.hpp"

#include "cvfem_hex8_ns_op.hpp"
#include "sfem_Function.hpp"
#include "sfem_context.hpp"
#include "smesh_mesh.hpp"
#include "smesh_sideset.hpp"

#include <memory>
#include <string>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int g_failures = 0;
// The communicator, kept here rather than threaded through every helper: sfem::Context has
// no accessor for the live instance and every mesh below wants the same one.
static std::shared_ptr<smesh::Communicator> g_comm;

static void check(const bool ok, const char *what) {
    std::printf("%-62s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) ++g_failures;
}

static constexpr int      NX = 6, NY = 4, NZ = 4;
static constexpr scalar_t LX = 3, LY = 1, LZ = 1;
static constexpr scalar_t RHO = 1, MU = 0.05;

// A state that is a function of position, so it does not depend on the node numbering.
static std::vector<real_t> make_state(const std::shared_ptr<smesh::Mesh> &mesh) {
    const ptrdiff_t     n = mesh->n_nodes();
    std::vector<real_t> x((size_t)n * N_FIELDS, 0);
    const auto *const   px = mesh->points()->data()[0];
    const auto *const   py = mesh->points()->data()[1];
    const auto *const   pz = mesh->points()->data()[2];
    for (ptrdiff_t i = 0; i < n; ++i) {
        x[(size_t)i * 4 + 0] = 0.7 + 0.13 * px[i] - 0.21 * py[i];
        x[(size_t)i * 4 + 1] = -0.3 + 0.11 * px[i] + 0.17 * py[i];
        x[(size_t)i * 4 + 2] = 0.2 - 0.07 * px[i] + 0.09 * pz[i];
        x[(size_t)i * 4 + 3] = 1.0 + 0.05 * px[i];
    }
    return x;
}

// One residual, for a given boundary configuration. `configure` is handed the operator
// before initialize(), which is the only point at which these are allowed to be set.
template <typename F>
static bool residual_with(F &&configure, std::vector<real_t> &out, ptrdiff_t &nnodes_out) {
    auto mesh = smesh::Mesh::create_hex8_cube(g_comm, NX, NY, NZ, 0, 0, 0, LX, LY, LZ);

    // Name the two surfaces the test uses. Sidesets rather than coordinate tests, because
    // that is what the operator consumes and what a driver would hand it.
    auto outlet = smesh::Sideset::create_from_plane(mesh, 1, 0, 0, (smesh::geom_t)LX, 1e-6);
    auto top    = smesh::Sideset::create_from_plane(mesh, 0, 1, 0, (smesh::geom_t)LY, 1e-6);
    if (outlet.empty() || top.empty()) return false;
    mesh->add_sideset("outlet", outlet.front());
    mesh->add_sideset("top", top.front());
    mesh->add_sideset("skin", smesh::skin_sideset(mesh));

    auto fs = sfem::FunctionSpace::create(mesh, N_FIELDS);
    auto op = std::make_shared<sfem::CVFEMNavierStokes>(fs);
    op->rho       = RHO;
    op->mu        = MU;
    op->pack_size = 0;
    configure(*op);
    if (op->initialize() != SFEM_SUCCESS) return false;

    const auto x = make_state(mesh);
    nnodes_out   = mesh->n_nodes();
    out.assign((size_t)nnodes_out * N_FIELDS, 0);
    return op->gradient(x.data(), out.data()) == SFEM_SUCCESS;
}

static scalar_t worst_diff(const std::vector<real_t> &a, const std::vector<real_t> &b) {
    scalar_t w = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) w = std::max(w, (scalar_t)std::fabs(a[i] - b[i]));
    return w;
}

// A misconfiguration reaches SFEM_ERROR, which calls MPI_Abort -- the convention
// throughout cvfem_hex8_ns_op.cpp, where the `return SFEM_FAILURE` after it is already
// unreachable. An aborting path cannot be checked in-process, and leaving it unchecked is
// how a refusal quietly stops refusing, so each one runs as a child of this binary and is
// judged on its exit status.
static const char *g_argv0 = nullptr;

static bool refused(const char *case_name) {
    char cmd[1024];
    std::snprintf(cmd, sizeof(cmd), "CVFEM_BC_REFUSAL_CASE=%s '%s' > /dev/null 2>&1", case_name, g_argv0);
    return std::system(cmd) != 0;
}

// One refusal case, run in the child. Returns only if the operator failed to refuse.
static int run_refusal_case(const std::string &name) {
    std::vector<real_t> dummy;
    ptrdiff_t           n = 0;
    if (name == "unknown_sideset") {
        residual_with([](sfem::CVFEMNavierStokes &op) { op.pressure_sideset = "no_such_set"; }, dummy, n);
    } else if (name == "port_is_also_outflow") {
        residual_with(
                [](sfem::CVFEMNavierStokes &op) {
                    op.natural_outflow_sideset = "outlet";
                    op.pressure_sideset        = "outlet";
                    op.pressure_value          = 1;
                },
                dummy, n);
    } else if (name == "no_boundary_mask") {
        setenv("SFEM_BOUNDARY_MASK", "0", 1);
        residual_with(
                [](sfem::CVFEMNavierStokes &op) {
                    op.pressure_sideset = "outlet";
                    op.pressure_value   = 1;
                },
                dummy, n);
    } else {
        std::fprintf(stderr, "unknown refusal case '%s'\n", name.c_str());
        return 2;
    }
    std::fprintf(stderr, "case '%s' was NOT refused\n", name.c_str());
    return 0;  // reaching here means no refusal, which the parent reads as a failure
}

int main(int argc, char **argv) {
    auto ctx = sfem::initialize(argc, argv);
    g_comm   = ctx->communicator();
    g_argv0  = argv[0];

    if (const char *c = std::getenv("CVFEM_BC_REFUSAL_CASE")) {
        setenv("SFEM_BOUNDARY_MASK", "1", 1);
        return run_refusal_case(c);
    }
    // Every one of these conditions is compiled from a topological mask, which is what this
    // switch turns on. The operator refuses the combination rather than running without one.
    setenv("SFEM_BOUNDARY_MASK", "1", 1);

    ptrdiff_t           n = 0;
    std::vector<real_t> r_plain, r_nat, r_trac, r_trac0, r_press;

    // 1. The baseline, and the do-nothing outflow that the value-carrying conditions
    //    generalise. If these two do not already differ, nothing below means anything.
    check(residual_with([](sfem::CVFEMNavierStokes &) {}, r_plain, n), "a closed box assembles");
    check(residual_with([](sfem::CVFEMNavierStokes &op) { op.natural_outflow_sideset = "outlet"; }, r_nat, n),
          "a do-nothing outflow assembles");
    check(worst_diff(r_plain, r_nat) > 1e-6, "the do-nothing outflow changes the residual");

    // 2. Traction. t = 0 on the same faces must reproduce the do-nothing outflow exactly --
    //    that is what makes traction a generalisation of it rather than a second mechanism
    //    -- and a non-zero t must move the residual.
    check(residual_with(
                  [](sfem::CVFEMNavierStokes &op) {
                      op.traction_sideset = "outlet";
                      op.traction[0] = op.traction[1] = op.traction[2] = 0;
                  },
                  r_trac0, n),
          "a zero traction assembles");
    // To round-off, NOT to the bit, and the difference is not the boundary condition.
    // pack_size = 0 is the atomic layout, whose scatter is a `#pragma omp atomic` and so
    // sums each node's contributions in whatever order the threads arrive; two runs of the
    // same operator differ in the last bits. Measured here: 2.08e-17 on this machine's
    // threads, and exactly 0.0 under OMP_NUM_THREADS=1, which is the actual statement --
    // the kernel takes the identical branch and adds a term that is identically zero.
    // Asserting == 0.0 therefore tests thread determinism rather than the condition, and it
    // fails. Please do not tighten it back.
    {
        const scalar_t d = worst_diff(r_nat, r_trac0);
        std::printf("  |zero-traction - do-nothing| = %.3e (atomic scatter reorders; 0 serial)\n", (double)d);
        check(d < 1e-14, "traction t = 0 is the do-nothing outflow, to round-off");
    }

    check(residual_with(
                  [](sfem::CVFEMNavierStokes &op) {
                      op.traction_sideset = "outlet";
                      op.traction[0]      = 0.4;
                      op.traction[1]      = -0.15;
                      op.traction[2]      = 0.05;
                  },
                  r_trac, n),
          "a prescribed traction assembles");
    check(worst_diff(r_trac, r_trac0) > 1e-6, "a prescribed traction reaches the kernel");

    // 3. And only where it is named. A traction on the top face must leave the outlet's
    //    rows alone; before tmask existed one scalar triple applied to every natural face,
    //    so this is the assertion that the mask is per face and not a flag.
    {
        std::vector<real_t> r_two;
        check(residual_with(
                      [](sfem::CVFEMNavierStokes &op) {
                          op.natural_outflow_sideset = "outlet";
                          op.traction_sideset        = "top";
                          op.traction[0]             = 0.4;
                          op.traction[1]             = -0.15;
                          op.traction[2]             = 0.05;
                      },
                      r_two, n),
              "an outflow and a separate traction surface coexist");
        // It differs from the plain outflow (the top is now natural and loaded) and from
        // the case where the same traction sits on the outlet instead.
        check(worst_diff(r_two, r_nat) > 1e-6, "the traction surface changes the residual");
        check(worst_diff(r_two, r_trac) > 1e-6, "traction on 'top' is not traction on 'outlet'");
    }

    // 4. Prescribed pressure.
    check(residual_with(
                  [](sfem::CVFEMNavierStokes &op) {
                      op.pressure_sideset = "outlet";
                      op.pressure_value   = 2.5;
                  },
                  r_press, n),
          "a prescribed pressure assembles");
    check(worst_diff(r_press, r_plain) > 1e-6, "a prescribed pressure reaches the kernel");
    check(worst_diff(r_press, r_nat) > 1e-6, "a pressure port is not a do-nothing outflow");

    // 5. The mistakes must stop the run. Each of these produced a silently wrong answer
    //    before the operator learned to refuse: a misspelled sideset did nothing, a face
    //    named as both port and outflow lost its port to the kernel's tie-break, and a
    //    condition set without SFEM_BOUNDARY_MASK compiled to no mask at all.
    check(refused("unknown_sideset"), "an unknown sideset name is refused");
    check(refused("port_is_also_outflow"), "a face that is both port and outflow is refused");
    check(refused("no_boundary_mask"), "a condition with no mask to compile into is refused");

    // 6. The pressure gauge must stand down exactly when a boundary fixes the level. Both
    //    ways round are silent failures -- over-determined one way, singular the other --
    //    so the operator answers the question rather than leaving the driver to guess.
    {
        auto mesh = smesh::Mesh::create_hex8_cube(g_comm, 2, 2, 2, 0, 0, 0, LX, LY, LZ);
        auto fs   = sfem::FunctionSpace::create(mesh, N_FIELDS);
        sfem::CVFEMNavierStokes op(fs);
        check(!op.fixes_pressure_level(), "a closed box does not fix the pressure level");
        op.natural_outflow_sideset = "outlet";
        check(op.fixes_pressure_level(), "a do-nothing outflow fixes it");
        op.natural_outflow_sideset.clear();
        op.traction_sideset = "top";
        check(op.fixes_pressure_level(), "a traction surface fixes it, whatever its value");
        op.traction_sideset.clear();
        op.pressure_sideset = "outlet";
        check(op.fixes_pressure_level(), "a pressure port fixes it");
    }

    if (g_failures) {
        std::fprintf(stderr, "\ncvfem_bc_sideset_test: %d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("\nall boundary-condition wiring checks passed\n");
    return 0;
}
