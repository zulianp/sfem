// The playground with a rigid obstacle: a free block dropped onto a hemisphere.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_ElasticityCase.hpp"
#include "sfem_Playground.hpp"
#include "sfem_playground_test_history.hpp"
#include "sfem_test.hpp"

namespace {
    const char *const DROP = R"(
mesh:
  box: {n: [4, 4, 4], min: [0, 0, 0.05], max: [2, 1, 1]}
  packed: false
material:
  type: GeneratedMooneyRivlinProny
  C10: 4000
  C01: 2000
  K: 160000
  relaxation: {iso: {prony: [{g: 0.15, tau: 0.1}, {g: 0.15, tau: 1.0}, {g: 0.10, tau: 10.0}, {g: 0.05, tau: 100.0}]}}
loads:
  body_force: {density: 10, g: [0, 0, -9.81]}
obstacle: {backend: node_sdf, nodes: skin, direction: z, shape: {type: hemisphere, plane: 0}}
regime: dynamic
dynamics: {scheme: newmark, density: 10}
time: {dt: 0.01, t_end: 0.3}
solver: {newton: {max_it: 19, atol: 1e-8, rtol: 1e-9, verbose: false}}
output:
  probes: [{type: contact}]
)";

    std::unique_ptr<sfem::Playground> assemble(const char *yaml, const std::vector<std::string> &overrides) {
        auto config = sfem::ElasticityCase::parse(yaml, "", overrides);
        if (!config) return nullptr;
        return sfem::Playground::create(sfem::Communicator::world(), *config);
    }
}  // namespace

// Every step converges on KKT. The block falls 0.05 in free flight, so it first touches at
// t = sqrt(0.1 / 9.81) ~ 0.101; then the obstacle pushes it up, its reaction on the block along
// +z. Under Newmark (stabilised) and under BDF2.
int test_block_lands_on_the_hemisphere() {
    for (const std::string scheme : {"newmark", "bdf2"}) {
        const std::string csv = "test_block_lands_on_the_hemisphere_" + scheme + ".csv";
        auto              pg  = assemble(DROP, {"dynamics.scheme=" + scheme, "output.history_csv=" + csv});
        SFEM_TEST_ASSERT(pg != nullptr);
        SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);

        auto   rows  = sfem::test::read_history(csv);
        double first = -1, impulse = 0;
        for (auto &row : rows) {
            if (row["status"] < 0) continue;
            if (row["touched"] > 0 && first < 0) first = row["t"];
            impulse += row["reaction_z"] * row["dt"];
        }
        printf("  %s: first contact at t = %g, impulse along z %g\n", scheme.c_str(), first, impulse);
        SFEM_TEST_ASSERT(first > 0.09 && first < 0.13);
        SFEM_TEST_ASSERT(impulse > 0);
        std::remove(csv.c_str());
    }
    return SFEM_TEST_SUCCESS;
}

// BDF2 retries a failed step at half its length, then grows the steps back at most twofold, the
// ratio its variable-step formula is stable for: the block dropped at dt 0.08 (which BDF2 takes
// whole) with Newton cut at two iterations, so that steps fail and are shortened.
int test_bdf2_steps_grow_back_twofold() {
    const std::string csv = "test_bdf2_steps_grow_back_twofold.csv";
    auto              pg  = assemble(DROP,
                                     {"dynamics.scheme=bdf2", "time.dt=0.08", "time.t_end=0.48", "solver.newton.max_it=2",
                                      "output.history_csv=" + csv});
    SFEM_TEST_ASSERT(pg != nullptr);
    SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);

    double last = 0, shortest = 1;
    int    steps = 0;
    for (auto &row : sfem::test::read_history(csv)) {
        if (row["status"] < 0 || row["step"] < 1) continue;
        if (last > 0) SFEM_TEST_ASSERT(row["dt"] <= 2 * last * (1 + 1e-12));
        shortest = std::min(shortest, row["dt"]);
        last     = row["dt"];
        steps++;
    }
    printf("  %d steps, the shortest %g, the last %g\n", steps, shortest, last);
    SFEM_TEST_ASSERT(shortest < 0.08 * 0.75);
    std::remove(csv.c_str());
    return SFEM_TEST_SUCCESS;
}

#ifdef SFEM_ENABLE_CUDA
// The same drop on the device: the contact, the QP and the time scheme there, and the run the
// host's to rounding.
int test_device_matches_host() {
    double first[2] = {-1, -1}, impulse[2] = {0, 0};
    int    newton[2] = {0, 0};
    for (int device = 0; device < 2; ++device) {
        const std::string csv = std::string("test_device_matches_host_") + (device ? "device" : "host") + ".csv";
        auto              pg  = assemble(DROP, {"output.history_csv=" + csv, std::string("execution_space=") + (device ? "device" : "host")});
        SFEM_TEST_ASSERT(pg != nullptr);
        SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);
        for (auto &row : sfem::test::read_history(csv)) {
            if (row["status"] < 0) continue;
            if (row["touched"] > 0 && first[device] < 0) first[device] = row["t"];
            impulse[device] += row["reaction_z"] * row["dt"];
            newton[device] += (int)row["newton"];
        }
        std::remove(csv.c_str());
    }
    printf("  host: contact at %g, impulse %.12g, %d Newton; device: contact at %g, impulse %.12g, %d Newton\n",
           first[0], impulse[0], newton[0], first[1], impulse[1], newton[1]);
    SFEM_TEST_ASSERT(first[0] == first[1]);
    SFEM_TEST_ASSERT(std::abs(impulse[1] - impulse[0]) <= 1e-6 * std::abs(impulse[0]));
    return SFEM_TEST_SUCCESS;
}
#endif

// The drop with the tangent stored in single precision (solver.linear.storage_bytes 4): the
// material's, the inertia's and the obstacle curvature's shares assembled into floats, on the host
// and the device. The Newton residual stays in double, so every step converges as in double, and
// the contact is the same to float's rounding.
int test_single_precision_tangent() {
#if SFEM_REAL_T_IS_FLOAT32
    return SFEM_TEST_SUCCESS;
#else
    struct Run {
        const char *space;
        int         bytes;
    };
    std::vector<Run> runs = {{"host", 8}, {"host", 4}};
#ifdef SFEM_ENABLE_CUDA
    runs.push_back({"device", 4});
#endif
    std::vector<double> first(runs.size(), -1), impulse(runs.size(), 0);
    for (size_t k = 0; k < runs.size(); ++k) {
        const std::string csv = std::string("test_single_precision_tangent_") + runs[k].space + std::to_string(runs[k].bytes) + ".csv";
        auto pg = assemble(DROP, {"output.history_csv=" + csv, std::string("execution_space=") + runs[k].space,
                                  "solver.linear.storage_bytes=" + std::to_string(runs[k].bytes)});
        SFEM_TEST_ASSERT(pg != nullptr);
        SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);
        for (auto &row : sfem::test::read_history(csv)) {
            if (row["status"] < 0) continue;
            if (row["touched"] > 0 && first[k] < 0) first[k] = row["t"];
            impulse[k] += row["reaction_z"] * row["dt"];
        }
        std::remove(csv.c_str());
        printf("  %s, %d bytes: contact at %g, impulse %.12g\n", runs[k].space, runs[k].bytes, first[k], impulse[k]);
    }
    for (size_t k = 1; k < runs.size(); ++k) {
        SFEM_TEST_ASSERT(first[k] == first[0]);
        SFEM_TEST_ASSERT(std::abs(impulse[k] - impulse[0]) <= 1e-6 * std::abs(impulse[0]));
    }
    return SFEM_TEST_SUCCESS;
#endif
}

// The ssmgc back-end on the same drop, linear and refined: in flight (t < 0.1) the body falls as it
// does with node_sdf, the two obstacle steps solving the same unconstrained problem.
int test_ssmgc_free_flight_matches_node_sdf() {
    const std::vector<std::string> linear = {"material={type: GeneratedLinearElasticity, mu: 12000, lmbda: 152000}",
                                             "mesh.refine=2",
                                             "time.t_end=0.05",
                                             "output.probes=[{type: control_point, at: [1, 0.5, 0.05]}]"};
    double uz[2] = {0, 0};
    for (int backend = 0; backend < 2; ++backend) {
        const std::string csv = std::string("test_ssmgc_free_flight_") + (backend ? "ssmgc" : "node_sdf") + ".csv";
        auto overrides = linear;
        overrides.push_back(std::string("obstacle.backend=") + (backend ? "ssmgc" : "node_sdf"));
        overrides.push_back("output.history_csv=" + csv);
        auto pg = assemble(DROP, overrides);
        SFEM_TEST_ASSERT(pg != nullptr);
        SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);
        for (auto &row : sfem::test::read_history(csv)) uz[backend] = row["uz"];
        std::remove(csv.c_str());
    }
    printf("  u_z at t = 0.05: node_sdf %.12g, ssmgc %.12g\n", uz[0], uz[1]);
    SFEM_TEST_ASSERT(uz[0] < -0.009);
    SFEM_TEST_ASSERT(std::abs(uz[1] - uz[0]) <= 1e-8 * std::abs(uz[0]));
    return SFEM_TEST_SUCCESS;
}

#ifdef SFEM_ENABLE_CUDA
// The ssmgc back-end through the landing on the device: the generated material's micro-element
// hierarchy, SSMGC and the step's contact loops there, every step's Newton count and the impulse
// the host's.
int test_ssmgc_device_matches_host() {
    const std::vector<std::string> linear = {"material={type: GeneratedLinearElasticity, mu: 12000, lmbda: 152000}",
                                             "mesh.refine=2",
                                             "obstacle.backend=ssmgc",
                                             "time.t_end=0.15"};
    double              impulse[2] = {0, 0};
    std::vector<double> newton[2];
    for (int device = 0; device < 2; ++device) {
        const std::string csv = std::string("test_ssmgc_device_matches_host_") + (device ? "device" : "host") + ".csv";
        auto overrides = linear;
        overrides.push_back("output.history_csv=" + csv);
        overrides.push_back(std::string("execution_space=") + (device ? "device" : "host"));
        auto pg = assemble(DROP, overrides);
        SFEM_TEST_ASSERT(pg != nullptr);
        SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);
        for (auto &row : sfem::test::read_history(csv)) {
            if (row["status"] < 0) continue;
            impulse[device] += row["reaction_z"] * row["dt"];
            newton[device].push_back(row["newton"]);
        }
        std::remove(csv.c_str());
    }
    printf("  ssmgc impulse: host %.12g, device %.12g\n", impulse[0], impulse[1]);
    SFEM_TEST_ASSERT(impulse[0] > 0);
    SFEM_TEST_ASSERT(newton[0] == newton[1]);
    SFEM_TEST_ASSERT(std::abs(impulse[1] - impulse[0]) <= 1e-6 * std::abs(impulse[0]));
    return SFEM_TEST_SUCCESS;
}
#endif

// SSMGC coarsens the material and the mesh: a nonlinear material, a mesh that is not refined, or
// the high-order basis is refused.
int test_ssmgc_refusals() {
    SFEM_TEST_ASSERT(assemble(DROP, {"obstacle.backend=ssmgc", "mesh.refine=2",
                                     "material={type: GeneratedNeoHookeanOgden, mu: 12000, lmbda: 152000}"}) == nullptr);
    SFEM_TEST_ASSERT(assemble(DROP, {"obstacle.backend=ssmgc", "material={type: GeneratedLinearElasticity}"}) == nullptr);
    SFEM_TEST_ASSERT(assemble(DROP, {"obstacle.backend=ssmgc", "material={type: GeneratedLinearElasticity}", "mesh.refine=2",
                                     "mesh.basis=high_order"}) == nullptr);
    return SFEM_TEST_SUCCESS;
}

// A traction of 1 on a disc of the block's top face, radius 0.3 about its centre: the plane
// restricted by center and radius, so the obstacle carries pi 0.3^2 of load (to the staircase of
// the faces' barycentres), not the whole face's 1. A center without a radius is refused.
int test_disc_traction() {
    const char *const yaml = R"(
mesh:
  box: {n: [20, 20, 4]}
  sidesets:
    - {name: top, plane: {axis: z, at: max}}
    - {name: disc, plane: {axis: z, at: max, center: [0.5, 0.5, 1], radius: 0.3}}
material: {type: GeneratedNeoHookeanOgden, mu: 1, lmbda: 2}
dirichlet_conditions:
  - {type: sideset, format: name, name: top, component: [0, 1], value: [0, 0]}
loads:
  neumann:
    - {type: sideset, format: name, name: disc, component: 2, value: -1}
obstacle: {backend: node_sdf, direction: z, shape: {type: plane, plane: 0}}
regime: steady
solver: {newton: {verbose: false}}
output:
  probes: [{type: contact}]
)";
    std::string no_radius = yaml;
    no_radius.replace(no_radius.find(", radius: 0.3"), std::string(", radius: 0.3").size(), "");
    SFEM_TEST_ASSERT(assemble(no_radius.c_str(), {}) == nullptr);
    const std::string csv = "test_disc_traction.csv";
    auto              pg  = assemble(yaml, {"output.history_csv=" + csv});
    SFEM_TEST_ASSERT(pg != nullptr);
    SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);
    auto         rows = sfem::test::read_history(csv);
    const double disc = M_PI * 0.09;
    printf("  reaction_z %.6g, disc pi 0.3^2 = %.6g\n", rows[1]["reaction_z"], disc);
    SFEM_TEST_ASSERT(std::abs(rows[1]["reaction_z"] - disc) < 0.05 * disc);
    std::remove(csv.c_str());
    return SFEM_TEST_SUCCESS;
}

// A block clamped at its top whose base starts 0.1 inside a plane, with no load: at the default
// gap_tol Newton pushes it out onto the plane; with obstacle.gap_tol = 0.2, more than the
// penetration, the start is admissible and stays as it is.
int test_gap_tol() {
    const char *const yaml = R"(
mesh:
  box: {n: [4, 4, 4]}
  sidesets: [{name: top, plane: {axis: z, at: max}}]
material: {type: GeneratedNeoHookeanOgden, mu: 1, lmbda: 2}
dirichlet_conditions:
  - {type: sideset, format: name, name: top, component: [0, 1, 2], value: [0, 0, 0]}
obstacle: {backend: node_sdf, direction: z, shape: {type: plane, plane: 0.1}}
regime: quasi_static
time: {dt: 1, t_end: 1}
solver: {newton: {verbose: false}}
)";
    real_t lowest[2] = {1, 1};
    for (int loose = 0; loose < 2; ++loose) {
        auto pg = assemble(yaml, loose ? std::vector<std::string>{"obstacle.gap_tol=0.2"} : std::vector<std::string>{});
        SFEM_TEST_ASSERT(pg != nullptr);
        SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);
        auto       mesh = pg->function()->space()->mesh_ptr();
        const auto z    = mesh->points()->data()[2];
        const auto u    = pg->solution()->data();
        for (ptrdiff_t i = 0; i < mesh->n_nodes(); ++i) lowest[loose] = std::min(lowest[loose], z[i] + u[i * 3 + 2]);
    }
    printf("  lowest point %.12g at the default gap_tol, %.12g at 0.2 (plane 0.1)\n", (double)lowest[0], (double)lowest[1]);
    SFEM_TEST_ASSERT(std::abs(lowest[0] - real_t(0.1)) < 1e-8);
    SFEM_TEST_ASSERT(lowest[1] == 0);
    return SFEM_TEST_SUCCESS;
}

// A block pressed onto a plane by a traction of 0.5 on its unit top face, held there only
// laterally: in a static step the obstacle's force is the whole reaction, and it balances the load,
// steady and quasi-static alike.
int test_static_reaction_balances_load() {
    const char *const yaml = R"(
mesh:
  box: {n: [4, 4, 4]}
  sidesets: [{name: top, plane: {axis: z, at: max}}]
material: {type: GeneratedNeoHookeanOgden, mu: 1, lmbda: 2}
dirichlet_conditions:
  - {type: sideset, format: name, name: top, component: [0, 1], value: [0, 0]}
loads:
  neumann:
    - {type: sideset, format: name, name: top, component: 2, value: -0.5}
obstacle: {backend: node_sdf, direction: z, shape: {type: plane, plane: 0}}
time: {dt: 1, t_end: 1}
solver: {newton: {verbose: false}}
output:
  probes: [{type: contact}]
)";
    for (const std::string regime : {"steady", "quasi_static"}) {
        const std::string csv = "test_static_reaction_" + regime + ".csv";
        auto              pg  = assemble(yaml, {"regime=" + regime, "output.history_csv=" + csv});
        SFEM_TEST_ASSERT(pg != nullptr);
        SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);
        auto rows = sfem::test::read_history(csv);
        SFEM_TEST_ASSERT(rows.size() == 2);
        printf("  %s: reaction_z %.12g (load 0.5)\n", regime.c_str(), rows[1]["reaction_z"]);
        SFEM_TEST_ASSERT(std::abs(rows[1]["reaction_z"] - 0.5) < 1e-8);
        std::remove(csv.c_str());
    }
    // Under inertia, loaded at once, the block rings and the scheme's dissipation settles it: then
    // the obstacle's force is the load. BDF2's reaction is the force; damped Newmark's force is 2
    // beta of its reaction (ContactStabilisedNewmarkScheme).
    const std::vector<std::vector<std::string>> dynamic = {
            {"dynamics={scheme: bdf2, density: 1}"},
            {"dynamics={scheme: newmark, density: 1, gamma: 0.6, beta: 0.3025}"},
    };
    for (const auto &scheme : dynamic) {
        const std::string csv       = "test_static_reaction_dynamic.csv";
        auto              overrides = scheme;
        overrides.insert(overrides.end(), {"regime=dynamic", "time={dt: 1, t_end: 120}", "output.history_csv=" + csv});
        auto pg = assemble(yaml, overrides);
        SFEM_TEST_ASSERT(pg != nullptr);
        SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);
        auto rows = sfem::test::read_history(csv);
        printf("  %s: reaction_z %.12g after %zu steps (load 0.5)\n", scheme[0].c_str(), rows.back()["reaction_z"], rows.size() - 1);
        SFEM_TEST_ASSERT(std::abs(rows.back()["reaction_z"] - 0.5) < 1e-6);
        std::remove(csv.c_str());
    }
    return SFEM_TEST_SUCCESS;
}

// A block clamped at its top whose base starts 0.1 inside a plane: the release lets it out in four
// quasi-static steps, each a converged equilibrium; at the end the base rests on the plane.
int test_release_from_penetration() {
    const char *const yaml = R"(
mesh:
  box: {n: [4, 4, 4]}
  sidesets: [{name: top, plane: {axis: z, at: max}}]
material: {type: GeneratedNeoHookeanOgden, mu: 1, lmbda: 2}
dirichlet_conditions:
  - {type: sideset, format: name, name: top, component: [0, 1, 2], value: [0, 0, 0]}
obstacle:
  backend: node_sdf
  direction: z
  shape: {type: plane, plane: 0.1}
  release: {type: linear_ramp, start_time: 0, end_time: 1, start_value: 0, end_value: 1}
regime: quasi_static
time: {dt: 0.25, t_end: 1, max_cutbacks: 4}
solver: {newton: {verbose: false}}
)";
    auto pg = assemble(yaml, {});
    SFEM_TEST_ASSERT(pg != nullptr);
    SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);

    auto       mesh   = pg->function()->space()->mesh_ptr();
    const auto z      = mesh->points()->data()[2];
    const auto u      = pg->solution()->data();
    real_t     lowest = 1;
    for (ptrdiff_t i = 0; i < mesh->n_nodes(); ++i) lowest = std::min(lowest, z[i] + u[i * 3 + 2]);
    printf("  lowest point %.12g (plane 0.1)\n", (double)lowest);
    SFEM_TEST_ASSERT(std::abs(lowest - real_t(0.1)) < 1e-8);
    return SFEM_TEST_SUCCESS;
}

// Utopia's scheme: the box 10% inside the plane, pushed out at once by four Newton steps damped by
// 0.8, the obstacle's curvature left out. The four steps end the solve: the plane's gap is linear,
// so each step's QP puts the box on it and 0.2^4 of the penetration is left. Without
// fixed_iterations the same four steps are a failed solve. A QP its max_it stops is the step: no
// retry, so the QP's iterations stay within its cap.
int test_utopia_scheme() {
    const char *const yaml = R"(
mesh:
  box: {n: [4, 4, 4]}
  sidesets: [{name: top, plane: {axis: z, at: max}}]
material: {type: GeneratedNeoHookeanOgden, mu: 1, lmbda: 2}
dirichlet_conditions:
  - {type: sideset, format: name, name: top, component: [0, 1, 2], value: [0, 0, 0]}
obstacle:
  backend: node_sdf
  direction: z
  shape: {type: plane, plane: 0.1}
  curvature: false
regime: steady
solver: {newton: {max_it: 4, fixed_iterations: true, verbose: false, line_search: {rule: damped, damping: 0.8}}}
)";
    SFEM_TEST_ASSERT(assemble(yaml, {"solver.newton.fixed_iterations=false"})->run() != SFEM_SUCCESS);

    auto pg = assemble(yaml, {});
    SFEM_TEST_ASSERT(pg != nullptr);
    SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);
    auto       mesh   = pg->function()->space()->mesh_ptr();
    const auto z      = mesh->points()->data()[2];
    const auto u      = pg->solution()->data();
    real_t     lowest = 1;
    for (ptrdiff_t i = 0; i < mesh->n_nodes(); ++i) lowest = std::min(lowest, z[i] + u[i * 3 + 2]);
    printf("  lowest point after four damped steps %.12g (expected %.12g)\n", (double)lowest, 0.1 - 0.1 * 0.0016);
    SFEM_TEST_ASSERT(std::abs(lowest - real_t(0.1 - 0.1 * 0.0016)) < 1e-8);

    const std::string history = "sfem_utopia_scheme_history.csv";
    auto capped = assemble(yaml, {"obstacle.qp={max_it: 2, accept_capped: true}", "output.history_csv=" + history});
    SFEM_TEST_ASSERT(capped != nullptr);
    SFEM_TEST_ASSERT(capped->run() == SFEM_SUCCESS);
    int linear = 0;
    for (auto &row : sfem::test::read_history(history)) {
        if (row["status"] >= 0) linear += (int)row["linear"];
    }
    printf("  QP iterations with max_it 2: %d over four steps\n", linear);
    SFEM_TEST_ASSERT(linear > 0 && linear <= 4 * 2);
    std::remove(history.c_str());
    return SFEM_TEST_SUCCESS;
}

// Under inertia node_sdf takes Newmark, which it stabilises, or BDF2; backward Euler, which has no
// inertia, is refused, and ssmgc takes Newmark alone. Without inertia node_sdf solves steady and
// quasi-static contact; ssmgc does not.
int test_obstacle_needs_newmark() {
    SFEM_TEST_ASSERT(assemble(DROP, {"dynamics.scheme=bdf2"}) != nullptr);
    SFEM_TEST_ASSERT(assemble(DROP, {"dynamics.scheme=backward_euler"}) == nullptr);
    SFEM_TEST_ASSERT(assemble(DROP, {"dynamics.scheme=bdf2", "obstacle.backend=ssmgc", "mesh.refine=2",
                                     "material={type: GeneratedLinearElasticity}"}) == nullptr);
    SFEM_TEST_ASSERT(assemble(DROP, {"regime=quasi_static"}) != nullptr);
    SFEM_TEST_ASSERT(assemble(DROP, {"regime=quasi_static", "obstacle.backend=ssmgc", "mesh.refine=2",
                                     "material={type: GeneratedLinearElasticity}"}) == nullptr);
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_block_lands_on_the_hemisphere);
    SFEM_RUN_TEST(test_bdf2_steps_grow_back_twofold);
    SFEM_RUN_TEST(test_single_precision_tangent);
    SFEM_RUN_TEST(test_obstacle_needs_newmark);
    SFEM_RUN_TEST(test_static_reaction_balances_load);
    SFEM_RUN_TEST(test_gap_tol);
    SFEM_RUN_TEST(test_disc_traction);
    SFEM_RUN_TEST(test_release_from_penetration);
    SFEM_RUN_TEST(test_utopia_scheme);
    SFEM_RUN_TEST(test_ssmgc_free_flight_matches_node_sdf);
    SFEM_RUN_TEST(test_ssmgc_refusals);
#ifdef SFEM_ENABLE_CUDA
    SFEM_RUN_TEST(test_device_matches_host);
    SFEM_RUN_TEST(test_ssmgc_device_matches_host);
#endif
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
