// Small cases through the whole playground, one per regime.

#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

#include "sfem_API.hpp"
#include "sfem_ElasticityCase.hpp"
#include "sfem_Function.hpp"
#include "sfem_Playground.hpp"
#include "sfem_test.hpp"

namespace {
    std::unique_ptr<sfem::Playground> assemble(const char *yaml, const std::vector<std::string> &overrides = {}) {
        auto config = sfem::ElasticityCase::parse(yaml, "", overrides);
        if (!config) return nullptr;
        return sfem::Playground::create(sfem::Communicator::world(), *config);
    }

    /// The rows of a step history CSV, each column by name.
    std::vector<std::map<std::string, double>> read_history(const std::string &path) {
        std::ifstream                              is(path);
        std::string                                line, cell;
        std::vector<std::string>                   names;
        std::vector<std::map<std::string, double>> rows;
        std::getline(is, line);
        for (std::stringstream ss(line); std::getline(ss, cell, ',');) names.push_back(cell);
        while (std::getline(is, line)) {
            std::map<std::string, double> row;
            std::stringstream             ss(line);
            for (const auto &name : names) {
                std::getline(ss, cell, ',');
                row[name] = name == "status" ? (cell == "initial" ? -1 : 0) : std::stod(cell);
            }
            rows.push_back(row);
        }
        return rows;
    }
}  // namespace

// Linear elasticity with the affine displacement u = A X on the whole boundary: trilinear elements
// represent it exactly, so the interior nodes take it to the solver's tolerance.
int test_steady_linear_patch() {
    const char *const yaml = R"(
mesh:
  box: {n: [3, 3, 3]}
  sidesets:
    - {name: x0, plane: {axis: x, at: min}}
    - {name: x1, plane: {axis: x, at: max}}
    - {name: y0, plane: {axis: y, at: min}}
    - {name: y1, plane: {axis: y, at: max}}
    - {name: z0, plane: {axis: z, at: min}}
    - {name: z1, plane: {axis: z, at: max}}
material: {type: GeneratedLinearElasticity, mu: 1, lmbda: 2}
regime: steady
solver: {newton: {verbose: false, rtol: 1e-12}}
)";
    // The boundary values are set below, node by node, from u = A X.
    auto pg = assemble(yaml);
    SFEM_TEST_ASSERT(pg != nullptr);

    const real_t A[3][3] = {{0.01, 0.02, -0.01}, {0.0, -0.015, 0.005}, {0.02, 0.0, 0.01}};
    auto         f       = pg->function();
    auto         mesh    = f->space()->mesh_ptr();
    auto         xyz     = mesh->points()->data();
    const auto   n       = mesh->n_nodes();

    std::vector<idx_t> boundary;
    for (ptrdiff_t i = 0; i < n; ++i) {
        bool on = false;
        for (int d = 0; d < 3; ++d) on = on || std::abs(xyz[d][i]) < 1e-8 || std::abs(xyz[d][i] - 1) < 1e-8;
        if (on) boundary.push_back((idx_t)i);
    }
    auto dc = std::make_shared<sfem::DirichletConditions>(f->space());
    for (int d = 0; d < 3; ++d) {
        auto values = (real_t *)malloc(boundary.size() * sizeof(real_t));
        auto nodes  = (idx_t *)malloc(boundary.size() * sizeof(idx_t));
        for (size_t k = 0; k < boundary.size(); ++k) {
            const idx_t i = boundary[k];
            nodes[k]      = i;
            values[k]     = A[d][0] * xyz[0][i] + A[d][1] * xyz[1][i] + A[d][2] * xyz[2][i];
        }
        dc->add_condition(boundary.size(), boundary.size(), nodes, d, values);
    }
    f->add_constraint(dc);

    SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);
    const real_t *const u = pg->solution()->data();
    for (ptrdiff_t i = 0; i < n; ++i) {
        for (int d = 0; d < 3; ++d) {
            const real_t exact = A[d][0] * xyz[0][i] + A[d][1] * xyz[1][i] + A[d][2] * xyz[2][i];
            SFEM_TEST_ASSERT(std::abs(u[i * 3 + d] - exact) < 1e-10);
        }
    }
    return SFEM_TEST_SUCCESS;
}

const char *const STRETCH = R"(
mesh:
  box: {n: [3, 3, 3]}
  sidesets:
    - {name: left, plane: {axis: x, at: min}}
    - {name: right, plane: {axis: x, at: max}}
material: {type: GeneratedNeoHookeanOgden, mu: 1, lmbda: 2}
dirichlet_conditions:
  - {type: sideset, format: name, name: left, component: [0, 1, 2], value: [0, 0, 0]}
  - type: sideset
    format: name
    name: right
    component: 0
    value: 0.4
    profile: {type: linear_ramp, start_time: 0, end_time: 1, start_value: 0, end_value: 1}
regime: quasi_static
time: {dt: 0.25, t_end: 1}
solver: {newton: {verbose: false}}
)";

// The load ramps with time and every step converges; the stretched face ends where it was put.
int test_quasi_static_stretch() {
    auto pg = assemble(STRETCH);
    SFEM_TEST_ASSERT(pg != nullptr);
    SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);

    auto       mesh = pg->function()->space()->mesh_ptr();
    auto       xyz  = mesh->points()->data();
    const auto u    = pg->solution()->data();
    real_t worst = 0;
    int    on_face = 0;
    for (ptrdiff_t i = 0; i < mesh->n_nodes(); ++i) {
        if (std::abs(xyz[0][i] - 1) < 1e-6) {
            worst = std::max(worst, std::abs(u[i * 3] - real_t(0.4)));
            on_face++;
        }
    }
    printf("  %d nodes on the pulled face, largest deviation %g\n", on_face, (double)worst);
    SFEM_TEST_ASSERT(on_face == 16);
    SFEM_TEST_ASSERT(worst < 1e-14);
    return SFEM_TEST_SUCCESS;
}

// A cantilever dropped under gravity, by each scheme with inertia: every step converges, and the
// tip falls.
int test_dynamic_cantilever() {
    const char *const yaml = R"(
mesh:
  box: {n: [6, 2, 2], min: [0, 0, 0], max: [3, 1, 1]}
  sidesets:
    - {name: clamp, plane: {axis: x, at: min}}
material: {type: GeneratedNeoHookeanOgden, mu: 50, lmbda: 100}
loads:
  body_force: {density: 1, g: [0, -1, 0]}
dirichlet_conditions:
  - {type: sideset, format: name, name: clamp, component: [0, 1, 2], value: [0, 0, 0]}
regime: dynamic
time: {dt: 0.05, steps: 5}
dynamics: {scheme: newmark, density: 1}
solver: {newton: {verbose: false}}
)";
    for (const char *scheme : {"newmark", "bdf2"}) {
        auto pg = assemble(yaml, {std::string("dynamics.scheme=") + scheme});
        SFEM_TEST_ASSERT(pg != nullptr);
        SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);

        auto   mesh = pg->function()->space()->mesh_ptr();
        auto   xyz  = mesh->points()->data();
        real_t tip  = 0;
        for (ptrdiff_t i = 0; i < mesh->n_nodes(); ++i) {
            if (std::abs(xyz[0][i] - 3) < 1e-8) tip = std::min(tip, pg->solution()->data()[i * 3 + 1]);
        }
        printf("  %s: tip deflection %g\n", scheme, (double)tip);
        SFEM_TEST_ASSERT(tip < 0);
    }
    return SFEM_TEST_SUCCESS;
}

// A bar twisted at one end by a rotation about its axis, ramped over two quasi-static steps: the
// twisted face sits at the rigid rotation of its reference positions.
int test_rotation_constraint() {
    const char *const yaml = R"(
mesh:
  box: {n: [4, 2, 2], min: [0, 0, 0], max: [2, 1, 1]}
  sidesets:
    - {name: left, plane: {axis: x, at: min}}
    - {name: right, plane: {axis: x, at: max}}
material: {type: GeneratedNeoHookeanOgden, mu: 1, lmbda: 2}
dirichlet_conditions:
  - {type: sideset, format: name, name: left, component: [0, 1, 2], value: [0, 0, 0]}
constraints:
  - type: rotation
    sideset: right
    axis: x
    center: [0, 0.5, 0.5]
    angle: 0.3
    profile: {type: linear_ramp, start_time: 0, end_time: 2, start_value: 0, end_value: 1}
regime: quasi_static
time: {dt: 1, steps: 2}
solver: {newton: {verbose: false}}
)";
    auto pg = assemble(yaml);
    SFEM_TEST_ASSERT(pg != nullptr);
    SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);

    auto         mesh = pg->function()->space()->mesh_ptr();
    auto         xyz  = mesh->points()->data();
    const auto   u    = pg->solution()->data();
    const real_t c = std::cos(real_t(0.3)), s = std::sin(real_t(0.3));
    int          twisted = 0;
    for (ptrdiff_t i = 0; i < mesh->n_nodes(); ++i) {
        if (std::abs(xyz[0][i] - 2) > 1e-6) continue;
        const real_t q1 = xyz[1][i] - real_t(0.5), q2 = xyz[2][i] - real_t(0.5);
        SFEM_TEST_ASSERT(std::abs(u[i * 3 + 1] - (c * q1 - s * q2 - q1)) < 1e-12);
        SFEM_TEST_ASSERT(std::abs(u[i * 3 + 2] - (s * q1 + c * q2 - q2)) < 1e-12);
        twisted++;
    }
    SFEM_TEST_ASSERT(twisted == 9);
    return SFEM_TEST_SUCCESS;
}

// A sine profile: the twisted face follows angle * sin(2 pi t / period).
int test_sine_rotation() {
    const char *const yaml = R"(
mesh:
  box: {n: [4, 2, 2], min: [0, 0, 0], max: [2, 1, 1]}
  sidesets:
    - {name: left, plane: {axis: x, at: min}}
    - {name: right, plane: {axis: x, at: max}}
material: {type: GeneratedNeoHookeanOgden, mu: 1, lmbda: 2}
dirichlet_conditions:
  - {type: sideset, format: name, name: left, component: [0, 1, 2], value: [0, 0, 0]}
constraints:
  - {type: rotation, sideset: right, axis: x, center: [0, 0.5, 0.5], angle: 0.3, profile: {type: sine, period: 4}}
regime: quasi_static
time: {dt: 1, steps: 3}
solver: {newton: {verbose: false}}
)";
    auto pg = assemble(yaml);
    SFEM_TEST_ASSERT(pg != nullptr);
    SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);

    // At t = 3 the angle is 0.3 sin(3 pi / 2) = -0.3.
    auto         mesh = pg->function()->space()->mesh_ptr();
    auto         xyz  = mesh->points()->data();
    const auto   u    = pg->solution()->data();
    const real_t c = std::cos(real_t(-0.3)), s = std::sin(real_t(-0.3));
    int          twisted = 0;
    for (ptrdiff_t i = 0; i < mesh->n_nodes(); ++i) {
        if (std::abs(xyz[0][i] - 2) > 1e-6) continue;
        const real_t q1 = xyz[1][i] - real_t(0.5), q2 = xyz[2][i] - real_t(0.5);
        SFEM_TEST_ASSERT(std::abs(u[i * 3 + 1] - (c * q1 - s * q2 - q1)) < 1e-12);
        SFEM_TEST_ASSERT(std::abs(u[i * 3 + 2] - (s * q1 + c * q2 - q2)) < 1e-12);
        twisted++;
    }
    SFEM_TEST_ASSERT(twisted == 9);
    return SFEM_TEST_SUCCESS;
}

const char *const PRONY_TORSION = R"(
mesh:
  box: {n: [6, 2, 2], min: [0, 0, 0], max: [3, 1, 1]}
  packed: false
  sidesets:
    - {name: left, plane: {axis: x, at: min}}
    - {name: right, plane: {axis: x, at: max}}
material:
  type: GeneratedMooneyRivlinProny
  C10: 0.3
  C01: 0.05
  K: 5
  relaxation: {iso: {prony: [{g: 0.5, tau: 1}]}}
dirichlet_conditions:
  - {type: sideset, format: name, name: left, component: [0, 1, 2], value: [0, 0, 0]}
regime: quasi_static
time: {dt: 0.5, t_end: 4}
solver: {newton: {verbose: false, atol: 1e-11, rtol: 0}}
output:
  probes: [{type: torque, constraint: 0}]
)";

// A Prony solid twisted, held and released: the torque grows over the ramp, relaxes at constant
// twist, and vanishes with the grip.
int test_prony_torsion_release() {
    const std::string csv = "test_prony_torsion_release.csv";
    auto              pg  = assemble(PRONY_TORSION,
                                     {"output.history_csv=" + csv,
                                      "constraints=[{type: rotation, sideset: right, axis: x, center: [0, 0.5, 0.5], angle: 0.4, "
                                      "profile: {type: linear_ramp, start_time: 0, end_time: 1, start_value: 0, end_value: 1}, "
                                      "release: 2.5}]"});
    SFEM_TEST_ASSERT(pg != nullptr);
    SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);

    auto rows = read_history(csv);
    SFEM_TEST_ASSERT(rows.size() == 9);
    // Rows 1..8 end at t = 0.5, 1, ..., 4: ramp to t = 1, hold to 2, released from 2.5.
    for (int k = 1; k <= 8; ++k) printf("  t = %g: angle %g, released %g, torque %g\n", rows[k]["t"], rows[k]["angle"], rows[k]["released"], rows[k]["torque"]);
    SFEM_TEST_ASSERT(rows[2]["torque"] > rows[1]["torque"] && rows[1]["torque"] > 0);
    SFEM_TEST_ASSERT(rows[3]["torque"] < rows[2]["torque"] && rows[4]["torque"] < rows[3]["torque"]);
    SFEM_TEST_ASSERT(rows[4]["angle"] == 0.4 && rows[4]["released"] == 0);
    SFEM_TEST_ASSERT(rows[5]["released"] == 1 && rows[5]["angle"] == 0);
    for (int k = 5; k <= 8; ++k) SFEM_TEST_ASSERT(std::abs(rows[k]["torque"]) < 1e-6 * rows[2]["torque"]);
    std::remove(csv.c_str());
    return SFEM_TEST_SUCCESS;
}

// The rotation driven by its torque: every step ends with the probe's torque on the ramped
// target, and under the held torque the Prony solid keeps turning.
int test_torque_control() {
    const std::string csv = "test_torque_control.csv";
    const double      T   = 2e-3;
    auto              pg  = assemble(PRONY_TORSION,
                                     {"output.history_csv=" + csv,
                                      "constraints=[{type: rotation, sideset: right, axis: x, center: [0, 0.5, 0.5], "
                                      "control: {torque: 2e-3, tol: 1e-8}, "
                                      "profile: {type: linear_ramp, start_time: 0, end_time: 1, start_value: 0, end_value: 1}}]"});
    SFEM_TEST_ASSERT(pg != nullptr);
    SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);

    auto rows = read_history(csv);
    SFEM_TEST_ASSERT(rows.size() == 9);
    for (int k = 1; k <= 8; ++k) {
        const double target = T * std::min(1.0, rows[k]["t"]);
        printf("  t = %g: angle %g, torque %g (target %g), %g Newton\n", rows[k]["t"], rows[k]["angle"], rows[k]["torque"], target, rows[k]["newton"]);
        SFEM_TEST_ASSERT(std::abs(rows[k]["torque"] - target) <= 1e-8 * target);
    }
    // Creep: the angle grows at constant torque.
    for (int k = 3; k <= 8; ++k) SFEM_TEST_ASSERT(rows[k]["angle"] > rows[k - 1]["angle"]);
    std::remove(csv.c_str());
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_steady_linear_patch);
    SFEM_RUN_TEST(test_quasi_static_stretch);
    SFEM_RUN_TEST(test_dynamic_cantilever);
    SFEM_RUN_TEST(test_rotation_constraint);
    SFEM_RUN_TEST(test_sine_rotation);
    SFEM_RUN_TEST(test_prony_torsion_release);
    SFEM_RUN_TEST(test_torque_control);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
