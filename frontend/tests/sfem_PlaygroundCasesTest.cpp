// Small cases through the whole playground, one per regime.

#include <cmath>
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

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_steady_linear_patch);
    SFEM_RUN_TEST(test_quasi_static_stretch);
    SFEM_RUN_TEST(test_dynamic_cantilever);
    SFEM_RUN_TEST(test_rotation_constraint);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
