// The stored-tangent operator of the rate-dependent materials against their exact apply.
//
// Prony relaxation: the tangent reads the relaxation's step weight and its history at every point,
// so after a few relaxing steps -- history nonzero, weight below one -- the stored f64 apply must
// still be the exact one.  Kelvin-Voigt: the tangent is not symmetric and reads the previous state;
// after a few Newmark steps the same holds, and CG refuses the stored operator.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_ElasticityCase.hpp"
#include "sfem_Function.hpp"
#include "sfem_Playground.hpp"
#include "sfem_test.hpp"

namespace {
    const char *const PRONY_TORSION = R"(
mesh:
  box: {n: [6, 2, 2], min: [0, 0, 0], max: [3, 1, 1]}
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
constraints:
  - {type: rotation, sideset: right, axis: x, center: [0, 0.5, 0.5], angle: 0.4, profile: {type: linear_ramp, start_time: 0, end_time: 1, start_value: 0, end_value: 1}}
regime: quasi_static
time: {dt: 0.5, t_end: 2}
solver: {newton: {verbose: false, atol: 1e-11, rtol: 0}, linear: {preconditioner: none}}
)";

    const char *const KELVIN_VOIGT_PULSE = R"(
mesh:
  box: {n: [6, 3, 3], min: [0, 0, 0], max: [2, 1, 1]}
  sidesets:
    - {name: left, plane: {axis: x, at: min}}
    - {name: right, plane: {axis: x, at: max}}
material: {type: GeneratedMooneyRivlinKelvinVoigt, mu: 1000, lmbda: 1000, eta_s: 0.00005, eta_b: 0}
loads:
  neumann:
    - {type: sideset, format: name, name: right, component: [1], value: [-20]}
  profile: {type: sine, value: 1, period: 0.1, end_time: 0.05}
dirichlet_conditions:
  - {type: sideset, format: name, name: left, component: [0, 1, 2], value: [0, 0, 0]}
regime: dynamic
dynamics: {scheme: newmark, density: 0.01}
time: {dt: 0.001, t_end: 0.005}
solver: {newton: {verbose: false}, linear: {type: bcgs, preconditioner: none}}
)";

    std::unique_ptr<sfem::Playground> assemble(const char *yaml, const std::vector<std::string> &overrides = {}) {
        auto config = sfem::ElasticityCase::parse(yaml, "", overrides);
        if (!config) return nullptr;
        return sfem::Playground::create(sfem::Communicator::world(), *config);
    }

    real_t stored_against_exact(sfem::Playground &pg, const char *const store, const char *const encoding) {
        auto                f = pg.function();
        const ptrdiff_t     n = f->space()->n_dofs();
        std::vector<real_t> u(pg.solution()->data(), pg.solution()->data() + n), h(n), exact(n, 0), stored(n, 0);
        unsigned            seed = 11;
        for (auto &v : h) {
            seed = seed * 1664525u + 1013904223u;
            v    = (real_t)(seed >> 8) / (real_t)(1u << 24) - real_t(0.5);
        }
        if (f->update(u.data()) != SFEM_SUCCESS) return -1;
        if (f->apply(u.data(), h.data(), exact.data()) != SFEM_SUCCESS) return -1;
        f->set_inexact_options(sfem::InexactOptions::from_strings(store, encoding));
        if (f->inexact_update(u.data()) != SFEM_SUCCESS) return -1;
        if (f->inexact_apply(h.data(), stored.data()) != SFEM_SUCCESS) return -1;
        real_t num = 0, den = 0;
        for (ptrdiff_t i = 0; i < n; ++i) {
            num += (exact[i] - stored[i]) * (exact[i] - stored[i]);
            den += exact[i] * exact[i];
        }
        return std::sqrt(num / den);
    }
}  // namespace

int test_prony_stored_tangent() {
    for (const char *t_end : {"1", "2"}) {
        auto pg = assemble(PRONY_TORSION, {std::string("time.t_end=") + t_end});
        SFEM_TEST_ASSERT(pg != nullptr);
        SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);
        for (const char *store : {"points", "element_matrix"}) {
            const real_t error = stored_against_exact(*pg, store, "f64");
            printf("Prony t = %s, %s f64: stored vs exact %.3e\n", t_end, store, (double)error);
            SFEM_TEST_ASSERT(error >= 0 && error < 1e-12);
        }
    }
    return SFEM_TEST_SUCCESS;
}

int test_kelvin_voigt_stored_tangent() {
    auto pg = assemble(KELVIN_VOIGT_PULSE);
    SFEM_TEST_ASSERT(pg != nullptr);
    SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(!pg->function()->tangent_is_symmetric());
    for (const char *store : {"points", "element_matrix"}) {
        const real_t error = stored_against_exact(*pg, store, "f64");
        printf("Kelvin-Voigt %s f64: stored vs exact %.3e\n", store, (double)error);
        SFEM_TEST_ASSERT(error >= 0 && error < 1e-12);
    }
    return SFEM_TEST_SUCCESS;
}

int test_kelvin_voigt_refuses_cg() {
    auto pg = assemble(KELVIN_VOIGT_PULSE, {"solver.linear.type=cg", "solver.linear.format=INEXACT"});
    SFEM_TEST_ASSERT(pg != nullptr);
    SFEM_TEST_ASSERT(pg->run() != SFEM_SUCCESS);
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_prony_stored_tangent);
    SFEM_RUN_TEST(test_kelvin_voigt_stored_tangent);
    SFEM_RUN_TEST(test_kelvin_voigt_refuses_cg);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
