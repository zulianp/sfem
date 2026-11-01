// The stored-tangent operator of the rate-dependent materials on the device, against the device's
// exact apply (sfem_InexactViscousTest.cpp on the host).
//
// Prony relaxation: after a few relaxing steps the tangent reads a nonzero history at every point
// and a step weight below one, from the relaxation's device state.  Kelvin-Voigt: after a few
// Newmark steps the tangent reads the previous state and is not symmetric.  In both the f64 stores
// must be the exact apply; and CG refuses the Kelvin-Voigt operator on the device as on the host.

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
execution_space: device
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
execution_space: device
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

    sfem::SharedBuffer<real_t> on_device(const std::vector<real_t> &v) {
        auto host = sfem::create_host_buffer<real_t>((ptrdiff_t)v.size());
        std::copy(v.begin(), v.end(), host->data());
        return smesh::to_device(host);
    }

    real_t stored_against_exact(sfem::Playground &pg, const char *const store, const char *const encoding) {
        auto            f = pg.function();
        const ptrdiff_t n = f->space()->n_dofs();
        std::vector<real_t> hv(n);
        unsigned            seed = 11;
        for (auto &v : hv) {
            seed = seed * 1664525u + 1013904223u;
            v    = (real_t)(seed >> 8) / (real_t)(1u << 24) - real_t(0.5);
        }
        auto       h     = on_device(hv);
        auto       exact = on_device(std::vector<real_t>(n, 0));
        auto       stored = on_device(std::vector<real_t>(n, 0));
        const real_t *const u = pg.solution()->data();
        if (f->update(u) != SFEM_SUCCESS) return -1;
        if (f->apply(u, h->data(), exact->data()) != SFEM_SUCCESS) return -1;
        f->set_inexact_options(sfem::InexactOptions::from_strings(store, encoding));
        if (f->inexact_update(u) != SFEM_SUCCESS) return -1;
        if (f->inexact_apply(h->data(), stored->data()) != SFEM_SUCCESS) return -1;
        auto   e = smesh::to_host(exact), s = smesh::to_host(stored);
        real_t num = 0, den = 0;
        for (ptrdiff_t i = 0; i < n; ++i) {
            num += (e->data()[i] - s->data()[i]) * (e->data()[i] - s->data()[i]);
            den += e->data()[i] * e->data()[i];
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
            printf("device Prony t = %s, %s f64: stored vs exact %.3e\n", t_end, store, (double)error);
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
        printf("device Kelvin-Voigt %s f64: stored vs exact %.3e\n", store, (double)error);
        SFEM_TEST_ASSERT(error >= 0 && error < 1e-12);
    }
    return SFEM_TEST_SUCCESS;
}

int test_kelvin_voigt_formats() {
    // Every store and encoding through Newton on the device: the run converges.
    const char *const formats[][2] = {{"points", "f32"}, {"points", "f16"}, {"element_matrix", "f16"}, {"bsr", "f16"}, {"bsr", "i8"}};
    for (const auto &fmt : formats) {
        auto pg = assemble(KELVIN_VOIGT_PULSE,
                           {"solver.linear.format=INEXACT",
                            std::string("solver.linear.inexact.store=") + fmt[0],
                            std::string("solver.linear.inexact.encoding=") + fmt[1]});
        SFEM_TEST_ASSERT(pg != nullptr);
        const int rc = pg->run();
        printf("device Kelvin-Voigt Newton with %s %s: %s\n", fmt[0], fmt[1], rc == SFEM_SUCCESS ? "converged" : "FAILED");
        SFEM_TEST_ASSERT(rc == SFEM_SUCCESS);
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
    SFEM_RUN_TEST(test_kelvin_voigt_formats);
    SFEM_RUN_TEST(test_kelvin_voigt_refuses_cg);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
