// The case reader: the schema check, the overrides, the paths, and the cases the playground
// refuses to assemble.

#include <string>

#include "sfem_API.hpp"
#include "sfem_ElasticityCase.hpp"
#include "sfem_Playground.hpp"
#include "sfem_test.hpp"

namespace {
    const char *const BASE = R"(
mesh:
  box: {n: [2, 2, 2]}
  sidesets:
    - {name: left, plane: {axis: x, at: min}}
material: {type: GeneratedNeoHookeanOgden, mu: 1, lmbda: 2}
dirichlet_conditions:
  - {type: sideset, format: name, name: left, component: [0, 1, 2], value: [0, 0, 0]}
regime: dynamic
time: {dt: 0.1, steps: 1}
dynamics: {scheme: newmark}
solver: {newton: {verbose: false}}
)";

    std::unique_ptr<sfem::Playground> assemble(const std::vector<std::string> &overrides) {
        auto config = sfem::ElasticityCase::parse(BASE, "/cases", overrides);
        if (!config) return nullptr;
        return sfem::Playground::create(sfem::Communicator::world(), *config);
    }
}  // namespace

int test_unknown_keys_are_errors() {
    SFEM_TEST_ASSERT(sfem::ElasticityCase::parse(BASE, "") != nullptr);
    SFEM_TEST_ASSERT(sfem::ElasticityCase::parse(std::string(BASE) + "solvr: {}\n", "") == nullptr);
    SFEM_TEST_ASSERT(sfem::ElasticityCase::parse(BASE, "", {"solver.newton.maxit=3"}) == nullptr);
    SFEM_TEST_ASSERT(sfem::ElasticityCase::parse(BASE, "", {"mesh.sidesets.0.plane.where=min"}) == nullptr);
    // The readers that own a subtree check it themselves.
    SFEM_TEST_ASSERT(sfem::ElasticityCase::parse(BASE, "", {"material.anything=1"}) != nullptr);
    return SFEM_TEST_SUCCESS;
}

int test_overrides() {
    auto config = sfem::ElasticityCase::parse(BASE, "", {"solver.newton.max_it=7", "mesh.box.n=[3,4,5]", "time.t_end=2"});
    SFEM_TEST_ASSERT(config != nullptr);
    auto root = config->root();

    int max_it = 0;
    root["solver"]["newton"]["max_it"].load(&max_it);
    SFEM_TEST_ASSERT(max_it == 7);

    std::vector<int> n;
    for (auto c : root["mesh"]["box"]["n"].children()) {
        int v = 0;
        c.load(&v);
        n.push_back(v);
    }
    SFEM_TEST_ASSERT(n.size() == 3 && n[0] == 3 && n[1] == 4 && n[2] == 5);
    SFEM_TEST_ASSERT(root["time"].has_child("t_end"));

    // An override reaches into a list by index.
    auto listed = sfem::ElasticityCase::parse(BASE, "", {"dirichlet_conditions.0.value=[0,0,1]"});
    SFEM_TEST_ASSERT(listed != nullptr);
    real_t z = 0;
    listed->root()["dirichlet_conditions"][0]["value"][2].load(&z);
    SFEM_TEST_ASSERT(z == 1);
    SFEM_TEST_ASSERT(sfem::ElasticityCase::parse(BASE, "", {"dirichlet_conditions.3.value=1"}) == nullptr);

    // A path through a missing key adds a mapping, one through a value replaces it by a mapping.
    auto nested = sfem::ElasticityCase::parse(BASE, "", {"solver.linear.rtol=1e-3", "material.type.kind=1"});
    SFEM_TEST_ASSERT(nested != nullptr);
    real_t rtol = 0;
    nested->root()["solver"]["linear"]["rtol"].load(&rtol);
    SFEM_TEST_ASSERT(rtol == real_t(1e-3));
    SFEM_TEST_ASSERT(!nested->root()["solver"]["linear"].has_val() && !nested->root()["material"]["type"].has_val());
    const std::string emitted = nested->emit();
    SFEM_TEST_ASSERT(emitted.find("rtol: 1e-3") != std::string::npos && emitted.find("kind: 1") != std::string::npos);
    return SFEM_TEST_SUCCESS;
}

int test_paths_resolve_against_the_case() {
    auto config = sfem::ElasticityCase::parse(BASE, "/cases");
    SFEM_TEST_ASSERT(config->resolve("mesh") == "/cases/mesh");
    SFEM_TEST_ASSERT(config->resolve("/abs/mesh") == "/abs/mesh");
    return SFEM_TEST_SUCCESS;
}

int test_refused_cases() {
    SFEM_TEST_ASSERT(assemble({}) != nullptr);
    // A sideset no one defined.
    SFEM_TEST_ASSERT(assemble({"dirichlet_conditions.0.name=right"}) == nullptr);
    // Newmark outside its unconditionally stable range.
    SFEM_TEST_ASSERT(assemble({"dynamics.gamma=0.6"}) == nullptr);
    SFEM_TEST_ASSERT(assemble({"dynamics.gamma=0.6", "dynamics.beta=0.64"}) != nullptr);
    // A material with a history and no time to relax in.
    SFEM_TEST_ASSERT(assemble({"regime=steady", "material.type=GeneratedMooneyRivlinProny", "material.C10=1", "material.C01=1",
                               "material.K=10"}) == nullptr);
    SFEM_TEST_ASSERT(assemble({"regime=bouncy"}) == nullptr);
    SFEM_TEST_ASSERT(assemble({"dynamics.scheme=leapfrog"}) == nullptr);
#ifndef SFEM_ENABLE_CUDA
    SFEM_TEST_ASSERT(assemble({"execution_space=device"}) == nullptr);
#endif
    // NCCL on the host, or in a build without it.
    SFEM_TEST_ASSERT(assemble({"nccl=true"}) == nullptr);
#if defined(SFEM_ENABLE_CUDA) && !defined(SMESH_ENABLE_NCCL)
    SFEM_TEST_ASSERT(assemble({"execution_space=device", "nccl=true"}) == nullptr);
#endif
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_unknown_keys_are_errors);
    SFEM_RUN_TEST(test_overrides);
    SFEM_RUN_TEST(test_paths_resolve_against_the_case);
    SFEM_RUN_TEST(test_refused_cases);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
