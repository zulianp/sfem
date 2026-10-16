// The playground with a rigid obstacle: a free block dropped onto a hemisphere.

#include <cmath>
#include <cstdio>
#include <string>

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
// +z.
int test_block_lands_on_the_hemisphere() {
    const std::string csv = "test_block_lands_on_the_hemisphere.csv";
    auto              pg  = assemble(DROP, {"output.history_csv=" + csv});
    SFEM_TEST_ASSERT(pg != nullptr);
    SFEM_TEST_ASSERT(pg->run() == SFEM_SUCCESS);

    auto   rows  = sfem::test::read_history(csv);
    double first = -1, impulse = 0;
    for (auto &row : rows) {
        if (row["status"] < 0) continue;
        if (row["touched"] > 0 && first < 0) first = row["t"];
        impulse += row["reaction_z"] * row["dt"];
    }
    printf("  first contact at t = %g, impulse along z %g\n", first, impulse);
    SFEM_TEST_ASSERT(first > 0.09 && first < 0.13);
    SFEM_TEST_ASSERT(impulse > 0);
    std::remove(csv.c_str());
    return SFEM_TEST_SUCCESS;
}

// The obstacle stabilises Newmark: a regime without it, or another scheme, is refused.
int test_obstacle_needs_newmark() {
    SFEM_TEST_ASSERT(assemble(DROP, {"dynamics.scheme=bdf2"}) == nullptr);
    SFEM_TEST_ASSERT(assemble(DROP, {"regime=quasi_static"}) == nullptr);
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_block_lands_on_the_hemisphere);
    SFEM_RUN_TEST(test_obstacle_needs_newmark);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
