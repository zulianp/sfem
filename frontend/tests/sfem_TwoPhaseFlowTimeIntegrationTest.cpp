#include "sfem_test.hpp"

#include "sfem_TwoPhaseFlowTimeIntegration.hpp"
#include "smesh_mesh.hpp"

#include <algorithm>
#include <cmath>
#include <unistd.h>
#include <vector>

namespace {
    int predictor(const real_t *const previous,
                  real_t *const trial,
                  const real_t time,
                  const real_t dt,
                  const ptrdiff_t ndofs) {
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            trial[i] = previous[i] + dt * time * (1 + i % 2);
        }
        return SFEM_SUCCESS;
    }
}

int test_zero_ramp_preserves_uniform_state() {
    auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 2, 1, 1);
    sfem::TwoPhaseFlowTimeConfig config;
    config.injection_co2_pressure = config.initial_co2_pressure;
    sfem::TwoPhaseFlowTimeIntegration integration(mesh, config);
    SFEM_TEST_ASSERT(integration.initialize() == SFEM_SUCCESS);
    const ptrdiff_t ndofs = 2 * mesh->n_nodes();
    for (int step = 0; step < 4; ++step) {
        SFEM_TEST_ASSERT(
                integration.advance(
                        0.25,
                        [=](const real_t *previous, real_t *trial, real_t, real_t) {
                            std::copy(previous, previous + ndofs, trial);
                            return SFEM_SUCCESS;
                        }) == SFEM_SUCCESS);
    }
    for (ptrdiff_t node = 0; node < mesh->n_nodes(); ++node) {
        SFEM_TEST_ASSERT(integration.accepted()->data()[2 * node] ==
                         config.initial_water_pressure);
        SFEM_TEST_ASSERT(integration.accepted()->data()[2 * node + 1] ==
                         config.initial_co2_pressure);
    }
    return SFEM_TEST_SUCCESS;
}

int test_restart_reproduces_uninterrupted_state() {
    auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 2, 1, 1);
    sfem::TwoPhaseFlowTimeConfig config;
    const ptrdiff_t ndofs = 2 * mesh->n_nodes();
    sfem::TwoPhaseFlowTimeIntegration uninterrupted(mesh, config);
    sfem::TwoPhaseFlowTimeIntegration first(mesh, config);
    uninterrupted.initialize();
    first.initialize();
    auto solve = [=](const real_t *previous, real_t *trial, real_t time, real_t dt) {
        return predictor(previous, trial, time, dt, ndofs);
    };
    for (int i = 0; i < 4; ++i) {
        uninterrupted.advance(0.25, solve);
    }
    for (int i = 0; i < 2; ++i) {
        first.advance(0.25, solve);
    }
    const smesh::Path checkpoint(
            std::string("/tmp/sfem_two_phase_restart_") +
            std::to_string(static_cast<long>(getpid())));
    SFEM_TEST_ASSERT(first.save_restart(checkpoint) == SFEM_SUCCESS);

    sfem::TwoPhaseFlowTimeIntegration resumed(mesh, config);
    SFEM_TEST_ASSERT(resumed.load_restart(checkpoint) == SFEM_SUCCESS);
    for (int i = 0; i < 2; ++i) {
        resumed.advance(0.25, solve);
    }
    SFEM_TEST_ASSERT(resumed.step() == uninterrupted.step());
    SFEM_TEST_ASSERT(resumed.time() == uninterrupted.time());
    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        SFEM_TEST_ASSERT(
                std::abs(resumed.accepted()->data()[i] -
                         uninterrupted.accepted()->data()[i]) < 1e-12);
    }
    return SFEM_TEST_SUCCESS;
}

int test_rejected_step_preserves_accepted_state() {
    auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 1, 1, 1);
    sfem::TwoPhaseFlowTimeConfig config;
    sfem::TwoPhaseFlowTimeIntegration integration(mesh, config);
    integration.initialize();
    const ptrdiff_t ndofs = 2 * mesh->n_nodes();
    std::vector<real_t> before(
            integration.accepted()->data(),
            integration.accepted()->data() + ndofs);
    SFEM_TEST_ASSERT(
            integration.advance(
                    0.25,
                    [=](const real_t *, real_t *trial, real_t, real_t) {
                        std::fill(trial, trial + ndofs, -1);
                        return SFEM_FAILURE;
                    }) == SFEM_FAILURE);
    SFEM_TEST_ASSERT(integration.step() == 0);
    SFEM_TEST_ASSERT(integration.time() == 0);
    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        SFEM_TEST_ASSERT(integration.accepted()->data()[i] == before[i]);
    }
    return SFEM_TEST_SUCCESS;
}

int test_balance_partitions_boundary_and_interior_nodes() {
    auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 2, 1, 1);
    sfem::TwoPhaseFlowTimeConfig config;
    sfem::TwoPhaseFlowTimeIntegration integration(mesh, config);
    integration.initialize();
    std::vector<real_t> residual(2 * mesh->n_nodes());
    for (ptrdiff_t node = 0; node < mesh->n_nodes(); ++node) {
        residual[2 * node + 0] = 1;
        residual[2 * node + 1] = 2;
    }
    const auto balance = integration.balance(residual.data());
    SFEM_TEST_ASSERT(balance.left[0] == 4);
    SFEM_TEST_ASSERT(balance.left[1] == 8);
    SFEM_TEST_ASSERT(balance.right[0] == 4);
    SFEM_TEST_ASSERT(balance.right[1] == 8);
    SFEM_TEST_ASSERT(balance.interior[0] == 4);
    SFEM_TEST_ASSERT(balance.interior[1] == 8);
    SFEM_TEST_ASSERT(balance.total[0] == 12);
    SFEM_TEST_ASSERT(balance.total[1] == 24);
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_zero_ramp_preserves_uniform_state);
    SFEM_RUN_TEST(test_restart_reproduces_uninterrupted_state);
    SFEM_RUN_TEST(test_rejected_step_preserves_accepted_state);
    SFEM_RUN_TEST(test_balance_partitions_boundary_and_interior_nodes);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
