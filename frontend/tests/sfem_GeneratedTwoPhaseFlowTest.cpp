#include "sfem_test.hpp"

#include "sfem_GeneratedTwoPhaseFlow.hpp"
#include "sfem_OpFactory.hpp"
#include "smesh_buffer.hpp"
#include "smesh_mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>

extern "C" {
int generated_two_phase_flow_hex8_residual_isoparametric_mesh_aos(
        ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t *,
        const real_t *, const real_t *, real_t *);
int generated_two_phase_flow_hex8_jacobian_action_isoparametric_mesh_aos(
        ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t *,
        const real_t *, const real_t *, const real_t *, real_t *);
}

namespace {
    std::array<real_t, 26> parameters() {
        return {0.2, 0.1, 1e5, 2.0, 1000.0, 1e-9, 1e5, 0.044, 1.0,
                8.314462618, 300.0, 1e-3, 1.5e-5, 2.0, 2.0, 2.0, 1.0,
                1e-12, 0.0, 0.0, 0.0, 1e-12, 0.0, 0.0, 0.0, 1e-12};
    }

    bool close(const real_t a, const real_t b) {
        return std::abs(a - b) <= 1e-11 * std::max<real_t>(1, std::max(std::abs(a), std::abs(b)));
    }
}

int test_generated_two_phase_flow_operator() {
    auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 2, 2, 2);
    auto space = sfem::FunctionSpace::create(mesh, 2);
    auto base = sfem::Factory::create_op(space, "GeneratedTwoPhaseFlow");
    auto op = std::dynamic_pointer_cast<sfem::GeneratedTwoPhaseFlow>(base);
    SFEM_TEST_ASSERT(op != nullptr);

    const ptrdiff_t ndofs = space->n_dofs();
    auto previous = sfem::create_host_buffer<real_t>(ndofs);
    auto current = sfem::create_host_buffer<real_t>(ndofs);
    auto direction = sfem::create_host_buffer<real_t>(ndofs);
    auto residual = sfem::create_host_buffer<real_t>(ndofs);
    auto residual_direct = sfem::create_host_buffer<real_t>(ndofs);
    auto action = sfem::create_host_buffer<real_t>(ndofs);
    auto action_direct = sfem::create_host_buffer<real_t>(ndofs);

    for (ptrdiff_t node = 0; node < mesh->n_nodes(); ++node) {
        previous->data()[2 * node + 0] = 15e6 + 100 * node;
        previous->data()[2 * node + 1] = 15.1e6 + 80 * node;
        current->data()[2 * node + 0] = previous->data()[2 * node + 0] + 10;
        current->data()[2 * node + 1] = previous->data()[2 * node + 1] + 15;
        direction->data()[2 * node + 0] = 0.01 * (node + 1);
        direction->data()[2 * node + 1] = -0.015 * (node + 1);
    }

    op->set_field("previous", previous, 0);
    op->reset_stats();
    op->gradient(current->data(), residual->data());
    op->apply(current->data(), direction->data(), action->data());
    auto stats = op->stats();
    SFEM_TEST_ASSERT(stats.residual_calls == 1);
    SFEM_TEST_ASSERT(stats.jacobian_calls == 1);
    SFEM_TEST_ASSERT(stats.residual_seconds >= 0);
    SFEM_TEST_ASSERT(stats.jacobian_seconds >= 0);

    const auto p = parameters();
    auto block = mesh->block(0);
    const auto points = const_cast<const geom_t *const *>(mesh->points()->data());
    generated_two_phase_flow_hex8_residual_isoparametric_mesh_aos(
            block->n_elements(), mesh->n_nodes(), block->elements()->data(), points,
            p.data(), current->data(), previous->data(), residual_direct->data());
    generated_two_phase_flow_hex8_jacobian_action_isoparametric_mesh_aos(
            block->n_elements(), mesh->n_nodes(), block->elements()->data(), points,
            p.data(), current->data(), previous->data(), direction->data(), action_direct->data());

    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        SFEM_TEST_ASSERT(close(residual->data()[i], residual_direct->data()[i]));
        SFEM_TEST_ASSERT(close(action->data()[i], action_direct->data()[i]));
    }

    auto linear = op->make_linear_operator(
            current->data(),
            [](const real_t *const in, real_t *const out) {
                out[0] = in[0];
                out[1] = in[1];
            });
    std::fill(action->data(), action->data() + ndofs, 0);
    linear->apply(direction->data(), action->data());
    SFEM_TEST_ASSERT(action->data()[0] == direction->data()[0]);
    SFEM_TEST_ASSERT(action->data()[1] == direction->data()[1]);
    for (ptrdiff_t i = 2; i < ndofs; ++i) {
        SFEM_TEST_ASSERT(close(action->data()[i], action_direct->data()[i]));
    }
    op->reset_stats();
    stats = op->stats();
    SFEM_TEST_ASSERT(stats.residual_calls == 0);
    SFEM_TEST_ASSERT(stats.jacobian_calls == 0);

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_generated_two_phase_flow_operator);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
