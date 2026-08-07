#include "sfem_test.hpp"

#include "sfem_API.hpp"
#include "sfem_GeneratedTwoPhaseFlow.hpp"
#include "sfem_OpFactory.hpp"
#include "smesh_buffer.hpp"
#include "smesh_mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

extern "C" {
int two_phase_flow_hex8_residual_isoparametric_mesh_aos(
        ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t *,
        const real_t *, const real_t *, real_t *);
int two_phase_flow_hex8_jacobian_action_isoparametric_mesh_aos(
        ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t *,
        const real_t *, const real_t *, real_t *);
}

namespace {
    std::array<real_t, 26> parameters() {
        return {1.8, 0.35, 0.52, 86.4, 0.0, 0.0, 0.0, 86.4, 0.0,
                0.0, 0.0, 86.4, 0.04401, 0.095, 8.314e-6, 0.39, 333.0,
                0.4252, 1.0, 0.000455, 4.2, 1.5, 5.2, 1.0, 0.1, 1100.0};
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
        previous->data()[2 * node + 0] = 15.0 + 1e-3 * node;
        previous->data()[2 * node + 1] = 15.1 + 8e-4 * node;
        current->data()[2 * node + 0] = previous->data()[2 * node + 0] + 1e-2;
        current->data()[2 * node + 1] = previous->data()[2 * node + 1] + 1.5e-2;
        direction->data()[2 * node + 0] = 1e-4 * (node + 1);
        direction->data()[2 * node + 1] = -1.5e-4 * (node + 1);
    }

    op->set_field("previous", previous, 0);
    op->update(previous->data(), current->data());
    std::fill(residual->data(), residual->data() + ndofs, 0);
    std::fill(action->data(), action->data() + ndofs, 0);
    SFEM_TEST_ASSERT(op->gradient(current->data(), residual->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(op->apply(current->data(), direction->data(), action->data()) == SFEM_SUCCESS);

    const auto p = parameters();
    auto block = mesh->block(0);
    const auto points = const_cast<const geom_t *const *>(mesh->points()->data());
    std::fill(residual_direct->data(), residual_direct->data() + ndofs, 0);
    std::fill(action_direct->data(), action_direct->data() + ndofs, 0);
    two_phase_flow_hex8_residual_isoparametric_mesh_aos(
            block->n_elements(), mesh->n_nodes(), block->elements()->data(), points,
            p.data(), current->data(), previous->data(), residual_direct->data());
    two_phase_flow_hex8_jacobian_action_isoparametric_mesh_aos(
            block->n_elements(), mesh->n_nodes(), block->elements()->data(), points,
            p.data(), current->data(), direction->data(), action_direct->data());

    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        if (!close(residual->data()[i], residual_direct->data()[i])) {
            fprintf(stderr,
                    "residual mismatch i=%td op=%.17g direct=%.17g diff=%.17g\n",
                    i,
                    static_cast<double>(residual->data()[i]),
                    static_cast<double>(residual_direct->data()[i]),
                    static_cast<double>(residual->data()[i] - residual_direct->data()[i]));
        }
        SFEM_TEST_ASSERT(close(residual->data()[i], residual_direct->data()[i]));
        if (!close(action->data()[i], action_direct->data()[i])) {
            fprintf(stderr,
                    "action mismatch i=%td op=%.17g direct=%.17g diff=%.17g\n",
                    i,
                    static_cast<double>(action->data()[i]),
                    static_cast<double>(action_direct->data()[i]),
                    static_cast<double>(action->data()[i] - action_direct->data()[i]));
        }
        SFEM_TEST_ASSERT(close(action->data()[i], action_direct->data()[i]));
    }

    auto linear = sfem::make_op<real_t>(
            ndofs,
            ndofs,
            [&](const real_t *const in, real_t *const out) {
                std::fill(out, out + ndofs, static_cast<real_t>(0));
                if (op->apply(nullptr, in, out) != SFEM_SUCCESS) {
                    SFEM_ERROR("GeneratedTwoPhaseFlow Jacobian action failed\n");
                }
                out[0] = in[0];
                out[1] = in[1];
            },
            sfem::EXECUTION_SPACE_HOST);
    std::fill(action->data(), action->data() + ndofs, 0);
    linear->apply(direction->data(), action->data());
    SFEM_TEST_ASSERT(action->data()[0] == direction->data()[0]);
    SFEM_TEST_ASSERT(action->data()[1] == direction->data()[1]);
    for (ptrdiff_t i = 2; i < ndofs; ++i) {
        SFEM_TEST_ASSERT(close(action->data()[i], action_direct->data()[i]));
    }

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_generated_two_phase_flow_operator);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
