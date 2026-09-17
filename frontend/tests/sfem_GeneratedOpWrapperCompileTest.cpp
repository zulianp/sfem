#include "sfem_test.hpp"

#include "sfem_API.hpp"
#include "sfem_NeumannConditions.hpp"
#include "sfem_OpFactory.hpp"

#include "generated/neumann/op/sfem_GeneratedNeumann.hpp"
#include "generated/neumann/op/sfem_GeneratedNeumann_c_abi.hpp"
#include "generated/neumann_general/op/sfem_GeneratedNeumannGeneral.hpp"
#include "generated/neumann_general/op/sfem_GeneratedNeumannGeneral_c_abi.hpp"
#include "sfem_GeneratedNeoHookeanOgden.hpp"
#include "sfem_GeneratedNeoHookeanOgden_c_abi.hpp"
#include "sfem_GeneratedMooneyRivlinKelvinVoigtNewmark.hpp"
#include "sfem_GeneratedTwoPhaseFlow.hpp"
#include "sfem_GeneratedTwoPhaseFlow_c_abi.hpp"
#include "smesh_sideset.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <type_traits>

namespace {
    template <typename T>
    void require_op_type() {
        static_assert(std::is_base_of<sfem::Op, T>::value, "generated wrapper must derive from sfem::Op");
    }

    std::shared_ptr<sfem::FunctionSpace> hex8_space(const int block_size) {
        auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 1, 1, 1);
        return sfem::FunctionSpace::create(mesh, block_size);
    }

    bool finite_vector(const real_t *const values, const ptrdiff_t n) {
        for (ptrdiff_t i = 0; i < n; ++i) {
            if (!std::isfinite(values[i])) {
                fprintf(stderr, "non-finite generated wrapper output at %td: %.17g\n", i, (double)values[i]);
                return false;
            }
        }
        return true;
    }

    bool has_nonzero(const real_t *const values, const ptrdiff_t n) {
        for (ptrdiff_t i = 0; i < n; ++i) {
            if (values[i] != 0) {
                return true;
            }
        }
        return false;
    }
}  // namespace

int test_generated_wrapper_headers_compile() {
    require_op_type<sfem::GeneratedNeoHookeanOgden>();
    require_op_type<sfem::GeneratedTwoPhaseFlow>();
    require_op_type<sfem::GeneratedNeumann>();
    require_op_type<sfem::GeneratedNeumannGeneral>();
    return SFEM_TEST_SUCCESS;
}

int test_generated_wrapper_factory_registration() {
    SFEM_TEST_ASSERT(sfem::Factory::create_op(hex8_space(3), "GeneratedNeoHookeanOgden") != nullptr);
    SFEM_TEST_ASSERT(sfem::Factory::create_op(hex8_space(2), "GeneratedTwoPhaseFlow") != nullptr);
    SFEM_TEST_ASSERT(sfem::Factory::create_op(hex8_space(3), "GeneratedNeumann") != nullptr);
    SFEM_TEST_ASSERT(sfem::Factory::create_op(hex8_space(3), "GeneratedNeumannGeneral") != nullptr);
    return SFEM_TEST_SUCCESS;
}

int test_generated_energy_wrapper_executes() {
    auto space = hex8_space(3);
    auto op    = sfem::Factory::create_op(space, "GeneratedNeoHookeanOgden");
    SFEM_TEST_ASSERT(op != nullptr);
    op->set_option("assume_affine", true);

    const ptrdiff_t ndofs     = space->n_dofs();
    auto            state     = sfem::create_host_buffer<real_t>(ndofs);
    auto            direction = sfem::create_host_buffer<real_t>(ndofs);
    auto            gradient  = sfem::create_host_buffer<real_t>(ndofs);
    auto            action    = sfem::create_host_buffer<real_t>(ndofs);
    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        state->data()[i]     = 1e-3 * ((i % 7) + 1);
        direction->data()[i] = 2e-4 * ((i % 5) + 1);
    }

    real_t value = -1;
    SFEM_TEST_ASSERT(op->value(state->data(), &value) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(std::isfinite(value));
    SFEM_TEST_ASSERT(op->gradient(state->data(), gradient->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(op->apply(state->data(), direction->data(), action->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(finite_vector(gradient->data(), ndofs));
    SFEM_TEST_ASSERT(finite_vector(action->data(), ndofs));
    return SFEM_TEST_SUCCESS;
}

int test_generated_residual_wrapper_executes() {
    auto space = hex8_space(2);
    auto op    = sfem::Factory::create_op(space, "GeneratedTwoPhaseFlow");
    SFEM_TEST_ASSERT(op != nullptr);

    const ptrdiff_t ndofs     = space->n_dofs();
    auto            previous  = sfem::create_host_buffer<real_t>(ndofs);
    auto            current   = sfem::create_host_buffer<real_t>(ndofs);
    auto            direction = sfem::create_host_buffer<real_t>(ndofs);
    auto            residual  = sfem::create_host_buffer<real_t>(ndofs);
    auto            action    = sfem::create_host_buffer<real_t>(ndofs);
    for (ptrdiff_t node = 0; node < space->mesh_ptr()->n_nodes(); ++node) {
        previous->data()[2 * node + 0]  = 15.0 + 1e-3 * node;
        previous->data()[2 * node + 1]  = 15.1 + 1e-3 * node;
        current->data()[2 * node + 0]   = previous->data()[2 * node + 0] + 1e-2;
        current->data()[2 * node + 1]   = previous->data()[2 * node + 1] + 2e-2;
        direction->data()[2 * node + 0] = 1e-4 * (node + 1);
        direction->data()[2 * node + 1] = -1e-4 * (node + 1);
    }

    op->set_field("previous", previous, 0);
    op->update(previous->data(), current->data());
    SFEM_TEST_ASSERT(op->gradient(current->data(), residual->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(op->apply(current->data(), direction->data(), action->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(finite_vector(residual->data(), ndofs));
    SFEM_TEST_ASSERT(finite_vector(action->data(), ndofs));
    return SFEM_TEST_SUCCESS;
}

int test_generated_boundary_wrapper_executes() {
    auto mesh  = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 1, 1, 1);
    auto space = sfem::FunctionSpace::create(mesh, 3);
    auto base  = sfem::Factory::create_op(space, "GeneratedNeumann");
    auto op    = std::dynamic_pointer_cast<sfem::GeneratedNeumann>(base);
    SFEM_TEST_ASSERT(op != nullptr);

    auto sidesets =
            sfem::Sideset::create_from_selector(mesh, [](const geom_t x, const geom_t, const geom_t) { return x <= 1e-12; });
    SFEM_TEST_ASSERT(!sidesets.empty());

    sfem::NeumannConditions::Condition condition;
    condition.sidesets          = sidesets;
    condition.values            = sfem::create_host_buffer<real_t>(3);
    condition.values->data()[0] = 0.5;
    condition.values->data()[1] = 0.25;
    condition.values->data()[2] = -0.125;
    op->add_condition(condition);

    const ptrdiff_t ndofs    = space->n_dofs();
    auto            residual = sfem::create_host_buffer<real_t>(ndofs);
    std::fill(residual->data(), residual->data() + ndofs, static_cast<real_t>(0));
    SFEM_TEST_ASSERT(op->gradient(nullptr, residual->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(finite_vector(residual->data(), ndofs));
    SFEM_TEST_ASSERT(has_nonzero(residual->data(), ndofs));
    return SFEM_TEST_SUCCESS;
}

/// The merit must be the norm of the residual the *Function* assembles.
///
/// This is the defect the node-wise reduction exists for.  The operator used to
/// compute `1/2*||R_op||^2` from its own `gradient`, so a traction -- which is a
/// separate `Op` and never appears in the interior operator's residual -- was
/// missing from it, and the scalar was not zero at the solution of the combined
/// system.  Asserting `value == 1/2*||Function::gradient||^2` pins the property
/// directly, without needing a converged state to see it.
int test_node_wise_merit_includes_the_forcing() {
    auto mesh  = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 1, 1, 1);
    // Block size 3: the traction operator is a displacement forcing and refuses
    // anything else, so the interior operator has to share that space.
    auto space = sfem::FunctionSpace::create(mesh, 3);

    auto interior = sfem::Factory::create_op(space, "GeneratedMooneyRivlinKelvinVoigtNewmark");
    SFEM_TEST_ASSERT(interior != nullptr);
    SFEM_TEST_ASSERT(interior->value_reduction() == sfem::Op::ValueReduction::NODE_WISE);

    const ptrdiff_t ndofs    = space->n_dofs();
    auto            previous = sfem::create_host_buffer<real_t>(ndofs);
    auto            current  = sfem::create_host_buffer<real_t>(ndofs);
    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        previous->data()[i] = 1e-3 * ((i % 7) + 1);
        current->data()[i]  = previous->data()[i] + 1e-4 * ((i % 5) + 1);
    }
    interior->set_field("previous", previous, 0);
    interior->update(previous->data(), current->data());

    auto base    = sfem::Factory::create_op(space, "GeneratedNeumann");
    auto forcing = std::dynamic_pointer_cast<sfem::GeneratedNeumann>(base);
    SFEM_TEST_ASSERT(forcing != nullptr);
    auto sidesets =
            sfem::Sideset::create_from_selector(mesh, [](const geom_t x, const geom_t, const geom_t) { return x <= 1e-12; });
    SFEM_TEST_ASSERT(!sidesets.empty());
    sfem::NeumannConditions::Condition condition;
    condition.sidesets          = sidesets;
    condition.values            = sfem::create_host_buffer<real_t>(3);
    condition.values->data()[0] = 0.5;
    condition.values->data()[1] = 0.25;
    condition.values->data()[2] = -0.125;
    forcing->add_condition(condition);

    auto f = sfem::Function::create(space);
    f->add_operator(interior);
    f->add_operator(forcing);
    SFEM_TEST_ASSERT(f->reduces_node_wise());

    real_t value = 0;
    SFEM_TEST_ASSERT(f->value(current->data(), &value) == SFEM_SUCCESS);

    auto residual = sfem::create_host_buffer<real_t>(ndofs);
    std::fill(residual->data(), residual->data() + ndofs, static_cast<real_t>(0));
    SFEM_TEST_ASSERT(f->gradient(current->data(), residual->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(has_nonzero(residual->data(), ndofs));

    real_t expected = 0;
    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        expected += residual->data()[i] * residual->data()[i];
    }
    expected *= static_cast<real_t>(0.5);

    SFEM_TEST_ASSERT(std::isfinite(value));
    SFEM_TEST_ASSERT(std::fabs(value - expected) <= 1e-10 * std::fabs(expected) + 1e-14);

    // And the forcing is genuinely in it: the interior operator's own residual
    // differs from the system's, which is the whole defect.  Without this the
    // test would still pass if the traction silently dropped out again.
    auto interior_only = sfem::create_host_buffer<real_t>(ndofs);
    std::fill(interior_only->data(), interior_only->data() + ndofs, static_cast<real_t>(0));
    SFEM_TEST_ASSERT(interior->gradient(current->data(), interior_only->data()) == SFEM_SUCCESS);
    real_t without_forcing = 0;
    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        without_forcing += interior_only->data()[i] * interior_only->data()[i];
    }
    without_forcing *= static_cast<real_t>(0.5);
    SFEM_TEST_ASSERT(std::fabs(value - without_forcing) > 1e-12 * std::fabs(value));
    return SFEM_TEST_SUCCESS;
}

/// A forcing operator reports the work it does, rather than nothing.
///
/// `GeneratedNeumann::value` returned `SFEM_SUCCESS` having written nothing, so
/// a traction contributed zero to the objective while contributing to the
/// residual -- the objective was then not a potential of what the gradient
/// computed.  The potential is the linear work `-t . u`, which equals `g . u`
/// for the `g` the operator assembles.
int test_forcing_operator_reports_its_linear_work() {
    auto mesh  = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 1, 1, 1);
    auto space = sfem::FunctionSpace::create(mesh, 3);
    auto base  = sfem::Factory::create_op(space, "GeneratedNeumann");
    auto op    = std::dynamic_pointer_cast<sfem::GeneratedNeumann>(base);
    SFEM_TEST_ASSERT(op != nullptr);

    auto sidesets =
            sfem::Sideset::create_from_selector(mesh, [](const geom_t x, const geom_t, const geom_t) { return x <= 1e-12; });
    SFEM_TEST_ASSERT(!sidesets.empty());
    sfem::NeumannConditions::Condition condition;
    condition.sidesets          = sidesets;
    condition.values            = sfem::create_host_buffer<real_t>(3);
    condition.values->data()[0] = 0.5;
    condition.values->data()[1] = 0.25;
    condition.values->data()[2] = -0.125;
    op->add_condition(condition);

    const ptrdiff_t ndofs = space->n_dofs();
    auto            state = sfem::create_host_buffer<real_t>(ndofs);
    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        state->data()[i] = 1e-3 * ((i % 5) + 1);
    }

    auto residual = sfem::create_host_buffer<real_t>(ndofs);
    std::fill(residual->data(), residual->data() + ndofs, static_cast<real_t>(0));
    SFEM_TEST_ASSERT(op->gradient(state->data(), residual->data()) == SFEM_SUCCESS);

    real_t expected = 0;
    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        expected += residual->data()[i] * state->data()[i];
    }
    SFEM_TEST_ASSERT(expected != 0);

    real_t value = 0;
    SFEM_TEST_ASSERT(op->value(state->data(), &value) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(std::fabs(value - expected) <= 1e-12 * std::fabs(expected) + 1e-16);

    // And it accumulates rather than clearing: `Function::value` sums every
    // operator into one scalar and does not zero between them.
    real_t accumulated = 1.25;
    SFEM_TEST_ASSERT(op->value(state->data(), &accumulated) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(std::fabs(accumulated - (1.25 + expected)) <= 1e-12 * std::fabs(expected) + 1e-16);
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_generated_wrapper_headers_compile);
    SFEM_RUN_TEST(test_generated_wrapper_factory_registration);
    SFEM_RUN_TEST(test_generated_energy_wrapper_executes);
    SFEM_RUN_TEST(test_generated_residual_wrapper_executes);
    SFEM_RUN_TEST(test_generated_boundary_wrapper_executes);
    SFEM_RUN_TEST(test_node_wise_merit_includes_the_forcing);
    SFEM_RUN_TEST(test_forcing_operator_reports_its_linear_work);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
