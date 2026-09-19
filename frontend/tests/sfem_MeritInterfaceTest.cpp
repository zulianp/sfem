// The two merits, named rather than inferred.
//
// The merit used to be chosen for the caller: one operator declaring NODE_WISE
// promoted the whole `Function`, and `Function::value` returned whichever merit
// that produced.  Worse, the stepped form re-assembled every operator at every
// trial step, so a traction whose `gradient` never reads the state was
// integrated once per alpha.
//
// These tests hold the replacement to the answer the old path gave -- the
// residual merit at each step must still be `1/2 * ||R(x + alpha*h)||^2` over
// the residual the *Function* assembles -- and to the properties the new shape
// is supposed to have: that the accumulator is really in the sum, that
// insertion order does not matter, and that a system with no potential says so
// instead of inventing an energy.

#include "sfem_test.hpp"

#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "generated/neumann/op/sfem_GeneratedNeumann.hpp"
#include "sfem_NeumannConditions.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

namespace {

    struct Fixture {
        std::shared_ptr<sfem::Mesh>          mesh;
        std::shared_ptr<sfem::FunctionSpace> space;
        ptrdiff_t                            ndofs{0};
    };

    Fixture make_fixture() {
        Fixture fixture;
        fixture.mesh  = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 2, 2, 2);
        fixture.space = sfem::FunctionSpace::create(fixture.mesh, 3);
        fixture.ndofs = fixture.space->n_dofs();
        return fixture;
    }

    void seed(const ptrdiff_t n, const int salt, real_t *const values) {
        for (ptrdiff_t i = 0; i < n; ++i) {
            values[i] = real_t(1e-3) * std::sin(real_t(0.5 * (i + 1) + salt));
        }
    }

    /// A material with no potential, so it mandates the residual merit, plus a
    /// traction, which has one and does not move with the state.
    std::shared_ptr<sfem::Function> make_residual_system(const Fixture &fixture,
                                                         const bool     material_first) {
        auto material = sfem::Factory::create_op(fixture.space, "GeneratedMooneyRivlinKelvinVoigt");
        if (!material) { fprintf(stderr, "no MooneyRivlinKelvinVoigt op\n"); std::abort(); }
        for (const auto &block : fixture.mesh->blocks()) {
            material->set_value_in_block(block->name(), "mu", real_t(3));
            material->set_value_in_block(block->name(), "lmbda", real_t(1));
            material->set_value_in_block(block->name(), "eta_s", real_t(0.05));
            material->set_value_in_block(block->name(), "eta_b", real_t(0.01));
        }

        auto previous = sfem::create_host_buffer<real_t>(fixture.ndofs);
        seed(fixture.ndofs, 9, previous->data());
        material->set_field("previous", previous, 0);

        auto forcing = std::dynamic_pointer_cast<sfem::GeneratedNeumann>(
                sfem::Factory::create_op(fixture.space, "GeneratedNeumann"));
        if (!forcing) { fprintf(stderr, "no GeneratedNeumann op\n"); std::abort(); }
        auto sidesets = sfem::Sideset::create_from_selector(
                fixture.mesh, [](const geom_t x, const geom_t, const geom_t) { return x < 1e-8; });
        if (sidesets.empty()) { fprintf(stderr, "no sideset selected\n"); std::abort(); }
        sfem::NeumannConditions::Condition condition;
        condition.sidesets          = sidesets;
        condition.values            = sfem::create_host_buffer<real_t>(3);
        condition.values->data()[0] = real_t(0.5);
        condition.values->data()[1] = real_t(0.25);
        condition.values->data()[2] = real_t(-0.125);
        forcing->add_condition(condition);

        auto f = sfem::Function::create(fixture.space);
        if (material_first) {
            f->add_operator(material);
            f->add_operator(forcing);
        } else {
            f->add_operator(forcing);
            f->add_operator(material);
        }
        return f;
    }

    /// `1/2 * ||Function::gradient(x + alpha*h)||^2`, the long way.
    real_t merit_the_long_way(const std::shared_ptr<sfem::Function> &f,
                              const ptrdiff_t                        ndofs,
                              const real_t *const                    x,
                              const real_t *const                    h,
                              const real_t                           alpha) {
        auto stepped  = sfem::create_host_buffer<real_t>(ndofs);
        auto residual = sfem::create_host_buffer<real_t>(ndofs);
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            stepped->data()[i] = x[i] + alpha * h[i];
        }
        SFEM_TEST_ASSERT(f->gradient(stepped->data(), residual->data()) == SFEM_SUCCESS);
        real_t acc = 0;
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            acc += residual->data()[i] * residual->data()[i];
        }
        return real_t(0.5) * acc;
    }

    const real_t kSteps[4] = {real_t(0), real_t(-0.5), real_t(0.25), real_t(1)};

    /// The property the old path had, kept: the merit at each trial step is the
    /// norm of the residual the *Function* assembles there -- the traction
    /// included, which is the whole reason an operator cannot compute it alone.
    int test_the_residual_merit_is_the_functions_own_residual() {
        auto fixture = make_fixture();
        auto f       = make_residual_system(fixture, /*material_first=*/true);

        auto x = sfem::create_host_buffer<real_t>(fixture.ndofs);
        auto h = sfem::create_host_buffer<real_t>(fixture.ndofs);
        seed(fixture.ndofs, 1, x->data());
        seed(fixture.ndofs, 2, h->data());

        std::vector<real_t> merits(4, 0);
        SFEM_TEST_ASSERT(f->residual_merit(x->data(), h->data(), 4, kSteps, merits.data()) == SFEM_SUCCESS);

        for (int step = 0; step < 4; ++step) {
            const real_t reference = merit_the_long_way(f, fixture.ndofs, x->data(), h->data(), kSteps[step]);
            SFEM_TEST_ASSERT(reference > 0);
            SFEM_TEST_ASSERT(std::abs(merits[step] - reference) <= real_t(1e-10) * reference);
        }
        return SFEM_TEST_SUCCESS;
    }

    /// The accumulator is genuinely in the sum.
    ///
    /// Without it the merit would be the material's own residual norm, which
    /// omits the traction and is not zero at the solution of the combined
    /// system.  Comparing against a Function with no traction at all shows the
    /// two are different numbers, so the test above cannot be passing by
    /// accident.
    int test_the_forcing_is_inside_the_norm() {
        auto fixture = make_fixture();
        auto with    = make_residual_system(fixture, true);

        auto x = sfem::create_host_buffer<real_t>(fixture.ndofs);
        auto h = sfem::create_host_buffer<real_t>(fixture.ndofs);
        seed(fixture.ndofs, 1, x->data());
        seed(fixture.ndofs, 2, h->data());

        std::vector<real_t> merits(4, 0);
        SFEM_TEST_ASSERT(with->residual_merit(x->data(), h->data(), 4, kSteps, merits.data()) == SFEM_SUCCESS);

        auto material = sfem::Factory::create_op(fixture.space, "GeneratedMooneyRivlinKelvinVoigt");
        auto previous = sfem::create_host_buffer<real_t>(fixture.ndofs);
        seed(fixture.ndofs, 9, previous->data());
        material->set_field("previous", previous, 0);
        auto bare = sfem::Function::create(fixture.space);
        bare->add_operator(material);

        std::vector<real_t> bare_merits(4, 0);
        SFEM_TEST_ASSERT(bare->residual_merit(x->data(), h->data(), 4, kSteps, bare_merits.data()) ==
                         SFEM_SUCCESS);

        bool any_differs = false;
        for (int step = 0; step < 4; ++step) {
            any_differs = any_differs || std::abs(merits[step] - bare_merits[step]) > real_t(1e-12);
        }
        SFEM_TEST_ASSERT(any_differs);
        return SFEM_TEST_SUCCESS;
    }

    /// The operator that reduces is kept last whatever order the caller adds in,
    /// so a driver does not have to know the rule.
    int test_insertion_order_does_not_matter() {
        auto fixture = make_fixture();

        auto x = sfem::create_host_buffer<real_t>(fixture.ndofs);
        auto h = sfem::create_host_buffer<real_t>(fixture.ndofs);
        seed(fixture.ndofs, 1, x->data());
        seed(fixture.ndofs, 2, h->data());

        std::vector<real_t> material_first(4, 0);
        std::vector<real_t> forcing_first(4, 0);
        SFEM_TEST_ASSERT(make_residual_system(fixture, true)->residual_merit(
                                 x->data(), h->data(), 4, kSteps, material_first.data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(make_residual_system(fixture, false)->residual_merit(
                                 x->data(), h->data(), 4, kSteps, forcing_first.data()) == SFEM_SUCCESS);

        for (int step = 0; step < 4; ++step) {
            SFEM_TEST_ASSERT(material_first[step] > 0);
            SFEM_TEST_ASSERT(std::abs(material_first[step] - forcing_first[step]) <=
                             real_t(1e-12) * material_first[step]);
        }
        return SFEM_TEST_SUCCESS;
    }

    /// A caller's buffer and the `Function`'s own must give the same answer, and
    /// the buffer must be zeroed rather than assumed zero: it is reused.
    int test_the_caller_may_own_the_accumulator() {
        auto fixture = make_fixture();
        auto f       = make_residual_system(fixture, true);

        auto x = sfem::create_host_buffer<real_t>(fixture.ndofs);
        auto h = sfem::create_host_buffer<real_t>(fixture.ndofs);
        seed(fixture.ndofs, 1, x->data());
        seed(fixture.ndofs, 2, h->data());

        auto accumulator = sfem::create_host_buffer<real_t>(fixture.ndofs);
        // Dirtied on purpose: a reused buffer is not zero, and the entry point
        // is what has to say so.
        for (ptrdiff_t i = 0; i < fixture.ndofs; ++i) {
            accumulator->data()[i] = real_t(7);
        }

        std::vector<real_t> owned(4, 0);
        std::vector<real_t> supplied(4, 0);
        SFEM_TEST_ASSERT(f->residual_merit(x->data(), h->data(), 4, kSteps, owned.data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(f->residual_merit(x->data(), h->data(), 4, kSteps, accumulator->data(),
                                           supplied.data()) == SFEM_SUCCESS);

        for (int step = 0; step < 4; ++step) {
            SFEM_TEST_ASSERT(owned[step] > 0);
            SFEM_TEST_ASSERT(std::abs(owned[step] - supplied[step]) <= real_t(1e-12) * owned[step]);
        }
        return SFEM_TEST_SUCCESS;
    }

    /// A system with no potential says so rather than inventing an energy, and
    /// refuses recoverably: the caller can fall back to the merit every system
    /// has.
    int test_a_system_without_a_potential_refuses_the_energy() {
        auto fixture = make_fixture();
        auto f       = make_residual_system(fixture, true);

        SFEM_TEST_ASSERT(!f->has_energy_merit());

        auto   x     = sfem::create_host_buffer<real_t>(fixture.ndofs);
        seed(fixture.ndofs, 1, x->data());
        real_t value = 0;
        SFEM_TEST_ASSERT(f->energy_merit(x->data(), &value) == SFEM_FAILURE);

        // And the other one works on the same Function.
        SFEM_TEST_ASSERT(f->residual_merit(x->data(), &value) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(value > 0);
        return SFEM_TEST_SUCCESS;
    }

    /// An energy system offers both: the potential it owns, and the merit every
    /// system has.
    int test_an_energy_system_offers_both_merits() {
        auto fixture = make_fixture();
        auto op      = sfem::Factory::create_op(fixture.space, "GeneratedNeoHookeanOgden");
        SFEM_TEST_ASSERT(op != nullptr);
        for (const auto &block : fixture.mesh->blocks()) {
            op->set_value_in_block(block->name(), "mu", real_t(3));
            op->set_value_in_block(block->name(), "lmbda", real_t(1));
        }
        auto f = sfem::Function::create(fixture.space);
        f->add_operator(op);

        SFEM_TEST_ASSERT(f->has_energy_merit());

        auto x = sfem::create_host_buffer<real_t>(fixture.ndofs);
        seed(fixture.ndofs, 3, x->data());

        real_t energy = 0;
        real_t merit  = 0;
        SFEM_TEST_ASSERT(f->energy_merit(x->data(), &energy) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(f->residual_merit(x->data(), &merit) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(energy != 0);
        SFEM_TEST_ASSERT(merit > 0);

        // They are different quantities, which is why the caller names one.
        SFEM_TEST_ASSERT(std::abs(energy - merit) > real_t(1e-14));
        return SFEM_TEST_SUCCESS;
    }

    /// A gravity line search used to abort.
    ///
    /// `energy_merit` asks every operator for its stepped 0-form, and the volume
    /// forcing declared only `value`, so it reached `Op::value_steps` and its
    /// `SFEM_ERROR`.  The work is linear in the state, so one assembly and two
    /// dots serve every step -- the same identity the traction uses -- and the
    /// test holds both that it runs and that the identity is exact.
    int test_a_body_force_can_be_line_searched() {
        auto fixture = make_fixture();
        auto op      = sfem::Factory::create_op(fixture.space, "GeneratedBodyForce");
        SFEM_TEST_ASSERT(op != nullptr);
        for (auto &block : fixture.mesh->blocks()) {
            op->set_value_in_block(block->name(), "density", real_t(1300));
            op->set_value_in_block(block->name(), "g0", real_t(0));
            op->set_value_in_block(block->name(), "g1", real_t(0));
            op->set_value_in_block(block->name(), "g2", real_t(-9.81));
        }

        auto f = sfem::Function::create(fixture.space);
        f->add_operator(op);
        SFEM_TEST_ASSERT(f->has_energy_merit());

        auto x = sfem::create_host_buffer<real_t>(fixture.ndofs);
        auto h = sfem::create_host_buffer<real_t>(fixture.ndofs);
        seed(fixture.ndofs, 4, x->data());
        seed(fixture.ndofs, 5, h->data());

        std::vector<real_t> stepped(4, 0);
        SFEM_TEST_ASSERT(f->energy_merit(x->data(), h->data(), 4, kSteps, stepped.data()) == SFEM_SUCCESS);

        // Against the value at each point formed explicitly: the split is exact,
        // not an approximation that happens to be close.
        for (int step = 0; step < 4; ++step) {
            auto point = sfem::create_host_buffer<real_t>(fixture.ndofs);
            for (ptrdiff_t i = 0; i < fixture.ndofs; ++i) {
                point->data()[i] = x->data()[i] + kSteps[step] * h->data()[i];
            }
            real_t direct = 0;
            SFEM_TEST_ASSERT(f->energy_merit(point->data(), &direct) == SFEM_SUCCESS);
            SFEM_TEST_ASSERT(std::abs(stepped[step] - direct) <= real_t(1e-12) * (1 + std::abs(direct)));
        }
        return SFEM_TEST_SUCCESS;
    }

}  // namespace

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_the_residual_merit_is_the_functions_own_residual);
    SFEM_RUN_TEST(test_the_forcing_is_inside_the_norm);
    SFEM_RUN_TEST(test_insertion_order_does_not_matter);
    SFEM_RUN_TEST(test_the_caller_may_own_the_accumulator);
    SFEM_RUN_TEST(test_a_system_without_a_potential_refuses_the_energy);
    SFEM_RUN_TEST(test_an_energy_system_offers_both_merits);
    SFEM_RUN_TEST(test_a_body_force_can_be_line_searched);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
