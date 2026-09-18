// One material, several schemes, no regeneration.
//
// `mooney_rivlin_kelvin_voigt` used to name Newmark in its own form: it
// declared a material parameter called `newmark_velocity_alpha` and multiplied
// the displacement gradient by it.  It now writes `gen.dt(u)` and names no
// scheme at all, which lowers to `u_dot = u_dt_shift * u + u_old`.  What fills
// those two in is a `TimeScheme` the operator holds.
//
// The tests below are the three properties that buys, in the order they
// matter: the scheme reaches the kernels, a different scheme reaches them
// differently without anything being recompiled, and the residual-based merit
// reads the same shift and history the residual does.

#include "sfem_test.hpp"

#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_NewmarkScheme.hpp"
#include "sfem_TimeScheme.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace {

    void set_material_parameter(const std::shared_ptr<sfem::Op>   &op,
                                const std::shared_ptr<sfem::Mesh> &mesh,
                                const char *const                  name,
                                const real_t                       value) {
        for (const auto &block : mesh->blocks()) {
            op->set_value_in_block(block->name(), name, value);
        }
    }

    struct Fixture {
        std::shared_ptr<sfem::Mesh>          mesh;
        std::shared_ptr<sfem::FunctionSpace> space;
        std::shared_ptr<sfem::Op>            op;
        ptrdiff_t                            ndofs{0};
    };

    Fixture make_fixture() {
        Fixture fixture;
        fixture.mesh  = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 2, 2, 2);
        fixture.space = sfem::FunctionSpace::create(fixture.mesh, 3);
        fixture.op    = sfem::Factory::create_op(fixture.space, "GeneratedMooneyRivlinKelvinVoigt");
        fixture.ndofs = fixture.space->n_dofs();

        set_material_parameter(fixture.op, fixture.mesh, "mu", real_t(3));
        set_material_parameter(fixture.op, fixture.mesh, "lmbda", real_t(1));
        set_material_parameter(fixture.op, fixture.mesh, "eta_s", real_t(0.05));
        set_material_parameter(fixture.op, fixture.mesh, "eta_b", real_t(0.01));
        return fixture;
    }

    /// A state, a velocity and an acceleration that are not multiples of one
    /// another, so a kernel reading the wrong one of them is visible.
    void seed_state(const ptrdiff_t n, const int salt, real_t *const values) {
        for (ptrdiff_t i = 0; i < n; ++i) {
            values[i] = real_t(1e-3) * std::sin(real_t(0.5 * (i + 1) + salt));
        }
    }

    int make_scheme(const Fixture                        &fixture,
                    const real_t                          beta,
                    const real_t                          gamma,
                    const real_t                          dt,
                    std::shared_ptr<sfem::NewmarkScheme> &out) {
        auto scheme = std::make_shared<sfem::NewmarkScheme>(fixture.space);
        scheme->set_beta(beta);
        scheme->set_gamma(gamma);
        scheme->set_density(real_t(2));
        SFEM_TEST_ASSERT(scheme->initialize() == SFEM_SUCCESS);

        seed_state(fixture.ndofs, 0, scheme->state()->data());
        seed_state(fixture.ndofs, 1, scheme->velocity()->data());
        seed_state(fixture.ndofs, 2, scheme->acceleration()->data());
        scheme->begin_step(0, dt);
        out = scheme;
        return SFEM_TEST_SUCCESS;
    }

    /// The residual of the one operator that holds the scheme, and the same
    /// thing assembled the long way: the material with the shift and the
    /// history set by hand, plus the scheme's own separable term added
    /// separately.  They have to agree, which is the statement that holding
    /// the scheme loses nothing and adds nothing.
    int gradients_held_and_pushed(const Fixture                              &fixture,
                                   const std::shared_ptr<sfem::NewmarkScheme> &scheme,
                                   const real_t *const                         x,
                                   real_t *const                               held,
                                   real_t *const                               pushed) {
        auto steppable = std::dynamic_pointer_cast<sfem::TimeSteppable>(fixture.op);
        SFEM_TEST_ASSERT(steppable != nullptr);

        std::fill(held, held + fixture.ndofs, real_t(0));
        steppable->set_time_scheme(scheme);
        SFEM_TEST_ASSERT(fixture.op->gradient(x, held) == SFEM_SUCCESS);

        // The same numbers, reached the old way.  The history has to be copied
        // out first: detaching the scheme is what makes the buffer the
        // operator's own again.
        auto history = sfem::create_host_buffer<real_t>(fixture.ndofs);
        std::copy(scheme->history(), scheme->history() + fixture.ndofs, history->data());

        steppable->set_time_scheme(nullptr);
        set_material_parameter(fixture.op, fixture.mesh, "u_dt_shift", scheme->shift());
        fixture.op->set_field("previous", history, 0);

        std::fill(pushed, pushed + fixture.ndofs, real_t(0));
        SFEM_TEST_ASSERT(fixture.op->gradient(x, pushed) == SFEM_SUCCESS);
        // The separable half, which the held form gets through the operator.
        SFEM_TEST_ASSERT(scheme->inertia_op()->gradient(x, pushed) == SFEM_SUCCESS);
        return SFEM_TEST_SUCCESS;
    }

    bool has_nonzero(const ptrdiff_t n, const real_t *const values) {
        for (ptrdiff_t i = 0; i < n; ++i) {
            if (values[i] != real_t(0)) return true;
        }
        return false;
    }

    int test_the_material_reads_the_scheme_it_was_handed() {
        auto fixture = make_fixture();
        SFEM_TEST_ASSERT(fixture.op != nullptr);
        SFEM_TEST_ASSERT(fixture.op->initialize() == SFEM_SUCCESS);

        std::shared_ptr<sfem::NewmarkScheme> scheme;
        SFEM_TEST_ASSERT(make_scheme(fixture, real_t(0.25), real_t(0.5), real_t(0.01), scheme) ==
                         SFEM_TEST_SUCCESS);

        auto x = sfem::create_host_buffer<real_t>(fixture.ndofs);
        seed_state(fixture.ndofs, 3, x->data());

        auto held   = sfem::create_host_buffer<real_t>(fixture.ndofs);
        auto pushed = sfem::create_host_buffer<real_t>(fixture.ndofs);
        SFEM_TEST_ASSERT(gradients_held_and_pushed(
                                 fixture, scheme, x->data(), held->data(), pushed->data()) ==
                         SFEM_TEST_SUCCESS);

        SFEM_TEST_ASSERT(has_nonzero(fixture.ndofs, held->data()));
        // To rounding, not bit for bit: the two assemblies add the material's
        // contribution and the scheme's in the opposite order.
        for (ptrdiff_t i = 0; i < fixture.ndofs; ++i) {
            SFEM_TEST_ASSERT(std::fabs(held->data()[i] - pushed->data()[i]) <=
                             1e-12 * std::fabs(pushed->data()[i]) + 1e-15);
        }
        return SFEM_TEST_SUCCESS;
    }

    int test_a_second_scheme_needs_no_regeneration() {
        auto fixture = make_fixture();
        SFEM_TEST_ASSERT(fixture.op->initialize() == SFEM_SUCCESS);

        auto x = sfem::create_host_buffer<real_t>(fixture.ndofs);
        seed_state(fixture.ndofs, 3, x->data());

        // Two Newmark parameterisations: the trapezoidal rule, and the
        // first-order-accurate one.  Different beta, gamma and dt, so
        // different shift and different history -- out of the same kernels.
        std::shared_ptr<sfem::NewmarkScheme> trapezoidal;
        std::shared_ptr<sfem::NewmarkScheme> damped;
        SFEM_TEST_ASSERT(make_scheme(fixture, real_t(0.25), real_t(0.5), real_t(0.01), trapezoidal) ==
                         SFEM_TEST_SUCCESS);
        SFEM_TEST_ASSERT(make_scheme(fixture, real_t(0.5), real_t(1.0), real_t(0.02), damped) ==
                         SFEM_TEST_SUCCESS);
        SFEM_TEST_ASSERT(trapezoidal->shift() != damped->shift());

        auto first_held   = sfem::create_host_buffer<real_t>(fixture.ndofs);
        auto first_pushed = sfem::create_host_buffer<real_t>(fixture.ndofs);
        SFEM_TEST_ASSERT(gradients_held_and_pushed(fixture,
                                                   trapezoidal,
                                                   x->data(),
                                                   first_held->data(),
                                                   first_pushed->data()) == SFEM_TEST_SUCCESS);

        auto second_held   = sfem::create_host_buffer<real_t>(fixture.ndofs);
        auto second_pushed = sfem::create_host_buffer<real_t>(fixture.ndofs);
        SFEM_TEST_ASSERT(gradients_held_and_pushed(fixture,
                                                   damped,
                                                   x->data(),
                                                   second_held->data(),
                                                   second_pushed->data()) == SFEM_TEST_SUCCESS);

        for (ptrdiff_t i = 0; i < fixture.ndofs; ++i) {
            SFEM_TEST_ASSERT(std::fabs(second_held->data()[i] - second_pushed->data()[i]) <=
                             1e-12 * std::fabs(second_pushed->data()[i]) + 1e-15);
        }

        // And the two schemes really do give different answers, or the test
        // above would pass with the scheme ignored entirely.
        bool differs = false;
        for (ptrdiff_t i = 0; i < fixture.ndofs; ++i) {
            differs = differs || first_held->data()[i] != second_held->data()[i];
        }
        SFEM_TEST_ASSERT(differs);
        return SFEM_TEST_SUCCESS;
    }

    int test_the_merit_reads_what_the_residual_reads() {
        auto fixture = make_fixture();
        SFEM_TEST_ASSERT(fixture.op->initialize() == SFEM_SUCCESS);

        std::shared_ptr<sfem::NewmarkScheme> scheme;
        SFEM_TEST_ASSERT(make_scheme(fixture, real_t(0.25), real_t(0.5), real_t(0.01), scheme) ==
                         SFEM_TEST_SUCCESS);
        auto steppable = std::dynamic_pointer_cast<sfem::TimeSteppable>(fixture.op);
        SFEM_TEST_ASSERT(steppable != nullptr);
        steppable->set_time_scheme(scheme);

        // One operator.  The scheme's own term rides in the material's
        // residual, because the material is what holds the scheme.
        auto f = sfem::Function::create(fixture.space);
        f->add_operator(fixture.op);
        SFEM_TEST_ASSERT(f->reduces_node_wise());

        auto x = sfem::create_host_buffer<real_t>(fixture.ndofs);
        seed_state(fixture.ndofs, 3, x->data());

        real_t merit = 0;
        SFEM_TEST_ASSERT(f->value(x->data(), &merit) == SFEM_SUCCESS);

        auto residual = sfem::create_host_buffer<real_t>(fixture.ndofs);
        SFEM_TEST_ASSERT(f->gradient(x->data(), residual->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(has_nonzero(fixture.ndofs, residual->data()));

        real_t expected = 0;
        for (ptrdiff_t i = 0; i < fixture.ndofs; ++i) {
            expected += residual->data()[i] * residual->data()[i];
        }
        expected *= real_t(0.5);
        SFEM_TEST_ASSERT(std::fabs(merit - expected) <= 1e-10 * std::fabs(expected) + 1e-14);

        // The shift and the history belong to the step, not to the iterate, so
        // a line search must not move them: `value_steps` re-assembles the
        // residual at each trial step length, and each answer has to be the
        // merit at that state and nothing else.  This is the property that a
        // scheme recomputing itself from `x` would break.
        auto                h      = sfem::create_host_buffer<real_t>(fixture.ndofs);
        seed_state(fixture.ndofs, 4, h->data());
        const real_t        steps[] = {real_t(0), real_t(-0.5), real_t(0.25), real_t(1)};
        const int           nsteps  = 4;
        std::vector<real_t> stepped(nsteps, real_t(0));
        SFEM_TEST_ASSERT(f->value_steps(x->data(), h->data(), nsteps, steps, stepped.data()) == SFEM_SUCCESS);

        auto trial = sfem::create_host_buffer<real_t>(fixture.ndofs);
        for (int step = 0; step < nsteps; ++step) {
            for (ptrdiff_t i = 0; i < fixture.ndofs; ++i) {
                trial->data()[i] = x->data()[i] + steps[step] * h->data()[i];
            }
            real_t at_trial = 0;
            SFEM_TEST_ASSERT(f->value(trial->data(), &at_trial) == SFEM_SUCCESS);
            SFEM_TEST_ASSERT(std::fabs(stepped[step] - at_trial) <=
                             1e-10 * std::fabs(at_trial) + 1e-14);
        }

        // And the merit at the original state is unchanged by having visited
        // the others, which is the same freezing property read backwards.
        real_t again = 0;
        SFEM_TEST_ASSERT(f->value(x->data(), &again) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(again == merit);

        // The inertia really is inside that residual, and not merely absent
        // without anyone noticing.  Detaching the scheme has to remove exactly
        // the separable term the scheme publishes -- which is the assertion
        // that would fail if the operator stopped forwarding to it.
        auto inertia = sfem::create_host_buffer<real_t>(fixture.ndofs);
        SFEM_TEST_ASSERT(scheme->inertia_op()->gradient(x->data(), inertia->data()) ==
                         SFEM_SUCCESS);
        SFEM_TEST_ASSERT(has_nonzero(fixture.ndofs, inertia->data()));

        steppable->set_time_scheme(nullptr);
        set_material_parameter(fixture.op, fixture.mesh, "u_dt_shift", scheme->shift());
        auto history = sfem::create_host_buffer<real_t>(fixture.ndofs);
        std::copy(scheme->history(), scheme->history() + fixture.ndofs, history->data());
        fixture.op->set_field("previous", history, 0);

        auto without = sfem::create_host_buffer<real_t>(fixture.ndofs);
        SFEM_TEST_ASSERT(f->gradient(x->data(), without->data()) == SFEM_SUCCESS);
        for (ptrdiff_t i = 0; i < fixture.ndofs; ++i) {
            const real_t expected = without->data()[i] + inertia->data()[i];
            SFEM_TEST_ASSERT(std::fabs(residual->data()[i] - expected) <=
                             1e-10 * std::fabs(expected) + 1e-14);
        }
        return SFEM_TEST_SUCCESS;
    }

    /// `advance` is the other half of the step boundary, and the only part of
    /// the scheme the tests above do not reach: they stop at the residual.
    /// Newmark's reconstruction is checked here against the same two numbers
    /// the scheme publishes, so a sign or a factor that drifts is visible
    /// without a time loop and without a mesh on disk.
    int test_advance_reconstructs_the_state() {
        auto fixture = make_fixture();

        const real_t dt    = real_t(0.01);
        const real_t beta  = real_t(0.25);
        const real_t gamma = real_t(0.5);

        std::shared_ptr<sfem::NewmarkScheme> scheme;
        SFEM_TEST_ASSERT(make_scheme(fixture, beta, gamma, dt, scheme) == SFEM_TEST_SUCCESS);

        // What the scheme carried into the step, kept before `advance`
        // overwrites it.
        auto u_before = sfem::create_host_buffer<real_t>(fixture.ndofs);
        auto v_before = sfem::create_host_buffer<real_t>(fixture.ndofs);
        auto a_before = sfem::create_host_buffer<real_t>(fixture.ndofs);
        auto history  = sfem::create_host_buffer<real_t>(fixture.ndofs);
        std::copy(scheme->state()->data(), scheme->state()->data() + fixture.ndofs, u_before->data());
        std::copy(scheme->velocity()->data(), scheme->velocity()->data() + fixture.ndofs, v_before->data());
        std::copy(scheme->acceleration()->data(),
                  scheme->acceleration()->data() + fixture.ndofs,
                  a_before->data());
        std::copy(scheme->history(), scheme->history() + fixture.ndofs, history->data());

        const real_t shift   = scheme->shift();
        const real_t alpha_a = scheme->weight("alpha_a");
        SFEM_TEST_ASSERT(std::fabs(shift - gamma / (beta * dt)) <= 1e-12 * shift);
        SFEM_TEST_ASSERT(std::fabs(alpha_a - 1 / (beta * dt * dt)) <= 1e-12 * alpha_a);
        SFEM_TEST_ASSERT(scheme->weight("dt") == dt);

        auto x = sfem::create_host_buffer<real_t>(fixture.ndofs);
        seed_state(fixture.ndofs, 5, x->data());
        scheme->advance(x->data());

        for (ptrdiff_t i = 0; i < fixture.ndofs; ++i) {
            // The predictor, rebuilt here from what went in.
            const real_t u_hat = u_before->data()[i] + dt * v_before->data()[i] +
                                 dt * dt * (real_t(0.5) - beta) * a_before->data()[i];
            const real_t a = alpha_a * (x->data()[i] - u_hat);
            const real_t v = shift * x->data()[i] + history->data()[i];

            SFEM_TEST_ASSERT(scheme->state()->data()[i] == x->data()[i]);
            SFEM_TEST_ASSERT(std::fabs(scheme->velocity()->data()[i] - v) <= 1e-10 * std::fabs(v) + 1e-14);
            SFEM_TEST_ASSERT(std::fabs(scheme->acceleration()->data()[i] - a) <=
                             1e-10 * std::fabs(a) + 1e-14);
        }

        // And the history it publishes really is the bracket of Newmark's
        // velocity update, not the previous displacement.
        for (ptrdiff_t i = 0; i < fixture.ndofs; ++i) {
            const real_t u_hat = u_before->data()[i] + dt * v_before->data()[i] +
                                 dt * dt * (real_t(0.5) - beta) * a_before->data()[i];
            const real_t expected =
                    v_before->data()[i] + dt * (1 - gamma) * a_before->data()[i] - shift * u_hat;
            SFEM_TEST_ASSERT(std::fabs(history->data()[i] - expected) <=
                             1e-10 * std::fabs(expected) + 1e-14);
        }
        return SFEM_TEST_SUCCESS;
    }

}  // namespace

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_the_material_reads_the_scheme_it_was_handed);
    SFEM_RUN_TEST(test_a_second_scheme_needs_no_regeneration);
    SFEM_RUN_TEST(test_the_merit_reads_what_the_residual_reads);
    SFEM_RUN_TEST(test_advance_reconstructs_the_state);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
