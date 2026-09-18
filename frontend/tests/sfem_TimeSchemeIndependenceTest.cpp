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
#include "sfem_BDF2Scheme.hpp"
#include "sfem_BackwardEulerScheme.hpp"
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

    /// The claim the whole arrangement exists to buy, stated across schemes of
    /// two different families rather than two parameterisations of one.
    ///
    /// `BackwardEulerScheme` is first order and publishes no separable term;
    /// `NewmarkScheme` is second order and publishes an inertia.  The material
    /// is the same object in both cases, compiled once, and neither scheme is
    /// named anywhere in its form.
    int test_a_first_order_scheme_runs_the_same_material() {
        auto fixture = make_fixture();
        SFEM_TEST_ASSERT(fixture.op->initialize() == SFEM_SUCCESS);

        auto steppable = std::dynamic_pointer_cast<sfem::TimeSteppable>(fixture.op);
        SFEM_TEST_ASSERT(steppable != nullptr);

        const real_t dt     = real_t(0.01);
        auto         scheme = std::make_shared<sfem::BackwardEulerScheme>(fixture.space);
        SFEM_TEST_ASSERT(scheme->initialize() == SFEM_SUCCESS);
        seed_state(fixture.ndofs, 0, scheme->state()->data());
        scheme->begin_step(0, dt);

        // Backward Euler is `shift = 1/dt`, `z = -u_n/dt`, and nothing else.
        SFEM_TEST_ASSERT(std::fabs(scheme->shift() - 1 / dt) <= 1e-12 / dt);
        for (ptrdiff_t i = 0; i < fixture.ndofs; ++i) {
            const real_t expected = -scheme->state()->data()[i] / dt;
            SFEM_TEST_ASSERT(std::fabs(scheme->history()[i] - expected) <=
                             1e-12 * std::fabs(expected) + 1e-15);
        }
        // No separable term, so nothing is added to the material's residual.
        SFEM_TEST_ASSERT(scheme->inertia_op() == nullptr);

        auto x = sfem::create_host_buffer<real_t>(fixture.ndofs);
        seed_state(fixture.ndofs, 3, x->data());

        // The same equality the Newmark cases assert: held equals the long-hand
        // assembly.  With no separable term the two are bit for bit.
        auto held = sfem::create_host_buffer<real_t>(fixture.ndofs);
        steppable->set_time_scheme(scheme);
        SFEM_TEST_ASSERT(fixture.op->gradient(x->data(), held->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(has_nonzero(fixture.ndofs, held->data()));

        auto history = sfem::create_host_buffer<real_t>(fixture.ndofs);
        std::copy(scheme->history(), scheme->history() + fixture.ndofs, history->data());
        steppable->set_time_scheme(nullptr);
        set_material_parameter(fixture.op, fixture.mesh, "u_dt_shift", scheme->shift());
        fixture.op->set_field("previous", history, 0);

        auto pushed = sfem::create_host_buffer<real_t>(fixture.ndofs);
        SFEM_TEST_ASSERT(fixture.op->gradient(x->data(), pushed->data()) == SFEM_SUCCESS);
        for (ptrdiff_t i = 0; i < fixture.ndofs; ++i) {
            SFEM_TEST_ASSERT(held->data()[i] == pushed->data()[i]);
        }

        // And it carries the step: `advance` is the whole of its memory.
        scheme->advance(x->data());
        for (ptrdiff_t i = 0; i < fixture.ndofs; ++i) {
            SFEM_TEST_ASSERT(scheme->state()->data()[i] == x->data()[i]);
        }
        return SFEM_TEST_SUCCESS;
    }

}  // namespace

/// BDF2 reproduces the algebra `hyperelasticity_bdf2` folded into its own step
/// loop as literal constants.
///
/// The driver predicted with `4/3 u_n - 1/3 u_nm1 + 8/9 dt v_n - 2/9 dt v_nm1`,
/// set the inertia's alpha to `9/(4 dt^2)`, and reconstructed the velocity as
/// `(3u - 4u_n + u_nm1)/(2dt)`, with a backward-Euler first step because BDF2
/// is not self-starting.  Every one of those is checked here against a value
/// written out by hand, because the point of moving them into a scheme is that
/// the driver stops restating them -- and a move is only safe if what arrives
/// is the same arithmetic.
int test_bdf2_matches_the_hand_written_algebra() {
    auto         fixture = make_fixture();
    const real_t dt      = real_t(0.05);

    auto scheme = std::make_shared<sfem::BDF2Scheme>(fixture.space);
    scheme->set_density(real_t(2));
    SFEM_TEST_ASSERT(scheme->initialize() == SFEM_SUCCESS);

    auto u0 = sfem::create_host_buffer<real_t>(fixture.ndofs);
    auto v0 = sfem::create_host_buffer<real_t>(fixture.ndofs);
    seed_state(fixture.ndofs, 0, u0->data());
    seed_state(fixture.ndofs, 1, v0->data());
    std::copy(u0->data(), u0->data() + fixture.ndofs, scheme->state()->data());
    std::copy(v0->data(), v0->data() + fixture.ndofs, scheme->velocity()->data());

    // Step one is backward Euler: v = (u - u_n)/dt, a = (v - v_n)/dt.
    scheme->begin_step(dt, dt);
    SFEM_TEST_ASSERT(std::abs(scheme->shift() - 1 / dt) <= 1e-12);
    SFEM_TEST_ASSERT(std::abs(scheme->weight("alpha_a") - 1 / (dt * dt)) <= 1e-12);
    SFEM_TEST_ASSERT(scheme->weight("order") == real_t(1));
    for (ptrdiff_t i = 0; i < fixture.ndofs; ++i) {
        SFEM_TEST_ASSERT(std::abs(scheme->history()[i] - (-u0->data()[i] / dt)) <= 1e-12);
    }

    auto x1 = sfem::create_host_buffer<real_t>(fixture.ndofs);
    seed_state(fixture.ndofs, 3, x1->data());

    auto v1 = sfem::create_host_buffer<real_t>(fixture.ndofs);
    auto a1 = sfem::create_host_buffer<real_t>(fixture.ndofs);
    scheme->reconstruct(x1->data(), v1->data(), a1->data());
    for (ptrdiff_t i = 0; i < fixture.ndofs; ++i) {
        const real_t v_be = (x1->data()[i] - u0->data()[i]) / dt;
        const real_t u_hat = u0->data()[i] + dt * v0->data()[i];
        SFEM_TEST_ASSERT(std::abs(v1->data()[i] - v_be) <= 1e-12);
        SFEM_TEST_ASSERT(std::abs(a1->data()[i] - (x1->data()[i] - u_hat) / (dt * dt)) <= 1e-12);
    }

    scheme->advance(x1->data());

    // Step two, and every step after it, is BDF2 proper.
    scheme->begin_step(2 * dt, dt);
    SFEM_TEST_ASSERT(std::abs(scheme->shift() - real_t(1.5) / dt) <= 1e-12);
    SFEM_TEST_ASSERT(std::abs(scheme->weight("alpha_a") - real_t(2.25) / (dt * dt)) <= 1e-12);
    SFEM_TEST_ASSERT(scheme->weight("order") == real_t(2));

    auto x2 = sfem::create_host_buffer<real_t>(fixture.ndofs);
    seed_state(fixture.ndofs, 4, x2->data());

    auto v2 = sfem::create_host_buffer<real_t>(fixture.ndofs);
    auto a2 = sfem::create_host_buffer<real_t>(fixture.ndofs);
    scheme->reconstruct(x2->data(), v2->data(), a2->data());
    for (ptrdiff_t i = 0; i < fixture.ndofs; ++i) {
        // The driver's `bdf2_predictor` and `update_velocity_bdf2`, written out.
        const real_t u_n = x1->data()[i], u_nm1 = u0->data()[i];
        const real_t v_n = v1->data()[i], v_nm1 = v0->data()[i];
        const real_t u_hat = real_t(4.0 / 3.0) * u_n - real_t(1.0 / 3.0) * u_nm1 +
                             real_t(8.0 / 9.0) * dt * v_n - real_t(2.0 / 9.0) * dt * v_nm1;
        const real_t v_bdf2 = (3 * x2->data()[i] - 4 * u_n + u_nm1) / (2 * dt);
        const real_t a_bdf2 = real_t(2.25) / (dt * dt) * (x2->data()[i] - u_hat);

        SFEM_TEST_ASSERT(std::abs(scheme->history()[i] - (-2 * u_n + real_t(0.5) * u_nm1) / dt) <= 1e-12);
        SFEM_TEST_ASSERT(std::abs(v2->data()[i] - v_bdf2) <= 1e-12);
        SFEM_TEST_ASSERT(std::abs(a2->data()[i] - a_bdf2) <= 1e-12);
    }

    return SFEM_TEST_SUCCESS;
}

/// The two schemes are of different order, and the published interface is
/// enough to show it.
///
/// `shift` and `history` are all a material is given, so integrating a problem
/// with them is exactly what a kernel does.  On `u' = -u` the discrete equation
/// `shift*u + z = -u` closes in one line, and the error at t=1 under halving dt
/// has to fall by 2 for backward Euler and by 4 for BDF2.  That is the claim
/// that makes having two schemes worth anything: they are not two spellings of
/// the same method.
int test_the_schemes_have_the_order_they_claim() {
    auto fixture = make_fixture();

    auto error_at_one = [&](const bool second_order, const int n_steps) -> real_t {
        const real_t dt = real_t(1) / n_steps;

        std::shared_ptr<sfem::TimeScheme> scheme;
        std::shared_ptr<sfem::Buffer<real_t>> state;
        if (second_order) {
            auto s = std::make_shared<sfem::BDF2Scheme>(fixture.space);
            s->set_density(real_t(1));
            if (s->initialize() != SFEM_SUCCESS) return real_t(-1);
            state  = s->state();
            scheme = s;
        } else {
            auto s = std::make_shared<sfem::BackwardEulerScheme>(fixture.space);
            if (s->initialize() != SFEM_SUCCESS) return real_t(-1);
            state  = s->state();
            scheme = s;
        }

        std::fill(state->data(), state->data() + fixture.ndofs, real_t(1));

        auto x = sfem::create_host_buffer<real_t>(fixture.ndofs);
        for (int step = 1; step <= n_steps; ++step) {
            scheme->begin_step(step * dt, dt);
            // u' = -u  =>  shift*u + z = -u  =>  u = -z/(shift + 1).
            const real_t s = scheme->shift();
            for (ptrdiff_t i = 0; i < fixture.ndofs; ++i) {
                x->data()[i] = -scheme->history()[i] / (s + 1);
            }
            scheme->advance(x->data());
        }
        return std::abs(x->data()[0] - std::exp(real_t(-1)));
    };

    for (int second = 0; second < 2; ++second) {
        const real_t coarse = error_at_one(second != 0, 40);
        const real_t fine   = error_at_one(second != 0, 80);
        SFEM_TEST_ASSERT(coarse > 0 && fine > 0);
        const real_t rate   = std::log2(coarse / fine);
        const real_t want   = second ? real_t(2) : real_t(1);
        SFEM_TEST_ASSERT(std::abs(rate - want) < real_t(0.15));
    }

    return SFEM_TEST_SUCCESS;
}

/// Whatever a driver exports has to be what the scheme is solving with.
///
/// This is the property `hyperelasticity_bdf2` broke: it wrote
/// `(v - v_n)/dt` as "acceleration" while the inertia in its own residual used
/// `9/(4 dt^2) * (u - u_hat)`.  Nothing fed the field back, so the trajectory
/// was right and the output was a different quantity from the one being solved.
///
/// Both halves are checked against the thing itself rather than against a
/// restatement of the method:
///
///   - the velocity must equal `shift * x + history`, which is what the
///     material's kernels read, element for element;
///   - the acceleration must be the one `inertia_op` assembles.  That is
///     checked by assembling it: the inertia's gradient is `M * a`, and its
///     Hessian action on `a / alpha` is `alpha * M * (a / alpha)`, the same
///     vector.  If the reported acceleration were a different discretisation
///     the two would part company.
///
/// Run over every scheme, so a scheme added later is covered by construction.
int test_the_diagnostic_matches_the_scheme() {
    auto         fixture = make_fixture();
    const real_t dt      = real_t(0.05);

    std::vector<std::shared_ptr<sfem::TimeScheme>> schemes;
    {
        auto newmark = std::make_shared<sfem::NewmarkScheme>(fixture.space);
        newmark->set_density(real_t(2));
        SFEM_TEST_ASSERT(newmark->initialize() == SFEM_SUCCESS);
        seed_state(fixture.ndofs, 0, newmark->state()->data());
        seed_state(fixture.ndofs, 1, newmark->velocity()->data());
        seed_state(fixture.ndofs, 2, newmark->acceleration()->data());
        schemes.push_back(newmark);

        auto bdf2 = std::make_shared<sfem::BDF2Scheme>(fixture.space);
        bdf2->set_density(real_t(2));
        SFEM_TEST_ASSERT(bdf2->initialize() == SFEM_SUCCESS);
        seed_state(fixture.ndofs, 0, bdf2->state()->data());
        seed_state(fixture.ndofs, 1, bdf2->velocity()->data());
        schemes.push_back(bdf2);

        auto be = std::make_shared<sfem::BackwardEulerScheme>(fixture.space);
        SFEM_TEST_ASSERT(be->initialize() == SFEM_SUCCESS);
        seed_state(fixture.ndofs, 0, be->state()->data());
        schemes.push_back(be);
    }

    auto x = sfem::create_host_buffer<real_t>(fixture.ndofs);
    seed_state(fixture.ndofs, 5, x->data());

    for (const auto &scheme : schemes) {
        scheme->begin_step(dt, dt);

        auto v = sfem::create_host_buffer<real_t>(fixture.ndofs);
        auto a = sfem::create_host_buffer<real_t>(fixture.ndofs);
        scheme->reconstruct(x->data(), v->data(), scheme->has_acceleration() ? a->data() : nullptr);

        // The velocity is the derivative the kernels read.
        const real_t shift = scheme->shift();
        for (ptrdiff_t i = 0; i < fixture.ndofs; ++i) {
            const real_t want = shift * x->data()[i] + scheme->history()[i];
            SFEM_TEST_ASSERT(std::abs(v->data()[i] - want) <= 1e-14 * (1 + std::abs(want)));
        }

        // A scheme with no second derivative must say so rather than invent one.
        SFEM_TEST_ASSERT(scheme->has_acceleration() == (scheme->inertia_op() != nullptr));
        if (!scheme->has_acceleration()) continue;

        // The acceleration is the one the inertia assembles.
        auto inertia = scheme->inertia_op();
        auto from_gradient = sfem::create_host_buffer<real_t>(fixture.ndofs);
        SFEM_TEST_ASSERT(inertia->gradient(x->data(), from_gradient->data()) == SFEM_SUCCESS);

        const real_t alpha = scheme->weight("alpha_a");
        SFEM_TEST_ASSERT(alpha != real_t(0));
        auto scaled = sfem::create_host_buffer<real_t>(fixture.ndofs);
        for (ptrdiff_t i = 0; i < fixture.ndofs; ++i) {
            scaled->data()[i] = a->data()[i] / alpha;
        }

        auto from_apply = sfem::create_host_buffer<real_t>(fixture.ndofs);
        SFEM_TEST_ASSERT(inertia->apply(nullptr, scaled->data(), from_apply->data()) == SFEM_SUCCESS);

        real_t scale = 0;
        for (ptrdiff_t i = 0; i < fixture.ndofs; ++i) scale = std::max(scale, std::abs(from_gradient->data()[i]));
        SFEM_TEST_ASSERT(scale > 0);
        for (ptrdiff_t i = 0; i < fixture.ndofs; ++i) {
            SFEM_TEST_ASSERT(std::abs(from_gradient->data()[i] - from_apply->data()[i]) <= 1e-12 * scale);
        }
    }

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_the_material_reads_the_scheme_it_was_handed);
    SFEM_RUN_TEST(test_a_second_scheme_needs_no_regeneration);
    SFEM_RUN_TEST(test_the_merit_reads_what_the_residual_reads);
    SFEM_RUN_TEST(test_advance_reconstructs_the_state);
    SFEM_RUN_TEST(test_a_first_order_scheme_runs_the_same_material);
    SFEM_RUN_TEST(test_bdf2_matches_the_hand_written_algebra);
    SFEM_RUN_TEST(test_the_schemes_have_the_order_they_claim);
    SFEM_RUN_TEST(test_the_diagnostic_matches_the_scheme);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
