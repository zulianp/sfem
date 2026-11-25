// NewtonSolver: one step for a linear problem, quadratic convergence for a nonlinear one, and
// stopping as soon as it knows it is failing.

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_NewtonSolver.hpp"
#include "sfem_UnconstrainedStep.hpp"
#include "sfem_elasticity_test_problem.hpp"
#include "sfem_test.hpp"

// A linear problem with an exact inner solve converges in one iteration, whose CG run is the
// one a direct CG solve of the same system makes.
int test_linear_one_iteration() {
    // Both CG runs on one thread: their dot products then sum in the same order (with threads a
    // run could take one iteration more than the other at rtol 1e-12).
    sfem::test::SerialScope serial;
    auto p = sfem::test::make_cube_problem("GeneratedLinearElasticity", 4, real_t(0.1));

    sfem::LinearSolverOptions linear;
    linear.rtol = 1e-12;
    auto step   = sfem::UnconstrainedStep::create(p.f, sfem::EXECUTION_SPACE_HOST, linear);

    sfem::NewtonOptions options;
    options.rtol = 1e-10;
    sfem::NewtonSolver newton(p.f, step, options);

    auto x0     = sfem::create_host_buffer<real_t>(p.fs->n_dofs());
    std::copy(p.x->data(), p.x->data() + p.fs->n_dofs(), x0->data());

    auto report = newton.solve(sfem::StepContext(), p.x->data());
    SFEM_TEST_ASSERT(report.status == sfem::NonlinearReport::Status::CONVERGED);
    SFEM_TEST_ASSERT(report.iterations == 1);

    // The same CG on the same system.
    const ptrdiff_t n   = p.fs->n_dofs();
    auto            op  = sfem::hessian_bsr(p.f, x0, sfem::EXECUTION_SPACE_HOST);
    auto            rhs = sfem::create_host_buffer<real_t>(n);
    auto            dx  = sfem::create_host_buffer<real_t>(n);
    auto            dg  = sfem::create_host_buffer<real_t>(n);
    p.f->gradient(x0->data(), rhs->data());
    p.f->set_value_to_constrained_dofs(0, rhs->data());
    for (ptrdiff_t i = 0; i < n; ++i) rhs->data()[i] = -rhs->data()[i];
    auto tangent = sfem::BSRTangent::create(p.f, sfem::EXECUTION_SPACE_HOST);
    tangent->assemble(x0->data());
    tangent->diagonal(dg->data());
    p.f->set_value_to_constrained_dofs(1, dg->data());
    auto cg     = sfem::create_cg<real_t>(op, sfem::EXECUTION_SPACE_HOST);
    cg->verbose = false;
    cg->set_rtol(1e-12);
    cg->set_atol(linear.atol);
    cg->set_max_it(linear.max_it);
    cg->set_preconditioner_op(sfem::create_shiftable_jacobi(dg, sfem::EXECUTION_SPACE_HOST));
    SFEM_TEST_ASSERT(cg->apply(rhs->data(), dx->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(cg->iterations() == report.linear_iterations);

    for (ptrdiff_t i = 0; i < n; ++i) {
        SFEM_TEST_ASSERT(std::abs(x0->data()[i] + dx->data()[i] - p.x->data()[i]) < 1e-14);
    }
    return SFEM_TEST_SUCCESS;
}

// With exact inner solves, Newton's residual contracts quadratically near the solution. Judged on
// the residual merit: at rtol 1e-12 the decrease of the energy falls below the energy's rounding.
int test_neohookean_quadratic() {
    auto p = sfem::test::make_cube_problem("GeneratedNeoHookeanOgden", 4, real_t(0.3));

    sfem::LinearSolverOptions linear;
    linear.rtol = 1e-12;
    auto step   = sfem::UnconstrainedStep::create(p.f, sfem::EXECUTION_SPACE_HOST, linear);

    sfem::NewtonOptions options;
    options.rtol  = 1e-12;
    options.merit = "residual";
    sfem::NewtonSolver newton(p.f, step, options);
    std::vector<real_t> r;
    newton.add_monitor([&r](const sfem::NewtonIteration &it) { r.push_back(it.measure.value); });

    auto report = newton.solve(sfem::StepContext(), p.x->data());
    printf("  %s after %d iterations\n", sfem::to_string(report.status), report.iterations);
    SFEM_TEST_ASSERT(report.status == sfem::NonlinearReport::Status::CONVERGED);
    SFEM_TEST_ASSERT(report.iterations <= 8);

    for (size_t k = 0; k + 1 < r.size(); ++k) {
        printf("  |r_%zu| / |r_0| = %.3e\n", k + 1, double(r[k + 1] / r[0]));
    }
    // Once the iterate is close (|r| below 1e-2 |r_0|), each step at least squares the error.
    int quadratic_steps = 0;
    for (size_t k = 1; k + 1 < r.size(); ++k) {
        const real_t e = r[k] / r[0], next = r[k + 1] / r[0];
        if (e < 1e-2 && next > 1e-13) {
            SFEM_TEST_ASSERT(next <= 10 * e * e);
            quadratic_steps++;
        }
    }
    SFEM_TEST_ASSERT(quadratic_steps >= 1);
    return SFEM_TEST_SUCCESS;
}

namespace {
    // A step whose measure and correction are scripted, to drive the solver into each way of
    // stopping.
    class ScriptedStep final : public sfem::LinearizedStep {
    public:
        explicit ScriptedStep(const ptrdiff_t ndofs) : ndofs_(ndofs) {}

        const char *name() const override { return "scripted"; }

        sfem::StepMeasure measure(const real_t *const, const real_t *const r) override {
            sfem::StepMeasure m;
            m.value = m.reference = measures.empty() ? 1 : measures[std::min(calls_++, measures.size() - 1)];
            if (measures.empty()) {
                real_t s = 0;
                for (ptrdiff_t i = 0; i < ndofs_; ++i) s += r[i] * r[i];
                m.value = m.reference = std::sqrt(s);
            }
            return m;
        }

        int linearize(const real_t *const, const real_t *const) override { return SFEM_SUCCESS; }

        int solve(const real_t *const, const real_t *const r, real_t *const dx) override {
            for (ptrdiff_t i = 0; i < ndofs_; ++i) dx[i] = direction * r[i];
            carried.push_back(carries_increment);
            return SFEM_SUCCESS;
        }

        void set_carries_increment(const bool on) override { carries_increment = on; }

        std::shared_ptr<sfem::Merit> merit() override { return own_merit; }

        int iterations() const override { return 0; }

        bool tangent_projection_supported() const override { return true; }
        void set_tangent_projection(const bool on) override { projected = on; }

        std::vector<real_t>          measures;
        real_t                       direction{0};
        bool                         projected{false};
        bool                         carries_increment{false};
        // Whether each solve carried the increment, in order.
        std::vector<int>             carried;
        // Null: the function's merit.
        std::shared_ptr<sfem::Merit> own_merit;

    private:
        ptrdiff_t ndofs_;
        size_t    calls_{0};
    };

    // A merit whose values are scripted: `values` at each iterate Newton measures it at, `trials`
    // at each step the line search tries, in order (the last repeated); its slope -1 along every
    // direction.
    class ScriptedMerit final : public sfem::Merit {
    public:
        const char *name() const override { return "scripted"; }

        int evaluate(const real_t *const, const real_t *const, const int nsteps, const real_t *const, real_t *const out) override {
            for (int i = 0; i < nsteps; ++i) out[i] = trials[std::min(trial_++, trials.size() - 1)];
            return SFEM_SUCCESS;
        }

        real_t at(const real_t *const, const real_t *const) override { return values[std::min(value_++, values.size() - 1)]; }

        real_t slope(const real_t *const, const real_t *const, const real_t *const, const real_t) override { return -1; }

        bool batched() const override { return false; }

        std::vector<real_t> values, trials;

    private:
        size_t value_{0};
        size_t trial_{0};
    };

    sfem::NonlinearReport run(sfem::test::CubeProblem &p, const std::shared_ptr<ScriptedStep> &step, sfem::NewtonOptions options) {
        sfem::NewtonSolver newton(p.f, step, options);
        return newton.solve(sfem::StepContext(), p.x->data());
    }
}  // namespace

int test_fails_fast() {
    auto p = sfem::test::make_cube_problem("GeneratedNeoHookeanOgden", 2, real_t(0.1));
    const ptrdiff_t n = p.fs->n_dofs();

    // Along +r the residual grows: no decrease, stopped after one iteration.
    {
        auto step       = std::make_shared<ScriptedStep>(n);
        step->direction = 1;
        auto report     = run(p, step, sfem::NewtonOptions());
        SFEM_TEST_ASSERT(report.status == sfem::NonlinearReport::Status::LINE_SEARCH_FAILED);
        SFEM_TEST_ASSERT(report.iterations == 1);
    }

    sfem::NewtonOptions no_search;
    no_search.line_search.enabled = false;

    // A measure that grows past the divergence factor.
    {
        auto step      = std::make_shared<ScriptedStep>(n);
        step->measures = {1, 10, 1e5};
        auto report    = run(p, step, no_search);
        SFEM_TEST_ASSERT(report.status == sfem::NonlinearReport::Status::DIVERGED);
        SFEM_TEST_ASSERT(report.iterations == 2);
    }

    // A measure that cannot be evaluated.
    {
        auto step      = std::make_shared<ScriptedStep>(n);
        step->measures = {1, std::numeric_limits<real_t>::quiet_NaN()};
        auto report    = run(p, step, no_search);
        SFEM_TEST_ASSERT(report.status == sfem::NonlinearReport::Status::NON_FINITE);
        SFEM_TEST_ASSERT(report.iterations == 1);
    }

    // Stopped by the cap, and said so.
    {
        auto step      = std::make_shared<ScriptedStep>(n);
        step->measures = {1};
        no_search.max_it = 3;
        auto report      = run(p, step, no_search);
        SFEM_TEST_ASSERT(report.status == sfem::NonlinearReport::Status::MAX_IT);
        SFEM_TEST_ASSERT(report.iterations == 3);
        SFEM_TEST_ASSERT(!report.converged());
    }

    // A correction that is not finite.
    {
        auto step       = std::make_shared<ScriptedStep>(n);
        step->direction = std::numeric_limits<real_t>::quiet_NaN();
        auto report     = run(p, step, sfem::NewtonOptions());
        SFEM_TEST_ASSERT(report.status == sfem::NonlinearReport::Status::STEP_FAILED);
    }
    return SFEM_TEST_SUCCESS;
}

// CG on an indefinite matrix stops at the first direction of non-positive curvature and returns
// a direction of descent for the quadratic 1/2 x'Ax - b'x: b'x > 0 from x = 0.
int test_cg_stops_on_negative_curvature() {
    for (const bool preconditioned : {false, true}) {
        for (const std::vector<real_t> &diagonal : {std::vector<real_t>{2, 1, -1, 3}, std::vector<real_t>{-1, -2, -1, -3}}) {
            const ptrdiff_t n  = diagonal.size();
            auto            op = sfem::make_op<real_t>(
                    n,
                    n,
                    [diagonal, n](const real_t *const x, real_t *const y) {
                        for (ptrdiff_t i = 0; i < n; ++i) y[i] += diagonal[i] * x[i];
                    },
                    sfem::EXECUTION_SPACE_HOST);
            auto cg     = sfem::create_cg<real_t>(op, sfem::EXECUTION_SPACE_HOST);
            cg->verbose = false;
            cg->set_n_dofs(n);
            cg->set_max_it(100);
            cg->set_rtol(1e-12);
            cg->set_stop_on_negative_curvature(true);
            if (preconditioned) {
                cg->set_preconditioner_op(sfem::make_op<real_t>(
                        n,
                        n,
                        [n](const real_t *const x, real_t *const y) {
                            for (ptrdiff_t i = 0; i < n; ++i) y[i] += x[i];
                        },
                        sfem::EXECUTION_SPACE_HOST));
            }

            std::vector<real_t> b(n, 1), x(n, 0);
            SFEM_TEST_ASSERT(cg->apply(b.data(), x.data()) == SFEM_SUCCESS);
            SFEM_TEST_ASSERT(cg->negative_curvature());
            real_t descent = 0;
            for (ptrdiff_t i = 0; i < n; ++i) descent += b[i] * x[i];
            SFEM_TEST_ASSERT(descent > 0);
        }
    }
    return SFEM_TEST_SUCCESS;
}

// tangent_projection: on_failure with project_below_alpha: a step the line search shortened below
// the fraction is not taken, and the iteration is repeated on the projected tangent; at 0 (the
// default), or without on_failure, it is taken. Every step here is a quarter of the direction.
int test_projects_below_alpha() {
    const sfem::TangentProjection projections[] = {
            sfem::TangentProjection::ON_FAILURE, sfem::TangentProjection::ON_FAILURE, sfem::TangentProjection::NONE};
    const real_t below[] = {0, real_t(0.5), real_t(0.5)};
    for (int c = 0; c < 3; ++c) {
        auto p          = sfem::test::make_cube_problem("GeneratedNeoHookeanOgden", 2, real_t(0.1));
        auto step       = std::make_shared<ScriptedStep>(p.fs->n_dofs());
        step->direction = -1e-3;  // a descent direction for the residual's merit, any step length
        sfem::NewtonOptions options;
        options.tangent_projection  = projections[c];
        options.line_search.steps   = {real_t(0.25)};
        options.project_below_alpha = below[c];
        options.max_it              = 3;

        sfem::NewtonSolver  newton(p.f, step, options);
        std::vector<real_t> measures, alphas;
        std::vector<int>    projected, discarded;
        newton.add_monitor([&](const sfem::NewtonIteration &it) {
            measures.push_back(it.measure.value);
            alphas.push_back(it.line_search.alpha);
            projected.push_back(it.projected);
            discarded.push_back(it.discarded);
        });
        auto report = newton.solve(sfem::StepContext(), p.x->data());
        printf("  tangent_projection %s, project_below_alpha %g: %s after %d iterations, %d projections\n",
               projections[c] == sfem::TangentProjection::NONE ? "none" : "on_failure",
               double(below[c]),
               sfem::to_string(report.status),
               report.iterations,
               report.projections);

        // Three linear solves, the cap; the iterate measured before each and after the last.
        SFEM_TEST_ASSERT(report.status == sfem::NonlinearReport::Status::MAX_IT);
        SFEM_TEST_ASSERT(report.iterations == 3);
        SFEM_TEST_ASSERT(measures.size() == 4);
        SFEM_TEST_ASSERT(report.min_alpha == real_t(0.25));
        if (c == 1) {
            // Projected once, and kept (no step is taken whole). The first direction is not
            // taken: the monitor shows x where it was, the step 0, then two projected steps.
            SFEM_TEST_ASSERT(report.projections == 1);
            SFEM_TEST_ASSERT(step->projected);
            SFEM_TEST_ASSERT(measures[1] == measures[0]);
            SFEM_TEST_ASSERT(alphas[1] == 0);
            SFEM_TEST_ASSERT(discarded[1] && !discarded[2] && !discarded[3]);
            SFEM_TEST_ASSERT(!projected[1] && projected[2] && projected[3]);
            SFEM_TEST_ASSERT(alphas[2] == real_t(0.25) && alphas[3] == real_t(0.25));
            SFEM_TEST_ASSERT(measures[2] < measures[1]);
        } else {
            // Three steps taken on the tangent itself.
            SFEM_TEST_ASSERT(report.projections == 0);
            SFEM_TEST_ASSERT(!step->projected);
            for (int k = 1; k < 4; ++k) {
                SFEM_TEST_ASSERT(alphas[k] == real_t(0.25));
                SFEM_TEST_ASSERT(!projected[k] && !discarded[k]);
                SFEM_TEST_ASSERT(measures[k] < measures[k - 1]);
            }
        }
    }
    return SFEM_TEST_SUCCESS;
}

// project_below_alpha with fixed_iterations: a step shortened below the fraction on the last
// iteration is taken, no iteration being left to repeat it projected; with max_it a cap instead,
// it is discarded as on any other iteration. One iteration, a quarter of the direction.
int test_below_alpha_last_fixed_iteration() {
    for (const bool fixed : {true, false}) {
        auto p          = sfem::test::make_cube_problem("GeneratedNeoHookeanOgden", 2, real_t(0.1));
        auto step       = std::make_shared<ScriptedStep>(p.fs->n_dofs());
        step->direction = -1e-3;
        sfem::NewtonOptions options;
        options.tangent_projection  = sfem::TangentProjection::ON_FAILURE;
        options.line_search.steps   = {real_t(0.25)};
        options.project_below_alpha = real_t(0.5);
        options.max_it              = 1;
        options.fixed_iterations    = fixed;

        sfem::NewtonSolver  newton(p.f, step, options);
        std::vector<real_t> measures, alphas;
        std::vector<int>    discarded;
        newton.add_monitor([&](const sfem::NewtonIteration &it) {
            measures.push_back(it.measure.value);
            alphas.push_back(it.line_search.alpha);
            discarded.push_back(it.discarded);
        });
        auto report = newton.solve(sfem::StepContext(), p.x->data());
        printf("  fixed_iterations %d: %s after %d iterations, %d projections\n",
               fixed,
               sfem::to_string(report.status),
               report.iterations,
               report.projections);

        SFEM_TEST_ASSERT(report.iterations == 1);
        SFEM_TEST_ASSERT(measures.size() == 2);
        if (fixed) {
            SFEM_TEST_ASSERT(report.status == sfem::NonlinearReport::Status::ITERATIONS_DONE);
            SFEM_TEST_ASSERT(report.projections == 0);
            SFEM_TEST_ASSERT(!step->projected);
            SFEM_TEST_ASSERT(alphas[1] == real_t(0.25) && !discarded[1]);
            SFEM_TEST_ASSERT(measures[1] < measures[0]);
        } else {
            SFEM_TEST_ASSERT(report.status == sfem::NonlinearReport::Status::MAX_IT);
            SFEM_TEST_ASSERT(report.projections == 1);
            SFEM_TEST_ASSERT(alphas[1] == 0 && discarded[1]);
            SFEM_TEST_ASSERT(measures[1] == measures[0]);
        }
    }
    return SFEM_TEST_SUCCESS;
}

// The iteration repeated after a discarded step (project_below_alpha) does not push its iterate's
// merit into the nonmonotone window again: a second copy would push out the older, larger merit
// that the repeated iteration's step needs as its reference. A scripted merit, a window of two,
// the steps 1 and 1/4:
//  - from 10, the full step to 5 is accepted;
//  - from 5, only 1/4 decreases it (to 4.9): below 1/2, discarded, the iteration repeated;
//  - from 5 again, the projected direction raises it (7 and 6), within Armijo of 10, the
//    window's largest: a nonmonotone full step. With 5 in the window twice, 10 is out of it, and
//    the line search finds no decrease.
int test_discarded_step_keeps_window() {
    auto p          = sfem::test::make_cube_problem("GeneratedNeoHookeanOgden", 2, real_t(0.1));
    auto step       = std::make_shared<ScriptedStep>(p.fs->n_dofs());
    auto merit      = std::make_shared<ScriptedMerit>();
    merit->values   = {10, 5, 5};
    merit->trials   = {5, 6, real_t(4.9), 7, 6};
    step->own_merit = merit;
    sfem::NewtonOptions options;
    options.tangent_projection         = sfem::TangentProjection::ON_FAILURE;
    options.line_search.steps          = {1, real_t(0.25)};
    options.line_search.nonmonotone    = 2;
    options.project_below_alpha        = real_t(0.5);
    options.max_it                     = 3;

    sfem::NewtonSolver                          newton(p.f, step, options);
    std::vector<sfem::LineSearchResult::Status> statuses;
    std::vector<real_t>                         alphas;
    std::vector<int>                            discarded;
    newton.add_monitor([&](const sfem::NewtonIteration &it) {
        statuses.push_back(it.line_search.status);
        alphas.push_back(it.line_search.alpha);
        discarded.push_back(it.discarded);
    });
    auto report = newton.solve(sfem::StepContext(), p.x->data());
    printf("  %s after %d iterations, %d projections\n", sfem::to_string(report.status), report.iterations, report.projections);

    SFEM_TEST_ASSERT(report.status == sfem::NonlinearReport::Status::MAX_IT);
    SFEM_TEST_ASSERT(report.iterations == 3);
    SFEM_TEST_ASSERT(report.projections == 1);
    SFEM_TEST_ASSERT(statuses.size() == 4);
    SFEM_TEST_ASSERT(statuses[1] == sfem::LineSearchResult::Status::ACCEPTED && alphas[1] == 1);
    SFEM_TEST_ASSERT(discarded[2] && alphas[2] == 0);
    SFEM_TEST_ASSERT(statuses[3] == sfem::LineSearchResult::Status::NONMONOTONE && alphas[3] == 1 && !discarded[3]);
    return SFEM_TEST_SUCCESS;
}

// The correction that carries a prescribed boundary increment is solved with the step told so
// (LinearizedStep::set_carries_increment), and Newton's iterations after it without: the cube
// from rest, its stretch prescribed, two fixed iterations after the increment. From the stretched
// state there is no increment, and no solve carries one.
int test_increment_announced() {
    for (const bool from_rest : {true, false}) {
        auto p = sfem::test::make_cube_problem("GeneratedNeoHookeanOgden", 2, real_t(0.1));
        if (from_rest) std::fill(p.x->data(), p.x->data() + p.fs->n_dofs(), real_t(0));
        auto step      = std::make_shared<ScriptedStep>(p.fs->n_dofs());
        step->measures = {1};
        sfem::NewtonOptions options;
        options.line_search.enabled = false;
        options.fixed_iterations    = true;
        options.max_it              = 2;
        auto report                 = run(p, step, options);
        printf("  from %s: %s, solves carrying the increment:", from_rest ? "rest" : "the stretch", sfem::to_string(report.status));
        for (const int c : step->carried) printf(" %d", c);
        printf("\n");

        SFEM_TEST_ASSERT(report.status == sfem::NonlinearReport::Status::ITERATIONS_DONE);
        SFEM_TEST_ASSERT(!step->carries_increment);
        if (from_rest) {
            SFEM_TEST_ASSERT(step->carried.size() == 3);
            SFEM_TEST_ASSERT(step->carried[0] && !step->carried[1] && !step->carried[2]);
        } else {
            SFEM_TEST_ASSERT(step->carried.size() == 2);
            SFEM_TEST_ASSERT(!step->carried[0] && !step->carried[1]);
        }
    }
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_linear_one_iteration);
    SFEM_RUN_TEST(test_neohookean_quadratic);
    SFEM_RUN_TEST(test_fails_fast);
    SFEM_RUN_TEST(test_cg_stops_on_negative_curvature);
    SFEM_RUN_TEST(test_projects_below_alpha);
    SFEM_RUN_TEST(test_below_alpha_last_fixed_iteration);
    SFEM_RUN_TEST(test_discarded_step_keeps_window);
    SFEM_RUN_TEST(test_increment_announced);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
