// NewtonSolver: one step for a linear problem, quadratic convergence for a nonlinear one, and
// stopping as soon as it knows it is failing.

#include <cmath>
#include <limits>
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
    SFEM_TEST_ASSERT(report.status == sfem::NonlinearReport::CONVERGED);
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
    SFEM_TEST_ASSERT(report.status == sfem::NonlinearReport::CONVERGED);
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
            return SFEM_SUCCESS;
        }

        int iterations() const override { return 0; }

        std::vector<real_t> measures;
        real_t              direction{0};

    private:
        ptrdiff_t ndofs_;
        size_t    calls_{0};
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
        SFEM_TEST_ASSERT(report.status == sfem::NonlinearReport::LINE_SEARCH_FAILED);
        SFEM_TEST_ASSERT(report.iterations == 1);
    }

    sfem::NewtonOptions no_search;
    no_search.line_search.enabled = false;

    // A measure that grows past the divergence factor.
    {
        auto step      = std::make_shared<ScriptedStep>(n);
        step->measures = {1, 10, 1e5};
        auto report    = run(p, step, no_search);
        SFEM_TEST_ASSERT(report.status == sfem::NonlinearReport::DIVERGED);
        SFEM_TEST_ASSERT(report.iterations == 2);
    }

    // A measure that cannot be evaluated.
    {
        auto step      = std::make_shared<ScriptedStep>(n);
        step->measures = {1, std::numeric_limits<real_t>::quiet_NaN()};
        auto report    = run(p, step, no_search);
        SFEM_TEST_ASSERT(report.status == sfem::NonlinearReport::NON_FINITE);
        SFEM_TEST_ASSERT(report.iterations == 1);
    }

    // Stopped by the cap, and said so.
    {
        auto step      = std::make_shared<ScriptedStep>(n);
        step->measures = {1};
        no_search.max_it = 3;
        auto report      = run(p, step, no_search);
        SFEM_TEST_ASSERT(report.status == sfem::NonlinearReport::MAX_IT);
        SFEM_TEST_ASSERT(report.iterations == 3);
        SFEM_TEST_ASSERT(!report.converged());
    }

    // A correction that is not finite.
    {
        auto step       = std::make_shared<ScriptedStep>(n);
        step->direction = std::numeric_limits<real_t>::quiet_NaN();
        auto report     = run(p, step, sfem::NewtonOptions());
        SFEM_TEST_ASSERT(report.status == sfem::NonlinearReport::STEP_FAILED);
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

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_linear_one_iteration);
    SFEM_RUN_TEST(test_neohookean_quadratic);
    SFEM_RUN_TEST(test_fails_fast);
    SFEM_RUN_TEST(test_cg_stops_on_negative_curvature);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
