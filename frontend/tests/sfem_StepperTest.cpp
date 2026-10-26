// The Stepper's sequence, recorded by stand-ins for the scheme, the history and the solver; and
// the carried states of the library's schemes.

#include <cmath>
#include <string>
#include <vector>

#include "sfem_BDF2Scheme.hpp"
#include "sfem_NewmarkScheme.hpp"
#include "sfem_Stepper.hpp"
#include "sfem_elasticity_test_problem.hpp"
#include "sfem_test.hpp"

namespace {
    using Log = std::vector<std::string>;

    std::string fmt(const char *what, const real_t a, const real_t b) {
        char buf[128];
        snprintf(buf, sizeof(buf), "%s(%g,%g)", what, (double)a, (double)b);
        return buf;
    }

    class RecordingScheme final : public sfem::TimeScheme {
    public:
        explicit RecordingScheme(Log &log) : log_(log) {}
        real_t        shift() const override { return 0; }
        const real_t *history() const override { return nullptr; }
        real_t        weight(const char *) const override { return 0; }
        void          begin_step(real_t t, real_t dt) override { log_.push_back(fmt("scheme.begin", t, dt)); }
        void          advance(const real_t *const) override { log_.push_back("scheme.advance"); }
        void          reconstruct(const real_t *const, real_t *const, real_t *const) const override {}
        void          start(real_t *const) const override { log_.push_back("scheme.start"); }
        void          carried_state(real_t *const, real_t *const) const override {}

    private:
        Log &log_;
    };

    class RecordingHistory final : public sfem::HistoryDependent {
    public:
        explicit RecordingHistory(Log &log) : log_(log) {}
        void begin_step(const real_t t, const real_t dt) override { log_.push_back(fmt("history.begin", t, dt)); }
        void advance(const real_t *const) override { log_.push_back("history.advance"); }
        void reset_history(const real_t *const) override {}

    private:
        Log &log_;
    };

    /// Fails the attempts listed in `failures` (counted from 1), converges otherwise.
    class RecordingSolver final : public sfem::NonlinearSolver {
    public:
        RecordingSolver(Log &log, std::vector<int> failures) : log_(log), failures_(std::move(failures)) {}

        sfem::NonlinearReport solve(const sfem::StepContext &context, real_t *const) override {
            log_.push_back(fmt("solve", context.t, context.dt));
            sfem::NonlinearReport report;
            attempt_++;
            report.status = sfem::NonlinearReport::Status::CONVERGED;
            for (int f : failures_) {
                if (f == attempt_) report.status = sfem::NonlinearReport::Status::LINE_SEARCH_FAILED;
            }
            report.iterations = 1;
            return report;
        }

        int commit(const real_t *const) override {
            log_.push_back("commit");
            return SFEM_SUCCESS;
        }

    private:
        Log             &log_;
        std::vector<int> failures_;
        int              attempt_{0};
    };

    class RecordingObserver final : public sfem::StepObserver {
    public:
        explicit RecordingObserver(Log &log) : log_(log) {}
        int initial(const sfem::StepRecord &r) override { return push("initial", r); }
        int converged(const sfem::StepRecord &r) override { return push("converged", r); }
        int committed(const sfem::StepRecord &r) override { return push("committed", r); }
        int failed(const sfem::StepRecord &r) override { return push("failed", r); }

    private:
        int push(const char *what, const sfem::StepRecord &r) {
            log_.push_back(fmt(what, r.t, r.step));
            return SFEM_SUCCESS;
        }
        Log &log_;
    };

    bool same(const Log &actual, const Log &expected) {
        bool ok = actual.size() == expected.size();
        for (size_t i = 0; ok && i < actual.size(); ++i) ok = actual[i] == expected[i];
        if (!ok) {
            printf("expected:\n");
            for (auto &e : expected) printf("  %s\n", e.c_str());
            printf("actual:\n");
            for (auto &a : actual) printf("  %s\n", a.c_str());
        }
        return ok;
    }
}  // namespace

// Conditions, scheme, history, start, solve, observers, commit, history, scheme: in that order.
int test_sequence() {
    auto p = sfem::test::make_cube_problem("GeneratedLinearElasticity", 1, 0);
    Log  log;

    sfem::StepperOptions options;
    options.dt    = real_t(0.5);
    options.steps = 2;
    sfem::Stepper stepper(p.f, std::make_shared<RecordingSolver>(log, std::vector<int>{}), options);
    stepper.set_time_scheme(std::make_shared<RecordingScheme>(log));
    stepper.add_history(std::make_shared<RecordingHistory>(log));
    stepper.add_time_dependent([&log](const real_t t) {
        log.push_back(fmt("set_time", t, 0));
        return SFEM_SUCCESS;
    });
    stepper.add_observer(std::make_shared<RecordingObserver>(log));

    SFEM_TEST_ASSERT(stepper.run(p.x->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(same(log,
                          {"set_time(0,0)",
                           "initial(0,0)",
                           "set_time(0.5,0)",
                           "scheme.begin(0,0.5)",
                           "history.begin(0.5,0.5)",
                           "scheme.start",
                           "solve(0,0.5)",
                           "converged(0.5,1)",
                           "commit",
                           "history.advance",
                           "scheme.advance",
                           "committed(0.5,1)",
                           "set_time(1,0)",
                           "scheme.begin(0.5,0.5)",
                           "history.begin(1,0.5)",
                           "scheme.start",
                           "solve(0.5,0.5)",
                           "converged(1,2)",
                           "commit",
                           "history.advance",
                           "scheme.advance",
                           "committed(1,2)"}));
    return SFEM_TEST_SUCCESS;
}

// A failed attempt is retried with half the step, which opens the step again; the next step is
// again of full length, and the last one ends at t_end.
int test_cutback() {
    auto p = sfem::test::make_cube_problem("GeneratedLinearElasticity", 1, 0);
    Log  log;

    sfem::StepperOptions options;
    options.dt           = 1;
    options.steps        = 2;
    options.max_cutbacks = 1;
    sfem::Stepper stepper(p.f, std::make_shared<RecordingSolver>(log, std::vector<int>{1}), options);
    stepper.add_history(std::make_shared<RecordingHistory>(log));
    stepper.add_observer(std::make_shared<RecordingObserver>(log));

    SFEM_TEST_ASSERT(stepper.run(p.x->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(same(log,
                          {"initial(0,0)",
                           "history.begin(1,1)",
                           "solve(0,1)",
                           "failed(1,1)",
                           "history.begin(0.5,0.5)",
                           "solve(0,0.5)",
                           "converged(0.5,1)",
                           "commit",
                           "history.advance",
                           "committed(0.5,1)",
                           "history.begin(1.5,1)",
                           "solve(0.5,1)",
                           "converged(1.5,2)",
                           "commit",
                           "history.advance",
                           "committed(1.5,2)",
                           "history.begin(2,0.5)",
                           "solve(1.5,0.5)",
                           "converged(2,3)",
                           "commit",
                           "history.advance",
                           "committed(2,3)"}));

    // Without cut-backs the failure ends the run.
    Log                  log2;
    sfem::StepperOptions strict = options;
    strict.max_cutbacks         = 0;
    sfem::Stepper failing(p.f, std::make_shared<RecordingSolver>(log2, std::vector<int>{1}), strict);
    SFEM_TEST_ASSERT(failing.run(p.x->data()) == SFEM_FAILURE);
    return SFEM_TEST_SUCCESS;
}

// A released constraint leaves the function at the first step ending at or after its time: the
// pulled face is constrained in steps 1 (t = 1) and not in steps 2 and 3.
int test_release() {
    auto p = sfem::test::make_cube_problem("GeneratedLinearElasticity", 2, real_t(0.1));

    // A shear of the pulled face, as its own constraint so it can be released (the pull along x
    // is the problem's own).
    auto right = sfem::Sideset::create_from_selector(p.mesh, [](const geom_t x, const geom_t, const geom_t) -> bool {
        return std::abs(x - 1) < 1e-8;
    });
    std::vector<sfem::DirichletConditions::Condition> conditions = {{.sidesets = {right}, .value = real_t(0.1), .component = 1}};
    auto pull = sfem::DirichletConditions::create(p.fs, conditions);

    Log                  log;
    sfem::StepperOptions options;
    options.steps = 3;
    sfem::Stepper stepper(p.f, std::make_shared<RecordingSolver>(log, std::vector<int>{}), options);
    stepper.add_release(pull, 2);
    p.f->add_constraint(pull);

    int constraints_at_step[4] = {0, 0, 0, 0};
    class Count final : public sfem::StepObserver {
    public:
        Count(sfem::Function &f, int *out) : f_(f), out_(out) {}
        int converged(const sfem::StepRecord &r) override {
            std::vector<mask_t> mask(mask_count(f_.space()->n_dofs()), 0);
            f_.constraints_mask(mask.data());
            int n = 0;
            for (ptrdiff_t i = 0; i < f_.space()->n_dofs(); ++i) n += mask_get(i, mask.data());
            out_[r.step] = n;
            return SFEM_SUCCESS;
        }

    private:
        sfem::Function &f_;
        int            *out_;
    };
    stepper.add_observer(std::make_shared<Count>(*p.f, constraints_at_step));

    SFEM_TEST_ASSERT(stepper.run(p.x->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(constraints_at_step[1] > constraints_at_step[2]);
    SFEM_TEST_ASSERT(constraints_at_step[2] == constraints_at_step[3]);
    return SFEM_TEST_SUCCESS;
}

// What a scheme carries out of a step is what `reconstruct` gives at its solution, for the
// schemes that carry it unmodified.
template <class Scheme>
int carried_matches_reconstruct() {
    auto            p     = sfem::test::make_cube_problem("GeneratedLinearElasticity", 2, 0);
    const ptrdiff_t n     = p.fs->n_dofs();
    auto            sch   = std::make_shared<Scheme>(p.fs);
    sch->set_density(2);
    SFEM_TEST_ASSERT(sch->initialize() == SFEM_SUCCESS);

    std::vector<real_t> x(n), v(n), a(n), cv(n), ca(n);
    for (int step = 0; step < 3; ++step) {
        sch->begin_step(step * real_t(0.1), real_t(0.1));
        for (ptrdiff_t i = 0; i < n; ++i) x[i] = std::sin(real_t(0.3) * (i + 1) + step);
        sch->reconstruct(x.data(), v.data(), a.data());
        sch->advance(x.data());
        sch->carried_state(cv.data(), ca.data());
        for (ptrdiff_t i = 0; i < n; ++i) {
            SFEM_TEST_ASSERT(std::abs(cv[i] - v[i]) <= 1e-12 * (1 + std::abs(v[i])));
            SFEM_TEST_ASSERT(std::abs(ca[i] - a[i]) <= 1e-12 * (1 + std::abs(a[i])));
        }
    }
    return SFEM_TEST_SUCCESS;
}

int test_newmark_carried_state() { return carried_matches_reconstruct<sfem::NewmarkScheme>(); }
int test_bdf2_carried_state() { return carried_matches_reconstruct<sfem::BDF2Scheme>(); }

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_sequence);
    SFEM_RUN_TEST(test_cutback);
    SFEM_RUN_TEST(test_release);
    SFEM_RUN_TEST(test_newmark_carried_state);
    SFEM_RUN_TEST(test_bdf2_carried_state);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
