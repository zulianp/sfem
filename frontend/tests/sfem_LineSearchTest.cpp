// The line search's one rule on a merit known in closed form: m(s) = (s - a)^2 along any
// direction, so m0 = a^2 and m'(0) = -2a.

#include <cmath>
#include <limits>
#include <set>

#include "sfem_LineSearch.hpp"
#include "sfem_test.hpp"

namespace {

    class Parabola final : public sfem::Merit {
    public:
        Parabola(const real_t a, const bool batched) : a_(a), batched_(batched) {}

        const char *name() const override { return "parabola"; }

        int evaluate(const real_t *const, const real_t *const, const int nsteps, const real_t *const steps, real_t *const out) override {
            calls++;
            for (int i = 0; i < nsteps; ++i) {
                out[i] = unevaluable.count(steps[i]) ? std::numeric_limits<real_t>::infinity() : (steps[i] - a_) * (steps[i] - a_);
            }
            return SFEM_SUCCESS;
        }

        real_t at(const real_t *const, const real_t *const) override { return a_ * a_; }
        real_t slope(const real_t *const, const real_t *const, const real_t *const, const real_t) override { return -2 * a_; }
        bool   batched() const override { return batched_; }

        std::set<real_t> unevaluable;
        int              calls{0};

    private:
        real_t a_;
        bool   batched_;
    };

    sfem::LineSearchResult search(Parabola &m, const sfem::LineSearchOptions &options = sfem::LineSearchOptions()) {
        sfem::LineSearch ls(options);
        return ls.search(m, nullptr, nullptr, m.at(nullptr, nullptr), m.slope(nullptr, nullptr, nullptr, 0));
    }

}  // namespace

// a = 0.3: s = 1 overshoots (0.49 > 0.09), s = 1/2 is the first to satisfy Armijo.
int test_first_armijo_step() {
    Parabola lazy(0.3, false), batched(0.3, true);
    auto     l = search(lazy);
    auto     b = search(batched);
    SFEM_TEST_ASSERT(l.status == sfem::LineSearchResult::Status::ACCEPTED);
    SFEM_TEST_ASSERT(l.alpha == real_t(0.5));
    SFEM_TEST_ASSERT(b.status == l.status && b.alpha == l.alpha && b.merit == l.merit);
    SFEM_TEST_ASSERT(l.evaluations == 2 && lazy.calls == 2);
    SFEM_TEST_ASSERT(batched.calls == 1);
    return SFEM_TEST_SUCCESS;
}

// Batching switched off evaluates a batched merit step by step, stopping at the accepted one, and
// takes the same step.
int test_batching_can_be_switched_off() {
    Parabola                m(0.3, true);
    sfem::LineSearchOptions options;
    options.batched = false;
    auto r          = search(m, options);
    SFEM_TEST_ASSERT(r.status == sfem::LineSearchResult::Status::ACCEPTED && r.alpha == real_t(0.5));
    SFEM_TEST_ASSERT(m.calls == 2 && r.evaluations == 2);
    return SFEM_TEST_SUCCESS;
}

// A trial that cannot be evaluated is passed over, never accepted.
int test_non_finite_trials_are_skipped() {
    Parabola m(1, false);
    m.unevaluable = {real_t(1), real_t(0.5)};
    auto r        = search(m);
    SFEM_TEST_ASSERT(r.status == sfem::LineSearchResult::Status::ACCEPTED);
    SFEM_TEST_ASSERT(r.alpha == real_t(0.25));

    for (real_t s : sfem::LineSearchOptions().steps) m.unevaluable.insert(s);
    SFEM_TEST_ASSERT(search(m).status == sfem::LineSearchResult::Status::NON_FINITE);
    return SFEM_TEST_SUCCESS;
}

// Along an ascent direction nothing decreases: the search fails rather than taking a step.
int test_ascent_direction_fails() {
    Parabola m(-1, true);
    auto     r = search(m);
    SFEM_TEST_ASSERT(r.status == sfem::LineSearchResult::Status::NO_DECREASE);
    SFEM_TEST_ASSERT(r.alpha == 0);
    return SFEM_TEST_SUCCESS;
}

// A predicted decrease below the merit's rounding is convergence, not failure.
int test_at_resolution() {
    Parabola m(1e-16, true);
    SFEM_TEST_ASSERT(search(m).status == sfem::LineSearchResult::Status::AT_RESOLUTION);
    return SFEM_TEST_SUCCESS;
}

// No step satisfies Armijo (the direction is not known to descend), but the best one decreases.
int test_best_decrease() {
    Parabola         m(0.3, true);
    sfem::LineSearch ls;
    auto             r = ls.search(m, nullptr, nullptr, m.at(nullptr, nullptr), 0);
    SFEM_TEST_ASSERT(r.status == sfem::LineSearchResult::Status::BEST_DECREASE);
    SFEM_TEST_ASSERT(r.alpha == real_t(0.25));
    return SFEM_TEST_SUCCESS;
}

// The damped rule takes damping times the step whatever the merit does (a = 0.05: the merit rises
// from 0.0025 to 0.5625 at s = 0.8), and shortens it only past a state it cannot evaluate.
int test_damped_ignores_the_merit() {
    Parabola                m(0.05, false);
    sfem::LineSearchOptions options;
    options.rule    = sfem::LineSearchOptions::Rule::DAMPED;
    options.damping = real_t(0.8);
    auto r          = search(m, options);
    SFEM_TEST_ASSERT(r.status == sfem::LineSearchResult::Status::ACCEPTED && r.alpha == real_t(0.8) && r.evaluations == 1);

    m.unevaluable = {real_t(0.8)};
    r             = search(m, options);
    SFEM_TEST_ASSERT(r.status == sfem::LineSearchResult::Status::ACCEPTED && r.alpha == real_t(0.4));

    for (real_t s : sfem::LineSearchOptions().steps) m.unevaluable.insert(real_t(0.8) * s);
    SFEM_TEST_ASSERT(search(m, options).status == sfem::LineSearchResult::Status::NON_FINITE);
    return SFEM_TEST_SUCCESS;
}

int test_disabled_takes_the_full_step() {
    Parabola                m(0.3, true);
    sfem::LineSearchOptions options;
    options.enabled = false;
    auto r          = search(m, options);
    SFEM_TEST_ASSERT(r.status == sfem::LineSearchResult::Status::ACCEPTED && r.alpha == 1);
    SFEM_TEST_ASSERT(m.calls == 0);
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_first_armijo_step);
    SFEM_RUN_TEST(test_batching_can_be_switched_off);
    SFEM_RUN_TEST(test_non_finite_trials_are_skipped);
    SFEM_RUN_TEST(test_damped_ignores_the_merit);
    SFEM_RUN_TEST(test_ascent_direction_fails);
    SFEM_RUN_TEST(test_at_resolution);
    SFEM_RUN_TEST(test_best_decrease);
    SFEM_RUN_TEST(test_disabled_takes_the_full_step);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
