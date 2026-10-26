#ifndef SFEM_LINE_SEARCH_HPP
#define SFEM_LINE_SEARCH_HPP

#include <vector>

#include "sfem_Merit.hpp"
#include "sfem_base.hpp"

namespace sfem {

    struct LineSearchOptions {
        /// ARMIJO: the first step of the schedule whose merit satisfies Armijo. DAMPED: the Newton
        /// step times `damping`, shortened along the schedule only where the merit cannot be
        /// evaluated (an element inverted); its value is not compared. The damped rule is the
        /// classical one for containment, where a start inside the obstacle makes the merit a
        /// poor judge of the first steps.
        enum class Rule { ARMIJO, DAMPED };

        /// Off: the full step, no merit evaluated.
        bool enabled{true};
        Rule rule{Rule::ARMIJO};
        real_t damping{1};
        /// The step lengths tried, in order.
        std::vector<real_t> steps{1, real_t(1) / 2, real_t(1) / 4, real_t(1) / 8, real_t(1) / 16, real_t(1) / 32, real_t(1) / 64, real_t(1) / 128};
        /// Armijo's constant: a step s is accepted when m(s) <= m0 + c1 s m'(0).
        real_t c1{real_t(1e-4)};
        /// Evaluate the whole schedule in one call, where the merit can (`Merit::batched`): one pass
        /// over the mesh with the steps vectorised in its inner loop. Off, the steps are evaluated
        /// one at a time, stopping at the first accepted; the step taken is the same either way.
        bool batched{true};
    };

    struct LineSearchResult {
        enum class Status {
            /// The first step of the schedule that satisfies Armijo.
            ACCEPTED,
            /// None does, but the best of them decreases the merit.
            BEST_DECREASE,
            /// None decreases it, and the decrease the slope predicts is below the merit's
            /// resolution: converged as far as this merit can tell.
            AT_RESOLUTION,
            /// None decreases it although a resolvable decrease was predicted.
            NO_DECREASE,
            /// No trial could be evaluated.
            NON_FINITE
        };

        Status status{Status::NO_DECREASE};
        real_t alpha{0};
        real_t merit{0};
        int    evaluations{0};
    };

    const char *to_string(const LineSearchResult::Status status);

    /// One rule for every Newton loop: the first Armijo step of a fixed schedule, evaluated in
    /// one call when batching is on and the merit is batched, step by step otherwise -- both give
    /// the same step.
    class LineSearch {
    public:
        explicit LineSearch(const LineSearchOptions &options = LineSearchOptions());

        LineSearchResult search(Merit              &merit,
                                const real_t *const x,
                                const real_t *const h,
                                const real_t        m0,
                                const real_t        slope);

        const LineSearchOptions &options() const { return options_; }

    private:
        LineSearchOptions   options_;
        std::vector<real_t> values_;
    };

}  // namespace sfem

#endif  // SFEM_LINE_SEARCH_HPP
