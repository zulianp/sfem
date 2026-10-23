#ifndef SFEM_NONLINEAR_SOLVER_HPP
#define SFEM_NONLINEAR_SOLVER_HPP

#include <memory>

#include "sfem_TimeScheme.hpp"
#include "sfem_base.hpp"

namespace sfem {

    /// The step a nonlinear solve belongs to: it ends at t + dt. `scheme` is the time scheme the
    /// material holds, or null without inertia.
    struct StepContext {
        real_t                      t{0};
        real_t                      dt{0};
        std::shared_ptr<TimeScheme> scheme;
    };

    struct NonlinearReport {
        enum Status {
            CONVERGED,
            /// The merit could no longer resolve a decrease (LineSearchResult::AT_RESOLUTION).
            CONVERGED_AT_RESOLUTION,
            /// The iteration cap, not a tolerance, stopped it.
            MAX_IT,
            /// The measure grew past divergence_factor times its first value.
            DIVERGED,
            NON_FINITE,
            LINE_SEARCH_FAILED,
            /// The linearisation or the step's inner solve failed outright.
            STEP_FAILED
        };

        Status status{STEP_FAILED};
        int    iterations{0};
        int    linear_iterations{0};
        /// Linearised solves that stopped on their cap rather than their tolerance.
        int    capped_linear_solves{0};
        real_t reference{0};
        real_t final_measure{0};
        real_t min_alpha{1};

        bool converged() const { return status == CONVERGED || status == CONVERGED_AT_RESOLUTION; }
    };

    const char *to_string(const NonlinearReport::Status status);

    /// Solves one step's nonlinear problem in place: x arrives with its constrained values and
    /// the starting guess, and leaves with the solution.
    class NonlinearSolver {
    public:
        virtual ~NonlinearSolver()                                                     = default;
        virtual NonlinearReport solve(const StepContext &context, real_t *const x) = 0;

        /// The step's solution x is accepted.
        virtual int commit(const real_t *const x) = 0;
    };

}  // namespace sfem

#endif  // SFEM_NONLINEAR_SOLVER_HPP
