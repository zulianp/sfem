#ifndef SFEM_STEPPER_HPP
#define SFEM_STEPPER_HPP

#include <functional>
#include <limits>
#include <memory>

#include "sfem_HistoryDependent.hpp"
#include "sfem_NonlinearSolver.hpp"
#include "sfem_TimeScheme.hpp"
#include "sfem_aliases.hpp"
#include "sfem_base.hpp"

namespace sfem {

    class Function;
    class Constraint;

    struct StepperOptions {
        real_t t0{0};
        real_t dt{1};
        /// Steps of length dt from t0; step k ends at t0 + k dt.
        int steps{1};
        /// How often a failed step is retried with half its length.
        int max_cutbacks{0};
        /// After a shortened step each step is at most this many times the last one's length, up
        /// to dt: a multistep scheme stays stable only for a bounded ratio (BDF2, 1 + sqrt(2)).
        real_t max_growth{std::numeric_limits<real_t>::infinity()};
    };

    /// A step as an observer sees it.
    struct StepRecord {
        int                         step{0};
        real_t                      t{0};
        real_t                      dt{0};
        const real_t               *x{nullptr};
        NonlinearReport             report;
        int                         attempts{0};
        /// The step that ends the run.
        bool                        last{false};
        std::shared_ptr<TimeScheme> scheme;
    };

    class StepObserver {
    public:
        virtual ~StepObserver() = default;
        /// The initial state, before the first step.
        virtual int initial(const StepRecord &) { return SFEM_SUCCESS; }
        /// A step's solution, before it is committed: the residual and the scheme are still
        /// the step's.
        virtual int converged(const StepRecord &) { return SFEM_SUCCESS; }
        /// After the step is committed and the scheme has carried its state out of it.
        virtual int committed(const StepRecord &) { return SFEM_SUCCESS; }
        /// A step that could not be solved, after its last attempt.
        virtual int failed(const StepRecord &) { return SFEM_SUCCESS; }
    };

    /// The one place the step sequence of a time-dependent elasticity problem is written:
    ///
    ///   1. the time-dependent conditions move to the step's end, scheduled releases happen;
    ///   2. the time scheme and the history-dependent materials open the step;
    ///   3. the nonlinear solve starts from the scheme's start, or from the last state;
    ///      the solver carries its constrained values onto the prescribed ones;
    ///   4. the solution is committed: the solver, the histories, the scheme;
    ///
    /// with the observers around it. A failed step is solved again from the last state, then with
    /// half its length up to `max_cutbacks` times.
    class Stepper {
    public:
        Stepper(const std::shared_ptr<Function>        &f,
                const std::shared_ptr<NonlinearSolver> &solver,
                const StepperOptions                   &options);
        ~Stepper();

        /// The scheme the material holds (TimeSteppable::set_time_scheme); none without inertia.
        void set_time_scheme(const std::shared_ptr<TimeScheme> &scheme);
        void add_history(const std::shared_ptr<HistoryDependent> &history);
        /// Called with the end time of every step attempt: Dirichlet and Neumann `set_time`.
        void add_time_dependent(const std::function<int(const real_t)> &set_time);
        /// The constraint is removed from the function at the first step ending at or after t.
        void add_release(const std::shared_ptr<Constraint> &constraint, const real_t t);
        void add_observer(const std::shared_ptr<StepObserver> &observer);

        /// Runs every step from the state in x, which it leaves at the last solution.
        /// SFEM_FAILURE when a step could not be solved.
        int run(real_t *const x);

        const StepperOptions &options() const;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

}  // namespace sfem

#endif  // SFEM_STEPPER_HPP
