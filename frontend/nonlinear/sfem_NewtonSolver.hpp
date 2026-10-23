#ifndef SFEM_NEWTON_SOLVER_HPP
#define SFEM_NEWTON_SOLVER_HPP

#include <functional>
#include <memory>
#include <string>

#include "sfem_LineSearch.hpp"
#include "sfem_LinearizedStep.hpp"
#include "sfem_NonlinearSolver.hpp"

namespace sfem {

    class Function;

    struct NewtonOptions {
        int max_it{30};
        /// Converged when the step's measure is at most max(atol, rtol * its first value).
        real_t atol{real_t(1e-9)};
        real_t rtol{real_t(1e-8)};
        /// Diverged when the measure exceeds this multiple of its first value.
        real_t divergence_factor{real_t(1e4)};
        /// A step whose prescribed values moved carries the increment in fractions, halving one
        /// whose solve fails at most this many times.
        int increment_halvings{6};
        /// "auto", "energy" or "residual"; a step with a merit of its own overrides it.
        std::string       merit{"auto"};
        LineSearchOptions line_search;
    };

    /// One measured Newton iterate as a monitor sees it. The linear solve and the line search are
    /// those of the step that led to it (none at iteration 0).
    struct NewtonIteration {
        int                      iteration{0};
        StepMeasure              measure;
        real_t                   tolerance{0};
        int                      linear_iterations{0};
        bool                     linear_capped{false};
        LineSearchResult         line_search;
        const real_t            *x{nullptr};
        const real_t            *residual{nullptr};
    };

    /// Newton's method on `f`, with the linearised problem and the convergence test delegated to
    /// a `LinearizedStep` and every step judged by a merit through one line search. It stops as
    /// soon as it knows it is failing: a non-finite measure, growth past `divergence_factor`, or a
    /// direction along which the merit cannot decrease.
    class NewtonSolver final : public NonlinearSolver {
    public:
        NewtonSolver(const std::shared_ptr<Function>       &f,
                     const std::shared_ptr<LinearizedStep> &step,
                     const NewtonOptions                   &options = NewtonOptions());
        ~NewtonSolver() override;

        NonlinearReport solve(const StepContext &context, real_t *const x) override;
        int             commit(const real_t *const x) override;

        /// Called at every iterate, including the last one measured.
        void add_monitor(const std::function<void(const NewtonIteration &)> &monitor);

        const NewtonOptions &options() const;
        NewtonOptions       &options();

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

}  // namespace sfem

#endif  // SFEM_NEWTON_SOLVER_HPP
