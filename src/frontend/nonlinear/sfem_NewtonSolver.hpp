#ifndef SFEM_NEWTON_SOLVER_HPP
#define SFEM_NEWTON_SOLVER_HPP

#include <functional>
#include <memory>
#include <string>

#include "sfem_LineSearch.hpp"
#include "sfem_LinearizedStep.hpp"
#include "sfem_NonlinearSolver.hpp"

namespace sfem {
    class MemoryLedger;

    class Function;

    /// Which tangent the Newton directions are computed on (LinearizedStep::set_tangent_projection).
    enum class TangentProjection {
        /// The tangent itself.
        NONE,
        /// The tangent itself, until it shows indefinite (CG meets negative curvature, or the step
        /// reports a linear solve that could not converge on it) or gives a direction along which
        /// the merit does not decrease (or barely: project_below_alpha): that iteration is
        /// repeated from the same iterate on the tangent projected to positive semidefinite, which is kept until a
        /// step is taken whole ("project on demand", Longva et al., "Pitfalls of Projection",
        /// 2023). Newton on the projected tangent converges only linearly, so the tangent itself
        /// is taken back once, as soon as the line search trusts a full step; a tangent that is
        /// positive definite where the iterates converge then keeps Newton's rate. Failing again,
        /// it is projected for the rest of the step.
        ON_FAILURE,
        /// The projected tangent at every iteration (projected Newton).
        ALWAYS
    };

    struct NewtonOptions {
        int max_it{30};
        /// max_it iterations are the scheme, not a cap: the step ends after them and its state is
        /// accepted unconverged (ITERATIONS_DONE), as a fixed number of damped iterations does.
        bool fixed_iterations{false};
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
        TangentProjection tangent_projection{TangentProjection::NONE};
        /// ON_FAILURE: a step the line search shortened below this fraction also counts as a
        /// failure of the tangent itself, the iteration repeated on the projected one. On an
        /// indefinite tangent the direction can stay a descent direction for the merit only
        /// barely, and Newton then creeps on with tiny steps instead of failing. 0: only a
        /// direction without decrease (or negative curvature) projects. On the last of
        /// fixed_iterations the short step is taken: no iteration is left to repeat.
        real_t project_below_alpha{0};
    };

    /// One measured Newton iterate as a monitor sees it. The linear solve and the line search are
    /// those of the step that led to it (none at iteration 0).
    struct NewtonIteration {
        int                      iteration{0};
        StepMeasure              measure;
        real_t                   tolerance{0};
        int                      linear_iterations{0};
        bool                     linear_capped{false};
        /// The direction was computed on the projected tangent.
        bool                     projected{false};
        /// The step was not taken (project_below_alpha): the iterate is the one before it, and the
        /// iteration is repeated on the projected tangent. line_search.alpha is 0.
        bool                     discarded{false};
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

        /// Its step's (LinearizedStep::account) and its own vectors (NEWTON).
        void account(MemoryLedger &ledger) const;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

}  // namespace sfem

#endif  // SFEM_NEWTON_SOLVER_HPP
