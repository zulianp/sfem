#include "sfem_NewtonSolver.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_Function.hpp"

namespace sfem {

    const char *to_string(const NonlinearReport::Status status) {
        switch (status) {
            case NonlinearReport::CONVERGED:
                return "converged";
            case NonlinearReport::CONVERGED_AT_RESOLUTION:
                return "converged_at_resolution";
            case NonlinearReport::MAX_IT:
                return "max_it";
            case NonlinearReport::DIVERGED:
                return "diverged";
            case NonlinearReport::NON_FINITE:
                return "non_finite";
            case NonlinearReport::LINE_SEARCH_FAILED:
                return "line_search_failed";
            case NonlinearReport::STEP_FAILED:
                return "step_failed";
        }
        return "unknown";
    }

    class NewtonSolver::Impl {
    public:
        std::shared_ptr<Function>                                 f;
        std::shared_ptr<LinearizedStep>                           step;
        NewtonOptions                                             options;
        std::shared_ptr<BLAS<real_t>>                             blas;
        ptrdiff_t                                                 ndofs{0};
        SharedBuffer<real_t>                                      r;
        SharedBuffer<real_t>                                      dx;
        std::shared_ptr<Merit>                                    default_merit;
        std::vector<std::function<void(const NewtonIteration &)>> monitors;

        void notify(const NewtonIteration &iteration) const {
            for (auto &m : monitors) m(iteration);
        }

        std::shared_ptr<Merit> merit() {
            if (auto own = step->merit()) return own;
            if (!default_merit) default_merit = create_merit(f, options.merit);
            return default_merit;
        }
    };

    NewtonSolver::NewtonSolver(const std::shared_ptr<Function>       &f,
                               const std::shared_ptr<LinearizedStep> &step,
                               const NewtonOptions                   &options)
        : impl_(std::make_unique<Impl>()) {
        auto &impl   = *impl_;
        impl.f       = f;
        impl.step    = step;
        impl.options = options;

        const ExecutionSpace es = f->execution_space();
        impl.blas               = blas<real_t>(es);
        impl.ndofs              = f->space()->n_dofs();
        impl.r                  = create_buffer<real_t>(impl.ndofs, es);
        impl.dx                 = create_buffer<real_t>(impl.ndofs, es);
    }

    NewtonSolver::~NewtonSolver() = default;

    void NewtonSolver::add_monitor(const std::function<void(const NewtonIteration &)> &monitor) {
        impl_->monitors.push_back(monitor);
    }

    int NewtonSolver::commit(const real_t *const x) { return impl_->step->end_step(x); }

    const NewtonOptions &NewtonSolver::options() const { return impl_->options; }
    NewtonOptions       &NewtonSolver::options() { return impl_->options; }

    NonlinearReport NewtonSolver::solve(const StepContext &context, real_t *const x) {
        SFEM_TRACE_SCOPE("NewtonSolver::solve");

        auto           &impl  = *impl_;
        auto           &opts  = impl.options;
        const ptrdiff_t ndofs = impl.ndofs;
        real_t *const   r     = impl.r->data();
        real_t *const   dx    = impl.dx->data();
        LineSearch      line_search(opts.line_search);

        NonlinearReport report;
        if (impl.step->begin_step(context) != SFEM_SUCCESS) {
            report.status = NonlinearReport::STEP_FAILED;
            return report;
        }

        std::shared_ptr<Merit> merit;
        if (opts.line_search.enabled) {
            merit = impl.merit();
            if (!merit) {
                report.status = NonlinearReport::STEP_FAILED;
                return report;
            }
        }

        // A linear function is solved by one full step whose linear solve met its tolerance: its
        // accuracy is the linear solver's, and a second step would only repeat the solve.
        const bool linear = impl.f->is_linear();
        bool       solved = false;

        NewtonIteration log;
        real_t          tol = 0;
        for (int it = 0;; ++it) {
            log.iteration = it;
            log.x         = x;
            log.residual  = r;

            impl.f->update(x);
            impl.blas->zeros(ndofs, r);
            if (impl.f->gradient(x, r) != SFEM_SUCCESS) {
                report.status = NonlinearReport::STEP_FAILED;
                return report;
            }
            impl.f->set_value_to_constrained_dofs(0, r);

            log.measure          = impl.step->measure(x, r);
            report.final_measure = log.measure.value;
            if (it == 0) {
                report.reference = log.measure.reference;
                tol              = std::max(opts.atol, opts.rtol * report.reference);
            }
            log.tolerance = tol;
            impl.notify(log);

            if (!std::isfinite(log.measure.value)) {
                report.status = NonlinearReport::NON_FINITE;
                return report;
            }

            if ((log.measure.value <= tol || solved) && log.measure.admissible) {
                report.status = NonlinearReport::CONVERGED;
                return report;
            }

            if (log.measure.value > opts.divergence_factor * std::max(report.reference, tol)) {
                report.status = NonlinearReport::DIVERGED;
                return report;
            }

            if (it == opts.max_it) {
                report.status = NonlinearReport::MAX_IT;
                return report;
            }

            {
                SFEM_TRACE_SCOPE("NewtonSolver::linearize");
                if (impl.step->linearize(x, r) != SFEM_SUCCESS) {
                    report.status = NonlinearReport::STEP_FAILED;
                    return report;
                }
            }

            {
                SFEM_TRACE_SCOPE("NewtonSolver::step");
                log.linear_capped = impl.step->solve(x, r, dx) != SFEM_SUCCESS;
            }
            log.linear_iterations = impl.step->iterations();
            report.linear_iterations += log.linear_iterations;
            report.capped_linear_solves += log.linear_capped;
            report.iterations = it + 1;

            if (!std::isfinite(impl.blas->norm2(ndofs, dx))) {
                report.status = NonlinearReport::STEP_FAILED;
                return report;
            }

            if (merit) {
                const real_t m0    = merit->at(x, r);
                const real_t slope = merit->slope(x, r, dx, m0);
                log.line_search    = line_search.search(*merit, x, dx, m0, slope);
            } else {
                log.line_search        = LineSearchResult();
                log.line_search.status = LineSearchResult::ACCEPTED;
                log.line_search.alpha  = 1;
            }

            switch (log.line_search.status) {
                case LineSearchResult::ACCEPTED:
                case LineSearchResult::BEST_DECREASE:
                    break;
                case LineSearchResult::AT_RESOLUTION:
                    report.status = NonlinearReport::CONVERGED_AT_RESOLUTION;
                    return report;
                case LineSearchResult::NO_DECREASE:
                case LineSearchResult::NON_FINITE:
                    report.status = NonlinearReport::LINE_SEARCH_FAILED;
                    return report;
            }

            report.min_alpha = std::min(report.min_alpha, log.line_search.alpha);
            impl.blas->axpy(ndofs, log.line_search.alpha, dx, x);
            solved = linear && !log.linear_capped && log.line_search.alpha == 1;
        }
    }

}  // namespace sfem
