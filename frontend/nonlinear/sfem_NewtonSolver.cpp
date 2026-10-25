#include "sfem_NewtonSolver.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "sfem_Function.hpp"
#include "sfem_OwnedReductions.hpp"
#include "sfem_blas_factory.hpp"
#include "sfem_operator_factory.hpp"

namespace sfem {

    const char *to_string(const NonlinearReport::Status status) {
        switch (status) {
            case NonlinearReport::CONVERGED:
                return "converged";
            case NonlinearReport::CONVERGED_AT_RESOLUTION:
                return "converged_at_resolution";
            case NonlinearReport::MAX_IT:
                return "max_it";
            case NonlinearReport::ITERATIONS_DONE:
                return "iterations_done";
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
        std::unique_ptr<OwnedReductions>                          reductions;
        SharedBuffer<real_t>                                      r;
        SharedBuffer<real_t>                                      dx;
        SharedBuffer<real_t>                                      moved;
        SharedBuffer<real_t>                                      increment;
        std::shared_ptr<Merit>                                    default_merit;
        std::vector<std::function<void(const NewtonIteration &)>> monitors;

        void notify(const NewtonIteration &iteration) const {
            for (auto &m : monitors) m(iteration);
        }

        /// The residual at x, the constrained rows zero, and its measure. With `known` the
        /// residual is already in r.
        int residual(const real_t *const x, StepMeasure &measure, const bool known = false) {
            f->update(x);
            if (!known) {
                blas->zeros(ndofs, r->data());
                if (!reductions->all(f->gradient(x, r->data()) == SFEM_SUCCESS)) return SFEM_FAILURE;
                f->set_value_to_constrained_dofs(0, r->data());
            }
            measure = step->measure(x, r->data());
            return SFEM_SUCCESS;
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
        impl.reductions         = std::make_unique<OwnedReductions>(f->space(), es);
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
        // Every failure below is decided by all ranks together (OwnedReductions::all).
        if (!impl.reductions->all(impl.step->begin_step(context) == SFEM_SUCCESS)) {
            report.status = NonlinearReport::STEP_FAILED;
            return report;
        }
        // Every evaluation reads x over all local elements: its ghost and aura copies from their
        // owners, once; the steps then keep them.
        if (impl.f->space()->synchronize(x, impl.f->execution_space()) != SFEM_SUCCESS) {
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
        const bool linear = impl.f->is_linear() && impl.step->solves_linear_function_exactly();

        NewtonIteration log;
        real_t          tol = 0;
        int             count = 0;
        log.x               = x;
        log.residual        = r;

        // Newton from x, its constrained values held where they are (the corrections are zero
        // there). `solved`: the step that led to x solved a linear problem.
        auto iterate = [&](bool solved) -> NonlinearReport::Status {
            bool known = false;
            for (int it = 0;; ++it) {
                log.iteration = count;

                if (impl.residual(x, log.measure, known) != SFEM_SUCCESS) return NonlinearReport::STEP_FAILED;
                report.final_measure = log.measure.value;
                if (count == 0) {
                    report.reference = log.measure.reference;
                    tol              = std::max(opts.atol, opts.rtol * report.reference);
                }
                log.tolerance = tol;
                impl.notify(log);

                if (!std::isfinite(log.measure.value)) return NonlinearReport::NON_FINITE;
                if ((log.measure.value <= tol || solved) && log.measure.admissible) return NonlinearReport::CONVERGED;
                if (log.measure.value > opts.divergence_factor * std::max(report.reference, tol)) return NonlinearReport::DIVERGED;
                if (it == opts.max_it) return opts.fixed_iterations ? NonlinearReport::ITERATIONS_DONE : NonlinearReport::MAX_IT;

                {
                    SFEM_TRACE_SCOPE("NewtonSolver::linearize");
                    if (!impl.reductions->all(impl.step->linearize(x, r) == SFEM_SUCCESS)) return NonlinearReport::STEP_FAILED;
                }

                {
                    SFEM_TRACE_SCOPE("NewtonSolver::step");
                    impl.blas->zeros(ndofs, dx);
                    log.linear_capped = !impl.reductions->all(impl.step->solve(x, r, dx) == SFEM_SUCCESS);
                }
                // The solve sets the owned rows; x + s dx keeps every copy consistent from here.
                if (impl.f->space()->synchronize(dx, impl.f->execution_space()) != SFEM_SUCCESS) return NonlinearReport::STEP_FAILED;
                log.linear_iterations = impl.step->iterations();
                report.linear_iterations += log.linear_iterations;
                report.capped_linear_solves += log.linear_capped;
                report.iterations = ++count;

                if (!std::isfinite(impl.reductions->norm2(dx))) return NonlinearReport::STEP_FAILED;

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
                        return NonlinearReport::CONVERGED_AT_RESOLUTION;
                    case LineSearchResult::NO_DECREASE:
                    case LineSearchResult::NON_FINITE:
                        return NonlinearReport::LINE_SEARCH_FAILED;
                }

                report.min_alpha = std::min(report.min_alpha, log.line_search.alpha);
                impl.blas->axpy(ndofs, log.line_search.alpha, dx, x);
                solved = linear && !log.linear_capped && log.line_search.alpha == 1;
                known  = merit && merit->residual_at(log.line_search.alpha, r);
            }
        };

        // x arrives with its constrained values where the last state left them. Setting them to
        // the prescribed ones alone shears the layer of elements at the boundary, by more than an
        // element when the increment is large against the mesh, so the increment is carried by
        // Newton itself: the tangent at x, a fraction of the increment in the constrained entries
        // of dx and the interior following it, taken whole, then Newton with the boundary held
        // there. A fraction whose solve fails is halved and tried again from the last converged
        // state, at most `increment_halvings` times. The tolerance keeps its reference at the
        // state with the boundary set.
        if (!impl.increment) impl.increment = create_buffer<real_t>(ndofs, impl.f->execution_space());
        real_t *const increment = impl.increment->data();
        impl.blas->copy(ndofs, x, increment);
        impl.f->apply_constraints(increment);
        impl.blas->axpy(ndofs, -1, x, increment);
        if (!(impl.reductions->norm2(increment) > 0)) {
            report.status = iterate(false);
            return report;
        }

        if (!impl.moved) impl.moved = create_buffer<real_t>(ndofs, impl.f->execution_space());
        real_t *const start = impl.moved->data();
        impl.blas->copy(ndofs, x, start);
        impl.f->apply_constraints(start);
        if (impl.residual(start, log.measure) != SFEM_SUCCESS) {
            report.status = NonlinearReport::STEP_FAILED;
            return report;
        }
        report.reference = log.measure.reference;
        StepMeasure at_x;
        if (impl.residual(x, at_x) != SFEM_SUCCESS) {
            report.status = NonlinearReport::STEP_FAILED;
            return report;
        }
        if (!std::isfinite(at_x.value)) {
            report.status = NonlinearReport::NON_FINITE;
            return report;
        }
        if (!std::isfinite(report.reference)) report.reference = at_x.reference;
        tol           = std::max(opts.atol, opts.rtol * report.reference);
        log.x         = start;
        log.tolerance = tol;
        impl.notify(log);
        log.x = x;
        count = 1;

        // From here `start` is the last converged state; the fractions are powers of two, so
        // their sum reaches 1 exactly.
        impl.blas->copy(ndofs, x, start);
        real_t done = 0, fraction = 1;
        for (int halvings = 0;;) {
            const real_t s = std::min(fraction, 1 - done);

            if (impl.residual(x, at_x) != SFEM_SUCCESS) {
                report.status = NonlinearReport::STEP_FAILED;
                return report;
            }
            {
                SFEM_TRACE_SCOPE("NewtonSolver::linearize");
                if (!impl.reductions->all(impl.step->linearize(x, r) == SFEM_SUCCESS)) {
                    report.status = NonlinearReport::STEP_FAILED;
                    return report;
                }
            }
            {
                SFEM_TRACE_SCOPE("NewtonSolver::step");
                impl.blas->zaxpby(ndofs, s, increment, 0, increment, dx);
                log.linear_capped = !impl.reductions->all(impl.step->solve(x, r, dx) == SFEM_SUCCESS);
            }
            if (impl.f->space()->synchronize(dx, impl.f->execution_space()) != SFEM_SUCCESS) {
                report.status = NonlinearReport::STEP_FAILED;
                return report;
            }
            log.linear_iterations = impl.step->iterations();
            report.linear_iterations += log.linear_iterations;
            report.capped_linear_solves += log.linear_capped;
            report.iterations = count;
            if (!std::isfinite(impl.reductions->norm2(dx))) {
                report.status = NonlinearReport::STEP_FAILED;
                return report;
            }

            impl.blas->axpy(ndofs, 1, dx, x);
            if (done + s == 1) impl.f->apply_constraints(x);
            log.line_search        = LineSearchResult();
            log.line_search.status = LineSearchResult::ACCEPTED;
            log.line_search.alpha  = 1;

            const auto status = iterate(linear && !log.linear_capped);
            if (status == NonlinearReport::CONVERGED || status == NonlinearReport::CONVERGED_AT_RESOLUTION ||
                status == NonlinearReport::ITERATIONS_DONE) {
                done += s;
                if (done == 1) {
                    report.status = status;
                    return report;
                }
                impl.blas->copy(ndofs, x, start);
                fraction = std::min(real_t(1), 2 * s);
            } else {
                if (++halvings > opts.increment_halvings) {
                    report.status = status;
                    return report;
                }
                impl.blas->copy(ndofs, start, x);
                fraction = s / 2;
            }
            count++;
        }
    }

}  // namespace sfem
