#include "sfem_Stepper.hpp"

#include <algorithm>
#include <vector>

#include "sfem_Function.hpp"
#include "sfem_blas_factory.hpp"

namespace sfem {

    class Stepper::Impl {
    public:
        struct Release {
            std::shared_ptr<Constraint> constraint;
            real_t                      t{0};
            bool                        released{false};
        };

        std::shared_ptr<Function>                       f;
        std::shared_ptr<NonlinearSolver>                solver;
        StepperOptions                                  options;
        std::shared_ptr<TimeScheme>                     scheme;
        std::vector<std::shared_ptr<HistoryDependent>>  histories;
        std::vector<std::function<int(const real_t)>>   time_dependent;
        std::vector<Release>                            releases;
        std::vector<std::shared_ptr<StepObserver>>      observers;

        template <typename Call>
        int notify(Call &&call) {
            for (auto &o : observers) {
                if (call(*o) != SFEM_SUCCESS) return SFEM_FAILURE;
            }
            return SFEM_SUCCESS;
        }

        /// Moves the conditions to `t`; releases a constraint once t reaches its time, and takes
        /// the release back if a shorter attempt ends before it.
        int set_time(const real_t t, const real_t tol) {
            for (auto &r : releases) {
                const bool due = t >= r.t - tol;
                if (due && !r.released) {
                    f->remove_constraint(r.constraint);
                    r.released = true;
                } else if (!due && r.released) {
                    f->add_constraint(r.constraint);
                    r.released = false;
                }
            }
            for (auto &set : time_dependent) {
                if (set(t) != SFEM_SUCCESS) return SFEM_FAILURE;
            }
            return SFEM_SUCCESS;
        }
    };

    Stepper::Stepper(const std::shared_ptr<Function>        &f,
                     const std::shared_ptr<NonlinearSolver> &solver,
                     const StepperOptions                   &options)
        : impl_(std::make_unique<Impl>()) {
        impl_->f       = f;
        impl_->solver  = solver;
        impl_->options = options;
    }

    Stepper::~Stepper() = default;

    void Stepper::set_time_scheme(const std::shared_ptr<TimeScheme> &scheme) { impl_->scheme = scheme; }
    void Stepper::add_history(const std::shared_ptr<HistoryDependent> &history) { impl_->histories.push_back(history); }
    void Stepper::add_time_dependent(const std::function<int(const real_t)> &set_time) {
        impl_->time_dependent.push_back(set_time);
    }
    void Stepper::add_release(const std::shared_ptr<Constraint> &constraint, const real_t t) {
        impl_->releases.push_back({constraint, t, false});
    }
    void Stepper::add_observer(const std::shared_ptr<StepObserver> &observer) { impl_->observers.push_back(observer); }

    const StepperOptions &Stepper::options() const { return impl_->options; }

    int Stepper::run(real_t *const x) {
        SFEM_TRACE_SCOPE("Stepper::run");

        auto           &impl  = *impl_;
        auto           &f     = impl.f;
        const auto      opts  = impl.options;
        const ptrdiff_t ndofs = f->space()->n_dofs();
        auto            blas  = sfem::blas<real_t>(f->execution_space());
        auto            x_n   = create_buffer<real_t>(ndofs, f->execution_space());

        const real_t t_end = opts.t0 + opts.steps * opts.dt;
        const real_t tol   = real_t(1e-10) * opts.dt;

        StepRecord record;
        record.t      = opts.t0;
        record.x      = x;
        record.scheme = impl.scheme;

        // x is the initial state as given: the solver of the first step carries it onto the
        // prescribed values (NewtonSolver's first iteration), rather than setting them here.
        if (impl.set_time(opts.t0, tol) != SFEM_SUCCESS) return SFEM_FAILURE;
        blas->copy(ndofs, x, x_n->data());
        if (impl.notify([&](StepObserver &o) { return o.initial(record); }) != SFEM_SUCCESS) return SFEM_FAILURE;

        real_t t    = opts.t0;
        int    step = 0;
        // While every step has its full length, step k ends at t0 + k dt exactly rather than at a
        // sum that has gathered rounding.
        bool   uniform = true;
        real_t last_dt = opts.dt;
        while (t < t_end - tol) {
            real_t dt   = uniform ? opts.dt : std::min({opts.dt, t_end - t, opts.max_growth * last_dt});
            int    cuts = 0;

            NonlinearReport report;
            int             attempts = 0;
            for (;;) {
                const real_t t_new = uniform && cuts == 0 ? opts.t0 + (step + 1) * opts.dt : t + dt;
                if (impl.set_time(t_new, tol) != SFEM_SUCCESS) return SFEM_FAILURE;
                if (impl.scheme) impl.scheme->begin_step(t, dt);
                for (auto &h : impl.histories) h->begin_step(t_new, dt);

                const StepContext context{t, dt, impl.scheme};

                // The scheme's start, and the last state if the start cannot be evaluated: right
                // after an impact the predicted start can invert elements the last state does not.
                const bool predicted = impl.scheme != nullptr;
                if (predicted) {
                    impl.scheme->start(x);
                } else {
                    blas->copy(ndofs, x_n->data(), x);
                }
                report = impl.solver->solve(context, x);
                attempts++;

                if (predicted && report.status == NonlinearReport::NON_FINITE && report.iterations == 0) {
                    blas->copy(ndofs, x_n->data(), x);
                    report = impl.solver->solve(context, x);
                    attempts++;
                }

                if (report.accepted()) break;

                record.step     = step + 1;
                record.t        = t_new;
                record.dt       = dt;
                record.report   = report;
                record.attempts = attempts;
                impl.notify([&](StepObserver &o) { return o.failed(record); });

                if (cuts == opts.max_cutbacks) {
                    blas->copy(ndofs, x_n->data(), x);
                    return SFEM_FAILURE;
                }
                cuts++;
                uniform = false;
                dt /= 2;
            }

            t = uniform ? opts.t0 + (step + 1) * opts.dt : t + dt;
            last_dt = dt;
            step++;

            record.step     = step;
            record.t        = t;
            record.dt       = dt;
            record.report   = report;
            record.attempts = attempts;
            record.last     = t >= t_end - tol;
            if (impl.notify([&](StepObserver &o) { return o.converged(record); }) != SFEM_SUCCESS) return SFEM_FAILURE;

            if (impl.solver->commit(x) != SFEM_SUCCESS) return SFEM_FAILURE;
            for (auto &h : impl.histories) h->advance(x);
            if (impl.scheme) impl.scheme->advance(x);
            blas->copy(ndofs, x, x_n->data());

            if (impl.notify([&](StepObserver &o) { return o.committed(record); }) != SFEM_SUCCESS) return SFEM_FAILURE;
        }

        return SFEM_SUCCESS;
    }

}  // namespace sfem
