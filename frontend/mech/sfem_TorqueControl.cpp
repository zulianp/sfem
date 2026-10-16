#include "sfem_TorqueControl.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_Rotation.hpp"

namespace sfem {

    int material_torque(Op                         &material,
                        const Rotation             &rotation,
                        const real_t *const         x,
                        const SharedBuffer<real_t> &scratch,
                        real_t                     &torque) {
        sfem::blas<real_t>(EXECUTION_SPACE_HOST)->zeros(scratch->size(), scratch->data());
        if (material.gradient(x, scratch->data()) != SFEM_SUCCESS) return SFEM_FAILURE;
        torque = rotation.torque(x, scratch->data());
        return SFEM_SUCCESS;
    }

    class TorqueControl::Impl {
    public:
        std::shared_ptr<Function>        f;
        std::shared_ptr<NonlinearSolver> inner;
        std::shared_ptr<Rotation>        rotation;
        std::shared_ptr<Op>              material;
        TorqueControlOptions             options;
        SharedBuffer<real_t>             scratch;

        // A small probe twist to start from: the response is very nearly linear near zero, so one
        // solve there and the exact pair torque(0) = 0 put the secant on the answer at once.
        real_t theta{real_t(1e-2)};
        // dT/dangle from the last secant, carried into the next step. A point from the previous
        // step cannot be: its torque was measured before the material relaxed further, and a
        // secant through it is noise that the clamp below turns into a doubled twist.
        real_t slope{0};
    };

    TorqueControl::TorqueControl(const std::shared_ptr<Function>        &f,
                                 const std::shared_ptr<NonlinearSolver> &inner,
                                 const std::shared_ptr<Rotation>        &rotation,
                                 const std::shared_ptr<Op>              &material,
                                 const TorqueControlOptions             &options)
        : impl_(std::make_unique<Impl>()) {
        impl_->f        = f;
        impl_->inner    = inner;
        impl_->rotation = rotation;
        impl_->material = material;
        impl_->options  = options;
        impl_->scratch  = create_host_buffer<real_t>(f->space()->n_dofs());
    }

    TorqueControl::~TorqueControl() = default;

    NonlinearReport TorqueControl::solve(const StepContext &context, real_t *const x) {
        SFEM_TRACE_SCOPE("TorqueControl::solve");

        auto &impl = *impl_;
        if (!impl.f->has_constraint(impl.rotation)) return impl.inner->solve(context, x);

        const real_t target = impl.options.torque * impl.options.profile.value(context.t + context.dt);

        NonlinearReport total;
        real_t          torque = 0, previous_theta = 0, previous_torque = 0;
        for (int it = 0; it < impl.options.max_it; ++it) {
            impl.rotation->set_angle(impl.theta);
            impl.f->apply_constraints(x);
            const NonlinearReport report = impl.inner->solve(context, x);

            if (it == 0) total.reference = report.reference;
            total.status = report.status;
            total.iterations += report.iterations;
            total.linear_iterations += report.linear_iterations;
            total.capped_linear_solves += report.capped_linear_solves;
            total.final_measure = report.final_measure;
            total.min_alpha     = std::min(total.min_alpha, report.min_alpha);
            if (!report.converged()) return total;

            if (material_torque(*impl.material, *impl.rotation, x, impl.scratch, torque) != SFEM_SUCCESS) {
                total.status = NonlinearReport::STEP_FAILED;
                return total;
            }
            if (std::abs(torque - target) <= impl.options.tol * std::abs(target)) return total;

            // Secant on the angle through this step's previous solve; the first solve of a step
            // uses the carried slope, and the first of the run the secant through the origin,
            // where torque(0) = 0.
            if (it > 0 && impl.theta != previous_theta && std::abs(torque - previous_torque) > std::abs(torque) * real_t(1e-14)) {
                impl.slope = (torque - previous_torque) / (impl.theta - previous_theta);
            } else if (impl.slope == 0 && impl.theta != 0) {
                impl.slope = torque / impl.theta;
            }
            real_t next = impl.slope != 0 ? impl.theta + (target - torque) / impl.slope : impl.theta + real_t(1e-3);

            // A secant step on a stiffening response can overshoot badly, so the change is clamped
            // to a span: a ratio would collapse to nothing at zero and crawl away from it.
            const real_t span = std::max(std::abs(impl.theta), real_t(1e-2));
            next              = std::min(std::max(next, impl.theta - span), impl.theta + span);
            previous_theta    = impl.theta;
            previous_torque   = torque;
            impl.theta        = next;
        }

        fprintf(stderr,
                "TorqueControl: the torque %g missed the target %g by more than %g of it in %d solves\n",
                (double)torque,
                (double)target,
                (double)impl.options.tol,
                impl.options.max_it);
        total.status = NonlinearReport::MAX_IT;
        return total;
    }

    int TorqueControl::commit(const real_t *const x) { return impl_->inner->commit(x); }

    // ---------------------------------------------------------------------------------------------

    namespace {
        class TorqueProbe final : public Probe {
        public:
            TorqueProbe(const std::shared_ptr<Function> &f, const std::shared_ptr<Rotation> &rotation, const std::shared_ptr<Op> &material)
                : f_(f), rotation_(rotation), material_(material), scratch_(create_host_buffer<real_t>(f->space()->n_dofs())) {}

            std::vector<std::string> columns() const override { return {"angle", "released", "torque"}; }

            int converged(const StepRecord &record) override {
                released_ = !f_->has_constraint(rotation_);
                angle_    = released_ ? real_t(0) : rotation_->angle();
                if (material_torque(*material_, *rotation_, record.x, scratch_, torque_) != SFEM_SUCCESS) return SFEM_FAILURE;

                flips_ = measured_ && torque_ * previous_ < 0 ? flips_ + 1 : 0;
                previous_ = torque_;
                measured_ = true;
                if (flips_ == 8) {
                    fprintf(stderr,
                            "TorqueProbe: the torque has changed sign on %d consecutive steps up to t = %g. That is the\n"
                            "  2 dt saw-tooth of a non-dissipative time integrator, not a response of the material:\n"
                            "  Newmark at gamma = 0.5 rings forever on anything impulsive (the start of a ramp or a\n"
                            "  sine, a release). Use gamma > 0.5 with beta = (1 + gamma)^2 / 4.\n",
                            flips_,
                            (double)record.t);
                }
                if (flips_ >= 32) {
                    fprintf(stderr,
                            "TorqueProbe: abandoning at t = %g: the torque has alternated for %d consecutive steps\n",
                            (double)record.t,
                            flips_);
                    return SFEM_FAILURE;
                }
                return SFEM_SUCCESS;
            }

            int sample(const StepRecord &, real_t *const values) override {
                values[0] = measured_ ? angle_ : rotation_->angle();
                values[1] = released_ ? 1 : 0;
                values[2] = measured_ ? torque_ : std::numeric_limits<real_t>::quiet_NaN();
                return SFEM_SUCCESS;
            }

        private:
            std::shared_ptr<Function> f_;
            std::shared_ptr<Rotation> rotation_;
            std::shared_ptr<Op>       material_;
            SharedBuffer<real_t>      scratch_;
            bool                      measured_{false};
            bool                      released_{false};
            real_t                    angle_{0};
            real_t                    torque_{0};
            real_t                    previous_{0};
            int                       flips_{0};
        };
    }  // namespace

    std::shared_ptr<Probe> create_torque_probe(const std::shared_ptr<Function> &f,
                                               const std::shared_ptr<Rotation> &rotation,
                                               const std::shared_ptr<Op>       &material) {
        return std::make_shared<TorqueProbe>(f, rotation, material);
    }

}  // namespace sfem
