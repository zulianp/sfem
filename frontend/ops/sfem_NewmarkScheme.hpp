#pragma once

#include <memory>

#include "sfem_InertiaPotential.hpp"
#include "sfem_TimeScheme.hpp"

namespace sfem {

    class FunctionSpace;

    /// Newmark-beta, as a `TimeScheme` a material can be handed.
    ///
    /// The method reconstructs the velocity of the new step as
    ///
    ///     v = gamma/(beta*dt) * u + [v_n + dt*(1-gamma)*a_n - gamma/(beta*dt) * u_hat]
    ///
    /// about the predictor `u_hat = u_n + dt*v_n + dt^2*(1/2-beta)*a_n`, which
    /// is exactly the `shift * u + history` shape `TimeScheme` publishes: the
    /// bracket is the history and the factor in front of `u` is the shift.  A
    /// material written with `gen.dt(u)` therefore runs under this scheme
    /// without being compiled for it, and would run under backward Euler or
    /// BDF2 the same way.
    ///
    /// The acceleration is the other half of the method and is separable, so it
    /// is not part of the material's form at all: it is
    /// `a = (u - u_hat)/(beta*dt^2)`, and `InertiaPotential` contributes
    /// `M*a` to the residual.  This class composes one rather than deriving
    /// from it -- the arithmetic there is tested and unchanged -- keeps its
    /// `alpha` and its `u_hat` current, and publishes it through `inertia_op`
    /// for the operator holding the scheme to contribute.  It has to reach the
    /// residual somehow: the node-wise merit is a norm of what
    /// `Function::gradient` assembles, so a term that no operator assembles is
    /// a term the line search cannot see.
    class NewmarkScheme final : public TimeScheme {
    public:
        /// `es` is where the caller's vectors live.  A driver that solves on
        /// the device hands `advance` a device vector, so the state this
        /// scheme carries has to live there too.
        explicit NewmarkScheme(const std::shared_ptr<FunctionSpace> &space,
                               ExecutionSpace es = EXECUTION_SPACE_HOST);
        ~NewmarkScheme() override;

        void set_beta(real_t beta);
        void set_gamma(real_t gamma);
        void set_density(real_t density);

        /// Build the mass and the state buffers.  `block_names` is forwarded to
        /// the lumped mass, which is the only thing here that integrates.
        int initialize(const std::vector<std::string> &block_names = {});

        real_t        shift() const override;
        const real_t *history() const override;
        real_t        weight(const char *name) const override;
        void          begin_step(real_t t, real_t dt) override;
        void          advance(const real_t *const x) override;

        std::shared_ptr<Op> inertia_op() const override;

        /// The velocity and acceleration the method implies at an arbitrary
        /// state, without advancing the step.
        ///
        /// Newmark's `v = shift * x + z` and `a = (x - u_hat)/(beta*dt^2)` are
        /// functions of the state, so an operator that takes them as fields --
        /// the hand-written `KelvinVoigtNewmark` does -- needs them refreshed
        /// at every Newton iterate, not once per step.  That is the same
        /// arithmetic `advance` ends with, which is why `advance` is this
        /// followed by a rotation rather than a second copy of it.
        void reconstruct(const real_t *const x, real_t *const velocity, real_t *const acceleration) const;

        /// The state the method carries between steps.  Writable, because the
        /// initial condition is the caller's to set.
        std::shared_ptr<Buffer<real_t>> state() const;
        std::shared_ptr<Buffer<real_t>> velocity() const;
        std::shared_ptr<Buffer<real_t>> acceleration() const;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

}  // namespace sfem
