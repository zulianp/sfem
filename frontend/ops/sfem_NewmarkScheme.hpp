#pragma once

#include <memory>

#include "sfem_NewmarkInertiaPotential.hpp"
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
    /// `a = (u - u_hat)/(beta*dt^2)`, and `NewmarkInertiaPotential` contributes
    /// `M*a` to the residual.  This class composes one rather than deriving
    /// from it -- the arithmetic there is tested and unchanged -- keeps its
    /// `alpha` and its `u_hat` current, and publishes it through `inertia_op`
    /// for the caller to add to the `Function`.  It has to go in the
    /// `Function`: the node-wise merit is a norm of what `Function::gradient`
    /// assembles, so a term that is not an operator is a term the line search
    /// cannot see.
    class NewmarkScheme final : public TimeScheme {
    public:
        explicit NewmarkScheme(const std::shared_ptr<FunctionSpace> &space);
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
