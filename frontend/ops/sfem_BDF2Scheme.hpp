#pragma once

#include <memory>
#include <string>
#include <vector>

#include "sfem_InertiaPotential.hpp"
#include "sfem_TimeScheme.hpp"

namespace sfem {

    class FunctionSpace;

    /// BDF2, as a `TimeScheme` a material can be handed.
    ///
    /// The method differentiates by the two-step backward formula
    ///
    ///     v = (3u - 4u_n + u_nm1) / (2dt)
    ///
    /// which is `shift * u + z` with `shift = 3/(2dt)` and the history
    /// `z = (-4u_n + u_nm1)/(2dt)`, so a material written with `gen.dt(u)`
    /// runs under this scheme without being compiled for it, exactly as it
    /// runs under Newmark or backward Euler.
    ///
    /// Applying the same formula to the velocity gives the acceleration, and
    /// that half is separable: substituting `v` into it leaves
    /// `a = alpha * (u - u_hat)` with `alpha = 9/(4dt^2)` and the predictor
    ///
    ///     u_hat = (4u_n - u_nm1)/3 + 2dt*(4v_n - v_nm1)/9
    ///
    /// so `InertiaPotential` contributes `M*a` and the material's form carries
    /// only the first derivative.  This is the same arrangement `NewmarkScheme`
    /// uses, with different coefficients -- which is the point of both being
    /// schemes rather than driver algebra.
    ///
    /// BDF2 is not self-starting: the first step has no `u_nm1`, so it is taken
    /// with backward Euler (`shift = 1/dt`, `alpha = 1/dt^2`,
    /// `u_hat = u_n + dt*v_n`).  That startup is part of the method, so it
    /// lives here rather than in the caller's step loop, which is what lets a
    /// driver's loop stop distinguishing the first step from the rest.
    class BDF2Scheme final : public TimeScheme {
    public:
        /// `es` is where the caller's vectors live; see `NewmarkScheme`.
        explicit BDF2Scheme(const std::shared_ptr<FunctionSpace> &space,
                            ExecutionSpace                        es = EXECUTION_SPACE_HOST);
        ~BDF2Scheme() override;

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
        /// state, without advancing the step.  See `NewmarkScheme::reconstruct`
        /// for why this is separate from `advance`.
        void reconstruct(const real_t *const x,
                         real_t *const       velocity,
                         real_t *const       acceleration) const override;

        /// Both halves of the method are reported, so the exported
        /// acceleration is the one `inertia_op` assembles.
        bool has_acceleration() const override { return true; }

        /// The state the method carries between steps.  Writable, because the
        /// initial condition is the caller's to set.
        std::shared_ptr<Buffer<real_t>> state() const;
        std::shared_ptr<Buffer<real_t>> velocity() const;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

}  // namespace sfem
