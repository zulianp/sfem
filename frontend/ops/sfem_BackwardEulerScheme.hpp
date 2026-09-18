#pragma once

#include <memory>

#include "sfem_TimeScheme.hpp"

namespace sfem {

    class FunctionSpace;

    /// Backward Euler, as a `TimeScheme` a material can be handed.
    ///
    /// The simplest thing that fits the shape `TimeScheme` publishes:
    /// `u_dot = (u - u_n)/dt` is `shift * u + z` with `shift = 1/dt` and
    /// `z = -u_n/dt`.  A material written with `gen.dt(u)` therefore runs under
    /// this without being compiled for it, exactly as it runs under Newmark --
    /// which is the property the whole arrangement exists to buy, and the
    /// reason this class is nine lines of arithmetic.
    ///
    /// It publishes no separable term.  Newmark has one because a second-order
    /// momentum balance carries an inertia the material's form does not, and a
    /// first-order problem has no such half: `inertia_op()` inherits the base's
    /// null, and the generated operator's forwarding already tolerates it.
    ///
    /// It also carries no derived state.  Newmark reconstructs a velocity and
    /// an acceleration at the end of a step; here the step's whole memory is
    /// the previous solution, so `advance` is a copy.
    class BackwardEulerScheme final : public TimeScheme {
    public:
        /// `es` is where the caller's vectors live; see `NewmarkScheme`.
        explicit BackwardEulerScheme(const std::shared_ptr<FunctionSpace> &space,
                                     ExecutionSpace es = EXECUTION_SPACE_HOST);
        ~BackwardEulerScheme() override;

        int initialize();

        real_t        shift() const override;
        const real_t *history() const override;
        real_t        weight(const char *name) const override;
        void          begin_step(real_t t, real_t dt) override;
        void          advance(const real_t *const x) override;

        /// The state carried between steps.  Writable, because the initial
        /// condition is the caller's to set.
        std::shared_ptr<Buffer<real_t>> state() const;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

}  // namespace sfem
