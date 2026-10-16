#ifndef SFEM_TORQUE_CONTROL_HPP
#define SFEM_TORQUE_CONTROL_HPP

#include <memory>

#include "sfem_LoadProfile.hpp"
#include "sfem_NonlinearSolver.hpp"
#include "sfem_Observers.hpp"
#include "sfem_aliases.hpp"

namespace sfem {

    class Function;
    class Op;
    class Rotation;

    /// The moment about the rotation's axis of the material's force at x, the scheme's inertia
    /// included when the material holds one: the torque the rotation applies at equilibrium.
    /// `scratch` holds n_dofs values on the host.
    int material_torque(Op                         &material,
                        const Rotation             &rotation,
                        const real_t *const         x,
                        const SharedBuffer<real_t> &scratch,
                        real_t                     &torque);

    struct TorqueControlOptions {
        /// The target moment, times `profile`(t) at the end of each step.
        real_t      torque{0};
        LoadProfile profile;
        /// Relative to the target.
        real_t      tol{real_t(1e-6)};
        int         max_it{20};
    };

    /// A rotation driven by the torque it applies rather than by its angle: each step solves
    /// torque(angle) = target for the angle by a secant iteration around the inner solve, warm
    /// started from the previous step. Once the rotation is released, the inner solve alone.
    class TorqueControl final : public NonlinearSolver {
    public:
        TorqueControl(const std::shared_ptr<Function>        &f,
                      const std::shared_ptr<NonlinearSolver> &inner,
                      const std::shared_ptr<Rotation>        &rotation,
                      const std::shared_ptr<Op>              &material,
                      const TorqueControlOptions             &options);
        ~TorqueControl() override;

        /// The iterations of every inner solve, summed. MAX_IT when the torque misses its target
        /// after `max_it` inner solves.
        NonlinearReport solve(const StepContext &context, real_t *const x) override;
        int             commit(const real_t *const x) override;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    /// Columns angle, released and torque of a rotation: the angle prescribed (0 once
    /// released) and material_torque, read before the step is committed. The torque is NaN in
    /// the initial row, which no step has measured.
    ///
    /// A torque that changes sign on every step is the 2 dt saw-tooth of a non-dissipative time
    /// integrator ringing on an impulse, not a response of the material, and every step still
    /// converges: it is reported after 8 steps and fails the run after 32.
    std::shared_ptr<Probe> create_torque_probe(const std::shared_ptr<Function> &f,
                                               const std::shared_ptr<Rotation> &rotation,
                                               const std::shared_ptr<Op>       &material);

}  // namespace sfem

#endif  // SFEM_TORQUE_CONTROL_HPP
