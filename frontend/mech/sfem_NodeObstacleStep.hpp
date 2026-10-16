#ifndef SFEM_NODE_OBSTACLE_STEP_HPP
#define SFEM_NODE_OBSTACLE_STEP_HPP

#include <memory>

#include "sfem_LinearizedStep.hpp"
#include "sfem_Observers.hpp"

namespace sfem {

    class Function;
    class NewmarkScheme;
    class NodeObstacleContact;

    struct NodeObstacleStepOptions {
        /// The Newton tolerances, max(atol, rtol * reference): the QP is solved to a tenth of them.
        real_t atol{real_t(1e-8)};
        real_t rtol{real_t(1e-9)};
        /// Penetration admitted, a length.
        real_t gap_tol{0};
        int    qp_max_it{20000};
        real_t qp_rtol{real_t(1e-10)};
    };

    /// Newton's step with a rigid obstacle (NodeObstacleContact): the linearised problem is the QP
    /// with n_i . dx_i <= gap_i, solved exactly by MPRGP in the contact nodes' frames; the tangent
    /// carries the constraints' curvature; convergence is measured on KKT (stationarity and
    /// complementarity as forces, penetration as a length).
    ///
    /// The inertia stiffness k = c0 M, which scales complementarity, is read from the Newmark the
    /// step's scheme composes, so it needs inertia. When MPRGP stops (an indefinite tangent during
    /// an impact) the step is solved again with the curvature term capped, then with the
    /// tangent's diagonal shifted by a growing fraction of itself. It runs where the function does.
    class NodeObstacleStep final : public LinearizedStep {
    public:
        static std::shared_ptr<NodeObstacleStep> create(const std::shared_ptr<Function>            &f,
                                                        const std::shared_ptr<NodeObstacleContact> &contact,
                                                        const std::shared_ptr<NewmarkScheme>       &newmark,
                                                        const NodeObstacleStepOptions              &options);
        ~NodeObstacleStep() override;

        const char *name() const override { return "node_obstacle"; }

        int                    begin_step(const StepContext &context) override;
        StepMeasure            measure(const real_t *const x, const real_t *const r) override;
        int                    linearize(const real_t *const x, const real_t *const r) override;
        int                    solve(const real_t *const x, const real_t *const r, real_t *const dx) override;
        std::shared_ptr<Merit> merit() override;
        int                    iterations() const override;

        /// The obstacle's reaction at the last measured state, -mu_i n_i: in the step's scaling,
        /// where the material force enters both ends of a trapezoidal step and the contact force
        /// once, the physical force is half of it.
        void reaction(real_t *const out) const;

    private:
        NodeObstacleStep();

        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    /// Columns reaction_x, reaction_y, reaction_z and touched: the resultant of the obstacle's
    /// physical force (half the step's reaction, see NodeObstacleStep::reaction) on the body, and
    /// whether a node is on the obstacle (its gap at most `touch_gap`; the reaction of a body in
    /// flight is the residual's rounding, not zero), read before the step is committed.
    std::shared_ptr<Probe> create_contact_probe(const std::shared_ptr<NodeObstacleContact> &contact,
                                                const ptrdiff_t                             ndofs,
                                                const real_t                                touch_gap);

}  // namespace sfem

#endif  // SFEM_NODE_OBSTACLE_STEP_HPP
