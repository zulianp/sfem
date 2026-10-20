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
        /// A QP stopped by qp_max_it is the step as it stands, not a failure to retry.
        bool qp_accept_capped{false};
        /// The obstacle's curvature in the tangent.
        bool curvature{true};
    };

    /// Newton's step with a rigid obstacle (NodeObstacleContact): the linearised problem is the QP
    /// with n_i . dx_i <= gap_i, solved exactly by MPRGP in the contact nodes' frames, Jacobi-scaled
    /// (the variables z = D^{1/2} y, D the tangent's diagonal, its mean over a contact node's three:
    /// the bounds stay a box; 1.06 to 1.43 times faster on the valve-containment cases, at the same
    /// accuracy: MPRGP stops on the unscaled projected gradient); the tangent
    /// carries the constraints' curvature; convergence is measured on KKT (stationarity and
    /// complementarity as forces, penetration as a length).
    ///
    /// Complementarity is scaled by each dof's stiffness k: the diagonal of the tangent at the
    /// reference configuration, plus the inertia c0 M of the Newmark the step's scheme composes
    /// when there is one (`newmark` null: a steady or quasi-static step). When MPRGP stops (an
    /// indefinite tangent during an impact) the step is solved again with the curvature term
    /// capped, then with the tangent's diagonal shifted by a growing fraction of itself. It runs
    /// where the function does.
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
        /// The obstacle's gap is linearised: Newton measures every iterate.
        bool solves_linear_function_exactly() const override { return false; }

        /// The obstacle's reaction at the last measured state, -mu_i n_i: in the step's scaling,
        /// where the material force enters both ends of a trapezoidal step and the contact force
        /// once, the physical force is half of it.
        void reaction(real_t *const out) const;

    private:
        NodeObstacleStep();

        class Impl;
        std::unique_ptr<Impl> impl_;
    };


}  // namespace sfem

#endif  // SFEM_NODE_OBSTACLE_STEP_HPP
