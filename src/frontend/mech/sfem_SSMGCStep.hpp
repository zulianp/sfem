#ifndef SFEM_SSMGC_STEP_HPP
#define SFEM_SSMGC_STEP_HPP

#include <memory>

#include "sfem_ContactStabilisedNewmarkScheme.hpp"
#include "sfem_LinearizedStep.hpp"

namespace sfem {

    class ContactConditions;
    class Function;
    class Input;
    class NewmarkScheme;

    struct SSMGCStepOptions {
        /// The Newton tolerances, max(atol, rtol * reference).
        real_t atol{real_t(1e-8)};
        real_t rtol{real_t(1e-9)};
        /// Penetration admitted, a length.
        real_t gap_tol{0};
        /// Adds the constraints' curvature times the reactions, mu_i C_i, to the tangent (the nodal
        /// gap only): without it Newton is linear on a curved obstacle. Off, nothing is allocated
        /// for it.
        bool curvature{true};
        /// SSMGC's own keys (max_it, penalty_param, coarse_op_type, ...); null for its defaults.
        std::shared_ptr<Input> input;
    };

    /// Newton's step with a rigid obstacle on a semi-structured mesh, the constrained linear problem
    /// solved by the shifted-penalty multigrid (SSMGC): K (x + dx) = K x - r with
    /// n_i . (x + dx)_i <= gap_i + n_i . x_i at the contact nodes. SSMGC coarsens the material
    /// (Function::derefine), so the material must be linear and the space a micro-element one; the
    /// hierarchy is rebuilt when the scheme's alpha changes (its coarse operators are assembled).
    /// With `curvature` the tangent is K + mu_i C_i (ContactConditions::curvature) and the step solves
    /// (K + C)(x + dx) = (K + C) x - r; when SSMGC fails with it, the step is solved again with the
    /// term capped as the node-SDF step caps it (node_obstacle_curvature).
    ///
    /// Convergence is measured on KKT as the node-SDF step does: the residual with the admissible
    /// reaction mu_i = max(-n_i . r_i, 0) removed, complementarity min(mu_i, k_i max(gap_i, 0)) with
    /// k = c0 M from the step's Newmark, penetration as a length (gap_tol: SSMGC is a penalty
    /// method). The gap is ContactConditions' (ContactConditions::Gap, nodal by default). It is also the contact the
    /// contact-stabilised Newmark scheme reads: the reaction and the velocity projection.
    class SSMGCStep final : public LinearizedStep, public ContactStabilisedNewmarkScheme::Contact {
    public:
        static std::shared_ptr<SSMGCStep> create(const std::shared_ptr<Function>          &f,
                                                 const std::shared_ptr<ContactConditions> &contact,
                                                 const std::shared_ptr<NewmarkScheme>     &newmark,
                                                 const SSMGCStepOptions                   &options);
        ~SSMGCStep() override;

        /// Before the first step: the step exists before the Newton tolerances are read, because
        /// the contact-stabilised scheme it serves is built first.
        void set_options(const SSMGCStepOptions &options);
        /// The gap times the normal at the contact nodes for x, zero elsewhere (output).
        int gap_field(const real_t *const x, real_t *const out);

        const char            *name() const override { return "ssmgc"; }
        int                    begin_step(const StepContext &context) override;
        StepMeasure            measure(const real_t *const x, const real_t *const r) override;
        int                    linearize(const real_t *const x, const real_t *const r) override;
        int                    solve(const real_t *const x, const real_t *const r, real_t *const dx) override;
        std::shared_ptr<Merit> merit() override;
        int                    iterations() const override;
        /// The obstacle's gap is linearised: Newton measures every iterate.
        bool solves_linear_function_exactly() const override { return false; }

        /// -mu_i n_i at the contact nodes with mu_i of the last measure, zero elsewhere: the
        /// residual's part that is the obstacle's reaction (the step's scaling, see
        /// create_contact_probe).
        void reaction(real_t *const out) const override;
        /// Moves each contact node of x + dt v that penetrates the obstacle back onto it along the
        /// normal, and v with it.
        void project_velocity(const real_t *const x, real_t *const v, const real_t dt) override;
        /// The smallest gap of the contact nodes at the last measure.
        real_t min_gap() const;

    private:
        SSMGCStep();
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

}  // namespace sfem

#endif  // SFEM_SSMGC_STEP_HPP
