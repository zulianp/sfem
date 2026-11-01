#ifndef SFEM_UNCONSTRAINED_STEP_HPP
#define SFEM_UNCONSTRAINED_STEP_HPP

#include <memory>
#include <string>

#include "sfem_InexactOptions.hpp"
#include "sfem_LinearizedStep.hpp"
#include "sfem_defs.hpp"

namespace sfem {

    class Function;

    struct LinearSolverOptions {
        /// "cg" or "bcgs".
        std::string type{"cg"};
        /// An `op_type`: "BSR" assembles into one `BSRTangent`, "MF" applies the function,
        /// "INEXACT" applies the stored tangent `Function::inexact_update` builds, and the other
        /// assembled formats go through `create_linear_operator` at each linearisation.
        std::string format{op_type::BSR};
        /// "jacobi" or "none". Jacobi reads the BSR diagonal, or `Function::hessian_diag` for the
        /// other formats.
        std::string preconditioner{"jacobi"};
        /// Inexact Newton: the step needs no more accuracy than the Newton direction has.
        real_t rtol{real_t(1e-3)};
        real_t atol{real_t(1e-16)};
        int    max_it{2000};
        /// CG stops at a direction of non-positive curvature and returns the descent direction it
        /// has built: a direction for a line search to scale, so only with one.
        bool stop_on_negative_curvature{true};
        /// Bytes per stored value of an assembled BSR tangent: 2, 4 or sizeof(real_t)
        /// (BSRTangent::create).
        int storage_bytes{(int)sizeof(real_t)};
        /// What format INEXACT stores (`solver.linear.inexact`).
        InexactOptions inexact;
    };

    /// The linearised step without constraints beyond the Dirichlet rows: K dx = -r by a Krylov
    /// method, converged on ||r||.
    class UnconstrainedStep final : public LinearizedStep {
    public:
        static std::shared_ptr<UnconstrainedStep> create(const std::shared_ptr<Function> &f,
                                                         const ExecutionSpace             es,
                                                         const LinearSolverOptions       &options = LinearSolverOptions());
        ~UnconstrainedStep() override;

        const char *name() const override { return "unconstrained"; }
        StepMeasure measure(const real_t *const x, const real_t *const r) override;
        int         linearize(const real_t *const x, const real_t *const r) override;
        int         solve(const real_t *const x, const real_t *const r, real_t *const dx) override;
        int         iterations() const override;

        /// Solve to `rtol` rather than the inexact tolerance, as a linear problem is solved by one
        /// step.
        void set_rtol(const real_t rtol);

        const LinearSolverOptions &options() const;

    private:
        UnconstrainedStep();

        class Impl;
        std::unique_ptr<Impl> impl_;
    };

}  // namespace sfem

#endif  // SFEM_UNCONSTRAINED_STEP_HPP
