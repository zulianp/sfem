#ifndef SFEM_LINEARIZED_STEP_HPP
#define SFEM_LINEARIZED_STEP_HPP

#include <memory>

#include "sfem_Merit.hpp"
#include "sfem_NonlinearSolver.hpp"
#include "sfem_base.hpp"

namespace sfem {

    /// What a step says about an iterate: the stationarity `value` the tolerance applies to, its
    /// scale `reference`, and whether the iterate is `admissible` (a constraint met to its own
    /// tolerance, `infeasibility` how far it is from that).
    struct StepMeasure {
        real_t value{0};
        real_t reference{0};
        bool   admissible{true};
        real_t infeasibility{0};
    };

    /// The linearised problem inside a Newton iteration, and what decides that the iteration is
    /// done. Without constraints it is a Krylov solve of the tangent; a constrained back-end (an
    /// obstacle, a penalty, a multigrid that bounds its corrections) solves its constrained
    /// counterpart and measures convergence on its optimality conditions.
    class LinearizedStep {
    public:
        virtual ~LinearizedStep() = default;

        virtual const char *name() const = 0;

        /// Once per step, before its first iterate.
        virtual int begin_step(const StepContext &) { return SFEM_SUCCESS; }

        /// The measure at x, whose residual is r (the constrained rows zero).
        virtual StepMeasure measure(const real_t *const x, const real_t *const r) = 0;

        /// Build the linearisation at x.
        virtual int linearize(const real_t *const x, const real_t *const r) = 0;

        /// The correction dx at x: K dx = -r, or the constrained problem it stands for. Returns
        /// SFEM_SUCCESS when the inner solver met its tolerance; a solve stopped by its cap
        /// returns SFEM_FAILURE with its last iterate in dx, for the line search to judge.
        virtual int solve(const real_t *const x, const real_t *const r, real_t *const dx) = 0;

        /// A merit of its own, or null for the function's (energy or residual).
        virtual std::shared_ptr<Merit> merit() { return nullptr; }

        /// Once per step, after its solution is accepted.
        virtual int end_step(const real_t *const) { return SFEM_SUCCESS; }

        /// Inner iterations of the last `solve`.
        virtual int iterations() const = 0;
    };

}  // namespace sfem

#endif  // SFEM_LINEARIZED_STEP_HPP
