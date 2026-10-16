#ifndef SFEM_MERIT_HPP
#define SFEM_MERIT_HPP

#include <memory>
#include <string>

#include "sfem_base.hpp"

namespace sfem {

    class Function;

    /// A scalar the Newton step must decrease, evaluated along a direction h from x.
    class Merit {
    public:
        virtual ~Merit() = default;

        virtual const char *name() const = 0;

        /// m(x + steps[i] h) into out[i], assigned; +inf where it cannot be evaluated (an
        /// inverted element, a non-finite value).
        virtual int evaluate(const real_t *const x,
                             const real_t *const h,
                             const int           nsteps,
                             const real_t *const steps,
                             real_t *const       out) = 0;

        /// m(x), given the residual r at x.
        virtual real_t at(const real_t *const x, const real_t *const r) = 0;

        /// dm/ds at s = 0 along h, given the residual r at x and m0 = m(x).
        virtual real_t slope(const real_t *const x, const real_t *const r, const real_t *const h, const real_t m0) = 0;

        /// Whether one call evaluates many steps for about the cost of one, so a line search
        /// asks for its whole schedule at once rather than step by step.
        virtual bool batched() const = 0;
    };

    /// The sum of the operators' potentials, `Function::energy_merit`. Its slope is r . h.
    std::shared_ptr<Merit> create_energy_merit(const std::shared_ptr<Function> &f);

    /// 1/2 ||R||^2, `Function::residual_merit`. Its slope along a Newton direction, K h = -r,
    /// is -||r||^2.
    std::shared_ptr<Merit> create_residual_merit(const std::shared_ptr<Function> &f);

    /// "energy", "residual", or "auto": the energy where `f` has one, otherwise the residual.
    std::shared_ptr<Merit> create_merit(const std::shared_ptr<Function> &f, const std::string &choice);

}  // namespace sfem

#endif  // SFEM_MERIT_HPP
