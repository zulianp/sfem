#pragma once

#include <memory>

#include "sfem_ForwardDeclarations.hpp"
#include "sfem_aliases.hpp"
#include "sfem_base.hpp"

namespace sfem {

    class Op;

    /// How a time derivative in a material's form is discretised.
    ///
    /// A material written with `gen.dt(u)` names no scheme.  Its kernels take
    /// the derivative as `u_dot = shift * u + z`: one scalar weight on the
    /// unknown and one pre-combined history vector.  Every one-step method the
    /// library needs is that shape -- backward Euler is `shift = 1/dt` with
    /// `z = -u_old/dt`, Newmark is `shift = gamma/(beta*dt)` with `z` built
    /// from the previous velocity and acceleration -- so the material is
    /// compiled once and the scheme arrives here.  PETSc's TS calls the weight
    /// the shift and forms `shift * F_u_dot + F_u` from it; the same Jacobian
    /// falls out of SFEM's differentiation with nothing added, because the
    /// shift is an ordinary scalar coefficient in the form.
    ///
    /// This is an object the material `Op` holds, not an operator beside it.
    /// The reason is the residual-based merit: `Function::value` reduces
    /// `0.5 * ||Function::gradient(x)||^2` whenever any operator reduces
    /// node-wise, so the residual and the merit are two assemblies of the same
    /// quantity.  They agree only if both read the same shift and the same
    /// history.  Handing those to the material through `set_value_in_block` and
    /// `set_field` would make that an ordering convention between operators;
    /// asking one object makes it a property of the object.
    ///
    /// `begin_step` and `advance` exist for the same reason.  The shift and the
    /// history belong to the *step*: they are built from the state at the start
    /// of it and never from the current iterate.  `Function::value_steps`
    /// re-assembles the residual at every trial step length of a line search,
    /// so anything that recomputed them from `x` would move the merit under the
    /// search.  Computing them in `Op::update(x)`, which runs once per Newton
    /// iteration and is handed the iterate, would leave that safety to
    /// convention; a step boundary that is its own call does not.
    ///
    /// A scheme whose method contributes a separable term of its own -- the
    /// inertia of an elastodynamic problem, say -- publishes it through
    /// `inertia_op`, and the caller adds that operator to the `Function`.  It
    /// has to be an operator: the node-wise merit sees only what
    /// `Function::gradient` assembles.
    class TimeScheme {
    public:
        virtual ~TimeScheme() = default;

        /// The weight on the unknown, `d(u_dot)/du`.
        virtual real_t shift() const = 0;

        /// The rest of `u_dot`, everything the history contributes.
        virtual const real_t *history() const = 0;

        /// A coefficient of the method itself, by name -- `dt`, and whatever
        /// else the method defines.  Present so a caller can read what the
        /// scheme is doing without going through the generic `Op` string ABI.
        virtual real_t weight(const char *name) const = 0;

        /// Open a step: build the shift and the history from the state carried
        /// out of the last one.  Everything the solve reads is fixed here.
        virtual void begin_step(real_t t, real_t dt) = 0;

        /// Close a step at the solution `x`: reconstruct whatever derived
        /// states the method carries, then rotate them.
        virtual void advance(const real_t *const x) = 0;

        /// The separable term this method contributes to the residual, if it
        /// has one, for the caller to add to the `Function`.
        virtual std::shared_ptr<Op> inertia_op() const { return nullptr; }
    };

    /// An `Op` whose form carries a time derivative, and can therefore be
    /// handed the scheme that discretises it.
    ///
    /// Separate from `Op` on purpose.  Most operators have no time derivative,
    /// and a method on the base class that almost nothing can honour is the
    /// string-keyed ABI again with a compiler-checked name on it.  Separate
    /// from the generated class too, so a caller attaches a scheme without
    /// naming the material: `dynamic_pointer_cast<TimeSteppable>` both asks
    /// whether this operator has a rate and yields the way to set it.
    class TimeSteppable {
    public:
        virtual ~TimeSteppable() = default;

        virtual void set_time_scheme(const std::shared_ptr<TimeScheme> &scheme) = 0;
    };

}  // namespace sfem
