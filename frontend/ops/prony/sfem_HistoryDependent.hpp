#pragma once

#include "sfem_base.hpp"

namespace sfem {

    /// An `Op` whose material carries integration-point history -- a Prony
    /// series, for one -- and therefore has to be told where steps begin and
    /// end.
    ///
    /// Separate from `Op` for the reason `TimeSteppable` is: most operators
    /// carry no history, and a caller asks whether this one does with
    /// `dynamic_pointer_cast<HistoryDependent>`, which also yields the way to
    /// drive it, without naming the material.
    ///
    /// The contract mirrors `TimeScheme`'s.  `begin_step` fixes everything a
    /// solve reads, so `gradient`, `apply` and `value` are pure functions of
    /// the iterate for the whole step -- a line search that re-assembles at
    /// every trial step, or a rejected trial, leaves no trace.  `advance` is
    /// the one call that mutates the history, and only at a converged `x`.
    class HistoryDependent {
    public:
        virtual ~HistoryDependent() = default;

        /// Open a step of length `dt` at time `t`.  May be called again to
        /// retry the same step with another `dt`.
        virtual void begin_step(const real_t t, const real_t dt) = 0;

        /// Commit the open step at the converged solution `x`.
        virtual void advance(const real_t *const x) = 0;

        /// Restart the history from rest at the configuration `x`.  Needed only
        /// when the initial configuration is not stress free.
        virtual void reset_history(const real_t *const x) = 0;
    };

}  // namespace sfem
