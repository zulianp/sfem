#pragma once

#include "sfem_base.hpp"

namespace sfem {

    /// Time-temperature superposition: the shift factor a_T(T).
    ///
    /// The convention is the standard one, tau_i(T) = a_T(T) * tau_i(T_ref):
    /// below the reference temperature a_T > 1 and relaxation slows down.  A
    /// step of length dt therefore advances the material by the reduced time
    /// dt / a_T, which is the only way a shift reaches the integrator.
    class TimeTemperatureShift {
    public:
        virtual ~TimeTemperatureShift() = default;
        virtual real_t factor(const real_t temperature) const = 0;
    };

    /// No time-temperature dependence: a_T = 1 at every temperature.
    class NoShift final : public TimeTemperatureShift {
    public:
        real_t factor(const real_t) const override { return 1; }
    };

    /// Williams-Landel-Ferry: log10 a_T = -C1 (T - T_ref) / (C2 + T - T_ref).
    ///
    /// The formula has a pole at T = T_ref - C2 and is meaningless below it, so
    /// a temperature there is an error rather than a silently unshifted step.
    class WLFShift final : public TimeTemperatureShift {
    public:
        WLFShift(const real_t C1, const real_t C2, const real_t T_ref);
        real_t factor(const real_t temperature) const override;

        real_t C1() const { return C1_; }
        real_t C2() const { return C2_; }
        real_t T_ref() const { return T_ref_; }

    private:
        real_t C1_;
        real_t C2_;
        real_t T_ref_;
    };

}  // namespace sfem
