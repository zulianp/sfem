#include "sfem_TimeTemperatureShift.hpp"

#include <cassert>
#include <cmath>

#include "sfem_logger.hpp"

namespace sfem {

    WLFShift::WLFShift(const real_t C1, const real_t C2, const real_t T_ref) : C1_(C1), C2_(C2), T_ref_(T_ref) {
        if (!(C2 > 0)) {
            SFEM_ERROR("WLFShift: C2 = %g must be positive\n", (double)C2);
        }
    }

    real_t WLFShift::factor(const real_t temperature) const {
        const real_t dT    = temperature - T_ref_;
        const real_t denom = C2_ + dT;
        if (!(denom > 0)) {
            SFEM_ERROR("WLFShift: T = %g is at or below the WLF pole T_ref - C2 = %g\n",
                       (double)temperature,
                       (double)(T_ref_ - C2_));
        }

        return std::pow(real_t(10), -C1_ * dT / denom);
    }

}  // namespace sfem
