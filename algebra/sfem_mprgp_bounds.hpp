#ifndef SFEM_MPRGP_BOUNDS_HPP
#define SFEM_MPRGP_BOUNDS_HPP

#include <cstddef>

#include "sfem_base.hpp"

namespace sfem {

    /// The per-dof rules of MPRGP for a box with a lower bound (LB), an upper bound (UB) or both,
    /// one definition for the host and the device passes.
    template <typename T, bool LB, bool UB>
    struct MPRGPBounds {
        const T* SFEM_RESTRICT lb;
        const T* SFEM_RESTRICT ub;

        static SFEM_DEVICE_INLINE T min(const T a, const T b) { return (b < a) ? b : a; }
        static SFEM_DEVICE_INLINE T max(const T a, const T b) { return (a < b) ? b : a; }
        static SFEM_DEVICE_INLINE bool near(const T a, const T b, const T eps) {
            const T d = a - b;
            return d < eps && -d < eps;
        }

        SFEM_DEVICE_INLINE T project(const ptrdiff_t i, T xi) const {
            if constexpr (UB) xi = min(xi, ub[i]);
            if constexpr (LB) xi = max(xi, lb[i]);
            return xi;
        }

        /// The gradient where no bound is active, zero on one.
        SFEM_DEVICE_INLINE T free(const ptrdiff_t i, const T xi, const T gi) const {
            if constexpr (LB) {
                if (xi <= lb[i]) return T(0);
            }
            if constexpr (UB) {
                if (xi >= ub[i]) return T(0);
            }
            return gi;
        }

        /// The gradient's part leaving an active bound, zero elsewhere.
        SFEM_DEVICE_INLINE T chopped(const ptrdiff_t i, const T xi, const T gi, const T eps) const {
            if constexpr (LB) {
                if (near(lb[i], xi, eps)) return min(T(0), gi);
            }
            if constexpr (UB) {
                if (near(ub[i], xi, eps)) return max(T(0), gi);
            }
            return T(0);
        }

        /// The longest step xi - alpha pi that stays inside the bounds, infty when none limits it.
        SFEM_DEVICE_INLINE T max_alpha(const ptrdiff_t i, const T xi, const T pi, const T infty) const {
            T alpha = infty;
            if constexpr (LB) {
                if (pi > 0) alpha = min(alpha, (xi - lb[i]) / pi);
            }
            if constexpr (UB) {
                if (pi < 0) alpha = min(alpha, (xi - ub[i]) / pi);
            }
            return alpha;
        }
    };

    /// Calls f with the MPRGPBounds of the bounds given (either may be null, not both).
    template <typename T, class F>
    void with_mprgp_bounds(const T* const lb, const T* const ub, F&& f) {
        if (lb && ub) {
            f(MPRGPBounds<T, true, true>{lb, ub});
        } else if (ub) {
            f(MPRGPBounds<T, false, true>{lb, ub});
        } else {
            f(MPRGPBounds<T, true, false>{lb, ub});
        }
    }

}  // namespace sfem

#endif  // SFEM_MPRGP_BOUNDS_HPP
