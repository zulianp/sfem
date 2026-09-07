#ifndef HEX8_HISTORY_SCALING_HPP
#define HEX8_HISTORY_SCALING_HPP

#include "hex8_history_diagnostics.hpp"
#include <algorithm>
#include <cmath>

template <class T>
static inline float fp16_history_scale(const T *const H, const int n, const bool check_history,
                                       const ptrdiff_t element, const ptrdiff_t qp, const int prony) {
    T max_abs = 0;
    for (int i = 0; i < n; ++i) {
        max_abs = fmax(max_abs, fabs(H[i]));
    }

    if (max_abs == 0) return 1;

    int exponent;
    const double mantissa = frexp(max_abs, &exponent);
    const double scale64 = ldexp(1.0, exponent - 15);
    // Floor BEFORE narrowing: avoid zero/subnormal FP32 scales and division by zero.
    // Extremely small histories can still round to zero when stored in FP16.
    const float scale32 = (float)std::max(scale64, double(std::numeric_limits<float>::min()));
    if (check_history) sfem_check_history_scale(max_abs, mantissa, exponent, scale64, scale32,
                                              element, qp, prony);
    return scale32;
}

#endif
