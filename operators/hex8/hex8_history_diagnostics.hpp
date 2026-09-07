#ifndef HEX8_HISTORY_DIAGNOSTICS_HPP
#define HEX8_HISTORY_DIAGNOSTICS_HPP

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

inline bool sfem_history_checks_enabled() {
    const char *value = std::getenv("SFEM_HISTORY_CHECK");
    return value && std::strcmp(value, "1") == 0;
}

inline bool sfem_history_finite(const double value) {
    // Inspect IEEE bits so Release/fast-math cannot assume away NaN checks.
    static_assert(sizeof(double) == sizeof(uint64_t) && std::numeric_limits<double>::is_iec559,
                  "History diagnostics require IEEE binary64");
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const volatile uint64_t observed = bits;
    return (observed & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

inline void sfem_check_history_scale(const double max_abs, const double mantissa, const int exponent,
                                     const double scale64, const float scale32,
                                     const ptrdiff_t element, const ptrdiff_t qp, const int prony) {
    if (sfem_history_finite(scale64) && scale64 > 0 &&
        sfem_history_finite(scale32) && scale32 > 0) return;

#pragma omp critical(sfem_history_diagnostic)
    {
        static_assert(sizeof(float) == sizeof(uint32_t) && std::numeric_limits<float>::is_iec559,
                      "Scale diagnostics require IEEE binary32");
        uint32_t bits;
        std::memcpy(&bits, &scale32, sizeof(bits));
        std::fprintf(stderr,
                     "[history-check] stage=scale_build element=%td qp=%td prony=%d "
                     "max_abs=%.17g mantissa=%.17g exponent=%d scale_exponent=%d "
                     "scale64=%.17g scale32=%.17g scale32_bits=0x%08x\n",
                     element, qp, prony, max_abs, mantissa, exponent, exponent - 15,
                     scale64, double(scale32), static_cast<unsigned int>(bits));
        std::fflush(stderr);
        std::_Exit(EXIT_FAILURE);
    }
}

inline void sfem_check_history(const char *stage, const ptrdiff_t element, const ptrdiff_t qp,
                               const int prony, const int component, const double value,
                               const double scale = 1, const bool require_fp16_range = false) {
    const double restored = value * scale;
    if (sfem_history_finite(value) && sfem_history_finite(scale) && scale > 0 &&
        sfem_history_finite(restored) &&
        (!require_fp16_range || (value >= -65504 && value <= 65504))) return;

    // Stop on the first reporting thread; indices are zero-based and domain-local.
#pragma omp critical(sfem_history_diagnostic)
    {
        std::fprintf(stderr,
                     "[history-check] stage=%s element=%td qp=%td prony=%d component=%d "
                     "value=%.17g scale=%.17g restored=%.17g fp16_range=%d\n",
                     stage, element, qp, prony, component, value, scale, restored,
                     int(require_fp16_range));
        std::fflush(stderr);
        std::_Exit(EXIT_FAILURE);  // Diagnostic failure: no multi-GB core dump.
    }
}

#endif
