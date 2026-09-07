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
