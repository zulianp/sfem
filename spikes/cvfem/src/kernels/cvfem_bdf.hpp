#pragma once

// BDF1/BDF2 coefficients for r += rho V (a0 u^{n+1} + a1 u^n + a2 u^{n-1}) / dt.
//
// BDF2 needs two levels of history, so the first step of a run has none and must fall back
// to BDF1. That is not an approximation to apologise for -- it is the standard start-up,
// and it costs one step of first-order error in a sequence that is otherwise second-order.
// The caller signals it through have_two.
//
// This is the one copy. There were three: the benchmark family's
// (frontend/staging/cvfem_hex8_best_common.hpp), the solver family's
// (frontend/op/cvfem_hex8_ns_core.hpp) -- byte-identical once comments and linkage are
// stripped -- and a third inlined into the semi-structured transient sweep. The two families
// never share a translation unit, so the duplication could not be caught by the compiler and
// drift would have shown up as two families integrating in time differently. What each of
// those sites keeps is the staging read: which members hold dt, the previous step and the
// history. The rule itself is arithmetic on four values and lives here.

#include <cstddef>

struct BdfCoeffs {
    scalar_t a0, a1, a2;
    int      order;
};

static inline BdfCoeffs cvfem_bdf_coeffs(const int      bdf_order,
                                         const scalar_t dt,
                                         const scalar_t dt_prev,
                                         const bool     history_has_two_levels) {
    const bool have_two = bdf_order >= 2 && history_has_two_levels;
    if (!have_two) return {scalar_t(1), scalar_t(-1), scalar_t(0), 1};

    // BDF2 on a VARIABLE step. With w = dt / dt_prev,
    //
    //     a0 = (1 + 2w)/(1 + w),   a1 = -(1 + w),   a2 = w^2/(1 + w)
    //
    // which is {3/2, -2, 1/2} at w = 1 and reduces to it exactly, so a run that never changes
    // its step is bit-for-bit what it was. dt_prev <= 0 means nothing has recorded a previous
    // step -- a fresh run, a coarse level built by clone_onto, any caller that does not adapt
    // -- and those take the uniform branch rather than a guess.
    //
    // The guard matters more than the formula. Using {3/2, -2, 1/2} after the step size has
    // changed is not an approximation, it is the wrong scheme: the truncation error stops
    // cancelling and BDF2 silently becomes first order while still reporting itself as second.
    if (dt_prev > scalar_t(0) && dt > scalar_t(0)) {
        const scalar_t w = dt / dt_prev;
        if (w != scalar_t(1)) {
            const scalar_t den = scalar_t(1) + w;
            return {(scalar_t(1) + scalar_t(2) * w) / den, -den, w * w / den, 2};
        }
    }
    return {scalar_t(1.5), scalar_t(-2), scalar_t(0.5), 2};
}
