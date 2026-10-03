#ifndef CVFEM_ENV_CONFIG_HPP
#define CVFEM_ENV_CONFIG_HPP

// WHERE A USER-LEVEL OPTION IS TURNED INTO A VALUE.
//
// DESIGN.md puts this boundary in one sentence: kernels take "only arguments that are actually
// used" and "no user level option flags are propagated down here, they are handled outside in
// the front-end". These two resolvers were on the wrong side of it -- they read the environment
// from inside src/kernels/, which is also required to carry no library dependencies. Neither was
// ever called from a kernel: both are called once per solve, from the residual's serial setup in
// src/hex8/cvfem_hex8_ns_core.hpp and src/ss/cvfem_sshex8_ns.hpp, which is what makes this a
// relocation rather than a change of behaviour.
//
// The value TYPES stay in the kernel headers, because a kernel consumes them: Hex8PecletConfig is
// a plain struct of four scalars and Hex8LimiterStats a plain struct of counters. What moves is
// the part that reads std::getenv and decides.
#include <cstdlib>

#include "kernels/microkernels/hex8/cvfem_hex8_ns_upwind_kernels.hpp"
#include "kernels/microkernels/limiters/cvfem_venkata_limiter.hpp"

inline void cvfem_limiter_stats_report() {
    const Hex8LimiterStats &s = cvfem_limiter_stats_object();
    if (s.n_samples == 0) return;
    const double out = 100.0 * (double)s.n_outside / (double)s.n_samples;
    const double chg = 100.0 * (double)s.n_changed / (double)s.n_samples;
    std::fprintf(stderr,
                 "limiter_stats: samples %ld  outside_bound %ld (%.4f%%)  changed %ld (%.4f%%)  "
                 "still_out %ld (%.4f%%)  max_excess %.6g\n",
                 s.n_samples, s.n_outside, out, s.n_changed, chg, s.n_still_out,
                 100.0 * (double)s.n_still_out / (double)s.n_samples, s.max_excess);
    if (s.n_outside == 0)
        std::fprintf(stderr,
                     "limiter_stats: the reconstruction never left the bound, so THIS CASE "
                     "CANNOT TEST A LIMITER -- any comparison of limited against unlimited on "
                     "it is a comparison of two schemes that did the same thing.\n");
}

// Returns the sink when the diagnostic is on, nullptr otherwise. Called from the residual's
// serial env-reading block, so the lazy initialisation below is not racing.
//
// The sink reaches the arithmetic as a template argument now, not as this pointer: an
// `#pragma omp atomic` inside a `#pragma omp simd` lane loop is a barrier that cost the
// higher-order kernels their vectorisation entirely -- see cvfem_hex8_scs_defcor's STATS
// parameter. The scalar sweep, which is what this diagnostic has always measured, is unaffected.
// Returns the sink when the diagnostic is on, nullptr otherwise. Called from the residual's
// serial env-reading block, so the lazy initialisation below is not racing.
inline Hex8LimiterStats *cvfem_limiter_stats_sink() {
    static int on = -1;
    if (on < 0) {
        const char *e = std::getenv("SFEM_LIMITER_STATS");
        on = (e && *e && *e != '0') ? 1 : 0;
        if (on) std::atexit(cvfem_limiter_stats_report);
    }
    return on ? &cvfem_limiter_stats_object() : nullptr;
}

template <typename scalar_t>
inline Hex8PecletConfig<scalar_t> cvfem_hex8_peclet_config() {
    // std::getenv and a function-local static, for the reason cvfem_hex8_rc_config gives:
    // the benchmark shares this header and has no Env to read through, and the value must be
    // read once rather than per face.
    static const Hex8PecletConfig<scalar_t> c = [] {
        Hex8PecletConfig<scalar_t> k;
        const char *const f = std::getenv("SFEM_PECLET_BLEND");
        if (!f || f[0] == '0' || f[0] == '\0') return k;            // off, eta = 1
        if (f[0] == 'c') k.form = 1;                                 // "classic"
        else if (f[0] == 't') k.form = 2;                            // "tanh"
        else return k;
        const char *const g = std::getenv("SFEM_PECLET_GAMMA");
        const char *const t = std::getenv("SFEM_PECLET_TRANS");
        const char *const w = std::getenv("SFEM_PECLET_WIDTH");
        // Nalu's velocity defaults, so asking for the tanh form without saying more gives the
        // scheme the reference implementation actually runs rather than a neutral one.
        k.gamma = g ? (scalar_t)std::atof(g) : scalar_t(1);
        k.trans = t ? (scalar_t)std::atof(t) : scalar_t(50000);
        k.width = w ? (scalar_t)std::atof(w) : scalar_t(200);
        if (k.width <= scalar_t(0)) k.width = scalar_t(1);
        return k;
    }();
    return c;
}

#endif  // CVFEM_ENV_CONFIG_HPP
