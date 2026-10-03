// The slope limiters in src/venkata, checked where reading them is not enough.
//
// The load-bearing claim is the zero-severity control: Venkatakrishnan's arm with K = 0 must
// reproduce the arm that preceded it BIT FOR BIT, not merely closely. That is what makes a
// sweep over K a sweep over one term rather than a comparison of two schemes, and it is a
// claim about floating-point evaluation order, so it is checked by evaluating both and
// comparing with ==. Reading the two expressions and agreeing that they match is exactly the
// check that does not catch a reassociation.

#include <cmath>
#include <cstdio>
#include <vector>

#include "kernels/microkernels/limiters/cvfem_venkata_limiter.hpp"

using scalar_t = double;

static int g_failures = 0;

static void check(const bool ok, const char *what, const double detail = 0.0) {
    std::printf("%-62s %s", what, ok ? "OK" : "FAIL");
    if (detail != 0.0) std::printf("   (%g)", detail);
    std::printf("\n");
    if (!ok) ++g_failures;
}

// The limiter exactly as it read before the eps^2 term was added, transcribed here and
// nowhere else. This is the one place in the tree a second copy of the formula is wanted:
// its whole purpose is to be the independent witness that the new one reduces to it.
static scalar_t legacy_vk(const scalar_t base, const scalar_t inc, const scalar_t lo, const scalar_t hi) {
    const scalar_t dp  = (inc >= scalar_t(0)) ? (hi - base) : (lo - base);
    const scalar_t num = dp * dp + scalar_t(2) * inc * dp;
    const scalar_t den = dp * dp + scalar_t(2) * inc * inc + inc * dp;
    return (den != scalar_t(0)) ? (num / den) * inc : inc;
}

int main() {
    // A grid that reaches the cases the arms differ on: increments far larger than the room
    // available, increments of both signs, zero room, zero increment, and a node pair that
    // agrees to round-off -- the flow-aligned edge that made the limiter fail in the first
    // place.
    const std::vector<scalar_t> vals = {-3.0, -1.0, -1e-14, 0.0, 1e-14, 0.25, 1.0, 7.0};

    // ---- the zero-severity control -------------------------------------------------
    {
        int n = 0;
        bool exact = true;
        for (const scalar_t a : vals)
            for (const scalar_t b : vals)
                for (const scalar_t inc : vals) {
                    const scalar_t lo = a < b ? a : b;
                    const scalar_t hi = a < b ? b : a;
                    const scalar_t got = cvfem_venkata_inc(a, inc, lo, hi, scalar_t(0));
                    const scalar_t ref = legacy_vk(a, inc, lo, hi);
                    // NaN is not expected from either; if one produced it the == would pass
                    // silently for the wrong reason, so it is rejected explicitly.
                    if (std::isnan(got) || std::isnan(ref) || got != ref) exact = false;
                    ++n;
                }
        check(exact, "K = 0 reproduces the preceding arm bit for bit", (double)n);
    }

    // ---- the eps^2 term is a deactivation switch, and only that ---------------------
    {
        // A flow-aligned edge: the two nodes agree to round-off, so the bound is a hair wide
        // and the unmodified limiter throws the increment away. This is the h^-2 failure.
        const scalar_t a = 1.0, b = 1.0 + 1e-14, inc = 0.1;
        const scalar_t lo = a, hi = b;
        const scalar_t off = cvfem_venkata_inc(a, inc, lo, hi, scalar_t(0));
        const scalar_t on  = cvfem_venkata_inc(a, inc, lo, hi, scalar_t(1.0));
        check(std::fabs(off) < 1e-12 * std::fabs(inc),
              "with no eps^2 a flow-aligned edge loses its increment", (double)(off / inc));
        check(on > scalar_t(0.9) * inc,
              "eps^2 above the local variation passes the increment through", (double)(on / inc));
    }
    {
        // And where the variation is genuinely large, eps^2 must NOT rescue the increment:
        // a switch that never switches off is the same defect in the other direction.
        const scalar_t a = 0.0, b = 1.0, inc = 50.0, lo = 0.0, hi = 1.0;
        const scalar_t off = cvfem_venkata_inc(a, inc, lo, hi, scalar_t(0));
        const scalar_t on  = cvfem_venkata_inc(a, inc, lo, hi, scalar_t(1e-6));
        check(std::fabs(on - off) < 1e-4 * std::fabs(off),
              "a small eps^2 leaves a large variation alone", (double)((on - off) / off));
        check(on < inc, "the large increment is still limited", (double)(on / inc));
    }

    // ---- the coefficient carries the units -----------------------------------------
    {
        check(cvfem_venkata_eps2_coeff<scalar_t>(0.0, 1.0, 1.0) == scalar_t(0),
              "K = 0 gives a zero coefficient, so the term cannot switch on");
        check(cvfem_venkata_eps2_coeff<scalar_t>(0.5, 1.0, 1.0) == scalar_t(0.125),
              "at uref = lref = 1 the coefficient is Venkatakrishnan's own K^3");
        // eps^2 has the units of the solution squared: doubling the reference velocity must
        // quadruple it, and doubling the box must divide it by eight.
        const scalar_t c1 = cvfem_venkata_eps2_coeff<scalar_t>(1.0, 1.0, 1.0);
        const scalar_t c2 = cvfem_venkata_eps2_coeff<scalar_t>(1.0, 2.0, 1.0);
        const scalar_t c3 = cvfem_venkata_eps2_coeff<scalar_t>(1.0, 1.0, 2.0);
        check(std::fabs(c2 - 4 * c1) < 1e-15, "doubling uref quadruples eps^2", (double)(c2 / c1));
        check(std::fabs(c3 - c1 / 8) < 1e-15, "doubling lref divides eps^2 by eight", (double)(c3 / c1));
    }

    // ---- the clip, which must not have moved ---------------------------------------
    {
        check(cvfem_limiter_clip_inc<scalar_t>(0.0, 5.0, 0.0, 1.0) == scalar_t(1.0),
              "the clip caps an increment at the upper bound");
        check(cvfem_limiter_clip_inc<scalar_t>(0.0, -5.0, 0.0, 1.0) == scalar_t(0.0),
              "the clip discards an increment that leaves the interval");
        check(cvfem_limiter_clip_inc<scalar_t>(0.0, 0.5, 0.0, 1.0) == scalar_t(0.5),
              "the clip passes an increment that stays inside");
    }

    // ---- Darwish-Moukalled, against van Leer's values -------------------------------
    {
        // r < 0: the reconstruction and the downwind difference disagree, and van Leer
        // returns nothing. Exactly zero, not nearly zero -- the cancellation is algebraic.
        check(cvfem_darwish_moukalled_inc<scalar_t>(0.0, 1.0, -1.0) == scalar_t(0),
              "opposite signs give exactly zero, which is psi(r < 0) = 0");
        // r = 1 is a = b, where psi = 1 and the increment is half the downwind difference.
        const scalar_t b = 0.8;
        const scalar_t at_one = cvfem_darwish_moukalled_inc<scalar_t>(0.0, b, b / 2);
        check(std::fabs(at_one - b / 2) < 1e-15, "psi(1) = 1 gives half the downwind difference",
              (double)(at_one / b));
        // r -> infinity: psi -> 2 and the increment saturates at the full downwind difference.
        const scalar_t big = cvfem_darwish_moukalled_inc<scalar_t>(0.0, b, 1e8);
        check(std::fabs(big - b) < 1e-6 * b, "psi saturates at the full downwind difference",
              (double)(big / b));
        // Both arguments zero is the only place the denominator vanishes.
        check(cvfem_darwish_moukalled_inc<scalar_t>(1.0, 1.0, 0.0) == scalar_t(0),
              "a flat edge returns zero rather than dividing by zero");
    }

    // ---- the transfer function, which is what separates "off" from "dormant" -------
    //
    // THE MEASUREMENT GAP THIS CLOSES. On the backward-facing step, K = 5 took the fine cell
    // from 303 Newton steps to 41 and K = 30 stopped converging by becoming the unlimited arm.
    // On the manufactured solution every arm recovered 95-129% of the gap to the unlimited
    // reconstruction. Neither measurement can distinguish a limiter that has SWITCHED ITSELF
    // OFF from one that is correctly DORMANT, because the manufactured field is smooth and a
    // correct limiter should do nothing there -- so an arm that does nothing anywhere scores
    // identically to a perfect one.
    //
    // What separates them is the only thing any of these arms actually sees: r = |increment| /
    // |room to the bound|. Small r is a smooth region, where every arm must pass the increment
    // through. Large r is the reconstruction overshooting its bound, which is the ONLY place a
    // limiter earns its name. Sweeping r needs no mesh and no solve, and it is a property of
    // the arm rather than of a case.
    //
    // `retained` below is the limited increment as a fraction of the room available. Bounded
    // means it does not exceed 1 as r grows: the face value stays inside the interval its own
    // two nodes span. An arm whose retained value runs away with r is not limiting.
    {
        const scalar_t base = 0.0, lo = 0.0, hi = 1.0;  // room = 1, so r = inc
        std::printf("\n  r = inc/room    clip   venkat K=0   venkat eps2=1   Darwish-Moukalled\n");
        for (const scalar_t r : {0.01, 0.1, 0.5, 1.0, 4.0, 20.0}) {
            std::printf("  %10.2f  %8.4f   %9.4f   %13.4f   %15.4f\n", (double)r,
                        (double)cvfem_limiter_clip_inc(base, r, lo, hi),
                        (double)cvfem_venkata_inc(base, r, lo, hi, scalar_t(0)),
                        (double)cvfem_venkata_inc(base, r, lo, hi, scalar_t(1)),
                        (double)cvfem_darwish_moukalled_inc(base, hi, r / 2));
        }
        std::printf("\n");

        // Every arm must be transparent where the field is smooth.
        for (const scalar_t r : {scalar_t(0.001), scalar_t(0.01)}) {
            check(cvfem_venkata_inc(base, r, lo, hi, scalar_t(0)) > scalar_t(0.9) * r,
                  "venkat K=0 passes a small increment through");
            check(cvfem_limiter_clip_inc(base, r, lo, hi) == r,
                  "the clip passes a small increment through untouched");
        }

        // And the ones that claim to bound must bound: retained <= room as r runs away.
        const scalar_t big = 20.0;
        check(cvfem_limiter_clip_inc(base, big, lo, hi) <= hi + scalar_t(1e-12),
              "the clip is bounded at large r");
        check(cvfem_venkata_inc(base, big, lo, hi, scalar_t(0)) <= hi + scalar_t(1e-12),
              "Venkatakrishnan at K = 0 is bounded at large r");
        check(cvfem_darwish_moukalled_inc(base, hi, big / 2) <= hi + scalar_t(1e-12),
              "Darwish-Moukalled is bounded at large r");

        // THE TRADE, ASSERTED SO IT CANNOT BE FORGOTTEN. A large eps^2 does not merely
        // deactivate the limiter in smooth regions -- past some size it deactivates it
        // everywhere, including where the reconstruction is overshooting badly. That is why
        // K = 30 reproduced the unlimited arm on both the step case and the MMS ladder, and
        // it is why K cannot be raised freely to buy convergence. The number here is what an
        // eps^2 far above the local variation costs: boundedness itself.
        const scalar_t off = cvfem_venkata_inc(base, big, lo, hi, scalar_t(1000));
        check(off > hi,
              "a large eps^2 breaks boundedness -- the deactivation trade, made explicit");
        // It approaches the unlimited scheme rather than reaching it at any particular eps^2:
        // psi -> 1 only as eps^2 dominates D-^2, which at r = 20 means eps^2 well past 400.
        // eps^2 = 1000 retains 11.4 of 20 -- already unbounded, still not transparent -- so
        // the limit is asserted where it actually holds, and monotonicity in between is what
        // says the two ends are joined by a dial rather than by a jump.
        const scalar_t further = cvfem_venkata_inc(base, big, lo, hi, scalar_t(1e6));
        check(further > off, "raising eps^2 further retains more of the increment");
        check(further > scalar_t(0.95) * big,
              "and in the limit the arm IS the unlimited scheme, which is how K = 30 reproduced it");
    }

    if (g_failures) {
        std::fprintf(stderr, "cvfem_venkata_test: %d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all limiter checks passed\n");
    return 0;
}
