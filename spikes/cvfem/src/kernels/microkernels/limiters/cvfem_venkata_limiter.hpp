#ifndef CVFEM_VENKATA_LIMITER_HPP
#define CVFEM_VENKATA_LIMITER_HPP

#include <cmath>
#include "kernels/cvfem_portability.hpp"
#include <cstdio>
#include <cstdlib>


// The slope limiters for the deferred-correction convective flux.
//
// One home for the limiter arms, and the ONLY one: cvfem_hex8_scs_defcor calls these rather
// than carrying its own copies. The arms have to be compared against each other, and two of
// them differ by a single term, so a second spelling of the same formula is how one arm
// quietly stops being the other arm plus a term.
//
// Every function here limits an INCREMENT rather than a face value. That is what the
// deferred correction needs -- it adds only the difference from first order to the residual,
// and the first-order value is the donor node's -- and it keeps the limiter's own zero
// exactly representable: an arm that switches off returns the increment unchanged, and one
// that clips entirely returns zero.
//
// None of these appears in the Jacobian. The deferred correction is lagged, so the limiter
// is a source term evaluated at the previous iterate; a limiter's contribution to an
// implicit Jacobian cannot readily be evaluated, which is the reason the structure is
// deferred correction in the first place and why the assembled Jacobian stays first order.
// What that does NOT buy is freedom from differentiability: the lagged term still has to
// reach a fixed point, and an arm with an active set that flips will chatter instead. That
// is measured, not assumed -- the clip below ran 399 Newton steps without converging on a
// case the unlimited arm cleared in 14.

// What the limiter actually did, counted over a solve.
//
// THE QUESTION THIS ANSWERS, and nothing else in the tree does. Every measurement so far
// compares limited against unlimited and finds them indistinguishable: the same Newton
// counts, order of accuracy within noise, and on the smooth manufactured solution every arm
// recovers 95-129% of the gap to the unlimited scheme. That is what a CORRECTLY DORMANT
// limiter looks like -- a smooth field needs no limiting -- and it is also exactly what a
// limiter that has SWITCHED ITSELF OFF looks like. The two are indistinguishable from
// outcomes alone, and telling them apart needs the limiter to be observed rather than
// inferred.
//
// n_outside is the honest measure of whether a case exercises the limiter at all: the count
// of reconstructions that would leave the interval their own two nodes span, which is the
// only place a limiter has anything to do. If it is zero, the case cannot test the limiter
// and no comparison run on it means anything -- which would be a finding about the test
// matrix, not about the scheme.
struct Hex8LimiterStats {
    long   n_samples{0};   // (sub-control surface, velocity component, donor) triples visited
    long   n_outside{0};   // ... where the UNLIMITED reconstruction leaves [lo, hi]
    long   n_changed{0};   // ... where the selected arm altered the increment
    // ... where the LIMITED face value is still outside [lo, hi]. This is the correctness
    // assertion, as opposed to the activity counts above: an arm that claims to bound must
    // leave nothing outside. Expected zero for the clip, for Venkatakrishnan at K = 0 and for
    // Darwish-Moukalled; expected NON-zero for the unlimited arm, which is the control that
    // says the counter can see a violation at all, and for any K > 0, which relaxes the bound
    // by construction.
    long   n_still_out{0};
    double max_excess{0};  // worst unlimited excursion past the bound, relative to the range
};

// Record one increment. `inc` is the unlimited reconstruction, `out` what the arm returned.
//
// Called only from inside the deferred correction, which is itself unreachable unless the
// nodal gradient exists, so the default path never evaluates the null test. Counting is
// per-sample rather than per-face because a face can be limited in one velocity component
// and untouched in the other two, and collapsing that would report a face as "limited" on
// the strength of one component.
template <typename scalar_t>
static SFEM_INLINE void cvfem_limiter_record(Hex8LimiterStats *const stats, const scalar_t base,
                                             const scalar_t inc, const scalar_t out,
                                             const scalar_t lo, const scalar_t hi) {
    if (!stats) return;
    const scalar_t face  = base + inc;
    const scalar_t ainc  = inc < scalar_t(0) ? -inc : inc;
    const scalar_t range = hi - lo;
    scalar_t       over  = scalar_t(0);
    if (face > hi) over = face - hi;
    else if (face < lo) over = lo - face;
    // Normalised by the LARGER of the bound's width and the increment, not by the width
    // alone. Dividing by the width is what the first version did and it reported a max excess
    // of 4.3e+16: on a flow-aligned edge the two nodes agree to round-off, the width is ~1e-17,
    // and any increment at all divides out to a meaningless number. That the metric blew up
    // there is itself the h^-2 pathology -- those edges are exactly where the two-node bound
    // collapses -- but a diagnostic that reports 1e16 measures its own denominator.
    //
    // With this denominator the value is bounded and readable: 1.0 means the reconstruction
    // overshot by as much as the increment it was making, which is total, and small values
    // mean the bound was missed narrowly.
    const scalar_t den  = range > ainc ? range : ainc;
    const double   rel = (over > scalar_t(0) && den > scalar_t(0)) ? (double)(over / den) : 0.0;
    // A RELATIVE test, not out != inc. The clip returns (base + inc) - base, which differs
    // from inc by rounding even when it clamps nothing, so exact inequality reported it as
    // changing 65% of samples against the 51% that actually leave the bound -- 14 points of
    // pure floating-point round-trip. The smooth arms are unaffected either way, which is
    // precisely why the artifact was invisible until the clip was measured beside them.
    const scalar_t adiff = (out > inc ? out - inc : inc - out);
    const scalar_t abase = base < scalar_t(0) ? -base : base;
    const scalar_t scale = (ainc > abase ? ainc : abase);
    const bool     changed = adiff > scalar_t(1e-12) * (scale > scalar_t(0) ? scale : scalar_t(1));
    // The limited face, tested against the same bound. A tolerance rather than a strict
    // comparison because the arms compute the increment and the face is base + out, so a
    // result that is mathematically exactly on the bound lands a rounding either side of it.
    const scalar_t fout = base + out;
    const scalar_t tol  = scalar_t(1e-12) * (scale > scalar_t(0) ? scale : scalar_t(1));
    const bool     still_out = fout > hi + tol || fout < lo - tol;
#pragma omp atomic
    stats->n_samples += 1;
    if (over > scalar_t(0)) {
#pragma omp atomic
        stats->n_outside += 1;
    }
    if (changed) {
#pragma omp atomic
        stats->n_changed += 1;
    }
    if (still_out) {
#pragma omp atomic
        stats->n_still_out += 1;
    }
    if (rel > 0.0) {
#pragma omp critical(cvfem_limiter_stats_max)
        if (rel > stats->max_excess) stats->max_excess = rel;
    }
}

// SFEM_LIMITER_STATS=1: accumulate over a whole solve and print once at exit.
//
// At exit rather than per residual because the question is about the run, not the iterate,
// and a per-call line would bury it. Accumulating across every Newton step of every
// continuation stage is the right denominator: a case exercises the limiter if the
// reconstruction ever leaves the bound anywhere, at any point in the solve.
inline Hex8LimiterStats &cvfem_limiter_stats_object() {
    static Hex8LimiterStats s;
    return s;
}

// The bound-preserving clip (Barth-Jespersen's bound on a two-node stencil).
//
// The reconstructed face value is confined to the range the edge's own two nodes span, so
// the face can introduce no new extremum. Kept because it is the arm every other one is
// measured against, not because it is recommended: it clips at a smooth extremum too, and
// its active set flips.
template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE scalar_t cvfem_limiter_clip_inc(const scalar_t base, const scalar_t inc,
                                                                    const scalar_t lo, const scalar_t hi) {
    scalar_t f = base + inc;
    f          = f < lo ? lo : (f > hi ? hi : f);
    return f - base;
}

// Venkatakrishnan's smooth limiter, WITH his epsilon-squared deactivation term.
//
// Write D- for the increment being limited and D+ for the room the bound leaves it. Then
//
//     psi = (D+^2 + eps^2 + 2 D- D+) / (D+^2 + 2 D-^2 + D- D+ + eps^2).
//
// The eps^2 term is the whole of what this function adds over the arm that preceded it, and
// it is the limiter's OFF switch: where the local variation is small compared with eps, both
// D+^2 and D-^2 are negligible beside it, numerator and denominator both tend to eps^2, and
// psi -> 1 -- the increment passes through untouched. Where the variation is large compared
// with eps the term is negligible and the limiter behaves exactly as it did before.
//
// That switch is the point. Without it the limiter is active everywhere, including across
// the flow-aligned edges that make up most of a channel core, where u_i and u_j agree to
// round-off while the true face value legitimately differs from both in the transverse
// direction. There the two-node bound is decided by round-off, the limiter clips a
// correction that was right, and refining makes it worse -- measured at exactly h^-2 between
// two meshes, which is the bound collapsing rather than any failure of smoothness.
//
// eps2 is passed in already dimensional, in the square of the solution variable's units.
// Venkatakrishnan writes eps^2 = (K dx)^3 for a non-dimensionalised solution, which is only
// consistent when the reference velocity and length are both one; the caller restores the
// scales (see cvfem_venkata_eps2_coeff). eps2 = 0 recovers the preceding arm BIT FOR BIT --
// adding a zero is exact, and the guard below is then the only branch either can take -- so
// K = 0 is a genuine zero-severity control rather than an approximation of one.
template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE scalar_t cvfem_venkata_inc(const scalar_t base, const scalar_t inc,
                                                               const scalar_t lo, const scalar_t hi,
                                                               const scalar_t eps2) {
    // The room the increment has before it leaves [lo, hi], signed the way the increment is,
    // so D+ and D- share a sign and psi is positive.
    const scalar_t dp  = (inc >= scalar_t(0)) ? (hi - base) : (lo - base);
    const scalar_t num = dp * dp + scalar_t(2) * inc * dp + eps2;
    const scalar_t den = dp * dp + scalar_t(2) * inc * inc + inc * dp + eps2;
    // den vanishes only when dp, inc and eps2 are all zero, and then the increment is zero
    // and the scaling is irrelevant; returning it unchanged is the psi = 1 limit. With any
    // eps2 > 0 the denominator is bounded below by eps2 and this branch is unreachable.
    return (den != scalar_t(0)) ? (num / den) * inc : inc;
}

// AND THE BOUND HOLDS, WHICH IS THE CORRECTNESS CLAIM. Same case, unfrozen so the limiter is
// re-evaluated at every residual, ~2M samples over a converged solve:
//
//   arm                  outside before   altered   OUTSIDE AFTER
//   unlimited (control)      60.96%         0.00%      60.96%
//   bounded-face clip        61.17%        61.17%       0.0000%
//   Venkatakrishnan K=0      61.11%        99.29%       0.0000%
//   Darwish-Moukalled        61.13%        99.26%       0.0000%
//   Venkatakrishnan K=5      60.98%        65.84%      60.86%
//
// The unlimited row is the control and it is what makes the zeros mean anything: it alters
// nothing, so every violation survives, and "outside after" equals "outside before" to the
// sample. Against that, three arms leave EXACTLY zero.
//
// The K = 5 row is the eps^2 trade measured rather than argued. It fixes 0.12 of the 60.98
// percentage points that were outside -- eps^2 at that size does not soften the bound, it
// very nearly abolishes it. K = 5 is also the value with the best convergence (41 Newton
// steps against 303), so the two properties point opposite ways, and that is exactly why the
// default is K = 0 with the convergence taken from freezing instead.
//
// WHAT THIS DOES NOT COVER. Under SFEM_CONV_FREEZE the correction is applied as a nodal
// source and this kernel is not called during the stage, so a frozen run reports zero and the
// zero means nothing. The limiter bounded the reconstruction against the stage-OPENING state;
// whether the held correction is still bounded against the converged one is unmeasured, and
// measuring it needs the per-face increments retained rather than collapsed into the nodal
// vector. Freezing is the default, so this is an open question about the default.
//
// MEASURED: THE LIMITER IS ACTIVE, NOT DORMANT. Cavity at Re 100, 7,060 dof, counted over
// the whole solve with SFEM_LIMITER_STATS=1:
//
//   arm                  reconstructions outside the bound   increments altered
//   unlimited                      50.8%                           0.0%
//   bounded-face clip              50.9%                          50.9%
//   Venkatakrishnan K=0            50.8%                          81.5%
//   Darwish-Moukalled              51.2%                          81.5%
//
// Half of all reconstructions leave the interval their own two nodes span, so this case does
// exercise a limiter, and max_excess saturates at 1 -- some faces overshoot by as much as the
// entire increment. Two rows validate the counter rather than the scheme: the unlimited arm
// alters exactly nothing, and the clip's altered count EQUALS its overshoot count, which is
// what a bound-preserving clip must do and was not arranged.
//
// It also resolves why limited and unlimited solutions looked identical in every earlier
// comparison. They are not identical because the limiter did nothing; they are close because
// the deferred correction it acts on is a small part of the total flux.
//
// MEASURED, AND THE TERM WORKS. Backward-facing step at Re = 40, FGMRES with Vanka, two
// meshes, Newton steps to reach the target (jobs/conv_limiter.sbatch, Grace, 2026-09-24):
//
//                        7,060 dof      47,268 dof
//   first order              15              12
//   unlimited                24          DID NOT CONVERGE
//   bounded-face clip       186             304
//   Venkatakrishnan K=0     103             303
//                  K=1       78             112
//                  K=5       27              41
//                  K=30      24          DID NOT CONVERGE
//   Darwish-Moukalled        96             134
//
// Read the K column downwards and it is a dial from "limiter fully on" to "limiter fully
// off": K = 0 is the arm that preceded this term, and K = 30 reproduces the UNLIMITED arm --
// including its failure on the finer mesh. That is the deactivation mechanism doing exactly
// what it is for, and it is also the warning. Fast convergence is what an arm that has
// stopped limiting looks like, so K cannot be chosen from this table alone; the order of
// accuracy against a manufactured solution is what separates "accurate" from "off", and it
// is measured by the conv group in scripts/verify_report.sh.
//
// The h^-2 penalty that motivated the term is largely gone with it. At K = 0 refining cost
// the limiter 103 -> 303 Newton steps, a factor of 2.9; at K = 5 it costs 27 -> 41, a factor
// of 1.5. What remains is not obviously the same effect.
//
// AND FREEZING SUPERSEDES IT. SFEM_CONV_FREEZE holds the correction fixed through a
// continuation stage (see sscvfem_residual). Measured beside the arms above:
//
//                            Newton steps        u L2 rate, Re = 100
//                         7,060     47,268           8/16/32
//   K = 0, unfrozen        103        303             2.120
//   K = 5, unfrozen         27         41             2.125
//   K = 0, FROZEN           15         12             2.272
//
// Frozen at K = 0 converges like first order, keeps the bound completely intact, and fits a
// HIGHER order than either unfrozen arm. So everything eps^2 was reached for is available
// without it, and the trade below does not have to be made. K is kept because it is a real
// mechanism and the sweep is reproducible, but nothing now requires spending it.
//
// AND THE TERM IS NOT FREE, which the convergence table alone does not show. Sweeping the
// ratio r = |increment| / |room to the bound| (tests/cvfem_venkata_test.cpp prints it) gives
// the limited increment as a fraction of the room:
//
//     r        clip   K=0     eps^2=1
//     0.5     0.5000  0.5000  0.5000
//     1.0     1.0000  0.7500  0.8000
//     4.0     1.0000  0.9730  1.0526
//     20.0    1.0000  0.9988  1.0219
//
// At eps^2 = 0 the arm stays at or below 1 -- the face value never leaves the interval its
// own two nodes span, which is what bounded means. At eps^2 = 1 it is already ABOVE 1 wherever
// the reconstruction overshoots. So eps^2 does not merely switch the limiter off in smooth
// regions; it relaxes the bound everywhere, continuously and in proportion. K buys convergence
// with boundedness, and the exchange rate is this table. That is asserted in the test rather
// than only written here, because it is the property a later K increase would silently spend.
//
// Two further things this table says. The unlimited arm is NOT a safe ceiling -- it fails at
// 47,268 dof -- so it can serve as the accuracy reference and not as a fallback. And the
// stalls previously recorded for these arms (the clip at Re 14.04, the smooth form at 21.92)
// do not reproduce: both now reach Re 40. The likely cause is the block-B Rhie-Chow term
// restored in kernels/semistructured/cvfem_sshex8_ns.hpp, which is what Vanka assembles, but that has not
// been confirmed by building the prior commit and is recorded here as a hypothesis.

// The dimensional bridge for the term above: eps^2 = coeff * h^3, with h a local length.
//
// coeff = uref^2 * (K / lref)^3 carries the units, so K stays the dimensionless knob
// Venkatakrishnan describes and a run on a box of side lref at velocity uref reproduces his
// non-dimensional numbers. Computed once by the caller from quantities the kernel does not
// have, which is also what keeps the local h in the kernel: on a graded mesh a single global
// eps^2 would deactivate the limiter in the fine region and not in the coarse one.
template <typename scalar_t>
static SFEM_INLINE scalar_t cvfem_venkata_eps2_coeff(const scalar_t k, const scalar_t uref, const scalar_t lref) {
    if (k <= scalar_t(0) || lref <= scalar_t(0)) return scalar_t(0);
    const scalar_t t = k / lref;
    return uref * uref * t * t * t;
}

// Darwish and Moukalled's virtual upwind node, as the comparison arm.
//
// The limiters above bound the face value against a two-node interval. This takes the other
// standard route on an unstructured grid: rebuild the one-dimensional TVD stencil that the
// classical limiters were written for, by placing a virtual upwind node at
// phi_U = phi_C - 2 grad phi_C . d, so that
//
//     r = (phi_C - phi_U) / (phi_D - phi_C) = 2 (grad phi_C . d) / (phi_D - phi_C),
//
// and the whole NVD/TVD family applies unchanged with no extra sweep and no neighbour
// min/max. It is here because it is what the codes running this same discretisation
// actually do -- Nalu applies van Leer to the extrapolated value on the two adjacent nodes,
// not a min/max over a node's neighbours -- so an arm that loses to it has lost to the
// state of the practice rather than to a straw man.
//
// van Leer's psi(r) = (r + |r|) / (1 + |r|), and the limited increment is psi(r)/2 times
// (phi_D - phi_C). Substituting r and clearing the fractions gives
//
//     inc = (a |b| + |a| b) / (2 (|a| + |b|)),   a = 2 grad phi_C . d,  b = phi_D - phi_C,
//
// which is one division instead of three and has no removable singularity: the denominator
// vanishes only when a and b are both zero, and the numerator vanishes with it. Opposite
// signs give exactly zero, which is psi(r < 0) = 0 -- the increment is discarded where the
// reconstruction and the downwind difference disagree about the direction of the gradient.
template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE scalar_t cvfem_darwish_moukalled_inc(const scalar_t phi_c, const scalar_t phi_d,
                                                                         const scalar_t grad_dot_d) {
    const scalar_t a  = scalar_t(2) * grad_dot_d;
    const scalar_t b  = phi_d - phi_c;
    const scalar_t aa = a < scalar_t(0) ? -a : a;
    const scalar_t ab = b < scalar_t(0) ? -b : b;
    const scalar_t den = scalar_t(2) * (aa + ab);
    return (den != scalar_t(0)) ? (a * ab + aa * b) / den : scalar_t(0);
}

// ---------------------------------------------------------------------------------------------
// DIRECTIONAL DERIVATIVES OF THE THREE LIMITERS.
//
// The deferred correction is applied lagged, so for years none of these was needed: the term is
// a constant within a Newton step and drops out of the Jacobian. They exist for the EXACT
// higher-order Jacobian action, which carries the correction and therefore has to differentiate
// the limiter that shaped it.
//
// Hand-written rather than generated, and the reason is measured rather than assumed. The
// element's exact action was built symbolically first (synthesize_cvfem_hex8_ns_upwind_sympy.py,
// defcor_action_exprs): the unlimited arm takes 8 s to build and 9 s to CSE into 2197 lines,
// which is fine and is generated; Venkatakrishnan takes 1124 s to CSE into 13502 lines and
// Darwish-Moukalled 2816 s to build. Straight-line blocks of that size are the shape this tree
// has already measured at 234 s and 4.45 GB of gcc on one kernel. So the limited arms keep the
// structure the residual has -- a small function per limiter, called from the face kernel -- and
// the generated unlimited arm becomes the independent check on this code rather than a
// replacement for it.
//
// Each takes the argument's own derivative in the Krylov direction and returns the derivative of
// the limited increment. They are the derivative where one exists; at a branch boundary they
// return the value of one side, which is a subgradient choice and the same one the first-order
// upwind switch makes with sgn.

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE scalar_t cvfem_limiter_clip_inc_d(const scalar_t base, const scalar_t inc,
                                                                      const scalar_t lo, const scalar_t hi,
                                                                      const scalar_t dbase, const scalar_t dinc,
                                                                      const scalar_t dlo, const scalar_t dhi) {
    const scalar_t f = base + inc;
    // Unclipped, the limiter is the identity and so is its derivative. Clipped, the increment is
    // pinned to a bound that is itself one of the two nodal values, so what survives is that
    // bound's derivative minus the base's.
    //
    // A BOUND COUNTS AS REACHED ONLY WHEN f PASSES IT BY MORE THAN A ROUNDING OF THE DATA, and
    // that band is what makes this derivative reproducible. Unlike the value above, the three
    // branches here differ by a FINITE amount, so an ulp of movement in f across a bound changes
    // the answer by O(1) rather than by an ulp. On a uniform mesh many faces sit exactly on a
    // bound -- 1024 of 6144 on one measured here -- and f is a dot product that the scalar and
    // the lane-blocked bodies contract into FMAs differently, so the two disagreed by 3.05e-3 on
    // the Jacobian action while the residual, whose clip moves the value by an ulp, agreed to the
    // last bit.
    //
    // Inside the band the interior subgradient is returned. That is the choice continuous with
    // the unclipped branch, it is a valid subgradient where the limiter has no derivative, and
    // both bodies now return it.
    const scalar_t scale = std::fabs(hi - lo) + std::fabs(f) + std::fabs(base);
    // Eight ulp of double precision, wide enough to cover the FMA reassociation of a three-term
    // dot product and far narrower than any clipping the limiter is meant to do.
    const scalar_t tol = scale * scalar_t(8) * scalar_t(2.220446049250313e-16);
    if (f < lo - tol) return dlo - dbase;
    if (f > hi + tol) return dhi - dbase;
    return dinc;
}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE scalar_t cvfem_venkata_inc_d(const scalar_t base, const scalar_t inc,
                                                                 const scalar_t lo, const scalar_t hi,
                                                                 const scalar_t eps2,
                                                                 const scalar_t dbase, const scalar_t dinc,
                                                                 const scalar_t dlo, const scalar_t dhi) {
    const scalar_t dp  = (inc >= scalar_t(0)) ? (hi - base) : (lo - base);
    const scalar_t ddp = (inc >= scalar_t(0)) ? (dhi - dbase) : (dlo - dbase);
    const scalar_t num = dp * dp + scalar_t(2) * inc * dp + eps2;
    const scalar_t den = dp * dp + scalar_t(2) * inc * inc + inc * dp + eps2;
    // THE inc >= 0 BRANCH NEEDS A BAND, and the note that used to sit beside this function
    // claiming otherwise was wrong in one case -- the common one.
    //
    // The claim was: "at inc == 0 its psi is exactly 1 and every term carrying the branch is
    // multiplied by inc, so the discontinuity cancels". That holds when dp is nonzero on both
    // sides of the branch. It fails when the base IS the bound the branch selects, which is
    // whenever the node is the local extremum: with eps2 = 0 and dp = hi - base = 0, num is 0,
    // so psi is 0 and not 1, and the result collapses to ddp -- while the other side of the
    // branch, where dp = lo - base is nonzero, gives dinc. The derivative therefore JUMPS by
    // dinc across inc = 0, and an ulp of movement in inc picks a side.
    //
    // eps2 is what normally prevents this: with eps2 > 0, num -> eps2 and den -> eps2 as
    // inc -> 0, so psi -> 1 and the cancellation is restored. That is why K > 0 is clean. But
    // K = 0 is a legitimate limiter for the VALUE and was every caller's default, so the
    // derivative cannot rely on K.
    //
    // Measured: the packed and atomic layouts, whose nodal-gradient reconstructions sum in
    // different orders and so reach this with inc differing in the last bits, disagreed on the
    // Jacobian action by 1.4e-04 to 2.8e-04 at K = 0 across every non-power-of-two cube size,
    // at one thread as well as 72, and agreed to 7.5e-16 at K = 0.01. The residual was clean
    // throughout, which is what identifies the derivative rather than the limiter.
    //
    // Inside the band dinc is returned: it is the psi = 1 limit, it is continuous with the
    // unlimited branch, and it is a valid subgradient where the limiter has none. Same choice
    // the clip makes, for the same reason.
    const scalar_t ainc  = inc < scalar_t(0) ? -inc : inc;
    const scalar_t scale = (hi - lo < scalar_t(0) ? lo - hi : hi - lo) + ainc +
                           (base < scalar_t(0) ? -base : base);
    if (ainc <= scale * scalar_t(8) * scalar_t(2.220446049250313e-16)) return dinc;
    // Reachable only with dp, inc and eps2 all exactly zero, which the band above already
    // covers; kept so the division is guarded on its own terms.
    if (den == scalar_t(0)) return dinc;
    const scalar_t dnum = scalar_t(2) * dp * ddp + scalar_t(2) * (dinc * dp + inc * ddp);
    const scalar_t dden = scalar_t(2) * dp * ddp + scalar_t(4) * inc * dinc + (dinc * dp + inc * ddp);
    const scalar_t psi  = num / den;
    // Quotient rule once, then the product with the increment the limiter scales.
    return ((dnum - psi * dden) / den) * inc + psi * dinc;
}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE scalar_t cvfem_darwish_moukalled_inc_d(const scalar_t phi_c, const scalar_t phi_d,
                                                                           const scalar_t grad_dot_d,
                                                                           const scalar_t dphi_c, const scalar_t dphi_d,
                                                                           const scalar_t dgrad_dot_d) {
    const scalar_t a   = scalar_t(2) * grad_dot_d;
    const scalar_t b   = phi_d - phi_c;
    const scalar_t da  = scalar_t(2) * dgrad_dot_d;
    const scalar_t db  = dphi_d - dphi_c;
    const scalar_t aa  = a < scalar_t(0) ? -a : a;
    const scalar_t ab  = b < scalar_t(0) ? -b : b;
    // THE SAME DEFECT THE CLIP HAD, at a == 0 and b == 0 rather than at a bound. The absolute
    // value has no derivative at zero, and the two sides of the sign reach dnum with a finite
    // difference: at a == 0 the result differs by 2*da*b/den between them, and at b == 0 by
    // 2*a*db/den. An ulp of noise in a dot product then flips an O(1) quantity, which is how the
    // clip made the Jacobian action layout dependent.
    //
    // Inside a rounding band the MINIMUM-NORM subgradient is taken, which is zero. It is the
    // symmetric element of the subdifferential of the absolute value at zero, so neither side is
    // preferred, and it is the only choice that does not depend on where an ulp lands.
    //
    // Venkatakrishnan beside it needs no such band although it also branches, on inc >= 0: at
    // inc == 0 its psi is exactly 1 and every term carrying the branch is multiplied by inc, so
    // the discontinuity cancels and the derivative is continuous there. That was checked rather
    // than assumed.
    // THE BAND HAS TO BE RELATIVE TO THE FIELD, NOT TO THE QUANTITY IT IS TESTING. It was
    // (aa + ab) * 8 ulp, which cannot fire when aa and ab are comparable -- the common case --
    // because the tolerance then shrinks with the very numbers it is bounding. And `den == 0` is
    // an equality test on a denominator this expression divides by twice, so a small den
    // amplifies a rounding of dnum into an O(1) change, exactly as Venkatakrishnan's does above.
    //
    // Measured the same way: the packed and atomic layouts disagreed on the Jacobian action by
    // 2.80e-05 with this limiter, on Grace, at non-power-of-two sizes, at one thread, while the
    // residual agreed. Venkatakrishnan's cure is its eps^2; this limiter has none -- its
    // denominator vanishes only where its numerator does, which makes the VALUE well behaved and
    // says nothing about the derivative -- so the floor is what it gets.
    //
    // The scale is the field's: the two nodal values and the reconstructed increment. None of
    // them vanishes with a or b, which is the property the old tolerance lacked.
    const scalar_t sc   = (phi_c < scalar_t(0) ? -phi_c : phi_c) + (phi_d < scalar_t(0) ? -phi_d : phi_d) +
                          (grad_dot_d < scalar_t(0) ? -grad_dot_d : grad_dot_d);
    const scalar_t tol = sc * scalar_t(8) * scalar_t(2.220446049250313e-16);
    const scalar_t daa = aa <= tol ? scalar_t(0) : (a < scalar_t(0) ? -da : da);
    const scalar_t dab = ab <= tol ? scalar_t(0) : (b < scalar_t(0) ? -db : db);
    const scalar_t den = scalar_t(2) * (aa + ab);
    if (den <= scalar_t(2) * tol) return scalar_t(0);
    const scalar_t num  = a * ab + aa * b;
    const scalar_t dnum = da * ab + a * dab + daa * b + aa * db;
    const scalar_t dden = scalar_t(2) * (daa + dab);
    return (dnum - (num / den) * dden) / den;
}

#endif  // CVFEM_VENKATA_LIMITER_HPP
