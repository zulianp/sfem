#ifndef CVFEM_VENKATA_LIMITER_HPP
#define CVFEM_VENKATA_LIMITER_HPP

#include <cmath>

#ifndef SFEM_INLINE
#define SFEM_INLINE inline
#endif
#ifndef SFEM_HOST_DEVICE
#define SFEM_HOST_DEVICE
#endif

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
// restored in src/ss/cvfem_sshex8_ns.hpp, which is what Vanka assembles, but that has not
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

#endif  // CVFEM_VENKATA_LIMITER_HPP
