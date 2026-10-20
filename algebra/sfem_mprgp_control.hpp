#ifndef SFEM_MPRGP_CONTROL_HPP
#define SFEM_MPRGP_CONTROL_HPP

#include <cmath>

#include "sfem_base.hpp"

namespace sfem {

    /// What a step along the current direction does, decided from the curvature pass.
    enum MPRGPMove { MPRGP_MOVE_STEP = 0, MPRGP_MOVE_EXPAND = 1, MPRGP_MOVE_UNBOUNDED = 2 };

    /// Which decision the curvature pass makes once its sums are final.
    enum MPRGPDecision { MPRGP_DECIDE_NONE = 0, MPRGP_DECIDE_CG = 1, MPRGP_DECIDE_PROPORTIONING = 2 };

    /// The scalars of one MPRGP iteration, in the execution space of its vectors: the reductions of
    /// the curvature and the step passes and the decisions taken from them. The passes read the
    /// step length and the next direction's coefficient from here, so on one device the iteration
    /// returns to the host once, to read the record.
    template <typename T>
    struct MPRGPControl {
        // The curvature pass along d: d.Ad, d.d, g.d, the longest feasible step.
        T dAd, dd, gd, alpha_feas;
        // The step pass at the iterate it reaches: Ad.gf, |gf|^2, |gc|^2, |w (gf + gc)|^2.
        T Adgf, gf2, gc2, gp2;
        // The step along d (MPRGP_MOVE_STEP), or the expansion's step to the bound.
        T alpha;
        // The CG step along d, infty on a direction without curvature.
        T alpha_cg;
        // The next direction gf - beta d after a CG step.
        T beta;
        // MPRGPMove, held as a T so that the record is an array of T.
        T move;
    };

    template <typename T>
    constexpr int mprgp_control_size() {
        return int(sizeof(MPRGPControl<T>) / sizeof(T));
    }

    /// The constants the decisions take.
    template <typename T>
    struct MPRGPParams {
        T max_eig;    // of the operator (bound or estimate)
        T alpha_bar;  // the expansion's gradient step
        T infty;      // stands for no bound
        T epsilon;    // the machine epsilon of T
    };

    /// Whether direction d (d.d = dd, d.Ad = dAd) sees no curvature to rounding: on a singular or
    /// indefinite operator (a body free to move) the energy does not rise along it, and only a
    /// bound ends the step.
    template <typename T>
    SFEM_DEVICE_INLINE bool mprgp_no_curvature(const T dAd, const T dd, const MPRGPParams<T>& p) {
        return dd > 0 && dAd <= 64 * p.epsilon * p.max_eig * dd;
    }

    /// Whether no bound stops a step of alpha_feas along d (d.d = dd): none limits it (infty), or
    /// only the bounds that stand for none (infty) do.
    template <typename T>
    SFEM_DEVICE_INLINE bool mprgp_unbounded(const T alpha_feas, const T dd, const MPRGPParams<T>& p) {
        using std::sqrt;
        return alpha_feas >= p.infty || alpha_feas * sqrt(dd) >= p.infty / 2;
    }

    /// The step along a direction without curvature: to the first bound, but no longer than the
    /// expansion step's alpha_bar. The energy falls along all of it; a bound far away (a free
    /// body's rigid motion barely touching one) would otherwise send the iterate as far.
    template <typename T>
    SFEM_DEVICE_INLINE T mprgp_flat_step(const T alpha_feas, const MPRGPParams<T>& p) {
        return (p.alpha_bar < alpha_feas) ? p.alpha_bar : alpha_feas;
    }

    /// Along the conjugate direction: the CG step if it stays feasible, else the expansion to the
    /// bound; unbounded where the direction has no curvature and no bound stops it.
    template <typename T>
    SFEM_DEVICE_INLINE void mprgp_decide_cg(MPRGPControl<T>& c, const MPRGPParams<T>& p) {
        const bool flat = mprgp_no_curvature(c.dAd, c.dd, p);
        if (flat && mprgp_unbounded(c.alpha_feas, c.dd, p)) {
            c.move  = T(MPRGP_MOVE_UNBOUNDED);
            c.alpha = 0;
            return;
        }
        c.alpha_cg = flat ? p.infty : c.gd / c.dAd;
        if (c.alpha_cg <= c.alpha_feas) {
            c.move  = T(MPRGP_MOVE_STEP);
            c.alpha = c.alpha_cg;
        } else {
            c.move  = T(MPRGP_MOVE_EXPAND);
            c.alpha = flat ? mprgp_flat_step(c.alpha_feas, p) : c.alpha_feas;
        }
    }

    /// Along the chopped gradient: the CG step, or to the first bound where it has no curvature;
    /// unbounded where no bound stops it either.
    template <typename T>
    SFEM_DEVICE_INLINE void mprgp_decide_proportioning(MPRGPControl<T>& c, const MPRGPParams<T>& p) {
        if (mprgp_no_curvature(c.dAd, c.dd, p)) {
            if (mprgp_unbounded(c.alpha_feas, c.dd, p)) {
                c.move  = T(MPRGP_MOVE_UNBOUNDED);
                c.alpha = 0;
                return;
            }
            c.alpha = mprgp_flat_step(c.alpha_feas, p);
        } else {
            c.alpha = c.gd / c.dAd;
        }
        c.alpha_cg = c.alpha;
        c.move     = T(MPRGP_MOVE_STEP);
    }

    template <typename T>
    SFEM_DEVICE_INLINE void mprgp_decide(MPRGPControl<T>& c, const MPRGPDecision decision, const MPRGPParams<T>& p) {
        if (decision == MPRGP_DECIDE_CG) {
            mprgp_decide_cg(c, p);
        } else if (decision == MPRGP_DECIDE_PROPORTIONING) {
            mprgp_decide_proportioning(c, p);
        }
    }

    /// After a CG step: the coefficient of the next direction gf - beta d.
    template <typename T>
    SFEM_DEVICE_INLINE void mprgp_after_cg_step(MPRGPControl<T>& c) {
        c.beta = c.Adgf / c.dAd;
    }

}  // namespace sfem

#endif  // SFEM_MPRGP_CONTROL_HPP
