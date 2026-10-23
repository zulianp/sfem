#pragma once

#include <memory>
#include <string>

#include "sfem_PronySeries.hpp"

namespace sfem {

    /// How one step advances the internal variables of a Prony series.
    ///
    /// Every one-step scheme in use for Simo-type viscoelasticity has the same
    /// shape,
    ///
    ///     H_i <- alpha_i H_i + b_i (S^{n+1} - S^n),   alpha_i = exp(-x_i),
    ///
    /// with x_i the reduced step over tau_i, and they differ only in the
    /// weight b_i on the stress increment.  So a scheme is the pair
    /// (alpha_i, b_i) and nothing else: the algorithmic stress of the step is
    /// gamma S^{n+1} + sum_i (alpha_i H_i - b_i S^n) with
    /// gamma = g_inf + sum_i b_i, and its tangent is gamma times the relaxed
    /// tangent plus the frozen history -- for every scheme alike.
    class RelaxationIntegrator {
    public:
        virtual ~RelaxationIntegrator() = default;

        /// Fill alpha[i] and b[i] for a step of reduced length `reduced_dt`.
        virtual void weights(const PronySeries &series, const real_t reduced_dt, real_t *const alpha, real_t *const b) const = 0;

        virtual const char *name() const = 0;
    };

    /// b_i = g_i (1 - exp(-x_i)) / x_i: the hereditary integral evaluated
    /// exactly for a stress that varies linearly over the step (Taylor, Pister
    /// & Goudreau 1970).  It is the recursion of SFEM's hand-written
    /// `MooneyRivlinVisco` and the one `prony_relaxation_3d` validates.
    class LinearInterpolationIntegrator final : public RelaxationIntegrator {
    public:
        void        weights(const PronySeries &series, const real_t reduced_dt, real_t *const alpha, real_t *const b) const override;
        const char *name() const override { return "linear"; }
    };

    /// b_i = g_i exp(-x_i / 2): the midpoint rule on the convolution kernel
    /// (Simo 1987; Holzapfel 2000, section 6.10; Liu, Latorre & Marsden 2021,
    /// eq. 3.15).
    class MidpointIntegrator final : public RelaxationIntegrator {
    public:
        void        weights(const PronySeries &series, const real_t reduced_dt, real_t *const alpha, real_t *const b) const override;
        const char *name() const override { return "midpoint"; }
    };

    /// "linear" or "midpoint".
    std::shared_ptr<RelaxationIntegrator> create_relaxation_integrator(const std::string &name);

}  // namespace sfem
