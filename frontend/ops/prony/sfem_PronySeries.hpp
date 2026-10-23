#pragma once

#include <vector>

#include "sfem_base.hpp"

namespace sfem {

    /// A relaxation spectrum: the Prony series G(t)/G_0 = g_inf + sum_i g_i exp(-t/tau_i).
    ///
    /// Pure data, validated once at construction.  Nothing here knows about a
    /// material, a mesh or a step; the weights a step uses come from a
    /// `RelaxationIntegrator`, and the time-temperature shift from a
    /// `TimeTemperatureShift`, so the spectrum stays the thing a fit produces.
    class PronySeries {
    public:
        PronySeries() = default;
        PronySeries(std::vector<real_t> g, std::vector<real_t> tau);

        int    n_terms() const { return static_cast<int>(g_.size()); }
        real_t g(const int i) const { return g_[i]; }
        real_t tau(const int i) const { return tau_[i]; }

        /// The equilibrium fraction, 1 - sum_i g_i.
        real_t g_inf() const { return g_inf_; }

    private:
        std::vector<real_t> g_;
        std::vector<real_t> tau_;
        real_t              g_inf_{1};
    };

}  // namespace sfem
