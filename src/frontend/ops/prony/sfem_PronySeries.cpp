#include "sfem_PronySeries.hpp"

#include <cassert>
#include <cmath>

#include "sfem_logger.hpp"

namespace sfem {

    PronySeries::PronySeries(std::vector<real_t> g, std::vector<real_t> tau) : g_(std::move(g)), tau_(std::move(tau)) {
        if (g_.size() != tau_.size()) {
            SFEM_ERROR("PronySeries: %d weights but %d relaxation times\n", (int)g_.size(), (int)tau_.size());
        }

        real_t sum_g = 0;
        for (int i = 0; i < n_terms(); i++) {
            if (!(g_[i] >= 0) || !std::isfinite(g_[i])) {
                SFEM_ERROR("PronySeries: g[%d] = %g must be finite and non-negative\n", i, (double)g_[i]);
            }

            if (!(tau_[i] > 0) || !std::isfinite(tau_[i])) {
                SFEM_ERROR("PronySeries: tau[%d] = %g must be finite and positive\n", i, (double)tau_[i]);
            }

            sum_g += g_[i];
        }

        // g_inf = 0 is a fluid; the solid this material describes needs an
        // equilibrium modulus.
        if (!(sum_g < 1)) {
            SFEM_ERROR("PronySeries: sum of g = %g must be below 1\n", (double)sum_g);
        }

        g_inf_ = 1 - sum_g;
    }

}  // namespace sfem
