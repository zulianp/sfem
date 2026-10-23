#include "sfem_RelaxationIntegrator.hpp"

#include <cassert>
#include <cmath>

#include "sfem_logger.hpp"

namespace sfem {

    void LinearInterpolationIntegrator::weights(const PronySeries &series,
                                                const real_t       reduced_dt,
                                                real_t *const      alpha,
                                                real_t *const      b) const {
        for (int i = 0; i < series.n_terms(); i++) {
            const real_t x = reduced_dt / series.tau(i);
            alpha[i]       = std::exp(-x);
            // (1 - exp(-x)) / x through expm1, which keeps it accurate for the
            // small steps where the difference cancels, and its limit g_i at
            // x = 0.
            b[i] = x > 0 ? series.g(i) * (-std::expm1(-x) / x) : series.g(i);
        }
    }

    void MidpointIntegrator::weights(const PronySeries &series,
                                     const real_t       reduced_dt,
                                     real_t *const      alpha,
                                     real_t *const      b) const {
        for (int i = 0; i < series.n_terms(); i++) {
            const real_t x = reduced_dt / series.tau(i);
            alpha[i]       = std::exp(-x);
            b[i]           = series.g(i) * std::exp(-x / 2);
        }
    }

    std::shared_ptr<RelaxationIntegrator> create_relaxation_integrator(const std::string &name) {
        if (name == "linear") return std::make_shared<LinearInterpolationIntegrator>();
        if (name == "midpoint") return std::make_shared<MidpointIntegrator>();
        SFEM_ERROR("Unknown relaxation integrator \"%s\" (expected linear or midpoint)\n", name.c_str());
        return nullptr;
    }

}  // namespace sfem
