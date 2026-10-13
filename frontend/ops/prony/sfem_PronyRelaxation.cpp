#include "sfem_PronyRelaxation.hpp"

#include <cassert>

#include "sfem_API.hpp"
#include "sfem_logger.hpp"
#include "sfem_prony_kernels.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_prony_kernels_cuda.hpp"
#endif

namespace sfem {

    namespace {
        PronyRelaxation::Options complete(PronyRelaxation::Options options) {
            if (!options.integrator) options.integrator = std::make_shared<LinearInterpolationIntegrator>();
            if (!options.shift) options.shift = std::make_shared<NoShift>();
            return options;
        }
    }  // namespace

    PronyRelaxation::PronyRelaxation(PronySeries series, const int n_components, const int n_qp, const ptrdiff_t n_elements)
        : PronyRelaxation(std::move(series), n_components, n_qp, n_elements, Options{}) {}

    PronyRelaxation::PronyRelaxation(PronySeries     series,
                                     const int       n_components,
                                     const int       n_qp,
                                     const ptrdiff_t n_elements,
                                     Options         options)
        : series_(std::move(series)),
          options_(complete(std::move(options))),
          n_components_(n_components),
          n_qp_(n_qp),
          n_elements_(n_elements),
          temperature_(options_.temperature),
          H_(series_.n_terms(), n_components, n_qp, n_elements, options_.history_storage, options_.execution_space),
          history_(create_buffer<real_t>(static_cast<ptrdiff_t>(n_components) * n_qp * n_elements, options_.execution_space)),
          alpha_(series_.n_terms(), 0),
          b_(series_.n_terms(), 0) {
#ifdef SFEM_ENABLE_CUDA
        if (options_.execution_space == EXECUTION_SPACE_DEVICE && series_.n_terms() > cu_prony_max_terms()) {
            SFEM_ERROR("PronyRelaxation: %d terms, the device loops take at most %d\n", series_.n_terms(), cu_prony_max_terms());
        }
#endif
        zero_history();
    }

    void PronyRelaxation::zero_history() {
        sfem::blas<real_t>(options_.execution_space)->zeros(history_->size(), history_->data());
    }

    void PronyRelaxation::set_temperature(const real_t temperature) { temperature_ = temperature; }

    void PronyRelaxation::begin_step(const real_t, const real_t dt, const real_t *const relaxed_stress_n) {
        SFEM_TRACE_SCOPE("PronyRelaxation::begin_step");

        if (!(dt > 0)) {
            SFEM_ERROR("PronyRelaxation::begin_step: dt = %g must be positive\n", (double)dt);
        }

        const real_t reduced_dt = dt / options_.shift->factor(temperature_);
        options_.integrator->weights(series_, reduced_dt, alpha_.data(), b_.data());

        real_t b_sum = 0;
        for (int i = 0; i < series_.n_terms(); i++) b_sum += b_[i];
        gamma_ = series_.g_inf() + b_sum;

        if (series_.n_terms() == 0) {
            // Nothing relaxes: the frozen history stays the zero it was made.
        } else if (!relaxed_stress_n) {
            SFEM_ERROR("PronyRelaxation::begin_step: a series with terms needs the relaxed stress S_r^n\n");
#ifdef SFEM_ENABLE_CUDA
        } else if (options_.execution_space == EXECUTION_SPACE_DEVICE) {
            if (cu_prony_frozen_history(H_.storage(),
                                        series_.n_terms(),
                                        H_.values_per_tensor(),
                                        n_elements_,
                                        alpha_.data(),
                                        b_sum,
                                        H_.raw(),
                                        H_.scales(),
                                        relaxed_stress_n,
                                        history_->data(),
                                        options_.stream) != SFEM_SUCCESS) {
                SFEM_ERROR("PronyRelaxation::begin_step failed on the device\n");
            }
#endif
        } else {
            dispatch_history_codec(H_.storage(), [&](auto h_codec) {
                using HCodec = decltype(h_codec);
                prony::frozen_history<HCodec>(series_.n_terms(),
                                              H_.values_per_tensor(),
                                              n_elements_,
                                              alpha_.data(),
                                              b_sum,
                                              H_.data<HCodec>(),
                                              H_.scales(),
                                              relaxed_stress_n,
                                              history_->data());
                return 0;
            });
        }

        step_open_ = true;
    }

    void PronyRelaxation::advance(const real_t *const relaxed_stress_n, const real_t *const relaxed_stress) {
        SFEM_TRACE_SCOPE("PronyRelaxation::advance");

        if (!step_open_) {
            SFEM_ERROR("PronyRelaxation::advance: no step is open; call begin_step first\n");
        }

        if (series_.n_terms() > 0 && (!relaxed_stress_n || !relaxed_stress)) {
            SFEM_ERROR("PronyRelaxation::advance: a series with terms needs both relaxed stresses\n");
#ifdef SFEM_ENABLE_CUDA
        } else if (series_.n_terms() > 0 && options_.execution_space == EXECUTION_SPACE_DEVICE) {
            if (cu_prony_advance(H_.storage(),
                                 series_.n_terms(),
                                 H_.values_per_tensor(),
                                 n_elements_,
                                 alpha_.data(),
                                 b_.data(),
                                 relaxed_stress_n,
                                 relaxed_stress,
                                 H_.raw(),
                                 H_.scales(),
                                 options_.stream) != SFEM_SUCCESS) {
                SFEM_ERROR("PronyRelaxation::advance failed on the device\n");
            }
#endif
        } else if (series_.n_terms() > 0) {
            dispatch_history_codec(H_.storage(), [&](auto h_codec) {
                using HCodec = decltype(h_codec);
                prony::advance<HCodec>(series_.n_terms(),
                                       H_.values_per_tensor(),
                                       n_elements_,
                                       alpha_.data(),
                                       b_.data(),
                                       relaxed_stress_n,
                                       relaxed_stress,
                                       H_.data<HCodec>(),
                                       H_.scales());
                return 0;
            });
        }

        step_open_ = false;
    }

    void PronyRelaxation::reset() {
        H_.zero(options_.stream);
        zero_history();
        gamma_     = 1;
        step_open_ = false;
    }

    std::size_t PronyRelaxation::bytes() const {
        return H_.bytes() + history_->size() * sizeof(real_t);
    }

}  // namespace sfem
