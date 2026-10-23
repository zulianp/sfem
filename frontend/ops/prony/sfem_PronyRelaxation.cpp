#include "sfem_PronyRelaxation.hpp"

#include <algorithm>
#include <cassert>
#include <limits>

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
          slot_(series_.n_terms(), -1),
          coefficient_(series_.n_terms(), 0),
          relaxed_(series_.n_terms(), 0),
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

    void PronyRelaxation::reserve(const int n_slots) {
        if (H_ && H_->n_tensors() >= n_slots) return;
        // Grown to exactly what the step stores: a term comes back only when
        // the step shrinks, so the store holds the most terms any step kept.
        auto grown = std::make_shared<QuadratureState>(
                n_slots, n_components_, n_qp_, n_elements_, options_.history_storage, options_.execution_space);
        for (int s = 0; s < n_live_; s++) grown->copy_tensor(*H_, s, s, options_.stream);
        H_ = grown;
    }

    void PronyRelaxation::store(const int i) {
        const int s = n_live_++;
        slot_[i]    = s;
        term_of_slot_.resize(n_live_);
        term_of_slot_[s] = i;
        if (coefficient_[i] == 0 || !D_) {
            H_->zero_tensor(s, options_.stream);
        } else {
            H_->copy_tensor(*D_, 0, s, options_.stream);
            H_->scale_tensor(s, coefficient_[i], options_.stream);
        }
    }

    void PronyRelaxation::unstore(const int i) {
        const int s    = slot_[i];
        const int last = --n_live_;
        if (s != last) {
            H_->copy_tensor(*H_, last, s, options_.stream);
            term_of_slot_[s]             = term_of_slot_[last];
            slot_[term_of_slot_[s]]      = s;
        }
        slot_[i] = -1;
        term_of_slot_.resize(n_live_);
    }

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

        // Below half a unit of precision, alpha_i H_i does not reach the last
        // bit of H_i: the term relaxes completely within this step.
        const real_t complete = std::numeric_limits<real_t>::epsilon() / 2;
        int n_stored = n_live_;
        for (int i = 0; i < series_.n_terms(); i++) {
            relaxed_[i] = alpha_[i] <= complete;
            if (!relaxed_[i] && slot_[i] < 0) n_stored++;
        }
        reserve(n_stored);
        for (int i = 0; i < series_.n_terms(); i++) {
            if (!relaxed_[i] && slot_[i] < 0) store(i);
        }

        slot_alpha_.resize(n_live_);
        slot_b_.resize(n_live_);
        for (int s = 0; s < n_live_; s++) {
            const int i    = term_of_slot_[s];
            slot_alpha_[s] = relaxed_[i] ? real_t(0) : alpha_[i];
            slot_b_[s]     = b_[i];
        }

        if (series_.n_terms() == 0) {
            // Nothing relaxes: the frozen history stays the zero it was made.
        } else if (!relaxed_stress_n) {
            SFEM_ERROR("PronyRelaxation::begin_step: a series with terms needs the relaxed stress S_r^n\n");
#ifdef SFEM_ENABLE_CUDA
        } else if (options_.execution_space == EXECUTION_SPACE_DEVICE) {
            if (cu_prony_frozen_history(options_.history_storage,
                                        n_live_,
                                        n_components_ * n_qp_,
                                        n_elements_,
                                        slot_alpha_.data(),
                                        b_sum,
                                        H_ ? H_->raw() : nullptr,
                                        H_ ? H_->scales() : nullptr,
                                        relaxed_stress_n,
                                        history_->data(),
                                        options_.stream) != SFEM_SUCCESS) {
                SFEM_ERROR("PronyRelaxation::begin_step failed on the device\n");
            }
#endif
        } else {
            dispatch_history_codec(options_.history_storage, [&](auto h_codec) {
                using HCodec = decltype(h_codec);
                prony::frozen_history<HCodec>(n_live_,
                                              n_components_ * n_qp_,
                                              n_elements_,
                                              slot_alpha_.data(),
                                              b_sum,
                                              H_ ? H_->data<HCodec>() : nullptr,
                                              H_ ? H_->scales() : nullptr,
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
        }

        bool any_relaxed = false;
        for (int i = 0; i < series_.n_terms(); i++) any_relaxed = any_relaxed || relaxed_[i];

        // The stored terms, and D = S_r^{n+1} - S_r^n, which is the update of
        // a store with alpha = 0 and b = 1.
        static const real_t zero = 0, one = 1;
        if (any_relaxed && !D_) {
            D_ = std::make_shared<QuadratureState>(
                    1, n_components_, n_qp_, n_elements_, options_.history_storage, options_.execution_space);
        }

        const ptrdiff_t rows = n_components_ * n_qp_;
#ifdef SFEM_ENABLE_CUDA
        if (options_.execution_space == EXECUTION_SPACE_DEVICE) {
            if ((n_live_ > 0 && cu_prony_advance(options_.history_storage,
                                                 n_live_,
                                                 rows,
                                                 n_elements_,
                                                 slot_alpha_.data(),
                                                 slot_b_.data(),
                                                 relaxed_stress_n,
                                                 relaxed_stress,
                                                 H_->raw(),
                                                 H_->scales(),
                                                 options_.stream) != SFEM_SUCCESS) ||
                (any_relaxed && cu_prony_advance(options_.history_storage,
                                                 1,
                                                 rows,
                                                 n_elements_,
                                                 &zero,
                                                 &one,
                                                 relaxed_stress_n,
                                                 relaxed_stress,
                                                 D_->raw(),
                                                 D_->scales(),
                                                 options_.stream) != SFEM_SUCCESS)) {
                SFEM_ERROR("PronyRelaxation::advance failed on the device\n");
            }
        } else
#endif
        {
            dispatch_history_codec(options_.history_storage, [&](auto h_codec) {
                using HCodec = decltype(h_codec);
                if (n_live_ > 0) {
                    prony::advance<HCodec>(n_live_,
                                           rows,
                                           n_elements_,
                                           slot_alpha_.data(),
                                           slot_b_.data(),
                                           relaxed_stress_n,
                                           relaxed_stress,
                                           H_->data<HCodec>(),
                                           H_->scales());
                }
                if (any_relaxed) {
                    prony::advance<HCodec>(
                            1, rows, n_elements_, &zero, &one, relaxed_stress_n, relaxed_stress, D_->data<HCodec>(), D_->scales());
                }
                return 0;
            });
        }

        // A term that relaxed completely is now b_i D, whatever it held.
        for (int i = 0; i < series_.n_terms(); i++) {
            if (!relaxed_[i]) continue;
            coefficient_[i] = b_[i];
            if (slot_[i] >= 0) unstore(i);
        }

        step_open_ = false;
    }

    void PronyRelaxation::reset() {
        n_live_ = 0;
        std::fill(slot_.begin(), slot_.end(), -1);
        std::fill(coefficient_.begin(), coefficient_.end(), real_t(0));
        term_of_slot_.clear();
        zero_history();
        gamma_     = 1;
        step_open_ = false;
    }

    void PronyRelaxation::decode_history(const int i, real_t *const values) const {
        const ptrdiff_t n = static_cast<ptrdiff_t>(n_components_) * n_qp_ * n_elements_;
        if (slot_[i] >= 0) {
            H_->decode(slot_[i], values);
        } else if (coefficient_[i] != 0 && D_) {
            D_->decode(0, values);
            for (ptrdiff_t k = 0; k < n; k++) values[k] *= coefficient_[i];
        } else {
            std::fill(values, values + n, real_t(0));
        }
    }

    std::size_t PronyRelaxation::bytes() const {
        return (H_ ? H_->bytes() : 0) + (D_ ? D_->bytes() : 0) + history_->size() * sizeof(real_t);
    }

}  // namespace sfem
