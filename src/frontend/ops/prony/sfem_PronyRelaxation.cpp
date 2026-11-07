#include "sfem_PronyRelaxation.hpp"

#include <algorithm>
#include <cassert>
#include <limits>
#include <type_traits>
#include <vector>

#include "sfem_blas_factory.hpp"
#include "sfem_logger.hpp"
#include "sfem_prony_kernels.hpp"

#ifdef SFEM_ENABLE_CUDA
#include <cuda_runtime_api.h>

#include "sfem_prony_kernels_cuda.hpp"
#endif

namespace sfem {

    namespace {
        /// `n` values of `src`, in `es`, as real_t on the host.
        template <typename T>
        std::vector<real_t> on_host(const T *const src, const ptrdiff_t n, const ExecutionSpace es) {
            std::vector<T> copy;
            const T       *host = src;
#ifdef SFEM_ENABLE_CUDA
            if (es == EXECUTION_SPACE_DEVICE) {
                copy.resize(n);
                if (cudaMemcpy(copy.data(), src, n * sizeof(T), cudaMemcpyDeviceToHost) != cudaSuccess) {
                    SFEM_ERROR("PronyRelaxation: copying the reference stress from the device failed\n");
                }
                host = copy.data();
            }
#else
            (void)es;
#endif
            return std::vector<real_t>(host, host + n);
        }

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
          beta_(series_.n_terms(), 0),
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
                n_slots,
                n_components_,
                n_qp_,
                n_elements_,
                options_.history_storage,
                options_.execution_space,
                options_.history_elements_per_mean);
        for (int s = 0; s < n_live_; s++) grown->copy_tensor(*H_, s, s, options_.stream);
        H_ = grown;
    }

    template <typename S_t>
    const S_t *PronyRelaxation::reference() const {
        if constexpr (std::is_same_v<S_t, real_t>) {
            return reference_full_ ? reference_full_->data() : nullptr;
        } else {
            return reference_narrow_ ? reference_narrow_->data() : nullptr;
        }
    }

    template <typename S_t>
    void PronyRelaxation::take_reference(const S_t *const relaxed_stress_n) {
        const ptrdiff_t n = static_cast<ptrdiff_t>(n_components_) * n_qp_ * n_elements_;
        S_t            *R = nullptr;
        if constexpr (std::is_same_v<S_t, real_t>) {
            reference_full_ = create_buffer<real_t>(n, options_.execution_space);
            R               = reference_full_->data();
        } else {
            reference_narrow_ = create_buffer<float>(n, options_.execution_space);
            R                 = reference_narrow_->data();
        }
        if (n == 0) return;
#ifdef SFEM_ENABLE_CUDA
        if (options_.execution_space == EXECUTION_SPACE_DEVICE) {
            if (cu_prony_copy(n * sizeof(S_t), relaxed_stress_n, R, 0, nullptr, nullptr, options_.stream) != SFEM_SUCCESS) {
                SFEM_ERROR("PronyRelaxation: copying the reference stress failed on the device\n");
            }
            return;
        }
#endif
        std::copy(relaxed_stress_n, relaxed_stress_n + n, R);
    }

    template <typename S_t>
    void PronyRelaxation::update_slots(const int           first,
                                       const int           count,
                                       const real_t *const alpha,
                                       const real_t *const b,
                                       const real_t *const beta,
                                       const S_t *const    S_n,
                                       const S_t *const    S_new) {
        if (count == 0 || n_elements_ == 0) return;
        const ptrdiff_t rows  = static_cast<ptrdiff_t>(n_components_) * n_qp_;
        const S_t *const R     = reference<S_t>();
        prony::Means     means = H_->means_view();
        if (means.data) means.data = means.tensor(first, rows);
        scaling_t *const scales = H_->scales() + first * n_elements_;
        uint8_t *const   data   = static_cast<uint8_t *>(H_->raw()) + first * rows * n_elements_ * bytes_per_value(options_.history_storage);
#ifdef SFEM_ENABLE_CUDA
        if (options_.execution_space == EXECUTION_SPACE_DEVICE) {
            if (cu_prony_advance(options_.history_storage,
                                 count,
                                 rows,
                                 n_elements_,
                                 alpha,
                                 b,
                                 (int)sizeof(S_t),
                                 S_n,
                                 S_new,
                                 data,
                                 scales,
                                 means,
                                 beta,
                                 R,
                                 options_.stream) != SFEM_SUCCESS) {
                SFEM_ERROR("PronyRelaxation: updating the history failed on the device\n");
            }
            return;
        }
#endif
        dispatch_history_codec(options_.history_storage, [&](auto h_codec) {
            using HCodec = decltype(h_codec);
            prony::advance<HCodec>(count,
                                   rows,
                                   n_elements_,
                                   alpha,
                                   b,
                                   S_n,
                                   S_new,
                                   reinterpret_cast<typename HCodec::storage_t *>(data),
                                   scales,
                                   means,
                                   beta,
                                   R);
            return 0;
        });
    }

    template <typename S_t>
    void PronyRelaxation::store(const int i, const S_t *const relaxed_stress_n, const S_t *const relaxed_stress_prev) {
        const int s = n_live_++;
        slot_[i]    = s;
        term_of_slot_.resize(n_live_);
        term_of_slot_[s] = i;
        slot_beta_.resize(n_live_);
        slot_beta_[s] = beta_[i];
        H_->zero_tensor(s, options_.stream);
        const real_t c = coefficient_[i];
        if (c == 0 && beta_[i] == 0) return;
        if (c != 0 && !relaxed_stress_prev && n_elements_ > 0) {
            SFEM_ERROR("PronyRelaxation::begin_step: a term comes back, and its history needs the last step's starting "
                       "stress\n");
        }
        // H_i = c_i (S_n - S_prev), in the store's form: the update of a zero
        // store with alpha = 0 and b = c_i from S_prev to S_n.
        static const real_t zero = 0;
        update_slots(s, 1, &zero, &c, &beta_[i], c != 0 ? relaxed_stress_prev : relaxed_stress_n, relaxed_stress_n);
    }

    void PronyRelaxation::unstore(const int i) {
        const int s    = slot_[i];
        const int last = --n_live_;
        if (s != last) {
            H_->copy_tensor(*H_, last, s, options_.stream);
            term_of_slot_[s]             = term_of_slot_[last];
            slot_[term_of_slot_[s]]      = s;
            slot_beta_[s]                = slot_beta_[last];
        }
        slot_[i] = -1;
        term_of_slot_.resize(n_live_);
        slot_beta_.resize(n_live_);
    }

    void PronyRelaxation::begin_step(const real_t,
                                     const real_t        dt,
                                     const real_t *const relaxed_stress_n,
                                     const real_t *const relaxed_stress_prev) {
        begin_step_typed(dt, relaxed_stress_n, relaxed_stress_prev);
    }

#if SFEM_REAL_T_IS_FLOAT64
    void PronyRelaxation::begin_step(const real_t,
                                     const real_t       dt,
                                     const float *const relaxed_stress_n,
                                     const float *const relaxed_stress_prev) {
        begin_step_typed(dt, relaxed_stress_n, relaxed_stress_prev);
    }
#endif

    void PronyRelaxation::advance(const real_t *const relaxed_stress_n, const real_t *const relaxed_stress) {
        advance_typed(relaxed_stress_n, relaxed_stress);
    }

#if SFEM_REAL_T_IS_FLOAT64
    void PronyRelaxation::advance(const float *const relaxed_stress_n, const float *const relaxed_stress) {
        advance_typed(relaxed_stress_n, relaxed_stress);
    }
#endif

    template <typename S_t>
    void PronyRelaxation::begin_step_typed(const real_t dt, const S_t *const relaxed_stress_n, const S_t *const relaxed_stress_prev) {
        SFEM_TRACE_SCOPE("PronyRelaxation::begin_step");

        if (!(dt > 0)) {
            SFEM_ERROR("PronyRelaxation::begin_step: dt = %g must be positive\n", (double)dt);
        }

        const real_t a_T        = options_.shift->factor(temperature_);
        const real_t reduced_dt = dt / a_T;
        options_.integrator->weights(series_, reduced_dt, alpha_.data(), b_.data());

        real_t b_sum = 0;
        for (int i = 0; i < series_.n_terms(); i++) b_sum += b_[i];
        gamma_ = series_.g_inf() + b_sum;

        if (series_.n_terms() > 0 && !relaxed_stress_n && n_elements_ > 0) {
            // (A rank holding none of the block's elements has an empty stress, null on the device.)
            SFEM_ERROR("PronyRelaxation::begin_step: a series with terms needs the relaxed stress S_r^n\n");
        }

        // Below half a unit of precision, alpha_i H_i does not reach the last
        // bit of H_i: the term relaxes completely within this step.  A term
        // slower than the horizon is stored as a remainder.
        const real_t complete  = std::numeric_limits<real_t>::epsilon() / 2;
        const bool   remainder = options_.remainder_horizon > 0 && options_.history_storage != HistoryStorage::FULL;
        bool         any_beta  = false;
        int          n_stored  = n_live_;
        for (int i = 0; i < series_.n_terms(); i++) {
            relaxed_[i] = alpha_[i] <= complete;
            beta_[i]    = remainder && !relaxed_[i] && series_.tau(i) * a_T > options_.remainder_horizon ? series_.g(i) : real_t(0);
            any_beta    = any_beta || beta_[i] != 0;
            if (!relaxed_[i] && slot_[i] < 0) n_stored++;
        }
        if (any_beta && !reference<S_t>()) take_reference(relaxed_stress_n);
        reserve(n_stored);
        for (int i = 0; i < series_.n_terms(); i++) {
            if (!relaxed_[i] && slot_[i] < 0) store(i, relaxed_stress_n, relaxed_stress_prev);
        }

        // A store whose beta changed (the shift moved a term across the horizon):
        // G += (beta_old - beta_new)(S_n - R), the update with alpha = 1, b = 0 and
        // beta = beta_old - beta_new from S_n to R.
        for (int s = 0; s < n_live_; s++) {
            const real_t target = beta_[term_of_slot_[s]];
            if (target == slot_beta_[s]) continue;
            static const real_t one = 1, zero = 0;
            const real_t        delta = slot_beta_[s] - target;
            update_slots(s, 1, &one, &zero, &delta, relaxed_stress_n, reference<S_t>());
            slot_beta_[s] = target;
        }

        slot_alpha_.resize(n_live_);
        slot_b_.resize(n_live_);
        for (int s = 0; s < n_live_; s++) {
            const int i    = term_of_slot_[s];
            slot_alpha_[s] = relaxed_[i] ? real_t(0) : alpha_[i];
            slot_b_[s]     = b_[i];
        }

        if (series_.n_terms() == 0 || n_elements_ == 0) {
            // Nothing relaxes, or nothing is here: the frozen history stays the zero it was made.
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
                                        (int)sizeof(S_t),
                                        relaxed_stress_n,
                                        history_->data(),
                                        H_ ? H_->means_view() : prony::Means{},
                                        slot_beta_.data(),
                                        reference<S_t>(),
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
                                              history_->data(),
                                              H_ ? H_->means_view() : prony::Means{},
                                              slot_beta_.data(),
                                              reference<S_t>());
                return 0;
            });
        }

        step_open_ = true;
    }

    template <typename S_t>
    void PronyRelaxation::advance_typed(const S_t *const relaxed_stress_n, const S_t *const relaxed_stress) {
        SFEM_TRACE_SCOPE("PronyRelaxation::advance");

        if (!step_open_) {
            SFEM_ERROR("PronyRelaxation::advance: no step is open; call begin_step first\n");
        }

        if (series_.n_terms() > 0 && n_elements_ > 0 && (!relaxed_stress_n || !relaxed_stress)) {
            SFEM_ERROR("PronyRelaxation::advance: a series with terms needs both relaxed stresses\n");
        }

        update_slots(0, n_live_, slot_alpha_.data(), slot_b_.data(), slot_beta_.data(), relaxed_stress_n, relaxed_stress);

        // A term that relaxed completely is now b_i (S_r^{n+1} - S_r^n), whatever it held.
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
        slot_beta_.clear();
        // The history is zero again: the next step's stress is the new R.
        reference_full_.reset();
        reference_narrow_.reset();
        zero_history();
        gamma_     = 1;
        step_open_ = false;
    }

    void PronyRelaxation::decode_history(const int           i,
                                         real_t *const       values,
                                         const real_t *const relaxed_stress_n,
                                         const real_t *const relaxed_stress_prev) const {
        const ptrdiff_t n = static_cast<ptrdiff_t>(n_components_) * n_qp_ * n_elements_;
        if (slot_[i] >= 0) {
            H_->decode(slot_[i], values);
            const real_t beta = slot_beta_[slot_[i]];
            if (beta == 0) return;
            // H = G + beta (S_n - R), R brought to the host.
            const std::vector<real_t> R = reference_full_ ? on_host(reference_full_->data(), n, options_.execution_space)
                                                          : on_host(reference_narrow_->data(), n, options_.execution_space);
            for (ptrdiff_t k = 0; k < n; k++) values[k] += beta * (relaxed_stress_n[k] - R[k]);
        } else if (coefficient_[i] != 0) {
            for (ptrdiff_t k = 0; k < n; k++) values[k] = coefficient_[i] * (relaxed_stress_n[k] - relaxed_stress_prev[k]);
        } else {
            std::fill(values, values + n, real_t(0));
        }
    }

    std::size_t PronyRelaxation::bytes() const { return history_bytes() + reference_bytes() + frozen_bytes(); }

    std::size_t PronyRelaxation::history_bytes() const { return H_ ? H_->bytes() : 0; }

    std::size_t PronyRelaxation::reference_bytes() const {
        return reference_full_ ? reference_full_->size() * sizeof(real_t)
                               : reference_narrow_ ? reference_narrow_->size() * sizeof(float) : 0;
    }

    std::size_t PronyRelaxation::frozen_bytes() const { return history_->size() * sizeof(real_t); }

}  // namespace sfem
