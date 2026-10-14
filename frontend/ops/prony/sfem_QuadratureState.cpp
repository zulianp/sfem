#include "sfem_QuadratureState.hpp"

#include <algorithm>
#include <cstring>

#include "sfem_prony_kernels.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_prony_kernels_cuda.hpp"
#endif

namespace sfem {

    QuadratureState::QuadratureState(const int            n_tensors,
                                     const int            n_components,
                                     const int            n_qp,
                                     const ptrdiff_t      n_elements,
                                     const HistoryStorage storage,
                                     const ExecutionSpace es)
        : n_tensors_(n_tensors),
          n_components_(n_components),
          n_qp_(n_qp),
          n_elements_(n_elements),
          storage_(storage),
          es_(es) {
        if (n_tensors < 0 || n_components <= 0 || n_qp <= 0 || n_elements < 0) {
            SFEM_ERROR("QuadratureState: invalid shape (%d tensors, %d components, %d points, %ld elements)\n",
                       n_tensors,
                       n_components,
                       n_qp,
                       (long)n_elements);
        }

        const ptrdiff_t n_values = n_tensors * values_per_tensor() * n_elements;
        data_   = create_buffer<uint8_t>(n_values * bytes_per_value(storage), es);
        scales_ = create_buffer<scaling_t>(n_tensors * n_elements, es);
        zero();
    }

    std::size_t QuadratureState::bytes() const {
        return data_->size() * sizeof(uint8_t) + scales_->size() * sizeof(scaling_t);
    }

    void QuadratureState::zero(void *const stream) {
#ifndef SFEM_ENABLE_CUDA
        (void)stream;
#else
        if (es_ == EXECUTION_SPACE_DEVICE) {
            if (cu_prony_zero(data_->size(), data_->data(), scales_->size(), scales_->data(), stream) != SFEM_SUCCESS) {
                SFEM_ERROR("QuadratureState::zero failed on the device\n");
            }
            return;
        }
#endif
        // Zero is the all-zero bit pattern in every storage format.
        std::memset(data_->data(), 0, data_->size());
        scaling_t *const s = scales_->data();
        std::fill(s, s + scales_->size(), scaling_t(1));
    }

    void QuadratureState::encode(const int t, const real_t *const values) {
        if (es_ != EXECUTION_SPACE_HOST) SFEM_ERROR("QuadratureState::encode is a host convenience\n");
        dispatch_history_codec(storage_, [&](auto codec) {
            using Codec = decltype(codec);
            prony::encode<Codec>(values_per_tensor(),
                                 n_elements_,
                                 values,
                                 &data<Codec>()[t * values_per_tensor() * n_elements_],
                                 &scales()[t * n_elements_]);
            return 0;
        });
    }

    void QuadratureState::decode(const int t, real_t *const values) const {
        if (es_ != EXECUTION_SPACE_HOST) SFEM_ERROR("QuadratureState::decode is a host convenience\n");
        dispatch_history_codec(storage_, [&](auto codec) {
            using Codec = decltype(codec);
            prony::decode<Codec>(values_per_tensor(),
                                 n_elements_,
                                 &data<Codec>()[t * values_per_tensor() * n_elements_],
                                 &scales()[t * n_elements_],
                                 values);
            return 0;
        });
    }

}  // namespace sfem
