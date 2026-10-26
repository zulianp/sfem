#include "sfem_QuadratureState.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

#include "sfem_prony_kernels.hpp"

#ifdef SFEM_ENABLE_CUDA
#include <cuda_runtime_api.h>

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

    void QuadratureState::zero_tensor(const int t, void *const stream) {
        const ptrdiff_t tensor_bytes = values_per_tensor() * n_elements_ * bytes_per_value(storage_);
        uint8_t *const   data        = data_->data() + t * tensor_bytes;
        scaling_t *const s           = scales_->data() + t * n_elements_;
#ifndef SFEM_ENABLE_CUDA
        (void)stream;
#else
        if (es_ == EXECUTION_SPACE_DEVICE) {
            if (cu_prony_zero(tensor_bytes, data, n_elements_, s, stream) != SFEM_SUCCESS) {
                SFEM_ERROR("QuadratureState::zero_tensor failed on the device\n");
            }
            return;
        }
#endif
        std::memset(data, 0, tensor_bytes);
        std::fill(s, s + n_elements_, scaling_t(1));
    }

    void QuadratureState::copy_tensor(const QuadratureState &src, const int from, const int to, void *const stream) {
        if (src.storage_ != storage_ || src.values_per_tensor() != values_per_tensor() || src.n_elements_ != n_elements_ ||
            src.es_ != es_) {
            SFEM_ERROR("QuadratureState::copy_tensor: the two stores do not hold the same kind of tensor\n");
        }
        const ptrdiff_t tensor_bytes = values_per_tensor() * n_elements_ * bytes_per_value(storage_);
        const uint8_t  *src_data     = src.data_->data() + from * tensor_bytes;
        uint8_t        *dst_data     = data_->data() + to * tensor_bytes;
        const scaling_t *src_scales  = src.scales_->data() + from * n_elements_;
        scaling_t       *dst_scales  = scales_->data() + to * n_elements_;
#ifndef SFEM_ENABLE_CUDA
        (void)stream;
#else
        if (es_ == EXECUTION_SPACE_DEVICE) {
            if (cu_prony_copy(tensor_bytes, src_data, dst_data, n_elements_, src_scales, dst_scales, stream) != SFEM_SUCCESS) {
                SFEM_ERROR("QuadratureState::copy_tensor failed on the device\n");
            }
            return;
        }
#endif
        std::memcpy(dst_data, src_data, tensor_bytes);
        std::memcpy(dst_scales, src_scales, n_elements_ * sizeof(scaling_t));
    }

    void QuadratureState::scale_tensor(const int t, const real_t factor, void *const stream) {
        void *const      data = data_->data() + t * values_per_tensor() * n_elements_ * bytes_per_value(storage_);
        scaling_t *const s    = scales_->data() + t * n_elements_;
#ifndef SFEM_ENABLE_CUDA
        (void)stream;
#else
        if (es_ == EXECUTION_SPACE_DEVICE) {
            if (cu_prony_scale(storage_, values_per_tensor(), n_elements_, factor, data, s, stream) != SFEM_SUCCESS) {
                SFEM_ERROR("QuadratureState::scale_tensor failed on the device\n");
            }
            return;
        }
#endif
        dispatch_history_codec(storage_, [&](auto codec) {
            using Codec = decltype(codec);
            prony::scale<Codec>(values_per_tensor(), n_elements_, factor, static_cast<typename Codec::storage_t *>(data), s);
            return 0;
        });
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
        const ptrdiff_t  tensor_bytes = values_per_tensor() * n_elements_ * bytes_per_value(storage_);
        const uint8_t   *data         = data_->data() + t * tensor_bytes;
        const scaling_t *s            = scales_->data() + t * n_elements_;
        // On the device the encoded tensor and its scales come to the host, which decodes them
        // as it does its own.
        std::vector<uint8_t>   host_data;
        std::vector<scaling_t> host_scales;
        if (es_ != EXECUTION_SPACE_HOST) {
#ifdef SFEM_ENABLE_CUDA
            host_data.resize(tensor_bytes);
            host_scales.resize(n_elements_);
            if (cudaMemcpy(host_data.data(), data, tensor_bytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
                cudaMemcpy(host_scales.data(), s, n_elements_ * sizeof(scaling_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
                SFEM_ERROR("QuadratureState::decode: copying the tensor from the device failed\n");
            }
            data = host_data.data();
            s    = host_scales.data();
#else
            SFEM_ERROR("QuadratureState::decode: device state without CUDA\n");
#endif
        }
        dispatch_history_codec(storage_, [&](auto codec) {
            using Codec = decltype(codec);
            prony::decode<Codec>(
                    values_per_tensor(), n_elements_, reinterpret_cast<const typename Codec::storage_t *>(data), s, values);
            return 0;
        });
    }

}  // namespace sfem
