#include "sfem_QuadratureState.hpp"
#include "sfem_StateFiles.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

#include "sfem_prony_kernels.hpp"
#include "smesh_codec_bits.hpp"

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
                                     const ExecutionSpace es,
                                     const int            elements_per_mean,
                                     const real_t         tolerance)
        : n_tensors_(n_tensors),
          n_components_(n_components),
          n_qp_(n_qp),
          n_elements_(n_elements),
          storage_(storage),
          es_(es),
          tolerance_(tolerance) {
        if (n_tensors < 0 || n_components <= 0 || n_qp <= 0 || n_elements < 0) {
            SFEM_ERROR("QuadratureState: invalid shape (%d tensors, %d components, %d points, %ld elements)\n",
                       n_tensors,
                       n_components,
                       n_qp,
                       (long)n_elements);
        }

        const ptrdiff_t n_values = n_tensors * values_per_tensor() * n_elements;
        data_ = create_buffer<uint8_t>(n_values * bytes_per_value(storage), es);
        if (is_scaled(storage)) scales_ = create_buffer<scaling_t>(n_tensors * n_elements, es);
        if (is_grid(storage)) {
            if (es != EXECUTION_SPACE_HOST) SFEM_ERROR("QuadratureState: %s is host only for now\n", to_string(storage));
            if (!(tolerance > 0)) SFEM_ERROR("QuadratureState: %s needs a positive tolerance\n", to_string(storage));
            scales_ = create_buffer<scaling_t>(2 * n_tensors, es);
        }

        if (is_mean_deviation(storage)) {
            if (n_components > 9) {
                SFEM_ERROR("QuadratureState: %s holds at most 9 components, not %d\n", to_string(storage), n_components);
            }
            elements_per_mean_ = elements_per_mean > 0 ? elements_per_mean : (n_qp > 1 ? 1 : DEFAULT_ELEMENTS_PER_MEAN);
            if (elements_per_mean_ > prony::MAX_ELEMENTS_PER_MEAN || prony::MAX_ELEMENTS_PER_MEAN % elements_per_mean_ != 0) {
                SFEM_ERROR("QuadratureState: elements per mean %d must be a power of two up to %d\n",
                           elements_per_mean_,
                           prony::MAX_ELEMENTS_PER_MEAN);
            }
            n_groups_ = (n_elements + elements_per_mean_ - 1) / elements_per_mean_;
            means_    = create_buffer<history_mean_t>(static_cast<ptrdiff_t>(n_tensors) * n_components * n_groups_, es);
        }
        zero();
    }

    std::size_t QuadratureState::bytes() const {
        return data_->size() * sizeof(uint8_t) + (scales_ ? scales_->size() * sizeof(scaling_t) : 0) +
               (means_ ? means_->size() * sizeof(history_mean_t) : 0);
    }

    prony::Means QuadratureState::means_view() const {
        prony::Means m;
        if (!means_) return m;
        m.data     = means_->data();
        m.n_qp     = n_qp_;
        m.elements = elements_per_mean_;
        m.n_groups = n_groups_;
        return m;
    }

    void QuadratureState::zero(void *const stream) {
#ifndef SFEM_ENABLE_CUDA
        (void)stream;
#else
        if (es_ == EXECUTION_SPACE_DEVICE) {
            if (cu_prony_zero(data_->size(), data_->data(), scales_ ? scales_->size() : 0, scales(), stream) != SFEM_SUCCESS ||
                (means_ && cu_prony_zero(means_->size() * sizeof(history_mean_t), means_->data(), 0, nullptr, stream) != SFEM_SUCCESS)) {
                SFEM_ERROR("QuadratureState::zero failed on the device\n");
            }
            return;
        }
#endif
        // Zero is the all-zero bit pattern in every storage format.
        std::memset(data_->data(), 0, data_->size());
        if (scales_ && is_grid(storage_)) {
            for (int t = 0; t < n_tensors_; t++) {
                scales_->data()[2 * t]     = 1;
                scales_->data()[2 * t + 1] = 0;
            }
        } else if (scales_) {
            std::fill(scales_->data(), scales_->data() + scales_->size(), scaling_t(1));
        }
        if (means_) std::memset(means_->data(), 0, means_->size() * sizeof(history_mean_t));
    }

    void QuadratureState::zero_tensor(const int t, void *const stream) {
        const ptrdiff_t tensor_bytes = values_per_tensor() * n_elements_ * bytes_per_value(storage_);
        uint8_t *const   data        = data_->data() + t * tensor_bytes;
        if (is_grid(storage_)) {
            std::memset(data, 0, tensor_bytes);
            scales_->data()[2 * t]     = 1;
            scales_->data()[2 * t + 1] = 0;
            return;
        }
        scaling_t *const s           = scales_ ? scales_->data() + t * n_elements_ : nullptr;
#ifndef SFEM_ENABLE_CUDA
        (void)stream;
#else
        if (es_ == EXECUTION_SPACE_DEVICE) {
            if (cu_prony_zero(tensor_bytes, data, s ? n_elements_ : 0, s, stream) != SFEM_SUCCESS ||
                (means_ && cu_prony_zero(n_components_ * n_groups_ * sizeof(history_mean_t),
                                         means_view().tensor(t, values_per_tensor()),
                                         0,
                                         nullptr,
                                         stream) != SFEM_SUCCESS)) {
                SFEM_ERROR("QuadratureState::zero_tensor failed on the device\n");
            }
            return;
        }
#endif
        std::memset(data, 0, tensor_bytes);
        if (s) std::fill(s, s + n_elements_, scaling_t(1));
        if (means_) std::memset(means_view().tensor(t, values_per_tensor()), 0, n_components_ * n_groups_ * sizeof(history_mean_t));
    }

    void QuadratureState::copy_tensor(const QuadratureState &src, const int from, const int to, void *const stream) {
        if (src.storage_ != storage_ || src.values_per_tensor() != values_per_tensor() || src.n_elements_ != n_elements_ ||
            src.es_ != es_ || src.n_qp_ != n_qp_ || src.elements_per_mean_ != elements_per_mean_) {
            SFEM_ERROR("QuadratureState::copy_tensor: the two stores do not hold the same kind of tensor\n");
        }
        const ptrdiff_t tensor_bytes = values_per_tensor() * n_elements_ * bytes_per_value(storage_);
        const uint8_t  *src_data     = src.data_->data() + from * tensor_bytes;
        uint8_t        *dst_data     = data_->data() + to * tensor_bytes;
        if (is_grid(storage_)) {
            std::memmove(dst_data, src_data, tensor_bytes);
            scales_->data()[2 * to]     = src.scales_->data()[2 * from];
            scales_->data()[2 * to + 1] = src.scales_->data()[2 * from + 1];
            return;
        }
        const scaling_t *src_scales  = src.scales_ ? src.scales_->data() + from * n_elements_ : nullptr;
        scaling_t       *dst_scales  = scales_ ? scales_->data() + to * n_elements_ : nullptr;
#ifndef SFEM_ENABLE_CUDA
        (void)stream;
#else
        if (es_ == EXECUTION_SPACE_DEVICE) {
            if (cu_prony_copy(tensor_bytes, src_data, dst_data, dst_scales ? n_elements_ : 0, src_scales, dst_scales, stream) !=
                        SFEM_SUCCESS ||
                (means_ && cu_prony_copy(n_components_ * n_groups_ * sizeof(history_mean_t),
                                         src.means_view().tensor(from, values_per_tensor()),
                                         means_view().tensor(to, values_per_tensor()),
                                         0,
                                         nullptr,
                                         nullptr,
                                         stream) != SFEM_SUCCESS)) {
                SFEM_ERROR("QuadratureState::copy_tensor failed on the device\n");
            }
            return;
        }
#endif
        std::memcpy(dst_data, src_data, tensor_bytes);
        if (dst_scales) std::memcpy(dst_scales, src_scales, n_elements_ * sizeof(scaling_t));
        if (means_) {
            std::memmove(means_view().tensor(to, values_per_tensor()),
                         src.means_view().tensor(from, values_per_tensor()),
                         n_components_ * n_groups_ * sizeof(history_mean_t));
        }
    }

    void QuadratureState::encode(const int t, const real_t *const values) {
        if (es_ != EXECUTION_SPACE_HOST) SFEM_ERROR("QuadratureState::encode is a host convenience\n");
        dispatch_history_codec(storage_, [&](auto codec) {
            using Codec = decltype(codec);
            prony::Means m = means_view();
            m.data         = means_ ? m.tensor(t, values_per_tensor()) : nullptr;
            prony::encode<Codec>(values_per_tensor(),
                                 n_elements_,
                                 values,
                                 &data<Codec>()[t * values_per_tensor() * n_elements_],
                                 scales_ ? &scales()[Codec::grid ? 2 * t : t * n_elements_] : nullptr,
                                 m,
                                 tolerance_);
            return 0;
        });
    }

    void QuadratureState::decode(const int t, real_t *const values) const {
        const ptrdiff_t  tensor_bytes = values_per_tensor() * n_elements_ * bytes_per_value(storage_);
        const uint8_t   *data         = data_->data() + t * tensor_bytes;
        const scaling_t *s            = scales_ ? scales_->data() + (is_grid(storage_) ? 2 * t : t * n_elements_) : nullptr;
        // On the device the encoded tensor and its scales come to the host, which decodes them
        // as it does its own.
        prony::Means m = means_view();
        m.data         = means_ ? m.tensor(t, values_per_tensor()) : nullptr;
        std::vector<uint8_t>        host_data;
        std::vector<scaling_t>      host_scales;
        std::vector<history_mean_t> host_means;
        if (es_ != EXECUTION_SPACE_HOST) {
#ifdef SFEM_ENABLE_CUDA
            host_data.resize(tensor_bytes);
            if (s) host_scales.resize(n_elements_);
            if (cudaMemcpy(host_data.data(), data, tensor_bytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
                (s && cudaMemcpy(host_scales.data(), s, n_elements_ * sizeof(scaling_t), cudaMemcpyDeviceToHost) != cudaSuccess)) {
                SFEM_ERROR("QuadratureState::decode: copying the tensor from the device failed\n");
            }
            data = host_data.data();
            if (s) s = host_scales.data();
            if (m.data) {
                host_means.resize(n_components_ * n_groups_);
                if (cudaMemcpy(host_means.data(), m.data, host_means.size() * sizeof(history_mean_t), cudaMemcpyDeviceToHost) !=
                    cudaSuccess) {
                    SFEM_ERROR("QuadratureState::decode: copying the means from the device failed\n");
                }
                m.data = host_means.data();
            }
#else
            SFEM_ERROR("QuadratureState::decode: device state without CUDA\n");
#endif
        }
        dispatch_history_codec(storage_, [&](auto codec) {
            using Codec = decltype(codec);
            prony::decode<Codec>(
                    values_per_tensor(), n_elements_, reinterpret_cast<const typename Codec::storage_t *>(data), s, values, m);
            return 0;
        });
    }

    std::size_t QuadratureState::packed_bytes() const {
        if (es_ != EXECUTION_SPACE_HOST || !is_integer(storage_)) return 0;
        const std::size_t side = (scales_ ? scales_->size() * sizeof(scaling_t) : 0) +
                                 (means_ ? means_->size() * sizeof(history_mean_t) : 0);
        const ptrdiff_t rows = static_cast<ptrdiff_t>(n_tensors_) * values_per_tensor();
        const ptrdiff_t n    = n_elements_;
        std::size_t     codes = 0;
        dispatch_history_codec(storage_, [&](auto codec) {
            using Codec = decltype(codec);
            if constexpr (Codec::integer) {
                const auto *const h = data<Codec>();
#pragma omp parallel for reduction(+ : codes)
                for (ptrdiff_t r = 0; r < rows; r++) {
                    for (ptrdiff_t e0 = 0; e0 < n; e0 += PACK_BLOCK) {
                        const ptrdiff_t nb = std::min(PACK_BLOCK, n - e0);
                        int64_t         lo = h[r * n + e0], hi = lo;
                        for (ptrdiff_t l = 1; l < nb; l++) {
                            const int64_t v = h[r * n + e0 + l];
                            lo              = std::min(lo, v);
                            hi              = std::max(hi, v);
                        }
                        const int w = smesh::compression::bit_width(static_cast<uint64_t>(hi - lo));
                        codes += 6 + (static_cast<std::size_t>(nb) * w + 7) / 8;
                    }
                }
            }
            return 0;
        });
        return codes + side;
    }

    int QuadratureState::write_state(const std::string &prefix) const {
        // The unscaled formats hold no scales (is_scaled), so they write no .scales file.
        if (write_state_buffer(smesh::Path(prefix + ".values"), data_) ||
            (scales_ && write_state_buffer(smesh::Path(prefix + ".scales"), scales_))) {
            return SFEM_FAILURE;
        }
        return means_ ? write_state_buffer(smesh::Path(prefix + ".means"), means_) : SFEM_SUCCESS;
    }

    int QuadratureState::read_state(const std::string &prefix) {
        if (read_state_buffer(smesh::Path(prefix + ".values"), data_) ||
            (scales_ && read_state_buffer(smesh::Path(prefix + ".scales"), scales_))) {
            return SFEM_FAILURE;
        }
        return means_ ? read_state_buffer(smesh::Path(prefix + ".means"), means_) : SFEM_SUCCESS;
    }

}  // namespace sfem
