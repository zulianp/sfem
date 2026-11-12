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
            if (es != EXECUTION_SPACE_HOST && !is_packed(storage)) {
                SFEM_ERROR("QuadratureState: %s is host only; quant_block runs on the device\n", to_string(storage));
            }
            if (!(tolerance > 0)) SFEM_ERROR("QuadratureState: %s needs a positive tolerance\n", to_string(storage));
            // A packed store's step and running maximum per tensor stay on the host, which sets
            // up every write and passes the step to the kernels.
            scales_ = is_packed(storage) ? create_host_buffer<scaling_t>(2 * n_tensors) : create_buffer<scaling_t>(2 * n_tensors, es);
        }
        if (is_packed(storage)) {
            const ptrdiff_t headers = values_per_tensor() * ((n_elements + prony::PACKED_BLOCK - 1) / prony::PACKED_BLOCK);
            packed_.resize(n_tensors);
            for (auto &p : packed_) {
                p.base   = create_buffer<int32_t>(headers, es);
                p.width  = create_buffer<uint8_t>(headers, es);
                p.offset = create_buffer<uint32_t>(headers, es);
            }
            if (es != EXECUTION_SPACE_HOST) {
                spare_.base   = create_buffer<int32_t>(headers, es);
                spare_.width  = create_buffer<uint8_t>(headers, es);
                spare_.offset = create_buffer<uint32_t>(headers, es);
                sizes_        = create_buffer<uint32_t>(headers, es);
                max_scratch_  = create_buffer<real_t>(1, es);
            }
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
        std::size_t packed = 0;
        for (const auto &p : packed_) {
            packed += p.codes->size() + p.base->size() * sizeof(int32_t) + p.width->size() + p.offset->size() * sizeof(uint32_t);
        }
        return data_->size() * sizeof(uint8_t) + (scales_ ? scales_->size() * sizeof(scaling_t) : 0) +
               (means_ ? means_->size() * sizeof(history_mean_t) : 0) + packed;
    }

    prony::PackedTensor QuadratureState::packed(const int t) const {
        const Packed       &p = packed_[t];
        prony::PackedTensor v;
        v.codes  = p.codes->data();
        v.base   = p.base->data();
        v.width  = p.width->data();
        v.offset = p.offset->data();
        v.rows   = values_per_tensor();
        v.n      = n_elements_;
        v.step   = scales_->data()[2 * t];
        return v;
    }

    namespace {
        /// `bytes` zero bytes at `data` in `es`.
        void zero_bytes(void *const data, const std::size_t bytes, const ExecutionSpace es) {
#ifdef SFEM_ENABLE_CUDA
            if (es == EXECUTION_SPACE_DEVICE) {
                if (cu_prony_zero(bytes, data, 0, nullptr, nullptr) != SFEM_SUCCESS) SFEM_ERROR("QuadratureState: zeroing on the device failed\n");
                return;
            }
#else
            (void)es;
#endif
            std::memset(data, 0, bytes);
        }

        /// `bytes` bytes from `src` to `dst`, both in `es`.
        void copy_bytes(void *const dst, const void *const src, const std::size_t bytes, const ExecutionSpace es) {
#ifdef SFEM_ENABLE_CUDA
            if (es == EXECUTION_SPACE_DEVICE) {
                if (cu_prony_copy(bytes, src, dst, 0, nullptr, nullptr, nullptr) != SFEM_SUCCESS) {
                    SFEM_ERROR("QuadratureState: copying on the device failed\n");
                }
                return;
            }
#else
            (void)es;
#endif
            std::memmove(dst, src, bytes);
        }

        /// A buffer's contents on the host, as `T`.
        template <typename T>
        std::vector<T> host_copy(const std::shared_ptr<Buffer<T>> &b, const ExecutionSpace es) {
            std::vector<T> out(b->size());
#ifdef SFEM_ENABLE_CUDA
            if (es == EXECUTION_SPACE_DEVICE) {
                if (cudaMemcpy(out.data(), b->data(), b->size() * sizeof(T), cudaMemcpyDeviceToHost) != cudaSuccess) {
                    SFEM_ERROR("QuadratureState: copying from the device failed\n");
                }
                return out;
            }
#else
            (void)es;
#endif
            std::copy(b->data(), b->data() + b->size(), out.begin());
            return out;
        }
    }  // namespace

    void QuadratureState::reset_packed(const int t) {
        // Every block empty: width 0, base 0, so every value decodes to zero.
        Packed &p = packed_[t];
        p.codes   = create_buffer<uint8_t>(static_cast<ptrdiff_t>(smesh::compression::kSlack), es_);
        zero_bytes(p.codes->data(), p.codes->size(), es_);
        zero_bytes(p.base->data(), p.base->size() * sizeof(int32_t), es_);
        zero_bytes(p.width->data(), p.width->size(), es_);
        zero_bytes(p.offset->data(), p.offset->size() * sizeof(uint32_t), es_);
        scales_->data()[2 * t]     = 1;
        scales_->data()[2 * t + 1] = 0;
    }

    void QuadratureState::write_packed_device(const int             t,
                                              const real_t          alpha,
                                              const real_t          b,
                                              const real_t          beta,
                                              const int             stress_bytes,
                                              const void *const     S_n,
                                              const void *const     S_new,
                                              const void *const     ref,
                                              const HistoryRounding rounding,
                                              const uint64_t        seed,
                                              const uint64_t        k0,
                                              void *const           stream) {
#ifdef SFEM_ENABLE_CUDA
        const prony::PackedTensor old = packed(t);
        const PackedUpdate        u{alpha, b, beta};
        const int                 stochastic = rounding == HistoryRounding::STOCHASTIC;
        real_t                    max_abs    = 0;
        if (cu_prony_packed_max(old, u, stress_bytes, S_n, S_new, ref, max_scratch_->data(), &max_abs, stream) != SFEM_SUCCESS) {
            SFEM_ERROR("QuadratureState::write_packed_device: the largest value failed\n");
        }
        max_abs           = std::max(max_abs, static_cast<real_t>(scales_->data()[2 * t + 1]));
        const real_t step = prony::grid_step(max_abs, tolerance_);
        if (cu_prony_packed_plan(old, u, stress_bytes, S_n, S_new, ref, step, stochastic, seed, k0, spare_.base->data(),
                                 spare_.width->data(), sizes_->data(), stream) != SFEM_SUCCESS) {
            SFEM_ERROR("QuadratureState::write_packed_device: the plan failed\n");
        }
        // The offsets, summed on the host: the sizes are the blocks', a few per thousand values.
        std::vector<uint32_t> sizes = host_copy(sizes_, es_);
        std::vector<uint32_t> offset(sizes.size());
        uint64_t              at = 0;
        for (std::size_t h = 0; h < sizes.size(); h++) {
            if (at > UINT32_MAX) SFEM_ERROR("QuadratureState::write_packed_device: a tensor's codes exceed 32 GB\n");
            offset[h] = static_cast<uint32_t>(at);
            at += sizes[h];
        }
        if (cudaMemcpy(spare_.offset->data(), offset.data(), offset.size() * sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess) {
            SFEM_ERROR("QuadratureState::write_packed_device: copying the offsets failed\n");
        }
        auto codes = create_buffer<uint8_t>(static_cast<ptrdiff_t>(at * 8 + smesh::compression::kSlack), es_);
        zero_bytes(codes->data(), codes->size(), es_);
        if (cu_prony_packed_pack(old, u, stress_bytes, S_n, S_new, ref, step, stochastic, seed, k0, spare_.base->data(),
                                 spare_.width->data(), spare_.offset->data(), codes->data(), stream) != SFEM_SUCCESS) {
            SFEM_ERROR("QuadratureState::write_packed_device: packing failed\n");
        }
        // The old codes go once the kernels that read them have finished.
        if (cudaStreamSynchronize((cudaStream_t)stream) != cudaSuccess) SFEM_ERROR("QuadratureState::write_packed_device failed\n");
        Packed &p = packed_[t];
        p.codes   = codes;
        std::swap(p.base, spare_.base);
        std::swap(p.width, spare_.width);
        std::swap(p.offset, spare_.offset);
        scales_->data()[2 * t]     = static_cast<scaling_t>(step);
        scales_->data()[2 * t + 1] = static_cast<scaling_t>(max_abs);
#else
        (void)t, (void)alpha, (void)b, (void)beta, (void)stress_bytes, (void)S_n, (void)S_new, (void)ref, (void)rounding, (void)seed,
                (void)k0, (void)stream;
        SFEM_ERROR("QuadratureState::write_packed_device: a build without CUDA\n");
#endif
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
        if (is_packed(storage_)) {
            for (int t = 0; t < n_tensors_; t++) reset_packed(t);
            return;
        }
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
        if (is_packed(storage_)) {
            reset_packed(t);
            return;
        }
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
        if (is_packed(storage_)) {
            const Packed &s = src.packed_[from];
            Packed       &d = packed_[to];
            // The code area by value: the source may be this store, and its slot be written next.
            auto codes = create_buffer<uint8_t>(s.codes->size(), es_);
            copy_bytes(codes->data(), s.codes->data(), s.codes->size(), es_);
            d.codes = codes;
            copy_bytes(d.base->data(), s.base->data(), s.base->size() * sizeof(int32_t), es_);
            copy_bytes(d.width->data(), s.width->data(), s.width->size(), es_);
            copy_bytes(d.offset->data(), s.offset->data(), s.offset->size() * sizeof(uint32_t), es_);
            scales_->data()[2 * to]     = src.scales_->data()[2 * from];
            scales_->data()[2 * to + 1] = src.scales_->data()[2 * from + 1];
            return;
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
        if (is_packed(storage_)) {
            const ptrdiff_t n = n_elements_;
            write_packed(t, [&](const ptrdiff_t r, const ptrdiff_t e) { return values[r * n + e]; });
            return;
        }
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
        if (is_packed(storage_)) {
            auto            v = packed(t);
            const ptrdiff_t n = n_elements_;
            // On the device the tensor comes to the host, which decodes it as its own.
            std::vector<uint8_t>  codes;
            std::vector<int32_t>  base;
            std::vector<uint8_t>  width;
            std::vector<uint32_t> offset;
            if (es_ != EXECUTION_SPACE_HOST) {
                const Packed &p = packed_[t];
                codes           = host_copy(p.codes, es_);
                base            = host_copy(p.base, es_);
                width           = host_copy(p.width, es_);
                offset          = host_copy(p.offset, es_);
                v.codes         = codes.data();
                v.base          = base.data();
                v.width         = width.data();
                v.offset        = offset.data();
            }
#pragma omp parallel for collapse(2)
            for (ptrdiff_t r = 0; r < v.rows; r++) {
                for (ptrdiff_t e = 0; e < n; e++) values[r * n + e] = v.value(r, e);
            }
            return;
        }
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
        if (is_packed(storage_)) return bytes();
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
        if (is_packed(storage_)) {
            // Per tensor its code area, whose size the .packed file records, and its headers.
            std::vector<std::pair<std::string, double>> sizes;
            for (int t = 0; t < n_tensors_; t++) {
                const Packed     &p = packed_[t];
                const std::string k = prefix + ".t" + std::to_string(t);
                sizes.push_back({"codes_" + std::to_string(t), (double)p.codes->size()});
                if (write_state_buffer(smesh::Path(k + ".codes"), p.codes) || write_state_buffer(smesh::Path(k + ".base"), p.base) ||
                    write_state_buffer(smesh::Path(k + ".width"), p.width) || write_state_buffer(smesh::Path(k + ".offset"), p.offset)) {
                    return SFEM_FAILURE;
                }
            }
            if (write_state_numbers(smesh::Path(prefix + ".packed"), sizes)) return SFEM_FAILURE;
            return write_state_buffer(smesh::Path(prefix + ".scales"), scales_);
        }
        // The unscaled formats hold no scales (is_scaled), so they write no .scales file.
        if (write_state_buffer(smesh::Path(prefix + ".values"), data_) ||
            (scales_ && write_state_buffer(smesh::Path(prefix + ".scales"), scales_))) {
            return SFEM_FAILURE;
        }
        return means_ ? write_state_buffer(smesh::Path(prefix + ".means"), means_) : SFEM_SUCCESS;
    }

    int QuadratureState::read_state(const std::string &prefix) {
        if (is_packed(storage_)) {
            std::map<std::string, double> sizes;
            if (read_state_numbers(smesh::Path(prefix + ".packed"), sizes)) return SFEM_FAILURE;
            for (int t = 0; t < n_tensors_; t++) {
                double bytes = 0;
                if (state_number(sizes, ("codes_" + std::to_string(t)).c_str(), &bytes)) return SFEM_FAILURE;
                Packed           &p = packed_[t];
                const std::string k = prefix + ".t" + std::to_string(t);
                p.codes             = create_buffer<uint8_t>(static_cast<ptrdiff_t>(bytes), es_);
                if (read_state_buffer(smesh::Path(k + ".codes"), p.codes) || read_state_buffer(smesh::Path(k + ".base"), p.base) ||
                    read_state_buffer(smesh::Path(k + ".width"), p.width) || read_state_buffer(smesh::Path(k + ".offset"), p.offset)) {
                    return SFEM_FAILURE;
                }
            }
            return read_state_buffer(smesh::Path(prefix + ".scales"), scales_);
        }
        if (read_state_buffer(smesh::Path(prefix + ".values"), data_) ||
            (scales_ && read_state_buffer(smesh::Path(prefix + ".scales"), scales_))) {
            return SFEM_FAILURE;
        }
        return means_ ? read_state_buffer(smesh::Path(prefix + ".means"), means_) : SFEM_SUCCESS;
    }

}  // namespace sfem
