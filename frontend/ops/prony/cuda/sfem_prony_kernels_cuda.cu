#include "sfem_prony_kernels_cuda.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "sfem_cuda_base.hpp"

namespace sfem {

    namespace {

        constexpr int MAX_TERMS = 64;

        /// The step's weights, by value in the launch: a spectrum is tens of
        /// terms, so this is under a kilobyte of kernel parameters.
        struct Weights {
            real_t alpha[MAX_TERMS];
            real_t b[MAX_TERMS];
        };

        /// One rounding to binary16, from whatever real_t is.  Going through
        /// float first would round twice and disagree with the host, which
        /// converts double to half directly.
        __device__ inline __half to_half(const real_t v) {
            if constexpr (std::is_same_v<real_t, float>) {
                return __float2half_rn(v);
            } else {
                return __double2half(v);
            }
        }

        // The device codecs.  The host ones (`sfem_HistoryStorage.hpp`) use the
        // host's half type and <cmath>; these spell the same encodings with the
        // device intrinsics.  The stored bits are identical: binary16 is binary16,
        // and the integer codes are the integer codes.

        template <typename T>
        struct DeviceCast {
            using storage_t              = T;
            static constexpr bool scaled = false;
            __device__ static real_t    decode(const T s, const real_t) { return static_cast<real_t>(s); }
            __device__ static T         encode(const real_t v, const real_t) { return static_cast<T>(v); }
        };

        struct DeviceHalf {
            using storage_t              = __half;
            static constexpr bool scaled = false;
            __device__ static real_t    decode(const __half s, const real_t) { return __half2float(s); }
            __device__ static __half    encode(const real_t v, const real_t) { return to_half(v); }
        };

        struct DeviceScaledHalf {
            using storage_t              = __half;
            static constexpr bool scaled = true;
            __device__ static real_t decode(const __half s, const real_t scale) { return __half2float(s) * scale; }
            __device__ static __half encode(const real_t v, const real_t inv_scale) { return to_half(v * inv_scale); }
            __device__ static scaling_t scale_for(const real_t max_abs) {
                if (!(max_abs > 0)) return 1;
                return static_cast<scaling_t>(exp2(ceil(log2(static_cast<double>(max_abs))) - 14));
            }
        };

        template <typename T, int LARGEST_CODE>
        struct DeviceScaledInteger {
            using storage_t              = T;
            static constexpr bool scaled = true;
            __device__ static real_t decode(const T s, const real_t scale) { return static_cast<real_t>(s) * scale; }
            __device__ static T      encode(const real_t v, const real_t inv_scale) {
                real_t code = nearbyint(v * inv_scale);
                code        = code > LARGEST_CODE ? LARGEST_CODE : (code < -LARGEST_CODE ? -LARGEST_CODE : code);
                return static_cast<T>(code);
            }
            __device__ static scaling_t scale_for(const real_t max_abs) {
                if (!(max_abs > 0)) return 1;
                return static_cast<scaling_t>(max_abs / LARGEST_CODE);
            }
        };

        template <class F>
        int dispatch(const HistoryStorage storage, F &&f) {
            switch (storage) {
                case HistoryStorage::FULL:
                    return f(DeviceCast<real_t>{});
                case HistoryStorage::FLOAT32:
                    return f(DeviceCast<float>{});
                case HistoryStorage::FLOAT16:
                    return f(DeviceHalf{});
                case HistoryStorage::FLOAT16_SCALED:
                    return f(DeviceScaledHalf{});
                case HistoryStorage::INT16_SCALED:
                    return f(DeviceScaledInteger<int16_t, 32767>{});
                case HistoryStorage::INT8_SCALED:
                    return f(DeviceScaledInteger<int8_t, 127>{});
            }
            return SFEM_FAILURE;
        }

        /// One thread per (row, element): the entries are independent.
        template <class Codec>
        __global__ void frozen_history_kernel(const int                                             n_terms,
                                              const ptrdiff_t                                       rows,
                                              const ptrdiff_t                                       n,
                                              const Weights                                         w,
                                              const real_t                                          b_sum,
                                              const typename Codec::storage_t *const SFEM_RESTRICT H,
                                              const scaling_t *const SFEM_RESTRICT                 H_scales,
                                              const real_t *const SFEM_RESTRICT                    S_n,
                                              real_t *const SFEM_RESTRICT                          out) {
            const ptrdiff_t total = rows * n;
            for (ptrdiff_t k = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; k < total;
                 k += (ptrdiff_t)blockDim.x * gridDim.x) {
                const ptrdiff_t r   = k / n;
                const ptrdiff_t e   = k - r * n;
                real_t          acc = -b_sum * S_n[k];
                for (int i = 0; i < n_terms; i++) {
                    acc += w.alpha[i] * Codec::decode(H[(i * rows + r) * n + e], H_scales[i * n + e]);
                }
                out[k] = acc;
            }
        }

        /// One thread per element: a scaled codec needs the max-abs of a whole
        /// tensor of the element before its first value is written, so the
        /// element's rows stay in one thread.  Adjacent threads read adjacent
        /// elements of every row, so the accesses coalesce.
        template <class Codec>
        __global__ void advance_kernel(const int                                       n_terms,
                                       const ptrdiff_t                                 rows,
                                       const ptrdiff_t                                 n,
                                       const Weights                                   w,
                                       const real_t *const SFEM_RESTRICT              S_n,
                                       const real_t *const SFEM_RESTRICT              S_new,
                                       typename Codec::storage_t *const SFEM_RESTRICT H,
                                       scaling_t *const SFEM_RESTRICT                 H_scales) {
            for (ptrdiff_t e = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; e < n;
                 e += (ptrdiff_t)blockDim.x * gridDim.x) {
                for (int i = 0; i < n_terms; i++) {
                    typename Codec::storage_t *const h         = &H[i * rows * n];
                    const real_t                     a         = w.alpha[i];
                    const real_t                     bi        = w.b[i];
                    const real_t                     old_scale = H_scales[i * n + e];

                    real_t inv = 1;
                    if constexpr (Codec::scaled) {
                        real_t max_abs = 0;
                        for (ptrdiff_t r = 0; r < rows; r++) {
                            const ptrdiff_t k = r * n + e;
                            const real_t    v = a * Codec::decode(h[k], old_scale) + bi * (S_new[k] - S_n[k]);
                            max_abs           = fmax(max_abs, fabs(v));
                        }
                        const scaling_t scale = Codec::scale_for(max_abs);
                        H_scales[i * n + e]   = scale;
                        inv                   = 1 / static_cast<real_t>(scale);
                    }

                    for (ptrdiff_t r = 0; r < rows; r++) {
                        const ptrdiff_t k = r * n + e;
                        const real_t    v = a * Codec::decode(h[k], old_scale) + bi * (S_new[k] - S_n[k]);
                        h[k]              = Codec::encode(v, inv);
                    }
                }
            }
        }

        template <class Codec>
        __global__ void scale_kernel(const ptrdiff_t                                 rows,
                                     const ptrdiff_t                                 n,
                                     const real_t                                    factor,
                                     typename Codec::storage_t *const SFEM_RESTRICT H,
                                     scaling_t *const SFEM_RESTRICT                 H_scales) {
            for (ptrdiff_t e = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; e < n;
                 e += (ptrdiff_t)blockDim.x * gridDim.x) {
                const real_t old_scale = H_scales[e];

                real_t inv = 1;
                if constexpr (Codec::scaled) {
                    real_t max_abs = 0;
                    for (ptrdiff_t r = 0; r < rows; r++) {
                        max_abs = fmax(max_abs, fabs(factor * Codec::decode(H[r * n + e], old_scale)));
                    }
                    const scaling_t scale = Codec::scale_for(max_abs);
                    H_scales[e]           = scale;
                    inv                   = 1 / static_cast<real_t>(scale);
                }

                for (ptrdiff_t r = 0; r < rows; r++) {
                    const ptrdiff_t k = r * n + e;
                    H[k]              = Codec::encode(factor * Codec::decode(H[k], old_scale), inv);
                }
            }
        }

        __global__ void fill_scales_kernel(const ptrdiff_t n, scaling_t *const SFEM_RESTRICT scales) {
            for (ptrdiff_t k = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; k < n;
                 k += (ptrdiff_t)blockDim.x * gridDim.x) {
                scales[k] = 1;
            }
        }

        /// The loops stride over the grid, so a capped grid covers any size.
        unsigned grid_for(const ptrdiff_t n, const int block) {
            const ptrdiff_t blocks = std::max(ptrdiff_t(1), (n + block - 1) / block);
            return (unsigned)std::min(blocks, ptrdiff_t(65535) * 32);
        }

        int launched() { return cudaGetLastError() == cudaSuccess ? SFEM_SUCCESS : SFEM_FAILURE; }

        Weights pack(const int n_terms, const real_t *const alpha, const real_t *const b) {
            Weights w{};
            for (int i = 0; i < n_terms; i++) {
                w.alpha[i] = alpha[i];
                w.b[i]     = b ? b[i] : real_t(0);
            }
            return w;
        }

    }  // namespace

    int cu_prony_max_terms() { return MAX_TERMS; }

    int cu_prony_frozen_history(const HistoryStorage   storage,
                                const int              n_terms,
                                const ptrdiff_t        rows,
                                const ptrdiff_t        n,
                                const real_t *const    alpha,
                                const real_t           b_sum,
                                const void *const      H,
                                const scaling_t *const H_scales,
                                const real_t *const    S_n,
                                real_t *const          out,
                                void *const            stream) {
        if (n_terms > MAX_TERMS) return SFEM_FAILURE;
        SFEM_DEBUG_SYNCHRONIZE(stream);
        const Weights w     = pack(n_terms, alpha, nullptr);
        const int     block = 128;
        const int     ret   = dispatch(storage, [&](auto codec) {
            using Codec = decltype(codec);
            frozen_history_kernel<Codec><<<grid_for(rows * n, block), block, 0, (cudaStream_t)stream>>>(
                    n_terms, rows, n, w, b_sum, static_cast<const typename Codec::storage_t *>(H), H_scales, S_n, out);
            return launched();
        });
        SFEM_DEBUG_SYNCHRONIZE(stream);
        return ret;
    }

    int cu_prony_advance(const HistoryStorage storage,
                         const int            n_terms,
                         const ptrdiff_t      rows,
                         const ptrdiff_t      n,
                         const real_t *const  alpha,
                         const real_t *const  b,
                         const real_t *const  S_n,
                         const real_t *const  S_new,
                         void *const          H,
                         scaling_t *const     H_scales,
                         void *const          stream) {
        if (n_terms > MAX_TERMS) return SFEM_FAILURE;
        SFEM_DEBUG_SYNCHRONIZE(stream);
        const Weights w     = pack(n_terms, alpha, b);
        const int     block = 128;
        const int     ret   = dispatch(storage, [&](auto codec) {
            using Codec = decltype(codec);
            advance_kernel<Codec><<<grid_for(n, block), block, 0, (cudaStream_t)stream>>>(
                    n_terms, rows, n, w, S_n, S_new, static_cast<typename Codec::storage_t *>(H), H_scales);
            return launched();
        });
        SFEM_DEBUG_SYNCHRONIZE(stream);
        return ret;
    }

    int cu_prony_scale(const HistoryStorage storage,
                       const ptrdiff_t      rows,
                       const ptrdiff_t      n,
                       const real_t         factor,
                       void *const          H,
                       scaling_t *const     H_scales,
                       void *const          stream) {
        SFEM_DEBUG_SYNCHRONIZE(stream);
        const int block = 128;
        const int ret   = dispatch(storage, [&](auto codec) {
            using Codec = decltype(codec);
            scale_kernel<Codec><<<grid_for(n, block), block, 0, (cudaStream_t)stream>>>(
                    rows, n, factor, static_cast<typename Codec::storage_t *>(H), H_scales);
            return launched();
        });
        SFEM_DEBUG_SYNCHRONIZE(stream);
        return ret;
    }

    int cu_prony_copy(const ptrdiff_t        bytes,
                      const void *const      src,
                      void *const            dst,
                      const ptrdiff_t        n_scales,
                      const scaling_t *const src_scales,
                      scaling_t *const       dst_scales,
                      void *const            stream) {
        if (bytes > 0 && cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, (cudaStream_t)stream) != cudaSuccess) {
            return SFEM_FAILURE;
        }
        if (n_scales > 0 &&
            cudaMemcpyAsync(dst_scales, src_scales, n_scales * sizeof(scaling_t), cudaMemcpyDeviceToDevice, (cudaStream_t)stream) !=
                    cudaSuccess) {
            return SFEM_FAILURE;
        }
        SFEM_DEBUG_SYNCHRONIZE(stream);
        return SFEM_SUCCESS;
    }

    int cu_prony_zero(const ptrdiff_t  bytes,
                      void *const      data,
                      const ptrdiff_t  n_scales,
                      scaling_t *const scales,
                      void *const      stream) {
        if (bytes > 0 && cudaMemsetAsync(data, 0, bytes, (cudaStream_t)stream) != cudaSuccess) return SFEM_FAILURE;
        int ret = SFEM_SUCCESS;
        if (n_scales > 0) {
            const int block = 128;
            fill_scales_kernel<<<grid_for(n_scales, block), block, 0, (cudaStream_t)stream>>>(n_scales, scales);
            ret = launched();
        }
        SFEM_DEBUG_SYNCHRONIZE(stream);
        return ret;
    }

}  // namespace sfem
