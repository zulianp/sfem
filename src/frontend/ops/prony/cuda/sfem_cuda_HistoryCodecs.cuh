#pragma once

#include <cuda_fp16.h>

#include <type_traits>

#include "sfem_HistoryStorage.hpp"
#include "sfem_base.hpp"

// The history codecs on the device (sfem_HistoryStorage.hpp has the host's), shared by the
// Prony history kernels and the stored-tangent stores (frontend/ops/inexact/cuda).

namespace sfem {
    namespace device_codecs {

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
            using storage_t                      = T;
            static constexpr bool scaled         = false;
            static constexpr bool mean_deviation = false;
            static constexpr bool integer        = false;
            __device__ static real_t    decode(const T s, const real_t) { return static_cast<real_t>(s); }
            __device__ static T         encode(const real_t v, const real_t) { return static_cast<T>(v); }
        };

        struct DeviceHalf {
            using storage_t                      = __half;
            static constexpr bool scaled         = false;
            static constexpr bool mean_deviation = false;
            static constexpr bool integer        = false;
            __device__ static real_t    decode(const __half s, const real_t) { return __half2float(s); }
            __device__ static __half    encode(const real_t v, const real_t) { return to_half(v); }
        };

        struct DeviceScaledHalf {
            using storage_t                      = __half;
            static constexpr bool scaled         = true;
            static constexpr bool mean_deviation = false;
            static constexpr bool integer        = false;
            __device__ static real_t decode(const __half s, const real_t scale) { return __half2float(s) * scale; }
            __device__ static __half encode(const real_t v, const real_t inv_scale) { return to_half(v * inv_scale); }
            __device__ static scaling_t scale_for(const real_t max_abs) {
                if (!(max_abs > 0)) return 1;
                return static_cast<scaling_t>(exp2(ceil(log2(static_cast<double>(max_abs))) - 14));
            }
        };

        template <typename T, int LARGEST_CODE>
        struct DeviceScaledInteger {
            using storage_t                      = T;
            static constexpr bool scaled         = true;
            static constexpr bool mean_deviation = false;
            static constexpr bool integer        = true;
            __device__ static real_t decode(const T s, const real_t scale) { return static_cast<real_t>(s) * scale; }
            __device__ static T      encode(const real_t v, const real_t inv_scale) { return clamp(nearbyint(v * inv_scale)); }
            /// Rounded up with the probability of the remainder, u uniform in [0, 1), as on the host.
            __device__ static T encode(const real_t v, const real_t inv_scale, const real_t u) {
                return clamp(floor(v * inv_scale + u));
            }
            __device__ static T clamp(real_t code) {
                code = code > LARGEST_CODE ? LARGEST_CODE : (code < -LARGEST_CODE ? -LARGEST_CODE : code);
                return static_cast<T>(code);
            }
            __device__ static scaling_t scale_for(const real_t max_abs) {
                if (!(max_abs > 0)) return 1;
                return static_cast<scaling_t>(max_abs / LARGEST_CODE);
            }
        };

        /// The deviations from the means a mean-deviation store keeps (sfem_HistoryStorage.hpp's
        /// MeanDeviationCodec), coded by the base codec.
        template <class Base>
        struct DeviceMeanDeviation : Base {
            static constexpr bool mean_deviation = true;
        };

        /// The code of v, rounded stochastically when asked and the codec is integer: the host's
        /// prony::detail::encode_rounded, with the same numbers (prony::uniform).
        template <class Codec>
        __device__ inline typename Codec::storage_t encode_rounded(const real_t    v,
                                                                   const real_t    inv,
                                                                   const bool      stochastic,
                                                                   const uint64_t  seed,
                                                                   const ptrdiff_t k) {
            if constexpr (Codec::integer) {
                if (stochastic) return Codec::encode(v, inv, prony::uniform(seed, static_cast<uint64_t>(k)));
            }
            return Codec::encode(v, inv);
        }

    }  // namespace device_codecs
}  // namespace sfem
