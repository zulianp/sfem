#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <string>

#include "sfem_base.hpp"
#include "sfem_config.h"
#include "sfem_logger.hpp"

namespace sfem {

    /// How integration-point history is stored.
    ///
    /// Arithmetic always runs in `real_t`; only the store is reduced.  The
    /// history of a long Prony series dominates the memory of a viscoelastic
    /// solve -- m terms cost m * n_components * n_qp values per element -- so
    /// the store is where precision is worth trading, and where it is traded
    /// is the caller's choice.  `FULL` is the default.
    ///
    /// The scaled formats carry one `scaling_t` per stored tensor and element,
    /// derived from the max-abs of that tensor over its components and
    /// integration points and recomputed whenever the tensor is written.
    enum class HistoryStorage {
        FULL,            ///< real_t
        FLOAT32,         ///< float
        FLOAT16,         ///< half_t, unscaled: overflows to inf above 65504, so only for values known to stay below it
        FLOAT16_SCALED,  ///< half_t with a power-of-two scale per tensor and element
        INT16_SCALED,    ///< symmetric 16-bit quantisation, one scale per tensor and element
        INT8_SCALED      ///< symmetric 8-bit quantisation, one scale per tensor and element
    };

    HistoryStorage history_storage_from_string(const std::string &name);
    const char    *to_string(const HistoryStorage storage);
    std::size_t    bytes_per_value(const HistoryStorage storage);
    bool           is_scaled(const HistoryStorage storage);

    /// A codec is a compile-time policy for the pointwise loops: the loops are
    /// templated on it and the runtime `HistoryStorage` picks one once, at the
    /// boundary, in `dispatch_history_codec` -- never per value.
    ///
    /// encode(v, inv_scale) and decode(s, scale) take the scale of the tensor
    /// the value belongs to; unscaled codecs ignore it.
    template <typename T>
    struct CastCodec {
        using storage_t                = T;
        static constexpr bool   scaled = false;
        static inline real_t    decode(const T s, const real_t) { return static_cast<real_t>(s); }
        static inline T         encode(const real_t v, const real_t) { return static_cast<T>(v); }
        static inline scaling_t scale_for(const real_t) { return 1; }
    };

    /// half_t under a power-of-two scale: the largest magnitude of the tensor
    /// is mapped to at most 2^14, which leaves headroom below FP16_MAX and keeps
    /// the smallest normal half at 2^-28 of it.  A power of two makes the
    /// rescaling exact, so the only error is half's own rounding.
    struct ScaledHalfCodec {
        using storage_t              = half_t;
        static constexpr bool scaled = true;

        static inline real_t decode(const half_t s, const real_t scale) { return static_cast<real_t>(s) * scale; }
        static inline half_t encode(const real_t v, const real_t inv_scale) { return static_cast<half_t>(v * inv_scale); }

        static inline scaling_t scale_for(const real_t max_abs) {
            if (!(max_abs > 0)) return 1;
            // In double, as on the device, so both pick the same exponent in a float build.
            return static_cast<scaling_t>(std::exp2(std::ceil(std::log2(static_cast<double>(max_abs))) - 14));
        }
    };

    /// Symmetric integer quantisation: max_abs maps to the largest code, and
    /// the rounding error is at most scale / 2.
    template <typename T, int LARGEST_CODE>
    struct ScaledIntegerCodec {
        using storage_t              = T;
        static constexpr bool scaled = true;

        static inline real_t decode(const T s, const real_t scale) { return static_cast<real_t>(s) * scale; }

        static inline T encode(const real_t v, const real_t inv_scale) {
            real_t code = std::nearbyint(v * inv_scale);
            code        = code > LARGEST_CODE ? LARGEST_CODE : (code < -LARGEST_CODE ? -LARGEST_CODE : code);
            return static_cast<T>(code);
        }

        static inline scaling_t scale_for(const real_t max_abs) {
            if (!(max_abs > 0)) return 1;
            return static_cast<scaling_t>(max_abs / LARGEST_CODE);
        }
    };

    using FullCodec    = CastCodec<real_t>;
    using Float32Codec = CastCodec<float>;
    using Float16Codec = CastCodec<half_t>;
    using Int16Codec   = ScaledIntegerCodec<int16_t, 32767>;
    using Int8Codec    = ScaledIntegerCodec<int8_t, 127>;

    /// Call `f(Codec{})` with the codec `storage` names.
    template <class F>
    decltype(auto) dispatch_history_codec(const HistoryStorage storage, F &&f) {
        switch (storage) {
            case HistoryStorage::FULL:
                return f(FullCodec{});
            case HistoryStorage::FLOAT32:
                return f(Float32Codec{});
            case HistoryStorage::FLOAT16:
                return f(Float16Codec{});
            case HistoryStorage::FLOAT16_SCALED:
                return f(ScaledHalfCodec{});
            case HistoryStorage::INT16_SCALED:
                return f(Int16Codec{});
            case HistoryStorage::INT8_SCALED:
                return f(Int8Codec{});
        }

        SFEM_ERROR("dispatch_history_codec: unknown storage %d\n", static_cast<int>(storage));
        return f(FullCodec{});
    }

}  // namespace sfem
