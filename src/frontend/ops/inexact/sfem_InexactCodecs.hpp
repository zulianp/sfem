#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "sfem_HistoryStorage.hpp"
#include "sfem_base.hpp"

namespace sfem {
    namespace inexact {

        /// How a stored tangent is encoded (docs/compressed_partial_assembly, "Compression").
        ///
        /// `Plain` keeps one scale per element and applies it to the element's outputs, which is
        /// the same number as scaling the decoded tangent, with `dim * n_nodes` multiplies rather
        /// than one per stored number.  `MeanDeviation` stores the element mean in single
        /// precision and encodes the deviations from it under one scale: the components of a
        /// tangent differ in magnitude far more than one component varies over an element, so
        /// the deviations spend an 8-bit code on what varies.
        enum class Scheme { Plain, MeanDeviation };

        /// The codes, as the encoder and the apply see them.  `double` and `float` are
        /// unscaled; `half_t` and `int8_t` reuse the history store's codecs
        /// (`sfem_HistoryStorage.hpp`), so the two stores round alike.
        template <typename code_t>
        struct CodeTraits {
            static constexpr bool scaled = false;
            static double         scale_for(double) { return 1; }
            static code_t         encode(double v, double) { return (code_t)v; }
        };

        template <>
        struct CodeTraits<half_t> {
            static constexpr bool scaled = true;
            static double         scale_for(double m) { return ScaledHalfCodec::scale_for((real_t)m); }
            static half_t encode(double v, double inv) { return ScaledHalfCodec::encode((real_t)v, (real_t)inv); }
        };

        template <>
        struct CodeTraits<int8_t> {
            static constexpr bool scaled = true;
            static double         scale_for(double m) { return Int8Codec::scale_for((real_t)m); }
            static int8_t encode(double v, double inv) { return Int8Codec::encode((real_t)v, (real_t)inv); }
        };

        /// The 8-bit factor of a node pair's scale, relative to the element's: `r = 2^-E (1 + m/8)`,
        /// exponent `E` in the high 5 bits and mantissa `m` in the low 3.  The node pairs of an
        /// element differ in magnitude by orders (corner-corner against corner-midside), which a
        /// single element scale spends an 8-bit code's range on; three mantissa bits keep the waste
        /// of rounding `r` up below one eighth, where a bare power of two wastes up to half.
        struct PairFactor {
            static SFEM_INLINE float value(const uint8_t b) {
                const int32_t bits = ((127 - (int32_t)(b >> 3)) << 23) | ((int32_t)(b & 7) << 20);
                float         r;
                std::memcpy(&r, &bits, sizeof(float));
                return r;
            }

            /// The smallest factor >= ratio (ratio in (0, 1]); ratios below 2^-31 take the smallest.
            static uint8_t encode(const double ratio) {
                if (!(ratio > std::ldexp(1.0, -31))) return uint8_t(31 << 3);
                int E = std::min(31, (int)std::ceil(-std::log2(ratio)));
                int m = (int)std::ceil((std::ldexp(ratio, E) - 1) * 8);
                if (m < 0) m = 0;
                // Guard the rounding of log2: step up the grid until the factor covers the ratio.
                while (E > 0 && (double)value((uint8_t)((E << 3) | std::min(m, 7))) < ratio) {
                    if (++m > 7) {
                        m = 0;
                        --E;
                    }
                }
                return (uint8_t)((E << 3) | std::min(m, 7));
            }
        };

    }  // namespace inexact
}  // namespace sfem
