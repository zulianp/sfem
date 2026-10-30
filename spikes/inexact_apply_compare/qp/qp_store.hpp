// The encoded quadrature-point store: what the set-up writes and the application reads.
//
// One template covers every encoding of docs/compressed_partial_assembly (Section
// "Compression"), selected at compile time:
//
//   code_t  double, float       uncompressed references, no scale
//           half_t, int8_t      the codecs of the Prony history store (sfem_HistoryStorage.hpp)
//   Scheme  Plain               one scale per element, applied to the element's outputs
//           PerComponent        one scale per (element, component), applied when decoding
//           MeanDeviation       element mean in f32 plus encoded deviations, one scale per element
//   Layout  SoA                 one array per (tensor, component) over the elements
//           Blocked             all numbers of a block of VS elements contiguous: tensor,
//                               component, lane
//   FOLD    true                the folded store of the tensor-product elements (Element::MASK):
//                               a point stores only the components its mask keeps.  The default
//                               wherever it applies (blocked, one scale per element): it is exact
//                               and measured faster for every encoding on Grace (RESULTS_QP.md)
//           false               45 numbers per point
//
// Decoding returns the value the contraction needs (eq. "decode" of the paper); for the plain
// scaled codecs the scale is left to the caller, which multiplies the outputs.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

#include "sfem_HistoryStorage.hpp"
#include "qp_contract.hpp"

namespace qp {

enum class Scheme { Plain, PerComponent, MeanDeviation };
enum class Layout { SoA, Blocked };

template <typename code_t>
struct CodeTraits {
    static constexpr bool scaled = false;
    static double scale_for(double) { return 1; }
    static code_t encode(double v, double) { return (code_t)v; }
};
template <>
struct CodeTraits<half_t> {
    static constexpr bool scaled = true;
    static double scale_for(double m) { return sfem::ScaledHalfCodec::scale_for((real_t)m); }
    static half_t encode(double v, double inv) { return sfem::ScaledHalfCodec::encode((real_t)v, (real_t)inv); }
};
template <>
struct CodeTraits<int8_t> {
    static constexpr bool scaled = true;
    static double scale_for(double m) { return sfem::Int8Codec::scale_for((real_t)m); }
    static int8_t encode(double v, double inv) { return sfem::Int8Codec::encode((real_t)v, (real_t)inv); }
};

template <class El, typename code_t, Scheme SCH, Layout LAY, int VS_,
          bool FOLD = El::foldable && LAY == Layout::Blocked && SCH != Scheme::PerComponent>
struct Store {
    using Element = El;
    using code_type = code_t;
    static constexpr int VS = VS_;
    static constexpr int NV = El::NV;
    static constexpr Scheme scheme = SCH;
    static constexpr Layout layout = LAY;
    static constexpr bool scaled = CodeTraits<code_t>::scaled;
    // Only the plain scaled codecs leave the scale to the outputs.
    static constexpr bool scale_on_outputs = scaled && SCH == Scheme::Plain;
    static constexpr bool has_aux = SCH == Scheme::MeanDeviation;
    static constexpr bool per_component = SCH == Scheme::PerComponent;
    static constexpr bool is_element_matrix = false;
    static constexpr bool folded = FOLD;
    // Stored numbers per element.
    static constexpr int NSE = FOLD ? El::NS : NV * 45;
    static_assert(SCH == Scheme::Plain || scaled, "the deviation and per-component schemes need a scaled code");
    static_assert(!FOLD || (El::foldable && LAY == Layout::Blocked && SCH != Scheme::PerComponent),
                  "the folded store is blocked, on a tensor-product element, with one scale per element");

    ptrdiff_t nelements = 0, stride = 0, nblocks = 0;
    std::vector<code_t> codes;
    std::vector<float> scales;  // per element, or per (element, component)
    std::vector<float> aux;     // mean, per (element, component)

    explicit Store(ptrdiff_t ne) : nelements(ne) {
        nblocks = (ne + VS - 1) / VS;
        stride = LAY == Layout::SoA ? ne + 64 : nblocks * VS;
        codes.assign((size_t)NSE * stride, code_t(0));
        if (scaled) scales.assign((size_t)(per_component ? 45 : 1) * stride, 1.0f);
        if (has_aux) aux.assign((size_t)45 * stride, 0.0f);
    }

    std::size_t bytes() const {
        return codes.size() * sizeof(code_t) + scales.size() * sizeof(float) + aux.size() * sizeof(float);
    }
    // Bytes per element as the model counts them, without padding.
    static constexpr std::size_t bytes_per_element() {
        return (std::size_t)NSE * sizeof(code_t) + (scaled ? (per_component ? 45 : 1) * sizeof(float) : 0) +
               (has_aux ? 45 * sizeof(float) : 0);
    }

    // Element e = e0 + lane.  In the blocked layout a run of lanes is unit-stride only while it
    // stays inside one store block; callers split their element blocks at multiples of VS.
    SFEM_INLINE std::size_t code_index(ptrdiff_t e0, int lane, int j, int c) const {
        const ptrdiff_t e = e0 + lane;
        if (LAY == Layout::SoA) return (size_t)(j * 45 + c) * stride + e;
        return ((size_t)(e / VS) * NV + j) * 45 * VS + (size_t)c * VS + (size_t)(e % VS);
    }
    // The folded store: the k-th number point j keeps.
    SFEM_INLINE std::size_t fold_index(ptrdiff_t e0, int lane, int j, int k) const {
        const ptrdiff_t e = e0 + lane;
        return ((size_t)(e / VS) * NSE + El::OFF[j] + k) * VS + (size_t)(e % VS);
    }
    SFEM_INLINE std::size_t comp_index(ptrdiff_t e0, int lane, int c) const {
        const ptrdiff_t e = e0 + lane;
        if (LAY == Layout::SoA) return (size_t)c * stride + e;
        return ((size_t)(e / VS) * 45 + c) * VS + (size_t)(e % VS);
    }

    // ------------------------------------------------------------------- set-up side

    // Called by the set-up kernel for every weighted point tangent v of component c.
    template <typename s_t>
    static SFEM_INLINE void accumulate(const int q, s_t bT[NV][45][VS], const int c, const int lane, const s_t v) {
        if constexpr (El::moments) {
            for (int j = 0; j < NV; ++j) bT[j][c][lane] += s_t(El::FACTOR[q][j]) * v;
        } else {
            bT[q][c][lane] = v;
        }
    }

    template <typename s_t>
    void encode_block(const ptrdiff_t evb, const int ne, const s_t bT[NV][45][VS]) {
        for (int lane = 0; lane < ne; ++lane) {
            double T[NV][45];
            for (int j = 0; j < NV; ++j)
                for (int c = 0; c < 45; ++c) T[j][c] = (double)bT[j][c][lane];

            if constexpr (has_aux) {
                double mean[45];
                for (int c = 0; c < 45; ++c) {
                    double acc = 0;
                    for (int j = 0; j < NV; ++j) acc += El::MULT[j] * T[j][c];
                    mean[c] = acc / El::KVOL;
                }
                for (int c = 0; c < 45; ++c) {
                    const float mf = (float)mean[c];
                    aux[comp_index(evb, lane, c)] = mf;
                    for (int j = 0; j < NV; ++j) T[j][c] -= El::X[j] * (double)mf;
                }
            }

            if constexpr (FOLD) {
                fold(T);
                double top = 0;
                for (int j = 0; j < NV; ++j)
                    for (int k = 0; k < El::NKEEP[El::MASK[j]]; ++k)
                        top = std::max(top, std::fabs(T[j][El::KEEP[El::MASK[j]][k]]));
                double inv = 1;
                if constexpr (scaled) {
                    const double s = CodeTraits<code_t>::scale_for(top);
                    scales[(size_t)evb + lane] = (float)s;
                    inv = 1.0 / (double)(float)s;
                }
                for (int j = 0; j < NV; ++j)
                    for (int k = 0; k < El::NKEEP[El::MASK[j]]; ++k)
                        codes[fold_index(evb, lane, j, k)] = CodeTraits<code_t>::encode(T[j][El::KEEP[El::MASK[j]][k]], inv);
                continue;
            }

            if constexpr (scaled) {
                if constexpr (per_component) {
                    for (int c = 0; c < 45; ++c) {
                        double top = 0;
                        for (int j = 0; j < NV; ++j) top = std::max(top, std::fabs(T[j][c]));
                        const double s = CodeTraits<code_t>::scale_for(top);
                        scales[comp_index(evb, lane, c)] = (float)s;
                        const double inv = 1.0 / (double)(float)s;
                        for (int j = 0; j < NV; ++j) codes[code_index(evb, lane, j, c)] = CodeTraits<code_t>::encode(T[j][c], inv);
                    }
                } else {
                    double top = 0;
                    for (int j = 0; j < NV; ++j)
                        for (int c = 0; c < 45; ++c) top = std::max(top, std::fabs(T[j][c]));
                    const double s = CodeTraits<code_t>::scale_for(top);
                    scales[(size_t)evb + lane] = (float)s;
                    const double inv = 1.0 / (double)(float)s;
                    for (int j = 0; j < NV; ++j)
                        for (int c = 0; c < 45; ++c) codes[code_index(evb, lane, j, c)] = CodeTraits<code_t>::encode(T[j][c], inv);
                }
            } else {
                for (int j = 0; j < NV; ++j)
                    for (int c = 0; c < 45; ++c) codes[code_index(evb, lane, j, c)] = (code_t)T[j][c];
            }
        }
    }

    // The fold (Element::MASK): along each line in direction d, the dropped point's m = n = d
    // components move to the line's kept points.  Linear, so it commutes with removing the mean.
    static void fold(double T[NV][45]) {
        if constexpr (FOLD) {
            for (int j = 0; j < NV; ++j)
                for (int c = 0; c < 45; ++c) {
                    const int d = El::DIR[c];
                    if (d >= 0 && !((El::MASK[j] >> d) & 1)) T[j][c] += El::FCOEF[j][d] * T[El::FSRC[j][d]][c];
                }
        }
    }

    // ------------------------------------------------------------- application side

    SFEM_INLINE double output_scale(const ptrdiff_t e) const { return scales[(size_t)e]; }
};

}  // namespace qp
