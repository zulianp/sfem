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
//           Whitened            deviations in the coordinates of the (regularised) element mean
//   Layout  SoA                 one array per (tensor, component) over the elements
//           Blocked             all numbers of a block of VS elements contiguous: tensor,
//                               component, lane
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

enum class Scheme { Plain, PerComponent, MeanDeviation, Whitened };
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

// Packed lower triangle of a 9x9 matrix, row-major: entry (r, c), c <= r.
static constexpr int tri(int r, int c) { return r * (r + 1) / 2 + c; }

template <class El, typename code_t, Scheme SCH, Layout LAY, int VS_>
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
    static constexpr bool has_aux = SCH == Scheme::MeanDeviation || SCH == Scheme::Whitened;
    static constexpr bool per_component = SCH == Scheme::PerComponent;
    static constexpr bool is_element_matrix = false;
    static_assert(SCH == Scheme::Plain || scaled, "the deviation and per-component schemes need a scaled code");

    ptrdiff_t nelements = 0, stride = 0, nblocks = 0;
    double whitening_shift = 1e-3;  // relative regularisation of the element mean (Whitened)
    std::vector<code_t> codes;
    std::vector<float> scales;  // per element, or per (element, component)
    std::vector<float> aux;     // mean, or packed Cholesky factor, per (element, component)

    explicit Store(ptrdiff_t ne) : nelements(ne) {
        nblocks = (ne + VS - 1) / VS;
        stride = LAY == Layout::SoA ? ne + 64 : nblocks * VS;
        codes.assign((size_t)NV * 45 * stride, code_t(0));
        if (scaled) scales.assign((size_t)(per_component ? 45 : 1) * stride, 1.0f);
        if (has_aux) aux.assign((size_t)45 * stride, 0.0f);
    }

    std::size_t bytes() const {
        return codes.size() * sizeof(code_t) + scales.size() * sizeof(float) + aux.size() * sizeof(float);
    }
    // Bytes per element as the model counts them, without padding.
    static constexpr std::size_t bytes_per_element() {
        return (std::size_t)NV * 45 * sizeof(code_t) + (scaled ? (per_component ? 45 : 1) * sizeof(float) : 0) +
               (has_aux ? 45 * sizeof(float) : 0);
    }

    // Element e = e0 + lane.  In the blocked layout a run of lanes is unit-stride only while it
    // stays inside one store block; callers split their element blocks at multiples of VS.
    SFEM_INLINE std::size_t code_index(ptrdiff_t e0, int lane, int j, int c) const {
        const ptrdiff_t e = e0 + lane;
        if (LAY == Layout::SoA) return (size_t)(j * 45 + c) * stride + e;
        return ((size_t)(e / VS) * NV + j) * 45 * VS + (size_t)c * VS + (size_t)(e % VS);
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
                if constexpr (SCH == Scheme::MeanDeviation) {
                    for (int c = 0; c < 45; ++c) {
                        const float mf = (float)mean[c];
                        aux[comp_index(evb, lane, c)] = mf;
                        for (int j = 0; j < NV; ++j) T[j][c] -= El::X[j] * (double)mf;
                    }
                } else {
                    whiten(evb, lane, mean, T);
                }
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

    // W_j = L^{-1} T_j L^{-T} - X_j I with L the Cholesky factor of the mean shifted by
    // whitening_shift * trace / 9 (the mean is singular on rotations at a stress-free state).
    // L is rounded to single precision first and the transform uses the rounded factor, so
    // that the application, which reads the rounded one, inverts exactly this.
    void whiten(const ptrdiff_t evb, const int lane, const double mean[45], double T[NV][45]) {
        double A[9][9];
        double tr = 0;
        for (int r = 0; r < 9; ++r) tr += mean[SLOT[r][r]];
        const double shift = whitening_shift * tr / 9;
        for (int r = 0; r < 9; ++r)
            for (int c = 0; c < 9; ++c) A[r][c] = mean[SLOT[r][c]] + (r == c ? shift : 0.0);
        double L[9][9] = {};
        for (int c = 0; c < 9; ++c) {
            double d = A[c][c];
            for (int k = 0; k < c; ++k) d -= L[c][k] * L[c][k];
            if (!(d > 0)) d = shift > 0 ? shift : 1e-300;  // indefinite mean: clamp the pivot
            L[c][c] = std::sqrt(d);
            for (int r = c + 1; r < 9; ++r) {
                double v = A[r][c];
                for (int k = 0; k < c; ++k) v -= L[r][k] * L[c][k];
                L[r][c] = v / L[c][c];
            }
        }
        for (int r = 0; r < 9; ++r)
            for (int c = 0; c <= r; ++c) {
                const float lf = (float)L[r][c];
                aux[comp_index(evb, lane, tri(r, c))] = lf;
                L[r][c] = (double)lf;
            }
        for (int j = 0; j < NV; ++j) {
            double M[9][9];
            for (int r = 0; r < 9; ++r)
                for (int c = 0; c < 9; ++c) M[r][c] = T[j][SLOT[r][c]];
            // M <- L^{-1} M  (forward substitution on columns), then M <- M L^{-T} (on rows).
            for (int c = 0; c < 9; ++c)
                for (int r = 0; r < 9; ++r) {
                    double v = M[r][c];
                    for (int k = 0; k < r; ++k) v -= L[r][k] * M[k][c];
                    M[r][c] = v / L[r][r];
                }
            for (int r = 0; r < 9; ++r)
                for (int c = 0; c < 9; ++c) {
                    double v = M[r][c];
                    for (int k = 0; k < c; ++k) v -= M[r][k] * L[c][k];
                    M[r][c] = v / L[c][c];
                }
            for (int r = 0; r < 9; ++r)
                for (int c = r; c < 9; ++c) T[j][SLOT[r][c]] = 0.5 * (M[r][c] + M[c][r]) - (r == c ? El::X[j] : 0.0);
        }
    }

    // ------------------------------------------------------------- application side

    template <typename tau>
    SFEM_INLINE void load_factor(const ptrdiff_t e0, const int lane, tau Lp[45]) const {
        for (int c = 0; c < 45; ++c) Lp[c] = tau(aux[comp_index(e0, lane, c)]);
    }

    SFEM_INLINE double output_scale(const ptrdiff_t e) const { return scales[(size_t)e]; }
};

}  // namespace qp
