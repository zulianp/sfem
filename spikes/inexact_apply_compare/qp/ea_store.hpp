// Element-matrix storage (element assembly), encoded like the quadrature-point store.
//
// The same set-up kernel feeds it: the point tangents (or moments) of a block arrive in
// `encode_block`, which builds each element matrix directly from them,
//
//   K[(i,p),(k,q)] = sum_tensors sum_{n,m} DT[b][n][p] S[(i,n),(k,m)] DT[a][m][q]   (+ (a,b) swapped),
//
// and encodes it.  Two forms:
//
//   Full        the upper triangle of the symmetric (3N)x(3N) matrix; y = K h
//   Difference  only the off-diagonal 3x3 node blocks K_pq, p < q.  An element matrix annihilates
//               rigid translations, so its diagonal blocks are minus the sums of the off-diagonal
//               ones, and y_p = sum_{q != p} K_pq (h_q - h_p): fewer numbers, and the translations
//               stay in the kernel of the encoded operator whatever the rounding.
//   PairScale   the difference form with a scale per node-pair block: the element scale times an
//               8-bit float factor r <= 1 (5 exponent bits, 3 mantissa bits, r = 2^-E (1 + m/8)),
//               rounded up so the pair's largest entry still fits the largest code.  The node pairs
//               of an element differ in magnitude by orders (corner-corner against corner-midside),
//               which a single element scale spends int8's range on; the 3 mantissa bits keep the
//               waste of rounding r up below 1/8 (a bare power of two wastes up to half the range).
//
// Codes as in qp_store.hpp; one scale per element, applied to the outputs (PairScale: the pair factors
// are applied to the differences).
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

#include "qp_contract.hpp"
#include "qp_element.hpp"
#include "qp_store.hpp"

namespace qp {

enum class EaForm { Full, Difference, PairScale };

template <class El, typename code_t, EaForm F, int VS_>
struct EaStore {
    using Element = El;
    using code_type = code_t;
    static constexpr int VS = VS_;
    static constexpr int NV = El::NV;
    static constexpr int N = El::N;
    static constexpr int NDOF = 3 * N;
    static constexpr EaForm form = F;
    static constexpr bool is_element_matrix = true;
    static constexpr bool per_pair = F == EaForm::PairScale;
    static constexpr bool scaled = CodeTraits<code_t>::scaled;  // one scale per element
    static constexpr int NP = N * (N - 1) / 2;
    static constexpr int NE = F == EaForm::Full ? NDOF * (NDOF + 1) / 2 : 9 * NP;
    static_assert(!per_pair || CodeTraits<code_t>::scaled, "pair exponents need a scaled code");

    ptrdiff_t nelements = 0, nblocks = 0;
    std::vector<code_t> codes;  // [block][entry][lane]
    std::vector<float> scales;  // per element
    std::vector<uint8_t> pair_factors;  // per node pair, [block][pair][lane] (PairScale)

    explicit EaStore(ptrdiff_t ne) : nelements(ne) {
        nblocks = (ne + VS - 1) / VS;
        codes.assign((size_t)nblocks * NE * VS, code_t(0));
        if (scaled) scales.assign((size_t)nblocks * VS, 1.0f);
        if (per_pair) pair_factors.assign((size_t)nblocks * NP * VS, uint8_t(0));
    }
    static constexpr std::size_t bytes_per_element() {
        return (std::size_t)NE * sizeof(code_t) + (scaled ? sizeof(float) : 0) + (per_pair ? NP : 0);
    }
    // The pair factor's value, from its bits: exponent E in the high 5 bits, mantissa m in the low 3.
    static SFEM_INLINE float pair_factor(const uint8_t b) {
        const int32_t bits = ((127 - (int32_t)(b >> 3)) << 23) | ((int32_t)(b & 7) << 20);
        float r;
        std::memcpy(&r, &bits, sizeof(float));
        return r;
    }
    // The smallest factor >= ratio (ratio in (0, 1]); ratios below 2^-31 take the smallest factor.
    static uint8_t pair_factor_for(const double ratio) {
        if (!(ratio > std::ldexp(1.0, -31))) return uint8_t(31 << 3);
        // ratio * 2^E in [1, 2), so ratio <= 2^-E (1 + m/8) with m in 0..8 (8 carries into E - 1).
        int E = std::min(31, (int)std::ceil(-std::log2(ratio)));
        int m = (int)std::ceil((std::ldexp(ratio, E) - 1) * 8);
        if (m < 0) m = 0;
        // Guard the rounding of log2: step up the grid until the factor covers the ratio.
        while (E > 0 && (double)pair_factor((uint8_t)((E << 3) | std::min(m, 7))) < ratio) {
            if (++m > 7) {
                m = 0;
                --E;
            }
        }
        return (uint8_t)((E << 3) | std::min(m, 7));
    }
    SFEM_INLINE std::size_t code_index(ptrdiff_t e, int entry) const {
        return ((size_t)(e / VS) * NE + entry) * VS + (size_t)(e % VS);
    }

    // Full form: entry -> (row, column), row <= column, rows and columns as i*N + p.
    // Difference form: entry t*9 + i*3 + k -> pair t = (p, q), p < q.
    struct Tables {
        std::array<int, (F == EaForm::Full ? NE : 1)> row{}, col{};
        std::array<int, NP + 1> p{}, q{};
    };
    static const Tables &tables() {
        static const Tables t = [] {
            Tables t;
            if constexpr (F == EaForm::Full) {
                int e = 0;
                for (int r = 0; r < NDOF; ++r)
                    for (int c = r; c < NDOF; ++c, ++e) {
                        t.row[e] = r;
                        t.col[e] = c;
                    }
            }
            int k = 0;
            for (int a = 0; a < N; ++a)
                for (int b = a + 1; b < N; ++b, ++k) {
                    t.p[k] = a;
                    t.q[k] = b;
                }
            return t;
        }();
        return t;
    }

    // ------------------------------------------------------------------- set-up side

    template <typename s_t>
    static SFEM_INLINE void accumulate(const int q, s_t bT[NV][45][VS], const int c, const int lane, const s_t v) {
        if constexpr (El::moments) {
            for (int j = 0; j < NV; ++j) bT[j][c][lane] += s_t(El::FACTOR[q][j]) * v;
        } else {
            bT[q][c][lane] = v;
        }
    }

    // The difference forms' off-diagonal blocks for every lane of the block at once:
    //   T[q][(i,n)][k] = sum_m S[(i,n),(k,m)] DT[a][m][q],   K_pq[i][k] += sum_n DT[b][n][p] T[q][(i,n)][k],
    // in single precision for the 16- and 8-bit codes, in double otherwise.  K is symmetric by
    // construction (both orders of a pair a != b are summed and S has major symmetry), so only the
    // blocks p < q are formed.
    template <typename s_t>
    void encode_block_difference(const ptrdiff_t evb, const int ne, const s_t bT[NV][45][VS]) {
        using w_t = std::conditional_t<(sizeof(code_t) <= 2), float, double>;
        const Tables &t = tables();
        static thread_local std::vector<w_t> work;
        work.resize((size_t)(N * 27 + NP * 9) * VS);
        w_t(*const T)[27][VS] = reinterpret_cast<w_t(*)[27][VS]>(work.data());
        w_t(*const Kd)[9][VS] = reinterpret_cast<w_t(*)[9][VS]>(work.data() + (size_t)N * 27 * VS);
        for (int k = 0; k < NP; ++k)
            for (int x = 0; x < 9; ++x)
#pragma omp simd
                for (int lane = 0; lane < VS; ++lane) Kd[k][x][lane] = 0;
        for (int j = 0; j < NV; ++j) {
            for (int pass = 0; pass < (El::PA[j] == El::PB[j] ? 1 : 2); ++pass) {
                const int a = pass == 0 ? El::PA[j] : El::PB[j];
                const int b = pass == 0 ? El::PB[j] : El::PA[j];
                for (int q = 0; q < N; ++q)
                    for (int in = 0; in < 9; ++in)
                        for (int k = 0; k < 3; ++k) {
                            const w_t d0 = w_t(El::DT[a][0][q]), d1 = w_t(El::DT[a][1][q]), d2 = w_t(El::DT[a][2][q]);
                            const s_t *const RSTR s0 = bT[j][SLOT[in][k * 3 + 0]];
                            const s_t *const RSTR s1 = bT[j][SLOT[in][k * 3 + 1]];
                            const s_t *const RSTR s2 = bT[j][SLOT[in][k * 3 + 2]];
#pragma omp simd
                            for (int lane = 0; lane < VS; ++lane)
                                T[q][in * 3 + k][lane] = d0 * w_t(s0[lane]) + d1 * w_t(s1[lane]) + d2 * w_t(s2[lane]);
                        }
                for (int k = 0; k < NP; ++k) {
                    const int p = t.p[k], q = t.q[k];
                    const w_t e0 = w_t(El::DT[b][0][p]), e1 = w_t(El::DT[b][1][p]), e2 = w_t(El::DT[b][2][p]);
                    for (int i = 0; i < 3; ++i)
                        for (int kk = 0; kk < 3; ++kk)
#pragma omp simd
                            for (int lane = 0; lane < VS; ++lane)
                                Kd[k][i * 3 + kk][lane] += e0 * T[q][(i * 3 + 0) * 3 + kk][lane] +
                                                           e1 * T[q][(i * 3 + 1) * 3 + kk][lane] +
                                                           e2 * T[q][(i * 3 + 2) * 3 + kk][lane];
                }
            }
        }
        for (int lane = 0; lane < ne; ++lane) {
            double vals[NE];
            for (int k = 0; k < NP; ++k)
                for (int x = 0; x < 9; ++x) vals[k * 9 + x] = (double)Kd[k][x][lane];
            encode_values(evb + lane, vals);
        }
    }

    // Scale and encode one element's stored numbers.
    void encode_values(const ptrdiff_t e, const double *const vals) {
        double s = 1, inv = 1;
        if constexpr (scaled) {
            double top = 0;
            for (int k = 0; k < NE; ++k) top = std::max(top, std::fabs(vals[k]));
            s = CodeTraits<code_t>::scale_for(top);
            scales[(size_t)e] = (float)s;
            inv = 1.0 / (double)(float)s;
        }
        if constexpr (per_pair) {
            double top = 0;
            for (int k = 0; k < NE; ++k) top = std::max(top, std::fabs(vals[k]));
            for (int k = 0; k < NP; ++k) {
                double pt = 0;
                for (int y = 0; y < 9; ++y) pt = std::max(pt, std::fabs(vals[k * 9 + y]));
                const uint8_t b = pair_factor_for(top > 0 ? pt / top : 0.0);
                pair_factors[((size_t)(e / VS) * NP + k) * VS + (size_t)(e % VS)] = b;
                const double pinv = inv / (double)pair_factor(b);
                for (int y = 0; y < 9; ++y) codes[code_index(e, k * 9 + y)] = CodeTraits<code_t>::encode(vals[k * 9 + y], pinv);
            }
            return;
        }
        for (int k = 0; k < NE; ++k) codes[code_index(e, k)] = CodeTraits<code_t>::encode(vals[k], inv);
    }

    template <typename s_t>
    void encode_block(const ptrdiff_t evb, const int ne, const s_t bT[NV][45][VS]) {
        if constexpr (F != EaForm::Full) {
            encode_block_difference(evb, ne, bT);
            return;
        }
        const Tables &t = tables();
        for (int lane = 0; lane < ne; ++lane) {
            static thread_local double K[NDOF][NDOF];
            for (int r = 0; r < NDOF; ++r)
                for (int c = 0; c < NDOF; ++c) K[r][c] = 0;
            for (int j = 0; j < NV; ++j) {
                for (int pass = 0; pass < (El::PA[j] == El::PB[j] ? 1 : 2); ++pass) {
                    const int a = pass == 0 ? El::PA[j] : El::PB[j];
                    const int b = pass == 0 ? El::PB[j] : El::PA[j];
                    // tmp[(i,n)][(k,q)] = sum_m S[(i,n),(k,m)] DT[a][m][q]
                    double tmp[9][NDOF];
                    for (int in = 0; in < 9; ++in)
                        for (int k = 0; k < 3; ++k)
                            for (int q = 0; q < N; ++q) {
                                double acc = 0;
                                for (int m = 0; m < 3; ++m) acc += (double)bT[j][SLOT[in][k * 3 + m]][lane] * El::DT[a][m][q];
                                tmp[in][k * N + q] = acc;
                            }
                    for (int i = 0; i < 3; ++i)
                        for (int p = 0; p < N; ++p)
                            for (int c = 0; c < NDOF; ++c) {
                                double acc = 0;
                                for (int n = 0; n < 3; ++n) acc += El::DT[b][n][p] * tmp[i * 3 + n][c];
                                K[i * N + p][c] += acc;
                            }
                }
            }
            double vals[NE];
            for (int e = 0; e < NE; ++e) vals[e] = 0.5 * (K[t.row[e]][t.col[e]] + K[t.col[e]][t.row[e]]);
            encode_values(evb + lane, vals);
        }
    }

    // ------------------------------------------------------------- application side

    // Elements evb .. evb + ne, inside one store block.  bh and out are [component][node][lane].
    template <typename tau>
    void block(const ptrdiff_t evb, const int ne, const double bh[3][N][VS], double out[3][N][VS]) const {
        const Tables &t = tables();
        const code_t *const RSTR c0 = codes.data() + (size_t)(evb / VS) * NE * VS + (size_t)(evb % VS);
        tau Y[NDOF][VS];
        for (int r = 0; r < NDOF; ++r)
#pragma omp simd
            for (int lane = 0; lane < VS; ++lane) Y[r][lane] = 0;
        if constexpr (F == EaForm::Full) {
            tau H[NDOF][VS];
            for (int i = 0; i < 3; ++i)
                for (int p = 0; p < N; ++p)
#pragma omp simd
                    for (int lane = 0; lane < ne; ++lane) H[i * N + p][lane] = tau(bh[i][p][lane]);
            for (int e = 0; e < NE; ++e) {
                const int r = t.row[e], c = t.col[e];
                const code_t *const RSTR ce = c0 + (size_t)e * VS;
                if (r == c) {
#pragma omp simd
                    for (int lane = 0; lane < ne; ++lane) Y[r][lane] += tau(ce[lane]) * H[c][lane];
                } else {
#pragma omp simd
                    for (int lane = 0; lane < ne; ++lane) {
                        const tau v = tau(ce[lane]);
                        Y[r][lane] += v * H[c][lane];
                        Y[c][lane] += v * H[r][lane];
                    }
                }
            }
        } else {
            const uint8_t *const RSTR r0 = per_pair ? pair_factors.data() + (size_t)(evb / VS) * NP * VS + (size_t)(evb % VS) : nullptr;
            for (int k = 0; k < NP; ++k) {
                const int p = t.p[k], q = t.q[k];
                const code_t *const RSTR ck = c0 + (size_t)k * 9 * VS;
#pragma omp simd
                for (int lane = 0; lane < ne; ++lane) {
                    // Differences in the vector's precision, so a translation cancels exactly.
                    tau d0 = tau(bh[0][q][lane] - bh[0][p][lane]);
                    tau d1 = tau(bh[1][q][lane] - bh[1][p][lane]);
                    tau d2 = tau(bh[2][q][lane] - bh[2][p][lane]);
                    if constexpr (per_pair) {
                        // The pair factor goes on the differences (3 multiplies).
                        const tau r = tau(pair_factor(r0[(size_t)k * VS + lane]));
                        d0 *= r;
                        d1 *= r;
                        d2 *= r;
                    }
                    tau K[9];
                    for (int x = 0; x < 9; ++x) K[x] = tau(ck[x * VS + lane]);
                    Y[0 * N + p][lane] += K[0] * d0 + K[1] * d1 + K[2] * d2;
                    Y[1 * N + p][lane] += K[3] * d0 + K[4] * d1 + K[5] * d2;
                    Y[2 * N + p][lane] += K[6] * d0 + K[7] * d1 + K[8] * d2;
                    Y[0 * N + q][lane] -= K[0] * d0 + K[3] * d1 + K[6] * d2;
                    Y[1 * N + q][lane] -= K[1] * d0 + K[4] * d1 + K[7] * d2;
                    Y[2 * N + q][lane] -= K[2] * d0 + K[5] * d1 + K[8] * d2;
                }
            }
        }
        for (int i = 0; i < 3; ++i)
            for (int p = 0; p < N; ++p)
#pragma omp simd
                for (int lane = 0; lane < ne; ++lane) {
                    double v = (double)Y[i * N + p][lane];
                    if constexpr (scaled) v *= (double)scales[(size_t)evb + lane];
                    out[i][p][lane] = v;
                }
    }
};

}  // namespace qp
