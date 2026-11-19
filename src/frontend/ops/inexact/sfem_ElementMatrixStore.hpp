#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "packed_thread_scratch.hpp"
#include "sfem_InexactCodecs.hpp"
#include "sfem_InexactElement.hpp"
#include "sfem_aliases.hpp"

namespace sfem {
    namespace inexact {

        /// Element matrices in difference form, encoded like the point store.
        ///
        /// The same set-up feeds it: a block's point tangents (or moments) arrive in
        /// `encode_block`, which builds each element matrix from them,
        ///
        ///   K[(i,p),(k,q)] = sum_tensors sum_{n,m} DT[b][n][p] S[(i,n),(k,m)] DT[a][m][q]  (+ (a,b) swapped),
        ///
        /// and stores only its off-diagonal 3x3 node blocks.  An operator that depends on the state
        /// only through its gradient annihilates translations, so a block row sums to zero, the
        /// diagonal block is minus the others and y_p = sum_{q != p} K_pq (h_q - h_p): fewer
        /// numbers, and translations stay in the kernel of the encoded operator whatever the
        /// rounding.  A symmetric tangent stores K_pq for p < q and applies its transpose to q; an
        /// unsymmetric one stores K_pq and K_qp.
        ///
        /// With `PAIR` every block carries an 8-bit factor of the element scale (`PairFactor`).
        template <class M, typename code_t, bool PAIR, int VS_>
        class ElementMatrixStore {
        public:
            using Unit        = M;
            using Element     = typename M::element;
            using Contraction = typename M::contraction;
            using code_type   = code_t;
            using Tables      = ElementTables<Element>;

            static constexpr int  VS        = VS_;
            static constexpr int  NV        = Element::NV;
            static constexpr int  N         = Element::N;
            static constexpr int  NC        = M::NC;
            static constexpr int  DIM       = Element::DIM;
            static constexpr int  ORDER     = DIM * DIM;
            static constexpr bool symmetric = M::symmetric;
            static constexpr int  NP        = N * (N - 1) / 2;
            static constexpr int  NB        = symmetric ? NP : 2 * NP;  // stored blocks
            static constexpr int  NE        = NB * DIM * DIM;
            static constexpr bool scaled    = CodeTraits<code_t>::scaled;

            static_assert(!PAIR || scaled, "pair factors need a scaled code");

            explicit ElementMatrixStore(const ptrdiff_t nelements) : nelements_(nelements) {
                nblocks_ = (nelements + VS - 1) / VS;
                codes_   = create_host_buffer<code_t>((size_t)NE * nblocks_ * VS);
                if (scaled) scales_ = create_host_buffer<float>((size_t)nblocks_ * VS);
                if (PAIR) factors_ = create_host_buffer<uint8_t>((size_t)NB * nblocks_ * VS);
                codes_p_   = codes_->data();
                scales_p_  = scales_ ? scales_->data() : nullptr;
                factors_p_ = factors_ ? factors_->data() : nullptr;
            }

            ptrdiff_t nelements() const { return nelements_; }

            std::size_t bytes() const {
                std::size_t ret = codes_->size() * sizeof(code_t);
                if (scales_) ret += scales_->size() * sizeof(float);
                if (factors_) ret += factors_->size();
                return ret;
            }

            /// Block `k`'s row and column node: p < q first, then (symmetric excepted) q > p.
            static constexpr NodePairs<N> PAIRS{};
            static SFEM_INLINE void block_nodes(const int k, int &row, int &col) {
                row = PAIRS.row[k];
                col = PAIRS.col[k];
            }

            // ---------------------------------------------------------------- set-up side

            template <typename s_t>
            static SFEM_INLINE void accumulate(const int q, s_t bT[NV][NC][VS], const int c, const int lane, const s_t v) {
                if constexpr (Element::moments) {
                    for (int j = 0; j < NV; ++j) bT[j][c][lane] += s_t(Element::FACTOR[q][j]) * v;
                } else {
                    bT[q][c][lane] = v;
                }
            }

            /// The blocks for every lane at once:
            ///   T[q][(i,n)][k] = sum_m S[(i,n),(k,m)] DT[a][m][q],   K_pq[i][k] += sum_n DT[b][n][p] T[q][(i,n)][k],
            /// in single precision for the 16- and 8-bit codes, in double otherwise.
            template <typename s_t>
            void encode_block(const ptrdiff_t evb, const int ne, const s_t bT[NV][NC][VS]) {
                using w_t = std::conditional_t<(sizeof(code_t) <= 2), float, double>;
                w_t *const work = codegen::thread_scratch<w_t>(2, (size_t)(N * DIM * ORDER + NE) * VS);
                w_t(*const T)[DIM * ORDER][VS] = reinterpret_cast<w_t(*)[DIM * ORDER][VS]>(work);
                w_t(*const Kd)[ORDER][VS]      = reinterpret_cast<w_t(*)[ORDER][VS]>(work + (size_t)N * DIM * ORDER * VS);
                for (int k = 0; k < NB; ++k)
                    for (int x = 0; x < ORDER; ++x)
#pragma omp simd
                        for (int lane = 0; lane < VS; ++lane) Kd[k][x][lane] = 0;
                for (int j = 0; j < NV; ++j) {
                    for (int pass = 0; pass < (Tables::pa(j) == Tables::pb(j) ? 1 : 2); ++pass) {
                        const int a = pass == 0 ? Tables::pa(j) : Tables::pb(j);
                        const int b = pass == 0 ? Tables::pb(j) : Tables::pa(j);
                        for (int q = 0; q < N; ++q)
                            for (int in = 0; in < ORDER; ++in)
                                for (int k = 0; k < DIM; ++k) {
                                    const w_t d0 = w_t(Element::DT[a][0][q]), d1 = w_t(Element::DT[a][1][q]),
                                              d2 = w_t(Element::DT[a][2][q]);
                                    const s_t *const RSTR s0 = bT[j][Contraction::slot(in, k * DIM + 0)];
                                    const s_t *const RSTR s1 = bT[j][Contraction::slot(in, k * DIM + 1)];
                                    const s_t *const RSTR s2 = bT[j][Contraction::slot(in, k * DIM + 2)];
#pragma omp simd
                                    for (int lane = 0; lane < VS; ++lane)
                                        T[q][in * DIM + k][lane] = d0 * w_t(s0[lane]) + d1 * w_t(s1[lane]) + d2 * w_t(s2[lane]);
                                }
                        for (int k = 0; k < NB; ++k) {
                            int p, q;
                            block_nodes(k, p, q);
                            const w_t e0 = w_t(Element::DT[b][0][p]), e1 = w_t(Element::DT[b][1][p]),
                                      e2 = w_t(Element::DT[b][2][p]);
                            for (int i = 0; i < DIM; ++i)
                                for (int kk = 0; kk < DIM; ++kk)
#pragma omp simd
                                    for (int lane = 0; lane < VS; ++lane)
                                        Kd[k][i * DIM + kk][lane] += e0 * T[q][(i * DIM + 0) * DIM + kk][lane] +
                                                                     e1 * T[q][(i * DIM + 1) * DIM + kk][lane] +
                                                                     e2 * T[q][(i * DIM + 2) * DIM + kk][lane];
                        }
                    }
                }
                for (int lane = 0; lane < ne; ++lane) {
                    double vals[NE];
                    for (int k = 0; k < NB; ++k)
                        for (int x = 0; x < ORDER; ++x) vals[k * ORDER + x] = (double)Kd[k][x][lane];
                    encode_values(evb + lane, vals);
                }
            }

            // ------------------------------------------------------------ application side

            /// The element matrix's diagonal for elements evb .. evb + ne: the diagonal blocks are
            /// minus the off-diagonal blocks of their row (K_pp = -sum_{q != p} K_pq), so their
            /// diagonal entries are minus the stored blocks' diagonals touching p.
            void diagonal_block(const ptrdiff_t evb, const int ne, real_t out[DIM][N][VS]) const {
                const code_t *const RSTR  codes   = codes_p_;
                const uint8_t *const RSTR factors = PAIR ? factors_p_ : nullptr;
                for (int lane = 0; lane < ne; ++lane) {
                    const ptrdiff_t e = evb + lane;
                    double          d[DIM][N] = {{0}};
                    const double    s = scaled ? (double)scales_p_[(size_t)e] : 1.0;
                    for (int k = 0; k < NB; ++k) {
                        int p, q;
                        block_nodes(k, p, q);
                        const double r =
                                PAIR ? (double)PairFactor::value(factors[((size_t)(e / VS) * NB + k) * VS + (size_t)(e % VS)]) : 1.0;
                        for (int i = 0; i < DIM; ++i) {
                            const double v = s * r * (double)codes[code_index(e, k * ORDER + i * DIM + i)];
                            d[i][p] -= v;
                            if constexpr (symmetric) d[i][q] -= v;
                        }
                    }
                    for (int i = 0; i < DIM; ++i)
                        for (int p = 0; p < N; ++p) out[i][p][lane] = (real_t)d[i][p];
                }
            }

            /// Elements evb .. evb + ne, inside one store block.
            template <typename tau>
            SFEM_INLINE void block(const ptrdiff_t evb, const int ne, const real_t bh[DIM][N][VS], real_t out[DIM][N][VS]) const {
                const code_t *const RSTR  c0 = codes_p_ + (size_t)(evb / VS) * NE * VS + (size_t)(evb % VS);
                const uint8_t *const RSTR r0 =
                        PAIR ? factors_p_ + (size_t)(evb / VS) * NB * VS + (size_t)(evb % VS) : nullptr;
                tau Y[DIM * N][VS];
                for (int r = 0; r < DIM * N; ++r)
#pragma omp simd
                    for (int lane = 0; lane < VS; ++lane) Y[r][lane] = 0;
                for (int k = 0; k < NB; ++k) {
                    int p, q;
                    block_nodes(k, p, q);
                    const code_t *const RSTR ck = c0 + (size_t)k * ORDER * VS;
#pragma omp simd
                    for (int lane = 0; lane < ne; ++lane) {
                        // Differences in the vector's precision, so a translation cancels exactly.
                        tau d0 = tau(bh[0][q][lane] - bh[0][p][lane]);
                        tau d1 = tau(bh[1][q][lane] - bh[1][p][lane]);
                        tau d2 = tau(bh[2][q][lane] - bh[2][p][lane]);
                        if constexpr (PAIR) {
                            // The pair factor goes on the differences (3 multiplies).
                            const tau r = tau(PairFactor::value(r0[(size_t)k * VS + lane]));
                            d0 *= r;
                            d1 *= r;
                            d2 *= r;
                        }
                        tau K[ORDER];
                        for (int x = 0; x < ORDER; ++x) K[x] = tau(ck[x * VS + lane]);
                        Y[0 * N + p][lane] += K[0] * d0 + K[1] * d1 + K[2] * d2;
                        Y[1 * N + p][lane] += K[3] * d0 + K[4] * d1 + K[5] * d2;
                        Y[2 * N + p][lane] += K[6] * d0 + K[7] * d1 + K[8] * d2;
                        if constexpr (symmetric) {
                            Y[0 * N + q][lane] -= K[0] * d0 + K[3] * d1 + K[6] * d2;
                            Y[1 * N + q][lane] -= K[1] * d0 + K[4] * d1 + K[7] * d2;
                            Y[2 * N + q][lane] -= K[2] * d0 + K[5] * d1 + K[8] * d2;
                        }
                    }
                }
                const float *const RSTR scales = scales_p_;
                for (int i = 0; i < DIM; ++i)
                    for (int p = 0; p < N; ++p)
#pragma omp simd
                        for (int lane = 0; lane < ne; ++lane) {
                            real_t v = (real_t)Y[i * N + p][lane];
                            if constexpr (scaled) v *= (real_t)scales[(size_t)evb + lane];
                            out[i][p][lane] = v;
                        }
            }

        private:
            ptrdiff_t             nelements_{0}, nblocks_{0};
            SharedBuffer<code_t>  codes_;    // [block][entry][lane]
            SharedBuffer<float>   scales_;   // per element
            SharedBuffer<uint8_t> factors_;  // [block][node block][lane]
            // The buffers' storage, resolved once: Buffer::data() is an out-of-line call into
            // smesh, and the blocks run per VS elements.
            code_t  *codes_p_{nullptr};
            float   *scales_p_{nullptr};
            uint8_t *factors_p_{nullptr};

            SFEM_INLINE std::size_t code_index(const ptrdiff_t e, const int entry) const {
                return ((size_t)(e / VS) * NE + entry) * VS + (size_t)(e % VS);
            }

            /// Scale and encode one element's stored numbers.
            void encode_values(const ptrdiff_t e, const double *const vals) {
                code_t *const RSTR codes = codes_p_;
                double             inv   = 1;
                double             top   = 0;
                if constexpr (scaled) {
                    for (int k = 0; k < NE; ++k) top = std::max(top, std::fabs(vals[k]));
                    const double s           = CodeTraits<code_t>::scale_for(top);
                    scales_p_[(size_t)e] = (float)s;
                    inv                        = 1.0 / (double)(float)s;
                }
                if constexpr (PAIR) {
                    uint8_t *const RSTR factors = factors_p_;
                    for (int k = 0; k < NB; ++k) {
                        double pt = 0;
                        for (int y = 0; y < ORDER; ++y) pt = std::max(pt, std::fabs(vals[k * ORDER + y]));
                        const uint8_t b = PairFactor::encode(top > 0 ? pt / top : 0.0);
                        factors[((size_t)(e / VS) * NB + k) * VS + (size_t)(e % VS)] = b;
                        const double pinv = inv / (double)PairFactor::value(b);
                        for (int y = 0; y < ORDER; ++y)
                            codes[code_index(e, k * ORDER + y)] = CodeTraits<code_t>::encode(vals[k * ORDER + y], pinv);
                    }
                    return;
                }
                for (int k = 0; k < NE; ++k) codes[code_index(e, k)] = CodeTraits<code_t>::encode(vals[k], inv);
            }
        };

    }  // namespace inexact
}  // namespace sfem
