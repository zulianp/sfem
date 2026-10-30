#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "sfem_InexactCodecs.hpp"
#include "sfem_InexactElement.hpp"
#include "sfem_aliases.hpp"

namespace sfem {
    namespace inexact {

        /// The point store: the weighted, pulled-back tangent at every point of the rule (its P2
        /// moments on TET10), encoded (docs/compressed_partial_assembly, "Compression").
        ///
        /// `M` is a generated unit (`<unit>_<element>_inexact`): its `element` gives the stages
        /// and tables, its `contraction` the packing of one tangent (45 numbers, or 81 without the
        /// major symmetry).  The layout is blocked, `[block][tensor][component][lane]`, so a block
        /// of `VS` elements reads one contiguous run.  With `FOLD` a point of a tensor-product
        /// element stores only the components its mask keeps (exact; `element::fold_*`).
        /// Stored numbers per element: the fold's, or every component of every tensor.
        template <bool FOLD, class Fold>
        constexpr int stored_numbers(const int unfolded) {
            if constexpr (FOLD) {
                return Fold::NS;
            } else {
                return unfolded;
            }
        }

        template <class M, typename code_t, Scheme SCH, int VS_, bool FOLD>
        class PointStore {
        public:
            using Unit        = M;
            using Element     = typename M::element;
            using Contraction = typename M::contraction;
            using Fold        = typename FoldOf<M>::type;
            using Tables      = ElementTables<Element>;
            using code_type   = code_t;

            static constexpr int  VS              = VS_;
            static constexpr int  NV              = Element::NV;
            static constexpr int  NC              = M::NC;
            static constexpr int  DIM             = Element::DIM;
            static constexpr int  ORDER           = DIM * DIM;
            static constexpr bool scaled          = CodeTraits<code_t>::scaled;
            static constexpr bool scale_on_output = scaled && SCH == Scheme::Plain;
            static constexpr bool has_mean        = SCH == Scheme::MeanDeviation;
            static constexpr bool folded          = FOLD;
            static constexpr int  NSE             = stored_numbers<FOLD, Fold>(NV * NC);

            static_assert(SCH == Scheme::Plain || scaled, "the deviation scheme needs a scaled code");
            static_assert(!FOLD || Element::foldable, "the folded store is a tensor-product element's");
            static_assert(NC == Fold::NC, "the fold's packing is the tangent's");

            explicit PointStore(const ptrdiff_t nelements) : nelements_(nelements) {
                nblocks_ = (nelements + VS - 1) / VS;
                codes_   = create_host_buffer<code_t>((size_t)NSE * nblocks_ * VS);
                if (scaled) scales_ = create_host_buffer<float>((size_t)nblocks_ * VS);
                if (has_mean) means_ = create_host_buffer<float>((size_t)NC * nblocks_ * VS);
            }

            ptrdiff_t nelements() const { return nelements_; }

            std::size_t bytes() const {
                std::size_t ret = codes_->size() * sizeof(code_t);
                if (scales_) ret += scales_->size() * sizeof(float);
                if (means_) ret += means_->size() * sizeof(float);
                return ret;
            }

            // ---------------------------------------------------------------- set-up side

            /// Point `q`'s weighted tangent for one lane, into the block's tensors: the point's own
            /// tensor, or every moment it contributes to.
            template <typename s_t>
            static SFEM_INLINE void accumulate(const int q, s_t bT[NV][NC][VS], const int c, const int lane, const s_t v) {
                if constexpr (Element::moments) {
                    for (int j = 0; j < NV; ++j) bT[j][c][lane] += s_t(Element::FACTOR[q][j]) * v;
                } else {
                    bT[q][c][lane] = v;
                }
            }

            template <typename s_t>
            void encode_block(const ptrdiff_t evb, const int ne, const s_t bT[NV][NC][VS]) {
                code_t *const RSTR codes  = codes_->data();
                float *const RSTR  scales = scales_ ? scales_->data() : nullptr;
                float *const RSTR  means  = means_ ? means_->data() : nullptr;
                for (int lane = 0; lane < ne; ++lane) {
                    double T[NV][NC];
                    for (int j = 0; j < NV; ++j)
                        for (int c = 0; c < NC; ++c) T[j][c] = (double)bT[j][c][lane];

                    if constexpr (has_mean) {
                        for (int c = 0; c < NC; ++c) {
                            double acc = 0;
                            for (int j = 0; j < NV; ++j) acc += Tables::mult(j) * T[j][c];
                            const float mean = (float)(acc / Element::KVOL);
                            means[mean_index(evb, lane, c)] = mean;
                            for (int j = 0; j < NV; ++j) T[j][c] -= Element::X[j] * (double)mean;
                        }
                    }

                    if constexpr (FOLD) {
                        fold(T);
                        double top = 0;
                        for (int j = 0; j < NV; ++j)
                            for (int k = 0; k < Fold::NKEEP[Fold::mask(j)]; ++k)
                                top = std::max(top, std::fabs(T[j][Fold::KEEP[Fold::mask(j)][k]]));
                        double inv = 1;
                        if constexpr (scaled) {
                            const double s           = CodeTraits<code_t>::scale_for(top);
                            scales[(size_t)evb + lane] = (float)s;
                            inv                        = 1.0 / (double)(float)s;
                        }
                        for (int j = 0; j < NV; ++j)
                            for (int k = 0; k < Fold::NKEEP[Fold::mask(j)]; ++k)
                                codes[fold_index(evb, lane, j, k)] =
                                        CodeTraits<code_t>::encode(T[j][Fold::KEEP[Fold::mask(j)][k]], inv);
                        continue;
                    }

                    double inv = 1;
                    if constexpr (scaled) {
                        double top = 0;
                        for (int j = 0; j < NV; ++j)
                            for (int c = 0; c < NC; ++c) top = std::max(top, std::fabs(T[j][c]));
                        const double s           = CodeTraits<code_t>::scale_for(top);
                        scales[(size_t)evb + lane] = (float)s;
                        inv                        = 1.0 / (double)(float)s;
                    }
                    for (int j = 0; j < NV; ++j)
                        for (int c = 0; c < NC; ++c) codes[code_index(evb, lane, j, c)] = CodeTraits<code_t>::encode(T[j][c], inv);
                }
            }

            /// The fold: along each line in direction d, the dropped point's m = n = d components
            /// move to the line's kept points.  Linear, so it commutes with removing the mean.
            static void fold(double T[NV][NC]) {
                if constexpr (FOLD) {
                    for (int j = 0; j < NV; ++j)
                        for (int c = 0; c < NC; ++c) {
                            const int d = Fold::DIR[c];
                            if (d >= 0 && !((Fold::mask(j) >> d) & 1)) T[j][c] += Fold::coefficient(j, d) * T[Fold::source(j, d)][c];
                        }
                }
            }

            // ------------------------------------------------------------ application side

            /// Stored tensor j of element e as the contraction reads it: decoded, the mean added
            /// back, the components a folded point does not store left at zero.  Unscaled when the
            /// scale goes on the outputs.
            void decode(const ptrdiff_t e, const int j, double S[NC]) const {
                const code_t *const RSTR codes = codes_->data();
                const double             mean_scale = has_mean ? (double)scales_->data()[(size_t)e] : 1.0;
                for (int c = 0; c < NC; ++c) S[c] = 0;
                auto value = [&](const int c, const double code, const double x) {
                    return has_mean ? x * (double)means_->data()[mean_index(e, 0, c)] + mean_scale * code : code;
                };
                if constexpr (FOLD) {
                    const int mask = Fold::mask(j);
                    for (int k = 0; k < Fold::NKEEP[mask]; ++k) {
                        const int c = Fold::KEEP[mask][k];
                        S[c]        = value(c, (double)codes[fold_index(e, 0, j, k)], Fold::XC[j][c]);
                    }
                } else {
                    for (int c = 0; c < NC; ++c) S[c] = value(c, (double)codes[code_index(e, 0, j, c)], Element::X[j]);
                }
            }

            /// The element matrix's diagonal, out[i][p] for elements evb .. evb + ne, computed from
            /// the stored tensors: sum_n sum_m dphi_p/dxi_n S[(i,n),(i,m)] dphi_p/dxi_m over the
            /// tensors and their gradient points.
            void diagonal_block(const ptrdiff_t evb, const int ne, real_t out[DIM][Element::N][VS]) const {
                for (int lane = 0; lane < ne; ++lane) {
                    const ptrdiff_t e = evb + lane;
                    double          d[DIM][Element::N] = {{0}};
                    for (int j = 0; j < NV; ++j) {
                        double S[NC];
                        decode(e, j, S);
                        const int a = Tables::pa(j), b = Tables::pb(j);
                        for (int pass = 0; pass < (a == b ? 1 : 2); ++pass) {
                            const int ga = pass == 0 ? a : b, gb = pass == 0 ? b : a;
                            for (int i = 0; i < DIM; ++i)
                                for (int p = 0; p < Element::N; ++p) {
                                    double acc = 0;
                                    for (int n = 0; n < DIM; ++n)
                                        for (int m = 0; m < DIM; ++m)
                                            acc += Element::DT[gb][n][p] * S[Contraction::slot(i * DIM + n, i * DIM + m)] *
                                                   Element::DT[ga][m][p];
                                    d[i][p] += acc;
                                }
                        }
                    }
                    const double s = scale_on_output ? (double)scales_->data()[(size_t)e] : 1.0;
                    for (int i = 0; i < DIM; ++i)
                        for (int p = 0; p < Element::N; ++p) out[i][p][lane] = (real_t)(s * d[i][p]);
                }
            }

            /// The element block: gathered increment in, element outputs out, for elements
            /// evb .. evb + ne, which lie inside one store block.  `tau` is the contraction's
            /// arithmetic; the stages and the outputs are in double.
            template <typename tau>
            SFEM_INLINE void block(const ptrdiff_t   evb,
                                   const int         ne,
                                   const real_t      bh[DIM][Element::N][VS],
                                   real_t            out[DIM][Element::N][VS]) const {
                const code_t *const RSTR codes = codes_->data();

                tau P[Element::R][ORDER][VS];
                Element::template stage_in<real_t, tau, VS>(ne, bh, P);

                tau Y[Element::R][ORDER][VS];
                for (int r = 0; r < Element::R; ++r)
                    for (int c = 0; c < ORDER; ++c)
#pragma omp simd
                        for (int lane = 0; lane < VS; ++lane) Y[r][c][lane] = tau(0);

                // What decoding needs besides the codes, read once per block: the mean and the
                // element scale.  The plain scaled codecs leave their scale to the outputs.
                tau A[has_mean ? NC : 1][VS];
                tau sc[VS];
                if constexpr (has_mean) {
                    const float *const RSTR means  = means_->data();
                    const float *const RSTR scales = scales_->data();
                    for (int c = 0; c < NC; ++c)
#pragma omp simd
                        for (int lane = 0; lane < ne; ++lane) A[c][lane] = tau(means[mean_index(evb, lane, c)]);
#pragma omp simd
                    for (int lane = 0; lane < ne; ++lane) sc[lane] = tau(scales[(size_t)evb + lane]);
                }

                for (int j = 0; j < NV; ++j) {
                    const int a = Tables::pa(j), b = Tables::pb(j);
                    if constexpr (FOLD) {
                        // A folded point stores only the numbers its mask keeps: decode them into
                        // their slots and contract without the terms it does not store.
                        const int                mask = Fold::mask(j);
                        const code_t *const RSTR cf =
                                codes + ((size_t)(evb / VS) * NSE + Fold::OFF[j]) * VS + (size_t)(evb % VS);
                        tau S[NC][VS];
                        for (int k = 0; k < Fold::NKEEP[mask]; ++k) {
                            const int                c  = Fold::KEEP[mask][k];
                            const code_t *const RSTR ck = cf + (size_t)k * VS;
                            const tau                xc = tau(Fold::XC[j][c]);
#pragma omp simd
                            for (int lane = 0; lane < ne; ++lane) {
                                const tau v = tau(ck[lane]);
                                if constexpr (has_mean) {
                                    S[c][lane] = xc * A[c][lane] + sc[lane] * v;
                                } else {
                                    S[c][lane] = v;
                                }
                            }
                        }
                        Contraction::template contract_lanes_masked<tau, VS>(mask, ne, S, P[a], Y[b]);
                        continue;
                    }
                    const tau                xj = tau(Element::X[j]);
                    const code_t *const RSTR cj = codes + ((size_t)(evb / VS) * NV + j) * NC * VS + (size_t)(evb % VS);
                    // Lane loop innermost, one statement per loop: the shape every compiler vectorises.
                    tau S[NC][VS];
                    for (int c = 0; c < NC; ++c) {
                        const code_t *const RSTR cjc = cj + c * VS;
#pragma omp simd
                        for (int lane = 0; lane < ne; ++lane) {
                            const tau v = tau(cjc[lane]);
                            if constexpr (has_mean) {
                                S[c][lane] = xj * A[c][lane] + sc[lane] * v;
                            } else {
                                S[c][lane] = v;
                            }
                        }
                    }
                    Contraction::template contract_lanes<tau, VS>(ne, S, P[a], Y[b]);
                    if (a != b) Contraction::template contract_lanes<tau, VS>(ne, S, P[b], Y[a]);
                }

                Element::template stage_out<tau, real_t, VS>(ne, Y, out);

                if constexpr (scale_on_output) {
                    const float *const RSTR scales = scales_->data();
                    for (int i = 0; i < DIM; ++i)
                        for (int p = 0; p < Element::N; ++p)
#pragma omp simd
                            for (int lane = 0; lane < ne; ++lane) out[i][p][lane] *= scales[(size_t)evb + lane];
                }
            }

        private:
            ptrdiff_t           nelements_{0}, nblocks_{0};
            SharedBuffer<code_t> codes_;
            SharedBuffer<float>  scales_;  // per element
            SharedBuffer<float>  means_;   // per (element, component), blocked

            SFEM_INLINE std::size_t code_index(const ptrdiff_t e0, const int lane, const int j, const int c) const {
                const ptrdiff_t e = e0 + lane;
                return ((size_t)(e / VS) * NV + j) * NC * VS + (size_t)c * VS + (size_t)(e % VS);
            }
            SFEM_INLINE std::size_t fold_index(const ptrdiff_t e0, const int lane, const int j, const int k) const {
                const ptrdiff_t e = e0 + lane;
                return ((size_t)(e / VS) * NSE + Fold::OFF[j] + k) * VS + (size_t)(e % VS);
            }
            SFEM_INLINE std::size_t mean_index(const ptrdiff_t e0, const int lane, const int c) const {
                const ptrdiff_t e = e0 + lane;
                return ((size_t)(e / VS) * NC + c) * VS + (size_t)(e % VS);
            }
        };

    }  // namespace inexact
}  // namespace sfem
