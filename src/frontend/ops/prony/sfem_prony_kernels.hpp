#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "sfem_HistoryStorage.hpp"
#include "sfem_prony_diagnostics.hpp"

// Pointwise host loops of the Prony infrastructure.
//
// Every array is SoA over the elements: row r of tensor t of an array with
// `rows` rows per tensor lives at [(t * rows + r) * n + e], so each row is a
// contiguous run and the innermost loop is over the elements of a block.  The
// loops are templated on the codec of what they read and write; the runtime
// storage format is resolved once, outside, by `dispatch_history_codec`.
//
// Scaled codecs need the max-abs of a whole tensor of an element before its
// first value can be written, so an update that re-encodes runs its arithmetic
// twice -- once for the scale and once to store.  The arithmetic is one fused
// multiply-add per value, which is cheaper than holding a block of decoded
// tensors in scratch.
//
// The mean-deviation codecs keep, beside the codes, a mean per (tensor,
// component, group of `Means::elements` consecutive elements' points), and code
// each value's deviation from it.  Re-encoding them runs the arithmetic three
// times -- for the means, the deviations' scale, and to store -- and a group
// never straddles a block (`elements` divides BLOCK), so each pass stays within
// the block that owns the group.

namespace sfem {
    namespace prony {

        static constexpr ptrdiff_t BLOCK = 64;

        static_assert(BLOCK % MAX_ELEMENTS_PER_MEAN == 0, "a group of elements must not straddle a block");

        namespace detail {
            static constexpr int MAX_COMPONENTS = 9;

            /// Store the values z(r, l) of one tensor (row r, element e0 + l) on the block of
            /// elements [e0, e0 + nb) as means and deviations: per component and group the mean
            /// of its values over the group's points, rounded to history_mean_t, and per element
            /// the base codec's scale of the largest deviation.  `z` may read the tensor being
            /// written: each value is read before it is overwritten, and the scales and means are
            /// written last.  With COUNT, the values whose stored value `old(r, l)` did not move
            /// although z did are counted (sfem_prony_diagnostics.hpp).
            template <class Codec, bool COUNT, class Z, class Old>
            void store_mean_deviation(const ptrdiff_t                          rows,
                                      const ptrdiff_t                          n,
                                      const ptrdiff_t                          e0,
                                      const ptrdiff_t                          nb,
                                      const Means                             &m,
                                      Z                                      &&z,
                                      Old                                    &&old,
                                      typename Codec::storage_t *const H,  // read by z: not restrict
                                      scaling_t *const                 scales,
                                      history_mean_t *const            means) {
                (void)old;
                const ptrdiff_t nc  = rows / m.n_qp;
                const ptrdiff_t g0  = e0 / m.elements;
                const ptrdiff_t ngb = (nb + m.elements - 1) / m.elements;

                real_t         mean[MAX_COMPONENTS][BLOCK];
                history_mean_t group_mean[MAX_COMPONENTS][BLOCK];
                for (ptrdiff_t c = 0; c < nc; c++) {
                    real_t sum[BLOCK];
                    for (ptrdiff_t l = 0; l < nb; l++) sum[l] = 0;
                    for (int q = 0; q < m.n_qp; q++) {
                        const ptrdiff_t r = c * m.n_qp + q;
                        for (ptrdiff_t l = 0; l < nb; l++) sum[l] += z(r, l);
                    }
                    for (ptrdiff_t g = 0; g < ngb; g++) {
                        const ptrdiff_t lo = g * m.elements, hi = std::min(lo + m.elements, nb);
                        real_t          s  = 0;
                        for (ptrdiff_t l = lo; l < hi; l++) s += sum[l];
                        group_mean[c][g] = static_cast<history_mean_t>(s / static_cast<real_t>((hi - lo) * m.n_qp));
                        for (ptrdiff_t l = lo; l < hi; l++) mean[c][l] = group_mean[c][g];
                    }
                }

                real_t max_abs[BLOCK];
                for (ptrdiff_t l = 0; l < nb; l++) max_abs[l] = 0;
                for (ptrdiff_t r = 0; r < rows; r++) {
                    const real_t *const mc = mean[r / m.n_qp];
                    for (ptrdiff_t l = 0; l < nb; l++) max_abs[l] = std::max(max_abs[l], std::abs(z(r, l) - mc[l]));
                }

                scaling_t scale[BLOCK];
                real_t    inv[BLOCK];
                for (ptrdiff_t l = 0; l < nb; l++) {
                    scale[l] = Codec::scale_for(max_abs[l]);
                    inv[l]   = 1 / static_cast<real_t>(scale[l]);
                }

                SFEM_PRONY_COUNTERS;
                for (ptrdiff_t r = 0; r < rows; r++) {
                    const real_t *const              mc = mean[r / m.n_qp];
                    typename Codec::storage_t *const h  = &H[r * n + e0];
                    for (ptrdiff_t l = 0; l < nb; l++) {
                        if constexpr (COUNT) {
                            SFEM_PRONY_DIAGNOSTIC(const real_t before = old(r, l));
                            const real_t v = z(r, l);
                            h[l]           = Codec::encode(v - mc[l], inv[l]);
                            SFEM_PRONY_COUNT(before, v, mc[l] + Codec::decode(h[l], scale[l]));
                        } else {
                            h[l] = Codec::encode(z(r, l) - mc[l], inv[l]);
                        }
                    }
                }
                if constexpr (COUNT) SFEM_PRONY_COMMIT_COUNTS;

                for (ptrdiff_t l = 0; l < nb; l++) scales[e0 + l] = scale[l];
                for (ptrdiff_t c = 0; c < nc; c++) {
                    for (ptrdiff_t g = 0; g < ngb; g++) means[c * m.n_groups + g0 + g] = group_mean[c][g];
                }
            }

            /// Value (r, e) of a mean-deviation tensor.
            template <class Codec>
            inline real_t mean_deviation_value(const typename Codec::storage_t code,
                                               const scaling_t                 scale,
                                               const history_mean_t *const     means,
                                               const Means                    &m,
                                               const ptrdiff_t                 r,
                                               const ptrdiff_t                 e) {
                return static_cast<real_t>(means[(r / m.n_qp) * m.n_groups + e / m.elements]) + Codec::decode(code, scale);
            }

            /// The group of each element of the block [e0, e0 + nb): computed once per block, not per value.
            inline void lane_groups(const ptrdiff_t e0, const ptrdiff_t nb, const Means &m, ptrdiff_t *const groups) {
                for (ptrdiff_t l = 0; l < nb; l++) groups[l] = (e0 + l) / m.elements;
            }

            /// The means of one tensor at each element of the block, by component: means[c][l].
            inline void lane_means(const ptrdiff_t        nc,
                                   const ptrdiff_t        nb,
                                   const Means           &m,
                                   const history_mean_t *const tensor_means,
                                   const ptrdiff_t *const groups,
                                   real_t (*const means)[BLOCK]) {
                for (ptrdiff_t c = 0; c < nc; c++) {
                    const history_mean_t *const row = &tensor_means[c * m.n_groups];
                    for (ptrdiff_t l = 0; l < nb; l++) means[c][l] = static_cast<real_t>(row[groups[l]]);
                }
            }
        }  // namespace detail

        /// Encode `rows` rows of real_t values into one tensor of a store.
        template <class Codec>
        void encode(const ptrdiff_t                  rows,
                    const ptrdiff_t                  n,
                    const real_t *const SFEM_RESTRICT values,
                    typename Codec::storage_t *const SFEM_RESTRICT out,
                    scaling_t *const SFEM_RESTRICT    scales,
                    const Means                      &m = Means{}) {
            if constexpr (Codec::mean_deviation) {
#pragma omp parallel for
                for (ptrdiff_t e0 = 0; e0 < n; e0 += BLOCK) {
                    const ptrdiff_t nb = std::min(BLOCK, n - e0);
                    detail::store_mean_deviation<Codec, false>(
                            rows,
                            n,
                            e0,
                            nb,
                            m,
                            [&](const ptrdiff_t r, const ptrdiff_t l) { return values[r * n + e0 + l]; },
                            [](const ptrdiff_t, const ptrdiff_t) { return real_t(0); },
                            out,
                            scales,
                            m.data);
                }
                return;
            }

#pragma omp parallel for
            for (ptrdiff_t e0 = 0; e0 < n; e0 += BLOCK) {
                const ptrdiff_t nb = std::min(BLOCK, n - e0);

                real_t inv[BLOCK];
                if constexpr (Codec::scaled) {
                    real_t max_abs[BLOCK];
                    for (ptrdiff_t l = 0; l < nb; l++) max_abs[l] = 0;

                    for (ptrdiff_t r = 0; r < rows; r++) {
                        const real_t *const v = &values[r * n + e0];
                        for (ptrdiff_t l = 0; l < nb; l++) max_abs[l] = std::max(max_abs[l], std::abs(v[l]));
                    }

                    for (ptrdiff_t l = 0; l < nb; l++) {
                        scales[e0 + l] = Codec::scale_for(max_abs[l]);
                        inv[l]         = 1 / static_cast<real_t>(scales[e0 + l]);
                    }
                } else {
                    for (ptrdiff_t l = 0; l < nb; l++) {
                        scales[e0 + l] = 1;
                        inv[l]         = 1;
                    }
                }

                for (ptrdiff_t r = 0; r < rows; r++) {
                    const real_t *const              v = &values[r * n + e0];
                    typename Codec::storage_t *const o = &out[r * n + e0];
                    for (ptrdiff_t l = 0; l < nb; l++) o[l] = Codec::encode(v[l], inv[l]);
                }
            }
        }

        /// Decode one tensor of a store into `rows` rows of real_t values.
        template <class Codec>
        void decode(const ptrdiff_t                               rows,
                    const ptrdiff_t                               n,
                    const typename Codec::storage_t *const SFEM_RESTRICT in,
                    const scaling_t *const SFEM_RESTRICT          scales,
                    real_t *const SFEM_RESTRICT                   values,
                    const Means                                  &m = Means{}) {
#pragma omp parallel for
            for (ptrdiff_t e0 = 0; e0 < n; e0 += BLOCK) {
                const ptrdiff_t nb = std::min(BLOCK, n - e0);
                for (ptrdiff_t r = 0; r < rows; r++) {
                    const typename Codec::storage_t *const s = &in[r * n + e0];
                    real_t *const                          v = &values[r * n + e0];
                    if constexpr (Codec::mean_deviation) {
                        for (ptrdiff_t l = 0; l < nb; l++) {
                            v[l] = detail::mean_deviation_value<Codec>(s[l], scales[e0 + l], m.data, m, r, e0 + l);
                        }
                    } else {
                        for (ptrdiff_t l = 0; l < nb; l++) v[l] = Codec::decode(s[l], scales[e0 + l]);
                    }
                }
            }
        }

        /// Multiply one tensor of a store by `factor` in place:
        ///     H <- factor * H,
        /// re-encoded under fresh scales.
        template <class Codec>
        void scale(const ptrdiff_t                            rows,
                   const ptrdiff_t                            n,
                   const real_t                               factor,
                   typename Codec::storage_t *const SFEM_RESTRICT H,
                   scaling_t *const SFEM_RESTRICT             H_scales,
                   const Means                               &m = Means{}) {
            if constexpr (Codec::mean_deviation) {
#pragma omp parallel for
                for (ptrdiff_t e0 = 0; e0 < n; e0 += BLOCK) {
                    const ptrdiff_t nb = std::min(BLOCK, n - e0);
                    ptrdiff_t groups[BLOCK];
                    real_t    old_mean[detail::MAX_COMPONENTS][BLOCK];
                    detail::lane_groups(e0, nb, m, groups);
                    detail::lane_means(rows / m.n_qp, nb, m, m.data, groups, old_mean);
                    detail::store_mean_deviation<Codec, false>(
                            rows,
                            n,
                            e0,
                            nb,
                            m,
                            [&](const ptrdiff_t r, const ptrdiff_t l) {
                                return factor * (old_mean[r / m.n_qp][l] + Codec::decode(H[r * n + e0 + l], H_scales[e0 + l]));
                            },
                            [](const ptrdiff_t, const ptrdiff_t) { return real_t(0); },
                            H,
                            H_scales,
                            m.data);
                }
                return;
            }

#pragma omp parallel for
            for (ptrdiff_t e0 = 0; e0 < n; e0 += BLOCK) {
                const ptrdiff_t nb = std::min(BLOCK, n - e0);

                real_t old_scale[BLOCK];
                real_t inv[BLOCK];
                for (ptrdiff_t l = 0; l < nb; l++) old_scale[l] = H_scales[e0 + l];

                if constexpr (Codec::scaled) {
                    real_t max_abs[BLOCK];
                    for (ptrdiff_t l = 0; l < nb; l++) max_abs[l] = 0;

                    for (ptrdiff_t r = 0; r < rows; r++) {
                        const typename Codec::storage_t *const h = &H[r * n + e0];
                        for (ptrdiff_t l = 0; l < nb; l++) {
                            max_abs[l] = std::max(max_abs[l], std::abs(factor * Codec::decode(h[l], old_scale[l])));
                        }
                    }

                    for (ptrdiff_t l = 0; l < nb; l++) {
                        H_scales[e0 + l] = Codec::scale_for(max_abs[l]);
                        inv[l]           = 1 / static_cast<real_t>(H_scales[e0 + l]);
                    }
                } else {
                    for (ptrdiff_t l = 0; l < nb; l++) inv[l] = 1;
                }

                for (ptrdiff_t r = 0; r < rows; r++) {
                    typename Codec::storage_t *const h = &H[r * n + e0];
                    for (ptrdiff_t l = 0; l < nb; l++) h[l] = Codec::encode(factor * Codec::decode(h[l], old_scale[l]), inv[l]);
                }
            }
        }

        /// The frozen history of a step:
        ///     out = sum_i alpha_i H_i - b_sum * S_n.
        ///
        /// `S_n` is the relaxed stress at the start of the step, evaluated from
        /// the converged displacement: real_t, or float when the stress slots are
        /// (`PronyRelaxation::Options::stress_storage`).
        template <class HCodec, typename S_t = real_t>
        void frozen_history(const int                                        n_terms,
                            const ptrdiff_t                                  rows,
                            const ptrdiff_t                                  n,
                            const real_t *const SFEM_RESTRICT                alpha,
                            const real_t                                     b_sum,
                            const typename HCodec::storage_t *const SFEM_RESTRICT H,
                            const scaling_t *const SFEM_RESTRICT             H_scales,
                            const S_t *const SFEM_RESTRICT                   S_n,
                            real_t *const SFEM_RESTRICT                      out,
                            const Means                                     &m = Means{}) {
#pragma omp parallel for
            for (ptrdiff_t e0 = 0; e0 < n; e0 += BLOCK) {
                const ptrdiff_t nb = std::min(BLOCK, n - e0);
                ptrdiff_t       groups[BLOCK];
                if constexpr (HCodec::mean_deviation) detail::lane_groups(e0, nb, m, groups);
                for (ptrdiff_t r = 0; r < rows; r++) {
                    real_t *const       o = &out[r * n + e0];
                    const S_t *const s = &S_n[r * n + e0];
                    for (ptrdiff_t l = 0; l < nb; l++) o[l] = -b_sum * static_cast<real_t>(s[l]);

                    for (int i = 0; i < n_terms; i++) {
                        const typename HCodec::storage_t *const h  = &H[(i * rows + r) * n + e0];
                        const scaling_t *const                  hs = &H_scales[i * n + e0];
                        const real_t                            a  = alpha[i];
                        if constexpr (HCodec::mean_deviation) {
                            const history_mean_t *const row = m.tensor(i, rows) + (r / m.n_qp) * m.n_groups;
                            for (ptrdiff_t l = 0; l < nb; l++) {
                                o[l] += a * (static_cast<real_t>(row[groups[l]]) + HCodec::decode(h[l], hs[l]));
                            }
                        } else {
                            for (ptrdiff_t l = 0; l < nb; l++) o[l] += a * HCodec::decode(h[l], hs[l]);
                        }
                    }
                }
            }
        }

        /// Close a step:
        ///     H_i <- alpha_i H_i + b_i (S_new - S_n).
        ///
        /// Both stresses are evaluated, in real_t, from the displacements at the
        /// two ends of the step; only H_i is stored, and only it is re-encoded.
        template <class HCodec, typename S_t = real_t>
        void advance(const int                                  n_terms,
                     const ptrdiff_t                            rows,
                     const ptrdiff_t                            n,
                     const real_t *const SFEM_RESTRICT          alpha,
                     const real_t *const SFEM_RESTRICT          b,
                     const S_t *const SFEM_RESTRICT             S_n,
                     const S_t *const SFEM_RESTRICT             S_new,
                     typename HCodec::storage_t *const SFEM_RESTRICT H,
                     scaling_t *const SFEM_RESTRICT             H_scales,
                     const Means                               &m = Means{}) {
            if constexpr (HCodec::mean_deviation) {
#pragma omp parallel for
                for (ptrdiff_t e0 = 0; e0 < n; e0 += BLOCK) {
                    const ptrdiff_t nb = std::min(BLOCK, n - e0);
                    ptrdiff_t       groups[BLOCK];
                    real_t          old_mean[detail::MAX_COMPONENTS][BLOCK];
                    detail::lane_groups(e0, nb, m, groups);
                    for (int i = 0; i < n_terms; i++) {
                        typename HCodec::storage_t *const h_i  = &H[i * rows * n];
                        scaling_t *const                  hs_i = &H_scales[i * n];
                        history_mean_t *const             m_i  = m.tensor(i, rows);
                        const real_t                      a    = alpha[i];
                        const real_t                      bi   = b[i];
                        detail::lane_means(rows / m.n_qp, nb, m, m_i, groups, old_mean);
                        const auto stored = [&](const ptrdiff_t r, const ptrdiff_t l) {
                            return old_mean[r / m.n_qp][l] + HCodec::decode(h_i[r * n + e0 + l], hs_i[e0 + l]);
                        };
                        detail::store_mean_deviation<HCodec, true>(
                                rows,
                                n,
                                e0,
                                nb,
                                m,
                                [&](const ptrdiff_t r, const ptrdiff_t l) {
                                    return a * stored(r, l) + bi * (static_cast<real_t>(S_new[r * n + e0 + l]) -
                                                                    static_cast<real_t>(S_n[r * n + e0 + l]));
                                },
                                stored,
                                h_i,
                                hs_i,
                                m_i);
                    }
                }
                return;
            }

#pragma omp parallel for
            for (ptrdiff_t e0 = 0; e0 < n; e0 += BLOCK) {
                const ptrdiff_t nb = std::min(BLOCK, n - e0);
                SFEM_PRONY_COUNTERS;

                for (int i = 0; i < n_terms; i++) {
                    typename HCodec::storage_t *const h_i  = &H[i * rows * n];
                    scaling_t *const                  hs_i = &H_scales[i * n + e0];
                    const real_t                      a    = alpha[i];
                    const real_t                      bi   = b[i];

                    real_t old_scale[BLOCK];
                    real_t inv[BLOCK];
                    for (ptrdiff_t l = 0; l < nb; l++) old_scale[l] = hs_i[l];

                    if constexpr (HCodec::scaled) {
                        real_t max_abs[BLOCK];
                        for (ptrdiff_t l = 0; l < nb; l++) max_abs[l] = 0;

                        for (ptrdiff_t r = 0; r < rows; r++) {
                            const typename HCodec::storage_t *const h  = &h_i[r * n + e0];
                            const S_t *const                        sp = &S_n[r * n + e0];
                            const S_t *const                        sn = &S_new[r * n + e0];
                            for (ptrdiff_t l = 0; l < nb; l++) {
                                const real_t v = a * HCodec::decode(h[l], old_scale[l]) + bi * (static_cast<real_t>(sn[l]) - static_cast<real_t>(sp[l]));
                                max_abs[l]     = std::max(max_abs[l], std::abs(v));
                            }
                        }

                        for (ptrdiff_t l = 0; l < nb; l++) {
                            hs_i[l] = HCodec::scale_for(max_abs[l]);
                            inv[l]  = 1 / static_cast<real_t>(hs_i[l]);
                        }
                    } else {
                        for (ptrdiff_t l = 0; l < nb; l++) inv[l] = 1;
                    }

                    for (ptrdiff_t r = 0; r < rows; r++) {
                        typename HCodec::storage_t *const h  = &h_i[r * n + e0];
                        const S_t *const                  sp = &S_n[r * n + e0];
                        const S_t *const                  sn = &S_new[r * n + e0];
                        for (ptrdiff_t l = 0; l < nb; l++) {
                            SFEM_PRONY_DIAGNOSTIC(const real_t before = HCodec::decode(h[l], old_scale[l]));
                            const real_t v = a * HCodec::decode(h[l], old_scale[l]) + bi * (static_cast<real_t>(sn[l]) - static_cast<real_t>(sp[l]));
                            h[l]           = HCodec::encode(v, inv[l]);
                            SFEM_PRONY_COUNT(before, v, HCodec::decode(h[l], hs_i[l]));
                        }
                    }
                }
                SFEM_PRONY_COMMIT_COUNTS;
            }
        }

    }  // namespace prony
}  // namespace sfem
