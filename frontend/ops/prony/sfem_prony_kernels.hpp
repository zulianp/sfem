#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "sfem_HistoryStorage.hpp"

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

namespace sfem {
    namespace prony {

        static constexpr ptrdiff_t BLOCK = 64;

        /// Encode `rows` rows of real_t values into one tensor of a store.
        template <class Codec>
        void encode(const ptrdiff_t                  rows,
                    const ptrdiff_t                  n,
                    const real_t *const SFEM_RESTRICT values,
                    typename Codec::storage_t *const SFEM_RESTRICT out,
                    scaling_t *const SFEM_RESTRICT    scales) {
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
                    real_t *const SFEM_RESTRICT                   values) {
#pragma omp parallel for
            for (ptrdiff_t e0 = 0; e0 < n; e0 += BLOCK) {
                const ptrdiff_t nb = std::min(BLOCK, n - e0);
                for (ptrdiff_t r = 0; r < rows; r++) {
                    const typename Codec::storage_t *const s = &in[r * n + e0];
                    real_t *const                          v = &values[r * n + e0];
                    for (ptrdiff_t l = 0; l < nb; l++) v[l] = Codec::decode(s[l], scales[e0 + l]);
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
                   scaling_t *const SFEM_RESTRICT             H_scales) {
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
        /// the converged displacement rather than stored, so it is real_t.
        template <class HCodec>
        void frozen_history(const int                                        n_terms,
                            const ptrdiff_t                                  rows,
                            const ptrdiff_t                                  n,
                            const real_t *const SFEM_RESTRICT                alpha,
                            const real_t                                     b_sum,
                            const typename HCodec::storage_t *const SFEM_RESTRICT H,
                            const scaling_t *const SFEM_RESTRICT             H_scales,
                            const real_t *const SFEM_RESTRICT                S_n,
                            real_t *const SFEM_RESTRICT                      out) {
#pragma omp parallel for
            for (ptrdiff_t e0 = 0; e0 < n; e0 += BLOCK) {
                const ptrdiff_t nb = std::min(BLOCK, n - e0);
                for (ptrdiff_t r = 0; r < rows; r++) {
                    real_t *const       o = &out[r * n + e0];
                    const real_t *const s = &S_n[r * n + e0];
                    for (ptrdiff_t l = 0; l < nb; l++) o[l] = -b_sum * s[l];

                    for (int i = 0; i < n_terms; i++) {
                        const typename HCodec::storage_t *const h  = &H[(i * rows + r) * n + e0];
                        const scaling_t *const                  hs = &H_scales[i * n + e0];
                        const real_t                            a  = alpha[i];
                        for (ptrdiff_t l = 0; l < nb; l++) o[l] += a * HCodec::decode(h[l], hs[l]);
                    }
                }
            }
        }

        /// Close a step:
        ///     H_i <- alpha_i H_i + b_i (S_new - S_n).
        ///
        /// Both stresses are evaluated, in real_t, from the displacements at the
        /// two ends of the step; only H_i is stored, and only it is re-encoded.
        template <class HCodec>
        void advance(const int                                  n_terms,
                     const ptrdiff_t                            rows,
                     const ptrdiff_t                            n,
                     const real_t *const SFEM_RESTRICT          alpha,
                     const real_t *const SFEM_RESTRICT          b,
                     const real_t *const SFEM_RESTRICT          S_n,
                     const real_t *const SFEM_RESTRICT          S_new,
                     typename HCodec::storage_t *const SFEM_RESTRICT H,
                     scaling_t *const SFEM_RESTRICT             H_scales) {
#pragma omp parallel for
            for (ptrdiff_t e0 = 0; e0 < n; e0 += BLOCK) {
                const ptrdiff_t nb = std::min(BLOCK, n - e0);

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
                            const real_t *const                     sp = &S_n[r * n + e0];
                            const real_t *const                     sn = &S_new[r * n + e0];
                            for (ptrdiff_t l = 0; l < nb; l++) {
                                const real_t v = a * HCodec::decode(h[l], old_scale[l]) + bi * (sn[l] - sp[l]);
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
                        const real_t *const               sp = &S_n[r * n + e0];
                        const real_t *const               sn = &S_new[r * n + e0];
                        for (ptrdiff_t l = 0; l < nb; l++) {
                            const real_t v = a * HCodec::decode(h[l], old_scale[l]) + bi * (sn[l] - sp[l]);
                            h[l]           = HCodec::encode(v, inv[l]);
                        }
                    }
                }
            }
        }

    }  // namespace prony
}  // namespace sfem
