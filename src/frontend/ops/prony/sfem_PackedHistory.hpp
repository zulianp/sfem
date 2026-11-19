#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "sfem_HistoryStorage.hpp"
#include "sfem_base.hpp"
#include "smesh_codec_bits.hpp"

// The packed history store (HistoryStorage::QUANT_BLOCK).
//
// A stored tensor is coded on one power-of-two grid, x = q * step, as QUANT_GRID's are; what
// differs is how the integer codes q are held.  Each row (component and point) of the tensor is
// cut into blocks of PACKED_BLOCK consecutive elements, and a block keeps its least code `base`,
// the bit width `width` of its codes' spread, and the codes' offsets from the base at that width
// (smesh's bit packing, smesh_codec_bits.hpp).  A block's codes start at `offset` * 8 bytes into
// the tensor's code area, and the area ends in smesh::compression::kSlack bytes, so a decoder may
// read whole 64-bit words.  The mesh is ordered along a space-filling curve, so the elements of a
// block are neighbours and the spread of their codes is the variation of the history over a
// neighbourhood, not its size: a finer grid costs only the bits that variation needs.
//
// The width of a block changes from one write to the next, so a write lays the code area out
// anew: pass one rounds every value and takes each block's least and largest code, the widths
// give the offsets, and pass two packs the codes into a fresh area, which then replaces the old
// one.  Rounding is the grid's (nearest, or stochastic from prony::uniform), and both passes
// round with the same numbers, so they agree on every code.

namespace sfem {
    namespace prony {

        static constexpr ptrdiff_t PACKED_BLOCK = 128;

        /// One packed tensor: `rows` rows of `n` elements in blocks of PACKED_BLOCK, the block
        /// headers indexed [r * n_blocks + b].
        struct PackedTensor {
            const uint8_t  *codes{nullptr};
            const int32_t  *base{nullptr};
            const uint8_t  *width{nullptr};
            const uint32_t *offset{nullptr};  ///< in units of 8 bytes
            ptrdiff_t       rows{0};
            ptrdiff_t       n{0};
            real_t          step{1};

            SFEM_DEVICE_INLINE ptrdiff_t n_blocks() const { return (n + PACKED_BLOCK - 1) / PACKED_BLOCK; }

            /// The code of value (r, e).
            SFEM_DEVICE_INLINE int64_t code(const ptrdiff_t r, const ptrdiff_t e) const {
                const ptrdiff_t h = r * n_blocks() + e / PACKED_BLOCK;
                const int       w = width[h];
                if (w == 0) return base[h];
                const uint64_t bit  = static_cast<uint64_t>(e % PACKED_BLOCK) * static_cast<uint64_t>(w);
                const uint64_t word = smesh::compression::load_u64(codes + static_cast<uint64_t>(offset[h]) * 8 + bit / 8);
                return static_cast<int64_t>(base[h]) +
                       static_cast<int64_t>((word >> (bit % 8)) & smesh::compression::mask_of<uint64_t>(w));
            }

            SFEM_DEVICE_INLINE real_t value(const ptrdiff_t r, const ptrdiff_t e) const {
                return static_cast<real_t>(code(r, e)) * step;
            }
        };

        /// The bytes a block of `count` codes of width `w` takes, rounded up to whole 64-bit words
        /// so that every block starts on one.
        SFEM_DEVICE_INLINE std::size_t packed_block_bytes(const ptrdiff_t count, const int w) {
            return (smesh::compression::packed_bytes(static_cast<size_t>(count), w) + 7) / 8 * 8;
        }

        /// The code of v on a grid of 1 / inv, rounded as `stochastic` says; k the value's
        /// position for the stochastic rounding's numbers.
        SFEM_DEVICE_INLINE int64_t grid_code(const real_t v, const real_t inv, const bool stochastic, const uint64_t seed, const uint64_t k) {
            const real_t x = v * inv;
#ifdef __CUDA_ARCH__
            return static_cast<int64_t>(stochastic ? floor(x + uniform(seed, k)) : nearbyint(x));
#else
            return static_cast<int64_t>(stochastic ? std::floor(x + uniform(seed, k)) : std::nearbyint(x));
#endif
        }

        /// The layout of a write: headers, and the size of the code area they describe.
        struct PackedLayout {
            std::vector<int32_t>  base;
            std::vector<uint8_t>  width;
            std::vector<uint32_t> offset;
            std::size_t           bytes{0};  ///< the code area, kSlack included
        };

        /// Pass one of a host write: every value z(r, e) rounded on the grid of `step`, each
        /// block's least code and width, and the offsets they give.  `k0` is the tensor's first
        /// value's position in the stochastic rounding's numbering.
        template <class Z>
        PackedLayout plan_packed(const ptrdiff_t rows,
                                 const ptrdiff_t n,
                                 const real_t    step,
                                 Z             &&z,
                                 const bool      stochastic,
                                 const uint64_t  seed,
                                 const uint64_t  k0) {
            const ptrdiff_t nb  = (n + PACKED_BLOCK - 1) / PACKED_BLOCK;
            const real_t    inv = 1 / step;
            PackedLayout    layout;
            layout.base.resize(rows * nb);
            layout.width.resize(rows * nb);
            layout.offset.resize(rows * nb);
            std::vector<std::size_t> sizes(rows * nb);
#pragma omp parallel for collapse(2)
            for (ptrdiff_t r = 0; r < rows; r++) {
                for (ptrdiff_t b = 0; b < nb; b++) {
                    const ptrdiff_t e0 = b * PACKED_BLOCK, e1 = std::min(n, e0 + PACKED_BLOCK);
                    int64_t         lo = INT64_MAX, hi = INT64_MIN;
                    for (ptrdiff_t e = e0; e < e1; e++) {
                        const int64_t q = grid_code(z(r, e), inv, stochastic, seed, k0 + r * n + e);
                        lo              = std::min(lo, q);
                        hi              = std::max(hi, q);
                    }
                    if (lo < INT32_MIN || hi > INT32_MAX || hi - lo > INT64_C(0xffffffff)) {
                        SFEM_ERROR("plan_packed: codes beyond 32 bits; the history's tolerance is too fine for its range\n");
                    }
                    const int w             = smesh::compression::bit_width(static_cast<uint64_t>(hi - lo));
                    layout.base[r * nb + b]  = static_cast<int32_t>(lo);
                    layout.width[r * nb + b] = static_cast<uint8_t>(w);
                    sizes[r * nb + b]        = packed_block_bytes(e1 - e0, w);
                }
            }
            std::size_t at = 0;
            for (ptrdiff_t h = 0; h < rows * nb; h++) {
                if (at / 8 > UINT32_MAX) SFEM_ERROR("plan_packed: a tensor's codes exceed 32 GB\n");
                layout.offset[h] = static_cast<uint32_t>(at / 8);
                at += sizes[h];
            }
            layout.bytes = at + smesh::compression::kSlack;
            return layout;
        }

        /// Pass two of a host write: the codes of z, as pass one rounded them, packed into
        /// `codes` (layout.bytes zeroed bytes).
        template <class Z>
        void pack_packed(const ptrdiff_t      rows,
                         const ptrdiff_t      n,
                         const real_t         step,
                         Z                  &&z,
                         const bool           stochastic,
                         const uint64_t       seed,
                         const uint64_t       k0,
                         const PackedLayout  &layout,
                         uint8_t *const       codes) {
            const ptrdiff_t nb  = (n + PACKED_BLOCK - 1) / PACKED_BLOCK;
            const real_t    inv = 1 / step;
#pragma omp parallel for collapse(2)
            for (ptrdiff_t r = 0; r < rows; r++) {
                for (ptrdiff_t b = 0; b < nb; b++) {
                    const ptrdiff_t h  = r * nb + b;
                    const int       w  = layout.width[h];
                    if (w == 0) continue;
                    const ptrdiff_t e0 = b * PACKED_BLOCK, e1 = std::min(n, e0 + PACKED_BLOCK);
                    uint64_t        v[PACKED_BLOCK];
                    for (ptrdiff_t e = e0; e < e1; e++) {
                        v[e - e0] = static_cast<uint64_t>(grid_code(z(r, e), inv, stochastic, seed, k0 + r * n + e) - layout.base[h]);
                    }
                    // bp_pack writes whole 64-bit words, which reach past a block's end into the
                    // next one: packed here, and only the block's own bytes copied, so blocks
                    // packed by other threads are not touched.
                    uint8_t           local[PACKED_BLOCK * 4 + smesh::compression::kSlack] = {};
                    const std::size_t bytes = packed_block_bytes(e1 - e0, w);
                    smesh::compression::bp_pack(local, v, e1 - e0, w);
                    std::memcpy(codes + static_cast<std::size_t>(layout.offset[h]) * 8, local, bytes);
                }
            }
        }

        /// prony::frozen_history over packed tensors: out = sum_s alpha_s H_s - b_sum S_n, H_s read
        /// as G_s + beta_s (S_n - R) for a remainder.
        template <typename S_t>
        void frozen_history_packed(const int                 n_terms,
                                   const PackedTensor *const H,
                                   const real_t *const       alpha,
                                   const real_t              b_sum,
                                   const S_t *const          S_n,
                                   real_t *const             out,
                                   const real_t *const       beta,
                                   const S_t *const          ref) {
            const ptrdiff_t rows = H[0].rows, n = H[0].n;
#pragma omp parallel for collapse(2)
            for (ptrdiff_t r = 0; r < rows; r++) {
                for (ptrdiff_t e = 0; e < n; e++) {
                    const ptrdiff_t k = r * n + e;
                    const real_t    s = static_cast<real_t>(S_n[k]);
                    real_t          o = -b_sum * s;
                    for (int i = 0; i < n_terms; i++) {
                        o += alpha[i] * H[i].value(r, e);
                        if (beta && beta[i] != 0) o += alpha[i] * beta[i] * (s - static_cast<real_t>(ref[k]));
                    }
                    out[k] = o;
                }
            }
        }

        /// prony::squared_norms over packed tensors.
        template <typename S_t>
        void squared_norms_packed(const int                 n_terms,
                                  const PackedTensor *const H,
                                  const int                 n_qp,
                                  const int                 n_diagonal,
                                  const real_t *const       beta,
                                  const S_t *const          S_n,
                                  const S_t *const          ref,
                                  real_t *const             out) {
            for (int i = 0; i < n_terms; i++) {
                const ptrdiff_t rows = H[i].rows, n = H[i].n;
                const real_t    be   = beta ? beta[i] : real_t(0);
                real_t          sum  = 0;
#pragma omp parallel for collapse(2) reduction(+ : sum)
                for (ptrdiff_t r = 0; r < rows; r++) {
                    for (ptrdiff_t e = 0; e < n; e++) {
                        const ptrdiff_t k = r * n + e;
                        real_t          v = H[i].value(r, e);
                        if (be != 0) v += be * (static_cast<real_t>(S_n[k]) - static_cast<real_t>(ref[k]));
                        sum += (r / n_qp < n_diagonal ? 1 : 2) * v * v;
                    }
                }
                out[i] = sum;
            }
        }

    }  // namespace prony
}  // namespace sfem
