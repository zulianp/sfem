// Assembled block matrices (BSR, 3x3 blocks) with quantized values: the assembled operator encoded
// like the element matrices (ea_store.hpp).  The values come from the generated assembly, in single
// precision, and are encoded after it.  Three forms:
//
//   Full        every block, the diagonal one included; y_i = sum_j K_ij x_j
//   Difference  only the off-diagonal blocks.  The material's tangent annihilates rigid translations,
//               K_ii = -sum_{j != i} K_ij, so y_i = sum_{j != i} K_ij (x_j - x_i): the translations
//               stay in the kernel of the encoded operator whatever the rounding.  The diagonal
//               blocks are neither stored nor indexed.
//   Upper       the difference form with each pair i < j stored once: d = x_j - x_i, y_i += K_ij d,
//               y_j -= K_ij^T d.  The off-diagonal blocks are symmetric by construction, so K_ij is
//               encoded as assembled and the lower block is never read; the implied diagonal blocks,
//               -sum_{j != i} K_ij, add independently rounded blocks and are symmetric only to the
//               encoding error.  The rows are cut into parts of whole sorting windows,
//               one per thread.  The SELL layout holds the blocks whose column lies in the part's
//               own rows and, with offset columns, within reach of the offset type; the others go
//               to the part's remainder (row, column, codes), applied after the SELL steps.  A
//               remainder block whose column lies beyond the part writes y_j into a halo slot; after
//               a barrier each part adds the slots that point into its rows, in a fixed order.
//               Deterministic, no atomics.  Columns are stored as indices (col_t = idx_t) or as
//               offsets j - i (an unsigned col_t, uint16_t: two bytes per block instead of four).
//
// Codes: int8 with one f32 scale per block (the block's largest entry maps to 127), or f16 with one
// power-of-two scale for the whole matrix.  K_ij and K_ji are encoded from the same symmetrised
// block (K_ij + K_ji^T) / 2, so they share the scale and their codes are transposes: the encoded
// off-diagonal blocks are symmetric whatever order the assembly summed in.  In the difference form
// the implied diagonal blocks are symmetric only to the encoding error.
//
// Layout: VS rows side by side (SELL-C-sigma).  The rows are sorted by length, longest first, within
// windows of SIGMA rows, cut into groups of VS, and each group is padded to its longest row; a step
// k of a group holds the k-th block of each of its rows, [step][lane] column indices and scales and
// [step][entry][lane] codes, so the application runs the lane loop innermost and widens the codes a
// vector at a time (a row at a time, the decode is one scalar conversion per entry and costs more
// than the bytes it saves).  A padding block is the row's own column with zero codes.  The store
// owns its structure, so its bytes are what the application reads.
//
// The codes and scales of a step form one record ([entry][lane] codes, then [lane] scales).  The
// upper form can write the records over the assembled values themselves (encode_in_place): a
// record holds 13 (int8) or 18 (f16) bytes per stored block against 36 bytes per assembled block,
// and only about half the blocks are stored.  The windows are encoded in rounds, a round taking the
// windows whose records end before the first value of the round's first window, so no record
// overwrites a value not yet read; the remainder is encoded before the records.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <numeric>
#include <omp.h>
#include <type_traits>
#include <vector>

#include "qp_store.hpp"

namespace qp {

enum class BsrForm { Full, Difference, Upper };

template <typename code_t, BsrForm F, int VS_, typename col_t = idx_t>
struct QuantBsr {
    static constexpr int VS = VS_;
    static constexpr ptrdiff_t SIGMA = 32 * VS_;
    static constexpr bool per_block = std::is_same<code_t, int8_t>::value;
    static constexpr bool difference = F != BsrForm::Full;
    static constexpr bool upper = F == BsrForm::Upper;
    static constexpr bool offsets = !std::is_same<col_t, idx_t>::value;
    static constexpr ptrdiff_t reach = offsets ? (ptrdiff_t)std::numeric_limits<col_t>::max() : PTRDIFF_MAX;
    static_assert(per_block || std::is_same<code_t, half_t>::value, "int8 or f16 codes");
    static_assert(!offsets || (upper && std::is_unsigned<col_t>::value), "offset columns: upper form, unsigned type");

    ptrdiff_t n_rows = 0, n_groups = 0;
    std::vector<idx_t> rows;        // [group][lane]: the row in each lane, -1 past the last row
    std::vector<count_t> group_ptr;  // steps of each group
    std::vector<col_t> colidx;      // [step][lane]
    // Step records: [entry][lane] codes (blocks row-major), then [lane] scales (int8), in the store's
    // own buffer or over the assembled values (encode_in_place).
    static constexpr size_t CODE_BYTES = 9 * (size_t)VS_ * sizeof(code_t);
    static constexpr size_t R = CODE_BYTES + (per_block ? (size_t)VS_ * sizeof(float) : 0);
    std::vector<uint8_t> own;
    uint8_t *rec = nullptr;
    float scale = 1;                // the matrix's (f16)
    // Upper: part p owns groups [part_group[p], part_group[p+1]) and rows [part_row[p], part_row[p+1]);
    // its remainder blocks are [rem_ptr[p], rem_ptr[p+1]) and its halo slots [halo_ptr[p], halo_ptr[p+1]),
    // slot s standing for row halo_row[s]; the slots that point into part p's rows are
    // halo_in[halo_in_ptr[p] .. halo_in_ptr[p+1]).
    std::vector<ptrdiff_t> part_group, part_row;
    std::vector<count_t> rem_ptr, halo_ptr, halo_in_ptr;
    std::vector<idx_t> rem_row, rem_col, rem_slot;  // rem_slot: -1 when the column is the part's own
    std::vector<code_t> rem_codes;                  // [block][entry]
    std::vector<float> rem_scales;                  // [block] (int8)
    std::vector<idx_t> halo_row, halo_in;
    mutable std::vector<double> halo_y;  // [slot][3]

    // Upper: the end of the rows of the part that owns row i.
    ptrdiff_t part_end(const ptrdiff_t i) const {
        return *std::upper_bound(part_row.begin(), part_row.end(), i);
    }
    bool in_sell(const ptrdiff_t i, const ptrdiff_t j, const ptrdiff_t end) const {
        return j > i && j < end && j - i <= reach;
    }
    count_t stored_blocks(const ptrdiff_t i, const count_t *const rp, const idx_t *const ci) const {
        if (upper) {
            const ptrdiff_t end = part_end(i);
            count_t c = 0;
            for (count_t k = rp[i]; k < rp[i + 1]; ++k) c += in_sell(i, ci[k], end);
            return c;
        }
        const bool diagonal = std::binary_search(ci + rp[i], ci + rp[i + 1], (idx_t)i);
        return (rp[i + 1] - rp[i]) - (difference && diagonal ? 1 : 0);
    }

    // The structure from a node graph with sorted rows.
    QuantBsr(const ptrdiff_t n, const count_t *const rp, const idx_t *const ci) : n_rows(n) {
        n_groups = (n + VS - 1) / VS;
        const ptrdiff_t parts = upper ? omp_get_max_threads() : 0;
        if (upper) {
            const ptrdiff_t windows = (n + SIGMA - 1) / SIGMA;
            part_group.resize((size_t)parts + 1);
            part_row.resize((size_t)parts + 1);
            for (ptrdiff_t p = 0; p <= parts; ++p) {
                const ptrdiff_t w = windows * p / parts;
                part_group[(size_t)p] = std::min(n_groups, w * (SIGMA / VS));
                part_row[(size_t)p] = std::min(n, w * SIGMA);
            }
        }
        std::vector<count_t> len((size_t)n);
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) len[(size_t)i] = stored_blocks(i, rp, ci);
        std::vector<idx_t> order((size_t)n);
        std::iota(order.begin(), order.end(), idx_t(0));
#pragma omp parallel for
        for (ptrdiff_t w = 0; w < n; w += SIGMA)
            std::stable_sort(order.begin() + w, order.begin() + std::min(n, w + SIGMA),
                             [&](const idx_t a, const idx_t b) { return len[(size_t)a] > len[(size_t)b]; });
        rows.assign((size_t)n_groups * VS, idx_t(-1));
        std::copy(order.begin(), order.end(), rows.begin());
        group_ptr.assign((size_t)n_groups + 1, 0);
        for (ptrdiff_t g = 0; g < n_groups; ++g) {
            count_t longest = 0;
            for (int l = 0; l < VS; ++l) {
                const idx_t i = rows[(size_t)g * VS + l];
                if (i >= 0) longest = std::max(longest, len[(size_t)i]);
            }
            group_ptr[(size_t)g + 1] = group_ptr[(size_t)g] + longest;
        }
        const size_t steps = (size_t)group_ptr[(size_t)n_groups];
        colidx.resize(steps * VS);
        if (!upper) return;

        // Remainder blocks and the rows beyond each part they reach.
        rem_ptr.assign((size_t)parts + 1, 0);
        std::vector<std::vector<idx_t>> beyond((size_t)parts);
#pragma omp parallel for schedule(static, 1)
        for (ptrdiff_t p = 0; p < parts; ++p) {
            const ptrdiff_t end = part_row[(size_t)p + 1];
            auto &b = beyond[(size_t)p];
            count_t c = 0;
            for (ptrdiff_t i = part_row[(size_t)p]; i < end; ++i)
                for (count_t k = rp[i]; k < rp[i + 1]; ++k)
                    if (ci[k] > i && !in_sell(i, ci[k], end)) {
                        ++c;
                        if (ci[k] >= end) b.push_back(ci[k]);
                    }
            rem_ptr[(size_t)p + 1] = c;
            std::sort(b.begin(), b.end());
            b.erase(std::unique(b.begin(), b.end()), b.end());
        }
        halo_ptr.assign((size_t)parts + 1, 0);
        for (ptrdiff_t p = 0; p < parts; ++p) {
            rem_ptr[(size_t)p + 1] += rem_ptr[(size_t)p];
            halo_ptr[(size_t)p + 1] = halo_ptr[(size_t)p] + (count_t)beyond[(size_t)p].size();
        }
        const size_t remainder = (size_t)rem_ptr[(size_t)parts], slots = (size_t)halo_ptr[(size_t)parts];
        rem_row.resize(remainder);
        rem_col.resize(remainder);
        rem_slot.resize(remainder);
        rem_codes.resize(remainder * 9);
        if (per_block) rem_scales.resize(remainder);
        halo_row.resize(slots);
        // The structure is fixed: the columns are set here, the encoding writes only codes and scales.
#pragma omp parallel for schedule(static, 1)
        for (ptrdiff_t p = 0; p < parts; ++p) {
            const auto &b = beyond[(size_t)p];
            const ptrdiff_t end = part_row[(size_t)p + 1];
            std::copy(b.begin(), b.end(), halo_row.begin() + halo_ptr[(size_t)p]);
            size_t e = (size_t)rem_ptr[(size_t)p];
            for (ptrdiff_t i = part_row[(size_t)p]; i < end; ++i)
                for (count_t k = rp[i]; k < rp[i + 1]; ++k) {
                    const idx_t j = ci[k];
                    if (j <= i || in_sell(i, j, end)) continue;
                    rem_row[e] = (idx_t)i;
                    rem_col[e] = j;
                    rem_slot[e] = j < end ? -1 : (idx_t)(halo_ptr[(size_t)p] + (std::lower_bound(b.begin(), b.end(), j) - b.begin()));
                    ++e;
                }
            for (ptrdiff_t g = part_group[(size_t)p]; g < part_group[(size_t)p + 1]; ++g)
                for (int l = 0; l < VS; ++l) {
                    const idx_t i = rows[(size_t)g * VS + l];
                    size_t step = (size_t)group_ptr[(size_t)g];
                    if (i >= 0)
                        for (count_t k = rp[i]; k < rp[i + 1]; ++k)
                            if (in_sell(i, ci[k], end)) colidx[step++ * VS + l] = offsets ? (col_t)(ci[k] - i) : (col_t)ci[k];
                    // Padding: the row itself (lane 0's past the last row), zero codes.
                    const col_t self = offsets ? col_t(0) : (col_t)(i >= 0 ? i : rows[(size_t)g * VS]);
                    for (; step < (size_t)group_ptr[(size_t)g + 1]; ++step) colidx[step * VS + l] = self;
                }
        }
        // Slots by the part that owns their row, in slot order.
        halo_in_ptr.assign((size_t)parts + 1, 0);
        std::vector<count_t> owner(slots);
        for (size_t t = 0; t < slots; ++t) {
            owner[t] = (count_t)(std::upper_bound(part_row.begin(), part_row.end(), (ptrdiff_t)halo_row[t]) - part_row.begin() - 1);
            ++halo_in_ptr[(size_t)owner[t] + 1];
        }
        for (ptrdiff_t p = 0; p < parts; ++p) halo_in_ptr[(size_t)p + 1] += halo_in_ptr[(size_t)p];
        halo_in.resize(slots);
        std::vector<count_t> fill(halo_in_ptr.begin(), halo_in_ptr.end() - 1);
        for (size_t t = 0; t < slots; ++t) halo_in[(size_t)fill[(size_t)owner[t]]++] = (idx_t)t;
        halo_y.resize(slots * 3);
    }

    code_t *step_codes(const size_t step) const { return reinterpret_cast<code_t *>(rec + step * R); }
    float *step_scales(const size_t step) const { return reinterpret_cast<float *>(rec + step * R + CODE_BYTES); }
    std::size_t record_bytes() const { return (size_t)group_ptr[(size_t)n_groups] * R; }

    std::size_t bytes() const {
        const std::size_t upper_bytes = (rem_ptr.size() + halo_ptr.size() + halo_in_ptr.size()) * sizeof(count_t) +
                                        (rem_row.size() + rem_col.size() + rem_slot.size()) * sizeof(idx_t) +
                                        rem_codes.size() * sizeof(code_t) + rem_scales.size() * sizeof(float) +
                                        (halo_row.size() + halo_in.size()) * sizeof(idx_t) + halo_y.size() * sizeof(double);
        return rows.size() * sizeof(idx_t) + group_ptr.size() * sizeof(count_t) + colidx.size() * sizeof(col_t) +
               record_bytes() + upper_bytes;
    }

    // values: the assembled matrix on (rp, ci), blocks row-major.  encode writes the records into the
    // store's own buffer; encode_in_place (upper form) writes them over the values, which the
    // application then reads and which must outlive it.
    template <typename V>
    void encode(const count_t *const rp, const idx_t *const ci, const V *const values) {
        own.resize(record_bytes());
        encode_into(rp, ci, values, own.data());
    }
    template <typename V>
    void encode_in_place(const count_t *const rp, const idx_t *const ci, V *const values) {
        static_assert(upper, "in place: only the upper form reads each block once, and only its own");
        std::vector<uint8_t>().swap(own);
        encode_into(rp, ci, values, reinterpret_cast<uint8_t *>(values));
    }

    template <typename V>
    void encode_into(const count_t *const rp, const idx_t *const ci, const V *const values, uint8_t *const dest) {
        if constexpr (!per_block) {
            double top = 0;
#pragma omp parallel for reduction(max : top)
            for (ptrdiff_t i = 0; i < n_rows; ++i)
                for (count_t k = rp[i]; k < rp[i + 1]; ++k)
                    if (upper ? ci[k] > i : (!difference || ci[k] != i))
                        for (int x = 0; x < 9; ++x) top = std::max(top, std::fabs((double)values[(size_t)k * 9 + x]));
            scale = (float)CodeTraits<half_t>::scale_for(top);
        }
        const double matrix_inv = 1.0 / (double)scale;
        // One block's codes (and scale) from S, int8 with a per-block scale or f16 with the matrix's.
        auto put = [&](const double *const S, code_t *const c, const size_t stride, float *const sc) {
            if constexpr (per_block) {
                double top = 0;
                for (int x = 0; x < 9; ++x) top = std::max(top, std::fabs(S[x]));
                const double s = CodeTraits<int8_t>::scale_for(top);
                *sc = (float)s;
                const double inv = 1.0 / (double)(float)s;
                for (int x = 0; x < 9; ++x) c[x * stride] = CodeTraits<int8_t>::encode(S[x], inv);
            } else {
                for (int x = 0; x < 9; ++x) c[x * stride] = CodeTraits<half_t>::encode(S[x], matrix_inv);
            }
        };
        // The remainder first: in place, its values may lie under the records.
        if constexpr (upper) {
            const ptrdiff_t parts = (ptrdiff_t)part_row.size() - 1;
#pragma omp parallel for schedule(static, 1)
            for (ptrdiff_t p = 0; p < parts; ++p) {
                const ptrdiff_t end = part_row[(size_t)p + 1];
                size_t e = (size_t)rem_ptr[(size_t)p];
                for (ptrdiff_t i = part_row[(size_t)p]; i < end; ++i)
                    for (count_t k = rp[i]; k < rp[i + 1]; ++k) {
                        if (ci[k] <= i || in_sell(i, ci[k], end)) continue;
                        double S[9];
                        for (int x = 0; x < 9; ++x) S[x] = (double)values[(size_t)k * 9 + x];
                        put(S, &rem_codes[e * 9], 1, per_block ? &rem_scales[e] : nullptr);
                        ++e;
                    }
            }
        }
        // The records of window w's groups, step s at base + (s - step0) * R.
        constexpr ptrdiff_t GW = SIGMA / VS;
        auto window = [&](const ptrdiff_t w, uint8_t *const base, const size_t step0) {
            for (ptrdiff_t g = w * GW; g < std::min(n_groups, (w + 1) * GW); ++g) {
                const ptrdiff_t end = upper ? part_end(rows[(size_t)g * VS]) : 0;
                for (int l = 0; l < VS; ++l) {
                    const idx_t i = rows[(size_t)g * VS + l];
                    size_t step = (size_t)group_ptr[(size_t)g];
                    auto c_at = [&](const size_t st) { return reinterpret_cast<code_t *>(base + (st - step0) * R) + l; };
                    auto s_at = [&](const size_t st) { return reinterpret_cast<float *>(base + (st - step0) * R + CODE_BYTES) + l; };
                    if (i >= 0) {
                        for (count_t k = rp[i]; k < rp[i + 1]; ++k) {
                            const idx_t j = ci[k];
                            if (upper ? !in_sell(i, j, end) : (difference && j == i)) continue;
                            double S[9];
                            if constexpr (upper) {
                                for (int x = 0; x < 9; ++x) S[x] = (double)values[(size_t)k * 9 + x];
                            } else {
                                const count_t kt = (count_t)(std::lower_bound(ci + rp[j], ci + rp[j + 1], (idx_t)i) - ci);
                                for (int r = 0; r < 3; ++r)
                                    for (int c = 0; c < 3; ++c)
                                        S[r * 3 + c] = 0.5 * ((double)values[(size_t)k * 9 + r * 3 + c] +
                                                              (double)values[(size_t)kt * 9 + c * 3 + r]);
                            }
                            if constexpr (!upper) colidx[step * VS + l] = j;
                            put(S, c_at(step), VS, per_block ? s_at(step) : nullptr);
                            ++step;
                        }
                    }
                    // Padding: the row's own column (lane 0's past the last row), zero codes.
                    const idx_t self = i >= 0 ? i : rows[(size_t)g * VS];
                    for (; step < (size_t)group_ptr[(size_t)g + 1]; ++step) {
                        if constexpr (!upper) colidx[step * VS + l] = self;
                        if constexpr (per_block) *s_at(step) = 0;
                        for (int x = 0; x < 9; ++x) c_at(step)[x * VS] = code_t(0);
                    }
                }
            }
        };
        rec = dest;
        const ptrdiff_t windows = (n_rows + SIGMA - 1) / SIGMA;
        auto rec_begin = [&](const ptrdiff_t w) { return (size_t)group_ptr[(size_t)std::min(n_groups, w * GW)] * R; };
        auto val_begin = [&](const ptrdiff_t w) { return (size_t)rp[std::min(n_rows, w * SIGMA)] * 9 * sizeof(V); };
        const bool aliased = (const void *)dest == (const void *)values;
        ptrdiff_t a = 0;
        while (a < windows) {
            ptrdiff_t b = windows;
            if (aliased)
                for (b = a; b < windows && rec_begin(b + 1) <= val_begin(a);) ++b;
            if (b == a) {
                // Window a alone, through a scratch: its records overlap its own values.
                if (rec_begin(a + 1) > val_begin(a + 1)) {
                    fprintf(stderr, "QuantBsr::encode_in_place: the records of window %td outgrow its values\n", a);
                    abort();
                }
                std::vector<uint8_t> scratch(rec_begin(a + 1) - rec_begin(a));
                window(a, scratch.data(), (size_t)group_ptr[(size_t)(a * GW)]);
                std::copy(scratch.begin(), scratch.end(), dest + rec_begin(a));
                ++a;
                continue;
            }
#pragma omp parallel for schedule(dynamic, 1)
            for (ptrdiff_t w = a; w < b; ++w) window(w, dest + rec_begin(w), (size_t)group_ptr[(size_t)(w * GW)]);
            a = b;
        }
    }

    // y = A x, or y += A x with ADD; components interleaved, arithmetic in tau.
    template <typename tau, bool ADD>
    void apply(const double *const RSTR x, double *const RSTR y) const {
        if constexpr (upper) {
            apply_upper<tau, ADD>(x, y);
            return;
        }
        const idx_t *const RSTR rw = rows.data();
        const count_t *const RSTR gp = group_ptr.data();
        const col_t *const RSTR ci = colidx.data();
        const tau out_scale = per_block ? tau(1) : tau(scale);
#pragma omp parallel for schedule(static)
        for (ptrdiff_t g = 0; g < n_groups; ++g) {
            const idx_t *const RSTR r = &rw[(size_t)g * VS];
            double xi0[VS], xi1[VS], xi2[VS];
            tau a0[VS], a1[VS], a2[VS];
            for (int l = 0; l < VS; ++l) {
                const size_t i = (size_t)(r[l] >= 0 ? r[l] : r[0]);
                xi0[l] = difference ? x[i * 3 + 0] : 0;
                xi1[l] = difference ? x[i * 3 + 1] : 0;
                xi2[l] = difference ? x[i * 3 + 2] : 0;
                a0[l] = a1[l] = a2[l] = 0;
            }
            for (count_t step = gp[g]; step < gp[g + 1]; ++step) {
                const col_t *const RSTR cj = &ci[(size_t)step * VS];
                const code_t *const RSTR b = step_codes((size_t)step);
                const float *const RSTR s = per_block ? step_scales((size_t)step) : nullptr;
#pragma omp simd
                for (int l = 0; l < VS; ++l) {
                    const double *const xj = &x[(size_t)cj[l] * 3];
                    // Differences in the vector's precision, so a translation cancels exactly.
                    tau d0 = tau(xj[0] - xi0[l]), d1 = tau(xj[1] - xi1[l]), d2 = tau(xj[2] - xi2[l]);
                    if constexpr (per_block) {
                        d0 *= tau(s[l]);
                        d1 *= tau(s[l]);
                        d2 *= tau(s[l]);
                    }
                    a0[l] += tau(b[0 * VS + l]) * d0 + tau(b[1 * VS + l]) * d1 + tau(b[2 * VS + l]) * d2;
                    a1[l] += tau(b[3 * VS + l]) * d0 + tau(b[4 * VS + l]) * d1 + tau(b[5 * VS + l]) * d2;
                    a2[l] += tau(b[6 * VS + l]) * d0 + tau(b[7 * VS + l]) * d1 + tau(b[8 * VS + l]) * d2;
                }
            }
            for (int l = 0; l < VS; ++l) {
                if (r[l] < 0) continue;
                const size_t i = (size_t)r[l];
                if constexpr (ADD) {
                    y[i * 3 + 0] += (double)(a0[l] * out_scale);
                    y[i * 3 + 1] += (double)(a1[l] * out_scale);
                    y[i * 3 + 2] += (double)(a2[l] * out_scale);
                } else {
                    y[i * 3 + 0] = (double)(a0[l] * out_scale);
                    y[i * 3 + 1] = (double)(a1[l] * out_scale);
                    y[i * 3 + 2] = (double)(a2[l] * out_scale);
                }
            }
        }
    }

    template <typename tau, bool ADD>
    void apply_upper(const double *const RSTR x, double *const RSTR y) const {
        const idx_t *const RSTR rw = rows.data();
        const count_t *const RSTR gp = group_ptr.data();
        const col_t *const RSTR ci = colidx.data();
        const tau out_scale = per_block ? tau(1) : tau(scale);
        const ptrdiff_t parts = (ptrdiff_t)part_row.size() - 1;
        const idx_t *const RSTR hr = halo_row.data();
        double *const RSTR hy = halo_y.data();
#pragma omp parallel
        {
#pragma omp for schedule(static, 1)
            for (ptrdiff_t p = 0; p < parts; ++p) {
                const ptrdiff_t end = part_row[(size_t)p + 1];
                if constexpr (!ADD)
                    for (ptrdiff_t v = part_row[(size_t)p] * 3; v < end * 3; ++v) y[v] = 0;
                for (ptrdiff_t v = (ptrdiff_t)halo_ptr[(size_t)p] * 3; v < (ptrdiff_t)halo_ptr[(size_t)p + 1] * 3; ++v) hy[v] = 0;
                for (ptrdiff_t g = part_group[(size_t)p]; g < part_group[(size_t)p + 1]; ++g) {
                    const idx_t *const RSTR r = &rw[(size_t)g * VS];
                    idx_t base[VS];
                    double xi0[VS], xi1[VS], xi2[VS];
                    tau a0[VS], a1[VS], a2[VS], t0[VS], t1[VS], t2[VS];
                    for (int l = 0; l < VS; ++l) {
                        base[l] = r[l] >= 0 ? r[l] : r[0];
                        const size_t i = (size_t)base[l];
                        xi0[l] = x[i * 3 + 0];
                        xi1[l] = x[i * 3 + 1];
                        xi2[l] = x[i * 3 + 2];
                        a0[l] = a1[l] = a2[l] = 0;
                    }
                    for (count_t step = gp[g]; step < gp[g + 1]; ++step) {
                        const col_t *const RSTR cj = &ci[(size_t)step * VS];
                        const code_t *const RSTR b = step_codes((size_t)step);
                        const float *const RSTR s = per_block ? step_scales((size_t)step) : nullptr;
                        idx_t jl[VS];
#pragma omp simd
                        for (int l = 0; l < VS; ++l) {
                            jl[l] = offsets ? base[l] + (idx_t)cj[l] : (idx_t)cj[l];
                            const double *const xj = &x[(size_t)jl[l] * 3];
                            tau d0 = tau(xj[0] - xi0[l]), d1 = tau(xj[1] - xi1[l]), d2 = tau(xj[2] - xi2[l]);
                            if constexpr (per_block) {
                                d0 *= tau(s[l]);
                                d1 *= tau(s[l]);
                                d2 *= tau(s[l]);
                            }
                            const tau c0 = tau(b[0 * VS + l]), c1 = tau(b[1 * VS + l]), c2 = tau(b[2 * VS + l]);
                            const tau c3 = tau(b[3 * VS + l]), c4 = tau(b[4 * VS + l]), c5 = tau(b[5 * VS + l]);
                            const tau c6 = tau(b[6 * VS + l]), c7 = tau(b[7 * VS + l]), c8 = tau(b[8 * VS + l]);
                            a0[l] += c0 * d0 + c1 * d1 + c2 * d2;
                            a1[l] += c3 * d0 + c4 * d1 + c5 * d2;
                            a2[l] += c6 * d0 + c7 * d1 + c8 * d2;
                            t0[l] = c0 * d0 + c3 * d1 + c6 * d2;
                            t1[l] = c1 * d0 + c4 * d1 + c7 * d2;
                            t2[l] = c2 * d0 + c5 * d1 + c8 * d2;
                        }
                        // Lanes in order: two rows of the group may share a neighbour.  Every column
                        // here is one of the part's own rows.
                        for (int l = 0; l < VS; ++l) {
                            double *const q = &y[(size_t)jl[l] * 3];
                            q[0] -= (double)(t0[l] * out_scale);
                            q[1] -= (double)(t1[l] * out_scale);
                            q[2] -= (double)(t2[l] * out_scale);
                        }
                    }
                    for (int l = 0; l < VS; ++l) {
                        if (r[l] < 0) continue;
                        const size_t i = (size_t)r[l];
                        y[i * 3 + 0] += (double)(a0[l] * out_scale);
                        y[i * 3 + 1] += (double)(a1[l] * out_scale);
                        y[i * 3 + 2] += (double)(a2[l] * out_scale);
                    }
                }
                // The remainder: columns beyond the part (into halo slots) or beyond the offsets' reach.
                for (count_t e = rem_ptr[(size_t)p]; e < rem_ptr[(size_t)p + 1]; ++e) {
                    const size_t i = (size_t)rem_row[(size_t)e], j = (size_t)rem_col[(size_t)e];
                    const code_t *const c = &rem_codes[(size_t)e * 9];
                    const tau sc = per_block ? tau(rem_scales[(size_t)e]) * out_scale : out_scale;
                    const tau d0 = tau(x[j * 3 + 0] - x[i * 3 + 0]) * sc, d1 = tau(x[j * 3 + 1] - x[i * 3 + 1]) * sc,
                              d2 = tau(x[j * 3 + 2] - x[i * 3 + 2]) * sc;
                    const tau c0 = tau(c[0]), c1 = tau(c[1]), c2 = tau(c[2]), c3 = tau(c[3]), c4 = tau(c[4]), c5 = tau(c[5]),
                              c6 = tau(c[6]), c7 = tau(c[7]), c8 = tau(c[8]);
                    y[i * 3 + 0] += (double)(c0 * d0 + c1 * d1 + c2 * d2);
                    y[i * 3 + 1] += (double)(c3 * d0 + c4 * d1 + c5 * d2);
                    y[i * 3 + 2] += (double)(c6 * d0 + c7 * d1 + c8 * d2);
                    const idx_t slot = rem_slot[(size_t)e];
                    double *const q = slot < 0 ? &y[j * 3] : &hy[(size_t)slot * 3];
                    q[0] -= (double)(c0 * d0 + c3 * d1 + c6 * d2);
                    q[1] -= (double)(c1 * d0 + c4 * d1 + c7 * d2);
                    q[2] -= (double)(c2 * d0 + c5 * d1 + c8 * d2);
                }
            }
            // The slots that point into this part's rows.
#pragma omp for schedule(static, 1)
            for (ptrdiff_t p = 0; p < parts; ++p)
                for (count_t e = halo_in_ptr[(size_t)p]; e < halo_in_ptr[(size_t)p + 1]; ++e) {
                    const size_t t = (size_t)halo_in[(size_t)e], i = (size_t)hr[t];
                    for (int c = 0; c < 3; ++c) y[i * 3 + c] += hy[t * 3 + c];
                }
        }
    }
};

}  // namespace qp
