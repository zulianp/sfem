#include "sfem_cuda_QuantBSRStore.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "sfem_base.hpp"
#include "sfem_cuda_InexactSetup.cuh"

namespace sfem {
    namespace inexact {
        namespace cuda {
            namespace {

                /// A block's record: int8 codes and the block's scale in 16 bytes (one 128-bit load), f16
                /// codes in 20 bytes under the row's scale, or the f32 values in 36 bytes.
                struct alignas(16) RecI8 {
                    int8_t  codes[9];
                    uint8_t pad[3];
                    float   scale;
                };
                struct alignas(4) RecF16 {
                    uint16_t codes[9];
                    uint16_t pad;
                };
                struct alignas(4) RecF32 {
                    float codes[9];
                };
                static_assert(sizeof(RecI8) == 16 && sizeof(RecF16) == 20 && sizeof(RecF32) == 36, "record sizes");

                template <typename T>
                struct Records;

                template <>
                struct Records<int8_t> {
                    using rec_t = RecI8;
                    using code  = Code<int8_t>;
                    __device__ static void load(const rec_t *const r, float K[9], float &sc) {
                        const uint4    w     = __ldg(reinterpret_cast<const uint4 *>(r));
                        const uint32_t ws[3] = {w.x, w.y, w.z};
                        for (int e = 0; e < 9; ++e) K[e] = float((int8_t)((ws[e >> 2] >> (8 * (e & 3))) & 0xff));
                        sc = __uint_as_float(w.w);
                    }
                };

                template <>
                struct Records<__half> {
                    using rec_t = RecF16;
                    using code  = Code<__half>;
                    __device__ static void load(const rec_t *const r, float K[9], float &sc) {
                        const uint32_t *const p = reinterpret_cast<const uint32_t *>(r);
                        uint32_t              ws[5];
                        for (int q = 0; q < 5; ++q) ws[q] = __ldg(p + q);
                        for (int e = 0; e < 9; ++e) K[e] = __half2float(__ushort_as_half((uint16_t)((ws[e >> 1] >> (16 * (e & 1))) & 0xffff)));
                        sc = 1;
                    }
                };

                template <>
                struct Records<float> {
                    using rec_t = RecF32;
                    using code  = Code<float>;
                    __device__ static void load(const rec_t *const r, float K[9], float &sc) {
                        for (int e = 0; e < 9; ++e) K[e] = __ldg(&r->codes[e]);
                        sc = 1;
                    }
                };

                template <typename T>
                constexpr bool scaled_by_row = std::is_same<T, __half>::value;  // f16: a scale per row

                __device__ __forceinline__ int64_t find(const count_t *const RSTR rp, const idx_t *const RSTR ci, const idx_t i, const idx_t j) {
                    int64_t lo = rp[i], hi = rp[i + 1];
                    while (lo < hi) {
                        const int64_t mid = (lo + hi) / 2;
                        if (ci[mid] < j)
                            lo = mid + 1;
                        else
                            hi = mid;
                    }
                    return lo;
                }

                /// The record of entry k of the node graph, (i, j) with j != i: the records hold the
                /// off-diagonal blocks only, in the graph's order, and every row has its diagonal.
                __device__ __forceinline__ int64_t slot(const int64_t k, const ptrdiff_t i, const idx_t j) { return k - i - (j > i ? 1 : 0); }

                // ------------------------------------------------------------------ the layouts
                // Where the record of graph entry k = (i, j) lies. GraphOrder: the graph's own order
                // (`slot`). SellOrder: SELL-C-sigma (C = 32, sigma = 1024), the rows sorted by length
                // within windows of sigma rows, C side by side; row i is lane l of group g at p = pos[i],
                // and its q-th off-diagonal block, in the graph's order, is the record of step gp[g] + q,
                // at (gp[g] + q) C + l.
                static constexpr int       SELL_C     = 32;
                static constexpr ptrdiff_t SELL_SIGMA = 1024;

                struct GraphOrder {
                    __device__ int64_t operator()(const int64_t k, const ptrdiff_t i, const idx_t j) const { return slot(k, i, j); }
                };

                struct SellOrder {
                    const count_t *RSTR rp;
                    const idx_t *RSTR   pos;
                    const int64_t *RSTR gp;
                    __device__ int64_t  operator()(const int64_t k, const ptrdiff_t i, const idx_t j) const {
                        const idx_t p = pos[i];
                        return (gp[p / SELL_C] + (slot(k, i, j) - ((int64_t)rp[i] - i))) * SELL_C + p % SELL_C;
                    }
                };

                // ------------------------------------------------------------------ the update, by windows
                // The tangent is assembled and encoded a window of rows at a time, so the single-precision
                // assembly never exists in full: a window's graph has the node graph's rows of the window
                // and no entries in the others (window_rowptr_kernel; its columns are the node graph's of
                // those rows), which the assembly kernels skip. `v` is the window's values, entry k of the
                // node graph at v[(k - k0) * 9], k0 = rowptr[r0].

                /// The window graph's row pointers: rowptr's of rows [r0, r1) less k0, empty rows elsewhere.
                __global__ void window_rowptr_kernel(const ptrdiff_t n, const count_t *const RSTR rp, const ptrdiff_t r0, const ptrdiff_t r1,
                                                     count_t *const RSTR out) {
                    const ptrdiff_t i = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x;
                    if (i > n) return;
                    const ptrdiff_t c = i < r0 ? r0 : (i > r1 ? r1 : i);
                    out[i]            = rp[c] - rp[r0];
                }

                /// The largest off-diagonal entry of the window's rows, as the bits of a non-negative float
                /// (atomicMax on int).
                __global__ void max_kernel(const ptrdiff_t r0, const ptrdiff_t r1, const count_t *const RSTR rp, const idx_t *const RSTR ci,
                                           const float *const RSTR v, int *const RSTR top) {
                    const count_t k0 = rp[r0];
                    float         m  = 0;
                    for (ptrdiff_t i = r0 + (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; i < r1; i += (ptrdiff_t)blockDim.x * gridDim.x)
                        for (int64_t k = rp[i]; k < rp[i + 1]; ++k)
                            if (ci[k] != i)
                                for (int y = 0; y < 9; ++y) m = fmaxf(m, fabsf(v[(size_t)(k - k0) * 9 + y]));
                    for (int o = 16; o > 0; o /= 2) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, o));
                    if ((threadIdx.x & 31) == 0) atomicMax(top, __float_as_int(m));
                }

                /// The window's rows' sums R_i = sum_j K_ij: a row whose sum is a term of the operator
                /// (row_sum_is_a_term; none for a material, the inertia being kept beside the store) gets the next correction,
                /// corr_of[i] its index and sums[c] its block; any other row annihilates translations and
                /// the difference form alone is its product (corr_of[i] = -1), the paper's form. With f16
                /// codes each row's scale, the window's.
                template <typename T>
                __global__ void rows_kernel(const ptrdiff_t r0, const ptrdiff_t r1, const count_t *const RSTR rp, const float *const RSTR v,
                                            float *const RSTR sums, idx_t *const RSTR corr_of, int *const RSTR count,
                                            const int *const RSTR top, float *const RSTR row_scale) {
                    const ptrdiff_t i = r0 + (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x;
                    if (i >= r1) return;
                    if constexpr (scaled_by_row<T>) {
                        const float t = __int_as_float(*top);
                        row_scale[i]  = t > 0 ? (float)Code<__half>::scale_for((double)t) : 1.0f;
                    }
                    const count_t k0 = rp[r0];
                    double        A[9] = {0}, largest = 0, sum = 0;
                    for (int64_t k = rp[i]; k < rp[i + 1]; ++k)
                        for (int x = 0; x < 9; ++x) {
                            const double e = (double)v[(size_t)(k - k0) * 9 + x];
                            A[x] += e;
                            largest = fmax(largest, fabs(e));
                        }
                    for (int x = 0; x < 9; ++x) sum = fmax(sum, fabs(A[x]));
                    if (!row_sum_is_a_term(sum, largest)) {
                        corr_of[i] = -1;
                        return;
                    }
                    const int c = atomicAdd(count, 1);
                    corr_of[i]  = (idx_t)c;
                    for (int x = 0; x < 9; ++x) sums[(size_t)c * 9 + x] = (float)A[x];
                }

                template <typename T>
                __device__ __forceinline__ void encode_block(const double S[9], const float row_scale, void *const records, const size_t k) {
                    using rec_t      = typename Records<T>::rec_t;
                    using code       = typename Records<T>::code;
                    rec_t *const r = reinterpret_cast<rec_t *>(records) + k;
                    if constexpr (std::is_same<T, float>::value) {
                        for (int x = 0; x < 9; ++x) r->codes[x] = (float)S[x];
                    } else if constexpr (std::is_same<T, int8_t>::value) {
                        double top = 0;
                        for (int x = 0; x < 9; ++x) top = fmax(top, fabs(S[x]));
                        const double sc  = code::scale_for(top);
                        const double inv = 1.0 / (double)(float)sc;
                        r->scale         = (float)sc;
                        for (int x = 0; x < 9; ++x) r->codes[x] = code::encode(S[x], inv);
                    } else {
                        const double inv = 1.0 / (double)row_scale;
                        for (int x = 0; x < 9; ++x) r->codes[x] = __half_as_ushort(code::encode(S[x], inv));
                    }
                }

                /// The records of the window's rows a symmetric operator encodes itself: the blocks right of
                /// the diagonal, from (K_ij + K_ji^T) / 2 where row j is in the window too and from K_ij where
                /// it is not. An unsymmetric operator's every block, as assembled.
                template <typename T, bool SYM, class L>
                __global__ void encode_kernel(const ptrdiff_t r0, const ptrdiff_t r1, const count_t *const RSTR rp, const idx_t *const RSTR ci,
                                              const float *const RSTR v, const float *const RSTR row_scale, void *const RSTR records,
                                              const L at) {
                    const ptrdiff_t i = r0 + (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x;
                    if (i >= r1) return;
                    const count_t k0 = rp[r0];
                    const float   rs = scaled_by_row<T> ? row_scale[i] : 1.0f;
                    for (int64_t k = rp[i]; k < rp[i + 1]; ++k) {
                        const idx_t j = ci[k];
                        if (j == i || (SYM && j < i)) continue;  // the diagonal has no record; the left ones are copies
                        double S[9];
                        if (SYM && j < r1) {
                            const size_t kt = (size_t)find(rp, ci, j, (idx_t)i);
                            for (int r = 0; r < 3; ++r)
                                for (int c = 0; c < 3; ++c)
                                    S[r * 3 + c] = 0.5 * ((double)v[(size_t)(k - k0) * 9 + r * 3 + c] + (double)v[(kt - k0) * 9 + c * 3 + r]);
                        } else {
                            for (int x = 0; x < 9; ++x) S[x] = (double)v[(size_t)(k - k0) * 9 + x];
                        }
                        encode_block<T>(S, rs, records, (size_t)at(k, i, j));
                    }
                }

                /// A symmetric operator's blocks left of the diagonal: the transposes of the records of
                /// rows j < i, encoded before (in an earlier window or by encode_kernel in this one), so the
                /// encoded operator is symmetric to the bit. f16 codes move between the two rows' scales,
                /// powers of two, which is exact.
                template <typename T, class L>
                __global__ void transpose_kernel(const ptrdiff_t r0, const ptrdiff_t r1, const count_t *const RSTR rp, const idx_t *const RSTR ci,
                                                 const float *const RSTR row_scale, void *const RSTR records, const L at) {
                    using rec_t       = typename Records<T>::rec_t;
                    const ptrdiff_t i = r0 + (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x;
                    if (i >= r1) return;
                    rec_t *const out = reinterpret_cast<rec_t *>(records);
                    for (int64_t k = rp[i]; k < rp[i + 1]; ++k) {
                        const idx_t j = ci[k];
                        if (j >= i) break;
                        const rec_t src = out[at(find(rp, ci, j, (idx_t)i), j, (idx_t)i)];
                        rec_t       dst = src;
                        if constexpr (!scaled_by_row<T>) {
                            for (int r = 0; r < 3; ++r)
                                for (int c = 0; c < 3; ++c) dst.codes[r * 3 + c] = src.codes[c * 3 + r];
                        } else {
                            const float factor = row_scale[j] / row_scale[i];
                            for (int r = 0; r < 3; ++r)
                                for (int c = 0; c < 3; ++c)
                                    dst.codes[r * 3 + c] =
                                            __half_as_ushort(__float2half(__half2float(__ushort_as_half(src.codes[c * 3 + r])) * factor));
                        }
                        out[at(k, i, j)] = dst;
                    }
                }

                // The records in the node graph's own order (rowptr, colidx), the diagonal left out
                // (`slot`); applied as BSR's group kernel applies its blocks (bsr_spmv_group_kernel): G
                // threads per row, each a stride of whole records, the partial sums added by shuffles in
                // a fixed order. A thread per row in SELL-32 order is latency bound on an unstructured
                // mesh's rows of uneven length (FF200: 60% occupancy, 37% of DRAM bandwidth against
                // BSR's 87% and 82%); on the FF200 valves this layout applies 1.4 to 1.8 times as fast.
                // On a mesh of regular rows (the structured cubes) SELL is the faster one (sell_kernel).

                /// Row i's output from its off-diagonal sum (a0, a1, a2): y_i (+)= a ms (+ R_i x_i with CORR, where
                /// row i has a correction), and with
                /// FUSED the extra blocks, the composed tangent's passes (zero y, product, correction,
                /// curvature, shift) in one, in their order.
                template <typename T, bool FUSED, bool CORR, bool DIAG>
                __device__ __forceinline__ void finish(const ptrdiff_t i, const float a0, const float a1, const float a2,
                                                       const float *const RSTR row_scale, const real_t *const RSTR x, real_t *const RSTR y,
                                                       const idx_t *const RSTR corr_of, const float *const RSTR corr,
                                                       const real_t *const RSTR diag, const DiagonalExtra &extra) {
                    const float  ms     = scaled_by_row<T> ? row_scale[i] : 1.0f;
                    const real_t xi[3]  = {x[(size_t)i * 3 + 0], x[(size_t)i * 3 + 1], x[(size_t)i * 3 + 2]};
                    real_t       out[3] = {FUSED ? real_t(0) : y[(size_t)i * 3 + 0], FUSED ? real_t(0) : y[(size_t)i * 3 + 1],
                                           FUSED ? real_t(0) : y[(size_t)i * 3 + 2]};
                    out[0] += (real_t)(a0 * ms);
                    out[1] += (real_t)(a1 * ms);
                    out[2] += (real_t)(a2 * ms);
                    if constexpr (DIAG)
                        for (int a = 0; a < 3; ++a) out[a] += diag[(size_t)i * 3 + a] * xi[a];
                    if constexpr (CORR) {
                        const idx_t c = corr_of[i];
                        if (c >= 0) {
                            const float *const R = &corr[(size_t)c * 9];
                            for (int a = 0; a < 3; ++a) out[a] += R[a * 3 + 0] * xi[0] + R[a * 3 + 1] * xi[1] + R[a * 3 + 2] * xi[2];
                        }
                    }
                    if constexpr (FUSED) {
                        if (extra.contact_of) {
                            const idx_t b = extra.contact_of[i];
                            if (b >= 0) packed_sym_apply(&extra.blocks[(size_t)b * 6], xi, out);
                        }
                        if (extra.shift != 0)
                            for (int a = 0; a < 3; ++a) out[a] += extra.shift * (extra.reference[(size_t)i * 3 + a] * xi[a]);
                    }
                    for (int a = 0; a < 3; ++a) y[(size_t)i * 3 + a] = out[a];
                }

                /// y_i += sum_{j != i} K_ij (x_j - x_i) (+ R_i x_i), G threads per row, the records in the
                /// graph's order.
                template <typename T, int G, bool FUSED, bool CORR, bool DIAG>
                __global__ void __launch_bounds__(THREADS) apply_kernel(const ptrdiff_t n, const count_t *const RSTR rp,
                                                                           const idx_t *const RSTR ci, const void *const RSTR records,
                                                                           const float *const RSTR row_scale, const real_t *const RSTR x,
                                                                           real_t *const RSTR y, const idx_t *const RSTR corr_of,
                                                                           const float *const RSTR corr, const real_t *const RSTR diag,
                                                                           const DiagonalExtra extra) {
                    using rec_t       = typename Records<T>::rec_t;
                    const ptrdiff_t t      = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x;
                    const ptrdiff_t i      = t / G;
                    const int       lane   = (int)(t % G);
                    const bool      active = i < n;
                    float           a0 = 0, a1 = 0, a2 = 0;
                    if (active) {
                        const real_t xi0 = x[(size_t)i * 3 + 0], xi1 = x[(size_t)i * 3 + 1], xi2 = x[(size_t)i * 3 + 2];
                        for (int64_t k = rp[i] + lane; k < rp[i + 1]; k += G) {
                            const idx_t j = __ldg(ci + k);
                            if (j == i) continue;
                            float K[9], sc;
                            Records<T>::load(reinterpret_cast<const rec_t *>(records) + slot(k, i, j), K, sc);
                            const float d0 = float(x[(size_t)j * 3 + 0] - xi0) * sc, d1 = float(x[(size_t)j * 3 + 1] - xi1) * sc,
                                        d2 = float(x[(size_t)j * 3 + 2] - xi2) * sc;
                            a0 += K[0] * d0 + K[1] * d1 + K[2] * d2;
                            a1 += K[3] * d0 + K[4] * d1 + K[5] * d2;
                            a2 += K[6] * d0 + K[7] * d1 + K[8] * d2;
                        }
                    }
                    if constexpr (G > 1) {
#pragma unroll
                        for (int o = G / 2; o > 0; o /= 2) {
                            a0 += __shfl_down_sync(0xffffffffu, a0, o, G);
                            a1 += __shfl_down_sync(0xffffffffu, a1, o, G);
                            a2 += __shfl_down_sync(0xffffffffu, a2, o, G);
                        }
                    }
                    if (active && lane == 0) finish<T, FUSED, CORR, DIAG>(i, a0, a1, a2, row_scale, x, y, corr_of, corr, diag, extra);
                }

                /// The same product in SELL-C-sigma: a thread per row, lane l of a group reading step s's
                /// record and column at s C + l, so a warp's loads are contiguous. A padding slot holds the
                /// row's own column (zero codes) and is skipped.
                template <typename T, bool FUSED, bool CORR, bool DIAG>
                __global__ void __launch_bounds__(THREADS) sell_kernel(const ptrdiff_t n_groups, const idx_t *const RSTR rows,
                                                                          const int64_t *const RSTR gp, const idx_t *const RSTR cols,
                                                                          const void *const RSTR records, const float *const RSTR row_scale,
                                                                          const real_t *const RSTR x, real_t *const RSTR y,
                                                                          const idx_t *const RSTR corr_of, const float *const RSTR corr,
                                                                          const real_t *const RSTR diag, const DiagonalExtra extra) {
                    using rec_t       = typename Records<T>::rec_t;
                    const ptrdiff_t t = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x;
                    const ptrdiff_t g = t / SELL_C;
                    if (g >= n_groups) return;
                    const idx_t i = rows[t];
                    if (i < 0) return;
                    const int    l   = (int)(t % SELL_C);
                    const real_t xi0 = x[(size_t)i * 3 + 0], xi1 = x[(size_t)i * 3 + 1], xi2 = x[(size_t)i * 3 + 2];
                    float        a0 = 0, a1 = 0, a2 = 0;
                    for (int64_t step = gp[g]; step < gp[g + 1]; ++step) {
                        const size_t s = (size_t)step * SELL_C + l;
                        const idx_t  j = __ldg(cols + s);
                        if (j == i) continue;
                        float K[9], sc;
                        Records<T>::load(reinterpret_cast<const rec_t *>(records) + s, K, sc);
                        const float d0 = float(x[(size_t)j * 3 + 0] - xi0) * sc, d1 = float(x[(size_t)j * 3 + 1] - xi1) * sc,
                                    d2 = float(x[(size_t)j * 3 + 2] - xi2) * sc;
                        a0 += K[0] * d0 + K[1] * d1 + K[2] * d2;
                        a1 += K[3] * d0 + K[4] * d1 + K[5] * d2;
                        a2 += K[6] * d0 + K[7] * d1 + K[8] * d2;
                    }
                    finish<T, FUSED, CORR, DIAG>(i, a0, a1, a2, row_scale, x, y, corr_of, corr, diag, extra);
                }

                /// The diagonal: minus the stored blocks' diagonals of each row, plus R_i's where it has a correction.
                template <typename T, class L>
                __global__ void diagonal_kernel(const ptrdiff_t n, const count_t *const RSTR rp, const idx_t *const RSTR ci,
                                                const void *const RSTR records, const float *const RSTR row_scale,
                                                const idx_t *const RSTR corr_of, const float *const RSTR corr, real_t *const RSTR diag,
                                                const L at) {
                    using rec_t       = typename Records<T>::rec_t;
                    const ptrdiff_t i = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x;
                    if (i >= n) return;
                    const double ms   = scaled_by_row<T> ? (double)row_scale[i] : 1.0;
                    double       d[3] = {0, 0, 0};
                    for (int64_t k = rp[i]; k < rp[i + 1]; ++k) {
                        const idx_t j = ci[k];
                        if (j == i) continue;
                        float K[9], sc;
                        Records<T>::load(reinterpret_cast<const rec_t *>(records) + at(k, i, j), K, sc);
                        const double f = (double)sc * ms;
                        for (int q = 0; q < 3; ++q) d[q] -= f * (double)K[q * 3 + q];
                    }
                    const idx_t c = corr_of[i];
                    if (c >= 0)
                        for (int q = 0; q < 3; ++q) d[q] += (double)corr[(size_t)c * 9 + q * 3 + q];
                    for (int q = 0; q < 3; ++q) diag[(size_t)i * 3 + q] += (real_t)d[q];
                }

                /// y += E x, E the extra diagonal blocks (curvature, then shift).
                __global__ void extra_kernel(const ptrdiff_t n, const DiagonalExtra extra, const real_t *const RSTR x, real_t *const RSTR y) {
                    const ptrdiff_t i = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x;
                    if (i >= n) return;
                    if (extra.contact_of && extra.contact_of[i] >= 0) packed_sym_apply(&extra.blocks[(size_t)extra.contact_of[i] * 6], &x[i * 3], &y[i * 3]);
                    if (extra.shift != 0)
                        for (int a = 0; a < 3; ++a) y[i * 3 + a] += extra.shift * (extra.reference[i * 3 + a] * x[i * 3 + a]);
                }

                /// The windows of a new store: as many as keep a window's single-precision values below half
                /// the records' bytes (f16 4, i8 5). FF200 TET10: 1, 2, 4, 8 windows hold 323, 161, 81, 40
                /// bytes per dof while the update runs and take 8.3, 8.4, 8.9, 11.3 ms; SFEM_QBSR_WINDOWS
                /// (1 to 12) overrides it for such a study.
                inline int window_count(const size_t record_bytes) { return qbsr_window_count(9 * sizeof(float), (double)record_bytes); }

                template <typename T, bool SYM, bool SELL>
                class DeviceBsrStore final : public BsrStore {
                public:
                    using rec_t = typename Records<T>::rec_t;

                    DeviceBsrStore(const std::shared_ptr<CRSGraph> &graph, std::string name, void *const stream)
                        : name_(std::move(name)), stream_((cudaStream_t)stream), n_(graph->n_nodes()) {
                        const count_t *const rp = graph->rowptr()->data();
                        const idx_t *const   ci = graph->colidx()->data();
                        // Threads per row from the rows' mean length (off-diagonal blocks): 4 where they
                        // are short (TET4, ~12), 8 where they are long (TET10, ~26); 16 was slower on both.
                        const double mean = n_ > 0 ? (double)(rp[n_] - n_) / (double)n_ : 0;
                        g_                = mean <= 16 ? 4 : 8;
                        // Windows of about equal entries.
                        const int    windows = window_count(sizeof(rec_t));
                        const double per     = (double)rp[n_] / windows;
                        bounds_.push_back(0);
                        for (ptrdiff_t i = 0; i < n_; ++i)
                            if ((double)rp[i + 1] >= per * (double)bounds_.size() && (int)bounds_.size() < windows) bounds_.push_back(i + 1);
                        if (bounds_.back() != n_) bounds_.push_back(n_);
                        count_t largest = 0;
                        for (size_t w = 0; w + 1 < bounds_.size(); ++w) largest = std::max(largest, rp[bounds_[w + 1]] - rp[bounds_[w]]);
                        for (const ptrdiff_t b : bounds_) host_rp_.push_back(rp[b]);
                        // The records skip the diagonal (`slot`), which assumes every row has it.
                        for (ptrdiff_t i = 0; i < n_; ++i) {
                            if (!std::binary_search(ci + rp[i], ci + rp[i + 1], (idx_t)i)) {
                                SFEM_ERROR("inexact::cuda::BsrStore: row %ld of the node graph has no diagonal\n", (long)i);
                            }
                        }
                        size_t records = (size_t)(rp[n_] - n_);
                        if constexpr (SELL) {
                            records = sell_structure(rp, ci);
                            name_ += ", SELL-" + std::to_string(SELL_C) + "-" + std::to_string(SELL_SIGMA);
                        } else {
                            name_ += ", " + std::to_string(g_) + " threads per row";
                        }
                        name_ += ", " + std::to_string(bounds_.size() - 1) + " windows";
                        records_ = DeviceArray<rec_t>(records);  // zero: the padding's codes (SELL)
                        rp_      = DeviceArray<count_t>::from(rp, (std::size_t)n_ + 1);
                        ci_      = DeviceArray<idx_t>::from(ci, (std::size_t)rp[n_]);
                        largest_ = (size_t)largest;
                        corr_of_ = DeviceArray<idx_t>((size_t)n_);
                        count_   = DeviceArray<int>(1);
                        if constexpr (scaled_by_row<T>) {
                            row_scale_ = DeviceArray<float>((size_t)n_);
                            top_       = DeviceArray<int>(1);
                        }
                    }

                    /// Window by window: its graph, the assembly into its values (both stream-ordered, freed at
                    /// the end), the rows' sums and scales, the records; then, symmetric, the copies left of
                    /// the diagonal. One synchronization at the end: the count of the rows with a correction,
                    /// which sizes their blocks and chooses the product (none: the difference form alone).
                    int update(const AssembleBSR &assemble) override {
                        SFEM_TRACE_SCOPE("inexact::cuda::BsrStore::update");
                        if (n_ == 0) return SFEM_SUCCESS;
                        float   *values    = nullptr;
                        count_t *window_rp = nullptr;
                        check(cudaMallocAsync((void **)&values, largest_ * 9 * sizeof(float), stream_), "bsr values");
                        check(cudaMallocAsync((void **)&window_rp, ((size_t)n_ + 1) * sizeof(count_t), stream_), "bsr window rowptr");
                        float *sums = nullptr;  // the corrections' blocks, at most one per row
                        check(cudaMallocAsync((void **)&sums, (size_t)n_ * 9 * sizeof(float), stream_), "bsr row sums");
                        check(cudaMemsetAsync(count_.data, 0, sizeof(int), stream_), "bsr corrections");
                        int        status = SFEM_SUCCESS;
                        const auto at     = layout();
                        for (size_t w = 0; w + 1 < bounds_.size() && status == SFEM_SUCCESS; ++w) {
                            const ptrdiff_t r0 = bounds_[w], r1 = bounds_[w + 1];
                            window_rowptr_kernel<<<grid(n_ + 1, THREADS), THREADS, 0, stream_>>>(n_, rp_.data, r0, r1, window_rp);
                            check(cudaMemsetAsync(values, 0, largest_ * 9 * sizeof(float), stream_), "bsr values");
                            const count_t k0 = host_rowptr(r0);
                            if (assemble(window_rp, ci_.data + k0, smesh::SMESH_FLOAT32, values) != SFEM_SUCCESS) {
                                status = SFEM_FAILURE;
                                break;
                            }
                            if constexpr (scaled_by_row<T>) {
                                check(cudaMemsetAsync(top_.data, 0, sizeof(int), stream_), "bsr scale");
                                max_kernel<<<1024, 256, 0, stream_>>>(r0, r1, rp_.data, ci_.data, values, top_.data);
                            }
                            rows_kernel<T><<<grid(r1 - r0, THREADS), THREADS, 0, stream_>>>(r0, r1, rp_.data, values, sums, corr_of_.data,
                                                                                           count_.data, top_.data, row_scale_.data);
                            encode_kernel<T, SYM><<<grid(r1 - r0, THREADS), THREADS, 0, stream_>>>(r0, r1, rp_.data, ci_.data, values,
                                                                                                  row_scale_.data, records_.data, at);
                            if constexpr (SYM)
                                transpose_kernel<T><<<grid(r1 - r0, THREADS), THREADS, 0, stream_>>>(r0, r1, rp_.data, ci_.data,
                                                                                                     row_scale_.data, records_.data, at);
                        }
                        check(cudaFreeAsync(values, stream_), "bsr values");
                        check(cudaFreeAsync(window_rp, stream_), "bsr window rowptr");
                        int count = 0;
                        check(cudaMemcpyAsync(&count, count_.data, sizeof(int), cudaMemcpyDeviceToHost, stream_), "bsr corrections");
                        check(cudaStreamSynchronize(stream_), "bsr corrections");
                        n_corr_ = status == SFEM_SUCCESS ? (ptrdiff_t)count : 0;
                        if ((size_t)n_corr_ * 9 > corr_.size) corr_ = DeviceArray<float>((size_t)n_corr_ * 9);
                        if (n_corr_ > 0)
                            check(cudaMemcpyAsync(corr_.data, sums, (size_t)n_corr_ * 9 * sizeof(float), cudaMemcpyDeviceToDevice, stream_),
                                  "bsr corrections");
                        check(cudaFreeAsync(sums, stream_), "bsr row sums");
                        check(cudaGetLastError(), "bsr encode");
                        SFEM_DEBUG_SYNCHRONIZE(stream_);
                        return status;
                    }

                    void apply(const real_t *const x, real_t *const y) const override { launch<false>(x, y, DiagonalExtra()); }

                    /// Overwriting: one kernel (the product, the row's correction, the extra blocks). Adding:
                    /// the plain product, then the extra blocks in a pass of their own.
                    void apply(const real_t *const x, real_t *const y, const DiagonalExtra &extra, const bool overwrite) const override {
                        if (overwrite) {
                            launch<true>(x, y, extra);
                            return;
                        }
                        launch<false>(x, y, DiagonalExtra());
                        if (n_ > 0) extra_kernel<<<grid(n_, THREADS), THREADS, 0, stream_>>>(n_, extra, x, y);
                        check(cudaGetLastError(), "bsr apply");
                        SFEM_DEBUG_SYNCHRONIZE(stream_);
                    }

                    void add_diagonal(real_t *const diag) const override {
                        if (n_ > 0)
                            diagonal_kernel<T><<<grid(n_, THREADS), THREADS, 0, stream_>>>(n_, rp_.data, ci_.data, records_.data, row_scale_.data,
                                                                                           corr_of_.data, corr_.data, diag, layout());
                        check(cudaGetLastError(), "bsr diagonal");
                        SFEM_DEBUG_SYNCHRONIZE(stream_);
                    }

                    /// The records, their structure (the node graph's device copy; SELL: its rows, steps,
                    /// columns and the rows' positions), the corrections (an index per row, a block per
                    /// corrected row) and the scales. A window's single-precision values and row pointers,
                    /// and the rows' sums, exist during the update only.
                    std::size_t bytes() const override {
                        return records_.bytes() + rp_.bytes() + ci_.bytes() + corr_of_.bytes() + (size_t)n_corr_ * 9 * sizeof(float) +
                               row_scale_.bytes() + top_.bytes() + count_.bytes() + rows_.bytes() + gp_.bytes() + cols_.bytes() + pos_.bytes();
                    }

                    std::string describe() const override { return name_ + ", " + std::to_string(n_corr_) + " corrected rows"; }

                    void set_diagonal(const real_t *const d) override { diag_ = d; }

                private:
                    std::string            name_;
                    cudaStream_t           stream_;
                    ptrdiff_t              n_;
                    int                    g_{8};
                    std::vector<ptrdiff_t> bounds_;    // the windows' first rows, and n
                    std::vector<count_t>   host_rp_;   // rowptr at the bounds
                    size_t                 largest_{0};
                    DeviceArray<rec_t>     records_;
                    DeviceArray<count_t>   rp_;
                    DeviceArray<idx_t>     ci_;
                    DeviceArray<int>       top_;
                    DeviceArray<idx_t>     corr_of_;    // per node: its correction's index, -1 for none
                    DeviceArray<float>     corr_;       // the corrections' blocks R_i (capacity; n_corr_ used)
                    DeviceArray<int>       count_;      // the update's count of corrections
                    ptrdiff_t              n_corr_{0};
                    const real_t          *diag_{nullptr};  // set_diagonal's, per dof
                    DeviceArray<float>     row_scale_;  // f16: per node, its window's scale
                    // SELL: [group][lane] rows (-1 past the last), the groups' first steps, [step][lane]
                    // columns (padding: the row itself), each row's position group * C + lane.
                    ptrdiff_t              n_groups_{0};
                    DeviceArray<idx_t>     rows_;
                    DeviceArray<int64_t>   gp_;
                    DeviceArray<idx_t>     cols_;
                    DeviceArray<idx_t>     pos_;

                    count_t host_rowptr(const ptrdiff_t row) const { return host_rp_[std::lower_bound(bounds_.begin(), bounds_.end(), row) - bounds_.begin()]; }

                    auto layout() const {
                        if constexpr (SELL)
                            return SellOrder{rp_.data, pos_.data, gp_.data};
                        else
                            return GraphOrder{};
                    }

                    /// SELL-C-sigma: the rows sorted by their off-diagonal count, longest first, within windows
                    /// of sigma rows, C side by side, each group padded to its longest row. Returns the slots.
                    size_t sell_structure(const count_t *const rp, const idx_t *const ci) {
                        constexpr ptrdiff_t  C = SELL_C;
                        std::vector<int32_t> len((size_t)n_);
                        for (ptrdiff_t i = 0; i < n_; ++i) len[(size_t)i] = (int32_t)(rp[i + 1] - rp[i] - 1);
                        std::vector<idx_t> order((size_t)n_);
                        for (ptrdiff_t i = 0; i < n_; ++i) order[(size_t)i] = (idx_t)i;
#pragma omp parallel for
                        for (ptrdiff_t w = 0; w < n_; w += SELL_SIGMA)
                            std::stable_sort(order.begin() + w, order.begin() + std::min(n_, w + SELL_SIGMA),
                                             [&](const idx_t x, const idx_t y) { return len[(size_t)x] > len[(size_t)y]; });
                        n_groups_ = (n_ + C - 1) / C;
                        std::vector<idx_t> rows((size_t)n_groups_ * C, -1), pos((size_t)n_);
                        std::copy(order.begin(), order.end(), rows.begin());
                        for (ptrdiff_t p = 0; p < n_; ++p) pos[(size_t)order[(size_t)p]] = (idx_t)p;
                        std::vector<int64_t> gp((size_t)n_groups_ + 1, 0);
                        for (ptrdiff_t g = 0; g < n_groups_; ++g) {
                            int32_t m = 0;
                            for (ptrdiff_t l = 0; l < C; ++l) {
                                const idx_t i = rows[(size_t)(g * C + l)];
                                if (i >= 0) m = std::max(m, len[(size_t)i]);
                            }
                            gp[(size_t)g + 1] = gp[(size_t)g] + m;
                        }
                        const size_t       slots = (size_t)gp[(size_t)n_groups_] * C;
                        std::vector<idx_t> cols(slots);
#pragma omp parallel for schedule(dynamic, 64)
                        for (ptrdiff_t g = 0; g < n_groups_; ++g)
                            for (ptrdiff_t l = 0; l < C; ++l) {
                                const idx_t i    = rows[(size_t)(g * C + l)];
                                size_t      step = (size_t)gp[(size_t)g];
                                if (i >= 0)
                                    for (count_t k = rp[i]; k < rp[i + 1]; ++k)
                                        if (ci[k] != i) cols[step++ * C + (size_t)l] = ci[k];
                                const idx_t self = i >= 0 ? i : rows[(size_t)(g * C)];
                                for (; step < (size_t)gp[(size_t)g + 1]; ++step) cols[step * C + (size_t)l] = self;
                            }
                        rows_ = DeviceArray<idx_t>::from(rows);
                        gp_   = DeviceArray<int64_t>::from(gp);
                        cols_ = DeviceArray<idx_t>::from(cols);
                        pos_  = DeviceArray<idx_t>::from(pos);
                        return slots;
                    }

                    /// The product: the difference form alone where no row has a correction (an operator
                    /// that annihilates translations), the corrections read where some do; the diagonal
                    /// (set_diagonal) in the same pass.
                    template <bool FUSED>
                    void launch(const real_t *const x, real_t *const y, const DiagonalExtra &extra) const {
                        if (n_ == 0) return;
                        if (n_corr_ > 0)
                            diag_ ? launch<FUSED, true, true>(x, y, extra) : launch<FUSED, true, false>(x, y, extra);
                        else
                            diag_ ? launch<FUSED, false, true>(x, y, extra) : launch<FUSED, false, false>(x, y, extra);
                    }

                    template <bool FUSED, bool CORR, bool DIAG>
                    void launch(const real_t *const x, real_t *const y, const DiagonalExtra &extra) const {
                        const idx_t *const  co = corr_of_.data;
                        const float *const  cb = corr_.data;
                        const real_t *const dg = diag_;
                        if constexpr (SELL) {
                            sell_kernel<T, FUSED, CORR, DIAG><<<grid(n_groups_ * SELL_C, THREADS), THREADS, 0, stream_>>>(
                                    n_groups_, rows_.data, gp_.data, cols_.data, records_.data, row_scale_.data, x, y, co, cb, dg, extra);
                        } else if (g_ == 4) {
                            apply_kernel<T, 4, FUSED, CORR, DIAG><<<grid(n_ * 4, THREADS), THREADS, 0, stream_>>>(
                                    n_, rp_.data, ci_.data, records_.data, row_scale_.data, x, y, co, cb, dg, extra);
                        } else {
                            apply_kernel<T, 8, FUSED, CORR, DIAG><<<grid(n_ * 8, THREADS), THREADS, 0, stream_>>>(
                                    n_, rp_.data, ci_.data, records_.data, row_scale_.data, x, y, co, cb, dg, extra);
                        }
                        check(cudaGetLastError(), "bsr apply");
                        SFEM_DEBUG_SYNCHRONIZE(stream_);
                    }
                };

                template <typename T, bool SELL>
                std::unique_ptr<BsrStore> make(const std::shared_ptr<CRSGraph> &graph, const bool symmetric, const char *const enc,
                                               void *const stream) {
                    const std::string name = std::string("bsr ") + enc + (symmetric ? "" : " unsymmetric") + " (device)";
                    if (symmetric) return std::make_unique<DeviceBsrStore<T, true, SELL>>(graph, name, stream);
                    return std::make_unique<DeviceBsrStore<T, false, SELL>>(graph, name, stream);
                }

                template <bool SELL>
                std::unique_ptr<BsrStore> make(const std::shared_ptr<CRSGraph> &graph, const InexactOptions::Encoding encoding,
                                               const bool symmetric, void *const stream) {
                    switch (encoding) {
                        case InexactOptions::Encoding::I8_BLOCK:
                            return make<int8_t, SELL>(graph, symmetric, "i8", stream);
                        case InexactOptions::Encoding::F16:
                            return make<__half, SELL>(graph, symmetric, "f16", stream);
                        case InexactOptions::Encoding::F32:
                            return make<float, SELL>(graph, symmetric, "f32", stream);
                        default:
                            SFEM_ERROR("inexact: the bsr store takes f32, f16 or i8\n");
                            return nullptr;
                    }
                }

            }  // namespace

            std::unique_ptr<BsrStore> create_bsr_store(const std::shared_ptr<CRSGraph> &graph,
                                                       const InexactOptions::Encoding   encoding,
                                                       const bool                       symmetric,
                                                       void *const                      stream,
                                                       const bool                       sell) {
                if (sell) return make<true>(graph, encoding, symmetric, stream);
                return make<false>(graph, encoding, symmetric, stream);
            }

        }  // namespace cuda
    }  // namespace inexact
}  // namespace sfem
