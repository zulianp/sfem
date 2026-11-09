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

#include "sfem_cuda_InexactSetup.cuh"

namespace sfem {
    namespace inexact {
        namespace cuda {
            namespace {

                /// A block's record: int8 codes and the block's scale in 16 bytes (one 128-bit load), or
                /// f16 codes in 20 bytes under the matrix's scale.
                struct alignas(16) RecI8 {
                    int8_t  codes[9];
                    uint8_t pad[3];
                    float   scale;
                };
                struct alignas(4) RecF16 {
                    uint16_t codes[9];
                    uint16_t pad;
                };
                static_assert(sizeof(RecI8) == 16 && sizeof(RecF16) == 20, "record sizes");

                template <bool I8>
                struct Records;

                template <>
                struct Records<true> {
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
                struct Records<false> {
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

                /// The window's rows whose blocks do not sum to zero beyond the assembly's rounding (the
                /// host's `correction`): their sum R_i into the next slot (`count`, over the windows), and
                /// each row's slot or -1 (corrected_of); with f16 codes each row's scale, the window's.
                template <bool I8>
                __global__ void rows_kernel(const ptrdiff_t r0, const ptrdiff_t r1, const count_t *const RSTR rp, const float *const RSTR v,
                                            int *const RSTR count, idx_t *const RSTR rows, float *const RSTR blocks,
                                            int32_t *const RSTR corrected_of, const int *const RSTR top, float *const RSTR row_scale) {
                    const ptrdiff_t i = r0 + (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x;
                    if (i >= r1) return;
                    if constexpr (!I8) {
                        const float t = __int_as_float(*top);
                        row_scale[i]  = t > 0 ? (float)Code<__half>::scale_for((double)t) : 1.0f;
                    }
                    const count_t k0 = rp[r0];
                    double        R[9] = {0}, m = 0;
                    for (int64_t k = rp[i]; k < rp[i + 1]; ++k)
                        for (int x = 0; x < 9; ++x) {
                            R[x] += (double)v[(size_t)(k - k0) * 9 + x];
                            m = fmax(m, fabs((double)v[(size_t)(k - k0) * 9 + x]));
                        }
                    double sum = 0;
                    for (int x = 0; x < 9; ++x) sum = fmax(sum, fabs(R[x]));
                    if (!(sum > 64 * FLT_EPSILON * m)) {
                        corrected_of[i] = -1;
                        return;
                    }
                    const int at    = atomicAdd(count, 1);
                    rows[at]        = (idx_t)i;
                    corrected_of[i] = at;
                    for (int x = 0; x < 9; ++x) blocks[(size_t)at * 9 + x] = (float)R[x];
                }

                template <bool I8>
                __device__ __forceinline__ void encode_block(const double S[9], const float row_scale, void *const records, const size_t k) {
                    using rec_t      = typename Records<I8>::rec_t;
                    using code       = typename Records<I8>::code;
                    rec_t *const r = reinterpret_cast<rec_t *>(records) + k;
                    if constexpr (I8) {
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
                template <bool I8, bool SYM>
                __global__ void encode_kernel(const ptrdiff_t r0, const ptrdiff_t r1, const count_t *const RSTR rp, const idx_t *const RSTR ci,
                                              const float *const RSTR v, const float *const RSTR row_scale, void *const RSTR records) {
                    const ptrdiff_t i = r0 + (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x;
                    if (i >= r1) return;
                    const count_t k0 = rp[r0];
                    const float   rs = I8 ? 1.0f : row_scale[i];
                    for (int64_t k = rp[i]; k < rp[i + 1]; ++k) {
                        const idx_t j = ci[k];
                        if (j == i || (SYM && j < i)) continue;  // the diagonal's codes stay zero; the left ones are copies
                        double S[9];
                        if (SYM && j < r1) {
                            const size_t kt = (size_t)find(rp, ci, j, (idx_t)i);
                            for (int r = 0; r < 3; ++r)
                                for (int c = 0; c < 3; ++c)
                                    S[r * 3 + c] = 0.5 * ((double)v[(size_t)(k - k0) * 9 + r * 3 + c] + (double)v[(kt - k0) * 9 + c * 3 + r]);
                        } else {
                            for (int x = 0; x < 9; ++x) S[x] = (double)v[(size_t)(k - k0) * 9 + x];
                        }
                        encode_block<I8>(S, rs, records, (size_t)k);
                    }
                }

                /// A symmetric operator's blocks left of the diagonal: the transposes of the records of
                /// rows j < i, encoded before (in an earlier window or by encode_kernel in this one), so the
                /// encoded operator is symmetric to the bit. f16 codes move between the two rows' scales,
                /// powers of two, which is exact.
                template <bool I8>
                __global__ void transpose_kernel(const ptrdiff_t r0, const ptrdiff_t r1, const count_t *const RSTR rp, const idx_t *const RSTR ci,
                                                 const float *const RSTR row_scale, void *const RSTR records) {
                    using rec_t       = typename Records<I8>::rec_t;
                    const ptrdiff_t i = r0 + (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x;
                    if (i >= r1) return;
                    rec_t *const out = reinterpret_cast<rec_t *>(records);
                    for (int64_t k = rp[i]; k < rp[i + 1]; ++k) {
                        const idx_t j = ci[k];
                        if (j >= i) break;
                        const rec_t src = out[find(rp, ci, j, (idx_t)i)];
                        rec_t       dst = src;
                        if constexpr (I8) {
                            for (int r = 0; r < 3; ++r)
                                for (int c = 0; c < 3; ++c) dst.codes[r * 3 + c] = src.codes[c * 3 + r];
                        } else {
                            const float factor = row_scale[j] / row_scale[i];
                            for (int r = 0; r < 3; ++r)
                                for (int c = 0; c < 3; ++c)
                                    dst.codes[r * 3 + c] =
                                            __half_as_ushort(__float2half(__half2float(__ushort_as_half(src.codes[c * 3 + r])) * factor));
                        }
                        out[k] = dst;
                    }
                }

                // The records in the node graph's own order (rowptr, colidx), the diagonal slot's codes
                // zero; applied as BSR's group kernel applies its blocks (bsr_spmv_group_kernel): G
                // threads per row, each a stride of whole records, the partial sums added by shuffles in
                // a fixed order. A thread per row in SELL-32 order, the layout before, is latency bound on
                // an unstructured mesh's rows of uneven length (FF200: 60% occupancy, 37% of DRAM
                // bandwidth against BSR's 87% and 82%); on the FF200 valves this layout applies 1.4 to
                // 1.8 times as fast.

                /// y_i += sum_j K_ij (x_j - x_i) + R_i x_i (the row's correction), G threads per row.
                ///
                /// FUSED: y = that + the extra blocks, the composed tangent's passes (zero y, product,
                /// correction, curvature, shift) in one, in their order.
                template <bool I8, int G, bool FUSED>
                __global__ void __launch_bounds__(THREADS) apply_kernel(const ptrdiff_t n, const count_t *const RSTR rp,
                                                                           const idx_t *const RSTR ci, const void *const RSTR records,
                                                                           const float *const RSTR row_scale, const real_t *const RSTR x,
                                                                           real_t *const RSTR y, const int32_t *const RSTR corrected_of,
                                                                           const float *const RSTR cblocks, const DiagonalExtra extra) {
                    using rec_t       = typename Records<I8>::rec_t;
                    const ptrdiff_t t      = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x;
                    const ptrdiff_t i      = t / G;
                    const int       lane   = (int)(t % G);
                    const bool      active = i < n;
                    float           a0 = 0, a1 = 0, a2 = 0;
                    if (active) {
                        const real_t xi0 = x[(size_t)i * 3 + 0], xi1 = x[(size_t)i * 3 + 1], xi2 = x[(size_t)i * 3 + 2];
                        for (int64_t k = rp[i] + lane; k < rp[i + 1]; k += G) {
                            const idx_t j = __ldg(ci + k);
                            float       K[9], sc;
                            Records<I8>::load(reinterpret_cast<const rec_t *>(records) + k, K, sc);
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
                    if (active && lane == 0) {
                        const float  ms     = I8 ? 1.0f : row_scale[i];
                        const real_t xi[3]  = {x[(size_t)i * 3 + 0], x[(size_t)i * 3 + 1], x[(size_t)i * 3 + 2]};
                        real_t       out[3] = {FUSED ? real_t(0) : y[(size_t)i * 3 + 0], FUSED ? real_t(0) : y[(size_t)i * 3 + 1],
                                               FUSED ? real_t(0) : y[(size_t)i * 3 + 2]};
                        out[0] += (real_t)(a0 * ms);
                        out[1] += (real_t)(a1 * ms);
                        out[2] += (real_t)(a2 * ms);
                        const int32_t c = corrected_of[i];
                        if (c >= 0) {
                            const float *const R = &cblocks[(size_t)c * 9];
                            for (int a = 0; a < 3; ++a) out[a] += R[a * 3 + 0] * xi[0] + R[a * 3 + 1] * xi[1] + R[a * 3 + 2] * xi[2];
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
                }

                /// The diagonal: minus the stored blocks' diagonals of each row, plus its correction's.
                template <bool I8>
                __global__ void diagonal_kernel(const ptrdiff_t n, const count_t *const RSTR rp, const idx_t *const RSTR ci,
                                                const void *const RSTR records, const float *const RSTR row_scale,
                                                const int32_t *const RSTR corrected_of, const float *const RSTR cblocks, real_t *const RSTR diag) {
                    using rec_t       = typename Records<I8>::rec_t;
                    const ptrdiff_t i = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x;
                    if (i >= n) return;
                    const double ms   = I8 ? 1.0 : (double)row_scale[i];
                    double       d[3] = {0, 0, 0};
                    for (int64_t k = rp[i]; k < rp[i + 1]; ++k) {
                        if (ci[k] == i) continue;
                        float K[9], sc;
                        Records<I8>::load(reinterpret_cast<const rec_t *>(records) + k, K, sc);
                        const double f = (double)sc * ms;
                        for (int q = 0; q < 3; ++q) d[q] -= f * (double)K[q * 3 + q];
                    }
                    for (int q = 0; q < 3; ++q) diag[(size_t)i * 3 + q] += (real_t)d[q];
                    const int32_t c = corrected_of[i];
                    if (c >= 0)
                        for (int q = 0; q < 3; ++q) diag[(size_t)i * 3 + q] += cblocks[(size_t)c * 9 + q * 3 + q];
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
                inline int window_count(const size_t record_bytes) {
                    const char *const v = getenv("SFEM_QBSR_WINDOWS");
                    if (v) return std::max(1, std::min(12, atoi(v)));
                    return std::max(1, std::min(12, (int)std::ceil(2.0 * 9 * sizeof(float) / (double)record_bytes)));
                }

                template <bool I8, bool SYM>
                class DeviceBsrStore final : public BsrStore {
                public:
                    using rec_t = typename Records<I8>::rec_t;

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
                        name_ += ", " + std::to_string(g_) + " threads per row, " + std::to_string(bounds_.size() - 1) + " windows";
                        records_      = DeviceArray<rec_t>((size_t)rp[n_]);  // zero: the diagonal slots' codes
                        rp_           = DeviceArray<count_t>::from(rp, (std::size_t)n_ + 1);
                        ci_           = DeviceArray<idx_t>::from(ci, (std::size_t)rp[n_]);
                        window_rp_    = DeviceArray<count_t>((size_t)n_ + 1);
                        largest_      = (size_t)largest;
                        top_          = DeviceArray<int>(1);
                        count_        = DeviceArray<int>(1);
                        crows_        = DeviceArray<idx_t>((size_t)n_);
                        cblocks_      = DeviceArray<float>((size_t)n_ * 9);
                        corrected_of_ = DeviceArray<int32_t>((size_t)n_);
                        if constexpr (!I8) row_scale_ = DeviceArray<float>((size_t)n_);
                    }

                    /// Window by window: its graph, the assembly into its values (stream-ordered, freed at the
                    /// end), the rows' corrections and scales, the records; then, symmetric, the copies left
                    /// of the diagonal. No synchronization.
                    int update(const AssembleBSR &assemble) override {
                        SFEM_TRACE_SCOPE("inexact::cuda::BsrStore::update");
                        if (n_ == 0) return SFEM_SUCCESS;
                        float *values = nullptr;
                        check(cudaMallocAsync((void **)&values, largest_ * 9 * sizeof(float), stream_), "bsr values");
                        check(cudaMemsetAsync(count_.data, 0, sizeof(int), stream_), "bsr correction");
                        int status = SFEM_SUCCESS;
                        for (size_t w = 0; w + 1 < bounds_.size() && status == SFEM_SUCCESS; ++w) {
                            const ptrdiff_t r0 = bounds_[w], r1 = bounds_[w + 1];
                            window_rowptr_kernel<<<grid(n_ + 1, THREADS), THREADS, 0, stream_>>>(n_, rp_.data, r0, r1, window_rp_.data);
                            check(cudaMemsetAsync(values, 0, largest_ * 9 * sizeof(float), stream_), "bsr values");
                            const count_t k0 = host_rowptr(r0);
                            if (assemble(window_rp_.data, ci_.data + k0, smesh::SMESH_FLOAT32, values) != SFEM_SUCCESS) {
                                status = SFEM_FAILURE;
                                break;
                            }
                            if constexpr (!I8) {
                                check(cudaMemsetAsync(top_.data, 0, sizeof(int), stream_), "bsr scale");
                                max_kernel<<<1024, 256, 0, stream_>>>(r0, r1, rp_.data, ci_.data, values, top_.data);
                            }
                            rows_kernel<I8><<<grid(r1 - r0, THREADS), THREADS, 0, stream_>>>(r0, r1, rp_.data, values, count_.data, crows_.data,
                                                                                           cblocks_.data, corrected_of_.data, top_.data,
                                                                                           row_scale_.data);
                            encode_kernel<I8, SYM><<<grid(r1 - r0, THREADS), THREADS, 0, stream_>>>(r0, r1, rp_.data, ci_.data, values,
                                                                                                   row_scale_.data, records_.data);
                            if constexpr (SYM)
                                transpose_kernel<I8><<<grid(r1 - r0, THREADS), THREADS, 0, stream_>>>(r0, r1, rp_.data, ci_.data,
                                                                                                      row_scale_.data, records_.data);
                        }
                        check(cudaFreeAsync(values, stream_), "bsr values");
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
                            diagonal_kernel<I8><<<grid(n_, THREADS), THREADS, 0, stream_>>>(n_, rp_.data, ci_.data, records_.data, row_scale_.data,
                                                                                            corrected_of_.data, cblocks_.data, diag);
                        check(cudaGetLastError(), "bsr diagonal");
                        SFEM_DEBUG_SYNCHRONIZE(stream_);
                    }

                    /// The records, their structure (the node graph's device copy), the corrections and the
                    /// rows' scales, and a window's row pointers. The window's single-precision values exist
                    /// during the update only (largest window: `scratch_bytes`).
                    std::size_t bytes() const override {
                        return records_.bytes() + rp_.bytes() + ci_.bytes() + window_rp_.bytes() + crows_.bytes() + cblocks_.bytes() +
                               corrected_of_.bytes() + row_scale_.bytes() + top_.bytes() + count_.bytes();
                    }

                    std::string describe() const override { return name_; }

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
                    DeviceArray<count_t>   window_rp_;
                    DeviceArray<int>       top_, count_;
                    DeviceArray<idx_t>     crows_;
                    DeviceArray<float>     cblocks_;
                    DeviceArray<int32_t>   corrected_of_;  // per node: its correction, -1 for none
                    DeviceArray<float>     row_scale_;     // f16: per node, its window's scale

                    count_t host_rowptr(const ptrdiff_t row) const { return host_rp_[std::lower_bound(bounds_.begin(), bounds_.end(), row) - bounds_.begin()]; }

                    template <bool FUSED>
                    void launch(const real_t *const x, real_t *const y, const DiagonalExtra &extra) const {
                        if (n_ == 0) return;
                        if (g_ == 4)
                            apply_kernel<I8, 4, FUSED><<<grid(n_ * 4, THREADS), THREADS, 0, stream_>>>(
                                    n_, rp_.data, ci_.data, records_.data, row_scale_.data, x, y, corrected_of_.data, cblocks_.data, extra);
                        else
                            apply_kernel<I8, 8, FUSED><<<grid(n_ * 8, THREADS), THREADS, 0, stream_>>>(
                                    n_, rp_.data, ci_.data, records_.data, row_scale_.data, x, y, corrected_of_.data, cblocks_.data, extra);
                        check(cudaGetLastError(), "bsr apply");
                        SFEM_DEBUG_SYNCHRONIZE(stream_);
                    }
                };

            }  // namespace

            std::unique_ptr<BsrStore> create_bsr_store(const std::shared_ptr<CRSGraph> &graph,
                                                       const bool                       int8,
                                                       const bool                       symmetric,
                                                       void *const                      stream) {
                if (symmetric) {
                    if (int8) return std::make_unique<DeviceBsrStore<true, true>>(graph, "bsr i8 (device)", stream);
                    return std::make_unique<DeviceBsrStore<false, true>>(graph, "bsr f16 (device)", stream);
                }
                if (int8) return std::make_unique<DeviceBsrStore<true, false>>(graph, "bsr i8 unsymmetric (device)", stream);
                return std::make_unique<DeviceBsrStore<false, false>>(graph, "bsr f16 unsymmetric (device)", stream);
            }

        }  // namespace cuda
    }  // namespace inexact
}  // namespace sfem
