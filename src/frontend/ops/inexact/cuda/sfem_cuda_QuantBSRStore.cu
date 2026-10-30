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

                constexpr int       C     = 32;    // rows side by side: one warp
                constexpr ptrdiff_t SIGMA = 1024;  // the sorting window

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

                struct Sell {
                    ptrdiff_t       n_groups{0};
                    const idx_t    *rows{nullptr};       // [group][lane], -1 past the last row
                    const int64_t  *group_ptr{nullptr};  // steps of each group
                    const int32_t  *cols{nullptr};       // [step][lane]; padding: the row's own
                    void           *records{nullptr};    // [step][lane]
                    float           matrix_scale{1};     // f16
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

                /// The largest off-diagonal entry, as the bits of a non-negative float (atomicMax on int).
                __global__ void max_kernel(const ptrdiff_t n, const count_t *const RSTR rp, const idx_t *const RSTR ci,
                                           const float *const RSTR v, int *const RSTR top) {
                    float m = 0;
                    for (ptrdiff_t i = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (ptrdiff_t)blockDim.x * gridDim.x)
                        for (int64_t k = rp[i]; k < rp[i + 1]; ++k)
                            if (ci[k] != i)
                                for (int y = 0; y < 9; ++y) m = fmaxf(m, fabsf(v[(size_t)k * 9 + y]));
                    for (int o = 16; o > 0; o /= 2) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, o));
                    if ((threadIdx.x & 31) == 0) atomicMax(top, __float_as_int(m));
                }

                /// The rows whose blocks do not sum to zero beyond the assembly's rounding (the host's
                /// `correction`): counted with `rows` null, written otherwise.
                __global__ void correction_kernel(const ptrdiff_t n, const count_t *const RSTR rp, const float *const RSTR v,
                                                  int *const RSTR count, idx_t *const RSTR rows, float *const RSTR blocks) {
                    const ptrdiff_t i = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x;
                    if (i >= n) return;
                    double R[9] = {0}, top = 0;
                    for (int64_t k = rp[i]; k < rp[i + 1]; ++k)
                        for (int x = 0; x < 9; ++x) {
                            R[x] += (double)v[(size_t)k * 9 + x];
                            top = fmax(top, fabs((double)v[(size_t)k * 9 + x]));
                        }
                    double sum = 0;
                    for (int x = 0; x < 9; ++x) sum = fmax(sum, fabs(R[x]));
                    if (!(sum > 64 * FLT_EPSILON * top)) return;
                    const int at = atomicAdd(count, 1);
                    if (!rows) return;
                    rows[at] = (idx_t)i;
                    for (int x = 0; x < 9; ++x) blocks[(size_t)at * 9 + x] = (float)R[x];
                }

                /// The records: a thread per row, its steps in order; a symmetric operator's block from
                /// (K_ij + K_ji^T) / 2, an unsymmetric one's as assembled.
                template <bool I8, bool SYM>
                __global__ void encode_kernel(const Sell s, const count_t *const RSTR rp, const idx_t *const RSTR ci,
                                              const float *const RSTR v, const double matrix_inv) {
                    using rec_t       = typename Records<I8>::rec_t;
                    using code        = typename Records<I8>::code;
                    const ptrdiff_t t = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x;
                    const ptrdiff_t g = t / C;
                    const int       l = (int)(t % C);
                    if (g >= s.n_groups) return;
                    const idx_t i = s.rows[g * C + l];
                    for (int64_t step = s.group_ptr[g]; step < s.group_ptr[g + 1]; ++step) {
                        const size_t slot = (size_t)step * C + l;
                        const idx_t  j    = s.cols[slot];
                        if (i < 0 || j == i) continue;  // padding: zero codes from the structure
                        const size_t k = (size_t)find(rp, ci, i, j);
                        double       S[9];
                        if constexpr (SYM) {
                            const size_t kt = (size_t)find(rp, ci, j, i);
                            for (int r = 0; r < 3; ++r)
                                for (int c = 0; c < 3; ++c) S[r * 3 + c] = 0.5 * ((double)v[k * 9 + r * 3 + c] + (double)v[kt * 9 + c * 3 + r]);
                        } else {
                            for (int x = 0; x < 9; ++x) S[x] = (double)v[k * 9 + x];
                        }
                        rec_t *const r = reinterpret_cast<rec_t *>(s.records) + slot;
                        if constexpr (I8) {
                            double top = 0;
                            for (int x = 0; x < 9; ++x) top = fmax(top, fabs(S[x]));
                            const double sc  = code::scale_for(top);
                            const double inv = 1.0 / (double)(float)sc;
                            r->scale         = (float)sc;
                            for (int x = 0; x < 9; ++x) r->codes[x] = code::encode(S[x], inv);
                        } else {
                            for (int x = 0; x < 9; ++x) r->codes[x] = __half_as_ushort(code::encode(S[x], matrix_inv));
                        }
                    }
                }

                /// y_i += sum_j K_ij (x_j - x_i): no scatter, deterministic.
                template <bool I8>
                __global__ void __launch_bounds__(THREADS) sell_kernel(const Sell s, const real_t *const RSTR x, real_t *const RSTR y) {
                    using rec_t       = typename Records<I8>::rec_t;
                    const ptrdiff_t t = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x;
                    const ptrdiff_t g = t / C;
                    const int       l = (int)(t % C);
                    if (g >= s.n_groups) return;
                    const idx_t i = s.rows[g * C + l];
                    if (i < 0) return;
                    const real_t xi0 = x[(size_t)i * 3 + 0], xi1 = x[(size_t)i * 3 + 1], xi2 = x[(size_t)i * 3 + 2];
                    float        a0 = 0, a1 = 0, a2 = 0;
                    for (int64_t step = s.group_ptr[g]; step < s.group_ptr[g + 1]; ++step) {
                        const size_t  slot = (size_t)step * C + l;
                        const int32_t j    = __ldg(s.cols + slot);
                        if (j == i) continue;  // padding
                        float K[9], sc;
                        Records<I8>::load(reinterpret_cast<const rec_t *>(s.records) + slot, K, sc);
                        const float d0 = float(x[(size_t)j * 3 + 0] - xi0) * sc, d1 = float(x[(size_t)j * 3 + 1] - xi1) * sc,
                                    d2 = float(x[(size_t)j * 3 + 2] - xi2) * sc;
                        a0 += K[0] * d0 + K[1] * d1 + K[2] * d2;
                        a1 += K[3] * d0 + K[4] * d1 + K[5] * d2;
                        a2 += K[6] * d0 + K[7] * d1 + K[8] * d2;
                    }
                    const float ms = s.matrix_scale;
                    y[(size_t)i * 3 + 0] += (real_t)(a0 * ms);
                    y[(size_t)i * 3 + 1] += (real_t)(a1 * ms);
                    y[(size_t)i * 3 + 2] += (real_t)(a2 * ms);
                }

                /// The diagonal: minus the stored blocks' diagonals of each row.
                template <bool I8>
                __global__ void diagonal_kernel(const Sell s, real_t *const RSTR diag) {
                    using rec_t       = typename Records<I8>::rec_t;
                    const ptrdiff_t t = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x;
                    const ptrdiff_t g = t / C;
                    const int       l = (int)(t % C);
                    if (g >= s.n_groups) return;
                    const idx_t i = s.rows[g * C + l];
                    if (i < 0) return;
                    double d[3] = {0, 0, 0};
                    for (int64_t step = s.group_ptr[g]; step < s.group_ptr[g + 1]; ++step) {
                        const size_t slot = (size_t)step * C + l;
                        if (s.cols[slot] == i) continue;
                        float K[9], sc;
                        Records<I8>::load(reinterpret_cast<const rec_t *>(s.records) + slot, K, sc);
                        const double f = (double)sc * (double)s.matrix_scale;
                        for (int q = 0; q < 3; ++q) d[q] -= f * (double)K[q * 3 + q];
                    }
                    for (int q = 0; q < 3; ++q) diag[(size_t)i * 3 + q] += (real_t)d[q];
                }

                /// y_i += R_i x_i, and diag_i += diag(R_i), over the corrected rows.
                __global__ void correction_apply_kernel(const int n, const idx_t *const RSTR rows, const float *const RSTR blocks,
                                                        const real_t *const RSTR x, real_t *const RSTR y) {
                    const int r = blockIdx.x * blockDim.x + threadIdx.x;
                    if (r >= n) return;
                    const ptrdiff_t     i  = rows[r];
                    const float *const  R  = &blocks[(size_t)r * 9];
                    const real_t *const xi = &x[i * 3];
                    for (int a = 0; a < 3; ++a) y[i * 3 + a] += R[a * 3 + 0] * xi[0] + R[a * 3 + 1] * xi[1] + R[a * 3 + 2] * xi[2];
                }

                __global__ void correction_diagonal_kernel(const int n, const idx_t *const RSTR rows, const float *const RSTR blocks,
                                                           real_t *const RSTR diag) {
                    const int r = blockIdx.x * blockDim.x + threadIdx.x;
                    if (r >= n) return;
                    for (int d = 0; d < 3; ++d) diag[(size_t)rows[r] * 3 + d] += blocks[(size_t)r * 9 + d * 3 + d];
                }

                template <bool I8, bool SYM>
                class DeviceBsrStore final : public BsrStore {
                public:
                    using rec_t = typename Records<I8>::rec_t;

                    DeviceBsrStore(const std::shared_ptr<CRSGraph> &graph, std::string name, void *const stream)
                        : name_(std::move(name)), stream_((cudaStream_t)stream), n_(graph->n_nodes()) {
                        const count_t *const rp = graph->rowptr()->data();
                        const idx_t *const   ci = graph->colidx()->data();
                        std::vector<int32_t> len((size_t)n_);
#pragma omp parallel for
                        for (ptrdiff_t i = 0; i < n_; ++i) {
                            int32_t c = 0;
                            for (count_t k = rp[i]; k < rp[i + 1]; ++k) c += ci[k] != i;
                            len[(size_t)i] = c;
                        }
                        std::vector<idx_t> order((size_t)n_);
                        for (ptrdiff_t i = 0; i < n_; ++i) order[(size_t)i] = (idx_t)i;
#pragma omp parallel for
                        for (ptrdiff_t w = 0; w < n_; w += SIGMA)
                            std::stable_sort(order.begin() + w, order.begin() + std::min(n_, w + SIGMA),
                                             [&](const idx_t x, const idx_t y) { return len[(size_t)x] > len[(size_t)y]; });
                        const ptrdiff_t    ng = (n_ + C - 1) / C;
                        std::vector<idx_t> rows((size_t)ng * C, -1);
                        std::copy(order.begin(), order.end(), rows.begin());
                        std::vector<int64_t> gp((size_t)ng + 1, 0);
                        for (ptrdiff_t g = 0; g < ng; ++g) {
                            int32_t m = 0;
                            for (int l = 0; l < C; ++l) {
                                const idx_t i = rows[(size_t)g * C + l];
                                if (i >= 0) m = std::max(m, len[(size_t)i]);
                            }
                            gp[(size_t)g + 1] = gp[(size_t)g] + m;
                        }
                        const size_t         slots = (size_t)gp[(size_t)ng] * C;
                        std::vector<int32_t> cols(slots);
#pragma omp parallel for schedule(dynamic, 64)
                        for (ptrdiff_t g = 0; g < ng; ++g)
                            for (int l = 0; l < C; ++l) {
                                const idx_t i    = rows[(size_t)g * C + l];
                                size_t      step = (size_t)gp[(size_t)g];
                                if (i >= 0)
                                    for (count_t k = rp[i]; k < rp[i + 1]; ++k)
                                        if (ci[k] != i) cols[step++ * C + l] = (int32_t)ci[k];
                                const idx_t self = i >= 0 ? i : rows[(size_t)g * C];
                                for (; step < (size_t)gp[(size_t)g + 1]; ++step) cols[step * C + l] = (int32_t)self;
                            }
                        rows_    = DeviceArray<idx_t>::from(rows);
                        gp_      = DeviceArray<int64_t>::from(gp);
                        cols_    = DeviceArray<int32_t>::from(cols);
                        records_ = DeviceArray<rec_t>(slots);  // zero: the padding's codes
                        rp_      = DeviceArray<count_t>::from(std::vector<count_t>(rp, rp + n_ + 1));
                        ci_      = DeviceArray<idx_t>::from(std::vector<idx_t>(ci, ci + rp[n_]));
                        values_  = DeviceArray<float>((size_t)rp[n_] * 9);
                        top_     = DeviceArray<int>(1);
                        count_   = DeviceArray<int>(1);
                        s_.n_groups  = ng;
                        s_.rows      = rows_.data;
                        s_.group_ptr = gp_.data;
                        s_.cols      = cols_.data;
                        s_.records   = records_.data;
                    }

                    int update(const AssembleBSR &assemble) override {
                        check(cudaMemsetAsync(values_.data, 0, values_.bytes(), stream_), "bsr values");
                        if (assemble(rp_.data, ci_.data, smesh::SMESH_FLOAT32, values_.data) != SFEM_SUCCESS) return SFEM_FAILURE;
                        correction();
                        double matrix_inv = 1;
                        if constexpr (!I8) {
                            check(cudaMemsetAsync(top_.data, 0, sizeof(int), stream_), "bsr scale");
                            max_kernel<<<1024, 256, 0, stream_>>>(n_, rp_.data, ci_.data, values_.data, top_.data);
                            int bits = 0;
                            check(cudaMemcpyAsync(&bits, top_.data, sizeof(int), cudaMemcpyDeviceToHost, stream_), "bsr scale");
                            check(cudaStreamSynchronize(stream_), "bsr scale");
                            float top;
                            std::memcpy(&top, &bits, sizeof(float));
                            // The host's matrix scale (CodeTraits<half_t>::scale_for, in double).
                            s_.matrix_scale = top > 0 ? (float)std::exp2(std::ceil(std::log2((double)top)) - 14) : 1.0f;
                            matrix_inv      = 1.0 / (double)s_.matrix_scale;
                        }
                        if (s_.n_groups > 0)
                            encode_kernel<I8, SYM><<<grid(s_.n_groups * C, THREADS), THREADS, 0, stream_>>>(s_, rp_.data, ci_.data, values_.data,
                                                                                                         matrix_inv);
                        check(cudaGetLastError(), "bsr encode");
                        SFEM_DEBUG_SYNCHRONIZE(stream_);
                        return SFEM_SUCCESS;
                    }

                    void apply(const real_t *const x, real_t *const y) const override {
                        if (s_.n_groups > 0) sell_kernel<I8><<<grid(s_.n_groups * C, THREADS), THREADS, 0, stream_>>>(s_, x, y);
                        if (n_corrected_ > 0)
                            correction_apply_kernel<<<grid(n_corrected_, THREADS), THREADS, 0, stream_>>>(n_corrected_, crows_.data,
                                                                                                         cblocks_.data, x, y);
                        check(cudaGetLastError(), "bsr apply");
                        SFEM_DEBUG_SYNCHRONIZE(stream_);
                    }

                    void add_diagonal(real_t *const diag) const override {
                        if (s_.n_groups > 0) diagonal_kernel<I8><<<grid(s_.n_groups * C, THREADS), THREADS, 0, stream_>>>(s_, diag);
                        if (n_corrected_ > 0)
                            correction_diagonal_kernel<<<grid(n_corrected_, THREADS), THREADS, 0, stream_>>>(n_corrected_, crows_.data,
                                                                                                            cblocks_.data, diag);
                        check(cudaGetLastError(), "bsr diagonal");
                        SFEM_DEBUG_SYNCHRONIZE(stream_);
                    }

                    /// The structure and records, the corrections, and the single-precision assembly
                    /// the records are encoded from.
                    std::size_t bytes() const override {
                        return rows_.bytes() + gp_.bytes() + cols_.bytes() + records_.bytes() + crows_.bytes() + cblocks_.bytes() +
                               values_.bytes();
                    }

                    std::string describe() const override { return name_ + ", " + std::to_string(n_corrected_) + " corrected rows"; }

                private:
                    std::string            name_;
                    cudaStream_t           stream_;
                    ptrdiff_t              n_;
                    Sell                   s_;
                    DeviceArray<idx_t>     rows_;
                    DeviceArray<int64_t>   gp_;
                    DeviceArray<int32_t>   cols_;
                    DeviceArray<rec_t>     records_;
                    DeviceArray<count_t>   rp_;
                    DeviceArray<idx_t>     ci_;
                    DeviceArray<float>     values_;
                    DeviceArray<int>       top_, count_;
                    DeviceArray<idx_t>     crows_;
                    DeviceArray<float>     cblocks_;
                    int                    n_corrected_{0};

                    /// Counts the rows to correct, sizes their arrays, then writes them.
                    void correction() {
                        if (n_ == 0) return;
                        check(cudaMemsetAsync(count_.data, 0, sizeof(int), stream_), "bsr correction");
                        correction_kernel<<<grid(n_, THREADS), THREADS, 0, stream_>>>(n_, rp_.data, values_.data, count_.data, nullptr, nullptr);
                        check(cudaMemcpyAsync(&n_corrected_, count_.data, sizeof(int), cudaMemcpyDeviceToHost, stream_), "bsr correction");
                        check(cudaStreamSynchronize(stream_), "bsr correction");
                        if (n_corrected_ == 0) return;
                        if ((size_t)n_corrected_ > crows_.size) {
                            crows_   = DeviceArray<idx_t>((size_t)n_corrected_);
                            cblocks_ = DeviceArray<float>((size_t)n_corrected_ * 9);
                        }
                        check(cudaMemsetAsync(count_.data, 0, sizeof(int), stream_), "bsr correction");
                        correction_kernel<<<grid(n_, THREADS), THREADS, 0, stream_>>>(n_, rp_.data, values_.data, count_.data, crows_.data,
                                                                                      cblocks_.data);
                        check(cudaGetLastError(), "bsr correction");
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
