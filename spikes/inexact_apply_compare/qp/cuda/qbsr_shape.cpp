// GPU kernel shapes for quantized BSR (qbsr_kernels.hpp): which layout the device format takes.
//
//   qbsr_shape [n] [repeats]
//
// The node graph of this build's element on the spike's mesh (n cells per direction), random
// symmetric 3x3 blocks quantized once (int8 with a scale per block, f16 with one scale), so every
// layout encodes the same matrix and must give the same product to single-precision rounding.
// Rows: the layout's MDOF/s, bytes per dof, effective GB/s (store plus vectors read and written
// once), the remainder's share of the stored blocks, the relative l2 error against a host double
// product of the quantized matrix, and whether two products agree bitwise.  Baselines: the
// library's BSR kernel (sfem::d_bsr_spmv_native, f32 and f64 values) and cuSPARSE bsrmv on the
// same matrix, its diagonal blocks minus the sums of the off-diagonal ones.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cusparse.h>
#include <omp.h>

#include "sfem_base.hpp"
#include "sfem_BSR_cuda.hpp"
#ifndef RSTR
#define RSTR SFEM_RESTRICT
#endif

#include "element_mesh.inc"
#include "bsr_matrix.inc"

#include "qbsr_kernels.hpp"

#define CHECK_CUDA(x)                                                                                    \
    do {                                                                                                 \
        const cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                         \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));                  \
            exit(1);                                                                                     \
        }                                                                                                \
    } while (0)

using qbsr::Code;
using qbsr::Cols;
using qbsr::Form;
using qbsr::Rec;
static constexpr int C = qbsr::C;

// ------------------------------------------------------------------------------- device memory

struct Dev {
    std::vector<void *> ptrs;
    size_t bytes = 0;
    template <typename T>
    T *put(const std::vector<T> &v) {
        if (v.empty()) return nullptr;
        void *p = nullptr;
        CHECK_CUDA(cudaMalloc(&p, v.size() * sizeof(T)));
        CHECK_CUDA(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
        ptrs.push_back(p);
        bytes += v.size() * sizeof(T);
        return (T *)p;
    }
    ~Dev() {
        for (void *p : ptrs) cudaFree(p);
    }
};

// ---------------------------------------------------------------------------- the quantized matrix

// One quantized matrix on the full off-diagonal pattern of the graph: codes per CSR entry (row-major
// blocks), K_ji the transpose of K_ij with the same scale.
struct Quantized {
    Code code;
    std::vector<uint8_t> codes;  // [k][9] code_t
    std::vector<float> scales;   // [k] (int8)
    float matrix_scale = 1;
    size_t code_size() const { return code == Code::I8 ? 1 : (code == Code::F16 ? 2 : 4); }
};

// The block of pair (a, b), a < b, from a hash: deterministic, so either row can make it.
static void pair_block(const idx_t a, const idx_t b, float B[9]) {
    uint64_t s = ((uint64_t)(uint32_t)a << 32) ^ (uint64_t)(uint32_t)b;
    for (int e = 0; e < 9; ++e) {
        s += 0x9E3779B97F4A7C15ull;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        B[e] = (float)((double)(z >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0);
    }
}

static Quantized quantize(const BsrGraph &g, const Code code) {
    Quantized q;
    q.code = code;
    const size_t nnz = (size_t)g.nnz();
    q.codes.assign(nnz * 9 * q.code_size(), 0);
    if (code == Code::I8) q.scales.assign(nnz, 0.0f);
#pragma omp parallel for schedule(dynamic, 1024)
    for (ptrdiff_t i = 0; i < g.n_rows; ++i)
        for (count_t k = g.rowptr[(size_t)i]; k < g.rowptr[(size_t)i + 1]; ++k) {
            const idx_t j = g.colidx[(size_t)k];
            if (j == i) continue;
            float B[9], K[9];
            pair_block(std::min<idx_t>((idx_t)i, j), std::max<idx_t>((idx_t)i, j), B);
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) K[r * 3 + c] = i < j ? B[r * 3 + c] : B[c * 3 + r];
            if (code == Code::I8) {
                float top = 0;
                for (int e = 0; e < 9; ++e) top = std::max(top, std::fabs(K[e]));
                const float s = top > 0 ? top / 127.0f : 1.0f;
                q.scales[(size_t)k] = s;
                for (int e = 0; e < 9; ++e)
                    q.codes[(size_t)k * 9 + e] = (uint8_t)(int8_t)std::max(-127.0f, std::min(127.0f, std::nearbyint(K[e] / s)));
            } else if (code == Code::F32) {
                std::memcpy(&q.codes[(size_t)k * 36], K, 36);
            } else {
                uint16_t *const c = reinterpret_cast<uint16_t *>(q.codes.data()) + (size_t)k * 9;
                for (int e = 0; e < 9; ++e) {
                    const _Float16 h = (_Float16)K[e];
                    std::memcpy(&c[e], &h, 2);
                }
            }
        }
    return q;
}

static double block_value(const Quantized &q, const size_t k, const int e) {
    if (q.code == Code::I8) return (double)(int8_t)q.codes[k * 9 + e] * (double)q.scales[k];
    if (q.code == Code::F32) {
        float v;
        std::memcpy(&v, &q.codes[(k * 9 + e) * 4], 4);
        return (double)v;
    }
    _Float16 h;
    std::memcpy(&h, &q.codes[(k * 9 + e) * 2], 2);
    return (double)h * (double)q.matrix_scale;
}

// y_i = sum_{j != i} K_ij (x_j - x_i), in double.
static std::vector<double> reference(const BsrGraph &g, const Quantized &q, const std::vector<double> &x) {
    std::vector<double> y((size_t)g.n_rows * 3, 0.0);
#pragma omp parallel for schedule(dynamic, 1024)
    for (ptrdiff_t i = 0; i < g.n_rows; ++i) {
        double a[3] = {0, 0, 0};
        for (count_t k = g.rowptr[(size_t)i]; k < g.rowptr[(size_t)i + 1]; ++k) {
            const idx_t j = g.colidx[(size_t)k];
            if (j == i) continue;
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    a[r] += block_value(q, (size_t)k, r * 3 + c) * (x[(size_t)j * 3 + c] - x[(size_t)i * 3 + c]);
        }
        for (int r = 0; r < 3; ++r) y[(size_t)i * 3 + r] = a[r];
    }
    return y;
}

// ------------------------------------------------------------------------------------ layouts

struct Layout {
    Form form;
    Cols cols;
    Rec rec;
    ptrdiff_t sigma;
    std::string name;
};

struct Built {
    qbsr::Sell s;
    Dev dev;
    size_t stored = 0, remainder = 0;  // blocks
};

// The SELL layout of the quantized matrix, uploaded.
static void build(const BsrGraph &g, const Quantized &q, const Layout &L, Built &out) {
    const ptrdiff_t n = g.n_rows;
    const bool upper = L.form != Form::Full, gather = L.form == Form::UpperGather;
    const ptrdiff_t reach = L.cols == Cols::I32 ? std::numeric_limits<ptrdiff_t>::max()
                                                : (upper ? 65535 : 32767);
    const size_t cs = q.code_size();
    auto in_sell = [&](const ptrdiff_t i, const ptrdiff_t j) {
        if (j == i) return false;
        if (upper && j < i) return false;
        return std::abs(j - i) <= reach;
    };
    auto in_lower = [&](const ptrdiff_t i, const ptrdiff_t j) { return gather && j < i && i - j <= reach; };
    std::vector<int32_t> ulen((size_t)n), llen((size_t)n);
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < n; ++i) {
        int32_t u = 0, l = 0;
        for (count_t k = g.rowptr[(size_t)i]; k < g.rowptr[(size_t)i + 1]; ++k) {
            u += in_sell(i, g.colidx[(size_t)k]);
            l += in_lower(i, g.colidx[(size_t)k]);
        }
        ulen[(size_t)i] = u;
        llen[(size_t)i] = l;
    }
    std::vector<int32_t> order((size_t)n);
    std::iota(order.begin(), order.end(), 0);
    if (L.sigma > 1) {
#pragma omp parallel for
        for (ptrdiff_t w = 0; w < n; w += L.sigma)
            std::stable_sort(order.begin() + w, order.begin() + std::min(n, w + L.sigma), [&](int32_t a, int32_t b) {
                return ulen[(size_t)a] + llen[(size_t)a] > ulen[(size_t)b] + llen[(size_t)b];
            });
    }
    const ptrdiff_t ng = (n + C - 1) / C;
    std::vector<int32_t> rows((size_t)ng * C, -1);
    std::copy(order.begin(), order.end(), rows.begin());
    std::vector<int64_t> gp((size_t)ng + 1, 0), lgp(gather ? (size_t)ng + 1 : 0, 0);
    for (ptrdiff_t gr = 0; gr < ng; ++gr) {
        int32_t mu = 0, ml = 0;
        for (int l = 0; l < C; ++l) {
            const int32_t i = rows[(size_t)gr * C + l];
            if (i < 0) continue;
            mu = std::max(mu, ulen[(size_t)i]);
            ml = std::max(ml, llen[(size_t)i]);
        }
        gp[(size_t)gr + 1] = gp[(size_t)gr] + mu;
        if (gather) lgp[(size_t)gr + 1] = lgp[(size_t)gr] + ml;
    }
    const size_t steps = (size_t)gp[(size_t)ng], slots = steps * C;
    const size_t rec_size = q.code == Code::I8 ? sizeof(qbsr::RecI8) : (q.code == Code::F16 ? sizeof(qbsr::RecF16) : sizeof(qbsr::RecF32));
    std::vector<int32_t> cols32(L.cols == Cols::I32 ? slots : 0);
    std::vector<uint16_t> cols16(L.cols == Cols::Off16 && L.rec == Rec::Scalar ? slots : 0);
    std::vector<uint8_t> codes(L.rec == Rec::Scalar ? slots * 9 * cs : slots * rec_size, 0);
    std::vector<float> scales(L.rec == Rec::Scalar && q.code == Code::I8 ? slots : 0, 0.0f);
    std::vector<int32_t> slot_of(gather ? (size_t)g.nnz() : 0, -1);
    // Upper (or full) steps.
#pragma omp parallel for schedule(dynamic, 64)
    for (ptrdiff_t gr = 0; gr < ng; ++gr)
        for (int l = 0; l < C; ++l) {
            const int32_t i = rows[(size_t)gr * C + l];
            size_t step = (size_t)gp[(size_t)gr];
            auto put = [&](const size_t st, const ptrdiff_t j, const size_t k) {
                const size_t slot = st * C + l;
                const ptrdiff_t off = j - (i >= 0 ? i : 0);
                if (L.cols == Cols::I32) cols32[slot] = (int32_t)j;
                if (!cols16.empty()) cols16[slot] = (uint16_t)off;
                if (L.rec == Rec::Scalar) {
                    for (int e = 0; e < 9; ++e)
                        std::memcpy(&codes[((st * 9 + e) * C + l) * cs], k == SIZE_MAX ? (const uint8_t *)"\0\0\0\0" : &q.codes[(k * 9 + e) * cs], cs);
                    if (!scales.empty()) scales[slot] = k == SIZE_MAX ? 0.0f : q.scales[k];
                } else if (q.code == Code::I8) {
                    qbsr::RecI8 r{};
                    if (k != SIZE_MAX) {
                        std::memcpy(r.codes, &q.codes[k * 9], 9);
                        r.scale = q.scales[k];
                    }
                    r.off_lo = (uint8_t)((uint16_t)off & 0xff);
                    r.off_hi = (uint8_t)((uint16_t)off >> 8);
                    std::memcpy(&codes[slot * rec_size], &r, rec_size);
                } else if (q.code == Code::F16) {
                    qbsr::RecF16 r{};
                    if (k != SIZE_MAX) std::memcpy(r.codes, &q.codes[k * 18], 18);
                    r.off = (uint16_t)off;
                    std::memcpy(&codes[slot * rec_size], &r, rec_size);
                } else {
                    qbsr::RecF32 r{};
                    if (k != SIZE_MAX) std::memcpy(r.codes, &q.codes[k * 36], 36);
                    r.off = (uint32_t)(uint16_t)off;
                    std::memcpy(&codes[slot * rec_size], &r, rec_size);
                }
            };
            if (i >= 0)
                for (count_t k = g.rowptr[(size_t)i]; k < g.rowptr[(size_t)i + 1]; ++k) {
                    const idx_t j = g.colidx[(size_t)k];
                    if (!in_sell(i, j)) continue;
                    if (gather) slot_of[(size_t)k] = (int32_t)(step * C + l);
                    put(step++, j, (size_t)k);
                }
            // Padding: the row itself (lane 0's row past the last), zero codes.
            const int32_t self = i >= 0 ? i : rows[(size_t)gr * C];
            for (; step < (size_t)gp[(size_t)gr + 1]; ++step) put(step, i >= 0 ? self : 0, SIZE_MAX);
        }
    // Lower steps (gather): the partner's slot.
    std::vector<int32_t> lcol, lrec;
    if (gather) {
        const size_t lslots = (size_t)lgp[(size_t)ng] * C;
        lcol.assign(lslots, 0);
        lrec.assign(lslots, -1);
#pragma omp parallel for schedule(dynamic, 64)
        for (ptrdiff_t gr = 0; gr < ng; ++gr)
            for (int l = 0; l < C; ++l) {
                const int32_t i = rows[(size_t)gr * C + l];
                if (i < 0) continue;
                size_t step = (size_t)lgp[(size_t)gr];
                for (count_t k = g.rowptr[(size_t)i]; k < g.rowptr[(size_t)i + 1]; ++k) {
                    const idx_t j = g.colidx[(size_t)k];
                    if (!in_lower(i, j)) continue;
                    const count_t *const b = &g.colidx[(size_t)g.rowptr[(size_t)j]] - 0;
                    (void)b;
                    const idx_t *const rj = &g.colidx[(size_t)g.rowptr[(size_t)j]];
                    const size_t kt = (size_t)g.rowptr[(size_t)j] +
                                      (size_t)(std::lower_bound(rj, rj + (g.rowptr[(size_t)j + 1] - g.rowptr[(size_t)j]), (idx_t)i) - rj);
                    lcol[step * C + l] = j;
                    lrec[step * C + l] = slot_of[kt];
                    ++step;
                }
            }
    }
    // Remainder.
    std::vector<int64_t> rem_rowptr(1, 0);
    std::vector<int32_t> rem_rows, rem_row, rem_col;
    std::vector<uint8_t> rem_codes;
    std::vector<float> rem_scales;
    if (L.cols == Cols::Off16) {
        for (ptrdiff_t i = 0; i < n; ++i) {
            const size_t before = rem_col.size();
            for (count_t k = g.rowptr[(size_t)i]; k < g.rowptr[(size_t)i + 1]; ++k) {
                const idx_t j = g.colidx[(size_t)k];
                if (j == i || std::abs((ptrdiff_t)j - i) <= reach) continue;
                if (L.form == Form::Upper && j < i) continue;
                if (L.form == Form::Upper) rem_row.push_back((int32_t)i);
                rem_col.push_back(j);
                rem_codes.insert(rem_codes.end(), &q.codes[(size_t)k * 9 * cs], &q.codes[((size_t)k + 1) * 9 * cs]);
                if (q.code == Code::I8) rem_scales.push_back(q.scales[(size_t)k]);
            }
            if (L.form != Form::Upper && rem_col.size() > before) {
                rem_rows.push_back((int32_t)i);
                rem_rowptr.push_back((int64_t)rem_col.size());
            }
        }
    }
    out.stored = 0;
    for (ptrdiff_t i = 0; i < n; ++i) out.stored += (size_t)ulen[(size_t)i];
    out.remainder = L.form == Form::Upper ? rem_row.size() : rem_col.size();
    qbsr::Sell &s = out.s;
    s.n_rows = n;
    s.n_groups = ng;
    s.rows = out.dev.put(rows);
    s.group_ptr = out.dev.put(gp);
    s.cols = L.cols == Cols::I32 ? (const void *)out.dev.put(cols32) : (const void *)out.dev.put(cols16);
    s.codes = out.dev.put(codes);
    s.scales = out.dev.put(scales);
    s.matrix_scale = q.matrix_scale;
    if (gather) {
        s.lgroup_ptr = out.dev.put(lgp);
        s.lcol = out.dev.put(lcol);
        s.lrec = out.dev.put(lrec);
    }
    s.n_rem = (ptrdiff_t)out.remainder;
    s.n_rem_rows = (ptrdiff_t)rem_rows.size();
    s.rem_rowptr = out.dev.put(rem_rowptr);
    s.rem_rows = out.dev.put(rem_rows);
    s.rem_row = out.dev.put(rem_row);
    s.rem_col = out.dev.put(rem_col);
    s.rem_codes = out.dev.put(rem_codes);
    s.rem_scales = out.dev.put(rem_scales);
}

// ------------------------------------------------------------------------------------- timing

template <typename F>
static double best_seconds(const int repeats, F &&fn) {
    cudaEvent_t a, b;
    CHECK_CUDA(cudaEventCreate(&a));
    CHECK_CUDA(cudaEventCreate(&b));
    fn();
    CHECK_CUDA(cudaDeviceSynchronize());
    double best = 1e300;
    for (int attempt = 0; attempt < 3; ++attempt) {
        CHECK_CUDA(cudaEventRecord(a));
        for (int r = 0; r < repeats; ++r) fn();
        CHECK_CUDA(cudaEventRecord(b));
        CHECK_CUDA(cudaEventSynchronize(b));
        float ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms, a, b));
        best = std::min(best, ms * 1e-3 / repeats);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return best;
}

struct Problem {
    ptrdiff_t n_rows = 0;
    std::vector<double> x;
    double *dx = nullptr, *dy = nullptr, *dy2 = nullptr;
};

static double rel_l2(const std::vector<double> &y, const std::vector<double> &ref) {
    double num = 0, den = 0;
    for (size_t k = 0; k < y.size(); ++k) {
        num += (y[k] - ref[k]) * (y[k] - ref[k]);
        den += ref[k] * ref[k];
    }
    return std::sqrt(num / den);
}

// Times fn, checks it against ref and against itself, prints the row.
template <typename F>
static void row(const Problem &P, const std::string &name, const double store_bytes, const double rem_share,
                const std::vector<double> &ref, const int repeats, F &&fn) {
    const size_t nv = (size_t)P.n_rows * 3;
    CHECK_CUDA(cudaMemset(P.dy, 0, nv * sizeof(double)));
    fn(P.dy);
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<double> y(nv), y2(nv);
    CHECK_CUDA(cudaMemcpy(y.data(), P.dy, nv * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemset(P.dy2, 0, nv * sizeof(double)));
    fn(P.dy2);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(y2.data(), P.dy2, nv * sizeof(double), cudaMemcpyDeviceToHost));
    const bool bitwise = std::memcmp(y.data(), y2.data(), nv * sizeof(double)) == 0;
    const double err = rel_l2(y, ref);
    const double t = best_seconds(repeats, [&] { fn(P.dy); });
    const double dof = (double)nv;
    const double moved = store_bytes + 2.0 * nv * sizeof(double);
    printf("  %-40s %10.1f %8.2f %8.1f %7.2f %10.2e %5s\n", name.c_str(), dof / t * 1e-6, store_bytes / dof,
           moved / t * 1e-9, 100 * rem_share, err, bitwise ? "yes" : "no");
    fflush(stdout);
}

int main(int argc, char **argv) {
    static const int sizes[] = SIZES;
    const int n = argc > 1 ? std::atoi(argv[1]) : sizes[2];
    const int repeats = argc > 2 ? std::atoi(argv[2]) : 10;
    const char *only = std::getenv("QP_ONLY");
    auto selected = [&](const std::string &name) {
        if (!only) return true;
        std::string all(only);
        for (size_t a = 0, b; a <= all.size(); a = b + 1) {
            b = all.find('|', a);
            if (b == std::string::npos) b = all.size();
            if (b > a && name.find(all.substr(a, b - a)) != std::string::npos) return true;
        }
        return false;
    };

    const Mesh m = build(n);
    const BsrGraph g = build_bsr_graph(m);
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    Problem P;
    P.n_rows = g.n_rows;
    const size_t nv = (size_t)g.n_rows * 3;
    P.x.resize(nv);
    std::mt19937 gen(12345);
    std::uniform_real_distribution<double> dist(-1, 1);
    for (auto &v : P.x) v = dist(gen);
    CHECK_CUDA(cudaMalloc(&P.dx, nv * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&P.dy, nv * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&P.dy2, nv * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(P.dx, P.x.data(), nv * sizeof(double), cudaMemcpyHostToDevice));

    printf("qbsr_shape %s: n=%d, %ld nodes, %ld dof, %ld blocks, %s\n", ELEMENT_NAME, n, (long)g.n_rows, (long)nv,
           (long)g.nnz(), prop.name);
    printf("  %-40s %10s %8s %8s %7s %10s %5s\n", "layout", "MDOF/s", "B/dof", "GB/s", "rem %", "rel l2", "det");

    // Baselines: the assembled matrix with its diagonal, f64 and f32 values, from the int8 matrix.
    {
        const Quantized q8 = quantize(g, Code::I8);
        const std::vector<double> ref = reference(g, q8, P.x);
        std::vector<double> v64((size_t)g.nnz() * 9, 0.0);
#pragma omp parallel for schedule(dynamic, 1024)
        for (ptrdiff_t i = 0; i < g.n_rows; ++i) {
            count_t kd = -1;
            double diag[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
            for (count_t k = g.rowptr[(size_t)i]; k < g.rowptr[(size_t)i + 1]; ++k) {
                if (g.colidx[(size_t)k] == i) {
                    kd = k;
                    continue;
                }
                for (int e = 0; e < 9; ++e) {
                    v64[(size_t)k * 9 + e] = block_value(q8, (size_t)k, e);
                    diag[e] -= v64[(size_t)k * 9 + e];
                }
            }
            for (int e = 0; e < 9; ++e) v64[(size_t)kd * 9 + e] = diag[e];
        }
        std::vector<float> v32(v64.begin(), v64.end());
        Dev d;
        const count_t *rp = d.put(g.rowptr);
        const idx_t *ci = d.put(g.colidx);
        const size_t graph_bytes = d.bytes;
        const double *d64 = d.put(v64);
        const float *d32 = d.put(v32);
        const double b64 = graph_bytes + v64.size() * 8.0, b32 = graph_bytes + v32.size() * 4.0;
        if (selected("library BSR f64"))
            row(P, "library BSR f64", b64, 0, ref, repeats, [&](double *y) {
                sfem::d_bsr_spmv_native<count_t, idx_t, double, double>(g.n_rows, 3, rp, ci, d64, 0.0, P.dx, y, nullptr);
            });
        if (selected("library BSR f32"))
            row(P, "library BSR f32", b32, 0, ref, repeats, [&](double *y) {
                sfem::d_bsr_spmv_native<count_t, idx_t, float, double>(g.n_rows, 3, rp, ci, d32, 0.0, P.dx, y, nullptr);
            });
        // The library's grouped product (sfem::d_bsr_spmv_grouped), a group of threads per block row.
        for (const int G : {1, 2, 4, 8, 16, 32}) {
            const std::string n32 = "library BSR f32 group " + std::to_string(G), n64 = "library BSR f64 group " + std::to_string(G);
            if (selected(n32))
                row(P, n32, b32, 0, ref, repeats, [&](double *y) {
                    sfem::d_bsr_spmv_grouped<count_t, idx_t, float, double>(g.n_rows, 3, rp, rp + 1, ci, d32, 0.0, P.dx, y, G, nullptr);
                });
            if (selected(n64))
                row(P, n64, b64, 0, ref, repeats, [&](double *y) {
                    sfem::d_bsr_spmv_grouped<count_t, idx_t, double, double>(g.n_rows, 3, rp, rp + 1, ci, d64, 0.0, P.dx, y, G, nullptr);
                });
        }
        if (selected("cusparse BSR f64")) {
            cusparseHandle_t h;
            cusparseCreate(&h);
            cusparseMatDescr_t descr;
            cusparseCreateMatDescr(&descr);
            const double one = 1, zero = 0;
            row(P, "cusparse BSR f64", b64, 0, ref, repeats, [&](double *y) {
                cusparseDbsrmv(h, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, (int)g.n_rows, (int)g.n_rows,
                               (int)g.nnz(), &one, descr, d64, rp, ci, 3, P.dx, &zero, y);
            });
            cusparseDestroyMatDescr(descr);
            cusparseDestroy(h);
        }
        // cuSPARSE f32: float vectors (no single-precision matrix with double vectors); the error of
        // the float product against the double reference.
        if (selected("cusparse BSR f32")) {
            cusparseHandle_t h;
            cusparseCreate(&h);
            cusparseMatDescr_t descr;
            cusparseCreateMatDescr(&descr);
            std::vector<float> xf(P.x.begin(), P.x.end());
            float *dxf = nullptr, *dyf = nullptr;
            CHECK_CUDA(cudaMalloc(&dxf, nv * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&dyf, nv * sizeof(float)));
            CHECK_CUDA(cudaMemcpy(dxf, xf.data(), nv * sizeof(float), cudaMemcpyHostToDevice));
            const float one = 1, zero = 0;
            auto product = [&] {
                cusparseSbsrmv(h, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, (int)g.n_rows, (int)g.n_rows,
                               (int)g.nnz(), &one, descr, d32, rp, ci, 3, dxf, &zero, dyf);
            };
            product();
            CHECK_CUDA(cudaDeviceSynchronize());
            std::vector<float> yf(nv);
            CHECK_CUDA(cudaMemcpy(yf.data(), dyf, nv * sizeof(float), cudaMemcpyDeviceToHost));
            const double err = rel_l2(std::vector<double>(yf.begin(), yf.end()), ref);
            const double t = best_seconds(repeats, product);
            printf("  %-40s %10.1f %8.2f %8.1f %7.2f %10.2e %5s\n", "cusparse BSR f32 (f32 vectors)", (double)nv / t * 1e-6,
                   b32 / (double)nv, (b32 + 2.0 * nv * sizeof(float)) / t * 1e-9, 0.0, err, "-");
            cudaFree(dxf);
            cudaFree(dyf);
            cusparseDestroyMatDescr(descr);
            cusparseDestroy(h);
        }
        // The same quantized blocks, one row per thread, no SELL.
        if (selected("csr i8 one row per thread")) {
            std::vector<int64_t> rp64(g.rowptr.begin(), g.rowptr.end());
            Dev dq;
            const int64_t *r64 = dq.put(rp64);
            const uint8_t *qc = dq.put(q8.codes);
            const float *qs = dq.put(q8.scales);
            const double bytes = graph_bytes + q8.codes.size() + q8.scales.size() * 4.0;
            row(P, "csr i8 one row per thread", bytes, 0, ref, repeats, [&](double *y) {
                qbsr::apply_csr(Code::I8, g.n_rows, r64, ci, qc, qs, 1.0f, P.dx, y, nullptr);
            });
        }
    }

    const std::vector<Layout> layouts = {
            {Form::Full, Cols::I32, Rec::Scalar, 1, "full i32 scalar s1"},
            {Form::Full, Cols::I32, Rec::Scalar, 1024, "full i32 scalar s1024"},
            {Form::Full, Cols::Off16, Rec::Scalar, 1024, "full off16 scalar s1024"},
            {Form::Full, Cols::I32, Rec::Packed, 1024, "full i32 packed s1024"},
            {Form::Full, Cols::Off16, Rec::Packed, 1024, "full off16 packed s1024"},
            {Form::Upper, Cols::I32, Rec::Scalar, 1024, "upper atomic i32 scalar s1024"},
            {Form::Upper, Cols::Off16, Rec::Scalar, 1024, "upper atomic off16 scalar s1024"},
            {Form::Upper, Cols::Off16, Rec::Packed, 1, "upper atomic off16 packed s1"},
            {Form::Upper, Cols::Off16, Rec::Packed, 1024, "upper atomic off16 packed s1024"},
            {Form::UpperGather, Cols::I32, Rec::Packed, 1024, "upper gather i32 packed s1024"},
            {Form::UpperGather, Cols::Off16, Rec::Packed, 1024, "upper gather off16 packed s1024"},
    };
    // f32: the full form only, to separate the layout from the bytes.
    for (const Code code : {Code::I8, Code::F16, Code::F32}) {
        const char *cn = code == Code::I8 ? "i8" : (code == Code::F16 ? "f16" : "f32");
        auto wanted = [&](const Layout &L) { return code != Code::F32 || L.form == Form::Full; };
        bool any = false;
        for (const auto &L : layouts) any = any || (wanted(L) && selected(std::string(cn) + " " + L.name));
        if (!any) continue;
        const Quantized q = quantize(g, code);
        const std::vector<double> ref = reference(g, q, P.x);
        for (const auto &L : layouts) {
            const std::string name = std::string(cn) + " " + L.name;
            if (!wanted(L) || !selected(name)) continue;
            Built b;
            build(g, q, L, b);
            const double share = b.stored + b.remainder > 0 ? (double)b.remainder / (double)(b.stored + b.remainder) : 0;
            row(P, name, (double)b.dev.bytes, share, ref, repeats, [&](double *y) {
                if (qbsr::apply(code, L.form, L.cols, L.rec, b.s, P.dx, y, nullptr)) {
                    fprintf(stderr, "qbsr_shape: %s failed\n", name.c_str());
                    exit(1);
                }
            });
        }
    }
    cudaFree(P.dx);
    cudaFree(P.dy);
    cudaFree(P.dy2);
    return 0;
}
