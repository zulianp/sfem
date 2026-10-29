// The compressed formats on the GPU (qp_device.hpp): set-up and application kernels for the
// quadrature-point stores and the element matrices of this build's element, and the quantized BSR
// store.  The element's stages, the contraction and the pointwise tangent are the generated host
// code (qp_gen/), compiled for the device with one lane; the stores are the host layouts with
// blocks of 32 elements, one warp, so a warp's loads of one code are one transaction.
//
// Set-up: one thread per element, in double as on the host: gather, reference gradients at the
// rule's points (setup_in), the pointwise tangent, the weighted points or moments, then the
// store's encoding (the host's encode_block, line by line).  Application: one thread per element,
// tau = float (double for the f64 reference), atomicAdd scatter.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <type_traits>
#include <vector>

#include "qbsr_kernels.hpp"
#include "qp_device.hpp"

#define SFEM_INLINE __forceinline__
#define RSTR __restrict__
#include "kernel_math.cuh"

#include "qp_contract.hpp"
#include "qp_element.hpp"
#include "qp_point.hpp"
#include "qp_device_tables.cuh"

#define QPD_CHECK(x)                                                                              \
    do {                                                                                          \
        const cudaError_t e_ = (x);                                                               \
        if (e_ != cudaSuccess) {                                                                  \
            fprintf(stderr, "qp_device: %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            return 1;                                                                             \
        }                                                                                         \
    } while (0)

namespace qpd {
namespace {

using El = qp::Element;
constexpr int N = El::N, R = El::R, NV = El::NV, NQ = El::NQ;
constexpr int NP = N * (N - 1) / 2;  // node pairs of the difference form
constexpr int V = 32;                // elements per store block: one warp
constexpr int THREADS = 128;

template <int B, int E, class F>
__device__ __forceinline__ void static_for(F &&f) {
    if constexpr (B < E) {
        f(std::integral_constant<int, B>{});
        static_for<B + 1, E>(f);
    }
}

// Pair k of the difference form: (p, q), p < q, in the host's order (p major).
__host__ __device__ constexpr int pair_p(int k) {
    int a = 0;
    while (k >= N - 1 - a) k -= N - 1 - a++;
    return a;
}
__host__ __device__ constexpr int pair_q(int k) {
    int a = 0;
    while (k >= N - 1 - a) k -= N - 1 - a++;
    return a + 1 + k;
}

// ----------------------------------------------------------------------------------------- codecs
// The host codecs (CodeTraits over sfem_HistoryStorage.hpp) with the device intrinsics: the same
// bits (binary16 is binary16, the integer codes are the integer codes).

struct F64 {
    using storage = double;
    using tau = double;
    static constexpr bool scaled = false;
    static __device__ double scale_for(double) { return 1; }
    static __device__ storage encode(double v, double) { return v; }
    static __device__ tau decode(storage c) { return c; }
};
struct F16 {
    using storage = uint16_t;
    using tau = float;
    static constexpr bool scaled = true;
    // In double, as the host's ScaledHalfCodec, then rounded to the f32 scale.
    static __device__ double scale_for(double m) {
        if (!(m > 0)) return 1;
        return (double)(float)exp2(ceil(log2(m)) - 14);
    }
    static __device__ storage encode(double v, double inv) { return __half_as_ushort(__double2half(v * inv)); }
    static __device__ tau decode(storage c) { return __half2float(__ushort_as_half(c)); }
};
struct I8 {
    using storage = int8_t;
    using tau = float;
    static constexpr bool scaled = true;
    static __device__ double scale_for(double m) {
        if (!(m > 0)) return 1;
        return (double)(float)(m / 127);
    }
    static __device__ storage encode(double v, double inv) {
        double c = rint(v * inv);
        c = c > 127 ? 127 : (c < -127 ? -127 : c);
        return (storage)c;
    }
    static __device__ tau decode(storage c) { return float(c); }
};

// The pair factor of the element matrices' PairScale form (ea_store.hpp): r = 2^-E (1 + m/8).
__host__ __device__ inline float pair_factor(const uint8_t b) {
    const int32_t bits = ((127 - (int32_t)(b >> 3)) << 23) | ((int32_t)(b & 7) << 20);
#ifdef __CUDA_ARCH__
    return __int_as_float(bits);
#else
    float r;
    std::memcpy(&r, &bits, sizeof(float));
    return r;
#endif
}
__device__ uint8_t pair_factor_for(const double ratio) {
    if (!(ratio > ldexp(1.0, -31))) return uint8_t(31 << 3);
    int E = min(31, (int)ceil(-log2(ratio)));
    int m = (int)ceil((ldexp(ratio, E) - 1) * 8);
    if (m < 0) m = 0;
    while (E > 0 && (double)pair_factor((uint8_t)((E << 3) | min(m, 7))) < ratio) {
        if (++m > 7) {
            m = 0;
            --E;
        }
    }
    return (uint8_t)((E << 3) | min(m, 7));
}

// ------------------------------------------------------------------------------------- set-up

__device__ __forceinline__ void gather(const Geometry &g, const ptrdiff_t e, const double *RSTR x, double h[3][N][1],
                                       int32_t node[N]) {
#pragma unroll
    for (int p = 0; p < N; ++p) {
        node[p] = g.elements[(size_t)p * g.nelements + e];
#pragma unroll
        for (int k = 0; k < 3; ++k) h[k][p][0] = x[(size_t)node[p] * 3 + k];
    }
}

// The weighted point tangents (points) or their moments, bT[j][c], of element e at state x.
__device__ void tangents(const Geometry &g, const ptrdiff_t e, const double *RSTR x, double bT[NV][45]) {
    double h[3][N][1];
    int32_t node[N];
    gather(g, e, x, h, node);
    double G[NQ][9][1];
    El::template setup_in<double, double, 1>(1, h, G);
    double adj[9];
#pragma unroll
    for (int c = 0; c < 9; ++c) adj[c] = (double)g.adj[(size_t)c * g.nelements + e];
    const double det = (double)g.det[e];
    if constexpr (El::moments) {
        for (int j = 0; j < NV; ++j)
            for (int c = 0; c < 45; ++c) bT[j][c] = 0;
    }
    for (int q = 0; q < NQ; ++q) {
        double gq[9], T[45];
#pragma unroll
        for (int c = 0; c < 9; ++c) gq[c] = G[q][c][0];
        qp::qp_point_tangent<double>(gq, adj, det, g.lmbda, g.mu, T);
        const double w = qp::dev::QW[q];
        if constexpr (El::moments) {
            for (int j = 0; j < NV; ++j) {
                const double f = qp::dev::FACTOR[q * NV + j];
                for (int c = 0; c < 45; ++c) bT[j][c] += f * (w * T[c]);
            }
        } else {
            for (int c = 0; c < 45; ++c) bT[q][c] = w * T[c];
        }
    }
}

// Point store: codes [block][j][c][lane], scales [element], means [block][c][lane] (MD).
template <class CODEC, bool MD>
__global__ void __launch_bounds__(THREADS) setup_points(const Geometry g, const double *RSTR x,
                                                        typename CODEC::storage *RSTR codes, float *RSTR scales,
                                                        float *RSTR aux) {
    const ptrdiff_t e = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= g.nelements) return;
    double T[NV][45];
    tangents(g, e, x, T);
    const size_t blk = (size_t)(e / V), lane = (size_t)(e % V);
    if constexpr (MD) {
        for (int c = 0; c < 45; ++c) {
            double acc = 0;
            for (int j = 0; j < NV; ++j) acc += qp::dev::MULT[j] * T[j][c];
            const float mf = (float)(acc / El::KVOL);
            aux[(blk * 45 + c) * V + lane] = mf;
            for (int j = 0; j < NV; ++j) T[j][c] -= qp::dev::X[j] * (double)mf;
        }
    }
    double inv = 1;
    if constexpr (CODEC::scaled) {
        double top = 0;
        for (int j = 0; j < NV; ++j)
            for (int c = 0; c < 45; ++c) top = fmax(top, fabs(T[j][c]));
        const double s = CODEC::scale_for(top);
        scales[e] = (float)s;
        inv = 1.0 / (double)(float)s;
    }
    for (int j = 0; j < NV; ++j)
        for (int c = 0; c < 45; ++c) codes[((blk * NV + j) * 45 + c) * V + lane] = CODEC::encode(T[j][c], inv);
}

// Element matrices, difference form: codes [block][pair * 9 + entry][lane], scales [element], pair
// factors [block][pair][lane] (PAIR).  The host's encode_block_difference and encode_values.
template <class CODEC, bool PAIR>
__global__ void __launch_bounds__(THREADS) setup_ea(const Geometry g, const double *RSTR x,
                                                    typename CODEC::storage *RSTR codes, float *RSTR scales,
                                                    uint8_t *RSTR pair_factors) {
    using w_t = std::conditional_t<(sizeof(typename CODEC::storage) <= 2), float, double>;
    const ptrdiff_t e = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= g.nelements) return;
    double bT[NV][45];
    tangents(g, e, x, bT);
    w_t T[N][27], Kd[NP][9];
    for (int k = 0; k < NP; ++k)
        for (int y = 0; y < 9; ++y) Kd[k][y] = 0;
    for (int j = 0; j < NV; ++j) {
        const int pa = qp::dev::PA[j], pb = qp::dev::PB[j];
        for (int pass = 0; pass < (pa == pb ? 1 : 2); ++pass) {
            const int a = pass == 0 ? pa : pb, b = pass == 0 ? pb : pa;
            for (int q = 0; q < N; ++q) {
                const w_t d0 = w_t(qp::dev::DT[(a * 3 + 0) * N + q]), d1 = w_t(qp::dev::DT[(a * 3 + 1) * N + q]),
                          d2 = w_t(qp::dev::DT[(a * 3 + 2) * N + q]);
                for (int in = 0; in < 9; ++in)
                    for (int k = 0; k < 3; ++k)
                        T[q][in * 3 + k] = d0 * w_t(bT[j][qp::dev::SLOT[in * 9 + k * 3 + 0]]) +
                                           d1 * w_t(bT[j][qp::dev::SLOT[in * 9 + k * 3 + 1]]) +
                                           d2 * w_t(bT[j][qp::dev::SLOT[in * 9 + k * 3 + 2]]);
            }
            for (int k = 0; k < NP; ++k) {
                const int p = qp::dev::PAIR_P[k], q = qp::dev::PAIR_Q[k];
                const w_t e0 = w_t(qp::dev::DT[(b * 3 + 0) * N + p]), e1 = w_t(qp::dev::DT[(b * 3 + 1) * N + p]),
                          e2 = w_t(qp::dev::DT[(b * 3 + 2) * N + p]);
                for (int i = 0; i < 3; ++i)
                    for (int kk = 0; kk < 3; ++kk)
                        Kd[k][i * 3 + kk] += e0 * T[q][(i * 3 + 0) * 3 + kk] + e1 * T[q][(i * 3 + 1) * 3 + kk] +
                                             e2 * T[q][(i * 3 + 2) * 3 + kk];
            }
        }
    }
    const size_t blk = (size_t)(e / V), lane = (size_t)(e % V);
    double top = 0;
    for (int k = 0; k < NP; ++k)
        for (int y = 0; y < 9; ++y) top = fmax(top, fabs((double)Kd[k][y]));
    const double s = CODEC::scale_for(top);
    scales[e] = (float)s;
    const double inv = 1.0 / (double)(float)s;
    for (int k = 0; k < NP; ++k) {
        double pinv = inv;
        if constexpr (PAIR) {
            double pt = 0;
            for (int y = 0; y < 9; ++y) pt = fmax(pt, fabs((double)Kd[k][y]));
            const uint8_t b = pair_factor_for(top > 0 ? pt / top : 0.0);
            pair_factors[(blk * NP + k) * V + lane] = b;
            pinv = inv / (double)pair_factor(b);
        }
        for (int y = 0; y < 9; ++y) codes[(blk * (NP * 9) + k * 9 + y) * V + lane] = CODEC::encode((double)Kd[k][y], pinv);
    }
}

// ------------------------------------------------------- the set-up, point-parallel (HEX8, TET10)
//
// One thread per element and set-up point: the thread's gradient (from the table of gradients at
// the set-up's points) and weighted tangent into shared memory; the moments summed by one thread per
// moment in the host's order; the element's encoding split over its threads.  No per-thread arrays,
// so nothing spills; the codes are the per-thread set-up's (the host's).

template <int EPB_>
struct Coop {
    static constexpr int EPB = EPB_, THREADS = EPB_ * NQ;
};
// Elements per block: as many as fit the shared memory of the store's kind.
constexpr int coop_epb(const size_t per_element) { return per_element * 8 <= 40000 ? 8 : (per_element * 4 <= 40000 ? 4 : 2); }

// Phase 1 (and the moments): bT[j][c] of each of the block's elements, in Ts.
template <int EPB>
__device__ __forceinline__ void coop_tangents(const Geometry &g, const double *RSTR x, double (*Ts)[NQ][45],
                                              double (*us)[3][N], double *sg, const int le, const int q,
                                              const ptrdiff_t e, const bool active) {
    const double *const src = qp::dev::SG_IS_DT ? qp::dev::DT : qp::dev::SG;
    for (int i = threadIdx.x; i < NQ * 3 * N; i += blockDim.x) sg[i] = src[i];
    if (active)
        for (int s = q; s < N; s += NQ) {
            const size_t v = (size_t)g.elements[(size_t)s * g.nelements + e] * 3;
            for (int k = 0; k < 3; ++k) us[le][k][s] = x[v + k];
        }
    __syncthreads();
    if (active) {
        double G[9], adj[9], T[45];
#pragma unroll
        for (int k = 0; k < 3; ++k)
#pragma unroll
            for (int m = 0; m < 3; ++m) {
                double acc = 0;
                for (int j = 0; j < N; ++j) acc += sg[(q * 3 + m) * N + j] * us[le][k][j];
                G[k * 3 + m] = acc;
            }
#pragma unroll
        for (int c = 0; c < 9; ++c) adj[c] = (double)g.adj[(size_t)c * g.nelements + e];
        qp::qp_point_tangent<double>(G, adj, (double)g.det[e], g.lmbda, g.mu, T);
        const double w = qp::dev::QW[q];
        for (int c = 0; c < 45; ++c) Ts[le][q][c] = w * T[c];
    }
    __syncthreads();
    if constexpr (El::moments) {
        double acc[45];
        if (active && q < NV) {
            for (int c = 0; c < 45; ++c) acc[c] = 0;
            for (int p = 0; p < NQ; ++p) {
                const double f = qp::dev::FACTOR[p * NV + q];
                for (int c = 0; c < 45; ++c) acc[c] += f * Ts[le][p][c];
            }
        }
        __syncthreads();
        if (active && q < NV)
            for (int c = 0; c < 45; ++c) Ts[le][q][c] = acc[c];
        __syncthreads();
    }
}

// The element's largest magnitude over its threads' partial maxima.
template <int EPB>
__device__ __forceinline__ double coop_max(double (*tops)[NQ], const int le, const int q, const double mine) {
    tops[le][q] = mine;
    __syncthreads();
    double top = 0;
    for (int t = 0; t < NQ; ++t) top = fmax(top, tops[le][t]);
    return top;
}

template <class CODEC, bool MD, int EPB>
__global__ void __launch_bounds__(EPB *NQ) setup_points_coop(const Geometry g, const double *RSTR x,
                                                              typename CODEC::storage *RSTR codes, float *RSTR scales,
                                                              float *RSTR aux) {
    __shared__ double sg[NQ * 3 * N];
    __shared__ double us[EPB][3][N];
    __shared__ double Ts[EPB][NQ][45];
    __shared__ double tops[EPB][NQ];
    const int le = threadIdx.x / NQ, q = threadIdx.x % NQ;
    const ptrdiff_t e = (ptrdiff_t)blockIdx.x * EPB + le;
    const bool active = e < g.nelements;
    coop_tangents<EPB>(g, x, Ts, us, sg, le, q, e, active);
    const size_t blk = (size_t)(e / V), lane = (size_t)(e % V);
    if constexpr (MD) {
        if (active)
            for (int c = q; c < 45; c += NQ) {
                double acc = 0;
                for (int j = 0; j < NV; ++j) acc += qp::dev::MULT[j] * Ts[le][j][c];
                const float mf = (float)(acc / El::KVOL);
                aux[(blk * 45 + c) * V + lane] = mf;
                for (int j = 0; j < NV; ++j) Ts[le][j][c] -= qp::dev::X[j] * (double)mf;
            }
        __syncthreads();
    }
    double inv = 1;
    if constexpr (CODEC::scaled) {
        double mine = 0;
        for (int i = q; i < NV * 45; i += NQ) mine = fmax(mine, fabs(Ts[le][i / 45][i % 45]));
        const double s = CODEC::scale_for(coop_max<EPB>(tops, le, q, mine));
        if (active && q == 0) scales[e] = (float)s;
        inv = 1.0 / (double)(float)s;
    }
    if (active)
        for (int i = q; i < NV * 45; i += NQ)
            codes[((blk * NV + i / 45) * 45 + i % 45) * V + lane] = CODEC::encode(Ts[le][i / 45][i % 45], inv);
}

template <class CODEC, bool PAIR, int EPB>
__global__ void __launch_bounds__(EPB *NQ) setup_ea_coop(const Geometry g, const double *RSTR x,
                                                          typename CODEC::storage *RSTR codes, float *RSTR scales,
                                                          uint8_t *RSTR pair_factors) {
    using w_t = std::conditional_t<(sizeof(typename CODEC::storage) <= 2), float, double>;
    __shared__ double sg[NQ * 3 * N];
    __shared__ double us[EPB][3][N];
    __shared__ double Ts[EPB][NQ][45];
    __shared__ double tops[EPB][NQ];
    __shared__ w_t Kd[EPB][NP][9];
    __shared__ w_t dt[R * 3 * N];
    const int le = threadIdx.x / NQ, q = threadIdx.x % NQ;
    const ptrdiff_t e = (ptrdiff_t)blockIdx.x * EPB + le;
    const bool active = e < g.nelements;
    for (int i = threadIdx.x; i < R * 3 * N; i += blockDim.x) dt[i] = w_t(qp::dev::DT[i]);
    coop_tangents<EPB>(g, x, Ts, us, sg, le, q, e, active);
    // The host's encode_block_difference, a pair at a time: the same products in the same order.
    double mine = 0;
    if (active)
        for (int k = q; k < NP; k += NQ) {
            const int p = qp::dev::PAIR_P[k], qq = qp::dev::PAIR_Q[k];
            w_t acc[9];
#pragma unroll
            for (int y = 0; y < 9; ++y) acc[y] = 0;
            for (int j = 0; j < NV; ++j) {
                const int pa = qp::dev::PA[j], pb = qp::dev::PB[j];
                for (int pass = 0; pass < (pa == pb ? 1 : 2); ++pass) {
                    const int a = pass == 0 ? pa : pb, b = pass == 0 ? pb : pa;
                    const w_t d0 = dt[(a * 3 + 0) * N + qq], d1 = dt[(a * 3 + 1) * N + qq], d2 = dt[(a * 3 + 2) * N + qq];
                    const w_t e0 = dt[(b * 3 + 0) * N + p], e1 = dt[(b * 3 + 1) * N + p], e2 = dt[(b * 3 + 2) * N + p];
                    w_t T[27];
#pragma unroll
                    for (int in = 0; in < 9; ++in)
#pragma unroll
                        for (int kk = 0; kk < 3; ++kk)
                            T[in * 3 + kk] = d0 * w_t(Ts[le][j][qp::dev::SLOT[in * 9 + kk * 3 + 0]]) +
                                             d1 * w_t(Ts[le][j][qp::dev::SLOT[in * 9 + kk * 3 + 1]]) +
                                             d2 * w_t(Ts[le][j][qp::dev::SLOT[in * 9 + kk * 3 + 2]]);
#pragma unroll
                    for (int i = 0; i < 3; ++i)
#pragma unroll
                        for (int kk = 0; kk < 3; ++kk)
                            acc[i * 3 + kk] += e0 * T[(i * 3 + 0) * 3 + kk] + e1 * T[(i * 3 + 1) * 3 + kk] +
                                               e2 * T[(i * 3 + 2) * 3 + kk];
                }
            }
#pragma unroll
            for (int y = 0; y < 9; ++y) {
                Kd[le][k][y] = acc[y];
                mine = fmax(mine, fabs((double)acc[y]));
            }
        }
    const double top = coop_max<EPB>(tops, le, q, mine);
    const double s = CODEC::scale_for(top);
    if (active && q == 0) scales[e] = (float)s;
    const double inv = 1.0 / (double)(float)s;
    const size_t blk = (size_t)(e / V), lane = (size_t)(e % V);
    if (active)
        for (int k = q; k < NP; k += NQ) {
            double pinv = inv;
            if constexpr (PAIR) {
                double pt = 0;
                for (int y = 0; y < 9; ++y) pt = fmax(pt, fabs((double)Kd[le][k][y]));
                const uint8_t b = pair_factor_for(top > 0 ? pt / top : 0.0);
                pair_factors[(blk * NP + k) * V + lane] = b;
                pinv = inv / (double)pair_factor(b);
            }
            for (int y = 0; y < 9; ++y)
                codes[(blk * (NP * 9) + k * 9 + y) * V + lane] = CODEC::encode((double)Kd[le][k][y], pinv);
        }
}

// Point-parallel when the element has more than one set-up point (HEX8, TET10).
constexpr bool COOP_SETUP = NQ > 1 && N != 27;
constexpr int COOP_EPB_POINTS = coop_epb((size_t)NQ * 45 * 8 + 3 * N * 8);
template <class CODEC>
constexpr int coop_epb_ea() {
    return coop_epb((size_t)NQ * 45 * 8 + 3 * N * 8 + (size_t)NP * 9 * (sizeof(typename CODEC::storage) <= 2 ? 4 : 8));
}

// -------------------------------------------------------------------------------- application

// The application's gather and scatter.  Atomic: each element reads its nodes from the vector and
// adds into the result with atomicAdd.  Block (g.local set): the block of THREADS elements first reads
// its distinct nodes into shared memory, the elements gather from there and accumulate there with
// shared-memory atomics, and the block adds each of its nodes into the result once.
template <bool BLOCK>
struct Traversal {
    const Geometry &g;
    const ptrdiff_t e;
    const bool active;
    double *hs = nullptr, *acc = nullptr;  // block: [local node][component]
    int32_t nb = 0;
    const int32_t *bn = nullptr;
    int32_t node[N];

    __device__ Traversal(const Geometry &g_, const double *RSTR hin, double *smem)
        : g(g_), e((ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x), active(e < g_.nelements) {
        if constexpr (BLOCK) {
            hs = smem;
            acc = smem + 3 * g.max_block_nodes;
            const int32_t b0 = g.block_ptr[blockIdx.x];
            nb = g.block_ptr[blockIdx.x + 1] - b0;
            bn = g.block_nodes + b0;
            for (int l = threadIdx.x; l < nb; l += blockDim.x) {
                const size_t v = (size_t)bn[l] * 3;
                hs[3 * l + 0] = hin[v + 0];
                hs[3 * l + 1] = hin[v + 1];
                hs[3 * l + 2] = hin[v + 2];
                acc[3 * l + 0] = acc[3 * l + 1] = acc[3 * l + 2] = 0;
            }
            __syncthreads();
        }
    }
    __device__ __forceinline__ void gather(const double *RSTR hin, double h[3][N][1]) {
        if (!active) return;
#pragma unroll
        for (int p = 0; p < N; ++p) {
            if constexpr (BLOCK) {
                node[p] = g.local[(size_t)p * g.nelements + e];
#pragma unroll
                for (int k = 0; k < 3; ++k) h[k][p][0] = hs[node[p] * 3 + k];
            } else {
                node[p] = g.elements[(size_t)p * g.nelements + e];
#pragma unroll
                for (int k = 0; k < 3; ++k) h[k][p][0] = hin[(size_t)node[p] * 3 + k];
            }
        }
    }
    __device__ __forceinline__ void scatter(const double o[3][N][1], double *RSTR out) {
        if constexpr (BLOCK) {
            if (active)
#pragma unroll
                for (int p = 0; p < N; ++p)
#pragma unroll
                    for (int k = 0; k < 3; ++k) atomicAdd(&acc[node[p] * 3 + k], o[k][p][0]);
            __syncthreads();
            for (int l = threadIdx.x; l < nb; l += blockDim.x) {
                const size_t v = (size_t)bn[l] * 3;
                atomicAdd(&out[v + 0], acc[3 * l + 0]);
                atomicAdd(&out[v + 1], acc[3 * l + 1]);
                atomicAdd(&out[v + 2], acc[3 * l + 2]);
            }
        } else {
            if (!active) return;
#pragma unroll
            for (int p = 0; p < N; ++p)
#pragma unroll
                for (int k = 0; k < 3; ++k) atomicAdd(&out[(size_t)node[p] * 3 + k], o[k][p][0]);
        }
    }
};

template <class CODEC, bool MD, bool BLOCK, int MINB>
__global__ void __launch_bounds__(THREADS, MINB) apply_points(const Geometry g, const typename CODEC::storage *RSTR codes,
                                                        const float *RSTR scales, const float *RSTR aux,
                                                        const double *RSTR hin, double *RSTR out) {
    using tau = typename CODEC::tau;
    extern __shared__ double smem[];
    Traversal<BLOCK> t(g, hin, smem);
    const ptrdiff_t e = t.e;
    double h[3][N][1];
    t.gather(hin, h);
    double o[3][N][1];
    if (t.active) {
    tau P[R][9][1], Y[R][9][1];
    El::template stage_in<double, tau, 1>(1, h, P);
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < 9; ++c) Y[r][c][0] = 0;
    const size_t blk = (size_t)(e / V), lane = (size_t)(e % V);
    tau A[MD ? 45 : 1], sc = 1;
    if constexpr (MD) {
#pragma unroll
        for (int c = 0; c < 45; ++c) A[c] = tau(__ldg(&aux[(blk * 45 + c) * V + lane]));
        sc = tau(__ldg(&scales[e]));
    }
    static_for<0, NV>([&](auto J) {
        constexpr int j = decltype(J)::value;
        constexpr int a = El::PA[j], b = El::PB[j];
        constexpr double xj = El::X[j];
        const typename CODEC::storage *const cj = codes + (blk * NV + j) * 45 * V + lane;
        tau S[45];
#pragma unroll
        for (int c = 0; c < 45; ++c) {
            const tau v = CODEC::decode(__ldg(&cj[c * V]));
            if constexpr (MD)
                S[c] = tau(xj) * A[c] + sc * v;
            else
                S[c] = v;
        }
        qp::contract(S, &P[a][0][0], &Y[b][0][0]);
        if constexpr (a != b) qp::contract(S, &P[b][0][0], &Y[a][0][0]);
    });
    El::template stage_out<tau, double, 1>(1, Y, o);
    if constexpr (CODEC::scaled && !MD) {
        const double s = (double)__ldg(&scales[e]);
#pragma unroll
        for (int k = 0; k < 3; ++k)
#pragma unroll
            for (int p = 0; p < N; ++p) o[k][p][0] *= s;
    }
    }
    t.scatter(o, out);
}

template <class CODEC, bool PAIR, bool BLOCK, int MINB>
__global__ void __launch_bounds__(THREADS, MINB) apply_ea(const Geometry g, const typename CODEC::storage *RSTR codes,
                                                    const float *RSTR scales, const uint8_t *RSTR pair_factors,
                                                    const double *RSTR hin, double *RSTR out) {
    using tau = typename CODEC::tau;
    extern __shared__ double smem[];
    Traversal<BLOCK> t(g, hin, smem);
    const ptrdiff_t e = t.e;
    double h[3][N][1];
    t.gather(hin, h);
    double o[3][N][1];
    if (t.active) {
    tau Y[3][N];
#pragma unroll
    for (int k = 0; k < 3; ++k)
#pragma unroll
        for (int p = 0; p < N; ++p) Y[k][p] = 0;
    const size_t blk = (size_t)(e / V), lane = (size_t)(e % V);
    static_for<0, NP>([&](auto KK) {
        constexpr int k = decltype(KK)::value;
        constexpr int p = pair_p(k), q = pair_q(k);
        // Differences in the vector's precision, so a translation cancels exactly.
        tau d0 = tau(h[0][q][0] - h[0][p][0]), d1 = tau(h[1][q][0] - h[1][p][0]), d2 = tau(h[2][q][0] - h[2][p][0]);
        if constexpr (PAIR) {
            const tau r = tau(pair_factor(__ldg(&pair_factors[(blk * NP + k) * V + lane])));
            d0 *= r;
            d1 *= r;
            d2 *= r;
        }
        const typename CODEC::storage *const ck = codes + (blk * (NP * 9) + k * 9) * V + lane;
        tau K[9];
#pragma unroll
        for (int y = 0; y < 9; ++y) K[y] = CODEC::decode(__ldg(&ck[y * V]));
        Y[0][p] += K[0] * d0 + K[1] * d1 + K[2] * d2;
        Y[1][p] += K[3] * d0 + K[4] * d1 + K[5] * d2;
        Y[2][p] += K[6] * d0 + K[7] * d1 + K[8] * d2;
        Y[0][q] -= K[0] * d0 + K[3] * d1 + K[6] * d2;
        Y[1][q] -= K[1] * d0 + K[4] * d1 + K[7] * d2;
        Y[2][q] -= K[2] * d0 + K[5] * d1 + K[8] * d2;
    });
    const double s = (double)__ldg(&scales[e]);
#pragma unroll
    for (int k = 0; k < 3; ++k)
#pragma unroll
        for (int p = 0; p < N; ++p) o[k][p][0] = (double)Y[k][p] * s;
    }
    t.scatter(o, out);
}

// ------------------------------------------------------------------------------- host classes

unsigned grid(const ptrdiff_t n) { return (unsigned)((n + THREADS - 1) / THREADS); }

// The application's shape: the scatter (block-local when the geometry has its map) and the
// resident blocks per SM the kernel is compiled for (1: as many registers as it likes; 2 and 4 cap
// them at 255 and 128).  QPD_MINB overrides the store's default.
int min_blocks(const int fallback) {
    const char *v = std::getenv("QPD_MINB");
    const int m = v ? std::atoi(v) : fallback;
    return m == 2 || m == 4 ? m : 1;
}
template <template <bool, int> class K, class... A>
void launch_shape(const bool block, const int minb, const unsigned blocks, const size_t shared, cudaStream_t st, A... args) {
    if (block) {
        if (minb == 4) K<true, 4>::run(blocks, shared, st, args...);
        else if (minb == 2) K<true, 2>::run(blocks, shared, st, args...);
        else K<true, 1>::run(blocks, shared, st, args...);
    } else {
        if (minb == 4) K<false, 4>::run(blocks, 0, st, args...);
        else if (minb == 2) K<false, 2>::run(blocks, 0, st, args...);
        else K<false, 1>::run(blocks, 0, st, args...);
    }
}
// The block-local scatter's shared memory: the block's nodes' values and sums.
size_t block_shared(const Geometry &g) { return (size_t)6 * g.max_block_nodes * sizeof(double); }

template <typename T>
T *device_alloc(const size_t n) {
    void *p = nullptr;
    if (n && cudaMalloc(&p, n * sizeof(T)) != cudaSuccess) return nullptr;
    if (n) cudaMemset(p, 0, n * sizeof(T));
    return (T *)p;
}

template <class CODEC, bool MD>
class PointStore final : public Kernels {
public:
    using storage = typename CODEC::storage;
    template <bool BLOCK, int MINB>
    struct Launch {
        template <class... A>
        static void run(const unsigned blocks, const size_t shared, cudaStream_t st, A... args) {
            apply_points<CODEC, MD, BLOCK, MINB><<<blocks, THREADS, shared, st>>>(args...);
        }
    };
    PointStore(const Geometry &g, void *stream, const int minb) : g_(g), st_((cudaStream_t)stream), minb_(min_blocks(minb)) {
        const size_t padded = (size_t)((g.nelements + V - 1) / V) * V;
        codes_ = device_alloc<storage>(padded * NV * 45);
        if (CODEC::scaled) scales_ = device_alloc<float>(padded);
        if (MD) aux_ = device_alloc<float>(padded * 45);
    }
    ~PointStore() override {
        cudaFree(codes_);
        cudaFree(scales_);
        cudaFree(aux_);
    }
    int update(const double *x) override {
        if constexpr (COOP_SETUP) {
            constexpr int EPB = COOP_EPB_POINTS;
            setup_points_coop<CODEC, MD, EPB><<<(unsigned)((g_.nelements + EPB - 1) / EPB), EPB * NQ, 0, st_>>>(
                    g_, x, codes_, scales_, aux_);
        } else {
            setup_points<CODEC, MD><<<grid(g_.nelements), THREADS, 0, st_>>>(g_, x, codes_, scales_, aux_);
        }
        QPD_CHECK(cudaGetLastError());
        QPD_CHECK(cudaStreamSynchronize(st_));
        return 0;
    }
    int apply(const double *h, double *out) override {
        launch_shape<Launch>(g_.local != nullptr, minb_, grid(g_.nelements), block_shared(g_), st_, g_,
                             (const storage *)codes_, (const float *)scales_, (const float *)aux_, h, out);
        QPD_CHECK(cudaGetLastError());
        QPD_CHECK(cudaStreamSynchronize(st_));
        return 0;
    }
    // As the host store counts it (bytes_per_element, no padding).
    size_t bytes() const override {
        return (size_t)g_.nelements *
               ((size_t)NV * 45 * sizeof(storage) + (CODEC::scaled ? sizeof(float) : 0) + (MD ? 45 * sizeof(float) : 0));
    }
    int download(std::vector<uint8_t> &codes, std::vector<float> &scales, std::vector<uint8_t> &extra) const override {
        const size_t padded = (size_t)((g_.nelements + V - 1) / V) * V;
        codes.resize(padded * NV * 45 * sizeof(storage));
        QPD_CHECK(cudaMemcpy(codes.data(), codes_, codes.size(), cudaMemcpyDeviceToHost));
        scales.assign(CODEC::scaled ? padded : 0, 0.0f);
        if (CODEC::scaled) QPD_CHECK(cudaMemcpy(scales.data(), scales_, padded * sizeof(float), cudaMemcpyDeviceToHost));
        extra.resize(MD ? padded * 45 * sizeof(float) : 0);
        if (MD) QPD_CHECK(cudaMemcpy(extra.data(), aux_, extra.size(), cudaMemcpyDeviceToHost));
        return 0;
    }

private:
    Geometry g_;
    cudaStream_t st_;
    int minb_;
    storage *codes_ = nullptr;
    float *scales_ = nullptr, *aux_ = nullptr;
};

template <class CODEC, bool PAIR>
class EaStore final : public Kernels {
public:
    using storage = typename CODEC::storage;
    template <bool BLOCK, int MINB>
    struct Launch {
        template <class... A>
        static void run(const unsigned blocks, const size_t shared, cudaStream_t st, A... args) {
            apply_ea<CODEC, PAIR, BLOCK, MINB><<<blocks, THREADS, shared, st>>>(args...);
        }
    };
    EaStore(const Geometry &g, void *stream, const int minb) : g_(g), st_((cudaStream_t)stream), minb_(min_blocks(minb)) {
        const size_t padded = (size_t)((g.nelements + V - 1) / V) * V;
        codes_ = device_alloc<storage>(padded * NP * 9);
        scales_ = device_alloc<float>(padded);
        if (PAIR) pair_factors_ = device_alloc<uint8_t>(padded * NP);
    }
    ~EaStore() override {
        cudaFree(codes_);
        cudaFree(scales_);
        cudaFree(pair_factors_);
    }
    int update(const double *x) override {
        if constexpr (COOP_SETUP) {
            constexpr int EPB = coop_epb_ea<CODEC>();
            setup_ea_coop<CODEC, PAIR, EPB><<<(unsigned)((g_.nelements + EPB - 1) / EPB), EPB * NQ, 0, st_>>>(
                    g_, x, codes_, scales_, pair_factors_);
        } else {
            setup_ea<CODEC, PAIR><<<grid(g_.nelements), THREADS, 0, st_>>>(g_, x, codes_, scales_, pair_factors_);
        }
        QPD_CHECK(cudaGetLastError());
        QPD_CHECK(cudaStreamSynchronize(st_));
        return 0;
    }
    int apply(const double *h, double *out) override {
        launch_shape<Launch>(g_.local != nullptr, minb_, grid(g_.nelements), block_shared(g_), st_, g_,
                             (const storage *)codes_, (const float *)scales_, (const uint8_t *)pair_factors_, h, out);
        QPD_CHECK(cudaGetLastError());
        QPD_CHECK(cudaStreamSynchronize(st_));
        return 0;
    }
    size_t bytes() const override {
        return (size_t)g_.nelements * ((size_t)NP * 9 * sizeof(storage) + sizeof(float) + (PAIR ? NP : 0));
    }
    int download(std::vector<uint8_t> &codes, std::vector<float> &scales, std::vector<uint8_t> &extra) const override {
        const size_t padded = (size_t)((g_.nelements + V - 1) / V) * V;
        codes.resize(padded * NP * 9 * sizeof(storage));
        QPD_CHECK(cudaMemcpy(codes.data(), codes_, codes.size(), cudaMemcpyDeviceToHost));
        scales.resize(padded);
        QPD_CHECK(cudaMemcpy(scales.data(), scales_, padded * sizeof(float), cudaMemcpyDeviceToHost));
        extra.resize(PAIR ? padded * NP : 0);
        if (PAIR) QPD_CHECK(cudaMemcpy(extra.data(), pair_factors_, extra.size(), cudaMemcpyDeviceToHost));
        return 0;
    }

private:
    Geometry g_;
    cudaStream_t st_;
    int minb_;
    storage *codes_ = nullptr;
    float *scales_ = nullptr;
    uint8_t *pair_factors_ = nullptr;
};

// ------------------------------------------------------------- PROTEUS_HEX27: point-parallel
//
// 64 points and 351 node pairs per element are too much work for one thread.  The point stores run
// one thread per point (two elements per block of 128): the element's nodes in shared memory, each
// thread its point's gradient, decoding and contraction, and the element's outputs summed over the
// points through shared memory.  Their codes are laid out [element][component][point], so a warp's
// loads are consecutive (download() returns the host layout).  The element matrices run one block of
// 128 threads per element: the set-up's weighted point tangents in shared memory, then the threads
// over the node pairs; the application one warp per element, a lane per node.  Codes
// [element][entry].  The reference derivatives come from the one-dimensional tables (B1, D1), the
// products in the generator's order, into a shared table per block.

constexpr int TPE = 64;  // threads per element: the points
constexpr int EPB = 2;   // elements per block of the point stores
constexpr int Q1 = qp::dev::Q1, S1 = qp::dev::S1;

// dphi_s/dxi_m at point j, as the generator multiplies it.
__device__ __forceinline__ double dt27(const int j, const int m, const int s) {
    const int jx = j % Q1, jy = (j / Q1) % Q1, jz = j / (Q1 * Q1);
    const int sx = s % S1, sy = (s / S1) % S1, sz = s / (S1 * S1);
    const double *B = qp::dev::B1, *D = qp::dev::D1;
    if (m == 0) return D[jx * S1 + sx] * B[jy * S1 + sy] * B[jz * S1 + sz];
    if (m == 1) return B[jx * S1 + sx] * D[jy * S1 + sy] * B[jz * S1 + sz];
    return B[jx * S1 + sx] * B[jy * S1 + sy] * D[jz * S1 + sz];
}

// The table dphi_s/dxi_m(j), [j][m][s], computed once per process (fill_dt27) in double and float.
__device__ double dt27_d[TPE * 3 * N];
__device__ float dt27_f[TPE * 3 * N];
__global__ void fill_dt27_kernel() {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < TPE * 3 * N; i += blockDim.x * gridDim.x) {
        dt27_d[i] = dt27(i / (3 * N), (i / N) % 3, i % N);
        dt27_f[i] = (float)dt27_d[i];
    }
}
int fill_dt27() {
    static bool done = false;
    if (done) return 0;
    fill_dt27_kernel<<<32, 256>>>();
    QPD_CHECK(cudaGetLastError());
    QPD_CHECK(cudaDeviceSynchronize());
    done = true;
    return 0;
}
template <typename T>
__device__ __forceinline__ const T *dt27_table() {
    if constexpr (std::is_same<T, float>::value)
        return dt27_f;
    else
        return dt27_d;
}
// Into shared memory by the block's threads.
template <typename T>
__device__ __forceinline__ void load_dt27(T *dt) {
    const T *const src = dt27_table<T>();
    for (int i = threadIdx.x; i < TPE * 3 * N; i += blockDim.x) dt[i] = src[i];
    __syncthreads();
}

// G[k*3+m] = sum_s dphi_s/dxi_m(j) u(k, s), u read through the accessor.
template <typename T, class U>
__device__ __forceinline__ void gradient27(const int j, U &&u, T G[9]) {
#pragma unroll
    for (int c = 0; c < 9; ++c) G[c] = 0;
    for (int s = 0; s < N; ++s) {
        const T w0 = T(dt27_d[(j * 3 + 0) * N + s]), w1 = T(dt27_d[(j * 3 + 1) * N + s]), w2 = T(dt27_d[(j * 3 + 2) * N + s]);
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            const T v = u(k, s);
            G[k * 3 + 0] += w0 * v;
            G[k * 3 + 1] += w1 * v;
            G[k * 3 + 2] += w2 * v;
        }
    }
}

__device__ __forceinline__ double warp_sum(double v) {
    for (int o = 16; o > 0; o /= 2) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ double warp_max(double v) {
    for (int o = 16; o > 0; o /= 2) v = fmax(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

template <class CODEC, bool MD>
__global__ void __launch_bounds__(TPE *EPB) setup_points27(const Geometry g, const double *RSTR x,
                                                           typename CODEC::storage *RSTR codes, float *RSTR scales,
                                                           float *RSTR aux) {
    __shared__ double us[EPB][3][N];
    __shared__ double red[EPB][2][45];
    __shared__ double tops[EPB][2];
    const int le = threadIdx.x / TPE, j = threadIdx.x % TPE, warp = j / 32;
    const ptrdiff_t e = (ptrdiff_t)blockIdx.x * EPB + le;
    const bool active = e < g.nelements;
    if (active && j < N) {
        const size_t v = (size_t)g.elements[(size_t)j * g.nelements + e] * 3;
        for (int k = 0; k < 3; ++k) us[le][k][j] = x[v + k];
    }
    __syncthreads();
    double v[45];
#pragma unroll
    for (int c = 0; c < 45; ++c) v[c] = 0;
    if (active) {
        double G[9], adj[9], T[45];
        gradient27<double>(j, [&](int k, int s) { return us[le][k][s]; }, G);
#pragma unroll
        for (int c = 0; c < 9; ++c) adj[c] = (double)g.adj[(size_t)c * g.nelements + e];
        qp::qp_point_tangent<double>(G, adj, (double)g.det[e], g.lmbda, g.mu, T);
        const double w = qp::dev::QW[j];
#pragma unroll
        for (int c = 0; c < 45; ++c) v[c] = w * T[c];
    }
    if constexpr (MD) {
        for (int c = 0; c < 45; ++c) {
            const double s = warp_sum(qp::dev::MULT[j] * v[c]);
            if ((j & 31) == 0) red[le][warp][c] = s;
        }
        __syncthreads();
        for (int c = 0; c < 45; ++c) {
            const float mf = (float)((red[le][0][c] + red[le][1][c]) / El::KVOL);
            if (active && j == c) aux[(size_t)e * 45 + c] = mf;
            v[c] -= qp::dev::X[j] * (double)mf;
        }
    }
    double inv = 1;
    if constexpr (CODEC::scaled) {
        double top = 0;
        for (int c = 0; c < 45; ++c) top = fmax(top, fabs(v[c]));
        top = warp_max(top);
        if ((j & 31) == 0) tops[le][warp] = top;
        __syncthreads();
        const double s = CODEC::scale_for(fmax(tops[le][0], tops[le][1]));
        if (active && j == 0) scales[e] = (float)s;
        inv = 1.0 / (double)(float)s;
    }
    if (active)
        for (int c = 0; c < 45; ++c) codes[((size_t)e * 45 + c) * TPE + j] = CODEC::encode(v[c], inv);
}

// The point's gradient from the one-dimensional tables (the thread's own 18 numbers), the output by
// the transposed stages in three one-dimensional sweeps over the element's threads (the generated
// tensor stage_out's sums, cooperatively): a[qx][qy][sz], then b[qx][sy][sz], then the nodes.
template <class CODEC, bool MD>
__global__ void __launch_bounds__(TPE *EPB) apply_points27(const Geometry g, const typename CODEC::storage *RSTR codes,
                                                           const float *RSTR scales, const float *RSTR aux,
                                                           const double *RSTR hin, double *RSTR out) {
    using tau = typename CODEC::tau;
    __shared__ double hs[EPB][3][N];
    __shared__ int32_t nodes[EPB][N];
    __shared__ tau Ys[EPB][TPE][9];
    __shared__ tau ax[EPB][3][Q1 * Q1 * S1], ay[EPB][3][Q1 * Q1 * S1], az[EPB][3][Q1 * Q1 * S1];
    __shared__ tau bx[EPB][3][Q1 * S1 * S1], cc[EPB][3][Q1 * S1 * S1];
    __shared__ tau B[Q1 * S1], D[Q1 * S1];
    const int le = threadIdx.x / TPE, j = threadIdx.x % TPE;
    const ptrdiff_t e = (ptrdiff_t)blockIdx.x * EPB + le;
    const bool active = e < g.nelements;
    if (threadIdx.x < Q1 * S1) {
        B[threadIdx.x] = tau(qp::dev::B1[threadIdx.x]);
        D[threadIdx.x] = tau(qp::dev::D1[threadIdx.x]);
    }
    if (active && j < N) {
        const int32_t node = g.elements[(size_t)j * g.nelements + e];
        nodes[le][j] = node;
        for (int k = 0; k < 3; ++k) hs[le][k][j] = hin[(size_t)node * 3 + k];
    }
    __syncthreads();
    if (active) {
        const int jx = j % Q1, jy = (j / Q1) % Q1, jz = j / (Q1 * Q1);
        tau bxv[S1], dxv[S1], byv[S1], dyv[S1], bzv[S1], dzv[S1];
#pragma unroll
        for (int t = 0; t < S1; ++t) {
            bxv[t] = B[jx * S1 + t], dxv[t] = D[jx * S1 + t];
            byv[t] = B[jy * S1 + t], dyv[t] = D[jy * S1 + t];
            bzv[t] = B[jz * S1 + t], dzv[t] = D[jz * S1 + t];
        }
        // The increment relative to the element's first node, as the host's stages take it.
        tau P[9], Y[9];
#pragma unroll
        for (int c = 0; c < 9; ++c) P[c] = 0;
#pragma unroll
        for (int sz = 0; sz < S1; ++sz)
#pragma unroll
            for (int sy = 0; sy < S1; ++sy)
#pragma unroll
                for (int sx = 0; sx < S1; ++sx) {
                    const int sn = sx + S1 * (sy + S1 * sz);
                    const tau w0 = dxv[sx] * byv[sy] * bzv[sz], w1 = bxv[sx] * dyv[sy] * bzv[sz], w2 = bxv[sx] * byv[sy] * dzv[sz];
#pragma unroll
                    for (int k = 0; k < 3; ++k) {
                        const tau v = tau(hs[le][k][sn] - hs[le][k][0]);
                        P[k * 3 + 0] += w0 * v;
                        P[k * 3 + 1] += w1 * v;
                        P[k * 3 + 2] += w2 * v;
                    }
                }
        tau S[45], sc = 1;
        if constexpr (MD) sc = tau(__ldg(&scales[e]));
        const tau xj = tau(qp::dev::X[j]);
#pragma unroll
        for (int c = 0; c < 45; ++c) {
            const tau v = CODEC::decode(__ldg(&codes[((size_t)e * 45 + c) * TPE + j]));
            if constexpr (MD)
                S[c] = xj * tau(__ldg(&aux[(size_t)e * 45 + c])) + sc * v;
            else
                S[c] = v;
        }
#pragma unroll
        for (int c = 0; c < 9; ++c) Y[c] = 0;
        qp::contract(S, P, Y);
#pragma unroll
        for (int c = 0; c < 9; ++c) Ys[le][j][c] = Y[c];
    }
    __syncthreads();
    // z: a[i][qx][qy][sz] (x and y parts with B, z part with D).
    if (active)
        for (int idx = j; idx < 3 * Q1 * Q1 * S1; idx += TPE) {
            const int i = idx / (Q1 * Q1 * S1), r = idx % (Q1 * Q1 * S1);
            const int qx = r % Q1, qy = (r / Q1) % Q1, sz = r / (Q1 * Q1);
            tau sx_ = 0, sy_ = 0, sz_ = 0;
#pragma unroll
            for (int qz = 0; qz < Q1; ++qz) {
                const int q = qx + Q1 * (qy + Q1 * qz);
                sx_ += B[qz * S1 + sz] * Ys[le][q][i * 3 + 0];
                sy_ += B[qz * S1 + sz] * Ys[le][q][i * 3 + 1];
                sz_ += D[qz * S1 + sz] * Ys[le][q][i * 3 + 2];
            }
            ax[le][i][r] = sx_, ay[le][i][r] = sy_, az[le][i][r] = sz_;
        }
    __syncthreads();
    // y: b[i][qx][sy][sz] = sum_qy B a_x, c[i][qx][sy][sz] = sum_qy D a_y + B a_z.
    if (active)
        for (int idx = j; idx < 3 * Q1 * S1 * S1; idx += TPE) {
            const int i = idx / (Q1 * S1 * S1), r = idx % (Q1 * S1 * S1);
            const int qx = r % Q1, sy = (r / Q1) % S1, sz = r / (Q1 * S1);
            tau b_ = 0, c_ = 0;
#pragma unroll
            for (int qy = 0; qy < Q1; ++qy) {
                const int a = qx + Q1 * (qy + Q1 * sz);
                b_ += B[qy * S1 + sy] * ax[le][i][a];
                c_ += D[qy * S1 + sy] * ay[le][i][a] + B[qy * S1 + sy] * az[le][i][a];
            }
            bx[le][i][r] = b_, cc[le][i][r] = c_;
        }
    __syncthreads();
    if (!active) return;
    // x: out[i][s] = sum_qx D b + B c.
    const double s = CODEC::scaled && !MD ? (double)__ldg(&scales[e]) : 1.0;
    for (int idx = j; idx < 3 * N; idx += TPE) {
        const int i = idx / N, node = idx % N;
        const int sx = node % S1, sy = (node / S1) % S1, sz = node / (S1 * S1);
        tau o = 0;
#pragma unroll
        for (int qx = 0; qx < Q1; ++qx) {
            const int r = qx + Q1 * (sy + S1 * sz);
            o += D[qx * S1 + sx] * bx[le][i][r] + B[qx * S1 + sx] * cc[le][i][r];
        }
        atomicAdd(&out[(size_t)nodes[le][node] * 3 + i], (double)o * s);
    }
}

constexpr int EA_THREADS = 128;

template <class CODEC, bool PAIR>
__global__ void __launch_bounds__(EA_THREADS) setup_ea27(const Geometry g, const double *RSTR x,
                                                         typename CODEC::storage *RSTR codes, float *RSTR scales,
                                                         uint8_t *RSTR pair_factors) {
    using w_t = std::conditional_t<(sizeof(typename CODEC::storage) <= 2), float, double>;
    // The point tangents in the arithmetic type (the host rounds them to it as it reads them).
    __shared__ double us[3][N];
    __shared__ w_t S[TPE][45];
    __shared__ w_t Kd[NP][9];
    __shared__ w_t dt[TPE * 3 * N];
    __shared__ double tops[EA_THREADS / 32];
    load_dt27(dt);
    const ptrdiff_t e = blockIdx.x;
    const int t = threadIdx.x;
    if (t < N) {
        const size_t v = (size_t)g.elements[(size_t)t * g.nelements + e] * 3;
        for (int k = 0; k < 3; ++k) us[k][t] = x[v + k];
    }
    __syncthreads();
    if (t < TPE) {
        double G[9], adj[9], T[45];
        gradient27<double>(t, [&](int k, int s) { return us[k][s]; }, G);
#pragma unroll
        for (int c = 0; c < 9; ++c) adj[c] = (double)g.adj[(size_t)c * g.nelements + e];
        qp::qp_point_tangent<double>(G, adj, (double)g.det[e], g.lmbda, g.mu, T);
        const double w = qp::dev::QW[t];
        for (int c = 0; c < 45; ++c) S[t][c] = w_t(w * T[c]);
    }
    __syncthreads();
    // K_pq[i][kk] = sum_j sum_n DT[j][n][p] sum_m S_j[(i,n),(kk,m)] DT[j][m][q], as the host's
    // encode_block_difference forms it.
    double top = 0;
    for (int k = t; k < NP; k += EA_THREADS) {
        const int p = qp::dev::PAIR_P[k], q = qp::dev::PAIR_Q[k];
        w_t acc[9];
#pragma unroll
        for (int y = 0; y < 9; ++y) acc[y] = 0;
        for (int j = 0; j < TPE; ++j) {
            const w_t d0 = dt[(j * 3 + 0) * N + q], d1 = dt[(j * 3 + 1) * N + q], d2 = dt[(j * 3 + 2) * N + q];
            const w_t e0 = dt[(j * 3 + 0) * N + p], e1 = dt[(j * 3 + 1) * N + p], e2 = dt[(j * 3 + 2) * N + p];
            w_t T[27];
#pragma unroll
            for (int in = 0; in < 9; ++in)
#pragma unroll
                for (int kk = 0; kk < 3; ++kk)
                    T[in * 3 + kk] = d0 * S[j][qp::dev::SLOT[in * 9 + kk * 3 + 0]] +
                                     d1 * S[j][qp::dev::SLOT[in * 9 + kk * 3 + 1]] +
                                     d2 * S[j][qp::dev::SLOT[in * 9 + kk * 3 + 2]];
#pragma unroll
            for (int i = 0; i < 3; ++i)
#pragma unroll
                for (int kk = 0; kk < 3; ++kk)
                    acc[i * 3 + kk] += e0 * T[(i * 3 + 0) * 3 + kk] + e1 * T[(i * 3 + 1) * 3 + kk] + e2 * T[(i * 3 + 2) * 3 + kk];
        }
#pragma unroll
        for (int y = 0; y < 9; ++y) {
            Kd[k][y] = acc[y];
            top = fmax(top, fabs((double)acc[y]));
        }
    }
    top = warp_max(top);
    if ((t & 31) == 0) tops[t / 32] = top;
    __syncthreads();
    for (int w = 0; w < EA_THREADS / 32; ++w) top = fmax(top, tops[w]);
    const double s = CODEC::scale_for(top);
    if (t == 0) scales[e] = (float)s;
    const double inv = 1.0 / (double)(float)s;
    for (int k = t; k < NP; k += EA_THREADS) {
        double pinv = inv;
        if constexpr (PAIR) {
            double pt = 0;
            for (int y = 0; y < 9; ++y) pt = fmax(pt, fabs((double)Kd[k][y]));
            const uint8_t b = pair_factor_for(top > 0 ? pt / top : 0.0);
            pair_factors[(size_t)e * NP + k] = b;
            pinv = inv / (double)pair_factor(b);
        }
        for (int y = 0; y < 9; ++y) codes[(size_t)e * (NP * 9) + k * 9 + y] = CODEC::encode((double)Kd[k][y], pinv);
    }
}

// One warp per element, lane p < 27 owning node p: y_p = sum_{q != p} M (h_q - h_p), M = K_pq for q > p
// and K_qp^T for q < p (y_q -= K_pq^T d of the stored pair).  Each pair's block is read by both its
// lanes, the second time from L1; no atomics within the element.
constexpr int EA_WARPS = 4;
__device__ __forceinline__ int pair_index(const int p, const int q) {  // p < q
    return p * (N - 1) - p * (p - 1) / 2 + (q - p - 1);
}
template <class CODEC, bool PAIR>
__global__ void __launch_bounds__(EA_WARPS * 32) apply_ea27(const Geometry g, const typename CODEC::storage *RSTR codes,
                                                            const float *RSTR scales, const uint8_t *RSTR pair_factors,
                                                            const double *RSTR hin, double *RSTR out) {
    using tau = typename CODEC::tau;
    __shared__ double hs[EA_WARPS][3][N];
    const int w = threadIdx.x / 32, p = threadIdx.x % 32;
    const ptrdiff_t e = (ptrdiff_t)blockIdx.x * EA_WARPS + w;
    if (e >= g.nelements) return;
    int32_t node = 0;
    if (p < N) {
        node = g.elements[(size_t)p * g.nelements + e];
        for (int k = 0; k < 3; ++k) hs[w][k][p] = hin[(size_t)node * 3 + k];
    }
    __syncwarp();
    if (p >= N) return;
    const typename CODEC::storage *const ce = codes + (size_t)e * (NP * 9);
    tau y0 = 0, y1 = 0, y2 = 0;
    for (int q = 0; q < N; ++q) {
        if (q == p) continue;
        const int k = p < q ? pair_index(p, q) : pair_index(q, p);
        tau d0 = tau(hs[w][0][q] - hs[w][0][p]), d1 = tau(hs[w][1][q] - hs[w][1][p]), d2 = tau(hs[w][2][q] - hs[w][2][p]);
        if constexpr (PAIR) {
            const tau r = tau(pair_factor(__ldg(&pair_factors[(size_t)e * NP + k])));
            d0 *= r;
            d1 *= r;
            d2 *= r;
        }
        tau K[9];
#pragma unroll
        for (int y = 0; y < 9; ++y) K[y] = CODEC::decode(__ldg(&ce[k * 9 + y]));
        if (p < q) {
            y0 += K[0] * d0 + K[1] * d1 + K[2] * d2;
            y1 += K[3] * d0 + K[4] * d1 + K[5] * d2;
            y2 += K[6] * d0 + K[7] * d1 + K[8] * d2;
        } else {
            y0 += K[0] * d0 + K[3] * d1 + K[6] * d2;
            y1 += K[1] * d0 + K[4] * d1 + K[7] * d2;
            y2 += K[2] * d0 + K[5] * d1 + K[8] * d2;
        }
    }
    const double s = (double)__ldg(&scales[e]);
    atomicAdd(&out[(size_t)node * 3 + 0], (double)y0 * s);
    atomicAdd(&out[(size_t)node * 3 + 1], (double)y1 * s);
    atomicAdd(&out[(size_t)node * 3 + 2], (double)y2 * s);
}

template <class CODEC, bool MD>
class PointStore27 final : public Kernels {
public:
    using storage = typename CODEC::storage;
    PointStore27(const Geometry &g, void *stream) : g_(g), st_((cudaStream_t)stream) {
        fill_dt27();
        codes_ = device_alloc<storage>((size_t)g.nelements * 45 * TPE);
        if (CODEC::scaled) scales_ = device_alloc<float>((size_t)g.nelements);
        if (MD) aux_ = device_alloc<float>((size_t)g.nelements * 45);
    }
    ~PointStore27() override {
        cudaFree(codes_);
        cudaFree(scales_);
        cudaFree(aux_);
    }
    int update(const double *x) override {
        setup_points27<CODEC, MD><<<(unsigned)((g_.nelements + EPB - 1) / EPB), TPE * EPB, 0, st_>>>(g_, x, codes_, scales_, aux_);
        QPD_CHECK(cudaGetLastError());
        QPD_CHECK(cudaStreamSynchronize(st_));
        return 0;
    }
    int apply(const double *h, double *out) override {
        apply_points27<CODEC, MD><<<(unsigned)((g_.nelements + EPB - 1) / EPB), TPE * EPB, 0, st_>>>(g_, codes_, scales_, aux_, h, out);
        QPD_CHECK(cudaGetLastError());
        QPD_CHECK(cudaStreamSynchronize(st_));
        return 0;
    }
    size_t bytes() const override {
        return (size_t)g_.nelements *
               ((size_t)NV * 45 * sizeof(storage) + (CODEC::scaled ? sizeof(float) : 0) + (MD ? 45 * sizeof(float) : 0));
    }
    // In the host store's layout: codes [block][point][component][lane], means [block][component][lane].
    int download(std::vector<uint8_t> &codes, std::vector<float> &scales, std::vector<uint8_t> &extra) const override {
        const size_t ne = (size_t)g_.nelements, padded = (ne + V - 1) / V * V;
        std::vector<storage> mine(ne * 45 * TPE);
        QPD_CHECK(cudaMemcpy(mine.data(), codes_, mine.size() * sizeof(storage), cudaMemcpyDeviceToHost));
        codes.assign(padded * NV * 45 * sizeof(storage), 0);
        storage *const hc = reinterpret_cast<storage *>(codes.data());
        for (size_t e = 0; e < ne; ++e)
            for (int c = 0; c < 45; ++c)
                for (int j = 0; j < TPE; ++j) hc[(((e / V) * NV + j) * 45 + c) * V + e % V] = mine[(e * 45 + c) * TPE + j];
        scales.assign(CODEC::scaled ? padded : 0, 0.0f);
        if (CODEC::scaled) QPD_CHECK(cudaMemcpy(scales.data(), scales_, ne * sizeof(float), cudaMemcpyDeviceToHost));
        extra.assign(MD ? padded * 45 * sizeof(float) : 0, 0);
        if (MD) {
            std::vector<float> a(ne * 45);
            QPD_CHECK(cudaMemcpy(a.data(), aux_, a.size() * sizeof(float), cudaMemcpyDeviceToHost));
            float *const ha = reinterpret_cast<float *>(extra.data());
            for (size_t e = 0; e < ne; ++e)
                for (int c = 0; c < 45; ++c) ha[((e / V) * 45 + c) * V + e % V] = a[e * 45 + c];
        }
        return 0;
    }

private:
    Geometry g_;
    cudaStream_t st_;
    storage *codes_ = nullptr;
    float *scales_ = nullptr, *aux_ = nullptr;
};

template <class CODEC, bool PAIR>
class EaStore27 final : public Kernels {
public:
    using storage = typename CODEC::storage;
    EaStore27(const Geometry &g, void *stream) : g_(g), st_((cudaStream_t)stream) {
        fill_dt27();
        codes_ = device_alloc<storage>((size_t)g.nelements * NP * 9);
        scales_ = device_alloc<float>((size_t)g.nelements);
        if (PAIR) pair_factors_ = device_alloc<uint8_t>((size_t)g.nelements * NP);
    }
    ~EaStore27() override {
        cudaFree(codes_);
        cudaFree(scales_);
        cudaFree(pair_factors_);
    }
    int update(const double *x) override {
        setup_ea27<CODEC, PAIR><<<(unsigned)g_.nelements, EA_THREADS, 0, st_>>>(g_, x, codes_, scales_, pair_factors_);
        QPD_CHECK(cudaGetLastError());
        QPD_CHECK(cudaStreamSynchronize(st_));
        return 0;
    }
    int apply(const double *h, double *out) override {
        apply_ea27<CODEC, PAIR><<<(unsigned)((g_.nelements + EA_WARPS - 1) / EA_WARPS), EA_WARPS * 32, 0, st_>>>(
                g_, codes_, scales_, pair_factors_, h, out);
        QPD_CHECK(cudaGetLastError());
        QPD_CHECK(cudaStreamSynchronize(st_));
        return 0;
    }
    size_t bytes() const override {
        return (size_t)g_.nelements * ((size_t)NP * 9 * sizeof(storage) + sizeof(float) + (PAIR ? NP : 0));
    }
    // In the host store's layout: codes [block][entry][lane], pair factors [block][pair][lane].
    int download(std::vector<uint8_t> &codes, std::vector<float> &scales, std::vector<uint8_t> &extra) const override {
        const size_t ne = (size_t)g_.nelements, padded = (ne + V - 1) / V * V, NE = (size_t)NP * 9;
        std::vector<storage> mine(ne * NE);
        QPD_CHECK(cudaMemcpy(mine.data(), codes_, mine.size() * sizeof(storage), cudaMemcpyDeviceToHost));
        codes.assign(padded * NE * sizeof(storage), 0);
        storage *const hc = reinterpret_cast<storage *>(codes.data());
        for (size_t e = 0; e < ne; ++e)
            for (size_t y = 0; y < NE; ++y) hc[((e / V) * NE + y) * V + e % V] = mine[e * NE + y];
        scales.assign(padded, 0.0f);
        QPD_CHECK(cudaMemcpy(scales.data(), scales_, ne * sizeof(float), cudaMemcpyDeviceToHost));
        extra.assign(PAIR ? padded * NP : 0, 0);
        if (PAIR) {
            std::vector<uint8_t> f(ne * NP);
            QPD_CHECK(cudaMemcpy(f.data(), pair_factors_, f.size(), cudaMemcpyDeviceToHost));
            for (size_t e = 0; e < ne; ++e)
                for (int k = 0; k < NP; ++k) extra[((e / V) * NP + k) * V + e % V] = f[e * NP + k];
        }
        return 0;
    }

private:
    Geometry g_;
    cudaStream_t st_;
    storage *codes_ = nullptr;
    float *scales_ = nullptr;
    uint8_t *pair_factors_ = nullptr;
};

// ---------------------------------------------------------------------------- quantized BSR
//
// The layout the GPU kernel-shape test chose (RESULTS_QP.md): full difference form, packed records,
// sigma = 1024 rows; int16 offsets plus a remainder by row, except on HEX27 (rows of 125 blocks),
// where int32 columns are faster.  The structure is built once on the host and uploaded; each
// update encodes the device f32 assembly into the records, each block from the symmetrised
// (K_ij + K_ji^T) / 2 as on the host, its position and its transpose's found by binary search in the
// sorted rows.

constexpr bool BSR_OFFSETS = N != 27;
constexpr ptrdiff_t BSR_SIGMA = 1024, BSR_REACH = 32767;

__device__ __forceinline__ int64_t find(const int32_t *RSTR rp, const int32_t *RSTR ci, const int32_t i, const int32_t j) {
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

// The symmetrised block of (i, j).
__device__ __forceinline__ void block_of(const int32_t *RSTR rp, const int32_t *RSTR ci, const float *RSTR v,
                                         const int32_t i, const int32_t j, double S[9]) {
    const size_t k = (size_t)find(rp, ci, i, j), kt = (size_t)find(rp, ci, j, i);
#pragma unroll
    for (int r = 0; r < 3; ++r)
#pragma unroll
        for (int c = 0; c < 3; ++c) S[r * 3 + c] = 0.5 * ((double)v[k * 9 + r * 3 + c] + (double)v[kt * 9 + c * 3 + r]);
}

// The largest off-diagonal entry, as the bits of a non-negative float (atomicMax on int).
__global__ void bsr_max_kernel(const ptrdiff_t n, const int32_t *RSTR rp, const int32_t *RSTR ci, const float *RSTR v,
                               int *RSTR top) {
    float m = 0;
    for (ptrdiff_t i = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (ptrdiff_t)blockDim.x * gridDim.x)
        for (int64_t k = rp[i]; k < rp[i + 1]; ++k)
            if (ci[k] != i)
                for (int y = 0; y < 9; ++y) m = fmaxf(m, fabsf(v[(size_t)k * 9 + y]));
    for (int o = 16; o > 0; o /= 2) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) atomicMax(top, __float_as_int(m));
}

template <class CODEC>
__device__ __forceinline__ void encode_block(const double S[9], const double matrix_inv, typename CODEC::storage *RSTR c,
                                             const size_t stride, float *RSTR sc) {
    if constexpr (std::is_same<CODEC, I8>::value) {
        double top = 0;
#pragma unroll
        for (int y = 0; y < 9; ++y) top = fmax(top, fabs(S[y]));
        const double s = I8::scale_for(top);
        *sc = (float)s;
        const double inv = 1.0 / (double)(float)s;
#pragma unroll
        for (int y = 0; y < 9; ++y) c[y * stride] = I8::encode(S[y], inv);
    } else {
#pragma unroll
        for (int y = 0; y < 9; ++y) c[y * stride] = CODEC::encode(S[y], matrix_inv);
    }
}

// The records: one thread per row (group and lane), its steps in order.
template <class CODEC>
__global__ void bsr_encode_kernel(const qbsr::Sell s, const int32_t *RSTR rp, const int32_t *RSTR ci,
                                  const float *RSTR v, const double matrix_inv, uint8_t *RSTR records) {
    using rec_t = std::conditional_t<std::is_same<CODEC, I8>::value, qbsr::RecI8, qbsr::RecF16>;
    const ptrdiff_t t = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x;
    const ptrdiff_t gr = t / qbsr::C;
    const int lane = (int)(t % qbsr::C);
    if (gr >= s.n_groups) return;
    const int32_t i = s.rows[gr * qbsr::C + lane];
    for (int64_t step = s.group_ptr[gr]; step < s.group_ptr[gr + 1]; ++step) {
        const size_t slot = (size_t)step * qbsr::C + lane;
        rec_t *const r = reinterpret_cast<rec_t *>(records) + slot;
        int32_t j;
        if constexpr (BSR_OFFSETS) {
            if constexpr (std::is_same<CODEC, I8>::value)
                j = i + (int32_t)(int16_t)(uint16_t)(r->off_lo | (r->off_hi << 8));
            else
                j = i + (int32_t)(int16_t)r->off;
        } else {
            j = reinterpret_cast<const int32_t *>(s.cols)[slot];
        }
        if (i < 0 || j == i) continue;  // padding: zero codes from the structure
        double S[9];
        block_of(rp, ci, v, i, j, S);
        float sc = 1;
        encode_block<CODEC>(S, matrix_inv, reinterpret_cast<typename CODEC::storage *>(r->codes), 1, &sc);
        if constexpr (std::is_same<CODEC, I8>::value) r->scale = sc;
    }
}

template <class CODEC>
__global__ void bsr_encode_remainder_kernel(const qbsr::Sell s, const int32_t *RSTR rp, const int32_t *RSTR ci,
                                            const float *RSTR v, const double matrix_inv,
                                            typename CODEC::storage *RSTR codes, float *RSTR scales) {
    const ptrdiff_t r = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= s.n_rem_rows) return;
    const int32_t i = s.rem_rows[r];
    for (int64_t e = s.rem_rowptr[r]; e < s.rem_rowptr[r + 1]; ++e) {
        double S[9];
        block_of(rp, ci, v, i, s.rem_col[e], S);
        float sc = 1;
        encode_block<CODEC>(S, matrix_inv, codes + (size_t)e * 9, 1, &sc);
        if (scales) scales[e] = sc;
    }
}

template <class CODEC>
class BsrStore final : public Kernels {
public:
    static constexpr bool I8CODE = std::is_same<CODEC, I8>::value;
    using rec_t = std::conditional_t<I8CODE, qbsr::RecI8, qbsr::RecF16>;

    BsrStore(const Assembly &a, void *stream) : a_(a), st_((cudaStream_t)stream) {
        const ptrdiff_t n = a.n_rows;
        const int32_t *rp = a.h_rowptr, *ci = a.h_colidx;
        auto in_sell = [&](const ptrdiff_t i, const ptrdiff_t j) {
            return j != i && (!BSR_OFFSETS || std::abs(j - i) <= BSR_REACH);
        };
        std::vector<int32_t> len((size_t)n);
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            int32_t c = 0;
            for (int32_t k = rp[i]; k < rp[i + 1]; ++k) c += in_sell(i, ci[k]);
            len[(size_t)i] = c;
        }
        std::vector<int32_t> order((size_t)n);
        for (ptrdiff_t i = 0; i < n; ++i) order[(size_t)i] = (int32_t)i;
#pragma omp parallel for
        for (ptrdiff_t w = 0; w < n; w += BSR_SIGMA)
            std::stable_sort(order.begin() + w, order.begin() + std::min(n, w + BSR_SIGMA),
                             [&](int32_t x, int32_t y) { return len[(size_t)x] > len[(size_t)y]; });
        const ptrdiff_t ng = (n + qbsr::C - 1) / qbsr::C;
        std::vector<int32_t> rows((size_t)ng * qbsr::C, -1);
        std::copy(order.begin(), order.end(), rows.begin());
        std::vector<int64_t> gp((size_t)ng + 1, 0);
        for (ptrdiff_t g = 0; g < ng; ++g) {
            int32_t m = 0;
            for (int l = 0; l < qbsr::C; ++l) {
                const int32_t i = rows[(size_t)g * qbsr::C + l];
                if (i >= 0) m = std::max(m, len[(size_t)i]);
            }
            gp[(size_t)g + 1] = gp[(size_t)g] + m;
        }
        const size_t slots = (size_t)gp[(size_t)ng] * qbsr::C;
        // Records with the offsets set and zero codes (padding stays so); columns when not offsets.
        std::vector<rec_t> records(slots);
        std::memset(records.data(), 0, slots * sizeof(rec_t));
        std::vector<int32_t> cols(BSR_OFFSETS ? 0 : slots);
#pragma omp parallel for schedule(dynamic, 64)
        for (ptrdiff_t g = 0; g < ng; ++g)
            for (int l = 0; l < qbsr::C; ++l) {
                const int32_t i = rows[(size_t)g * qbsr::C + l];
                size_t step = (size_t)gp[(size_t)g];
                if (i >= 0)
                    for (int32_t k = rp[i]; k < rp[i + 1]; ++k) {
                        if (!in_sell(i, ci[k])) continue;
                        const size_t slot = step++ * qbsr::C + l;
                        if constexpr (BSR_OFFSETS) {
                            const uint16_t off = (uint16_t)(int16_t)(ci[k] - i);
                            if constexpr (I8CODE) {
                                records[slot].off_lo = (uint8_t)(off & 0xff);
                                records[slot].off_hi = (uint8_t)(off >> 8);
                            } else {
                                records[slot].off = off;
                            }
                        } else {
                            cols[slot] = ci[k];
                        }
                    }
                const int32_t self = i >= 0 ? i : rows[(size_t)g * qbsr::C];
                for (; step < (size_t)gp[(size_t)g + 1]; ++step)
                    if (!BSR_OFFSETS) cols[step * qbsr::C + l] = self;
            }
        std::vector<int64_t> rem_rowptr(1, 0);
        std::vector<int32_t> rem_rows, rem_col;
        if (BSR_OFFSETS)
            for (ptrdiff_t i = 0; i < n; ++i) {
                const size_t before = rem_col.size();
                for (int32_t k = rp[i]; k < rp[i + 1]; ++k)
                    if (ci[k] != i && !in_sell(i, ci[k])) rem_col.push_back(ci[k]);
                if (rem_col.size() > before) {
                    rem_rows.push_back((int32_t)i);
                    rem_rowptr.push_back((int64_t)rem_col.size());
                }
            }
        put(rows, rows_);
        put(gp, gp_);
        put(records, records_);
        put(cols, cols_);
        put(rem_rowptr, rem_rowptr_);
        put(rem_rows, rem_rows_);
        put(rem_col, rem_col_);
        rem_codes_ = device_alloc<typename CODEC::storage>(rem_col.size() * 9);
        if (I8CODE) rem_scales_ = device_alloc<float>(rem_col.size());
        bytes_ += rem_col.size() * 9 * sizeof(typename CODEC::storage) + (I8CODE ? rem_col.size() * sizeof(float) : 0);
        values_ = device_alloc<float>((size_t)a.h_rowptr[n] * 9);
        top_ = device_alloc<int>(1);
        s_.n_rows = n;
        s_.n_groups = ng;
        s_.rows = rows_;
        s_.group_ptr = gp_;
        s_.cols = cols_;
        s_.codes = records_;
        s_.n_rem = (ptrdiff_t)rem_col.size();
        s_.n_rem_rows = (ptrdiff_t)rem_rows.size();
        s_.rem_rowptr = rem_rowptr_;
        s_.rem_rows = rem_rows_;
        s_.rem_col = rem_col_;
        s_.rem_codes = rem_codes_;
        s_.rem_scales = rem_scales_;
        s_.accumulate = true;
    }
    ~BsrStore() override {
        for (void *p : owned_) cudaFree(p);
        cudaFree(rem_codes_);
        cudaFree(rem_scales_);
        cudaFree(values_);
        cudaFree(top_);
    }
    int update(const double *x) override {
        if (a_.assemble(x, values_)) return 1;
        double matrix_inv = 1;
        if (!I8CODE) {
            QPD_CHECK(cudaMemsetAsync(top_, 0, sizeof(int), st_));
            bsr_max_kernel<<<1024, 256, 0, st_>>>(a_.n_rows, a_.rowptr, a_.colidx, values_, top_);
            int bits = 0;
            QPD_CHECK(cudaMemcpyAsync(&bits, top_, sizeof(int), cudaMemcpyDeviceToHost, st_));
            QPD_CHECK(cudaStreamSynchronize(st_));
            float top;
            std::memcpy(&top, &bits, sizeof(float));
            // The host's ScaledHalfCodec::scale_for, in double.
            const float scale = top > 0 ? (float)std::exp2(std::ceil(std::log2((double)top)) - 14) : 1.0f;
            s_.matrix_scale = scale;
            matrix_inv = 1.0 / (double)scale;
        }
        bsr_encode_kernel<CODEC><<<grid(s_.n_groups * qbsr::C), THREADS, 0, st_>>>(
                s_, a_.rowptr, a_.colidx, values_, matrix_inv, reinterpret_cast<uint8_t *>(records_));
        if (s_.n_rem_rows > 0)
            bsr_encode_remainder_kernel<CODEC><<<grid(s_.n_rem_rows), THREADS, 0, st_>>>(
                    s_, a_.rowptr, a_.colidx, values_, matrix_inv, rem_codes_, rem_scales_);
        QPD_CHECK(cudaGetLastError());
        QPD_CHECK(cudaStreamSynchronize(st_));
        return 0;
    }
    int apply(const double *h, double *out) override {
        if (qbsr::apply(I8CODE ? qbsr::Code::I8 : qbsr::Code::F16, qbsr::Form::Full,
                        BSR_OFFSETS ? qbsr::Cols::Off16 : qbsr::Cols::I32, qbsr::Rec::Packed, s_, h, out, st_))
            return 1;
        QPD_CHECK(cudaStreamSynchronize(st_));
        return 0;
    }
    size_t bytes() const override { return bytes_; }

private:
    template <typename T>
    void put(const std::vector<T> &v, T *&d) {
        d = device_alloc<T>(v.size());
        if (!v.empty()) cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
        owned_.push_back(d);
        bytes_ += v.size() * sizeof(T);
    }
    Assembly a_;
    cudaStream_t st_;
    qbsr::Sell s_;
    std::vector<void *> owned_;
    size_t bytes_ = 0;
    int32_t *rows_ = nullptr, *cols_ = nullptr, *rem_rows_ = nullptr, *rem_col_ = nullptr;
    int64_t *gp_ = nullptr, *rem_rowptr_ = nullptr;
    rec_t *records_ = nullptr;
    typename CODEC::storage *rem_codes_ = nullptr;
    float *rem_scales_ = nullptr, *values_ = nullptr;
    int *top_ = nullptr;
};

}  // namespace

std::shared_ptr<void> attach_blocks(Geometry &g, const int32_t *h_elements) {
    const ptrdiff_t ne = g.nelements, nblocks = (ne + THREADS - 1) / THREADS;
    std::vector<int32_t> ptr((size_t)nblocks + 1, 0);
    std::vector<std::vector<int32_t>> nodes((size_t)nblocks);
    std::vector<uint16_t> local((size_t)N * ne);
#pragma omp parallel for schedule(dynamic, 16)
    for (ptrdiff_t b = 0; b < nblocks; ++b) {
        auto &v = nodes[(size_t)b];
        const ptrdiff_t e0 = b * THREADS, e1 = std::min(ne, e0 + THREADS);
        for (int p = 0; p < N; ++p)
            for (ptrdiff_t e = e0; e < e1; ++e) v.push_back(h_elements[(size_t)p * ne + e]);
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        for (int p = 0; p < N; ++p)
            for (ptrdiff_t e = e0; e < e1; ++e)
                local[(size_t)p * ne + e] =
                        (uint16_t)(std::lower_bound(v.begin(), v.end(), h_elements[(size_t)p * ne + e]) - v.begin());
    }
    int most = 0;
    for (ptrdiff_t b = 0; b < nblocks; ++b) {
        ptr[(size_t)b + 1] = ptr[(size_t)b] + (int32_t)nodes[(size_t)b].size();
        most = std::max(most, (int)nodes[(size_t)b].size());
    }
    std::vector<int32_t> flat((size_t)ptr[(size_t)nblocks]);
    for (ptrdiff_t b = 0; b < nblocks; ++b) std::copy(nodes[(size_t)b].begin(), nodes[(size_t)b].end(), flat.begin() + ptr[(size_t)b]);
    struct Owner {
        std::vector<void *> p;
        ~Owner() {
            for (void *q : p) cudaFree(q);
        }
    };
    auto owner = std::make_shared<Owner>();
    auto up = [&](const auto &v) {
        using T = typename std::decay_t<decltype(v)>::value_type;
        T *d = device_alloc<T>(v.size());
        cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
        owner->p.push_back(d);
        return d;
    };
    g.block_ptr = up(ptr);
    g.block_nodes = up(flat);
    g.local = up(local);
    g.max_block_nodes = most;
    return owner;
}

const char *element_name() { return El::name; }
int element_nodes() { return N; }

namespace {
// One element's kernels only: the other family is never instantiated (if constexpr in a template).
template <int NN>
std::unique_ptr<Kernels> create_for(const std::string &variant, const Geometry &g, void *stream) {
    if constexpr (NN == 27) {
        if (variant == "f64") return std::make_unique<PointStore27<F64, false>>(g, stream);
        if (variant == "f16") return std::make_unique<PointStore27<F16, false>>(g, stream);
        if (variant == "i8md") return std::make_unique<PointStore27<I8, true>>(g, stream);
        if (variant == "ead16") return std::make_unique<EaStore27<F16, false>>(g, stream);
        if (variant == "eadi8p") return std::make_unique<EaStore27<I8, true>>(g, stream);
        return nullptr;
    } else {
        // The kernel-shape test's choice per element and store (RESULTS_QP.md, measured_gpu/eshape_*):
        // the scatter (block-local, or atomicAdd per element node) and the resident blocks per SM.
        // QPD_SCATTER=atomic|block and QPD_MINB override them.
        struct Shape {
            bool block;
            int minb;
        };
        Shape shape{false, 1};
        if (NN == 8) {  // HEX8
            if (variant == "f16" || variant == "ead16") shape = {true, 4};
        } else if (NN == 4) {  // TET4
            shape = {true, 1};
        } else if (NN == 10) {  // TET10
            if (variant == "f16") shape = {true, 4};
        }
        if (const char *v = std::getenv("QPD_SCATTER")) shape.block = !std::strcmp(v, "block");
        Geometry gs = g;
        if (!shape.block || !g.local) gs.local = nullptr;
        if (variant == "f64") return std::make_unique<PointStore<F64, false>>(gs, stream, shape.minb);
        if (variant == "f16") return std::make_unique<PointStore<F16, false>>(gs, stream, shape.minb);
        if (variant == "i8md" && NV > 1) return std::make_unique<PointStore<I8, true>>(gs, stream, shape.minb);
        if (variant == "ead16") return std::make_unique<EaStore<F16, false>>(gs, stream, shape.minb);
        if (variant == "eadi8p") return std::make_unique<EaStore<I8, true>>(gs, stream, shape.minb);
        return nullptr;
    }
}
}  // namespace

std::unique_ptr<Kernels> create(const std::string &variant, const Geometry &g, void *stream) {
    return create_for<N>(variant, g, stream);
}

std::unique_ptr<Kernels> create_bsr(const std::string &variant, const Assembly &a, void *stream) {
    if (variant == "bsro16") return std::make_unique<BsrStore<F16>>(a, stream);
    if (variant == "bsroi8") return std::make_unique<BsrStore<I8>>(a, stream);
    return nullptr;
}

}  // namespace qpd
