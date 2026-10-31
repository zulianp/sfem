#pragma once

// The device stores' common half (sfem_cuda_InexactStore.cuh): the codes, the element's tables
// where a kernel indexes them at run time, and the set-up.
//
// The set-up runs a thread per element and point in double: the element's reference gradients at
// the points (each thread its own, from the generated derivative table), the material's point tangent (the generated `M::point`),
// the weighted points or moments in shared memory, then the store's encoding (its `Sink`) in the
// host's order, split over the element's threads.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "sfem_InexactElement.hpp"
#include "sfem_InexactInterface.hpp"
#include "sfem_InexactOptions.hpp"
#include "sfem_InexactSetup.hpp"
#include "sfem_cuda_HistoryCodecs.cuh"
#include "sfem_cuda_base.hpp"

namespace sfem {
    namespace inexact {
        namespace cuda {

            static constexpr int THREADS    = 128;  // a thread-per-element kernel's block
            static constexpr int V          = 32;   // elements per store block: one warp
            static constexpr int MAX_PARAMS = 32;
            static constexpr int MAX_FIELDS = 4;
            static constexpr int MAX_FIELD_COMPONENTS = 9;
            static constexpr int MAX_NODES  = 27;

            template <int B, int E, class F>
            __device__ __forceinline__ void static_for(F &&f) {
                if constexpr (B < E) {
                    f(std::integral_constant<int, B>{});
                    static_for<B + 1, E>(f);
                }
            }

            inline unsigned grid(const ptrdiff_t n, const int threads) { return (unsigned)((n + threads - 1) / threads); }

            inline void check(const cudaError_t e, const char *const what) {
                if (e != cudaSuccess) SFEM_ERROR("inexact (device): %s: %s\n", what, cudaGetErrorString(e));
            }

            /// The element's node arrays in the staging order, by value: a kernel reads a node's
            /// index with one load (through the mesh-order device array and the permutation, it
            /// was three dependent loads ahead of the first barrier: HEX27 points on the H100 2%).
            struct Connectivity {
                const idx_t *node[MAX_NODES]{};
            };

            /// `Connectivity` from the device array of device arrays (mesh order) the operator
            /// passes: bound at the set-up, its N pointers copied to the host once per array.
            template <class El>
            class StagedConnectivity {
            public:
                static_assert(El::N <= MAX_NODES, "more nodes per element than a kernel's connectivity carries");

                const Connectivity &bind(const idx_t *const *const elements) {
                    if (elements != bound_) {
                        const idx_t *pointers[El::N];
                        check(cudaMemcpy(pointers, elements, El::N * sizeof(idx_t *), cudaMemcpyDeviceToHost), "connectivity");
                        for (int s = 0; s < El::N; ++s) staged_.node[s] = pointers[ElementTables<El>::shape_node(s)];
                        bound_ = elements;
                    }
                    return staged_;
                }

                const Connectivity &operator()(const idx_t *const *const elements) const {
                    if (elements != bound_) SFEM_ERROR("inexact (device): applied on another connectivity than the set-up's\n");
                    return staged_;
                }

            private:
                const idx_t *const *bound_{nullptr};
                Connectivity        staged_;
            };

            // ------------------------------------------------------------------- codes

            /// The host's codes (CodeTraits, sfem_InexactCodecs.hpp) with the device codecs: the same
            /// bits, the same scales.  `tau` is the arithmetic the application contracts in.
            template <typename code_t>
            struct Code;

            template <>
            struct Code<double> {
                using storage                = double;
                using tau                    = double;
                static constexpr bool scaled = false;
                __device__ static double  scale_for(double) { return 1; }
                __device__ static storage encode(const double v, double) { return v; }
                __device__ static tau     decode(const storage c) { return c; }
            };

            template <>
            struct Code<float> {
                using storage                = float;
                using tau                    = float;
                static constexpr bool scaled = false;
                __device__ static double  scale_for(double) { return 1; }
                __device__ static storage encode(const double v, double) { return (float)v; }
                __device__ static tau     decode(const storage c) { return c; }
            };

            template <>
            struct Code<__half> {
                using storage                = __half;
                using tau                    = float;
                static constexpr bool scaled = true;
                using codec                  = device_codecs::DeviceScaledHalf;
                __device__ static double  scale_for(const double m) { return (double)codec::scale_for((real_t)m); }
                __device__ static storage encode(const double v, const double inv) { return codec::encode((real_t)v, (real_t)inv); }
                __device__ static tau     decode(const storage c) { return __half2float(c); }
            };

            template <>
            struct Code<int8_t> {
                using storage                = int8_t;
                using tau                    = float;
                static constexpr bool scaled = true;
                using codec                  = device_codecs::DeviceScaledInteger<int8_t, 127>;
                __device__ static double  scale_for(const double m) { return (double)codec::scale_for((real_t)m); }
                __device__ static storage encode(const double v, const double inv) { return codec::encode((real_t)v, (real_t)inv); }
                __device__ static tau     decode(const storage c) { return float(c); }
            };

            /// The host's PairFactor (sfem_InexactCodecs.hpp) with the device intrinsics.
            struct PairFactor {
                __device__ static float value(const uint8_t b) {
                    return __int_as_float(((127 - (int32_t)(b >> 3)) << 23) | ((int32_t)(b & 7) << 20));
                }
                __device__ static uint8_t encode(const double ratio) {
                    if (!(ratio > ldexp(1.0, -31))) return uint8_t(31 << 3);
                    int E = min(31, (int)ceil(-log2(ratio)));
                    int m = (int)ceil((ldexp(ratio, E) - 1) * 8);
                    if (m < 0) m = 0;
                    while (E > 0 && (double)value((uint8_t)((E << 3) | min(m, 7))) < ratio) {
                        if (++m > 7) {
                            m = 0;
                            --E;
                        }
                    }
                    return (uint8_t)((E << 3) | min(m, 7));
                }
            };

            // ------------------------------------------------------------------ memory

            template <typename T>
            struct DeviceArray {
                T          *data{nullptr};
                std::size_t size{0};

                DeviceArray() = default;
                explicit DeviceArray(const std::size_t n) : size(n) {
                    if (n) {
                        check(cudaMalloc((void **)&data, n * sizeof(T)), "cudaMalloc");
                        check(cudaMemset(data, 0, n * sizeof(T)), "cudaMemset");
                    }
                }
                DeviceArray(const DeviceArray &)            = delete;
                DeviceArray &operator=(const DeviceArray &) = delete;
                DeviceArray(DeviceArray &&o) noexcept : data(o.data), size(o.size) { o.data = nullptr, o.size = 0; }
                DeviceArray &operator=(DeviceArray &&o) noexcept {
                    std::swap(data, o.data);
                    std::swap(size, o.size);
                    return *this;
                }
                ~DeviceArray() {
                    if (data) cudaFree(data);
                }
                static DeviceArray from(const T *const v, const std::size_t n) {
                    DeviceArray a(n);
                    if (n) check(cudaMemcpy(a.data, v, n * sizeof(T), cudaMemcpyHostToDevice), "upload");
                    return a;
                }
                static DeviceArray from(const std::vector<T> &v) { return from(v.data(), v.size()); }
                std::size_t bytes() const { return size * sizeof(T); }
            };

            // ------------------------------------------------------------------ tables

            /// The element's tables where a kernel indexes them at run time (the set-up, the
            /// diagonal, HEX27's point threads): a device copy of the generated header's constants.
            /// The unrolled kernels read the header's own constants, as constant expressions.
            struct TablesView {
                const double *qw{nullptr};           // [NQ]
                const double *factor{nullptr};       // [NQ][NV], moments
                const double *mult{nullptr};         // [NV]
                const double *x{nullptr};            // [NV]
                const double *dt{nullptr};           // [R][3][N]
                const double *dq{nullptr};           // [NQ][N][4], at the set-up's points (the 4th zero)
                const int    *pa{nullptr}, *pb{nullptr};  // [NV]
                const int    *slot{nullptr};         // [9][9]
                const int    *mask{nullptr};         // [NV], folded
                const int    *source{nullptr};       // [NV][3]
                const double *coefficient{nullptr};  // [NV][3]
                const int    *dir{nullptr};          // [NC]
                const int    *off{nullptr};          // [NV + 1]
                const int    *nkeep{nullptr};        // [8]
                const int    *keep{nullptr};         // [8][NC]
                const double *xc{nullptr};           // [NV][NC]
                const double *b1{nullptr}, *d1{nullptr};  // [Q1][S1], tensor-product sweeps
            };

            template <class M, bool FOLD>
            class Tables {
            public:
                using El                = typename M::element;
                using Fold              = typename FoldOf<M>::type;
                using T                 = ElementTables<El>;
                static constexpr int N  = El::N, NQ = El::NQ, NV = El::NV, R = El::R, NC = M::NC;

                Tables() {
                    std::vector<double> d;
                    std::vector<int>    i;
                    auto add_d = [&](std::size_t &at, const std::vector<double> &v) { at = d.size(), d.insert(d.end(), v.begin(), v.end()); };
                    auto add_i = [&](std::size_t &at, const std::vector<int> &v) { at = i.size(), i.insert(i.end(), v.begin(), v.end()); };
                    std::vector<int>    pa(NV), pb(NV), slot(81);
                    std::vector<double> qw(NQ), mult(NV), x(NV), dt((size_t)R * 3 * N), dq((size_t)NQ * N * 4), factor;
                    for (int q = 0; q < NQ; ++q) qw[q] = El::QW[q];
                    for (int j = 0; j < NV; ++j) pa[j] = T::pa(j), pb[j] = T::pb(j), mult[j] = T::mult(j), x[j] = El::X[j];
                    for (int r = 0; r < R; ++r)
                        for (int m = 0; m < 3; ++m)
                            for (int s = 0; s < N; ++s) dt[((size_t)r * 3 + m) * N + s] = El::DT[r][m][s];
                    // The set-up's points are the stored ones but for the moments, which integrate
                    // over the rule's own (`DQ`).
                    static_assert(El::moments || El::R == NQ, "a point store's set-up points are its gradient points");
                    for (int q = 0; q < NQ; ++q)
                        for (int m = 0; m < 3; ++m)
                            for (int s = 0; s < N; ++s) {
                                if constexpr (El::moments) {
                                    dq[((size_t)q * N + s) * 4 + m] = El::DQ[q][m][s];
                                } else {
                                    dq[((size_t)q * N + s) * 4 + m] = El::DT[q][m][s];
                                }
                            }
                    for (int r = 0; r < 9; ++r)
                        for (int c = 0; c < 9; ++c) slot[r * 9 + c] = M::contraction::slot(r, c);
                    if constexpr (El::moments) {
                        for (int q = 0; q < NQ; ++q)
                            for (int j = 0; j < NV; ++j) factor.push_back(El::FACTOR[q][j]);
                    }
                    std::size_t o_pa, o_pb, o_slot, o_qw, o_mult, o_x, o_dt, o_dq, o_factor;
                    add_i(o_pa, pa), add_i(o_pb, pb), add_i(o_slot, slot);
                    add_d(o_qw, qw), add_d(o_mult, mult), add_d(o_x, x), add_d(o_dt, dt);
                    d.resize((d.size() + 1) / 2 * 2);  // 16-byte aligned: a node's derivatives load as double2
                    add_d(o_dq, dq), add_d(o_factor, factor);
                    std::size_t o_mask = 0, o_source = 0, o_coef = 0, o_dir = 0, o_off = 0, o_nkeep = 0, o_keep = 0, o_xc = 0;
                    if constexpr (FOLD) {
                        std::vector<int>    mask(NV), source((size_t)NV * 3), dir(NC), off(NV + 1), nkeep(8), keep((size_t)8 * NC, -1);
                        std::vector<double> coef((size_t)NV * 3), xc((size_t)NV * NC);
                        for (int j = 0; j < NV; ++j) {
                            mask[j] = Fold::mask(j);
                            off[j]  = Fold::OFF[j];
                            for (int dd = 0; dd < 3; ++dd) source[j * 3 + dd] = Fold::source(j, dd), coef[j * 3 + dd] = Fold::coefficient(j, dd);
                            for (int c = 0; c < NC; ++c) xc[(size_t)j * NC + c] = Fold::XC[j][c];
                        }
                        off[NV] = Fold::NS;
                        for (int c = 0; c < NC; ++c) dir[c] = Fold::DIR[c];
                        for (int m = 0; m < 8; ++m) {
                            nkeep[m] = Fold::NKEEP[m];
                            for (int k = 0; k < Fold::NKEEP[m]; ++k) keep[(size_t)m * NC + k] = Fold::KEEP[m][k];
                        }
                        add_i(o_mask, mask), add_i(o_source, source), add_i(o_dir, dir), add_i(o_off, off), add_i(o_nkeep, nkeep),
                                add_i(o_keep, keep);
                        add_d(o_coef, coef), add_d(o_xc, xc);
                    }
                    std::size_t o_b1 = 0, o_d1 = 0;
                    if constexpr (N == 27) {
                        std::vector<double> b1, d1;
                        for (int q = 0; q < El::Q1; ++q)
                            for (int s = 0; s < El::S1; ++s) b1.push_back(El::B1[q][s]), d1.push_back(El::D1[q][s]);
                        add_d(o_b1, b1), add_d(o_d1, d1);
                    }
                    doubles_ = DeviceArray<double>::from(d);
                    ints_    = DeviceArray<int>::from(i);
                    const double *D = doubles_.data;
                    const int    *I = ints_.data;
                    view_.pa = I + o_pa, view_.pb = I + o_pb, view_.slot = I + o_slot;
                    view_.qw = D + o_qw, view_.mult = D + o_mult, view_.x = D + o_x, view_.dt = D + o_dt, view_.dq = D + o_dq;
                    view_.factor = El::moments ? D + o_factor : nullptr;
                    if constexpr (FOLD) {
                        view_.mask = I + o_mask, view_.source = I + o_source, view_.dir = I + o_dir, view_.off = I + o_off;
                        view_.nkeep = I + o_nkeep, view_.keep = I + o_keep, view_.coefficient = D + o_coef, view_.xc = D + o_xc;
                    }
                    if constexpr (N == 27) view_.b1 = D + o_b1, view_.d1 = D + o_d1;
                }

                const TablesView &view() const { return view_; }

            private:
                DeviceArray<double> doubles_;
                DeviceArray<int>    ints_;
                TablesView          view_;
            };

            // ----------------------------------------------------------------- the set-up

            /// What the set-up kernel reads, by value.
            struct SetupArgs {
                ptrdiff_t             nelements{0};
                Connectivity          elements;
                const geom_t *const  *points{nullptr};    // device array, when not affine
                const geom_t         *adjugate[9]{};      // affine
                const geom_t         *determinant{nullptr};
                bool                  affine{false};
                real_t                parameters[MAX_PARAMS]{};
                ptrdiff_t             u_stride{3};
                const real_t         *u{nullptr};
                const real_t         *z{nullptr};
                const real_t         *quadrature[MAX_FIELDS]{};
                int                   quadrature_components[MAX_FIELDS]{};
                ptrdiff_t             quadrature_stride{0};
            };

            template <class M>
            SetupArgs setup_args(const SetupInput &in, const Connectivity &elements) {
                static_assert(M::n_parameters <= MAX_PARAMS, "more material constants than the set-up carries");
                static_assert(M::n_quadrature_fields <= MAX_FIELDS && M::max_quadrature_components <= MAX_FIELD_COMPONENTS,
                              "more integration-point fields than the set-up carries");
                SetupArgs a;
                a.nelements = in.nelements;
                a.elements  = elements;
                a.affine    = in.adjugate != nullptr;
                if (a.affine) {
                    for (int c = 0; c < 9; ++c) a.adjugate[c] = in.adjugate[c];
                    a.determinant = in.determinant;
                } else if (!in.points) {
                    SFEM_ERROR("inexact (device): neither the affine geometry nor the coordinates were given\n");
                } else {
                    a.points = in.points;
                }
                for (int p = 0; p < M::n_parameters; ++p) a.parameters[p] = in.parameters[p];
                a.u_stride = in.u_stride;
                a.u        = in.u;
                a.z        = in.z;
                if (M::reads_previous && !in.z) SFEM_ERROR("inexact (device): the tangent reads a previous state that was not given\n");
                if (M::n_quadrature_fields > 0 && !in.quadrature) {
                    SFEM_ERROR("inexact (device): the tangent reads integration-point fields that were not given\n");
                }
                if constexpr (M::n_quadrature_fields > 0) {
                    for (int f = 0; f < M::n_quadrature_fields; ++f) {
                        a.quadrature[f]            = in.quadrature[f];
                        a.quadrature_components[f] = M::quadrature_components[f];
                    }
                }
                a.quadrature_stride = in.quadrature_stride;
                return a;
            }

            /// The set-up's shared memory per element, in doubles: the gathered state, previous
            /// state and coordinates, the weighted point tangents, the stored tensors (the moments,
            /// on TET10), the partial maxima, then the store's own (`Sink::extra_doubles`).  The
            /// gathered values are dead once the point tangents are written: the store's scratch
            /// (`Sink::scratch_doubles`) reuses them.
            template <class M, class Sink>
            struct SetupShared {
                using El               = typename M::element;
                static constexpr int N = El::N, NQ = El::NQ, NV = El::NV, NC = M::NC;
                static constexpr int H        = 0;
                static constexpr int TS       = H + std::max(9 * N, Sink::scratch_doubles);
                static constexpr int S        = TS + NQ * NC;
                static constexpr int TOPS     = S + (El::moments ? NV * NC : 0);
                static constexpr int EXTRA    = TOPS + NQ;
                static constexpr int per_element = EXTRA + Sink::extra_doubles;
            };

            /// The reference gradient at one point, G[k*3+m], from the shape derivatives there
            /// (`d[j][m]`, padded to four), in the stages' difference form: the nodal values less the first.  A
            /// node's derivatives and differences are read once (with the node innermost, each
            /// was read per direction and per component: HEX27 a quarter of the set-up's
            /// instructions); every entry sums over the nodes in the same order.
            template <int N>
            __device__ __forceinline__ void point_gradient(const double *const RSTR d, const double (*const h)[N], real_t G[9]) {
                double acc[9] = {0};
#pragma unroll
                for (int j = 1; j < N; ++j) {
                    const double2 w01 = reinterpret_cast<const double2 *>(d)[2 * j];
                    const double  w0 = w01.x, w1 = w01.y, w2 = d[4 * j + 2];
#pragma unroll
                    for (int k = 0; k < 3; ++k) {
                        const double v = h[k][j] - h[k][0];
                        acc[k * 3 + 0] += w0 * v;
                        acc[k * 3 + 1] += w1 * v;
                        acc[k * 3 + 2] += w2 * v;
                    }
                }
#pragma unroll
                for (int c = 0; c < 9; ++c) G[c] = acc[c];
            }

            template <class M, class Sink>
            __global__ void setup_kernel(const SetupArgs a, const TablesView t, const Sink sink) {
                using El               = typename M::element;
                using L                = SetupShared<M, Sink>;
                static constexpr int N = El::N, NQ = El::NQ, NV = El::NV, NC = M::NC;
                extern __shared__ double smem[];
                const int       epb    = blockDim.x / NQ;
                const int       le     = threadIdx.x / NQ, q = threadIdx.x % NQ;
                const ptrdiff_t e      = (ptrdiff_t)blockIdx.x * epb + le;
                const bool      active = le < epb && e < a.nelements;
                double *const   base   = smem + (size_t)min(le, epb - 1) * L::per_element;
                double(*const hu)[N]   = reinterpret_cast<double(*)[N]>(base + L::H);
                double(*const hz)[N]   = hu + 3;
                double(*const hx)[N]   = hz + 3;
                double(*const Ts)[NC]  = reinterpret_cast<double(*)[NC]>(base + L::TS);
                double(*const S)[NC]   = El::moments ? reinterpret_cast<double(*)[NC]>(base + L::S) : Ts;
                double *const tops     = base + L::TOPS;

                if (active) {
                    for (int s = q; s < N; s += NQ) {
                        const ptrdiff_t node = a.elements.node[s][e];
                        for (int k = 0; k < 3; ++k) {
                            if constexpr (M::reads_state) hu[k][s] = a.u[node * a.u_stride + k];
                            if constexpr (M::reads_previous) hz[k][s] = a.z[node * a.u_stride + k];
                            if (!a.affine) hx[k][s] = (double)a.points[k][node];
                        }
                    }
                }
                __syncthreads();
                if (active) {
                    // Each thread differentiates at its own point.
                    const double *const RSTR d = t.dq + (size_t)q * N * 4;
                    real_t                   gu[9] = {0}, gz[9] = {0}, adj[9], det;
                    if constexpr (M::reads_state) point_gradient<N>(d, hu, gu);
                    if constexpr (M::reads_previous) point_gradient<N>(d, hz, gz);
                    if (a.affine) {
                        for (int c = 0; c < 9; ++c) adj[c] = (real_t)a.adjugate[c][e];
                        det = (real_t)a.determinant[e];
                    } else {
                        real_t J[9];
                        point_gradient<N>(d, hx, J);
                        detail::adjugate_and_determinant(J, adj, det);
                    }
                    real_t        qf[MAX_FIELDS][MAX_FIELD_COMPONENTS];
                    const real_t *qp[MAX_FIELDS];
                    for (int f = 0; f < M::n_quadrature_fields; ++f) {
                        for (int c = 0; c < a.quadrature_components[f]; ++c)
                            qf[f][c] = a.quadrature[f][((ptrdiff_t)c * NQ + q) * a.quadrature_stride + e];
                        qp[f] = qf[f];
                    }
                    real_t T[NC];
                    M::template point<real_t>(a.parameters, gu, gz, adj, det, qp, T);
                    const real_t w = t.qw[q];
                    for (int c = 0; c < NC; ++c) Ts[q][c] = w * T[c];
                }
                __syncthreads();
                if constexpr (El::moments) {
                    // The moments in the host's order: the points ascending, from zero.
                    if (active)
                        for (int j = q; j < NV; j += NQ)
                            for (int c = 0; c < NC; ++c) {
                                double acc = 0;
                                for (int p = 0; p < NQ; ++p) acc += t.factor[p * NV + j] * Ts[p][c];
                                S[j][c] = acc;
                            }
                    __syncthreads();
                }
                sink.encode(t, e, active, q, S, tops, base + L::EXTRA, base + L::H);
            }

            /// The element's largest magnitude over its threads' partial maxima.  Whole warps per
            /// element (HEX27) reduce by shuffles, then over the element's warps; otherwise each
            /// thread reads the element's partial maxima.
            template <int NQ>
            __device__ __forceinline__ double element_max(double *const tops, const int q, double mine) {
                if constexpr (NQ % 32 == 0) {
                    for (int o = 16; o > 0; o /= 2) mine = fmax(mine, __shfl_xor_sync(0xffffffffu, mine, o));
                    if (q % 32 == 0) tops[q / 32] = mine;
                    __syncthreads();
                    double top = 0;
                    for (int w = 0; w < NQ / 32; ++w) top = fmax(top, tops[w]);
                    return top;
                } else {
                    tops[q] = mine;
                    __syncthreads();
                    double top = 0;
                    for (int p = 0; p < NQ; ++p) top = fmax(top, tops[p]);
                    return top;
                }
            }

            /// Launches the set-up with as many elements per block as fit, up to `THREADS` threads.
            template <class M, class Sink>
            void launch_setup(const SetupArgs &a, const TablesView &t, const Sink &sink, cudaStream_t stream) {
                using L                     = SetupShared<M, Sink>;
                static constexpr int  NQ    = M::element::NQ;
                static constexpr std::size_t per = (std::size_t)L::per_element * sizeof(double);
                static constexpr std::size_t cap = 96 * 1024;
                static_assert(per <= 200 * 1024, "one element's set-up does not fit a block's shared memory");
                int epb = std::max(1, THREADS / NQ);
                while (epb > 1 && epb * per > cap) --epb;
                const std::size_t shared = epb * per;
                if (shared > 48 * 1024) {
                    check(cudaFuncSetAttribute(setup_kernel<M, Sink>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared),
                          "set-up shared memory");
                }
                if (a.nelements == 0) return;
                const unsigned blocks = (unsigned)((a.nelements + epb - 1) / epb);
                setup_kernel<M, Sink><<<blocks, epb * NQ, shared, stream>>>(a, t, sink);
                check(cudaGetLastError(), "set-up launch");
                SFEM_DEBUG_SYNCHRONIZE(stream);
            }

        }  // namespace cuda
    }  // namespace inexact
}  // namespace sfem

