// Quantized BSR kernels for the GPU kernel-shape test (see qbsr_kernels.hpp).
#include "qbsr_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <type_traits>

namespace qbsr {
namespace {

constexpr int THREADS = 128;

template <Code CODE>
struct Codes;
template <>
struct Codes<Code::I8> {
    using code_t = int8_t;
    using rec_t = RecI8;
    static __device__ __forceinline__ float value(const code_t c) { return float(c); }
};
template <>
struct Codes<Code::F16> {
    using code_t = uint16_t;
    using rec_t = RecF16;
    static __device__ __forceinline__ float value(const code_t c) { return __half2float(__ushort_as_half(c)); }
};
template <>
struct Codes<Code::F32> {
    using code_t = float;
    using rec_t = RecF32;
    static __device__ __forceinline__ float value(const code_t c) { return c; }
};

// One block: its codes as floats, its scale (int8; 1 for f16) and its column.
template <Code CODE, Cols COLS, Rec REC, bool UPPER, bool COLUMN = true>
__device__ __forceinline__ int32_t load_block(const Sell &s, const size_t slot, const size_t step, const int lane,
                                              const int32_t i, float K[9], float &sc) {
    using T = Codes<CODE>;
    using code_t = typename T::code_t;
    int32_t j = i;
    if constexpr (REC == Rec::Packed) {
        if constexpr (CODE == Code::I8) {
            const uint4 w = __ldg(reinterpret_cast<const uint4 *>(s.codes) + slot);
            const uint32_t ws[3] = {w.x, w.y, w.z};
#pragma unroll
            for (int e = 0; e < 9; ++e) K[e] = float((int8_t)((ws[e >> 2] >> (8 * (e & 3))) & 0xff));
            sc = __uint_as_float(w.w);
            if constexpr (COLS == Cols::Off16) {
                const uint32_t off = (w.z >> 8) & 0xffff;  // bytes 9 and 10
                j = UPPER ? i + (int32_t)off : i + (int32_t)(int16_t)off;
            }
        } else if constexpr (CODE == Code::F32) {
            const uint32_t *const r = reinterpret_cast<const uint32_t *>(s.codes) + slot * 10;
#pragma unroll
            for (int e = 0; e < 9; ++e) K[e] = __uint_as_float(__ldg(r + e));
            sc = 1;
            if constexpr (COLS == Cols::Off16) {
                const uint32_t off = __ldg(r + 9);
                j = UPPER ? i + (int32_t)(off & 0xffff) : i + (int32_t)(int16_t)(off & 0xffff);
            }
        } else {
            const uint32_t *const r = reinterpret_cast<const uint32_t *>(s.codes) + slot * 5;
            uint32_t ws[5];
#pragma unroll
            for (int q = 0; q < 5; ++q) ws[q] = __ldg(r + q);
#pragma unroll
            for (int e = 0; e < 9; ++e) K[e] = T::value((uint16_t)((ws[e >> 1] >> (16 * (e & 1))) & 0xffff));
            sc = 1;
            if constexpr (COLS == Cols::Off16) {
                const uint32_t off = ws[4] >> 16;
                j = UPPER ? i + (int32_t)off : i + (int32_t)(int16_t)off;
            }
        }
    } else {
        const code_t *const c = reinterpret_cast<const code_t *>(s.codes) + step * 9 * C + lane;
#pragma unroll
        for (int e = 0; e < 9; ++e) K[e] = T::value(__ldg(c + e * C));
        sc = CODE == Code::I8 ? __ldg(s.scales + slot) : 1.0f;
    }
    if constexpr (!COLUMN) {
    } else if constexpr (COLS == Cols::I32) {
        j = __ldg(reinterpret_cast<const int32_t *>(s.cols) + slot);
    } else if constexpr (REC == Rec::Scalar) {
        if constexpr (UPPER)
            j = i + (int32_t)__ldg(reinterpret_cast<const uint16_t *>(s.cols) + slot);
        else
            j = i + (int32_t)__ldg(reinterpret_cast<const int16_t *>(s.cols) + slot);
    }
    return j;
}

template <Code CODE, Form F, Cols COLS, Rec REC>
__global__ void __launch_bounds__(THREADS) sell_kernel(const Sell s, const double *__restrict__ x,
                                                       double *__restrict__ y) {
    constexpr bool UPPER = F != Form::Full;
    const ptrdiff_t t = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x;
    const ptrdiff_t g = t / C;
    const int lane = (int)(t % C);
    if (g >= s.n_groups) return;
    const int32_t i = s.rows[g * C + lane];
    if (i < 0) return;
    const double xi0 = x[(size_t)i * 3 + 0], xi1 = x[(size_t)i * 3 + 1], xi2 = x[(size_t)i * 3 + 2];
    const float ms = s.matrix_scale;
    float a0 = 0, a1 = 0, a2 = 0;
    for (int64_t step = s.group_ptr[g]; step < s.group_ptr[g + 1]; ++step) {
        const size_t slot = (size_t)step * C + lane;
        float K[9], sc;
        const int32_t j = load_block<CODE, COLS, REC, UPPER>(s, slot, (size_t)step, lane, i, K, sc);
        if (j == i) continue;  // padding
        const float d0 = float(x[(size_t)j * 3 + 0] - xi0) * sc, d1 = float(x[(size_t)j * 3 + 1] - xi1) * sc,
                    d2 = float(x[(size_t)j * 3 + 2] - xi2) * sc;
        a0 += K[0] * d0 + K[1] * d1 + K[2] * d2;
        a1 += K[3] * d0 + K[4] * d1 + K[5] * d2;
        a2 += K[6] * d0 + K[7] * d1 + K[8] * d2;
        if constexpr (F == Form::Upper) {
            atomicAdd(&y[(size_t)j * 3 + 0], -(double)((K[0] * d0 + K[3] * d1 + K[6] * d2) * ms));
            atomicAdd(&y[(size_t)j * 3 + 1], -(double)((K[1] * d0 + K[4] * d1 + K[7] * d2) * ms));
            atomicAdd(&y[(size_t)j * 3 + 2], -(double)((K[2] * d0 + K[5] * d1 + K[8] * d2) * ms));
        }
    }
    if constexpr (F == Form::UpperGather) {
        // The lower neighbours: K_ji^T (x_j - x_i) from the partner's record, which holds K_ji.
        for (int64_t step = s.lgroup_ptr[g]; step < s.lgroup_ptr[g + 1]; ++step) {
            const size_t slot = (size_t)step * C + lane;
            const int32_t rec = __ldg(s.lrec + slot);
            if (rec < 0) continue;
            const int32_t j = __ldg(s.lcol + slot);
            float K[9], sc;
            load_block<CODE, Cols::I32, REC, true, false>(s, (size_t)rec, (size_t)rec / C, rec % C, j, K, sc);
            const float d0 = float(x[(size_t)j * 3 + 0] - xi0) * sc, d1 = float(x[(size_t)j * 3 + 1] - xi1) * sc,
                        d2 = float(x[(size_t)j * 3 + 2] - xi2) * sc;
            a0 += K[0] * d0 + K[3] * d1 + K[6] * d2;
            a1 += K[1] * d0 + K[4] * d1 + K[7] * d2;
            a2 += K[2] * d0 + K[5] * d1 + K[8] * d2;
        }
    }
    if constexpr (F == Form::Upper) {
        atomicAdd(&y[(size_t)i * 3 + 0], (double)(a0 * ms));
        atomicAdd(&y[(size_t)i * 3 + 1], (double)(a1 * ms));
        atomicAdd(&y[(size_t)i * 3 + 2], (double)(a2 * ms));
    } else {
        double *const yi = &y[(size_t)i * 3];
        yi[0] = (s.accumulate ? yi[0] : 0.0) + (double)(a0 * ms);
        yi[1] = (s.accumulate ? yi[1] : 0.0) + (double)(a1 * ms);
        yi[2] = (s.accumulate ? yi[2] : 0.0) + (double)(a2 * ms);
    }
}

// Remainder by row (Full, UpperGather): added after the SELL kernel, in a fixed order.
template <Code CODE>
__global__ void remainder_rows_kernel(const Sell s, const double *__restrict__ x, double *__restrict__ y) {
    using T = Codes<CODE>;
    using code_t = typename T::code_t;
    const ptrdiff_t r = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= s.n_rem_rows) return;
    const int32_t i = s.rem_rows[r];
    const double xi0 = x[(size_t)i * 3 + 0], xi1 = x[(size_t)i * 3 + 1], xi2 = x[(size_t)i * 3 + 2];
    float a0 = 0, a1 = 0, a2 = 0;
    for (int64_t e = s.rem_rowptr[r]; e < s.rem_rowptr[r + 1]; ++e) {
        const int32_t j = s.rem_col[e];
        const code_t *const c = reinterpret_cast<const code_t *>(s.rem_codes) + (size_t)e * 9;
        const float sc = CODE == Code::I8 ? s.rem_scales[e] : 1.0f;
        const float d0 = float(x[(size_t)j * 3 + 0] - xi0) * sc, d1 = float(x[(size_t)j * 3 + 1] - xi1) * sc,
                    d2 = float(x[(size_t)j * 3 + 2] - xi2) * sc;
        a0 += T::value(c[0]) * d0 + T::value(c[1]) * d1 + T::value(c[2]) * d2;
        a1 += T::value(c[3]) * d0 + T::value(c[4]) * d1 + T::value(c[5]) * d2;
        a2 += T::value(c[6]) * d0 + T::value(c[7]) * d1 + T::value(c[8]) * d2;
    }
    const float ms = s.matrix_scale;
    y[(size_t)i * 3 + 0] += (double)(a0 * ms);
    y[(size_t)i * 3 + 1] += (double)(a1 * ms);
    y[(size_t)i * 3 + 2] += (double)(a2 * ms);
}

// Remainder by pair (Upper): both rows, atomically.
template <Code CODE>
__global__ void remainder_pairs_kernel(const Sell s, const double *__restrict__ x, double *__restrict__ y) {
    using T = Codes<CODE>;
    using code_t = typename T::code_t;
    const ptrdiff_t e = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= s.n_rem) return;
    const int32_t i = s.rem_row[e], j = s.rem_col[e];
    const code_t *const c = reinterpret_cast<const code_t *>(s.rem_codes) + (size_t)e * 9;
    float K[9];
#pragma unroll
    for (int q = 0; q < 9; ++q) K[q] = T::value(c[q]);
    const float sc = (CODE == Code::I8 ? s.rem_scales[e] : 1.0f) * s.matrix_scale;
    const float d0 = float(x[(size_t)j * 3 + 0] - x[(size_t)i * 3 + 0]) * sc,
                d1 = float(x[(size_t)j * 3 + 1] - x[(size_t)i * 3 + 1]) * sc,
                d2 = float(x[(size_t)j * 3 + 2] - x[(size_t)i * 3 + 2]) * sc;
    atomicAdd(&y[(size_t)i * 3 + 0], (double)(K[0] * d0 + K[1] * d1 + K[2] * d2));
    atomicAdd(&y[(size_t)i * 3 + 1], (double)(K[3] * d0 + K[4] * d1 + K[5] * d2));
    atomicAdd(&y[(size_t)i * 3 + 2], (double)(K[6] * d0 + K[7] * d1 + K[8] * d2));
    atomicAdd(&y[(size_t)j * 3 + 0], -(double)(K[0] * d0 + K[3] * d1 + K[6] * d2));
    atomicAdd(&y[(size_t)j * 3 + 1], -(double)(K[1] * d0 + K[4] * d1 + K[7] * d2));
    atomicAdd(&y[(size_t)j * 3 + 2], -(double)(K[2] * d0 + K[5] * d1 + K[8] * d2));
}

template <Code CODE>
__global__ void csr_kernel(const ptrdiff_t n, const int64_t *__restrict__ rp, const int32_t *__restrict__ ci,
                           const void *__restrict__ codes, const float *__restrict__ scales, const float ms,
                           const double *__restrict__ x, double *__restrict__ y) {
    using T = Codes<CODE>;
    using code_t = typename T::code_t;
    const ptrdiff_t i = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double xi0 = x[(size_t)i * 3 + 0], xi1 = x[(size_t)i * 3 + 1], xi2 = x[(size_t)i * 3 + 2];
    float a0 = 0, a1 = 0, a2 = 0;
    for (int64_t k = rp[i]; k < rp[i + 1]; ++k) {
        const int32_t j = ci[k];
        if (j == i) continue;
        const code_t *const c = reinterpret_cast<const code_t *>(codes) + (size_t)k * 9;
        const float sc = CODE == Code::I8 ? scales[k] : 1.0f;
        const float d0 = float(x[(size_t)j * 3 + 0] - xi0) * sc, d1 = float(x[(size_t)j * 3 + 1] - xi1) * sc,
                    d2 = float(x[(size_t)j * 3 + 2] - xi2) * sc;
        a0 += T::value(c[0]) * d0 + T::value(c[1]) * d1 + T::value(c[2]) * d2;
        a1 += T::value(c[3]) * d0 + T::value(c[4]) * d1 + T::value(c[5]) * d2;
        a2 += T::value(c[6]) * d0 + T::value(c[7]) * d1 + T::value(c[8]) * d2;
    }
    y[(size_t)i * 3 + 0] = (double)(a0 * ms);
    y[(size_t)i * 3 + 1] = (double)(a1 * ms);
    y[(size_t)i * 3 + 2] = (double)(a2 * ms);
}

template <Code CODE, Form F, Cols COLS, Rec REC>
int launch(const Sell &s, const double *x, double *y, cudaStream_t stream) {
    if (F == Form::Upper) cudaMemsetAsync(y, 0, (size_t)s.n_rows * 3 * sizeof(double), stream);
    const ptrdiff_t threads = s.n_groups * C;
    sell_kernel<CODE, F, COLS, REC><<<(unsigned)((threads + THREADS - 1) / THREADS), THREADS, 0, stream>>>(s, x, y);
    if (COLS == Cols::Off16) {
        if (F == Form::Upper) {
            if (s.n_rem > 0)
                remainder_pairs_kernel<CODE><<<(unsigned)((s.n_rem + THREADS - 1) / THREADS), THREADS, 0, stream>>>(s, x, y);
        } else if (s.n_rem_rows > 0) {
            remainder_rows_kernel<CODE><<<(unsigned)((s.n_rem_rows + THREADS - 1) / THREADS), THREADS, 0, stream>>>(s, x, y);
        }
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

template <Code CODE, Form F, Cols COLS>
int by_rec(Rec rec, const Sell &s, const double *x, double *y, cudaStream_t st) {
    return rec == Rec::Scalar ? launch<CODE, F, COLS, Rec::Scalar>(s, x, y, st) : launch<CODE, F, COLS, Rec::Packed>(s, x, y, st);
}
template <Code CODE, Form F>
int by_cols(Cols cols, Rec rec, const Sell &s, const double *x, double *y, cudaStream_t st) {
    return cols == Cols::I32 ? by_rec<CODE, F, Cols::I32>(rec, s, x, y, st) : by_rec<CODE, F, Cols::Off16>(rec, s, x, y, st);
}
template <Code CODE>
int by_form(Form form, Cols cols, Rec rec, const Sell &s, const double *x, double *y, cudaStream_t st) {
    switch (form) {
        case Form::Full: return by_cols<CODE, Form::Full>(cols, rec, s, x, y, st);
        case Form::Upper: return by_cols<CODE, Form::Upper>(cols, rec, s, x, y, st);
        default: return by_cols<CODE, Form::UpperGather>(cols, rec, s, x, y, st);
    }
}

}  // namespace

int apply(Code code, Form form, Cols cols, Rec rec, const Sell &s, const double *x, double *y, void *stream) {
    // The gather reads the partner's record by index: packed records only (a scalar record would be
    // nine scattered single-byte loads).
    if (form == Form::UpperGather && rec != Rec::Packed) return 1;
    const cudaStream_t st = (cudaStream_t)stream;
    switch (code) {
        case Code::I8: return by_form<Code::I8>(form, cols, rec, s, x, y, st);
        case Code::F16: return by_form<Code::F16>(form, cols, rec, s, x, y, st);
        default: return by_form<Code::F32>(form, cols, rec, s, x, y, st);
    }
}

int apply_csr(Code code, ptrdiff_t n, const int64_t *rp, const int32_t *ci, const void *codes, const float *scales,
              float ms, const double *x, double *y, void *stream) {
    const unsigned blocks = (unsigned)((n + THREADS - 1) / THREADS);
    if (code == Code::I8)
        csr_kernel<Code::I8><<<blocks, THREADS, 0, (cudaStream_t)stream>>>(n, rp, ci, codes, scales, ms, x, y);
    else if (code == Code::F16)
        csr_kernel<Code::F16><<<blocks, THREADS, 0, (cudaStream_t)stream>>>(n, rp, ci, codes, scales, ms, x, y);
    else
        csr_kernel<Code::F32><<<blocks, THREADS, 0, (cudaStream_t)stream>>>(n, rp, ci, codes, scales, ms, x, y);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

}  // namespace qbsr
