// Quantized BSR on the GPU: the layouts the kernel-shape test (qbsr_shape.cpp) compares, behind a
// plain host interface (all pointers in the structures are device pointers).
//
// Every layout is SELL-32: 32 rows side by side, one warp, a step k of a group holding the k-th
// stored block of each of its rows.  The rows are sorted by length within windows of sigma rows.
//
//   Form   Full         every off-diagonal block, difference form y_i = sum_j K_ij (x_j - x_i); no
//                       scatter, deterministic
//          Upper        each pair i < j once: y_i += K d, y_j -= K^T d (atomicAdd), d = x_j - x_i
//          UpperGather  each pair once; a row also lists its lower neighbours, whose blocks it reads
//                       transposed from the partner's record (one more index per block); no scatter,
//                       deterministic
//   Cols   I32          column indices
//          Off16        offsets j - i (int16 for Full, uint16 for the upper forms); blocks beyond
//                       reach go to a remainder applied by a second kernel
//   Rec    Scalar       [step][entry][lane] codes, [step][lane] scales and columns: one load per
//                       entry
//          Packed       [step][lane] records holding codes, offset and scale: int8 16 bytes (one
//                       128-bit load), f16 20 bytes (five 32-bit loads)
//   Codes  int8 with one f32 scale per block, or f16 with one scale for the matrix; f32 values in
//          the same layouts, to tell the layout's share of a speed-up from the bytes'
#pragma once

#include <cstddef>
#include <cstdint>

namespace qbsr {

enum class Code { I8, F16, F32 };
enum class Form { Full, Upper, UpperGather };
enum class Cols { I32, Off16 };
enum class Rec { Scalar, Packed };

constexpr int C = 32;

// The packed int8 record: codes, the offset (Off16) and the block's scale in 16 bytes.
struct alignas(16) RecI8 {
    int8_t   codes[9];
    uint8_t  off_lo, off_hi, pad;
    float    scale;
};
// The packed f16 record: codes and the offset in 20 bytes.
struct alignas(4) RecF16 {
    uint16_t codes[9];
    uint16_t off;
};
// The packed f32 record: values and the offset in 40 bytes.
struct alignas(4) RecF32 {
    float    codes[9];
    uint32_t off;
};
static_assert(sizeof(RecI8) == 16 && sizeof(RecF16) == 20 && sizeof(RecF32) == 40, "record sizes");

struct Sell {
    ptrdiff_t n_rows = 0, n_groups = 0;
    const int32_t *rows = nullptr;       // [group][lane], -1 past the last row
    const int64_t *group_ptr = nullptr;  // upper (or full) steps of each group
    const void    *cols = nullptr;       // [step][lane]: int32, int16 or uint16; unused by packed Off16
    const void    *codes = nullptr;      // Scalar: [step][9][lane] codes; Packed: [step][lane] records
    const float   *scales = nullptr;     // Scalar int8: [step][lane]
    float          matrix_scale = 1;     // f16
    // UpperGather: the lower neighbours, [lstep][lane] column and record index (step * 32 + lane of
    // the partner's upper block; -1 padding).
    const int64_t *lgroup_ptr = nullptr;
    const int32_t *lcol = nullptr, *lrec = nullptr;
    // Remainder (Off16): the blocks beyond reach.  Full and UpperGather: by row, both orientations
    // ([rem_rowptr[r], rem_rowptr[r + 1]) of row rem_rows[r]); Upper: one entry per pair (rem_row
    // < rem_col).  Codes [entry][9], scales [entry] (int8).
    ptrdiff_t      n_rem = 0, n_rem_rows = 0;
    const int64_t *rem_rowptr = nullptr;
    const int32_t *rem_rows = nullptr, *rem_row = nullptr, *rem_col = nullptr;
    const void    *rem_codes = nullptr;
    const float   *rem_scales = nullptr;
    // Full and UpperGather: y += A x instead of y = A x.
    bool accumulate = false;
};

// y = A x (overwrites y), components interleaved, on `stream` (a cudaStream_t).  The upper form
// clears y first.  Returns 0 on success.
int apply(Code code, Form form, Cols cols, Rec rec, const Sell &s, const double *x, double *y, void *stream);

// The CSR kernel with one row per thread on the same quantized blocks (full difference form,
// int32 columns, [block][entry] codes): the layout without SELL.
int apply_csr(Code code, ptrdiff_t n_rows, const int64_t *rowptr, const int32_t *colidx, const void *codes,
              const float *scales, float matrix_scale, const double *x, double *y, void *stream);

}  // namespace qbsr
