#include "tet10_resample_field_V2.h"

#include "quadratures_rule.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tet10_resample_field.h"
#include "tet10_vec.h"
#include "tet10_weno_V.h"

// #define UNROLL_ZERO _Pragma("GCC unroll(0)")
#define UNROLL_ZERO _Pragma("unroll(1)")

//////////////////////////////////////////////////////////
// Casting from int64 to double SIMD vector
vec_real vec_indices_to_real(const vec_indices a) {
#if _VL_ == 8  //// 512 bits SIMD in double precision or 256 bits SIMD in single precision
    return (vec_real){(real_t)a[0],
                      (real_t)a[1],
                      (real_t)a[2],
                      (real_t)a[3],
                      (real_t)a[4],
                      (real_t)a[5],
                      (real_t)a[6],
                      (real_t)a[7]};

#elif _VL_ == 4  //// 256 bits SIMD in double precision or 128 bits SIMD in single precision

    return (vec_real){(real_t)a[0], (real_t)a[1], (real_t)a[2], (real_t)a[3]};

#elif _VL_ == 16  //// 512 bits SIMD in single precision

    return (vec_real){(real_t)a[0],
                      (real_t)a[1],
                      (real_t)a[2],
                      (real_t)a[3],
                      (real_t)a[4],
                      (real_t)a[5],
                      (real_t)a[6],
                      (real_t)a[7],
                      (real_t)a[8],
                      (real_t)a[9],
                      (real_t)a[10],
                      (real_t)a[11],
                      (real_t)a[12],
                      (real_t)a[13],
                      (real_t)a[14],
                      (real_t)a[15]};

#endif
}

#define ASSIGN_QUADRATURE_POINT_MACRO(_q, _qx_V, _qy_V, _qz_V, _qw_V) \
    {                                                                 \
        _qx_V = *((vec_real*)(&tet4_qx[q]));                          \
        _qy_V = *((vec_real*)(&tet4_qy[q]));                          \
        _qz_V = *((vec_real*)(&tet4_qz[q]));                          \
        _qw_V = *((vec_real*)(&tet4_qw[q]));                          \
    }

//////////////////////////////////////////////////////////
/// Macros for the cases of the SIMD implementation
#if _VL_ == 8  //// AVX512

#define ASSIGN_QUADRATURE_POINT_MACRO_TAIL(_q, _qx_V, _qy_V, _qz_V, _qw_V)       \
    {                                                                            \
        _qx_V = (vec_real){(_q + 0) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 0],  \
                           (_q + 1) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 1],  \
                           (_q + 2) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 2],  \
                           (_q + 3) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 3],  \
                           (_q + 4) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 4],  \
                           (_q + 5) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 5],  \
                           (_q + 6) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 6],  \
                           (_q + 7) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 7]}; \
                                                                                 \
        _qy_V = (vec_real){(_q + 0) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 0],  \
                           (_q + 1) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 1],  \
                           (_q + 2) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 2],  \
                           (_q + 3) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 3],  \
                           (_q + 4) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 4],  \
                           (_q + 5) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 5],  \
                           (_q + 6) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 6],  \
                           (_q + 7) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 7]}; \
                                                                                 \
        _qz_V = (vec_real){(_q + 0) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 0],  \
                           (_q + 1) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 1],  \
                           (_q + 2) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 2],  \
                           (_q + 3) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 3],  \
                           (_q + 4) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 4],  \
                           (_q + 5) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 5],  \
                           (_q + 6) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 6],  \
                           (_q + 7) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 7]}; \
                                                                                 \
        _qw_V = (vec_real){(_q + 0) >= TET4_NQP ? 0.0 : tet4_qw[_q + 0],         \
                           (_q + 1) >= TET4_NQP ? 0.0 : tet4_qw[_q + 1],         \
                           (_q + 2) >= TET4_NQP ? 0.0 : tet4_qw[_q + 2],         \
                           (_q + 3) >= TET4_NQP ? 0.0 : tet4_qw[_q + 3],         \
                           (_q + 4) >= TET4_NQP ? 0.0 : tet4_qw[_q + 4],         \
                           (_q + 5) >= TET4_NQP ? 0.0 : tet4_qw[_q + 5],         \
                           (_q + 6) >= TET4_NQP ? 0.0 : tet4_qw[_q + 6],         \
                           (_q + 7) >= TET4_NQP ? 0.0 : tet4_qw[_q + 7]};        \
    }

#elif _VL_ == 4  //// AVX512

#define ASSIGN_QUADRATURE_POINT_MACRO_TAIL(_q, _qx_V, _qy_V, _qz_V, _qw_V)       \
    {                                                                            \
        _qx_V = (vec_real){(_q + 0) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 0],  \
                           (_q + 1) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 1],  \
                           (_q + 2) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 2],  \
                           (_q + 3) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 3]}; \
                                                                                 \
        _qy_V = (vec_real){(_q + 0) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 0],  \
                           (_q + 1) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 1],  \
                           (_q + 2) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 2],  \
                           (_q + 3) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 3]}; \
                                                                                 \
        _qz_V = (vec_real){(_q + 0) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 0],  \
                           (_q + 1) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 1],  \
                           (_q + 2) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 2],  \
                           (_q + 3) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 3]}; \
                                                                                 \
        _qw_V = (vec_real){(_q + 0) >= TET4_NQP ? 0.0 : tet4_qw[_q + 0],         \
                           (_q + 1) >= TET4_NQP ? 0.0 : tet4_qw[_q + 1],         \
                           (_q + 2) >= TET4_NQP ? 0.0 : tet4_qw[_q + 2],         \
                           (_q + 3) >= TET4_NQP ? 0.0 : tet4_qw[_q + 3]};        \
    }

#elif _VL_ == 16

#define _qx_V                                                         \
    (vec_real){(_q + 0) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 0],   \
               (_q + 1) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 1],   \
               (_q + 2) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 2],   \
               (_q + 3) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 3],   \
               (_q + 4) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 4],   \
               (_q + 5) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 5],   \
               (_q + 6) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 6],   \
               (_q + 7) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 7],   \
               (_q + 8) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 8],   \
               (_q + 9) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 9],   \
               (_q + 10) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 10], \
               (_q + 11) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 11], \
               (_q + 12) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 12], \
               (_q + 13) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 13], \
               (_q + 14) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 14], \
               (_q + 15) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 15]};

#define _qy_V                                                         \
    (vec_real){(_q + 0) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 0],   \
               (_q + 1) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 1],   \
               (_q + 2) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 2],   \
               (_q + 3) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 3],   \
               (_q + 4) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 4],   \
               (_q + 5) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 5],   \
               (_q + 6) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 6],   \
               (_q + 7) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 7],   \
               (_q + 8) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 8],   \
               (_q + 9) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 9],   \
               (_q + 10) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 10], \
               (_q + 11) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 11], \
               (_q + 12) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 12], \
               (_q + 13) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 13], \
               (_q + 14) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 14], \
               (_q + 15) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 15]};

#define _qz_V                                                         \
    (vec_real){(_q + 0) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 0],   \
               (_q + 1) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 1],   \
               (_q + 2) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 2],   \
               (_q + 3) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 3],   \
               (_q + 4) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 4],   \
               (_q + 5) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 5],   \
               (_q + 6) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 6],   \
               (_q + 7) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 7],   \
               (_q + 8) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 8],   \
               (_q + 9) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 9],   \
               (_q + 10) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 10], \
               (_q + 11) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 11], \
               (_q + 12) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 12], \
               (_q + 13) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 13], \
               (_q + 14) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 14], \
               (_q + 15) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 15]};

#define _qw_V                                                  \
    (vec_real){(_q + 0) >= TET4_NQP ? 0.0 : tet4_qw[_q + 0],   \
               (_q + 1) >= TET4_NQP ? 0.0 : tet4_qw[_q + 1],   \
               (_q + 2) >= TET4_NQP ? 0.0 : tet4_qw[_q + 2],   \
               (_q + 3) >= TET4_NQP ? 0.0 : tet4_qw[_q + 3],   \
               (_q + 4) >= TET4_NQP ? 0.0 : tet4_qw[_q + 4],   \
               (_q + 5) >= TET4_NQP ? 0.0 : tet4_qw[_q + 5],   \
               (_q + 6) >= TET4_NQP ? 0.0 : tet4_qw[_q + 6],   \
               (_q + 7) >= TET4_NQP ? 0.0 : tet4_qw[_q + 7],   \
               (_q + 8) >= TET4_NQP ? 0.0 : tet4_qw[_q + 8],   \
               (_q + 9) >= TET4_NQP ? 0.0 : tet4_qw[_q + 9],   \
               (_q + 10) >= TET4_NQP ? 0.0 : tet4_qw[_q + 10], \
               (_q + 11) >= TET4_NQP ? 0.0 : tet4_qw[_q + 11], \
               (_q + 12) >= TET4_NQP ? 0.0 : tet4_qw[_q + 12], \
               (_q + 13) >= TET4_NQP ? 0.0 : tet4_qw[_q + 13], \
               (_q + 14) >= TET4_NQP ? 0.0 : tet4_qw[_q + 14], \
               (_q + 15) >= TET4_NQP ? 0.0 : tet4_qw[_q + 15]};

#define ASSIGN_QUADRATURE_POINT_MACRO_TAIL(_q, _qx_V, _qy_V, _qz_V, _qw_V)         \
    {                                                                              \
        _qx_V = (vec_real){(_q + 0) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 0],    \
                           (_q + 1) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 1],    \
                           (_q + 2) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 2],    \
                           (_q + 3) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 3],    \
                           (_q + 4) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 4],    \
                           (_q + 5) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 5],    \
                           (_q + 6) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 6],    \
                           (_q + 7) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 7],    \
                           (_q + 8) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 8],    \
                           (_q + 9) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 9],    \
                           (_q + 10) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 10],  \
                           (_q + 11) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 11],  \
                           (_q + 12) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 12],  \
                           (_q + 13) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 13],  \
                           (_q + 14) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 14],  \
                           (_q + 15) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 15]}; \
                                                                                   \
        _qy_V = (vec_real){(_q + 0) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 0],    \
                           (_q + 1) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 1],    \
                           (_q + 2) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 2],    \
                           (_q + 3) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 3],    \
                           (_q + 4) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 4],    \
                           (_q + 5) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 5],    \
                           (_q + 6) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 6],    \
                           (_q + 7) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 7],    \
                           (_q + 8) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 8],    \
                           (_q + 9) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 9],    \
                           (_q + 10) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 10],  \
                           (_q + 11) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 11],  \
                           (_q + 12) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 12],  \
                           (_q + 13) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 13],  \
                           (_q + 14) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 14],  \
                           (_q + 15) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 15]}; \
                                                                                   \
        _qz_V = (vec_real){(_q + 0) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 0],    \
                           (_q + 1) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 1],    \
                           (_q + 2) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 2],    \
                           (_q + 3) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 3],    \
                           (_q + 4) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 4],    \
                           (_q + 5) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 5],    \
                           (_q + 6) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 6],    \
                           (_q + 7) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 7],    \
                           (_q + 8) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 8],    \
                           (_q + 9) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 9],    \
                           (_q + 10) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 10],  \
                           (_q + 11) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 11],  \
                           (_q + 12) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 12],  \
                           (_q + 13) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 13],  \
                           (_q + 14) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 14],  \
                           (_q + 15) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 15]}; \
                                                                                   \
        _qw_V = (vec_real){(_q + 0) >= TET4_NQP ? 0.0 : tet4_qw[_q + 0],           \
                           (_q + 1) >= TET4_NQP ? 0.0 : tet4_qw[_q + 1],           \
                           (_q + 2) >= TET4_NQP ? 0.0 : tet4_qw[_q + 2],           \
                           (_q + 3) >= TET4_NQP ? 0.0 : tet4_qw[_q + 3],           \
                           (_q + 4) >= TET4_NQP ? 0.0 : tet4_qw[_q + 4],           \
                           (_q + 5) >= TET4_NQP ? 0.0 : tet4_qw[_q + 5],           \
                           (_q + 6) >= TET4_NQP ? 0.0 : tet4_qw[_q + 6],           \
                           (_q + 7) >= TET4_NQP ? 0.0 : tet4_qw[_q + 7],           \
                           (_q + 8) >= TET4_NQP ? 0.0 : tet4_qw[_q + 8],           \
                           (_q + 9) >= TET4_NQP ? 0.0 : tet4_qw[_q + 9],           \
                           (_q + 10) >= TET4_NQP ? 0.0 : tet4_qw[_q + 10],         \
                           (_q + 11) >= TET4_NQP ? 0.0 : tet4_qw[_q + 11],         \
                           (_q + 12) >= TET4_NQP ? 0.0 : tet4_qw[_q + 12],         \
                           (_q + 13) >= TET4_NQP ? 0.0 : tet4_qw[_q + 13],         \
                           (_q + 14) >= TET4_NQP ? 0.0 : tet4_qw[_q + 14],         \
                           (_q + 15) >= TET4_NQP ? 0.0 : tet4_qw[_q + 15]};        \
    }

#endif  //// end SIMD implementation

void assert_vec_real(const vec_real a, const vec_real b) {
    for (int i = 0; i < _VL_; i++) {
        if (a[i] == b[i]) {
            printf("Error: %f == %f\n", a[i], b[i]);
            exit(1);
        }
    }
}

//////////////////////////////////////////////////////////
/// Macros for the cases of the SIMD implementation
#if _VL_ == 8  //// AVX512

//// ZEROS for AVX512
#define ZEROS_VEC \
    (vec_real) { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }

//// SIMD_REDUCE_SUM for AVX512
#define SIMD_REDUCE_SUM_MACRO(_out, _in) \
    { _out = _in[0] + _in[1] + _in[2] + _in[3] + _in[4] + _in[5] + _in[6] + _in[7]; }

//// SIMD floor for AVX512
vec_indices floor_V(vec_real a) {
    // vec_indices r = (vec_indices){0, 0, 0, 0, 0, 0, 0, 0};
    // for(int ii = 0; ii < _VL_; ii++) {
    //     r[ii] = floor(a[ii]);
    // }

    // return r;
    return (vec_indices){(ptrdiff_t)a[0],
                         (ptrdiff_t)a[1],
                         (ptrdiff_t)a[2],
                         (ptrdiff_t)a[3],
                         (ptrdiff_t)a[4],
                         (ptrdiff_t)a[5],
                         (ptrdiff_t)a[6],
                         (ptrdiff_t)a[7]};
}

#elif _VL_ == 4  //// AVX2

//// ZEROS for AVX2
#define ZEROS_VEC \
    (vec_real) { 0.0, 0.0, 0.0, 0.0 }

//// SIMD_REDUCE_SUM for AVX2
#define SIMD_REDUCE_SUM_MACRO(_out, _in) \
    { _out = _in[0] + _in[1] + _in[2] + _in[3]; }

//// SIMD floor for AVX2
vec_indices floor_V(vec_real a) {
    return (vec_indices){(ptrdiff_t)a[0], (ptrdiff_t)a[1], (ptrdiff_t)a[2], (ptrdiff_t)a[3]};
}

#elif _VL_ == 16

//// ZEROS for AVX512
#define ZEROS_VEC \
    (vec_real) { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }

//// SIMD_REDUCE_SUM for AVX512
#define SIMD_REDUCE_SUM_MACRO(_out, _in)                                                        \
    {                                                                                           \
        _out = _in[0] + _in[1] + _in[2] + _in[3] + _in[4] + _in[5] + _in[6] + _in[7] + _in[8] + \
               _in[9] + _in[10] + _in[11] + _in[12] + _in[13] + _in[14] + _in[15];              \
    }

vec_indices floor_V(vec_real a) {
    return (vec_indices){(ptrdiff_t)a[0],
                         (ptrdiff_t)a[1],
                         (ptrdiff_t)a[2],
                         (ptrdiff_t)a[3],
                         (ptrdiff_t)a[4],
                         (ptrdiff_t)a[5],
                         (ptrdiff_t)a[6],
                         (ptrdiff_t)a[7],
                         (ptrdiff_t)a[8],
                         (ptrdiff_t)a[9],
                         (ptrdiff_t)a[10],
                         (ptrdiff_t)a[11],
                         (ptrdiff_t)a[12],
                         (ptrdiff_t)a[13],
                         (ptrdiff_t)a[14],
                         (ptrdiff_t)a[15]};
}

#endif  //// end SIMD implementation

///////////////////////////////////////////////////////////////////////
// isoparametric_lumped_mass_kernel_hrt
///////////////////////////////////////////////////////////////////////
SFEM_INLINE static void isoparametric_lumped_mass_kernel_hrt_V(
        const vec_real dV,
        // Quadrature
        const vec_real qx, const vec_real qy, const vec_real qz,
        vec_real* const SFEM_RESTRICT element_diag) {
    //
    const vec_real x0 = 4 * qx;
    const vec_real x1 = qy * qz;
    const vec_real x2 = 2 * qx - 1;
    const vec_real x3 = qx * x2;
    const vec_real x4 = 2 * qy;
    const vec_real x5 = qy * (x4 - 1);
    const vec_real x6 = 2 * qz;
    const vec_real x7 = qz * (x6 - 1);
    const vec_real x8 = qx + qy + qz - 1;
    const vec_real x9 = qx * x8;
    const vec_real x10 = qy * x8;
    const vec_real x11 = qz * x8;
    const vec_real x12 = x8 * (x2 + x4 + x6);
    const vec_real x13 = (real_t)(5.0 / 18.0) * dV;
    const vec_real x14 = POW2_VEC(qx);
    const vec_real x15 = POW2_VEC(qy);
    const vec_real x16 = POW2_VEC(qz);
    const vec_real x17 = 11 * qx;
    const vec_real x18 = -qy * x17 + 11 * x11 + x12 + 4 * x7;
    const vec_real x19 = -qz * x17 + 11 * x10 + 4 * x5;
    const vec_real x20 = (real_t)(40.0 / 27.0) * dV;
    const vec_real x21 = qx * qy;
    const vec_real x22 = -24 * qx + 21 * x14 + 4;
    const vec_real x23 = -24 * qy + 21 * x15;
    const vec_real x24 = qx * qz;
    const vec_real x25 = -11 * x1 + 4 * x3 + 11 * x9;
    const vec_real x26 = -24 * qz + 21 * x16;

    element_diag[0] +=
            x12 * x13 * (qy * x0 + qz * x0 + 4 * x1 - x10 - x11 + 10 * x12 + x3 + x5 + x7 - x9);
    element_diag[1] += qx * x13 * (42 * (POW3_VEC(qx)) + 14 * qx - 45 * x14 - 1);
    element_diag[2] += qy * x13 * (42 * (POW3_VEC(qy)) + 14 * qy - 45 * x15 - 1);
    element_diag[3] += qz * x13 * (42 * (POW3_VEC(qz)) + 14 * qz - 45 * x16 - 1);
    element_diag[4] += -x20 * x9 * (16 * x1 + x18 + x19 + x3 - 46 * x9);
    element_diag[5] += x20 * x21 * (84 * x21 + x22 + x23);
    element_diag[6] += -x10 * x20 * (-46 * x10 + x18 + 16 * x24 + x25 + x5);
    element_diag[7] += -x11 * x20 * (-46 * x11 + x12 + x19 + 16 * x21 + x25 + x7);
    element_diag[8] += x20 * x24 * (x22 + 84 * x24 + x26);
    element_diag[9] += x1 * x20 * (84 * x1 + x23 + x26 + 4);
}

/**
 * @brief iso-parametric version
 *
 * @param x
 * @param y
 * @param z
 * @param qx
 * @param qy
 * @param qz
 * @return SFEM_INLINE
 */
SFEM_INLINE static vec_real tet10_measure_V(const real_t* const SFEM_RESTRICT x,
                                            const real_t* const SFEM_RESTRICT y,
                                            const real_t* const SFEM_RESTRICT z,
                                            // Quadrature point
                                            const vec_real qx, const vec_real qy,
                                            const vec_real qz) {
    const vec_real x0 = 4 * qz;
    const vec_real x1 = x0 - 1;
    const vec_real x2 = 4 * qy;
    const vec_real x3 = 4 * qx;
    const vec_real x4 = x3 - 4;
    const vec_real x5 = -8 * qz - x2 - x4;
    const vec_real x6 = -x3 * y[4];
    const vec_real x7 = x0 + x2;
    const vec_real x8 = x3 + x7 - 3;
    const vec_real x9 = x8 * y[0];
    const vec_real x10 = -x2 * y[6] + x9;
    const vec_real x11 = x1 * y[3] + x10 + x2 * y[9] + x3 * y[8] + x5 * y[7] + x6;
    const vec_real x12 = -x2 * z[6];
    const vec_real x13 = -x0 * z[7];
    const vec_real x14 = x3 - 1;
    const vec_real x15 = x8 * z[0];
    const vec_real x16 = -8 * qx - x7 + 4;
    const vec_real x17 = x0 * z[8] + x12 + x13 + x14 * z[1] + x15 + x16 * z[4] + x2 * z[5];
    const vec_real x18 = x2 - 1;
    const vec_real x19 = -8 * qy - x0 - x4;
    const vec_real x20 = -x3 * x[4];
    const vec_real x21 = x8 * x[0];
    const vec_real x22 = -x0 * x[7] + x21;
    const real_t sixth = 1.0 / 6.0;
    const vec_real x23 = sixth * x0 * x[9] + sixth * x18 * x[2] + sixth * x19 * x[6] + sixth * x20 +
                         sixth * x22 + sixth * x3 * x[5];
    const vec_real x24 = -x0 * y[7];
    const vec_real x25 = x0 * y[8] + x10 + x14 * y[1] + x16 * y[4] + x2 * y[5] + x24;
    const vec_real x26 = x15 - x3 * z[4];
    const vec_real x27 = x1 * z[3] + x12 + x2 * z[9] + x26 + x3 * z[8] + x5 * z[7];
    const vec_real x28 = x0 * y[9] + x18 * y[2] + x19 * y[6] + x24 + x3 * y[5] + x6 + x9;
    const vec_real x29 = -x2 * x[6];
    const vec_real x30 = sixth * x1 * x[3] + sixth * x2 * x[9] + sixth * x20 + sixth * x21 +
                         sixth * x29 + sixth * x3 * x[8] + sixth * x5 * x[7];
    const vec_real x31 = x0 * z[9] + x13 + x18 * z[2] + x19 * z[6] + x26 + x3 * z[5];
    const vec_real x32 = sixth * x0 * x[8] + sixth * x14 * x[1] + sixth * x16 * x[4] +
                         sixth * x2 * x[5] + sixth * x22 + sixth * x29;

    return x11 * x17 * x23 - x11 * x31 * x32 - x17 * x28 * x30 - x23 * x25 * x27 + x25 * x30 * x31 +
           x27 * x28 * x32;
}

SFEM_INLINE static void tet10_transform_V(const real_t* const SFEM_RESTRICT x,
                                          const real_t* const SFEM_RESTRICT y,
                                          const real_t* const SFEM_RESTRICT z,
                                          // Quadrature point
                                          const vec_real qx, const vec_real qy, const vec_real qz,
                                          // Output
                                          vec_real* const SFEM_RESTRICT out_x,
                                          vec_real* const SFEM_RESTRICT out_y,
                                          vec_real* const SFEM_RESTRICT out_z) {
    const vec_real x0 = 4.0 * qx;
    const vec_real x1 = qy * x0;
    const vec_real x2 = qz * x0;
    const vec_real x3 = 4.0 * qy;
    const vec_real x4 = qz * x3;
    const vec_real x5 = 2.0 * qx - 1.0;
    const vec_real x6 = qx * x5;
    const vec_real x7 = 2.0 * qy;
    const vec_real x8 = qy * (x7 - 1.0);
    const vec_real x9 = 2.0 * qz;
    const vec_real x10 = qz * (x9 - 1.0);
    const vec_real x11 = -4 * qz - x0 - x3 + 4.0;
    const vec_real x12 = qx * x11;
    const vec_real x13 = qy * x11;
    const vec_real x14 = qz * x11;
    const vec_real x15 = (-x5 - x7 - x9) * (-qx - qy - qz + 1.0);

    *out_x = x[0] * x15 + x[1] * x6 + x[2] * x8 + x[3] * x10 + x[4] * x12 + x[5] * x1 + x[6] * x13 +
             x[7] * x14 + x[8] * x2 + x[9] * x4;
    *out_y = y[0] * x15 + y[1] * x6 + y[2] * x8 + y[3] * x10 + y[4] * x12 + y[5] * x1 + y[6] * x13 +
             y[7] * x14 + y[8] * x2 + y[9] * x4;
    *out_z = z[0] * x15 + z[1] * x6 + z[2] * x8 + z[3] * x10 + z[4] * x12 + z[5] * x1 + z[6] * x13 +
             z[7] * x14 + z[8] * x2 + z[9] * x4;
}

SFEM_INLINE static void tet10_dual_basis_hrt_V(const vec_real qx, const vec_real qy,
                                               const vec_real qz, vec_real* const f) {
    static const real_t c5_18 = 5.0 / 18.0;
    static const real_t c5_72 = 5.0 / 72.0;
    static const real_t c10_9 = 10.0 / 9.0;
    static const real_t c25_9 = 25.0 / 9.0;
    static const real_t c40_27 = 40.0 / 27.0;
    static const real_t c115_27 = 115.0 / 27.0;
    static const real_t c110_27 = 110.0 / 27.0;
    static const real_t c55_54 = 55.0 / 54.0;
    static const real_t c10_27 = 10.0 / 27.0;
    static const real_t c160_27 = 160.0 / 27.0;
    static const real_t c460_27 = 460.0 / 27.0;

    const vec_real x0 = 2 * qy;
    const vec_real x1 = 2 * qz;
    const vec_real x2 = 2 * qx - 1;
    const vec_real x3 = (-x0 - x1 - x2) * (-qx - qy - qz + 1);
    const vec_real x4 = x0 - 1;
    const vec_real x5 = c5_18 * qy;
    const vec_real x6 = x4 * x5;
    const vec_real x7 = x1 - 1;
    const vec_real x8 = c5_18 * qz;
    const vec_real x9 = x7 * x8;
    const vec_real x10 = -4 * qx - 4 * qy - 4 * qz + 4;
    const vec_real x11 = c5_72 * x10;
    const vec_real x12 = qy * qz;
    const vec_real x13 = qx * x11 + c10_9 * x12 + x6 + x9;
    const vec_real x14 = c5_18 * qx;
    const vec_real x15 = x14 * x2;
    const vec_real x16 = c10_9 * qx;
    const vec_real x17 = qy * x11 + qz * x16 + x15;
    const vec_real x18 = qy * x16 + qz * x11;
    const vec_real x19 = qx * x2;
    const vec_real x20 = c5_18 * x3;
    const vec_real x21 = qy * x14 + x10 * x8 + x20;
    const vec_real x22 = qz * x14 + x10 * x5;
    const vec_real x23 = qy * x4;
    const vec_real x24 = qz * x5 + x10 * x14;
    const vec_real x25 = qz * x7;
    const vec_real x26 = c40_27 * x23;
    const vec_real x27 = c115_27 * x10;
    const vec_real x28 = c110_27 * qx;
    const vec_real x29 = -qz * x28;
    const vec_real x30 = c55_54 * x10;
    const vec_real x31 = -qy * x30;
    const vec_real x32 = c10_27 * x19;
    const vec_real x33 = c40_27 * x25;
    const vec_real x34 = x29 + x31 + x32 + x33;
    const vec_real x35 = -qy * x28;
    const vec_real x36 = -qz * x30;
    const vec_real x37 = c10_27 * x3;
    const vec_real x38 = x35 + x36 + x37;
    const vec_real x39 = c40_27 * x10;
    const vec_real x40 = qx * qy;
    const vec_real x41 = -qx * x30 - c110_27 * x12;
    const vec_real x42 = c10_27 * x23;
    const vec_real x43 = c40_27 * x3;
    const vec_real x44 = x42 + x43;
    const vec_real x45 = qx * qz;
    const vec_real x46 = c40_27 * x19;
    const vec_real x47 = x41 + x46;
    const vec_real x48 = c10_27 * x25;
    const vec_real x49 = x26 + x48;
    const vec_real x50 = x29 + x31;
    const vec_real x51 = x35 + x36;

    f[0] = x13 + x17 + x18 + c25_9 * x3;
    f[1] = x13 + c25_9 * x19 + x21 + x22;
    f[2] = x17 + x21 + c25_9 * x23 + x24 + x9;
    f[3] = x15 + x18 + x20 + x22 + x24 + c25_9 * x25 + x6;
    f[4] = qx * x27 + c160_27 * x12 + x26 + x34 + x38;
    f[5] = qz * x39 + x34 + c460_27 * x40 + x41 + x44;
    f[6] = qy * x27 + x33 + x38 + x42 + c160_27 * x45 + x47;
    f[7] = qz * x27 + x37 + c160_27 * x40 + x47 + x49 + x50;
    f[8] = qy * x39 + x32 + x41 + x43 + c460_27 * x45 + x49 + x51;
    f[9] = qx * x39 + c460_27 * x12 + x44 + x46 + x48 + x50 + x51;
}

/**
 * @brief
 *
 * @param x
 * @param y
 * @param z
 * @param f
 * @return SFEM_INLINE
 */
SFEM_INLINE static void hex_aa_8_eval_fun_V(
        // Quadrature point (local coordinates)
        // With respect to the hat functions of a cube element
        // In a local coordinate system
        const vec_real x, const vec_real y, const vec_real z,
        // Output
        vec_real* const SFEM_RESTRICT f) {
    //
    f[0] = (1.0 - x) * (1.0 - y) * (1.0 - z);
    f[1] = x * (1.0 - y) * (1.0 - z);
    f[2] = x * y * (1.0 - z);
    f[3] = (1.0 - x) * y * (1.0 - z);
    f[4] = (1.0 - x) * (1.0 - y) * z;
    f[5] = x * (1.0 - y) * z;
    f[6] = x * y * z;
    f[7] = (1.0 - x) * y * z;
}

//////////////////////////////////////////////////////////
/// Macros for the data collection
#if _VL_ == 8

#define GET_DATA_MACRO(_out, _data, _indx_V)  \
    {                                         \
        _out = (vec_real){_data[_indx_V[0]],  \
                          _data[_indx_V[1]],  \
                          _data[_indx_V[2]],  \
                          _data[_indx_V[3]],  \
                          _data[_indx_V[4]],  \
                          _data[_indx_V[5]],  \
                          _data[_indx_V[6]],  \
                          _data[_indx_V[7]]}; \
    }

#elif _VL_ == 4

#define GET_DATA_MACRO(_out, _data, _indx_V)                                                 \
    {                                                                                        \
        _out = (vec_real){                                                                   \
                _data[_indx_V[0]], _data[_indx_V[1]], _data[_indx_V[2]], _data[_indx_V[3]]}; \
    }

#elif _VL_ == 16

#define GET_DATA_MACRO(_out, _data, _indx_V)   \
    {                                          \
        _out = (vec_real){_data[_indx_V[0]],   \
                          _data[_indx_V[1]],   \
                          _data[_indx_V[2]],   \
                          _data[_indx_V[3]],   \
                          _data[_indx_V[4]],   \
                          _data[_indx_V[5]],   \
                          _data[_indx_V[6]],   \
                          _data[_indx_V[7]],   \
                          _data[_indx_V[8]],   \
                          _data[_indx_V[9]],   \
                          _data[_indx_V[10]],  \
                          _data[_indx_V[11]],  \
                          _data[_indx_V[12]],  \
                          _data[_indx_V[13]],  \
                          _data[_indx_V[14]],  \
                          _data[_indx_V[15]]}; \
    }

#endif

/**
 * @brief Pick the data from the
 *
 * @param stride0
 * @param stride1
 * @param stride2
 * @param i
 * @param j
 * @param k
 * @param data
 * @param out
 */
SFEM_INLINE static void hex_aa_8_collect_coeffs_V(
        const ptrdiff_t stride0, const ptrdiff_t stride1, const ptrdiff_t stride2,

        const vec_indices i, const vec_indices j, const vec_indices k,

        // Attention this is geometric data transformed to solver data!
        const real_t* const SFEM_RESTRICT data, vec_real* const SFEM_RESTRICT out) {
    //

    ptrdiff_t_sfem stride0_sfem = (ptrdiff_t_sfem)stride0;
    ptrdiff_t_sfem stride1_sfem = (ptrdiff_t_sfem)stride1;
    ptrdiff_t_sfem stride2_sfem = (ptrdiff_t_sfem)stride2;

    const vec_indices i0 = i * stride0_sfem + j * stride1_sfem + k * stride2_sfem;
    const vec_indices i1 = (i + 1) * stride0_sfem + j * stride1_sfem + k * stride2_sfem;
    const vec_indices i2 = (i + 1) * stride0_sfem + (j + 1) * stride1_sfem + k * stride2_sfem;
    const vec_indices i3 = i * stride0_sfem + (j + 1) * stride1_sfem + k * stride2_sfem;
    const vec_indices i4 = i * stride0_sfem + j * stride1_sfem + (k + 1) * stride2_sfem;
    const vec_indices i5 = (i + 1) * stride0_sfem + j * stride1_sfem + (k + 1) * stride2_sfem;
    const vec_indices i6 = (i + 1) * stride0_sfem + (j + 1) * stride1_sfem + (k + 1) * stride2_sfem;
    const vec_indices i7 = i * stride0_sfem + (j + 1) * stride1_sfem + (k + 1) * stride2_sfem;

    GET_DATA_MACRO(out[0], data, i0);
    GET_DATA_MACRO(out[1], data, i1);
    GET_DATA_MACRO(out[2], data, i2);
    GET_DATA_MACRO(out[3], data, i3);
    GET_DATA_MACRO(out[4], data, i4);
    GET_DATA_MACRO(out[5], data, i5);
    GET_DATA_MACRO(out[6], data, i6);
    GET_DATA_MACRO(out[7], data, i7);
}

int vec_real_check_nan(const vec_real a) {
    for (int i = 0; i < _VL_; i++) {
        if (isnan(a[i])) {
            return 1;
        }
    }
    return 0;
}

/**
 * @brief
 *
 * @param nelements
 * @param nnodes
 * @param elems
 * @param xyz
 * @param n
 * @param stride
 * @param origin
 * @param delta
 * @param data
 * @param weighted_field
 * @return int
 */
int hex8_to_isoparametric_tet10_resample_field_local_V(
        // Mesh
        const ptrdiff_t nelements,          // number of elements
        const ptrdiff_t nnodes,             // number of nodes
        idx_t** const SFEM_RESTRICT elems,  // connectivity
        geom_t** const SFEM_RESTRICT xyz,   // coordinates
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n,       // number of nodes in each direction
        const ptrdiff_t* const SFEM_RESTRICT stride,  // stride of the data
        const geom_t* const SFEM_RESTRICT origin,     // origin of the domain
        const geom_t* const SFEM_RESTRICT delta,      // delta of the domain
        const real_t* const SFEM_RESTRICT data,       // SDF
        // Output
        real_t* const SFEM_RESTRICT weighted_field) {
    //
    PRINT_CURRENT_FUNCTION;

    printf("============================================================\n");
    printf("Start: hex8_to_isoparametric_tet10_resample_field_local_V\n");
    printf("============================================================\n");

    const real_t ox = (real_t)origin[0];
    const real_t oy = (real_t)origin[1];
    const real_t oz = (real_t)origin[2];

    const real_t dx = (real_t)delta[0];
    const real_t dy = (real_t)delta[1];
    const real_t dz = (real_t)delta[2];

    const ptrdiff_t stride0 = stride[0];
    const ptrdiff_t stride1 = stride[1];
    const ptrdiff_t stride2 = stride[2];

    // #pragma omp parallel
    //     {
    // #pragma omp for  // nowait
    /// Loop over the elements of the mesh

    for (ptrdiff_t i = 0; i < nelements; ++i) {
        // printf("element = %d\n", i);

        idx_t ev[10];

        // ISOPARAMETRIC
        real_t x[10], y[10], z[10];

        vec_real hex8_f[8];
        vec_real coeffs[8];

        vec_real tet10_f[10];
        vec_real element_field[10];

        // loop over the 4 vertices of the tetrahedron
        UNROLL_ZERO
        for (int v = 0; v < 10; ++v) {
            ev[v] = elems[v][i];
        }

        // ISOPARAMETRIC
        for (int v = 0; v < 10; ++v) {
            x[v] = (real_t)(xyz[0][ev[v]]);  // x-coordinates
            y[v] = (real_t)(xyz[1][ev[v]]);  // y-coordinates
            z[v] = (real_t)(xyz[2][ev[v]]);  // z-coordinates
        }

        // memset(element_field, 0, 10 * sizeof(real_t));

        // set to zero the element field
        for (int v = 0; v < 10; v++) {
            element_field[v] = (vec_real)ZEROS_VEC;
        }

        // SUBPARAMETRIC (for iso-parametric tassellation of tet10 might be necessary)
        for (int q = 0; q < TET4_NQP; q += (_VL_)) {  // loop over the quadrature points

            vec_real tet4_qx_V, tet4_qy_V, tet4_qz_V, tet4_qw_V;

            const int q_next = q + _VL_;
            // printf("q + % d,  qq = %d\n", q, qq);

            if (q_next < TET4_NQP) {
                ASSIGN_QUADRATURE_POINT_MACRO(q, tet4_qx_V, tet4_qy_V, tet4_qz_V, tet4_qw_V);

            } else {
                ASSIGN_QUADRATURE_POINT_MACRO_TAIL(q, tet4_qx_V, tet4_qy_V, tet4_qz_V, tet4_qw_V);

                // printf("tail\n");
                // for (int ii = 0; ii < _VL_; ii++) {
                //     printf("tet4_qw_V[%d] = %f\n", ii, tet4_qw_V[ii]);
                // }

                // if (vec_real_check_nan(tet4_qx_V) || vec_real_check_nan(tet4_qy_V) ||
                //     vec_real_check_nan(tet4_qz_V) || vec_real_check_nan(tet4_qw_V)) {
                //     printf("ERROR: tet4_qx_V file: %s:%d\n", __FILE__, __LINE__);
                //     exit(1);
                // }
            }

            const vec_real measure = tet10_measure_V(x,  //
                                                     y,
                                                     z,
                                                     tet4_qx_V,
                                                     tet4_qy_V,
                                                     tet4_qz_V);

            const vec_real dV = measure * tet4_qw_V;

            vec_real g_qx, g_qy, g_qz;
            // Transform quadrature point to physical space
            // g_qx, g_qy, g_qz are the coordinates of the quadrature point in the physical
            // space
            tet10_transform_V(x, y, z, tet4_qx_V, tet4_qy_V, tet4_qz_V, &g_qx, &g_qy, &g_qz);
            tet10_dual_basis_hrt_V(tet4_qx_V, tet4_qy_V, tet4_qz_V, tet10_f);

            ///// ======================================================

            const vec_real grid_x = (g_qx - ox) / dx;
            const vec_real grid_y = (g_qy - oy) / dy;
            const vec_real grid_z = (g_qz - oz) / dz;

            const vec_indices i = floor_V(grid_x);
            const vec_indices j = floor_V(grid_y);
            const vec_indices k = floor_V(grid_z);

            // Get the reminder [0, 1]
            vec_real l_x = (grid_x - vec_indices_to_real(i));
            vec_real l_y = (grid_y - vec_indices_to_real(j));
            vec_real l_z = (grid_z - vec_indices_to_real(k));

            // assert(l_x >= -1e-8); /// Maybe define a macro for the assert in SIMD version
            // assert(l_y >= -1e-8);
            // assert(l_z >= -1e-8);

            // assert(l_x <= 1 + 1e-8);
            // assert(l_y <= 1 + 1e-8);
            // assert(l_z <= 1 + 1e-8);

            hex_aa_8_eval_fun_V(l_x, l_y, l_z, hex8_f);
            hex_aa_8_collect_coeffs_V(stride0, stride1, stride2, i, j, k, data, coeffs);

            // Integrate field
            {
                vec_real eval_field = (vec_real)ZEROS_VEC;
                // UNROLL_ZERO
                for (int edof_j = 0; edof_j < 8; edof_j++) {
                    eval_field += hex8_f[edof_j] * coeffs[edof_j];
                    // eval_field += coeffs[edof_j];
                }

                // UNROLL_ZERO
                for (int edof_i = 0; edof_i < 10; edof_i++) {
                    element_field[edof_i] += eval_field * tet10_f[edof_i] * dV;
                }  // end edof_i loop
            }
        }  // end quadrature loop

        ///// QUI ======================================================

        UNROLL_ZERO
        for (int v = 0; v < 10; ++v) {
            // #pragma omp atomic update

            real_t element_field_v = 0.0;
            SIMD_REDUCE_SUM_MACRO(element_field_v, element_field[v]);

            weighted_field[ev[v]] += element_field_v;

            // for (int ii = 0; ii < _VL_; ii++) {
            //     if (isnan(element_field[v][ii])) {
            //         printf("ERROR: element_field[v][ii] is NaN %f, v = %d, ii = %d file:
            //         %s:%d\n",
            //                element_field[v][ii],
            //                v,
            //                ii,
            //                __FILE__,
            //                __LINE__);

            //         exit(1);
            //     }
            // }

        }  // end vertex loop
    }      // end element loop
    // }          // end parallel region

    return 0;
}  // end function hex8_to_tet10_resample_field_local
//////////////////////////////////////////////////////////

// ////////////////////////////////////////////////////////////////////////
// // hex_aa_8_eval_fun_V
// ////////////////////////////////////////////////////////////////////////
// SFEM_INLINE static void hex_aa_8_eval_fun_V(
//         // Quadrature point (local coordinates)
//         // With respect to the hat functions of a cube element
//         // In a local coordinate system
//         const vec_real x, const vec_real y, const vec_real z,
//         // Output
//         vec_real* const SFEM_RESTRICT f) {
//     //
//     f[0] = (1.0 - x) * (1.0 - y) * (1.0 - z);
//     f[1] = x * (1.0 - y) * (1.0 - z);
//     f[2] = x * y * (1.0 - z);
//     f[3] = (1.0 - x) * y * (1.0 - z);
//     f[4] = (1.0 - x) * (1.0 - y) * z;
//     f[5] = x * (1.0 - y) * z;
//     f[6] = x * y * z;
//     f[7] = (1.0 - x) * y * z;
// }

SFEM_INLINE static real_t tet4_measure_V(
        // X-coordinates
        const real_t px0, const real_t px1, const real_t px2, const real_t px3,
        // Y-coordinates
        const real_t py0, const real_t py1, const real_t py2, const real_t py3,
        // Z-coordinates
        const real_t pz0, const real_t pz1, const real_t pz2, const real_t pz3) {
    //
    // determinant of the Jacobian
    // M = [px0, py0, pz0, 1]
    //     [px1, py1, pz1, 1]
    //     [px2, py2, pz2, 1]
    //     [px3, py3, pz3, 1]
    //
    // V = (1/6) * det(M)

    const real_t x0 = -pz0 + pz3;
    const real_t x1 = -py0 + py2;
    const real_t x2 = -1.0 / 6.0 * px0 + (1.0 / 6.0) * px1;
    const real_t x3 = -py0 + py3;
    const real_t x4 = -pz0 + pz2;
    const real_t x5 = -py0 + py1;
    const real_t x6 = -1.0 / 6.0 * px0 + (1.0 / 6.0) * px2;
    const real_t x7 = -pz0 + pz1;
    const real_t x8 = -1.0 / 6.0 * px0 + (1.0 / 6.0) * px3;

    return x0 * x1 * x2 - x0 * x5 * x6 - x1 * x7 * x8 - x2 * x3 * x4 + x3 * x6 * x7 + x4 * x5 * x8;
}

SFEM_INLINE static void tet4_transform_VV(
        /**
         ****************************************************************************************
        \begin{bmatrix}
        out_x \\
        out_y \\
        out_z
        \end{bmatrix}
        =
        \begin{bmatrix}
        px_0 \\
        py_0 \\
        pz_0
        \end{bmatrix}
        +
        \begin{bmatrix}
        px_1 - px_0 & px_2 - px_0 & px_3 - px_0 \\
        py_1 - py_0 & py_2 - py_0 & py_3 - py_0 \\
        pz_1 - pz_0 & pz_2 - pz_0 & pz_3 - pz_0
        \end{bmatrix}
        \cdot
        \begin{bmatrix}
        qx \\
        qy \\
        qz
        \end{bmatrix}
        *************************************************************************************************
*/

        // X-coordinates
        const real_t px0, const real_t px1, const real_t px2, const real_t px3,
        // Y-coordinates
        const real_t py0, const real_t py1, const real_t py2, const real_t py3,
        // Z-coordinates
        const real_t pz0, const real_t pz1, const real_t pz2, const real_t pz3,
        // Quadrature point
        const vec_real qx, const vec_real qy, const vec_real qz,
        // Output
        vec_real* const SFEM_RESTRICT out_x,    //
        vec_real* const SFEM_RESTRICT out_y,    //
        vec_real* const SFEM_RESTRICT out_z) {  //
    //
    //
    *out_x = px0 + qx * (-px0 + px1) + qy * (-px0 + px2) + qz * (-px0 + px3);
    *out_y = py0 + qx * (-py0 + py1) + qy * (-py0 + py2) + qz * (-py0 + py3);
    *out_z = pz0 + qx * (-pz0 + pz1) + qy * (-pz0 + pz2) + qz * (-pz0 + pz3);
}  // end tet4_transform_v2

/**
 * @brief Compute the dual basis of the tet10 element
 *
 * @param qx
 * @param qy
 * @param qz
 * @param f
 * @return SFEM_INLINE
 */
SFEM_INLINE static void tet10_dual_basis_hrt_VV(const vec_real qx,    //
                                                const vec_real qy,    //
                                                const vec_real qz,    //
                                                vec_real* const f) {  //
    const vec_real x0 = 2 * qy;
    const vec_real x1 = 2 * qz;
    const vec_real x2 = 2 * qx - 1;
    const vec_real x3 = (-x0 - x1 - x2) * (-qx - qy - qz + 1);
    const vec_real x4 = x0 - 1;
    const vec_real x5 = (5.0 / 18.0) * qy;
    const vec_real x6 = x4 * x5;
    const vec_real x7 = x1 - 1;
    const vec_real x8 = (5.0 / 18.0) * qz;
    const vec_real x9 = x7 * x8;
    const vec_real x10 = -4 * qx - 4 * qy - 4 * qz + 4;
    const vec_real x11 = (5.0 / 72.0) * x10;
    const vec_real x12 = qy * qz;
    const vec_real x13 = qx * x11 + (10.0 / 9.0) * x12 + x6 + x9;
    const vec_real x14 = (5.0 / 18.0) * qx;
    const vec_real x15 = x14 * x2;
    const vec_real x16 = (10.0 / 9.0) * qx;
    const vec_real x17 = qy * x11 + qz * x16 + x15;
    const vec_real x18 = qy * x16 + qz * x11;
    const vec_real x19 = qx * x2;
    const vec_real x20 = (5.0 / 18.0) * x3;
    const vec_real x21 = qy * x14 + x10 * x8 + x20;
    const vec_real x22 = qz * x14 + x10 * x5;
    const vec_real x23 = qy * x4;
    const vec_real x24 = qz * x5 + x10 * x14;
    const vec_real x25 = qz * x7;
    const vec_real x26 = (40.0 / 27.0) * x23;
    const vec_real x27 = (115.0 / 27.0) * x10;
    const vec_real x28 = (110.0 / 27.0) * qx;
    const vec_real x29 = -qz * x28;
    const vec_real x30 = (55.0 / 54.0) * x10;
    const vec_real x31 = -qy * x30;
    const vec_real x32 = (10.0 / 27.0) * x19;
    const vec_real x33 = (40.0 / 27.0) * x25;
    const vec_real x34 = x29 + x31 + x32 + x33;
    const vec_real x35 = -qy * x28;
    const vec_real x36 = -qz * x30;
    const vec_real x37 = (10.0 / 27.0) * x3;
    const vec_real x38 = x35 + x36 + x37;
    const vec_real x39 = (40.0 / 27.0) * x10;
    const vec_real x40 = qx * qy;
    const vec_real x41 = -qx * x30 - 110.0 / 27.0 * x12;
    const vec_real x42 = (10.0 / 27.0) * x23;
    const vec_real x43 = (40.0 / 27.0) * x3;
    const vec_real x44 = x42 + x43;
    const vec_real x45 = qx * qz;
    const vec_real x46 = (40.0 / 27.0) * x19;
    const vec_real x47 = x41 + x46;
    const vec_real x48 = (10.0 / 27.0) * x25;
    const vec_real x49 = x26 + x48;
    const vec_real x50 = x29 + x31;
    const vec_real x51 = x35 + x36;

    f[0] = x13 + x17 + x18 + (25.0 / 9.0) * x3;
    f[1] = x13 + (25.0 / 9.0) * x19 + x21 + x22;
    f[2] = x17 + x21 + (25.0 / 9.0) * x23 + x24 + x9;
    f[3] = x15 + x18 + x20 + x22 + x24 + (25.0 / 9.0) * x25 + x6;
    f[4] = qx * x27 + (160.0 / 27.0) * x12 + x26 + x34 + x38;
    f[5] = qz * x39 + x34 + (460.0 / 27.0) * x40 + x41 + x44;
    f[6] = qy * x27 + x33 + x38 + x42 + (160.0 / 27.0) * x45 + x47;
    f[7] = qz * x27 + x37 + (160.0 / 27.0) * x40 + x47 + x49 + x50;
    f[8] = qy * x39 + x32 + x41 + x43 + (460.0 / 27.0) * x45 + x49 + x51;
    f[9] = qx * x39 + (460.0 / 27.0) * x12 + x44 + x46 + x48 + x50 + x51;
}

////////////////////////////////////////////////////////////////////////
// hex8_to_tet10_resample_field_local_cube1_V
////////////////////////////////////////////////////////////////////////
int hex8_to_subparametric_tet10_resample_field_local_V(
        // Mesh
        const ptrdiff_t nelements,          // number of elements
        const ptrdiff_t nnodes,             // number of nodes
        idx_t** const SFEM_RESTRICT elems,  // connectivity
        geom_t** const SFEM_RESTRICT xyz,   // coordinates
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n,       // number of nodes in each direction
        const ptrdiff_t* const SFEM_RESTRICT stride,  // stride of the data
        const geom_t* const SFEM_RESTRICT origin,     // origin of the domain
        const geom_t* const SFEM_RESTRICT delta,      // delta of the domain
        const real_t* const SFEM_RESTRICT data,       // SDF
        // Output
        real_t* const SFEM_RESTRICT weighted_field) {
    //
    PRINT_CURRENT_FUNCTION;
    // printf("============================================================\n");
    // printf("Start: hex8_to_tet10_resample_field_local\n");
    // printf("============================================================\n");
    //
    const real_t ox = (real_t)origin[0];
    const real_t oy = (real_t)origin[1];
    const real_t oz = (real_t)origin[2];

    const real_t dx = (real_t)delta[0];
    const real_t dy = (real_t)delta[1];
    const real_t dz = (real_t)delta[2];

    const ptrdiff_t stride0 = stride[0];
    const ptrdiff_t stride1 = stride[1];
    const ptrdiff_t stride2 = stride[2];

    /// Loop over the elements of the mesh
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        //
        idx_t ev[10];

        // SUBPARAMETRIC
        real_t x[4], y[4], z[4];

        vec_real hex8_f[8];
        vec_real coeffs[8];

        vec_real tet10_f[10];
        vec_real element_field[10];

        // loop over the 4 vertices of the tetrahedron
        // UNROLL_ZERO?
        for (int v = 0; v < 10; ++v) {
            ev[v] = elems[v][i];
        }

        // SUBPARAMETRIC (for isoparametric we collect all 10 points instead of 4)
        // copy the coordinates of the vertices
        for (int v = 0; v < 4; ++v) {
            x[v] = (real_t)xyz[0][ev[v]];  // x-coordinates
            y[v] = (real_t)xyz[1][ev[v]];  // y-coordinates
            z[v] = (real_t)xyz[2][ev[v]];  // z-coordinates
        }

        // set to zero the element field
        for (int v = 0; v < 10; v++) {
            element_field[v] = (vec_real)ZEROS_VEC;
        }

        // SUBPARAMETRIC (for isoparametric this is a nonlinear map computed for each qp)
        const real_t measure = tet4_measure_V(x[0],
                                              x[1],
                                              x[2],
                                              x[3],
                                              //
                                              y[0],
                                              y[1],
                                              y[2],
                                              y[3],
                                              //
                                              z[0],
                                              z[1],
                                              z[2],
                                              z[3]);

        assert(measure > 0);

        // SUBPARAMETRIC (for iso-parametric tassellation of tet10 might be necessary)
        for (int q = 0; q < TET4_NQP; q += _VL_) {  // loop over the quadrature points

            vec_real qx_V, qy_V, qz_V, qw_V;
            const int q_next = q + _VL_;
            if (q_next < TET4_NQP) {
                ASSIGN_QUADRATURE_POINT_MACRO(q, qx_V, qy_V, qz_V, qw_V);
            } else {
                ASSIGN_QUADRATURE_POINT_MACRO_TAIL(q, qx_V, qy_V, qz_V, qw_V);
            }

            // det of jacobian
            vec_real g_qx_V, g_qy_V, g_qz_V;
            // Transform quadrature point to physical space
            // g_qx, g_qy, g_qz are the coordinates of the quadrature point in the physical
            // space

            // SUBPARAMETRIC (for isoparametric this is a nonlinear map)
            tet4_transform_VV(x[0],
                              x[1],
                              x[2],
                              x[3],
                              //
                              y[0],
                              y[1],
                              y[2],
                              y[3],
                              //
                              z[0],
                              z[1],
                              z[2],
                              z[3],
                              //
                              qx_V,  //
                              qy_V,  //
                              qz_V,  //
                              //
                              &g_qx_V,
                              &g_qy_V,
                              &g_qz_V);

            // No standard basis function (tet10 cannot lump as for tet4, needs special
            // treatment)
            // tet10_dual_basis_popp(tet4_qx[q], tet4_qy[q], tet4_qz[q], tet10_f);
            tet10_dual_basis_hrt_VV(qx_V, qy_V, qz_V, tet10_f);

            const vec_real dV = measure * qw_V;

            const vec_real grid_x_V = (g_qx_V - ox) / dx;
            const vec_real grid_y_V = (g_qy_V - oy) / dy;
            const vec_real grid_z_V = (g_qz_V - oz) / dz;

            const vec_indices i_V = floor_V(grid_x_V);
            const vec_indices j_V = floor_V(grid_y_V);
            const vec_indices k_V = floor_V(grid_z_V);

            // // If outside
            // if (i < 0 || j < 0 || k < 0 || (i + 1 >= n[0]) || (j + 1 >= n[1]) || (k + 1 >= n[2]))
            // {
            //     fprintf(stderr,
            //             "warning (%g, %g, %g) (%ld, %ld, %ld) outside domain  (%ld, %ld, "
            //             "%ld)!\n",
            //             g_qx,
            //             g_qy,
            //             g_qz,
            //             i,
            //             j,
            //             k,
            //             n[0],
            //             n[1],
            //             n[2]);
            //     continue;
            // }

            // Get the reminder [0, 1]
            vec_real l_x_V = (grid_x_V - (vec_real)i_V);
            vec_real l_y_V = (grid_y_V - (vec_real)j_V);
            vec_real l_z_V = (grid_z_V - (vec_real)k_V);

            // assert(l_x >= -1e-8);
            // assert(l_y >= -1e-8);
            // assert(l_z >= -1e-8);

            // assert(l_x <= 1 + 1e-8);
            // assert(l_y <= 1 + 1e-8);
            // assert(l_z <= 1 + 1e-8);

            hex_aa_8_eval_fun_V(l_x_V, l_y_V, l_z_V, hex8_f);
            hex_aa_8_collect_coeffs_V(stride0,
                                      stride1,
                                      stride2,  //
                                      i_V,
                                      j_V,
                                      k_V,
                                      data,
                                      coeffs);

            // Integrate field
            {
                vec_real eval_field = (vec_real)ZEROS_VEC;
                // UNROLL_ZERO?
                for (int edof_j = 0; edof_j < 8; edof_j++) {
                    eval_field += hex8_f[edof_j] * coeffs[edof_j];
                }

                // UNROLL_ZERO?
                for (int edof_i = 0; edof_i < 10; edof_i++) {
                    element_field[edof_i] += eval_field * tet10_f[edof_i] * dV;
                }  // end edof_i loop
            }
        }  // end quadrature loop

        // UNROLL_ZERO?
        for (int v = 0; v < 10; ++v) {
            real_t element_field_v = 0.0;

            SIMD_REDUCE_SUM_MACRO(element_field_v, element_field[v]);
            weighted_field[ev[v]] += element_field_v;

        }  // end vertex loop
    }      // end element loop

    return 0;
}  // end function hex8_to_subparametric_tet10_resample_field_local_V

////////////////////////////////////////////////////////////////////////
// hex_aa_8_eval_weno4_3D
////////////////////////////////////////////////////////////////////////
SFEM_INLINE static vec_real hex_aa_8_eval_weno4_3D_Unit_V(  //
        const vec_real x_unit,                              //
        const vec_real y_unit,                              //
        const vec_real z_unit,                              //
        const vec_real ox_unit,                             //
        const vec_real oy_unit,                             //
        const vec_real oz_unit,                             //
        const vec_indices i,                                // it must be the absolute index
        const vec_indices j,                                // Used to get the data
        const vec_indices k,                                // From the data array
        const ptrdiff_t* stride,                            //
        const real_t* const SFEM_RESTRICT data) {           //

    // collect the data for the WENO interpolation

    // const int stride_x = stride[0];
    // const int stride_y = stride[1];
    // const int stride_z = stride[2];

    // real_t* out = NULL;
    real_t* first_ptrs_array[_VL_];
    hex_aa_8_collect_coeffs_O3_ptr_vec(stride, i, j, k, data, first_ptrs_array);

    // ////// Compute the local indices
    // vec_indices i_local, j_local, k_local;

    const vec_real ones_vec = CONST_VEC(1.0);

    const vec_indices i_local = floor_V(x_unit - ox_unit);
    const vec_indices j_local = floor_V(y_unit - oy_unit);
    const vec_indices k_local = floor_V(z_unit - oz_unit);

    const vec_real i_local_vec = vec_indices_to_real(i_local);
    const vec_real x = (x_unit - ox_unit) - i_local_vec + ones_vec;

    const vec_real j_local_vec = vec_indices_to_real(j_local);
    const vec_real y = (y_unit - oy_unit) - j_local_vec + ones_vec;

    const vec_real k_local_vec = vec_indices_to_real(k_local);
    const vec_real z = (z_unit - oz_unit) - k_local_vec + ones_vec;

    // // printf("x = %f, x_ = %f, i = %d\n", x, x_, i);
    // // printf("y = %f, y_ = %f, j = %d\n", y, y_, j);
    // // printf("z = %f, z_ = %f, k = %d\n", z, z_, k);

    // // printf("delta = %f\n", h);

    const vec_real w4 = weno4_3D_HOne_V(stride, x, y, z, (const real_t**)first_ptrs_array);

    return w4;
}

//////////////////////////////////////////////////////////
/// hex8_to_tet10_resample_field_local_cube1_V2
//////////////////////////////////////////////////////////
int hex8_to_isoparametric_tet10_resample_field_local_cube1_V(
        /// Mesh
        const ptrdiff_t nelements,          // number of elements
        const ptrdiff_t nnodes,             // number of nodes
        idx_t** const SFEM_RESTRICT elems,  // connectivity
        geom_t** const SFEM_RESTRICT xyz,   // coordinates
        /// SDF
        const ptrdiff_t* const SFEM_RESTRICT n,       // number of nodes in each direction
        const ptrdiff_t* const SFEM_RESTRICT stride,  // stride of the data
        const geom_t* const SFEM_RESTRICT origin,     // origin of the domain
        const geom_t* const SFEM_RESTRICT delta,      // delta of the domain
        const real_t* const SFEM_RESTRICT data,       // SDF
        // Output                                     //
        real_t* const SFEM_RESTRICT weighted_field) {  //
    //
    PRINT_CURRENT_FUNCTION;

    const real_t ox = (real_t)origin[0];
    const real_t oy = (real_t)origin[1];
    const real_t oz = (real_t)origin[2];

    const real_t dx = (real_t)delta[0];
    const real_t dy = (real_t)delta[1];
    const real_t dz = (real_t)delta[2];

    const real_t cVolume = dx * dy * dz;

    const ptrdiff_t stride0 = stride[0];
    const ptrdiff_t stride1 = stride[1];
    const ptrdiff_t stride2 = stride[2];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
        // printf("element = %d\n", i);

        idx_t ev[10];

        // ISOPARAMETRIC
        real_t x[10], y[10], z[10];
        real_t x_unit[10], y_unit[10], z_unit[10];

        vec_real hex8_f[8];
        vec_real coeffs[8];

        vec_real tet10_f[10];
        vec_real element_field[10];

        // loop over the 4 vertices of the tetrahedron
        UNROLL_ZERO
        for (int v = 0; v < 10; ++v) {
            ev[v] = elems[v][i];
        }

        // ISOPARAMETRIC
        // search for the vertex with the minimum distance to the origin

        int v_orig = 0;
        real_t dist_min = 1e14;

        for (int v = 0; v < 10; ++v) {
            x[v] = (real_t)(xyz[0][ev[v]]);  // x-coordinates
            y[v] = (real_t)(xyz[1][ev[v]]);  // y-coordinates
            z[v] = (real_t)(xyz[2][ev[v]]);  // z-coordinates

            const real_t dist = sqrt((x[v] - ox) * (x[v] - ox) +  //
                                     (y[v] - oy) * (y[v] - oy) +  //
                                     (z[v] - oz) * (z[v] - oz));  //

            if (dist < dist_min) {
                dist_min = dist;
                v_orig = v;
            }
        }

        const real_t grid_x_orig = (x[v_orig] - ox) / dx;
        const real_t grid_y_orig = (y[v_orig] - oy) / dy;
        const real_t grid_z_orig = (z[v_orig] - oz) / dz;

        const ptrdiff_t i_orig = floor(grid_x_orig);
        const ptrdiff_t j_orig = floor(grid_y_orig);
        const ptrdiff_t k_orig = floor(grid_z_orig);

        const real_t x_orig = ox + ((real_t)i_orig) * dx;
        const real_t y_orig = oy + ((real_t)j_orig) * dy;
        const real_t z_orig = oz + ((real_t)k_orig) * dz;

        // memset(element_field, 0, 10 * sizeof(real_t));

        // set to zero the element field
        for (int ii = 0; ii < 10; ii++) {
            element_field[ii] = ZEROS_VEC;
        }

        // Map element to the grid based on unitary spacing
        for (int v = 0; v < 10; ++v) {
            x_unit[v] = (x[v] - x_orig) / dx;
            y_unit[v] = (y[v] - y_orig) / dy;
            z_unit[v] = (z[v] - z_orig) / dz;
        }

        // set to zero the element field
        // memset(element_field, 0, 10 * sizeof(real_t));

        for (int q = 0; q < TET4_NQP; q += (_VL_)) {
            vec_real tet4_qx_V, tet4_qy_V, tet4_qz_V, tet4_qw_V;

            const int q_next = q + (_VL_);

            if (q_next < TET4_NQP) {
                ASSIGN_QUADRATURE_POINT_MACRO(q, tet4_qx_V, tet4_qy_V, tet4_qz_V, tet4_qw_V);
            } else {
                ASSIGN_QUADRATURE_POINT_MACRO_TAIL(q, tet4_qx_V, tet4_qy_V, tet4_qz_V, tet4_qw_V);
            }

            const vec_real measure_V = tet10_measure_V(x_unit,  //
                                                       y_unit,
                                                       z_unit,
                                                       tet4_qx_V,
                                                       tet4_qy_V,
                                                       tet4_qz_V);

            const vec_real dV = measure_V * tet4_qw_V * cVolume;

            vec_real g_qx_glob_V, g_qy_glob_V, g_qz_glob_V;
            tet10_transform_V(x,
                              y,
                              z,
                              tet4_qx_V,
                              tet4_qy_V,
                              tet4_qz_V,
                              &g_qx_glob_V,
                              &g_qy_glob_V,
                              &g_qz_glob_V);

            tet10_dual_basis_hrt_V(tet4_qx_V, tet4_qy_V, tet4_qz_V, tet10_f);

            // Transform quadrature point to unitary space
            // g_qx_unit, g_qy_unit, g_qz_unit are the coordinates of the quadrature point in
            // the unitary space
            vec_real g_qx_unit_V, g_qy_unit_V, g_qz_unit_V;
            tet10_transform_V(x_unit,
                              y_unit,
                              z_unit,
                              tet4_qx_V,
                              tet4_qy_V,
                              tet4_qz_V,
                              &g_qx_unit_V,
                              &g_qy_unit_V,
                              &g_qz_unit_V);

            ///// ======================================================

            // Get the global grid coordinates of the inner cube
            // In the global space

            const vec_real grid_x_V = (g_qx_glob_V - ox) / dx;
            const vec_real grid_y_V = (g_qy_glob_V - oy) / dy;
            const vec_real grid_z_V = (g_qz_glob_V - oz) / dz;

            const vec_indices i_glob_V = floor_V(grid_x_V);
            const vec_indices j_glob_V = floor_V(grid_y_V);
            const vec_indices k_glob_V = floor_V(grid_z_V);

            // If outside the domain, omit the control at the moment

            // This is the WENO direct approach
            // The main approach is omitted in this version

            // Calculate the origin of the 4x4x4 cube in the global space
            // And transform the coordinates to the the unitary space
            const vec_real x_cube_origin_V = (ox + ((vec_real)i_glob_V - (real_t)1.0) * dx) / dx;
            const vec_real y_cube_origin_V = (oy + ((vec_real)j_glob_V - (real_t)1.0) * dy) / dy;
            const vec_real z_cube_origin_V = (oz + ((vec_real)k_glob_V - (real_t)1.0) * dz) / dz;

            //// Compute the WENO interpolation
            vec_real eval_field;
            eval_field = hex_aa_8_eval_weno4_3D_Unit_V(g_qx_unit_V,      //
                                                       g_qy_unit_V,      //
                                                       g_qz_unit_V,      //
                                                       x_cube_origin_V,  //
                                                       y_cube_origin_V,  //
                                                       z_cube_origin_V,  //  //
                                                       i_glob_V,         //
                                                       j_glob_V,         //
                                                       k_glob_V,         //
                                                       stride,           //
                                                       data);            //

            // for (int iii = 0; iii < _VL_; iii++) {
            //     double eval_field_t = hex_aa_8_eval_weno4_3D_Unit(g_qx_unit_V[iii],      //
            //                                                       g_qy_unit_V[iii],      //
            //                                                       g_qz_unit_V[iii],      //
            //                                                       x_cube_origin_V[iii],  //
            //                                                       y_cube_origin_V[iii],  //
            //                                                       z_cube_origin_V[iii],  //
            //                                                       i_glob_V[iii],         //
            //                                                       j_glob_V[iii],         //
            //                                                       k_glob_V[iii],         //
            //                                                       stride,                //
            //                                                       data);                 //

            //     // double eval_field_t = pinco_p();

            //     // printf("eval_field = %f\n", eval_field_t);
            //     eval_field[iii] = eval_field_t;
            // }

            // eval_field = (vec_real)CONST_VEC(1.0);

            // TODO: Check if this is correct
            // TODO: Check if this is correct
            // vec_real eval_field = ZEROS_VEC;

            // Integrate field
            // {
            //     // vec_real eval_field = ZEROS_VEC;
            //     // UNROLL_ZERO
            //     for (int edof_j = 0; edof_j < 8; edof_j++) {
            //         eval_field += hex8_f[edof_j] * coeffs[edof_j];
            //     }

            // UNROLL_ZERO
            for (int edof_i = 0; edof_i < 10; edof_i++) {
                element_field[edof_i] += eval_field * tet10_f[edof_i] * dV;
            }  // end edof_i loop
            // }

        }  // end quadrature loop

        ///// QUI ======================================================
        UNROLL_ZERO
        for (int v = 0; v < 10; ++v) {
            // #pragma omp atomic update

            real_t element_field_v = 0.0;
            SIMD_REDUCE_SUM_MACRO(element_field_v, element_field[v]);

            weighted_field[ev[v]] += element_field_v;
        }  // end vertex loop

    }  // end element loop

    RETURN_FROM_FUNCTION(0);
}  // end function hex8_to_tet10_resample_field_local_cube1_V2

////////////////////////////////////////////////////////////////////////
// hex8_to_tet10_resample_field_local_cube1_V
////////////////////////////////////////////////////////////////////////
int isoparametric_tet10_assemble_dual_mass_vector_V(const ptrdiff_t nelements,
                                                    const ptrdiff_t nnodes,
                                                    idx_t** const SFEM_RESTRICT elems,
                                                    geom_t** const SFEM_RESTRICT xyz,
                                                    real_t* const diag) {
    PRINT_CURRENT_FUNCTION;

    idx_t ev[10];  // Element indices
    vec_real element_diag[10];
    real_t x[10], y[10], z[10];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
        //

        for (int v = 0; v < 10; ++v) {
            ev[v] = elems[v][i];
        }

        for (int v = 0; v < 10; ++v) {
            x[v] = xyz[0][ev[v]];  // x-coordinates
            y[v] = xyz[1][ev[v]];  // y-coordinates
            z[v] = xyz[2][ev[v]];  // z-coordinates
        }

        for (int v = 0; v < 10; ++v) {
            element_diag[v] = (vec_real)ZEROS_VEC;
        }

        // We do this numerical integration due to the det J
        for (int q = 0; q < TET4_NQP; q += _VL_) {  // loop over the quadrature points
                                                    //
            vec_real tet4_qx_V, tet4_qy_V, tet4_qz_V, tet4_qw_V;

            const int q_next = q + (_VL_);

            if (q_next < TET4_NQP) {
                ASSIGN_QUADRATURE_POINT_MACRO(q, tet4_qx_V, tet4_qy_V, tet4_qz_V, tet4_qw_V);
            } else {
                ASSIGN_QUADRATURE_POINT_MACRO_TAIL(q, tet4_qx_V, tet4_qy_V, tet4_qz_V, tet4_qw_V);
            }

            vec_real dV = tet10_measure_V(x, y, z, tet4_qx_V, tet4_qy_V, tet4_qz_V) * tet4_qw_V;

            isoparametric_lumped_mass_kernel_hrt_V(dV,
                                                   // Quadrature
                                                   tet4_qx_V,
                                                   tet4_qy_V,
                                                   tet4_qz_V,
                                                   element_diag);
        }  // end quadrature loop

        for (int v = 0; v < 10; ++v) {
            const idx_t idx = ev[v];

            float_t diag_v = 0.0;
            SIMD_REDUCE_SUM_MACRO(diag_v, element_diag[v]);

            diag[idx] += diag_v;

        }  // end vertex loop
    }      // end element loop

    RETURN_FROM_FUNCTION(0);
}  // end isoparametric_tet10_assemble_dual_mass_vector

/**
 * @brief
 *
 * @param nelements
 * @param nnodes
 * @param elems
 * @param xyz
 * @param n
 * @param stride
 * @param origin
 * @param delta
 * @param data
 * @param weighted_field
 * @return int
 */
int hex8_to_tet10_resample_field_local_V2(
        // Mesh
        const ptrdiff_t nelements,          // number of elements
        const ptrdiff_t nnodes,             // number of nodes
        idx_t** const SFEM_RESTRICT elems,  // connectivity
        geom_t** const SFEM_RESTRICT xyz,   // coordinates
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n,       // number of nodes in each direction
        const ptrdiff_t* const SFEM_RESTRICT stride,  // stride of the data
        const geom_t* const SFEM_RESTRICT origin,     // origin of the domain
        const geom_t* const SFEM_RESTRICT delta,      // delta of the domain
        const real_t* const SFEM_RESTRICT data,       // SDF
        // Output
        real_t* const SFEM_RESTRICT weighted_field) {
    //
    int SFEM_ENABLE_ISOPARAMETRIC = 0;
    SFEM_READ_ENV(SFEM_ENABLE_ISOPARAMETRIC, atoi);

    // const ptrdiff_t nelements_aligned = nelements - (nelements % (_VL_));
    // const ptrdiff_t nelements_tail = nelements % (_VL_);

    if (SFEM_ENABLE_ISOPARAMETRIC) {
        int a = 0;

#if SFEM_TET10_WENO == OFF
        a = hex8_to_isoparametric_tet10_resample_field_local_V(nelements,  //
                                                               nnodes,
                                                               elems,
                                                               xyz,
                                                               n,
                                                               stride,
                                                               origin,
                                                               delta,
                                                               data,
                                                               weighted_field);

#else
        a = hex8_to_isoparametric_tet10_resample_field_local_cube1_V(nelements,  //
                                                                     nnodes,
                                                                     elems,
                                                                     xyz,
                                                                     n,
                                                                     stride,
                                                                     origin,
                                                                     delta,
                                                                     data,
                                                                     weighted_field);
#endif
        return a;
    } else {
        // return hex8_to_subparametric_tet10_resample_field_local(nelements,  //
        //                                                         nnodes,
        //                                                         elems,
        //                                                         xyz,
        //                                                         n,
        //                                                         stride,
        //                                                         origin,
        //                                                         delta,
        //                                                         data,
        //                                                         weighted_field);
    }
}