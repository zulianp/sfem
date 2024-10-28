#include "tet10_resample_field_V2.h"

#include "quadratures_rule.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tet10_weno_V.h"

#if SFEM_VEC_SIZE == 8
#define AVX512
#elif SFEM_VEC_SIZE == 4
#define AVX2
#endif

#ifdef AVX512
#define _VL_ 8
#elif defined(AVX2)
#define _VL_ 4
#endif

// #define UNROLL_ZERO _Pragma("GCC unroll(0)")
#define UNROLL_ZERO _Pragma("unroll(1)")

typedef double vec_double __attribute__((vector_size(_VL_ * sizeof(double)),  //
                                         aligned(sizeof(double))));

typedef ptrdiff_t vec_int64 __attribute__((vector_size(_VL_ * sizeof(ptrdiff_t)),  //
                                           aligned(sizeof(ptrdiff_t))));

// Casting from int64 to double SIMD vector
vec_double vec_int64_to_double(const vec_int64 a) {
#ifdef AVX512  //// AVX512
    return (vec_double){(double)a[0],
                        (double)a[1],
                        (double)a[2],
                        (double)a[3],
                        (double)a[4],
                        (double)a[5],
                        (double)a[6],
                        (double)a[7]};
#elif defined(AVX2)  //// AVX2
    return (vec_double){(double)a[0], (double)a[1], (double)a[2], (double)a[3]};
#endif
}

#define ASSIGN_QUADRATURE_POINT_MACRO(_q, _qx_V, _qy_V, _qz_V, _qw_V) \
    {                                                                 \
        _qx_V = *((vec_double*)(&tet4_qx[q]));                        \
        _qy_V = *((vec_double*)(&tet4_qy[q]));                        \
        _qz_V = *((vec_double*)(&tet4_qz[q]));                        \
        _qw_V = *((vec_double*)(&tet4_qw[q]));                        \
    }

//////////////////////////////////////////////////////////
/// Macros for the cases of the SIMD implementation
#ifdef AVX512  //// AVX512

#define ASSIGN_QUADRATURE_POINT_MACRO_TAIL(_q, _qx_V, _qy_V, _qz_V, _qw_V)         \
    {                                                                              \
        _qx_V = (vec_double){(_q + 0) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 0],  \
                             (_q + 1) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 1],  \
                             (_q + 2) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 2],  \
                             (_q + 3) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 3],  \
                             (_q + 4) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 4],  \
                             (_q + 5) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 5],  \
                             (_q + 6) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 6],  \
                             (_q + 7) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 7]}; \
                                                                                   \
        _qy_V = (vec_double){(_q + 0) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 0],  \
                             (_q + 1) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 1],  \
                             (_q + 2) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 2],  \
                             (_q + 3) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 3],  \
                             (_q + 4) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 4],  \
                             (_q + 5) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 5],  \
                             (_q + 6) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 6],  \
                             (_q + 7) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 7]}; \
                                                                                   \
        _qz_V = (vec_double){(_q + 0) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 0],  \
                             (_q + 1) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 1],  \
                             (_q + 2) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 2],  \
                             (_q + 3) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 3],  \
                             (_q + 4) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 4],  \
                             (_q + 5) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 5],  \
                             (_q + 6) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 6],  \
                             (_q + 7) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 7]}; \
                                                                                   \
        _qw_V = (vec_double){(_q + 0) >= TET4_NQP ? 0.0 : tet4_qw[_q + 0],         \
                             (_q + 1) >= TET4_NQP ? 0.0 : tet4_qw[_q + 1],         \
                             (_q + 2) >= TET4_NQP ? 0.0 : tet4_qw[_q + 2],         \
                             (_q + 3) >= TET4_NQP ? 0.0 : tet4_qw[_q + 3],         \
                             (_q + 4) >= TET4_NQP ? 0.0 : tet4_qw[_q + 4],         \
                             (_q + 5) >= TET4_NQP ? 0.0 : tet4_qw[_q + 5],         \
                             (_q + 6) >= TET4_NQP ? 0.0 : tet4_qw[_q + 6],         \
                             (_q + 7) >= TET4_NQP ? 0.0 : tet4_qw[_q + 7]};        \
    }

#elif defined(AVX2)  //// AVX2

#define ASSIGN_QUADRATURE_POINT_MACRO_TAIL(_q, _qx_V, _qy_V, _qz_V, _qw_V)         \
    {                                                                              \
        _qx_V = (vec_double){(_q + 0) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 0],  \
                             (_q + 1) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 1],  \
                             (_q + 2) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 2],  \
                             (_q + 3) >= TET4_NQP ? tet4_qx[0] : tet4_qx[_q + 3]}; \
                                                                                   \
        _qy_V = (vec_double){(_q + 0) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 0],  \
                             (_q + 1) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 1],  \
                             (_q + 2) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 2],  \
                             (_q + 3) >= TET4_NQP ? tet4_qy[0] : tet4_qy[_q + 3]}; \
                                                                                   \
        _qz_V = (vec_double){(_q + 0) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 0],  \
                             (_q + 1) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 1],  \
                             (_q + 2) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 2],  \
                             (_q + 3) >= TET4_NQP ? tet4_qz[0] : tet4_qz[_q + 3]}; \
                                                                                   \
        _qw_V = (vec_double){(_q + 0) >= TET4_NQP ? 0.0 : tet4_qw[_q + 0],         \
                             (_q + 1) >= TET4_NQP ? 0.0 : tet4_qw[_q + 1],         \
                             (_q + 2) >= TET4_NQP ? 0.0 : tet4_qw[_q + 2],         \
                             (_q + 3) >= TET4_NQP ? 0.0 : tet4_qw[_q + 3]};        \
    }

#endif  //// end SIMD implementation

void assert_vec_double(const vec_double a, const vec_double b) {
    for (int i = 0; i < _VL_; i++) {
        if (a[i] == b[i]) {
            printf("Error: %f == %f\n", a[i], b[i]);
            exit(1);
        }
    }
}

//////////////////////////////////////////////////////////
/// Macros for the cases of the SIMD implementation
#ifdef AVX512  //// AVX512

//// ZEROS for AVX512
#define ZEROS_SIMD_MACRO \
    (vec_double) { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }

//// SIMD_REDUCE_SUM for AVX512
#define SIMD_REDUCE_SUM_MACRO(_out, _in) \
    { _out = _in[0] + _in[1] + _in[2] + _in[3] + _in[4] + _in[5] + _in[6] + _in[7]; }

//// SIMD floor for AVX512
vec_int64 floor_V(vec_double a) {
    // vec_int64 r = (vec_int64){0, 0, 0, 0, 0, 0, 0, 0};
    // for(int ii = 0; ii < _VL_; ii++) {
    //     r[ii] = floor(a[ii]);
    // }

    // return r;
    return (vec_int64){(ptrdiff_t)a[0],
                       (ptrdiff_t)a[1],
                       (ptrdiff_t)a[2],
                       (ptrdiff_t)a[3],
                       (ptrdiff_t)a[4],
                       (ptrdiff_t)a[5],
                       (ptrdiff_t)a[6],
                       (ptrdiff_t)a[7]};
}

#elif defined(AVX2)  //// AVX2 ////////////////////////////

//// ZEROS for AVX2
#define ZEROS_SIMD_MACRO \
    (vec_double) { 0.0, 0.0, 0.0, 0.0 }

//// SIMD_REDUCE_SUM for AVX2
#define SIMD_REDUCE_SUM_MACRO(_out, _in) \
    { _out = _in[0] + _in[1] + _in[2] + _in[3]; }

//// SIMD floor for AVX2
vec_int64 floor_V(vec_double a) {
    return (vec_int64){(ptrdiff_t)a[0], (ptrdiff_t)a[1], (ptrdiff_t)a[2], (ptrdiff_t)a[3]};
}
#endif  //// end SIMD implementation

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
SFEM_INLINE static vec_double tet10_measure_V(const real_t* const SFEM_RESTRICT x,
                                              const real_t* const SFEM_RESTRICT y,
                                              const real_t* const SFEM_RESTRICT z,
                                              // Quadrature point
                                              const vec_double qx, const vec_double qy,
                                              const vec_double qz) {
    const vec_double x0 = 4 * qz;
    const vec_double x1 = x0 - 1;
    const vec_double x2 = 4 * qy;
    const vec_double x3 = 4 * qx;
    const vec_double x4 = x3 - 4;
    const vec_double x5 = -8 * qz - x2 - x4;
    const vec_double x6 = -x3 * y[4];
    const vec_double x7 = x0 + x2;
    const vec_double x8 = x3 + x7 - 3;
    const vec_double x9 = x8 * y[0];
    const vec_double x10 = -x2 * y[6] + x9;
    const vec_double x11 = x1 * y[3] + x10 + x2 * y[9] + x3 * y[8] + x5 * y[7] + x6;
    const vec_double x12 = -x2 * z[6];
    const vec_double x13 = -x0 * z[7];
    const vec_double x14 = x3 - 1;
    const vec_double x15 = x8 * z[0];
    const vec_double x16 = -8 * qx - x7 + 4;
    const vec_double x17 = x0 * z[8] + x12 + x13 + x14 * z[1] + x15 + x16 * z[4] + x2 * z[5];
    const vec_double x18 = x2 - 1;
    const vec_double x19 = -8 * qy - x0 - x4;
    const vec_double x20 = -x3 * x[4];
    const vec_double x21 = x8 * x[0];
    const vec_double x22 = -x0 * x[7] + x21;
    const vec_double x23 = (1.0 / 6.0) * x0 * x[9] + (1.0 / 6.0) * x18 * x[2] +
                           (1.0 / 6.0) * x19 * x[6] + (1.0 / 6.0) * x20 + (1.0 / 6.0) * x22 +
                           (1.0 / 6.0) * x3 * x[5];
    const vec_double x24 = -x0 * y[7];
    const vec_double x25 = x0 * y[8] + x10 + x14 * y[1] + x16 * y[4] + x2 * y[5] + x24;
    const vec_double x26 = x15 - x3 * z[4];
    const vec_double x27 = x1 * z[3] + x12 + x2 * z[9] + x26 + x3 * z[8] + x5 * z[7];
    const vec_double x28 = x0 * y[9] + x18 * y[2] + x19 * y[6] + x24 + x3 * y[5] + x6 + x9;
    const vec_double x29 = -x2 * x[6];
    const vec_double x30 = (1.0 / 6.0) * x1 * x[3] + (1.0 / 6.0) * x2 * x[9] + (1.0 / 6.0) * x20 +
                           (1.0 / 6.0) * x21 + (1.0 / 6.0) * x29 + (1.0 / 6.0) * x3 * x[8] +
                           (1.0 / 6.0) * x5 * x[7];
    const vec_double x31 = x0 * z[9] + x13 + x18 * z[2] + x19 * z[6] + x26 + x3 * z[5];
    const vec_double x32 = (1.0 / 6.0) * x0 * x[8] + (1.0 / 6.0) * x14 * x[1] +
                           (1.0 / 6.0) * x16 * x[4] + (1.0 / 6.0) * x2 * x[5] + (1.0 / 6.0) * x22 +
                           (1.0 / 6.0) * x29;

    return x11 * x17 * x23 - x11 * x31 * x32 - x17 * x28 * x30 - x23 * x25 * x27 + x25 * x30 * x31 +
           x27 * x28 * x32;
}

SFEM_INLINE static void tet10_transform_V(const real_t* const SFEM_RESTRICT x,
                                          const real_t* const SFEM_RESTRICT y,
                                          const real_t* const SFEM_RESTRICT z,
                                          // Quadrature point
                                          const vec_double qx, const vec_double qy,
                                          const vec_double qz,
                                          // Output
                                          vec_double* const SFEM_RESTRICT out_x,
                                          vec_double* const SFEM_RESTRICT out_y,
                                          vec_double* const SFEM_RESTRICT out_z) {
    const vec_double x0 = 4.0 * qx;
    const vec_double x1 = qy * x0;
    const vec_double x2 = qz * x0;
    const vec_double x3 = 4.0 * qy;
    const vec_double x4 = qz * x3;
    const vec_double x5 = 2.0 * qx - 1.0;
    const vec_double x6 = qx * x5;
    const vec_double x7 = 2.0 * qy;
    const vec_double x8 = qy * (x7 - 1.0);
    const vec_double x9 = 2.0 * qz;
    const vec_double x10 = qz * (x9 - 1.0);
    const vec_double x11 = -4 * qz - x0 - x3 + 4.0;
    const vec_double x12 = qx * x11;
    const vec_double x13 = qy * x11;
    const vec_double x14 = qz * x11;
    const vec_double x15 = (-x5 - x7 - x9) * (-qx - qy - qz + 1.0);

    *out_x = x[0] * x15 + x[1] * x6 + x[2] * x8 + x[3] * x10 + x[4] * x12 + x[5] * x1 + x[6] * x13 +
             x[7] * x14 + x[8] * x2 + x[9] * x4;
    *out_y = y[0] * x15 + y[1] * x6 + y[2] * x8 + y[3] * x10 + y[4] * x12 + y[5] * x1 + y[6] * x13 +
             y[7] * x14 + y[8] * x2 + y[9] * x4;
    *out_z = z[0] * x15 + z[1] * x6 + z[2] * x8 + z[3] * x10 + z[4] * x12 + z[5] * x1 + z[6] * x13 +
             z[7] * x14 + z[8] * x2 + z[9] * x4;
}

SFEM_INLINE static void tet10_dual_basis_hrt_V(const vec_double qx, const vec_double qy,
                                               const vec_double qz, vec_double* const f) {
    const vec_double x0 = 2 * qy;
    const vec_double x1 = 2 * qz;
    const vec_double x2 = 2 * qx - 1;
    const vec_double x3 = (-x0 - x1 - x2) * (-qx - qy - qz + 1);
    const vec_double x4 = x0 - 1;
    const vec_double x5 = (5.0 / 18.0) * qy;
    const vec_double x6 = x4 * x5;
    const vec_double x7 = x1 - 1;
    const vec_double x8 = (5.0 / 18.0) * qz;
    const vec_double x9 = x7 * x8;
    const vec_double x10 = -4 * qx - 4 * qy - 4 * qz + 4;
    const vec_double x11 = (5.0 / 72.0) * x10;
    const vec_double x12 = qy * qz;
    const vec_double x13 = qx * x11 + (10.0 / 9.0) * x12 + x6 + x9;
    const vec_double x14 = (5.0 / 18.0) * qx;
    const vec_double x15 = x14 * x2;
    const vec_double x16 = (10.0 / 9.0) * qx;
    const vec_double x17 = qy * x11 + qz * x16 + x15;
    const vec_double x18 = qy * x16 + qz * x11;
    const vec_double x19 = qx * x2;
    const vec_double x20 = (5.0 / 18.0) * x3;
    const vec_double x21 = qy * x14 + x10 * x8 + x20;
    const vec_double x22 = qz * x14 + x10 * x5;
    const vec_double x23 = qy * x4;
    const vec_double x24 = qz * x5 + x10 * x14;
    const vec_double x25 = qz * x7;
    const vec_double x26 = (40.0 / 27.0) * x23;
    const vec_double x27 = (115.0 / 27.0) * x10;
    const vec_double x28 = (110.0 / 27.0) * qx;
    const vec_double x29 = -qz * x28;
    const vec_double x30 = (55.0 / 54.0) * x10;
    const vec_double x31 = -qy * x30;
    const vec_double x32 = (10.0 / 27.0) * x19;
    const vec_double x33 = (40.0 / 27.0) * x25;
    const vec_double x34 = x29 + x31 + x32 + x33;
    const vec_double x35 = -qy * x28;
    const vec_double x36 = -qz * x30;
    const vec_double x37 = (10.0 / 27.0) * x3;
    const vec_double x38 = x35 + x36 + x37;
    const vec_double x39 = (40.0 / 27.0) * x10;
    const vec_double x40 = qx * qy;
    const vec_double x41 = -qx * x30 - 110.0 / 27.0 * x12;
    const vec_double x42 = (10.0 / 27.0) * x23;
    const vec_double x43 = (40.0 / 27.0) * x3;
    const vec_double x44 = x42 + x43;
    const vec_double x45 = qx * qz;
    const vec_double x46 = (40.0 / 27.0) * x19;
    const vec_double x47 = x41 + x46;
    const vec_double x48 = (10.0 / 27.0) * x25;
    const vec_double x49 = x26 + x48;
    const vec_double x50 = x29 + x31;
    const vec_double x51 = x35 + x36;

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
        const vec_double x, const vec_double y, const vec_double z,
        // Output
        vec_double* const SFEM_RESTRICT f) {
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
#ifdef AVX512

#define GET_DATA_MACRO(_out, _data, _indx_V)    \
    {                                           \
        _out = (vec_double){_data[_indx_V[0]],  \
                            _data[_indx_V[1]],  \
                            _data[_indx_V[2]],  \
                            _data[_indx_V[3]],  \
                            _data[_indx_V[4]],  \
                            _data[_indx_V[5]],  \
                            _data[_indx_V[6]],  \
                            _data[_indx_V[7]]}; \
    }

#elif defined(AVX2)

#define GET_DATA_MACRO(_out, _data, _indx_V)                                                 \
    {                                                                                        \
        _out = (vec_double){                                                                 \
                _data[_indx_V[0]], _data[_indx_V[1]], _data[_indx_V[2]], _data[_indx_V[3]]}; \
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

        const vec_int64 i, const vec_int64 j, const vec_int64 k,

        // Attention this is geometric data transformed to solver data!
        const real_t* const SFEM_RESTRICT data, vec_double* const SFEM_RESTRICT out) {
    //
    const vec_int64 i0 = i * stride0 + j * stride1 + k * stride2;
    const vec_int64 i1 = (i + 1) * stride0 + j * stride1 + k * stride2;
    const vec_int64 i2 = (i + 1) * stride0 + (j + 1) * stride1 + k * stride2;
    const vec_int64 i3 = i * stride0 + (j + 1) * stride1 + k * stride2;
    const vec_int64 i4 = i * stride0 + j * stride1 + (k + 1) * stride2;
    const vec_int64 i5 = (i + 1) * stride0 + j * stride1 + (k + 1) * stride2;
    const vec_int64 i6 = (i + 1) * stride0 + (j + 1) * stride1 + (k + 1) * stride2;
    const vec_int64 i7 = i * stride0 + (j + 1) * stride1 + (k + 1) * stride2;

    GET_DATA_MACRO(out[0], data, i0);
    GET_DATA_MACRO(out[1], data, i1);
    GET_DATA_MACRO(out[2], data, i2);
    GET_DATA_MACRO(out[3], data, i3);
    GET_DATA_MACRO(out[4], data, i4);
    GET_DATA_MACRO(out[5], data, i5);
    GET_DATA_MACRO(out[6], data, i6);
    GET_DATA_MACRO(out[7], data, i7);
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

        vec_double hex8_f[8];
        vec_double coeffs[8];

        vec_double tet10_f[10];
        vec_double element_field[10];

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
        for (int ii = 0; ii < 10; ii++) {
            element_field[ii] = ZEROS_SIMD_MACRO;
        }

        // SUBPARAMETRIC (for iso-parametric tassellation of tet10 might be necessary)
        for (int q = 0; q < TET4_NQP; q += (_VL_)) {  // loop over the quadrature points

            vec_double tet4_qx_V, tet4_qy_V, tet4_qz_V, tet4_qw_V;

            const int q_next = q + _VL_;
            // printf("q + % d,  qq = %d\n", q, qq);

            if (q_next <= TET4_NQP) {
                ASSIGN_QUADRATURE_POINT_MACRO(q, tet4_qx_V, tet4_qy_V, tet4_qz_V, tet4_qw_V);

            } else {
                ASSIGN_QUADRATURE_POINT_MACRO_TAIL(q, tet4_qx_V, tet4_qy_V, tet4_qz_V, tet4_qw_V);
            }

            const vec_double measure = tet10_measure_V(x, y, z, tet4_qx_V, tet4_qy_V, tet4_qz_V);

            const vec_double dV = measure * tet4_qw_V;

            vec_double g_qx, g_qy, g_qz;
            // Transform quadrature point to physical space
            // g_qx, g_qy, g_qz are the coordinates of the quadrature point in the physical
            // space
            tet10_transform_V(x, y, z, tet4_qx_V, tet4_qy_V, tet4_qz_V, &g_qx, &g_qy, &g_qz);
            tet10_dual_basis_hrt_V(tet4_qx_V, tet4_qy_V, tet4_qz_V, tet10_f);

            ///// ======================================================

            const vec_double grid_x = (g_qx - ox) / dx;
            const vec_double grid_y = (g_qy - oy) / dy;
            const vec_double grid_z = (g_qz - oz) / dz;

            const vec_int64 i = floor_V(grid_x);
            const vec_int64 j = floor_V(grid_y);
            const vec_int64 k = floor_V(grid_z);

            // Get the reminder [0, 1]
            vec_double l_x = (grid_x - vec_int64_to_double(i));
            vec_double l_y = (grid_y - vec_int64_to_double(j));
            vec_double l_z = (grid_z - vec_int64_to_double(k));

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
                vec_double eval_field = ZEROS_SIMD_MACRO;
                // UNROLL_ZERO
                for (int edof_j = 0; edof_j < 8; edof_j++) {
                    eval_field += hex8_f[edof_j] * coeffs[edof_j];
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

            double element_field_v = 0.0;
            SIMD_REDUCE_SUM_MACRO(element_field_v, element_field[v]);

            weighted_field[ev[v]] += element_field_v;

        }  // end vertex loop
    }      // end element loop
    // }          // end parallel region

    return 0;
}  // end function hex8_to_tet10_resample_field_local
//////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
// hex_aa_8_eval_weno4_3D
////////////////////////////////////////////////////////////////////////
SFEM_INLINE static real_t hex_aa_8_eval_weno4_3D_Unit_V(  //
        const vec_double x_unit,                          //
        const vec_double y_unit,                          //
        const vec_double z_unit,                          //
        const real_t ox_unit,                             //
        const real_t oy_unit,                             //
        const real_t oz_unit,                             //
        const vec_int64 i,                                // it must be the absolute index
        const vec_int64 j,                                // Used to get the data
        const vec_int64 k,                                // From the data array
        const ptrdiff_t* stride,                          //
        const real_t* const SFEM_RESTRICT data) {         //

    // collect the data for the WENO interpolation

    const int stride_x = stride[0];
    const int stride_y = stride[1];
    const int stride_z = stride[2];

    real_t w4 = 0.0;

    // real_t* out = NULL;
    ptr_array first_ptrs_array;
    hex_aa_8_collect_coeffs_O3_ptr_vec(stride, i, j, k, data, &first_ptrs_array);

    // ////// Compute the local indices
    // vec_int64 i_local, j_local, k_local;

    const vec_double ones_vec = CONST_VEC(1.0);

    const vec_int64 i_local = floor_V(x_unit - (vec_double)CONST_VEC(ox_unit));
    const vec_int64 j_local = floor_V(y_unit - (vec_double)CONST_VEC(oy_unit));
    const vec_int64 k_local = floor_V(z_unit - (vec_double)CONST_VEC(oz_unit));

    const vec_double i_local_vec = vec_int64_to_double(i_local);
    const vec_double x = (x_unit - (vec_double)CONST_VEC(ox_unit)) - i_local_vec + ones_vec;

    const vec_double j_local_vec = vec_int64_to_double(j_local);
    const vec_double y = (y_unit - (vec_double)CONST_VEC(oy_unit)) - j_local_vec + ones_vec;

    const vec_double k_local_vec = vec_int64_to_double(k_local);
    const vec_double z = (z_unit - (vec_double)CONST_VEC(oz_unit)) - k_local_vec + ones_vec;

    // // printf("x = %f, x_ = %f, i = %d\n", x, x_, i);
    // // printf("y = %f, y_ = %f, j = %d\n", y, y_, j);
    // // printf("z = %f, z_ = %f, k = %d\n", z, z_, k);

    // // printf("delta = %f\n", h);

    // const real_t w4 = weno4_3D_HOne(x,  //
    //                                 y,
    //                                 z,
    //                                 out,
    //                                 stride_x,
    //                                 stride_y,
    //                                 stride_z);

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

        vec_double hex8_f[8];
        vec_double coeffs[8];

        vec_double tet10_f[10];
        vec_double element_field[10];

        // loop over the 4 vertices of the tetrahedron
        UNROLL_ZERO
        for (int v = 0; v < 10; ++v) {
            ev[v] = elems[v][i];
        }

        // ISOPARAMETRIC
        // search for the vertex with the minimum distance to the origin

        int v_orig = 0;
        double dist_min = 1e14;

        for (int v = 0; v < 10; ++v) {
            x[v] = (real_t)(xyz[0][ev[v]]);  // x-coordinates
            y[v] = (real_t)(xyz[1][ev[v]]);  // y-coordinates
            z[v] = (real_t)(xyz[2][ev[v]]);  // z-coordinates

            const double dist = sqrt((x[v] - ox) * (x[v] - ox) +  //
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
            element_field[ii] = ZEROS_SIMD_MACRO;
        }

        // Map element to the grid based on unitary spacing
        for (int v = 0; v < 10; ++v) {
            x_unit[v] = (x[v] - x_orig) / dx;
            y_unit[v] = (y[v] - y_orig) / dy;
            z_unit[v] = (z[v] - z_orig) / dz;
        }

        // set to zero the element field
        memset(element_field, 0, 10 * sizeof(real_t));

        for (int q = 0; q < TET4_NQP; q += (_VL_)) {
            vec_double tet4_qx_V, tet4_qy_V, tet4_qz_V, tet4_qw_V;

            const int q_next = q + (_VL_);

            if (q_next <= TET4_NQP) {
                ASSIGN_QUADRATURE_POINT_MACRO(q, tet4_qx_V, tet4_qy_V, tet4_qz_V, tet4_qw_V);
            } else {
                ASSIGN_QUADRATURE_POINT_MACRO_TAIL(q, tet4_qx_V, tet4_qy_V, tet4_qz_V, tet4_qw_V);
            }

            const vec_double measure_V = tet10_measure_V(x, y, z, tet4_qx_V, tet4_qy_V, tet4_qz_V);

            const vec_double dV = measure_V * tet4_qw_V * cVolume;

            vec_double g_qx_glob_V, g_qy_glob_V, g_qz_glob_V;
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
            vec_double g_qx_unit_V, g_qy_unit_V, g_qz_unit_V;
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

            const vec_double grid_x_V = (g_qx_glob_V - ox) / dx;
            const vec_double grid_y_V = (g_qy_glob_V - oy) / dy;
            const vec_double grid_z_V = (g_qz_glob_V - oz) / dz;

            const vec_int64 i_glob_V = floor_V(grid_x_V);
            const vec_int64 j_glob_V = floor_V(grid_y_V);
            const vec_int64 k_glob_V = floor_V(grid_z_V);

            // If outside the domain, omit the control at the moment

            // This is the WENO direct approach
            // The main approach is omitted in this version
            const vec_double x_cube_origin = (ox + ((vec_double)i_glob_V - 1) * dx) / dx;
            const vec_double y_cube_origin = (oy + ((vec_double)j_glob_V - 1) * dy) / dy;
            const vec_double z_cube_origin = (oz + ((vec_double)k_glob_V - 1) * dz) / dz;

            // TODO: Check if this is correct
            // TODO: Check if this is correct
            vec_double eval_field = ZEROS_SIMD_MACRO;

            // Integrate field
            {
                vec_double eval_field = ZEROS_SIMD_MACRO;
                // UNROLL_ZERO
                for (int edof_j = 0; edof_j < 8; edof_j++) {
                    eval_field += hex8_f[edof_j] * coeffs[edof_j];
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

            double element_field_v = 0.0;
            SIMD_REDUCE_SUM_MACRO(element_field_v, element_field[v]);

            weighted_field[ev[v]] += element_field_v;
        }  // end vertex loop

    }  // end element loop

    RETURN_FROM_FUNCTION(0);
}  // end function hex8_to_tet10_resample_field_local_cube1_V2

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