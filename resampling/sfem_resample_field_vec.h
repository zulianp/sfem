#ifndef __SFEM_RESAMPLE_FIELD_VEC_H__
#define __SFEM_RESAMPLE_FIELD_VEC_H__

#include <math.h>
#include <stddef.h>

#include "sfem_base.h"

#if SFEM_VEC_SIZE == 8 && SIZEOF_REAL_T == 8

#pragma message "USING SIMD 512 double"

#define SIMD_512_DOUBLE
#define _VL_ 8

#define ptrdiff_t_sfem_tet4 int64_t

#elif SFEM_VEC_SIZE == 4 && SIZEOF_REAL_T == 8

#pragma message "USING SIMD 256 double"

#define SIMD_256_DOUBLE
#define _VL_ 4

#define ptrdiff_t_sfem_tet4 int64_t

#elif SFEM_VEC_SIZE == 4 && SIZEOF_REAL_T == 4

#pragma message "USING SIMD 256 float"

#define SIMD_256_FLOAT
#define _VL_ 8

#define ptrdiff_t_sfem_tet4 int32_t

#elif SFEM_VEC_SIZE == 8 && SIZEOF_REAL_T == 4

#pragma message "USING SIMD 512 float"

#define SIMD_512_FLOAT

#define _VL_ 16

#define ptrdiff_t_sfem_tet4 int32_t

#endif

typedef real_t vec8_t __attribute__((vector_size(8 * sizeof(real_t)),  //
                                     aligned(sizeof(real_t))));        //

typedef real_t vec4_t __attribute__((vector_size(4 * sizeof(real_t)),  //
                                     aligned(sizeof(real_t))));        //

typedef real_t vec16_t __attribute__((vector_size(16 * sizeof(real_t)),  //
                                      aligned(sizeof(real_t))));         //

typedef ptrdiff_t_sfem_tet4 vec_indices8_t
        __attribute__((vector_size(8 * sizeof(ptrdiff_t_sfem_tet4)),  //
                       aligned(sizeof(ptrdiff_t_sfem_tet4))));        //

typedef ptrdiff_t_sfem_tet4 vec_indices4_t
        __attribute__((vector_size(4 * sizeof(ptrdiff_t_sfem_tet4)),  //
                       aligned(sizeof(ptrdiff_t_sfem_tet4))));        //

typedef ptrdiff_t_sfem_tet4 vec_indices16_t
        __attribute__((vector_size(16 * sizeof(ptrdiff_t_sfem_tet4)),  //
                       aligned(sizeof(ptrdiff_t_sfem_tet4))));         //

////////////////////////////////////////////////////////////////////////
typedef real_t vec_real __attribute__((vector_size(_VL_ * sizeof(real_t)),  //
                                       aligned(sizeof(real_t))));           //

typedef ptrdiff_t_sfem_tet4 vec_indices                                  //
        __attribute__((vector_size(_VL_ * sizeof(ptrdiff_t_sfem_tet4)),  //
                       aligned(sizeof(ptrdiff_t_sfem_tet4))));           //

// __attribute__((vector_size(_VL_ * sizeof(ptrdiff_t_sfem_tet4)),  //
//                aligned(sizeof(ptrdiff_t_sfem_tet4))));           //

#if defined(SIMD_512_DOUBLE) || defined(SIMD_256_FLOAT)

#define CONST_VEC(_val_) \
    { (_val_), (_val_), (_val_), (_val_), (_val_), (_val_), (_val_), (_val_) }

#define INCREMENT_VEC(_val_) \
    { _val_ + 0, _val_ + 1, _val_ + 2, _val_ + 3, _val_ + 4, _val_ + 5, _val_ + 6, _val_ + 7 }

#define ZEROS_VEC() \
    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }

// #include <immintrin.h>
// union vec_union {
//     vec_real v;
//     __m512d m;
// };

// union vec_union_indices {
//     vec_indices v;
//     __m512d m;
// };

// vec_double gather_vec_dbl(const double *ptr, vec_int64 indices) {
//     vec_union result;

//     __m512i i_index = _mm512_load_epi64(&indices);
//     result.m = _mm512_i64gather_pd(i_index, (void const *)ptr, sizeof(double));

//     return result.v;
// }

// vec_indices gather_vec_indices(const int *ptr, vec_int64 indices) {
//     vec_union_indices result;

//     __m512i i_index = _mm512_load_epi64(&indices);
//     result.m = _mm512_i64gather_pd(i_index, (void const *)ptr, sizeof(double));

//     return result.v;
// }

#elif defined(SIMD_256_DOUBLE)

#define CONST_VEC(_val_) \
    { (_val_), (_val_), (_val_), (_val_) }

#define INCREMENT_VEC(_val_) \
    { _val_ + 0, _val_ + 1, _val_ + 2, _val_ + 3 }

#define ZEROS_VEC() \
    { 0.0, 0.0, 0.0, 0.0 }

#elif defined(SIMD_512_FLOAT)

#define CONST_VEC(_val_)                                                                          \
    {                                                                                             \
        (_val_), (_val_), (_val_), (_val_), (_val_), (_val_), (_val_), (_val_), (_val_), (_val_), \
                (_val_), (_val_), (_val_), (_val_), (_val_), (_val_)                              \
    }

#define INCREMENT_VEC(_val_)                                                                      \
    {                                                                                             \
        _val_ + 0, _val_ + 1, _val_ + 2, _val_ + 3, _val_ + 4, _val_ + 5, _val_ + 6, _val_ + 7,   \
                _val_ + 8, _val_ + 9, _val_ + 10, _val_ + 11, _val_ + 12, _val_ + 13, _val_ + 14, \
                _val_ + 15                                                                        \
    }

#define ZEROS_VEC() \
    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }

#endif

#endif  // __SFEM_RESAMPLE_FIELD_VEC_H__