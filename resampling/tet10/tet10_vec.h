#ifndef __TET10_VEC_H__
#define __TET10_VEC_H__

#include <math.h>
#include <stddef.h>

#include "sfem_base.h"
#include "tet10_weno.h"

#if SFEM_VEC_SIZE == 8 && real_t == double
#pragma message "SIMD 512 double"

#define SIMD_512_DOUBLE
#define _VL_ 8

#define ptrdiff_t_sfem int64_t

#elif SFEM_VEC_SIZE == 4 && real_t == double
#pragma message "SIMD 256 double"

#define SIMD_256_DOUBLE
#define _VL_ 4

#define ptrdiff_t_sfem int64_t

#elif SFEM_VEC_SIZE == 4 && real_t == float

#pragma message "SIMD 256 float"

#define SIMD_256_FLOAT
#define _VL_ 8

#define ptrdiff_t_sfem int32_t

#elif SFEM_VEC_SIZE == 8 && real_t == float

#pragma message "SIMD 512 float"

#define SIMD_512_FLOAT

#define _VL_ 16

#define ptrdiff_t_sfem int32_t

#endif

typedef real_t vec_real __attribute__((vector_size(_VL_ * sizeof(real_t)),  //
                                       aligned(sizeof(real_t))));

typedef ptrdiff_t_sfem vec_indices __attribute__((vector_size(_VL_ * sizeof(ptrdiff_t_sfem)),  //
                                                  aligned(sizeof(ptrdiff_t_sfem))));

#if _VL_ == 8

#define ZEROS_VEC \
    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }

#define CONST_VEC(__X__) \
    { __X__, __X__, __X__, __X__, __X__, __X__, __X__, __X__ }

#elif _VL_ == 4

#define ZEROS_VEC \
    { 0.0, 0.0, 0.0, 0.0 }
#define CONST_VEC(__X__) \
    { __X__, __X__, __X__, __X__ }

#elif _VL_ == 16

#define ZEROS_VEC(__X__)                                                                           \
    {                                                                                              \
        __X__, __X__, __X__, __X__, __X__, __X__, __X__, __X__, __X__, __X__, __X__, __X__, __X__, \
                __X__, __X__, __X__                                                                \
    }

#endif

#endif  // __TET10_VEC_H__