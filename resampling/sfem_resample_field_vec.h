#ifndef __SFEM_RESAMPLE_FIELD_VEC_H__
#define __SFEM_RESAMPLE_FIELD_VEC_H__

#include <math.h>
#include <stddef.h>

#include "sfem_base.h"

#if SFEM_VEC_SIZE == 8 && real_t == double

#pragma message "SIMD 512 double"

#define SIMD_512_DOUBLE
#define _VL_ 8

#define ptrdiff_t_sfem_tet4 int64_t

#elif SFEM_VEC_SIZE == 4 && real_t == double

#pragma message "SIMD 256 double"

#define SIMD_256_DOUBLE
#define _VL_ 4

#define ptrdiff_t_sfem_tet4 int64_t

#elif SFEM_VEC_SIZE == 4 && real_t == float

#pragma message "SIMD 256 float"

#define SIMD_256_FLOAT
#define _VL_ 8

#define ptrdiff_t_sfem_tet4 int32_t

#elif SFEM_VEC_SIZE == 8 && real_t == float

#pragma message "SIMD 512 float"

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

#endif  // __SFEM_RESAMPLE_FIELD_VEC_H__