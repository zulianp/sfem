#ifndef __TET10_WENO_V_H__
#define __TET10_WENO_V_H__

#include <math.h>
#include <stddef.h>

#include "sfem_base.h"
#include "sfem_defs.h"

#if SFEM_VEC_SIZE == 8
#define AVX512
#elif SFEM_VEC_SIZE == 4
#define AVX2
#endif

#ifdef AVX512
#define _VL_ 8
#define ZEROS_VEC \
    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }
#define CONST_VEC(__X__) \
    { __X__, __X__, __X__, __X__, __X__, __X__, __X__, __X__ }
#elif defined(AVX2)
#define _VL_ 4
#define ZEROS_VEC \
    { 0.0, 0.0, 0.0, 0.0 }
#define CONST_VEC(__X__) \
    { __X__, __X__, __X__, __X__ }
#endif

// #define UNROLL_ZERO _Pragma("GCC unroll(0)")
#define UNROLL_ZERO _Pragma("unroll(1)")

typedef double vec_double __attribute__((vector_size(_VL_ * sizeof(double)),  //
                                         aligned(sizeof(double))));

typedef ptrdiff_t vec_indices __attribute__((vector_size(_VL_ * sizeof(ptrdiff_t)),  //
                                             aligned(sizeof(ptrdiff_t))));

typedef float_t* ptr_array[_VL_];

#define List2_V(ARRAY, AA, BB) \
    {                          \
        ARRAY[0] = (AA);       \
        ARRAY[1] = (BB);       \
    }

#define List3_V(ARRAY, AA, BB, CC) \
    {                              \
        ARRAY[0] = (AA);           \
        ARRAY[1] = (BB);           \
        ARRAY[2] = (CC);           \
    }

// void getLinearWeightsConstH(const vec_double x, const vec_double h, vec_double *linear_weights);

// void getNonLinearWeightsConstH(const double x, const double h,    //
//                                const double y0, const double y1,  //
//                                const double y2, const double y3,  //
//                                double *non_linear_weights,        //
//                                const double eps);

// double weno4ConstH(const double x, const double h,                                       //
//                    const double y0, const double y1, const double y2, const double y3);  //

// double weno4_2D_ConstH(const double x, const double y, const double h,  //
//                                                                         //
//                        const double y00, const double y10, const double y20, const double y30,
//                        //
//                        const double y01, const double y11, const double y21, const double y31,
//                        //
//                        const double y02, const double y12, const double y22, const double y32,
//                        //
//                        const double y03, const double y13, const double y23, const double y33);

// double weno4_3D_ConstH(const double x, const double y, const double z,  //
//                        const double h, const double *f,                 //
//                        const int stride_x,                              //
//                        const int stride_y,                              //
//                        const int stride_z);                             //

vec_double weno4_3D_HOne_V(const vec_double x, const vec_double y, const vec_double z,  //
                           const double* f,                                             //
                           const vec_indices stride_x,                                  //
                           const vec_indices stride_y,                                  //
                           const vec_indices stride_z);                                 //

#endif  // __TET10_WENO_V_H__