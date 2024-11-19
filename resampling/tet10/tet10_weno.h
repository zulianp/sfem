#ifndef __TET10_WENO_H__
#define __TET10_WENO_H__

#include <math.h>
#include "sfem_base.h"

#define List2(ARRAY, AA, BB) \
    {                        \
        ARRAY[0] = (AA);     \
        ARRAY[1] = (BB);     \
    }

#define List3(ARRAY, AA, BB, CC) \
    {                            \
        ARRAY[0] = (AA);         \
        ARRAY[1] = (BB);         \
        ARRAY[2] = (CC);         \
    }

#define Abs(x) fabs(x)

void getLinearWeightsConstH(const real_t x, const real_t h, real_t *linear_weights);

void getNonLinearWeightsConstH(const real_t x, const real_t h,    //
                               const real_t y0, const real_t y1,  //
                               const real_t y2, const real_t y3,  //
                               real_t *non_linear_weights,        //
                               const real_t eps);

real_t weno4ConstH(const real_t x, const real_t h,                                       //
                   const real_t y0, const real_t y1, const real_t y2, const real_t y3);  //

real_t weno4_2D_ConstH(const real_t x, const real_t y, const real_t h,  //
                                                                        //
                       const real_t y00, const real_t y10, const real_t y20, const real_t y30,
                       //
                       const real_t y01, const real_t y11, const real_t y21, const real_t y31,
                       //
                       const real_t y02, const real_t y12, const real_t y22, const real_t y32,
                       //
                       const real_t y03, const real_t y13, const real_t y23, const real_t y33);

real_t weno4_3D_ConstH(const real_t x, const real_t y, const real_t z,  //
                       const real_t h, const real_t *f,                 //
                       const int stride_x,                              //
                       const int stride_y,                              //
                       const int stride_z);                             //

real_t weno4_3D_HOne(const real_t x, const real_t y, const real_t z,  //
                     const real_t *f,                                 //
                     const int stride_x,                              //
                     const int stride_y,                              //
                     const int stride_z);

#endif  // __TET10_WENO_H__