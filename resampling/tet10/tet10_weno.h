#ifndef __TET10_WENO_H__
#define __TET10_WENO_H__

#include <math.h>

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

void getLinearWeightsConstH(const double x, const double h, double *linear_weights);

void getNonLinearWeightsConstH(const double x, const double h,    //
                               const double y0, const double y1,  //
                               const double y2, const double y3,  //
                               double *non_linear_weights,        //
                               const double eps);

double weno4ConstH(const double x, const double h,                                       //
                   const double y0, const double y1, const double y2, const double y3);  //

double weno4_2D_ConstH(const double x, const double y, const double h,  //
                                                                        //
                       const double y00, const double y10, const double y20, const double y30,
                       //
                       const double y01, const double y11, const double y21, const double y31,
                       //
                       const double y02, const double y12, const double y22, const double y32,
                       //
                       const double y03, const double y13, const double y23, const double y33);

double weno4_3D_ConstH(const double x, const double y, const double z,  //
                       const double h, const double *f,                 //
                       const int stride_x,                              //
                       const int stride_y,                              //
                       const int stride_z);                             //

#endif  // __TET10_WENO_H__