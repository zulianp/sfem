#ifndef __TET10_WENO_CUDA_H__
#define __TET10_WENO_CUDA_H__

#include "sfem_config.h"

#if SFEM_REAL_T_IS_FLOAT64

#define real_type double
#define Abs(x) fabs(x)

#elif SFEM_REAL_T_IS_FLOAT32

#define real_type float
#define Abs(x) fabsf(x)

#endif

// #ifndef real_type
// #define real_type double
// #endif

// #if real_type == double
// #define Abs(x) fabs(x)
// #elif real_type == float
// #define Abs(x) fabsf(x)
// #endif

#define List2_cu(ARRAY, AA, BB) \
    {                           \
        ARRAY[0] = (AA);        \
        ARRAY[1] = (BB);        \
    }

#define List3_cu(ARRAY, AA, BB, CC) \
    {                               \
        ARRAY[0] = (AA);            \
        ARRAY[1] = (BB);            \
        ARRAY[2] = (CC);            \
    }

/**
 * @brief Compute the power of x raised to the power of y
 *
 * @param x
 * @param y
 * @return real_type
 */
__device__ real_type Power_cuda(const real_type x, const real_type y) {
#if SFEM_REAL_T_IS_FLOAT64
    return pow(x, y);
#else
    return powf(x, y);
#endif
}

/**
 * @brief Compute the power of x raised to the power of 2
 *
 * @param x
 * @return real_type
 */
__device__ real_type Power2_cuda(const real_type x) { return x * x; }

/**
 * @brief Compute the power of x raised to the power of 1.5
 *
 * @param x
 * @return real_type
 */
__device__ real_type Power1p5_cuda(const real_type x) {
#if SFEM_REAL_T_IS_FLOAT64
    return x * sqrt(x);
#else
    return x * sqrtf(x);
#endif
}

__device__ real_type Power_m1p5(const real_type x) {
#if SFEM_REAL_T_IS_FLOAT64
    return 1.0 / (x * sqrt(x));
#else
    const real_type r1 = 1.0;
    return r1 / (x * sqrtf(x));
#endif
}

/**
 * @brief Compute the power of x raised to the power of -1.5
 *
 * @param x
 * @return real_type
 */
__device__ real_type Power_m1p5_cuda(const real_type x) {
#if SFEM_REAL_T_IS_FLOAT64
    return 1.0 / (x * sqrt(x));
#else
    return 1.0 / (x * sqrtf(x));
#endif
}

/**
 * @brief Compute the Lagrange Polynomials of order 2 at x for h constant
 *
 * @param x
 * @return real_type
 */
__device__ void                                                //
LagrangePolyArrayConstH_cuda(const real_type x,                //
                             const real_type h,                //
                             real_type      *lagrange_poly_0,  //
                             real_type      *lagrange_poly_1) {     //

    List3_cu(lagrange_poly_0,                                    //
             ((-2 * h + x) * (-h + x)) / (2. * Power2_cuda(h)),  //
             -((x * (-2 * h + x)) / Power2_cuda(h)),             //
             (x * (-h + x)) / (2. * Power2_cuda(h)));            //

    List3_cu(lagrange_poly_1,                                        //
             ((-3 * h + x) * (-2 * h + x)) / (2. * Power2_cuda(h)),  //
             -(((-3 * h + x) * (-h + x)) / Power2_cuda(h)),          //
             ((-2 * h + x) * (-h + x)) / (2. * Power2_cuda(h)));     //
}

/**
 * @brief Get the Linear Weights object by assuming the stencil starts at 0 and
 * h is constant
 *
 * @param x : value at which the weights are to be calculated
 * @param h : grid spacing
 * @param linear_weights : array to store the linear weights of size 2
 */
__device__ void  //
getLinearWeightsConstH_cuda(const real_type x, const real_type h, real_type *linear_weights) {
    //

    const real_type r3 = 3.0;

    linear_weights[0] = (r3 * h - x) / (r3 * h);
    linear_weights[1] = x / (r3 * h);
}

/**
 * @brief Get the Non Linear Weights for the WENO scheme by assuming the stencil starts at 0 and h
 * is constant
 *
 * @param x
 * @param h
 * @param y0
 * @param y1
 * @param y2
 * @param y3
 * @param non_linear_weights
 * @param eps
 */
__device__ void                                                     //
getNonLinearWeightsConstH_cuda(const real_type x,                   //
                               const real_type h,                   //
                               const real_type y0,                  //
                               const real_type y1,                  //
                               const real_type y2,                  //
                               const real_type y3,                  //
                               real_type      *non_linear_weights,  //
                               const real_type eps) {               //

    real_type alpha[2];

    const real_type r1   = 1.0;
    const real_type r2   = 2.0;
    const real_type r3   = 3.0;
    const real_type r5   = 5.0;
    const real_type r6   = 6.0;
    const real_type r7   = 7.0;
    const real_type r9   = 9.0;
    const real_type r11  = 11.0;
    const real_type r18  = 18.0;
    const real_type r1_9 = 1.0 / 9.0;
    const real_type r1_3 = 1.0 / 3.0;

    const real_type inv_h = r1 / h;

    const real_type a = -r3 * y0 + r7 * y1 - r5 * y2 + y3;
    const real_type b = y0 - r6 * (r1 + h) * y1 + r3 * y2 + r2 * y3;

    List2_cu(alpha,
             // alpha[0]
             Power_m1p5_cuda(eps + r1_9 * Power2_cuda(-r3 * Abs((a)*inv_h) + Abs((b)*inv_h))) -
                     (r1_3 * x) / (h * Power1p5_cuda(eps + r1_9 * Power2_cuda(-r3 * Abs((a)*inv_h) + Abs((b)*inv_h)))),
             //
             // alpha[1]
             (r1_3 * x) /
                     (h * Power1p5_cuda(eps +
                                        r1_9 *  //
                                                Power2_cuda(Abs((r6 * y0 - r9 * y1 + r18 * y2 + r6 * h * y2 - r11 * y3) * inv_h) -
                                                            r3 * Abs((y0 - r1 * y1 - r1 * y2 + y3) * inv_h)  //  end of Abs
                                                            )                                                // end of Power2
                                        )));

    real_type den = alpha[0] + alpha[1];

    non_linear_weights[0] = alpha[0] / den;
    non_linear_weights[1] = alpha[1] / den;
}

/**
 * @brief WENO for 4th order scheme with constant h and stencil starting at 0
 *
 * @param x
 * @param h
 * @param y0
 * @param y1
 * @param y2
 * @param y3
 * @return __device__
 */
__device__ real_type                    //
weno4ConstH_cuda(const real_type x,     //
                 const real_type h,     //
                 const real_type y0,    //
                 const real_type y1,    //
                 const real_type y2,    //
                 const real_type y3) {  //

    const real_type eps = 1e-6;

    real_type lagrange_poly_0[3];
    real_type lagrange_poly_1[3];
    real_type non_linear_weights[2];

    LagrangePolyArrayConstH_cuda(x, h, lagrange_poly_0, lagrange_poly_1);

    getNonLinearWeightsConstH_cuda(x, h, y0, y1, y2, y3, non_linear_weights, eps);

    const real_type weno4_a =
            (lagrange_poly_0[0] * y0 + lagrange_poly_0[1] * y1 + lagrange_poly_0[2] * y2) * non_linear_weights[0];

    const real_type weno4_b =
            (lagrange_poly_1[0] * y1 + lagrange_poly_1[1] * y2 + lagrange_poly_1[2] * y3) * non_linear_weights[1];

    return weno4_a + weno4_b;
}

/**
 * @brief Compute the WENO scheme for 4th order for 2D with constant h.
 * The f(x,y) values are arranged in a 4x4 matrix where y00 is the bottom left corner (x = 0, y = 0)
 * and y33 is the top right corner (x = 3h, y = 3h)
 *
 * @param x
 * @param y
 * @param h
 * @param y00
 * @param y10
 * @param y20
 * @param y30
 * @param y01
 * @param y11
 * @param y21
 * @param y31
 * @param y02
 * @param y12
 * @param y22
 * @param y32
 * @param y03
 * @param y13
 * @param y23
 * @param y33
 * @return real_type
 */
__device__ real_type weno4_2D_ConstH_cuda(const real_type x, const real_type y,
                                          const real_type h,                                                                   //
                                                                                                                               //
                                          const real_type y00, const real_type y10, const real_type y20, const real_type y30,  //
                                          //
                                          const real_type y01, const real_type y11, const real_type y21, const real_type y31,  //
                                          //
                                          const real_type y02, const real_type y12, const real_type y22, const real_type y32,  //
                                          //
                                          const real_type y03, const real_type y13, const real_type y23,
                                          const real_type y33) {  //
                                                                  //
    real_type yw0 = weno4ConstH_cuda(x, h, y00, y10, y20, y30);
    real_type yw1 = weno4ConstH_cuda(x, h, y01, y11, y21, y31);
    real_type yw2 = weno4ConstH_cuda(x, h, y02, y12, y22, y32);
    real_type yw3 = weno4ConstH_cuda(x, h, y03, y13, y23, y33);

    real_type yw = weno4ConstH_cuda(y, h, yw0, yw1, yw2, yw3);

    return yw;
}

/**
 * @brief Compute the WENO scheme for 4th order for 3D with constant h.
 * The f(x,y,z) values are arranged in a 4x4x4 matrix where f000 is the bottom left corner (x = 0, y
 * = 0, z = 0) and f333 is the top right corner (x = 3h, y = 3h, z = 3h)
 *
 * @param x
 * @param y
 * @param z
 * @param h
 * @param f
 * @param stride_x
 * @param stride_y
 * @param stride_z
 * @return real_type
 */
__device__ real_type weno4_3D_ConstH_cuda(const real_type x, const real_type y,
                                          const real_type z,                      //
                                          const real_type h, const real_type *f,  //
                                          const int stride_x,                     //
                                          const int stride_y,                     //
                                          const int stride_z) {                   //

    real_type w1 = weno4_2D_ConstH_cuda(x,
                                        y,
                                        h,  //
                                        f[0 * stride_x + 0 * stride_y + 0 * stride_z],
                                        f[1 * stride_x + 0 * stride_y + 0 * stride_z],
                                        f[2 * stride_x + 0 * stride_y + 0 * stride_z],
                                        f[3 * stride_x + 0 * stride_y + 0 * stride_z],
                                        //
                                        f[0 * stride_x + 1 * stride_y + 0 * stride_z],
                                        f[1 * stride_x + 1 * stride_y + 0 * stride_z],
                                        f[2 * stride_x + 1 * stride_y + 0 * stride_z],
                                        f[3 * stride_x + 1 * stride_y + 0 * stride_z],
                                        //
                                        f[0 * stride_x + 2 * stride_y + 0 * stride_z],
                                        f[1 * stride_x + 2 * stride_y + 0 * stride_z],
                                        f[2 * stride_x + 2 * stride_y + 0 * stride_z],
                                        f[3 * stride_x + 2 * stride_y + 0 * stride_z],
                                        //
                                        f[0 * stride_x + 3 * stride_y + 0 * stride_z],
                                        f[1 * stride_x + 3 * stride_y + 0 * stride_z],
                                        f[2 * stride_x + 3 * stride_y + 0 * stride_z],
                                        f[3 * stride_x + 3 * stride_y + 0 * stride_z]);

    real_type w2 = weno4_2D_ConstH_cuda(x,
                                        y,
                                        h,  //
                                        f[0 * stride_x + 0 * stride_y + 1 * stride_z],
                                        f[1 * stride_x + 0 * stride_y + 1 * stride_z],
                                        f[2 * stride_x + 0 * stride_y + 1 * stride_z],
                                        f[3 * stride_x + 0 * stride_y + 1 * stride_z],
                                        //
                                        f[0 * stride_x + 1 * stride_y + 1 * stride_z],
                                        f[1 * stride_x + 1 * stride_y + 1 * stride_z],
                                        f[2 * stride_x + 1 * stride_y + 1 * stride_z],
                                        f[3 * stride_x + 1 * stride_y + 1 * stride_z],
                                        //
                                        f[0 * stride_x + 2 * stride_y + 1 * stride_z],
                                        f[1 * stride_x + 2 * stride_y + 1 * stride_z],
                                        f[2 * stride_x + 2 * stride_y + 1 * stride_z],
                                        f[3 * stride_x + 2 * stride_y + 1 * stride_z],
                                        //
                                        f[0 * stride_x + 3 * stride_y + 1 * stride_z],
                                        f[1 * stride_x + 3 * stride_y + 1 * stride_z],
                                        f[2 * stride_x + 3 * stride_y + 1 * stride_z],
                                        f[3 * stride_x + 3 * stride_y + 1 * stride_z]);

    real_type w3 = weno4_2D_ConstH_cuda(x,
                                        y,
                                        h,  //
                                        f[0 * stride_x + 0 * stride_y + 2 * stride_z],
                                        f[1 * stride_x + 0 * stride_y + 2 * stride_z],
                                        f[2 * stride_x + 0 * stride_y + 2 * stride_z],
                                        f[3 * stride_x + 0 * stride_y + 2 * stride_z],
                                        //
                                        f[0 * stride_x + 1 * stride_y + 2 * stride_z],
                                        f[1 * stride_x + 1 * stride_y + 2 * stride_z],
                                        f[2 * stride_x + 1 * stride_y + 2 * stride_z],
                                        f[3 * stride_x + 1 * stride_y + 2 * stride_z],
                                        //
                                        f[0 * stride_x + 2 * stride_y + 2 * stride_z],
                                        f[1 * stride_x + 2 * stride_y + 2 * stride_z],
                                        f[2 * stride_x + 2 * stride_y + 2 * stride_z],
                                        f[3 * stride_x + 2 * stride_y + 2 * stride_z],
                                        //
                                        f[0 * stride_x + 3 * stride_y + 2 * stride_z],
                                        f[1 * stride_x + 3 * stride_y + 2 * stride_z],
                                        f[2 * stride_x + 3 * stride_y + 2 * stride_z],
                                        f[3 * stride_x + 3 * stride_y + 2 * stride_z]);

    real_type w4 = weno4_2D_ConstH_cuda(x,
                                        y,
                                        h,  //
                                        f[0 * stride_x + 0 * stride_y + 3 * stride_z],
                                        f[1 * stride_x + 0 * stride_y + 3 * stride_z],
                                        f[2 * stride_x + 0 * stride_y + 3 * stride_z],
                                        f[3 * stride_x + 0 * stride_y + 3 * stride_z],
                                        //
                                        f[0 * stride_x + 1 * stride_y + 3 * stride_z],
                                        f[1 * stride_x + 1 * stride_y + 3 * stride_z],
                                        f[2 * stride_x + 1 * stride_y + 3 * stride_z],
                                        f[3 * stride_x + 1 * stride_y + 3 * stride_z],
                                        //
                                        f[0 * stride_x + 2 * stride_y + 3 * stride_z],
                                        f[1 * stride_x + 2 * stride_y + 3 * stride_z],
                                        f[2 * stride_x + 2 * stride_y + 3 * stride_z],
                                        f[3 * stride_x + 2 * stride_y + 3 * stride_z],
                                        //
                                        f[0 * stride_x + 3 * stride_y + 3 * stride_z],
                                        f[1 * stride_x + 3 * stride_y + 3 * stride_z],
                                        f[2 * stride_x + 3 * stride_y + 3 * stride_z],
                                        f[3 * stride_x + 3 * stride_y + 3 * stride_z]);

    real_type wz = weno4ConstH_cuda(z, h, w1, w2, w3, w4);

    return wz;
}

/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
// WENO4 interpolation for 1D data with constant grid spacing set to 1 by default
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Get the Lagrange Poly Array object by assuming the h is equal to 1
 *
 * @param x
 * @param lagrange_poly_0
 * @param lagrange_poly_1
 */
__device__ void LagrangePolyArrayHOne(const real_type x,                //
                                      real_type      *lagrange_poly_0,  //
                                      real_type      *lagrange_poly_1) {     //

    // const real_type h = 1.0;

    const real_type xx = x * x;

    const real_type r1 = 1.0;
    const real_type r2 = 2.0;
    const real_type r3 = 3.0;
    const real_type r4 = 4.0;
    const real_type r5 = 5.0;
    const real_type r6 = 6.0;

    List3_cu(lagrange_poly_0,          //
             (r2 - r3 * x + xx) / r2,  //
             -((-r2 + x) * x),         //
             ((-r1 + x) * x) / r2      //
    );                                 //

    List3_cu(lagrange_poly_1,           //
             (r6 - r5 * x + xx) / r2,   //
             -r3 + r4 * x - xx,         //
             (r2 - r3 * x + xx) / r2);  //
}

/**
 * @brief Get the Non Linear Weights object by assuming the h is constant and
 * equal to 1
 *
 * @param x
 * @param y0
 * @param y1
 * @param y2
 * @param y3
 * @param non_linear_weights
 * @param eps
 */
__device__ void                                                       //
getNonLinearWeightsHOne_cuda(const real_type x,                       //
                             const real_type y0, const real_type y1,  //
                             const real_type y2, const real_type y3,  //
                             real_type      *non_linear_weights,      //
                             const real_type eps) {                   //

    real_type alpha[2];
    // const real_type h = 1.0;

    const real_type r3     = 3.0;
    const real_type r4     = 4.0;
    const real_type r7     = 7.0;
    const real_type r5     = 5.0;
    const real_type r1     = 1.0;
    const real_type r12    = 12.0;
    const real_type r2     = 2.0;
    const real_type r1_9   = 1.0 / 9.0;
    const real_type r1_3   = 1.0 / 3.0;
    const real_type r1p5   = 1.5;
    const real_type r1p833 = 1.8333333333333333;

    const real_type a = Abs(r3 * y0 - r7 * y1 + r5 * y2 - r1 * y3);
    const real_type b = Power2_cuda(-r3 * (a) + Abs(y0 - r12 * y1 + r3 * y2 + r2 * y3));

    List2_cu(alpha,
             //
             Power_m1p5(eps + r1_9 * (b)) - (r1_3 * x) / Power1p5_cuda(eps + r1_9 * (b))
             //
             ,
             //
             (r1_3 * x) / Power1p5_cuda(eps + Power2_cuda(-r2 * Abs(y0 - r1p5 * y1 + r4 * y2 - r1p833 * y3) +
                                                          Abs(y0 - r1 * y1 - r1 * y2 + y3)))
             //
    );

    real_type den = alpha[0] + alpha[1];

    // printf("alpha[0]=%f, alpha[1]=%f, den=%f\n", alpha[0], alpha[1], den);

    non_linear_weights[0] = alpha[0] / den;
    non_linear_weights[1] = alpha[1] / den;
}

/**
 * @brief WENO4 interpolation for constant grid spacing and h = 1
 *
 * @param x
 * @param y0
 * @param y1
 * @param y2
 * @param y3
 * @return real_type
 */
__device__ real_type weno4_HOne_cuda(const real_type x,                         //
                                     const real_type y0, const real_type y1,    //
                                     const real_type y2, const real_type y3) {  //

    const real_type eps = 1e-6;

    real_type lagrange_poly_0[3];
    real_type lagrange_poly_1[3];
    real_type non_linear_weights[2];

    LagrangePolyArrayHOne(x, lagrange_poly_0, lagrange_poly_1);

    getNonLinearWeightsHOne_cuda(x, y0, y1, y2, y3, non_linear_weights, eps);
    // getNonLinearWeightsConstH(x, 1.0, y0, y1, y2, y3, non_linear_weights, eps);

    const real_type weno4_a =
            (lagrange_poly_0[0] * y0 + lagrange_poly_0[1] * y1 + lagrange_poly_0[2] * y2) * non_linear_weights[0];

    const real_type weno4_b =
            (lagrange_poly_1[0] * y1 + lagrange_poly_1[1] * y2 + lagrange_poly_1[2] * y3) * non_linear_weights[1];

    return weno4_a + weno4_b;
}

/**
 * @brief WENO4 interpolation for 2D data with constant grid spacing and h = 1
 *
 * @param x
 * @param y
 * @param y00
 * @param y10
 * @param y20
 * @param y30
 * @param y01
 * @param y11
 * @param y21
 * @param y31
 * @param y02
 * @param y12
 * @param y22
 * @param y32
 * @param y03
 * @param y13
 * @param y23
 * @param y33
 * @return real_type
 */
__device__ real_type                                                                                    //
weno4_2D_HOne_cuda(const real_type x, const real_type y,                                                //
                                                                                                        //
                   const real_type y00, const real_type y10, const real_type y20, const real_type y30,  //
                   //
                   const real_type y01, const real_type y11, const real_type y21, const real_type y31,  //
                   //
                   const real_type y02, const real_type y12, const real_type y22, const real_type y32,  //
                   //
                   const real_type y03, const real_type y13, const real_type y23, const real_type y33) {  //

    real_type yw0 = weno4_HOne_cuda(x, y00, y10, y20, y30);
    real_type yw1 = weno4_HOne_cuda(x, y01, y11, y21, y31);
    real_type yw2 = weno4_HOne_cuda(x, y02, y12, y22, y32);
    real_type yw3 = weno4_HOne_cuda(x, y03, y13, y23, y33);

    real_type yw = weno4_HOne_cuda(y, yw0, yw1, yw2, yw3);

    return yw;
}

/**
 * @brief WENO4 interpolation for 3D data with constant grid spacing and h = 1
 *
 * @param x
 * @param y
 * @param z
 * @param f
 * @param stride_x
 * @param stride_y
 * @param stride_z
 * @return real_type
 */
__device__ real_type                           //
weno4_3D_HOne_cuda(const real_type  x,         //
                   const real_type  y,         //
                   const real_type  z,         //
                   const real_type *f,         //
                   const int        stride_x,  //
                   const int        stride_y,  //
                   const int        stride_z) {       //

    real_type w1 = weno4_2D_HOne_cuda(x,
                                      y,
                                      f[0 * stride_x + 0 * stride_y + 0 * stride_z],
                                      f[1 * stride_x + 0 * stride_y + 0 * stride_z],
                                      f[2 * stride_x + 0 * stride_y + 0 * stride_z],
                                      f[3 * stride_x + 0 * stride_y + 0 * stride_z],
                                      //
                                      f[0 * stride_x + 1 * stride_y + 0 * stride_z],
                                      f[1 * stride_x + 1 * stride_y + 0 * stride_z],
                                      f[2 * stride_x + 1 * stride_y + 0 * stride_z],
                                      f[3 * stride_x + 1 * stride_y + 0 * stride_z],
                                      //
                                      f[0 * stride_x + 2 * stride_y + 0 * stride_z],
                                      f[1 * stride_x + 2 * stride_y + 0 * stride_z],
                                      f[2 * stride_x + 2 * stride_y + 0 * stride_z],
                                      f[3 * stride_x + 2 * stride_y + 0 * stride_z],
                                      //
                                      f[0 * stride_x + 3 * stride_y + 0 * stride_z],
                                      f[1 * stride_x + 3 * stride_y + 0 * stride_z],
                                      f[2 * stride_x + 3 * stride_y + 0 * stride_z],
                                      f[3 * stride_x + 3 * stride_y + 0 * stride_z]);

    real_type w2 = weno4_2D_HOne_cuda(x,
                                      y,  //
                                      f[0 * stride_x + 0 * stride_y + 1 * stride_z],
                                      f[1 * stride_x + 0 * stride_y + 1 * stride_z],
                                      f[2 * stride_x + 0 * stride_y + 1 * stride_z],
                                      f[3 * stride_x + 0 * stride_y + 1 * stride_z],
                                      //
                                      f[0 * stride_x + 1 * stride_y + 1 * stride_z],
                                      f[1 * stride_x + 1 * stride_y + 1 * stride_z],
                                      f[2 * stride_x + 1 * stride_y + 1 * stride_z],
                                      f[3 * stride_x + 1 * stride_y + 1 * stride_z],
                                      //
                                      f[0 * stride_x + 2 * stride_y + 1 * stride_z],
                                      f[1 * stride_x + 2 * stride_y + 1 * stride_z],
                                      f[2 * stride_x + 2 * stride_y + 1 * stride_z],
                                      f[3 * stride_x + 2 * stride_y + 1 * stride_z],
                                      //
                                      f[0 * stride_x + 3 * stride_y + 1 * stride_z],
                                      f[1 * stride_x + 3 * stride_y + 1 * stride_z],
                                      f[2 * stride_x + 3 * stride_y + 1 * stride_z],
                                      f[3 * stride_x + 3 * stride_y + 1 * stride_z]);

    real_type w3 = weno4_2D_HOne_cuda(x,
                                      y,  //
                                      f[0 * stride_x + 0 * stride_y + 2 * stride_z],
                                      f[1 * stride_x + 0 * stride_y + 2 * stride_z],
                                      f[2 * stride_x + 0 * stride_y + 2 * stride_z],
                                      f[3 * stride_x + 0 * stride_y + 2 * stride_z],
                                      //
                                      f[0 * stride_x + 1 * stride_y + 2 * stride_z],
                                      f[1 * stride_x + 1 * stride_y + 2 * stride_z],
                                      f[2 * stride_x + 1 * stride_y + 2 * stride_z],
                                      f[3 * stride_x + 1 * stride_y + 2 * stride_z],
                                      //
                                      f[0 * stride_x + 2 * stride_y + 2 * stride_z],
                                      f[1 * stride_x + 2 * stride_y + 2 * stride_z],
                                      f[2 * stride_x + 2 * stride_y + 2 * stride_z],
                                      f[3 * stride_x + 2 * stride_y + 2 * stride_z],
                                      //
                                      f[0 * stride_x + 3 * stride_y + 2 * stride_z],
                                      f[1 * stride_x + 3 * stride_y + 2 * stride_z],
                                      f[2 * stride_x + 3 * stride_y + 2 * stride_z],
                                      f[3 * stride_x + 3 * stride_y + 2 * stride_z]);

    real_type w4 = weno4_2D_HOne_cuda(x,
                                      y,  //
                                      f[0 * stride_x + 0 * stride_y + 3 * stride_z],
                                      f[1 * stride_x + 0 * stride_y + 3 * stride_z],
                                      f[2 * stride_x + 0 * stride_y + 3 * stride_z],
                                      f[3 * stride_x + 0 * stride_y + 3 * stride_z],
                                      //
                                      f[0 * stride_x + 1 * stride_y + 3 * stride_z],
                                      f[1 * stride_x + 1 * stride_y + 3 * stride_z],
                                      f[2 * stride_x + 1 * stride_y + 3 * stride_z],
                                      f[3 * stride_x + 1 * stride_y + 3 * stride_z],
                                      //
                                      f[0 * stride_x + 2 * stride_y + 3 * stride_z],
                                      f[1 * stride_x + 2 * stride_y + 3 * stride_z],
                                      f[2 * stride_x + 2 * stride_y + 3 * stride_z],
                                      f[3 * stride_x + 2 * stride_y + 3 * stride_z],
                                      //
                                      f[0 * stride_x + 3 * stride_y + 3 * stride_z],
                                      f[1 * stride_x + 3 * stride_y + 3 * stride_z],
                                      f[2 * stride_x + 3 * stride_y + 3 * stride_z],
                                      f[3 * stride_x + 3 * stride_y + 3 * stride_z]);

    real_type wz = weno4_HOne_cuda(z, w1, w2, w3, w4);

    return wz;
}

#endif  // __TET10_WENO_CUDA_H__