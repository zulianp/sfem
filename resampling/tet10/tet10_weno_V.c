#include "tet10_weno_V.h"

/**
 * @brief Compute the square of a vector
 *
 * @param x
 * @return vec_double
 */
vec_double Power2_V(const vec_double x) { return x * x; }

/**
 * @brief Compute the absolute value of a vector
 *
 * @param __X__
 * @return vec_double
 */
vec_double Abs_V(const vec_double __X__) {
    vec_double __R__;
    for (int ii = 0; ii < _VL_; ii++) {
        __R__[ii] = fabs(__X__[ii]);
    }
    return __R__;
}

static SFEM_INLINE vec_double Power1p5_V(const vec_double x) {
    vec_double sqrt_x;
    for (int ii = 0; ii < _VL_; ii++) {
        sqrt_x[ii] = sqrt(x[ii]);
    }

    return x * sqrt_x;
}

static SFEM_INLINE vec_double Power_m1p5_V(const vec_double x) {
    vec_double sqrt_x;
    for (int ii = 0; ii < _VL_; ii++) {
        sqrt_x[ii] = sqrt(x[ii]);
    }

    return 1.0 / (x * sqrt_x);
}

/**
 * @brief Get the Lagrange Poly Array object by assuming the h is equal to 1
 *
 * @param x
 * @param lagrange_poly_0
 * @param lagrange_poly_1
 */
void LagrangePolyArrayHOne_V(const vec_double x,             //
                             vec_double *lagrange_poly_0,    //
                             vec_double *lagrange_poly_1) {  //

    // const double h = 1.0;

    const vec_double xx = Power2_V(x);

    List3_V(lagrange_poly_0,  //
            (2.0 - 3.0 * x + xx) / 2.,
            -((-2.0 + x) * x),
            ((-1.0 + x) * x) / 2.  //
    );                             //

    List3_V(lagrange_poly_1,  //
            (6.0 - 5.0 * x + xx) / 2.,
            -3.0 + 4.0 * x - xx,         //
            (2.0 - 3.0 * x + xx) / 2.);  //
}

/**
 * @brief Get the Non Linear Weights object by assuming the h is constant and equal to 1
 *
 * @param x
 * @param y0
 * @param y1
 * @param y2
 * @param y3
 * @param non_linear_weights
 * @param eps
 */
void getNonLinearWeightsHOne_V(const vec_double x,                        //
                               const vec_double y0, const vec_double y1,  //
                               const vec_double y2, const vec_double y3,  //
                               vec_double *non_linear_weights,            //
                               const vec_double eps) {                    //

    vec_double alpha[2];
    // const double h = 1.0;

    const vec_double a = Abs_V(3. * y0 - 7. * y1 + 5. * y2 - 1. * y3);
    const vec_double b = Power2_V(-3. * (a) + Abs_V(y0 - 12. * y1 + 3. * y2 + 2. * y3));

    List2(alpha,
          //
          Power_m1p5_V(eps + 0.1111111111111111 * (b)) -
                  (0.3333333333333333 * x) / Power1p5_V(eps + 0.1111111111111111 * (b))
          //
          ,
          //
          (0.3333333333333333 *
           x) / Power1p5_V(eps +
                           Power2_V(-2. * Abs_V(y0 - 1.5 * y1 + 4. * y2 - 1.8333333333333333 * y3) +
                                    Abs_V(y0 - 1. * y1 - 1. * y2 + y3)))
          //
    );

    vec_double den = alpha[0] + alpha[1];

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
 * @return vec_double
 */
vec_double weno4_HOne_V(const vec_double x,                          //
                        const vec_double y0, const vec_double y1,    //
                        const vec_double y2, const vec_double y3) {  //

    const vec_double eps = CONST_VEC(1e-6);

    vec_double lagrange_poly_0[3];
    vec_double lagrange_poly_1[3];
    vec_double non_linear_weights[2];

    LagrangePolyArrayHOne_V(x, lagrange_poly_0, lagrange_poly_1);

    getNonLinearWeightsHOne(x, y0, y1, y2, y3, non_linear_weights, eps);
    // getNonLinearWeightsConstH(x, 1.0, y0, y1, y2, y3, non_linear_weights, eps);

    const vec_double weno4_a =
            (lagrange_poly_0[0] * y0 + lagrange_poly_0[1] * y1 + lagrange_poly_0[2] * y2) *
            non_linear_weights[0];

    const vec_double weno4_b =
            (lagrange_poly_1[0] * y1 + lagrange_poly_1[1] * y2 + lagrange_poly_1[2] * y3) *
            non_linear_weights[1];

    return weno4_a + weno4_b;
}


double weno4_2D_HOne_V(const double x, const double y,                                          //
                                                                                              //
                     const double y00, const double y10, const double y20, const double y30,  //
                     //
                     const double y01, const double y11, const double y21, const double y31,  //
                     //
                     const double y02, const double y12, const double y22, const double y32,  //
                     //
                     const double y03, const double y13, const double y23, const double y33) {  //

    double yw0 = weno4_HOne(x, y00, y10, y20, y30);
    double yw1 = weno4_HOne(x, y01, y11, y21, y31);
    double yw2 = weno4_HOne(x, y02, y12, y22, y32);
    double yw3 = weno4_HOne(x, y03, y13, y23, y33);

    double yw = weno4_HOne(y, yw0, yw1, yw2, yw3);

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
 * @return vec_double 
 */
vec_double weno4_3D_HOne_V(const vec_double x, const vec_double y, const vec_double z,  //
                           const vec_double *f,                                         //
                           const int stride_x,                                          //
                           const int stride_y,                                          //
                           const int stride_z) {                                        //

    vec_double ret = ZEROS_VEC;

    // double w1 = weno4_2D_HOne(x,
    //                           y,  //
    //                           f[0 * stride_x + 0 * stride_y + 0 * stride_z],
    //                           f[1 * stride_x + 0 * stride_y + 0 * stride_z],
    //                           f[2 * stride_x + 0 * stride_y + 0 * stride_z],
    //                           f[3 * stride_x + 0 * stride_y + 0 * stride_z],
    //                           //
    //                           f[0 * stride_x + 1 * stride_y + 0 * stride_z],
    //                           f[1 * stride_x + 1 * stride_y + 0 * stride_z],
    //                           f[2 * stride_x + 1 * stride_y + 0 * stride_z],
    //                           f[3 * stride_x + 1 * stride_y + 0 * stride_z],
    //                           //
    //                           f[0 * stride_x + 2 * stride_y + 0 * stride_z],
    //                           f[1 * stride_x + 2 * stride_y + 0 * stride_z],
    //                           f[2 * stride_x + 2 * stride_y + 0 * stride_z],
    //                           f[3 * stride_x + 2 * stride_y + 0 * stride_z],
    //                           //
    //                           f[0 * stride_x + 3 * stride_y + 0 * stride_z],
    //                           f[1 * stride_x + 3 * stride_y + 0 * stride_z],
    //                           f[2 * stride_x + 3 * stride_y + 0 * stride_z],
    //                           f[3 * stride_x + 3 * stride_y + 0 * stride_z]);

    // double w2 = weno4_2D_HOne(x,
    //                           y,  //
    //                           f[0 * stride_x + 0 * stride_y + 1 * stride_z],
    //                           f[1 * stride_x + 0 * stride_y + 1 * stride_z],
    //                           f[2 * stride_x + 0 * stride_y + 1 * stride_z],
    //                           f[3 * stride_x + 0 * stride_y + 1 * stride_z],
    //                           //
    //                           f[0 * stride_x + 1 * stride_y + 1 * stride_z],
    //                           f[1 * stride_x + 1 * stride_y + 1 * stride_z],
    //                           f[2 * stride_x + 1 * stride_y + 1 * stride_z],
    //                           f[3 * stride_x + 1 * stride_y + 1 * stride_z],
    //                           //
    //                           f[0 * stride_x + 2 * stride_y + 1 * stride_z],
    //                           f[1 * stride_x + 2 * stride_y + 1 * stride_z],
    //                           f[2 * stride_x + 2 * stride_y + 1 * stride_z],
    //                           f[3 * stride_x + 2 * stride_y + 1 * stride_z],
    //                           //
    //                           f[0 * stride_x + 3 * stride_y + 1 * stride_z],
    //                           f[1 * stride_x + 3 * stride_y + 1 * stride_z],
    //                           f[2 * stride_x + 3 * stride_y + 1 * stride_z],
    //                           f[3 * stride_x + 3 * stride_y + 1 * stride_z]);

    // double w3 = weno4_2D_HOne(x,
    //                           y,  //
    //                           f[0 * stride_x + 0 * stride_y + 2 * stride_z],
    //                           f[1 * stride_x + 0 * stride_y + 2 * stride_z],
    //                           f[2 * stride_x + 0 * stride_y + 2 * stride_z],
    //                           f[3 * stride_x + 0 * stride_y + 2 * stride_z],
    //                           //
    //                           f[0 * stride_x + 1 * stride_y + 2 * stride_z],
    //                           f[1 * stride_x + 1 * stride_y + 2 * stride_z],
    //                           f[2 * stride_x + 1 * stride_y + 2 * stride_z],
    //                           f[3 * stride_x + 1 * stride_y + 2 * stride_z],
    //                           //
    //                           f[0 * stride_x + 2 * stride_y + 2 * stride_z],
    //                           f[1 * stride_x + 2 * stride_y + 2 * stride_z],
    //                           f[2 * stride_x + 2 * stride_y + 2 * stride_z],
    //                           f[3 * stride_x + 2 * stride_y + 2 * stride_z],
    //                           //
    //                           f[0 * stride_x + 3 * stride_y + 2 * stride_z],
    //                           f[1 * stride_x + 3 * stride_y + 2 * stride_z],
    //                           f[2 * stride_x + 3 * stride_y + 2 * stride_z],
    //                           f[3 * stride_x + 3 * stride_y + 2 * stride_z]);

    // double w4 = weno4_2D_HOne(x,
    //                           y,  //
    //                           f[0 * stride_x + 0 * stride_y + 3 * stride_z],
    //                           f[1 * stride_x + 0 * stride_y + 3 * stride_z],
    //                           f[2 * stride_x + 0 * stride_y + 3 * stride_z],
    //                           f[3 * stride_x + 0 * stride_y + 3 * stride_z],
    //                           //
    //                           f[0 * stride_x + 1 * stride_y + 3 * stride_z],
    //                           f[1 * stride_x + 1 * stride_y + 3 * stride_z],
    //                           f[2 * stride_x + 1 * stride_y + 3 * stride_z],
    //                           f[3 * stride_x + 1 * stride_y + 3 * stride_z],
    //                           //
    //                           f[0 * stride_x + 2 * stride_y + 3 * stride_z],
    //                           f[1 * stride_x + 2 * stride_y + 3 * stride_z],
    //                           f[2 * stride_x + 2 * stride_y + 3 * stride_z],
    //                           f[3 * stride_x + 2 * stride_y + 3 * stride_z],
    //                           //
    //                           f[0 * stride_x + 3 * stride_y + 3 * stride_z],
    //                           f[1 * stride_x + 3 * stride_y + 3 * stride_z],
    //                           f[2 * stride_x + 3 * stride_y + 3 * stride_z],
    //                           f[3 * stride_x + 3 * stride_y + 3 * stride_z]);

    // double wz = weno4_HOne(z, w1, w2, w3, w4);

    // return wz;
}