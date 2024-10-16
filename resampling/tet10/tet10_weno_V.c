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

    // const vec_double h = 1.0;

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
    // const vec_double h = 1.0;

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
 * @return vec_double
 */
vec_double weno4_2D_HOne_V(const vec_double x, const vec_double y,  //
                                                                    //
                           const vec_double y00, const vec_double y10, const vec_double y20,
                           const vec_double y30,  //
                           //
                           const vec_double y01, const vec_double y11, const vec_double y21,
                           const vec_double y31,  //
                           //
                           const vec_double y02, const vec_double y12, const vec_double y22,
                           const vec_double y32,  //
                           //
                           const vec_double y03, const vec_double y13, const vec_double y23,
                           const vec_double y33) {  //

    vec_double yw0 = weno4_HOne_V(x, y00, y10, y20, y30);
    vec_double yw1 = weno4_HOne_V(x, y01, y11, y21, y31);
    vec_double yw2 = weno4_HOne_V(x, y02, y12, y22, y32);
    vec_double yw3 = weno4_HOne_V(x, y03, y13, y23, y33);

    vec_double yw = weno4_HOne_V(y, yw0, yw1, yw2, yw3);

    return yw;
}

vec_indices copy_strides(const vec_indices stride_x,  //
                         const vec_indices stride_y,  //
                         const vec_indices stride_z,  //
                         const int side_x, const int side_y, const int side_z) {
    vec_indices ret = CONST_VEC(0);

    for (int ii = 0; ii < _VL_; ii++) {
        ret[ii] = stride_x[ii] * side_x + stride_y[ii] * side_y + stride_z[ii] * side_z;
    }

    return ret;
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
                           const double *f,                                             //
                           const vec_indices stride_x,                                  //
                           const vec_indices stride_y,                                  //
                           const vec_indices stride_z) {                                //

    vec_double ret = ZEROS_VEC;
    vec_double w1, w2, w3, w4;

    {
        vec_indices stride_x_000 = copy_strides(stride_x, stride_y, stride_z, 0, 0, 0);
        vec_indices stride_x_100 = copy_strides(stride_x, stride_y, stride_z, 1, 0, 0);
        vec_indices stride_x_200 = copy_strides(stride_x, stride_y, stride_z, 2, 0, 0);
        vec_indices stride_x_300 = copy_strides(stride_x, stride_y, stride_z, 3, 0, 0);

        vec_indices stride_x_010 = copy_strides(stride_x, stride_y, stride_z, 0, 1, 0);
        vec_indices stride_x_110 = copy_strides(stride_x, stride_y, stride_z, 1, 1, 0);
        vec_indices stride_x_210 = copy_strides(stride_x, stride_y, stride_z, 2, 1, 0);
        vec_indices stride_x_310 = copy_strides(stride_x, stride_y, stride_z, 3, 1, 0);

        vec_indices stride_x_020 = copy_strides(stride_x, stride_y, stride_z, 0, 2, 0);
        vec_indices stride_x_120 = copy_strides(stride_x, stride_y, stride_z, 1, 2, 0);
        vec_indices stride_x_220 = copy_strides(stride_x, stride_y, stride_z, 2, 2, 0);
        vec_indices stride_x_320 = copy_strides(stride_x, stride_y, stride_z, 3, 2, 0);

        vec_indices stride_x_030 = copy_strides(stride_x, stride_y, stride_z, 0, 3, 0);
        vec_indices stride_x_130 = copy_strides(stride_x, stride_y, stride_z, 1, 3, 0);
        vec_indices stride_x_230 = copy_strides(stride_x, stride_y, stride_z, 2, 3, 0);
        vec_indices stride_x_330 = copy_strides(stride_x, stride_y, stride_z, 3, 3, 0);

        w1 = weno4_2D_HOne_V(x,
                             y,  //
                             stride_x_000,
                             stride_x_100,
                             stride_x_200,
                             stride_x_300,  //
                             //
                             stride_x_010,
                             stride_x_110,
                             stride_x_210,
                             stride_x_310,  //
                             //
                             stride_x_020,
                             stride_x_120,
                             stride_x_220,
                             stride_x_320,  //
                             //
                             stride_x_030,
                             stride_x_130,
                             stride_x_230,
                             stride_x_330);
    }

    {
        vec_indices stride_x_001 = copy_strides(stride_x, stride_y, stride_z, 0, 0, 1);
        vec_indices stride_x_101 = copy_strides(stride_x, stride_y, stride_z, 1, 0, 1);
        vec_indices stride_x_201 = copy_strides(stride_x, stride_y, stride_z, 2, 0, 1);
        vec_indices stride_x_301 = copy_strides(stride_x, stride_y, stride_z, 3, 0, 1);

        vec_indices stride_x_011 = copy_strides(stride_x, stride_y, stride_z, 0, 1, 1);
        vec_indices stride_x_111 = copy_strides(stride_x, stride_y, stride_z, 1, 1, 1);
        vec_indices stride_x_211 = copy_strides(stride_x, stride_y, stride_z, 2, 1, 1);
        vec_indices stride_x_311 = copy_strides(stride_x, stride_y, stride_z, 3, 1, 1);

        vec_indices stride_x_021 = copy_strides(stride_x, stride_y, stride_z, 0, 2, 1);
        vec_indices stride_x_121 = copy_strides(stride_x, stride_y, stride_z, 1, 2, 1);
        vec_indices stride_x_221 = copy_strides(stride_x, stride_y, stride_z, 2, 2, 1);
        vec_indices stride_x_321 = copy_strides(stride_x, stride_y, stride_z, 3, 2, 1);

        vec_indices stride_x_031 = copy_strides(stride_x, stride_y, stride_z, 0, 3, 1);
        vec_indices stride_x_131 = copy_strides(stride_x, stride_y, stride_z, 1, 3, 1);
        vec_indices stride_x_231 = copy_strides(stride_x, stride_y, stride_z, 2, 3, 1);
        vec_indices stride_x_331 = copy_strides(stride_x, stride_y, stride_z, 3, 3, 1);

        w2 = weno4_2D_HOne_V(x,
                             y,
                             stride_x_001,
                             stride_x_101,
                             stride_x_201,
                             stride_x_301,  //
                             //
                             stride_x_011,
                             stride_x_111,
                             stride_x_211,
                             stride_x_311,  //
                             //
                             stride_x_021,
                             stride_x_121,
                             stride_x_221,
                             stride_x_321,  //
                             //
                             stride_x_031,
                             stride_x_131,
                             stride_x_231,
                             stride_x_331);
    }

    {
        vec_indices stride_x_002 = copy_strides(stride_x, stride_y, stride_z, 0, 0, 2);
        vec_indices stride_x_102 = copy_strides(stride_x, stride_y, stride_z, 1, 0, 2);
        vec_indices stride_x_202 = copy_strides(stride_x, stride_y, stride_z, 2, 0, 2);
        vec_indices stride_x_302 = copy_strides(stride_x, stride_y, stride_z, 3, 0, 2);

        vec_indices stride_x_012 = copy_strides(stride_x, stride_y, stride_z, 0, 1, 2);
        vec_indices stride_x_112 = copy_strides(stride_x, stride_y, stride_z, 1, 1, 2);
        vec_indices stride_x_212 = copy_strides(stride_x, stride_y, stride_z, 2, 1, 2);
        vec_indices stride_x_312 = copy_strides(stride_x, stride_y, stride_z, 3, 1, 2);

        vec_indices stride_x_022 = copy_strides(stride_x, stride_y, stride_z, 0, 2, 2);
        vec_indices stride_x_122 = copy_strides(stride_x, stride_y, stride_z, 1, 2, 2);
        vec_indices stride_x_222 = copy_strides(stride_x, stride_y, stride_z, 2, 2, 2);
        vec_indices stride_x_322 = copy_strides(stride_x, stride_y, stride_z, 3, 2, 2);

        vec_indices stride_x_032 = copy_strides(stride_x, stride_y, stride_z, 0, 3, 2);
        vec_indices stride_x_132 = copy_strides(stride_x, stride_y, stride_z, 1, 3, 2);
        vec_indices stride_x_232 = copy_strides(stride_x, stride_y, stride_z, 2, 3, 2);
        vec_indices stride_x_332 = copy_strides(stride_x, stride_y, stride_z, 3, 3, 2);

        w3 = weno4_2D_HOne_V(x,
                             y,  //
                             stride_x_002,
                             stride_x_102,
                             stride_x_202,
                             stride_x_302,  //
                             //
                             stride_x_012,
                             stride_x_112,
                             stride_x_212,
                             stride_x_312,  //
                             //
                             stride_x_022,
                             stride_x_122,
                             stride_x_222,
                             stride_x_322,  //
                             //
                             stride_x_032,
                             stride_x_132,
                             stride_x_232,
                             stride_x_332);
    }

    {
        vec_indices stride_x_003 = copy_strides(stride_x, stride_y, stride_z, 0, 0, 3);
        vec_indices stride_x_103 = copy_strides(stride_x, stride_y, stride_z, 1, 0, 3);
        vec_indices stride_x_203 = copy_strides(stride_x, stride_y, stride_z, 2, 0, 3);
        vec_indices stride_x_303 = copy_strides(stride_x, stride_y, stride_z, 3, 0, 3);

        vec_indices stride_x_013 = copy_strides(stride_x, stride_y, stride_z, 0, 1, 3);
        vec_indices stride_x_113 = copy_strides(stride_x, stride_y, stride_z, 1, 1, 3);
        vec_indices stride_x_213 = copy_strides(stride_x, stride_y, stride_z, 2, 1, 3);
        vec_indices stride_x_313 = copy_strides(stride_x, stride_y, stride_z, 3, 1, 3);

        vec_indices stride_x_023 = copy_strides(stride_x, stride_y, stride_z, 0, 2, 3);
        vec_indices stride_x_123 = copy_strides(stride_x, stride_y, stride_z, 1, 2, 3);
        vec_indices stride_x_223 = copy_strides(stride_x, stride_y, stride_z, 2, 2, 3);
        vec_indices stride_x_323 = copy_strides(stride_x, stride_y, stride_z, 3, 2, 3);

        vec_indices stride_x_033 = copy_strides(stride_x, stride_y, stride_z, 0, 3, 3);
        vec_indices stride_x_133 = copy_strides(stride_x, stride_y, stride_z, 1, 3, 3);
        vec_indices stride_x_233 = copy_strides(stride_x, stride_y, stride_z, 2, 3, 3);
        vec_indices stride_x_333 = copy_strides(stride_x, stride_y, stride_z, 3, 3, 3);

        w4 = weno4_2D_HOne_V(x,  //
                             y,  //
                             stride_x_003,
                             stride_x_103,
                             stride_x_203,
                             stride_x_303,  //
                             //
                             stride_x_013,
                             stride_x_113,
                             stride_x_213,
                             stride_x_313,  //
                             //
                             stride_x_023,
                             stride_x_123,
                             stride_x_223,
                             stride_x_323,  //
                             //
                             stride_x_033,
                             stride_x_133,
                             stride_x_233,
                             stride_x_333);
    }

    vec_double wz = weno4_HOne_V(z, w1, w2, w3, w4);

    return wz;
}