#include "tet10_weno_V.h"

/**
 * @brief Compute the square of a vector
 *
 * @param x
 * @return vec_double
 */
SFEM_INLINE static
vec_double Power2_V(const vec_double x) { return x * x; }

/**
 * @brief Compute the absolute value of a vector
 *
 * @param __X__
 * @return vec_double
 */
SFEM_INLINE static
vec_double Abs_V(const vec_double __X__) {
    vec_double __R__;
    for (int ii = 0; ii < _VL_; ii++) {
        __R__[ii] = fabs(__X__[ii]);
    }
    return __R__;
}

SFEM_INLINE static
vec_double Power1p5_V(const vec_double x) {
    vec_double sqrt_x;
    for (int ii = 0; ii < _VL_; ii++) {
        sqrt_x[ii] = sqrt(x[ii]);
    }

    return x * sqrt_x;
}

SFEM_INLINE static
vec_double Power_m1p5_V(const vec_double x) {
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
SFEM_INLINE static
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
SFEM_INLINE static
void getNonLinearWeightsHOne_V(const vec_double x,                        //
                               const vec_double y0, const vec_double y1,  //
                               const vec_double y2, const vec_double y3,  //
                               vec_double *non_linear_weights,            //
                               const vec_double eps) {                    //

    vec_double alpha[2];
    // const vec_double h = 1.0;

    const vec_double a = Abs_V(3. * y0 - 7. * y1 + 5. * y2 - 1. * y3);
    const vec_double b = Power2_V(-3. * (a) + Abs_V(y0 - 12. * y1 + 3. * y2 + 2. * y3));

    List2_V(alpha,
            //
            Power_m1p5_V(eps + 0.1111111111111111 * (b)) -
                    (0.3333333333333333 * x) / Power1p5_V(eps + 0.1111111111111111 * (b))
            //
            ,
            //
            (0.3333333333333333 * x) /
                    Power1p5_V(eps + Power2_V(-2. * Abs_V(y0 - 1.5 * y1 + 4. * y2 -
                                                          1.8333333333333333 * y3) +
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
SFEM_INLINE static
vec_double weno4_HOne_V(const vec_double x,                          //
                        const vec_double y0, const vec_double y1,    //
                        const vec_double y2, const vec_double y3) {  //

    const vec_double eps = CONST_VEC(1e-6);

    vec_double lagrange_poly_0[3];
    vec_double lagrange_poly_1[3];
    vec_double non_linear_weights[2];

    LagrangePolyArrayHOne_V(x, lagrange_poly_0, lagrange_poly_1);

    getNonLinearWeightsHOne_V(x, y0, y1, y2, y3, non_linear_weights, eps);
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
SFEM_INLINE static
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

SFEM_INLINE static
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

SFEM_INLINE static
vec_double copy_f(const double *f,             //
                  const vec_indices stride_x,  //
                  const vec_indices stride_y,  //
                  const vec_indices stride_z,  //
                  const int side_x, const int side_y, const int side_z) {
    //
    vec_double ret;
    vec_indices indx = stride_x * side_x + stride_y * side_y + stride_z * side_z;

    for (int ii = 0; ii < _VL_; ii++) {
        ret[ii] = f[indx[ii]];
    }

    return ret;
}

/**
 * @brief Compute the first index of the field for third order interpolation
 *
 * @param stride
 * @param i
 * @param j
 * @param k
 * @return SFEM_INLINE
 */
SFEM_INLINE static
vec_indices hex_aa_8_indices_O3_first_index_vec(const ptrdiff_t *const stride,             //
                                                const vec_indices i, const vec_indices j,  //
                                                const vec_indices k) {                     //
    //
    vec_indices ret = CONST_VEC(0);

    for (int ii = 0; ii < _VL_; ii++) {
        ret[ii] = (i[ii] - 1) * stride[0] + (j[ii] - 1) * stride[1] + (k[ii] - 1) * stride[2];
    }

    return ret;
}

/**
 * @brief Compute the coefficients of the field for third order interpolation
 *
 * @param stride
 * @param i
 * @param j
 * @param k
 * @param data
 * @param out
 * @return SFEM_INLINE
 */
void hex_aa_8_collect_coeffs_O3_ptr_vec(const ptrdiff_t *const stride,  //
                                        const vec_indices i,            //
                                        const vec_indices j,            //
                                        const vec_indices k,            //
                                        const real_t *const data,
                                        real_t *first_ptrs_array[]) {  //

    const vec_indices first_indices = hex_aa_8_indices_O3_first_index_vec(stride, i, j, k);

    for (int ii = 0; ii < _VL_; ii++) {
        first_ptrs_array[ii] = &data[first_indices[ii]];
    }
}

/**
 * @brief Copy a vector of pointers to a vector of pointers
 * 
 * @param f 
 * @param stride_x 
 * @param stride_y 
 * @param stride_z 
 * @param side_x 
 * @param side_y 
 * @param side_z 
 * @return vec_double 
 */
SFEM_INLINE static
vec_double copy_f_vec(const real_t *f[],         //
                      const ptrdiff_t stride_x,  //
                      const ptrdiff_t stride_y,  //
                      const ptrdiff_t stride_z,  //
                      const int side_x,          //
                      const int side_y,          //
                      const int side_z) {        //
    //
    vec_double res = CONST_VEC(0);

    for (int ii = 0; ii < _VL_; ii++) {
        res[ii] = f[ii][side_x * stride_x + side_y * stride_y + side_z * stride_z];
    }

    return res;
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
vec_double weno4_3D_HOne_V(const ptrdiff_t *const stride,                               //
                           const vec_double x, const vec_double y, const vec_double z,  //
                           const real_t *f[]) {                                         //

    const ptrdiff_t stride_x = stride[0];
    const ptrdiff_t stride_y = stride[1];
    const ptrdiff_t stride_z = stride[2];

    vec_double w1, w2, w3, w4;

    {  // begin Z = 0
        vec_double y00, y10, y20, y30;
        vec_double y01, y11, y21, y31;
        vec_double y02, y12, y22, y32;
        vec_double y03, y13, y23, y33;

        y00 = copy_f_vec(f, stride_x, stride_y, stride_z, 0, 0, 0);
        y10 = copy_f_vec(f, stride_x, stride_y, stride_z, 1, 0, 0);
        y20 = copy_f_vec(f, stride_x, stride_y, stride_z, 2, 0, 0);
        y30 = copy_f_vec(f, stride_x, stride_y, stride_z, 3, 0, 0);

        y01 = copy_f_vec(f, stride_x, stride_y, stride_z, 0, 1, 0);
        y11 = copy_f_vec(f, stride_x, stride_y, stride_z, 1, 1, 0);
        y21 = copy_f_vec(f, stride_x, stride_y, stride_z, 2, 1, 0);
        y31 = copy_f_vec(f, stride_x, stride_y, stride_z, 3, 1, 0);

        y02 = copy_f_vec(f, stride_x, stride_y, stride_z, 0, 2, 0);
        y12 = copy_f_vec(f, stride_x, stride_y, stride_z, 1, 2, 0);
        y22 = copy_f_vec(f, stride_x, stride_y, stride_z, 2, 2, 0);
        y32 = copy_f_vec(f, stride_x, stride_y, stride_z, 3, 2, 0);

        y03 = copy_f_vec(f, stride_x, stride_y, stride_z, 0, 3, 0);
        y13 = copy_f_vec(f, stride_x, stride_y, stride_z, 1, 3, 0);
        y23 = copy_f_vec(f, stride_x, stride_y, stride_z, 2, 3, 0);
        y33 = copy_f_vec(f, stride_x, stride_y, stride_z, 3, 3, 0);

        w1 = weno4_2D_HOne_V(x,
                             y,
                             y00,
                             y10,
                             y20,
                             y30,
                             y01,
                             y11,
                             y21,
                             y31,
                             y02,
                             y12,
                             y22,
                             y32,
                             y03,
                             y13,
                             y23,
                             y33);
    }  // end Z = 0

    {  // begin Z = 1

        vec_double y00, y10, y20, y30;
        vec_double y01, y11, y21, y31;
        vec_double y02, y12, y22, y32;
        vec_double y03, y13, y23, y33;

        y00 = copy_f_vec(f, stride_x, stride_y, stride_z, 0, 0, 1);
        y10 = copy_f_vec(f, stride_x, stride_y, stride_z, 1, 0, 1);
        y20 = copy_f_vec(f, stride_x, stride_y, stride_z, 2, 0, 1);
        y30 = copy_f_vec(f, stride_x, stride_y, stride_z, 3, 0, 1);

        y01 = copy_f_vec(f, stride_x, stride_y, stride_z, 0, 1, 1);
        y11 = copy_f_vec(f, stride_x, stride_y, stride_z, 1, 1, 1);
        y21 = copy_f_vec(f, stride_x, stride_y, stride_z, 2, 1, 1);
        y31 = copy_f_vec(f, stride_x, stride_y, stride_z, 3, 1, 1);

        y02 = copy_f_vec(f, stride_x, stride_y, stride_z, 0, 2, 1);
        y12 = copy_f_vec(f, stride_x, stride_y, stride_z, 1, 2, 1);
        y22 = copy_f_vec(f, stride_x, stride_y, stride_z, 2, 2, 1);
        y32 = copy_f_vec(f, stride_x, stride_y, stride_z, 3, 2, 1);

        y03 = copy_f_vec(f, stride_x, stride_y, stride_z, 0, 3, 1);
        y13 = copy_f_vec(f, stride_x, stride_y, stride_z, 1, 3, 1);
        y23 = copy_f_vec(f, stride_x, stride_y, stride_z, 2, 3, 1);
        y33 = copy_f_vec(f, stride_x, stride_y, stride_z, 3, 3, 1);

        w2 = weno4_2D_HOne_V(x,
                             y,
                             y00,
                             y10,
                             y20,
                             y30,
                             y01,
                             y11,
                             y21,
                             y31,
                             y02,
                             y12,
                             y22,
                             y32,
                             y03,
                             y13,
                             y23,
                             y33);
    }  // end Z = 1

    {  // begin Z = 2

        vec_double y00, y10, y20, y30;
        vec_double y01, y11, y21, y31;
        vec_double y02, y12, y22, y32;
        vec_double y03, y13, y23, y33;

        y00 = copy_f_vec(f, stride_x, stride_y, stride_z, 0, 0, 2);
        y10 = copy_f_vec(f, stride_x, stride_y, stride_z, 1, 0, 2);
        y20 = copy_f_vec(f, stride_x, stride_y, stride_z, 2, 0, 2);
        y30 = copy_f_vec(f, stride_x, stride_y, stride_z, 3, 0, 2);

        y01 = copy_f_vec(f, stride_x, stride_y, stride_z, 0, 1, 2);
        y11 = copy_f_vec(f, stride_x, stride_y, stride_z, 1, 1, 2);
        y21 = copy_f_vec(f, stride_x, stride_y, stride_z, 2, 1, 2);
        y31 = copy_f_vec(f, stride_x, stride_y, stride_z, 3, 1, 2);

        y02 = copy_f_vec(f, stride_x, stride_y, stride_z, 0, 2, 2);
        y12 = copy_f_vec(f, stride_x, stride_y, stride_z, 1, 2, 2);
        y22 = copy_f_vec(f, stride_x, stride_y, stride_z, 2, 2, 2);
        y32 = copy_f_vec(f, stride_x, stride_y, stride_z, 3, 2, 2);

        y03 = copy_f_vec(f, stride_x, stride_y, stride_z, 0, 3, 2);
        y13 = copy_f_vec(f, stride_x, stride_y, stride_z, 1, 3, 2);
        y23 = copy_f_vec(f, stride_x, stride_y, stride_z, 2, 3, 2);
        y33 = copy_f_vec(f, stride_x, stride_y, stride_z, 3, 3, 2);

        w3 = weno4_2D_HOne_V(x,
                             y,
                             y00,
                             y10,
                             y20,
                             y30,
                             y01,
                             y11,
                             y21,
                             y31,
                             y02,
                             y12,
                             y22,
                             y32,
                             y03,
                             y13,
                             y23,
                             y33);

    }  // end Z = 2

    {  // begin Z = 3

        vec_double y00, y10, y20, y30;
        vec_double y01, y11, y21, y31;
        vec_double y02, y12, y22, y32;
        vec_double y03, y13, y23, y33;

        y00 = copy_f_vec(f, stride_x, stride_y, stride_z, 0, 0, 3);
        y10 = copy_f_vec(f, stride_x, stride_y, stride_z, 1, 0, 3);
        y20 = copy_f_vec(f, stride_x, stride_y, stride_z, 2, 0, 3);
        y30 = copy_f_vec(f, stride_x, stride_y, stride_z, 3, 0, 3);

        y01 = copy_f_vec(f, stride_x, stride_y, stride_z, 0, 1, 3);
        y11 = copy_f_vec(f, stride_x, stride_y, stride_z, 1, 1, 3);
        y21 = copy_f_vec(f, stride_x, stride_y, stride_z, 2, 1, 3);
        y31 = copy_f_vec(f, stride_x, stride_y, stride_z, 3, 1, 3);

        y02 = copy_f_vec(f, stride_x, stride_y, stride_z, 0, 2, 3);
        y12 = copy_f_vec(f, stride_x, stride_y, stride_z, 1, 2, 3);
        y22 = copy_f_vec(f, stride_x, stride_y, stride_z, 2, 2, 3);
        y32 = copy_f_vec(f, stride_x, stride_y, stride_z, 3, 2, 3);

        y03 = copy_f_vec(f, stride_x, stride_y, stride_z, 0, 3, 3);
        y13 = copy_f_vec(f, stride_x, stride_y, stride_z, 1, 3, 3);
        y23 = copy_f_vec(f, stride_x, stride_y, stride_z, 2, 3, 3);
        y33 = copy_f_vec(f, stride_x, stride_y, stride_z, 3, 3, 3);

        w4 = weno4_2D_HOne_V(x,
                             y,
                             y00,
                             y10,
                             y20,
                             y30,
                             y01,
                             y11,
                             y21,
                             y31,
                             y02,
                             y12,
                             y22,
                             y32,
                             y03,
                             y13,
                             y23,
                             y33);

    }  // end Z = 3

    const vec_double wz = weno4_HOne_V(z, w1, w2, w3, w4);

    return wz;
}