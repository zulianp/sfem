
#include <stdio.h>

#include "tet10_weno_cuda.h"

/**
 * @brief Compute the power of x raised to the power of y
 *
 * @param x
 * @param y
 * @return real_type
 */
__device__ real_type Power(const real_type x, const real_type y) { return pow(x, y); }

/**
 * @brief Compute the power of x raised to the power of 2
 *
 * @param x
 * @return real_type
 */
__device__ real_type Power2(const real_type x) { return x * x; }

/**
 * @brief Compute the power of x raised to the power of 1.5
 *
 * @param x
 * @return real_type
 */
__device__ real_type Power1p5(const real_type x) { return x * sqrt(x); }

/**
 * @brief Compute the power of x raised to the power of -1.5
 *
 * @param x
 * @return real_type
 */
__device__ real_type Power_m1p5(const real_type x) { return 1.0 / (x * sqrt(x)); }

/**
 * @brief Compute the Lagrange Polynomials of order 2 at x for h constant
 *
 * @param x
 * @return real_type
 */
__device__ void LagrangePolyArrayConstH(const real_type x, const real_type h,  //
                                        real_type *lagrange_poly_0,
                                        real_type *lagrange_poly_1) {  //

    List3_cu(lagrange_poly_0,                               //
             ((-2 * h + x) * (-h + x)) / (2. * Power2(h)),  //
             -((x * (-2 * h + x)) / Power2(h)),             //
             (x * (-h + x)) / (2. * Power2(h)));            //

    List3_cu(lagrange_poly_1,                                   //
             ((-3 * h + x) * (-2 * h + x)) / (2. * Power2(h)),  //
             -(((-3 * h + x) * (-h + x)) / Power2(h)),          //
             ((-2 * h + x) * (-h + x)) / (2. * Power2(h)));     //
}

/**
 * @brief Get the Linear Weights object by assuming the stencil starts at 0 and
 * h is constant
 *
 * @param x : value at which the weights are to be calculated
 * @param h : grid spacing
 * @param linear_weights : array to store the linear weights of size 2
 */
__device__ void getLinearWeightsConstH(const real_type x, const real_type h,
                                       real_type *linear_weights) {
    //
    linear_weights[0] = (3 * h - x) / (3. * h);
    linear_weights[1] = x / (3. * h);
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
__device__ void getNonLinearWeightsConstH(const real_type x, const real_type h,    //
                                          const real_type y0, const real_type y1,  //
                                          const real_type y2, const real_type y3,  //
                                          real_type *non_linear_weights,           //
                                          const real_type eps) {                   //

    real_type alpha[2];

    const real_type inv_h = 1. / h;

    const real_type a = -3. * y0 + 7. * y1 - 5. * y2 + y3;
    const real_type b = y0 - 6. * (1. + h) * y1 + 3. * y2 + 2. * y3;

    List2_cu(alpha,
             // alpha[0]
             Power_m1p5(eps + 0.1111111111111111 * Power2(-3. * Abs((a)*inv_h) + Abs((b)*inv_h))) -
                     (0.3333333333333333 * x) /
                             (h * Power1p5(eps + 0.1111111111111111 * Power2(-3. * Abs((a)*inv_h) +
                                                                             Abs((b)*inv_h)))),
             //
             // alpha[1]
             (0.3333333333333333 * x) /
                     (h * Power1p5(eps + 0.1111111111111111 *  //
                                                 Power2(Abs((6. * y0 - 9. * y1 + 18. * y2 +
                                                             6. * h * y2 - 11. * y3) *
                                                            inv_h) -
                                                        3. * Abs((y0 - 1. * y1 - 1. * y2 + y3) *
                                                                 inv_h)  //  end of Abs
                                                        )                // end of Power2
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
__device__ real_type weno4ConstH(const real_type x, const real_type h,                        //
                                 const real_type y0, const real_type y1, const real_type y2,  //
                                 const real_type y3) {                                        //
    const real_type eps = 1e-6;

    real_type lagrange_poly_0[3];
    real_type lagrange_poly_1[3];
    real_type non_linear_weights[2];

    LagrangePolyArrayConstH(x, h, lagrange_poly_0, lagrange_poly_1);

    getNonLinearWeightsConstH(x, h, y0, y1, y2, y3, non_linear_weights, eps);

    const real_type weno4_a =
            (lagrange_poly_0[0] * y0 + lagrange_poly_0[1] * y1 + lagrange_poly_0[2] * y2) *
            non_linear_weights[0];

    const real_type weno4_b =
            (lagrange_poly_1[0] * y1 + lagrange_poly_1[1] * y2 + lagrange_poly_1[2] * y3) *
            non_linear_weights[1];

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
__device__ real_type weno4_2D_ConstH(
        const real_type x, const real_type y, const real_type h,                             //
                                                                                             //
        const real_type y00, const real_type y10, const real_type y20, const real_type y30,  //
        //
        const real_type y01, const real_type y11, const real_type y21, const real_type y31,  //
        //
        const real_type y02, const real_type y12, const real_type y22, const real_type y32,  //
        //
        const real_type y03, const real_type y13, const real_type y23, const real_type y33) {  //
                                                                                               //
    real_type yw0 = weno4ConstH(x, h, y00, y10, y20, y30);
    real_type yw1 = weno4ConstH(x, h, y01, y11, y21, y31);
    real_type yw2 = weno4ConstH(x, h, y02, y12, y22, y32);
    real_type yw3 = weno4ConstH(x, h, y03, y13, y23, y33);

    real_type yw = weno4ConstH(y, h, yw0, yw1, yw2, yw3);

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
__device__ real_type weno4_3D_ConstH(const real_type x, const real_type y, const real_type z,  //
                                     const real_type h, const real_type *f,                    //
                                     const int stride_x,                                       //
                                     const int stride_y,                                       //
                                     const int stride_z) {                                     //

    real_type w1 = weno4_2D_ConstH(x,
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

    real_type w2 = weno4_2D_ConstH(x,
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

    real_type w3 = weno4_2D_ConstH(x,
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

    real_type w4 = weno4_2D_ConstH(x,
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

    real_type wz = weno4ConstH(z, h, w1, w2, w3, w4);

    return wz;
}


#define __TESTING_TET10_WENO_CUDA__

#ifdef __TESTING_TET10_WENO_CUDA__
int main() {

    printf("Testing WENO 4th order for 3D with constant h\n");

    return 0;
}
#endif
