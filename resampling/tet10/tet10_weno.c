#include <math.h>

#include "tet10_weno.h"

// double Power(const double x, const double y) { return exp(y * log(x)); }
inline double Power(const double x, const double y) { return pow(x, y); }

inline double Power2(const double x) { return x * x; }

// double Power1p5(const double x) { return exp(1.5 * log(x)); }
inline double Power1p5(const double x) { return x * sqrt(x); }

inline double Power_m1p5(const double x) { return 1.0 / (x * sqrt(x)); }

void LagrangePolyArrayConstH(const double x, const double h,  //
                             double *lagrange_poly_0,
                             double *lagrange_poly_1) {  //

    List3(lagrange_poly_0,                               //
          ((-2 * h + x) * (-h + x)) / (2. * Power2(h)),  //
          -((x * (-2 * h + x)) / Power2(h)),             //
          (x * (-h + x)) / (2. * Power2(h)));            //

    List3(lagrange_poly_1,                                   //
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
void getLinearWeightsConstH(const double x, const double h, double *linear_weights) {
    //
    linear_weights[0] = (3 * h - x) / (3. * h);
    linear_weights[1] = x / (3. * h);
}

/**
 * @brief Get the Non Linear Weights object by assuming the h is constant
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
void getNonLinearWeightsConstH(const double x, const double h,    //
                               const double y0, const double y1,  //
                               const double y2, const double y3,  //
                               double *non_linear_weights,        //
                               const double eps) {                //

    double alpha[2];

    const double inv_h = 1. / h;

    const double a = -3. * y0 + 7. * y1 - 5. * y2 + y3;
    const double b = y0 - 6. * (1. + h) * y1 + 3. * y2 + 2. * y3;

    List2(alpha,
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

    double den = alpha[0] + alpha[1];

    non_linear_weights[0] = alpha[0] / den;
    non_linear_weights[1] = alpha[1] / den;
}

double weno4ConstH(const double x, const double h,                                        //
                   const double y0, const double y1, const double y2, const double y3) {  //
    const double eps = 1e-6;

    double lagrange_poly_0[3];
    double lagrange_poly_1[3];
    double non_linear_weights[2];

    LagrangePolyArrayConstH(x, h, lagrange_poly_0, lagrange_poly_1);

    getNonLinearWeightsConstH(x, h, y0, y1, y2, y3, non_linear_weights, eps);

    const double weno4_a =
            (lagrange_poly_0[0] * y0 + lagrange_poly_0[1] * y1 + lagrange_poly_0[2] * y2) *
            non_linear_weights[0];

    const double weno4_b =
            (lagrange_poly_1[0] * y1 + lagrange_poly_1[1] * y2 + lagrange_poly_1[2] * y3) *
            non_linear_weights[1];

    return weno4_a + weno4_b;
}

double weno4_2D_ConstH(const double x, const double y, const double h,                          //
                                                                                                //
                       const double y00, const double y10, const double y20, const double y30,  //
                       //
                       const double y01, const double y11, const double y21, const double y31,  //
                       //
                       const double y02, const double y12, const double y22, const double y32,  //
                       //
                       const double y03, const double y13, const double y23, const double y33) {  //
                                                                                                  //
    double yw0 = weno4ConstH(x, h, y00, y10, y20, y30);
    double yw1 = weno4ConstH(x, h, y01, y11, y21, y31);
    double yw2 = weno4ConstH(x, h, y02, y12, y22, y32);
    double yw3 = weno4ConstH(x, h, y03, y13, y23, y33);

    double yw = weno4ConstH(y, h, yw0, yw1, yw2, yw3);

    return yw;
}

/**
 * @brief WENO4 interpolation for 3D data with constant grid spacing
 *
 * @param x
 * @param y
 * @param z
 * @param h
 * @param f
 * @param stride_y
 * @param stride_z
 * @return double
 */
double weno4_3D_ConstH(const double x, const double y, const double z,  //
                       const double h, const double *f,                 //
                       const int stride_x,                              //
                       const int stride_y,                              //
                       const int stride_z) {                            //

    double w1 = weno4_2D_ConstH(x,
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

    double w2 = weno4_2D_ConstH(x,
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

    double w3 = weno4_2D_ConstH(x,
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

    double w4 = weno4_2D_ConstH(x,
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

    double wz = weno4ConstH(z, h, w1, w2, w3, w4);

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
void LagrangePolyArrayHOne(const double x,             //
                           double *lagrange_poly_0,    //
                           double *lagrange_poly_1) {  //

    // const double h = 1.0;

    const double xx = Power2(x);

    List3(lagrange_poly_0,  //
          (2.0 - 3.0 * x + xx) / 2.,
          -((-2.0 + x) * x),
          ((-1.0 + x) * x) / 2.  //
    );                           //

    List3(lagrange_poly_1,  //
          (6.0 - 5.0 * x + xx) / 2.,
          -3.0 + 4.0 * x - xx,         //
          (2.0 - 3.0 * x + xx) / 2.);  //
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
void getNonLinearWeightsHOne(const double x,                    //
                             const double y0, const double y1,  //
                             const double y2, const double y3,  //
                             double *non_linear_weights,        //
                             const double eps) {                //

    double alpha[2];
    // const double h = 1.0;

    const double a = Abs(3. * y0 - 7. * y1 + 5. * y2 - 1. * y3);
    const double b = Power2(-3. * (a) + Abs(y0 - 12. * y1 + 3. * y2 + 2. * y3));

    List2(alpha,
          //
          Power_m1p5(eps + 0.1111111111111111 * (b)) -
                  (0.3333333333333333 * x) / Power1p5(eps + 0.1111111111111111 * (b))
          //
          ,
          //
          (0.3333333333333333 *
           x) / Power1p5(eps + Power2(-2. * Abs(y0 - 1.5 * y1 + 4. * y2 - 1.8333333333333333 * y3) +
                                      Abs(y0 - 1. * y1 - 1. * y2 + y3)))
          //
    );

    double den = alpha[0] + alpha[1];

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
 * @return double
 */
double weno4_HOne(const double x,                      //
                  const double y0, const double y1,    //
                  const double y2, const double y3) {  //

    const double eps = 1e-6;

    double lagrange_poly_0[3];
    double lagrange_poly_1[3];
    double non_linear_weights[2];

    LagrangePolyArrayHOne(x, lagrange_poly_0, lagrange_poly_1);

    getNonLinearWeightsHOne(x, y0, y1, y2, y3, non_linear_weights, eps);
    // getNonLinearWeightsConstH(x, 1.0, y0, y1, y2, y3, non_linear_weights, eps);

    const double weno4_a =
            (lagrange_poly_0[0] * y0 + lagrange_poly_0[1] * y1 + lagrange_poly_0[2] * y2) *
            non_linear_weights[0];

    const double weno4_b =
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
 * @return double
 */
double weno4_2D_HOne(const double x, const double y,                                          //
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
 * @return double
 */
double weno4_3D_HOne(const double x, const double y, const double z,  //
                     const double *f,                                 //
                     const int stride_x,                              //
                     const int stride_y,                              //
                     const int stride_z) {                            //

    double w1 = weno4_2D_HOne(x,
                              y,  //
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

    double w2 = weno4_2D_HOne(x,
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

    double w3 = weno4_2D_HOne(x,
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

    double w4 = weno4_2D_HOne(x,
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

    double wz = weno4_HOne(z, w1, w2, w3, w4);

    return wz;
}
