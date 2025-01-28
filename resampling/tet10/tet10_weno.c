#include <math.h>

#include "tet10_weno.h"
#include "sfem_base.h"

// real_t Power(const real_t x, const real_t y) { return exp(y * log(x)); }
static SFEM_INLINE real_t Power(const real_t x, const real_t y) { return pow(x, y); }

static SFEM_INLINE real_t Power2(const real_t x) { return x * x; }

// real_t Power1p5(const real_t x) { return exp(1.5 * log(x)); }
static SFEM_INLINE real_t Power1p5(const real_t x) { return x * sqrt(x); }

static SFEM_INLINE real_t Power_m1p5(const real_t x) { return 1.0 / (x * sqrt(x)); }

void LagrangePolyArrayConstH(const real_t x, const real_t h,  //
                             real_t *lagrange_poly_0,
                             real_t *lagrange_poly_1) {  //

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
void getLinearWeightsConstH(const real_t x, const real_t h, real_t *linear_weights) {
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
void getNonLinearWeightsConstH(const real_t x, const real_t h,    //
                               const real_t y0, const real_t y1,  //
                               const real_t y2, const real_t y3,  //
                               real_t *non_linear_weights,        //
                               const real_t eps) {                //

    real_t alpha[2];

    const real_t inv_h = 1. / h;

    const real_t a = -3. * y0 + 7. * y1 - 5. * y2 + y3;
    const real_t b = y0 - 6. * (1. + h) * y1 + 3. * y2 + 2. * y3;

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

    real_t den = alpha[0] + alpha[1];

    non_linear_weights[0] = alpha[0] / den;
    non_linear_weights[1] = alpha[1] / den;
}

real_t weno4ConstH(const real_t x, const real_t h,                                        //
                   const real_t y0, const real_t y1, const real_t y2, const real_t y3) {  //
    const real_t eps = 1e-6;

    real_t lagrange_poly_0[3];
    real_t lagrange_poly_1[3];
    real_t non_linear_weights[2];

    LagrangePolyArrayConstH(x, h, lagrange_poly_0, lagrange_poly_1);

    getNonLinearWeightsConstH(x, h, y0, y1, y2, y3, non_linear_weights, eps);

    const real_t weno4_a =
            (lagrange_poly_0[0] * y0 + lagrange_poly_0[1] * y1 + lagrange_poly_0[2] * y2) *
            non_linear_weights[0];

    const real_t weno4_b =
            (lagrange_poly_1[0] * y1 + lagrange_poly_1[1] * y2 + lagrange_poly_1[2] * y3) *
            non_linear_weights[1];

    return weno4_a + weno4_b;
}

real_t weno4_2D_ConstH(const real_t x, const real_t y, const real_t h,                          //
                                                                                                //
                       const real_t y00, const real_t y10, const real_t y20, const real_t y30,  //
                       //
                       const real_t y01, const real_t y11, const real_t y21, const real_t y31,  //
                       //
                       const real_t y02, const real_t y12, const real_t y22, const real_t y32,  //
                       //
                       const real_t y03, const real_t y13, const real_t y23, const real_t y33) {  //
                                                                                                  //
    real_t yw0 = weno4ConstH(x, h, y00, y10, y20, y30);
    real_t yw1 = weno4ConstH(x, h, y01, y11, y21, y31);
    real_t yw2 = weno4ConstH(x, h, y02, y12, y22, y32);
    real_t yw3 = weno4ConstH(x, h, y03, y13, y23, y33);

    real_t yw = weno4ConstH(y, h, yw0, yw1, yw2, yw3);

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
 * @return real_t
 */
real_t weno4_3D_ConstH(const real_t x, const real_t y, const real_t z,  //
                       const real_t h, const real_t *f,                 //
                       const int stride_x,                              //
                       const int stride_y,                              //
                       const int stride_z) {                            //

    real_t w1 = weno4_2D_ConstH(x,
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

    real_t w2 = weno4_2D_ConstH(x,
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

    real_t w3 = weno4_2D_ConstH(x,
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

    real_t w4 = weno4_2D_ConstH(x,
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

    real_t wz = weno4ConstH(z, h, w1, w2, w3, w4);

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
void LagrangePolyArrayHOne(const real_t x,             //
                           real_t *lagrange_poly_0,    //
                           real_t *lagrange_poly_1) {  //

    // const real_t h = 1.0;

    const real_t xx = Power2(x);

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
void getNonLinearWeightsHOne(const real_t x,                    //
                             const real_t y0, const real_t y1,  //
                             const real_t y2, const real_t y3,  //
                             real_t *non_linear_weights,        //
                             const real_t eps) {                //

    real_t alpha[2];
    // const real_t h = 1.0;

    const real_t a = Abs(3. * y0 - 7. * y1 + 5. * y2 - 1. * y3);
    const real_t b = Power2(-3. * (a) + Abs(y0 - 12. * y1 + 3. * y2 + 2. * y3));

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

    real_t den = alpha[0] + alpha[1];

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
 * @return real_t
 */
real_t weno4_HOne(const real_t x,                      //
                  const real_t y0, const real_t y1,    //
                  const real_t y2, const real_t y3) {  //

    const real_t eps = 1e-6;

    real_t lagrange_poly_0[3];
    real_t lagrange_poly_1[3];
    real_t non_linear_weights[2];

    LagrangePolyArrayHOne(x, lagrange_poly_0, lagrange_poly_1);

    getNonLinearWeightsHOne(x, y0, y1, y2, y3, non_linear_weights, eps);
    // getNonLinearWeightsConstH(x, 1.0, y0, y1, y2, y3, non_linear_weights, eps);

    const real_t weno4_a =
            (lagrange_poly_0[0] * y0 + lagrange_poly_0[1] * y1 + lagrange_poly_0[2] * y2) *
            non_linear_weights[0];

    const real_t weno4_b =
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
 * @return real_t
 */
real_t weno4_2D_HOne(const real_t x, const real_t y,                                          //
                                                                                              //
                     const real_t y00, const real_t y10, const real_t y20, const real_t y30,  //
                     //
                     const real_t y01, const real_t y11, const real_t y21, const real_t y31,  //
                     //
                     const real_t y02, const real_t y12, const real_t y22, const real_t y32,  //
                     //
                     const real_t y03, const real_t y13, const real_t y23, const real_t y33) {  //

    real_t yw0 = weno4_HOne(x, y00, y10, y20, y30);
    real_t yw1 = weno4_HOne(x, y01, y11, y21, y31);
    real_t yw2 = weno4_HOne(x, y02, y12, y22, y32);
    real_t yw3 = weno4_HOne(x, y03, y13, y23, y33);

    real_t yw = weno4_HOne(y, yw0, yw1, yw2, yw3);

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
 * @return real_t
 */
real_t weno4_3D_HOne(const real_t x, const real_t y, const real_t z,  //
                     const real_t *f,                                 //
                     const int stride_x,                              //
                     const int stride_y,                              //
                     const int stride_z) {                            //

    real_t w1 = weno4_2D_HOne(x,
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

    real_t w2 = weno4_2D_HOne(x,
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

    real_t w3 = weno4_2D_HOne(x,
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

    real_t w4 = weno4_2D_HOne(x,
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

    real_t wz = weno4_HOne(z, w1, w2, w3, w4);

    return wz;
}
