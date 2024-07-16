#include "tet10_resample_field.h"

#include "quadratures_rule.h"
#include "tet10_weno.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define POW2(x) ((x) * (x))
#define POW3(x) ((x) * (x) * (x))

SFEM_INLINE static void tet10_dual_basis_popp(const real_t qx, const real_t qy, const real_t qz,
                                              real_t* const f) {
    const real_t x0 = -qx - qy - qz + 1;
    const real_t x1 = 2 * qy;
    const real_t x2 = 2 * qz;
    const real_t x3 = 2 * qx - 1;
    const real_t x4 = -x1 - x2 - x3;
    const real_t x5 = x0 * x4;
    const real_t x6 = x1 - 1;
    const real_t x7 = (7.0 / 10.0) * qy;
    const real_t x8 = x6 * x7;
    const real_t x9 = x2 - 1;
    const real_t x10 = (7.0 / 10.0) * qz;
    const real_t x11 = x10 * x9;
    const real_t x12 = -4 * qx - 4 * qy - 4 * qz + 4;
    const real_t x13 = (7.0 / 40.0) * x12;
    const real_t x14 = qy * qz;
    const real_t x15 = qx * x13 + x11 + (14.0 / 5.0) * x14 + x8;
    const real_t x16 = qx * x3;
    const real_t x17 = (7.0 / 10.0) * x16;
    const real_t x18 = (14.0 / 5.0) * qx;
    const real_t x19 = qy * x13 + qz * x18 + x17;
    const real_t x20 = qy * x18 + qz * x13;
    const real_t x21 = (7.0 / 10.0) * x5;
    const real_t x22 = qx * x7 + x10 * x12 + x21;
    const real_t x23 = qx * x10 + x12 * x7;
    const real_t x24 = qy * x6;
    const real_t x25 = qx * x12;
    const real_t x26 = qz * x7 + (7.0 / 10.0) * x25;
    const real_t x27 = qz * x9;
    const real_t x28 = -6.0 / 5.0 * qy * x6;
    const real_t x29 = -6.0 / 5.0 * qz * x9;
    const real_t x30 = (15.0 / 2.0) * qx;
    const real_t x31 = qz * x30;
    const real_t x32 = (15.0 / 8.0) * x12;
    const real_t x33 = qy * x32;
    const real_t x34 = (39.0 / 10.0) * x16;
    const real_t x35 = x29 + x31 + x33 + x34;
    const real_t x36 = qy * x30;
    const real_t x37 = qz * x32;
    const real_t x38 = (39.0 / 10.0) * x5;
    const real_t x39 = x36 + x37 + x38;
    const real_t x40 = (15.0 / 2.0) * x14 + (15.0 / 8.0) * x25;
    const real_t x41 = -6.0 / 5.0 * x0 * x4;
    const real_t x42 = (39.0 / 10.0) * x24;
    const real_t x43 = x41 + x42;
    const real_t x44 = -6.0 / 5.0 * qx * x3;
    const real_t x45 = x40 + x44;
    const real_t x46 = (39.0 / 10.0) * x27;
    const real_t x47 = x28 + x46;
    const real_t x48 = x31 + x33;
    const real_t x49 = x36 + x37;
    f[0] = x15 + x19 + x20 + 7 * x5;
    f[1] = x15 + 7 * x16 + x22 + x23;
    f[2] = x11 + x19 + x22 + 7 * x24 + x26;
    f[3] = x17 + x20 + x21 + x23 + x26 + 7 * x27 + x8;
    f[4] = (111.0 / 20.0) * qx * x12 + (24.0 / 5.0) * qy * qz - x28 - x35 - x39;
    f[5] = (111.0 / 5.0) * qx * qy + (6.0 / 5.0) * qz * x12 - x35 - x40 - x43;
    f[6] = (24.0 / 5.0) * qx * qz + (111.0 / 20.0) * qy * x12 - x29 - x39 - x42 - x45;
    f[7] = (24.0 / 5.0) * qx * qy + (111.0 / 20.0) * qz * x12 - x38 - x45 - x47 - x48;
    f[8] = (111.0 / 5.0) * qx * qz + (6.0 / 5.0) * qy * x12 - x34 - x40 - x41 - x47 - x49;
    f[9] = (6.0 / 5.0) * qx * x12 + (111.0 / 5.0) * qy * qz - x43 - x44 - x46 - x48 - x49;
}

SFEM_INLINE static void tet10_dual_basis_hrt(const real_t qx, const real_t qy, const real_t qz,
                                             real_t* const f) {
    const real_t x0 = 2 * qy;
    const real_t x1 = 2 * qz;
    const real_t x2 = 2 * qx - 1;
    const real_t x3 = (-x0 - x1 - x2) * (-qx - qy - qz + 1);
    const real_t x4 = x0 - 1;
    const real_t x5 = (5.0 / 18.0) * qy;
    const real_t x6 = x4 * x5;
    const real_t x7 = x1 - 1;
    const real_t x8 = (5.0 / 18.0) * qz;
    const real_t x9 = x7 * x8;
    const real_t x10 = -4 * qx - 4 * qy - 4 * qz + 4;
    const real_t x11 = (5.0 / 72.0) * x10;
    const real_t x12 = qy * qz;
    const real_t x13 = qx * x11 + (10.0 / 9.0) * x12 + x6 + x9;
    const real_t x14 = (5.0 / 18.0) * qx;
    const real_t x15 = x14 * x2;
    const real_t x16 = (10.0 / 9.0) * qx;
    const real_t x17 = qy * x11 + qz * x16 + x15;
    const real_t x18 = qy * x16 + qz * x11;
    const real_t x19 = qx * x2;
    const real_t x20 = (5.0 / 18.0) * x3;
    const real_t x21 = qy * x14 + x10 * x8 + x20;
    const real_t x22 = qz * x14 + x10 * x5;
    const real_t x23 = qy * x4;
    const real_t x24 = qz * x5 + x10 * x14;
    const real_t x25 = qz * x7;
    const real_t x26 = (40.0 / 27.0) * x23;
    const real_t x27 = (115.0 / 27.0) * x10;
    const real_t x28 = (110.0 / 27.0) * qx;
    const real_t x29 = -qz * x28;
    const real_t x30 = (55.0 / 54.0) * x10;
    const real_t x31 = -qy * x30;
    const real_t x32 = (10.0 / 27.0) * x19;
    const real_t x33 = (40.0 / 27.0) * x25;
    const real_t x34 = x29 + x31 + x32 + x33;
    const real_t x35 = -qy * x28;
    const real_t x36 = -qz * x30;
    const real_t x37 = (10.0 / 27.0) * x3;
    const real_t x38 = x35 + x36 + x37;
    const real_t x39 = (40.0 / 27.0) * x10;
    const real_t x40 = qx * qy;
    const real_t x41 = -qx * x30 - 110.0 / 27.0 * x12;
    const real_t x42 = (10.0 / 27.0) * x23;
    const real_t x43 = (40.0 / 27.0) * x3;
    const real_t x44 = x42 + x43;
    const real_t x45 = qx * qz;
    const real_t x46 = (40.0 / 27.0) * x19;
    const real_t x47 = x41 + x46;
    const real_t x48 = (10.0 / 27.0) * x25;
    const real_t x49 = x26 + x48;
    const real_t x50 = x29 + x31;
    const real_t x51 = x35 + x36;
    f[0] = x13 + x17 + x18 + (25.0 / 9.0) * x3;
    f[1] = x13 + (25.0 / 9.0) * x19 + x21 + x22;
    f[2] = x17 + x21 + (25.0 / 9.0) * x23 + x24 + x9;
    f[3] = x15 + x18 + x20 + x22 + x24 + (25.0 / 9.0) * x25 + x6;
    f[4] = qx * x27 + (160.0 / 27.0) * x12 + x26 + x34 + x38;
    f[5] = qz * x39 + x34 + (460.0 / 27.0) * x40 + x41 + x44;
    f[6] = qy * x27 + x33 + x38 + x42 + (160.0 / 27.0) * x45 + x47;
    f[7] = qz * x27 + x37 + (160.0 / 27.0) * x40 + x47 + x49 + x50;
    f[8] = qy * x39 + x32 + x41 + x43 + (460.0 / 27.0) * x45 + x49 + x51;
    f[9] = qx * x39 + (460.0 / 27.0) * x12 + x44 + x46 + x48 + x50 + x51;
}

SFEM_INLINE static real_t tet4_measure(
        // X-coordinates
        const real_t px0, const real_t px1, const real_t px2, const real_t px3,
        // Y-coordinates
        const real_t py0, const real_t py1, const real_t py2, const real_t py3,
        // Z-coordinates
        const real_t pz0, const real_t pz1, const real_t pz2, const real_t pz3) {
    //
    // determinant of the Jacobian
    // M = [px0, py0, pz0, 1]
    //     [px1, py1, pz1, 1]
    //     [px2, py2, pz2, 1]
    //     [px3, py3, pz3, 1]
    //
    // V = (1/6) * det(M)

    const real_t x0 = -pz0 + pz3;
    const real_t x1 = -py0 + py2;
    const real_t x2 = -1.0 / 6.0 * px0 + (1.0 / 6.0) * px1;
    const real_t x3 = -py0 + py3;
    const real_t x4 = -pz0 + pz2;
    const real_t x5 = -py0 + py1;
    const real_t x6 = -1.0 / 6.0 * px0 + (1.0 / 6.0) * px2;
    const real_t x7 = -pz0 + pz1;
    const real_t x8 = -1.0 / 6.0 * px0 + (1.0 / 6.0) * px3;

    return x0 * x1 * x2 - x0 * x5 * x6 - x1 * x7 * x8 - x2 * x3 * x4 + x3 * x6 * x7 + x4 * x5 * x8;
}

SFEM_INLINE static void tet4_transform(
        /**
         ****************************************************************************************
        \begin{bmatrix}
        out_x \\
        out_y \\
        out_z
        \end{bmatrix}
        =
        \begin{bmatrix}
        px_0 \\
        py_0 \\
        pz_0
        \end{bmatrix}
        +
        \begin{bmatrix}
        px_1 - px_0 & px_2 - px_0 & px_3 - px_0 \\
        py_1 - py_0 & py_2 - py_0 & py_3 - py_0 \\
        pz_1 - pz_0 & pz_2 - pz_0 & pz_3 - pz_0
        \end{bmatrix}
        \cdot
        \begin{bmatrix}
        qx \\
        qy \\
        qz
        \end{bmatrix}
        *************************************************************************************************
      */

        // X-coordinates
        const real_t px0, const real_t px1, const real_t px2, const real_t px3,
        // Y-coordinates
        const real_t py0, const real_t py1, const real_t py2, const real_t py3,
        // Z-coordinates
        const real_t pz0, const real_t pz1, const real_t pz2, const real_t pz3,
        // Quadrature point
        const real_t qx, const real_t qy, const real_t qz,
        // Output
        real_t* const SFEM_RESTRICT out_x, real_t* const SFEM_RESTRICT out_y,
        real_t* const SFEM_RESTRICT out_z) {
    //
    //
    *out_x = px0 + qx * (-px0 + px1) + qy * (-px0 + px2) + qz * (-px0 + px3);
    *out_y = py0 + qx * (-py0 + py1) + qy * (-py0 + py2) + qz * (-py0 + py3);
    *out_z = pz0 + qx * (-pz0 + pz1) + qy * (-pz0 + pz2) + qz * (-pz0 + pz3);
}

SFEM_INLINE static void hex_aa_8_eval_fun(
        // Quadrature point (local coordinates)
        // With respect to the hat functions of a cube element
        // In a local coordinate system
        const real_t x, const real_t y, const real_t z,
        // Output
        real_t* const SFEM_RESTRICT f) {
    //
    f[0] = (1.0 - x) * (1.0 - y) * (1.0 - z);
    f[1] = x * (1.0 - y) * (1.0 - z);
    f[2] = x * y * (1.0 - z);
    f[3] = (1.0 - x) * y * (1.0 - z);
    f[4] = (1.0 - x) * (1.0 - y) * z;
    f[5] = x * (1.0 - y) * z;
    f[6] = x * y * z;
    f[7] = (1.0 - x) * y * z;
}

SFEM_INLINE static void hex_aa_8_collect_coeffs(
        const ptrdiff_t* const SFEM_RESTRICT stride, const ptrdiff_t i, const ptrdiff_t j,
        const ptrdiff_t k,
        // Attention this is geometric data transformed to solver data!
        const real_t* const SFEM_RESTRICT data, real_t* const SFEM_RESTRICT out) {
    const ptrdiff_t i0 = i * stride[0] + j * stride[1] + k * stride[2];
    const ptrdiff_t i1 = (i + 1) * stride[0] + j * stride[1] + k * stride[2];
    const ptrdiff_t i2 = (i + 1) * stride[0] + (j + 1) * stride[1] + k * stride[2];
    const ptrdiff_t i3 = i * stride[0] + (j + 1) * stride[1] + k * stride[2];
    const ptrdiff_t i4 = i * stride[0] + j * stride[1] + (k + 1) * stride[2];
    const ptrdiff_t i5 = (i + 1) * stride[0] + j * stride[1] + (k + 1) * stride[2];
    const ptrdiff_t i6 = (i + 1) * stride[0] + (j + 1) * stride[1] + (k + 1) * stride[2];
    const ptrdiff_t i7 = i * stride[0] + (j + 1) * stride[1] + (k + 1) * stride[2];

    out[0] = data[i0];
    out[1] = data[i1];
    out[2] = data[i2];
    out[3] = data[i3];
    out[4] = data[i4];
    out[5] = data[i5];
    out[6] = data[i6];
    out[7] = data[i7];
}

/**
 * @brief Compute the indices of the field for third order interpolation
 *
 * @param stride
 * @param i
 * @param j
 * @param k
 * @param i0 .. i15
 * @return SFEM_INLINE
 */
SFEM_INLINE static void hex_aa_8_indices_O3(const ptrdiff_t* const SFEM_RESTRICT stride,  //
                                            const ptrdiff_t i, const ptrdiff_t j, const ptrdiff_t k,
                                            // Output
                                            ptrdiff_t* i0, ptrdiff_t* i1, ptrdiff_t* i2,
                                            ptrdiff_t* i3, ptrdiff_t* i4, ptrdiff_t* i5,
                                            ptrdiff_t* i6, ptrdiff_t* i7, ptrdiff_t* i8,
                                            ptrdiff_t* i9, ptrdiff_t* i10, ptrdiff_t* i11,
                                            ptrdiff_t* i12, ptrdiff_t* i13, ptrdiff_t* i14,
                                            ptrdiff_t* i15) {
    const ptrdiff_t stride_x = stride[0];
    const ptrdiff_t stride_y = stride[1];
    const ptrdiff_t stride_z = stride[2];

    *i0 = (i - 1) * stride_x + (j - 1) * stride_y + (k)*stride_z;
    *i1 = (i)*stride_x + (j - 1) * stride_y + (k)*stride_z;
    *i2 = (i + 1) * stride_x + (j - 1) * stride_y + (k)*stride_z;
    *i3 = (i + 2) * stride_x + (j - 1) * stride_y + (k)*stride_z;

    *i4 = (i - 1) * stride_x + (j)*stride_y + (k)*stride_z;
    *i5 = (i)*stride_x + (j)*stride_y + (k)*stride_z;
    *i6 = (i + 1) * stride_x + (j)*stride_y + (k)*stride_z;
    *i7 = (i + 2) * stride_x + (j)*stride_y + (k)*stride_z;

    *i8 = (i - 1) * stride_x + (j + 1) * stride_y + (k)*stride_z;
    *i9 = (i)*stride_x + (j + 1) * stride_y + (k)*stride_z;
    *i10 = (i + 1) * stride_x + (j + 1) * stride_y + (k)*stride_z;
    *i11 = (i + 2) * stride_x + (j + 1) * stride_y + (k)*stride_z;

    *i12 = (i - 1) * stride_x + (j + 2) * stride_y + (k)*stride_z;
    *i13 = (i)*stride_x + (j + 2) * stride_y + (k)*stride_z;
    *i14 = (i + 1) * stride_x + (j + 2) * stride_y + (k)*stride_z;
    *i15 = (i + 2) * stride_x + (j + 2) * stride_y + (k)*stride_z;
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
SFEM_INLINE static void hex_aa_8_collect_coeffs_O3(
        const ptrdiff_t* const SFEM_RESTRICT stride,  //
        const ptrdiff_t i, const ptrdiff_t j, const ptrdiff_t k,
        // Attention this is geometric data transformed to solver data!
        const real_t* const SFEM_RESTRICT data, real_t* const SFEM_RESTRICT out) {
    //
    ptrdiff_t i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15;

    ptrdiff_t i16, i17, i18, i19, i20, i21, i22, i23, i24, i25, i26, i27, i28, i29, i30, i31;

    ptrdiff_t i32, i33, i34, i35, i36, i37, i38, i39, i40, i41, i42, i43, i44, i45, i46, i47;

    ptrdiff_t i48, i49, i50, i51, i52, i53, i54, i55, i56, i57, i58, i59, i60, i61, i62, i63;

    hex_aa_8_indices_O3(stride,
                        i,
                        j,
                        k,
                        &i0,
                        &i1,
                        &i2,
                        &i3,
                        &i4,
                        &i5,
                        &i6,
                        &i7,
                        &i8,
                        &i9,
                        &i10,
                        &i11,
                        &i12,
                        &i13,
                        &i14,
                        &i15);

    hex_aa_8_indices_O3(stride,
                        i,
                        j,
                        k + 1,
                        &i16,
                        &i17,
                        &i18,
                        &i19,
                        &i20,
                        &i21,
                        &i22,
                        &i23,
                        &i24,
                        &i25,
                        &i26,
                        &i27,
                        &i28,
                        &i29,
                        &i30,
                        &i31);

    hex_aa_8_indices_O3(stride,
                        i,
                        j,
                        k + 2,
                        &i32,
                        &i33,
                        &i34,
                        &i35,
                        &i36,
                        &i37,
                        &i38,
                        &i39,
                        &i40,
                        &i41,
                        &i42,
                        &i43,
                        &i44,
                        &i45,
                        &i46,
                        &i47);

    hex_aa_8_indices_O3(stride,
                        i,
                        j,
                        k + 3,
                        &i48,
                        &i49,
                        &i50,
                        &i51,
                        &i52,
                        &i53,
                        &i54,
                        &i55,
                        &i56,
                        &i57,
                        &i58,
                        &i59,
                        &i60,
                        &i61,
                        &i62,
                        &i63);

    out[0] = data[i0];
    out[1] = data[i1];
    out[2] = data[i2];
    out[3] = data[i3];
    out[4] = data[i4];
    out[5] = data[i5];
    out[6] = data[i6];
    out[7] = data[i7];
    out[8] = data[i8];
    out[9] = data[i9];
    out[10] = data[i10];
    out[11] = data[i11];
    out[12] = data[i12];
    out[13] = data[i13];
    out[14] = data[i14];
    out[15] = data[i15];
    out[16] = data[i16];
    out[17] = data[i17];
    out[18] = data[i18];
    out[19] = data[i19];
    out[20] = data[i20];
    out[21] = data[i21];
    out[22] = data[i22];
    out[23] = data[i23];
    out[24] = data[i24];
    out[25] = data[i25];
    out[26] = data[i26];
    out[27] = data[i27];
    out[28] = data[i28];
    out[29] = data[i29];
    out[30] = data[i30];
    out[31] = data[i31];
    out[32] = data[i32];
    out[33] = data[i33];
    out[34] = data[i34];
    out[35] = data[i35];
    out[36] = data[i36];
    out[37] = data[i37];
    out[38] = data[i38];
    out[39] = data[i39];
    out[40] = data[i40];
    out[41] = data[i41];
    out[42] = data[i42];
    out[43] = data[i43];
    out[44] = data[i44];
    out[45] = data[i45];
    out[46] = data[i46];
    out[47] = data[i47];
    out[48] = data[i48];
    out[49] = data[i49];
    out[50] = data[i50];
    out[51] = data[i51];
    out[52] = data[i52];
    out[53] = data[i53];
    out[54] = data[i54];
    out[55] = data[i55];
    out[56] = data[i56];
    out[57] = data[i57];
    out[58] = data[i58];
    out[59] = data[i59];
    out[60] = data[i60];
    out[61] = data[i61];
    out[62] = data[i62];
    out[63] = data[i63];
}

SFEM_INLINE static real_t hex_aa_8_eval_weno4_3D(const real_t x_,                           //
                                                 const real_t y_,                           //
                                                 const real_t z_,                           //
                                                 const real_t ox,                           //
                                                 const real_t oy,                           //
                                                 const real_t oz,                           //
                                                 const real_t h,                            //
                                                 const ptrdiff_t i,                         //
                                                 const ptrdiff_t j,                         //
                                                 const ptrdiff_t k,                         //
                                                 const ptrdiff_t* stride,                   //
                                                 const real_t* const SFEM_RESTRICT data) {  //

    real_t out[64];
    hex_aa_8_collect_coeffs_O3(stride, i, j, k, data, &out[0]);

    double x = (x_ - ox) - (real_t)i * h + h;
    double y = (y_ - oy) - (real_t)j * h + h;
    double z = (z_ - oz) - (real_t)k * h + h;

    // printf("x = %f, x_ = %f, i = %d\n", x, x_, i);
    // printf("y = %f, y_ = %f, j = %d\n", y, y_, j);
    // printf("z = %f, z_ = %f, k = %d\n", z, z_, k);

    // printf("delta = %f\n", h);

    const real_t w4 = weno4_3D_ConstH(x,  //
                                      y,
                                      z,
                                      h,
                                      out,
                                      1,
                                      4,
                                      16);

    // printf("data[%d] = %f\n",
    //        (i * stride[0] + j * stride[1] + k * stride[2]),
    //        data[i * stride[0] + j * stride[1] + k * stride[2]]);

    // printf("w4 = %f\n", w4);

    return w4;
}

/**
 * @brief Compute the indices of the field for third order interpolation
 *
 * @param nelements
 * @param nnodes
 * @param elems
 * @param xyz
 * @param n
 * @param stride
 * @param origin
 * @param delta
 * @param data
 * @param weighted_field
 * @return int
 */
int hex8_to_subparametric_tet10_resample_field_local(
        // Mesh
        const ptrdiff_t nelements,          // number of elements
        const ptrdiff_t nnodes,             // number of nodes
        idx_t** const SFEM_RESTRICT elems,  // connectivity
        geom_t** const SFEM_RESTRICT xyz,   // coordinates
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n,       // number of nodes in each direction
        const ptrdiff_t* const SFEM_RESTRICT stride,  // stride of the data
        const geom_t* const SFEM_RESTRICT origin,     // origin of the domain
        const geom_t* const SFEM_RESTRICT delta,      // delta of the domain
        const real_t* const SFEM_RESTRICT data,       // SDF
        // Output
        real_t* const SFEM_RESTRICT weighted_field) {
    //
    // printf("============================================================\n");
    // printf("Start: hex8_to_tet10_resample_field_local\n");
    // printf("============================================================\n");
    //
    const real_t ox = (real_t)origin[0];
    const real_t oy = (real_t)origin[1];
    const real_t oz = (real_t)origin[2];

    const real_t dx = (real_t)delta[0];
    const real_t dy = (real_t)delta[1];
    const real_t dz = (real_t)delta[2];

#pragma omp parallel
    {
#pragma omp for  // nowait

        /// Loop over the elements of the mesh
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[10];

            // SUBPARAMETRIC
            geom_t x[4], y[4], z[4];

            real_t hex8_f[8];
            real_t coeffs[8];

            real_t tet10_f[10];
            real_t element_field[10];

            // loop over the 4 vertices of the tetrahedron
            // UNROLL_ZERO?
            for (int v = 0; v < 10; ++v) {
                ev[v] = elems[v][i];
            }

            // SUBPARAMETRIC (for isoparametric we collect all 10 points instead of 4)
            // copy the coordinates of the vertices
            for (int v = 0; v < 4; ++v) {
                x[v] = xyz[0][ev[v]];  // x-coordinates
                y[v] = xyz[1][ev[v]];  // y-coordinates
                z[v] = xyz[2][ev[v]];  // z-coordinates
            }

            memset(element_field, 0,
                   10 * sizeof(real_t));  // set to zero the element field

            // SUBPARAMETRIC (for isoparametric this is a nonlinear map computed for each qp)
            const real_t measure = tet4_measure(x[0],
                                                x[1],
                                                x[2],
                                                x[3],
                                                //
                                                y[0],
                                                y[1],
                                                y[2],
                                                y[3],
                                                //
                                                z[0],
                                                z[1],
                                                z[2],
                                                z[3]);

            assert(measure > 0);

            // SUBPARAMETRIC (for iso-parametric tassellation of tet10 might be necessary)
            for (int q = 0; q < TET4_NQP; q++) {  // loop over the quadrature points

                // det of jacobian
                real_t g_qx, g_qy, g_qz;
                // Transform quadrature point to physical space
                // g_qx, g_qy, g_qz are the coordinates of the quadrature point in the physical
                // space

                // SUBPARAMETRIC (for isoparametric this is a nonlinear map)
                tet4_transform(x[0],
                               x[1],
                               x[2],
                               x[3],
                               //
                               y[0],
                               y[1],
                               y[2],
                               y[3],
                               //
                               z[0],
                               z[1],
                               z[2],
                               z[3],
                               //
                               tet4_qx[q],
                               tet4_qy[q],
                               tet4_qz[q],
                               //
                               &g_qx,
                               &g_qy,
                               &g_qz);

                // No standard basis function (tet10 cannot lump as for tet4, needs special
                // treatment)
                // tet10_dual_basis_popp(tet4_qx[q], tet4_qy[q], tet4_qz[q], tet10_f);
                tet10_dual_basis_hrt(tet4_qx[q], tet4_qy[q], tet4_qz[q], tet10_f);

                const real_t dV = measure * tet4_qw[q];

                const real_t grid_x = (g_qx - ox) / dx;
                const real_t grid_y = (g_qy - oy) / dy;
                const real_t grid_z = (g_qz - oz) / dz;

                const ptrdiff_t i = floor(grid_x);
                const ptrdiff_t j = floor(grid_y);
                const ptrdiff_t k = floor(grid_z);

                // If outside
                if (i < 0 || j < 0 || k < 0 || (i + 1 >= n[0]) || (j + 1 >= n[1]) ||
                    (k + 1 >= n[2])) {
                    fprintf(stderr,
                            "warning (%g, %g, %g) (%ld, %ld, %ld) outside domain  (%ld, %ld, "
                            "%ld)!\n",
                            g_qx,
                            g_qy,
                            g_qz,
                            i,
                            j,
                            k,
                            n[0],
                            n[1],
                            n[2]);
                    continue;
                }

                // Get the reminder [0, 1]
                real_t l_x = (grid_x - i);
                real_t l_y = (grid_y - j);
                real_t l_z = (grid_z - k);

                assert(l_x >= -1e-8);
                assert(l_y >= -1e-8);
                assert(l_z >= -1e-8);

                assert(l_x <= 1 + 1e-8);
                assert(l_y <= 1 + 1e-8);
                assert(l_z <= 1 + 1e-8);

                hex_aa_8_eval_fun(l_x, l_y, l_z, hex8_f);
                hex_aa_8_collect_coeffs(stride, i, j, k, data, coeffs);

                // Integrate field
                {
                    real_t eval_field = 0;
                    // UNROLL_ZERO?
                    for (int edof_j = 0; edof_j < 8; edof_j++) {
                        eval_field += hex8_f[edof_j] * coeffs[edof_j];
                    }

                    // UNROLL_ZERO?
                    for (int edof_i = 0; edof_i < 10; edof_i++) {
                        element_field[edof_i] += eval_field * tet10_f[edof_i] * dV;
                    }  // end edof_i loop
                }
            }  // end quadrature loop

            // UNROLL_ZERO?
            for (int v = 0; v < 10; ++v) {
#pragma omp atomic update
                weighted_field[ev[v]] += element_field[v];

            }  // end vertex loop
        }      // end element loop
    }          // end parallel region

    return 0;
}

static SFEM_INLINE void lumped_mass_kernel_popp(
        const real_t px0, const real_t px1, const real_t px2, const real_t px3, const real_t py0,
        const real_t py1, const real_t py2, const real_t py3, const real_t pz0, const real_t pz1,
        const real_t pz2, const real_t pz3, real_t* const SFEM_RESTRICT element_matrix_diag) {
    //
    const real_t x0 = px0 - px1;
    const real_t x1 = py0 - py2;
    const real_t x2 = pz0 - pz3;
    const real_t x3 = x1 * x2;
    const real_t x4 = py0 - py3;
    const real_t x5 = pz0 - pz2;
    const real_t x6 = px0 - px2;
    const real_t x7 = py0 - py1;
    const real_t x8 = pz0 - pz1;
    const real_t x9 = x4 * x8;
    const real_t x10 = px0 - px3;
    const real_t x11 = x5 * x7;
    const real_t x12 = -7.0 / 600.0 * x0 * x3 + (7.0 / 600.0) * x0 * x4 * x5 +
                       (7.0 / 600.0) * x1 * x10 * x8 - 7.0 / 600.0 * x10 * x11 +
                       (7.0 / 600.0) * x2 * x6 * x7 - 7.0 / 600.0 * x6 * x9;
    const real_t x13 = -1.0 / 50.0 * x0 * x3 + (1.0 / 50.0) * x0 * x4 * x5 +
                       (1.0 / 50.0) * x1 * x10 * x8 - 1.0 / 50.0 * x10 * x11 +
                       (1.0 / 50.0) * x2 * x6 * x7 - 1.0 / 50.0 * x6 * x9;
    element_matrix_diag[0] = x12;
    element_matrix_diag[1] = x12;
    element_matrix_diag[2] = x12;
    element_matrix_diag[3] = x12;
    element_matrix_diag[4] = x13;
    element_matrix_diag[5] = x13;
    element_matrix_diag[6] = x13;
    element_matrix_diag[7] = x13;
    element_matrix_diag[8] = x13;
    element_matrix_diag[9] = x13;
}

static SFEM_INLINE void lumped_mass_kernel_hrt(const real_t px0, const real_t px1, const real_t px2,
                                               const real_t px3, const real_t py0, const real_t py1,
                                               const real_t py2, const real_t py3, const real_t pz0,
                                               const real_t pz1, const real_t pz2, const real_t pz3,
                                               real_t* const SFEM_RESTRICT diag) {
    // Symbolic integration (for iso-parametric tet10, this is not possible)
    const real_t x0 = -px0 + px1;
    const real_t x1 = (1.0 / 216.0) * x0;
    const real_t x2 = -pz0 + pz3;
    const real_t x3 = -py0 + py2;
    const real_t x4 = x2 * x3;
    const real_t x5 = -py0 + py3;
    const real_t x6 = -pz0 + pz2;
    const real_t x7 = x5 * x6;
    const real_t x8 = -px0 + px2;
    const real_t x9 = (1.0 / 216.0) * x8;
    const real_t x10 = -py0 + py1;
    const real_t x11 = x10 * x2;
    const real_t x12 = -pz0 + pz1;
    const real_t x13 = x12 * x5;
    const real_t x14 = -px0 + px3;
    const real_t x15 = (1.0 / 216.0) * x14;
    const real_t x16 = x10 * x6;
    const real_t x17 = x12 * x3;
    const real_t x18 = x1 * x4 - x1 * x7 - x11 * x9 + x13 * x9 + x15 * x16 - x15 * x17;
    const real_t x19 = (2.0 / 81.0) * x0;
    const real_t x20 = (2.0 / 81.0) * x8;
    const real_t x21 = (2.0 / 81.0) * x14;
    const real_t x22 = -x11 * x20 + x13 * x20 + x16 * x21 - x17 * x21 + x19 * x4 - x19 * x7;
    diag[0] = x18;
    diag[1] = x18;
    diag[2] = x18;
    diag[3] = x18;
    diag[4] = x22;
    diag[5] = x22;
    diag[6] = x22;
    diag[7] = x22;
    diag[8] = x22;
    diag[9] = x22;
}

int subparametric_tet10_assemble_dual_mass_vector(const ptrdiff_t nelements, const ptrdiff_t nnodes,
                                                  idx_t** const SFEM_RESTRICT elems,
                                                  geom_t** const SFEM_RESTRICT xyz,
                                                  real_t* const diag) {
    const geom_t* x = xyz[0];
    const geom_t* y = xyz[1];
    const geom_t* z = xyz[2];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[10];  // Element indices
        real_t element_diag[10];

#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elems[v][i];
        }

        // lumped_mass_kernel_popp (this method has been studied in the literature, but requires a
        // matrix transformation at the end)
        lumped_mass_kernel_hrt(  // (this method is kinda (epsilon) novel, perfect for matrix-free
                                 // and parallel computing, unclear on the properties) X-coordinates
                x[ev[0]],
                x[ev[1]],
                x[ev[2]],
                x[ev[3]],
                // Y-coordinates
                y[ev[0]],
                y[ev[1]],
                y[ev[2]],
                y[ev[3]],
                // Z-coordinates
                z[ev[0]],
                z[ev[1]],
                z[ev[2]],
                z[ev[3]],
                element_diag);

        for (int v = 0; v < 10; ++v) {
            const idx_t idx = ev[v];

#pragma omp atomic update
            diag[idx] += element_diag[v];
        }
    }

    return 0;
}

//-------------------------------------------
/// iso-parametric version
//-------------------------------------------

SFEM_INLINE static real_t tet10_measure(const geom_t* const SFEM_RESTRICT x,
                                        const geom_t* const SFEM_RESTRICT y,
                                        const geom_t* const SFEM_RESTRICT z,
                                        // Quadrature point
                                        const real_t qx, const real_t qy, const real_t qz) {
    const real_t x0 = 4 * qz;
    const real_t x1 = x0 - 1;
    const real_t x2 = 4 * qy;
    const real_t x3 = 4 * qx;
    const real_t x4 = x3 - 4;
    const real_t x5 = -8 * qz - x2 - x4;
    const real_t x6 = -x3 * y[4];
    const real_t x7 = x0 + x2;
    const real_t x8 = x3 + x7 - 3;
    const real_t x9 = x8 * y[0];
    const real_t x10 = -x2 * y[6] + x9;
    const real_t x11 = x1 * y[3] + x10 + x2 * y[9] + x3 * y[8] + x5 * y[7] + x6;
    const real_t x12 = -x2 * z[6];
    const real_t x13 = -x0 * z[7];
    const real_t x14 = x3 - 1;
    const real_t x15 = x8 * z[0];
    const real_t x16 = -8 * qx - x7 + 4;
    const real_t x17 = x0 * z[8] + x12 + x13 + x14 * z[1] + x15 + x16 * z[4] + x2 * z[5];
    const real_t x18 = x2 - 1;
    const real_t x19 = -8 * qy - x0 - x4;
    const real_t x20 = -x3 * x[4];
    const real_t x21 = x8 * x[0];
    const real_t x22 = -x0 * x[7] + x21;
    const real_t x23 = (1.0 / 6.0) * x0 * x[9] + (1.0 / 6.0) * x18 * x[2] +
                       (1.0 / 6.0) * x19 * x[6] + (1.0 / 6.0) * x20 + (1.0 / 6.0) * x22 +
                       (1.0 / 6.0) * x3 * x[5];
    const real_t x24 = -x0 * y[7];
    const real_t x25 = x0 * y[8] + x10 + x14 * y[1] + x16 * y[4] + x2 * y[5] + x24;
    const real_t x26 = x15 - x3 * z[4];
    const real_t x27 = x1 * z[3] + x12 + x2 * z[9] + x26 + x3 * z[8] + x5 * z[7];
    const real_t x28 = x0 * y[9] + x18 * y[2] + x19 * y[6] + x24 + x3 * y[5] + x6 + x9;
    const real_t x29 = -x2 * x[6];
    const real_t x30 = (1.0 / 6.0) * x1 * x[3] + (1.0 / 6.0) * x2 * x[9] + (1.0 / 6.0) * x20 +
                       (1.0 / 6.0) * x21 + (1.0 / 6.0) * x29 + (1.0 / 6.0) * x3 * x[8] +
                       (1.0 / 6.0) * x5 * x[7];
    const real_t x31 = x0 * z[9] + x13 + x18 * z[2] + x19 * z[6] + x26 + x3 * z[5];
    const real_t x32 = (1.0 / 6.0) * x0 * x[8] + (1.0 / 6.0) * x14 * x[1] +
                       (1.0 / 6.0) * x16 * x[4] + (1.0 / 6.0) * x2 * x[5] + (1.0 / 6.0) * x22 +
                       (1.0 / 6.0) * x29;
    return x11 * x17 * x23 - x11 * x31 * x32 - x17 * x28 * x30 - x23 * x25 * x27 + x25 * x30 * x31 +
           x27 * x28 * x32;
}

SFEM_INLINE static void tet10_transform(const geom_t* const SFEM_RESTRICT x,
                                        const geom_t* const SFEM_RESTRICT y,
                                        const geom_t* const SFEM_RESTRICT z,
                                        // Quadrature point
                                        const real_t qx, const real_t qy, const real_t qz,
                                        // Output
                                        real_t* const SFEM_RESTRICT out_x,
                                        real_t* const SFEM_RESTRICT out_y,
                                        real_t* const SFEM_RESTRICT out_z) {
    const real_t x0 = 4 * qx;
    const real_t x1 = qy * x0;
    const real_t x2 = qz * x0;
    const real_t x3 = 4 * qy;
    const real_t x4 = qz * x3;
    const real_t x5 = 2 * qx - 1;
    const real_t x6 = qx * x5;
    const real_t x7 = 2 * qy;
    const real_t x8 = qy * (x7 - 1);
    const real_t x9 = 2 * qz;
    const real_t x10 = qz * (x9 - 1);
    const real_t x11 = -4 * qz - x0 - x3 + 4;
    const real_t x12 = qx * x11;
    const real_t x13 = qy * x11;
    const real_t x14 = qz * x11;
    const real_t x15 = (-x5 - x7 - x9) * (-qx - qy - qz + 1);

    *out_x = x[0] * x15 + x[1] * x6 + x[2] * x8 + x[3] * x10 + x[4] * x12 + x[5] * x1 + x[6] * x13 +
             x[7] * x14 + x[8] * x2 + x[9] * x4;
    *out_y = y[0] * x15 + y[1] * x6 + y[2] * x8 + y[3] * x10 + y[4] * x12 + y[5] * x1 + y[6] * x13 +
             y[7] * x14 + y[8] * x2 + y[9] * x4;
    *out_z = z[0] * x15 + z[1] * x6 + z[2] * x8 + z[3] * x10 + z[4] * x12 + z[5] * x1 + z[6] * x13 +
             z[7] * x14 + z[8] * x2 + z[9] * x4;
}

int hex8_to_isoparametric_tet10_resample_field_local(
        // Mesh
        const ptrdiff_t nelements,          // number of elements
        const ptrdiff_t nnodes,             // number of nodes
        idx_t** const SFEM_RESTRICT elems,  // connectivity
        geom_t** const SFEM_RESTRICT xyz,   // coordinates
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n,       // number of nodes in each direction
        const ptrdiff_t* const SFEM_RESTRICT stride,  // stride of the data
        const geom_t* const SFEM_RESTRICT origin,     // origin of the domain
        const geom_t* const SFEM_RESTRICT delta,      // delta of the domain
        const real_t* const SFEM_RESTRICT data,       // SDF
        // Output
        real_t* const SFEM_RESTRICT weighted_field) {
    //

#define WENO 1

    printf("============================================================\n");
    printf("Start: hex8_to_isoparametric_tet10_resample_field_local. Interpolation: %s\n",
           (WENO == 1 ? "Weno" : "Linear"));
    printf("============================================================\n");
    printf("dx = %e, dy = %e, dz = %e\n", delta[0], delta[1], delta[2]);
    printf("============================================================\n");

    const real_t ox = (real_t)origin[0];
    const real_t oy = (real_t)origin[1];
    const real_t oz = (real_t)origin[2];

    const real_t dx = (real_t)delta[0];
    const real_t dy = (real_t)delta[1];
    const real_t dz = (real_t)delta[2];

#pragma omp parallel
    {
#pragma omp for  // nowait
        /// Loop over the elements of the mesh
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            // printf("element = %d\n", i);

            idx_t ev[10];

            // ISOPARAMETRIC
            geom_t x[10], y[10], z[10];

            real_t hex8_f[8];
            real_t coeffs[8];

            real_t tet10_f[10];
            real_t element_field[10];

            // loop over the 4 vertices of the tetrahedron
            // UNROLL_ZERO ?
            for (int v = 0; v < 10; ++v) {
                ev[v] = elems[v][i];
            }

            // ISOPARAMETRIC
            for (int v = 0; v < 10; ++v) {
                x[v] = xyz[0][ev[v]];  // x-coordinates
                y[v] = xyz[1][ev[v]];  // y-coordinates
                z[v] = xyz[2][ev[v]];  // z-coordinates
            }

            memset(element_field, 0,
                   10 * sizeof(real_t));  // set to zero the element field

            // SUBPARAMETRIC (for iso-parametric tassellation of tet10 might be necessary)
            for (int q = 0; q < TET4_NQP; q++) {  // loop over the quadrature points

                // if (q == 8) {
                //     break;
                // }

                const real_t measure = tet10_measure(x, y, z, tet4_qx[q], tet4_qy[q], tet4_qz[q]);

                assert(measure > 0);
                const real_t dV = measure * tet4_qw[q];
                // printf("dV[%d]: %e\n", q, dV);

                real_t g_qx, g_qy, g_qz;
                // Transform quadrature point to physical space
                // g_qx, g_qy, g_qz are the coordinates of the quadrature point in the physical
                // space
                tet10_transform(x, y, z, tet4_qx[q], tet4_qy[q], tet4_qz[q], &g_qx, &g_qy, &g_qz);
                tet10_dual_basis_hrt(tet4_qx[q], tet4_qy[q], tet4_qz[q], tet10_f);

                ///// ======================================================

                const real_t grid_x = (g_qx - ox) / dx;
                const real_t grid_y = (g_qy - oy) / dy;
                const real_t grid_z = (g_qz - oz) / dz;

                const ptrdiff_t i = floor(grid_x);
                const ptrdiff_t j = floor(grid_y);
                const ptrdiff_t k = floor(grid_z);

                // If outside
                if (i < 0 || j < 0 || k < 0 || (i + 1 >= n[0]) || (j + 1 >= n[1]) ||
                    (k + 1 >= n[2])) {
                    fprintf(stderr,
                            "warning (%g, %g, %g) (%ld, %ld, %ld) outside domain  (%ld, %ld, "
                            "%ld)!\n",
                            g_qx,
                            g_qy,
                            g_qz,
                            i,
                            j,
                            k,
                            n[0],
                            n[1],
                            n[2]);
                    continue;
                }

                // Integrate field
                {
#if WENO == 1

                    // printf("origin = (%f, %f, %f)\n", ox, oy, oz);
                    real_t eval_field = hex_aa_8_eval_weno4_3D(g_qx,
                                                               g_qy,
                                                               g_qz,  //
                                                               ox,
                                                               oy,
                                                               oz,                            //
                                                               (dx + dy + dz) * (1.0 / 3.0),  //
                                                               i,
                                                               j,
                                                               k,
                                                               stride,
                                                               data);  //
#else

                    // Get the reminder [0, 1]
                    real_t l_x = (grid_x - i);
                    real_t l_y = (grid_y - j);
                    real_t l_z = (grid_z - k);

                    assert(l_x >= -1e-8);
                    assert(l_y >= -1e-8);
                    assert(l_z >= -1e-8);

                    assert(l_x <= 1 + 1e-8);
                    assert(l_y <= 1 + 1e-8);
                    assert(l_z <= 1 + 1e-8);

                    hex_aa_8_eval_fun(l_x, l_y, l_z, hex8_f);
                    hex_aa_8_collect_coeffs(stride, i, j, k, data, coeffs);

                    real_t eval_field = 0;
                    // UNROLL_ZERO?
                    for (int edof_j = 0; edof_j < 8; edof_j++) {
                        eval_field += hex8_f[edof_j] * coeffs[edof_j];
                    }
#endif
                    // UNROLL_ZERO?
                    for (int edof_i = 0; edof_i < 10; edof_i++) {
                        element_field[edof_i] += eval_field * tet10_f[edof_i] * dV;
                    }  // end edof_i loop
                }
            }  // end quadrature loop

            // UNROLL_ZERO?
            for (int v = 0; v < 10; ++v) {
#pragma omp atomic update
                weighted_field[ev[v]] += element_field[v];

            }  // end vertex loop
        }      // end element loop
    }          // end parallel region

    return 0;
}

SFEM_INLINE static void isoparametric_lumped_mass_kernel_hrt(
        const real_t dV,
        // Quadrature
        const real_t qx, const real_t qy, const real_t qz,
        real_t* const SFEM_RESTRICT element_diag) {
    const real_t x0 = 4 * qx;
    const real_t x1 = qy * qz;
    const real_t x2 = 2 * qx - 1;
    const real_t x3 = qx * x2;
    const real_t x4 = 2 * qy;
    const real_t x5 = qy * (x4 - 1);
    const real_t x6 = 2 * qz;
    const real_t x7 = qz * (x6 - 1);
    const real_t x8 = qx + qy + qz - 1;
    const real_t x9 = qx * x8;
    const real_t x10 = qy * x8;
    const real_t x11 = qz * x8;
    const real_t x12 = x8 * (x2 + x4 + x6);
    const real_t x13 = (5.0 / 18.0) * dV;
    const real_t x14 = POW2(qx);
    const real_t x15 = POW2(qy);
    const real_t x16 = POW2(qz);
    const real_t x17 = 11 * qx;
    const real_t x18 = -qy * x17 + 11 * x11 + x12 + 4 * x7;
    const real_t x19 = -qz * x17 + 11 * x10 + 4 * x5;
    const real_t x20 = (40.0 / 27.0) * dV;
    const real_t x21 = qx * qy;
    const real_t x22 = -24 * qx + 21 * x14 + 4;
    const real_t x23 = -24 * qy + 21 * x15;
    const real_t x24 = qx * qz;
    const real_t x25 = -11 * x1 + 4 * x3 + 11 * x9;
    const real_t x26 = -24 * qz + 21 * x16;
    element_diag[0] +=
            x12 * x13 * (qy * x0 + qz * x0 + 4 * x1 - x10 - x11 + 10 * x12 + x3 + x5 + x7 - x9);
    element_diag[1] += qx * x13 * (42 * (POW3(qx)) + 14 * qx - 45 * x14 - 1);
    element_diag[2] += qy * x13 * (42 * (POW3(qy)) + 14 * qy - 45 * x15 - 1);
    element_diag[3] += qz * x13 * (42 * (POW3(qz)) + 14 * qz - 45 * x16 - 1);
    element_diag[4] += -x20 * x9 * (16 * x1 + x18 + x19 + x3 - 46 * x9);
    element_diag[5] += x20 * x21 * (84 * x21 + x22 + x23);
    element_diag[6] += -x10 * x20 * (-46 * x10 + x18 + 16 * x24 + x25 + x5);
    element_diag[7] += -x11 * x20 * (-46 * x11 + x12 + x19 + 16 * x21 + x25 + x7);
    element_diag[8] += x20 * x24 * (x22 + 84 * x24 + x26);
    element_diag[9] += x1 * x20 * (84 * x1 + x23 + x26 + 4);
}

int isoparametric_tet10_assemble_dual_mass_vector(const ptrdiff_t nelements, const ptrdiff_t nnodes,
                                                  idx_t** const SFEM_RESTRICT elems,
                                                  geom_t** const SFEM_RESTRICT xyz,
                                                  real_t* const diag) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[10];  // Element indices
        real_t element_diag[10];
        geom_t x[10], y[10], z[10];

#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elems[v][i];
        }

        for (int v = 0; v < 10; ++v) {
            x[v] = xyz[0][ev[v]];  // x-coordinates
            y[v] = xyz[1][ev[v]];  // y-coordinates
            z[v] = xyz[2][ev[v]];  // z-coordinates
        }

        memset(element_diag, 0, 10 * sizeof(real_t));

        // We do this numerical integration due to the det J
        for (int q = 0; q < TET4_NQP; q++) {  // loop over the quadrature points
            real_t dV = tet10_measure(x, y, z, tet4_qx[q], tet4_qy[q], tet4_qz[q]) * tet4_qw[q];
            isoparametric_lumped_mass_kernel_hrt(dV,
                                                 // Quadrature
                                                 tet4_qx[q],
                                                 tet4_qy[q],
                                                 tet4_qz[q],
                                                 element_diag);
        }

        for (int v = 0; v < 10; ++v) {
            const idx_t idx = ev[v];

#pragma omp atomic update
            diag[idx] += element_diag[v];
        }
    }

    return 0;
}

int tet10_assemble_dual_mass_vector(const ptrdiff_t nelements, const ptrdiff_t nnodes,
                                    idx_t** const SFEM_RESTRICT elems,
                                    geom_t** const SFEM_RESTRICT xyz, real_t* const mass_vector) {
    int SFEM_ENABLE_ISOPARAMETRIC = 0;
    SFEM_READ_ENV(SFEM_ENABLE_ISOPARAMETRIC, atoi);

    if (SFEM_ENABLE_ISOPARAMETRIC) {
        return isoparametric_tet10_assemble_dual_mass_vector(
                nelements, nnodes, elems, xyz, mass_vector);
    } else {
        return subparametric_tet10_assemble_dual_mass_vector(
                nelements, nnodes, elems, xyz, mass_vector);
    }
}

int hex8_to_tet10_resample_field_local(
        // Mesh
        const ptrdiff_t nelements,          // number of elements
        const ptrdiff_t nnodes,             // number of nodes
        idx_t** const SFEM_RESTRICT elems,  // connectivity
        geom_t** const SFEM_RESTRICT xyz,   // coordinates
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n,       // number of nodes in each direction
        const ptrdiff_t* const SFEM_RESTRICT stride,  // stride of the data
        const geom_t* const SFEM_RESTRICT origin,     // origin of the domain
        const geom_t* const SFEM_RESTRICT delta,      // delta of the domain
        const real_t* const SFEM_RESTRICT data,       // SDF
        // Output
        real_t* const SFEM_RESTRICT weighted_field) {
    int SFEM_ENABLE_ISOPARAMETRIC = 0;
    SFEM_READ_ENV(SFEM_ENABLE_ISOPARAMETRIC, atoi);

    if (1 | SFEM_ENABLE_ISOPARAMETRIC) {
        return hex8_to_isoparametric_tet10_resample_field_local(
                nelements, nnodes, elems, xyz, n, stride, origin, delta, data, weighted_field);
    } else {
        return hex8_to_subparametric_tet10_resample_field_local(
                nelements, nnodes, elems, xyz, n, stride, origin, delta, data, weighted_field);
    }
}
