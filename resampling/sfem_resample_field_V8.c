#include "sfem_resample_field.h"

#include "mass.h"
#include "matrixio_array.h"
#include "sfem_resample_V.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#define real_t double

#include "quadratures_rule.h"


#define real_type real_t
#define SFEM_RESTRICT __restrict__

#define SFEM_RESAMPLE_GAP_DUAL

#define _VL8_ 8
typedef double vec8_double
        __attribute__((vector_size(_VL8_ * sizeof(double)), aligned(sizeof(double))));

typedef ptrdiff_t vec8_int64
        __attribute__((vector_size(_VL8_ * sizeof(ptrdiff_t)), aligned(sizeof(ptrdiff_t))));

SFEM_INLINE vec8_int64 floor_V8(const vec8_double x) {
    const vec8_int64 res = __builtin_convertvector(x, vec8_int64);
    return res;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_measure_V8 ///////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
SFEM_INLINE static vec8_double tet4_measure_V8(
        // X-coordinates
        const vec8_double px0,
        const vec8_double px1,
        const vec8_double px2,
        const vec8_double px3,
        // Y-coordinates
        const vec8_double py0,
        const vec8_double py1,
        const vec8_double py2,
        const vec8_double py3,
        // Z-coordinates
        const vec8_double pz0,
        const vec8_double pz1,
        const vec8_double pz2,
        const vec8_double pz3) {
    //
    // determinant of the Jacobian
    // M = [px0, py0, pz0, 1]
    //     [px1, py1, pz1, 1]
    //     [px2, py2, pz2, 1]
    //     [px3, py3, pz3, 1]
    //
    // V = (1/6) * det(M)

    const vec8_double x0 = -pz0 + pz3;
    const vec8_double x1 = -py0 + py2;
    const vec8_double x2 = -(1.0 / 6.0) * px0 + (1.0 / 6.0) * px1;
    const vec8_double x3 = -py0 + py3;
    const vec8_double x4 = -pz0 + pz2;
    const vec8_double x5 = -py0 + py1;
    const vec8_double x6 = -(1.0 / 6.0) * px0 + (1.0 / 6.0) * px2;
    const vec8_double x7 = -pz0 + pz1;
    const vec8_double x8 = -(1.0 / 6.0) * px0 + (1.0 / 6.0) * px3;

    return x0 * x1 * x2 - x0 * x5 * x6 - x1 * x7 * x8 - x2 * x3 * x4 + x3 * x6 * x7 + x4 * x5 * x8;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_transform_V8 /////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
SFEM_INLINE static void tet4_transform_V8(
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
        const vec8_double px0,
        const vec8_double px1,
        const vec8_double px2,
        const vec8_double px3,
        // Y-coordinates
        const vec8_double py0,
        const vec8_double py1,
        const vec8_double py2,
        const vec8_double py3,
        // Z-coordinates
        const vec8_double pz0,
        const vec8_double pz1,
        const vec8_double pz2,
        const vec8_double pz3,
        // Quadrature point
        const vec8_double qx,
        const vec8_double qy,
        const vec8_double qz,
        // Output
        vec8_double* const SFEM_RESTRICT out_x,
        vec8_double* const SFEM_RESTRICT out_y,
        vec8_double* const SFEM_RESTRICT out_z) {
    //
    //
    *out_x = px0 + qx * (-px0 + px1) + qy * (-px0 + px2) + qz * (-px0 + px3);
    *out_y = py0 + qx * (-py0 + py1) + qy * (-py0 + py2) + qz * (-py0 + py3);
    *out_z = pz0 + qx * (-pz0 + pz1) + qy * (-pz0 + pz2) + qz * (-pz0 + pz3);
}  // end tet4_transform_v2

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// hex_aa_8_collect_coeffs ////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
SFEM_INLINE static void hex_aa_8_eval_fun_V8(
        // Quadrature point (local coordinates)
        // With respect to the hat functions of a cube element
        // In a local coordinate system
        const vec8_double x,
        const vec8_double y,
        const vec8_double z,

        // Output
        vec8_double* const SFEM_RESTRICT f0,
        vec8_double* const SFEM_RESTRICT f1,
        vec8_double* const SFEM_RESTRICT f2,
        vec8_double* const SFEM_RESTRICT f3,
        vec8_double* const SFEM_RESTRICT f4,
        vec8_double* const SFEM_RESTRICT f5,
        vec8_double* const SFEM_RESTRICT f6,
        vec8_double* const SFEM_RESTRICT f7) {
    //
    *f0 = (1.0 - x) * (1.0 - y) * (1.0 - z);
    *f1 = x * (1.0 - y) * (1.0 - z);
    *f2 = x * y * (1.0 - z);
    *f3 = (1.0 - x) * y * (1.0 - z);
    *f4 = (1.0 - x) * (1.0 - y) * z;
    *f5 = x * (1.0 - y) * z;
    *f6 = x * y * z;
    *f7 = (1.0 - x) * y * z;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// hex_aa_8_collect_coeffs_V8 ////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
SFEM_INLINE static void hex_aa_8_collect_coeffs_V8(
        const vec8_int64 stride0,
        const vec8_int64 stride1,
        const vec8_int64 stride2,
        const vec8_int64 i,
        const vec8_int64 j,
        const vec8_int64 k,
        // Attention this is geometric data transformed to solver data!
        const real_t* const SFEM_RESTRICT data,
        //
        vec8_double* SFEM_RESTRICT out0,
        vec8_double* SFEM_RESTRICT out1,
        vec8_double* SFEM_RESTRICT out2,
        vec8_double* SFEM_RESTRICT out3,
        vec8_double* SFEM_RESTRICT out4,
        vec8_double* SFEM_RESTRICT out5,
        vec8_double* SFEM_RESTRICT out6,
        vec8_double* SFEM_RESTRICT out7) {
    //

    const vec8_int64 i0 = i * stride0 + j * stride1 + k * stride2;
    const vec8_int64 i1 = (i + 1) * stride0 + j * stride1 + k * stride2;
    const vec8_int64 i2 = (i + 1) * stride0 + (j + 1) * stride1 + k * stride2;
    const vec8_int64 i3 = i * stride0 + (j + 1) * stride1 + k * stride2;
    const vec8_int64 i4 = i * stride0 + j * stride1 + (k + 1) * stride2;
    const vec8_int64 i5 = (i + 1) * stride0 + j * stride1 + (k + 1) * stride2;
    const vec8_int64 i6 = (i + 1) * stride0 + (j + 1) * stride1 + (k + 1) * stride2;
    const vec8_int64 i7 = i * stride0 + (j + 1) * stride1 + (k + 1) * stride2;

    *out0 = (vec8_double){data[i0[0]],
                          data[i0[1]],
                          data[i0[2]],
                          data[i0[3]],
                          data[i0[4]],
                          data[i0[5]],
                          data[i0[6]],
                          data[i0[7]]};

    *out1 = (vec8_double){data[i1[0]],
                          data[i1[1]],
                          data[i1[2]],
                          data[i1[3]],
                          data[i1[4]],
                          data[i1[5]],
                          data[i1[6]],
                          data[i1[7]]};

    *out2 = (vec8_double){data[i2[0]],
                          data[i2[1]],
                          data[i2[2]],
                          data[i2[3]],
                          data[i2[4]],
                          data[i2[5]],
                          data[i2[6]],
                          data[i2[7]]};

    *out3 = (vec8_double){data[i3[0]],
                          data[i3[1]],
                          data[i3[2]],
                          data[i3[3]],
                          data[i3[4]],
                          data[i3[5]],
                          data[i3[6]],
                          data[i3[7]]};

    *out4 = (vec8_double){data[i4[0]],
                          data[i4[1]],
                          data[i4[2]],
                          data[i4[3]],
                          data[i4[4]],
                          data[i4[5]],
                          data[i4[6]],
                          data[i4[7]]};

    *out5 = (vec8_double){data[i5[0]],
                          data[i5[1]],
                          data[i5[2]],
                          data[i5[3]],
                          data[i5[4]],
                          data[i5[5]],
                          data[i5[6]],
                          data[i5[7]]};

    *out6 = (vec8_double){data[i6[0]],
                          data[i6[1]],
                          data[i6[2]],
                          data[i6[3]],
                          data[i6[4]],
                          data[i6[5]],
                          data[i6[6]],
                          data[i6[7]]};

    *out7 = (vec8_double){data[i7[0]],
                          data[i7[1]],
                          data[i7[2]],
                          data[i7[3]],
                          data[i7[4]],
                          data[i7[5]],
                          data[i7[6]],
                          data[i7[7]]};
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_local_v2 //////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int tet4_resample_field_local_V8_aligned(
        // Mesh
        const ptrdiff_t start_nelement,
        const ptrdiff_t end_nelement,
        const ptrdiff_t nnodes,
        idx_t** const SFEM_RESTRICT elems,
        geom_t** const SFEM_RESTRICT xyz,
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n,
        const ptrdiff_t* const SFEM_RESTRICT stride,
        const geom_t* const SFEM_RESTRICT origin,
        const geom_t* const SFEM_RESTRICT delta,
        const real_type* const SFEM_RESTRICT data,
        // Output
        real_type* const SFEM_RESTRICT weighted_field) {
    //
    printf("============================================================\n");
    printf("Start: tet4_resample_field_local_V8_aligned  V8 [%s] \n", __FILE__);
    printf("============================================================\n");
    //
    const double ox = (double)origin[0];
    const double oy = (double)origin[1];
    const double oz = (double)origin[2];

    const double dx = (double)delta[0];
    const double dy = (double)delta[1];
    const double dz = (double)delta[2];

    vec8_int64 stride0 = {stride[0],  //
                          stride[0],
                          stride[0],
                          stride[0],
                          stride[0],
                          stride[0],
                          stride[0],
                          stride[0]};

    vec8_int64 stride1 = {stride[1],  //
                          stride[1],
                          stride[1],
                          stride[1],
                          stride[1],
                          stride[1],
                          stride[1],
                          stride[1]};

    vec8_int64 stride2 = {stride[2],  //
                          stride[2],
                          stride[2],
                          stride[2],
                          stride[2],
                          stride[2],
                          stride[2],
                          stride[2]};

    //////////////////////////////////////////////////////////////////////
    // Loop over the elements
    for (ptrdiff_t element_i = start_nelement; element_i < end_nelement; element_i += 8) {
        //
        vec8_int64 ev0 = {elems[0][element_i + 0],
                          elems[0][element_i + 1],
                          elems[0][element_i + 2],
                          elems[0][element_i + 3],
                          elems[0][element_i + 4],
                          elems[0][element_i + 5],
                          elems[0][element_i + 6],
                          elems[0][element_i + 7]};

        vec8_int64 ev1 = {elems[1][element_i + 0],
                          elems[1][element_i + 1],
                          elems[1][element_i + 2],
                          elems[1][element_i + 3],
                          elems[1][element_i + 4],
                          elems[1][element_i + 5],
                          elems[1][element_i + 6],
                          elems[1][element_i + 7]};

        vec8_int64 ev2 = {elems[2][element_i + 0],
                          elems[2][element_i + 1],
                          elems[2][element_i + 2],
                          elems[2][element_i + 3],
                          elems[2][element_i + 4],
                          elems[2][element_i + 5],
                          elems[2][element_i + 6],
                          elems[2][element_i + 7]};

        vec8_int64 ev3 = {elems[3][element_i + 0],
                          elems[3][element_i + 1],
                          elems[3][element_i + 2],
                          elems[3][element_i + 3],
                          elems[3][element_i + 4],
                          elems[3][element_i + 5],
                          elems[3][element_i + 6],
                          elems[3][element_i + 7]};

        // real_type x[4], y[4], z[4];

        const vec8_double zeros = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        // Vertices coordinates of the tetrahedron
        vec8_double x0 = zeros, x1 = zeros, x2 = zeros, x3 = zeros;

        vec8_double y0 = zeros, y1 = zeros, y2 = zeros, y3 = zeros;
        vec8_double z0 = zeros, z1 = zeros, z2 = zeros, z3 = zeros;

        // real_type hex8_f[8];
        vec8_double hex8_f0 = zeros, hex8_f1 = zeros, hex8_f2 = zeros, hex8_f3 = zeros,
                    hex8_f4 = zeros, hex8_f5 = zeros, hex8_f6 = zeros, hex8_f7 = zeros;

        // real_type coeffs[8];
        vec8_double coeffs0 = zeros, coeffs1 = zeros, coeffs2 = zeros, coeffs3 = zeros,
                    coeffs4 = zeros, coeffs5 = zeros, coeffs6 = zeros, coeffs7 = zeros;

        // real_type tet4_f[4];
        vec8_double tet4_f0 = zeros, tet4_f1 = zeros, tet4_f2 = zeros, tet4_f3 = zeros;

        // real_type element_field[4];
        vec8_double element_field0 = zeros, element_field1 = zeros, element_field2 = zeros,
                    element_field3 = zeros;

        // copy the coordinates of the vertices
        {
            x0 = (vec8_double){(double)xyz[0][ev0[0]],
                               (double)xyz[0][ev0[1]],
                               (double)xyz[0][ev0[2]],
                               (double)xyz[0][ev0[3]],
                               (double)xyz[0][ev0[4]],
                               (double)xyz[0][ev0[5]],
                               (double)xyz[0][ev0[6]],
                               (double)xyz[0][ev0[7]]};

            x1 = (vec8_double){(double)xyz[0][ev1[0]],
                               (double)xyz[0][ev1[1]],
                               (double)xyz[0][ev1[2]],
                               (double)xyz[0][ev1[3]],
                               (double)xyz[0][ev1[4]],
                               (double)xyz[0][ev1[5]],
                               (double)xyz[0][ev1[6]],
                               (double)xyz[0][ev1[7]]};

            x2 = (vec8_double){(double)xyz[0][ev2[0]],
                               (double)xyz[0][ev2[1]],
                               (double)xyz[0][ev2[2]],
                               (double)xyz[0][ev2[3]],
                               (double)xyz[0][ev2[4]],
                               (double)xyz[0][ev2[5]],
                               (double)xyz[0][ev2[6]],
                               (double)xyz[0][ev2[7]]};

            x3 = (vec8_double){(double)xyz[0][ev3[0]],
                               (double)xyz[0][ev3[1]],
                               (double)xyz[0][ev3[2]],
                               (double)xyz[0][ev3[3]],
                               (double)xyz[0][ev3[4]],
                               (double)xyz[0][ev3[5]],
                               (double)xyz[0][ev3[6]],
                               (double)xyz[0][ev3[7]]};

            y0 = (vec8_double){(double)xyz[1][ev0[0]],
                               (double)xyz[1][ev0[1]],
                               (double)xyz[1][ev0[2]],
                               (double)xyz[1][ev0[3]],
                               (double)xyz[1][ev0[4]],
                               (double)xyz[1][ev0[5]],
                               (double)xyz[1][ev0[6]],
                               (double)xyz[1][ev0[7]]};

            y1 = (vec8_double){(double)xyz[1][ev1[0]],
                               (double)xyz[1][ev1[1]],
                               (double)xyz[1][ev1[2]],
                               (double)xyz[1][ev1[3]],
                               (double)xyz[1][ev1[4]],
                               (double)xyz[1][ev1[5]],
                               (double)xyz[1][ev1[6]],
                               (double)xyz[1][ev1[7]]};

            y2 = (vec8_double){(double)xyz[1][ev2[0]],
                               (double)xyz[1][ev2[1]],
                               (double)xyz[1][ev2[2]],
                               (double)xyz[1][ev2[3]],
                               (double)xyz[1][ev2[4]],
                               (double)xyz[1][ev2[5]],
                               (double)xyz[1][ev2[6]],
                               (double)xyz[1][ev2[7]]};

            y3 = (vec8_double){(double)xyz[1][ev3[0]],
                               (double)xyz[1][ev3[1]],
                               (double)xyz[1][ev3[2]],
                               (double)xyz[1][ev3[3]],
                               (double)xyz[1][ev3[4]],
                               (double)xyz[1][ev3[5]],
                               (double)xyz[1][ev3[6]],
                               (double)xyz[1][ev3[7]]};

            z0 = (vec8_double){(double)xyz[2][ev0[0]],
                               (double)xyz[2][ev0[1]],
                               (double)xyz[2][ev0[2]],
                               (double)xyz[2][ev0[3]],
                               (double)xyz[2][ev0[4]],
                               (double)xyz[2][ev0[5]],
                               (double)xyz[2][ev0[6]],
                               (double)xyz[2][ev0[7]]};

            z1 = (vec8_double){(double)xyz[2][ev1[0]],
                               (double)xyz[2][ev1[1]],
                               (double)xyz[2][ev1[2]],
                               (double)xyz[2][ev1[3]],
                               (double)xyz[2][ev1[4]],
                               (double)xyz[2][ev1[5]],
                               (double)xyz[2][ev1[6]],
                               (double)xyz[2][ev1[7]]};

            z2 = (vec8_double){(double)xyz[2][ev2[0]],
                               (double)xyz[2][ev2[1]],
                               (double)xyz[2][ev2[2]],
                               (double)xyz[2][ev2[3]],
                               (double)xyz[2][ev2[4]],
                               (double)xyz[2][ev2[5]],
                               (double)xyz[2][ev2[6]],
                               (double)xyz[2][ev2[7]]};

            z3 = (vec8_double){(double)xyz[2][ev3[0]],
                               (double)xyz[2][ev3[1]],
                               (double)xyz[2][ev3[2]],
                               (double)xyz[2][ev3[3]],
                               (double)xyz[2][ev3[4]],
                               (double)xyz[2][ev3[5]],
                               (double)xyz[2][ev3[6]],
                               (double)xyz[2][ev3[7]]};
        }  // end copy the coordinates of the vertices

        // Volume of the tetrahedrons (8 at a time)
        const vec8_double theta_volume = tet4_measure_V8(x0,
                                                         x1,
                                                         x2,
                                                         x3,
                                                         //
                                                         y0,
                                                         y1,
                                                         y2,
                                                         y3,
                                                         //
                                                         z0,
                                                         z1,
                                                         z2,
                                                         z3);

        //////////////////////////////////////////////////////////////////////
        // loop over the quadrature points
        for (int quad_i = 0; quad_i < TET4_NQP; quad_i++) {
            //
            vec8_double g_qx, g_qy, g_qz;

            vec8_double tet4_qx_v = {tet4_qx[quad_i],
                                     tet4_qx[quad_i],
                                     tet4_qx[quad_i],
                                     tet4_qx[quad_i],
                                     tet4_qx[quad_i],
                                     tet4_qx[quad_i],
                                     tet4_qx[quad_i],
                                     tet4_qx[quad_i]};

            vec8_double tet4_qy_v = {tet4_qy[quad_i],
                                     tet4_qy[quad_i],
                                     tet4_qy[quad_i],
                                     tet4_qy[quad_i],
                                     tet4_qy[quad_i],
                                     tet4_qy[quad_i],
                                     tet4_qy[quad_i],
                                     tet4_qy[quad_i]};

            vec8_double tet4_qz_v = {tet4_qz[quad_i],
                                     tet4_qz[quad_i],
                                     tet4_qz[quad_i],
                                     tet4_qz[quad_i],
                                     tet4_qz[quad_i],
                                     tet4_qz[quad_i],
                                     tet4_qz[quad_i],
                                     tet4_qz[quad_i]};

            tet4_transform_V8(x0,
                              x1,
                              x2,
                              x3,
                              //
                              y0,
                              y1,
                              y2,
                              y3,
                              //
                              z0,
                              z1,
                              z2,
                              z3,
                              //
                              tet4_qx_v,
                              tet4_qy_v,
                              tet4_qz_v,
                              //
                              &g_qx,
                              &g_qy,
                              &g_qz);

#ifndef SFEM_RESAMPLE_GAP_DUAL
            // Standard basis function
            {
                // tet4_f[0] = 1 - tet4_qx[q] - tet4_qy[q] - tet4_qz[q];
                // tet4_f[1] = tet4_qx[q];
                // tet4_f[2] = tet4_qy[q];
                // tet4_f[2] = tet4_qz[q];
            }
#else
            // DUAL basis function
            {
                const vec8_double f0 = 1.0 - tet4_qx_v - tet4_qy_v - tet4_qz_v;
                const vec8_double f1 = tet4_qx_v;
                const vec8_double f2 = tet4_qy_v;
                const vec8_double f3 = tet4_qz_v;

                tet4_f0 = 4.0 * f0 - f1 - f2 - f3;
                tet4_f1 = -f0 + 4.0 * f1 - f2 - f3;
                tet4_f2 = -f0 - f1 + 4.0 * f2 - f3;
                tet4_f3 = -f0 - f1 - f2 + 4.0 * f3;
            }
#endif

            const vec8_double grid_x = (g_qx - ox) / dx;
            const vec8_double grid_y = (g_qy - oy) / dy;
            const vec8_double grid_z = (g_qz - oz) / dz;

            const vec8_int64 i = floor_V8(grid_x);
            const vec8_int64 j = floor_V8(grid_y);
            const vec8_int64 k = floor_V8(grid_z);

            //     // If outside
            //     if (i < 0 || j < 0 || k < 0 || (i + 1 >= n[0]) || (j + 1 >= n[1]) || (k + 1 >=
            //     n[2])) {
            //         fprintf(stderr,
            //                 "warning (%g, %g, %g) (%ld, %ld, %ld) outside domain  (%ld, %ld, "
            //                 "%ld)!\n",
            //                 g_qx,
            //                 g_qy,
            //                 g_qz,
            //                 i,
            //                 j,
            //                 k,
            //                 n[0],
            //                 n[1],
            //                 n[2]);
            //         continue;
            //     } // end if outside

            // Get the reminder [0, 1]
            vec8_double l_x = (grid_x - __builtin_convertvector(i, vec8_double));
            vec8_double l_y = (grid_y - __builtin_convertvector(j, vec8_double));
            vec8_double l_z = (grid_z - __builtin_convertvector(k, vec8_double));

            // Critical point
            hex_aa_8_eval_fun_V8(l_x,
                                 l_y,
                                 l_z,
                                 &hex8_f0,
                                 &hex8_f1,
                                 &hex8_f2,
                                 &hex8_f3,
                                 &hex8_f4,
                                 &hex8_f5,
                                 &hex8_f6,
                                 &hex8_f7);

            hex_aa_8_collect_coeffs_V8(stride0,
                                       stride1,
                                       stride2,
                                       i,
                                       j,
                                       k,
                                       data,
                                       &coeffs0,
                                       &coeffs1,
                                       &coeffs2,
                                       &coeffs3,
                                       &coeffs4,
                                       &coeffs5,
                                       &coeffs6,
                                       &coeffs7);

            // Integrate gap function
            {
                vec8_double eval_field = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                // UNROLL_ZERO
                // for (int edof_j = 0; edof_j < 8; edof_j++) {
                //     eval_field += hex8_f[edof_j] * coeffs[edof_j];
                // }
                eval_field += hex8_f0 * coeffs0;
                eval_field += hex8_f1 * coeffs1;
                eval_field += hex8_f2 * coeffs2;
                eval_field += hex8_f3 * coeffs3;
                eval_field += hex8_f4 * coeffs4;
                eval_field += hex8_f5 * coeffs5;
                eval_field += hex8_f6 * coeffs6;
                eval_field += hex8_f7 * coeffs7;

                // UNROLL_ZERO
                // for (int edof_i = 0; edof_i < 4; edof_i++) {
                //     element_field[edof_i] += eval_field * tet4_f[edof_i] * dV;
                // }  // end edof_i loop

                vec8_double dV = theta_volume * tet4_qw[quad_i];
                // dV = (vec8_double){1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

                element_field0 += eval_field * tet4_f0 * dV;
                element_field1 += eval_field * tet4_f1 * dV;
                element_field2 += eval_field * tet4_f2 * dV;
                element_field3 += eval_field * tet4_f3 * dV;

            }  // end integrate gap function

        }  // end for over the quadrature points

        // for (int v = 0; v < 4; ++v) {
        //     // Invert sign since distance field is negative insdide and positive outside

        //      weighted_field[ev[v]] += element_field[v];
        // }  // end vertex loop
        //   weighted_field[ev[0]] += element_field0;
        //   weighted_field[ev[1]] += element_field1;
        //   weighted_field[ev[2]] += element_field2;
        //   weighted_field[ev[3]] += element_field3;

        // // loop over the 4 vertices of the tetrahedron
        // idx_t ev[4];
        // for (int v = 0; v < 4; ++v) {
        //     ev[v] = elems[v][element_i];
        // }

        // for (int v = 0; v < 4; v++) {
        // Invert sign since distance field is negative insdide and positive outside
        weighted_field[elems[0][element_i + 0]] += element_field0[0];
        weighted_field[elems[0][element_i + 1]] += element_field0[1];
        weighted_field[elems[0][element_i + 2]] += element_field0[2];
        weighted_field[elems[0][element_i + 3]] += element_field0[3];
        weighted_field[elems[0][element_i + 4]] += element_field0[4];
        weighted_field[elems[0][element_i + 5]] += element_field0[5];
        weighted_field[elems[0][element_i + 6]] += element_field0[6];
        weighted_field[elems[0][element_i + 7]] += element_field0[7];

        weighted_field[elems[1][element_i + 0]] += element_field1[0];
        weighted_field[elems[1][element_i + 1]] += element_field1[1];
        weighted_field[elems[1][element_i + 2]] += element_field1[2];
        weighted_field[elems[1][element_i + 3]] += element_field1[3];
        weighted_field[elems[1][element_i + 4]] += element_field1[4];
        weighted_field[elems[1][element_i + 5]] += element_field1[5];
        weighted_field[elems[1][element_i + 6]] += element_field1[6];
        weighted_field[elems[1][element_i + 7]] += element_field1[7];

        weighted_field[elems[2][element_i + 0]] += element_field2[0];
        weighted_field[elems[2][element_i + 1]] += element_field2[1];
        weighted_field[elems[2][element_i + 2]] += element_field2[2];
        weighted_field[elems[2][element_i + 3]] += element_field2[3];
        weighted_field[elems[2][element_i + 4]] += element_field2[4];
        weighted_field[elems[2][element_i + 5]] += element_field2[5];
        weighted_field[elems[2][element_i + 6]] += element_field2[6];
        weighted_field[elems[2][element_i + 7]] += element_field2[7];

        weighted_field[elems[3][element_i + 0]] += element_field3[0];
        weighted_field[elems[3][element_i + 1]] += element_field3[1];
        weighted_field[elems[3][element_i + 2]] += element_field3[2];
        weighted_field[elems[3][element_i + 3]] += element_field3[3];
        weighted_field[elems[3][element_i + 4]] += element_field3[4];
        weighted_field[elems[3][element_i + 5]] += element_field3[5];
        weighted_field[elems[3][element_i + 6]] += element_field3[6];
        weighted_field[elems[3][element_i + 7]] += element_field3[7];

        // }  // end vertex loop

    }  // end for over elements

    //

    return 0;
}  // end tet4_resample_field_local_V8_aligned


//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_local_v2 //////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int tet4_resample_field_local_V8(
        // Mesh
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t** const SFEM_RESTRICT elems,
        geom_t** const SFEM_RESTRICT xyz,
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n,
        const ptrdiff_t* const SFEM_RESTRICT stride,
        const geom_t* const SFEM_RESTRICT origin,
        const geom_t* const SFEM_RESTRICT delta,
        const real_type* const SFEM_RESTRICT data,
        // Output
        real_type* const SFEM_RESTRICT weighted_field) {
    //
    const ptrdiff_t nelements_aligned = nelements - (nelements % 8);
    const ptrdiff_t nelements_tail = nelements % 8;

    printf("=============================================\n");
    printf("nelements_aligned = %ld\n", nelements_aligned);
    printf("nelements_tail =    %ld\n", nelements_tail);
    printf("=============================================\n");

    tet4_resample_field_local_V8_aligned(0,
                                         nelements_aligned,
                                         nnodes,
                                         elems,
                                         xyz,
                                         n,
                                         stride,
                                         origin,
                                         delta,
                                         data,
                                         weighted_field);

    if (nelements_tail > 0) {
        tet4_resample_field_local_v2(nelements_aligned,
                                     nelements,
                                     nnodes,
                                     elems,
                                     xyz,
                                     n,
                                     stride,
                                     origin,
                                     delta,
                                     data,
                                     weighted_field);
    }

    return 0;
}