#include "sfem_resample_field.h"

#include "sfem_resample_V.h"

#include "mass.h"
// #include "read_mesh.h"
#include "matrixio_array.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#define real_t double

#include "quadratures_rule.h"


#define real_type real_t
#define SFEM_RESTRICT __restrict__

#define SFEM_RESAMPLE_GAP_DUAL

#define _VL4_ 4
typedef double vec4_double
        __attribute__((vector_size(_VL4_ * sizeof(double)), aligned(sizeof(double))));

typedef ptrdiff_t vec4_int64
        __attribute__((vector_size(_VL4_ * sizeof(ptrdiff_t)), aligned(sizeof(ptrdiff_t))));

SFEM_INLINE vec4_int64 floor_V4(const vec4_double x) {
    const vec4_int64 res = __builtin_convertvector(x, vec4_int64);
    return res;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_measure_V8 ///////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
SFEM_INLINE static vec4_double tet4_measure_V4(
        // X-coordinates
        const vec4_double px0,
        const vec4_double px1,
        const vec4_double px2,
        const vec4_double px3,
        // Y-coordinates
        const vec4_double py0,
        const vec4_double py1,
        const vec4_double py2,
        const vec4_double py3,
        // Z-coordinates
        const vec4_double pz0,
        const vec4_double pz1,
        const vec4_double pz2,
        const vec4_double pz3) {
    //
    // determinant of the Jacobian
    // M = [px0, py0, pz0, 1]
    //     [px1, py1, pz1, 1]
    //     [px2, py2, pz2, 1]
    //     [px3, py3, pz3, 1]
    //
    // V = (1/6) * det(M)

    const vec4_double x0 = -pz0 + pz3;
    const vec4_double x1 = -py0 + py2;
    const vec4_double x2 = -(1.0 / 6.0) * px0 + (1.0 / 6.0) * px1;
    const vec4_double x3 = -py0 + py3;
    const vec4_double x4 = -pz0 + pz2;
    const vec4_double x5 = -py0 + py1;
    const vec4_double x6 = -(1.0 / 6.0) * px0 + (1.0 / 6.0) * px2;
    const vec4_double x7 = -pz0 + pz1;
    const vec4_double x8 = -(1.0 / 6.0) * px0 + (1.0 / 6.0) * px3;

    return x0 * x1 * x2 - x0 * x5 * x6 - x1 * x7 * x8 - x2 * x3 * x4 + x3 * x6 * x7 + x4 * x5 * x8;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_transform_V8 /////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
SFEM_INLINE static void tet4_transform_V4(
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
        const vec4_double px0,
        const vec4_double px1,
        const vec4_double px2,
        const vec4_double px3,
        // Y-coordinates
        const vec4_double py0,
        const vec4_double py1,
        const vec4_double py2,
        const vec4_double py3,
        // Z-coordinates
        const vec4_double pz0,
        const vec4_double pz1,
        const vec4_double pz2,
        const vec4_double pz3,
        // Quadra4ure point
        const vec4_double qx,
        const vec4_double qy,
        const vec4_double qz,
        // Output
        vec4_double* const SFEM_RESTRICT out_x,
        vec4_double* const SFEM_RESTRICT out_y,
        vec4_double* const SFEM_RESTRICT out_z) {
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
SFEM_INLINE static void hex_aa_8_eval_fun_V4(
        // Quadrature point (local coordinates)
        // With respect to the hat functions of a cube element
        // In a local coordinate system
        const vec4_double x,
        const vec4_double y,
        const vec4_double z,

        // Output
        vec4_double* const SFEM_RESTRICT f0,
        vec4_double* const SFEM_RESTRICT f1,
        vec4_double* const SFEM_RESTRICT f2,
        vec4_double* const SFEM_RESTRICT f3,
        vec4_double* const SFEM_RESTRICT f4,
        vec4_double* const SFEM_RESTRICT f5,
        vec4_double* const SFEM_RESTRICT f6,
        vec4_double* const SFEM_RESTRICT f7) {
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
SFEM_INLINE static void hex_aa_8_collect_coeffs_V4(
        const vec4_int64 stride0,
        const vec4_int64 stride1,
        const vec4_int64 stride2,
        const vec4_int64 i,
        const vec4_int64 j,
        const vec4_int64 k,
        // Attention this is geometric data transformed to solver data!
        const real_t* const SFEM_RESTRICT data,
        //
        vec4_double* SFEM_RESTRICT out0,
        vec4_double* SFEM_RESTRICT out1,
        vec4_double* SFEM_RESTRICT out2,
        vec4_double* SFEM_RESTRICT out3,
        vec4_double* SFEM_RESTRICT out4,
        vec4_double* SFEM_RESTRICT out5,
        vec4_double* SFEM_RESTRICT out6,
        vec4_double* SFEM_RESTRICT out7) {
    //

    const vec4_int64 i0 = i * stride0 + j * stride1 + k * stride2;
    const vec4_int64 i1 = (i + 1) * stride0 + j * stride1 + k * stride2;
    const vec4_int64 i2 = (i + 1) * stride0 + (j + 1) * stride1 + k * stride2;
    const vec4_int64 i3 = i * stride0 + (j + 1) * stride1 + k * stride2;
    const vec4_int64 i4 = i * stride0 + j * stride1 + (k + 1) * stride2;
    const vec4_int64 i5 = (i + 1) * stride0 + j * stride1 + (k + 1) * stride2;
    const vec4_int64 i6 = (i + 1) * stride0 + (j + 1) * stride1 + (k + 1) * stride2;
    const vec4_int64 i7 = i * stride0 + (j + 1) * stride1 + (k + 1) * stride2;

    *out0 = (vec4_double){data[i0[0]], data[i0[1]], data[i0[2]], data[i0[3]]};

    *out1 = (vec4_double){data[i1[0]], data[i1[1]], data[i1[2]], data[i1[3]]};

    *out2 = (vec4_double){data[i2[0]], data[i2[1]], data[i2[2]], data[i2[3]]};

    *out3 = (vec4_double){data[i3[0]], data[i3[1]], data[i3[2]], data[i3[3]]};

    *out4 = (vec4_double){data[i4[0]], data[i4[1]], data[i4[2]], data[i4[3]]};

    *out5 = (vec4_double){data[i5[0]], data[i5[1]], data[i5[2]], data[i5[3]]};

    *out6 = (vec4_double){data[i6[0]], data[i6[1]], data[i6[2]], data[i6[3]]};

    *out7 = (vec4_double){data[i7[0]], data[i7[1]], data[i7[2]], data[i7[3]]};
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_local_v4 //////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int tet4_resample_field_local_V4_aligned(
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
    printf("Start: tet4_resample_field_local_V4_aligned  V4 [%s] \n", __FILE__);
    printf("============================================================\n");
    //
    const double ox = (double)origin[0];
    const double oy = (double)origin[1];
    const double oz = (double)origin[2];

    const double dx = (double)delta[0];
    const double dy = (double)delta[1];
    const double dz = (double)delta[2];

    vec4_int64 stride0 = {stride[0], stride[0], stride[0], stride[0]};
    vec4_int64 stride1 = {stride[1], stride[1], stride[1], stride[1]};
    vec4_int64 stride2 = {stride[2], stride[2], stride[2], stride[2]};

    //////////////////////////////////////////////////////////////////////
    // Loop over the elements
    for (ptrdiff_t element_i = start_nelement; element_i < end_nelement; element_i += 4) {
        vec4_int64 ev0 = {elems[0][element_i + 0],
                          elems[0][element_i + 1],
                          elems[0][element_i + 2],
                          elems[0][element_i + 3]};

        vec4_int64 ev1 = {elems[1][element_i + 0],
                          elems[1][element_i + 1],
                          elems[1][element_i + 2],
                          elems[1][element_i + 3]};

        vec4_int64 ev2 = {elems[2][element_i + 0],
                          elems[2][element_i + 1],
                          elems[2][element_i + 2],
                          elems[2][element_i + 3]};

        vec4_int64 ev3 = {elems[3][element_i + 0],
                          elems[3][element_i + 1],
                          elems[3][element_i + 2],
                          elems[3][element_i + 3]};

        const vec4_double zeros = {0.0, 0.0, 0.0, 0.0};

        // Vertices coordinates of the tetrahedron
        vec4_double x0 = zeros,  //
                x1 = zeros,      //
                x2 = zeros,      //
                x3 = zeros;      //

        vec4_double y0 = zeros, y1 = zeros, y2 = zeros, y3 = zeros;
        vec4_double z0 = zeros, z1 = zeros, z2 = zeros, z3 = zeros;

        // real_type hex8_f[8];
        vec4_double hex8_f0 = zeros, hex8_f1 = zeros, hex8_f2 = zeros, hex8_f3 = zeros,
                    hex8_f4 = zeros, hex8_f5 = zeros, hex8_f6 = zeros, hex8_f7 = zeros;

        // real_type coeffs[8];
        vec4_double coeffs0 = zeros, coeffs1 = zeros, coeffs2 = zeros, coeffs3 = zeros,
                    coeffs4 = zeros, coeffs5 = zeros, coeffs6 = zeros, coeffs7 = zeros;

        // real_type tet4_f[4];
        vec4_double tet4_f0 = zeros, tet4_f1 = zeros, tet4_f2 = zeros, tet4_f3 = zeros;

        // real_type element_field[4];
        vec4_double element_field0 = zeros, element_field1 = zeros, element_field2 = zeros,
                    element_field3 = zeros;

        // copy the coordinates of the vertices
        {
            x0 = (vec4_double){(double)xyz[0][ev0[0]],
                               (double)xyz[0][ev0[1]],
                               (double)xyz[0][ev0[2]],
                               (double)xyz[0][ev0[3]]};

            x1 = (vec4_double){(double)xyz[0][ev1[0]],
                               (double)xyz[0][ev1[1]],
                               (double)xyz[0][ev1[2]],
                               (double)xyz[0][ev1[3]]};

            x2 = (vec4_double){(double)xyz[0][ev2[0]],
                               (double)xyz[0][ev2[1]],
                               (double)xyz[0][ev2[2]],
                               (double)xyz[0][ev2[3]]};

            x3 = (vec4_double){(double)xyz[0][ev3[0]],
                               (double)xyz[0][ev3[1]],
                               (double)xyz[0][ev3[2]],
                               (double)xyz[0][ev3[3]]};

            y0 = (vec4_double){(double)xyz[1][ev0[0]],
                               (double)xyz[1][ev0[1]],
                               (double)xyz[1][ev0[2]],
                               (double)xyz[1][ev0[3]]};

            y1 = (vec4_double){(double)xyz[1][ev1[0]],
                               (double)xyz[1][ev1[1]],
                               (double)xyz[1][ev1[2]],
                               (double)xyz[1][ev1[3]]};

            y2 = (vec4_double){(double)xyz[1][ev2[0]],
                               (double)xyz[1][ev2[1]],
                               (double)xyz[1][ev2[2]],
                               (double)xyz[1][ev2[3]]};

            y3 = (vec4_double){(double)xyz[1][ev3[0]],
                               (double)xyz[1][ev3[1]],
                               (double)xyz[1][ev3[2]],
                               (double)xyz[1][ev3[3]]};

            z0 = (vec4_double){(double)xyz[2][ev0[0]],
                               (double)xyz[2][ev0[1]],
                               (double)xyz[2][ev0[2]],
                               (double)xyz[2][ev0[3]]};

            z1 = (vec4_double){(double)xyz[2][ev1[0]],
                               (double)xyz[2][ev1[1]],
                               (double)xyz[2][ev1[2]],
                               (double)xyz[2][ev1[3]]};

            z2 = (vec4_double){(double)xyz[2][ev2[0]],
                               (double)xyz[2][ev2[1]],
                               (double)xyz[2][ev2[2]],
                               (double)xyz[2][ev2[3]]};

            z3 = (vec4_double){(double)xyz[2][ev3[0]],
                               (double)xyz[2][ev3[1]],
                               (double)xyz[2][ev3[2]],
                               (double)xyz[2][ev3[3]]};

            //////////////////////////////////////////////////////////////////////
        }  // end copy the coordinates of the vertices

        // Volume of the tetrahedrons (8 at a time)
        const vec4_double theta_volume = tet4_measure_V4(x0,
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
            vec4_double g_qx, g_qy, g_qz;

            vec4_double tet4_qx_v = {tet4_qx[quad_i],  //
                                     tet4_qx[quad_i],
                                     tet4_qx[quad_i],
                                     tet4_qx[quad_i]};

            vec4_double tet4_qy_v = {tet4_qy[quad_i],  //
                                     tet4_qy[quad_i],
                                     tet4_qy[quad_i],
                                     tet4_qy[quad_i]};

            vec4_double tet4_qz_v = {tet4_qz[quad_i],  //
                                     tet4_qz[quad_i],
                                     tet4_qz[quad_i],
                                     tet4_qz[quad_i]};

            tet4_transform_V4(x0,
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
                const vec4_double f0 = 1.0 - tet4_qx_v - tet4_qy_v - tet4_qz_v;
                const vec4_double f1 = tet4_qx_v;
                const vec4_double f2 = tet4_qy_v;
                const vec4_double f3 = tet4_qz_v;

                tet4_f0 = 4.0 * f0 - f1 - f2 - f3;
                tet4_f1 = -f0 + 4.0 * f1 - f2 - f3;
                tet4_f2 = -f0 - f1 + 4.0 * f2 - f3;
                tet4_f3 = -f0 - f1 - f2 + 4.0 * f3;
            }
#endif

            const vec4_double grid_x = (g_qx - ox) / dx;
            const vec4_double grid_y = (g_qy - oy) / dy;
            const vec4_double grid_z = (g_qz - oz) / dz;

            const vec4_int64 i = floor_V4(grid_x);
            const vec4_int64 j = floor_V4(grid_y);
            const vec4_int64 k = floor_V4(grid_z);

            // Get the reminder [0, 1]
            vec4_double l_x = (grid_x - __builtin_convertvector(i, vec4_double));
            vec4_double l_y = (grid_y - __builtin_convertvector(j, vec4_double));
            vec4_double l_z = (grid_z - __builtin_convertvector(k, vec4_double));

            // Critical point
            hex_aa_8_eval_fun_V4(l_x,
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

            hex_aa_8_collect_coeffs_V4(stride0,
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
                vec4_double eval_field = {0.0, 0.0, 0.0, 0.0};
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

                vec4_double dV = theta_volume * tet4_qw[quad_i];
                // dV = (vec8_double){1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

                element_field0 += eval_field * tet4_f0 * dV;
                element_field1 += eval_field * tet4_f1 * dV;
                element_field2 += eval_field * tet4_f2 * dV;
                element_field3 += eval_field * tet4_f3 * dV;

            }  // end integrate gap function

            //////////////////////////////////////////////////////////////////////
        }  // end loop over the quadrature points

        weighted_field[elems[0][element_i + 0]] += element_field0[0];
        weighted_field[elems[0][element_i + 1]] += element_field0[1];
        weighted_field[elems[0][element_i + 2]] += element_field0[2];
        weighted_field[elems[0][element_i + 3]] += element_field0[3];

        weighted_field[elems[1][element_i + 0]] += element_field1[0];
        weighted_field[elems[1][element_i + 1]] += element_field1[1];
        weighted_field[elems[1][element_i + 2]] += element_field1[2];
        weighted_field[elems[1][element_i + 3]] += element_field1[3];

        weighted_field[elems[2][element_i + 0]] += element_field2[0];
        weighted_field[elems[2][element_i + 1]] += element_field2[1];
        weighted_field[elems[2][element_i + 2]] += element_field2[2];
        weighted_field[elems[2][element_i + 3]] += element_field2[3];

        weighted_field[elems[3][element_i + 0]] += element_field3[0];
        weighted_field[elems[3][element_i + 1]] += element_field3[1];
        weighted_field[elems[3][element_i + 2]] += element_field3[2];
        weighted_field[elems[3][element_i + 3]] += element_field3[3];

        //////////////////////////////////////////////////////////////////////
    }  // end loop over the elements

    return 0;
    //////////////////////////////////////////////////////////////////////
}  // end tet4_resample_field_local_V4_aligned

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_local_v4 //////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int tet4_resample_field_local_V4(
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
    const ptrdiff_t nelements_aligned = nelements - (nelements % 4);
    const ptrdiff_t nelements_tail = nelements % 4;

    printf("=============================================\n");
    printf("nelements_aligned = %ld\n", nelements_aligned);
    printf("nelements_tail =    %ld\n", nelements_tail);
    printf("=============================================\n");

    tet4_resample_field_local_V4_aligned(0,
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
