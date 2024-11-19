#include "sfem_resample_field.h"



#include "mass.h"
// #include "read_mesh.h"
#include "matrixio_array.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

// #define real_t double

#include "quadratures_rule.h"


#define real_type real_t
#define SFEM_RESTRICT __restrict__

#define SFEM_RESAMPLE_GAP_DUAL

SFEM_INLINE static void tet4_transform_v2(
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
        const real_type px0,
        const real_type px1,
        const real_type px2,
        const real_type px3,
        // Y-coordinates
        const real_type py0,
        const real_type py1,
        const real_type py2,
        const real_type py3,
        // Z-coordinates
        const real_type pz0,
        const real_type pz1,
        const real_type pz2,
        const real_type pz3,
        // Quadrature point
        const real_type qx,
        const real_type qy,
        const real_type qz,
        // Output
        real_type* const SFEM_RESTRICT out_x,
        real_type* const SFEM_RESTRICT out_y,
        real_type* const SFEM_RESTRICT out_z) {
    //
    //
    *out_x = px0 + qx * (-px0 + px1) + qy * (-px0 + px2) + qz * (-px0 + px3);
    *out_y = py0 + qx * (-py0 + py1) + qy * (-py0 + py2) + qz * (-py0 + py3);
    *out_z = pz0 + qx * (-pz0 + pz1) + qy * (-pz0 + pz2) + qz * (-pz0 + pz3);
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_local /////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
SFEM_INLINE static real_type tet4_measure_v2(
        // X-coordinates
        const real_type px0,
        const real_type px1,
        const real_type px2,
        const real_type px3,
        // Y-coordinates
        const real_type py0,
        const real_type py1,
        const real_type py2,
        const real_type py3,
        // Z-coordinates
        const real_type pz0,
        const real_type pz1,
        const real_type pz2,
        const real_type pz3) {
    //
    // determinant of the Jacobian
    // M = [px0, py0, pz0, 1]
    //     [px1, py1, pz1, 1]
    //     [px2, py2, pz2, 1]
    //     [px3, py3, pz3, 1]
    //
    // V = (1/6) * det(M)

    const real_type x0 = -pz0 + pz3;
    const real_type x1 = -py0 + py2;
    const real_type x2 = -(1.0 / 6.0) * px0 + (1.0 / 6.0) * px1;
    const real_type x3 = -py0 + py3;
    const real_type x4 = -pz0 + pz2;
    const real_type x5 = -py0 + py1;
    const real_type x6 = -(1.0 / 6.0) * px0 + (1.0 / 6.0) * px2;
    const real_type x7 = -pz0 + pz1;
    const real_type x8 = -(1.0 / 6.0) * px0 + (1.0 / 6.0) * px3;

    return x0 * x1 * x2 - x0 * x5 * x6 - x1 * x7 * x8 - x2 * x3 * x4 + x3 * x6 * x7 + x4 * x5 * x8;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// hex_aa_8_collect_coeffs ////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
SFEM_INLINE static void hex_aa_8_eval_fun_V(
        // Quadrature point (local coordinates)
        // With respect to the hat functions of a cube element
        // In a local coordinate system
        const real_t x,
        const real_t y,
        const real_t z,

        // Output
        real_t* const SFEM_RESTRICT f0,
        real_t* const SFEM_RESTRICT f1,
        real_t* const SFEM_RESTRICT f2,
        real_t* const SFEM_RESTRICT f3,
        real_t* const SFEM_RESTRICT f4,
        real_t* const SFEM_RESTRICT f5,
        real_t* const SFEM_RESTRICT f6,
        real_t* const SFEM_RESTRICT f7) {
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
// hex_aa_8_collect_coeffs ////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
SFEM_INLINE static void hex_aa_8_collect_coeffs_V(
        const ptrdiff_t* const SFEM_RESTRICT stride,
        const ptrdiff_t i,
        const ptrdiff_t j,
        const ptrdiff_t k,
        // Attention this is geometric data transformed to solver data!
        const real_t* const SFEM_RESTRICT data,
        //
        real_t* SFEM_RESTRICT out0,
        real_t* SFEM_RESTRICT out1,
        real_t* SFEM_RESTRICT out2,
        real_t* SFEM_RESTRICT out3,
        real_t* SFEM_RESTRICT out4,
        real_t* SFEM_RESTRICT out5,
        real_t* SFEM_RESTRICT out6,
        real_t* SFEM_RESTRICT out7) {
    //
    const ptrdiff_t i0 = i * stride[0] + j * stride[1] + k * stride[2];
    const ptrdiff_t i1 = (i + 1) * stride[0] + j * stride[1] + k * stride[2];
    const ptrdiff_t i2 = (i + 1) * stride[0] + (j + 1) * stride[1] + k * stride[2];
    const ptrdiff_t i3 = i * stride[0] + (j + 1) * stride[1] + k * stride[2];
    const ptrdiff_t i4 = i * stride[0] + j * stride[1] + (k + 1) * stride[2];
    const ptrdiff_t i5 = (i + 1) * stride[0] + j * stride[1] + (k + 1) * stride[2];
    const ptrdiff_t i6 = (i + 1) * stride[0] + (j + 1) * stride[1] + (k + 1) * stride[2];
    const ptrdiff_t i7 = i * stride[0] + (j + 1) * stride[1] + (k + 1) * stride[2];

    *out0 = data[i0];
    *out1 = data[i1];
    *out2 = data[i2];
    *out3 = data[i3];
    *out4 = data[i4];
    *out5 = data[i5];
    *out6 = data[i6];
    *out7 = data[i7];
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_local_v2 //////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int tet4_resample_field_local_v2(
        // Mesh
        const ptrdiff_t start_element,
        const ptrdiff_t end_element,
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
    printf("Start: tet4_resample_field_local  v2 [%s] \n", __FILE__);
    printf("============================================================\n");
    //
    const real_type ox = (real_type)origin[0];
    const real_type oy = (real_type)origin[1];
    const real_type oz = (real_type)origin[2];

    const real_type dx = (real_type)delta[0];
    const real_type dy = (real_type)delta[1];
    const real_type dz = (real_type)delta[2];

    for (ptrdiff_t element_i = start_element; element_i < end_element; element_i++) {
        // real_type x[4], y[4], z[4];
        // Vertices coordinates of the tetrahedron
        real_type x0 = 0.0, x1 = 0.0, x2 = 0.0, x3 = 0.0;
        real_type y0 = 0.0, y1 = 0.0, y2 = 0.0, y3 = 0.0;
        real_type z0 = 0.0, z1 = 0.0, z2 = 0.0, z3 = 0.0;

        // real_type hex8_f[8];
        real_type hex8_f0 = 0.0, hex8_f1 = 0.0, hex8_f2 = 0.0, hex8_f3 = 0.0, hex8_f4 = 0.0,
                  hex8_f5 = 0.0, hex8_f6 = 0.0, hex8_f7 = 0.0;

        // real_type coeffs[8];
        real_type coeffs0 = 0.0, coeffs1 = 0.0, coeffs2 = 0.0, coeffs3 = 0.0, coeffs4 = 0.0,
                  coeffs5 = 0.0, coeffs6 = 0.0, coeffs7 = 0.0;

        // real_type tet4_f[4];
        real_type tet4_f0 = 0.0, tet4_f1 = 0.0, tet4_f2 = 0.0, tet4_f3 = 0.0;

        // real_type element_field[4];
        real_type element_field0 = 0.0, element_field1 = 0.0, element_field2 = 0.0,
                  element_field3 = 0.0;

        // loop over the 4 vertices of the tetrahedron
        idx_t ev[4];
        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][element_i];
        }

        // copy the coordinates of the vertices
        // for (int v = 0; v < 4; ++v) {
        //     x[v] = xyz[0][ev[v]];  // x-coordinates
        //     y[v] = xyz[1][ev[v]];  // y-coordinates
        //     z[v] = xyz[2][ev[v]];  // z-coordinates
        // }
        {
            x0 = xyz[0][ev[0]];
            x1 = xyz[0][ev[1]];
            x2 = xyz[0][ev[2]];
            x3 = xyz[0][ev[3]];

            y0 = xyz[1][ev[0]];
            y1 = xyz[1][ev[1]];
            y2 = xyz[1][ev[2]];
            y3 = xyz[1][ev[3]];

            z0 = xyz[2][ev[0]];
            z1 = xyz[2][ev[1]];
            z2 = xyz[2][ev[2]];
            z3 = xyz[2][ev[3]];
        }

        // Volume of the tetrahedron
        const real_type theta_volume = tet4_measure_v2(x0,
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

        /////////////////////////////////////////////  
        // loop over the quadrature points
        for (int quad_i = 0; quad_i < TET4_NQP; quad_i++) {  // loop over the quadrature points

            real_type g_qx, g_qy, g_qz;

            tet4_transform_v2(x0,
                              x1,
                              x2,
                              x3,

                              y0,
                              y1,
                              y2,
                              y3,

                              z0,
                              z1,
                              z2,
                              z3,

                              tet4_qx[quad_i],
                              tet4_qy[quad_i],
                              tet4_qz[quad_i],

                              &g_qx,
                              &g_qy,
                              &g_qz);

#ifndef SFEM_RESAMPLE_GAP_DUAL
            // Standard basis function
            {
                tet4_f[0] = 1 - tet4_qx[q] - tet4_qy[q] - tet4_qz[q];
                tet4_f[1] = tet4_qx[q];
                tet4_f[2] = tet4_qy[q];
                tet4_f[2] = tet4_qz[q];
            }
#else
            // DUAL basis function
            {
                const real_type f0 = 1.0 - tet4_qx[quad_i] - tet4_qy[quad_i] - tet4_qz[quad_i];
                const real_type f1 = tet4_qx[quad_i];
                const real_type f2 = tet4_qy[quad_i];
                const real_type f3 = tet4_qz[quad_i];

                tet4_f0 = 4.0 * f0 - f1 - f2 - f3;
                tet4_f1 = -f0 + 4.0 * f1 - f2 - f3;
                tet4_f2 = -f0 - f1 + 4.0 * f2 - f3;
                tet4_f3 = -f0 - f1 - f2 + 4.0 * f3;
            }
#endif

            const real_type grid_x = (g_qx - ox) / dx;
            const real_type grid_y = (g_qy - oy) / dy;
            const real_type grid_z = (g_qz - oz) / dz;

            const ptrdiff_t i = floor(grid_x);
            const ptrdiff_t j = floor(grid_y);
            const ptrdiff_t k = floor(grid_z);

            // printf("i = %ld grid_x = %g\n", i, grid_x);
            // printf("j = %ld grid_y = %g\n", j, grid_y);
            // printf("k = %ld grid_z = %g\n", k, grid_z);

            // If outside
            if (i < 0 || j < 0 || k < 0 || (i + 1 >= n[0]) || (j + 1 >= n[1]) || (k + 1 >= n[2])) {
                fprintf(stderr,
                        "WARNING: (%g, %g, %g) (%ld, %ld, %ld) outside domain  (%ld, %ld, "
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
                exit(1);
            }

            // Get the reminder [0, 1]
            real_type l_x = (grid_x - (double)i);
            real_type l_y = (grid_y - (double)j);
            real_type l_z = (grid_z - (double)k);

            assert(l_x >= -1e-8);
            assert(l_y >= -1e-8);
            assert(l_z >= -1e-8);

            assert(l_x <= 1 + 1e-8);
            assert(l_y <= 1 + 1e-8);
            assert(l_z <= 1 + 1e-8);

            // Critical point
            hex_aa_8_eval_fun_V(l_x,
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

            hex_aa_8_collect_coeffs_V(stride,
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
                real_type eval_field = 0.0;
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

                real_type dV = theta_volume * tet4_qw[quad_i];
                // dV = 1.0;

                element_field0 += eval_field * tet4_f0 * dV;
                element_field1 += eval_field * tet4_f1 * dV;
                element_field2 += eval_field * tet4_f2 * dV;
                element_field3 += eval_field * tet4_f3 * dV;

            }  // end integrate gap function

        }  // end for quad_i over quadrature points

        // for (int v = 0; v < 4; ++v) {
        //     // Invert sign since distance field is negative insdide and positive outside

        //      weighted_field[ev[v]] += element_field[v];
        // }  // end vertex loop
        weighted_field[ev[0]] += element_field0;
        weighted_field[ev[1]] += element_field1;
        weighted_field[ev[2]] += element_field2;
        weighted_field[ev[3]] += element_field3;

    }  // end for i over elements

    return 0;
}  // tet4_resample_field_local
