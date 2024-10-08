#include "sfem_resample_field.h"

#include "matrixio_array.h"

#include "mass.h"

#include "sfem_resample_V.h"
#include "tet10_resample_field.h"
#include "tet10_resample_field_V2.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "quadratures_rule.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define SFEM_RESAMPLE_GAP_DUAL

static SFEM_INLINE real_t put_inside(const real_t v) { return MIN(MAX(1e-7, v), 1 - 1e-7); }

SFEM_INLINE static int hex_aa_8_contains(
        // X-coordinates
        const real_t xmin, const real_t xmax,
        // Y-coordinates
        const real_t ymin, const real_t ymax,
        // Z-coordinates
        const real_t zmin, const real_t zmax, const real_t x, const real_t y, const real_t z) {
    int outside = (x < xmin) | (x > xmax) | (y < ymin) | (y > ymax) | (z < zmin) | (x > zmax);
    return !outside;
}

SFEM_INLINE static real_t tri_shell_3_measure(
        // X-coordinates
        const real_t px0, const real_t px1, const real_t px2,
        // Y-coordinates
        const real_t py0, const real_t py1, const real_t py2,
        // Z-coordinates
        const real_t pz0, const real_t pz1, const real_t pz2) {
    const real_t x0 = -px0 + px1;
    const real_t x1 = -px0 + px2;
    const real_t x2 = -py0 + py1;
    const real_t x3 = -py0 + py2;
    const real_t x4 = -pz0 + pz1;
    const real_t x5 = -pz0 + pz2;
    return (1.0 / 2.0) *
           sqrt((pow(x0, 2) + pow(x2, 2) + pow(x4, 2)) * (pow(x1, 2) + pow(x3, 2) + pow(x5, 2)) -
                pow(x0 * x1 + x2 * x3 + x4 * x5, 2));
}

SFEM_INLINE static void tri_shell_3_transform(
        // X-coordinates
        const real_t x0, const real_t x1, const real_t x2,
        // Y-coordinates
        const real_t y0, const real_t y1, const real_t y2,
        // Z-coordinates
        const real_t z0, const real_t z1, const real_t z2,
        // Quadrature point
        const real_t x, const real_t y,
        // Output
        real_t* const SFEM_RESTRICT out_x, real_t* const SFEM_RESTRICT out_y,
        real_t* const SFEM_RESTRICT out_z) {
    const real_t phi0 = 1 - x - y;
    const real_t phi1 = x;
    const real_t phi2 = y;

    *out_x = phi0 * x0 + phi1 * x1 + phi2 * x2;
    *out_y = phi0 * y0 + phi1 * y1 + phi2 * y2;
    *out_z = phi0 * z0 + phi1 * z1 + phi2 * z2;
}

SFEM_INLINE static real_t beam2_measure(
        // X-coordinates
        const real_t px0, const real_t px1,
        // Y-coordinates
        const real_t py0, const real_t py1,
        // Z-coordinates
        const real_t pz0, const real_t pz1) {
    return sqrt(pow(-px0 + px1, 2) + pow(-py0 + py1, 2) + pow(-pz0 + pz1, 2));
}

SFEM_INLINE static void beam2_transform(
        // X-coordinates
        const real_t px0, const real_t px1,
        // Y-coordinates
        const real_t py0, const real_t py1,
        // Z-coordinates
        const real_t pz0, const real_t pz1,
        // Quadrature point
        const real_t x,
        // Output
        real_t* const SFEM_RESTRICT out_x, real_t* const SFEM_RESTRICT out_y,
        real_t* const SFEM_RESTRICT out_z) {
    *out_x = px0 + x * (-px0 + px1);
    *out_y = py0 + x * (-py0 + py1);
    *out_z = pz0 + x * (-pz0 + pz1);
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

SFEM_INLINE static void hex_aa_8_eval_grad(
        // Quadrature point (local coordinates)
        const real_t x, const real_t y, const real_t z,
        // Output
        real_t* const SFEM_RESTRICT gx, real_t* const SFEM_RESTRICT gy,
        real_t* const SFEM_RESTRICT gz) {
    //
    // Transformation to ref element
    gx[0] = -(1.0 - y) * (1.0 - z);
    gy[0] = -(1.0 - x) * (1.0 - z);
    gz[0] = -(1.0 - x) * (1.0 - y);

    gx[1] = (1.0 - y) * (1.0 - z);
    gy[1] = -x * (1.0 - z);
    gz[1] = -x * (1.0 - y);

    gx[2] = y * (1.0 - z);
    gy[2] = x * (1.0 - z);
    gz[2] = -x * y;

    gx[3] = -y * (1.0 - z);
    gy[3] = (1.0 - x) * (1.0 - z);
    gz[3] = -(1.0 - x) * y;

    gx[4] = -(1.0 - y) * z;
    gy[4] = -(1.0 - x) * z;
    gz[4] = (1.0 - x) * (1.0 - y);

    gx[5] = (1.0 - y) * z;
    gy[5] = -x * z;
    gz[5] = x * (1.0 - y);

    gx[6] = y * z;
    gy[6] = x * z;
    gz[6] = x * y;

    gx[7] = -y * z;
    gy[7] = (1.0 - x) * z;
    gz[7] = (1.0 - x) * y;
}

// GCC unroll(0) does not compile on Grace
#define UNROLL_ZERO _Pragma("GCC unroll(1)")

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_local /////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int tet4_resample_field_local(
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
    printf("============================================================\n");
    printf("Start: tet4_resample_field_local\n");
    printf("============================================================\n");
    //
    const real_t ox = (real_t)origin[0];
    const real_t oy = (real_t)origin[1];
    const real_t oz = (real_t)origin[2];

    const real_t dx = (real_t)delta[0];
    const real_t dy = (real_t)delta[1];
    const real_t dz = (real_t)delta[2];

    // printf("\nnumber of elements %ld  +++++++++++++++++++++++++++++++++++ \n", nelements);

#pragma omp parallel
    {
#pragma omp for  // nowait

        /// Loop over the elements of the mesh
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[4];
            geom_t x[4], y[4], z[4];

            real_t hex8_f[8];
            real_t coeffs[8];

            real_t tet4_f[4];
            real_t element_field[4];

            // loop over the 4 vertices of the tetrahedron
            UNROLL_ZERO
            for (int v = 0; v < 4; ++v) {
                ev[v] = elems[v][i];
            }

            // copy the coordinates of the vertices
            for (int v = 0; v < 4; ++v) {
                x[v] = xyz[0][ev[v]];  // x-coordinates
                y[v] = xyz[1][ev[v]];  // y-coordinates
                z[v] = xyz[2][ev[v]];  // z-coordinates
            }

            memset(element_field, 0,
                   4 * sizeof(real_t));  // set to zero the element field

            // Area of the tetrahedron
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

            for (int q = 0; q < TET4_NQP; q++) {  // loop over the quadrature points

                real_t g_qx, g_qy, g_qz;
                // Transform quadrature point to physical space
                // g_qx, g_qy, g_qz are the coordinates of the quadrature point in the physical
                // space
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
                    const real_t f0 = 1.0 - tet4_qx[q] - tet4_qy[q] - tet4_qz[q];
                    const real_t f1 = tet4_qx[q];
                    const real_t f2 = tet4_qy[q];
                    const real_t f3 = tet4_qz[q];

                    tet4_f[0] = 4 * f0 - f1 - f2 - f3;
                    tet4_f[1] = -f0 + 4 * f1 - f2 - f3;
                    tet4_f[2] = -f0 - f1 + 4 * f2 - f3;
                    tet4_f[3] = -f0 - f1 - f2 + 4 * f3;
                }
#endif
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

                // Integrate gap function
                {
                    real_t eval_field = 0;
                    UNROLL_ZERO
                    for (int edof_j = 0; edof_j < 8; edof_j++) {
                        eval_field += hex8_f[edof_j] * coeffs[edof_j];
                    }

                    UNROLL_ZERO
                    for (int edof_i = 0; edof_i < 4; edof_i++) {
                        element_field[edof_i] += eval_field * tet4_f[edof_i] * dV;
                    }  // end edof_i loop
                }
            }  // end quadrature loop

            UNROLL_ZERO
            for (int v = 0; v < 4; ++v) {
                // Invert sign since distance field is negative insdide and positive outside
#pragma omp critical
                { weighted_field[ev[v]] += element_field[v]; }

            }  // end vertex loop
        }      // end element loop
    }          // end parallel region

    return 0;
}

int trishell3_resample_field_local(
        // Mesh
        const ptrdiff_t nelements, const ptrdiff_t nnodes, idx_t** const SFEM_RESTRICT elems,
        geom_t** const SFEM_RESTRICT xyz,
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n, const ptrdiff_t* const SFEM_RESTRICT stride,
        const geom_t* const SFEM_RESTRICT origin, const geom_t* const SFEM_RESTRICT delta,
        const real_t* const SFEM_RESTRICT data,
        // Output
        real_t* const SFEM_RESTRICT weighted_field) {
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
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[3];
            geom_t x[3], y[3], z[3];

            real_t hex8_f[8];
            real_t coeffs[8];

            real_t tri3_f[3];
            real_t element_field[3];

            UNROLL_ZERO
            for (int v = 0; v < 3; ++v) {
                ev[v] = elems[v][i];
            }

            for (int v = 0; v < 3; ++v) {
                x[v] = xyz[0][ev[v]];
                y[v] = xyz[1][ev[v]];
                z[v] = xyz[2][ev[v]];
            }

            memset(element_field, 0, 3 * sizeof(real_t));

            const real_t measure =
                    tri_shell_3_measure(x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2]);

            assert(measure > 0);

            for (int q = 0; q < TRI3_NQP; q++) {
                real_t g_qx, g_qy, g_qz;
                tri_shell_3_transform(x[0],
                                      x[1],
                                      x[2],
                                      y[0],
                                      y[1],
                                      y[2],
                                      z[0],
                                      z[1],
                                      z[2],
                                      tri3_qx[q],
                                      tri3_qy[q],
                                      &g_qx,
                                      &g_qy,
                                      &g_qz);

#ifndef SFEM_RESAMPLE_GAP_DUAL
                // Standard basis function
                {
                    tri3_f[0] = 1 - tri3_qx[q] - tri3_qy[q];
                    tri3_f[1] = tri3_qx[q];
                    tri3_f[2] = tri3_qy[q];
                }
#else
                // DUAL basis function
                {
                    const real_t f0 = 1 - tri3_qx[q] - tri3_qy[q];
                    const real_t f1 = tri3_qx[q];
                    const real_t f2 = tri3_qy[q];

                    tri3_f[0] = 3.0 * f0 - f1 - f2;
                    tri3_f[1] = -f0 + 3.0 * f1 - f2;
                    tri3_f[2] = -f0 - f1 + 3.0 * f2;
                }
#endif

                const real_t dV = measure * tri3_qw[q];

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

                // Integrate gap function
                {
                    real_t eval_field = 0;

                    UNROLL_ZERO
                    for (int edof_j = 0; edof_j < 8; edof_j++) {  // loop over the 8 vertices
                        eval_field += hex8_f[edof_j] * coeffs[edof_j];
                    }

                    UNROLL_ZERO
                    for (int edof_i = 0; edof_i < 3; edof_i++) {  // loop over the 3 vertices
                        element_field[edof_i] += eval_field * tri3_f[edof_i] * dV;
                    }
                }
            }

            UNROLL_ZERO
            for (int v = 0; v < 3; ++v) {
                // Invert sign since distance field is negative insdide and positive outside
#pragma omp critical
                { weighted_field[ev[v]] += element_field[v]; }

            }  // end vertex loop
        }      // end element loop
    }          // end parallel region

    return 0;
}  // end trishell3_resample_field_local

int beam2_resample_field_local(const ptrdiff_t nelements, const ptrdiff_t nnodes,
                               idx_t** const SFEM_RESTRICT elems, geom_t** const SFEM_RESTRICT xyz,
                               // SDF
                               const ptrdiff_t* const SFEM_RESTRICT n,
                               const ptrdiff_t* const SFEM_RESTRICT stride,
                               const geom_t* const SFEM_RESTRICT origin,
                               const geom_t* const SFEM_RESTRICT delta,
                               const real_t* const SFEM_RESTRICT data,
                               // Output
                               real_t* const SFEM_RESTRICT weighted_field) {
    printf("beam2_resample_field_local!\n");

    const real_t ox = (real_t)origin[0];
    const real_t oy = (real_t)origin[1];
    const real_t oz = (real_t)origin[2];

    const real_t dx = (real_t)delta[0];
    const real_t dy = (real_t)delta[1];
    const real_t dz = (real_t)delta[2];

#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[3];
            geom_t x[3], y[3], z[3];

            real_t hex8_f[8];
            real_t coeffs[8];

            real_t beam2_f[2];
            real_t element_field[2];

            UNROLL_ZERO
            for (int v = 0; v < 2; ++v) {
                ev[v] = elems[v][i];
            }

            for (int v = 0; v < 2; ++v) {
                x[v] = xyz[0][ev[v]];
                y[v] = xyz[1][ev[v]];
                z[v] = xyz[2][ev[v]];
            }

            memset(element_field, 0, 2 * sizeof(real_t));

            const real_t measure = beam2_measure(x[0], x[1], y[0], y[1], z[0], z[1]);

            assert(measure > 0);

            for (int q = 0; q < EDGE2_NQP; q++) {
                real_t g_qx, g_qy, g_qz;
                beam2_transform(
                        x[0], x[1], y[0], y[1], z[0], z[1], edge2_qx[q], &g_qx, &g_qy, &g_qz);

#ifndef SFEM_RESAMPLE_GAP_DUAL
                // Standard basis function
                {
                    beam2_f[0] = 1 - edge2_qx[q];
                    beam2_f[1] = edge2_qx[q];
                }
#else
                // DUAL basis function
                {
                    const real_t f0 = 1 - edge2_qx[q];
                    const real_t f1 = edge2_qx[q];
                    beam2_f[0] = 2 * f0 - f1;
                    beam2_f[1] = -f0 + 2 * f1;
                }
#endif

                const real_t dV = measure * edge2_qw[q];

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

                // Integrate gap function
                {
                    real_t eval_field = 0;

                    UNROLL_ZERO
                    for (int edof_j = 0; edof_j < 8; edof_j++) {
                        eval_field += hex8_f[edof_j] * coeffs[edof_j];
                    }

                    UNROLL_ZERO
                    for (int edof_i = 0; edof_i < 2; edof_i++) {
                        element_field[edof_i] += eval_field * beam2_f[edof_i] * dV;
                    }
                }  // end integrate gap function
            }      // end quadrature loop

            UNROLL_ZERO
            for (int v = 0; v < 2; ++v) {
                // Invert sign since distance field is negative insdide and positive outside
#pragma omp critical
                { weighted_field[ev[v]] += element_field[v]; }

            }  // end vertex loop

        }  // end element loop
    }      // end parallel region

    return 0;
}  // end beam2_resample_field_local
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

#define MY_RESTRICT __restrict__
#define real_type double

int tet4_resample_field_local_CUDA(  // Mesh
        const ptrdiff_t nelements, const ptrdiff_t nnodes, int** const MY_RESTRICT elems,
        float** const MY_RESTRICT xyz,
        // SDF
        const ptrdiff_t* const MY_RESTRICT n, const ptrdiff_t* const MY_RESTRICT stride,
        const float* const MY_RESTRICT origin, const float* const MY_RESTRICT delta,
        const real_type* const MY_RESTRICT data,
        // Output
        real_type* const MY_RESTRICT weighted_field);

int tet4_resample_field_local_reduce_CUDA(  // Mesh
        const ptrdiff_t nelements, const ptrdiff_t nnodes, int** const MY_RESTRICT elems,
        float** const MY_RESTRICT xyz,
        // SDF
        const ptrdiff_t* const MY_RESTRICT n, const ptrdiff_t* const MY_RESTRICT stride,
        const float* const MY_RESTRICT origin, const float* const MY_RESTRICT delta,
        const real_type* const MY_RESTRICT data,
        // Output
        real_type* const MY_RESTRICT weighted_field);

int tet4_resample_field_local_V8(
        // Mesh
        const ptrdiff_t nelements, const ptrdiff_t nnodes, int** const MY_RESTRICT elems,
        float** const MY_RESTRICT xyz,
        // SDF
        const ptrdiff_t* const MY_RESTRICT n, const ptrdiff_t* const MY_RESTRICT stride,
        const float* const MY_RESTRICT origin, const float* const MY_RESTRICT delta,
        const real_type* const MY_RESTRICT data,
        // Output
        real_type* const MY_RESTRICT weighted_field);

int hex8_to_tet10_resample_field_local_CUDA(
        // Mesh
        const ptrdiff_t nelements,  // number of elements
        const ptrdiff_t nnodes,     // number of nodes
        const idx_t** const elems,  // connectivity
        const geom_t** const xyz,   // coordinates
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n,       // number of nodes in each direction
        const ptrdiff_t* const SFEM_RESTRICT stride,  // stride of the data

        const geom_t* const SFEM_RESTRICT origin,  // origin of the domain
        const geom_t* const SFEM_RESTRICT delta,   // delta of the domain
        const real_t* const SFEM_RESTRICT data,    // SDF
        // Output //
        real_t* const SFEM_RESTRICT weighted_field);

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// resample_field_local ////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

#define USE_TET4_V4 0
#define USE_TET4_V8 1
#define USE_TET4_CUDA 2

#define USE_TET4_MODEL USE_TET4_V8

int resample_field_local(
        // Mesh
        const enum ElemType element_type, const ptrdiff_t nelements, const ptrdiff_t nnodes,
        idx_t** const SFEM_RESTRICT elems, geom_t** const SFEM_RESTRICT xyz,
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n, const ptrdiff_t* const SFEM_RESTRICT stride,
        const geom_t* const SFEM_RESTRICT origin, const geom_t* const SFEM_RESTRICT delta,
        const real_t* const SFEM_RESTRICT data,
        // Output
        real_t* const SFEM_RESTRICT weighted_field, sfem_resample_field_info* info) {
    //
    switch (TET10) {
        case TET4: {
            info->quad_nodes_cnt = TET4_NQP;
            info->nelements = nelements;

            // tet4_resample_field_local_reduce_CUDA

#if USE_TET4_MODEL == USE_TET4_V4
            return tet4_resample_field_local_V4(
                    nelements, nnodes, elems, xyz, n, stride, origin, delta, data, weighted_field);
#elif USE_TET4_MODEL == USE_TET4_V8
            return tet4_resample_field_local_V8(
                    nelements, nnodes, elems, xyz, n, stride, origin, delta, data, weighted_field);
#elif USE_TET4_MODEL == USE_TET4_CUDA
            return tet4_resample_field_local_reduce_CUDA(
                    nelements, nnodes, elems, xyz, n, stride, origin, delta, data, weighted_field);
#endif
        }

        case TET10: {
// #define TET10_V2

// { /// DEBUG ///
// double norm_data = 0.0;
// for (ptrdiff_t i = 0; i < n[0] * n[1] * n[2]; i++) {
//     norm_data += data[i] * data[i];
//     if(i % 50000 == 0) {
//         printf("norm_data[%ld] = %g, %s:%d\n", i, norm_data, __FILE__, __LINE__);
//     }
// }

// norm_data = sqrt(norm_data);
// printf("\nFunction: %s\n", __FUNCTION__);
// printf("\nnorm_data input = %g   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< %s:%d\n\n", norm_data, __FILE__, __LINE__);
// } /// end DEBUG ///

#ifdef TET10_V2  // V2
            return hex8_to_tet10_resample_field_local_CUDA(
                    nelements, nnodes, elems, xyz, n, stride, origin, delta, data, weighted_field);
#else
            return hex8_to_tet10_resample_field_local(
                    nelements, nnodes, elems, xyz, n, stride, origin, delta, data, weighted_field);
#endif
        }

        default:
            break;
    }

    enum ElemType st = shell_type(element_type);

    switch (st) {
        case TRISHELL3:
            return trishell3_resample_field_local(
                    nelements, nnodes, elems, xyz, n, stride, origin, delta, data, weighted_field);
        case BEAM2:
            return beam2_resample_field_local(
                    nelements, nnodes, elems, xyz, n, stride, origin, delta, data, weighted_field);

        default: {
            assert(0);
            fprintf(stderr, "Unknown element type %d\n", st);
            MPI_Abort(MPI_COMM_WORLD, -1);
            return EXIT_FAILURE;
        }
    }
}

int resample_field(
        // Mesh
        const enum ElemType element_type, const ptrdiff_t nelements, const ptrdiff_t nnodes,
        idx_t** const SFEM_RESTRICT elems, geom_t** const SFEM_RESTRICT xyz,
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n, const ptrdiff_t* const SFEM_RESTRICT stride,
        const geom_t* const SFEM_RESTRICT origin, const geom_t* const SFEM_RESTRICT delta,
        const real_t* const SFEM_RESTRICT data,
        // Output
        real_t* const SFEM_RESTRICT g, sfem_resample_field_info* info) {
    //
    real_t* weighted_field = calloc(nnodes, sizeof(real_t));

    // { /// DEBUG ///

    //     double norm_data = 0.0;

    //     printf("\nFunction: %s\n", __FUNCTION__);
    //     printf("data (ptr): %p, %s:%d\n", (void *)data, __FILE__, __LINE__);
    //     printf("n[0] = %ld, n[1] = %ld, n[2] = %ld, %s:%d\n", n[0], n[1], n[2], __FILE__, __LINE__);

        
    //     for (ptrdiff_t i = 0; i < (n[0] * n[1] * n[2]); i++) {
    //         norm_data += (data[i] * data[i]);
    //     //     // search a nan value in the data
    //         if ( i % 50000 == 0 ) {
    //             printf("norm_data[%ld] = %g, %s:%d\n", i,norm_data, __FILE__, __LINE__);
    //         }
    //     }
    //     const double sqrt_norm_data = sqrt(norm_data);
    //     printf("\nnorm_data input = %e   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< %s:%d\n\n", sqrt_norm_data,
    //            __FILE__,
    //            __LINE__);

    //     int indices[3] = {22, 55, 111};
    //                 printf("data[%d] = %g, %s:%d\n", indices[0], data[indices[0]], __FILE__, __LINE__);
    //                 printf("data[%d] = %g, %s:%d\n", indices[1], data[indices[1]], __FILE__, __LINE__);
    //                 printf("data[%d] = %g, %s:%d\n", indices[2], data[indices[2]], __FILE__, __LINE__);
    // } /// end DEBUG ///


    resample_field_local(element_type,
                         nelements,
                         nnodes,
                         elems,
                         xyz,
                         n,
                         stride,
                         origin,
                         delta,
                         data,
                         weighted_field,
                         info);

    enum ElemType st = shell_type(element_type);

    if (INVALID == st) {
        // FIXME
        if (element_type == TET10) {
            real_t* mass_vector = calloc(nnodes, sizeof(real_t));
            tet10_assemble_dual_mass_vector(nelements, nnodes, elems, xyz, mass_vector);

            for (ptrdiff_t i = 0; i < nnodes; i++) {
                assert(mass_vector[i] != 0);
                g[i] = weighted_field[i] / mass_vector[i];
            }

            free(mass_vector);
        } else {
            // Removing the mass-contributions from the weighted gap function "weighted_field"
            apply_inv_lumped_mass(element_type, nelements, nnodes, elems, xyz, weighted_field, g);
        }
    } else {
        apply_inv_lumped_mass(st, nelements, nnodes, elems, xyz, weighted_field, g);
    }

    free(weighted_field);
    return 0;
}

int interpolate_field(const ptrdiff_t nnodes, geom_t** const SFEM_RESTRICT xyz,
                      // SDF
                      const ptrdiff_t* const SFEM_RESTRICT n,
                      const ptrdiff_t* const SFEM_RESTRICT stride,
                      const geom_t* const SFEM_RESTRICT origin,
                      const geom_t* const SFEM_RESTRICT delta,
                      const real_t* const SFEM_RESTRICT data,
                      // Output
                      real_t* const SFEM_RESTRICT g) {
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
        for (ptrdiff_t node = 0; node < nnodes; ++node) {
            real_t hex8_f[8];
            real_t hex8_grad_x[8];
            real_t hex8_grad_y[8];
            real_t hex8_grad_z[8];
            real_t coeffs[8];

            const real_t x = xyz[0][node];
            const real_t y = xyz[1][node];
            const real_t z = xyz[2][node];

            const real_t grid_x = (x - ox) / dx;
            const real_t grid_y = (y - oy) / dy;
            const real_t grid_z = (z - oz) / dz;

            const ptrdiff_t i = floor(grid_x);
            const ptrdiff_t j = floor(grid_y);
            const ptrdiff_t k = floor(grid_z);

            // If outside
            if (i < 0 || j < 0 || k < 0 || (i + 1 >= n[0]) || (j + 1 >= n[1]) || (k + 1 >= n[2])) {
                int rank;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                fprintf(stderr,
                        "[%d] warning (%g, %g, %g) (%ld, %ld, %ld) outside domain  (%ld, %ld, "
                        "%ld)!\n",
                        rank,
                        x,
                        y,
                        z,
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

            // Interpolate gap function
            {
                real_t eval_field = 0;

                UNROLL_ZERO
                for (int edof_j = 0; edof_j < 8; edof_j++) {
                    eval_field += hex8_f[edof_j] * coeffs[edof_j];
                }

                g[node] = eval_field;
            }
        }
    }

    return 0;
}

SFEM_INLINE static void minmax(const ptrdiff_t n, const geom_t* const SFEM_RESTRICT x, geom_t* xmin,
                               geom_t* xmax) {
    *xmin = x[0];
    *xmax = x[0];
    for (ptrdiff_t i = 1; i < n; i++) {
        *xmin = MIN(*xmin, x[i]);
        *xmax = MAX(*xmax, x[i]);
    }
}

int field_view(MPI_Comm comm, const ptrdiff_t nnodes, const geom_t* SFEM_RESTRICT z_coordinate,
               const ptrdiff_t* const nlocal, const ptrdiff_t* const SFEM_RESTRICT nglobal,
               const ptrdiff_t* const SFEM_RESTRICT stride, const geom_t* const origin,
               const geom_t* const SFEM_RESTRICT delta, const real_t* const field,
               real_t** field_out, ptrdiff_t* z_nlocal_out,
               geom_t* const SFEM_RESTRICT z_origin_out) {
    return field_view_ensure_margin(comm,
                                    nnodes,
                                    z_coordinate,
                                    nlocal,
                                    nglobal,
                                    stride,
                                    origin,
                                    delta,
                                    field,
                                    2,
                                    field_out,
                                    z_nlocal_out,
                                    z_origin_out);
}

int field_view_ensure_margin(MPI_Comm comm, const ptrdiff_t nnodes,
                             const geom_t* SFEM_RESTRICT z_coordinate,
                             const ptrdiff_t* const nlocal,
                             const ptrdiff_t* const SFEM_RESTRICT nglobal,
                             const ptrdiff_t* const SFEM_RESTRICT stride,
                             const geom_t* const origin, const geom_t* const SFEM_RESTRICT delta,
                             const real_t* const field, const ptrdiff_t z_margin,
                             real_t** field_out, ptrdiff_t* z_nlocal_out,
                             geom_t* const SFEM_RESTRICT z_origin_out) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size == 1) {
        if (!rank) {
            fprintf(stderr, "[%d] resample_grid_view cannot be used in serial runs!\n", rank);
        }

        MPI_Abort(comm, -1);
        return 1;
    }

    double field_view_tick = MPI_Wtime();

    geom_t zmin, zmax;
    minmax(nnodes, z_coordinate, &zmin, &zmax);

    // Z is distributed
    ptrdiff_t zoffset = 0;
    MPI_Exscan(&nlocal[2], &zoffset, 1, MPI_LONG, MPI_SUM, comm);

    // // Compute Local z-tile
    ptrdiff_t field_start = (zmin - origin[2]) / delta[2];
    ptrdiff_t field_end = (zmax - origin[2]) / delta[2];

    // Make sure we are inside the grid and get also the required margin for resampling
    field_start = MAX(0, field_start - 1 - z_margin);
    field_end = MIN(
            nglobal[2],
            field_end + 2 + z_margin);  // 1 for the rightside of the cell 1 for the exclusive range

    ptrdiff_t pnlocal_z = (field_end - field_start);
    real_t* pfield = malloc(pnlocal_z * stride[2] * sizeof(real_t));

    array_range_select(comm,
                       SFEM_MPI_REAL_T,
                       (void*)field,
                       (void*)pfield,
                       // Size of z-slice
                       nlocal[2] * stride[2],
                       // starting offset
                       field_start * stride[2],
                       // ending offset
                       field_end * stride[2]);

    *field_out = pfield;
    *z_nlocal_out = pnlocal_z;
    *z_origin_out = origin[2] + field_start * delta[2];

    double field_view_tock = MPI_Wtime();

    if (!rank) {
        printf("[%d] field_view %g (seconds)\n", rank, field_view_tock - field_view_tick);
    }

    return 0;
}
