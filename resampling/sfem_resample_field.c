#include <float.h>  // Include for FLT_MAX, DBL_MAX
#include <math.h>
#include <stddef.h>
#include <stddef.h>  // Add this for ptrdiff_t type
#include <stdio.h>
#include <string.h>
#include "sfem_base.h"
#include "sfem_config.h"  // Include the generated config header

#ifdef SFEM_ENABLE_CUDA
#include "sfem_adjoint_mini_tet_gpu_wrapper.h"
#endif

#include "matrixio_array.h"
#include "sfem_resample_field.h"

#include "mass.h"

#include "sfem_resample_V.h"
#include "tet10_resample_field.h"
#include "tet10_resample_field_V2.h"

#include "mesh_aura.h"
#include "quadratures_rule.h"
#include "sfem_defs.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// apply_fun_to_mesh ///////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
int                                                              //
apply_fun_to_mesh(const ptrdiff_t                    nnodes,     // Mesh
                  const geom_t** const SFEM_RESTRICT xyz,        // Mesh
                  const function_XYZ_t               fun,        // Function
                  real_t* const SFEM_RESTRICT        weighted_field) {  //   Output (weighted field)
    PRINT_CURRENT_FUNCTION;

    for (ptrdiff_t node = 0; node < nnodes; node++) {
        // Get the coordinates of the node
        const real_t x = (real_t)xyz[0][node];
        const real_t y = (real_t)xyz[1][node];
        const real_t z = (real_t)xyz[2][node];

        const real_t v1 = fun(x, y, z);

        weighted_field[node] = v1;
    }

    RETURN_FROM_FUNCTION(0);
}

/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

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
    return (1.0 / 2.0) * sqrt((pow(x0, 2) + pow(x2, 2) + pow(x4, 2)) * (pow(x1, 2) + pow(x3, 2) + pow(x5, 2)) -
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
        real_t* const SFEM_RESTRICT out_x, real_t* const SFEM_RESTRICT out_y, real_t* const SFEM_RESTRICT out_z) {
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
        real_t* const SFEM_RESTRICT out_x, real_t* const SFEM_RESTRICT out_y, real_t* const SFEM_RESTRICT out_z) {
    *out_x = px0 + x * (-px0 + px1);
    *out_y = py0 + x * (-py0 + py1);
    *out_z = pz0 + x * (-pz0 + pz1);
}

real_t tet4_measure(
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

SFEM_INLINE static void                            //
tet4_transform(const real_t                px0,    // X-coordinates
               const real_t                px1,    //
               const real_t                px2,    //
               const real_t                px3,    //
               const real_t                py0,    // Y-coordinates
               const real_t                py1,    //
               const real_t                py2,    //
               const real_t                py3,    //
               const real_t                pz0,    // Z-coordinates
               const real_t                pz1,    //
               const real_t                pz2,    //
               const real_t                pz3,    //
               const real_t                qx,     // Quadrature point
               const real_t                qy,     //
               const real_t                qz,     //
               real_t* const SFEM_RESTRICT out_x,  // Output
               real_t* const SFEM_RESTRICT out_y,  //
               real_t* const SFEM_RESTRICT out_z) {
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
    //
    //
    *out_x = px0 + qx * (-px0 + px1) + qy * (-px0 + px2) + qz * (-px0 + px3);
    *out_y = py0 + qx * (-py0 + py1) + qy * (-py0 + py2) + qz * (-py0 + py3);
    *out_z = pz0 + qx * (-pz0 + pz1) + qy * (-pz0 + pz2) + qz * (-pz0 + pz3);
}

void hex_aa_8_eval_fun(
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

SFEM_INLINE static void hex_aa_8_collect_coeffs(const ptrdiff_t* const SFEM_RESTRICT stride, const ptrdiff_t i, const ptrdiff_t j,
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
        real_t* const SFEM_RESTRICT gx, real_t* const SFEM_RESTRICT gy, real_t* const SFEM_RESTRICT gz) {
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
int                                                                        //
tet4_resample_field_local(const ptrdiff_t                      nelements,  // Mesh: number of elements
                          const ptrdiff_t                      nnodes,     // Mesh: number of nodes
                          idx_t** const SFEM_RESTRICT          elems,      // Mesh: connectivity
                          geom_t** const SFEM_RESTRICT         xyz,        // Mesh: coordinates
                          const ptrdiff_t* const SFEM_RESTRICT n,          // SDF: number of nodes in each direction
                          const ptrdiff_t* const SFEM_RESTRICT stride,     // SDF: stride of the data
                          const geom_t* const SFEM_RESTRICT    origin,     // SDF: origin of the domain
                          const geom_t* const SFEM_RESTRICT    delta,      // SDF: delta of the domain
                          const real_t* const SFEM_RESTRICT    data,       // SDF
                          real_t* const SFEM_RESTRICT          weighted_field) {    // Output
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
            idx_t  ev[4];
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

            for (int q = 0; q < TET_QUAD_NQP; q++) {  // loop over the quadrature points

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
                               tet_qx[q],
                               tet_qy[q],
                               tet_qz[q],
                               //
                               &g_qx,
                               &g_qy,
                               &g_qz);

#ifndef SFEM_RESAMPLE_GAP_DUAL
                // Standard basis function
                {
                    tet4_f[0] = 1 - tet_qx[q] - tet_qy[q] - tet_qz[q];
                    tet4_f[1] = tet_qx[q];
                    tet4_f[2] = tet_qy[q];
                    tet4_f[2] = tet_qz[q];
                }
#else
                // DUAL basis function
                {
                    const real_t f0 = 1.0 - tet_qx[q] - tet_qy[q] - tet_qz[q];
                    const real_t f1 = tet_qx[q];
                    const real_t f2 = tet_qy[q];
                    const real_t f3 = tet_qz[q];

                    tet4_f[0] = 4 * f0 - f1 - f2 - f3;
                    tet4_f[1] = -f0 + 4 * f1 - f2 - f3;
                    tet4_f[2] = -f0 - f1 + 4 * f2 - f3;
                    tet4_f[3] = -f0 - f1 - f2 + 4 * f3;
                }
#endif
                // const real_t dV = measure * tet_qw[q];
                const real_t dV = tet_qw[q];

                const real_t grid_x = (g_qx - ox) / dx;
                const real_t grid_y = (g_qy - oy) / dy;
                const real_t grid_z = (g_qz - oz) / dz;

                const ptrdiff_t i = floor(grid_x);
                const ptrdiff_t j = floor(grid_y);
                const ptrdiff_t k = floor(grid_z);

                // If outside
                if (i < 0 || j < 0 || k < 0 || (i + 1 >= n[0]) || (j + 1 >= n[1]) || (k + 1 >= n[2])) {
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

            for (int edof_i = 0; edof_i < 4; edof_i++) {
                element_field[edof_i] *= measure;
            }

            UNROLL_ZERO
            for (int v = 0; v < 4; ++v) {
                // Invert sign since distance field is negative insdide and positive outside
#pragma omp critical
                {
                    weighted_field[ev[v]] += element_field[v];
                }

            }  // end vertex loop
        }  // end element loop
    }  // end parallel region

    return 0;
}

int trishell3_resample_field_local(
        // Mesh
        const ptrdiff_t nelements, const ptrdiff_t nnodes, idx_t** const SFEM_RESTRICT elems, geom_t** const SFEM_RESTRICT xyz,
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n, const ptrdiff_t* const SFEM_RESTRICT stride,
        const geom_t* const SFEM_RESTRICT origin, const geom_t* const SFEM_RESTRICT delta, const real_t* const SFEM_RESTRICT data,
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
            idx_t  ev[3];
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

            const real_t measure = tri_shell_3_measure(x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2]);

            assert(measure > 0);

            for (int q = 0; q < TRI3_NQP; q++) {
                real_t g_qx, g_qy, g_qz;
                tri_shell_3_transform(
                        x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2], tri3_qx[q], tri3_qy[q], &g_qx, &g_qy, &g_qz);

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
                if (i < 0 || j < 0 || k < 0 || (i + 1 >= n[0]) || (j + 1 >= n[1]) || (k + 1 >= n[2])) {
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
                {
                    weighted_field[ev[v]] += element_field[v];
                }

            }  // end vertex loop
        }  // end element loop
    }  // end parallel region

    return 0;
}  // end trishell3_resample_field_local

int beam2_resample_field_local(const ptrdiff_t nelements, const ptrdiff_t nnodes, idx_t** const SFEM_RESTRICT elems,
                               geom_t** const SFEM_RESTRICT xyz,
                               // SDF
                               const ptrdiff_t* const SFEM_RESTRICT n, const ptrdiff_t* const SFEM_RESTRICT stride,
                               const geom_t* const SFEM_RESTRICT origin, const geom_t* const SFEM_RESTRICT delta,
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
            idx_t  ev[3];
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
                beam2_transform(x[0], x[1], y[0], y[1], z[0], z[1], edge2_qx[q], &g_qx, &g_qy, &g_qz);

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
                    beam2_f[0]      = 2 * f0 - f1;
                    beam2_f[1]      = -f0 + 2 * f1;
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
                if (i < 0 || j < 0 || k < 0 || (i + 1 >= n[0]) || (j + 1 >= n[1]) || (k + 1 >= n[2])) {
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
            }  // end quadrature loop

            UNROLL_ZERO
            for (int v = 0; v < 2; ++v) {
                // Invert sign since distance field is negative insdide and positive outside
#pragma omp critical
                {
                    weighted_field[ev[v]] += element_field[v];
                }

            }  // end vertex loop

        }  // end element loop
    }  // end parallel region

    return 0;
}  // end beam2_resample_field_local
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

#define MY_RESTRICT __restrict__
#define real_type real_t

int                                                    //
tet4_resample_field_local_CUDA(                        //
        const ptrdiff_t                    nelements,  // Mesh
        const ptrdiff_t                    nnodes,     // Mesh
        int** const MY_RESTRICT            elems,      // Mesh
        float** const MY_RESTRICT          xyz,        // Mesh
        const ptrdiff_t* const MY_RESTRICT n,          // SDF
        const ptrdiff_t* const MY_RESTRICT stride,     // SDF
        const float* const MY_RESTRICT     origin,     // SDF
        const float* const MY_RESTRICT     delta,      // SDF
        const real_type* const MY_RESTRICT data,       // SDF
        real_type* const MY_RESTRICT       weighted_field);  // Output

int                                                    //
tet4_resample_field_local_reduce_CUDA(                 //
        const ptrdiff_t                    nelements,  // Mesh
        const ptrdiff_t                    nnodes,     // Mesh
        int** const MY_RESTRICT            elems,      // Mesh
        float** const MY_RESTRICT          xyz,        // Mesh
        const ptrdiff_t* const MY_RESTRICT n,          // SDF
        const ptrdiff_t* const MY_RESTRICT stride,     // SDF
        const float* const MY_RESTRICT     origin,     // SDF
        const float* const MY_RESTRICT     delta,      // SDF
        const real_type* const MY_RESTRICT data,       // SDF
        real_type* const MY_RESTRICT       weighted_field);  // Output

int                                                                                         //
tet4_resample_field_local_reduce_CUDA_wrapper(const int mpi_size,                           // MPI size
                                              const int mpi_rank,                           // MPI rank
                                              mesh_t*   mesh,                               // Mesh
                                              int*      bool_assemble_dual_mass_vector,     // assemble dual mass vector
                                              const ptrdiff_t* const SFEM_RESTRICT n,       // number of nodes in each direction
                                              const ptrdiff_t* const SFEM_RESTRICT stride,  // stride of the data
                                              const geom_t* const SFEM_RESTRICT    origin,  // origin of the domain
                                              const geom_t* const SFEM_RESTRICT    delta,   // delta of the domain
                                              const real_t* const SFEM_RESTRICT    data,    // SDF
                                              real_t* const SFEM_RESTRICT          g_host);          // Output

int tet4_resample_field_local_V8(
        // Mesh
        const ptrdiff_t nelements, const ptrdiff_t nnodes, int** const MY_RESTRICT elems, float** const MY_RESTRICT xyz,
        // SDF
        const ptrdiff_t* const MY_RESTRICT n, const ptrdiff_t* const MY_RESTRICT stride, const float* const MY_RESTRICT origin,
        const float* const MY_RESTRICT delta, const real_type* const MY_RESTRICT data,
        // Output
        real_type* const MY_RESTRICT weighted_field);

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// resample_field_local ////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

#include "sfem_resample_field_vec.h"

#define USE_TET4_V4 0
#define USE_TET4_V8 1
#define USE_TET4_V16 2
#define USE_TET4_CUDA 10

#if SFEM_TET4_CUDA == ON
#define USE_TET4_MODEL USE_TET4_CUDA

#elif _VL_ == 8
#define USE_TET4_MODEL USE_TET4_V8
#elif _VL_ == 4
#define USE_TET4_MODEL USE_TET4_V4
#elif _VL_ == 16
#define USE_TET4_MODEL USE_TET4_V16
#endif

// #if SFEM_TET4_CUDA == ON
// #define USE_TET4_MODEL USE_TET4_CUDA
// #endif

int resample_field_local(
        // Mesh
        const enum ElemType element_type, const ptrdiff_t nelements, const ptrdiff_t nnodes, idx_t** const SFEM_RESTRICT elems,
        geom_t** const SFEM_RESTRICT xyz,
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n, const ptrdiff_t* const SFEM_RESTRICT stride,
        const geom_t* const SFEM_RESTRICT origin, const geom_t* const SFEM_RESTRICT delta, const real_t* const SFEM_RESTRICT data,
        // Output
        real_t* const SFEM_RESTRICT weighted_field, sfem_resample_field_info* info) {
    //

    PRINT_CURRENT_FUNCTION;

    switch (info->element_type) {
        case TET4: {
            info->quad_nodes_cnt = TET_QUAD_NQP;
            info->nelements      = nelements;

            // tet4_resample_field_local_reduce_CUDA

#if USE_TET4_MODEL == USE_TET4_V4 || USE_TET4_MODEL == USE_TET4_V8 || USE_TET4_MODEL == USE_TET4_V16
            return tet4_resample_field_local_V(nelements,        //
                                               nnodes,           //
                                               elems,            //
                                               xyz,              //
                                               n,                //
                                               stride,           //
                                               origin,           //
                                               delta,            //
                                               data,             //
                                               weighted_field);  //
// #elif USE_TET4_MODEL == USE_TET4_V8
//             return tet4_resample_field_local_V(
//                     nelements, nnodes, elems, xyz, n, stride, origin, delta, data,
//                     weighted_field);
#elif USE_TET4_MODEL == USE_TET4_CUDA
            return tet4_resample_field_local_reduce_CUDA(
                    nelements, nnodes, elems, xyz, n, stride, origin, delta, data, weighted_field);
#endif
        }

        break;

        case TET10: {
// #define TET10_V2
#if SFEM_TET10_CUDA == ON
            const int ret = hex8_to_tet10_resample_field_local_CUDA(
                    nelements, nnodes, 1, elems, xyz, n, stride, origin, delta, data, weighted_field);

            RETURN_FROM_FUNCTION(ret);
#else
            const int ret = hex8_to_tet10_resample_field_local(
                    nelements, nnodes, elems, xyz, n, stride, origin, delta, data, weighted_field);

            RETURN_FROM_FUNCTION(ret);
#endif
        }

        break;

        default:
            break;
    }

    enum ElemType st = shell_type(element_type);

    switch (st) {
        case TRISHELL3:
            return trishell3_resample_field_local(nelements, nnodes, elems, xyz, n, stride, origin, delta, data, weighted_field);
        case BEAM2:
            return beam2_resample_field_local(nelements, nnodes, elems, xyz, n, stride, origin, delta, data, weighted_field);

        default: {
            assert(0);
            fprintf(stderr, "Unknown element type %d\n", st);
            MPI_Abort(MPI_COMM_WORLD, -1);
            return EXIT_FAILURE;
        }
    }

    RETURN_FROM_FUNCTION(0);
}  // end resample_field_local

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// resample_field /////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
int resample_field(const enum ElemType                  element_type,  // Mesh: element type
                   const ptrdiff_t                      nelements,     // Mesh: number of elements
                   const ptrdiff_t                      nnodes,        // Mesh: number of nodes
                   idx_t** const SFEM_RESTRICT          elems,         // Mesh: connectivity
                   geom_t** const SFEM_RESTRICT         xyz,           // Mesh: coordinates
                   const ptrdiff_t* const SFEM_RESTRICT n,             // Sdf: number of nodes in each direction
                   const ptrdiff_t* const SFEM_RESTRICT stride,        // Sdf: stride of the data
                   const geom_t* const SFEM_RESTRICT    origin,        // Sdf: origin ox oy oz
                   const geom_t* const SFEM_RESTRICT    delta,         // Sdf: delta dx dy dz
                   const real_t* const SFEM_RESTRICT    data,          // Sdf: data
                   real_t* const SFEM_RESTRICT          g,             // Sdf: gap function (output)
                   sfem_resample_field_info*            info) {                   // Output
    //
    PRINT_CURRENT_FUNCTION;

    real_t* weighted_field = calloc(nnodes, sizeof(real_t));

    if (element_type == TET10 && SFEM_TET10_CUDA == ON) {
#if SFEM_TET10_CUDA == ON
        const int ret =
                hex8_to_tet10_resample_field_local_CUDA(nelements, nnodes, 1, elems, xyz, n, stride, origin, delta, data, g);

        RETURN_FROM_FUNCTION(ret);
#endif
    }

    resample_field_local(element_type, nelements, nnodes, elems, xyz, n, stride, origin, delta, data, weighted_field, info);

    enum ElemType st = shell_type(element_type);

    if (INVALID == st) {
        // FIXME
        if (element_type == TET10) {
            real_t* mass_vector = calloc(nnodes, sizeof(real_t));

            // // set the mass vector to zeros
            // for (ptrdiff_t i = 0; i < nnodes; i++) {
            //     mass_vector[i] = 0;
            // }

            tet10_assemble_dual_mass_vector(nelements, nnodes, elems, xyz, mass_vector);

            for (ptrdiff_t i = 0; i < nnodes; i++) {
                assert(mass_vector[i] != 0);
                g[i] = weighted_field[i] / mass_vector[i];
            }  // end for (i) loop

            free(mass_vector);
        } else {
            // Removing the mass-contributions from the weighted gap function "weighted_field"
            apply_inv_lumped_mass(element_type, nelements, nnodes, elems, xyz, weighted_field, g);
        }  // end if (TET10 == element_type)
    } else {
        apply_inv_lumped_mass(st, nelements, nnodes, elems, xyz, weighted_field, g);
    }  // end if (INVALID == st)

    free(weighted_field);

    RETURN_FROM_FUNCTION(0);
    // return 0;
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// perform_exchange_operations /////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
void                                              //
perform_exchange_operations(mesh_t* mesh,         //
                            real_t* mass_vector,  //
                            real_t* g) {          //

    send_recv_t slave_to_master;
    mesh_create_nodal_send_recv(mesh, &slave_to_master);

    ptrdiff_t count       = mesh_exchange_master_buffer_count(&slave_to_master);
    real_t*   real_buffer = (real_t*)calloc(count, sizeof(real_t));

    exchange_add(mesh, &slave_to_master, mass_vector, real_buffer);
    exchange_add(mesh, &slave_to_master, g, real_buffer);

    free(real_buffer);
    real_buffer = NULL;

    send_recv_destroy(&slave_to_master);
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// resample_field_mesh ////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
int                                                                      //
resample_field_mesh_tet4(const int                            mpi_size,  // MPI size
                         const int                            mpi_rank,  // MPI rank
                         const mesh_t* const SFEM_RESTRICT    mesh,      // Mesh: mesh_t struct
                         const ptrdiff_t* const SFEM_RESTRICT nlocal,    // SDF: nlocal[3]
                         const ptrdiff_t* const SFEM_RESTRICT stride,    // SDF: stride[3]
                         const geom_t* const SFEM_RESTRICT    origin,    // SDF: origin[3]
                         const geom_t* const SFEM_RESTRICT    delta,     // SDF: delta[3]
                         const real_t* const SFEM_RESTRICT    data,      // SDF: data
                         real_t* const SFEM_RESTRICT          g,         // Output
                         sfem_resample_field_info*            info) {               //
    //
    PRINT_CURRENT_FUNCTION;

    int assemble_dual_mass_vector = 0;

    int ret = 1;

    {  // Begin of the calls to the resample_field_local
#if USE_TET4_MODEL == USE_TET4_V4 || USE_TET4_MODEL == USE_TET4_V8 || USE_TET4_MODEL == USE_TET4_V16
        ret = tet4_resample_field_local_V(mesh->nelements,  //
                                          mesh->nnodes,     //
                                          mesh->elements,   //
                                          mesh->points,     //
                                          nlocal,           //
                                          stride,           //
                                          origin,           //
                                          delta,            //
                                          data,             //
                                          g);               //

#elif USE_TET4_MODEL == USE_TET4_CUDA

        ret = tet4_resample_field_local_reduce_CUDA_wrapper(mpi_size,                    //
                                                            mpi_rank,                    //
                                                            mesh,                        //
                                                            &assemble_dual_mass_vector,  //
                                                            nlocal,                      //
                                                            stride,                      //
                                                            origin,                      //
                                                            delta,                       //
                                                            data,                        //
                                                            g);                          //

        if (assemble_dual_mass_vector == 1) {
            // the exchange was mede in the kernel
            RETURN_FROM_FUNCTION(ret);
        }

#endif
    }

    real_t* mass_vector = calloc(mesh->nnodes, sizeof(real_t));

    {
        enum ElemType st = shell_type(mesh->element_type);  // The only possible outcome for TET4 is INVALID
        st               = (st == INVALID) ? mesh->element_type : st;
        assemble_lumped_mass(st,               //
                             mesh->nelements,  //
                             mesh->nnodes,     //
                             mesh->elements,   //
                             mesh->points,     //
                             mass_vector);     //
    }

    {
        //// TODO In CPU must be called.
        //// TODO In GPU should be calculated in the kernel calls in case of unified and Managed memory
        //// TODO In GPU is calculated here in case of host memory and more than one MPI rank (at the moment)

        // exchange ghost nodes and add contribution
        if (mpi_size > 1) {
            perform_exchange_operations((mesh_t*)mesh, mass_vector, g);
        }  // end if mpi_size > 1

        // divide by the mass vector
        for (ptrdiff_t i = 0; i < mesh->n_owned_nodes; i++) {
            if (mass_vector[i] == 0)
                fprintf(stderr, "Found 0 mass at %ld, info (%ld, %ld)\n", i, mesh->n_owned_nodes, mesh->nnodes);

            assert(mass_vector[i] != 0);
            g[i] /= mass_vector[i];
        }  // end for i < mesh.n_owned_nodes
    }

    free(mass_vector);
    mass_vector = NULL;

    RETURN_FROM_FUNCTION(ret);
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// resample_field_adjoint //////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
int                                                                             //
resample_field_adjoint_tet4(const int                            mpi_size,      // MPI size
                            const int                            mpi_rank,      // MPI rank
                            const mesh_t* const SFEM_RESTRICT    mesh,          // Mesh: mesh_t struct
                            const ptrdiff_t* const SFEM_RESTRICT n,             // SDF: n[3]
                            const ptrdiff_t* const SFEM_RESTRICT stride,        // SDF: stride[3]
                            const geom_t* const SFEM_RESTRICT    origin,        // SDF: origin[3]
                            const geom_t* const SFEM_RESTRICT    delta,         // SDF: delta[3]
                            const real_t* const SFEM_RESTRICT    g,             // Weighted field
                            const function_XYZ_t                 fun_XYZ,       // Function to apply
                            real_t* const SFEM_RESTRICT          data,          // SDF: data (output)
                            unsigned int*                        data_cnt,      // SDF: data count (output)
                            real_t const*                        alpha,         // SDF: tet alpha
                            real_t const*                        volume,        // SDF: tet volume
                            real_t const*                        data_fun_XYZ,  // SDF: data for fun_XYZ
                            sfem_resample_field_info*            info,          // Info struct with options and flags
                            const mini_tet_parameters_t          mini_tet_parameters) {  // Info struct with options and flags
    //
    PRINT_CURRENT_FUNCTION;
    int ret = 0;

    real_t* mass_vector = calloc(mesh->nnodes, sizeof(real_t));

    {  // Apply the mass matrix to the adjoint field

        {
            enum ElemType st = shell_type(mesh->element_type);  // The only possible outcome for TET4 is INVALID
            st               = (st == INVALID) ? mesh->element_type : st;
            assemble_lumped_mass(st,               //
                                 mesh->nelements,  //
                                 mesh->nnodes,     //
                                 mesh->elements,   //
                                 mesh->points,     //
                                 mass_vector);     //
        }

        {
            // exchange ghost nodes and add contribution
            // if (mpi_size > 1) {
            //     perform_exchange_operations((mesh_t*)mesh, mass_vector, g);
            // }  // end if mpi_size > 1

            // divide by the mass vector
            for (ptrdiff_t i = 0; i < mesh->n_owned_nodes; i++) {
                if (mass_vector[i] == 0)
                    fprintf(stderr, "Found 0 mass at %ld, info (%ld, %ld)\n", i, mesh->n_owned_nodes, mesh->nnodes);

                assert(mass_vector[i] != 0);
                // g[i] /= mass_vector[i];
                mass_vector[i] = g[i] / mass_vector[i];
                // printf("mass_vector[%ld] = %g\n", i, mass_vector[i]);

                // DEBUG: - to be removed
                mass_vector[i] = g[i];  // DEBUG: - to be removed: it directly pass the weighted field

            }  // end for i < mesh.n_owned_nodes
        }

    }  // end Apply the mass matrix to the adjoint field

    const real_t alpha_th = info->alpha_th;

    switch (info->adjoint_refine_type) {
        case ADJOINT_REFINE_ITERATIVE:
            ret = tet4_resample_field_local_ref_iter_adjoint_stack(0,                              //
                                                                   mesh->nelements,                //
                                                                   mesh->nnodes,                   //
                                                                   (const idx_t**)mesh->elements,  //
                                                                   (const geom_t**)mesh->points,   //
                                                                   n,                              //
                                                                   stride,                         //
                                                                   origin,                         //
                                                                   delta,                          //
                                                                   mass_vector,                    //
                                                                   alpha_th,                       //
                                                                   data);                          //

            break;

        case ADJOINT_REFINE_ITERATIVE_QUEUE:
            ret = tet4_resample_field_local_ref_iter_adjoint_queue(0,                              //
                                                                   mesh->nelements,                //
                                                                   mesh->nnodes,                   //
                                                                   (const idx_t**)mesh->elements,  //
                                                                   (const geom_t**)mesh->points,   //
                                                                   n,                              //
                                                                   stride,                         //
                                                                   origin,                         //
                                                                   delta,                          //
                                                                   mass_vector,                    //
                                                                   alpha_th,                       //
                                                                   data);                          //
            break;

        case ADJOINT_REFINE_ONE_STEP:
            ret = tet4_resample_field_local_refine_adjoint(0,                              //
                                                           mesh->nelements,                //
                                                           mesh->nnodes,                   //
                                                           (const idx_t**)mesh->elements,  //
                                                           (const geom_t**)mesh->points,   //
                                                           n,                              //
                                                           stride,                         //
                                                           origin,                         //
                                                           delta,                          //
                                                           mass_vector,                    //
                                                           alpha_th,                       //
                                                           data);                          //
            break;

        case ADJOINT_REFINE_HYTEG_REFINEMENT:

            // #define TEST_GPU_HYTEG_REFINEMENT
            // #define COMPUTE_FUN_XYZ_HEX

#if defined(TEST_GPU_HYTEG_REFINEMENT) && defined(SFEM_ENABLE_CUDA)

            ret = tet4_resample_field_local_refine_adjoint_hyteg_gpu(0,                              //
                                                                     mesh->nelements,                //
                                                                     mesh->nnodes,                   //
                                                                     (const idx_t**)mesh->elements,  //
                                                                     (const geom_t**)mesh->points,   //
                                                                     n,                              //
                                                                     stride,                         //
                                                                     origin,                         //
                                                                     delta,                          //
                                                                     mass_vector,                    //
                                                                     mini_tet_parameters,            //
                                                                     data);                          //

            break;

#else

            ret = tet4_resample_field_adjoint_hex_quad_d  //

                    //  ret = tet4_resample_field_local_refine_adjoint_hyteg_d  //
                    (0,                              //
                     mesh->nelements,                //
                     mesh->nnodes,                   //
                     (const idx_t**)mesh->elements,  //
                     (const geom_t**)mesh->points,   //
                     n,                              //
                     stride,                         //
                     origin,                         //
                     delta,                          //
                     mass_vector,                    //
                     mini_tet_parameters,            //
                     data);                          //
#ifdef COMPUTE_FUN_XYZ_HEX
            if (fun_XYZ != NULL && data_fun_XYZ != NULL) {
                ret = tet4_resample_field_apply_fun_to_hexa_d(0,                              //
                                                              mesh->nelements,                //
                                                              mesh->nnodes,                   //
                                                              (const idx_t**)mesh->elements,  //
                                                              (const geom_t**)mesh->points,   //
                                                              n,                              //
                                                              stride,                         //
                                                              origin,                         //
                                                              delta,                          //
                                                              mass_vector,                    //
                                                              mini_tet_parameters,            //
                                                              fun_XYZ,                        //
                                                              data_fun_XYZ);                  //
            }
#endif

            break;

#endif

        case ADJOINT_BASE:
        default:
            ret = tet4_resample_field_local_adjoint(0,                              //
                                                    mesh->nelements,                //
                                                    mesh->nnodes,                   //
                                                    (const idx_t**)mesh->elements,  //
                                                    (const geom_t**)mesh->points,   //
                                                    n,                              //
                                                    stride,                         //
                                                    origin,                         //
                                                    delta,                          //
                                                    mass_vector,                    //
                                                    data);                          //

            break;
    }

    if (data_cnt != NULL) {
        // ret = tet4_cnt_mesh_adjoint(0,                              //
        //                             mesh->nelements,                //
        //                             mesh->nnodes,                   //
        //                             (const idx_t**)mesh->elements,  //
        //                             (const geom_t**)mesh->points,   //
        //                             n,                              //
        //                             stride,                         //
        //                             origin,                         //
        //                             delta,                          //
        //                             mass_vector,                    //
        //                             data_cnt);                      //
    }

    if (alpha != NULL) {
        // tet4_alpha_volume_mesh_adjoint(0,                              //
        //                                mesh->nelements,                //
        //                                mesh->nnodes,                   //
        //                                (const idx_t**)mesh->elements,  //
        //                                (const geom_t**)mesh->points,   //
        //                                n,                              //
        //                                stride,                         //
        //                                origin,                         //
        //                                delta,                          //
        //                                mass_vector,                    //
        //                                alpha,                          //
        //                                volume);                        //
    }

    free(mass_vector);
    mass_vector = NULL;

    RETURN_FROM_FUNCTION(ret);
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// resample_field_TEST_adjoint_tet4 ////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
int                                                                              //
resample_field_TEST_adjoint_tet4(const int                            mpi_size,  // MPI size
                                 const int                            mpi_rank,  // MPI rank
                                 const mesh_t* const SFEM_RESTRICT    mesh,      // Mesh: mesh_t struct
                                 const ptrdiff_t* const SFEM_RESTRICT n,         // SDF: n[3]
                                 const ptrdiff_t* const SFEM_RESTRICT stride,    // SDF: stride[3]
                                 const geom_t* const SFEM_RESTRICT    origin,    // SDF: origin[3]
                                 const geom_t* const SFEM_RESTRICT    delta,     // SDF: delta[3]
                                 const real_t* const SFEM_RESTRICT    in_data,   // Weighted field
                                 real_t* const SFEM_RESTRICT          out_data,  // SDF: data (output)
                                 real_t* const SFEM_RESTRICT          g,         // Weighted field (output)
                                 sfem_resample_field_info*            info) {               // Info struct with options and flags

    PRINT_CURRENT_FUNCTION;

    int ret = 0;

    ret = tet4_resample_field_local_V(mesh->nelements,  //
                                      mesh->nnodes,     //
                                      mesh->elements,   //
                                      mesh->points,     //
                                      n,                //
                                      stride,           //
                                      origin,           //
                                      delta,            //
                                      in_data,          //
                                      g);               //

    real_t* mass_vector = calloc(mesh->nnodes, sizeof(real_t));

    {
        enum ElemType st = shell_type(mesh->element_type);  // The only possible outcome for TET4 is INVALID
        st               = (st == INVALID) ? mesh->element_type : st;
        assemble_lumped_mass(st,               //
                             mesh->nelements,  //
                             mesh->nnodes,     //
                             mesh->elements,   //
                             mesh->points,     //
                             mass_vector);     //
    }

    {
        //// TODO In CPU must be called.
        //// TODO In GPU should be calculated in the kernel calls in case of unified and Managed memory
        //// TODO In GPU is calculated here in case of host memory and more than one MPI rank (at the moment)

        // exchange ghost nodes and add contribution
        if (mpi_size > 1) {
            perform_exchange_operations((mesh_t*)mesh, mass_vector, g);
        }  // end if mpi_size > 1

        // divide by the mass vector
        for (ptrdiff_t i = 0; i < mesh->n_owned_nodes; i++) {
            if (mass_vector[i] == 0)
                fprintf(stderr, "Found 0 mass at %ld, info (%ld, %ld)\n", i, mesh->n_owned_nodes, mesh->nnodes);

            assert(mass_vector[i] != 0);
            g[i] /= mass_vector[i];
        }  // end for i < mesh.n_owned_nodes
    }

    free(mass_vector);
    mass_vector = NULL;

    ret = tet4_resample_field_local_adjoint(0,                              //
                                            mesh->nelements,                //
                                            mesh->nnodes,                   //
                                            (const idx_t**)mesh->elements,  //
                                            (const geom_t**)mesh->points,   //
                                            n,                              //
                                            stride,                         //
                                            origin,                         //
                                            delta,                          //
                                            g,                              //
                                            out_data);                      //

    RETURN_FROM_FUNCTION(ret);
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// resample_field_adjoint_tet10 ////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
int                                                                               //
resample_field_mesh_adjoint_tet10(const int                            mpi_size,  // MPI size
                                  const int                            mpi_rank,  // MPI rank
                                  const mesh_t* const SFEM_RESTRICT    mesh,      // Mesh: mesh_t struct
                                  const ptrdiff_t* const SFEM_RESTRICT n,         // SDF: n[3]
                                  const ptrdiff_t* const SFEM_RESTRICT stride,    // SDF: stride[3]
                                  const geom_t* const SFEM_RESTRICT    origin,    // SDF: origin[3]
                                  const geom_t* const SFEM_RESTRICT    delta,     // SDF: delta[3]
                                  const real_t* const SFEM_RESTRICT    g,         // Weighted field
                                  real_t* const SFEM_RESTRICT          data,      // SDF: data (output)
                                  unsigned int*                        data_cnt,  // SDF: data count (output)
                                  sfem_resample_field_info*            info) {               // Info struct with options and flags

    PRINT_CURRENT_FUNCTION;
    int ret = 0;

    enum ElemType st             = shell_type(mesh->element_type);
    real_t*       mass_vector    = calloc(mesh->nnodes, sizeof(real_t));
    real_t*       weighted_field = malloc(mesh->nnodes * sizeof(real_t));
    memcpy(weighted_field, g, mesh->nnodes * sizeof(real_t));

    if (st == INVALID) {
        // printf("INVALID == st\n");

        tet10_assemble_dual_mass_vector(mesh->nelements,  //
                                        mesh->nnodes,     //
                                        mesh->elements,   //
                                        mesh->points,     //
                                        mass_vector);     //

        // // exchange ghost nodes and add contribution
        // if (mpi_size > 1) {                                                     //
        //     printf("perform_exchange_operations %s:%d\n", __FILE__, __LINE__);  //
        //     perform_exchange_operations(mesh,                                   //
        //                                 mass_vector,                            //
        //                                 weighted_field);                        //
        // }

        for (ptrdiff_t i = 0; i < mesh->nnodes; i++) {  //
            // assert(mass_vector[i] != 0);                             //

            // DEBUG: - to be uncommented after the tests
            // weighted_field[i] = weighted_field[i] / mass_vector[i];  //

        }  // end for (i) loop

    } else {
        apply_inv_lumped_mass(st,               //
                              mesh->nelements,  //
                              mesh->nnodes,     //
                              mesh->elements,   //
                              mesh->points,     //
                              weighted_field,   //
                              g);               //

    }  // end if (INVALID == st)

    free(mass_vector);
    mass_vector = NULL;

    const real_t alpha_th = 2.0;

#define TEST_REFINE_ADJOINT 3
#if TEST_REFINE_ADJOINT == 0

    hex8_to_isoparametric_tet10_resample_field_adjoint(0,                //
                                                       mesh->nelements,  //
                                                       mesh->nnodes,     //
                                                       mesh->elements,   //
                                                       mesh->points,     //
                                                       n,                //
                                                       stride,           //
                                                       origin,           //
                                                       delta,            //
                                                       weighted_field,   //
                                                       data);            //

#elif TEST_REFINE_ADJOINT == 1

#pragma message "ATTENTIN Using TEST_REFINE_ADJOINT for TET10"

    hex8_to_isoparametric_tet10_resample_field_iterative_ref_adjoint(0,                                    //
                                                                     mesh->nelements,                      //
                                                                     mesh->nnodes,                         //
                                                                     (const idx_t** const)mesh->elements,  //
                                                                     (const geom_t** const)mesh->points,   //
                                                                     n,                                    //
                                                                     stride,                               //
                                                                     origin,                               //
                                                                     delta,                                //
                                                                     weighted_field,                       //
                                                                     alpha_th,                             //
                                                                     data);                                //

#elif TEST_REFINE_ADJOINT == 2

    hex8_to_isoparametric_tet10_resample_field_refine_adjoint(0,                //
                                                              mesh->nelements,  //
                                                              mesh->nnodes,     //
                                                              mesh->elements,   //
                                                              mesh->points,     //
                                                              n,                //
                                                              stride,           //
                                                              origin,           //
                                                              delta,            //
                                                              weighted_field,   //
                                                              alpha_th,         //
                                                              data);            //

#else

    mini_tet_parameters_t mini_tet_parameters;

    mini_tet_parameters.alpha_min_threshold = 1.0;
    mini_tet_parameters.alpha_max_threshold = 8.0;
    mini_tet_parameters.min_refinement_L    = 1;
    mini_tet_parameters.max_refinement_L    = 15;

#if defined(SFEM_ENABLE_CUDA)
    hex8_to_isoparametric_tet10_resample_field_hyteg_mt_adjoint_gpu(0,                              //
                                                                    mesh->nelements,                //
                                                                    mesh->nnodes,                   //
                                                                    (const idx_t**)mesh->elements,  //
                                                                    (const geom_t**)mesh->points,   //
                                                                    n,                              //
                                                                    stride,                         //
                                                                    origin,                         //
                                                                    delta,                          //
                                                                    weighted_field,                 //
                                                                    data,                           //
                                                                    mini_tet_parameters);           //
#endif

#endif

    // const real_t    volume_hex = delta[0] * delta[1] * delta[2];
    // const ptrdiff_t data_size  = n[0] * n[1] * n[2];
    // for (ptrdiff_t i = 0; i < data_size; i++) {
    //     data[i] /= volume_hex;
    // }
    free(weighted_field);
    weighted_field = NULL;

    RETURN_FROM_FUNCTION(ret);
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// resample_field_mesh_tet10 ///////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
int                                                                       //
resample_field_mesh_tet10(const int                            mpi_size,  // MPI size
                          const int                            mpi_rank,  // MPI rank
                          const mesh_t* const SFEM_RESTRICT    mesh,      // Mesh: mesh_t struct
                          const ptrdiff_t* const SFEM_RESTRICT n,         // SDF: n[3]
                          const ptrdiff_t* const SFEM_RESTRICT stride,    // SDF: stride[3]
                          const geom_t* const SFEM_RESTRICT    origin,    // SDF: origin[3]
                          const geom_t* const SFEM_RESTRICT    delta,     // SDF: delta[3]
                          const real_t* const SFEM_RESTRICT    data,      // SDF: data
                          real_t* const SFEM_RESTRICT          g,         // Output: g
                          sfem_resample_field_info*            info) {               // info struct with options and flags
    //
    PRINT_CURRENT_FUNCTION;

    int assemble_dual_mass_vector = 1;
    // This indicates if the exchange was made in the kernel
    // The default is 1, which means that the exchange was not made in the kernel
    // If the exchange was made in the kernel, this variable will be set to 0 in the CUDA wrapper.

    int ret = 0;

    const int mesh_nnodes = mpi_size > 1 ? mesh->nnodes : mesh->n_owned_nodes;
    // const int mesh_nnodes    = mesh->nnodes;

    real_t* weighted_field = NULL;

    if (info->use_accelerator == SFEM_ACCELERATOR_TYPE_CUDA && (SFEM_TET10_CUDA == ON)) {
#if SFEM_TET10_CUDA == ON
        if (SFEM_CUDA_MEMORY_MODEL == CUDA_HOST_MEMORY) assemble_dual_mass_vector = 0;

        ret = hex8_to_tet10_resample_field_local_CUDA_wrapper(mpi_size,                    //
                                                              mpi_rank,                    //
                                                              mesh,                        //
                                                              &assemble_dual_mass_vector,  //
                                                              n,                           //
                                                              stride,                      //
                                                              origin,                      //
                                                              delta,                       //
                                                              data,                        //
                                                              g);                          //

        if (assemble_dual_mass_vector == 1) {
            // the exchange was mede in the kernel
            RETURN_FROM_FUNCTION(ret);
        }

        weighted_field = g;  // for the cases where the exchange was not made in the kernel
#else
        fprintf(stderr, "SFEM_TET10_CUDA is OFF,  %s:%d\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
#endif
    }  // end if info->use_accelerator == SFEM_ACCELERATOR_TYPE_CUDA && (SFEM_TET10_CUDA == ON)

    if (info->use_accelerator == SFEM_ACCELERATOR_TYPE_CPU) {  // CPU

        weighted_field = calloc(mesh->nnodes, sizeof(real_t));

        ret = hex8_to_tet10_resample_field_local(mesh->nelements,  //
                                                 mesh_nnodes,      //
                                                 mesh->elements,   //
                                                 mesh->points,     //
                                                 n,                //
                                                 stride,           //
                                                 origin,           //
                                                 delta,            //
                                                 data,             //
                                                 weighted_field);  //

    }  // end if SFEM_TET10_CUDA == OFF

    enum ElemType st = shell_type(mesh->element_type);

    real_t* mass_vector = calloc(mesh->nnodes, sizeof(real_t));

    if (st == INVALID) {
        tet10_assemble_dual_mass_vector(mesh->nelements,  //
                                        mesh->nnodes,     //
                                        mesh->elements,   //
                                        mesh->points,     //
                                        mass_vector);     //

        // exchange ghost nodes and add contribution
        if (mpi_size > 1) {                                                     //
            printf("perform_exchange_operations %s:%d\n", __FILE__, __LINE__);  //
            perform_exchange_operations(mesh,                                   //
                                        mass_vector,                            //
                                        weighted_field);                        //
        }

        for (ptrdiff_t i = 0; i < mesh->nnodes; i++) {  //
            assert(mass_vector[i] != 0);                //
            g[i] = weighted_field[i] / mass_vector[i];  //
        }  // end for (i) loop

    } else {
        apply_inv_lumped_mass(st,               //
                              mesh->nelements,  //
                              mesh->nnodes,     //
                              mesh->elements,   //
                              mesh->points,     //
                              weighted_field,   //
                              g);               //

    }  // end if (INVALID == st)

    free(mass_vector);
    mass_vector = NULL;

    if (weighted_field != NULL && weighted_field != g) {
        free(weighted_field);
        weighted_field = NULL;
    }

    RETURN_FROM_FUNCTION(ret);
}  // end resample_field_mesh

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// interpolate_field ///////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
int                                                             //
interpolate_field(const ptrdiff_t                      nnodes,  // Mesh: nnodes
                  geom_t** const SFEM_RESTRICT         xyz,     // Mesh: xyz[3][nnodes]
                  const ptrdiff_t* const SFEM_RESTRICT n,       // SDF: n[3]
                  const ptrdiff_t* const SFEM_RESTRICT stride,  // SDF: stride[3]
                  const geom_t* const SFEM_RESTRICT    origin,  // SDF: origin[3]
                  const geom_t* const SFEM_RESTRICT    delta,   // SDF: delta[3]
                  const real_t* const SFEM_RESTRICT    data,    // SDF: data
                  real_t* const SFEM_RESTRICT          g) {              // Output: g

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

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// minmax //////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
SFEM_INLINE static void                         //
minmax(const ptrdiff_t                   n,     //
       const geom_t* const SFEM_RESTRICT x,     //
       geom_t*                           xmin,  //
       geom_t*                           xmax) {                          //

    *xmin = x[0];  //
    *xmax = x[0];  //

    for (ptrdiff_t i = 1; i < n; i++) {
        *xmin = MIN(*xmin, x[i]);
        *xmax = MAX(*xmax, x[i]);
    }
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// resample_wiew ///////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
int field_view(MPI_Comm                             comm,          //
               const ptrdiff_t                      nnodes,        //
               const geom_t* SFEM_RESTRICT          z_coordinate,  //
               const ptrdiff_t* const               nlocal,        //
               const ptrdiff_t* const SFEM_RESTRICT nglobal,       //
               const ptrdiff_t* const SFEM_RESTRICT stride,        //
               const geom_t* const                  origin,        //
               const geom_t* const SFEM_RESTRICT    delta,         //
               const real_t* const                  field,         //
               real_t**                             field_out,     //
               ptrdiff_t*                           z_nlocal_out,  //
               geom_t* const SFEM_RESTRICT          z_origin_out) {         //

    return field_view_ensure_margin(comm,           //
                                    nnodes,         //
                                    z_coordinate,   //
                                    nlocal,         //
                                    nglobal,        //
                                    stride,         //
                                    origin,         //
                                    delta,          //
                                    field,          //
                                    3,              //
                                    field_out,      //
                                    z_nlocal_out,   //
                                    z_origin_out);  //
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// resample_wiew_ensure_margin /////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
int                                                                          //
field_view_ensure_margin(MPI_Comm                             comm,          //
                         const ptrdiff_t                      nnodes,        //
                         const geom_t* SFEM_RESTRICT          z_coordinate,  //
                         const ptrdiff_t* const               nlocal,        //
                         const ptrdiff_t* const SFEM_RESTRICT nglobal,       //
                         const ptrdiff_t* const SFEM_RESTRICT stride,        //
                         const geom_t* const                  origin,        //
                         const geom_t* const SFEM_RESTRICT    delta,         //
                         const real_t* const                  field,         //
                         const ptrdiff_t                      z_margin,      //
                         real_t**                             field_out,     //
                         ptrdiff_t*                           z_nlocal_out,  //
                         geom_t* const SFEM_RESTRICT          z_origin_out) {         //

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
    ptrdiff_t field_end   = (zmax - origin[2]) / delta[2];

    // Make sure we are inside the grid and get also the required margin for resampling
    field_start = MAX(0, field_start - 1 - z_margin);
    field_end   = MIN(nglobal[2],
                    field_end + 2 + z_margin);  // 1 for the rightside of the cell 1 for the exclusive range

    ptrdiff_t pnlocal_z = (field_end - field_start);
    real_t*   pfield    = malloc(pnlocal_z * stride[2] * sizeof(real_t));

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

    *field_out    = pfield;
    *z_nlocal_out = pnlocal_z;
    *z_origin_out = origin[2] + field_start * delta[2];

    double field_view_tock = MPI_Wtime();

    if (!rank) {
        printf("[%d] field_view %g (seconds)\n", rank, field_view_tock - field_view_tick);
    }

    return 0;
}

// Function to normalize the field by hexahedron volume and find min/max
void                                                             //
normalize_field_and_find_min_max(real_t*         field,          // Input/Output: Field data to normalize
                                 const ptrdiff_t n_zyx,          // Input: Total size of the field array
                                 const geom_t    delta[3],       // Input: Grid spacing
                                 real_t*         out_min,        // Output: Minimum value found in the field
                                 real_t*         out_max,        //
                                 int*            out_min_index,  //
                                 int*            out_max_index) {           // Output: Maximum value found in the field

    int max_field_index = -1;
    int min_field_index = -1;

    if (!field || !delta || !out_min || !out_max) {
        fprintf(stderr, "Error: Invalid arguments provided to %s\n", __func__);
        // Optionally set outputs to sentinel values based on real_t type
        if (out_min) {
#ifdef SFEM_REAL_T_IS_FLOAT32
            *out_min = FLT_MAX;
#elif defined(SFEM_REAL_T_IS_FLOAT64)
            *out_min = DBL_MAX;
#else
            // Fallback or error for unsupported types
            *out_min = 0;  // Or some other indicator
            fprintf(stderr, "Warning: Unsupported real_t type for sentinel min value in %s\n", __func__);
#endif
        }
        if (out_max) {
#ifdef SFEM_REAL_T_IS_FLOAT32
            *out_max = -FLT_MAX;
#elif defined(SFEM_REAL_T_IS_FLOAT64)
            *out_max = -DBL_MAX;
#else
            // Fallback or error for unsupported types
            *out_max = 0;  // Or some other indicator
            fprintf(stderr, "Warning: Unsupported real_t type for sentinel max value in %s\n", __func__);
#endif
        }
        return;
    }

// Initialize min/max based on the actual type of real_t using config macros
#ifdef SFEM_REAL_T_IS_FLOAT32
    // real_t is float (32-bit)
    real_t min_val = FLT_MAX;
    real_t max_val = -FLT_MAX;
#elif defined(SFEM_REAL_T_IS_FLOAT64)
    // real_t is double (64-bit)
    real_t min_val = DBL_MAX;
    real_t max_val = -DBL_MAX;  // Use negative max for correct comparison
#else
// Fallback for unknown or unsupported real_t type
#error "SFEM_REAL_T_IS_FLOAT32 or SFEM_REAL_T_IS_FLOAT64 must be defined in sfem_config.h"
    // Define dummy values to allow compilation attempt, though it will fail at #error
    real_t min_val = 0;
    real_t max_val = 0;
#endif

    const real_t hexa_volume = delta[0] * delta[1] * delta[2];

    // Avoid division by zero or near-zero volume
    if (hexa_volume <= 1e-16) {  // Use a small threshold instead of exact zero
        fprintf(stderr,
                "Warning: hexa_volume is zero or close to zero (%g) in %s. Skipping normalization.\n",
                (double)hexa_volume,  // Cast to double for printf
                __func__);
        // Find min/max without normalization
        for (ptrdiff_t i = 0; i < n_zyx; i++) {
            if (field[i] > max_val) {
                max_val         = field[i];
                max_field_index = i;
            }
            if (field[i] < min_val) {
                min_val         = field[i];
                min_field_index = i;
            }
        }
    } else {
        // Normalize and find min/max
        for (ptrdiff_t i = 0; i < n_zyx; i++) {
            field[i] /= hexa_volume;

            if (field[i] > max_val) {
                max_val         = field[i];
                max_field_index = i;
            }
            if (field[i] < min_val) {
                min_val         = field[i];
                min_field_index = i;
            }
        }
    }

    *out_min       = min_val;
    *out_max       = max_val;
    *out_min_index = min_field_index;
    *out_max_index = max_field_index;
}
