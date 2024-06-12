#include "tet10_resample_field.h"

#include "quadratures_rule.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

SFEM_INLINE static void tet10_dual_basis_functions(const real_t qx,
                                                   const real_t qy,
                                                   const real_t qz,
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

SFEM_INLINE static real_t tet4_measure(
        // X-coordinates
        const real_t px0,
        const real_t px1,
        const real_t px2,
        const real_t px3,
        // Y-coordinates
        const real_t py0,
        const real_t py1,
        const real_t py2,
        const real_t py3,
        // Z-coordinates
        const real_t pz0,
        const real_t pz1,
        const real_t pz2,
        const real_t pz3) {
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
        const real_t px0,
        const real_t px1,
        const real_t px2,
        const real_t px3,
        // Y-coordinates
        const real_t py0,
        const real_t py1,
        const real_t py2,
        const real_t py3,
        // Z-coordinates
        const real_t pz0,
        const real_t pz1,
        const real_t pz2,
        const real_t pz3,
        // Quadrature point
        const real_t qx,
        const real_t qy,
        const real_t qz,
        // Output
        real_t* const SFEM_RESTRICT out_x,
        real_t* const SFEM_RESTRICT out_y,
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
        const real_t x,
        const real_t y,
        const real_t z,
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
        const ptrdiff_t* const SFEM_RESTRICT stride,
        const ptrdiff_t i,
        const ptrdiff_t j,
        const ptrdiff_t k,
        // Attention this is geometric data transformed to solver data!
        const real_t* const SFEM_RESTRICT data,
        real_t* const SFEM_RESTRICT out) {
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
    printf("============================================================\n");
    printf("Start: hex8_to_tet10_resample_field_local\n");
    printf("============================================================\n");
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
            tet4_measure(x[0],
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
                const real_t measure = 0;
                assert(0);

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
                tet10_dual_basis_functions(tet4_qx[q], tet4_qy[q], tet4_qz[q], tet10_f);

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

static SFEM_INLINE void lumped_mass_kernel(const real_t px0,
                                           const real_t px1,
                                           const real_t px2,
                                           const real_t px3,
                                           const real_t py0,
                                           const real_t py1,
                                           const real_t py2,
                                           const real_t py3,
                                           const real_t pz0,
                                           const real_t pz1,
                                           const real_t pz2,
                                           const real_t pz3,
                                           real_t* const SFEM_RESTRICT element_matrix_diag) {
    // generated code
    // FLOATING POINT OPS!
    //       - Result: 10*ASSIGNMENT
    //       - Subexpressions: 4*ADD + 12*DIV + 27*MUL + 15*SUB
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

int subparametric_tet10_assemble_dual_mass_vector(const ptrdiff_t nelements,
                                                  const ptrdiff_t nnodes,
                                                  idx_t** const SFEM_RESTRICT elems,
                                                  geom_t** const SFEM_RESTRICT xyz,
                                                  real_t* const values) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[10];
        real_t element_weights[10];
#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elems[v][i];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];
        const idx_t i3 = ev[3];

        lumped_mass_kernel(
                // X-coordinates
                xyz[0][i0],
                xyz[0][i1],
                xyz[0][i2],
                xyz[0][i3],
                // Y-coordinates
                xyz[1][i0],
                xyz[1][i1],
                xyz[1][i2],
                xyz[1][i3],
                // Z-coordinates
                xyz[2][i0],
                xyz[2][i1],
                xyz[2][i2],
                xyz[2][i3],
                element_weights);

        for (int v = 0; v < 10; ++v) {
            const idx_t idx = ev[v];

#pragma omp atomic update
            values[idx] += element_weights[v];
        }
    }

    return 0;
}
