#include "hex8_mass.h"
#include "hex8_quadrature.h"

#include <stdio.h>

#ifndef POW2
#define POW2(a) ((a) * (a))
#endif

static SFEM_INLINE void hex8_mass_apply_points(const scalar_t *const SFEM_RESTRICT x,
                                               const scalar_t *const SFEM_RESTRICT y,
                                               const scalar_t *const SFEM_RESTRICT z,
                                               const scalar_t qx,
                                               const scalar_t qy,
                                               const scalar_t qz,
                                               const scalar_t qw,
                                               const scalar_t *SFEM_RESTRICT u,
                                               accumulator_t *SFEM_RESTRICT element_vector) {
    const scalar_t x0 = 1 - qz;
    const scalar_t x1 = 1 - qy;
    const scalar_t x2 = 1 - qx;
    const scalar_t x3 = x1 * x2;
    const scalar_t x4 = x0 * x3;
    const scalar_t x5 = qx * qy;
    const scalar_t x6 = qx * x1;
    const scalar_t x7 = qy * x2;
    const scalar_t x8 = qx * qy * x[6] + qx * x1 * x[5] + qy * x2 * x[7] + x1 * x2 * x[4] -
                        x3 * x[0] - x5 * x[2] - x6 * x[1] - x7 * x[3];
    const scalar_t x9 = qx * qz;
    const scalar_t x10 = qx * x0;
    const scalar_t x11 = qz * x2;
    const scalar_t x12 = x0 * x2;
    const scalar_t x13 = qx * qz * y[6] + qx * x0 * y[2] + qz * x2 * y[7] + x0 * x2 * y[3] -
                         x10 * y[1] - x11 * y[4] - x12 * y[0] - x9 * y[5];
    const scalar_t x14 = qy * qz;
    const scalar_t x15 = qy * x0;
    const scalar_t x16 = qz * x1;
    const scalar_t x17 = x0 * x1;
    const scalar_t x18 = x14 * z[6] - x14 * z[7] + x15 * z[2] - x15 * z[3] - x16 * z[4] +
                         x16 * z[5] - x17 * z[0] + x17 * z[1];
    const scalar_t x19 = qx * qy * y[6] + qx * x1 * y[5] + qy * x2 * y[7] + x1 * x2 * y[4] -
                         x3 * y[0] - x5 * y[2] - x6 * y[1] - x7 * y[3];
    const scalar_t x20 = qx * qz * z[6] + qx * x0 * z[2] + qz * x2 * z[7] + x0 * x2 * z[3] -
                         x10 * z[1] - x11 * z[4] - x12 * z[0] - x9 * z[5];
    const scalar_t x21 = x14 * x[6] - x14 * x[7] + x15 * x[2] - x15 * x[3] - x16 * x[4] +
                         x16 * x[5] - x17 * x[0] + x17 * x[1];
    const scalar_t x22 = qx * qy * z[6] + qx * x1 * z[5] + qy * x2 * z[7] + x1 * x2 * z[4] -
                         x3 * z[0] - x5 * z[2] - x6 * z[1] - x7 * z[3];
    const scalar_t x23 = qx * qz * x[6] + qx * x0 * x[2] + qz * x2 * x[7] + x0 * x2 * x[3] -
                         x10 * x[1] - x11 * x[4] - x12 * x[0] - x9 * x[5];
    const scalar_t x24 = x14 * y[6] - x14 * y[7] + x15 * y[2] - x15 * y[3] - x16 * y[4] +
                         x16 * y[5] - x17 * y[0] + x17 * y[1];
    const scalar_t x25 = qz * x5;
    const scalar_t x26 = x0 * x5;
    const scalar_t x27 = qz * x6;
    const scalar_t x28 = qz * x7;
    const scalar_t x29 = x0 * x6;
    const scalar_t x30 = x0 * x7;
    const scalar_t x31 = qz * x3;
    const scalar_t x32 = qw *
                         (-x13 * x18 * x8 + x13 * x21 * x22 + x18 * x19 * x23 - x19 * x20 * x21 +
                          x20 * x24 * x8 - x22 * x23 * x24) *
                         (u[0] * x4 + u[1] * x29 + u[2] * x26 + u[3] * x30 + u[4] * x31 +
                          u[5] * x27 + u[6] * x25 + u[7] * x28);
    element_vector[0] += x32 * x4;
    element_vector[1] += x29 * x32;
    element_vector[2] += x26 * x32;
    element_vector[3] += x30 * x32;
    element_vector[4] += x31 * x32;
    element_vector[5] += x27 * x32;
    element_vector[6] += x25 * x32;
    element_vector[7] += x28 * x32;
}

static SFEM_INLINE void hex8_lumped_mass_points(const scalar_t *const SFEM_RESTRICT x,
                                                const scalar_t *const SFEM_RESTRICT y,
                                                const scalar_t *const SFEM_RESTRICT z,
                                                const scalar_t qx,
                                                const scalar_t qy,
                                                const scalar_t qz,
                                                const scalar_t qw,
                                                accumulator_t *SFEM_RESTRICT element_matrix_diag) {
    const scalar_t x0 = 1 - qx;
    const scalar_t x1 = POW2(x0);
    const scalar_t x2 = 1 - qy;
    const scalar_t x3 = POW2(x2);
    const scalar_t x4 = 1 - qz;
    const scalar_t x5 = qx * qy;
    const scalar_t x6 = qx * x2;
    const scalar_t x7 = qy * x0;
    const scalar_t x8 = x0 * x2;
    const scalar_t x9 = qx * qy * x[6] + qx * x2 * x[5] + qy * x0 * x[7] + x0 * x2 * x[4] -
                        x5 * x[2] - x6 * x[1] - x7 * x[3] - x8 * x[0];
    const scalar_t x10 = qx * qz;
    const scalar_t x11 = qx * x4;
    const scalar_t x12 = qz * x0;
    const scalar_t x13 = x0 * x4;
    const scalar_t x14 = qx * qz * y[6] + qx * x4 * y[2] + qz * x0 * y[7] + x0 * x4 * y[3] -
                         x10 * y[5] - x11 * y[1] - x12 * y[4] - x13 * y[0];
    const scalar_t x15 = qy * qz;
    const scalar_t x16 = qy * x4;
    const scalar_t x17 = qz * x2;
    const scalar_t x18 = x2 * x4;
    const scalar_t x19 = x15 * z[6] - x15 * z[7] + x16 * z[2] - x16 * z[3] - x17 * z[4] +
                         x17 * z[5] - x18 * z[0] + x18 * z[1];
    const scalar_t x20 = qx * qy * y[6] + qx * x2 * y[5] + qy * x0 * y[7] + x0 * x2 * y[4] -
                         x5 * y[2] - x6 * y[1] - x7 * y[3] - x8 * y[0];
    const scalar_t x21 = qx * qz * z[6] + qx * x4 * z[2] + qz * x0 * z[7] + x0 * x4 * z[3] -
                         x10 * z[5] - x11 * z[1] - x12 * z[4] - x13 * z[0];
    const scalar_t x22 = x15 * x[6] - x15 * x[7] + x16 * x[2] - x16 * x[3] - x17 * x[4] +
                         x17 * x[5] - x18 * x[0] + x18 * x[1];
    const scalar_t x23 = qx * qy * z[6] + qx * x2 * z[5] + qy * x0 * z[7] + x0 * x2 * z[4] -
                         x5 * z[2] - x6 * z[1] - x7 * z[3] - x8 * z[0];
    const scalar_t x24 = qx * qz * x[6] + qx * x4 * x[2] + qz * x0 * x[7] + x0 * x4 * x[3] -
                         x10 * x[5] - x11 * x[1] - x12 * x[4] - x13 * x[0];
    const scalar_t x25 = x15 * y[6] - x15 * y[7] + x16 * y[2] - x16 * y[3] - x17 * y[4] +
                         x17 * y[5] - x18 * y[0] + x18 * y[1];
    const scalar_t x26 = qw * (-x14 * x19 * x9 + x14 * x22 * x23 + x19 * x20 * x24 -
                               x20 * x21 * x22 + x21 * x25 * x9 - x23 * x24 * x25);
    const scalar_t x27 = x26 * POW2(x4);
    const scalar_t x28 = x27 * x3;
    const scalar_t x29 = x26 * x3;
    const scalar_t x30 = qz * x4;
    const scalar_t x31 = x29 * x30;
    const scalar_t x32 = x1 * x31;
    const scalar_t x33 = qx * x0;
    const scalar_t x34 = x10 * x13;
    const scalar_t x35 = x29 * x34;
    const scalar_t x36 = x5 * x8;
    const scalar_t x37 = x26 * x30 * x36;
    const scalar_t x38 = x27 * x36 + x37;
    const scalar_t x39 = x28 * x33 + x35 + x38;
    const scalar_t x40 = qy * x2;
    const scalar_t x41 = x27 * x40;
    const scalar_t x42 = x1 * x26;
    const scalar_t x43 = x15 * x18;
    const scalar_t x44 = x42 * x43;
    const scalar_t x45 = x1 * x41 + x44;
    const scalar_t x46 = POW2(qx);
    const scalar_t x47 = x31 * x46;
    const scalar_t x48 = x26 * x46;
    const scalar_t x49 = x43 * x48;
    const scalar_t x50 = x41 * x46 + x49;
    const scalar_t x51 = POW2(qy);
    const scalar_t x52 = x27 * x51;
    const scalar_t x53 = x30 * x51;
    const scalar_t x54 = x26 * x34 * x51;
    const scalar_t x55 = x48 * x53 + x54;
    const scalar_t x56 = x33 * x52 + x38;
    const scalar_t x57 = x42 * x53 + x54;
    const scalar_t x58 = POW2(qz) * x26;
    const scalar_t x59 = x3 * x58;
    const scalar_t x60 = x36 * x58 + x37;
    const scalar_t x61 = x33 * x59 + x35 + x60;
    const scalar_t x62 = x40 * x58;
    const scalar_t x63 = x1 * x62 + x44;
    const scalar_t x64 = x46 * x62 + x49;
    const scalar_t x65 = x51 * x58;
    const scalar_t x66 = x33 * x65 + x60;
    element_matrix_diag[0] += x1 * x28 + x32 + x39 + x45;
    element_matrix_diag[1] += x28 * x46 + x39 + x47 + x50;
    element_matrix_diag[2] += x46 * x52 + x50 + x55 + x56;
    element_matrix_diag[3] += x1 * x52 + x45 + x56 + x57;
    element_matrix_diag[4] += x1 * x59 + x32 + x61 + x63;
    element_matrix_diag[5] += x46 * x59 + x47 + x61 + x64;
    element_matrix_diag[6] += x46 * x65 + x55 + x64 + x66;
    element_matrix_diag[7] += x1 * x65 + x57 + x63 + x66;
}

int hex8_apply_mass(const ptrdiff_t nelements,
                    const ptrdiff_t nnodes,
                    idx_t **const SFEM_RESTRICT elements,
                    geom_t **const SFEM_RESTRICT points,
                    const ptrdiff_t stride_u,
                    const real_t *const SFEM_RESTRICT u,
                    const ptrdiff_t stride_values,
                    real_t *const SFEM_RESTRICT values) {
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    int n_qp = q27_n;
    const scalar_t *qx = q27_x;
    const scalar_t *qy = q27_y;
    const scalar_t *qz = q27_z;
    const scalar_t *qw = q27_w;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[8];
        accumulator_t element_vector[8] = {0};
        scalar_t element_u[8];

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 8; ++v) {
            element_u[v] = u[ev[v] * stride_u];
        }

        const scalar_t lx[8] = {
                x[ev[0]], x[ev[1]], x[ev[2]], x[ev[3]], x[ev[4]], x[ev[5]], x[ev[6]], x[ev[7]]};

        const scalar_t ly[8] = {
                y[ev[0]], y[ev[1]], y[ev[2]], y[ev[3]], y[ev[4]], y[ev[5]], y[ev[6]], y[ev[7]]};

        const scalar_t lz[8] = {
                z[ev[0]], z[ev[1]], z[ev[2]], z[ev[3]], z[ev[4]], z[ev[5]], z[ev[6]], z[ev[7]]};

        for (int k = 0; k < n_qp; k++) {
            hex8_mass_apply_points(
                    lx, ly, lz, qx[k], qy[k], qz[k], qw[k], element_u, element_vector);
        }

        for (int edof_i = 0; edof_i < 8; ++edof_i) {
            const idx_t dof_i = ev[edof_i] * stride_values;

#pragma omp atomic update
            values[dof_i] += element_vector[edof_i];
        }
    }

    return SFEM_SUCCESS;
}

int hex8_assemble_lumped_mass(const ptrdiff_t nelements,
                              const ptrdiff_t nnodes,
                              idx_t **const SFEM_RESTRICT elements,
                              geom_t **const SFEM_RESTRICT points,
                              const ptrdiff_t stride_values,
                              real_t *const SFEM_RESTRICT values) {
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    int n_qp = q27_n;
    const scalar_t *qx = q27_x;
    const scalar_t *qy = q27_y;
    const scalar_t *qz = q27_z;
    const scalar_t *qw = q27_w;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[8];
        accumulator_t element_vector[8] = {0};

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i];
        }

        const scalar_t lx[8] = {
                x[ev[0]], x[ev[1]], x[ev[2]], x[ev[3]], x[ev[4]], x[ev[5]], x[ev[6]], x[ev[7]]};

        const scalar_t ly[8] = {
                y[ev[0]], y[ev[1]], y[ev[2]], y[ev[3]], y[ev[4]], y[ev[5]], y[ev[6]], y[ev[7]]};

        const scalar_t lz[8] = {
                z[ev[0]], z[ev[1]], z[ev[2]], z[ev[3]], z[ev[4]], z[ev[5]], z[ev[6]], z[ev[7]]};

        for (int k = 0; k < n_qp; k++) {
            hex8_lumped_mass_points(lx, ly, lz, qx[k], qy[k], qz[k], qw[k], element_vector);
        }

        for (int edof_i = 0; edof_i < 8; ++edof_i) {
            const ptrdiff_t dof_i = ev[edof_i] * stride_values;
#pragma omp atomic update
            values[dof_i] += element_vector[edof_i];
        }
    }

    return SFEM_SUCCESS;
}
