#include "node_interpolate.h"

#include <math.h>
#include <stdio.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

static SFEM_INLINE real_t put_inside(const real_t v) { return MIN(MAX(1e-7, v), 1 - 1e-7); }

SFEM_INLINE static void hex_aa_8_eval_fun(
        // Quadrature point (local coordinates)
        const real_t x,
        const real_t y,
        const real_t z,
        // Output
        real_t* const SFEM_RESTRICT f) {
    f[0] = (1.0 - x) * (1.0 - y) * (1.0 - z);
    f[1] = x * (1.0 - y) * (1.0 - z);
    f[2] = x * y * (1.0 - z);
    f[3] = (1.0 - x) * y * (1.0 - z);
    f[4] = (1.0 - x) * (1.0 - y) * z;
    f[5] = x * (1.0 - y) * z;
    f[6] = x * y * z;
    f[7] = (1.0 - x) * y * z;
}

SFEM_INLINE static void hex_aa_8_collect_coeffs(const ptrdiff_t* const SFEM_RESTRICT stride,
                                                const ptrdiff_t                      i,
                                                const ptrdiff_t                      j,
                                                const ptrdiff_t                      k,
                                                // Attention this is geometric data transformed to solver data!
                                                const geom_t* const SFEM_RESTRICT data,
                                                real_t* const SFEM_RESTRICT       out) {
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
        const real_t x,
        const real_t y,
        const real_t z,
        // Output
        real_t* const SFEM_RESTRICT gx,
        real_t* const SFEM_RESTRICT gy,
        real_t* const SFEM_RESTRICT gz) {
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

int interpolate_gap(const ptrdiff_t              nnodes,
                    geom_t** const SFEM_RESTRICT xyz,
                    // SDF
                    const ptrdiff_t* const SFEM_RESTRICT n,
                    const ptrdiff_t* const SFEM_RESTRICT stride,
                    const geom_t* const SFEM_RESTRICT    origin,
                    const geom_t* const SFEM_RESTRICT    delta,
                    const geom_t* const SFEM_RESTRICT    data,
                    // Output
                    real_t* const SFEM_RESTRICT g,
                    real_t* const SFEM_RESTRICT xnormal,
                    real_t* const SFEM_RESTRICT ynormal,
                    real_t* const SFEM_RESTRICT znormal) {
    if (!nnodes) return 0;

    const real_t ox = (real_t)origin[0];
    const real_t oy = (real_t)origin[1];
    const real_t oz = (real_t)origin[2];

    const real_t dx = (real_t)delta[0];
    const real_t dy = (real_t)delta[1];
    const real_t dz = (real_t)delta[2];

#pragma omp parallel for
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
            SFEM_ERROR(
                    "(%g, %g, %g) (%ld, %ld, %ld) outside domain  (%ld, %ld, "
                    "%ld)!\n",
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
        hex_aa_8_eval_grad(put_inside(l_x), put_inside(l_y), put_inside(l_z), hex8_grad_x, hex8_grad_y, hex8_grad_z);

        hex_aa_8_collect_coeffs(stride, i, j, k, data, coeffs);

        // Interpolate gap function
        {
            real_t eval_gap = 0;

#pragma unroll(8)
            for (int edof_j = 0; edof_j < 8; edof_j++) {
                eval_gap += hex8_f[edof_j] * coeffs[edof_j];
            }

            g[node] = -eval_gap;
        }

        // Interpolate normals
        {
            real_t eval_xnormal = 0;
            real_t eval_ynormal = 0;
            real_t eval_znormal = 0;

#pragma unroll(8)
            for (int edof_j = 0; edof_j < 8; edof_j++) {
                eval_xnormal += hex8_grad_x[edof_j] * coeffs[edof_j];
                eval_ynormal += hex8_grad_y[edof_j] * coeffs[edof_j];
                eval_znormal += hex8_grad_z[edof_j] * coeffs[edof_j];
            }

            {
                // Normalize
                real_t denom = sqrt(eval_xnormal * eval_xnormal + eval_ynormal * eval_ynormal + eval_znormal * eval_znormal);

                assert(denom != 0);

                eval_xnormal /= denom;
                eval_ynormal /= denom;
                eval_znormal /= denom;
            }

            xnormal[node] = eval_xnormal;
            ynormal[node] = eval_ynormal;
            znormal[node] = eval_znormal;
        }
    }

    return 0;
}

int interpolate_gap_normals(const ptrdiff_t              nnodes,
                            geom_t** const SFEM_RESTRICT xyz,
                            // SDF
                            const ptrdiff_t* const SFEM_RESTRICT n,
                            const ptrdiff_t* const SFEM_RESTRICT stride,
                            const geom_t* const SFEM_RESTRICT    origin,
                            const geom_t* const SFEM_RESTRICT    delta,
                            const geom_t* const SFEM_RESTRICT    data,
                            // Output
                            real_t* const SFEM_RESTRICT xnormal,
                            real_t* const SFEM_RESTRICT ynormal,
                            real_t* const SFEM_RESTRICT znormal) {
    if (!nnodes) return 0;

    const real_t ox = (real_t)origin[0];
    const real_t oy = (real_t)origin[1];
    const real_t oz = (real_t)origin[2];

    const real_t dx = (real_t)delta[0];
    const real_t dy = (real_t)delta[1];
    const real_t dz = (real_t)delta[2];

#pragma omp parallel for  // nowait
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
            SFEM_ERROR(
                    "(%g, %g, %g) (%ld, %ld, %ld) outside domain  (%ld, %ld, "
                    "%ld)!\n",
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
        hex_aa_8_eval_grad(put_inside(l_x), put_inside(l_y), put_inside(l_z), hex8_grad_x, hex8_grad_y, hex8_grad_z);

        hex_aa_8_collect_coeffs(stride, i, j, k, data, coeffs);

        // Interpolate normals
        {
            real_t eval_xnormal = 0;
            real_t eval_ynormal = 0;
            real_t eval_znormal = 0;

#pragma unroll(8)
            for (int edof_j = 0; edof_j < 8; edof_j++) {
                eval_xnormal += hex8_grad_x[edof_j] * coeffs[edof_j];
                eval_ynormal += hex8_grad_y[edof_j] * coeffs[edof_j];
                eval_znormal += hex8_grad_z[edof_j] * coeffs[edof_j];
            }

            {
                // Normalize
                real_t denom = sqrt(eval_xnormal * eval_xnormal + eval_ynormal * eval_ynormal + eval_znormal * eval_znormal);

                assert(denom != 0);

                eval_xnormal /= denom;
                eval_ynormal /= denom;
                eval_znormal /= denom;
            }

            xnormal[node] = eval_xnormal;
            ynormal[node] = eval_ynormal;
            znormal[node] = eval_znormal;
        }
    }

    return SFEM_SUCCESS;
}

int interpolate_gap_value(const ptrdiff_t              nnodes,
                          geom_t** const SFEM_RESTRICT xyz,
                          // SDF
                          const ptrdiff_t* const SFEM_RESTRICT n,
                          const ptrdiff_t* const SFEM_RESTRICT stride,
                          const geom_t* const SFEM_RESTRICT    origin,
                          const geom_t* const SFEM_RESTRICT    delta,
                          const geom_t* const SFEM_RESTRICT    data,
                          // Output
                          real_t* const SFEM_RESTRICT g) {
    if (!nnodes) return 0;

    const real_t ox = (real_t)origin[0];
    const real_t oy = (real_t)origin[1];
    const real_t oz = (real_t)origin[2];

    const real_t dx = (real_t)delta[0];
    const real_t dy = (real_t)delta[1];
    const real_t dz = (real_t)delta[2];

#pragma omp parallel for
    for (ptrdiff_t node = 0; node < nnodes; ++node) {
        real_t hex8_f[8];
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
            SFEM_ERROR("(%g, %g, %g) (%ld, %ld, %ld) outside domain  (%ld, %ld, %ld)!\n", x, y, z, i, j, k, n[0], n[1], n[2]);
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
            real_t eval_gap = 0;

#pragma unroll(8)
            for (int edof_j = 0; edof_j < 8; edof_j++) {
                eval_gap += hex8_f[edof_j] * coeffs[edof_j];
            }

            g[node] = -eval_gap;
        }
    }

    return SFEM_SUCCESS;
}
