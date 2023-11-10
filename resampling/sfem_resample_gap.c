#include "sfem_resample_gap.h"

#include "mass.h"
#include "read_mesh.h"

#include "matrixio_array.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

static SFEM_INLINE real_t put_inside(const real_t v) { return MIN(MAX(1e-7, v), 1 - 1e-7); }

// TRI3 6th order quadrature rule
static real_t qw[12] = { 0.050844906370206816920936809106869,
    0.050844906370206816920936809106869,
    0.050844906370206816920936809106869,
    0.11678627572637936602528961138558,
    0.11678627572637936602528961138558,
    0.11678627572637936602528961138558,
    0.082851075618373575193553456420442,
    0.082851075618373575193553456420442,
    0.082851075618373575193553456420442,
    0.082851075618373575193553456420442,
    0.082851075618373575193553456420442,
    0.082851075618373575193553456420442 };

static real_t qx[12] = { 0.063089014491502228340331602870819,
    0.063089014491502228340331602870819,
    0.87382197101699554331933679425836,
    0.24928674517091042129163855310702,
    0.24928674517091042129163855310702,
    0.50142650965817915741672289378596,
    0.053145049844816947353249671631398,
    0.053145049844816947353249671631398,
    0.31035245103378440541660773395655,
    0.31035245103378440541660773395655,
    0.63650249912139864723014259441205,
    0.63650249912139864723014259441205 };

static real_t qy[12] = { 0.063089014491502228340331602870819,
    0.87382197101699554331933679425836,
    0.063089014491502228340331602870819,
    0.24928674517091042129163855310702,
    0.50142650965817915741672289378596,
    0.24928674517091042129163855310702,
    0.31035245103378440541660773395655,
    0.63650249912139864723014259441205,
    0.053145049844816947353249671631398,
    0.63650249912139864723014259441205,
    0.053145049844816947353249671631398,
    0.31035245103378440541660773395655 };

SFEM_INLINE static int hex_aa_8_contains(
    // X-coordinates
    const real_t xmin,
    const real_t xmax,
    // Y-coordinates
    const real_t ymin,
    const real_t ymax,
    // Z-coordinates
    const real_t zmin,
    const real_t zmax,
    const real_t x,
    const real_t y,
    const real_t z)
{
    int outside = (x < xmin) | (x > xmax) | (y < ymin) | (y > ymax) | (z < zmin) | (x > zmax);
    return !outside;
}

SFEM_INLINE static real_t tri_shell_3_measure(
    // X-coordinates
    const real_t px0,
    const real_t px1,
    const real_t px2,
    // Y-coordinates
    const real_t py0,
    const real_t py1,
    const real_t py2,
    // Z-coordinates
    const real_t pz0,
    const real_t pz1,
    const real_t pz2)
{
    const real_t x0 = -px0 + px1;
    const real_t x1 = -px0 + px2;
    const real_t x2 = -py0 + py1;
    const real_t x3 = -py0 + py2;
    const real_t x4 = -pz0 + pz1;
    const real_t x5 = -pz0 + pz2;
    return (1.0 / 2.0) * sqrt((pow(x0, 2) + pow(x2, 2) + pow(x4, 2)) * (pow(x1, 2) + pow(x3, 2) + pow(x5, 2)) - pow(x0 * x1 + x2 * x3 + x4 * x5, 2));
}

SFEM_INLINE static void tri_shell_3_transform(
    // X-coordinates
    const real_t x0,
    const real_t x1,
    const real_t x2,
    // Y-coordinates
    const real_t y0,
    const real_t y1,
    const real_t y2,
    // Z-coordinates
    const real_t z0,
    const real_t z1,
    const real_t z2,
    // Quadrature point
    const real_t x,
    const real_t y,
    // Output
    real_t* const SFEM_RESTRICT out_x,
    real_t* const SFEM_RESTRICT out_y,
    real_t* const SFEM_RESTRICT out_z)
{
    const real_t phi0 = 1 - x - y;
    const real_t phi1 = x;
    const real_t phi2 = y;

    *out_x = phi0 * x0 + phi1 * x1 + phi2 * x2;
    *out_y = phi0 * y0 + phi1 * y1 + phi2 * y2;
    *out_z = phi0 * z0 + phi1 * z1 + phi2 * z2;
}

SFEM_INLINE static void hex_aa_8_eval_fun(
    // Quadrature point (local coordinates)
    const real_t x,
    const real_t y,
    const real_t z,
    // Output
    real_t* const SFEM_RESTRICT f)
{
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
    const geom_t* const SFEM_RESTRICT data,
    real_t* const SFEM_RESTRICT out)
{
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
    real_t* const SFEM_RESTRICT gz)
{
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

int resample_gap(
    // Mesh
    const enum ElemType element_type,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t** const SFEM_RESTRICT elems,
    geom_t** const SFEM_RESTRICT xyz,
    // SDF
    const ptrdiff_t* const SFEM_RESTRICT n,
    const ptrdiff_t* const SFEM_RESTRICT stride,
    const geom_t* const SFEM_RESTRICT origin,
    const geom_t* const SFEM_RESTRICT delta,
    const geom_t* const SFEM_RESTRICT data,
    // Output
    real_t* const SFEM_RESTRICT g,
    real_t* const SFEM_RESTRICT xnormal,
    real_t* const SFEM_RESTRICT ynormal,
    real_t* const SFEM_RESTRICT znormal)
{
    assert(element_type == TRI3 || element_type == TRISHELL3); // only triangles supported for now
    if(element_type != TRI3 && element_type != TRISHELL3) {
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    memset(g, 0, nnodes * sizeof(real_t));
    memset(xnormal, 0, nnodes * sizeof(real_t));
    memset(ynormal, 0, nnodes * sizeof(real_t));
    memset(znormal, 0, nnodes * sizeof(real_t));

    real_t* wg = calloc(nnodes, sizeof(real_t));

    const real_t ox = (real_t)origin[0];
    const real_t oy = (real_t)origin[1];
    const real_t oz = (real_t)origin[2];

    const real_t dx = (real_t)delta[0];
    const real_t dy = (real_t)delta[1];
    const real_t dz = (real_t)delta[2];

#pragma omp parallel
    {
#pragma omp for // nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[3];
            geom_t x[3], y[3], z[3];

            real_t hex8_f[8];
            real_t hex8_grad_x[8];
            real_t hex8_grad_y[8];
            real_t hex8_grad_z[8];
            real_t coeffs[8];

            real_t tri3_f[3];
            real_t element_gap[3];
            real_t element_xnormal[3];
            real_t element_ynormal[3];
            real_t element_znormal[3];

#pragma unroll(3)
            for (int v = 0; v < 3; ++v) {
                ev[v] = elems[v][i];
            }

            for (int v = 0; v < 3; ++v) {
                x[v] = xyz[0][ev[v]];
                y[v] = xyz[1][ev[v]];
                z[v] = xyz[2][ev[v]];
            }

            memset(element_gap, 0, 3 * sizeof(real_t));
            memset(element_xnormal, 0, 3 * sizeof(real_t));
            memset(element_ynormal, 0, 3 * sizeof(real_t));
            memset(element_znormal, 0, 3 * sizeof(real_t));

            const real_t measure = tri_shell_3_measure(x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2]);

            for (int q = 0; q < 12; q++) {
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
                    qx[q],
                    qy[q],
                    &g_qx,
                    &g_qy,
                    &g_qz);

                tri3_f[0] = 1 - qx[q] - qy[q];
                tri3_f[1] = qx[q];
                tri3_f[2] = qy[q];

                const real_t dV = measure * qw[q];

                const real_t grid_x = (g_qx - ox) / dx;
                const real_t grid_y = (g_qy - oy) / dy;
                const real_t grid_z = (g_qz - oz) / dz;

                const ptrdiff_t i = floor(grid_x);
                const ptrdiff_t j = floor(grid_y);
                const ptrdiff_t k = floor(grid_z);

                // If outside
                if (i < 0 || j < 0 || k < 0 || (i + 1 >= n[0]) || (j + 1 >= n[1]) || (k + 1 >= n[2])) {
                    fprintf(stderr,
                        "warning (%ld, %ld, %ld) outside domain  (%ld, %ld, %ld)!\n",
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
                hex_aa_8_eval_grad(put_inside(l_x),
                    put_inside(l_y),
                    put_inside(l_z),
                    hex8_grad_x,
                    hex8_grad_y,
                    hex8_grad_z);
                hex_aa_8_collect_coeffs(stride, i, j, k, data, coeffs);

                // Integrate gap function
                {
                    real_t eval_gap = 0;

#pragma unroll(8)
                    for (int edof_j = 0; edof_j < 8; edof_j++) {
                        eval_gap += hex8_f[edof_j] * coeffs[edof_j];
                    }

#pragma unroll(3)
                    for (int edof_i = 0; edof_i < 3; edof_i++) {
                        element_gap[edof_i] += eval_gap * tri3_f[edof_i] * dV;
                    }
                }

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

#pragma unroll(3)
                    for (int edof_i = 0; edof_i < 3; edof_i++) {
                        element_xnormal[edof_i] += eval_xnormal * tri3_f[edof_i] * dV;
                        element_ynormal[edof_i] += eval_ynormal * tri3_f[edof_i] * dV;
                        element_znormal[edof_i] += eval_znormal * tri3_f[edof_i] * dV;
                    }
                }
            }

#pragma unroll(3)
            for (int v = 0; v < 3; ++v) {
                // Invert sign since distance field is negative insdide and positive outside
#pragma omp critical
                {
                    wg[ev[v]] -= element_gap[v];
                    xnormal[ev[v]] += element_xnormal[v];
                    ynormal[ev[v]] += element_ynormal[v];
                    znormal[ev[v]] += element_znormal[v];
                }
            }
        }
    }

    apply_inv_lumped_mass(TRISHELL3, nelements, nnodes, elems, xyz, wg, g);

    // Normalize!
    for (ptrdiff_t i = 0; i < nnodes; i++) {
        real_t denom = sqrt(xnormal[i] * xnormal[i] + ynormal[i] * ynormal[i] + znormal[i] * znormal[i]);
        xnormal[i] /= denom;
        ynormal[i] /= denom;
        znormal[i] /= denom;
    }

    free(wg);

    return 0;
}

int interpolate_gap(const ptrdiff_t nnodes,
    geom_t** const SFEM_RESTRICT xyz,
    // SDF
    const ptrdiff_t* const SFEM_RESTRICT n,
    const ptrdiff_t* const SFEM_RESTRICT stride,
    const geom_t* const SFEM_RESTRICT origin,
    const geom_t* const SFEM_RESTRICT delta,
    const geom_t* const SFEM_RESTRICT data,
    // Output
    real_t* const SFEM_RESTRICT g,
    real_t* const SFEM_RESTRICT xnormal,
    real_t* const SFEM_RESTRICT ynormal,
    real_t* const SFEM_RESTRICT znormal)
{

    const real_t ox = (real_t)origin[0];
    const real_t oy = (real_t)origin[1];
    const real_t oz = (real_t)origin[2];

    const real_t dx = (real_t)delta[0];
    const real_t dy = (real_t)delta[1];
    const real_t dz = (real_t)delta[2];

#pragma omp parallel
    {
#pragma omp for // nowait
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
                    "[%d] warning (%ld, %ld, %ld) outside domain  (%ld, %ld, %ld)!\n",
                    rank,
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
            hex_aa_8_eval_grad(put_inside(l_x),
                put_inside(l_y),
                put_inside(l_z),
                hex8_grad_x,
                hex8_grad_y,
                hex8_grad_z);
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

            // Interpolate gap function
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
    }

    return 0;
}

SFEM_INLINE static void minmax(
    const ptrdiff_t n,
    const geom_t* const SFEM_RESTRICT x, geom_t* xmin, geom_t* xmax)
{
    *xmin = x[0];
    *xmax = x[0];
    for (ptrdiff_t i = 1; i < n; i++) {
        *xmin = MIN(*xmin, x[i]);
        *xmax = MAX(*xmax, x[i]);
    }
}

int sdf_view(
    MPI_Comm comm,
    const ptrdiff_t nnodes,
    const geom_t* SFEM_RESTRICT z_coordinate,
    const ptrdiff_t* const nlocal,
    const ptrdiff_t* const SFEM_RESTRICT nglobal,
    const ptrdiff_t* const SFEM_RESTRICT stride,
    const geom_t* const origin,
    const geom_t* const SFEM_RESTRICT delta,
    const geom_t* const sdf,
    geom_t** sdf_out,
    ptrdiff_t* z_nlocal_out,
    geom_t* const SFEM_RESTRICT z_origin_out)
{
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

    double sdf_view_tick = MPI_Wtime();

    geom_t zmin, zmax;
    minmax(nnodes, z_coordinate, &zmin, &zmax);

    // Z is distributed
    ptrdiff_t zoffset = 0;
    MPI_Exscan(&nlocal[2], &zoffset, 1, MPI_LONG, MPI_SUM, comm);

    // // Compute Local z-tile
    ptrdiff_t sdf_start = (zmin - origin[2]) / delta[2];
    ptrdiff_t sdf_end = (zmax - origin[2]) / delta[2];

    // Make sure we are inside the grid
    sdf_start = MAX(0, sdf_start);
    sdf_end = MIN(nglobal[2], sdf_end + 1 + 1); // 1 for the rightside of the cell 1 for the exclusive range

    ptrdiff_t pnlocal_z = (sdf_end - sdf_start);
    geom_t* psdf = malloc(pnlocal_z * stride[2] * sizeof(geom_t));

    array_range_select(comm, SFEM_MPI_GEOM_T, 
        (void*)sdf, 
        (void*)psdf,
        // Size of z-slice
        nlocal[2] * stride[2],
        // starting offset
        sdf_start * stride[2],
        // ending offset
        sdf_end * stride[2]);

    *sdf_out = psdf;
    *z_nlocal_out = pnlocal_z;
    *z_origin_out = origin[2] + sdf_start * delta[2];

    double sdf_view_tock = MPI_Wtime();

    if (!rank) {
        printf("[%d] sdf_view %g (seconds)\n", rank, sdf_view_tock - sdf_view_tick);
    }

    return 0;
}
