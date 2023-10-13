#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "matrix.io/matrixio_ndarray.h"

#include "crs_graph.h"
#include "sfem_base.h"

#include "isotropic_phasefield_for_fracture.h"

#include "read_mesh.h"
#include "sfem_defs.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// TRI3 6th order quadrature rule
static real_t qw[12] = {0.050844906370206816920936809106869,
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
                        0.082851075618373575193553456420442};

static real_t qx[12] = {0.063089014491502228340331602870819,
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
                        0.63650249912139864723014259441205};

static real_t qy[12] = {0.063089014491502228340331602870819,
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
                        0.31035245103378440541660773395655};

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
    const real_t z) {
    int outside = (x < xmin) | (x > xmax) | (y < ymin) | (y > ymax) | (z < zmin) | (x > zmax);
    return !outside;
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
    real_t *const SFEM_RESTRICT out_x,
    real_t *const SFEM_RESTRICT out_y,
    real_t *const SFEM_RESTRICT out_z) {
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
    real_t *const SFEM_RESTRICT f) {
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
    const ptrdiff_t *const SFEM_RESTRICT stride,
    const ptrdiff_t i,
    const ptrdiff_t j,
    const ptrdiff_t k,
    // Attention this is geometric data transformed to solver data!
    const geom_t *const SFEM_RESTRICT data,
    real_t *const SFEM_RESTRICT out) {
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
    real_t *const SFEM_RESTRICT gx,
    real_t *const SFEM_RESTRICT gy,
    real_t *const SFEM_RESTRICT gz) {
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

int resample_gap(MPI_Comm comm,
                 // Mesh
                 const enum ElemType element_type,
                 const ptrdiff_t nelements,
                 const ptrdiff_t nnodes,
                 idx_t **const SFEM_RESTRICT elems,
                 geom_t **const SFEM_RESTRICT xyz,
                 // SDF
                 const ptrdiff_t *const SFEM_RESTRICT nlocal,
                 const ptrdiff_t *const SFEM_RESTRICT nglobal,
                 const ptrdiff_t *const SFEM_RESTRICT stride,
                 const geom_t *const SFEM_RESTRICT origin,
                 const geom_t *const SFEM_RESTRICT delta,
                 const geom_t *const SFEM_RESTRICT data,
                 // Output
                 real_t *const SFEM_RESTRICT g,
                 real_t *const SFEM_RESTRICT xnormal,
                 real_t *const SFEM_RESTRICT ynormal,
                 real_t *const SFEM_RESTRICT znormal) {
    assert(element_type == TRI3);  // only triangles supported for now

    memset(g, 0, nnodes * sizeof(real_t));
    memset(xnormal, 0, nnodes * sizeof(real_t));
    memset(ynormal, 0, nnodes * sizeof(real_t));
    memset(znormal, 0, nnodes * sizeof(real_t));

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

            // TODO
            real_t det_jac = 0;
            real_t dV = det_jac * qw[q] / 2;

            const ptrdiff_t i = (g_qx - origin[0]) / delta[0];
            const ptrdiff_t j = (g_qy - origin[1]) / delta[1];
            const ptrdiff_t k = (g_qz - origin[2]) / delta[2];

            const real_t xmin = origin[0] + i * delta[0];
            const real_t xmax = origin[0] + (i + 1) * delta[0];

            const real_t ymin = origin[1] + j * delta[1];
            const real_t ymax = origin[1] + (j + 1) * delta[1];

            const real_t zmin = origin[2] + k * delta[2];
            const real_t zmax = origin[2] + (k + 1) * delta[2];

            const real_t l_x = (g_qx - xmin) / (xmax - xmin);
            const real_t l_y = (g_qy - ymin) / (ymax - ymin);
            const real_t l_z = (g_qz - zmin) / (zmax - zmin);

            hex_aa_8_eval_fun(l_x, l_y, l_z, hex8_f);
            hex_aa_8_eval_grad(l_x, l_y, l_z, hex8_grad_x, hex8_grad_y, hex8_grad_z);
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
                    real_t denom = sqrt(eval_xnormal * eval_xnormal + eval_ynormal * eval_ynormal +
                                        eval_znormal * eval_znormal);
                    eval_xnormal /= denom;
                    eval_ynormal /= denom;
                    eval_znormal /= denom;
                }

#pragma unroll(8)
                for (int edof_i = 0; edof_i < 8; edof_i++) {
                    element_xnormal[edof_i] += eval_xnormal * tri3_f[edof_i] * dV;
                    element_ynormal[edof_i] += eval_ynormal * tri3_f[edof_i] * dV;
                    element_znormal[edof_i] += eval_znormal * tri3_f[edof_i] * dV;
                }
            }
        }

#pragma unroll(3)
        for (int v = 0; v < 3; ++v) {
            g[ev[v]] += element_gap[v];
            xnormal[ev[v]] += element_xnormal[v];
            ynormal[ev[v]] += element_ynormal[v];
            znormal[ev[v]] += element_znormal[v];
        }
    }

    // TODO for Shell elements!
    // apply_inv_lumped_mass(element_type, nelements, nnodes, elems, xyz, g, g);    
    return 0;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size != 1) {
        fprintf(stderr, "Parallel execution not supported!\n");
        return EXIT_FAILURE;
    }

    if (argc != 13) {
        fprintf(stderr,
                "usage: %s <folder> <nx> <ny> <nz> <ox> <oy> <oz> <dx> <dy> <dz> "
                "<data.float32.raw> <output_folder>\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    const char *folder = argv[1];
    ptrdiff_t nglobal[3] = {atol(argv[2]), atol(argv[3]), atol(argv[4])};
    geom_t origin[3] = {atof(argv[5]), atof(argv[6]), atof(argv[7])};
    geom_t delta[3] = {atof(argv[8]), atof(argv[9]), atof(argv[10])};
    const char *data_path = argv[11];
    const char *output_folder = argv[12];

    struct stat st = {0};
    if (stat(output_folder, &st) == -1) {
        mkdir(output_folder, 0700);
    }

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    ptrdiff_t n = nglobal[0] * nglobal[1] * nglobal[2];
    geom_t *sdf = (geom_t *)malloc(n * sizeof(geom_t));
    ptrdiff_t nlocal[3];

    if (ndarray_read(comm, data_path, SFEM_MPI_GEOM_T, 3, sdf, nlocal, nglobal)) {
        return EXIT_FAILURE;
    }

    // X is contiguous
    ptrdiff_t stride[3] = {1, nlocal[0], nlocal[0] * nlocal[1]};

    real_t *g = malloc(mesh.nnodes * sizeof(real_t));
    real_t *xnormal = malloc(mesh.nnodes * sizeof(real_t));
    real_t *ynormal = malloc(mesh.nnodes * sizeof(real_t));
    real_t *znormal = malloc(mesh.nnodes * sizeof(real_t));

    resample_gap(comm,
                 // Mesh
                 mesh.element_type,
                 mesh.nelements,
                 mesh.nnodes,
                 mesh.elements,
                 mesh.points,
                 // SDF
                 nlocal,
                 nglobal,
                 stride,
                 origin,
                 delta,
                 sdf,
                 // Output
                 g,
                 xnormal,
                 ynormal,
                 znormal);

    // Free resources
    free(sdf);
    free(g);
    free(xnormal);
    free(ynormal);
    free(znormal);
    mesh_destroy(&mesh);
    return MPI_Finalize();
}
