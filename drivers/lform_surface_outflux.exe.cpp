#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "array_dtof.h"
#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "utils.h"

#include "crs_graph.h"
#include "sfem_base.h"

#include "read_mesh.h"

#include "tet4_grad.h"

#include "sfem_defs.h"
#include "sfem_macros.h"

#include "sfem_API.hpp"

static SFEM_INLINE void normalize(real_t *const vec3) {
    const real_t len = sqrt(vec3[0] * vec3[0] + vec3[1] * vec3[1] + vec3[2] * vec3[2]);
    vec3[0] /= len;
    vec3[1] /= len;
    vec3[2] /= len;
}

void normals(const ptrdiff_t nelements,
             const ptrdiff_t nnodes,
             idx_t **const SFEM_RESTRICT elems,
             geom_t **const SFEM_RESTRICT xyz,
             geom_t **const SFEM_RESTRICT normals_xyz) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

    idx_t ev[4];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(4)
        for (int v = 0; v < 3; ++v) {
            ev[v] = elems[v][i];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];

        real_t u[3] = {xyz[0][i1] - xyz[0][i0], xyz[1][i1] - xyz[1][i0], xyz[2][i1] - xyz[2][i0]};
        real_t v[3] = {xyz[0][i2] - xyz[0][i0], xyz[1][i2] - xyz[1][i0], xyz[2][i2] - xyz[2][i0]};

        normalize(u);
        normalize(v);

        real_t n[3] = {u[1] * v[2] - u[2] * v[1],  //
                       u[2] * v[0] - u[0] * v[2],  //
                       u[0] * v[1] - u[1] * v[0]};

        normalize(n);

        normals_xyz[0][i] = n[0];
        normals_xyz[1][i] = n[1];
        normals_xyz[2][i] = n[2];
    }

    double tock = MPI_Wtime();
    printf("lform_surface_outflux.c: normals\t%g seconds\n", tock - tick);
}

static SFEM_INLINE void p1_p1_surface_outflux_kernel(const real_t px0,
                                                     const real_t px1,
                                                     const real_t px2,
                                                     const real_t py0,
                                                     const real_t py1,
                                                     const real_t py2,
                                                     const real_t pz0,
                                                     const real_t pz1,
                                                     const real_t pz2,
                                                     // Normals
                                                     const real_t nx,
                                                     const real_t ny,
                                                     const real_t nz,
                                                     // Data
                                                     const real_t *const SFEM_RESTRICT vx,
                                                     const real_t *const SFEM_RESTRICT vy,
                                                     const real_t *const SFEM_RESTRICT vz,
                                                     // Output
                                                     real_t *const SFEM_RESTRICT element_vector) {
    // FLOATING POINT OPS!
    //      - Result: 3*ADD + 3*ASSIGNMENT + 9*MUL
    //      - Subexpressions: 41*ADD + 3*DIV + 99*MUL + 10*POW + 27*SUB
    const real_t x0 = 2 * px0;
    const real_t x1 = px1 * x0;
    const real_t x2 = py0 * x1;
    const real_t x3 = py1 * py2;
    const real_t x4 = pz0 * pz1;
    const real_t x5 = pz0 * pz2;
    const real_t x6 = pz1 * pz2;
    const real_t x7 = px2 * x0;
    const real_t x8 = py0 * x7;
    const real_t x9 = px1 * px2;
    const real_t x10 = 2 * py0;
    const real_t x11 = py1 * x10;
    const real_t x12 = py2 * x10;
    const real_t x13 = 2 * x9;
    const real_t x14 = 2 * pz0;
    const real_t x15 = pz1 * x14;
    const real_t x16 = pz2 * x14;
    const real_t x17 = POW2(px0);
    const real_t x18 = POW2(py1);
    const real_t x19 = POW2(py2);
    const real_t x20 = POW2(pz1);
    const real_t x21 = POW2(pz2);
    const real_t x22 = POW2(px1);
    const real_t x23 = POW2(py0);
    const real_t x24 = POW2(pz0);
    const real_t x25 = POW2(px2);
    const real_t x26 = 2 * x17;
    const real_t x27 = 2 * x23;
    const real_t x28 = 2 * x24;
    const real_t x29 =
        sqrt(-py1 * x2 + py1 * x8 + py2 * x2 - py2 * x8 - x1 * x19 - x1 * x21 + x1 * x3 - x1 * x4 + x1 * x5 + x1 * x6 -
             x11 * x21 - x11 * x25 - x11 * x4 + x11 * x5 + x11 * x6 + x11 * x9 - x12 * x20 - x12 * x22 + x12 * x4 -
             x12 * x5 + x12 * x6 + x12 * x9 - x13 * x3 - x13 * x6 - x15 * x19 - x15 * x25 + x15 * x3 + x15 * x9 -
             x16 * x18 - x16 * x22 + x16 * x3 + x16 * x9 + x17 * x18 + x17 * x19 + x17 * x20 + x17 * x21 + x18 * x21 +
             x18 * x24 + x18 * x25 - x18 * x7 + x19 * x20 + x19 * x22 + x19 * x24 + x20 * x23 + x20 * x25 - x20 * x7 +
             x21 * x22 + x21 * x23 + x22 * x23 + x22 * x24 + x23 * x25 + x24 * x25 - x26 * x3 - x26 * x6 - x27 * x6 -
             x27 * x9 - x28 * x3 - x28 * x9 - 2 * x3 * x6 + x3 * x7 + x4 * x7 - x5 * x7 + x6 * x7);
    const real_t x30 = (1.0 / 12.0) * x29;
    const real_t x31 = nx * x30;
    const real_t x32 = ny * x30;
    const real_t x33 = nz * x30;
    const real_t x34 = (1.0 / 24.0) * x29;
    const real_t x35 = nx * x34;
    const real_t x36 = ny * x34;
    const real_t x37 = nz * x34;
    const real_t x38 = vx[2] * x35 + vy[2] * x36 + vz[2] * x37;
    const real_t x39 = vx[1] * x35 + vy[1] * x36 + vz[1] * x37;
    const real_t x40 = vx[0] * x35 + vy[0] * x36 + vz[0] * x37;
    element_vector[0] = vx[0] * x31 + vy[0] * x32 + vz[0] * x33 + x38 + x39;
    element_vector[1] = vx[1] * x31 + vy[1] * x32 + vz[1] * x33 + x38 + x40;
    element_vector[2] = vx[2] * x31 + vy[2] * x32 + vz[2] * x33 + x39 + x40;
}

void tri3_p1_p1_surface_outflux(const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const SFEM_RESTRICT elems,
                                geom_t **const SFEM_RESTRICT xyz,
                                geom_t **const SFEM_RESTRICT normals_xyz,
                                real_t *const SFEM_RESTRICT vector_field_x,
                                real_t *const SFEM_RESTRICT vector_field_y,
                                real_t *const SFEM_RESTRICT vector_field_z,
                                real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

    real_t element_vector_x[3];
    real_t element_vector_y[3];
    real_t element_vector_z[3];
    real_t element_vector[3];

    idx_t ev[3];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(3)
        for (int v = 0; v < 3; ++v) {
            ev[v] = elems[v][i];
        }

        // Global to local
        for (int v = 0; v < 3; ++v) {
            element_vector_x[v] = vector_field_x[ev[v]];
            element_vector_y[v] = vector_field_y[ev[v]];
            element_vector_z[v] = vector_field_z[ev[v]];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];

        p1_p1_surface_outflux_kernel(
            // X-coordinates
            xyz[0][i0],
            xyz[0][i1],
            xyz[0][i2],
            // Y-coordinates
            xyz[1][i0],
            xyz[1][i1],
            xyz[1][i2],
            // Z-coordinates
            xyz[2][i0],
            xyz[2][i1],
            xyz[2][i2],
            // Normal,
            normals_xyz[0][i],
            normals_xyz[1][i],
            normals_xyz[2][i],
            // Data
            element_vector_x,
            element_vector_y,
            element_vector_z,
            // Output
            element_vector);

        values[i0] += element_vector[0];
        values[i1] += element_vector[1];
        values[i2] += element_vector[2];
    }

    double tock = MPI_Wtime();
    printf("surface_outflux.c: tri3_surface_outflux\t%g seconds\n", tock - tick);
}

static SFEM_INLINE void tri6_p2_p2_surface_outflux_kernel(const real_t px0,
                                                          const real_t px1,
                                                          const real_t px2,
                                                          const real_t py0,
                                                          const real_t py1,
                                                          const real_t py2,
                                                          const real_t pz0,
                                                          const real_t pz1,
                                                          const real_t pz2,
                                                          // Normals
                                                          const real_t nx,
                                                          const real_t ny,
                                                          const real_t nz,
                                                          // Data
                                                          const real_t *const SFEM_RESTRICT vx,
                                                          const real_t *const SFEM_RESTRICT vy,
                                                          const real_t *const SFEM_RESTRICT vz,
                                                          // Output
                                                          real_t *const SFEM_RESTRICT element_vector) {
    // FLOATING POINT OPS!
    //       - Result: 6*ADD + 6*ASSIGNMENT + 42*MUL
    //       - Subexpressions: 47*ADD + 5*DIV + 116*MUL + 10*POW + 27*SUB
    const real_t x0 = 2 * px0;
    const real_t x1 = px1 * x0;
    const real_t x2 = py0 * x1;
    const real_t x3 = py1 * py2;
    const real_t x4 = pz0 * pz1;
    const real_t x5 = pz0 * pz2;
    const real_t x6 = pz1 * pz2;
    const real_t x7 = px2 * x0;
    const real_t x8 = py0 * x7;
    const real_t x9 = px1 * px2;
    const real_t x10 = 2 * py0;
    const real_t x11 = py1 * x10;
    const real_t x12 = py2 * x10;
    const real_t x13 = 2 * x9;
    const real_t x14 = 2 * pz0;
    const real_t x15 = pz1 * x14;
    const real_t x16 = pz2 * x14;
    const real_t x17 = POW2(px0);
    const real_t x18 = POW2(py1);
    const real_t x19 = POW2(py2);
    const real_t x20 = POW2(pz1);
    const real_t x21 = POW2(pz2);
    const real_t x22 = POW2(px1);
    const real_t x23 = POW2(py0);
    const real_t x24 = POW2(pz0);
    const real_t x25 = POW2(px2);
    const real_t x26 = 2 * x17;
    const real_t x27 = 2 * x23;
    const real_t x28 = 2 * x24;
    const real_t x29 =
        sqrt(-py1 * x2 + py1 * x8 + py2 * x2 - py2 * x8 - x1 * x19 - x1 * x21 + x1 * x3 - x1 * x4 + x1 * x5 + x1 * x6 -
             x11 * x21 - x11 * x25 - x11 * x4 + x11 * x5 + x11 * x6 + x11 * x9 - x12 * x20 - x12 * x22 + x12 * x4 -
             x12 * x5 + x12 * x6 + x12 * x9 - x13 * x3 - x13 * x6 - x15 * x19 - x15 * x25 + x15 * x3 + x15 * x9 -
             x16 * x18 - x16 * x22 + x16 * x3 + x16 * x9 + x17 * x18 + x17 * x19 + x17 * x20 + x17 * x21 + x18 * x21 +
             x18 * x24 + x18 * x25 - x18 * x7 + x19 * x20 + x19 * x22 + x19 * x24 + x20 * x23 + x20 * x25 - x20 * x7 +
             x21 * x22 + x21 * x23 + x22 * x23 + x22 * x24 + x23 * x25 + x24 * x25 - x26 * x3 - x26 * x6 - x27 * x6 -
             x27 * x9 - x28 * x3 - x28 * x9 - 2 * x3 * x6 + x3 * x7 + x4 * x7 - x5 * x7 + x6 * x7);
    const real_t x30 = (1.0 / 90.0) * x29;
    const real_t x31 = nx * x30;
    const real_t x32 = ny * x30;
    const real_t x33 = nz * x30;
    const real_t x34 = (1.0 / 360.0) * x29;
    const real_t x35 = nx * x34;
    const real_t x36 = ny * x34;
    const real_t x37 = nz * x34;
    const real_t x38 = vx[2] * x35 + vy[2] * x36 + vz[2] * x37;
    const real_t x39 = vx[1] * x35 + vy[1] * x36 + vz[1] * x37;
    const real_t x40 = vx[0] * x35 + vy[0] * x36 + vz[0] * x37;
    const real_t x41 = (4.0 / 45.0) * x29;
    const real_t x42 = nx * x41;
    const real_t x43 = ny * x41;
    const real_t x44 = nz * x41;
    const real_t x45 = (2.0 / 45.0) * x29;
    const real_t x46 = nx * x45;
    const real_t x47 = ny * x45;
    const real_t x48 = nz * x45;
    const real_t x49 = vx[5] * x46 + vy[5] * x47 + vz[5] * x48;
    const real_t x50 = vx[4] * x46 + vy[4] * x47 + vz[4] * x48;
    const real_t x51 = vx[3] * x46 + vy[3] * x47 + vz[3] * x48;
    element_vector[0] = (1.0 / 60.0) * nx * vx[0] * x29 + (1.0 / 60.0) * ny * vy[0] * x29 +
                        (1.0 / 60.0) * nz * vz[0] * x29 - vx[4] * x31 - vy[4] * x32 - vz[4] * x33 - x38 - x39;
    element_vector[1] = (1.0 / 60.0) * nx * vx[1] * x29 + (1.0 / 60.0) * ny * vy[1] * x29 +
                        (1.0 / 60.0) * nz * vz[1] * x29 - vx[5] * x31 - vy[5] * x32 - vz[5] * x33 - x38 - x40;
    element_vector[2] = (1.0 / 60.0) * nx * vx[2] * x29 + (1.0 / 60.0) * ny * vy[2] * x29 +
                        (1.0 / 60.0) * nz * vz[2] * x29 - vx[3] * x31 - vy[3] * x32 - vz[3] * x33 - x39 - x40;
    element_vector[3] = -vx[2] * x31 + vx[3] * x42 - vy[2] * x32 + vy[3] * x43 - vz[2] * x33 + vz[3] * x44 + x49 + x50;
    element_vector[4] = -vx[0] * x31 + vx[4] * x42 - vy[0] * x32 + vy[4] * x43 - vz[0] * x33 + vz[4] * x44 + x49 + x51;
    element_vector[5] = -vx[1] * x31 + vx[5] * x42 - vy[1] * x32 + vy[5] * x43 - vz[1] * x33 + vz[5] * x44 + x50 + x51;
}

void tri6_p2_p2_surface_outflux(const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const SFEM_RESTRICT elems,
                                geom_t **const SFEM_RESTRICT xyz,
                                geom_t **const SFEM_RESTRICT normals_xyz,
                                real_t *const SFEM_RESTRICT vector_field_x,
                                real_t *const SFEM_RESTRICT vector_field_y,
                                real_t *const SFEM_RESTRICT vector_field_z,
                                real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

    real_t element_vector_x[6];
    real_t element_vector_y[6];
    real_t element_vector_z[6];
    real_t element_vector[6];
    idx_t ev[6];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(6)
        for (int v = 0; v < 6; ++v) {
            ev[v] = elems[v][i];
        }

        // Global to local
        for (int v = 0; v < 6; ++v) {
            element_vector_x[v] = vector_field_x[ev[v]];
            element_vector_y[v] = vector_field_y[ev[v]];
            element_vector_z[v] = vector_field_z[ev[v]];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];

        tri6_p2_p2_surface_outflux_kernel(
            // X-coordinates
            xyz[0][i0],
            xyz[0][i1],
            xyz[0][i2],
            // Y-coordinates
            xyz[1][i0],
            xyz[1][i1],
            xyz[1][i2],
            // Z-coordinates
            xyz[2][i0],
            xyz[2][i1],
            xyz[2][i2],
            // Normal,
            normals_xyz[0][i],
            normals_xyz[1][i],
            normals_xyz[2][i],
            // Data
            element_vector_x,
            element_vector_y,
            element_vector_z,
            // Output
            element_vector);

#pragma unroll(6)
        for (int v = 0; v < 6; ++v) {
            values[ev[v]] += element_vector[v];
        }
    }

    double tock = MPI_Wtime();
    printf("surface_outflux.c: tri6_surface_outflux\t%g seconds\n", tock - tick);
}

void surface_outflux(const enum ElemType element_type,
                     const ptrdiff_t nelements,
                     const ptrdiff_t nnodes,
                     idx_t **const SFEM_RESTRICT elems,
                     geom_t **const SFEM_RESTRICT xyz,
                     geom_t **const SFEM_RESTRICT normals_xyz,
                     real_t *const SFEM_RESTRICT vector_field_x,
                     real_t *const SFEM_RESTRICT vector_field_y,
                     real_t *const SFEM_RESTRICT vector_field_z,
                     real_t *const SFEM_RESTRICT values)

{
    switch (element_type) {
        case TRI3: {
            tri3_p1_p1_surface_outflux(
                nelements, nnodes, elems, xyz, normals_xyz, vector_field_x, vector_field_y, vector_field_z, values);
            break;
        }
        case TRI6: {
            tri6_p2_p2_surface_outflux(
                nelements, nnodes, elems, xyz, normals_xyz, vector_field_x, vector_field_y, vector_field_z, values);
            break;
        }
        default: {
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, SFEM_FAILURE);
        }
    }
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

    if (argc != 6) {
        fprintf(stderr, "usage: %s <folder> <vx.raw> <vy.raw> <vz.raw> <outflux.raw>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *folder = argv[1];
    const char *path_vector_field[3] = {argv[2], argv[3], argv[4]};
    const char *path_output = argv[5];

    printf("%s %s %s %s %s %s\n",
           argv[0],
           folder,
           path_vector_field[0],
           path_vector_field[1],
           path_vector_field[2],
           path_output);

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    auto mesh = sfem::Mesh::create_from_file(sfem::Communicator::wrap(comm), folder);
    const ptrdiff_t n_elements = mesh->n_elements();
    const ptrdiff_t n_nodes = mesh->n_nodes();

    real_t *vector_field[3];
    ptrdiff_t vector_field_size_local, vector_field_size_global;

    for (int d = 0; d < mesh->spatial_dimension(); ++d) {
        array_create_from_file(comm,
                               path_vector_field[d],
                               SFEM_MPI_REAL_T,
                               (void **)&vector_field[d],
                               &vector_field_size_local,
                               &vector_field_size_global);
    }

    geom_t *normals_xyz[3];
    for (int d = 0; d < mesh->spatial_dimension(); ++d) {
        normals_xyz[d] = (geom_t *)malloc(n_elements * sizeof(geom_t));
    }

    normals(n_elements, n_nodes, mesh->elements()->data(), mesh->points()->data(), normals_xyz);

    real_t *outflux = (real_t *)malloc(n_nodes * sizeof(real_t));
    memset(outflux, 0, n_nodes * sizeof(real_t));

    surface_outflux(mesh->element_type(),
                    n_elements,
                    n_nodes,
                    mesh->elements()->data(),
                    mesh->points()->data(),
                    normals_xyz,
                    vector_field[0],
                    vector_field[1],
                    vector_field[2],
                    outflux);

    int SFEM_EXPORT_NORMALS = 0;
    SFEM_READ_ENV(SFEM_EXPORT_NORMALS, atoi);

    if (SFEM_EXPORT_NORMALS) {
        array_write(comm, "normalx.raw", SFEM_MPI_GEOM_T, normals_xyz[0], n_elements, n_elements);
        array_write(comm, "normaly.raw", SFEM_MPI_GEOM_T, normals_xyz[1], n_elements, n_elements);
        array_write(comm, "normalz.raw", SFEM_MPI_GEOM_T, normals_xyz[2], n_elements, n_elements);
    }

    array_write(comm, path_output, SFEM_MPI_REAL_T, outflux, n_nodes, n_nodes);

    free(outflux);

    for (int d = 0; d < mesh->spatial_dimension(); ++d) {
        free(normals_xyz[d]);
        free(vector_field[d]);
    }

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld\n", (long)n_elements, (long)n_nodes);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}

