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

static SFEM_INLINE void surface_projection_p0_to_p1_kernel(const real_t px0,
                                                           const real_t px1,
                                                           const real_t px2,
                                                           const real_t py0,
                                                           const real_t py1,
                                                           const real_t py2,
                                                           const real_t pz0,
                                                           const real_t pz1,
                                                           const real_t pz2,
                                                           // Data
                                                           const real_t *const SFEM_RESTRICT u_p0,
                                                           // Output
                                                           real_t *const SFEM_RESTRICT u_p1,
                                                           real_t *const SFEM_RESTRICT weight) {
    // FLOATING POINT OPS!
    //       - Result: 6*ASSIGNMENT
    //       - Subexpressions: 6*ADD + 2*DIV + 5*MUL + 8*POW + 7*SUB
    const real_t x0 = -px0 + px1;
    const real_t x1 = -px0 + px2;
    const real_t x2 = -py0 + py1;
    const real_t x3 = -py0 + py2;
    const real_t x4 = -pz0 + pz1;
    const real_t x5 = -pz0 + pz2;
    const real_t x6 =
        (1.0 / 6.0) * sqrt((pow(x0, 2) + pow(x2, 2) + pow(x4, 2)) * (pow(x1, 2) + pow(x3, 2) + pow(x5, 2)) -
                           pow(x0 * x1 + x2 * x3 + x4 * x5, 2));
    const real_t x7 = u_p0[0] * x6;
    weight[0] = x6;
    u_p1[0] = x7;
    weight[1] = x6;
    u_p1[1] = x7;
    weight[2] = x6;
    u_p1[2] = x7;
}

static SFEM_INLINE void integrate_p0_to_p1_kernel(const real_t px0,
                                                  const real_t px1,
                                                  const real_t px2,
                                                  const real_t py0,
                                                  const real_t py1,
                                                  const real_t py2,
                                                  const real_t pz0,
                                                  const real_t pz1,
                                                  const real_t pz2,
                                                  // Data
                                                  const real_t *const SFEM_RESTRICT u_p0,
                                                  // Output
                                                  real_t *const SFEM_RESTRICT u_p1) {
    // FLOATING POINT OPS!
    //       - Result: 3*ASSIGNMENT
    //       - Subexpressions: 6*ADD + 2*DIV + 5*MUL + 8*POW + 7*SUB
    const real_t x0 = -px0 + px1;
    const real_t x1 = -px0 + px2;
    const real_t x2 = -py0 + py1;
    const real_t x3 = -py0 + py2;
    const real_t x4 = -pz0 + pz1;
    const real_t x5 = -pz0 + pz2;
    const real_t x6 = (1.0 / 6.0) * u_p0[0] *
                      sqrt((pow(x0, 2) + pow(x2, 2) + pow(x4, 2)) * (pow(x1, 2) + pow(x3, 2) + pow(x5, 2)) -
                           pow(x0 * x1 + x2 * x3 + x4 * x5, 2));
    u_p1[0] = x6;
    u_p1[1] = x6;
    u_p1[2] = x6;
}

void surface_projection_p0_to_p1(const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elems,
                                 geom_t **const SFEM_RESTRICT xyz,
                                 const real_t *const SFEM_RESTRICT p0,
                                 real_t *const SFEM_RESTRICT p1) {
    double tick = MPI_Wtime();

    idx_t ev[3];

    real_t element_p0;
    real_t element_p1[3];
    real_t element_weights[3];

    real_t *weights = (real_t *)malloc(nnodes * sizeof(real_t));
    memset(weights, 0, nnodes * sizeof(real_t));
    memset(p1, 0, nnodes * sizeof(real_t));

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(3)
        for (int v = 0; v < 3; ++v) {
            ev[v] = elems[v][i];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];

        element_p0 = p0[i];

        surface_projection_p0_to_p1_kernel(
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
            // Data
            &element_p0,
            // Output
            element_p1,
            element_weights);

        for (int v = 0; v < 3; ++v) {
            const idx_t idx = ev[v];
            p1[idx] += element_p1[v];
            weights[idx] += element_weights[v];
        }
    }

    for (ptrdiff_t i = 0; i < nnodes; i++) {
        p1[i] /= weights[i];
    }

    free(weights);

    double tock = MPI_Wtime();
    printf("surface_projection_p0_to_p1.c: surface_projection_p0_to_p1\t%g seconds\n", tock - tick);
}

void surface_assemble_p0_to_p1(const ptrdiff_t nelements,
                               const ptrdiff_t nnodes,
                               idx_t **const SFEM_RESTRICT elems,
                               geom_t **const SFEM_RESTRICT xyz,
                               const real_t *const SFEM_RESTRICT p0,
                               real_t *const SFEM_RESTRICT p1) {
    double tick = MPI_Wtime();

    idx_t ev[3];

    real_t element_p0;
    real_t element_p1[3];

    memset(p1, 0, nnodes * sizeof(real_t));

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(3)
        for (int v = 0; v < 3; ++v) {
            ev[v] = elems[v][i];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];

        element_p0 = p0[i];

        integrate_p0_to_p1_kernel(
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
            // Data
            &element_p0,
            // Output
            element_p1);

        for (int v = 0; v < 3; ++v) {
            const idx_t idx = ev[v];
            p1[idx] += element_p1[v];
        }
    }

    double tock = MPI_Wtime();
    printf("surface_projection_p0_to_p1.c: surface_assemble_p0_to_p1\t%g seconds\n", tock - tick);
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

    if (argc != 4) {
        fprintf(stderr, "usage: %s <folder> <in_p0.raw> <out_p1.raw>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *folder = argv[1];
    const char *path_p0 = argv[2];
    const char *path_p1 = argv[3];

    printf("%s %s %s %s\n", argv[0], folder, path_p0, path_p1);

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    mesh_t mesh;
    if (mesh_surf_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    real_t *p0;
    ptrdiff_t p0_n_local, p0_n_global;
    array_create_from_file(comm, path_p0, SFEM_MPI_REAL_T, (void **)&p0, &p0_n_local, &p0_n_global);

    ptrdiff_t nelements = mesh.nelements;
    ptrdiff_t nnodes = mesh.nnodes;

    assert(p0_n_local == nelements);

    real_t *p1 = (real_t *)malloc(nnodes * sizeof(real_t));
    memset(p1, 0, nnodes * sizeof(real_t));

    ///////////////////////////////////////////////////////////////////////////////
    // Compute surface_projection
    ///////////////////////////////////////////////////////////////////////////////

    int SFEM_COMPUTE_COEFFICIENTS = 1;

    SFEM_READ_ENV(SFEM_COMPUTE_COEFFICIENTS, atoi);

    if (SFEM_COMPUTE_COEFFICIENTS) {
        surface_projection_p0_to_p1(mesh.nelements, mesh.nnodes, mesh.elements, mesh.points, p0, p1);
    } else {
        surface_assemble_p0_to_p1(mesh.nelements, mesh.nnodes, mesh.elements, mesh.points, p0, p1);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Write cell data
    ///////////////////////////////////////////////////////////////////////////////

    array_write(comm, path_p1, SFEM_MPI_REAL_T, p1, nnodes, nnodes);

    ///////////////////////////////////////////////////////////////////////////////
    // Free resources
    ///////////////////////////////////////////////////////////////////////////////

    free(p0);
    free(p1);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld\n", (long)mesh.nelements, (long)mesh.nnodes);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
