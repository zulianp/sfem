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

static SFEM_INLINE void assemble_p0_to_p1(const real_t px0,
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
                                          // Data
                                          const real_t *const SFEM_RESTRICT u_p0,
                                          // Output
                                          real_t *const SFEM_RESTRICT u_p1,
                                          real_t *const SFEM_RESTRICT weight) {
    // FLOATING POINT OPS!
    //       - Result: 8*ASSIGNMENT
    //       - Subexpressions: 2*ADD + 6*DIV + 13*MUL + 12*SUB
    const real_t x0 = py0 - py2;
    const real_t x1 = pz0 - pz3;
    const real_t x2 = px0 - px1;
    const real_t x3 = py0 - py3;
    const real_t x4 = pz0 - pz2;
    const real_t x5 = px0 - px2;
    const real_t x6 = py0 - py1;
    const real_t x7 = pz0 - pz1;
    const real_t x8 = px0 - px3;
    const real_t x9 = -1.0 / 24.0 * x0 * x1 * x2 + (1.0 / 24.0) * x0 * x7 * x8 + (1.0 / 24.0) * x1 * x5 * x6 +
                      (1.0 / 24.0) * x2 * x3 * x4 - 1.0 / 24.0 * x3 * x5 * x7 - 1.0 / 24.0 * x4 * x6 * x8;
    const real_t x10 = u_p0[0] * x9;
    weight[0] = x9;
    u_p1[0] = x10;
    weight[1] = x9;
    u_p1[1] = x10;
    weight[2] = x9;
    u_p1[2] = x10;
    weight[3] = x9;
    u_p1[3] = x10;
}

void projection_p0_to_p1(const ptrdiff_t nelements,
                         const ptrdiff_t nnodes,
                         idx_t **const SFEM_RESTRICT elems,
                         geom_t **const SFEM_RESTRICT xyz,
                         const real_t *const SFEM_RESTRICT p0,
                         real_t *const SFEM_RESTRICT p1) {
    double tick = MPI_Wtime();

    idx_t ev[4];

    real_t element_p0;
    real_t element_p1[4];
    real_t element_weights[4];

    real_t *weights = (real_t *)malloc(nnodes * sizeof(real_t));
    memset(weights, 0, nnodes * sizeof(real_t));
    memset(p1, 0, nnodes * sizeof(real_t));

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][i];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];
        const idx_t i3 = ev[3];

        element_p0 = p0[i];

        assemble_p0_to_p1(
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
            // Data
            &element_p0,
            // Output
            element_p1,
            element_weights);

        for (int v = 0; v < 4; ++v) {
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
    printf("projection_p0_to_p1.c: projection_p0_to_p1\t%g seconds\n", tock - tick);
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
    if (mesh_read(comm, folder, &mesh)) {
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
    // Compute projection
    ///////////////////////////////////////////////////////////////////////////////

    projection_p0_to_p1(mesh.nelements, mesh.nnodes, mesh.elements, mesh.points, p0, p1);

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
