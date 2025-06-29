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

#include "operators/div.h"

#include "read_mesh.h"

#include "sfem_API.hpp"

static SFEM_INLINE void volume(const real_t px0,
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
                          real_t *const element_value) {
    // FLOATING POINT OPS!
    //       - Result: ADD + ASSIGNMENT + 6*MUL
    //       - Subexpressions: 9*SUB
    const real_t x0 = -px0 + px1;
    const real_t x1 = -py0 + py2;
    const real_t x2 = -pz0 + pz3;
    const real_t x3 = -px0 + px2;
    const real_t x4 = -py0 + py3;
    const real_t x5 = -pz0 + pz1;
    const real_t x6 = -px0 + px3;
    const real_t x7 = -py0 + py1;
    const real_t x8 = -pz0 + pz2;
    element_value[0] = x0 * x1 * x2 - x0 * x4 * x8 - x1 * x5 * x6 - x2 * x3 * x7 + x3 * x4 * x5 + x6 * x7 * x8;
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

    if (argc != 3) {
        fprintf(stderr, "usage: %s <folder> <output.raw>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *folder = argv[1];
    const char *path_output = argv[2];

    printf("%s %s %s\n", argv[0], folder, path_output);

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    auto mesh = sfem::Mesh::create_from_file(comm, folder);

    real_t *volumes = (real_t *)malloc(mesh->n_elements() * sizeof(real_t));
    memset(volumes, 0, mesh->n_elements() * sizeof(real_t));

    geom_t ** xyz = mesh->points()->data();
    auto elements = mesh->elements()->data();

    for (ptrdiff_t i = 0; i < mesh->n_elements(); ++i) {
        const idx_t i0 = elements[0][i];
        const idx_t i1 = elements[1][i];
        const idx_t i2 = elements[2][i];
        const idx_t i3 = elements[3][i];

        real_t vol = 0;
        volume(
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
            &vol);

        volumes[i] = vol;
    }

    int SFEM_VERBOSE = 1;
    SFEM_READ_ENV(SFEM_VERBOSE, atoi);
    array_write(comm, path_output, SFEM_MPI_REAL_T, volumes, mesh->n_elements(), mesh->n_elements());
    free(volumes);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld\n", (long)mesh->n_elements(), (long)mesh->n_nodes());
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
