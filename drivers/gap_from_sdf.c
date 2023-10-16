#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "matrix.io/matrixio_array.h"
#include "matrix.io/matrixio_ndarray.h"

#include "read_mesh.h"
#include "sfem_resample_gap.h"

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

    int SFEM_INTERPOLATE = 1;
    SFEM_READ_ENV(SFEM_INTERPOLATE, atoi);

    double tick = MPI_Wtime();

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
    ptrdiff_t stride[3] = {1, nglobal[0], nglobal[0] * nglobal[1]};

    real_t *g = malloc(mesh.nnodes * sizeof(real_t));
    real_t *xnormal = malloc(mesh.nnodes * sizeof(real_t));
    real_t *ynormal = malloc(mesh.nnodes * sizeof(real_t));
    real_t *znormal = malloc(mesh.nnodes * sizeof(real_t));

    if (SFEM_INTERPOLATE) {
        interpolate_gap(comm,
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
    } else {
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
    }

    // Write result to disk
    {
        char path[1024 * 10];
        sprintf(path, "%s/gap.float64.raw", output_folder);
        array_write(comm, path, SFEM_MPI_REAL_T, g, mesh.nnodes, mesh.nnodes);

        sprintf(path, "%s/xnormal.float64.raw", output_folder);
        array_write(comm, path, SFEM_MPI_REAL_T, xnormal, mesh.nnodes, mesh.nnodes);

        sprintf(path, "%s/ynormal.float64.raw", output_folder);
        array_write(comm, path, SFEM_MPI_REAL_T, ynormal, mesh.nnodes, mesh.nnodes);

        sprintf(path, "%s/znormal.float64.raw", output_folder);
        array_write(comm, path, SFEM_MPI_REAL_T, znormal, mesh.nnodes, mesh.nnodes);
    }

    ptrdiff_t nelements = mesh.nelements;
    ptrdiff_t nnodes = mesh.nnodes;

    // Free resources
    {
        free(sdf);
        free(g);
        free(xnormal);
        free(ynormal);
        free(znormal);
        mesh_destroy(&mesh);
    }

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld #grid (%ld x %ld x %ld)\n",
               (long)nelements,
               (long)nnodes,
               nglobal[0],
               nglobal[1],
               nglobal[2]);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
