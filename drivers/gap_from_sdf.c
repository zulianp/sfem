#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "matrixio_array.h"
#include "matrixio_ndarray.h"

#include "read_mesh.h"
#include "sfem_mesh_write.h"
#include "sfem_resample_gap.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

static void minmax(
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

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // if (size != 1) {
    //     fprintf(stderr, "Parallel execution not supported!\n");
    //     return EXIT_FAILURE;
    // }

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

    const char* folder = argv[1];
    ptrdiff_t nglobal[3] = { atol(argv[2]), atol(argv[3]), atol(argv[4]) };
    geom_t origin[3] = { atof(argv[5]), atof(argv[6]), atof(argv[7]) };
    geom_t delta[3] = { atof(argv[8]), atof(argv[9]), atof(argv[10]) };
    const char* data_path = argv[11];
    const char* output_folder = argv[12];

    struct stat st = { 0 };
    if (stat(output_folder, &st) == -1) {
        mkdir(output_folder, 0700);
    }

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }


    ptrdiff_t n = nglobal[0] * nglobal[1] * nglobal[2];
    geom_t* sdf = (geom_t*)malloc(n * sizeof(geom_t));
    ptrdiff_t nlocal[3];

    // if (ndarray_read(comm, data_path, SFEM_MPI_GEOM_T, 3, sdf, nlocal, nglobal)) {
    //     return EXIT_FAILURE;
    // }

    if (ndarray_read(comm, data_path, SFEM_MPI_GEOM_T, 3, sdf, nlocal, nglobal)) {
        return EXIT_FAILURE;
    }

    // X is contiguous
    ptrdiff_t stride[3] = { 1, nlocal[0], nlocal[0] * nlocal[1] };

    real_t* g = malloc(mesh.nnodes * sizeof(real_t));
    real_t* xnormal = malloc(mesh.nnodes * sizeof(real_t));
    real_t* ynormal = malloc(mesh.nnodes * sizeof(real_t));
    real_t* znormal = malloc(mesh.nnodes * sizeof(real_t));

    if (size > 1) {
        geom_t zmin, zmax;
        minmax(mesh.nnodes, mesh.points[2], &zmin, &zmax);

        // Z is distributed
        ptrdiff_t zoffset = 0;
        MPI_Exscan(&nlocal[2], &zoffset, 1, MPI_LONG, MPI_SUM, comm);

        // Compute Local z-tile
        ptrdiff_t sdf_start = (zmin - origin[2]) / delta[2];
        ptrdiff_t sdf_end   = (zmax - origin[2]) / delta[2];

        // ptrdiff_t sdf_start = 0;
        // ptrdiff_t sdf_end   = nglobal[2];

        // Make sure we are inside the grid
        sdf_start = MAX(0, sdf_start);
        sdf_end = MIN(nglobal[2], sdf_end  + 1);

        ptrdiff_t pnlocal_z = (sdf_end - sdf_start);
        geom_t* psdf = malloc(pnlocal_z * stride[2] * sizeof(geom_t));
        array_range_select(comm, SFEM_MPI_GEOM_T, sdf, psdf, nlocal[2], sdf_start * stride[2], sdf_end * stride[2]);

        printf("[%d] sdf_start=%ld, sdf_end=%ld\n", rank, sdf_start, sdf_end);
        origin[2] += sdf_start * delta[2];;
        nlocal[2] = pnlocal_z;

        free(sdf);
        sdf = psdf;
    }

    if (SFEM_INTERPOLATE) {
        interpolate_gap(
            // Mesh
            mesh.n_owned_nodes,
            mesh.points,
            // SDF
            nlocal,
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
        resample_gap(
            // Mesh
            mesh.element_type,
            mesh.nelements,
            mesh.n_owned_nodes,
            mesh.elements,
            mesh.points,
            // SDF
            nlocal,
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
        mesh_write_nodal_field(&mesh, path, SFEM_MPI_REAL_T, g);

        sprintf(path, "%s/xnormal.float64.raw", output_folder);
        mesh_write_nodal_field(&mesh, path, SFEM_MPI_REAL_T, xnormal);

        sprintf(path, "%s/ynormal.float64.raw", output_folder);
        mesh_write_nodal_field(&mesh, path, SFEM_MPI_REAL_T, ynormal);

        sprintf(path, "%s/znormal.float64.raw", output_folder);
        mesh_write_nodal_field(&mesh, path, SFEM_MPI_REAL_T, znormal);
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
