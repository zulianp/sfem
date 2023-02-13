#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../matrix.io/array_dtof.h"
#include "../matrix.io/matrixio_array.h"
#include "../matrix.io/matrixio_crs.h"
#include "../matrix.io/utils.h"

#include "crs_graph.h"
#include "read_mesh.h"
#include "sfem_base.h"
#include "sfem_mesh_write.h"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc < 6) {
        if (!rank) {
            fprintf(stderr, "usage: %s <folder> <x> <y> <z> <max_vertices> [output_folder=./]", argv[0]);
        }

        return EXIT_FAILURE;
    }

    geom_t roi[3] = {atof(argv[2]), atof(argv[3]), atof(argv[4])};
    ptrdiff_t max_vertices = atol(argv[5]);

    const char *output_folder = "./";
    if (argc > 6) {
        output_folder = argv[6];
    }

    if (!rank) {
        printf("%s %s %g %g %g %ld %s\n",
               argv[0],
               argv[1],
               (double)roi[0],
               (double)roi[1],
               (double)roi[2],
               (long)max_vertices,
               output_folder);
    }

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];
    char path[1024 * 10];

    mesh_t mesh;
    if (mesh_surf_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    double tack = MPI_Wtime();

    mesh_t selection; 
    geom_t closest_sq_dist = 1000000;
    ptrdiff_t closest_node = -1;

    const int dim = mesh.spatial_dim;

    for (ptrdiff_t node = 0; node < mesh.nnodes; ++node) {
        geom_t sq_dist = 0;
        for(int d = 0; d < dim; ++d) {
            const real_t m_x = mesh.points[d][node];
            const real_t roi_x = roi[d];
            const real_t diff = m_x - roi_x;
            sq_dist += diff * diff;
        }

        if(sq_dist < closest_sq_dist) {
            closest_sq_dist = sq_dist;
            closest_node = node;
        }
    }

    printf("found: %ld %g\n", closest_node, closest_sq_dist);

    // char output_path[2048];
    // sprintf(output_path, "part_%0.5d", rank);
    // // Everyone independent
    // mesh.comm = MPI_COMM_SELF;
    // mesh_write(output_path, &mesh);

    // for (int r = 0; r < size; ++r) {
    //     if (r == rank) {
    //         printf("[%d] #elements %ld #nodes %ld\n", rank, (long)mesh.nelements, (long)mesh.nnodes);
    //     }

    //     fflush(stdout);

    //     MPI_Barrier(comm);
    // }

    // MPI_Barrier(comm);

    // mesh_destroy(&mesh);
    // double tock = MPI_Wtime();

    // if (!rank) {
    //     printf("----------------------------------------\n");
    //     printf("TTS:\t\t\t%g seconds\n", tock - tick);
    // }

    return MPI_Finalize();
}
