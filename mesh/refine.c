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


// Generic indexing, but maybe lets stick to exodus
// {
// template <Integer ManifoldDim>
// inline Integer midpoint_index(const Integer i, const Integer j) {
//     const auto ip1 = i + 1;
//     const auto jp1 = j + 1;
//     return ((ip1 - 1) * (ManifoldDim - (ip1 / 2.)) + jp1 + ManifoldDim) - 1;
// }
//     template <Integer Dim>
//     inline void fixed_red_refinement(std::array<Simplex<Dim, 3>, 8> &sub_simplices) {
//         // corner tets
//         sub_simplices[0].nodes = {0, 4, 5, 6};
//         sub_simplices[1].nodes = {1, 7, 4, 8};
//         sub_simplices[2].nodes = {2, 5, 7, 9};
//         sub_simplices[3].nodes = {6, 8, 9, 3};

//         // octahedron tets
//         sub_simplices[4].nodes = {4, 8, 7, 5};
//         sub_simplices[5].nodes = {6, 5, 8, 4};
//         sub_simplices[6].nodes = {6, 8, 5, 9};
//         sub_simplices[7].nodes = {8, 7, 5, 9};
//     }

// }

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc < 2) {
        if (!rank) {
            fprintf(stderr, "usage: %s <folder> [output_folder=./]", argv[0]);
        }

        return EXIT_FAILURE;
    }

    const char *output_folder = "./";
    if (argc > 2) {
        output_folder = argv[2];
    }

    if (!rank) {
        printf("%s %s %s\n", argv[0], argv[1], output_folder);
    }

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];
    // char path[1024 * 10];

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    // double tack = MPI_Wtime();

    mesh_t fine_mesh;

    //TODO Refine mesh

    mesh_write(output_folder, &fine_mesh);    

    mesh_destroy(&mesh);
    mesh_destroy(&fine_mesh);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
