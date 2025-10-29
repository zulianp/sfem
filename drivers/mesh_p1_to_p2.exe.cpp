#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "sfem_Mesh.hpp"
#include "sfem_P1toP2.hpp"
#include "sfem_glob.hpp"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    if (argc != 3) {
        if (!rank) {
            fprintf(stderr, "usage: %s <folder> <output_folder>\n", argv[0]);
        }

        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const char *output_folder = argv[2];

    if (!rank) {
        printf("%s %s %s\n", argv[0], argv[1], output_folder);
    }

    double tick = MPI_Wtime();

    sfem::create_directory(output_folder);

    auto p1_mesh = sfem::Mesh::create_from_file(sfem::Communicator::wrap(comm), argv[1]);
    auto p2_mesh = sfem::convert_p1_mesh_to_p2(p1_mesh);

    if (!p2_mesh) {
        if (!rank) {
            fprintf(stderr, "mesh_p1_to_p2: conversion failed\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (p2_mesh->write(output_folder) != SFEM_SUCCESS) {
        if (!rank) {
            fprintf(stderr, "mesh_p1_to_p2: failed to write output mesh to %s\n", output_folder);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (!rank) {
        printf("----------------------------------------\n");
        printf("mesh_p1_to_p2.c: #elements %ld, nodes #p1 %ld, #p2 %ld\n",
               (long)p1_mesh->n_elements(),
               (long)p1_mesh->n_nodes(),
               (long)p2_mesh->n_nodes());
        printf("----------------------------------------\n");
    }

    double tock = MPI_Wtime();
    if (!rank) {
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
