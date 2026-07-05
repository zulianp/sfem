#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"

#include "sfem_base.hpp"

#include "tet4_neohookean.hpp"

#include "sfem_API.hpp"

int compute_cauchy_stress(const std::shared_ptr<sfem::Communicator> &comm, int argc, char *argv[]) {
    if (comm->size() != 1) {
        fprintf(stderr, "Parallel execution not supported!\n");
        return EXIT_FAILURE;
    }

    if (argc != 7 && argc != 9) {
        fprintf(stderr, "usage: (input can be AoS or SoA. output is always SoA)\n");
        fprintf(stderr, " (AoS): %s <material> <mu> <lambda> <folder> <uxyz.raw> <stress_prefix>\n", argv[0]);
        fprintf(stderr, " (SoA): %s <material> <mu> <lambda> <folder> <ux.raw> <uy.raw> <uz.raw> <stress_prefix>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *material = argv[1];

    real_t mu     = atof(argv[2]);
    real_t lambda = atof(argv[3]);

    const char *folder = argv[4];
    const char *path_u[3];
    const char *output_prefix;

    const char *SFEM_OUTPUT_POSTFIX = "";
    SFEM_READ_ENV(SFEM_OUTPUT_POSTFIX, );

    int is_AoS = argc == 7;

    if (is_AoS) {
        path_u[0]     = argv[5];
        output_prefix = argv[6];

        printf("(AoS) %s %s %g %g %s %s %s\n", argv[0], material, (double)mu, (double)lambda, folder, path_u[0], output_prefix);

    } else {
        path_u[0]     = argv[5];
        path_u[1]     = argv[6];
        path_u[2]     = argv[7];
        output_prefix = argv[8];

        printf("(SoA) %s %s %g %g %s %s %s %s %s\n",
               argv[0],
               material,
               (double)mu,
               (double)lambda,
               folder,
               path_u[0],
               path_u[1],
               path_u[2],
               output_prefix);
    }

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    auto mesh = sfem::Mesh::create_from_file(comm, smesh::Path(folder));

    std::shared_ptr<sfem::Buffer<real_t>> stress_bufs[6];
    real_t                               *stress[6];
    for (int d = 0; d < 6; ++d) {
        stress_bufs[d] = sfem::create_host_buffer<real_t>(mesh->n_elements());
        stress[d]      = stress_bufs[d]->data();
    }

    // TODO
    // if(strcmp(material, "neohookean") == 0) { }

    if (is_AoS) {
        auto u = sfem::Buffer<real_t>::from_file(smesh::Path(path_u[0]));
        if (!u) {
            SFEM_ERROR("Failed to read file %s\n", path_u[0]);
        }
        neohookean_cauchy_stress_aos(
                mesh->n_elements(), mesh->n_nodes(), mesh->elements(0)->data(), mesh->points()->data(), mu, lambda, u->data(), stress);
    } else {
        auto u0 = sfem::Buffer<real_t>::from_file(smesh::Path(path_u[0]));
        auto u1 = sfem::Buffer<real_t>::from_file(smesh::Path(path_u[1]));
        auto u2 = sfem::Buffer<real_t>::from_file(smesh::Path(path_u[2]));
        if (!u0 || !u1 || !u2) {
            SFEM_ERROR("Failed to read displacement files\n");
        }
        real_t *u[3] = {u0->data(), u1->data(), u2->data()};
        neohookean_cauchy_stress_soa(
                mesh->n_elements(), mesh->n_nodes(), mesh->elements(0)->data(), mesh->points()->data(), mu, lambda, u, stress);
    }

    char path[2048];
    for (int d = 0; d < 6; ++d) {
        snprintf(path, sizeof(path), "%s.%d%s.raw", output_prefix, d, SFEM_OUTPUT_POSTFIX);
        stress_bufs[d]->to_file(smesh::Path(path));
    }

    double tock = MPI_Wtime();

    if (!comm->rank()) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld\n", (long)mesh->n_elements(), (long)mesh->n_nodes());
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return 0;
}

int main(int argc, char *argv[]) {
    auto ctx = sfem::initialize(argc, argv);
    return compute_cauchy_stress(ctx->communicator(), argc, argv);
}
