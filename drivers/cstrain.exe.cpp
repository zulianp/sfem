#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"

#include "sfem_base.hpp"

#include "tet4_strain.hpp"

#include "sfem_API.hpp"

int solve_cstrain(const std::shared_ptr<sfem::Communicator> &comm, int argc, char *argv[]) {
    if (comm->size() != 1) {
        fprintf(stderr, "Parallel execution not supported!\n");
        return EXIT_FAILURE;
    }

    if (argc < 5) {
        fprintf(stderr, "usage: %s <folder> <ux.raw> <uy.raw> <uz.raw> <strain_prefix>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *SFEM_OUTPUT_POSTFIX = "";
    SFEM_READ_ENV(SFEM_OUTPUT_POSTFIX, );

    const char *folder        = argv[1];
    const char *path_u[3]     = {argv[2], argv[3], argv[4]};
    const char *output_prefix = argv[5];

    printf("%s %s %s %s %s %s\n", argv[0], folder, path_u[0], path_u[1], path_u[2], output_prefix);

    const double tick = smesh::time_seconds();

    auto            mesh       = sfem::Mesh::create_from_file(comm, smesh::Path(folder));
    const ptrdiff_t n_elements = mesh->n_elements();
    const ptrdiff_t n_nodes    = mesh->n_nodes();

    auto u0 = sfem::Buffer<real_t>::from_file(smesh::Path(path_u[0]));
    auto u1 = sfem::Buffer<real_t>::from_file(smesh::Path(path_u[1]));
    auto u2 = sfem::Buffer<real_t>::from_file(smesh::Path(path_u[2]));
    if (!u0 || !u1 || !u2) {
        SFEM_ERROR("Failed to read displacement files\n");
    }

    real_t *u[3] = {u0->data(), u1->data(), u2->data()};

    std::shared_ptr<sfem::Buffer<real_t>> strain_bufs[6];
    real_t                               *strain_6[6];
    for (int d = 0; d < 6; ++d) {
        strain_bufs[d] = sfem::create_host_buffer<real_t>(n_elements);
        strain_6[d]    = strain_bufs[d]->data();
    }

    strain(n_elements,
           n_nodes,
           mesh->elements(0)->data(),
           mesh->points()->data(),
           u[0],
           u[1],
           u[2],
           strain_6[0],
           strain_6[1],
           strain_6[2],
           strain_6[3],
           strain_6[4],
           strain_6[5]);

    char path[2048];
    for (int d = 0; d < 6; ++d) {
        snprintf(path, sizeof(path), "%s.%d%s.raw", output_prefix, d, SFEM_OUTPUT_POSTFIX);
        strain_bufs[d]->to_file(smesh::Path(path));
    }

    const double tock = smesh::time_seconds();

    if (!comm->rank()) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld\n", (long)n_elements, (long)n_nodes);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return 0;
}

int main(int argc, char *argv[]) {
    auto ctx = sfem::initialize(argc, argv);
    return solve_cstrain(ctx->communicator(), argc, argv);
}
