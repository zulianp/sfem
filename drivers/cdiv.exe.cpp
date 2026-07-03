#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"

#include "sfem_base.hpp"

#include "operators/div.hpp"

#include "sfem_API.hpp"

int compute_cdiv(const std::shared_ptr<sfem::Communicator> &comm, int argc, char *argv[]) {
    if (comm->size() != 1) {
        fprintf(stderr, "Parallel execution not supported!\n");
        return EXIT_FAILURE;
    }

    if (argc != 6) {
        fprintf(stderr, "usage: %s <folder> <ux.raw> <uy.raw> <uz.raw> <output.raw>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *folder      = argv[1];
    const char *path_u[3]   = {argv[2], argv[3], argv[4]};
    const char *path_output = argv[5];

    printf("%s %s %s %s %s %s\n", argv[0], folder, path_u[0], path_u[1], path_u[2], path_output);

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

    auto    div_u_buf = sfem::create_host_buffer<real_t>(mesh->n_elements());
    real_t *div_u     = div_u_buf->data();

    cdiv(mesh->element_type(0), n_elements, n_nodes, mesh->elements(0)->data(), mesh->points()->data(), u[0], u[1], u[2], div_u);

    real_t SFEM_SCALE = 1;
    SFEM_READ_ENV(SFEM_SCALE, atof);

    if (SFEM_SCALE != 1) {
        for (ptrdiff_t i = 0; i < n_elements; ++i) {
            div_u[i] *= SFEM_SCALE;
        }
    }

    div_u_buf->to_file(smesh::Path(path_output));

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
    return compute_cdiv(ctx->communicator(), argc, argv);
}
