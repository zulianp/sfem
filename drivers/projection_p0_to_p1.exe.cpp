#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"

#include "sfem_base.hpp"

#include "tet4_l2_projection_p0_p1.hpp"

#include "sfem_API.hpp"

int solve_projection_p0_to_p1(const std::shared_ptr<sfem::Communicator> &comm, int argc, char *argv[]) {
    if (comm->size() != 1) {
        fprintf(stderr, "Parallel execution not supported!\n");
        return EXIT_FAILURE;
    }

    if (argc != 4) {
        fprintf(stderr, "usage: %s <folder> <in_p0.raw> <out_p1.raw>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *folder  = argv[1];
    const char *path_p0 = argv[2];
    const char *path_p1 = argv[3];

    printf("%s %s %s %s\n", argv[0], folder, path_p0, path_p1);

    const double tick = smesh::time_seconds();

    auto            mesh       = sfem::Mesh::create_from_file(comm, smesh::Path(folder));
    const ptrdiff_t n_elements = mesh->n_elements();
    const ptrdiff_t n_nodes    = mesh->n_nodes();

    auto p0 = sfem::Buffer<real_t>::from_file(smesh::Path(path_p0));
    if (!p0) {
        SFEM_ERROR("Failed to read file %s\n", path_p0);
    }

    assert((ptrdiff_t)p0->size() == n_elements);

    auto p1_buf = sfem::create_host_buffer<real_t>(n_nodes);
    real_t *p1  = p1_buf->data();

    int SFEM_COMPUTE_COEFFICIENTS = 1;

    SFEM_READ_ENV(SFEM_COMPUTE_COEFFICIENTS, atoi);

    if (SFEM_COMPUTE_COEFFICIENTS) {
        tet4_p0_p1_projection_coeffs(n_elements, n_nodes, mesh->elements(0)->data(), mesh->points()->data(), p0->data(), p1);
    } else {
        tet4_p0_p1_l2_projection_apply(n_elements, n_nodes, mesh->elements(0)->data(), mesh->points()->data(), p0->data(), p1);
    }

    p1_buf->to_file(smesh::Path(path_p1));

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
    return solve_projection_p0_to_p1(ctx->communicator(), argc, argv);
}
