#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"

#include "sfem_base.hpp"

#include "surface_l2_projection.hpp"

#include "sfem_API.hpp"

int solve_surface_projection(const std::shared_ptr<sfem::Communicator> &comm, int argc, char *argv[]) {
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
    const ptrdiff_t nelements  = mesh->n_elements();
    const ptrdiff_t nnodes     = mesh->n_nodes();

    auto p0_buf = sfem::Buffer<real_t>::from_file(smesh::Path(path_p0));
    if (!p0_buf) {
        SFEM_ERROR("Failed to read file %s\n", path_p0);
    }

    assert((ptrdiff_t)p0_buf->size() == nelements);

    auto p1_buf = sfem::create_host_buffer<real_t>(nnodes);
    real_t *p1  = p1_buf->data();

    int SFEM_COMPUTE_COEFFICIENTS = 1;
    SFEM_READ_ENV(SFEM_COMPUTE_COEFFICIENTS, atoi);

    if (SFEM_COMPUTE_COEFFICIENTS) {
        surface_e_projection_coeffs(
                mesh->element_type(0), nelements, nnodes, mesh->elements(0)->data(), mesh->points()->data(), p0_buf->data(), p1);
    } else {
        surface_e_projection_apply(
                mesh->element_type(0), nelements, nnodes, mesh->elements(0)->data(), mesh->points()->data(), p0_buf->data(), p1);
    }

    p1_buf->to_file(smesh::Path(path_p1));

    const double tock = smesh::time_seconds();

    if (!comm->rank()) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld\n", (long)nelements, (long)nnodes);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return 0;
}

int main(int argc, char *argv[]) {
    auto ctx = sfem::initialize(argc, argv);
    return solve_surface_projection(ctx->communicator(), argc, argv);
}
