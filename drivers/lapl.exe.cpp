#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"

#include "sfem_base.hpp"

#include "operators/laplacian.hpp"

#include "sfem_API.hpp"

int solve_lapl(const std::shared_ptr<sfem::Communicator> &comm, int argc, char *argv[]) {
    if (comm->size() != 1) {
        fprintf(stderr, "Parallel execution not supported!\n");
        return EXIT_FAILURE;
    }

    if (argc != 4) {
        fprintf(stderr, "usage: %s <folder> <u.raw> <output.raw>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *folder      = argv[1];
    const char *path_u      = argv[2];
    const char *path_output = argv[3];

    printf("%s %s %s %s\n", argv[0], folder, path_u, path_output);

    const double tick = smesh::time_seconds();

    auto            mesh       = sfem::Mesh::create_from_file(comm, smesh::Path(folder));
    const ptrdiff_t n_elements = mesh->n_elements();
    const ptrdiff_t n_nodes    = mesh->n_nodes();

    auto u = sfem::Buffer<real_t>::from_file(smesh::Path(path_u));
    if (!u) {
        SFEM_ERROR("Failed to read file %s\n", path_u);
    }

    if ((ptrdiff_t)u->size() != n_nodes) {
        fprintf(stderr, "Input field does not have correct size. Expected %ld, actual = %ld", (long)n_nodes, (long)u->size());
        return EXIT_FAILURE;
    }

    auto lapl_u_buf = sfem::create_host_buffer<real_t>(n_nodes);
    real_t *lapl_u  = lapl_u_buf->data();

    laplacian_apply(mesh->element_type(0), n_elements, n_nodes, mesh->elements(0)->data(), mesh->points()->data(), u->data(), lapl_u);

    real_t SFEM_SCALE = 1;
    SFEM_READ_ENV(SFEM_SCALE, atof);

    if (SFEM_SCALE != 1) {
        for (ptrdiff_t i = 0; i < n_nodes; ++i) {
            lapl_u[i] *= SFEM_SCALE;
        }
    }

    lapl_u_buf->to_file(smesh::Path(path_output));

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
    return solve_lapl(ctx->communicator(), argc, argv);
}
