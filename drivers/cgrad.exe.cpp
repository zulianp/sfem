#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"

#include "sfem_base.hpp"

#include "tet4_grad.hpp"

#include "sfem_API.hpp"

int compute_cgrad(const std::shared_ptr<sfem::Communicator> &comm, int argc, char *argv[]) {
    if (comm->size() != 1) {
        fprintf(stderr, "Parallel execution not supported!\n");
        return EXIT_FAILURE;
    }

    if (argc < 6) {
        fprintf(stderr, "usage: %s <folder> <f.raw> <dfdx.raw> <dfdy.raw> <dfdz.raw>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *folder          = argv[1];
    const char *path_f          = argv[2];
    const char *path_outputs[3] = {argv[3], argv[4], argv[5]};

    printf("%s %s %s %s %s %s\n", argv[0], folder, path_f, path_outputs[0], path_outputs[1], path_outputs[2]);

    const double tick = smesh::time_seconds();

    auto            mesh       = sfem::Mesh::create_from_file(comm, smesh::Path(folder));
    const ptrdiff_t n_elements = mesh->n_elements();
    const ptrdiff_t n_nodes    = mesh->n_nodes();

    auto f = sfem::Buffer<real_t>::from_file(smesh::Path(path_f));
    if (!f) {
        SFEM_ERROR("Failed to read file %s\n", path_f);
    }

    std::shared_ptr<sfem::Buffer<real_t>> df_bufs[3];
    real_t                               *df[3];
    for (int d = 0; d < 3; ++d) {
        df_bufs[d] = sfem::create_host_buffer<real_t>(n_elements);
        df[d]      = df_bufs[d]->data();
    }

    tet4_grad(n_elements, n_nodes, mesh->elements(0)->data(), mesh->points()->data(), f->data(), df[0], df[1], df[2]);

    real_t SFEM_SCALE = 1;
    SFEM_READ_ENV(SFEM_SCALE, atof);

    if (SFEM_SCALE != 1.) {
        for (int d = 0; d < 3; ++d) {
            for (ptrdiff_t i = 0; i < n_elements; i++) {
                df[d][i] *= SFEM_SCALE;
            }
        }
    }

    for (int d = 0; d < 3; ++d) {
        df_bufs[d]->to_file(smesh::Path(path_outputs[d]));
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
    return compute_cgrad(ctx->communicator(), argc, argv);
}
