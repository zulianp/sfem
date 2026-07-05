#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"

#include "sfem_base.hpp"

#include "operators/div.hpp"

#include "sfem_API.hpp"

int compute_u_dot_grad_q(const std::shared_ptr<sfem::Communicator> &comm, int argc, char *argv[]) {
    if (comm->size() != 1) {
        fprintf(stderr, "Parallel execution not supported!\n");
        return EXIT_FAILURE;
    }

    if (argc < 5) {
        fprintf(stderr, "usage: %s <folder> <ux.raw> <uy.raw> <uz.raw> <output.raw>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *folder      = argv[1];
    const char *path_u[3]   = {argv[2], argv[3], argv[4]};
    const char *path_output = argv[5];

    printf("%s %s %s %s %s %s\n", argv[0], folder, path_u[0], path_u[1], path_u[2], path_output);

    const double tick = smesh::time_seconds();

    auto            mesh       = sfem::Mesh::create_from_file(comm, smesh::Path(folder));
    const ptrdiff_t nelements  = mesh->n_elements();
    const ptrdiff_t nnodes     = mesh->n_nodes();

    std::shared_ptr<sfem::Buffer<real_t>> u_bufs[3];
    real_t                               *u[3];
    ptrdiff_t                             nu_p0_or_p1 = 0;

    for (int d = 0; d < 3; ++d) {
        u_bufs[d] = sfem::Buffer<real_t>::from_file(smesh::Path(path_u[d]));
        if (!u_bufs[d]) {
            SFEM_ERROR("Failed to read file %s\n", path_u[d]);
        }
        u[d] = u_bufs[d]->data();
        if (d == 0) {
            nu_p0_or_p1 = (ptrdiff_t)u_bufs[d]->size();
        }
    }

    assert(nu_p0_or_p1 == nelements || nu_p0_or_p1 == nnodes);

    auto div_u_buf = sfem::create_host_buffer<real_t>(nnodes);
    real_t *div_u  = div_u_buf->data();

    if (nu_p0_or_p1 == nelements) {
        p0_u_dot_grad_q_apply(nelements, nnodes, mesh->elements(0)->data(), mesh->points()->data(), u[0], u[1], u[2], div_u);
    } else {
        p1_u_dot_grad_q_apply(nelements, nnodes, mesh->elements(0)->data(), mesh->points()->data(), u[0], u[1], u[2], div_u);
    }

    real_t SFEM_SCALE = 1;
    SFEM_READ_ENV(SFEM_SCALE, atof);

    if (SFEM_SCALE != 1) {
        for (ptrdiff_t i = 0; i < nnodes; ++i) {
            div_u[i] *= SFEM_SCALE;
        }
    }

    int SFEM_VERBOSE = 0;
    SFEM_READ_ENV(SFEM_VERBOSE, atoi);

    if (SFEM_VERBOSE) {
        real_t integral = 0.;
        for (ptrdiff_t i = 0; i < nnodes; ++i) {
            integral += div_u[i];
        }

        if (!comm->rank()) {
            printf("integral div(u) = %g\n", (double)integral);
        }
    }

    div_u_buf->to_file(smesh::Path(path_output));

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
    return compute_u_dot_grad_q(ctx->communicator(), argc, argv);
}
