#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"

#include "sfem_base.hpp"

#include "tet4_neohookean.hpp"
#include "sfem_API.hpp"

int compute_vonmises(const std::shared_ptr<sfem::Communicator> &comm, int argc, char *argv[]) {
    if (comm->size() != 1) {
        fprintf(stderr, "Parallel execution not supported!\n");
        return EXIT_FAILURE;
    }

    if (argc != 7 && argc != 9) {
        fprintf(stderr, "usage: (input can be AoS or SoA. output is always SoA)\n");
        fprintf(stderr, " (AoS): %s <material> <mu> <lambda> <folder> <uxyz.raw> <vonmises.raw>\n", argv[0]);
        fprintf(stderr,
                " (SoA): %s <material> <mu> <lambda> <folder> <ux.raw> <uy.raw> <uz.raw> <vonmises.raw>\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    const char *material = argv[1];

    real_t mu     = atof(argv[2]);
    real_t lambda = atof(argv[3]);

    const char *folder = argv[4];
    const char *path_u[3];
    const char *output_path;

    int is_AoS = argc == 7;

    if (is_AoS) {
        path_u[0]     = argv[5];
        output_path   = argv[6];

        printf("(AoS) %s %s %g %g %s %s %s\n",
               argv[0],
               material,
               (double)mu,
               (double)lambda,
               folder,
               path_u[0],
               output_path);

    } else {
        path_u[0]   = argv[5];
        path_u[1]   = argv[6];
        path_u[2]   = argv[7];
        output_path = argv[8];

        printf("(SoA) %s %s %g %g %s %s %s %s %s\n",
               argv[0],
               material,
               (double)mu,
               (double)lambda,
               folder,
               path_u[0],
               path_u[1],
               path_u[2],
               output_path);
    }

    const double tick = smesh::time_seconds();

    auto            mesh      = sfem::Mesh::create_from_file(comm, smesh::Path(folder));
    const ptrdiff_t nelements = mesh->n_elements();
    const ptrdiff_t nnodes    = mesh->n_nodes();

    auto stress_buf = sfem::create_host_buffer<real_t>(nelements);
    real_t *stress  = stress_buf->data();

    if (is_AoS) {
        SFEM_ERROR("AoS not supported yet!\n");
    } else {
        std::shared_ptr<sfem::Buffer<real_t>> u_bufs[3];
        real_t                               *u[3];

        for (int d = 0; d < 3; d++) {
            u_bufs[d] = sfem::Buffer<real_t>::from_file(smesh::Path(path_u[d]));
            if (!u_bufs[d]) {
                SFEM_ERROR("Failed to read file %s\n", path_u[d]);
            }
            u[d] = u_bufs[d]->data();
        }

        neohookean_vonmises_soa(
                nelements, nnodes, mesh->elements(0)->data(), mesh->points()->data(), mu, lambda, u, stress);
    }

    stress_buf->to_file(smesh::Path(output_path));

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
    return compute_vonmises(ctx->communicator(), argc, argv);
}
