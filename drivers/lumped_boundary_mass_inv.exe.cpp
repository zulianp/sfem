#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"

#include "sfem_base.hpp"
#include "boundary_mass.hpp"

#include "sfem_API.hpp"

int solve_lumped_boundary_mass_inv(const std::shared_ptr<sfem::Communicator> &comm, int argc, char *argv[]) {
    if (comm->size() != 1) {
        fprintf(stderr, "Parallel execution not supported!\n");
        return EXIT_FAILURE;
    }

    if (argc != 4) {
        fprintf(stderr, "usage: %s <folder> <in.raw> <out.raw>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *folder      = argv[1];
    const char *path_input  = argv[2];
    const char *path_output = argv[3];

    printf("%s %s %s %s\n", argv[0], folder, path_input, path_output);

    const double tick = smesh::time_seconds();

    auto            mesh       = sfem::Mesh::create_from_file(comm, smesh::Path(folder));
    const ptrdiff_t n_elements = mesh->n_elements();
    const ptrdiff_t n_nodes    = mesh->n_nodes();

    auto input = sfem::Buffer<real_t>::from_file(smesh::Path(path_input));
    if (!input) {
        SFEM_ERROR("Failed to read file %s\n", path_input);
    }

    assert((ptrdiff_t)input->size() == n_nodes);

    auto output_buf = sfem::create_host_buffer<real_t>(n_nodes);
    real_t *output  = output_buf->data();

    assemble_lumped_boundary_mass(n_elements, n_nodes, mesh->elements(0)->data(), mesh->points()->data(), output);

    for (ptrdiff_t i = 0; i < n_nodes; i++) {
        output[i] = input->data()[i] / output[i];
    }

    output_buf->to_file(smesh::Path(path_output));

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
    return solve_lumped_boundary_mass_inv(ctx->communicator(), argc, argv);
}
