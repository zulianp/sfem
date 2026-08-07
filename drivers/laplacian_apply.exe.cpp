#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"

#include "sfem_base.hpp"
#include "sfem_defs.hpp"

#include "laplacian.hpp"
#include "tet4_fff.hpp"

#include "sfem_API.hpp"

int solve_laplacian_apply(const std::shared_ptr<sfem::Communicator> &comm, int argc, char *argv[]) {
    if (comm->size() != 1) {
        fprintf(stderr, "Parallel execution not supported!\n");
        return EXIT_FAILURE;
    }

    if (argc < 4) {
        fprintf(stderr, "usage: %s <folder> <x.raw> <y.raw>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int SFEM_REPEAT    = 1;
    int SFEM_USE_OPT   = 1;
    int SFEM_USE_MACRO = 1;

    SFEM_READ_ENV(SFEM_REPEAT, atoi);
    SFEM_READ_ENV(SFEM_USE_OPT, atoi);
    SFEM_READ_ENV(SFEM_USE_MACRO, atoi);

    const char *folder      = argv[1];
    const char *path_f      = argv[2];
    const char *path_output = argv[3];

    printf("%s %s %s %s\n", argv[0], folder, path_f, path_output);

    const double tick = smesh::time_seconds();

    auto            mesh       = sfem::Mesh::create_from_file(comm, smesh::Path(folder));
    const ptrdiff_t n_elements = mesh->n_elements();
    const ptrdiff_t n_nodes    = mesh->n_nodes();

    auto element_type = mesh->element_type(0);
    if (SFEM_USE_MACRO) {
        element_type = macro_type_variant(element_type);
    }

    std::shared_ptr<sfem::Buffer<real_t>> x_buf;
    real_t                               *x = nullptr;

    if (strcmp("gen:ones", path_f) == 0) {
        x_buf = sfem::create_host_buffer<real_t>(n_nodes);
        x     = x_buf->data();
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n_nodes; ++i) {
            x[i] = 1;
        }
    } else {
        x_buf = sfem::Buffer<real_t>::from_file(smesh::Path(path_f));
        if (!x_buf) {
            SFEM_ERROR("Failed to read file %s\n", path_f);
        }
        x = x_buf->data();
    }

    auto y_buf = sfem::create_host_buffer<real_t>(n_nodes);
    real_t *y  = y_buf->data();

    if (!laplacian_is_opt(element_type)) {
        SFEM_USE_OPT = 0;
    }

    fff_t fff;
    if (SFEM_USE_OPT) {
        tet4_fff_create(&fff, n_elements, mesh->elements(0)->data(), mesh->points()->data());
    }

    const double spmv_tick = smesh::time_seconds();

    for (int repeat = 0; repeat < SFEM_REPEAT; repeat++) {
        if (SFEM_USE_OPT) {
            laplacian_apply_opt(element_type, fff.nelements, fff.elements, fff.data, x, y);
        } else {
            laplacian_apply(element_type, n_elements, n_nodes, mesh->elements(0)->data(), mesh->points()->data(), x, y);
        }
    }

    const double spmv_tock = smesh::time_seconds();

    y_buf->to_file(smesh::Path(path_output));

    if (SFEM_USE_OPT) {
        tet4_fff_destroy(&fff);
    }

    const double tock   = smesh::time_seconds();
    const float  TTS    = (float)(tock - tick);
    const float  TTS_op = (float)((spmv_tock - spmv_tick) / SFEM_REPEAT);

    if (!comm->rank()) {
        printf("----------------------------------------\n");
        printf("SUMMARY (%s): %s\n", type_to_string(element_type), argv[0]);
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld\n", (long)n_elements, (long)n_nodes);
        printf("Operator TTS:\t\t%.4f\t[s]\n", TTS_op);
        printf("Operator throughput:\t%.1f\t[ME/s]\n", 1e-6f * (float)n_elements / TTS_op);
        printf("Operator throughput:\t%.1f\t[MDOF/s]\n", 1e-6f * (float)n_nodes / TTS_op);
        printf("Total:\t\t\t%.4f\t[s]\n", TTS);
        printf("----------------------------------------\n");
    }

    return 0;
}

int main(int argc, char *argv[]) {
    auto ctx = sfem::initialize(argc, argv);
    return solve_laplacian_apply(ctx->communicator(), argc, argv);
}
