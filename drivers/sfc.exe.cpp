#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "array_dtof.h"
#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "utils.h"

#include "crs_graph.h"
#include "read_mesh.h"
#include "sfem_base.h"
#include "sfem_defs.h"
#include "sfem_mesh_write.h"

#include "argsort.h"

#include "sfem_API.hpp"
#include "sfem_Env.hpp"
#include "sfem_SFC.hpp"


int sfc_reorder(const std::shared_ptr<sfem::Communicator> &comm, int argc, char *argv[]) {
    if (argc != 3) {
        if (!comm->rank()) {
            fprintf(stderr, "usage: %s <folder> <output_folder>\n", argv[0]);
        }

        return EXIT_FAILURE;
    }


    const char *folder = argv[1];
    const char *output_folder = argv[2];

    int SFEM_ORDER_WITH_COORDINATE = sfem::Env::read<int>("SFEM_ORDER_WITH_COORDINATE", -1);

    if (!comm->rank()) {
        printf("%s %s %s\n", argv[0], folder, output_folder);
        printf("SFEM_ORDER_WITH_COORDINATE=%d\n", SFEM_ORDER_WITH_COORDINATE);
    }

    double tick = MPI_Wtime();

    auto mesh = sfem::Mesh::create_from_file(comm, folder);
    auto sfc = sfem::SFC::create_from_env();
    sfc->reorder(*mesh);

    double tock = MPI_Wtime();

    if (!comm->rank()) {
        printf("----------------------------------------\n");
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return SFEM_SUCCESS;
}


int main(int argc, char *argv[]) {
    auto ctx = sfem::initialize(argc, argv);
    return sfc_reorder(ctx->communicator(), argc, argv);
}
