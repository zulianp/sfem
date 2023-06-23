#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "utils.h"

#include "dirichlet.h"

#include "sfem_base.h"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size != 1) {
        fprintf(stderr, "Parallel execution not supported!\n");
        return EXIT_FAILURE;
    }

    if (argc != 7) {
        fprintf(stderr,
                "usage: %s <block_rows> <block_cols> <crs_folder> <nodes.raw> <diag_value> "
                "<output_folder>\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    const int block_rows = atoi(argv[1]);
    const int block_cols = atoi(argv[2]);

    const char *crs_folder = argv[3];
    const char *nodes_path = argv[4];
    const real_t diag_value = atof(argv[5]);
    const char *output_folder = argv[6];

    double tick = MPI_Wtime();

    char rowptr_path[1024 * 10];
    char colidx_path[1024 * 10];
    char values_path[1024 * 10];

    sprintf(rowptr_path, "%s/rowptr.raw", crs_folder);
    sprintf(colidx_path, "%s/colidx.raw", crs_folder);

    ptrdiff_t _nope_, ndirichlet;
    idx_t *dirichlet_nodes = 0;
    array_create_from_file(
        comm, nodes_path, SFEM_MPI_IDX_T, (void **)&dirichlet_nodes, &_nope_, &ndirichlet);

    if (block_rows * block_cols > 1) {
        sprintf(values_path, "%s/values.*.raw", crs_folder);
        block_crs_t crs;
        block_crs_read(comm,
                       rowptr_path,
                       colidx_path,
                       values_path,
                       SFEM_MPI_COUNT_T,
                       SFEM_MPI_IDX_T,
                       SFEM_MPI_REAL_T,
                       &crs);

        assert(block_rows * block_cols == crs.block_size);
        if (block_rows * block_cols != crs.block_size) {
            fprintf(stderr, "Wrong number of blocks for block_crs\n");
            return EXIT_FAILURE;
        }

        for (int bi = 0; bi < block_rows; bi++) {
            for (int bj = 0; bj < block_cols; bj++) {
                real_t *vals = (real_t *)crs.values[bi * block_cols + bj];

                crs_constraint_nodes_to_identity(ndirichlet,
                                                 dirichlet_nodes,
                                                 (bi == bj) * diag_value,
                                                 (const idx_t *)crs.rowptr,
                                                 (const idx_t *)crs.colidx,
                                                 vals);
            }
        }

        sprintf(rowptr_path, "%s/rowptr.raw", output_folder);
        sprintf(colidx_path, "%s/colidx.raw", output_folder);
        sprintf(values_path, "%s/values.%%d.raw", output_folder);

        block_crs_write(comm, rowptr_path, colidx_path, values_path, &crs);
        // block_crs_free(&crs); //TODO

    } else {
        sprintf(values_path, "%s/values.raw", crs_folder);

        crs_t crs;
        crs_read(comm,
                 rowptr_path,
                 colidx_path,
                 values_path,
                 SFEM_MPI_COUNT_T,
                 SFEM_MPI_IDX_T,
                 SFEM_MPI_REAL_T,
                 &crs);

        crs_constraint_nodes_to_identity(ndirichlet,
                                         dirichlet_nodes,
                                         diag_value,
                                         (const idx_t *)crs.rowptr,
                                         (const idx_t *)crs.colidx,
                                         (real_t *)crs.values);


        crs_write_folder(comm, output_folder, &crs);
        crs_free(&crs);
    }

    free(dirichlet_nodes);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("TTS: %g seconds\n", tock - tick);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
