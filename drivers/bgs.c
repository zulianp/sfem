#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "utils.h"

#include "dirichlet.h"

#include "sfem_base.h"

int bgs(const ptrdiff_t nnodes,
        const int block_rows,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        real_t **const SFEM_RESTRICT values,
        real_t **const rhs,
        real_t **const x,
        const int num_sweeps) {
    for (int s = 0; s < num_sweeps; s++) {
        for (ptrdiff_t i = 0; i < nnodes; i++) {
            const count_t r_begin = rowptr[i];
            const count_t r_end = rowptr[i + 1];
            const count_t r_extent = r_end - r_begin;
            const idx_t *const r_colidx = colidx[r_begin];

            count_t diag_idx = -1;
            for(count_t k = 0; k < r_extent; k++) {
                // 
                const idx_t col = r_colidx[k];

                if(col == i) {
                    diag_idx = k;
                } else {

                }
            }
        }
    }

    return 0;
}

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

    if (argc != 5) {
        fprintf(stderr, "usage: %s <block_rows> <crs_folder> <rhs.raw> <output.raw>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const int block_rows = atoi(argv[1]);
    const char *crs_folder = argv[2];
    const char *rhs_prefix = argv[3];
    const char *output_path = argv[4];

    struct stat st = {0};
    if (stat(output_path, &st) == -1) {
        mkdir(output_path, 0700);
    }

    const double tick = MPI_Wtime();

    char rowptr_path[10240];
    char colidx_path[10240];
    char values_path[10240];
    char rhs_path[10240];

    sprintf(rowptr_path, "%s/rowptr.raw", crs_folder);
    sprintf(colidx_path, "%s/colidx.raw", crs_folder);
    sprintf(values_path, "%s/values.*.raw", crs_folder);

    ptrdiff_t _nope_, ndofs;
    real_t **rhs = (real_t **)malloc(block_rows * sizeof(real_t *));
    for (int d = 0; d < block_rows; d++) {
        sprintf(rhs_path, "%s.%d.raw", rhs_prefix, d);
        array_create_from_file(comm, rhs_path, SFEM_MPI_REAL_T, (void **)&rhs[d], &_nope_, &ndofs);
    }

    block_crs_t crs;
    block_crs_read(comm,
                   rowptr_path,
                   colidx_path,
                   values_path,
                   SFEM_MPI_COUNT_T,
                   SFEM_MPI_IDX_T,
                   SFEM_MPI_REAL_T,
                   &crs);

    assert(block_rows * block_rows == crs.block_size);
    if (block_rows * block_rows != crs.block_size) {
        fprintf(stderr, "Wrong number of blocks for block_crs\n");
        return EXIT_FAILURE;
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Free resources
    ///////////////////////////////////////////////////////////////////////////////

    block_crs_free(&crs);
    free(rhs);

    const double tock = MPI_Wtime();

    if (!rank) {
        printf("TTS: %g seconds\n", tock - tick);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
