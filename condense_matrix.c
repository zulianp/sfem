#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "../matrix.io/matrixio_array.h"
#include "../matrix.io/matrixio_crs.h"
#include "../matrix.io/utils.h"

#include "sfem_base.h"

#ifdef NDEBUG
#define INLINE inline
#else
#define INLINE
#endif

ptrdiff_t read_file(MPI_Comm comm, const char *path, void **data) {
    MPI_Status status;
    MPI_Offset nbytes;
    MPI_File file;
    CATCH_MPI_ERROR(MPI_File_open(comm, path, MPI_MODE_RDONLY, MPI_INFO_NULL, &file));
    CATCH_MPI_ERROR(MPI_File_get_size(file, &nbytes));
    *data = malloc(nbytes);

    CATCH_MPI_ERROR(MPI_File_read_at_all(file, 0, *data, nbytes, MPI_CHAR, &status));
    return nbytes;
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

    const char * help = "usage: %s <crs_folder> <dirichlet_nodes.raw> [output_folder=./condensed]";

    if (argc < 3) {
        fprintf(stderr, help, argv[0]);
        return EXIT_FAILURE;
    }

    const char *output_folder = "./condensed";

    if (argc > 3) {
        output_folder = argv[3];
    }

    if (strcmp(output_folder, argv[1]) == 0) {
        fprintf(stderr, "Input and output folder are the same! Quitting!\n");
        fprintf(stderr, help, argv[0]);
        return EXIT_FAILURE;
    }

    struct stat st = {0};
    if (stat(output_folder, &st) == -1) {
        mkdir(output_folder, 0700);
    }

    printf("%s %s %s %s\n", argv[0], argv[1], argv[2], output_folder);

    double tick = MPI_Wtime();

    crs_t crs_in;
    crs_read_folder(comm, argv[1], MPI_INT, MPI_INT, MPI_DOUBLE, &crs_in);
    ptrdiff_t nnodes = crs_in.grows;

    idx_t *is_dirichlet = 0;
    ptrdiff_t new_nnodes = 0;
    {
        is_dirichlet = (idx_t *)malloc(nnodes * sizeof(idx_t));
        memset(is_dirichlet, 0, nnodes * sizeof(idx_t));

        idx_t *dirichlet_nodes = 0;

        ptrdiff_t nlocal_, ndrichlet;
        array_read(comm, argv[2], MPI_INT, (void **)&dirichlet_nodes, &nlocal_, &ndrichlet);

        new_nnodes = nnodes - ndrichlet;

        for (ptrdiff_t node = 0; node < ndrichlet; ++node) {
            idx_t i = dirichlet_nodes[node];
            assert(i < nnodes);
            is_dirichlet[i] = 1;
        }

        free(dirichlet_nodes);
    }

    const count_t *rowptr = (const count_t *)crs_in.rowptr;
    const idx_t *colidx = (const idx_t *)crs_in.colidx;
    const real_t *values = (const real_t *)crs_in.values;
    const idx_t nrows = crs_in.grows;
    const idx_t nnz = crs_in.gnnz;

    count_t *new_rowptr = (count_t *)malloc((new_nnodes + 1) * sizeof(count_t));
    new_rowptr[0] = 0;

    // change name for meaning but reuse memory by overwriting linearly
    idx_t *mapper = is_dirichlet;

    ptrdiff_t overestimated_new_nnz = 0;
    for (ptrdiff_t node = 0, new_node_idx = 0; node < nnodes; ++node) {
        if (!is_dirichlet[node]) {
            mapper[node] = new_node_idx;

            idx_t range = rowptr[node + 1] - rowptr[node];
            assert(range > 0);
            overestimated_new_nnz += range;
            new_rowptr[++new_node_idx] = range;
        } else {
            // nrows is an invalid value
            mapper[node] = nrows;
        }
    }

    idx_t *new_colidx = (idx_t *)malloc(overestimated_new_nnz * sizeof(idx_t));
    real_t *new_values = (real_t *)malloc(overestimated_new_nnz * sizeof(real_t));

    ptrdiff_t new_nnz = 0;
    for (ptrdiff_t node = 0, new_node_idx = 0; node < nnodes; ++node) {
        if (mapper[node] == nrows) continue;
        // Only valid rows

        count_t start = rowptr[node];
        count_t end = rowptr[node + 1];

        // idx_t range = end - rowptr[node];

        for (count_t k = start; k < end; ++k) {
            idx_t col = colidx[k];
            idx_t new_col = mapper[col];

            if (new_col == nrows) continue;

            // Only valid columns
            new_colidx[new_nnz] = new_col;
            new_values[new_nnz] = values[k];
            new_nnz++;
        }

        new_rowptr[++new_node_idx] = new_nnz;
    }

    free(is_dirichlet); // mapper is invalidated here!

    // Free input CRS
    crs_free(&crs_in);

    {
        crs_t crs_out;
        crs_out.rowptr = (char *)new_rowptr;
        crs_out.colidx = (char *)new_colidx;
        crs_out.values = (char *)new_values;
        crs_out.grows = new_nnodes;
        crs_out.lrows = new_nnodes;
        crs_out.lnnz = new_nnz;
        crs_out.gnnz = new_nnz;
        crs_out.start = 0;
        crs_out.rowoffset = 0;
        crs_out.rowptr_type = SFEM_MPI_COUNT_T;
        crs_out.colidx_type = SFEM_MPI_IDX_T;
        crs_out.values_type = SFEM_MPI_REAL_T;

        crs_write_folder(comm, output_folder, &crs_out);
    }

    double tock = MPI_Wtime();

    if (!rank) {
        printf("Condensed dofs: from %ld to %ld\n (nnz: %ld to %ld)\n",
               (long)nnodes,
               (long)new_nnodes,
               (long)nnz,
               (long)new_nnz);
        printf("TTS: %g seconds\n", tock - tick);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
