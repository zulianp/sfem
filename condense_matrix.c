#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../matrix.io/matrixio_array.h"
#include "../matrix.io/matrixio_crs.h"
#include "../matrix.io/utils.h"

typedef float geom_t;
typedef int idx_t;
typedef double real_t;

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

    if (argc < 3) {
        fprintf(stderr, "usage: %s <crs_folder> <dirichlet_nodes.raw> [output_folder=./condensed]", argv[0]);
        return EXIT_FAILURE;
    }

    const char *output_folder = "./condensed";

    if (argc > 3) {
        output_folder = argv[3];
    }

    if(strcmp(output_folder, argv[1])==0) {
        fprintf(stderr, "Input and output folder are the same! Quitting!\n");
        fprintf(stderr, "usage: %s <crs_folder> <dirichlet_nodes.raw> [output_folder=./condensed]", argv[0]);
        return EXIT_FAILURE;
    }

    struct stat st = {0};
    if (stat(output_folder, &st) == -1) {
        mkdir(output_folder, 0700);
    }

    double tick = MPI_Wtime();

    crs_t crs_in;
    crs_read_folder(comm, argv[1], MPI_INT, MPI_INT, MPI_DOUBLE, &crs_in);
    ptrdiff_t nnodes = crs_in.grows;

    idx_t *dof_map = 0;
    ptrdiff_t new_nnodes = 0;
    {
        dof_map = (idx_t *)malloc(nnodes * sizeof(idx_t));
        memset(dof_map, 0, nnodes * sizeof(idx_t));

        idx_t *dirichlet_nodes = 0;
        ptrdiff_t nn = read_file(comm, argv[2], (void **)&dirichlet_nodes);
        assert((nn / sizeof(idx_t)) * sizeof(idx_t) == nn);
        nn /= sizeof(idx_t);

        new_nnodes = nnodes - nn;

        for (ptrdiff_t node = 0; node < nn; ++node) {
            idx_t i = dirichlet_nodes[node];
            dof_map[i] = 1;
        }

        free(dirichlet_nodes);
    }

    idx_t *rowptr = (idx_t *)crs_in.rowptr;
    idx_t *colidx = (idx_t *)crs_in.colidx;
    idx_t *values = (idx_t *)crs_in.values;
    idx_t nrows = crs_in.grows;
    idx_t nnz = crs_in.gnnz;

    idx_t *new_rowptr = (idx_t*)malloc((new_nnodes + 1) * sizeof(idx_t));
    new_rowptr[0] = 0;

    ptrdiff_t new_nnz = 0;

    for (ptrdiff_t node = 0, new_node_idx = 0; node < nnodes; ++node) {
        if (dof_map[node]) {
            idx_t range = rowptr[node + 1] - rowptr[node];
            new_nnz += range;
            new_rowptr[++new_node_idx] = range;
        }
    }

    for (ptrdiff_t node = 0; node < new_nnodes; ++node) {
        new_rowptr[node + 1] += new_rowptr[node];
    }

    assert(new_nnz == new_rowptr[new_nnodes]);

    idx_t *new_colidx = (idx_t *)malloc(new_nnz * sizeof(idx_t));
    real_t *new_values = (real_t *)malloc(new_nnz * sizeof(real_t));

    for (ptrdiff_t node = 0, new_node_idx = 0; node < nnodes; ++node) {
        if (dof_map[node]) {
            idx_t start = rowptr[node];
            idx_t range = rowptr[node + 1] - rowptr[node];

            idx_t new_start = new_rowptr[new_node_idx];
            idx_t new_range = new_rowptr[new_node_idx + 1] - new_rowptr[new_node_idx];

            assert(new_range == range);
            assert(new_start <= start);

            memcpy(&new_colidx[new_start], &colidx[start], range * sizeof(idx_t));
            memcpy(&new_values[new_start], &values[start], range * sizeof(real_t));

            new_node_idx++;
        }
    }

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

       crs_write_folder(comm, output_folder, MPI_INT, MPI_INT, MPI_DOUBLE, &crs_out);
    }

    double tock = MPI_Wtime();

    if (!rank) {
        printf("Condensed dofs: from %ld to %ld\n", (long)nnodes, (long)new_nnodes);
        printf("TTS: %g seconds\n", tock - tick);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
