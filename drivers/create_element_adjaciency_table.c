#include <glob.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "../matrix.io/array_dtof.h"
#include "../matrix.io/matrixio_array.h"
#include "../matrix.io/matrixio_crs.h"
#include "../matrix.io/utils.h"

#include "crs_graph.h"
#include "read_mesh.h"
#include "sfem_base.h"
#include "sfem_mesh_write.h"

#include "extract_surface_graph.h"

#include "sfem_defs.h"

#include "argsort.h"

inline static int count_files(const char *pattern) {
    glob_t gl;
    glob(pattern, GLOB_MARK, NULL, &gl);

    int n_files = gl.gl_pathc;

    printf("n_files (%d):\n", n_files);
    for (int np = 0; np < n_files; np++) {
        printf("%s\n", gl.gl_pathv[np]);
    }

    globfree(&gl);
    return n_files;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc < 2) {
        if (!rank) {
            fprintf(stderr, "usage: %s <folder> <output_folder>\n", argv[0]);
        }

        return EXIT_FAILURE;
    }

    if (size != 1) {
        fprintf(stderr, "Only serial runs supported!\n");
        return EXIT_FAILURE;
    }

    const char *output_folder = argv[2];

    {
        struct stat st = {0};
        if (stat(output_folder, &st) == -1) {
            mkdir(output_folder, 0700);
        }
    }

    if (!rank) {
        printf("%s %s %s\n", argv[0], argv[1], output_folder);
    }

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];
    char path[SFEM_MAX_PATH_LENGTH];
    sprintf(path, "%s/i*.raw", folder);
    int nnxe = count_files(path);
    idx_t **elems = (idx_t **)malloc(sizeof(idx_t *) * nnxe);

    ptrdiff_t n_local_elements, n_elements;
    mesh_read_elements(comm, nnxe, folder, elems, &n_local_elements, &n_elements);

    ptrdiff_t n_nodes = 0;
    for (int d = 0; d < nnxe; d++) {
        for (ptrdiff_t i = 0; i < n_elements; i++) {
            n_nodes = MAX(n_nodes, elems[d][i]);
        }
    }

    n_nodes += 1;

    count_t *adj_ptr = 0;
    element_idx_t *adj_idx = 0;

    sprintf(path, "%s/adj_ptr.raw", output_folder);
    ptrdiff_t adj_ptr_size_local, adj_ptr_size;
    array_create_from_file(comm, path, SFEM_MPI_COUNT_T, (void**)&adj_ptr, &adj_ptr_size_local, &adj_ptr_size);

    sprintf(path, "%s/adj_idx.raw", output_folder);
    ptrdiff_t ennz_local, ennz;
    array_create_from_file(comm, path, SFEM_MPI_ELEMENT_IDX_T, (void**)&adj_idx, &ennz_local, &ennz);

    for (int d = 0; d < nnxe; d++) {
        free(elems[d]);
    }

    free(adj_ptr);
    free(adj_idx);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
