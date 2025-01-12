#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "array_dtof.h"
#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "utils.h"

#include "crs_graph.h"
#include "read_mesh.h"
#include "sfem_base.h"
#include "sfem_mesh_write.h"

#include "adj_table.h"

#include "sfem_defs.h"

#include "argsort.h"
#include "sfem_glob.hpp"

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
    char        path[SFEM_MAX_PATH_LENGTH];
    sprintf(path, "%s/i*.raw", folder);
    int nnxe = sfem::count_files(path);

    // FIXME
    enum ElemType element_type          = (enum ElemType)nnxe;
    enum ElemType element_type_for_algo = element_type;
    if (element_type == TET10) {
        element_type_for_algo = TET4;
    } else if (element_type == TRI6) {
        element_type_for_algo = TRI3;
    }

    nnxe = elem_num_nodes(element_type_for_algo);
    // Read only the data we need
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

    count_t       *adj_ptr = 0;
    element_idx_t *adj_idx = 0;

    sprintf(path, "%s/adj_ptr.raw", output_folder);
    ptrdiff_t adj_ptr_size_local, adj_ptr_size;
    array_create_from_file(comm, path, SFEM_MPI_COUNT_T, (void **)&adj_ptr, &adj_ptr_size_local, &adj_ptr_size);

    sprintf(path, "%s/adj_idx.raw", output_folder);
    ptrdiff_t ennz_local, ennz;
    array_create_from_file(comm, path, SFEM_MPI_ELEMENT_IDX_T, (void **)&adj_idx, &ennz_local, &ennz);

    int             ns    = elem_num_sides(element_type);
    element_idx_t **table = (element_idx_t **)malloc(ns * sizeof(element_idx_t *));

    for (int s = 0; s < ns; s++) {
        table[s] = (element_idx_t *)malloc(n_elements * sizeof(element_idx_t));
    }

    create_element_adj_table_from_dual_graph_soa(n_elements, n_nodes, element_type, elems, adj_ptr, adj_idx, table);

    for (int s = 0; s < ns; s++) {
        sprintf(path, "%s/a.%d.raw", output_folder, s);
        array_write(comm, path, SFEM_MPI_ELEMENT_IDX_T, table[s], n_elements, n_elements);
    }

    // Free resources

    for (int d = 0; d < nnxe; d++) {
        free(elems[d]);
    }

    for (int s = 0; s < ns; s++) {
        free(table[s]);
    }

    free(table);

    free(adj_ptr);
    free(adj_idx);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
