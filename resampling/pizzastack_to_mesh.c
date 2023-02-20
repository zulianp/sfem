#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrix.io/array_dtof.h"
#include "matrix.io/matrixio_array.h"
#include "matrix.io/matrixio_crs.h"
#include "matrix.io/utils.h"

#include "crs_graph.h"
#include "read_mesh.h"
#include "sfem_base.h"
#include "sfem_mesh_write.h"

typedef struct {
    ptrdiff_t *cell_ptr;
    ptrdiff_t *idx;
} cell_list_1D;

static void histogram(const ptrdiff_t nnodes,
                      const geom_t *x,
                      const geom_t shift,
                      const geom_t scaling,
                      const ptrdiff_t nbins,
                      ptrdiff_t *histo) {
    memset(histo, 0, nbins * sizeof(ptrdiff_t));
    for (ptrdiff_t i = 0; i < nnodes; ++i) {
        ptrdiff_t idx = scaling * (x[i] + shift);
        histo[idx] += 1;
    }
}

static void bounding_intervals(const ptrdiff_t n_elements,
                               const int n_nodes_per_elem,
                               idx_t **const elems,
                               geom_t *const SFEM_RESTRICT x,
                               geom_t *const SFEM_RESTRICT bi_min,
                               geom_t *const SFEM_RESTRICT bi_max) {
    for (ptrdiff_t e = 0; e < n_elements; e++) {
        const idx_t i = elems[0][e];
        bi_min[e] = x[i];
        bi_max[e] = x[i];
    }

    for (int d = 1; d < n_nodes_per_elem; ++d) {
        const idx_t *idx = elems[d];

        for (ptrdiff_t e = 1; e < n_elements; e++) {
            const idx_t i = idx[e];
            bi_min[e] = MIN(bi_min[e], x[i]);
            bi_max[e] = MAX(bi_max[e], x[i]);
        }
    }
}

void resample_box_to_tetra_mesh(const count_t n[3],
                                const count_t ld[3],
                                const real_t *SFEM_RESTRICT box_field,
                                const ptrdiff_t n_elements,
                                const ptrdiff_t n_nodes,
                                idx_t **const elems,
                                geom_t **const xyz,
                                real_t *const SFEM_RESTRICT mesh_field) {
    geom_t *zbi_min = (geom_t *)malloc(n_elements * sizeof(geom_t));
    geom_t *zbi_max = (geom_t *)malloc(n_elements * sizeof(geom_t));

    bounding_intervals(n_elements, 4, elems, xyz[2], zbi_min, zbi_max);

    
    free(zbi_min);
    free(zbi_max);
}

// void cell_list_1D_create(cell_list_1D *cl, 
//     const ptrdiff_t n, 
//     const geom_t *SFEM_RESTRICT zbi_min,
//     const geom_t *SFEM_RESTRICT zbi_max) {
//     ptrdiff_t *zhisto = malloc(nbins * sizeof(ptrdiff_t));
//     histogram(x, shift, scaling, nbins, zhisto);
// }


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc < 2) {
        if (!rank) {
            fprintf(stderr, "usage: %s <folder> [output_folder=./]", argv[0]);
        }

        return EXIT_FAILURE;
    }

    const char *output_folder = "./";
    if (argc > 2) {
        output_folder = argv[2];
    }

    if (!rank) {
        printf("%s %s %s\n", argv[0], argv[1], output_folder);
    }

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];
    // char path[1024 * 10];

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    MPI_Barrier(comm);

    // double tack = MPI_Wtime();

    char output_path[2048];
    sprintf(output_path, "%s/part_%0.5d", output_folder, rank);
    // Everyone independent
    mesh.comm = MPI_COMM_SELF;
    mesh_write(output_path, &mesh);

    for (int r = 0; r < size; ++r) {
        if (r == rank) {
            printf("[%d] #elements %ld #nodes %ld #owned_nodes %ld #shared_elements %ld\n",
                   rank,
                   (long)mesh.nelements,
                   (long)mesh.nnodes,
                   (long)mesh.n_owned_nodes,
                   (long)mesh.n_shared_elements);
        }

        fflush(stdout);

        MPI_Barrier(comm);
    }

    MPI_Barrier(comm);

    mesh_destroy(&mesh);
    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
