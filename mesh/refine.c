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

#include "sortreduce.h"

static const int refine_pattern[8][4] = {
    // Corner tests
    {0, 4, 6, 7},
    {4, 1, 5, 8},
    {6, 5, 2, 9},
    {7, 8, 9, 3},
    // Octahedron tets
    {4, 5, 6, 8},
    {7, 4, 6, 8},
    {6, 5, 9, 8},
    {7, 6, 9, 8}};

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc != 3) {
        if (!rank) {
            fprintf(stderr, "usage: %s <folder> <output_folder>", argv[0]);
        }

        return EXIT_FAILURE;
    }

    const char *output_folder = argv[2];

    if (!rank) {
        printf("%s %s %s\n", argv[0], argv[1], output_folder);
    }

    double tick = MPI_Wtime();

    struct stat st = {0};
    if (stat(output_folder, &st) == -1) {
        mkdir(output_folder, 0700);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];
    // char path[1024 * 10];

    mesh_t coarse_mesh;
    if (mesh_read(comm, folder, &coarse_mesh)) {
        return EXIT_FAILURE;
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Build CRS graph of P1 mesh
    ///////////////////////////////////////////////////////////////////////////////

    count_t *rowptr = 0;
    idx_t *colidx = 0;

    // This only works for TET4 or TRI3
    build_crs_graph(coarse_mesh.nelements, coarse_mesh.nnodes, coarse_mesh.elements, &rowptr, &colidx);

    ///////////////////////////////////////////////////////////////////////////////

    mesh_t refined_mesh;

    const count_t nnz = rowptr[coarse_mesh.nnodes];
    idx_t *p2idx = (idx_t *)malloc(nnz * sizeof(idx_t));
    memset(p2idx, 0, nnz * sizeof(idx_t));

    ptrdiff_t fine_nodes = 0;
    idx_t next_id = coarse_mesh.nnodes;
    for (ptrdiff_t i = 0; i < coarse_mesh.nnodes; i++) {
        const count_t begin = rowptr[i];
        const count_t end = rowptr[i + 1];

        for (count_t k = begin; k < end; k++) {
            const idx_t j = colidx[k];

            if (i < j) {
                fine_nodes += 1;
                p2idx[k] = next_id++;
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Create refined mesh
    ///////////////////////////////////////////////////////////////////////////////

    refined_mesh.comm = comm;
    refined_mesh.mem_space = coarse_mesh.mem_space;
    refined_mesh.nelements = coarse_mesh.nelements * 8;
    refined_mesh.nnodes = coarse_mesh.nnodes + fine_nodes;
    refined_mesh.spatial_dim = coarse_mesh.spatial_dim;

    refined_mesh.n_owned_nodes = refined_mesh.nnodes;
    refined_mesh.n_owned_elements = refined_mesh.nelements;
    refined_mesh.n_shared_elements = 0;

    refined_mesh.node_mapping = 0;
    refined_mesh.node_owner = 0;
    refined_mesh.element_mapping = 0;

    refined_mesh.element_type = coarse_mesh.element_type;
    refined_mesh.elements = (idx_t **)malloc(refined_mesh.element_type * sizeof(idx_t *));
    refined_mesh.points = (geom_t **)malloc(refined_mesh.spatial_dim * sizeof(geom_t *));

    // Allocate space for refined nodes
    for (int d = 0; d < refined_mesh.element_type; d++) {
        refined_mesh.elements[d] = (idx_t *)malloc(refined_mesh.nelements * sizeof(idx_t));

        // Copy p1 portion
        memcpy(refined_mesh.elements[d], coarse_mesh.elements[d], coarse_mesh.nelements * sizeof(idx_t));
    }

    for (int d = 0; d < refined_mesh.spatial_dim; d++) {
        refined_mesh.points[d] = (geom_t *)malloc(refined_mesh.nnodes * sizeof(geom_t));

        // Copy p1 portion
        memcpy(refined_mesh.points[d], coarse_mesh.points[d], coarse_mesh.nnodes * sizeof(geom_t));
    }

    if (coarse_mesh.element_type == 4) {
        // TODO fill p2 node indices in elements
        for (ptrdiff_t e = 0; e < coarse_mesh.nelements; e++) {
            idx_t macro_element[10];
            for (int k = 0; k < 4; k++) {
                macro_element[k] = coarse_mesh.elements[k][e];
            }

            // Ordering of edges compliant to exodusII spec
            idx_t row[6];
            row[0] = MIN(macro_element[0], macro_element[1]);
            row[1] = MIN(macro_element[1], macro_element[2]);
            row[2] = MIN(macro_element[0], macro_element[2]);
            row[3] = MIN(macro_element[0], macro_element[3]);
            row[4] = MIN(macro_element[1], macro_element[3]);
            row[5] = MIN(macro_element[2], macro_element[3]);

            idx_t key[6];
            key[0] = MAX(macro_element[0], macro_element[1]);
            key[1] = MAX(macro_element[1], macro_element[2]);
            key[2] = MAX(macro_element[0], macro_element[2]);
            key[3] = MAX(macro_element[0], macro_element[3]);
            key[4] = MAX(macro_element[1], macro_element[3]);
            key[5] = MAX(macro_element[2], macro_element[3]);

            for (int l = 0; l < 6; l++) {
                const idx_t r = row[l];
                const count_t row_begin = rowptr[r];
                const count_t len_row = rowptr[r + 1] - row_begin;
                const idx_t *cols = &colidx[row_begin];
                const idx_t k = find_idx_binary_search(key[l], cols, len_row);
                macro_element[l + 4] = p2idx[row_begin + k];
            }

            // distribute macro_element to fine_mesh
            ptrdiff_t element_offset = e * 8;
            for (int k = 0; k < 4; k++) {
                for (int sub_e = 0; sub_e < 8; sub_e++) {
                    const idx_t ik = macro_element[refine_pattern[sub_e][k]];
                    refined_mesh.elements[k][element_offset + sub_e] = ik;
                }
            }
        }

    } else {
        fprintf(stderr, "Implement for element_type %d\n!", coarse_mesh.element_type);
        return EXIT_FAILURE;
    }

    // Generate p2 coordinates
    for (ptrdiff_t i = 0; i < coarse_mesh.nnodes; i++) {
        const count_t begin = rowptr[i];
        const count_t end = rowptr[i + 1];

        for (count_t k = begin; k < end; k++) {
            const idx_t j = colidx[k];

            if (i < j) {
                const idx_t nidx = p2idx[k];

                for (int d = 0; d < refined_mesh.spatial_dim; d++) {
                    geom_t xi = refined_mesh.points[d][i];
                    geom_t xj = refined_mesh.points[d][j];

                    // Midpoint
                    refined_mesh.points[d][nidx] = (xi + xj) / 2;
                }

                // printf("%d -> %f %f %f\n", nidx, refined_mesh.points[0][nidx], refined_mesh.points[1][nidx],
                // refined_mesh.points[2][nidx]);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Write P2 mesh to disk
    ///////////////////////////////////////////////////////////////////////////////

    mesh_write(output_folder, &refined_mesh);

    // Make sure we do not delete the same array twice
    for (int d = 0; d < coarse_mesh.element_type; d++) {
        coarse_mesh.elements[d] = 0;
    }

    if (!rank) {
        printf("----------------------------------------\n");
        printf("refine.c: elements #coarse %ld, #fine %ld, nodes #coarse %ld, #fine %ld\n",
               (long)coarse_mesh.nelements,
               (long)refined_mesh.nelements,
               (long)coarse_mesh.nnodes,
               (long)refined_mesh.nnodes);
        printf("----------------------------------------\n");
    }

    mesh_destroy(&coarse_mesh);
    mesh_destroy(&refined_mesh);

    double tock = MPI_Wtime();
    if (!rank) {
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
