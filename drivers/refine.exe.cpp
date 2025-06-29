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
#include "sfem_defs.h"
#include "sfem_mesh_write.h"

#include "sfem_glob.hpp"
#include "sortreduce.h"

#include "sfem_API.hpp"

static const int tet4_refine_pattern[8][4] = {
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

static const int tri3_refine_pattern[4][3] = {
        // Corner triangles
        {0, 3, 5},
        {3, 1, 4},
        {5, 4, 2},
    // Center triangle
    {3, 4, 5}};

static const int quad4_refine_pattern[4][4] = {
    {0, 4, 8, 7},
    {4, 1, 5, 8},
    {8, 5, 2, 6},
    {7, 8, 6, 3}};

static const int quad4_remove_connections[4][1] = {
    {2},
    {3},
    {0},
    {1}
};

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc != 3) {
        if (!rank) {
            fprintf(stderr, "usage: %s <folder> <output_folder>\n", argv[0]);
        }

        return EXIT_FAILURE;
    }

    const char *output_folder = argv[2];

    if (!rank) {
        printf("%s %s %s\n", argv[0], argv[1], output_folder);
    }

    double tick = MPI_Wtime();
    sfem::create_directory(output_folder);

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];
    // char path[1024 * 10];

    auto            coarse_mesh         = sfem::Mesh::create_from_file(sfem::Communicator::wrap(comm), folder);
    const ptrdiff_t n_elements          = coarse_mesh->n_elements();
    const ptrdiff_t n_nodes             = coarse_mesh->n_nodes();
    const int       n_nodes_per_element = coarse_mesh->n_nodes_per_element();
    const int       spatial_dim         = coarse_mesh->spatial_dimension();

    ///////////////////////////////////////////////////////////////////////////////
    // Build CRS graph of P1 mesh
    ///////////////////////////////////////////////////////////////////////////////

    count_t *rowptr = 0;
    idx_t   *colidx = 0;

    // This only works for TET4 or TRI3
    build_crs_graph_for_elem_type(
            coarse_mesh->element_type(), n_elements, n_nodes, coarse_mesh->elements()->data(), &rowptr, &colidx);

    if (coarse_mesh->element_type() == QUAD4) {
        // TODO Clean up the extra connectivity
    }

    ///////////////////////////////////////////////////////////////////////////////

    const count_t nnz   = rowptr[n_nodes];
    idx_t        *p2idx = (idx_t *)malloc(nnz * sizeof(idx_t));
    memset(p2idx, 0, nnz * sizeof(idx_t));

    ptrdiff_t fine_nodes = 0;
    idx_t     next_id    = n_nodes;
    for (ptrdiff_t i = 0; i < n_nodes; i++) {
        const count_t begin = rowptr[i];
        const count_t end   = rowptr[i + 1];

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

    auto refined_elements_buffer =
            sfem::create_host_buffer<idx_t>(n_nodes_per_element, n_elements * (coarse_mesh->element_type() == TET4 ? 8 : 4));
    auto refined_points_buffer = sfem::create_host_buffer<geom_t>(spatial_dim, coarse_mesh->n_nodes() + fine_nodes);

    auto refined_mesh = std::make_shared<sfem::Mesh>(sfem::Communicator::wrap(comm),
                                                     spatial_dim,
                                                     coarse_mesh->element_type(),
                                                     refined_elements_buffer->extent(1),
                                                     refined_elements_buffer,
                                                     refined_points_buffer->extent(1),
                                                     refined_points_buffer);

    auto refined_elements = refined_elements_buffer->data();
    auto refined_points   = refined_points_buffer->data();

    auto coarse_elements = coarse_mesh->elements()->data();
    auto coarse_points   = coarse_mesh->points()->data();

    // Allocate space for refined nodes
    for (int d = 0; d < n_nodes_per_element; d++) {
        // Copy p1 portion
        memcpy(refined_elements[d], coarse_elements[d], n_elements * sizeof(idx_t));
    }

    for (int d = 0; d < spatial_dim; d++) {
        // Copy p1 portion
        memcpy(refined_points[d], coarse_points[d], n_nodes * sizeof(geom_t));
    }

  

    if (coarse_mesh->element_type() == TET4) {
        // TODO fill p2 node indices in elements
        for (ptrdiff_t e = 0; e < n_elements; e++) {
            idx_t macro_element[10];
            for (int k = 0; k < 4; k++) {
                macro_element[k] = coarse_elements[k][e];
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
                const idx_t   r         = row[l];
                const count_t row_begin = rowptr[r];
                const count_t len_row   = rowptr[r + 1] - row_begin;
                const idx_t  *cols      = &colidx[row_begin];
                const idx_t   k         = find_idx_binary_search(key[l], cols, len_row);
                macro_element[l + 4]    = p2idx[row_begin + k];
            }

            // distribute macro_element to fine_mesh
            ptrdiff_t element_offset = e * 8;
            for (int k = 0; k < 4; k++) {
                for (int sub_e = 0; sub_e < 8; sub_e++) {
                    const idx_t ik                              = macro_element[tet4_refine_pattern[sub_e][k]];
                    refined_elements[k][element_offset + sub_e] = ik;
                }
            }
        }

    } else if (coarse_mesh->element_type() == TRI3) {
        // TODO fill p2 node indices in elements
        for (ptrdiff_t e = 0; e < n_elements; e++) {
            idx_t macro_element[6];
            for (int k = 0; k < 3; k++) {
                macro_element[k] = coarse_elements[k][e];
            }

            // Ordering of edges compliant to exodusII spec
            idx_t row[3];
            row[0] = MIN(macro_element[0], macro_element[1]);
            row[1] = MIN(macro_element[1], macro_element[2]);
            row[2] = MIN(macro_element[0], macro_element[2]);

            idx_t key[3];
            key[0] = MAX(macro_element[0], macro_element[1]);
            key[1] = MAX(macro_element[1], macro_element[2]);
            key[2] = MAX(macro_element[0], macro_element[2]);

            for (int l = 0; l < 3; l++) {
                const idx_t   r         = row[l];
                const count_t row_begin = rowptr[r];
                const count_t len_row   = rowptr[r + 1] - row_begin;
                const idx_t  *cols      = &colidx[row_begin];
                const idx_t   k         = find_idx_binary_search(key[l], cols, len_row);
                macro_element[l + 3]    = p2idx[row_begin + k];
            }

            // distribute macro_element to fine_mesh
            ptrdiff_t element_offset = e * 4;
            for (int k = 0; k < 3; k++) {
                for (int sub_e = 0; sub_e < 4; sub_e++) {
                    const idx_t ik                              = macro_element[tri3_refine_pattern[sub_e][k]];
                    refined_elements[k][element_offset + sub_e] = ik;
                }
            }
        }
    } else {
        fprintf(stderr, "Implement for element_type %d\n!", coarse_mesh->element_type());
        return EXIT_FAILURE;
    }

    // Generate p2 coordinates
    for (ptrdiff_t i = 0; i < n_nodes; i++) {
        const count_t begin = rowptr[i];
        const count_t end   = rowptr[i + 1];

        for (count_t k = begin; k < end; k++) {
            const idx_t j = colidx[k];

            if (i < j) {
                const idx_t nidx = p2idx[k];

                for (int d = 0; d < spatial_dim; d++) {
                    geom_t xi = refined_points[d][i];
                    geom_t xj = refined_points[d][j];

                    // Midpoint
                    refined_points[d][nidx] = (xi + xj) / 2;
                }
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Write P2 mesh to disk
    ///////////////////////////////////////////////////////////////////////////////

    refined_mesh->write(output_folder);

    // Make sure we do not delete the same array twice
    for (int d = 0; d < n_nodes_per_element; d++) {
        coarse_elements[d] = 0;
    }

    if (!rank) {
        printf("----------------------------------------\n");
        printf("refine.c: elements #coarse %ld, #fine %ld, nodes #coarse %ld, #fine %ld\n",
               (long)n_elements,
               (long)refined_mesh->n_elements(),
               (long)n_nodes,
               (long)refined_mesh->n_nodes());
        printf("----------------------------------------\n");
    }

 

    double tock = MPI_Wtime();
    if (!rank) {
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
