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

#include "sortreduce.h"
#include "sfem_glob.hpp"

#include "sfem_API.hpp"

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

    auto p1_mesh = sfem::Mesh::create_from_file(comm, folder);
    const ptrdiff_t n_elements = p1_mesh->n_elements();
    const ptrdiff_t n_nodes = p1_mesh->n_nodes();
    int p1_nxe = elem_num_nodes(p1_mesh->element_type());

    ///////////////////////////////////////////////////////////////////////////////
    // Build CRS graph of P1 mesh
    ///////////////////////////////////////////////////////////////////////////////

    count_t *rowptr = 0;
    idx_t *colidx = 0;

    // This only works for TET4 or TRI3
    build_crs_graph_for_elem_type(p1_mesh->element_type(),
                                  n_elements,
                                  n_nodes,
                                  p1_mesh->elements()->data(),
                                  &rowptr,
                                  &colidx);

    ///////////////////////////////////////////////////////////////////////////////

    mesh_t p2_mesh;
    mesh_init(&p2_mesh);

    const count_t nnz = rowptr[n_nodes];
    idx_t *p2idx = (idx_t *)malloc(nnz * sizeof(idx_t));
    memset(p2idx, 0, nnz * sizeof(idx_t));

    ptrdiff_t p2_nodes = 0;
    idx_t next_id = n_nodes;
    for (ptrdiff_t i = 0; i < n_nodes; i++) {
        const count_t begin = rowptr[i];
        const count_t end = rowptr[i + 1];

        for (count_t k = begin; k < end; k++) {
            const idx_t j = colidx[k];

            if (i < j) {
                p2_nodes += 1;
                p2idx[k] = next_id++;
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Create P2 mesh
    ///////////////////////////////////////////////////////////////////////////////

    const int n_p2_node_x_element = (p1_mesh->element_type() == TET4 ? 6 : 3);

    p2_mesh.comm = comm;
    p2_mesh.nelements = n_elements;
    p2_mesh.nnodes = n_nodes + p2_nodes;
    p2_mesh.spatial_dim = p1_mesh->spatial_dimension();

    p2_mesh.n_owned_nodes = p2_mesh.nnodes;
    p2_mesh.n_owned_elements = p2_mesh.nelements;
    p2_mesh.n_shared_elements = 0;

    p2_mesh.node_mapping = 0;
    p2_mesh.node_owner = 0;
    p2_mesh.element_mapping = 0;

    p2_mesh.element_type = elem_higher_order(p1_mesh->element_type());
    p2_mesh.elements = (idx_t **)malloc(p2_mesh.element_type * sizeof(idx_t *));
    p2_mesh.points = (geom_t **)malloc(p2_mesh.spatial_dim * sizeof(geom_t *));

    const int p2_nxe = elem_num_nodes(p2_mesh.element_type);

    for (int d = 0; d < p1_nxe; d++) {
        p2_mesh.elements[d] = p1_mesh->elements()->data()[d];
    }

    // Allocate space for p2 nodes
    for (int d = p1_nxe; d < p2_nxe; d++) {
        p2_mesh.elements[d] = (idx_t *)malloc(p2_mesh.nelements * sizeof(idx_t));
    }

    for (int d = 0; d < p2_mesh.spatial_dim; d++) {
        p2_mesh.points[d] = (geom_t *)malloc(p2_mesh.nnodes * sizeof(geom_t));

        // Copy p1 portion
        memcpy(p2_mesh.points[d], p1_mesh->points()->data()[d], n_nodes * sizeof(geom_t));
    }

    if (p1_mesh->element_type() == TET4) {
        // TODO fill p2 node indices in elements
        for (ptrdiff_t e = 0; e < p2_mesh.nelements; e++) {
            // Ordering of edges compliant to exodusII spec
            idx_t row[6];
            row[0] = MIN(p2_mesh.elements[0][e], p2_mesh.elements[1][e]);
            row[1] = MIN(p2_mesh.elements[1][e], p2_mesh.elements[2][e]);
            row[2] = MIN(p2_mesh.elements[0][e], p2_mesh.elements[2][e]);
            row[3] = MIN(p2_mesh.elements[0][e], p2_mesh.elements[3][e]);
            row[4] = MIN(p2_mesh.elements[1][e], p2_mesh.elements[3][e]);
            row[5] = MIN(p2_mesh.elements[2][e], p2_mesh.elements[3][e]);

            idx_t key[6];
            key[0] = MAX(p2_mesh.elements[0][e], p2_mesh.elements[1][e]);
            key[1] = MAX(p2_mesh.elements[1][e], p2_mesh.elements[2][e]);
            key[2] = MAX(p2_mesh.elements[0][e], p2_mesh.elements[2][e]);
            key[3] = MAX(p2_mesh.elements[0][e], p2_mesh.elements[3][e]);
            key[4] = MAX(p2_mesh.elements[1][e], p2_mesh.elements[3][e]);
            key[5] = MAX(p2_mesh.elements[2][e], p2_mesh.elements[3][e]);

            for (int l = 0; l < 6; l++) {
                const idx_t r = row[l];
                const count_t row_begin = rowptr[r];
                const count_t len_row = rowptr[r + 1] - row_begin;
                const idx_t *cols = &colidx[row_begin];
                const idx_t k = find_idx_binary_search(key[l], cols, len_row);
                p2_mesh.elements[l + p1_nxe][e] = p2idx[row_begin + k];
            }
        }

    } else if (p1_mesh->element_type() == TRI3) {
        for (ptrdiff_t e = 0; e < p2_mesh.nelements; e++) {
            // Ordering of edges compliant to exodusII spec
            idx_t row[3];
            row[0] = MIN(p2_mesh.elements[0][e], p2_mesh.elements[1][e]);
            row[1] = MIN(p2_mesh.elements[1][e], p2_mesh.elements[2][e]);
            row[2] = MIN(p2_mesh.elements[0][e], p2_mesh.elements[2][e]);

            idx_t key[3];
            key[0] = MAX(p2_mesh.elements[0][e], p2_mesh.elements[1][e]);
            key[1] = MAX(p2_mesh.elements[1][e], p2_mesh.elements[2][e]);
            key[2] = MAX(p2_mesh.elements[0][e], p2_mesh.elements[2][e]);

            for (int l = 0; l < 3; l++) {
                const idx_t r = row[l];
                const count_t row_begin = rowptr[r];
                const count_t len_row = rowptr[r + 1] - row_begin;
                const idx_t *cols = &colidx[row_begin];
                const idx_t k = find_idx_binary_search(key[l], cols, len_row);
                p2_mesh.elements[l + p1_nxe][e] = p2idx[row_begin + k];
            }
        }
    } else {
        fprintf(stderr, "Implement for element_type %d\n!", p1_mesh->element_type());
        return EXIT_FAILURE;
    }

    // Generate p2 coordinates
    for (ptrdiff_t i = 0; i < n_nodes; i++) {
        const count_t begin = rowptr[i];
        const count_t end = rowptr[i + 1];

        for (count_t k = begin; k < end; k++) {
            const idx_t j = colidx[k];

            if (i < j) {
                const idx_t nidx = p2idx[k];

                for (int d = 0; d < p2_mesh.spatial_dim; d++) {
                    geom_t xi = p2_mesh.points[d][i];
                    geom_t xj = p2_mesh.points[d][j];

                    // Midpoint
                    p2_mesh.points[d][nidx] = (xi + xj) / 2;
                }
            }
        }
    }

    { //This is a hack
        int SFEM_MAP_TO_SPHERE = 0;
        SFEM_READ_ENV(SFEM_MAP_TO_SPHERE, atoi);
        if (SFEM_MAP_TO_SPHERE) {
            float SFEM_SPERE_RADIUS = 0.5;
            SFEM_READ_ENV(SFEM_SPERE_RADIUS, atof);

            double SFEM_SPERE_TOL = 1e-8;
            SFEM_READ_ENV(SFEM_SPERE_TOL, atof);

            for (ptrdiff_t i = 0; i < n_nodes; i++) {
                const count_t begin = rowptr[i];
                const count_t end = rowptr[i + 1];

                for (count_t k = begin; k < end; k++) {
                    const idx_t j = colidx[k];

                    if (i < j) {
                        const idx_t nidx = p2idx[k];

                        geom_t r1 = 0;
                        geom_t r2 = 0;
                        geom_t mr = 0;

                        for (int d = 0; d < p2_mesh.spatial_dim; d++) {
                            const geom_t xi = p2_mesh.points[d][i];
                            const geom_t xj = p2_mesh.points[d][j];
                            r1 += xi * xi;
                            r2 += xj * xj;

                            const geom_t mxi = p2_mesh.points[d][nidx];
                            mr += mxi * mxi;
                        }

                        r1 = sqrt(r1);
                        r2 = sqrt(r2);
                        mr = r1 / sqrt(mr);

                        // On the surface
                        if (fabs(r1 - SFEM_SPERE_RADIUS) < SFEM_SPERE_TOL &&
                            fabs(r2 - SFEM_SPERE_RADIUS) < SFEM_SPERE_TOL) {
                            for (int d = 0; d < p2_mesh.spatial_dim; d++) {
                                p2_mesh.points[d][nidx] *= mr;
                            }
                        }
                    }
                }
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Write P2 mesh to disk
    ///////////////////////////////////////////////////////////////////////////////

    mesh_write(output_folder, &p2_mesh);

    // Make sure we do not delete the same array twice
    for(int d = 0; d < p1_nxe; d++) {
        p1_mesh->elements()->data()[d] = nullptr;
    }
   
    if (!rank) {
        printf("----------------------------------------\n");
        printf("mesh_p1_to_p2.c: #elements %ld, nodes #p1 %ld, #p2 %ld\n",
               (long)n_elements,
               (long)n_nodes,
               (long)p2_mesh.nnodes);
        printf("----------------------------------------\n");
    }

    mesh_destroy(&p2_mesh);

    double tock = MPI_Wtime();
    if (!rank) {
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
