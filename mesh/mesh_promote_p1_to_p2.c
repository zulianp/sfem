// #include <math.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>
// #include <sys/stat.h>

// #include "array_dtof.h"
// #include "matrixio_array.h"
// #include "matrixio_crs.h"
// #include "utils.h"

// #include "crs_graph.h"
// #include "read_mesh.h"
// #include "sfem_base.h"
// #include "sfem_mesh_write.h"

// #include "sortreduce.h"

// void mesh_promote_p1_to_p2(const mesh_t *const p1_mesh, mesh_t *const p2_mesh, const int
// steal_p1_nodes) {
//     ///////////////////////////////////////////////////////////////////////////////
//     // Build CRS graph of P1 mesh
//     ///////////////////////////////////////////////////////////////////////////////

//     count_t *rowptr = 0;
//     idx_t *colidx = 0;

//     // This only works for TET4 or TRI3
//     build_crs_graph(p1_mesh->nelements, p1_mesh->nnodes, p1_mesh->elements, &rowptr, &colidx);

//     ///////////////////////////////////////////////////////////////////////////////

//     mesh_t p2_mesh;

//     const count_t nnz = rowptr[p1_mesh->nnodes];
//     idx_t *p2idx = (idx_t *)malloc(nnz * sizeof(idx_t));
//     memset(p2idx, 0, nnz * sizeof(idx_t));

//     ptrdiff_t p2_nodes = 0;
//     idx_t next_id = p1_mesh->nnodes;
//     for (ptrdiff_t i = 0; i < p1_mesh->nnodes; i++) {
//         const count_t begin = rowptr[i];
//         const count_t end = rowptr[i + 1];

//         for (count_t k = begin; k < end; k++) {
//             const idx_t j = colidx[k];

//             if (i < j) {
//                 p2_nodes += 1;
//                 p2idx[k] = next_id++;
//             }
//         }
//     }

//     ///////////////////////////////////////////////////////////////////////////////
//     // Create P2 mesh
//     ///////////////////////////////////////////////////////////////////////////////

//     const int n_p2_node_x_element = (p1_mesh->element_type == 4 ? 6 : 3);

//     p2_mesh->comm = comm;
//     p2_mesh->mem_space = p1_mesh->mem_space;
//     p2_mesh->nelements = p1_mesh->nelements;
//     p2_mesh->nnodes = p1_mesh->nnodes + p2_nodes;
//     p2_mesh->spatial_dim = p1_mesh->spatial_dim;

//     p2_mesh->n_owned_nodes = p2_mesh->nnodes;
//     p2_mesh->n_owned_elements = p2_mesh->nelements;
//     p2_mesh->n_shared_elements = 0;

//     p2_mesh->node_mapping = 0;
//     p2_mesh->node_owner = 0;
//     p2_mesh->element_mapping = 0;

//     p2_mesh->element_type = p1_mesh->element_type + n_p2_node_x_element;
//     p2_mesh->elements = (idx_t **)malloc(p2_mesh->element_type * sizeof(idx_t *));
//     p2_mesh->points = (geom_t **)malloc(p2_mesh->spatial_dim * sizeof(geom_t *));

//     if(steal_p1_nodes) {
//         for (int d = 0; d < p1_mesh->element_type; d++) {
//             p2_mesh->elements[d] = p1_mesh->elements[d];
//         }
//     } else {
//         for (int d = 0; d < p1_mesh->element_type; d++) {
//             p2_mesh->elements[d] =  (idx_t *)malloc(p2_mesh->nelements * sizeof(idx_t));
//             memcpy(p2_mesh->elements[d],  p1_mesh->elements[d], p2_mesh->nelements *
//             sizeof(idx_t));
//         }
//     }

//     // Allocate space for p2 nodes
//     for (int d = p1_mesh->element_type; d < p2_mesh->element_type; d++) {
//         p2_mesh->elements[d] = (idx_t *)malloc(p2_mesh->nelements * sizeof(idx_t));
//     }

//     for (int d = 0; d < p2_mesh->spatial_dim; d++) {
//         p2_mesh->points[d] = (geom_t *)malloc(p2_mesh->nnodes * sizeof(geom_t));

//         // Copy p1 portion
//         memcpy(p2_mesh->points[d], p1_mesh->points[d], p1_mesh->nnodes * sizeof(geom_t));
//     }

//     if (p1_mesh->element_type == 4) {
//         // TODO fill p2 node indices in elements
//         for (ptrdiff_t e = 0; e < p2_mesh->nelements; e++) {
//             // Ordering of edges compliant to exodusII spec
//             idx_t row[6];
//             row[0] = MIN(p2_mesh->elements[0][e], p2_mesh->elements[1][e]);
//             row[1] = MIN(p2_mesh->elements[1][e], p2_mesh->elements[2][e]);
//             row[2] = MIN(p2_mesh->elements[0][e], p2_mesh->elements[2][e]);
//             row[3] = MIN(p2_mesh->elements[0][e], p2_mesh->elements[3][e]);
//             row[4] = MIN(p2_mesh->elements[1][e], p2_mesh->elements[3][e]);
//             row[5] = MIN(p2_mesh->elements[2][e], p2_mesh->elements[3][e]);

//             idx_t key[6];
//             key[0] = MAX(p2_mesh->elements[0][e], p2_mesh->elements[1][e]);
//             key[1] = MAX(p2_mesh->elements[1][e], p2_mesh->elements[2][e]);
//             key[2] = MAX(p2_mesh->elements[0][e], p2_mesh->elements[2][e]);
//             key[3] = MAX(p2_mesh->elements[0][e], p2_mesh->elements[3][e]);
//             key[4] = MAX(p2_mesh->elements[1][e], p2_mesh->elements[3][e]);
//             key[5] = MAX(p2_mesh->elements[2][e], p2_mesh->elements[3][e]);

//             for (int l = 0; l < 6; l++) {
//                 const idx_t r = row[l];
//                 const count_t row_begin = rowptr[r];
//                 const count_t len_row = rowptr[r + 1] - row_begin;
//                 const idx_t *cols = &colidx[row_begin];
//                 const idx_t k = find_idx_binary_search(key[l], cols, len_row);
//                 p2_mesh->elements[l + p1_mesh->element_type][e] = p2idx[row_begin + k];

//                 // printf("%d, %d -> %d\n", r, key[l], p2idx[row_begin + k]);
//             }

//             // printf("\n");
//         }

//     } else {
//         fprintf(stderr, "Implement for element_type %d\n!", p1_mesh->element_type);
//         return EXIT_FAILURE;
//     }

//     // Generate p2 coordinates
//     for (ptrdiff_t i = 0; i < p1_mesh->nnodes; i++) {
//         const count_t begin = rowptr[i];
//         const count_t end = rowptr[i + 1];

//         for (count_t k = begin; k < end; k++) {
//             const idx_t j = colidx[k];

//             if (i < j) {
//                 const idx_t nidx = p2idx[k];

//                 for (int d = 0; d < p2_mesh->spatial_dim; d++) {
//                     geom_t xi = p2_mesh->points[d][i];
//                     geom_t xj = p2_mesh->points[d][j];

//                     // Midpoint
//                     p2_mesh->points[d][nidx] = (xi + xj) / 2;
//                 }

//                 // printf("%d -> %f %f %f\n", nidx, p2_mesh->points[0][nidx],
//                 p2_mesh->points[1][nidx],
//                 // p2_mesh->points[2][nidx]);
//             }
//         }
//     }

//     ///////////////////////////////////////////////////////////////////////////////
//     // Write P2 mesh to disk
//     ///////////////////////////////////////////////////////////////////////////////

//     mesh_write(output_folder, &p2_mesh);

//     if (steal_p1_nodes) {
//         // Make sure we do not delete the same array twice
//         for (int d = 0; d < p1_mesh->element_type; d++) {
//             p1_mesh->elements[d] = 0;
//         }
//     }

//     if (!rank) {
//         printf("----------------------------------------\n");
//         printf("mesh_p1_to_p2.c: #elements %ld, nodes #p1 %ld, #p2 %ld\n",
//                (long)p1_mesh->nelements,
//                (long)p1_mesh->nnodes,
//                (long)p2_mesh->nnodes);
//         printf("----------------------------------------\n");
//     }

//     mesh_destroy(&p1_mesh);
//     mesh_destroy(&p2_mesh);

//     double tock = MPI_Wtime();
//     if (!rank) {
//         printf("TTS:\t\t\t%g seconds\n", tock - tick);
//     }

//     return MPI_Finalize();
// }
