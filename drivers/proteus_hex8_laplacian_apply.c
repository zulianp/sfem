#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "array_dtof.h"
#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "utils.h"

#include "crs_graph.h"
#include "sfem_base.h"
#include "sfem_defs.h"

#include "read_mesh.h"

#include "laplacian.h"
// #include "proteus_hex8_laplacian.h"
// #include "hex8_fff.h"

#include "sortreduce.h"

#include "adj_table.h"

static SFEM_INLINE int hex8_linear_search(const idx_t target,
                                          const idx_t *const arr,
                                          const int size) {
    int i;
    for (i = 0; i < size - 4; i += 4) {
        if (arr[i] == target) return i;
        if (arr[i + 1] == target) return i + 1;
        if (arr[i + 2] == target) return i + 2;
        if (arr[i + 3] == target) return i + 3;
    }
    for (; i < size; i++) {
        if (arr[i] == target) return i;
    }
    return -1;
}

static SFEM_INLINE int hex8_find_col(const idx_t key, const idx_t *const row, const int lenrow) {
    if (lenrow <= 32) {
        return hex8_linear_search(key, row, lenrow);
    } else {
        // Use this for larger number of dofs per row
        return find_idx_binary_search(key, row, lenrow);
    }
}

static SFEM_INLINE void hex8_find_corner_cols(const idx_t *targets,
                                              const idx_t *const row,
                                              const int lenrow,
                                              int *ks) {
    if (lenrow > 32) {
        for (int d = 0; d < 3; ++d) {
            ks[d] = hex8_find_col(targets[d], row, lenrow);
        }
    } else {
#pragma unroll(3)
        for (int d = 0; d < 3; ++d) {
            ks[d] = 0;
        }

        for (int i = 0; i < lenrow; ++i) {
#pragma unroll(3)
            for (int d = 0; d < 3; ++d) {
                ks[d] += row[i] < targets[d];
            }
        }
    }
}

#define SFEM_INVALID_IDX (-1)

int proteus_hex8_nxe(int level) {
    const int corners = 8;
    const int edge_nodes = 12 * (level - 1);
    const int face_nodes = 6 * (level - 1) * (level - 1);
    const int vol_nodes = (level - 1) * (level - 1) * (level - 1);
    return corners + edge_nodes + face_nodes + vol_nodes;
}

int proteus_hex8_txe(int level) { return level * level * level; }

int proteus_hex8_lidx(const int L, const int x, const int y, const int z) {
    int Lp1 = L + 1;
    int ret = z * (Lp1 * Lp1) + y * Lp1 + x;
    assert(ret < proteus_hex8_nxe(L));
    assert(ret >= 0);
    return ret;
}

static void proteus_hex8_create_full_idx(const int L, mesh_t *mesh, idx_t **elements) {
    assert(L >= 2);

    double tick = MPI_Wtime();

    const int nxe = proteus_hex8_nxe(L);

    // 1) Get the node indices from the TET4 mesh
    int corner_lidx[8] = {// Bottom
                          proteus_hex8_lidx(L, 0, 0, 0),
                          proteus_hex8_lidx(L, L, 0, 0),
                          proteus_hex8_lidx(L, 0, L, 0),
                          proteus_hex8_lidx(L, L, L, 0),
                          // Top
                          proteus_hex8_lidx(L, 0, 0, L),
                          proteus_hex8_lidx(L, L, 0, L),
                          proteus_hex8_lidx(L, 0, L, L),
                          proteus_hex8_lidx(L, L, L, L)};

#ifndef NDEBUG
    for (int i = 0; i < 8; i++) {
        assert(corner_lidx[i] < nxe);
    }
#endif

    int *coords[3];
    for (int d = 0; d < 3; d++) {
        coords[d] = malloc(nxe * sizeof(int));
    }

    for (int zi = 0; zi <= L; zi++) {
        for (int yi = 0; yi <= L; yi++) {
            for (int xi = 0; xi <= L; xi++) {
                int lidx = proteus_hex8_lidx(L, xi, yi, zi);
                assert(lidx < nxe);
                coords[0][lidx] = xi;
                coords[1][lidx] = yi;
                coords[2][lidx] = zi;
            }
        }
    }

    // ------------------------------
    // Corner nodes
    // ------------------------------
    for (int d = 0; d < 8; d++) {
        for (ptrdiff_t e = 0; e < mesh->nelements; e++) {
            elements[corner_lidx[d]][e] = mesh->elements[d][e];
            // printf("%d) e[%d][%ld] = %d\n", d, corner_lidx[d], e, elements[corner_lidx[d]][e]);
        }
    }

    idx_t index_base = mesh->nnodes;

    // 2) Compute the unique edge-node indices using the CRSGraph
    // A unique edge index can be used and use the multiple to store all indices
    // as consecutive

    // Number of nodes in the edge interior
    ptrdiff_t nxedge = L - 1;  // L == 0 (is this correct?)

    if (nxedge) {
        count_t *rowptr;
        idx_t *colidx;
        build_crs_graph_for_elem_type(mesh->element_type,
                                      mesh->nelements,
                                      mesh->nnodes,
                                      mesh->elements,
                                      &rowptr,
                                      &colidx);

        ptrdiff_t nedges = (rowptr[mesh->nnodes] - mesh->nnodes) / 2;

        ptrdiff_t nnz = rowptr[mesh->nnodes];
        idx_t *edge_idx = (idx_t *)malloc(nnz * sizeof(idx_t));
        memset(edge_idx, 0, nnz * sizeof(idx_t));

        ptrdiff_t edge_count = 0;
        idx_t next_id = 0;
        for (ptrdiff_t i = 0; i < mesh->nnodes; i++) {
            const count_t begin = rowptr[i];
            const count_t end = rowptr[i + 1];

            for (count_t k = begin; k < end; k++) {
                const idx_t j = colidx[k];

                if (i < j) {
                    edge_count += 1;
                    edge_idx[k] = next_id++;
                }
            }
        }

        assert(edge_count == nedges);

        // node-to-node for the hex edges in local indexing
        idx_t connectivity[8][3] = {{1, 2, 4},
                                    {0, 3, 5},
                                    {0, 3, 6},
                                    {1, 2, 7},
                                    {0, 5, 6},
                                    {1, 4, 7},
                                    {2, 4, 7},
                                    {3, 5, 6}};

        for (ptrdiff_t e = 0; e < mesh->nelements; e++) {
            idx_t nodes[8];
            for (int d = 0; d < 8; d++) {
                nodes[d] = mesh->elements[d][e];
            }

            for (int d1 = 0; d1 < 8; d1++) {
                idx_t node1 = nodes[d1];
                const idx_t *const columns = &colidx[rowptr[node1]];
                const idx_t *const edge_view = &edge_idx[rowptr[node1]];

                idx_t g_edges[3];
                idx_t g_neigh[3];

                for (int k = 0; k < 3; k++) {
                    g_neigh[k] = nodes[connectivity[d1][k]];
                }

                idx_t offsets[3];
                hex8_find_corner_cols(g_neigh, columns, rowptr[node1 + 1] - rowptr[node1], offsets);

                for (int d = 0; d < 3; d++) {
                    g_edges[d] = edge_view[offsets[d]];
                }

                // printf("//-----------\n");
                // for(int i = rowptr[node1]; i < rowptr[node1+1]; i++) {
                //     printf("%d ", colidx[i]);
                // }
                // printf("\n");
                // printf("%d => %d [%d %d %d] -> [%d %d %d]\n", d1, node1, g_neigh[0], g_neigh[1],
                // g_neigh[2],
                //      g_edges[0], g_edges[1], g_edges[2]);

                for (int d2 = 0; d2 < 3; d2++) {
                    const idx_t node2 = g_neigh[d2];

                    // direction of edge is always smaller node id to greater node id
                    if (node1 > node2) continue;

                    const int lid1 = corner_lidx[d1];
                    const int lid2 = corner_lidx[connectivity[d1][d2]];

                    int start[3], end[3];

                    int invert_dir = 0;
                    for (int d = 0; d < 3; d++) {
                        int s = coords[d][lid1];
                        int e = coords[d][lid2];
                        int x = e - s;

                        // If handle local inverted traversals
                        if (x < 0) {
                            invert_dir = 1;
                            int temp = e;
                            e = s;
                            s = temp;
                        }

                        // We make sure we can iterate over the 0 dims once
                        e += x == 0;

                        // We remove the corner node
                        s += x != 0;

                        start[d] = s;
                        end[d] = e;
                    }

                    idx_t edge_start = index_base + g_edges[d2] * nxedge;
                    idx_t edge_increment = 1;

                    if (invert_dir) {
                        edge_start += nxedge - 1;
                        edge_increment = -1;
                    }

                    int en = 0;
                    for (int zi = start[2]; zi < end[2]; zi++) {
                        for (int yi = start[1]; yi < end[1]; yi++) {
                            for (int xi = start[0]; xi < end[0]; xi++) {
                                const int lidx_edge = proteus_hex8_lidx(L, xi, yi, zi);
                                elements[lidx_edge][e] = edge_start + en;
                                // printf("e[%d][%ld] = %d\n", lidx_edge, e,
                                // elements[lidx_edge][e]);
                                en += edge_increment;
                            }
                        }
                    }

                    assert(en * edge_increment == nxedge);
                }
            }
        }

        free(rowptr);
        free(colidx);

        index_base += (nedges * nxedge);
    }

    // 3) Compute the unique face-node indices using the adjacency table
    // Two elements share a face, figure out the ordering
    int nxf = 0;  // TODO number of nodes in the face interior
    if (nxf) {
        element_idx_t *adj_table = 0;
        create_element_adj_table(
                mesh->nelements, mesh->nnodes, mesh->element_type, mesh->elements, &adj_table);

        idx_t n_unique_faces = 0;
        for (ptrdiff_t e = 0; e < mesh->nelements; e++) {
            for (int f = 0; f < 4; f++) {
                element_idx_t neigh_element = adj_table[e * 4 + f];
                if (neigh_element == SFEM_INVALID_IDX) {
                    // Is boundary face

                    for (int fn = 0; fn < nxf; fn++) {
                        idx_t idx = index_base + n_unique_faces * nxf + fn;
                        // ...
                        // TODO we need to generate the index
                    }

                    n_unique_faces++;
                } else if (e < neigh_element) {
                    // TODO we need to generate the index
                    n_unique_faces++;
                } else {
                    // TODO we need to replicate the index
                }
            }
        }

        index_base += n_unique_faces;

        // Clean-up
        free(adj_table);
    }

    // TODO Consistent ordering with implicit looping scheme needs to be figured out
    // (orientation is reflected like mirror)

    // 4) Compute the unique internal nodes implicitly using the element id and the idx offset
    // of the total number of explicit indices (offset + element_id * n_internal_nodes +
    // local_internal_node_id) ptrdiff_t n_internal_nodes = ?;
    int nxelement = 0;  // TODO number of nodes in the volume interior
    if (nxelement) {
        for (ptrdiff_t e = 0; e < mesh->nelements; e++) {
            for (int ne = 0; ne < nxe; ne++) {
                idx_t idx = index_base + e * nxelement + ne;
                // TODO figure out the local index/ordering of the interior nodes and assign to
                // elements
            }
        }
    }

    for (int d = 0; d < 3; d++) {
        free(coords[d]);
    }

    double tock = MPI_Wtime();

    printf("Create idx (%s) took %g [s]\n", type_to_string(mesh->element_type), tock - tick);
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

    if (argc < 4) {
        fprintf(stderr, "usage: %s <folder> <x.raw> <y.raw>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int SFEM_REPEAT = 1;
    int SFEM_USE_IDX = 0;

    SFEM_READ_ENV(SFEM_REPEAT, atoi);
    SFEM_READ_ENV(SFEM_USE_IDX, atoi);

    const char *folder = argv[1];
    const char *path_f = argv[2];
    const char *path_output = argv[3];

    printf("%s %s %s %s\n", argv[0], folder, path_f, path_output);

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Set-up (read and init)
    ///////////////////////////////////////////////////////////////////////////////

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    int L = 2;

    const int nxe = proteus_hex8_nxe(L);
    const int txe = proteus_hex8_txe(L);
    ptrdiff_t nnodes_discont = mesh.nelements * nxe;

    idx_t **elements = 0;

    // if (SFEM_USE_IDX) {
    elements = malloc(nxe * sizeof(idx_t *));
    for (int d = 0; d < nxe; d++) {
        elements[d] = malloc(mesh.nelements * sizeof(idx_t));
    }

    for (int d = 0; d < nxe; d++) {
        for (ptrdiff_t i = 0; i < mesh.nelements; i++) {
            elements[d][i] = -1;
        }
    }

    proteus_hex8_create_full_idx(L, &mesh, elements);

    printf("ORIGINAL\n");

    for (int d = 0; d < 8; d++) {
        for (ptrdiff_t i = 0; i < mesh.nelements; i++) {
            printf("%d\t", mesh.elements[d][i]);
        }
        printf("\n");
    }

    printf("MACRO\n");

    for (int d = 0; d < nxe; d++) {
        for (ptrdiff_t i = 0; i < mesh.nelements; i++) {
            printf("%d\t", elements[d][i]);
        }
        printf("\n");
    }
    // }

    // real_t *x = calloc(nnodes_discont, sizeof(real_t));
    // real_t *y = calloc(nnodes_discont, sizeof(real_t));

    // if (!x || !y) {
    //     fprintf(stderr, "Unable to allocate memory!\n");
    //     MPI_Abort(MPI_COMM_WORLD, -1);
    // }

    // fff_t fff;
    // int err = hex8_fff_create(&fff, mesh.nelements, mesh.elements, mesh.points);
    // if (err) {
    //     fprintf(stderr, "Unable to create FFFs!\n");
    //     MPI_Abort(MPI_COMM_WORLD, -1);
    // }

    // for (ptrdiff_t i = 0; i < nnodes_discont; i++) {
    //     x[i] = 1;
    // }

    // ///////////////////////////////////////////////////////////////////////////////
    // // Measure
    // ///////////////////////////////////////////////////////////////////////////////

    // double spmv_tick = MPI_Wtime();

    // for (int repeat = 0; repeat < SFEM_REPEAT; repeat++) {
    //     proteus_hex8_laplacian_apply(L, fff.nelements, fff.data, x, y);
    // }

    // double spmv_tock = MPI_Wtime();
    // long nelements = mesh.nelements;
    // int element_type = mesh.element_type;

    // ///////////////////////////////////////////////////////////////////////////////
    // // Output for testing
    // ///////////////////////////////////////////////////////////////////////////////

    // real_t sq_nrm = 0;
    // for (ptrdiff_t i = 0; i < nnodes_discont; i++) {
    //     sq_nrm += y[i] * y[i];
    // }

    // printf("sq_nrm = %g\n", sq_nrm);

    // // array_write(comm, path_output, SFEM_MPI_REAL_T, y, nnodes_discont, u_n_global);

    // ///////////////////////////////////////////////////////////////////////////////
    // // Free resources
    // ///////////////////////////////////////////////////////////////////////////////

    // hex8_fff_destroy(&fff);

    // free(x);
    // free(y);
    // mesh_destroy(&mesh);

    // if (SFEM_USE_IDX) {
    //     for (int d = 0; d < nxe; d++) {
    //         free(elements[d]);
    //     }

    //     free(elements);
    // }

    // ///////////////////////////////////////////////////////////////////////////////
    // // Stats
    // ///////////////////////////////////////////////////////////////////////////////

    // double tock = MPI_Wtime();
    // float TTS = tock - tick;
    // float TTS_op = (spmv_tock - spmv_tick) / SFEM_REPEAT;

    // if (!rank) {
    //     float mem_coeffs = 2 * nnodes_discont * sizeof(real_t) * 1e-9;
    //     float mem_jacs = 6 * nelements * sizeof(jacobian_t) * 1e-9;
    //     float mem_idx = nelements * nxe * sizeof(idx_t) * 1e-9;
    //     printf("----------------------------------------\n");
    //     printf("SUMMARY (%s): %s\n", type_to_string(element_type), argv[0]);
    //     printf("----------------------------------------\n");
    //     printf("#elements %ld #microelements %ld #nodes %ld\n",
    //            nelements,
    //            nelements * txe,
    //            nnodes_discont);
    //     printf("#nodexelement %d #microelementsxelement %d\n", nxe, txe);
    //     printf("Operator TTS:\t\t%.4f\t[s]\n", TTS_op);
    //     printf("Operator throughput:\t%.1f\t[ME/s]\n", 1e-6f * nelements / TTS_op);
    //     printf("Operator throughput:\t%.1f\t[MmicroE/s]\n", 1e-6f * nelements * txe /
    //     TTS_op); printf("Operator throughput:\t%.1f\t[MDOF/s]\n", 1e-6f * nnodes_discont /
    //     TTS_op); printf("Operator memory %g (2 x coeffs) + %g (FFFs) + %g (index) = %g
    //     [GB]\n",
    //            mem_coeffs,
    //            mem_jacs,
    //            mem_idx,
    //            mem_coeffs + mem_jacs + mem_idx);
    //     printf("Total:\t\t\t%.4f\t[s]\n", TTS);
    //     printf("----------------------------------------\n");
    // }

    return MPI_Finalize();
}
