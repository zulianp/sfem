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
#include "sfem_hex8_mesh_graph.h"
// #include "hex8_fff.h"

#include "sortreduce.h"

#include "adj_table.h"

ptrdiff_t my_max_node_id(const ptrdiff_t nelements,
                         const int nxe,
                         idx_t **const SFEM_RESTRICT elements) {
    ptrdiff_t ret = 0;
    for (int i = 0; i < nxe; i++) {
        for (ptrdiff_t e = 0; e < nelements; e++) {
            ret = MAX(ret, elements[i][e]);
        }
    }
    return ret;
}

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

static void index_face(const int L,
                       mesh_t *mesh,
                       const int *const local_side_table,
                       int *lagr_to_proteus_corners,
                       int **coords,
                       const idx_t global_face_offset,
                       const ptrdiff_t e,
                       const int f,
                       idx_t **const elements) {
    int argmin = 0;
    idx_t valmin = mesh->elements[local_side_table[f * 4 + 0]][e];
    for (int i = 0; i < 4; i++) {
        idx_t temp = mesh->elements[local_side_table[f * 4 + i]][e];
        if (temp < valmin) {
            argmin = i;
            valmin = temp;
        }
    }

    int lst_o = argmin;
    int lst_u = ((lst_o + 1) % 4);
    int lst_v = ((lst_o + 3) % 4);
    if (mesh->elements[local_side_table[f * 4 + lst_u]][e] >
        mesh->elements[local_side_table[f * 4 + lst_v]][e]) {
        int temp = lst_v;
        lst_v = lst_u;
        lst_u = temp;
    }

    // o, u, v are sorted based on global indices
    int lidx_o = lagr_to_proteus_corners[local_side_table[f * 4 + lst_o]];
    int lidx_u = lagr_to_proteus_corners[local_side_table[f * 4 + lst_u]];
    int lidx_v = lagr_to_proteus_corners[local_side_table[f * 4 + lst_v]];

    // printf("lst   (%d, %d, %d)\n", lst_o, lst_u, lst_v);

    // printf("lst[] (%d, %d, %d)\n",
    //        local_side_table[f * 4 + lst_o],
    //        local_side_table[f * 4 + lst_u],
    //        local_side_table[f * 4 + lst_v]);

    // printf("lidx  (%d, %d, %d)\n", lidx_o, lidx_u, lidx_v);

    int o_start[3];
    int u_len[3], u_dir[3];
    int v_len[3], v_dir[3];

    // O
    for (int d = 0; d < 3; d++) {
        int o = coords[d][lidx_o];
        o_start[d] = o;
    }

    // printf("po = [%d %d %d]\n", coords[0][lidx_o], coords[1][lidx_o], coords[2][lidx_o]);
    // printf("pu = [%d %d %d]\n", coords[0][lidx_u], coords[1][lidx_u], coords[2][lidx_u]);
    // printf("pv = [%d %d %d]\n", coords[0][lidx_v], coords[1][lidx_v], coords[2][lidx_v]);

    // U
    for (int d = 0; d < 3; d++) {
        int x = coords[d][lidx_u] - coords[d][lidx_o];

        u_dir[d] = 1;
        u_len[d] = 1;

        if (x > 0) {
            x -= 1;
            u_len[d] = x;
            o_start[d] = 1;
        } else if (x < 0) {
            x += 1;
            u_len[d] = x;
            u_dir[d] = -1;
            o_start[d] = L - 1;
        }
    }

    // V
    for (int d = 0; d < 3; d++) {
        int x = coords[d][lidx_v] - coords[d][lidx_o];

        v_dir[d] = 1;
        v_len[d] = 1;

        if (x > 0) {
            x -= 1;
            v_len[d] = x;
            o_start[d] = 1;

        } else if (x < 0) {
            x += 1;
            v_len[d] = x;
            v_dir[d] = -1;
            o_start[d] = L - 1;
        }
    }

    // printf("global_face_offset = %d\n", global_face_offset);
    // printf("o  (%d, %d, %d)\n", o_start[0], o_start[1], o_start[2]);
    // printf("u_dir  (%d, %d, %d)\n", u_dir[0], u_dir[1], u_dir[2]);
    // printf("u_len    (%d, %d, %d)\n", u_len[0], u_len[1], u_len[2]);
    // printf("v_dir  (%d, %d, %d)\n", v_dir[0], v_dir[1], v_dir[2]);
    // printf("v_len    (%d, %d, %d)\n", v_len[0], v_len[1], v_len[2]);
    // fflush(stdout);

    int local_offset = 0;
    for (int vzi = 0; vzi != v_len[2]; vzi += v_dir[2]) {
        for (int vyi = 0; vyi != v_len[1]; vyi += v_dir[1]) {
            for (int vxi = 0; vxi != v_len[0]; vxi += v_dir[0]) {
                //
                for (int uzi = 0; uzi != u_len[2]; uzi += u_dir[2]) {
                    for (int uyi = 0; uyi != u_len[1]; uyi += u_dir[1]) {
                        for (int uxi = 0; uxi != u_len[0]; uxi += u_dir[0]) {
                            // printf("u = [%d %d %d]\n", uxi, uyi, uzi);
                            // printf("v = [%d %d %d]\n", vxi, vyi, vzi);
                            // printf("p = [%d %d %d]\n",
                            //        uxi + vxi + o_start[0],
                            //        uyi + vyi + o_start[1],
                            //        uzi + vzi + o_start[2]);

                            int pidx = proteus_hex8_lidx(L,
                                                         uxi + vxi + o_start[0],
                                                         uyi + vyi + o_start[1],
                                                         uzi + vzi + o_start[2]);

                            // idx_t u_offset = u_face_start + (uxi + uyi + uzi) * u_increment;
                            // idx_t v_offset = v_face_start + (vxi + vyi + vzi) * v_increment;

                            idx_t fidx = global_face_offset + local_offset++;
                            // v_offset * (L - 1) + u_offset;
                            elements[pidx][e] = fidx;

                            // printf("offsets %d %d\n", u_offset, v_offset);
                            // printf("p %d => %d\n", pidx, fidx);
                        }
                    }
                }
            }
        }
    }
}

static void proteus_hex8_create_full_idx(const int L, mesh_t *mesh, idx_t **elements) {
    assert(L >= 2);

    double tick = MPI_Wtime();

    const int nxe = proteus_hex8_nxe(L);

    // 1) Get the node indices from the TET4 mesh
    int lagr_to_proteus_corners[8] = {// Bottom
                                      proteus_hex8_lidx(L, 0, 0, 0),
                                      proteus_hex8_lidx(L, L, 0, 0),
                                      proteus_hex8_lidx(L, L, L, 0),
                                      proteus_hex8_lidx(L, 0, L, 0),
                                      // Top
                                      proteus_hex8_lidx(L, 0, 0, L),
                                      proteus_hex8_lidx(L, L, 0, L),
                                      proteus_hex8_lidx(L, L, L, L),
                                      proteus_hex8_lidx(L, 0, L, L)};

#ifndef NDEBUG
    for (int i = 0; i < 8; i++) {
        assert(lagr_to_proteus_corners[i] < nxe);
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
            elements[lagr_to_proteus_corners[d]][e] = mesh->elements[d][e];
        }
    }

    idx_t index_base = mesh->nnodes;

    double tack = MPI_Wtime();
    printf("NODES\t%g [s]\n", tack - tick);

    // 2) Compute the unique edge-node indices using the CRSGraph
    // A unique edge index can be used and use the multiple to store all indices
    // as consecutive

    // Number of nodes in the edge interior
    ptrdiff_t nxedge = L - 1;  // L == 0 (is this correct?)

    if (nxedge) {
        double temp_tick = MPI_Wtime();

        count_t *rowptr;
        idx_t *colidx;
        hex8_build_edge_graph(mesh->nelements, mesh->nnodes, mesh->elements, &rowptr, &colidx);

        ptrdiff_t nedges = rowptr[mesh->nnodes] / 2;

        ptrdiff_t nnz = rowptr[mesh->nnodes];
        idx_t *edge_idx = (idx_t *)malloc(nnz * sizeof(idx_t));
        memset(edge_idx, 0, nnz * sizeof(idx_t));

        // node-to-node for the hex edges in local indexing
        idx_t lagr_connectivity[8][3] = {// BOTTOM
                                         {1, 3, 4},
                                         {0, 2, 5},
                                         {1, 3, 6},
                                         {0, 2, 7},
                                         // TOP
                                         {0, 5, 7},
                                         {1, 4, 6},
                                         {2, 5, 7},
                                         {3, 4, 6}};

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
                    g_neigh[k] = nodes[lagr_connectivity[d1][k]];
                }

                idx_t offsets[3];
                hex8_find_corner_cols(g_neigh, columns, rowptr[node1 + 1] - rowptr[node1], offsets);

                for (int d = 0; d < 3; d++) {
                    g_edges[d] = edge_view[offsets[d]];
                }

                for (int d2 = 0; d2 < 3; d2++) {
                    const idx_t node2 = g_neigh[d2];

                    // direction of edge is always smaller node id to greater node id
                    if (node1 > node2) continue;

                    const int lid1 = lagr_to_proteus_corners[d1];
                    const int lid2 = lagr_to_proteus_corners[lagr_connectivity[d1][d2]];

                    int start[3], len[3], dir[3];
                    for (int d = 0; d < 3; d++) {
                        int o = coords[d][lid1];
                        start[d] = o;
                    }

                    int invert_dir = 0;
                    for (int d = 0; d < 3; d++) {
                        int x = coords[d][lid2] - coords[d][lid1];
                        dir[d] = 1;
                        len[d] = 1;

                        if (x > 0) {
                            x -= 1;
                            len[d] = x;
                            start[d] = 1;
                        } else if (x < 0) {
                            x += 1;
                            len[d] = x;
                            dir[d] = -1;
                            start[d] = L - 1;
                        }
                    }

                    idx_t edge_start = index_base + g_edges[d2] * nxedge;

                    // printf("//----------\n");
                    // printf("po(%d) = [%d %d %d]\n",
                    //        lid1,
                    //        coords[0][lid1],
                    //        coords[1][lid1],
                    //        coords[2][lid1]);
                    // printf("pu(%d) = [%d %d %d]\n",
                    //        lid2,
                    //        coords[0][lid2],
                    //        coords[1][lid2],
                    //        coords[2][lid2]);

                    // printf("start  = [%d %d %d]\n", start[0], start[1], start[2]);
                    // printf("dir    = [%d %d %d]\n", dir[0], dir[1], dir[2]);
                    // printf("len    = [%d %d %d]\n", len[0], len[1], len[2]);

                    int en = 0;
                    for (int zi = 0; zi != len[2]; zi += dir[2]) {
                        for (int yi = 0; yi != len[1]; yi += dir[1]) {
                            for (int xi = 0; xi != len[0]; xi += dir[0]) {
                                const int lidx_edge = proteus_hex8_lidx(
                                        L, start[0] + xi, start[1] + yi, start[2] + zi);
                                elements[lidx_edge][e] = edge_start + en;
                                en += 1;

                                // printf("%d) => %d\n", lidx_edge, elements[lidx_edge][e]);
                            }
                        }
                    }
                }
            }
        }

        free(rowptr);
        free(colidx);

        // printf("----------------------------\n");
        // printf("index_base %d + %ld * %d\n", index_base, nedges, nxedge);
        // printf("----------------------------\n");
        index_base += (nedges * nxedge);

        tack = MPI_Wtime();
        printf("EDGES\t%g [s]\n", tack - temp_tick);
    }

    // 3) Compute the unique face-node indices using the adjacency table
    // Two elements share a face, figure out the ordering
    int nxf = (L - 1) * (L - 1);  // TODO number of nodes in the face interior
    if (nxf) {
        double temp_tick = MPI_Wtime();

        int local_side_table[6 * 4];
        fill_local_side_table(HEX8, local_side_table);

        element_idx_t *adj_table = 0;
        create_element_adj_table(
                mesh->nelements, mesh->nnodes, mesh->element_type, mesh->elements, &adj_table);

        idx_t n_unique_faces = 0;
        for (ptrdiff_t e = 0; e < mesh->nelements; e++) {
            for (int f = 0; f < 6; f++) {
                element_idx_t neigh_element = adj_table[e * 6 + f];
                // If this face is not boundary and it has already been processed continue
                if (neigh_element != -1 && neigh_element < e) continue;

                idx_t global_face_offset = index_base + n_unique_faces * nxf;
                index_face(L,
                           mesh,
                           local_side_table,
                           lagr_to_proteus_corners,
                           coords,
                           global_face_offset,
                           e,
                           f,
                           elements);

                if (neigh_element != -1) {
                    // find same face on neigh element
                    int neigh_f;
                    for (neigh_f = 0; neigh_f < 6; neigh_f++) {
                        if (e == adj_table[neigh_element * 6 + neigh_f]) {
                            break;
                        }
                    }

                    assert(neigh_f != 6);

                    index_face(L,
                               mesh,
                               local_side_table,
                               lagr_to_proteus_corners,
                               coords,
                               global_face_offset,
                               neigh_element,
                               neigh_f,
                               elements);
                }

                // Next id
                n_unique_faces++;
            }
        }

        index_base += n_unique_faces * nxf;

        // Clean-up
        free(adj_table);

        tack = MPI_Wtime();
        printf("FACES\t%g [s]\n", tack - temp_tick);
    }

    // 4) Compute the unique internal nodes implicitly using the element id and the idx offset
    // of the total number of explicit indices (offset + element_id * n_internal_nodes +
    // local_internal_node_id) ptrdiff_t n_internal_nodes = ?;
    int nxelement = (L - 1) * (L - 1) * (L - 1);
    if (nxelement) {
        double temp_tick = MPI_Wtime();

        for (int zi = 1; zi < L - 1; zi++) {
            for (int yi = 1; yi < L - 1; yi++) {
                for (int xi = 1; xi < L - 1; xi++) {
                    const int lidx_vol = proteus_hex8_lidx(L, xi, yi, zi);
                    int en = (zi - 1) * (L - 1) * (L - 1) + (yi - 1) * (L - 1) + xi - 1;

#pragma omp parallel for
                    for (ptrdiff_t e = 0; e < mesh->nelements; e++) {
                        elements[lidx_vol][e] = index_base + e * nxelement + en;
                    }
                }
            }
        }

        tack = MPI_Wtime();
        printf("ELEMS\t%g [s]\n", tack - temp_tick);
    }

    for (int d = 0; d < 3; d++) {
        free(coords[d]);
    }

    double tock = MPI_Wtime();
    printf("Create idx (%s) took\t%g [s]\n", type_to_string(mesh->element_type), tock - tick);
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

    int L = 6;

    const int nxe = proteus_hex8_nxe(L);
    const int txe = proteus_hex8_txe(L);
    ptrdiff_t nnodes_discont = mesh.nelements * nxe;

    printf("nelements %ld\n", mesh.nelements);
    printf("nnodes    %ld\n", mesh.nnodes);
    printf("nxe       %d\n", nxe);
    printf("txe       %d\n", txe);

    idx_t **elements = 0;

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

    if (0) {
        printf("//--------------- \n");
        printf("ORIGINAL\n");

        for (int d = 0; d < 8; d++) {
            for (ptrdiff_t i = 0; i < mesh.nelements; i++) {
                printf("%d\t", mesh.elements[d][i]);
            }
            printf("\n");
        }

        printf("//--------------- \n");
        printf("MACRO\n");

        for (int d = 0; d < nxe; d++) {
            printf("%d)\t", d);
            for (ptrdiff_t i = 0; i < mesh.nelements; i++) {
                printf("%d\t", elements[d][i]);
            }
            printf("\n");
        }
        printf("//--------------- \n");
    }

    idx_t nunique_nodes = my_max_node_id(mesh.nelements, nxe, elements) + 1;

    ptrdiff_t internal_nodes = mesh.nelements * (L - 1) * (L - 1) * (L - 1);
    printf("n unique nodes %d\n", nunique_nodes);
    printf("vol nodes %ld\n", internal_nodes);
    printf("vol %f%%\n", 100 * (float)internal_nodes / nunique_nodes);

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
