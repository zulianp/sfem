#include "sfem_hex8_mesh_graph.h"

#include "crs_graph.h"
#include "sortreduce.h"

#include "adj_table.h"
#include "proteus_hex8.h"  //FIXME

#include <assert.h>
#include <mpi.h>
#include <stdio.h>

#ifndef MAX
#define MAX(a, b) ((a < b) ? (b) : (a))
#endif

#define A3SET(a, x, y, z) \
    do {                  \
        a[0] = x;         \
        a[1] = y;         \
        a[2] = z;         \
    } while (0)

// According to exodus doc
enum HEX8_Sides {
    HEX8_LEFT = 3,
    HEX8_RIGHT = 1,
    HEX8_BOTTOM = 4,
    HEX8_TOP = 5,
    HEX8_FRONT = 0,
    HEX8_BACK = 2
};

// PROTEUS
// static idx_t hex8_edge_connectivity[8][3] =
//         {{1, 2, 4},
//          {0, 3, 5},
//          {0, 3, 6},
//          {1, 2, 7},
//          {0, 5, 6},
//          {1, 4, 7},
//          {2, 4, 7},
//          {3, 5, 6}};

// LAGRANGE
static idx_t hex8_edge_connectivity[8][3] = {
        // BOTTOM
        {1, 3, 4},
        {0, 2, 5},
        {1, 3, 6},
        {0, 2, 7},
        // TOP
        {0, 5, 7},
        {1, 4, 6},
        {2, 5, 7},
        {3, 4, 6}};

static int hex8_build_edge_graph_from_n2e(const ptrdiff_t nelements,
                                          const ptrdiff_t nnodes,
                                          idx_t **const SFEM_RESTRICT elems,
                                          const count_t *const SFEM_RESTRICT n2eptr,
                                          const element_idx_t *const SFEM_RESTRICT elindex,
                                          count_t **out_rowptr,
                                          idx_t **out_colidx) {
    count_t *rowptr = (count_t *)malloc((nnodes + 1) * sizeof(count_t));
    idx_t *colidx = 0;

    static const int nnodesxelem = 8;

    {
        rowptr[0] = 0;

#pragma omp parallel for
        for (ptrdiff_t node = 0; node < nnodes; ++node) {
            idx_t n2nbuff[2048];

            count_t ebegin = n2eptr[node];
            count_t eend = n2eptr[node + 1];

            idx_t nneighs = 0;

            for (count_t e = ebegin; e < eend; ++e) {
                element_idx_t eidx = elindex[e];
                assert(eidx < nelements);

                int lidx = -1;
                for (int edof_i = 0; edof_i < nnodesxelem; ++edof_i) {
                    if (elems[edof_i][eidx] == node) {
                        lidx = edof_i;
                        break;
                    }
                }

                assert(lidx != -1);
                assert(lidx < 8);

                for (int d = 0; d < 3; d++) {
                    idx_t neighnode = elems[hex8_edge_connectivity[lidx][d]][eidx];
                    assert(nneighs < 2048);
                    n2nbuff[nneighs++] = neighnode;
                }
            }

            nneighs = sortreduce(n2nbuff, nneighs);
            rowptr[node + 1] = nneighs;
        }

        // Cumulative sum
        for (ptrdiff_t node = 0; node < nnodes; ++node) {
            rowptr[node + 1] += rowptr[node];
        }

        const ptrdiff_t nnz = rowptr[nnodes];
        colidx = (idx_t *)malloc(nnz * sizeof(idx_t));

#pragma omp parallel for
        for (ptrdiff_t node = 0; node < nnodes; ++node) {
            idx_t n2nbuff[2048];

            count_t ebegin = n2eptr[node];
            count_t eend = n2eptr[node + 1];

            idx_t nneighs = 0;

            for (count_t e = ebegin; e < eend; ++e) {
                element_idx_t eidx = elindex[e];
                assert(eidx < nelements);

                int lidx = 0;
                for (int edof_i = 0; edof_i < nnodesxelem; ++edof_i) {
                    if (elems[edof_i][eidx] == node) {
                        lidx = edof_i;
                        break;
                    }
                }

                for (int d = 0; d < 3; d++) {
                    idx_t neighnode = elems[hex8_edge_connectivity[lidx][d]][eidx];
                    assert(nneighs < 2048);
                    n2nbuff[nneighs++] = neighnode;
                }
            }

            nneighs = sortreduce(n2nbuff, nneighs);

            for (idx_t i = 0; i < nneighs; ++i) {
                colidx[rowptr[node] + i] = n2nbuff[i];
            }
        }
    }

    *out_rowptr = rowptr;
    *out_colidx = colidx;
    return 0;
}

int hex8_build_edge_graph(const ptrdiff_t nelements,
                          const ptrdiff_t nnodes,
                          idx_t **const elems,
                          count_t **out_rowptr,
                          idx_t **out_colidx) {
    double tick = MPI_Wtime();

    count_t *n2eptr;
    element_idx_t *elindex;
    build_n2e(nelements, nnodes, 8, elems, &n2eptr, &elindex);

    int err = hex8_build_edge_graph_from_n2e(
            nelements, nnodes, elems, n2eptr, elindex, out_rowptr, out_colidx);

    free(n2eptr);
    free(elindex);

    double tock = MPI_Wtime();
    printf("crs_graph.c: build nz (mem conservative) structure\t%g seconds\n", tock - tick);
    return err;
}

#define SFEM_INVALID_IDX (-1)

ptrdiff_t nxe_max_node_id(const ptrdiff_t nelements,
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

int proteus_hex8_create_full_idx(const int L,
                                 mesh_t *mesh,
                                 idx_t **elements,
                                 ptrdiff_t *n_unique_nodes_out,
                                 ptrdiff_t *interior_start_out) {
    assert(L >= 2);
    static int verbose = 0;

    double tick = MPI_Wtime();

    const int nxe = proteus_hex8_nxe(L);

    // 1) Get the node indices from the HEX8 mesh
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

    if (verbose) printf("NODES\t%g [s]\n", tack - tick);

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
        if (verbose) printf("EDGES\t%g [s]\n", tack - temp_tick);
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
        if (verbose) printf("FACES\t%g [s]\n", tack - temp_tick);
    }

    // 4) Compute the unique internal nodes implicitly using the element id and the idx offset
    // of the total number of explicit indices (offset + element_id * n_internal_nodes +
    // local_internal_node_id) ptrdiff_t n_internal_nodes = ?;
    int nxelement = (L - 1) * (L - 1) * (L - 1);
    ptrdiff_t interior_start = index_base;
    if (nxelement) {
        double temp_tick = MPI_Wtime();

        for (int zi = 1; zi < L; zi++) {
            for (int yi = 1; yi < L; yi++) {
                for (int xi = 1; xi < L; xi++) {
                    const int lidx_vol = proteus_hex8_lidx(L, xi, yi, zi);
                    int Lm1 = L - 1;
                    int en = (zi - 1) * Lm1 * Lm1 + (yi - 1) * Lm1 + xi - 1;

#pragma omp parallel for
                    for (ptrdiff_t e = 0; e < mesh->nelements; e++) {
                        elements[lidx_vol][e] = index_base + e * nxelement + en;
                        // printf("elements[%d][%ld] = %d + %ld * %d + %d\n", lidx_vol, e,
                        // index_base, e, nxelement, en);
                    }
                }
            }
        }

        tack = MPI_Wtime();
        if (verbose) printf("ELEMS\t%g [s]\n", tack - temp_tick);
    }

    for (int d = 0; d < 3; d++) {
        free(coords[d]);
    }

    *n_unique_nodes_out = interior_start + mesh->nelements * nxelement;
    *interior_start_out = interior_start;

    double tock = MPI_Wtime();
    printf("Create idx (%s) took\t%g [s]\n", type_to_string(mesh->element_type), tock - tick);

    return SFEM_SUCCESS;
}

int proteus_hex8_mesh_skin(const int L,
                           const ptrdiff_t nelements,
                           idx_t **elements,
                           ptrdiff_t *n_surf_elements,
                           idx_t **const surf_elements,
                           element_idx_t **parent_element) {
    const int proteus_to_std[8] = {// Bottom
                                   proteus_hex8_lidx(L, 0, 0, 0),
                                   proteus_hex8_lidx(L, L, 0, 0),
                                   proteus_hex8_lidx(L, L, L, 0),
                                   proteus_hex8_lidx(L, 0, L, 0),
                                   // Top
                                   proteus_hex8_lidx(L, 0, 0, L),
                                   proteus_hex8_lidx(L, L, 0, L),
                                   proteus_hex8_lidx(L, L, L, L),
                                   proteus_hex8_lidx(L, 0, L, L)};

    idx_t *hex8_elements[8] = {elements[proteus_to_std[0]],
                               elements[proteus_to_std[1]],
                               elements[proteus_to_std[2]],
                               elements[proteus_to_std[3]],
                               elements[proteus_to_std[4]],
                               elements[proteus_to_std[5]],
                               elements[proteus_to_std[6]],
                               elements[proteus_to_std[7]]};

    ptrdiff_t hex8_n_nodes = nxe_max_node_id(nelements, 8, hex8_elements) + 1;

    const int ns = elem_num_sides(HEX8);
    element_idx_t *table = 0;
    create_element_adj_table(nelements, hex8_n_nodes, HEX8, hex8_elements, &table);

    // Num side-nodes
    const int nn = (L + 1) * (L + 1);

    *n_surf_elements = 0;
    for (ptrdiff_t e = 0; e < nelements; e++) {
        for (int s = 0; s < ns; s++) {
            // Array of structures
            const element_idx_t e_adj = table[e * ns + s];
            if (e_adj == SFEM_INVALID_IDX) {
                (*n_surf_elements)++;
            }
        }
    }

    *parent_element = malloc((*n_surf_elements) * sizeof(element_idx_t));
    for (int s = 0; s < nn; s++) {
        surf_elements[s] = malloc((*n_surf_elements) * sizeof(idx_t));
    }

    ptrdiff_t side_offset = 0;
    for (ptrdiff_t e = 0; e < nelements; e++) {
        for (int s = 0; s < ns; s++) {
            // Array of structures
            const element_idx_t e_adj = table[e * ns + s];
            if (e_adj == SFEM_INVALID_IDX) {
                int start[3] = {0, 0, 0};
                int end[3] = {L + 1, L + 1, L + 1};
                int increment[3] = {1, 1, 1};

                // printf("side %d: ", s);

                switch (s) {
                    case HEX8_LEFT: {
                        A3SET(start, 0, L, 0);
                        A3SET(end, 1, -1, L + 1);
                        A3SET(increment, 1, -1, 1);
                        // printf("HEX8_LEFT\n");
                        break;
                    }
                    case HEX8_RIGHT: {
                        start[0] = L;
                        end[0] = L + 1;
                        // printf("HEX8_RIGHT\n");
                        break;
                    }
                    case HEX8_BOTTOM: {
                        start[2] = 0;
                        end[2] = 1;
                        A3SET(start, 0, L, 0);
                        A3SET(end,  L+1, -1, 1);
                        A3SET(increment, 1, -1, 1);
                        // printf("HEX8_BOTTOM\n");
                        break;
                    }
                    case HEX8_TOP: {
                        start[2] = L;
                        end[2] = L + 1;
                        // printf("HEX8_TOP\n");
                        break;
                    }
                    case HEX8_FRONT: {
                        start[1] = 0;
                        end[1] = 1;
                        // printf("HEX8_FRONT\n");
                        break;
                    }
                    case HEX8_BACK: {
                        A3SET(start, L, L, 0);
                        A3SET(end,  -1, L+1, L + 1);
                        A3SET(increment, -1, 1, 1);
                        // printf("HEX8_BACK\n");
                        break;
                    }
                    default: {
                        assert(0);
                        break;
                    }
                }

                int n = 0;
                for (int zi = start[2]; zi != end[2]; zi += increment[2]) {
                    for (int yi = start[1]; yi != end[1]; yi += increment[1]) {
                        for (int xi = start[0]; xi != end[0]; xi += increment[0]) {
                            const int lidx = proteus_hex8_lidx(L, xi, yi, zi);
                            const idx_t node = elements[lidx][e];
                            surf_elements[n++][side_offset] = node;
                            // printf("(%d, %d, %d), l: %d => g:\t%d\n", xi, yi, zi, lidx, node);
                        }
                    }
                }

                (*parent_element)[side_offset] = e;
                side_offset++;
            }
        }
    }

    free(table);
    return SFEM_SUCCESS;
}
