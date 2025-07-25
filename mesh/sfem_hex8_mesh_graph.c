#include "sfem_hex8_mesh_graph.h"

#include "crs_graph.h"
#include "sortreduce.h"

#include "adj_table.h"
#include "sshex8.h"   //FIXME
#include "ssquad4.h"  //FIXME

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

static int hex8_build_edge_graph_from_n2e(const ptrdiff_t                          nelements,
                                          const ptrdiff_t                          nnodes,
                                          idx_t **const SFEM_RESTRICT              elems,
                                          const count_t *const SFEM_RESTRICT       n2eptr,
                                          const element_idx_t *const SFEM_RESTRICT elindex,
                                          count_t                                **out_rowptr,
                                          idx_t                                  **out_colidx) {
    count_t *rowptr = (count_t *)malloc((nnodes + 1) * sizeof(count_t));
    idx_t   *colidx = 0;

    static const int nnodesxelem = 8;

    {
        rowptr[0] = 0;

#pragma omp parallel for
        for (ptrdiff_t node = 0; node < nnodes; ++node) {
            idx_t n2nbuff[2048];

            count_t ebegin = n2eptr[node];
            count_t eend   = n2eptr[node + 1];

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

            nneighs          = sortreduce(n2nbuff, nneighs);
            rowptr[node + 1] = nneighs;
        }

        // Cumulative sum
        for (ptrdiff_t node = 0; node < nnodes; ++node) {
            rowptr[node + 1] += rowptr[node];
        }

        const ptrdiff_t nnz = rowptr[nnodes];
        colidx              = (idx_t *)malloc(nnz * sizeof(idx_t));

#pragma omp parallel for
        for (ptrdiff_t node = 0; node < nnodes; ++node) {
            idx_t n2nbuff[2048];

            count_t ebegin = n2eptr[node];
            count_t eend   = n2eptr[node + 1];

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
                          idx_t **const   elems,
                          count_t       **out_rowptr,
                          idx_t         **out_colidx) {
    double tick = MPI_Wtime();

    count_t       *n2eptr;
    element_idx_t *elindex;
    build_n2e(nelements, nnodes, 8, elems, &n2eptr, &elindex);

    int err = hex8_build_edge_graph_from_n2e(nelements, nnodes, elems, n2eptr, elindex, out_rowptr, out_colidx);

    free(n2eptr);
    free(elindex);

    double tock = MPI_Wtime();
    printf("crs_graph.c: build nz (mem conservative) structure\t%g seconds\n", tock - tick);
    return err;
}

static int sshex8_skeleton_build_edge_graph_from_hex8_n2e(const int                                level,
                                                          const ptrdiff_t                          nelements,
                                                          const ptrdiff_t                          nnodes,
                                                          idx_t **const SFEM_RESTRICT              elems,
                                                          const count_t *const SFEM_RESTRICT       n2eptr,
                                                          const element_idx_t *const SFEM_RESTRICT elindex,
                                                          count_t                                **out_rowptr,
                                                          idx_t                                  **out_colidx) {
//     count_t *rowptr = (count_t *)malloc((nnodes + 1) * sizeof(count_t));
//     idx_t   *colidx = 0;

//     static const int nnodesxelem = 8;

//     {
//         rowptr[0] = 0;

// #pragma omp parallel for
//         for (ptrdiff_t node = 0; node < nnodes; ++node) {
//             idx_t n2nbuff[2048];

//             count_t ebegin = n2eptr[node];
//             count_t eend   = n2eptr[node + 1];

//             idx_t nneighs = 0;

//             for (count_t e = ebegin; e < eend; ++e) {
//                 element_idx_t eidx = elindex[e];
//                 assert(eidx < nelements);

//                 int lidx = -1;
//                 for (int edof_i = 0; edof_i < nnodesxelem; ++edof_i) {
//                     if (elems[edof_i][eidx] == node) {
//                         lidx = edof_i;
//                         break;
//                     }
//                 }

//                 assert(lidx != -1);
//                 assert(lidx < 8);

//                 for (int d = 0; d < 3; d++) {
//                     idx_t neighnode = elems[hex8_edge_connectivity[lidx][d]][eidx];
//                     assert(nneighs < 2048);
//                     n2nbuff[nneighs++] = neighnode;
//                 }
//             }

//             nneighs          = sortreduce(n2nbuff, nneighs);
//             rowptr[node + 1] = nneighs;
//         }

//         // Cumulative sum
//         for (ptrdiff_t node = 0; node < nnodes; ++node) {
//             rowptr[node + 1] += rowptr[node];
//         }

        //         const ptrdiff_t nnz = rowptr[nnodes];
        //         colidx              = (idx_t *)malloc(nnz * sizeof(idx_t));

        // #pragma omp parallel for
        //         for (ptrdiff_t node = 0; node < nnodes; ++node) {
        //             idx_t n2nbuff[2048];

        //             count_t ebegin = n2eptr[node];
        //             count_t eend   = n2eptr[node + 1];

        //             idx_t nneighs = 0;

        //             for (count_t e = ebegin; e < eend; ++e) {
        //                 element_idx_t eidx = elindex[e];
        //                 assert(eidx < nelements);

        //                 int lidx = 0;
        //                 for (int edof_i = 0; edof_i < nnodesxelem; ++edof_i) {
        //                     if (elems[edof_i][eidx] == node) {
        //                         lidx = edof_i;
        //                         break;
        //                     }
        //                 }

        //                 for (int d = 0; d < 3; d++) {
        //                     idx_t neighnode = elems[hex8_edge_connectivity[lidx][d]][eidx];
        //                     assert(nneighs < 2048);
        //                     n2nbuff[nneighs++] = neighnode;
        //                 }
        //             }

        //             nneighs = sortreduce(n2nbuff, nneighs);

        //             for (idx_t i = 0; i < nneighs; ++i) {
        //                 colidx[rowptr[node] + i] = n2nbuff[i];
        //             }
        //         }
    // }

    // *out_rowptr = rowptr;
    // *out_colidx = colidx;
    return SFEM_SUCCESS;
}

int sshex8_skeleton_crs_graph(const int       L,
                              const ptrdiff_t nelements,
                              const ptrdiff_t nnodes,
                              idx_t **const   elements,
                              count_t       **out_rowptr,
                              idx_t         **out_colidx) {
    double tick = MPI_Wtime();

    count_t       *n2eptr;
    element_idx_t *elindex;
    build_n2e(nelements, nnodes, 8, elements, &n2eptr, &elindex);

    sshex8_skeleton_build_edge_graph_from_hex8_n2e(L, nelements, nnodes, elements, n2eptr, elindex, out_rowptr, out_colidx);

    free(n2eptr);
    free(elindex);

    double tock = MPI_Wtime();
    printf("sshex8_skeleton_crs_graph.c: build nz (mem conservative) structure\t%g seconds\n", tock - tick);
    return SFEM_SUCCESS;
}

ptrdiff_t nxe_max_node_id(const ptrdiff_t nelements, const int nxe, idx_t **const SFEM_RESTRICT elements) {
    ptrdiff_t ret = 0;
    for (int i = 0; i < nxe; i++) {
        for (ptrdiff_t e = 0; e < nelements; e++) {
            ret = MAX(ret, elements[i][e]);
        }
    }
    return ret;
}

static SFEM_INLINE idx_t hex8_linear_search(const idx_t target, const idx_t *const arr, const int size) {
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
    return SFEM_IDX_INVALID;
}

static SFEM_INLINE idx_t hex8_find_col(const idx_t key, const idx_t *const row, const int lenrow) {
    if (lenrow <= 32) {
        return hex8_linear_search(key, row, lenrow);
    } else {
        // Use this for larger number of dofs per row
        return find_idx_binary_search(key, row, lenrow);
    }
}

static SFEM_INLINE void hex8_find_corner_cols(const idx_t *targets, const idx_t *const row, const int lenrow, idx_t *ks) {
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

static void index_face(const int        L,
                       idx_t **const    m_elements,
                       const int *const local_side_table,
                       int             *lagr_to_proteus_corners,
                       int            **coords,
                       const idx_t      global_face_offset,
                       const ptrdiff_t  e,
                       const int        f,
                       idx_t **const    elements) {
    int   argmin = 0;
    idx_t valmin = m_elements[local_side_table[f * 4 + 0]][e];
    for (int i = 0; i < 4; i++) {
        idx_t temp = m_elements[local_side_table[f * 4 + i]][e];
        if (temp < valmin) {
            argmin = i;
            valmin = temp;
        }
    }

    int lst_o = argmin;
    int lst_u = ((lst_o + 1) % 4);
    int lst_v = ((lst_o + 3) % 4);
    if (m_elements[local_side_table[f * 4 + lst_u]][e] > m_elements[local_side_table[f * 4 + lst_v]][e]) {
        int temp = lst_v;
        lst_v    = lst_u;
        lst_u    = temp;
    }

    // o, u, v are sorted based on global indices
    int lidx_o = lagr_to_proteus_corners[local_side_table[f * 4 + lst_o]];
    int lidx_u = lagr_to_proteus_corners[local_side_table[f * 4 + lst_u]];
    int lidx_v = lagr_to_proteus_corners[local_side_table[f * 4 + lst_v]];

    int o_start[3];
    int u_len[3], u_dir[3];
    int v_len[3], v_dir[3];

    // O
    for (int d = 0; d < 3; d++) {
        int o      = coords[d][lidx_o];
        o_start[d] = o;
    }

    // U
    for (int d = 0; d < 3; d++) {
        int x = coords[d][lidx_u] - coords[d][lidx_o];

        u_dir[d] = 1;
        u_len[d] = 1;

        if (x > 0) {
            x -= 1;
            u_len[d]   = x;
            o_start[d] = 1;
        } else if (x < 0) {
            x += 1;
            u_len[d]   = x;
            u_dir[d]   = -1;
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
            v_len[d]   = x;
            o_start[d] = 1;

        } else if (x < 0) {
            x += 1;
            v_len[d]   = x;
            v_dir[d]   = -1;
            o_start[d] = L - 1;
        }
    }

    int local_offset = 0;
    for (int vzi = 0; vzi != v_len[2]; vzi += v_dir[2]) {
        for (int vyi = 0; vyi != v_len[1]; vyi += v_dir[1]) {
            for (int vxi = 0; vxi != v_len[0]; vxi += v_dir[0]) {
                //
                for (int uzi = 0; uzi != u_len[2]; uzi += u_dir[2]) {
                    for (int uyi = 0; uyi != u_len[1]; uyi += u_dir[1]) {
                        for (int uxi = 0; uxi != u_len[0]; uxi += u_dir[0]) {
                            int pidx = sshex8_lidx(L, uxi + vxi + o_start[0], uyi + vyi + o_start[1], uzi + vzi + o_start[2]);
                            idx_t fidx = global_face_offset + local_offset++;
                            elements[pidx][e] = fidx;
                        }
                    }
                }
            }
        }
    }
}

int sshex8_generate_elements(const int       L,
                             const ptrdiff_t m_nelements,
                             const ptrdiff_t m_nnodes,
                             idx_t **const   m_elements,
                             idx_t         **elements,
                             ptrdiff_t      *n_unique_nodes_out,
                             ptrdiff_t      *interior_start_out) {
    static const enum ElemType m_element_type = HEX8;
    assert(L >= 2);
    static int verbose = 0;

    double tick = MPI_Wtime();

    const int nxe = sshex8_nxe(L);

    // 1) Get the node indices from the HEX8 mesh
    int lagr_to_proteus_corners[8] = {// Bottom
                                      sshex8_lidx(L, 0, 0, 0),
                                      sshex8_lidx(L, L, 0, 0),
                                      sshex8_lidx(L, L, L, 0),
                                      sshex8_lidx(L, 0, L, 0),
                                      // Top
                                      sshex8_lidx(L, 0, 0, L),
                                      sshex8_lidx(L, L, 0, L),
                                      sshex8_lidx(L, L, L, L),
                                      sshex8_lidx(L, 0, L, L)};

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
                int lidx = sshex8_lidx(L, xi, yi, zi);
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
        for (ptrdiff_t e = 0; e < m_nelements; e++) {
            elements[lagr_to_proteus_corners[d]][e] = m_elements[d][e];
        }
    }

    idx_t index_base = m_nnodes;

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
        idx_t   *colidx;
        hex8_build_edge_graph(m_nelements, m_nnodes, m_elements, &rowptr, &colidx);

        ptrdiff_t nedges = rowptr[m_nnodes] / 2;

        ptrdiff_t nnz      = rowptr[m_nnodes];
        idx_t    *edge_idx = (idx_t *)malloc(nnz * sizeof(idx_t));
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
        idx_t     next_id    = 0;
        for (ptrdiff_t i = 0; i < m_nnodes; i++) {
            const count_t begin = rowptr[i];
            const count_t end   = rowptr[i + 1];

            for (count_t k = begin; k < end; k++) {
                const idx_t j = colidx[k];

                if (i < j) {
                    edge_count += 1;
                    edge_idx[k] = next_id++;
                }
            }
        }

        assert(edge_count == nedges);

        for (ptrdiff_t e = 0; e < m_nelements; e++) {
            idx_t nodes[8];
            for (int d = 0; d < 8; d++) {
                nodes[d] = m_elements[d][e];
            }

            for (int d1 = 0; d1 < 8; d1++) {
                idx_t              node1     = nodes[d1];
                const idx_t *const columns   = &colidx[rowptr[node1]];
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
                        int o    = coords[d][lid1];
                        start[d] = o;
                    }

                    int invert_dir = 0;
                    for (int d = 0; d < 3; d++) {
                        int x  = coords[d][lid2] - coords[d][lid1];
                        dir[d] = 1;
                        len[d] = 1;

                        if (x > 0) {
                            x -= 1;
                            len[d]   = x;
                            start[d] = 1;
                        } else if (x < 0) {
                            x += 1;
                            len[d]   = x;
                            dir[d]   = -1;
                            start[d] = L - 1;
                        }
                    }

                    idx_t edge_start = index_base + g_edges[d2] * nxedge;

                    int en = 0;
                    for (int zi = 0; zi != len[2]; zi += dir[2]) {
                        for (int yi = 0; yi != len[1]; yi += dir[1]) {
                            for (int xi = 0; xi != len[0]; xi += dir[0]) {
                                const int lidx_edge    = sshex8_lidx(L, start[0] + xi, start[1] + yi, start[2] + zi);
                                elements[lidx_edge][e] = edge_start + en;
                                en += 1;
                            }
                        }
                    }
                }
            }
        }

        free(rowptr);
        free(colidx);

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
        create_element_adj_table(m_nelements, m_nnodes, m_element_type, m_elements, &adj_table);

        idx_t n_unique_faces = 0;
        for (ptrdiff_t e = 0; e < m_nelements; e++) {
            for (int f = 0; f < 6; f++) {
                element_idx_t neigh_element = adj_table[e * 6 + f];
                // If this face is not boundary and it has already been processed continue
                if (neigh_element != SFEM_ELEMENT_IDX_INVALID && neigh_element < e) continue;

                idx_t global_face_offset = index_base + n_unique_faces * nxf;
                index_face(L, m_elements, local_side_table, lagr_to_proteus_corners, coords, global_face_offset, e, f, elements);

                if (neigh_element != SFEM_ELEMENT_IDX_INVALID) {
                    // find same face on neigh element
                    int neigh_f;
                    for (neigh_f = 0; neigh_f < 6; neigh_f++) {
                        if (e == adj_table[neigh_element * 6 + neigh_f]) {
                            break;
                        }
                    }

                    assert(neigh_f != 6);

                    index_face(L,
                               m_elements,
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
    int       nxelement      = (L - 1) * (L - 1) * (L - 1);
    ptrdiff_t interior_start = index_base;
    if (nxelement) {
        double temp_tick = MPI_Wtime();

#pragma omp parallel for collapse(3)
        for (int zi = 1; zi < L; zi++) {
            for (int yi = 1; yi < L; yi++) {
                for (int xi = 1; xi < L; xi++) {
                    const int lidx_vol = sshex8_lidx(L, xi, yi, zi);
                    int       Lm1      = L - 1;
                    int       en       = (zi - 1) * Lm1 * Lm1 + (yi - 1) * Lm1 + xi - 1;
                    for (ptrdiff_t e = 0; e < m_nelements; e++) {
                        elements[lidx_vol][e] = index_base + e * nxelement + en;
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

    *n_unique_nodes_out = interior_start + m_nelements * nxelement;
    *interior_start_out = interior_start;

    double tock = MPI_Wtime();
    printf("Create idx (%s) took\t%g [s]\n", type_to_string(m_element_type), tock - tick);
    printf("#macroelements %ld, #macronodes %ld\n", m_nelements, m_nnodes);
    printf("#microelements %ld, #micronodes %ld\n", m_nelements * (L * L * L), *n_unique_nodes_out);

    return SFEM_SUCCESS;
}

int sshex8_build_n2e(const int       L,
                     const ptrdiff_t nelements,
                     const ptrdiff_t nnodes,
                     idx_t **const   elems,
                     count_t       **out_n2eptr,
                     element_idx_t **out_elindex) {
    double tick = MPI_Wtime();

#ifdef SFEM_ENABLE_MEM_DIAGNOSTICS
    printf("build_n2e: allocating %g GB\n", (nnodes + 1) * sizeof(count_t) * 1e-9);
#endif

    count_t *n2eptr = (count_t *)malloc((nnodes + 1) * sizeof(count_t));
    memset(n2eptr, 0, (nnodes + 1) * sizeof(count_t));

    int *book_keeping = (int *)malloc((nnodes) * sizeof(int));
    memset(book_keeping, 0, (nnodes) * sizeof(int));

    const int txe = L * L * L;
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        for (int zi = 0; zi < L; zi++) {
            for (int yi = 0; yi < L; yi++) {
                for (int xi = 0; xi < L; xi++) {
                    const idx_t lev[8] = {sshex8_lidx(L, xi, yi, zi),
                                          sshex8_lidx(L, xi + 1, yi, zi),
                                          sshex8_lidx(L, xi, yi + 1, zi),
                                          sshex8_lidx(L, xi + 1, yi + 1, zi),
                                          sshex8_lidx(L, xi, yi, zi + 1),
                                          sshex8_lidx(L, xi + 1, yi, zi + 1),
                                          sshex8_lidx(L, xi, yi + 1, zi + 1),
                                          sshex8_lidx(L, xi + 1, yi + 1, zi + 1)};

                    for (int edof_i = 0; edof_i < 8; ++edof_i) {
                        assert(elems[lev[edof_i]][i] < nnodes);
                        assert(elems[lev[edof_i]][i] >= 0);
                        ++n2eptr[elems[lev[edof_i]][i] + 1];
                    }
                }
            }
        }
    }

    for (ptrdiff_t i = 0; i < nnodes; ++i) {
        n2eptr[i + 1] += n2eptr[i];
    }

#ifdef SFEM_ENABLE_MEM_DIAGNOSTICS
    printf("build_n2e: allocating %g GB\n", n2eptr[nnodes] * sizeof(element_idx_t) * 1e-9);
#endif
    element_idx_t *elindex = (element_idx_t *)malloc(n2eptr[nnodes] * sizeof(element_idx_t));

    for (ptrdiff_t i = 0; i < nelements; ++i) {
        for (int zi = 0; zi < L; zi++) {
            for (int yi = 0; yi < L; yi++) {
                for (int xi = 0; xi < L; xi++) {
                    const idx_t lev[8] = {sshex8_lidx(L, xi, yi, zi),
                                          sshex8_lidx(L, xi + 1, yi, zi),
                                          sshex8_lidx(L, xi, yi + 1, zi),
                                          sshex8_lidx(L, xi + 1, yi + 1, zi),
                                          sshex8_lidx(L, xi, yi, zi + 1),
                                          sshex8_lidx(L, xi + 1, yi, zi + 1),
                                          sshex8_lidx(L, xi, yi + 1, zi + 1),
                                          sshex8_lidx(L, xi + 1, yi + 1, zi + 1)};

                    const ptrdiff_t elidx = i * txe + zi * L * L + yi * L + xi;

                    for (int edof_i = 0; edof_i < 8; ++edof_i) {
                        const element_idx_t node = elems[lev[edof_i]][i];
                        assert(n2eptr[node] + book_keeping[node] < n2eptr[node + 1]);
                        elindex[n2eptr[node] + book_keeping[node]++] = elidx;
                    }
                }
            }
        }
    }

    free(book_keeping);

    *out_n2eptr  = n2eptr;
    *out_elindex = elindex;

    double tock = MPI_Wtime();
    printf("crs_graph.c: build_n2e\t\t%g seconds\n", tock - tick);
    return SFEM_SUCCESS;
}

static int sshex8_build_crs_graph_from_n2e(const int                                L,
                                           const ptrdiff_t                          nelements,
                                           const ptrdiff_t                          nnodes,
                                           idx_t **const SFEM_RESTRICT              elems,
                                           const count_t *const SFEM_RESTRICT       n2eptr,
                                           const element_idx_t *const SFEM_RESTRICT elindex,
                                           count_t                                **out_rowptr,
                                           idx_t                                  **out_colidx) {
    count_t *rowptr = (count_t *)malloc((nnodes + 1) * sizeof(count_t));
    idx_t   *colidx = 0;

    const int txe = L * L * L;
    {
        rowptr[0] = 0;

#pragma omp parallel
        {
            idx_t n2nbuff[4096];
#pragma omp for
            for (ptrdiff_t node = 0; node < nnodes; ++node) {
                const count_t ebegin = n2eptr[node];
                const count_t eend   = n2eptr[node + 1];

                idx_t nneighs = 0;

                for (count_t e = ebegin; e < eend; ++e) {
                    const element_idx_t eidx = elindex[e];
                    assert(eidx < nelements * txe);

                    const ptrdiff_t e_macro = eidx / txe;
                    const ptrdiff_t zi      = (eidx - e_macro * txe) / (L * L);
                    const ptrdiff_t yi      = (eidx - e_macro * txe - zi * L * L) / L;
                    const ptrdiff_t xi      = eidx - e_macro * txe - zi * L * L - yi * L;

                    const idx_t lev[8] = {sshex8_lidx(L, xi, yi, zi),
                                          sshex8_lidx(L, xi + 1, yi, zi),
                                          sshex8_lidx(L, xi, yi + 1, zi),
                                          sshex8_lidx(L, xi + 1, yi + 1, zi),
                                          sshex8_lidx(L, xi, yi, zi + 1),
                                          sshex8_lidx(L, xi + 1, yi, zi + 1),
                                          sshex8_lidx(L, xi, yi + 1, zi + 1),
                                          sshex8_lidx(L, xi + 1, yi + 1, zi + 1)};

                    for (int edof_i = 0; edof_i < 8; ++edof_i) {
                        idx_t neighnode = elems[lev[edof_i]][e_macro];
                        assert(nneighs < 4096);
                        n2nbuff[nneighs++] = neighnode;
                    }
                }

                nneighs          = sortreduce(n2nbuff, nneighs);
                rowptr[node + 1] = nneighs;
            }
        }

        // Cumulative sum
        for (ptrdiff_t node = 0; node < nnodes; ++node) {
            rowptr[node + 1] += rowptr[node];
        }

        const ptrdiff_t nnz = rowptr[nnodes];
        colidx              = (idx_t *)malloc(nnz * sizeof(idx_t));

#pragma omp parallel
        {
            idx_t n2nbuff[4096];
#pragma omp for
            for (ptrdiff_t node = 0; node < nnodes; ++node) {
                const count_t ebegin = n2eptr[node];
                const count_t eend   = n2eptr[node + 1];

                idx_t nneighs = 0;

                for (count_t e = ebegin; e < eend; ++e) {
                    const element_idx_t eidx = elindex[e];
                    assert(eidx < nelements * txe);

                    const ptrdiff_t e_macro = eidx / txe;
                    const ptrdiff_t zi      = (eidx - e_macro * txe) / (L * L);
                    const ptrdiff_t yi      = (eidx - e_macro * txe - zi * L * L) / L;
                    const ptrdiff_t xi      = eidx - e_macro * txe - zi * L * L - yi * L;

                    const idx_t lev[8] = {sshex8_lidx(L, xi, yi, zi),
                                          sshex8_lidx(L, xi + 1, yi, zi),
                                          sshex8_lidx(L, xi, yi + 1, zi),
                                          sshex8_lidx(L, xi + 1, yi + 1, zi),
                                          sshex8_lidx(L, xi, yi, zi + 1),
                                          sshex8_lidx(L, xi + 1, yi, zi + 1),
                                          sshex8_lidx(L, xi, yi + 1, zi + 1),
                                          sshex8_lidx(L, xi + 1, yi + 1, zi + 1)};

                    for (int edof_i = 0; edof_i < 8; ++edof_i) {
                        idx_t neighnode = elems[lev[edof_i]][e_macro];
                        assert(nneighs < 4096);
                        n2nbuff[nneighs++] = neighnode;
                    }
                }

                nneighs = sortreduce(n2nbuff, nneighs);
                for (idx_t i = 0; i < nneighs; ++i) {
                    assert(rowptr[node] + i < nnz);
                    colidx[rowptr[node] + i] = n2nbuff[i];
                }
            }
        }
    }

    *out_rowptr = rowptr;
    *out_colidx = colidx;
    return SFEM_SUCCESS;
}

int sshex8_crs_graph(const int       L,
                     const ptrdiff_t nelements,
                     const ptrdiff_t nnodes,
                     idx_t **const   elements,
                     count_t       **out_rowptr,
                     idx_t         **out_colidx) {
    double tick = MPI_Wtime();

    count_t       *n2eptr;
    element_idx_t *elindex;
    sshex8_build_n2e(L, nelements, nnodes, elements, &n2eptr, &elindex);

    int err = sshex8_build_crs_graph_from_n2e(L, nelements, nnodes, elements, n2eptr, elindex, out_rowptr, out_colidx);

    free(n2eptr);
    free(elindex);

    double tock = MPI_Wtime();
    printf("sshex8_crs_graph \t%g seconds\n", tock - tick);
    return err;
}

int sshex8_hierarchical_n_levels(const int L) {
    int count = 0;
    int l     = L;
    for (; l > 1 && l % 2 == 0; l /= 2) {
        count++;
    }

    if (l >= 1) {
        count++;
    }

    return count;
}

void sshex8_hierarchical_mesh_levels(const int L, const int nlevels, int *const levels) {
    assert(sshex8_hierarchical_n_levels(L) == nlevels);

    int count = 0;
    int l     = L;
    for (; l > 1 && l % 2 == 0; l /= 2) {
        levels[nlevels - 1 - count] = l;
        count++;
    }

    if (l >= 1) {
        count++;
        levels[0] = 1;
    }
}

int sshex8_hierarchical_renumbering(const int       L,
                                    const int       nlevels,
                                    int *const      levels,
                                    const ptrdiff_t nelements,
                                    const ptrdiff_t nnodes,
                                    idx_t **const   elements) {
    idx_t *node_mapping = malloc(nnodes * sizeof(idx_t));
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nnodes; i++) {
        node_mapping[i] = SFEM_IDX_INVALID;
    }

    idx_t next_id = 0;
    // Preserve original ordering for base HEX8 mesh
    for (int zi = 0; zi <= 1; zi++) {
        for (int yi = 0; yi <= 1; yi++) {
            for (int xi = 0; xi <= 1; xi++) {
                for (ptrdiff_t e = 0; e < nelements; e++) {
                    const int   v   = sshex8_lidx(L, xi * L, yi * L, zi * L);
                    const idx_t idx = elements[v][e];

                    assert(idx < nnodes);
                    node_mapping[idx] = idx;
                    next_id           = MAX(next_id, idx);
                }
            }
        }
    }

    next_id++;

    int stride = 1;
    for (int k = 1; k < nlevels; k++) {
        const int l           = levels[k];
        const int step_factor = L / l;

        for (ptrdiff_t e = 0; e < nelements; e++) {
            for (int zi = 0; zi <= l; zi += stride) {
                for (int yi = 0; yi <= l; yi += stride) {
                    for (int xi = 0; xi <= l; xi += stride) {
                        const int   v   = sshex8_lidx(L, xi * step_factor, yi * step_factor, zi * step_factor);
                        const idx_t idx = elements[v][e];
                        assert(idx < nnodes);
                        if (node_mapping[idx] == SFEM_IDX_INVALID) {
                            node_mapping[idx] = next_id++;
                        }
                    }
                }
            }
        }
    }

    for (int zi = 0; zi <= L; zi++) {
        for (int yi = 0; yi <= L; yi++) {
            for (int xi = 0; xi <= L; xi++) {
                for (ptrdiff_t e = 0; e < nelements; e++) {
                    const int   v   = sshex8_lidx(L, xi, yi, zi);
                    const idx_t idx = elements[v][e];

                    if (node_mapping[idx] == SFEM_IDX_INVALID) {
                        printf("%d %d %d [%ld]\n", xi, yi, zi, e);
                        SFEM_ERROR("Uninitialized node mapping\n");
                    }

                    elements[v][e] = node_mapping[idx];
                }
            }
        }
    }

    free(node_mapping);
    return SFEM_SUCCESS;
}

static void sshex8_x_lside(const int                   L,
                           const ptrdiff_t             e,
                           idx_t **const SFEM_RESTRICT elems,
                           const ptrdiff_t             i,
                           idx_t **const SFEM_RESTRICT sides) {
    int lidx = 0;
    for (int zi = 0; zi <= L; zi++) {
        for (int yi = L; yi >= 0; yi--) {
            sides[lidx++][i] = elems[sshex8_lidx(L, 0, yi, zi)][e];
        }
    }
}

static void sshex8_x_rside(const int                   L,
                           const ptrdiff_t             e,
                           idx_t **const SFEM_RESTRICT elems,
                           const ptrdiff_t             i,
                           idx_t **const SFEM_RESTRICT sides) {
    int lidx = 0;
    for (int zi = 0; zi <= L; zi++) {
        for (int yi = 0; yi <= L; yi++) {
            sides[lidx++][i] = elems[sshex8_lidx(L, L, yi, zi)][e];
        }
    }
}

//////////////

static void sshex8_y_lside(const int                   L,
                           const ptrdiff_t             e,
                           idx_t **const SFEM_RESTRICT elems,
                           const ptrdiff_t             i,
                           idx_t **const SFEM_RESTRICT sides) {
    int lidx = 0;
    for (int zi = 0; zi <= L; zi++) {
        for (int xi = L; xi >= 0; xi--) {
            sides[lidx++][i] = elems[sshex8_lidx(L, xi, 0, zi)][e];
        }
    }
}

static void sshex8_y_rside(const int                   L,
                           const ptrdiff_t             e,
                           idx_t **const SFEM_RESTRICT elems,
                           const ptrdiff_t             i,
                           idx_t **const SFEM_RESTRICT sides) {
    int lidx = 0;
    for (int zi = 0; zi <= L; zi++) {
        for (int xi = 0; xi <= L; xi++) {
            sides[lidx++][i] = elems[sshex8_lidx(L, xi, L, zi)][e];
        }
    }
}

//////////////

static void sshex8_z_lside(const int                   L,
                           const ptrdiff_t             e,
                           idx_t **const SFEM_RESTRICT elems,
                           const ptrdiff_t             i,
                           idx_t **const SFEM_RESTRICT sides) {
    int lidx = 0;
    for (int yi = L; yi >= 0; yi--) {
        for (int xi = 0; xi <= L; xi++) {
            sides[lidx++][i] = elems[sshex8_lidx(L, xi, yi, 0)][e];
        }
    }
}

static void sshex8_z_rside(const int                   L,
                           const ptrdiff_t             e,
                           idx_t **const SFEM_RESTRICT elems,
                           const ptrdiff_t             i,
                           idx_t **const SFEM_RESTRICT sides) {
    int lidx = 0;
    for (int yi = 0; yi <= L; yi++) {
        for (int xi = 0; xi <= L; xi++) {
            sides[lidx++][i] = elems[sshex8_lidx(L, xi, yi, L)][e];
        }
    }
}

//////////////

int         sshex8_extract_surface_from_sideset(const int                                L,
                                                idx_t **const SFEM_RESTRICT              elems,
                                                const ptrdiff_t                          n_surf_elements,
                                                const element_idx_t *const SFEM_RESTRICT parent_element,
                                                const int16_t *const SFEM_RESTRICT       side_idx,
                                                idx_t **const SFEM_RESTRICT              sides) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < n_surf_elements; i++) {
        const ptrdiff_t e = parent_element[i];
        const int       s = side_idx[i];

        switch (s) {
            case 0: {
                sshex8_y_lside(L, e, elems, i, sides);
                break;
            }
            case 1: {
                sshex8_x_rside(L, e, elems, i, sides);
                break;
            }
            case 2: {
                sshex8_y_rside(L, e, elems, i, sides);
                break;
            }
            case 3: {
                sshex8_x_lside(L, e, elems, i, sides);
                break;
            }
            case 4: {
                sshex8_z_lside(L, e, elems, i, sides);
                break;
            }
            case 5: {
                sshex8_z_rside(L, e, elems, i, sides);
                break;
            }
            default: {
                assert(0);
                break;
            }
        }
    }

    return SFEM_SUCCESS;
}

static void sshex8_x_lside_nodes(const int                   L,
                                 const ptrdiff_t             e,
                                 idx_t **const SFEM_RESTRICT elems,
                                 idx_t *const SFEM_RESTRICT  nodes) {
    int lidx = 0;
    for (int zi = 0; zi <= L; zi++) {
        for (int yi = L; yi >= 0; yi--) {
            nodes[lidx++] = elems[sshex8_lidx(L, 0, yi, zi)][e];
        }
    }
}

static void sshex8_x_rside_nodes(const int                   L,
                                 const ptrdiff_t             e,
                                 idx_t **const SFEM_RESTRICT elems,
                                 idx_t *const SFEM_RESTRICT  nodes) {
    int lidx = 0;
    for (int zi = 0; zi <= L; zi++) {
        for (int yi = 0; yi <= L; yi++) {
            nodes[lidx++] = elems[sshex8_lidx(L, L, yi, zi)][e];
        }
    }
}

//////////////

static void sshex8_y_lside_nodes(const int                   L,
                                 const ptrdiff_t             e,
                                 idx_t **const SFEM_RESTRICT elems,
                                 idx_t *const SFEM_RESTRICT  nodes) {
    int lidx = 0;
    for (int zi = 0; zi <= L; zi++) {
        for (int xi = L; xi >= 0; xi--) {
            nodes[lidx++] = elems[sshex8_lidx(L, xi, 0, zi)][e];
        }
    }
}

static void sshex8_y_rside_nodes(const int                   L,
                                 const ptrdiff_t             e,
                                 idx_t **const SFEM_RESTRICT elems,
                                 idx_t *const SFEM_RESTRICT  nodes) {
    int lidx = 0;
    for (int zi = 0; zi <= L; zi++) {
        for (int xi = 0; xi <= L; xi++) {
            nodes[lidx++] = elems[sshex8_lidx(L, xi, L, zi)][e];
        }
    }
}

//////////////

static void sshex8_z_lside_nodes(const int                   L,
                                 const ptrdiff_t             e,
                                 idx_t **const SFEM_RESTRICT elems,
                                 idx_t *const SFEM_RESTRICT  nodes) {
    int lidx = 0;
    for (int yi = L; yi >= 0; yi--) {
        for (int xi = 0; xi <= L; xi++) {
            nodes[lidx++] = elems[sshex8_lidx(L, xi, yi, 0)][e];
        }
    }
}

static void sshex8_z_rside_nodes(const int                   L,
                                 const ptrdiff_t             e,
                                 idx_t **const SFEM_RESTRICT elems,
                                 idx_t *const SFEM_RESTRICT  nodes) {
    int lidx = 0;
    for (int yi = 0; yi <= L; yi++) {
        for (int xi = 0; xi <= L; xi++) {
            nodes[lidx++] = elems[sshex8_lidx(L, xi, yi, L)][e];
        }
    }
}

int sshex8_extract_nodeset_from_sideset(const int                                L,
                                        idx_t **const SFEM_RESTRICT              elems,
                                        const ptrdiff_t                          n_surf_elements,
                                        const element_idx_t *const SFEM_RESTRICT parent_element,
                                        const int16_t *const SFEM_RESTRICT       side_idx,
                                        ptrdiff_t                               *n_nodes_out,
                                        idx_t **SFEM_RESTRICT                    nodes_out)

{
    const int       nnxs  = (L + 1) * (L + 1);
    const ptrdiff_t n     = nnxs * n_surf_elements;
    idx_t          *nodes = malloc(n * sizeof(idx_t));

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < n_surf_elements; i++) {
        const ptrdiff_t e = parent_element[i];
        const int       s = side_idx[i];

        switch (s) {
            case 0: {
                sshex8_y_lside_nodes(L, e, elems, &nodes[i * nnxs]);
                break;
            }
            case 1: {
                sshex8_x_rside_nodes(L, e, elems, &nodes[i * nnxs]);
                break;
            }
            case 2: {
                sshex8_y_rside_nodes(L, e, elems, &nodes[i * nnxs]);
                break;
            }
            case 3: {
                sshex8_x_lside_nodes(L, e, elems, &nodes[i * nnxs]);
                break;
            }
            case 4: {
                sshex8_z_lside_nodes(L, e, elems, &nodes[i * nnxs]);
                break;
            }
            case 5: {
                sshex8_z_rside_nodes(L, e, elems, &nodes[i * nnxs]);
                break;
            }
            default: {
                assert(0);
                break;
            }
        }
    }

    *n_nodes_out = sortreduce(nodes, n);
    *nodes_out   = realloc(nodes, *n_nodes_out * sizeof(idx_t));
    return SFEM_SUCCESS;
}

// 1) 1, 2, 6, 5, ---  9, 14, 17, 13, 26 (Y left)
// 2) 2, 3, 7, 6, --- 10, 15, 18, 14, 25 (X right)
// 3) 3, 4, 8, 7, --- 11, 16, 19, 15, 27 (Y right)
// 4) 1, 5, 8, 4, --- 13, 20, 16, 12, 24 (X left)
// 5) 1, 4, 3, 2, --- 12, 11, 10, 9,  22 (Z left)
// 6) 5, 6, 7, 8, --- 17, 18, 19, 20, 23 (Z right)

// Front face
// 5 ---------- 6
// |            |
// |            |
// |            |
// |            |
// 1/___________2/  x -->

// Back face

// /8 ----------/7
// |            |
// |            |
// |            |
// |            |
// /4___________/3

static void sshex8_x_lside_quadshell4(const int                   L,
                                      const ptrdiff_t             e,
                                      idx_t **const SFEM_RESTRICT elems,
                                      const ptrdiff_t             offset,
                                      idx_t **const SFEM_RESTRICT sides) {
    int lidx = 0;
    for (int zi = 0; zi < L; zi++) {
        for (int yi = 0; yi < L; yi++) {
            // 1, 5, 8, 4,
            const idx_t i0 = sshex8_lidx(L, 0, yi, zi);
            const idx_t i1 = sshex8_lidx(L, 0, yi, zi + 1);
            const idx_t i2 = sshex8_lidx(L, 0, yi + 1, zi + 1);
            const idx_t i3 = sshex8_lidx(L, 0, yi + 1, zi);

            sides[0][offset + lidx] = elems[i0][e];
            sides[1][offset + lidx] = elems[i1][e];
            sides[2][offset + lidx] = elems[i2][e];
            sides[3][offset + lidx] = elems[i3][e];
            lidx++;
        }
    }
}

static void sshex8_x_rside_quadshell4(const int                   L,
                                      const ptrdiff_t             e,
                                      idx_t **const SFEM_RESTRICT elems,
                                      const ptrdiff_t             offset,
                                      idx_t **const SFEM_RESTRICT sides) {
    int lidx = 0;
    for (int zi = 0; zi < L; zi++) {
        for (int yi = 0; yi < L; yi++) {
            // 2, 3, 7, 6
            const idx_t i0 = sshex8_lidx(L, L, yi, zi);
            const idx_t i1 = sshex8_lidx(L, L, yi + 1, zi);
            const idx_t i2 = sshex8_lidx(L, L, yi + 1, zi + 1);
            const idx_t i3 = sshex8_lidx(L, L, yi, zi + 1);

            sides[0][offset + lidx] = elems[i0][e];
            sides[1][offset + lidx] = elems[i1][e];
            sides[2][offset + lidx] = elems[i2][e];
            sides[3][offset + lidx] = elems[i3][e];
            lidx++;
        }
    }
}

//////////////

static void sshex8_y_lside_quadshell4(const int                   L,
                                      const ptrdiff_t             e,
                                      idx_t **const SFEM_RESTRICT elems,
                                      const ptrdiff_t             offset,
                                      idx_t **const SFEM_RESTRICT sides) {
    int lidx = 0;
    for (int zi = 0; zi < L; zi++) {
        for (int xi = 0; xi < L; xi++) {
            // 1, 2, 6, 5
            const idx_t i0 = sshex8_lidx(L, xi, 0, zi);
            const idx_t i1 = sshex8_lidx(L, xi + 1, 0, zi);
            const idx_t i2 = sshex8_lidx(L, xi + 1, 0, zi + 1);
            const idx_t i3 = sshex8_lidx(L, xi, 0, zi + 1);

            sides[0][offset + lidx] = elems[i0][e];
            sides[1][offset + lidx] = elems[i1][e];
            sides[2][offset + lidx] = elems[i2][e];
            sides[3][offset + lidx] = elems[i3][e];
            lidx++;
        }
    }
}

static void sshex8_y_rside_quadshell4(const int                   L,
                                      const ptrdiff_t             e,
                                      idx_t **const SFEM_RESTRICT elems,
                                      const ptrdiff_t             offset,
                                      idx_t **const SFEM_RESTRICT sides) {
    int lidx = 0;
    for (int zi = 0; zi < L; zi++) {
        for (int xi = 0; xi < L; xi++) {
            // 3, 4, 8, 7
            const idx_t i0 = sshex8_lidx(L, xi + 1, L, zi);
            const idx_t i1 = sshex8_lidx(L, xi, L, zi);
            const idx_t i2 = sshex8_lidx(L, xi, L, zi + 1);
            const idx_t i3 = sshex8_lidx(L, xi + 1, L, zi + 1);

            sides[0][offset + lidx] = elems[i0][e];
            sides[1][offset + lidx] = elems[i1][e];
            sides[2][offset + lidx] = elems[i2][e];
            sides[3][offset + lidx] = elems[i3][e];
            lidx++;
        }
    }
}

//////////////

static void sshex8_z_lside_quadshell4(const int                   L,
                                      const ptrdiff_t             e,
                                      idx_t **const SFEM_RESTRICT elems,
                                      const ptrdiff_t             offset,
                                      idx_t **const SFEM_RESTRICT sides) {
    int lidx = 0;
    for (int yi = 0; yi < L; yi++) {
        for (int xi = 0; xi < L; xi++) {
            // 1, 4, 3, 2
            const idx_t i0 = sshex8_lidx(L, xi, yi, 0);
            const idx_t i1 = sshex8_lidx(L, xi, yi + 1, 0);
            const idx_t i2 = sshex8_lidx(L, xi + 1, yi + 1, 0);
            const idx_t i3 = sshex8_lidx(L, xi + 1, yi, 0);

            sides[0][offset + lidx] = elems[i0][e];
            sides[1][offset + lidx] = elems[i1][e];
            sides[2][offset + lidx] = elems[i2][e];
            sides[3][offset + lidx] = elems[i3][e];
            lidx++;
        }
    }
}

static void sshex8_z_rside_quadshell4(const int                   L,
                                      const ptrdiff_t             e,
                                      idx_t **const SFEM_RESTRICT elems,
                                      const ptrdiff_t             offset,
                                      idx_t **const SFEM_RESTRICT sides) {
    int lidx = 0;
    for (int yi = 0; yi < L; yi++) {
        for (int xi = 0; xi < L; xi++) {
            // 5, 6, 7, 8
            const idx_t i0 = sshex8_lidx(L, xi, yi, L);
            const idx_t i1 = sshex8_lidx(L, xi + 1, yi, L);
            const idx_t i2 = sshex8_lidx(L, xi + 1, yi + 1, L);
            const idx_t i3 = sshex8_lidx(L, xi, yi + 1, L);

            sides[0][offset + lidx] = elems[i0][e];
            sides[1][offset + lidx] = elems[i1][e];
            sides[2][offset + lidx] = elems[i2][e];
            sides[3][offset + lidx] = elems[i3][e];
            lidx++;
        }
    }
}

//////////////

int sshex8_extract_quadshell4_surface_from_sideset(const int                                L,
                                                   idx_t **const SFEM_RESTRICT              elems,
                                                   const ptrdiff_t                          n_surf_elements,
                                                   const element_idx_t *const SFEM_RESTRICT parent_element,
                                                   const int16_t *const SFEM_RESTRICT       side_idx,
                                                   idx_t **const SFEM_RESTRICT              sides) {
    const int nquadsxside = L * L;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < n_surf_elements; i++) {
        const ptrdiff_t e = parent_element[i];
        const int       s = side_idx[i];

        switch (s) {
            case 0: {
                sshex8_y_lside_quadshell4(L, e, elems, i * nquadsxside, sides);
                break;
            }
            case 1: {
                sshex8_x_rside_quadshell4(L, e, elems, i * nquadsxside, sides);
                break;
            }
            case 2: {
                sshex8_y_rside_quadshell4(L, e, elems, i * nquadsxside, sides);
                break;
            }
            case 3: {
                sshex8_x_lside_quadshell4(L, e, elems, i * nquadsxside, sides);
                break;
            }
            case 4: {
                sshex8_z_lside_quadshell4(L, e, elems, i * nquadsxside, sides);
                break;
            }
            case 5: {
                sshex8_z_rside_quadshell4(L, e, elems, i * nquadsxside, sides);
                break;
            }
            default: {
                assert(0);
                break;
            }
        }
    }

    return SFEM_SUCCESS;
}

int ssquad4_hierarchical_remapping(const int                   L,
                                   const int                   nlevels,
                                   int *const                  levels,
                                   const ptrdiff_t             nelements,
                                   const ptrdiff_t             nnodes,
                                   idx_t **const SFEM_RESTRICT elements,
                                   idx_t **SFEM_RESTRICT       node_mapping_out,
                                   ptrdiff_t                  *count_out) {
    idx_t *node_mapping = malloc(nnodes * sizeof(idx_t));

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nnodes; i++) {
        node_mapping[i] = SFEM_IDX_INVALID;
    }

    idx_t next_id = 0;
    // Preserve original ordering for base SSQUAD4 mesh

    for (int yi = 0; yi <= 1; yi++) {
        for (int xi = 0; xi <= 1; xi++) {
            for (ptrdiff_t e = 0; e < nelements; e++) {
                const int   v   = ssquad4_lidx(L, xi * L, yi * L);
                const idx_t idx = elements[v][e];

                assert(idx < nnodes);
                if (node_mapping[idx] == SFEM_IDX_INVALID) {
                    node_mapping[idx] = next_id++;
                }
            }
        }
    }

    int stride = 1;
    for (int k = 1; k < nlevels; k++) {
        const int l           = levels[k];
        const int step_factor = L / l;

        for (ptrdiff_t e = 0; e < nelements; e++) {
            for (int yi = 0; yi <= l; yi += stride) {
                for (int xi = 0; xi <= l; xi += stride) {
                    const int   v   = ssquad4_lidx(L, xi * step_factor, yi * step_factor);
                    const idx_t idx = elements[v][e];
                    assert(idx < nnodes);
                    if (node_mapping[idx] == SFEM_IDX_INVALID) {
                        node_mapping[idx] = next_id++;
                    }
                }
            }
        }
    }

    for (int yi = 0; yi <= L; yi++) {
        for (int xi = 0; xi <= L; xi++) {
            for (ptrdiff_t e = 0; e < nelements; e++) {
                const int   v   = ssquad4_lidx(L, xi, yi);
                const idx_t idx = elements[v][e];

                if (node_mapping[idx] == SFEM_IDX_INVALID) {
                    printf("%d %d[%ld]\n", xi, yi, e);
                    SFEM_ERROR("Uninitialized node mapping\n");
                }

                elements[v][e] = node_mapping[idx];
            }
        }
    }

    *count_out        = next_id;
    *node_mapping_out = malloc(*count_out * sizeof(idx_t));

#ifndef NDEBUG
    for (ptrdiff_t i = 0; i < *count_out; i++) {
        (*node_mapping_out)[i] = SFEM_IDX_INVALID;
    }

#endif

    for (ptrdiff_t i = 0; i < nnodes; i++) {
        if (node_mapping[i] != SFEM_IDX_INVALID) {
            (*node_mapping_out)[node_mapping[i]] = i;
        }
    }

#ifndef NDEBUG
    for (ptrdiff_t i = 0; i < *count_out; i++) {
        assert((*node_mapping_out)[i] != SFEM_IDX_INVALID);
    }

#endif

    free(node_mapping);
    return SFEM_SUCCESS;
}
