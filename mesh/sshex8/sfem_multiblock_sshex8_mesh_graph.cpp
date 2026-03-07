

// int sshex8_generate_elements_multi_block(const int           L,
//                                          const int           n_blocks,
//                                          const smesh::ElemType element_types[],
//                                          const ptrdiff_t     m_nelements[],
//                                          const ptrdiff_t     m_nnodes[],
//                                          idx_t **const       m_elements[],
//                                          idx_t             **elements[],
//                                          ptrdiff_t          *n_unique_nodes_out,
//                                          ptrdiff_t          *interior_start_out) {
//     assert(L >= 2);
//     static int verbose = 0;

//     for (int b = 0; b < n_blocks; b++) {
//         assert(element_types[b] == smesh::HEX8);
//     }

//     double tick = MPI_Wtime();

//     const int nxe = sshex8_nxe(L);

//     // 1) Get the node indices from the smesh::HEX8 mesh
//     int lagr_to_proteus_corners[8] = {// Bottom
//                                       sshex8_lidx(L, 0, 0, 0),
//                                       sshex8_lidx(L, L, 0, 0),
//                                       sshex8_lidx(L, L, L, 0),
//                                       sshex8_lidx(L, 0, L, 0),
//                                       // Top
//                                       sshex8_lidx(L, 0, 0, L),
//                                       sshex8_lidx(L, L, 0, L),
//                                       sshex8_lidx(L, L, L, L),
//                                       sshex8_lidx(L, 0, L, L)};

// #ifndef NDEBUG
//     for (int i = 0; i < 8; i++) {
//         assert(lagr_to_proteus_corners[i] < nxe);
//     }
// #endif

//     int *coords[3];
//     for (int d = 0; d < 3; d++) {
//         coords[d] = malloc(nxe * sizeof(int));
//     }

//     for (int zi = 0; zi <= L; zi++) {
//         for (int yi = 0; yi <= L; yi++) {
//             for (int xi = 0; xi <= L; xi++) {
//                 int lidx = sshex8_lidx(L, xi, yi, zi);
//                 assert(lidx < nxe);
//                 coords[0][lidx] = xi;
//                 coords[1][lidx] = yi;
//                 coords[2][lidx] = zi;
//             }
//         }
//     }

//     // ------------------------------
//     // Corner nodes
//     // ------------------------------

//     for (int b = 0; b < n_blocks; b++) {
//         for (int d = 0; d < 8; d++) {
//             for (ptrdiff_t e = 0; e < m_nelements[b]; e++) {
//                 elements[b][lagr_to_proteus_corners[d]][e] = m_elements[b][d][e];
//             }
//         }
//     }

//     idx_t index_base = m_nnodes;

//     double tack = MPI_Wtime();

//     if (verbose) printf("NODES\t%g [s]\n", tack - tick);

//     // 2) Compute the unique edge-node indices using the CRSGraph
//     // A unique edge index can be used and use the multiple to store all indices
//     // as consecutive

//     // Number of nodes in the edge interior
//     ptrdiff_t nxedge = L - 1;  // L == 0 (is this correct?)

//     if (nxedge) {
//         double temp_tick = MPI_Wtime();

//         count_t *rowptr;
//         idx_t   *colidx;
//         build_multiblock_crs_graph(n_blocks, element_types, m_nelements, m_elements, m_nnodes, &rowptr, &colidx);

//         ptrdiff_t nedges = rowptr[m_nnodes] / 2;

//         ptrdiff_t nnz      = rowptr[m_nnodes];
//         idx_t    *edge_idx = (idx_t *)malloc(nnz * sizeof(idx_t));
//         memset(edge_idx, 0, nnz * sizeof(idx_t));

//         // node-to-node for the hex edges in local indexing
//         idx_t lagr_connectivity[8][3] = {// BOTTOM
//                                          {1, 3, 4},
//                                          {0, 2, 5},
//                                          {1, 3, 6},
//                                          {0, 2, 7},
//                                          // TOP
//                                          {0, 5, 7},
//                                          {1, 4, 6},
//                                          {2, 5, 7},
//                                          {3, 4, 6}};

//         ptrdiff_t edge_count = 0;
//         idx_t     next_id    = 0;
//         for (ptrdiff_t i = 0; i < m_nnodes; i++) {
//             const count_t begin = rowptr[i];
//             const count_t end   = rowptr[i + 1];

//             for (count_t k = begin; k < end; k++) {
//                 const idx_t j = colidx[k];

//                 if (i < j) {
//                     edge_count += 1;
//                     edge_idx[k] = next_id++;
//                 }
//             }
//         }

//         assert(edge_count == nedges);

//         for (int b = 0; b < n_blocks; b++) {
//             for (ptrdiff_t e = 0; e < m_nelements[b]; e++) {
//                 idx_t nodes[8];
//                 for (int d = 0; d < 8; d++) {
//                     nodes[d] = m_elements[b][d][e];
//                 }

//                 for (int d1 = 0; d1 < 8; d1++) {
//                     idx_t              node1     = nodes[d1];
//                     const idx_t *const columns   = &colidx[rowptr[node1]];
//                     const idx_t *const edge_view = &edge_idx[rowptr[node1]];

//                     idx_t g_edges[3];
//                     idx_t g_neigh[3];

//                     for (int k = 0; k < 3; k++) {
//                         g_neigh[k] = nodes[lagr_connectivity[d1][k]];
//                     }

//                     idx_t offsets[3];
//                     hex8_find_corner_cols(g_neigh, columns, rowptr[node1 + 1] - rowptr[node1], offsets);

//                     for (int d = 0; d < 3; d++) {
//                         g_edges[d] = edge_view[offsets[d]];
//                     }

//                     for (int d2 = 0; d2 < 3; d2++) {
//                         const idx_t node2 = g_neigh[d2];

//                         // direction of edge is always smaller node id to greater node id
//                         if (node1 > node2) continue;

//                         const int lid1 = lagr_to_proteus_corners[d1];
//                         const int lid2 = lagr_to_proteus_corners[lagr_connectivity[d1][d2]];

//                         int start[3], len[3], dir[3];
//                         for (int d = 0; d < 3; d++) {
//                             int o    = coords[d][lid1];
//                             start[d] = o;
//                         }

//                         int invert_dir = 0;
//                         for (int d = 0; d < 3; d++) {
//                             int x  = coords[d][lid2] - coords[d][lid1];
//                             dir[d] = 1;
//                             len[d] = 1;

//                             if (x > 0) {
//                                 x -= 1;
//                                 len[d]   = x;
//                                 start[d] = 1;
//                             } else if (x < 0) {
//                                 x += 1;
//                                 len[d]   = x;
//                                 dir[d]   = -1;
//                                 start[d] = L - 1;
//                             }
//                         }

//                         idx_t edge_start = index_base + g_edges[d2] * nxedge;

//                         int en = 0;
//                         for (int zi = 0; zi != len[2]; zi += dir[2]) {
//                             for (int yi = 0; yi != len[1]; yi += dir[1]) {
//                                 for (int xi = 0; xi != len[0]; xi += dir[0]) {
//                                     const int lidx_edge    = sshex8_lidx(L, start[0] + xi, start[1] + yi, start[2] + zi);
//                                     elements[lidx_edge][e] = edge_start + en;
//                                     en += 1;
//                                 }
//                             }
//                         }
//                     }
//                 }
//             }
//         }

//         free(rowptr);
//         free(colidx);

//         index_base += (nedges * nxedge);

//         tack = MPI_Wtime();
//         if (verbose) printf("EDGES\t%g [s]\n", tack - temp_tick);
//     }

//     // 3) Compute the unique face-node indices using the adjacency table
//     // Two elements share a face, figure out the ordering
//     int nxf = (L - 1) * (L - 1);  // TODO number of nodes in the face interior
//     if (nxf) {
//         double temp_tick = MPI_Wtime();

//         int local_side_table[6 * 4];
//         fill_local_side_table(smesh::HEX8, local_side_table);

//         element_idx_t *adj_table = 0;
//         create_element_adj_table(m_nelements, m_nnodes, m_element_type, m_elements, &adj_table);

//         idx_t n_unique_faces = 0;
//         for (ptrdiff_t e = 0; e < m_nelements; e++) {
//             for (int f = 0; f < 6; f++) {
//                 element_idx_t neigh_element = adj_table[e * 6 + f];
//                 // If this face is not boundary and it has already been processed continue
//                 if (neigh_element != SFEM_ELEMENT_IDX_INVALID && neigh_element < e) continue;

//                 idx_t global_face_offset = index_base + n_unique_faces * nxf;
//                 index_face(L, m_elements, local_side_table, lagr_to_proteus_corners, coords, global_face_offset, e, f, elements);

//                 if (neigh_element != SFEM_ELEMENT_IDX_INVALID) {
//                     // find same face on neigh element
//                     int neigh_f;
//                     for (neigh_f = 0; neigh_f < 6; neigh_f++) {
//                         if (e == adj_table[neigh_element * 6 + neigh_f]) {
//                             break;
//                         }
//                     }

//                     assert(neigh_f != 6);

//                     index_face(L,
//                                m_elements,
//                                local_side_table,
//                                lagr_to_proteus_corners,
//                                coords,
//                                global_face_offset,
//                                neigh_element,
//                                neigh_f,
//                                elements);
//                 }

//                 // Next id
//                 n_unique_faces++;
//             }
//         }

//         index_base += n_unique_faces * nxf;

//         // Clean-up
//         free(adj_table);

//         tack = MPI_Wtime();
//         if (verbose) printf("FACES\t%g [s]\n", tack - temp_tick);
//     }

//     // 4) Compute the unique internal nodes implicitly using the element id and the idx offset
//     // of the total number of explicit indices (offset + element_id * n_internal_nodes +
//     // local_internal_node_id) ptrdiff_t n_internal_nodes = ?;
//     int       nxelement      = (L - 1) * (L - 1) * (L - 1);
//     ptrdiff_t interior_start = index_base;
//     if (nxelement) {
//         double temp_tick = MPI_Wtime();

// #pragma omp parallel for collapse(3)
//         for (int zi = 1; zi < L; zi++) {
//             for (int yi = 1; yi < L; yi++) {
//                 for (int xi = 1; xi < L; xi++) {
//                     const int lidx_vol = sshex8_lidx(L, xi, yi, zi);
//                     int       Lm1      = L - 1;
//                     int       en       = (zi - 1) * Lm1 * Lm1 + (yi - 1) * Lm1 + xi - 1;
//                     for (ptrdiff_t e = 0; e < m_nelements; e++) {
//                         elements[lidx_vol][e] = index_base + e * nxelement + en;
//                     }
//                 }
//             }
//         }

//         tack = MPI_Wtime();
//         if (verbose) printf("ELEMS\t%g [s]\n", tack - temp_tick);
//     }

//     for (int d = 0; d < 3; d++) {
//         free(coords[d]);
//     }

//     *n_unique_nodes_out = interior_start + m_nelements * nxelement;
//     *interior_start_out = interior_start;

//     double tock = MPI_Wtime();
//     printf("Create idx (%s) took\t%g [s]\n", type_to_string(m_element_type), tock - tick);
//     printf("#macroelements %ld, #macronodes %ld\n", m_nelements, m_nnodes);
//     printf("#microelements %ld, #micronodes %ld\n", m_nelements * (L * L * L), *n_unique_nodes_out);

//     return SFEM_SUCCESS;
// }
