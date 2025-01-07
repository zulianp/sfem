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
#include "sstet4_laplacian.h"
#include "tet4_fff.h"

#include "adj_table.h"

#include "tet4_inline_cpu.h"

// Adj Table
// (Face, node)
// LST(0, 0) = 1 - 1;
// LST(0, 1) = 2 - 1;
// LST(0, 2) = 4 - 1;

// LST(1, 0) = 2 - 1;
// LST(1, 1) = 3 - 1;
// LST(1, 2) = 4 - 1;

// LST(2, 0) = 1 - 1;
// LST(2, 1) = 4 - 1;
// LST(2, 2) = 3 - 1;

// LST(3, 0) = 1 - 1;
// LST(3, 1) = 3 - 1;
// LST(3, 2) = 2 - 1;

#define SFEM_INVALID_IDX (-1)

static void sstet4_create_full_idx(const int L, mesh_t *mesh, idx_t **elements) {
    double tick = MPI_Wtime();

    const int nxe = sstet4_nxe(L);

    // 1) Get the node indices from the TET4 mesh
    int corner_lidx[4] = {0, L, (L + 1) * (L + 2) / 2 - 1, nxe - 1};  // TODO check if correct
    printf("level %d, nxe %d, (%d, %d, %d, %d)\n",
           L,
           nxe,
           corner_lidx[0],
           corner_lidx[1],
           corner_lidx[2],
           corner_lidx[3]);

    for (int d = 0; d < 4; d++) {
        for (ptrdiff_t e = 0; e < mesh->nelements; e++) {
            elements[corner_lidx[d]][e] = mesh->elements[d][e];
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
        idx_t next_id = index_base;
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
            idx_t nodes[4];
            for (int d = 0; d < 4; d++) {
                nodes[d] = mesh->elements[d][e];
            }

            for (int d = 0; d < 4; d++) {
                idx_t node = nodes[d];

                idx_t edges[4];  // for same node edge is 0
                {
                    const idx_t *const columns = &colidx[rowptr[node]];

                    idx_t offsets[4];
                    tet4_find_cols(nodes, columns, rowptr[node + 1] - rowptr[node], offsets);

                    for (int d = 0; d < 4; d++) {
                        edges[d] = columns[offsets[d]];
                    }
                }

                for (int d1 = 0; d1 < 4; d1++) {
                    idx_t node1 = nodes[d1];

                    for (int d2 = 0; d2 < 4; d2++) {
                        idx_t node2 = nodes[d2];
                        if (node1 > node2) continue;

                        // direction of edge is always smaller node id to greater node id
                        for (int en = 0; en < nxedge; en++) {
                            assert(edges[d2]);  // edge must not be 0
                            idx_t en_id = edges[d2] * nxedge + en;

                            // TODO find out local ordering and place it in elements
                        }
                    }
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

    // TODO Consistent ordering with implicit looping scheme needs to be figured out (orientation is
    // reflected like mirror)

    // 4) Compute the unique internal nodes implicitly using the element id and the idx offset of
    // the total number of explicit indices (offset + element_id * n_internal_nodes +
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

    int L = 8;

    const int nxe = sstet4_nxe(L);
    const int txe = sstet4_txe(L);
    ptrdiff_t nnodes_discont = mesh.nelements * nxe;

    idx_t **elements = 0;

    if (SFEM_USE_IDX) {
        elements = malloc(nxe * sizeof(idx_t *));
        for (int d = 0; d < nxe; d++) {
            elements[d] = malloc(mesh.nelements * sizeof(idx_t));
        }

        sstet4_create_full_idx(L, &mesh, elements);
    }

    real_t *x = calloc(nnodes_discont, sizeof(real_t));
    real_t *y = calloc(nnodes_discont, sizeof(real_t));

    if (!x || !y) {
        fprintf(stderr, "Unable to allocate memory!\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    fff_t fff;
    int err = tet4_fff_create(&fff, mesh.nelements, mesh.elements, mesh.points);
    if (err) {
        fprintf(stderr, "Unable to create FFFs!\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    for (ptrdiff_t i = 0; i < nnodes_discont; i++) {
        x[i] = 1;
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Measure
    ///////////////////////////////////////////////////////////////////////////////

    double spmv_tick = MPI_Wtime();

    for (int repeat = 0; repeat < SFEM_REPEAT; repeat++) {
        sstet4_laplacian_apply(L, fff.nelements, fff.data, x, y);
    }

    double spmv_tock = MPI_Wtime();
    long nelements = mesh.nelements;
    int element_type = mesh.element_type;

    ///////////////////////////////////////////////////////////////////////////////
    // Output for testing
    ///////////////////////////////////////////////////////////////////////////////

    real_t sq_nrm = 0;
    for (ptrdiff_t i = 0; i < nnodes_discont; i++) {
        sq_nrm += y[i] * y[i];
    }

    printf("sq_nrm = %g\n", sq_nrm);

    // array_write(comm, path_output, SFEM_MPI_REAL_T, y, nnodes_discont, u_n_global);

    ///////////////////////////////////////////////////////////////////////////////
    // Free resources
    ///////////////////////////////////////////////////////////////////////////////

    tet4_fff_destroy(&fff);

    free(x);
    free(y);
    mesh_destroy(&mesh);

    if (SFEM_USE_IDX) {
        for (int d = 0; d < nxe; d++) {
            free(elements[d]);
        }

        free(elements);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Stats
    ///////////////////////////////////////////////////////////////////////////////

    double tock = MPI_Wtime();
    float TTS = tock - tick;
    float TTS_op = (spmv_tock - spmv_tick) / SFEM_REPEAT;

    if (!rank) {
        float mem_coeffs = 2 * nnodes_discont * sizeof(real_t) * 1e-9;
        float mem_jacs = 6 * nelements * sizeof(jacobian_t) * 1e-9;
        float mem_idx = nelements * nxe * sizeof(idx_t) * 1e-9;
        printf("----------------------------------------\n");
        printf("SUMMARY (%s): %s\n", type_to_string(element_type), argv[0]);
        printf("----------------------------------------\n");
        printf("#elements %ld #microelements %ld #nodes %ld\n",
               nelements,
               nelements * txe,
               nnodes_discont);
        printf("#nodexelement %d #microelementsxelement %d\n", nxe, txe);
        printf("Operator TTS:\t\t%.4f\t[s]\n", TTS_op);
        printf("Operator throughput:\t%.1f\t[ME/s]\n", 1e-6f * nelements / TTS_op);
        printf("Operator throughput:\t%.1f\t[MmicroE/s]\n", 1e-6f * nelements * txe / TTS_op);
        printf("Operator throughput:\t%.1f\t[MDOF/s]\n", 1e-6f * nnodes_discont / TTS_op);
        printf("Operator memory %g (2 x coeffs) + %g (FFFs) + %g (index) = %g [GB]\n",
               mem_coeffs,
               mem_jacs,
               mem_idx,
               mem_coeffs + mem_jacs + mem_idx);
        printf("Total:\t\t\t%.4f\t[s]\n", TTS);
        printf("----------------------------------------\n");
    }

    return MPI_Finalize();
}
