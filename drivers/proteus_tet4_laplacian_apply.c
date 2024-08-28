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
#include "proteus_tet4_laplacian.h"
#include "tet4_fff.h"

#include "adj_table.h"

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

static void create_idx(const int L, mesh_t *mesh /*, ..output*/) {
    double tick = MPI_Wtime();

    // 1) Get the node indices from the TET4 mesh

    // 2) Compute the unique edge-node indices using the CRSGraph
    // A unique edge index can be used and use the multiple to store all indices
    // as consecutive

    count_t *rowptr;
    idx_t *colidx;
    build_crs_graph_for_elem_type(
            mesh->element_type, mesh->nelements, mesh->nnodes, mesh->elements, &rowptr, &colidx);

    ptrdiff_t nxedge = L - 1; // L == 0 (is this correct?)
    ptrdiff_t nedges = (rowptr[mesh->nnodes] - mesh->nnodes) / 2;


    // TODO

    // 3) Compute the unique face-node indices using the adjacency table
    // Two elements share a face, figure out the ordering 

    element_idx_t *adj_table = 0;
    create_element_adj_table(
            mesh->nelements, mesh->nnodes, mesh->element_type, mesh->elements, &adj_table);

    // TODO Consistent ordering with implicit looping scheme needs to be figured out (orientation is reflected like mirror)

    // 4) Compute the unique internal nodes implicitly using the element id and the idx offset of
    // the total number of explicit indices (offset + element_id * n_internal_nodes + local_internal_node_id)
    // ptrdiff_t n_internal_nodes = ?;

    // TODO


    // Clean-up
    free(rowptr);
    free(colidx);
    free(adj_table);
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
    int SFEM_USE_OPT = 1;
    int SFEM_USE_MACRO = 1;

    SFEM_READ_ENV(SFEM_REPEAT, atoi);
    SFEM_READ_ENV(SFEM_USE_OPT, atoi);
    SFEM_READ_ENV(SFEM_USE_MACRO, atoi);

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

    int L = 4;

    const int nxe = proteus_tet4_nxe(L);
    const int txe = proteus_tet4_txe(L);
    ptrdiff_t nnodes_discont = mesh.nelements * nxe;

    // create_idx(L, &mesh);

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
        proteus_tet4_laplacian_apply(L, fff.nelements, fff.data, x, y);
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

    if (SFEM_USE_OPT) {
        tet4_fff_destroy(&fff);
    }

    free(x);
    free(y);
    mesh_destroy(&mesh);

    ///////////////////////////////////////////////////////////////////////////////
    // Stats
    ///////////////////////////////////////////////////////////////////////////////

    double tock = MPI_Wtime();
    float TTS = tock - tick;
    float TTS_op = (spmv_tock - spmv_tick) / SFEM_REPEAT;

    if (!rank) {
        double mem_coeffs = 2 * nnodes_discont * sizeof(real_t) * 1e-9;
        double mem_jacs = 6 * nelements * sizeof(jacobian_t) * 1e-9;
        printf("----------------------------------------\n");
        printf("SUMMARY (%s): %s\n", type_to_string(element_type), argv[0]);
        printf("----------------------------------------\n");
        printf("#elements %ld #microelements %ld #nodes %ld\n", nelements, nelements * txe, nnodes_discont);
        printf("#nodexelement %d #microelementsxelement %d\n", nxe, txe);
        printf("Operator TTS:\t\t%.4f\t[s]\n", TTS_op);
        printf("Operator throughput:\t%.1f\t[ME/s]\n", 1e-6f * nelements / TTS_op);
        printf("Operator throughput:\t%.1f\t[MmicroE/s]\n", 1e-6f * nelements * txe / TTS_op);
        printf("Operator throughput:\t%.1f\t[MDOF/s]\n", 1e-6f * nnodes_discont / TTS_op);
        printf("Operator memory %g (2 x coeffs) + %g (FFFs) = %g [GB]\n",
               mem_coeffs,
               mem_jacs,
               mem_coeffs + mem_jacs);
        printf("Total:\t\t\t%.4f\t[s]\n", TTS);
        printf("----------------------------------------\n");
    }

    return MPI_Finalize();
}
