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
#include "sshex8.h"
#include "sshex8_laplacian.h"
#include "sfem_hex8_mesh_graph.h"

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

    int SFEM_ELEMENT_REFINE_LEVEL = 2;
    SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);

    int SFEM_HEX8_ASSUME_AFFINE = 0;
    SFEM_READ_ENV(SFEM_HEX8_ASSUME_AFFINE, atoi);

    const int nxe = sshex8_nxe(SFEM_ELEMENT_REFINE_LEVEL);
    const int txe = sshex8_txe(SFEM_ELEMENT_REFINE_LEVEL);
    // ptrdiff_t nnodes_discont = mesh.nelements * nxe;

    printf("nelements %ld\n", mesh.nelements);
    printf("nnodes    %ld\n", mesh.nnodes);
    printf("nxe       %d\n", nxe);
    printf("txe       %d\n", txe);

    idx_t **elements = 0;

    elements = (idx_t**)malloc(nxe * sizeof(idx_t *));
    for (int d = 0; d < nxe; d++) {
        elements[d] = (idx_t*)malloc(mesh.nelements * sizeof(idx_t));
    }

    for (int d = 0; d < nxe; d++) {
        for (ptrdiff_t i = 0; i < mesh.nelements; i++) {
            elements[d][i] = SFEM_IDX_INVALID;
        }
    }

    ptrdiff_t n_unique_nodes, interior_start;
    sshex8_generate_elements(
            SFEM_ELEMENT_REFINE_LEVEL, mesh.nelements, mesh.nnodes, mesh.elements, elements, &n_unique_nodes, &interior_start);

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

    ptrdiff_t internal_nodes = mesh.nelements * (SFEM_ELEMENT_REFINE_LEVEL - 1) *
                               (SFEM_ELEMENT_REFINE_LEVEL - 1) * (SFEM_ELEMENT_REFINE_LEVEL - 1);
                               
    printf("n unique nodes %ld\n", n_unique_nodes);
    printf("vol nodes %ld\n", internal_nodes);
    printf("vol %f%%\n", 100 * (float)internal_nodes / n_unique_nodes);

    real_t *x = (real_t *)calloc(n_unique_nodes, sizeof(real_t));
    real_t *y = (real_t *)calloc(n_unique_nodes, sizeof(real_t));

    for (ptrdiff_t i = 0; i < n_unique_nodes; i++) {
        x[i] = 1;
    }

    // ///////////////////////////////////////////////////////////////////////////////
    // // Measure
    // ///////////////////////////////////////////////////////////////////////////////

    double spmv_tick = MPI_Wtime();

    for (int repeat = 0; repeat < SFEM_REPEAT; repeat++) {
        if (SFEM_HEX8_ASSUME_AFFINE) {
            affine_sshex8_laplacian_apply(SFEM_ELEMENT_REFINE_LEVEL,
                                                mesh.nelements,
                                                interior_start,
                                                elements,
                                                mesh.points,
                                                x,
                                                y);
        } else {
            sshex8_laplacian_apply(SFEM_ELEMENT_REFINE_LEVEL,
                                         mesh.nelements,
                                         interior_start,
                                         elements,
                                         mesh.points,
                                         x,
                                         y);
        }
    }

    double spmv_tock = MPI_Wtime();
    long nelements = mesh.nelements;
    enum ElemType element_type = mesh.element_type;

    // ///////////////////////////////////////////////////////////////////////////////
    // // Output for testing
    // ///////////////////////////////////////////////////////////////////////////////

    real_t sq_nrm = 0;
    for (ptrdiff_t i = 0; i < n_unique_nodes; i++) {
        sq_nrm += y[i] * y[i];
    }

    printf("sq_nrm = %g\n", sq_nrm);

    // // array_write(comm, path_output, SFEM_MPI_REAL_T, y, nnodes_discont, u_n_global);

    // ///////////////////////////////////////////////////////////////////////////////
    // // Free resources
    // ///////////////////////////////////////////////////////////////////////////////

    free(x);
    free(y);
    mesh_destroy(&mesh);

    for (int d = 0; d < nxe; d++) {
        free(elements[d]);
    }

    free(elements);

    // ///////////////////////////////////////////////////////////////////////////////
    // // Stats
    // ///////////////////////////////////////////////////////////////////////////////

    double tock = MPI_Wtime();
    float TTS = tock - tick;
    float TTS_op = (spmv_tock - spmv_tick) / SFEM_REPEAT;

    if (!rank) {
        float mem_coeffs = 2 * n_unique_nodes * sizeof(real_t) * 1e-9;
        float mem_jacs = 6 * nelements * sizeof(jacobian_t) * 1e-9;
        float mem_idx = nelements * nxe * sizeof(idx_t) * 1e-9;
        printf("----------------------------------------\n");
        printf("SUMMARY (%s): %s\n", type_to_string(element_type), argv[0]);
        printf("----------------------------------------\n");
        printf("#elements %ld #microelements %ld #nodes %ld\n",
               nelements,
               nelements * txe,
               n_unique_nodes);
        printf("#nodexelement %d #microelementsxelement %d\n", nxe, txe);
        printf("Operator TTS:\t\t%.4f\t[s]\n", TTS_op);
        printf("Operator throughput:\t%.1f\t[ME/s]\n", 1e-6f * nelements / TTS_op);
        printf("Operator throughput:\t%.1f\t[MmicroE/s]\n", 1e-6f * nelements * txe / TTS_op);
        printf("Operator throughput:\t%.1f\t[MDOF/s]\n", 1e-6f * n_unique_nodes / TTS_op);
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
