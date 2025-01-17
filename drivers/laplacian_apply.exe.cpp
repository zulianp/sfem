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
#include "tet4_fff.h"

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

    if (SFEM_USE_MACRO) {
        mesh.element_type = macro_type_variant(mesh.element_type);
    }

    ptrdiff_t u_n_local, u_n_global;

    real_t *x = 0;
    if (strcmp("gen:ones", path_f) == 0) {
        x = (real_t*)malloc(mesh.nnodes * sizeof(real_t));
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < mesh.nnodes; ++i) {
            x[i] = 1;
        }

        u_n_local = mesh.nnodes;
        u_n_global = mesh.nnodes;

    } else {
        array_create_from_file(comm, path_f, SFEM_MPI_REAL_T, (void **)&x, &u_n_local, &u_n_global);
    }

    real_t *y = (real_t*)calloc(u_n_local, sizeof(real_t));

    if (!laplacian_is_opt(mesh.element_type)) {
        SFEM_USE_OPT = 0;
    }

    fff_t fff;
    if (SFEM_USE_OPT) {
        // FIXME!
        tet4_fff_create(&fff, mesh.nelements, mesh.elements, mesh.points);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Measure
    ///////////////////////////////////////////////////////////////////////////////

    double spmv_tick = MPI_Wtime();

    for (int repeat = 0; repeat < SFEM_REPEAT; repeat++) {
        if (SFEM_USE_OPT) {
            laplacian_apply_opt(mesh.element_type, fff.nelements, fff.elements, fff.data, x, y);
        } else {
            laplacian_apply(mesh.element_type,
                            mesh.nelements,
                            mesh.nnodes,
                            mesh.elements,
                            mesh.points,
                            x,
                            y);
        }
    }

    double spmv_tock = MPI_Wtime();
    long nelements = mesh.nelements;
    long nnodes = mesh.nnodes;
    enum ElemType element_type = mesh.element_type;

    ///////////////////////////////////////////////////////////////////////////////
    // Output for testing
    ///////////////////////////////////////////////////////////////////////////////

    array_write(comm, path_output, SFEM_MPI_REAL_T, y, u_n_local, u_n_global);

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
        printf("----------------------------------------\n");
        printf("SUMMARY (%s): %s\n", type_to_string(element_type), argv[0]);
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld\n", nelements, nnodes);
        printf("Operator TTS:\t\t%.4f\t[s]\n", TTS_op);
        printf("Operator throughput:\t%.1f\t[ME/s]\n", 1e-6f * nelements / TTS_op);
        printf("Operator throughput:\t%.1f\t[MDOF/s]\n", 1e-6f * nnodes / TTS_op);
        printf("Total:\t\t\t%.4f\t[s]\n", TTS);
        printf("----------------------------------------\n");
    }

    return MPI_Finalize();
}
