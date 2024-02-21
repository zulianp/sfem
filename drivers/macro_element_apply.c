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

#include "macro_tet4_laplacian.h"
#include "macro_tri3_laplacian.h"

// Tet4 on Arm (MF Slower than naive SPMV), 8 cores
// Serial    MF: 0.131053,          SPMV: 0.0386919 (3.38x)
// OpenMP    MF: 0.031211 (0.03891), SPMV: 0.0115319 (2.7x)
// SpeedUp   MF: 4.198x,            SPMV: 3.355x
// Tet4 on Zen4 (MF-AVX2 faster with 16 cores, SPMV does not scale with TLP)
// Serial    MF: 0.106794           SPMV: 0.0571115 
// OpenMP(8) MF: 0.0287701          SPMV: 0.0268007
// OpenMP(16)MF: 0.0163542          SPMV: 0.0292629 

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
    SFEM_READ_ENV(SFEM_REPEAT, atoi);

    int SFEM_USE_OPT = 1;
    SFEM_READ_ENV(SFEM_USE_OPT, atoi);

    const char *folder = argv[1];
    const char *path_f = argv[2];
    const char *path_output = argv[3];

    printf("%s %s %s %s\n", argv[0], folder, path_f, path_output);

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    real_t *x;
    ptrdiff_t u_n_local, u_n_global;
    array_create_from_file(comm, path_f, SFEM_MPI_REAL_T, (void **)&x, &u_n_local, &u_n_global);

    real_t *y = calloc(u_n_local, sizeof(real_t));

    macro_tet4_laplacian_t tet4_ctx;
    if (mesh.element_type == TET10) {
        macro_tet4_laplacian_init(&tet4_ctx, mesh.nelements, mesh.elements, mesh.points);
    }

    double spmv_tick = MPI_Wtime();

    for (int repeat = 0; repeat < SFEM_REPEAT; repeat++) {
        if (mesh.element_type == TRI6) {
            // mesh.element_type = MACRO_TRI3;
            macro_tri3_laplacian_apply(
                mesh.nelements, mesh.nnodes, mesh.elements, mesh.points, x, y);
        } else if (mesh.element_type == TET10) {

            if(SFEM_USE_OPT) {
                macro_tet4_laplacian_apply_opt(
                   &tet4_ctx, x, y);
            } else {
                macro_tet4_laplacian_apply(
                mesh.nelements, mesh.nnodes, mesh.elements, mesh.points, x, y);
            }
        } else {
            return EXIT_FAILURE;
        }
    }

    double spmv_tock = MPI_Wtime();
    printf("macro_element_apply: %g (seconds)\n", (spmv_tock - spmv_tick) / SFEM_REPEAT);

    if (mesh.element_type == TET10) {
        macro_tet4_laplacian_destroy(&tet4_ctx);
    }

    array_write(comm, path_output, SFEM_MPI_REAL_T, y, u_n_local, u_n_global);

    ///////////////////////////////////////////////////////////////////////////////
    // Free resources
    ///////////////////////////////////////////////////////////////////////////////

    free(x);
    free(y);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld\n", (long)mesh.nelements, (long)mesh.nnodes);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
