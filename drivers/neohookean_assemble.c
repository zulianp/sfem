#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../matrix.io/array_dtof.h"
#include "../matrix.io/matrixio_array.h"
#include "../matrix.io/matrixio_crs.h"
#include "../matrix.io/utils.h"

#include "crs_graph.h"
#include "sfem_base.h"

#include "neohookean.h"

#include "read_mesh.h"

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

    if (argc != 3) {
        fprintf(stderr, "usage: %s <mesh> <matrices_and_vectors>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *output_folder = argv[2];
    printf("%s %s %s\n", argv[0], argv[1], output_folder);

    int SFEM_HANDLE_DIRICHLET = 0;
    int SFEM_EXPORT_FP32 = 0;

    SFEM_READ_ENV(SFEM_HANDLE_DIRICHLET, atoi);
    SFEM_READ_ENV(SFEM_EXPORT_FP32, atoi);

    printf("----------------------------------------\n");
    printf("Environment variables:\n- SFEM_HANDLE_DIRICHLET=%d\n- SFEM_EXPORT_FP32=%d\n",
           SFEM_HANDLE_DIRICHLET,
           SFEM_EXPORT_FP32);
    printf("----------------------------------------\n");

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    ptrdiff_t nnodes = mesh.nnodes;
    ptrdiff_t nelements = mesh.nelements;
    const int dims = mesh.spatial_dim;

    real_t **displacement = (real_t **)malloc(dims * sizeof(real_t *));

    for (int b = 0; b < dims * dims; b++) {
        displacement[b] = (real_t *)calloc(nnodes, sizeof(real_t));
    }

    real_t SFEM_MU = 1;
    real_t SFEM_LAMBDA = 1;
    SFEM_READ_ENV(SFEM_MU, atof);
    SFEM_READ_ENV(SFEM_LAMBDA, atof);

    const char * path_pattern_disp = 0;
    SFEM_READ_ENV(path_pattern_disp, );

    if(path_pattern_disp) {
        // TODO
    }

    double tack = MPI_Wtime();
    printf("neohookean_assemble.c: read\t\t%g seconds\n", tack - tick);

    ///////////////////////////////////////////////////////////////////////////////
    // Build CRS graph
    ///////////////////////////////////////////////////////////////////////////////

    ptrdiff_t nnz = 0;
    count_t *rowptr = 0;
    idx_t *colidx = 0;
    build_crs_graph(mesh.nelements, mesh.nnodes, mesh.elements, &rowptr, &colidx);
    nnz = rowptr[nnodes];

    real_t **values = (real_t **)malloc(dims * dims * sizeof(real_t *));

    for (int b = 0; b < dims * dims; b++) {
        values[b] = (real_t *)calloc(nnz, sizeof(real_t));
    }

    double tock = MPI_Wtime();
    printf("neohookean_assemble.c: build crs\t\t%g seconds\n", tock - tack);
    tack = tock;

    ///////////////////////////////////////////////////////////////////////////////
    // Operator assembly
    ///////////////////////////////////////////////////////////////////////////////

    neohookean_assemble_hessian_soa(
        // Mesh
        mesh.nelements,
        mesh.nnodes,
        mesh.elements,
        mesh.points,
        // Material
        SFEM_MU,
        SFEM_LAMBDA,
        displacement,
        // Output
        rowptr,
        colidx,
        values);

    tock = MPI_Wtime();
    printf("neohookean_assemble.c: assembly\t\t%g seconds\n", tock - tack);
    tack = tock;

    ///////////////////////////////////////////////////////////////////////////////
    // Write CRS matrix and rhs vector
    ///////////////////////////////////////////////////////////////////////////////

    MPI_Datatype value_type = SFEM_EXPORT_FP32 ? MPI_FLOAT : SFEM_MPI_REAL_T;

    if (SFEM_EXPORT_FP32) {
        for (int b = 0; b < dims * dims; b++) {
            array_dtof(nnz, (const real_t *)&values[b], (float *)&values[b]);
        }

        // for (int b = 0; b < dims; b++) {
        //     array_dtof(nnodes, (const real_t *)&rhs[b], (float *)&values[b]);
        // }
    }

    {
        block_crs_t crs_out;
        crs_out.rowptr = (char *)rowptr;
        crs_out.colidx = (char *)colidx;
        crs_out.values = (char **)values;
        crs_out.grows = nnodes;
        crs_out.lrows = nnodes;
        crs_out.lnnz = nnz;
        crs_out.gnnz = nnz;
        crs_out.start = 0;
        crs_out.rowoffset = 0;
        crs_out.rowptr_type = SFEM_MPI_COUNT_T;
        crs_out.colidx_type = SFEM_MPI_IDX_T;
        crs_out.values_type = value_type;

        char path_rowptr[1024 * 10];
        sprintf(path_rowptr, "%s/rowptr.raw", output_folder);

        char path_colidx[1024 * 10];
        sprintf(path_colidx, "%s/colidx.raw", output_folder);

        char format_values[1024 * 10];
        sprintf(format_values, "%s/values.%%d.raw", output_folder);
        block_crs_write(comm, path_rowptr, path_colidx, format_values, &crs_out);
    }

    // {
    //     char path[1024 * 10];
    //     for(int b = 0; b < 3; b++) {
    //         sprintf(path, "%s/rhs.%d.raw", output_folder, b);
    //         array_write(comm, path, value_type, &rhs[b], nnodes, nnodes);
    //     }
    // }

    tock = MPI_Wtime();
    printf("neohookean_assemble.c: write\t\t%g seconds\n", tock - tack);
    tack = tock;

    ///////////////////////////////////////////////////////////////////////////////
    // Free resources
    ///////////////////////////////////////////////////////////////////////////////

    free(rowptr);
    free(colidx);
    free(values);

    mesh_destroy(&mesh);

    tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld #nz %ld\n", (long)nelements, (long)nnodes, (long)nnz);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
