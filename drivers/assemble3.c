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

// ptrdiff_t read_file(MPI_Comm comm, const char *path, void **data) {
//     MPI_Status status;
//     MPI_Offset nbytes;
//     MPI_File file;
//     CATCH_MPI_ERROR(MPI_File_open(comm, path, MPI_MODE_RDONLY, MPI_INFO_NULL, &file));
//     CATCH_MPI_ERROR(MPI_File_get_size(file, &nbytes));
//     *data = malloc(nbytes);

//     CATCH_MPI_ERROR(MPI_File_read_at_all(file, 0, *data, nbytes, MPI_CHAR, &status));
//     return nbytes;
// }

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

    if (argc < 2) {
        fprintf(stderr, "usage: %s <folder> [output_folder=./]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *output_folder = "./";
    if (argc > 2) {
        output_folder = argv[2];
    }

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
    // char path[1024 * 10];
    ptrdiff_t nnodes = 0;
    geom_t *xyz[3];

    ptrdiff_t nelements = 0;
    idx_t *elems[4];

    if (serial_read_tet_mesh(folder, &nelements, elems, &nnodes, xyz)) {
        return EXIT_FAILURE;
    }

    // TODO read displacement from file
    real_t *displacement = malloc(nnodes * 3 * sizeof(real_t));
    memset(displacement, 0, nnodes * 3 * sizeof(real_t));

    // TODO read params
    const real_t mu = 1;
    const real_t lambda = 1;

    double tack = MPI_Wtime();
    printf("assemble3.c: read\t\t%g seconds\n", tack - tick);

    ///////////////////////////////////////////////////////////////////////////////
    // Build CRS graph
    ///////////////////////////////////////////////////////////////////////////////

    ptrdiff_t nnz = 0;
    count_t *rowptr = 0;
    idx_t *colidx = 0;
    real_t *values = 0;

    build_crs_graph(nelements, nnodes, elems, &rowptr, &colidx);

    nnz = rowptr[nnodes];
    values = (real_t *)malloc(nnz * 9 * sizeof(real_t));
    memset(values, 0, nnz * 9 * sizeof(real_t));

    double tock = MPI_Wtime();
    printf("assemble3.c: build crs\t\t%g seconds\n", tock - tack);
    tack = tock;

    ///////////////////////////////////////////////////////////////////////////////
    // Operator assembly
    ///////////////////////////////////////////////////////////////////////////////

    neohookean_assemble_hessian(
        // Mesh
        nelements,
        nnodes,
        elems,
        xyz,
        // Material
        mu,
        lambda,
        displacement,
        // Output
        rowptr,
        colidx,
        values);

    tock = MPI_Wtime();
    printf("assemble3.c: assembly\t\t%g seconds\n", tock - tack);
    tack = tock;

    ///////////////////////////////////////////////////////////////////////////////
    // Block to scalar operator
    ///////////////////////////////////////////////////////////////////////////////

    count_t *new_rowptr = (count_t *)malloc(((nnodes)*3 + 1)*sizeof(count_t));
    idx_t *new_colidx = (idx_t *)malloc(nnz * 9 * sizeof(idx_t));
    real_t *new_values = (real_t *)malloc(nnz * 9 * sizeof(real_t));

    block_crs_to_crs(nnodes,
                     3,
                     // Block matrix
                     rowptr,
                     colidx,
                     values,
                     // Scalar matrix
                     new_rowptr,
                     new_colidx,
                     new_values);

    // substitute arrays
    free(rowptr);
    free(colidx);
    free(values);

    rowptr = new_rowptr;
    colidx = new_colidx;
    values = new_values;

    // for (ptrdiff_t i = 0; i < nnodes * 3; ++i) {
    //     idx_t begin = rowptr[i];
    //     idx_t end = rowptr[i + 1];

    //     printf("%d) %d-%d\n", (int)i, begin, end);

    //     for (idx_t k = begin; k < end; ++k) {
    //         printf("(%d, %g) ", colidx[k], values[k]);
    //     }

    //     printf("\n---\n");
    // }

    // printf("bnnz=%d nnz=%d == %d\n-----------------\n", (int)nnz, (int)rowptr[nnodes * 3], (int)(nnz * 9));

    tock = MPI_Wtime();
    printf("assemble3.c: block to scalar\t\t%g seconds\n", tock - tack);
    tack = tock;

    ///////////////////////////////////////////////////////////////////////////////
    // Boundary conditions
    ///////////////////////////////////////////////////////////////////////////////

    // real_t *rhs = (real_t *)malloc(nnodes * sizeof(real_t));
    // memset(rhs, 0, nnodes * sizeof(real_t));

    // {  // Neumann
    //     sprintf(path, "%s/on.raw", folder);
    //     idx_t *faces_neumann = 0;
    //     ptrdiff_t nfacesx3 = read_file(comm, path, (void **)&faces_neumann);
    //     idx_t nfaces = (nfacesx3 / 3) / sizeof(idx_t);
    //     assert(nfaces * 3 * sizeof(idx_t) == nfacesx3);

    //     free(faces_neumann);
    // }

    // if (SFEM_HANDLE_DIRICHLET) {
    //     // Dirichlet
    //     sprintf(path, "%s/zd.raw", folder);
    //     idx_t *dirichlet_nodes = 0;
    //     ptrdiff_t nn = read_file(comm, path, (void **)&dirichlet_nodes);
    //     assert((nn / sizeof(idx_t)) * sizeof(idx_t) == nn);
    //     nn /= sizeof(idx_t);

    // }

    // tock = MPI_Wtime();
    // printf("assemble3.c: boundary\t\t%g seconds\n", tock - tack);
    // tack = tock;

    ///////////////////////////////////////////////////////////////////////////////
    // Write CRS matrix and rhs vector
    ///////////////////////////////////////////////////////////////////////////////

    MPI_Datatype value_type = SFEM_EXPORT_FP32 ? MPI_FLOAT : SFEM_MPI_REAL_T;

    if (SFEM_EXPORT_FP32) {
        array_dtof(nnz * 9, (const real_t *)values, (float *)values);
        // array_dtof(nnodes, (const real_t *)rhs, (float*)rhs);
    }

    {
        crs_t crs_out;
        crs_out.rowptr = (char *)rowptr;
        crs_out.colidx = (char *)colidx;
        crs_out.values = (char *)values;
        crs_out.grows = (nnodes * 3);
        crs_out.lrows = (nnodes * 3);
        crs_out.lnnz = nnz * 9;
        crs_out.gnnz = nnz * 9;
        crs_out.start = 0;
        crs_out.rowoffset = 0;
        crs_out.rowptr_type = SFEM_MPI_COUNT_T;
        crs_out.colidx_type = SFEM_MPI_IDX_T;
        crs_out.values_type = value_type;
        crs_write_folder(comm, output_folder, &crs_out);
    }

    // {
    //     sprintf(path, "%s/rhs.raw", output_folder);
    //     array_write(comm, path, value_type, rhs, nnodes, nnodes);
    // }

    tock = MPI_Wtime();
    printf("assemble3.c: write\t\t%g seconds\n", tock - tack);
    tack = tock;

    ///////////////////////////////////////////////////////////////////////////////
    // Free resources
    ///////////////////////////////////////////////////////////////////////////////

    free(rowptr);
    free(colidx);
    free(values);
    // free(rhs);

    for (int d = 0; d < 3; ++d) {
        free(xyz[d]);
    }

    for (int i = 0; i < 4; ++i) {
        free(elems[i]);
    }

    tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld #nz %ld\n", (long)nelements, (long)nnodes, (long)nnz);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
