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

#include "isotropic_phasefield_for_fracture.h"

#include "read_mesh.h"

#include "sfem_API.hpp"

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
    char *SFEM_INPUT = 0;

    real_t SFEM_MU = 1.0;
    real_t SFEM_LAMBDA = 1.0;
    real_t SFEM_FRACTURE_TOUGHNESS = 1.0;
    real_t SFEM_LENGTH_SCALE_PARAMETER = 1.0;

    SFEM_READ_ENV(SFEM_HANDLE_DIRICHLET, atoi);
    SFEM_READ_ENV(SFEM_EXPORT_FP32, atoi);

    SFEM_READ_ENV(SFEM_MU, atof);
    SFEM_READ_ENV(SFEM_LAMBDA, atof);
    SFEM_READ_ENV(SFEM_FRACTURE_TOUGHNESS, atof);
    SFEM_READ_ENV(SFEM_LENGTH_SCALE_PARAMETER, atof);

    SFEM_READ_ENV(SFEM_INPUT, );

    printf("----------------------------------------\n");
    printf(
        "Environment variables:\n"
        "- SFEM_HANDLE_DIRICHLET=%d\n"
        "- SFEM_EXPORT_FP32=%d\n"
        "- SFEM_MU=%g\n"
        "- SFEM_LAMBDA=%g\n"
        "- SFEM_FRACTURE_TOUGHNESS=%g\n"
        "- SFEM_LENGTH_SCALE_PARAMETER=%g\n"
        "- SFEM_INPUT=%s\n",
        SFEM_HANDLE_DIRICHLET,
        SFEM_EXPORT_FP32,
        SFEM_MU,
        SFEM_LAMBDA,
        SFEM_FRACTURE_TOUGHNESS,
        SFEM_LENGTH_SCALE_PARAMETER,
        SFEM_INPUT);
    printf("----------------------------------------\n");

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];
    char path[1024 * 10];

    auto mesh = sfem::Mesh::create_from_file(sfem::Communicator::wrap(comm), folder);

    const ptrdiff_t nnodes = mesh->n_nodes();
    const ptrdiff_t nelements = mesh->n_elements();

    static const int block_size = 4;
    static const int mat_block_size = 4 * 4;

    // TODO read displacement from file
    real_t *u = (real_t*)malloc(nnodes * block_size * sizeof(real_t));

    if (SFEM_INPUT) {
        ptrdiff_t nlocal, nglobal;
        array_create_from_file(comm, SFEM_INPUT, SFEM_MPI_REAL_T, (void **)&u, &nlocal, &nglobal);

        assert(nlocal == nnodes * block_size);
    } else {
        memset(u, 0, nnodes * block_size * sizeof(real_t));
    }

    const real_t mu = SFEM_MU;
    const real_t lambda = SFEM_LAMBDA;
    const real_t Gc = SFEM_FRACTURE_TOUGHNESS;
    const real_t ls = SFEM_LENGTH_SCALE_PARAMETER;

    double tack = MPI_Wtime();
    printf("assemble4.c: read\t\t%g seconds\n", tack - tick);

    ///////////////////////////////////////////////////////////////////////////////
    // Build CRS graph
    ///////////////////////////////////////////////////////////////////////////////

    ptrdiff_t nnz = 0;
    count_t *rowptr = 0;
    idx_t *colidx = 0;
    real_t *values = 0;

    build_crs_graph(nelements, nnodes, mesh->elements()->data(), &rowptr, &colidx);

    nnz = rowptr[nnodes];
    values = (real_t *)malloc(nnz * mat_block_size * sizeof(real_t));
    memset(values, 0, nnz * mat_block_size * sizeof(real_t));

    double tock = MPI_Wtime();
    printf("assemble4.c: build crs\t\t%g seconds\n", tock - tack);
    tack = tock;

    ///////////////////////////////////////////////////////////////////////////////
    // Operator assembly
    ///////////////////////////////////////////////////////////////////////////////

    isotropic_phasefield_for_fracture_assemble_hessian(nelements,
                                                       nnodes,
                                                       mesh->elements()->data(),
                                                       mesh->points()->data(),
                                                       mu,
                                                       lambda,
                                                       Gc,
                                                       ls,
                                                       u,
                                                       // Output
                                                       rowptr,
                                                       colidx,
                                                       values);

    tock = MPI_Wtime();
    printf("assemble4.c: assembly\t\t%g seconds\n", tock - tack);
    tack = tock;

    ///////////////////////////////////////////////////////////////////////////////
    // Block to scalar operator
    ///////////////////////////////////////////////////////////////////////////////

    count_t *new_rowptr = (count_t *)malloc(((nnodes)*block_size + 1) * sizeof(count_t));
    idx_t *new_colidx = (idx_t *)malloc(nnz * mat_block_size * sizeof(idx_t));
    real_t *new_values = (real_t *)malloc(nnz * mat_block_size * sizeof(real_t));

    block_crs_to_crs(nnodes,
                     block_size,
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

    tock = MPI_Wtime();
    printf("assemble4.c: block to scalar\t\t%g seconds\n", tock - tack);
    tack = tock;

    ///////////////////////////////////////////////////////////////////////////////
    // RHS and Boundary conditions
    ///////////////////////////////////////////////////////////////////////////////

    real_t *rhs = (real_t *)malloc(nnodes * block_size * sizeof(real_t));
    memset(rhs, 0, nnodes * block_size * sizeof(real_t));

    isotropic_phasefield_for_fracture_assemble_gradient(
        nelements, nnodes, mesh->elements()->data(), mesh->points()->data(), mu, lambda, Gc, ls, u, rhs);

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
    // printf("assemble4.c: boundary\t\t%g seconds\n", tock - tack);
    // tack = tock;

    ///////////////////////////////////////////////////////////////////////////////
    // Energy
    ///////////////////////////////////////////////////////////////////////////////

    real_t energy = 0;
    isotropic_phasefield_for_fracture_assemble_value(
        nelements, nnodes, mesh->elements()->data(), mesh->points()->data(), mu, lambda, Gc, ls, u, &energy);

    ///////////////////////////////////////////////////////////////////////////////
    // Write CRS matrix and rhs vector
    ///////////////////////////////////////////////////////////////////////////////

    MPI_Datatype value_type = SFEM_EXPORT_FP32 ? MPI_FLOAT : SFEM_MPI_REAL_T;

    if (SFEM_EXPORT_FP32) {
        array_dtof(nnz * mat_block_size, (const real_t *)values, (float *)values);
        array_dtof(nnodes * block_size, (const real_t *)rhs, (float *)rhs);
        array_dtof(1, (const real_t *)&energy, (float *)&energy);
    }

    {
        crs_t crs_out;
        crs_out.rowptr = (char *)rowptr;
        crs_out.colidx = (char *)colidx;
        crs_out.values = (char *)values;
        crs_out.grows = (nnodes * block_size);
        crs_out.lrows = (nnodes * block_size);
        crs_out.lnnz = nnz * mat_block_size;
        crs_out.gnnz = nnz * mat_block_size;
        crs_out.start = 0;
        crs_out.rowoffset = 0;
        crs_out.rowptr_type = SFEM_MPI_COUNT_T;
        crs_out.colidx_type = SFEM_MPI_IDX_T;
        crs_out.values_type = value_type;
        crs_write_folder(comm, output_folder, &crs_out);
    }

    {
        snprintf(path, sizeof(path), "%s/rhs.raw", output_folder);
        array_write(comm, path, value_type, rhs, nnodes * block_size, nnodes * block_size);
    }

    snprintf(path, sizeof(path), "%s/value.raw", output_folder);
    if (!rank) {
        array_write(MPI_COMM_SELF, path, value_type, &energy, 1, 1);
    }

    tock = MPI_Wtime();
    printf("assemble4.c: write\t\t%g seconds\n", tock - tack);
    tack = tock;

    ///////////////////////////////////////////////////////////////////////////////
    // Free resources
    ///////////////////////////////////////////////////////////////////////////////

    free(rowptr);
    free(colidx);
    free(values);
    // free(rhs);

    tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf(
            "#elements %ld #nodes %ld #nzblocks %ld\n", (long)nelements, (long)nnodes, (long)nnz);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
