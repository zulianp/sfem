#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"

#include "sfem_base.hpp"
#include "smesh_graph.impl.hpp"

#include "isotropic_phasefield_for_fracture.hpp"

#include "sfem_API.hpp"

int assemble4(const std::shared_ptr<sfem::Communicator> &comm, int argc, char *argv[]) {
    if (comm->size() != 1) {
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

    int   SFEM_HANDLE_DIRICHLET = 0;
    int   SFEM_EXPORT_FP32      = 0;
    char *SFEM_INPUT            = 0;

    real_t SFEM_MU                     = 1.0;
    real_t SFEM_LAMBDA                 = 1.0;
    real_t SFEM_FRACTURE_TOUGHNESS     = 1.0;
    real_t SFEM_LENGTH_SCALE_PARAMETER = 1.0;

    SFEM_READ_ENV(SFEM_HANDLE_DIRICHLET, atoi);
    SFEM_READ_ENV(SFEM_EXPORT_FP32, atoi);

    SFEM_READ_ENV(SFEM_MU, atof);
    SFEM_READ_ENV(SFEM_LAMBDA, atof);
    SFEM_READ_ENV(SFEM_FRACTURE_TOUGHNESS, atof);
    SFEM_READ_ENV(SFEM_LENGTH_SCALE_PARAMETER, atof);

    SFEM_READ_ENV(SFEM_INPUT, );

    printf("----------------------------------------\n");
    printf("Environment variables:\n"
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

    auto mesh = sfem::Mesh::create_from_file(comm, smesh::Path(folder));

    const ptrdiff_t nnodes    = mesh->n_nodes();
    const ptrdiff_t nelements = mesh->n_elements();

    static const int block_size     = 4;
    static const int mat_block_size = 4 * 4;

    std::shared_ptr<sfem::Buffer<real_t>> u_buf;
    if (SFEM_INPUT) {
        u_buf = sfem::Buffer<real_t>::from_file(smesh::Path(SFEM_INPUT));
        if (!u_buf) {
            SFEM_ERROR("Failed to read file %s\n", SFEM_INPUT);
        }
        assert((ptrdiff_t)u_buf->size() == nnodes * block_size);
    } else {
        u_buf = sfem::create_host_buffer<real_t>(nnodes * block_size);
    }
    real_t *const u = u_buf->data();

    const real_t mu     = SFEM_MU;
    const real_t lambda = SFEM_LAMBDA;
    const real_t Gc     = SFEM_FRACTURE_TOUGHNESS;
    const real_t ls     = SFEM_LENGTH_SCALE_PARAMETER;

    double tack = MPI_Wtime();
    printf("assemble4.c: read\t\t%g seconds\n", tack - tick);

    ///////////////////////////////////////////////////////////////////////////////
    // Build CRS graph
    ///////////////////////////////////////////////////////////////////////////////

    ptrdiff_t nnz    = 0;
    count_t  *rowptr = 0;
    idx_t    *colidx = 0;
    real_t   *values = 0;

    smesh::create_crs_graph(nelements, nnodes, mesh->elements(0)->data(), &rowptr, &colidx);

    nnz    = rowptr[nnodes];
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
                                                       mesh->elements(0)->data(),
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
    idx_t   *new_colidx = (idx_t *)malloc(nnz * mat_block_size * sizeof(idx_t));
    real_t  *new_values = (real_t *)malloc(nnz * mat_block_size * sizeof(real_t));

    smesh::block_crs_to_crs(nnodes,
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

    auto rhs_buf = sfem::create_host_buffer<real_t>(nnodes * block_size);

    isotropic_phasefield_for_fracture_assemble_gradient(
            nelements, nnodes, mesh->elements(0)->data(), mesh->points()->data(), mu, lambda, Gc, ls, u, rhs_buf->data());

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
            nelements, nnodes, mesh->elements(0)->data(), mesh->points()->data(), mu, lambda, Gc, ls, u, &energy);

    ///////////////////////////////////////////////////////////////////////////////
    // Write CRS matrix, rhs vector, and energy
    ///////////////////////////////////////////////////////////////////////////////

    const ptrdiff_t n_dofs     = nnodes * block_size;
    const ptrdiff_t scalar_nnz = nnz * mat_block_size;

    smesh::create_directory(output_folder);

    auto rowptr_buf = sfem::Buffer<count_t>::wrap(n_dofs + 1, rowptr);
    auto colidx_buf = sfem::Buffer<idx_t>::wrap(scalar_nnz, colidx);
    auto values_buf = sfem::Buffer<real_t>::wrap(scalar_nnz, values);

    if (SFEM_EXPORT_FP32) {
        auto crs = sfem::h_crs_spmv<count_t, idx_t, float>(
                n_dofs, n_dofs, rowptr_buf, colidx_buf, sfem::astype<float>(values_buf), (float)1);
        crs->to_file(smesh::Path(output_folder));
    } else {
        auto crs = sfem::h_crs_spmv<count_t, idx_t, real_t>(n_dofs, n_dofs, rowptr_buf, colidx_buf, values_buf, (real_t)1);
        crs->to_file(smesh::Path(output_folder));
    }

    {
        const smesh::Path rhs_path = smesh::Path(output_folder) / "rhs.raw";
        if (SFEM_EXPORT_FP32) {
            sfem::astype<float>(rhs_buf)->to_file(rhs_path);
        } else {
            rhs_buf->to_file(rhs_path);
        }
    }

    if (!comm->rank()) {
        auto energy_buf              = sfem::create_host_buffer<real_t>(1);
        energy_buf->data()[0]        = energy;
        const smesh::Path value_path = smesh::Path(output_folder) / "value.raw";
        if (SFEM_EXPORT_FP32) {
            sfem::astype<float>(energy_buf)->to_file(value_path);
        } else {
            energy_buf->to_file(value_path);
        }
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

    tock = MPI_Wtime();

    if (!comm->rank()) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld #nzblocks %ld\n", (long)nelements, (long)nnodes, (long)nnz);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return 0;
}

int main(int argc, char *argv[]) {
    auto ctx = sfem::initialize(argc, argv);
    return assemble4(ctx->communicator(), argc, argv);
}
