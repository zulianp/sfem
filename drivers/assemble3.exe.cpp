#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"

#include "sfem_base.hpp"
#include "smesh_graph.impl.hpp"

// #include "tet4_neohookean.hpp"
#include "linear_elasticity.hpp"

#include "sfem_API.hpp"

int solve_assemble3(const std::shared_ptr<sfem::Communicator> &comm, int argc, char *argv[]) {
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

    int SFEM_HANDLE_DIRICHLET = 0;
    int SFEM_EXPORT_FP32      = 0;

    SFEM_READ_ENV(SFEM_HANDLE_DIRICHLET, atoi);
    SFEM_READ_ENV(SFEM_EXPORT_FP32, atoi);

    int SFEM_USE_MACRO = 0;

    SFEM_READ_ENV(SFEM_USE_MACRO, atoi);

    printf("----------------------------------------\n");
    printf("Environment variables:\n- SFEM_HANDLE_DIRICHLET=%d\n- SFEM_EXPORT_FP32=%d\n",
           SFEM_HANDLE_DIRICHLET,
           SFEM_EXPORT_FP32);
    printf("----------------------------------------\n");

    double tick = smesh::time_seconds();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];

    auto mesh = sfem::Mesh::create_from_file(comm, smesh::Path(folder));

    auto element_type = mesh->element_type(0);
    if (SFEM_USE_MACRO) {
        element_type = macro_type_variant(element_type);
    }

    const ptrdiff_t nnodes    = mesh->n_nodes();
    const ptrdiff_t nelements = mesh->n_elements();

    // TODO read displacement from file
    real_t *displacement = (real_t *)malloc((size_t)nnodes * mesh->spatial_dimension() * sizeof(real_t));
    memset(displacement, 0, (size_t)nnodes * mesh->spatial_dimension() * sizeof(real_t));

    // TODO read params
    const real_t mu     = 1;
    const real_t lambda = 1;

    double tack = smesh::time_seconds();
    printf("assemble3.c: read\t\t%g seconds\n", tack - tick);

    ///////////////////////////////////////////////////////////////////////////////
    // Build CRS graph
    ///////////////////////////////////////////////////////////////////////////////

    ptrdiff_t nnz    = 0;
    count_t  *rowptr = 0;
    idx_t    *colidx = 0;
    real_t   *values = 0;

    smesh::create_crs_graph_for_elem_type(element_type, nelements, nnodes, mesh->elements(0)->data(), &rowptr, &colidx);

    nnz    = rowptr[nnodes];
    values = (real_t *)malloc((size_t)nnz * 9 * sizeof(real_t));
    memset(values, 0, (size_t)nnz * 9 * sizeof(real_t));

    double tock = smesh::time_seconds();
    printf("assemble3.c: build crs\t\t%g seconds\n", tock - tack);
    tack = tock;

    ///////////////////////////////////////////////////////////////////////////////
    // Operator assembly
    ///////////////////////////////////////////////////////////////////////////////

    // neohookean_assemble_hessian(
    linear_elasticity_crs_aos(element_type,
                              // Mesh
                              nelements,
                              nnodes,
                              mesh->elements(0)->data(),
                              mesh->points()->data(),
                              // Material
                              mu,
                              lambda,
                              // Output
                              rowptr,
                              colidx,
                              values);

    tock = smesh::time_seconds();
    printf("assemble3.c: assembly\t\t%g seconds\n", tock - tack);
    tack = tock;

    ///////////////////////////////////////////////////////////////////////////////
    // Block to scalar operator
    ///////////////////////////////////////////////////////////////////////////////

    count_t *new_rowptr = (count_t *)malloc(((nnodes)*mesh->spatial_dimension() + 1) * sizeof(count_t));
    idx_t   *new_colidx = (idx_t *)malloc((size_t)nnz * mesh->spatial_dimension() * mesh->spatial_dimension() * sizeof(idx_t));
    real_t  *new_values = (real_t *)malloc((size_t)nnz * mesh->spatial_dimension() * mesh->spatial_dimension() * sizeof(real_t));

    smesh::block_crs_to_crs(nnodes,
                            mesh->spatial_dimension(),
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

    // printf("bnnz=%d nnz=%d == %d\n-----------------\n", (int)nnz, (int)rowptr[nnodes * 3],
    // (int)(nnz * 9));

    tock = smesh::time_seconds();
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

    // tock = smesh::time_seconds();
    // printf("assemble3.c: boundary\t\t%g seconds\n", tock - tack);
    // tack = tock;

    ///////////////////////////////////////////////////////////////////////////////
    // Write CRS matrix
    ///////////////////////////////////////////////////////////////////////////////

    const ptrdiff_t n_dofs     = nnodes * mesh->spatial_dimension();
    const ptrdiff_t scalar_nnz = nnz * mesh->spatial_dimension() * mesh->spatial_dimension();

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

    tock = smesh::time_seconds();
    printf("assemble3.c: write\t\t%g seconds\n", tock - tack);
    tack = tock;

    ///////////////////////////////////////////////////////////////////////////////
    // Free resources
    ///////////////////////////////////////////////////////////////////////////////

    free(rowptr);
    free(colidx);
    free(values);

    tock = smesh::time_seconds();

    if (!comm->rank()) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld #nz %ld\n", (long)nelements, (long)nnodes, (long)nnz);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return 0;
}

int main(int argc, char *argv[]) {
    auto ctx = sfem::initialize(argc, argv);
    return solve_assemble3(ctx->communicator(), argc, argv);
}
